#include <metal_stdlib>
using namespace metal;

// log_2 of the band texture size. Must match the dimensions of the provided texture.
// Could be supplied as a function constant instead.
constexpr constant int kLogBandTextureWidth = 12;

struct ViewConstants {
    float4x4 modelViewProjectionMatrix;
    float2 viewportSize;
};

struct VertexIn {
    float4 posAndNorm [[attribute(0)]]; // xy = object-space position, zw = outward normal
    float4 texAndAtlasOffsets [[attribute(1)]]; // xy = em-space coords, zw = packed glyph/band data
    float4 invJacobian [[attribute(2)]]; // inverse Jacobian for moving from object space to viewport space
    float4 bandTransform [[attribute(3)]]; // 2D scales and offsets for band transform
    float4 color [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 texCoords;
    float4 bandTransform [[flat]];
    int4 glyph [[flat]];
};

// Calculate the dynamic dilation for the em-space glyph coordinates and vertex position.
// Derivation: https://terathon.com/blog/decade-slug.html#:~:text=derivation
static float2 dilate(float2 position, float2 vertNorm, float2 tex,
                     float2x2 invJacobian, float4x4 mvpTranspose, float2 viewportSize,
                     thread float2 &outDilatedPosition)
{
    float4 m0 = mvpTranspose[0], m1 = mvpTranspose[1], m3 = mvpTranspose[3];

    float2 norm = normalize(vertNorm);
    float s = dot(m3.xy, position.xy) + m3.w;
    float t = dot(m3.xy, norm);

    float u = (s * dot(m0.xy, norm) - t * (dot(m0.xy, position.xy) + m0.w)) * viewportSize.x;
    float v = (s * dot(m1.xy, norm) - t * (dot(m1.xy, position.xy) + m1.w)) * viewportSize.y;

    float s2 = s * s;
    float st = s * t;
    float uv = u * u + v * v;
    float2 dir = vertNorm * (s2 * (st + sqrt(uv)) / (uv - st * st));

    outDilatedPosition = position + dir;
    return tex + invJacobian * dir;
}

static inline int4 unpack(float2 packed) {
    uint2 g = as_type<uint2>(packed);
    return int4(int(g.x & 0xFFFFu), int(g.x >> 16u), int(g.y & 0xFFFFu), int(g.y >> 16u));
}

[[vertex]]
VertexOut glyph_vertex(VertexIn in [[stage_in]], constant ViewConstants &view [[buffer(1)]]) {
    float2x2 invJacobian { in.invJacobian.xy, in.invJacobian.zw };
    float2 dilatedPosition{};
    float2 dilatedUV = dilate(in.posAndNorm.xy, in.posAndNorm.zw, in.texAndAtlasOffsets.xy, invJacobian,
                              transpose(view.modelViewProjectionMatrix), view.viewportSize,
                              dilatedPosition);
    VertexOut out {
        .position = view.modelViewProjectionMatrix * float4(dilatedPosition, 0, 1),
        .color = in.color,
        .texCoords = dilatedUV,
        .bandTransform = in.bandTransform,
        .glyph = unpack(in.texAndAtlasOffsets.zw),
    };
    return out;
}

static uint classify_roots(float y1, float y2, float y3) {
    uint i1 = as_type<uint>(y1) >> 31u;
    uint i2 = as_type<uint>(y2) >> 30u;
    uint i3 = as_type<uint>(y3) >> 29u;
    uint shift = (i2 & 2u) | (i1 & ~2u);
    shift = (i3 & 4u) | (shift & ~4u);
    return (0x2E74u >> shift) & 0x0101u;
}

static float2 solve_poly(float2 p1, float2 p2, float2 p3, int axis/* = 0 for x, 1 for y*/) {
    float2 a = p1 - p2 * 2.0f + p3;
    float2 b = p1 - p2;
    // Clamp discriminant to non-negative; this only occurs in equivalence classes C & F
    float disc = sqrt(max(b[1 - axis] * b[1 - axis] - a[1 - axis] * p1[1 - axis], 0.0f));
    float ra = 1.0f / a[1 - axis];
    float t1 = (b[1 - axis] - disc) * ra;
    float t2 = (b[1 - axis] + disc) * ra;
    if (abs(a[1 - axis]) < 1e-5) { // Almost linear; switch to t1 = t2 = c / 2b for stability
        float r2b = 1.0f / (b[1 - axis] * 2.0f);
        t1 = p1[1 - axis] * r2b;
        t2 = p1[1 - axis] * r2b;
    }
    return float2((a[axis] * t1 - b[axis] * 2.0f) * t1 + p1[axis],
                  (a[axis] * t2 - b[axis] * 2.0f) * t2 + p1[axis]);
}

static int2 lookup_band(int2 glyphLoc, uint offset) {
    int2 loc = int2(glyphLoc.x + int(offset), glyphLoc.y);
    loc.y += loc.x >> kLogBandTextureWidth;
    loc.x &= (1 << kLogBandTextureWidth) - 1;
    return loc;
}

[[fragment]]
float4 glyph_fragment(VertexOut in [[stage_in]],
                    texture2d<float> curveData [[texture(0)]],
                    texture2d<uint>  bandData  [[texture(1)]])
{
    float2 emUV = in.texCoords;
    float2 emsPerPixel = fwidth(emUV);
    float2 pixelsPerEm = 1.0f / emsPerPixel;

    int2 glyphLoc = in.glyph.xy;
    int2 bandMax = in.glyph.zw;
    bandMax.y &= 0x00FF;

    float2 bandScale = in.bandTransform.xy;
    float2 bandOffset = in.bandTransform.zw;

    int2 bandIndex = clamp(int2(emUV * bandScale + bandOffset), int2(0, 0), bandMax);

    // Horizontal cast (along +X)
    uint2 hBand = bandData.read(uint2(glyphLoc.x + bandIndex.y, glyphLoc.y)).xy;
    int2 hl = lookup_band(glyphLoc, hBand.y);
    float xCoverage = 0.0f, xWeight = 0.0f;
    for (int ci = 0; ci < int(hBand.x); ci++) {
        int2 cl = int2(bandData.read(uint2(hl.x + ci, hl.y)).xy);

        // Retrieve Bezier control points and shift origin to current sample point
        float4 p12 = curveData.read(uint2(cl)) - float4(emUV, emUV);
        float2 p1 = p12.xy, p2 = p12.zw;
        float2 p3 = curveData.read(uint2(cl.x + 1, cl.y)).xy - emUV;

        // Early out if we can't possibly intersect with this curve
        if (max(max(p1.x, p2.x), p3.x) * pixelsPerEm.x < -0.5f) {
            break;
        }

        // Determine eligibility of roots
        uint code = classify_roots(p1.y, p2.y, p3.y);
        if (code != 0u) {
            // We have at least one eligible root for this curve; solve for intersections
            float2 r = solve_poly(p12.xy, p12.zw, p3, 0) * pixelsPerEm.x;
            if ((code & 1u) != 0u) {
                // Root t1 affects coverage
                xCoverage += saturate(r.x + 0.5f);
                xWeight = max(xWeight, saturate(1.0f - abs(r.x) * 2.0f));
            }
            if (code > 1u) {
                // Root t2 affects coverage
                xCoverage -= saturate(r.y + 0.5f);
                xWeight  = max(xWeight, saturate(1.0f - abs(r.y) * 2.0f));
            }
        }
    }

    // Vertical cast (along +Y)
    uint2 vd = bandData.read(uint2(glyphLoc.x + bandMax.y + 1 + bandIndex.x, glyphLoc.y)).xy;
    int2  vl = lookup_band(glyphLoc, vd.y);
    float yCoverage = 0.0f, yWeight = 0.0f;
    for (int ci = 0; ci < int(vd.x); ci++) {
        int2 cl = int2(bandData.read(uint2(vl.x + ci, vl.y)).xy);

        // Retrieve Bezier control points and shift origin to current sample point
        float4 p12 = curveData.read(uint2(cl)) - float4(emUV, emUV);
        float2 p1 = p12.xy, p2 = p12.zw;
        float2 p3  = curveData.read(uint2(cl.x + 1, cl.y)).xy - emUV;

        // Early out if we can't possibly intersect with this curve
        if (max(max(p12.y, p12.w), p3.y) * pixelsPerEm.y < -0.5f) {
            break;
        }

        // Determine eligibility of roots
        uint code = classify_roots(p1.x, p2.x, p3.x);
        if (code != 0u) {
            // We have at least one eligible root for this curve; solve for intersections
            float2 r = solve_poly(p1, p2, p3, 1) * pixelsPerEm.y;
            if ((code & 1u) != 0u) {
                // Root t1 affects coverage
                yCoverage -= saturate(r.x + 0.5f);
                yWeight  = max(yWeight, saturate(1.0f - abs(r.x) * 2.0f));
            }
            if (code > 1u) {
                // Root t2 affects coverage
                yCoverage += saturate(r.y + 0.5f);
                yWeight  = max(yWeight, saturate(1.0f - abs(r.y) * 2.0f));
            }
        }
    }

    // Final coverage
    float denom = max(xWeight + yWeight, 1.0f / 65536.0f);
    float coverage = max(abs(xCoverage * xWeight + yCoverage * yWeight) / denom, min(abs(xCoverage), abs(yCoverage)));
    coverage = saturate(coverage);

    return in.color * coverage;
}

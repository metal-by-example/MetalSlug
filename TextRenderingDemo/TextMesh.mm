#include "TextMesh.h"

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <span>
#include <unordered_map>
#include <vector>

// TODO: Switch to std::float16_t when <stdfloat> becomes available in Clang
// https://github.com/llvm/llvm-project/issues/105196
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

namespace {

#ifdef FLT16_MIN
    using half = _Float16; // Favor _Float16 if available
#else
    using half = __fp16;   // Fall back on the more widely available __fp16 otherwise
#endif

static const int kCurveTexWidth = 4096;
static const int kBandTexWidth  = 4096; // must equal 2^kLogBandTextureWidth in shader

struct GlyphVertex {
    simd_float4 posAndNorm;
    simd_float4 texAndAtlasOffsets;
    simd_float4 invJacobian;
    simd_float4 bandTransform;
    simd_float4 color;
};

struct QuadBezier {
    simd_float2 p0, p1, p2;

    float getMinX() const { return fmin(p0.x, fmin(p1.x, p2.x)); }
    float getMaxX() const { return fmax(p0.x, fmax(p1.x, p2.x)); }
    float getMinY() const { return fmin(p0.y, fmin(p1.y, p2.y)); }
    float getMaxY() const { return fmax(p0.y, fmax(p1.y, p2.y)); }

    bool isStraightHorizontal() const {
        return fabs(p0.y - p2.y) < 1e-5 && fabs(p1.y - (p0.y + p2.y) * 0.5) < 1e-5;
    }

    bool isStraightVertical() const {
        return fabs(p0.x - p2.x) < 1e-5 && fabs(p1.x - (p0.x + p2.x) * 0.5) < 1e-5;
    }
};

struct GlyphInfo {
    float advanceWidth; // in em units (normalized)
    float xMin, yMin, xMax, yMax;
    int curveTexStart; // first texel index (linear) in curve texture
    int curveCount;
    int bandTexX, bandTexY; // top-left texel of this glyph's band data
    int numHorizBands;
    int numVertBands;
    float bandScaleX, bandScaleY;
    float bandOffsetX, bandOffsetY;
};

CGPoint lerp(CGPoint a, CGPoint b, double t) {
    return CGPointMake(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
}

double distSq(CGPoint a, CGPoint b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Cubic (P0,C1,C2,P3) → two quadratics via subdivision at t=0.5
std::array<QuadBezier, 2> cubicToQuadratic(CGPoint p0, CGPoint c1, CGPoint c2, CGPoint p3) {
    CGPoint m01  = lerp(p0, c1, 0.5);
    CGPoint m12  = lerp(c1, c2, 0.5);
    CGPoint m23  = lerp(c2, p3, 0.5);
    CGPoint m012 = lerp(m01, m12, 0.5);
    CGPoint m123 = lerp(m12, m23, 0.5);
    CGPoint mid  = lerp(m012, m123, 0.5);
    return {
        QuadBezier {
            simd_float2 { (float)p0.x, (float)p0.y },
            simd_float2 { (float)m012.x, (float)m012.y },
            simd_float2 { (float)mid.x, (float)mid.y }
        },
        QuadBezier {
            simd_float2 { (float)mid.x, (float)mid.y },
            simd_float2 { (float)m123.x, (float)m123.y },
            simd_float2 { (float)p3.x, (float)p3.y }
        }
    };
}

class FontAtlas {
private:
    id<MTLDevice> device;
    CTFontRef font;
    std::unordered_map<CGGlyph, GlyphInfo> glyphCache;
    float unitsPerEm;

    half *curvePixels; // RGBA16F, 4 channels per texel
    uint16_t *bandPixels; // RG16Uint, 2 channels per texel
    int maxCurveTexels = 4096 * 64;
    int maxBandTexels  = 4096 * 64;
    int curveTexCursor = 0; // next free texel index (linear)
    int bandTexCursor  = 0; // next free texel index (linear)

public:
    id<MTLTexture> curveTexture;
    id<MTLTexture> bandTexture;

    FontAtlas(const char *fontName, id<MTLDevice> device) : device(device) {
        CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, fontName, kCFStringEncodingUTF8);
        font = CTFontCreateWithName(name, 1.0, NULL);
        CFRelease(name);
        unitsPerEm = CTFontGetUnitsPerEm(font);

        (void)posix_memalign((void **)&curvePixels, getpagesize(), sizeof(half) * maxCurveTexels * 4);
        (void)posix_memalign((void **)&bandPixels, getpagesize(), sizeof(uint16_t) * maxBandTexels * 2);
    }

    ~FontAtlas() {
        free(bandPixels);
        free(curvePixels);
        CFRelease(font);
    }

    void insertGlyphs(std::span<CGGlyph> glyphs) {
        if (ensureGlyphs(glyphs)) {
            uploadTextures(device);
        }
    }

    GlyphInfo glyphInfoForGlyph(CGGlyph glyph) {
        auto const& infoIter = glyphCache.find(glyph);
        if (infoIter != glyphCache.end()) {
            return infoIter->second;
        }
        return GlyphInfo{};
    }

private:

    void ensureGlyph(unsigned short &glyph) {
        CGSize adv;
        CTFontGetAdvancesForGlyphs(font, kCTFontOrientationHorizontal, &glyph, &adv, 1);
        float advanceEm = static_cast<float>(adv.width);

        CGPathRef path = CTFontCreatePathForGlyph(font, glyph, NULL);
        if (path == NULL) {
            // Trivial glyph without a path, such as whitespace
            glyphCache.insert({ glyph, GlyphInfo { advanceEm, 0, 0, advanceEm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } });
            return;
        }

        std::vector<QuadBezier> curves = extractQuadBeziersFromPath(path);
        CGPathRelease(path);

        if (curves.size() == 0) {
            // Trivial glyph without a path, such as whitespace
            glyphCache.insert({ glyph, GlyphInfo { advanceEm, 0, 0, advanceEm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } });
            return;
        }

        CGFloat xMin = std::reduce(begin(curves), end(curves), FLT_MAX, [](float partial, QuadBezier const& b) {
            return fmin(partial, b.getMinX());
        });
        CGFloat xMax = std::reduce(begin(curves), end(curves), -FLT_MAX, [](float partial, QuadBezier const& b) {
            return fmax(partial, b.getMaxX());
        });
        CGFloat yMin = std::reduce(begin(curves), end(curves), FLT_MAX, [](float partial, QuadBezier const& b) {
            return fmin(partial, b.getMinY());
        });
        CGFloat yMax = std::reduce(begin(curves), end(curves), -FLT_MAX, [](float partial, QuadBezier const& b) {
            return fmax(partial, b.getMaxY());
        });

        int curveStart = curveTexCursor;

        std::vector<simd_float2> curveTexCoords;
        for (auto const& curve : curves) {
            int idx = curveTexCursor;
            int tx = idx % kCurveTexWidth;
            int ty = idx / kCurveTexWidth;
            curveTexCoords.push_back(simd_make_float2(tx, ty));

            // Texel 0: (p0.x, p0.y, p1.x, p1.y)
            writeCurveTexel(curve.p0.x, curve.p0.y, curve.p1.x, curve.p1.y);
            // Texel 1: (p2.x, p2.y, 0, 0)
            writeCurveTexel(curve.p2.x, curve.p2.y, 0, 0);
        }

        int numH = 8;
        int numV = 8;
        float eps = 1.0 / 1024.0;

        float hBandHeight = (yMax - yMin) / static_cast<float>(numH);
        float vBandWidth  = (xMax - xMin) / static_cast<float>(numV);

        std::vector<std::vector<int>> hBands;
        for (int b = 0; b < numH; ++b) {
            float bYMin = yMin + (float)b * hBandHeight - eps;
            float bYMax = yMin + (float)(b + 1) * hBandHeight + eps;
            std::vector<int> indices;
            for (int i = 0; i < curves.size(); ++i) {
                const auto &c = curves[i];
                if (c.isStraightHorizontal()) { continue; }
                if (c.getMaxY() >= bYMin && c.getMinY() <= bYMax) { indices.push_back(i); }
            }
            // Sort descending by max X (early-out optimization)
            std::sort(begin(indices), end(indices), [&](int i0, int i1) {
                return curves[i0].getMaxX() > curves[i1].getMaxX();
            });
            hBands.push_back(indices);
        }

        std::vector<std::vector<int>> vBands;
        for (int b = 0; b < numV; ++b) {
            float bXMin = xMin + (float)b * vBandWidth  - eps;
            float bXMax = xMin + (float)(b + 1) * vBandWidth + eps;
            std::vector<int> indices;
            for (int i = 0; i < curves.size(); ++i) {
                const auto &c = curves[i];
                if (c.isStraightVertical()) { continue; }
                if (c.getMaxX() >= bXMin && c.getMinX() <= bXMax) { indices.push_back(i); }
            }
            // Sort descending by max Y
            std::sort(begin(indices), end(indices), [&](int i0, int i1) {
                return curves[i0].getMaxY() > curves[i1].getMaxY();
            });
            vBands.push_back(indices);
        }

        int bandStartLinear = bandTexCursor;
        int bandStartX = bandStartLinear % kBandTexWidth;
        int bandStartY = bandStartLinear / kBandTexWidth;

        // Headers: numH horiz headers, then numV vert headers
        int headerCount = numH + numV;
        int headerOffset = bandTexCursor;

        // Reserve header slots (fill in offsets later)
        bandTexCursor += headerCount;

        std::vector<int> hCurveListOffsets; // relative offset from bandStart
        for (int b = 0; b < numH; ++b) {
            int relOffset = bandTexCursor - bandStartLinear;
            hCurveListOffsets.push_back(relOffset);
            for (int ci : hBands[b]) {
                simd_float2 t = curveTexCoords[ci];
                writeBandTexel(static_cast<uint16_t>(t.x), static_cast<uint16_t>(t.y));
            }
        }

        std::vector<int> vCurveListOffsets;
        for (int b = 0; b < numV; ++b) {
            int relOffset = bandTexCursor - bandStartLinear;
            vCurveListOffsets.push_back(relOffset);
            for (int ci : vBands[b]) {
                simd_float2 t = curveTexCoords[ci];
                writeBandTexel(static_cast<uint16_t>(t.x), static_cast<uint16_t>(t.y));
            }
        }

        // Write headers (count, relativeOffset)
        int ptr = headerOffset;
        for (int b = 0; b < numH; ++b) {
            int count  = (int)hBands[b].size();
            int offset = hCurveListOffsets[b];
            bandPixels[ptr * 2 + 0] = static_cast<uint16_t>(count);
            bandPixels[ptr * 2 + 1] = static_cast<uint16_t>(offset);
            ptr += 1;
        }
        for (int b = 0; b < numV; ++b) {
            int count  = (int)vBands[b].size();
            int offset = vCurveListOffsets[b];
            bandPixels[ptr * 2 + 0] = static_cast<uint16_t>(count);
            bandPixels[ptr * 2 + 1] = static_cast<uint16_t>(offset);
            ptr += 1;
        }

        // Band transform: maps em-space -> band index
        // bandIndex.x (vert) = (x - xMin) * numV / (xMax - xMin)  → clamped 0..numV-1
        // bandIndex.y (horiz) = (y - yMin) * numH / (yMax - yMin)  → clamped 0..numH-1
        float bSX = (float)numV / fmax(xMax - xMin, 1e-6);
        float bSY = (float)numH / fmax(yMax - yMin, 1e-6);
        float bOX = -xMin * bSX;
        float bOY = -yMin * bSY;

        glyphCache.insert({ glyph,
            {
                advanceEm, (float)xMin, (float)yMin, (float)xMax, (float)yMax,
                curveStart, (int)curves.size(),
                bandStartX, bandStartY, numH, numV,
                bSX, bSY, bOX, bOY
            }
        });
    }

    bool ensureGlyphs(std::span<CGGlyph> glyphs) {
        bool didUpdate = false;
        for (auto glyph : glyphs) {
            if (glyphCache.find(glyph) != glyphCache.end()) { continue; }
            ensureGlyph(glyph);
            didUpdate = true;
        }
        return didUpdate;
    }

    void writeCurveTexel(float r, float g, float b, float a) {
        int i = curveTexCursor;
        curvePixels[i * 4 + 0] = static_cast<half>(r);
        curvePixels[i * 4 + 1] = static_cast<half>(g);
        curvePixels[i * 4 + 2] = static_cast<half>(b);
        curvePixels[i * 4 + 3] = static_cast<half>(a);
        curveTexCursor += 1;
    }

    void writeBandTexel(uint16_t r, uint16_t g) {
        int i = bandTexCursor;
        bandPixels[i * 2 + 0] = r;
        bandPixels[i * 2 + 1] = g;
        bandTexCursor += 1;
    }

    void uploadTextures(id<MTLDevice> device) {
        int cw = kCurveTexWidth;
        int ch = std::max(1, (curveTexCursor + cw - 1) / cw);
        int bw = kBandTexWidth;
        int bh = std::max(1, (bandTexCursor  + bw - 1) / bw);

        MTLTextureDescriptor *curveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                             width:cw
                                                                                            height:ch
                                                                                         mipmapped:NO];
        curveDesc.usage = MTLTextureUsageShaderRead;
        curveDesc.storageMode = MTLStorageModeShared; // N.B. This won't work on older Macs without unified memory!
        id<MTLTexture> ct = [device newTextureWithDescriptor:curveDesc];
        [ct replaceRegion:MTLRegionMake2D(0, 0, cw, ch) mipmapLevel:0 withBytes:curvePixels bytesPerRow:sizeof(half) * cw * 4];
        curveTexture = ct;

        MTLTextureDescriptor *bandDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Uint
                                                                                            width:bw
                                                                                           height:bh
                                                                                        mipmapped:NO];
        bandDesc.usage = MTLTextureUsageShaderRead;
        bandDesc.storageMode = MTLStorageModeShared; // N.B. This won't work on older Macs without unified memory!
        id<MTLTexture> bt = [device newTextureWithDescriptor:bandDesc];
        [bt replaceRegion:MTLRegionMake2D(0, 0, bw, bh)
              mipmapLevel:0
                withBytes:bandPixels
              bytesPerRow:sizeof(uint16_t) * bw * 2];
        bandTexture = bt;
    }

    std::vector<QuadBezier> extractQuadBeziersFromPath(CGPathRef path) {
        CGFloat scale = 1.0;
        __block CGPoint subpathStart;
        __block CGPoint current;

        __block std::vector<QuadBezier> result;
        CGPathApplyWithBlock(path, ^(const CGPathElement *element) {
            switch (element->type) {
                case kCGPathElementMoveToPoint: {
                    CGPoint p = element->points[0];
                    subpathStart = CGPointMake(p.x * scale, p.y * scale);
                    current = subpathStart;
                    break;
                }
                case kCGPathElementAddLineToPoint: {
                    CGPoint p1 = CGPointMake(element->points[0].x * scale, element->points[0].y * scale);
                    CGPoint mid = CGPointMake((current.x + p1.x) * 0.5, (current.y + p1.y) * 0.5);
                    result.emplace_back(simd_float2 { (float)current.x, (float)current.y },
                                        simd_float2 { (float)mid.x, (float)mid.y },
                                        simd_float2 { (float)p1.x, (float)p1.y });
                    current = p1;
                    break;
                }
                case kCGPathElementAddQuadCurveToPoint: {
                    CGPoint ctrl = CGPointMake(element->points[0].x * scale, element->points[0].y * scale);
                    CGPoint end  = CGPointMake(element->points[1].x * scale, element->points[1].y * scale);
                    result.emplace_back(simd_float2 { (float)current.x, (float)current.y },
                                        simd_float2 { (float)ctrl.x, (float)ctrl.y },
                                        simd_float2 { (float)end.x, (float)end.y });
                    current = end;
                    break;
                }
                case kCGPathElementAddCurveToPoint: {
                    CGPoint c1 = CGPointMake(element->points[0].x * scale, element->points[0].y * scale);
                    CGPoint c2 = CGPointMake(element->points[1].x * scale, element->points[1].y * scale);
                    CGPoint ep = CGPointMake(element->points[2].x * scale, element->points[2].y * scale);
                    auto quads = cubicToQuadratic(current, c1, c2, ep);
                    result.push_back(quads[0]);
                    result.push_back(quads[1]);
                    current = ep;
                    break;
                }
                case kCGPathElementCloseSubpath:
                    if (distSq(current, subpathStart) > 1e-10) {
                        CGPoint mid = CGPointMake((current.x + subpathStart.x) * 0.5, (current.y + subpathStart.y) * 0.5);
                        result.emplace_back(simd_float2 { (float)current.x, (float)current.y },
                                            simd_float2 { (float)mid.x, (float)mid.y },
                                            simd_float2 { (float)subpathStart.x, (float)subpathStart.y });
                    }
                    current = subpathStart;
                    break;
            }
        });

        return result;
    }
};

} // end anonymous namespace

struct TextSubmesh {
    FontAtlas *atlas;
    int indexBufferOffset;
    int indexCount;
    MTLIndexType indexType;
};

struct TextMesh {
    id<MTLRenderPipelineState> renderPipeline;
    id<MTLBuffer> vertexBuffer;
    int vertexBufferOffset;
    id<MTLBuffer> indexBuffer;
    std::vector<TextSubmesh> submeshes;
    CGRect bounds;
};

class TextRenderer {
private:
    id<MTLDevice> device;
    id<MTLRenderPipelineState> renderPipeline;
    std::unordered_map<std::string, std::unique_ptr<FontAtlas>> fontAtlasCache;

public:
    TextRenderer(TextRendererDescriptor const& desc) {
        device = desc.device;
        if (device == nil) {
            device = MTLCreateSystemDefaultDevice();
        }
        makePipeline(desc);
    }

    FontAtlas *atlasForFontNamed(const char *fontName) {
        auto cached = fontAtlasCache.find(fontName);
        if (cached != fontAtlasCache.end()) {
            return cached->second.get();
        }
        auto atlas = std::make_unique<FontAtlas>(fontName, device);
        FontAtlas *atlasPtr = atlas.get();
        fontAtlasCache.insert({ std::string(fontName), std::move(atlas) });
        NSLog(@"Created glyph atlas for %s", fontName);
        return atlasPtr;
    }

private:
    bool metalPixelFormatHasStencilAspect(MTLPixelFormat format) {
        switch (format) {
            case MTLPixelFormatStencil8: [[fallthrough]];
            case MTLPixelFormatDepth24Unorm_Stencil8: [[fallthrough]];
            case MTLPixelFormatDepth32Float_Stencil8: [[fallthrough]];
            case MTLPixelFormatX32_Stencil8: [[fallthrough]];
            case MTLPixelFormatX24_Stencil8:
                return true;
            default:
                return false;
        }
    }
    bool makePipeline(TextRendererDescriptor const& textDesc) {
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (library == nil) { return false; }

        id<MTLFunction> vertexFunction = [library newFunctionWithName:@"glyph_vertex"];
        if (vertexFunction == nil) { return false; }
        id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"glyph_fragment"];
        if (fragmentFunction == nil) { return false; }

        MTLVertexDescriptor *vertexDesc = [MTLVertexDescriptor vertexDescriptor];
        int stride = sizeof(GlyphVertex); // 80

        // 5 float4 attributes, each at offset = attrib * 16
        for (int i = 0; i < 5; ++i) {
            vertexDesc.attributes[i].format = MTLVertexFormatFloat4;
            vertexDesc.attributes[i].offset = i * 16;
            vertexDesc.attributes[i].bufferIndex = 0;
        }
        vertexDesc.layouts[0].stride = stride;
        vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

        MTLRenderPipelineDescriptor *pipelineDesc = [MTLRenderPipelineDescriptor new];
        pipelineDesc.vertexFunction = vertexFunction;
        pipelineDesc.fragmentFunction = fragmentFunction;
        pipelineDesc.vertexDescriptor = vertexDesc;
        pipelineDesc.colorAttachments[0].pixelFormat = textDesc.colorPixelFormat;
        pipelineDesc.colorAttachments[0].blendingEnabled = YES;
        pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDesc.depthAttachmentPixelFormat = textDesc.depthStencilPixelFormat;
        if (metalPixelFormatHasStencilAspect(textDesc.depthStencilPixelFormat)) {
            pipelineDesc.stencilAttachmentPixelFormat = textDesc.depthStencilPixelFormat;
        }
        pipelineDesc.rasterSampleCount = textDesc.rasterSampleCount;

        NSError *error = nil;
        renderPipeline = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (renderPipeline == nil) {
            return false;
        }

        return true;
    }

public:
    TextMesh *makeTextMesh(CFAttributedStringRef string, CGSize maximumSize) {
        std::vector<GlyphVertex> vertices;
        std::vector<uint16_t> indices;
        std::vector<TextSubmesh> submeshes;

        CTFramesetterRef framesetter = CTFramesetterCreateWithAttributedString(string);
        CGSize suggestedSize = CTFramesetterSuggestFrameSizeWithConstraints(framesetter,
            CFRangeMake(0, 0), NULL, maximumSize, NULL);
        CGPathRef framePath = CGPathCreateWithRect(CGRectMake(0, 0, suggestedSize.width, suggestedSize.height), NULL);
        CTFrameRef frame = CTFramesetterCreateFrame(framesetter, CFRangeMake(0, 0), framePath, NULL);
        CFArrayRef lines = CTFrameGetLines(frame);

        CFIndex lineCount = CFArrayGetCount(lines);
        std::vector<CGPoint> lineOrigins(lineCount);
        if (lineCount > 0) {
            CTFrameGetLineOrigins(frame, CFRangeMake(0, 0), lineOrigins.data());
        }

        // Normalize so the first line's baseline is at Y=0
        CGFloat firstLineY = (lineCount > 0) ? lineOrigins[0].y : 0;

        float boundsMinX = FLT_MAX, boundsMinY = FLT_MAX;
        float boundsMaxX = -FLT_MAX, boundsMaxY = -FLT_MAX;

        for (CFIndex lineIdx = 0; lineIdx < lineCount; ++lineIdx) {
            CTLineRef line = (CTLineRef)CFArrayGetValueAtIndex(lines, lineIdx);
            CGPoint lineOrigin = lineOrigins[lineIdx];

            CFArrayRef runs = CTLineGetGlyphRuns(line);

            for (CFIndex runIdx = 0; runIdx < CFArrayGetCount(runs); ++runIdx) {
                CTRunRef run = (CTRunRef)CFArrayGetValueAtIndex(runs, runIdx);
                CFDictionaryRef attrs = CTRunGetAttributes(run);
                CTFontRef font = (CTFontRef)CFDictionaryGetValue(attrs, kCTFontAttributeName);

                CGFloat fontSize = font ? CTFontGetSize(font) : 1.0;

                CFStringRef fontName = CTFontCopyPostScriptName(font);
                char fontNameCstr[256];
                CFStringGetCString(fontName, fontNameCstr, 256, kCFStringEncodingUTF8);
                CFRelease(fontName);

                FontAtlas *atlas = atlasForFontNamed(fontNameCstr);

                CFIndex glyphCount = CTRunGetGlyphCount(run);
                std::vector<CGGlyph> glyphs(glyphCount);
                CTRunGetGlyphs(run, CFRangeMake(0, 0), glyphs.data());

                std::vector<CGPoint> positions(glyphCount);
                CTRunGetPositions(run, CFRangeMake(0, 0), positions.data());

                atlas->insertGlyphs({ glyphs.data(), (size_t)glyphCount });

                // Extract foreground color, converting to linear sRGB
                simd_float4 runColor = simd_make_float4(1, 1, 1, 1);
                CGColorRef fgColor = (CGColorRef)CFDictionaryGetValue(attrs, kCTForegroundColorAttributeName);
                if (fgColor) {
                    CGColorSpaceRef linearSRGB = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);
                    CGColorRef linearColor = CGColorCreateCopyByMatchingToColorSpace(linearSRGB,
                        kCGRenderingIntentDefault, fgColor, NULL);
                    if (linearColor) {
                        const CGFloat *c = CGColorGetComponents(linearColor);
                        size_t n = CGColorGetNumberOfComponents(linearColor);
                        if (n >= 4) {
                            runColor = simd_make_float4((float)c[0], (float)c[1], (float)c[2], (float)c[3]);
                        } else if (n >= 2) {
                            // Grayscale + alpha
                            runColor = simd_make_float4((float)c[0], (float)c[0], (float)c[0], (float)c[1]);
                        }
                        CGColorRelease(linearColor);
                    }
                    CGColorSpaceRelease(linearSRGB);
                }

                int submeshIndexStart = (int)indices.size();
                int submeshIndexCount = 0;
                for (CFIndex glyphIdx = 0; glyphIdx < glyphCount; ++glyphIdx) {
                    CGGlyph glyph = glyphs[glyphIdx];

                    GlyphInfo info = atlas->glyphInfoForGlyph(glyph);

                    // Skip glyphs with no visible outline (whitespace, etc.)
                    if (info.curveCount == 0) {
                        continue;
                    }

                    // Glyph position in the frame's coordinate space (points)
                    float posX = (float)(lineOrigin.x + positions[glyphIdx].x);
                    float posY = (float)(lineOrigin.y - firstLineY + positions[glyphIdx].y);

                    // Em-space glyph bbox with margin for dilation
                    const float margin = 0.02f;
                    float ex0 = info.xMin - margin;
                    float ex1 = info.xMax + margin;
                    float ey0 = info.yMin - margin;
                    float ey1 = info.yMax + margin;

                    // Point-space glyph bbox (scaled from em to points)
                    float fontScale = (float)fontSize;
                    float px0 = ex0 * fontScale;
                    float py0 = ey0 * fontScale;
                    float px1 = ex1 * fontScale;
                    float py1 = ey1 * fontScale;

                    boundsMinX = fmin(boundsMinX, posX + px0);
                    boundsMinY = fmin(boundsMinY, posY + py0);
                    boundsMaxX = fmax(boundsMaxX, posX + px1);
                    boundsMaxY = fmax(boundsMaxY, posY + py1);

                    // Packed glyph data for tex.z and tex.w
                    uint32_t glocX = static_cast<uint32_t>(info.bandTexX);
                    uint32_t glocY = static_cast<uint32_t>(info.bandTexY);
                    uint32_t texZPacked = glocX | (glocY << 16);
                    float texZ = std::bit_cast<float>(texZPacked);

                    uint32_t bmaxX = static_cast<uint32_t>(info.numVertBands  - 1);
                    uint32_t bmaxY = static_cast<uint32_t>(info.numHorizBands - 1);
                    uint32_t texWPacked = bmaxX | (bmaxY << 16);
                    float texW = std::bit_cast<float>(texWPacked);

                    simd_float4 band = simd_make_float4(info.bandScaleX, info.bandScaleY,
                                                        info.bandOffsetX, info.bandOffsetY);

                    // Inverse Jacobian: maps world-space (points) to em-space
                    float invScale = 1.0f / fontScale;
                    simd_float4 invJacobian = simd_make_float4(invScale, 0, 0, invScale);

                    // Four corners (BL, BR, TR, TL)
                    // px/py = point-space offsets for pos.xy, ex/ey = em-space for tex.xy
                    struct { float px, py, ex, ey; } corners[4] = {
                        { px0, py0, ex0, ey0 },
                        { px1, py0, ex1, ey0 },
                        { px1, py1, ex1, ey1 },
                        { px0, py1, ex0, ey1 },
                    };

                    int baseIndex = (int)vertices.size();

                    for (auto const& c : corners) {
                        simd_float2 norm = simd_normalize(simd_make_float2(c.ex, c.ey));
                        GlyphVertex vert {
                            simd_make_float4(posX + c.px, posY + c.py, norm.x, norm.y),
                            simd_make_float4(c.ex, c.ey, texZ, texW),
                            invJacobian,
                            band,
                            runColor,
                        };
                        vertices.push_back(vert);
                    }

                    // Two triangles: BL-BR-TR and BL-TR-TL
                    indices.push_back(baseIndex);
                    indices.push_back(baseIndex + 1);
                    indices.push_back(baseIndex + 2);
                    indices.push_back(baseIndex);
                    indices.push_back(baseIndex + 2);
                    indices.push_back(baseIndex + 3);
                    submeshIndexCount += 6;
                }

                if (submeshIndexCount > 0) {
                    submeshes.emplace_back(atlas,
                                           submeshIndexStart * (int)sizeof(uint16_t),
                                           submeshIndexCount,
                                           MTLIndexTypeUInt16);
                }
            }
        }
        CFRelease(frame);
        CFRelease(framePath);
        CFRelease(framesetter);

        CGRect bounds = CGRectZero;
        if (boundsMaxX > boundsMinX && boundsMaxY > boundsMinY) {
            bounds = CGRectMake(boundsMinX, boundsMinY, boundsMaxX - boundsMinX, boundsMaxY - boundsMinY);
        }

        int vertexBufferSize = sizeof(GlyphVertex) * (int)vertices.size();
        id<MTLBuffer> vertexBuffer = [device newBufferWithBytes:vertices.data()
                                                         length:vertexBufferSize
                                                        options:MTLResourceStorageModeShared];
        int indexBufferSize = sizeof(uint16_t) * (int)indices.size();
        id<MTLBuffer> indexBuffer = [device newBufferWithBytes:indices.data()
                                                        length:indexBufferSize
                                                       options:MTLResourceStorageModeShared];

        return new TextMesh { renderPipeline, vertexBuffer, 0, indexBuffer, submeshes, bounds };
    }
};

TextRendererRef TextRendererCreate(TextRendererDescriptor const *desc) {
    return new TextRenderer(*desc);
}

void TextRendererDestroy(TextRendererRef context) {
    if (context) {
        delete context;
    }
}

TextMeshRef TextMeshCreate(TextRendererRef context, const char *string, const char *fontName, CGSize maximumSize) {
    CFStringRef str = CFStringCreateWithCString(kCFAllocatorDefault, string, kCFStringEncodingUTF8);

    CFStringRef fontNameStr = CFStringCreateWithCString(kCFAllocatorDefault, fontName, kCFStringEncodingUTF8);
    CTFontRef font = CTFontCreateWithName(fontNameStr, 12.0, NULL);
    CFRelease(fontNameStr);

    CFMutableDictionaryRef attr = CFDictionaryCreateMutable(kCFAllocatorDefault, 1,
                                                            &kCFTypeDictionaryKeyCallBacks,
                                                            &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(attr, kCTFontAttributeName, font);

    CFAttributedStringRef attrString = CFAttributedStringCreate(kCFAllocatorDefault, str, attr);

    TextMesh *mesh = context->makeTextMesh(attrString, maximumSize);

    CFRelease(attrString);
    CFRelease(font);
    CFRelease(attr);
    CFRelease(str);

    return mesh;
}

TextMeshRef TextMeshCreateFromAttributedString(TextRendererRef context, CFAttributedStringRef str, CGSize maximumSize) {
    return context->makeTextMesh(str, maximumSize);
}

CGRect TextMeshGetBounds(TextMeshRef mesh) {
    return mesh->bounds;
}

void TextMeshRender(TextMeshRef mesh, TextViewConstants const *view, id<MTLRenderCommandEncoder> renderEncoder) {
    if (mesh->submeshes.empty()) {
        return;
    }
    [renderEncoder setRenderPipelineState:mesh->renderPipeline];
    [renderEncoder setVertexBuffer:mesh->vertexBuffer offset:mesh->vertexBufferOffset atIndex:0];
    [renderEncoder setVertexBytes:view length:sizeof(TextViewConstants) atIndex:1];
    [renderEncoder setCullMode:MTLCullModeNone];

    // Cache bound textures to avoid redundant bindings. We could probably do this
    // at a higher level to avoid redundant vertex buffer bindings.
    auto boundAtlas = mesh->submeshes[0].atlas;
    [renderEncoder setFragmentTexture:boundAtlas->curveTexture atIndex:0];
    [renderEncoder setFragmentTexture:boundAtlas->bandTexture atIndex:1];

    for (auto const& submesh : mesh->submeshes) {
        if (submesh.atlas != boundAtlas) {
            boundAtlas = submesh.atlas;
            [renderEncoder setFragmentTexture:boundAtlas->curveTexture atIndex:0];
            [renderEncoder setFragmentTexture:boundAtlas->bandTexture atIndex:1];
        }
        [renderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                  indexCount:submesh.indexCount
                                   indexType:submesh.indexType
                                 indexBuffer:mesh->indexBuffer
                           indexBufferOffset:submesh.indexBufferOffset];
    }
}

void TextMeshDestroy(TextMeshRef mesh) {
    if (mesh) {
        delete mesh;
    }
}

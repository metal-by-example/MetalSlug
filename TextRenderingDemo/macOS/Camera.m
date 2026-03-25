#import "Camera.h"

static simd_float4x4 lookAt(simd_float3 eye, simd_float3 center, simd_float3 up);
static simd_float4x4 perspectiveReverseZInfinite(float fovY, float aspect, float near);

@interface Camera ()
@property (nonatomic, assign) simd_float3 center; // look-at point
@property (nonatomic, assign) float distance;     // camera distance from center
@property (nonatomic, assign) float yaw;          // radians, around Y axis
@property (nonatomic, assign) float pitch;        // radians
@property (nonatomic, assign) float fovY;         // perspective vertical FOV
@end

@implementation Camera

- (instancetype)init {
    if (self = [super init]) {
        _center = simd_make_float3(0.0);
        _distance = 4;
        _fovY = M_PI_4;
    }
    return self;
}

- (simd_float4x4)viewMatrix {
    simd_float3 eye = self.eyePosition;
    return lookAt(eye, self.center, simd_make_float3(0, 1, 0));
}

- (simd_float4x4)projectionMatrixForAspectRatio:(float)aspectRatio {
    return perspectiveReverseZInfinite(self.fovY, aspectRatio, 0.01);
}

- (simd_float3)eyePosition {
    float cosP = cos(self.pitch), sinP = sin(self.pitch);
    float cosY = cos(self.yaw),   sinY = sin(self.yaw);
    simd_float3 dir = simd_make_float3(cosP * sinY, sinP, cosP * cosY);
    return self.center + dir * self.distance;
}

- (void)frameBoundsOfSize:(CGSize)size forAspectRatio:(float)aspectRatio {
    float tanHalfFov = tan(self.fovY * 0.5);
    float distForHeight = (float)(size.height * 0.5) / tanHalfFov;
    float distForWidth  = (float)(size.width  * 0.5) / (tanHalfFov * aspectRatio);
    self.distance = fmax(distForHeight, distForWidth) * 1.1; // 10% padding
}

- (void)scroll:(CGFloat)delta {
    self.distance = fmax(0.1, self.distance * (1.0 - delta * 0.01));
}

- (void)truck:(CGVector)delta {
    float cosP = cos(self.pitch), sinP = sin(self.pitch);
    float cosY = cos(self.yaw),   sinY = sin(self.yaw);

    // Camera-local axes from the current orientation
    simd_float3 fwd = simd_make_float3(-cosP * sinY, -sinP, -cosP * cosY);
    simd_float3 worldUp = simd_make_float3(0, 1, 0);
    simd_float3 right = simd_normalize(simd_cross(fwd, worldUp));
    simd_float3 up = simd_cross(right, fwd);

    // Scale by distance so the motion feels proportional at any zoom level
    float scale = self.distance * 0.002;
    self.center = self.center - right * (float)(delta.dx * scale) - up * (float)(delta.dy * scale);
}

- (void)rotate:(CGVector)delta {
    simd_float3 eye = self.eyePosition;

    // Direction from center toward eye (unit vector)
    float cosP = cos(self.pitch), sinP = sin(self.pitch);
    float cosY = cos(self.yaw),   sinY = sin(self.yaw);
    simd_float3 dir = simd_make_float3(cosP * sinY, sinP, cosP * cosY);

    // Cast the view ray (eye along -dir) to the XY plane.
    // Ray: P(t) = eye - t*dir;  Z=0 => t = eye.z / dir.z
    simd_float3 pivot = self.center;
    float pivotDist = self.distance;
    if (fabsf(dir.z) > 1e-6f) {
        float t = eye.z / dir.z;
        if (t > 0.0f) {
            pivot = eye - dir * t;
            pivotDist = t;
        }
    }

    // Update orientation
    self.yaw += delta.dx * -0.005;
    self.pitch = fmax(-M_PI / 2 + 0.01, fmin(M_PI / 2 - 0.01, self.pitch - delta.dy * 0.005));

    // Orbit the eye around the pivot at the same distance
    float cosP2 = cos(self.pitch), sinP2 = sin(self.pitch);
    float cosY2 = cos(self.yaw),   sinY2 = sin(self.yaw);
    simd_float3 newDir = simd_make_float3(cosP2 * sinY2, sinP2, cosP2 * cosY2);
    simd_float3 newEye = pivot + newDir * pivotDist;

    // Pin center to the Z=0 text plane so truck/scroll stay well-scaled.
    // Solve: (newEye - newDir * d).z = 0  =>  d = newEye.z / newDir.z
    if (fabsf(newDir.z) > 0.01f) {
        float d = newEye.z / newDir.z;
        if (d > 0.1f) {
            self.distance = d;
            self.center = newEye - newDir * d;
        }
    }
}

@end

simd_float4x4 lookAt(simd_float3 eye, simd_float3 center, simd_float3 up) {
    simd_float3 f = simd_normalize(center - eye);
    simd_float3 r = simd_normalize(simd_cross(f, up));
    simd_float3 u = simd_cross(r, f);
    return (simd_float4x4){
        {
            { r.x,  u.x, -f.x, 0 },
            { r.y,  u.y, -f.y, 0 },
            { r.z,  u.z, -f.z, 0 },
            { -simd_dot(r, eye), -simd_dot(u, eye), simd_dot(f, eye), 1 }
        }
    };
}

simd_float4x4 perspectiveReverseZInfinite(float fovY, float aspect, float near) {
    float ys = 1 / tanf(fovY * 0.5);
    float xs = ys / aspect;
    return (simd_float4x4){
        {
            { xs, 0,  0,    0 },
            { 0,  ys, 0,    0 },
            { 0,  0,  0,   -1 },
            { 0,  0,  near, 0 }
        }
    };
}

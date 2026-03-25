#import <Foundation/Foundation.h>
#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@interface Camera : NSObject

- (simd_float4x4)viewMatrix;
- (simd_float4x4)projectionMatrixForAspectRatio:(float)aspectRatio;
- (simd_float3)eyePosition;

- (void)frameBoundsOfSize:(CGSize)size forAspectRatio:(float)aspectRatio;
- (void)scroll:(CGFloat)delta;
- (void)truck:(CGVector)delta;
- (void)rotate:(CGVector)delta;

@end

NS_ASSUME_NONNULL_END

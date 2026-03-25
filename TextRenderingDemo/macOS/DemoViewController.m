#import "DemoViewController.h"

#import <MetalKit/MetalKit.h>

#import "Camera.h"
#import "TextMesh.h"

@interface InteractiveMTKView: MTKView
@property (nonatomic, strong) Camera *camera;
@property (nonatomic, assign) CGPoint lastDragPoint;
@end

@implementation InteractiveMTKView

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)scrollWheel:(NSEvent *)event {
    [self.camera scroll:event.scrollingDeltaY];
}

- (void)mouseDown:(NSEvent *)event {
    self.lastDragPoint = event.locationInWindow;
}

- (void)mouseDragged:(NSEvent *)event {
    CGPoint cur = event.locationInWindow;
    [self.camera truck:CGVectorMake(cur.x - _lastDragPoint.x, cur.y - _lastDragPoint.y)];
    self.lastDragPoint = cur;
}

- (void)mouseUp:(NSEvent *)event {
}

- (void)rightMouseDown:(NSEvent *)event {
    self.lastDragPoint = event.locationInWindow;
}

- (void)rightMouseDragged:(NSEvent *)event {
    CGPoint cur = event.locationInWindow;
    [self.camera rotate:CGVectorMake(cur.x - _lastDragPoint.x, cur.y - _lastDragPoint.y)];
    self.lastDragPoint = cur;
}

- (void)rightMouseUp:(NSEvent *)event {
}

@end

@interface DemoViewController () <MTKViewDelegate>
@property (nonatomic, strong) MTKView *metalView;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) Camera *camera;
@property (nonatomic, assign) TextRendererRef textContext;
@property (nonatomic, assign) TextMeshRef textMesh;
@end

static simd_float3 rgb_from_hue(float hue) {
    hue = fmod(hue, 1.0);
    float r = fmax(0.0f, fmin(fabsf(hue * 6 - 3) - 1, 1.0f));
    float g = fmax(0.0f, fmin(2 - fabsf(hue * 6 - 2), 1.0f));
    float b = fmax(0.0f, fmin(2 - fabsf(hue * 6 - 4), 1.0f));
    return (simd_float3){ r, g, b };
}

@implementation DemoViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    self.camera = [Camera new];

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    self.commandQueue = [device newCommandQueue];

    InteractiveMTKView *metalView = [[InteractiveMTKView alloc] initWithFrame:self.view.bounds device:device];
    metalView.camera = self.camera;

    self.metalView = metalView;
    self.metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.metalView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.metalView];

    self.metalView.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    self.metalView.clearColor = MTLClearColorMake(0.02, 0.02, 0.02, 1.0);
    self.metalView.sampleCount = 4;
    self.metalView.delegate = self;

    TextRendererDescriptor textDesc;
    textDesc.device = device;
    textDesc.colorPixelFormat = self.metalView.colorPixelFormat;
    textDesc.depthStencilPixelFormat = self.metalView.depthStencilPixelFormat;
    textDesc.rasterSampleCount = (int)self.metalView.sampleCount;
    self.textContext = TextRendererCreate(&textDesc);

    self.textMesh = TextMeshCreateFromAttributedString(self.textContext,
                                                       (__bridge CFAttributedStringRef)self.attributedDemoString,
                                                       CGSizeMake(CGFLOAT_MAX, CGFLOAT_MAX));

    // Set camera distance so all text is visible
    CGRect meshBounds = TextMeshGetBounds(self.textMesh);
    CGSize viewSize = self.view.bounds.size;
    float aspectRatio = (float)(viewSize.width / viewSize.height);
    [self.camera frameBoundsOfSize:meshBounds.size forAspectRatio:aspectRatio];
}

- (NSAttributedString *)attributedDemoString {
    NSFont *fancyFont = [NSFont fontWithName:@"Zapfino" size:12.0];
    NSMutableAttributedString *text = [[NSMutableAttributedString alloc] initWithString:@"This is Slug rendered with Metal\n"
                                                                             attributes:@{}];
    int codepointCount = (int)[text length];
    for (int i = 0; i < codepointCount; ++i) {
        float hue = (float)i / (codepointCount - 1);
        simd_float3 rgb = rgb_from_hue(hue);
        CGColorRef color = CGColorCreateSRGB(rgb[0], rgb[1], rgb[2], 1.0);
        NSDictionary *colorAttr = @{
            (__bridge id)kCTFontAttributeName : fancyFont,
            (__bridge id)kCTForegroundColorAttributeName : (__bridge id)color
        };
        [text setAttributes:colorAttr range:NSMakeRange(i, 1)];
        CGColorRelease(color);
    }

    NSFont *defaultFont = [NSFont fontWithName:@"HelveticaNeue" size:12.0];
    NSDictionary *defaultAttrs = @{
        (__bridge id)kCTFontAttributeName : defaultFont,
        (__bridge id)kCTForegroundColorAttributeName : (__bridge id)NSColor.whiteColor.CGColor
    };
    // Various translations of "I can eat glass..." from https://www.kermitproject.org/utf8.html#glass
    NSArray<NSString *> *phrases = @[
        @"我能吞下玻璃而不伤身体。\n",
        @"私はガラスを食べられます。それは私を傷つけません。\n",
        @"Μπορῶ νὰ φάω σπασμένα γυαλιὰ χωρὶς νὰ πάθω τίποτα.\n",
        @"ᛁᚳ᛫ᛗᚨᚷ᛫ᚷᛚᚨᛋ᛫ᛖᚩᛏᚪᚾ᛫ᚩᚾᛞ᛫ᚻᛁᛏ᛫ᚾᛖ᛫ᚻᛖᚪᚱᛗᛁᚪᚧ᛫ᛗᛖ᛬\n",
        @"زه شيشه خوړلې شم، هغه ما نه خوږوي\n",
        @".من می توانم بدونِ احساس درد شيشه بخورم\n",
        @"میں کانچ کھا سکتا ہوں اور مجھے تکلیف نہیں ہوتی ۔\n",
        @"Я могу есть стекло, оно мне не вредит.\n",
        @"𐌼𐌰𐌲 𐌲𐌻𐌴𐍃 𐌹̈𐍄𐌰𐌽, 𐌽𐌹 𐌼𐌹𐍃 𐍅𐌿 𐌽𐌳𐌰𐌽 𐌱𐍂𐌹𐌲𐌲𐌹𐌸.\n",
        @"Կրնամ ապակի ուտել և ինծի անհանգիստ չըներ։\n",
        @"എനിക്ക് ഗ്ലാസ് തിന്നാം. അതെന്നെ വേദനിപ്പിക്കില്ല.\n",
        @"මට වීදුරු කෑමට හැකියි. එයින් මට කිසි හානියක් සිදු නොවේ.\n",
        @"אני יכול לאכול זכוכית וזה לא מזיק לי.\n",
        @"ကျွန်တော် ကျွန်မ မှန်စားနိုင်တယ်။ ၎င်းကြောင့် ထိခိုက်မှုမရှိပါ။\n",
        @"ខ្ញុំអាចញុំកញ្ចក់បាន ដោយគ្មានបញ្ហារ\n",
        @"ຂອ້ຍກິນແກ້ວໄດ້ໂດຍທີ່ມັນບໍ່ໄດ້ເຮັດໃຫ້ຂອ້ຍເຈັບ.\n",
        @"ฉันกินกระจกได้ แต่มันไม่ทำให้ฉันเจ็บ\n",
        @"ཤེལ་སྒོ་ཟ་ནས་ང་ན་གི་མ་རེད།\n",
    ];

    for (NSString *phrase in phrases) {
        [text appendAttributedString:[[NSAttributedString alloc] initWithString:phrase attributes:defaultAttrs]];
    }

    return text;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];

    MTLRenderPassDescriptor *passDescriptor = view.currentRenderPassDescriptor;
    if (passDescriptor == nil) {
        return;
    }

    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];

    // Construct a model matrix that centers the text mesh about the origin
    CGRect meshBounds = TextMeshGetBounds(self.textMesh);
    float cx = CGRectGetMidX(meshBounds);
    float cy = CGRectGetMidY(meshBounds);
    simd_float4x4 modelMatrix = {{
        { 1, 0, 0, 0 },
        { 0, 1, 0, 0 },
        { 0, 0, 1, 0 },
        { -cx, -cy, 0, 1 },
    }};

    // Compute the rest of the model-view-projection transform
    CGSize drawableSize = self.metalView.drawableSize;
    float aspectRatio = (float)(drawableSize.width / drawableSize.height);
    simd_float4x4 projectionMatrix = [self.camera projectionMatrixForAspectRatio:aspectRatio];
    simd_float4x4 viewMatrix = self.camera.viewMatrix;
    simd_float4x4 viewProjectionMatrix = simd_mul(projectionMatrix, viewMatrix);
    simd_float4x4 modelViewProjectionMatrix = simd_mul(viewProjectionMatrix, modelMatrix);

    // Draw.
    TextViewConstants viewConstants = {
        .transform = modelViewProjectionMatrix,
        .viewportSize = simd_make_float2(drawableSize.width, drawableSize.height)
    };
    TextMeshRender(self.textMesh, &viewConstants, renderEncoder);

    [renderEncoder endEncoding];

    [commandBuffer presentDrawable:view.currentDrawable];

    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
}

@end

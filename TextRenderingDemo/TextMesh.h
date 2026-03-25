#import <Metal/Metal.h>
#import <simd/simd.h>

#ifdef __cplusplus
extern "C" {
#endif

/// An opaque reference to a rendering context.
typedef struct TextRenderer *TextRendererRef;

/// An opaque reference to a text mesh.
typedef struct TextMesh *TextMeshRef;

/// Configuration for creating a text rendering context.
///
/// The pixel formats and sample count must match those of the render pass in which
/// Slug text meshes will be drawn.
typedef struct {
    /// The Metal device used to create GPU resources.
    id<MTLDevice> device;
    /// The pixel format of the color attachment.
    MTLPixelFormat colorPixelFormat;
    /// The pixel format of the depth/stencil attachment, or ``MTLPixelFormatInvalid`` if none.
    MTLPixelFormat depthStencilPixelFormat;
    /// The number of samples per pixel for multisampled rendering.
    int rasterSampleCount;
} TextRendererDescriptor;

/// Per-frame constants passed to the text vertex shader.
typedef struct {
    /// The model-view-projection matrix.
    simd_float4x4 transform;
    /// The viewport dimensions in pixels.
    simd_float2 viewportSize;
} TextViewConstants;

/// Creates a new rendering context.
///
/// The returned context owns shared GPU resources (pipeline state, font atlas cache) and must be
/// released with ``TextRendererDestroy`` when no longer needed.
///
/// - Parameter desc: The descriptor specifying the Metal device and render pass configuration.
/// - Returns: A new context reference, or `NULL` if pipeline creation fails.
TextRendererRef TextRendererCreate(const TextRendererDescriptor *desc);

/// Destroys a Slug rendering context and releases its GPU resources.
///
/// - Parameter context: The context to destroy. Does nothing if `NULL`.
void TextRendererDestroy(TextRendererRef context);

/// Creates a text mesh from a plain C string and a PostScript font name.
///
/// The font is created at size 12.0 (em units). Glyph layout is performed by Core Text within
/// the area defined by `maximumSize`. Pass a large size (e.g. `CGFLOAT_MAX` for both dimensions)
/// to allow natural, unconstrained layout.
///
/// - Parameters:
///   - context: The text rendering context.
///   - string: A null-terminated UTF-8 string to render.
///   - fontName: A PostScript font name (e.g. `"HelveticaNeue"`).
///   - maximumSize: The maximum width and height available for text layout.
/// - Returns: A new text mesh reference, or `NULL` if the string is empty.
TextMeshRef TextMeshCreate(TextRendererRef context, const char *string, const char *fontName, CGSize maximumSize);

/// Creates a text mesh from a Core Foundation attributed string.
///
/// The attributed string should have ``kCTFontAttributeName`` set on each run. Runs may use
/// different fonts and sizes. Color is read from ``kCTForegroundColorAttributeName`` and
/// converted to linear sRGB; runs without a color attribute default to opaque white.
///
/// - Parameters:
///   - context: The text rendering context.
///   - str: A `CFAttributedStringRef` describing the styled text to render.
///   - maximumSize: The maximum width and height available for text layout.
/// - Returns: A new text mesh reference, or `NULL` if the string is empty.
TextMeshRef TextMeshCreateFromAttributedString(TextRendererRef context, CFAttributedStringRef str, CGSize maximumSize);

/// Returns the bounding rectangle of the text mesh in its local coordinate space.
///
/// The bounds enclose all glyph quads (including dilation margins). The coordinate space is
/// in typographic points, matching the positions produced by Core Text.
///
/// - Parameter mesh: The text mesh to query.
/// - Returns: The bounding rectangle, or ``CGRectZero`` if the mesh contains no visible glyphs.
CGRect TextMeshGetBounds(TextMeshRef mesh);

/// Encodes draw commands for a text mesh into a render command encoder.
///
/// The caller is responsible for creating the command encoder and ending encoding afterward.
/// This method sets the render pipeline state, vertex buffer, and fragment textures on the
/// encoder; some of this state (notably the cull mode) is not restored by this call.
///
/// - Parameters:
///   - mesh: The text mesh to render.
///   - view: A pointer to the per-frame view constants (transform and viewport size).
///   - renderEncoder: The render command encoder to encode draw calls into.
void TextMeshRender(TextMeshRef mesh, const TextViewConstants *view, id<MTLRenderCommandEncoder> renderEncoder);

/// Destroys a text mesh and releases its GPU buffers.
///
/// - Parameter mesh: The text mesh to destroy. Does nothing if `NULL`.
void TextMeshDestroy(TextMeshRef mesh);

#ifdef __cplusplus
}
#endif

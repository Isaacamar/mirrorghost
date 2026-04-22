import ARKit
import UIKit
import CoreImage

enum FaceIsolator {

    // CIContext is thread-safe but doesn't conform to Sendable — nonisolated(unsafe) is correct here.
    nonisolated(unsafe) private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Isolation mask (512×512, white = face region)

    static func mask(
        geometry: ARFaceGeometry,
        screenVertices: [CGPoint],
        canvasSize: CGSize
    ) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: canvasSize)
        return renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(origin: .zero, size: canvasSize))

            let cg = ctx.cgContext
            cg.setFillColor(UIColor.white.cgColor)

            // Fill every triangle of the face mesh — more accurate than convex hull
            // because the mesh tracks concavities at temples and under chin.
            let indices = geometry.triangleIndices   // UnsafeBufferPointer<Int16>
            for t in 0..<geometry.triangleCount {
                let base = t * 3
                cg.move(to:     screenVertices[Int(indices[base])])
                cg.addLine(to:  screenVertices[Int(indices[base + 1])])
                cg.addLine(to:  screenVertices[Int(indices[base + 2])])
                cg.closePath()
            }
            cg.fillPath()
        }
    }

    // MARK: - Eye-aligned face crop

    // Projects both eye centers using ARKit camera projection, then crops a square
    // region sized 2.8× the inter-ocular distance, with the eye-line sitting ~38%
    // from the top of the crop.  This is the same alignment convention InsightFace
    // uses for its ArcFace embeddings.
    //
    // Returns nil when the eyes are too close together in screen space (extreme
    // head angle, face very close to lens, etc.).
    //
    // Orientation note: capturedImage is always a landscape pixel buffer from the
    // front TrueDepth sensor.  We orient it with CGImagePropertyOrientation.right
    // (90° CCW display rotation) to get portrait-upright pixels.  If the crop
    // appears rotated on a specific device, change .right to .left here.

    static func alignedCrop(anchor: ARFaceAnchor, frame: ARFrame) -> UIImage? {
        let pb   = frame.capturedImage
        let bufW = CGFloat(CVPixelBufferGetWidth(pb))    // landscape buffer width  (e.g. 1440)
        let bufH = CGFloat(CVPixelBufferGetHeight(pb))   // landscape buffer height (e.g. 1080)

        // Portrait viewport: landscape W and H are swapped
        let viewport = CGSize(width: bufH, height: bufW)   // e.g. 1080×1440

        let leftWorld  = extractTranslation(anchor.leftEyeTransform)
        let rightWorld = extractTranslation(anchor.rightEyeTransform)

        let leftPt  = frame.camera.projectPoint(leftWorld,  orientation: .portrait, viewportSize: viewport)
        let rightPt = frame.camera.projectPoint(rightWorld, orientation: .portrait, viewportSize: viewport)

        let eyeDist = hypot(rightPt.x - leftPt.x, rightPt.y - leftPt.y)
        guard eyeDist > 20 else { return nil }

        let midX     = (leftPt.x + rightPt.x) / 2
        let midY     = (leftPt.y + rightPt.y) / 2
        let cropSide = eyeDist * 2.8
        let originX  = midX - cropSide / 2
        let originY  = midY - cropSide * 0.38     // eyes ~38% from top

        // capturedImage is landscape; .oriented(.right) rotates 90° CCW → portrait upright.
        // After orientation, CIImage extent = (0, 0, bufH, bufW).
        // CIImage Y is bottom-up; projectPoint Y is top-down — flip accordingly.
        let ciBase     = CIImage(cvPixelBuffer: pb)
        let ciPortrait = ciBase.oriented(.right)

        let ciY      = bufW - (originY + cropSide)          // top-down → bottom-up
        let cropRect = CGRect(x: originX, y: ciY, width: cropSide, height: cropSide)
        let validRect = cropRect.intersection(ciPortrait.extent)
        guard validRect.width > 20 && validRect.height > 20 else { return nil }

        let scale  = 512.0 / validRect.width
        let output = ciPortrait
            .cropped(to: validRect)
            .transformed(by: CGAffineTransform(translationX: -validRect.minX,
                                               y:            -validRect.minY))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        guard let cg = ciContext.createCGImage(
            output,
            from: CGRect(x: 0, y: 0, width: 512, height: 512)
        ) else { return nil }

        return UIImage(cgImage: cg)
    }

    // MARK: - Helpers

    private static func extractTranslation(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }
}

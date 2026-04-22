import ARKit
import UIKit
import simd

enum FaceMeshRenderer {

    struct Output {
        let image: UIImage
        let screenVertices: [CGPoint]   // 1,220 points projected onto the output canvas
    }

    static func render(
        geometry: ARFaceGeometry,
        transform: simd_float4x4,
        camera: ARCamera,
        outputSize: CGSize = CGSize(width: 512, height: 512)
    ) -> Output {

        // Project all 1,220 vertices: face-local → world → viewport
        let verts  = geometry.vertices    // UnsafeBufferPointer<simd_float3>
        var screen = [CGPoint](repeating: .zero, count: verts.count)
        for i in 0..<verts.count {
            let v      = verts[i]
            let world4 = transform * simd_float4(v.x, v.y, v.z, 1.0)
            let world3 = SIMD3<Float>(world4.x, world4.y, world4.z)
            let p = camera.projectPoint(world3, orientation: .portrait, viewportSize: outputSize)
            screen[i] = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
        }

        let renderer = UIGraphicsImageRenderer(size: outputSize)
        let image = renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(origin: .zero, size: outputSize))

            let cg      = ctx.cgContext
            let indices = geometry.triangleIndices   // UnsafeBufferPointer<Int16>

            cg.setStrokeColor(UIColor(white: 0.35, alpha: 1.0).cgColor)
            cg.setLineWidth(0.5)
            for t in 0..<geometry.triangleCount {
                let base = t * 3
                let p0 = screen[Int(indices[base])]
                let p1 = screen[Int(indices[base + 1])]
                let p2 = screen[Int(indices[base + 2])]
                cg.move(to: p0); cg.addLine(to: p1)
                cg.move(to: p1); cg.addLine(to: p2)
                cg.move(to: p2); cg.addLine(to: p0)
            }
            cg.strokePath()

            cg.setFillColor(UIColor(white: 0.9, alpha: 0.8).cgColor)
            let r: CGFloat = 1.5
            for p in screen {
                cg.fillEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
            }
        }

        return Output(image: image, screenVertices: screen)
    }
}

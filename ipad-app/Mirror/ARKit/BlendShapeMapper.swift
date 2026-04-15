import ARKit

enum BlendShapeMapper {
    /// Extract all 52 ARKit blend shapes as [String: Float].
    /// Returns keys matching the JSON protocol (camelCase, no "AR" prefix).
    static func extract(from anchor: ARFaceAnchor) -> [String: Float] {
        var result: [String: Float] = [:]
        for (key, value) in anchor.blendShapes {
            result[key.rawValue] = value.floatValue
        }
        return result
    }
}

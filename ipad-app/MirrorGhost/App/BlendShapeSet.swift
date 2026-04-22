import ARKit

struct BlendShapeSet {
    // Brow
    var browDownLeft:     Float = 0
    var browDownRight:    Float = 0
    var browInnerUp:      Float = 0
    var browOuterUpLeft:  Float = 0
    var browOuterUpRight: Float = 0
    // Cheek
    var cheekPuff:        Float = 0
    var cheekSquintLeft:  Float = 0
    var cheekSquintRight: Float = 0
    // Eye
    var eyeBlinkLeft:     Float = 0
    var eyeBlinkRight:    Float = 0
    var eyeLookDownLeft:  Float = 0
    var eyeLookDownRight: Float = 0
    var eyeLookInLeft:    Float = 0
    var eyeLookInRight:   Float = 0
    var eyeLookOutLeft:   Float = 0
    var eyeLookOutRight:  Float = 0
    var eyeLookUpLeft:    Float = 0
    var eyeLookUpRight:   Float = 0
    var eyeSquintLeft:    Float = 0
    var eyeSquintRight:   Float = 0
    var eyeWideLeft:      Float = 0
    var eyeWideRight:     Float = 0
    // Jaw
    var jawForward:       Float = 0
    var jawLeft:          Float = 0
    var jawRight:         Float = 0
    var jawOpen:          Float = 0
    // Mouth
    var mouthClose:          Float = 0
    var mouthDimpleLeft:     Float = 0
    var mouthDimpleRight:    Float = 0
    var mouthFrownLeft:      Float = 0
    var mouthFrownRight:     Float = 0
    var mouthFunnel:         Float = 0
    var mouthLeft:           Float = 0
    var mouthRight:          Float = 0
    var mouthLowerDownLeft:  Float = 0
    var mouthLowerDownRight: Float = 0
    var mouthPressLeft:      Float = 0
    var mouthPressRight:     Float = 0
    var mouthPucker:         Float = 0
    var mouthRollLower:      Float = 0
    var mouthRollUpper:      Float = 0
    var mouthShrugLower:     Float = 0
    var mouthShrugUpper:     Float = 0
    var mouthSmileLeft:      Float = 0
    var mouthSmileRight:     Float = 0
    var mouthStretchLeft:    Float = 0
    var mouthStretchRight:   Float = 0
    var mouthUpperUpLeft:    Float = 0
    var mouthUpperUpRight:   Float = 0
    // Nose
    var noseSneerLeft:  Float = 0
    var noseSneerRight: Float = 0
    // Tongue
    var tongueOut: Float = 0

    init(from anchor: ARFaceAnchor) {
        let s = anchor.blendShapes
        func v(_ k: ARFaceAnchor.BlendShapeLocation) -> Float { s[k]?.floatValue ?? 0 }
        browDownLeft     = v(.browDownLeft);     browDownRight    = v(.browDownRight)
        browInnerUp      = v(.browInnerUp);      browOuterUpLeft  = v(.browOuterUpLeft)
        browOuterUpRight = v(.browOuterUpRight)
        cheekPuff        = v(.cheekPuff);        cheekSquintLeft  = v(.cheekSquintLeft)
        cheekSquintRight = v(.cheekSquintRight)
        eyeBlinkLeft     = v(.eyeBlinkLeft);     eyeBlinkRight    = v(.eyeBlinkRight)
        eyeLookDownLeft  = v(.eyeLookDownLeft);  eyeLookDownRight = v(.eyeLookDownRight)
        eyeLookInLeft    = v(.eyeLookInLeft);    eyeLookInRight   = v(.eyeLookInRight)
        eyeLookOutLeft   = v(.eyeLookOutLeft);   eyeLookOutRight  = v(.eyeLookOutRight)
        eyeLookUpLeft    = v(.eyeLookUpLeft);    eyeLookUpRight   = v(.eyeLookUpRight)
        eyeSquintLeft    = v(.eyeSquintLeft);    eyeSquintRight   = v(.eyeSquintRight)
        eyeWideLeft      = v(.eyeWideLeft);      eyeWideRight     = v(.eyeWideRight)
        jawForward       = v(.jawForward);       jawLeft          = v(.jawLeft)
        jawRight         = v(.jawRight);         jawOpen          = v(.jawOpen)
        mouthClose          = v(.mouthClose);       mouthDimpleLeft    = v(.mouthDimpleLeft)
        mouthDimpleRight    = v(.mouthDimpleRight); mouthFrownLeft     = v(.mouthFrownLeft)
        mouthFrownRight     = v(.mouthFrownRight);  mouthFunnel        = v(.mouthFunnel)
        mouthLeft           = v(.mouthLeft);        mouthRight         = v(.mouthRight)
        mouthLowerDownLeft  = v(.mouthLowerDownLeft); mouthLowerDownRight = v(.mouthLowerDownRight)
        mouthPressLeft      = v(.mouthPressLeft);   mouthPressRight    = v(.mouthPressRight)
        mouthPucker         = v(.mouthPucker);      mouthRollLower     = v(.mouthRollLower)
        mouthRollUpper      = v(.mouthRollUpper);   mouthShrugLower    = v(.mouthShrugLower)
        mouthShrugUpper     = v(.mouthShrugUpper);  mouthSmileLeft     = v(.mouthSmileLeft)
        mouthSmileRight     = v(.mouthSmileRight);  mouthStretchLeft   = v(.mouthStretchLeft)
        mouthStretchRight   = v(.mouthStretchRight); mouthUpperUpLeft  = v(.mouthUpperUpLeft)
        mouthUpperUpRight   = v(.mouthUpperUpRight)
        noseSneerLeft  = v(.noseSneerLeft);  noseSneerRight = v(.noseSneerRight)
        tongueOut      = v(.tongueOut)
    }

    // Returns all 52 as (name, value) sorted descending — used for debug display.
    func sorted() -> [(name: String, value: Float)] {
        allPairs.sorted { $0.value > $1.value }
    }

    func top(_ n: Int) -> [(name: String, value: Float)] {
        Array(sorted().prefix(n))
    }

    private var allPairs: [(name: String, value: Float)] {[
        ("browDownLeft", browDownLeft), ("browDownRight", browDownRight),
        ("browInnerUp", browInnerUp), ("browOuterUpLeft", browOuterUpLeft),
        ("browOuterUpRight", browOuterUpRight), ("cheekPuff", cheekPuff),
        ("cheekSquintLeft", cheekSquintLeft), ("cheekSquintRight", cheekSquintRight),
        ("eyeBlinkLeft", eyeBlinkLeft), ("eyeBlinkRight", eyeBlinkRight),
        ("eyeLookDownLeft", eyeLookDownLeft), ("eyeLookDownRight", eyeLookDownRight),
        ("eyeLookInLeft", eyeLookInLeft), ("eyeLookInRight", eyeLookInRight),
        ("eyeLookOutLeft", eyeLookOutLeft), ("eyeLookOutRight", eyeLookOutRight),
        ("eyeLookUpLeft", eyeLookUpLeft), ("eyeLookUpRight", eyeLookUpRight),
        ("eyeSquintLeft", eyeSquintLeft), ("eyeSquintRight", eyeSquintRight),
        ("eyeWideLeft", eyeWideLeft), ("eyeWideRight", eyeWideRight),
        ("jawForward", jawForward), ("jawLeft", jawLeft),
        ("jawRight", jawRight), ("jawOpen", jawOpen),
        ("mouthClose", mouthClose), ("mouthDimpleLeft", mouthDimpleLeft),
        ("mouthDimpleRight", mouthDimpleRight), ("mouthFrownLeft", mouthFrownLeft),
        ("mouthFrownRight", mouthFrownRight), ("mouthFunnel", mouthFunnel),
        ("mouthLeft", mouthLeft), ("mouthRight", mouthRight),
        ("mouthLowerDownLeft", mouthLowerDownLeft), ("mouthLowerDownRight", mouthLowerDownRight),
        ("mouthPressLeft", mouthPressLeft), ("mouthPressRight", mouthPressRight),
        ("mouthPucker", mouthPucker), ("mouthRollLower", mouthRollLower),
        ("mouthRollUpper", mouthRollUpper), ("mouthShrugLower", mouthShrugLower),
        ("mouthShrugUpper", mouthShrugUpper), ("mouthSmileLeft", mouthSmileLeft),
        ("mouthSmileRight", mouthSmileRight), ("mouthStretchLeft", mouthStretchLeft),
        ("mouthStretchRight", mouthStretchRight), ("mouthUpperUpLeft", mouthUpperUpLeft),
        ("mouthUpperUpRight", mouthUpperUpRight), ("noseSneerLeft", noseSneerLeft),
        ("noseSneerRight", noseSneerRight), ("tongueOut", tongueOut)
    ]}
}

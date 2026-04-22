import SwiftUI

struct FaceDisplayView: View {
    let image: UIImage?

    var body: some View {
        GeometryReader { geo in
            if let img = image {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
            } else {
                Color.black
            }
        }
        .ignoresSafeArea()
    }
}

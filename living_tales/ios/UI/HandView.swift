import SwiftUI

struct HandView: View {
    let hand: [UnknownToken]

    var body: some View {
        HStack(spacing: 8) {
            ForEach(hand.indices, id: \.self) { idx in
                let token = hand[idx].asToken(nDims: 3)
                TokenCardView(token: token)
            }
        }
    }
}

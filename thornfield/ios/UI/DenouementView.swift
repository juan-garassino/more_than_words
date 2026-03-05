import SwiftUI

struct DenouementView: View {
    let tokens: [Token]

    var body: some View {
        VStack(spacing: 12) {
            ForEach(tokens, id: \.id) { token in
                TokenCardView(token: token)
            }
        }
    }
}

import SwiftUI

struct TriadView: View {
    let triad: Triad

    var body: some View {
        HStack(spacing: 6) {
            ForEach(triad.tokens, id: \.id) { token in
                TokenCardView(token: token)
            }
        }
    }
}

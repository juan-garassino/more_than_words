import SwiftUI

struct WorldView: View {
    @ObservedObject var viewModel: GameViewModel
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(viewModel.world.currentLocation)
                .font(Theme.Fonts.caption)
                .foregroundColor(.gray)
                .padding(.bottom, 4)
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(viewModel.world.visibleTokens, id: \.self) { token in
                        Text(token)
                            .font(Theme.Fonts.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(4)
                            .overlay(
                                RoundedRectangle(cornerRadius: 4)
                                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
                            )
                    }
                }
            }
        }
        .padding()
        // Fog Visualization based on Lambda
        .overlay(
            LinearGradient(
                gradient: Gradient(colors: [.black.opacity(viewModel.world.lambda), .clear]),
                startPoint: .top,
                endPoint: .bottom
            )
            .allowsHitTesting(false)
        )
    }
}

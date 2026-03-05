import SwiftUI

struct PetView: View {
    @ObservedObject var viewModel: GameViewModel
    
    var body: some View {
        ZStack {
            // Placeholder for the "Creature"
            // In the future this would be a Shader or Lottie animation
            VStack {
                Text(getEmojiForState())
                    .font(.system(size: 80))
                    .shadow(color: Theme.Colors.neonBlue, radius: 10)
                    .scaleEffect(viewModel.pet.isAsleep ? 0.9 : 1.0)
                    .animation(Animation.easeInOut(duration: 2).repeatForever(autoreverses: true), value: viewModel.pet.isAsleep)
                
                if let suggestion = viewModel.aiSuggestion {
                     Text(suggestion)
                        .font(Theme.Fonts.caption)
                        .padding(8)
                        .background(Theme.Colors.neonPink.opacity(0.2))
                        .cornerRadius(8)
                        .transition(.scale.combined(with: .opacity))
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    func getEmojiForState() -> String {
        if viewModel.pet.health < 0.3 { return "🤢" }
        if viewModel.pet.isAsleep { return "😴" }
        if viewModel.pet.happiness < 0.3 { return "😢" }
        if viewModel.pet.hunger > 0.7 { return "🤤" }
        if viewModel.pet.dirt > 0.7 { return "💩" }
        return "👾" // Default happy alien
    }
}

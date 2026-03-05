import SwiftUI

struct ActionsView: View {
    @ObservedObject var viewModel: GameViewModel
    
    let columns = [
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible())
    ]
    
    var body: some View {
        VStack {
            // Interaction Grid
            LazyVGrid(columns: columns, spacing: 12) {
                ActionButton(label: "FEED", icon: "fork.knife", color: Theme.Colors.neonGreen) {
                    viewModel.performAction("FEED")
                }
                ActionButton(label: "CLEAN", icon: "soap", color: Theme.Colors.neonBlue) {
                    viewModel.performAction("CLEAN")
                }
                ActionButton(label: "PLAY", icon: "gamecontroller", color: Theme.Colors.neonPink) {
                    viewModel.performAction("PLAY")
                }
                ActionButton(label: "SLEEP", icon: "bed.double", color: .purple) {
                    viewModel.performAction(viewModel.pet.isAsleep ? "WAKE" : "SLEEP")
                }
                ActionButton(label: "MEDS", icon: "cross.case", color: .red) {
                    viewModel.performAction("MEDICINE")
                }
                ActionButton(label: "WAIT", icon: "hourglass", color: .gray) {
                    viewModel.performAction("WAIT")
                }
            }
            .padding()
            
            // Movement Controls
            HStack(spacing: 20) {
                Button(action: { viewModel.move(.west) }) {
                    Image(systemName: "arrow.left")
                }
                VStack(spacing: 8) {
                    Button(action: { viewModel.move(.north) }) {
                        Image(systemName: "arrow.up")
                    }
                    Button(action: { viewModel.move(.south) }) {
                        Image(systemName: "arrow.down")
                    }
                }
                Button(action: { viewModel.move(.east) }) {
                    Image(systemName: "arrow.right")
                }
            }
            .font(.title2)
            .foregroundColor(.white)
            .padding(.top, 10)
        }
        .glassmorphic()
    }
}

struct ActionButton: View {
    var label: String
    var icon: String
    var color: Color
    var action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack {
                Image(systemName: icon)
                    .font(.system(size: 20))
                Text(label)
                    .font(Theme.Fonts.caption)
                    .fontWeight(.bold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(color.opacity(0.2))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(color.opacity(0.8), lineWidth: 1)
            )
            .cornerRadius(8)
            .foregroundColor(.white)
        }
    }
}

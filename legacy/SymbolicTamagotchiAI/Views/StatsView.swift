import SwiftUI

struct StatsView: View {
    @ObservedObject var viewModel: GameViewModel
    
    var body: some View {
        VStack(spacing: 8) {
            // Lambda Meter (The "Ambiguity" of the world)
            HStack {
                Text("λ")
                    .font(Theme.Fonts.title)
                    .foregroundColor(Theme.Colors.neonPink)
                
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .frame(width: geometry.size.width, height: 10)
                            .opacity(0.3)
                            .foregroundColor(Color.gray)
                        
                        Rectangle()
                            .frame(width: min(CGFloat(self.viewModel.world.lambda) * geometry.size.width, geometry.size.width), height: 10)
                            .foregroundColor(Theme.Colors.neonPink)
                            .animation(.linear, value: viewModel.world.lambda)
                    }
                }
                .frame(height: 10)
                .cornerRadius(5)
                
                Text(String(format: "%.2f", viewModel.world.lambda))
                    .font(Theme.Fonts.caption)
                    .foregroundColor(.white)
            }
            .padding(.bottom, 8)
            
            // Standard Stats
            StatRow(label: "HUNGER", value: viewModel.pet.hunger, inverted: true)
            StatRow(label: "DIRT", value: viewModel.pet.dirt, inverted: true)
            StatRow(label: "ENERGY", value: 1.0 - viewModel.pet.tiredness, inverted: false) // Display as Energy (inverse of Tiredness)
            StatRow(label: "HAPPY", value: viewModel.pet.happiness, inverted: false)
            StatRow(label: "HEALTH", value: viewModel.pet.health, inverted: false)
        }
        .padding()
        .glassmorphic()
    }
}

struct StatRow: View {
    var label: String
    var value: Double
    var inverted: Bool
    
    var body: some View {
        HStack {
            Text(label)
                .font(Theme.Fonts.caption)
                .frame(width: 60, alignment: .leading)
                .foregroundColor(Theme.Colors.secondaryText)
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .frame(width: geometry.size.width, height: 6)
                        .opacity(0.3)
                        .foregroundColor(Color.gray)
                    
                    Rectangle()
                        .frame(width: min(CGFloat(self.value) * geometry.size.width, geometry.size.width), height: 6)
                        .foregroundColor(Theme.Colors.Colors.forStat(value, inverted: inverted))
                        .animation(.linear, value: value)
                }
            }
            .frame(height: 6)
            .cornerRadius(3)
        }
    }
}

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var viewModel: GameViewModel
    
    var body: some View {
        ZStack {
            // Background
            Theme.Colors.background
                .ignoresSafeArea()
            
            // Dynamic Background based on Low Health/High Lambda
            if viewModel.pet.isSick {
                Color.red.opacity(0.1).ignoresSafeArea()
                    .blendMode(.multiply)
            }
            
            VStack {
                // Top: Header & Stats
                Text("MORE THAN WORDS")
                    .font(Theme.Fonts.title)
                    .foregroundColor(Theme.Colors.neonBlue)
                    .padding(.top, 40)
                
                StatsView(viewModel: viewModel)
                    .padding(.top, 10)
                
                // Middle: World & Pet
                ZStack {
                   WorldView(viewModel: viewModel)
                        .zIndex(0) // Behind pet
                    
                   PetView(viewModel: viewModel)
                        .zIndex(1)
                }
                .frame(maxHeight: .infinity)
                
                // Bottom: Actions
                ActionsView(viewModel: viewModel)
                    .padding(.bottom)
            }
        }
        .preferredColorScheme(.dark)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(GameViewModel())
    }
}

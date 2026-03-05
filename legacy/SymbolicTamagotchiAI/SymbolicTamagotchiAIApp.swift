import SwiftUI

@main
struct SymbolicTamagotchiAIApp: App {
    // Initialize the persistence controller.
    // For now, we use the shared instance.
    let persistenceController = PersistenceController.shared
    
    // Initialize the GameViewModel as a StateObject to own the source of truth.
    // In a real app, this might be injected or created here.
    @StateObject private var gameViewModel = GameViewModel()
    
    // Handle scene phases for saving state when backgrounded.
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
                .environmentObject(gameViewModel)
        }
        .onChange(of: scenePhase) { newPhase in
            switch newPhase {
            case .background, .inactive:
                // Save game state when app goes to background
                gameViewModel.saveState()
                persistenceController.save()
            default:
                break
            }
        }
    }
}

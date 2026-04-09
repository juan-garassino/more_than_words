import Foundation

struct CasebookPlacement: Identifiable {
    let id = UUID()
    let triad: Triad
    let position: GridPosition
}

import SwiftUI

struct Theme {
    // MARK: - Colors
    struct Colors {
        static let background = Color("BackgroundColor") // Need to define in Assets usually, fallback to system
        static let primaryText = Color.white
        static let secondaryText = Color.gray
        static let accent = Color.cyan
        static let danger = Color.red
        static let warning = Color.orange
        static let success = Color.green
        
        static let neonPink = Color(red: 1.0, green: 0.0, blue: 0.8)
        static let neonBlue = Color(red: 0.0, green: 0.8, blue: 1.0)
        static let neonGreen = Color(red: 0.0, green: 1.0, blue: 0.4)
        
        // Semantic colors for stats
        static func forStat(_ value: Double, inverted: Bool = false) -> Color {
            let normalized = max(0, min(1, value))
            if inverted {
                // High is bad (e.g. Hunger)
                return normalized > 0.7 ? danger : (normalized > 0.4 ? warning : success)
            } else {
                // High is good (e.g. Health)
                return normalized < 0.3 ? danger : (normalized < 0.6 ? warning : success)
            }
        }
    }
    
    // MARK: - Fonts
    struct Fonts {
        static let title = Font.system(size: 24, weight: .bold, design: .monospaced)
        static let body = Font.system(size: 16, weight: .regular, design: .monospaced)
        static let caption = Font.system(size: 12, weight: .light, design: .monospaced)
        static let pixel = Font.custom("Courier New", size: 14) // Fallback for pixel style
    }
    
    // MARK: - Layout
    struct Layout {
        static let padding: CGFloat = 16
        static let cornerRadius: CGFloat = 12
        static let iconSize: CGFloat = 24
    }
    
    // MARK: - Modifiers
    struct Glassmorphism: ViewModifier {
        func body(content: Content) -> some View {
            content
                .background(.ultraThinMaterial)
                .cornerRadius(Layout.cornerRadius)
                .shadow(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 2)
        }
    }
}

extension View {
    func glassmorphic() -> some View {
        self.modifier(Theme.Glassmorphism())
    }
}

import SwiftUI

struct ConvergenceMeterView: View {
    let dimensions: [Float]

    var body: some View {
        HStack(spacing: 6) {
            ForEach(dimensions.indices, id: \.self) { idx in
                let value = Double(dimensions[idx])
                Rectangle()
                    .fill(LinearGradient(
                        colors: [Color(red: 0.55, green: 0.41, blue: 0.08), Color(red: 0.83, green: 0.63, blue: 0.09)],
                        startPoint: .leading,
                        endPoint: .trailing
                    ))
                    .frame(height: 4)
                    .opacity(value > 0.0 ? value : 0.0)
                    .cornerRadius(2)
            }
        }
        .padding(.horizontal, 8)
        .opacity(dimensions.count > 0 ? 1.0 : 0.0)
    }
}

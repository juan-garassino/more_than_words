import SwiftUI

struct TokenCardView: View {
    let token: Token

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(token.tokenClass.rawValue.prefix(1))
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(Color(red: 0.8, green: 0.75, blue: 0.65))
                Spacer()
                Text(token.id)
                    .font(.system(size: 9, weight: .regular, design: .monospaced))
                    .foregroundColor(Color(red: 0.4, green: 0.35, blue: 0.3))
            }
            Text(token.surfaceExpression.isEmpty ? token.id : token.surfaceExpression)
                .font(.system(size: 14, weight: .regular, design: .serif))
                .foregroundColor(Color(red: 0.78, green: 0.72, blue: 0.6))
                .frame(maxWidth: .infinity, alignment: .leading)
            Rectangle()
                .fill(LinearGradient(
                    colors: [Color(red: 0.2, green: 0.25, blue: 0.3), Color(red: 0.85, green: 0.7, blue: 0.2)],
                    startPoint: .leading,
                    endPoint: .trailing))
                .frame(height: 2)
                .opacity(Double(max(0.1, min(1.0, token.temperature))))
        }
        .padding(8)
        .background(Color(red: 0.1, green: 0.09, blue: 0.07))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color(red: 0.24, green: 0.19, blue: 0.16), lineWidth: 1)
        )
        .cornerRadius(6)
    }
}

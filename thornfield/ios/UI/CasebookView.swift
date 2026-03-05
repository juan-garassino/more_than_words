import SwiftUI

struct CasebookView: View {
    let casebook: CasebookState

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let height = proxy.size.height
            let cellW = width / 6
            let cellH = height / 8

            ZStack {
                ForEach(0..<8, id: \.self) { row in
                    ForEach(0..<6, id: \.self) { col in
                        Rectangle()
                            .stroke(Color(red: 0.18, green: 0.14, blue: 0.12), lineWidth: 1)
                            .frame(width: cellW, height: cellH)
                            .position(x: cellW * (CGFloat(col) + 0.5), y: cellH * (CGFloat(row) + 0.5))
                    }
                }

                ForEach(Array(casebook.placedTriads.keys), id: \.self) { pos in
                    if let triad = casebook.placedTriads[pos] {
                        TriadView(triad: triad)
                            .frame(width: cellW * 2.5, height: cellH * 0.9)
                            .position(x: cellW * (CGFloat(pos.col) + 0.5), y: cellH * (CGFloat(pos.row) + 0.5))
                    }
                }
            }
        }
        .frame(height: 360)
    }
}

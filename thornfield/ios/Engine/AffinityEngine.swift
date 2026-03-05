import Foundation

final class AffinityEngine {
    let affinityTable: AffinityTable
    let triadIndex: TriadIndex

    init(affinityTable: AffinityTable, triadIndex: TriadIndex) {
        self.affinityTable = affinityTable
        self.triadIndex = triadIndex
    }

    func computeJointEnergy(triad: Triad, casebook: CasebookState) -> Float {
        var energy: Float = 1.0
        let placedTokens = casebook.allPlacedTokens

        let pairs = combinations(triad.tokens, 2)
        for (a, b) in pairs {
            if a.tokenClass == .unknown || b.tokenClass == .unknown { continue }
            let bonus = affinityTable[TokenPairKey(a.id, b.id)] ?? 0.0
            energy -= bonus * 0.4
        }

        for placed in placedTokens {
            for candidate in triad.tokens {
                if placed.tokenClass == .unknown || candidate.tokenClass == .unknown { continue }
                let bonus = affinityTable[TokenPairKey(placed.id, candidate.id)] ?? 0.0
                energy -= bonus * 0.3
            }
        }

        for placed in placedTokens {
            for candidate in triad.tokens {
                if placed.tokenClass == .unknown || candidate.tokenClass == .unknown { continue }
                let repulsion = !Set(placed.repulsionTags).isDisjoint(with: Set(candidate.affinityTags))
                if repulsion { energy += 0.5 }
            }
        }

        let adjacent = casebook.spatiallyAdjacentTokens(to: triad.suggestedPosition, radius: 2)
        for candidate in triad.tokens {
            if candidate.tokenClass == .unknown { continue }
            for adj in adjacent {
                if adj.tokenClass == .unknown { continue }
                let bonus = affinityTable[TokenPairKey(candidate.id, adj.id)] ?? 0.0
                energy -= bonus * 0.15
            }
        }

        for candidate in triad.tokens {
            if candidate.tokenClass == .unknown { continue }
            if !candidate.phase.isAvailable(at: casebook.turnCount) {
                energy += 0.8
            }
        }

        if triad.isClassDiverse { energy -= 0.2 }

        let gradientBonus = triad.tokens.reduce(0) { $0 + $1.narrativeGradient } / 3.0
        energy -= gradientBonus * 0.15

        return max(0.0, min(1.0, energy))
    }

    func surfaceTriads(count: Int = 3, casebook: CasebookState) -> [Triad] {
        let candidates = triadIndex.candidates(for: casebook, turnCount: casebook.turnCount)

        var scored = candidates.map { triad -> (Triad, Float) in
            let energy = computeJointEnergy(triad: triad, casebook: casebook)
            var t = triad
            t.jointEnergy = energy
            t.convergenceDelta = projectConvergenceDelta(triad: t, casebook: casebook)
            return (t, energy)
        }

        scored.sort { $0.1 < $1.1 }
        return selectDiverseTop(scored.map(\.0), count: count)
    }

    private func selectDiverseTop(_ sorted: [Triad], count: Int) -> [Triad] {
        var selected: [Triad] = []
        var usedClasses: Set<TokenClass> = []

        for triad in sorted {
            if selected.count >= count { break }
            let newClasses = Set(triad.tokens.map(\.tokenClass))
            if newClasses.isDisjoint(with: usedClasses) || selected.count < 2 {
                selected.append(triad)
                usedClasses.formUnion(newClasses)
            }
        }

        while selected.count < count {
            if let next = sorted.first(where: { t in
                !selected.contains { $0.id == t.id }
            }) {
                selected.append(next)
            } else { break }
        }

        return selected
    }

    private func projectConvergenceDelta(triad: Triad, casebook: CasebookState) -> [Float] {
        let contribution = triad.attractorContribution
        return contribution.map { v in
            min(1.0, v * 0.25)
        }
    }

    private func combinations<T>(_ items: [T], _ k: Int) -> [(T, T)] {
        guard k == 2 else { return [] }
        var result: [(T, T)] = []
        for i in 0..<items.count {
            for j in (i + 1)..<items.count {
                result.append((items[i], items[j]))
            }
        }
        return result
    }
}

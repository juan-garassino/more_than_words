import Foundation

final class TriadIndex {
    private var tagIndex: [String: [Triad]] = [:]
    private var allValidTriads: [Triad] = []

    init() {}

    func build(from tokens: [Token], affinityTable: AffinityTable) {
        tagIndex.removeAll()
        allValidTriads.removeAll()

        let nonInvariant = tokens.filter { !$0.isInvariant }

        for i in 0..<nonInvariant.count {
            for j in (i + 1)..<nonInvariant.count {
                for k in (j + 1)..<nonInvariant.count {
                    let trio = [nonInvariant[i], nonInvariant[j], nonInvariant[k]]
                    if isValidTriad(trio, affinityTable: affinityTable) {
                        let triad = Triad(tokens: trio)
                        allValidTriads.append(triad)
                        for token in trio {
                            for tag in token.affinityTags {
                                tagIndex[tag, default: []].append(triad)
                            }
                        }
                    }
                }
            }
        }
    }

    func candidates(for casebook: CasebookState, turnCount: Int) -> [Triad] {
        let activeTags = casebook.activeAffinityTags
        var seen: Set<UUID> = []
        var result: [Triad] = []

        for tag in activeTags {
            for triad in tagIndex[tag] ?? [] {
                if seen.insert(triad.id).inserted {
                    result.append(triad)
                }
            }
        }

        let placedIds = casebook.placedTokenIds
        return result.filter { triad in
            !triad.tokens.contains { placedIds.contains($0.id) } &&
            triad.tokens.allSatisfy { $0.phase.isAvailable(at: turnCount) }
        }
    }

    private func isValidTriad(_ tokens: [Token], affinityTable: AffinityTable) -> Bool {
        guard Set(tokens.map(\.tokenClass)).count == 3 else { return false }

        let pairs = [(tokens[0], tokens[1]), (tokens[0], tokens[2]), (tokens[1], tokens[2])]
        let hasAffinity = pairs.contains { pair in
            (affinityTable[TokenPairKey(pair.0.id, pair.1.id)] ?? 0.0) > 0.1
        }
        guard hasAffinity else { return false }

        for (a, b) in pairs {
            let aRepelsB = !Set(a.repulsionTags).isDisjoint(with: Set(b.affinityTags))
            let bRepelsA = !Set(b.repulsionTags).isDisjoint(with: Set(a.affinityTags))
            if aRepelsB || bRepelsA { return false }
        }

        return true
    }
}

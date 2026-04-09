import Foundation

final class CartridgeLoader {
    func load(from directoryURL: URL) throws -> CartridgeSpec {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        let tokensURL = directoryURL.appendingPathComponent("tokens.json")
        let graphURL = directoryURL.appendingPathComponent("graph.json")

        let manifestData = try Data(contentsOf: manifestURL)
        let tokensData = try Data(contentsOf: tokensURL)
        let graphData = try Data(contentsOf: graphURL)

        let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any] ?? [:]
        let tokensArray = try JSONSerialization.jsonObject(with: tokensData) as? [[String: Any]] ?? []
        let graph = try JSONSerialization.jsonObject(with: graphData) as? [String: Any] ?? [:]

        let type = CartridgeType(rawValue: manifest["cartridge_type"] as? String ?? "MYSTERY") ?? .mystery
        let caseId = manifest["case_id"] as? String ?? ""
        let title = manifest["title"] as? String ?? ""
        let mode = manifest["mode"] as? String ?? "converging"
        let nDims = manifest["n_attractor_dims"] as? Int ?? 3
        let convergenceThreshold = Float(manifest["convergence_threshold"] as? Double ?? 0.75)
        let convergenceRate = Float(manifest["convergence_rate"] as? Double ?? 0.25)
        let minTurns = manifest["min_turns"] as? Int ?? 10
        let maxTurns = manifest["max_turns"] as? Int ?? 18
        let initialDimensionValue = Float(manifest["initial_dimension_value"] as? Double ?? 0.0)
        let dimensionLowerBound = Float(manifest["dimension_lower_bound"] as? Double ?? 0.0)
        let dimensionUpperBound = Float(manifest["dimension_upper_bound"] as? Double ?? 1.0)

        let tokens: [Token] = tokensArray.compactMap { item in
            guard
                let id = item["id"] as? String,
                let tokenClassRaw = item["token_class"] as? String,
                let phaseRaw = item["phase"] as? String,
                let weights = item["attractor_weights"] as? [Double]
            else { return nil }

            let affinityTags = item["affinity_tags"] as? [String] ?? []
            let repulsionTags = item["repulsion_tags"] as? [String] ?? []
            let temperature = Float(item["temperature"] as? Double ?? 0.5)
            let narrativeGradient = Float(item["narrative_gradient"] as? Double ?? 0.0)
            let isInvariant = item["is_invariant"] as? Bool ?? false

            return Token(
                id: id,
                tokenClass: TokenClass(rawValue: tokenClassRaw) ?? .unknown,
                phase: TokenPhase(rawValue: phaseRaw) ?? .any,
                attractorWeights: weights.map { Float($0) },
                affinityTags: affinityTags,
                repulsionTags: repulsionTags,
                temperature: temperature,
                narrativeGradient: narrativeGradient,
                isInvariant: isInvariant,
                surfaceExpression: item["surface_expression"] as? String ?? ""
            )
        }

        let affinityTable = buildAffinityTable(graph: graph)

        let invariantIds = tokens.filter { $0.isInvariant }.map(\.id)
        let invariantTokens = tokens.filter { $0.isInvariant }
        let openingTokenIds = manifest["opening_token_ids"] as? [String] ?? []

        return CartridgeSpec(
            type: type,
            caseId: caseId,
            title: title,
            mode: mode,
            nAttractorDims: nDims,
            convergenceThreshold: convergenceThreshold,
            convergenceRate: convergenceRate,
            minTurns: minTurns,
            maxTurns: maxTurns,
            initialDimensionValue: initialDimensionValue,
            dimensionLowerBound: dimensionLowerBound,
            dimensionUpperBound: dimensionUpperBound,
            tokens: tokens,
            affinityTable: affinityTable,
            invariantTokens: invariantTokens,
            openingTokenIds: openingTokenIds.isEmpty ? invariantIds : openingTokenIds
        )
    }

    private func buildAffinityTable(graph: [String: Any]) -> AffinityTable {
        var table: AffinityTable = [:]
        let edges = graph["edges"] as? [[String: Any]] ?? []

        for edge in edges {
            guard let from = edge["from"] as? String,
                  let to = edge["to"] as? String else { continue }
            let weight = Float(edge["weight"] as? Double ?? 0.0)
            table[TokenPairKey(from, to)] = weight
        }
        return table
    }
}

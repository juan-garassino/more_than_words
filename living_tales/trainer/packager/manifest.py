from __future__ import annotations

from typing import Dict


def build_manifest(spec, proof_report: Dict) -> Dict:
    return {
        "cartridge_type": "MYSTERY",
        "case_id": spec.case_id,
        "title": spec.title,
        "mode": spec.mode,
        "version": "1.0.0",
        "vocab_size": spec.vocab_size,
        "embedding_dim": spec.embedding_dim,
        "context_dim": spec.context_dim,
        "n_attractor_dims": spec.n_attractor_dims,
        "convergence_threshold": spec.convergence_threshold,
        "convergence_rate": spec.convergence_rate,
        "min_turns": spec.min_turns,
        "max_turns": spec.max_turns,
        "initial_dimension_value": spec.initial_dimension_value,
        "dimension_lower_bound": spec.dimension_lower_bound,
        "dimension_upper_bound": spec.dimension_upper_bound,
        "proof_convergence_rate": proof_report.get("convergence_rate", 0.0),
        "proof_invariant_accuracy": proof_report.get("invariant_accuracy", 0.0),
        "hopfield_basin_coverage": proof_report.get("basin_coverage", 0.0),
    }

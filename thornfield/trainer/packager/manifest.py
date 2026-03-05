from __future__ import annotations

from typing import Dict


def build_manifest(spec, proof_report: Dict) -> Dict:
    return {
        "cartridge_type": "MYSTERY",
        "case_id": spec.case_id,
        "title": spec.title,
        "version": "1.0.0",
        "vocab_size": spec.vocab_size,
        "embedding_dim": spec.embedding_dim,
        "context_dim": spec.context_dim,
        "n_attractor_dims": spec.n_attractor_dims,
        "convergence_threshold": spec.convergence_threshold,
        "convergence_rate": spec.convergence_rate,
        "min_turns": spec.min_turns,
        "max_turns": spec.max_turns,
        "proof_convergence_rate": proof_report.get("convergence_rate", 0.0),
        "proof_invariant_accuracy": proof_report.get("invariant_accuracy", 0.0),
        "hopfield_basin_coverage": proof_report.get("basin_coverage", 0.0),
    }

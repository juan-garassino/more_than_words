from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from packager.manifest import build_manifest


def export_mystery_cartridge(model, spec, output_path: str, proof_report: dict) -> None:
    if not proof_report.get("passed", False):
        raise RuntimeError("Convergence proof failed. Cartridge export blocked.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(spec, proof_report)

    tokens = [
        {
            "id": t.id,
            "token_class": t.token_class.value,
            "phase": t.phase.value,
            "attractor_weights": t.attractor_weights.tolist(),
            "affinity_tags": t.affinity_tags,
            "repulsion_tags": t.repulsion_tags,
            "temperature": t.temperature,
            "narrative_gradient": t.narrative_gradient,
            "is_invariant": t.is_invariant,
        }
        for t in spec.tokens
    ]

    graph = {
        "nodes": spec.token_graph.nodes,
        "edges": [
            {"from": a, "to": b, "weight": w} for (a, b), w in spec.token_graph.edges.items()
        ],
    }

    precomputed = {"valid_triads": []}

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("tokens.json", json.dumps(tokens, indent=2))
        zf.writestr("graph.json", json.dumps(graph, indent=2))
        zf.writestr("attractor.json", json.dumps({"invariants": spec.invariant_token_ids}, indent=2))
        zf.writestr("phases.json", json.dumps({"min_turns": spec.min_turns, "max_turns": spec.max_turns}, indent=2))
        zf.writestr("precomputed.json", json.dumps(precomputed, indent=2))
        zf.writestr("expressions.json", json.dumps({}, indent=2))
        zf.writestr("proof.json", json.dumps(proof_report, indent=2))
        buffer = io.BytesIO()
        np.savez(buffer, weights=np.array([0.0], dtype=np.float32))
        zf.writestr("weights.npz", buffer.getvalue())

    _ = model


def export_tamagotchi_cartridge(model, spec, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cartridge_type": "TAMAGOTCHI",
        "case_id": spec.case_id,
        "title": spec.title,
        "version": "1.0.0",
        "vocab_size": spec.vocab_size,
        "embedding_dim": spec.embedding_dim,
        "n_needs": spec.n_needs,
        "n_personality_dims": spec.n_personality_dims,
    }

    tokens = [
        {
            "id": t.id,
            "token_class": t.token_class.value,
            "phase": t.phase.value,
            "attractor_weights": t.attractor_weights.tolist(),
            "affinity_tags": t.affinity_tags,
            "repulsion_tags": t.repulsion_tags,
            "temperature": t.temperature,
            "narrative_gradient": t.narrative_gradient,
            "is_invariant": t.is_invariant,
        }
        for t in spec.tokens
    ]

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("tokens.json", json.dumps(tokens, indent=2))
        zf.writestr("expressions.json", json.dumps({}, indent=2))
        buffer = io.BytesIO()
        np.savez(buffer, weights=np.array([0.0], dtype=np.float32))
        zf.writestr("weights.npz", buffer.getvalue())

    _ = model

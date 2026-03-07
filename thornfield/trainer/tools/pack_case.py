import argparse
import json
from pathlib import Path


def _load_case(case_path: Path) -> dict:
    with case_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def pack_case(case_path: Path, output_dir: Path) -> None:
    case = _load_case(case_path)

    tokens = case["tokens"]
    expressions = case.get("expressions", {})

    # Derive n_attractor_dims from the attractor dimensions array if present
    attractor_dims = case.get("attractor", {}).get("dimensions", [])
    n_attractor_dims = len(attractor_dims) if attractor_dims else 3

    # Read convergence params from case JSON, fall back to defaults
    convergence_rate = case.get("convergence_rate", 0.40)
    convergence_threshold = case.get("convergence_threshold", 0.75)
    min_turns = case.get("min_turns", 10)
    max_turns = case.get("max_turns", 18)

    spec = {
        "cartridge_type": "MYSTERY",
        "case_id": case["case_id"],
        "title": case["title"],
        "version": "1.0.0",
        "vocab_size": len(tokens),
        "embedding_dim": 64,
        "context_dim": 128,
        "n_attractor_dims": n_attractor_dims,
        "convergence_threshold": convergence_threshold,
        "convergence_rate": convergence_rate,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "opening_token_ids": case["opening_token_ids"],
        "invariant_token_ids": case["invariant_token_ids"],
    }

    tokens_out = []
    for token in tokens:
        token_id = token["id"]
        tokens_out.append(
            {
                "id": token_id,
                "token_class": token["class"],
                "phase": token["phase"],
                "attractor_weights": token["attractor_weights"],
                "affinity_tags": token.get("affinity_tags", []),
                "repulsion_tags": token.get("repulsion_tags", []),
                "temperature": token.get("temperature", 0.5),
                "is_invariant": token.get("is_invariant", False),
                "surface_expression": expressions.get(token_id, ""),
                "stream": token.get("stream", "EVIDENCE"),
                "agency": token.get("agency", "SHARED"),
            }
        )

    graph = case["graph"]
    phases = {
        "min_turns": min_turns,
        "max_turns": max_turns,
    }
    attractor = {
        "invariants": case["invariant_token_ids"],
    }

    _write_json(output_dir / "spec.json", spec)
    _write_json(output_dir / "tokens.json", tokens_out)
    _write_json(output_dir / "graph.json", graph)
    _write_json(output_dir / "expressions.json", expressions)
    _write_json(output_dir / "phases.json", phases)
    _write_json(output_dir / "attractor.json", attractor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a single-case JSON into trainer case files.")
    parser.add_argument("case_json", type=Path, help="Path to case JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for case files (default: trainer/cases/<case_id>)",
    )
    args = parser.parse_args()

    case = _load_case(args.case_json)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "cases" / case["case_id"]

    pack_case(args.case_json, output_dir)
    print(f"Packed case into {output_dir}")


if __name__ == "__main__":
    main()

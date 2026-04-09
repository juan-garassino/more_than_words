#!/usr/bin/env python3
"""Convert a packed trainer case into an iOS cartridge bundle.

Usage:
    python3 living_tales/trainer/tools/pack_ios_cartridge.py amber_cipher

Input:  living_tales/trainer/cases/<case_id>/  (spec.json, tokens.json, graph.json)
Output: living_tales/ios/Cartridge/Bundles/<Title>.cartridge/
        manifest.json, tokens.json, graph.json
"""

import argparse
import json
import shutil
from pathlib import Path


def _title_to_bundle_name(title: str) -> str:
    """'The Amber Cipher' -> 'AmberCipher'"""
    return "".join(word.capitalize() for word in title.split())


def pack_ios_cartridge(case_id: str, repo_root: Path) -> Path:
    cases_dir = repo_root / "living_tales" / "trainer" / "cases" / case_id
    if not cases_dir.exists():
        raise FileNotFoundError(f"Packed case not found: {cases_dir}")

    spec = json.loads((cases_dir / "spec.json").read_text())
    tokens = json.loads((cases_dir / "tokens.json").read_text())
    graph = json.loads((cases_dir / "graph.json").read_text())

    title = spec.get("title", case_id)
    bundle_name = _title_to_bundle_name(title)
    out_dir = repo_root / "living_tales" / "ios" / "Cartridge" / "Bundles" / f"{bundle_name}.cartridge"
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json — iOS CartridgeLoader fields
    manifest = {
        "cartridge_type": spec.get("cartridge_type", "MYSTERY"),
        "case_id": spec["case_id"],
        "title": title,
        "mode": spec.get("mode", "converging"),
        "vocab_size": spec.get("vocab_size", len(tokens)),
        "n_attractor_dims": spec.get("n_attractor_dims", 3),
        "convergence_threshold": spec.get("convergence_threshold", 0.75),
        "convergence_rate": spec.get("convergence_rate", 0.40),
        "min_turns": spec.get("min_turns", 10),
        "max_turns": spec.get("max_turns", 18),
        "initial_dimension_value": spec.get("initial_dimension_value", 0.0),
        "dimension_lower_bound": spec.get("dimension_lower_bound", 0.0),
        "dimension_upper_bound": spec.get("dimension_upper_bound", 1.0),
        "opening_token_ids": spec.get("opening_token_ids", []),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # tokens.json — already in iOS CartridgeLoader format
    (out_dir / "tokens.json").write_text(json.dumps(tokens, indent=2))

    # graph.json — already in iOS CartridgeLoader format
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2))

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a trainer case into an iOS cartridge bundle.")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher)")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root or Path(__file__).resolve().parents[3]
    out_dir = pack_ios_cartridge(args.case_id, repo_root)
    print(f"iOS cartridge written to: {out_dir}")
    print(f"  manifest.json: {(out_dir / 'manifest.json').stat().st_size} bytes")
    print(f"  tokens.json:   {(out_dir / 'tokens.json').stat().st_size} bytes")
    print(f"  graph.json:    {(out_dir / 'graph.json').stat().st_size} bytes")


if __name__ == "__main__":
    main()

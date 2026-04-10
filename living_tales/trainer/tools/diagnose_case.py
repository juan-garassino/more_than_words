#!/usr/bin/env python3
"""
Diagnostic tool for Living Tales case JSON analysis.

Analyses attractor weight distributions, head vocab masks, and token
specialisation to identify cases that will train well vs. cases with
flat or degenerate weight structures.

Usage:
    python diagnose_case.py amber_cipher
    python diagnose_case.py amber_cipher --compare little_creature
    python diagnose_case.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TRAINER_ROOT = _HERE.parent
_PROJECT_ROOT = _TRAINER_ROOT.parents[1]
_CASES_JSON = _PROJECT_ROOT / "cases"
_CASES_PACKED = _TRAINER_ROOT / "cases"

sys.path.insert(0, str(_TRAINER_ROOT))

from core.cartridge import CartridgeSpec
from trainer.training_profile import build_head_vocab_masks
from tools.pack_case import pack_case as pack_case_fn

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_spec(case_id: str) -> CartridgeSpec:
    """Load a CartridgeSpec, packing the case JSON if needed."""
    spec_path = _CASES_PACKED / case_id / "spec.json"
    if spec_path.exists():
        return CartridgeSpec.load(str(spec_path))

    case_json = _CASES_JSON / f"{case_id}.json"
    if not case_json.exists():
        raise FileNotFoundError(
            f"No packed case at {spec_path} and no source JSON at {case_json}"
        )

    with tempfile.TemporaryDirectory(prefix=f"{case_id}_diag_") as tmpdir:
        out_dir = Path(tmpdir) / case_id
        pack_case_fn(case_json, out_dir)
        return CartridgeSpec.load(str(out_dir / "spec.json"))


def _load_dimension_names(case_id: str) -> list[str]:
    """Try to read human-readable dimension labels from the raw case JSON."""
    case_json = _CASES_JSON / f"{case_id}.json"
    if not case_json.exists():
        return []
    with open(case_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dims = raw.get("attractor", {}).get("dimensions", [])
    return [d.get("label") or d.get("id", f"dim_{i}") for i, d in enumerate(dims)]


def _engine_tokens(spec: CartridgeSpec):
    return [t for t in spec.tokens if t.agency.value == "ENGINE" and not t.is_invariant]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _build_weight_matrix(spec: CartridgeSpec) -> np.ndarray:
    """Return (n_tokens, n_dims) matrix of absolute attractor weights."""
    n_dims = spec.n_attractor_dims
    rows = []
    for tok in spec.tokens:
        w = np.abs(tok.attractor_weights[:n_dims].astype(np.float64))
        rows.append(w)
    return np.array(rows)


def analyse_case(case_id: str):
    """Return a dict of diagnostic information for a single case."""
    spec = _load_spec(case_id)
    dim_names = _load_dimension_names(case_id)
    threshold = 0.10 if spec.mode == "converging" else 0.05
    masks = build_head_vocab_masks(spec, threshold=threshold)  # (n_dims, vocab_size) bool

    n_dims = spec.n_attractor_dims
    weight_matrix = _build_weight_matrix(spec)  # (V, n_dims)
    engine_idx = [
        i for i, t in enumerate(spec.tokens)
        if t.agency.value == "ENGINE" and not t.is_invariant
    ]

    # Per-dimension stats
    dim_stats = []
    for d in range(n_dims):
        mask_size = int(masks[d].sum().item())
        dim_weights = weight_matrix[:, d]  # all tokens

        # Top 10 tokens by abs weight
        sorted_idx = np.argsort(-dim_weights)
        top10 = []
        for idx in sorted_idx[:10]:
            top10.append((spec.tokens[idx].id, float(dim_weights[idx])))

        nonzero = dim_weights[dim_weights > 1e-9]
        max_w = float(dim_weights.max()) if len(dim_weights) else 0.0
        mean_w = float(nonzero.mean()) if len(nonzero) else 0.0

        label = dim_names[d] if d < len(dim_names) else f"dim_{d}"

        dim_stats.append({
            "dim": d,
            "label": label,
            "invariant": spec.invariant_token_ids[d] if d < len(spec.invariant_token_ids) else "?",
            "mask_size": mask_size,
            "top10": top10,
            "max_weight": max_w,
            "mean_weight": mean_w,
        })

    # Specialists: tokens where (max_weight_across_dims - min_weight_across_dims) > 0.15
    specialists = []
    for i, tok in enumerate(spec.tokens):
        w = weight_matrix[i]
        spread = float(w.max() - w.min())
        if spread > 0.15:
            specialists.append((tok.id, spread))
    specialists.sort(key=lambda x: -x[1])

    # Overlap matrix
    overlap = np.zeros((n_dims, n_dims), dtype=int)
    for a in range(n_dims):
        for b in range(n_dims):
            overlap[a, b] = int((masks[a] & masks[b]).sum().item())

    # Health flags
    flags = []
    for ds in dim_stats:
        if ds["mask_size"] > 50:
            flags.append(f"BROAD HEAD  dim {ds['dim']} ({ds['label']}): {ds['mask_size']} tokens in mask")
        if ds["max_weight"] < 0.25:
            flags.append(f"FLAT DIM    dim {ds['dim']} ({ds['label']}): max weight {ds['max_weight']:.3f}")
        if len(ds["top10"]) >= 2:
            w1 = ds["top10"][0][1]
            w2 = ds["top10"][1][1]
            if w2 > 0 and w1 / w2 > 5.0:
                flags.append(
                    f"DOMINANT TOKEN  dim {ds['dim']} ({ds['label']}): "
                    f"{ds['top10'][0][0]} = {w1:.3f} vs next = {w2:.3f} (ratio {w1/w2:.1f}x)"
                )

    n_engine = len(engine_idx)
    specialist_pct = (len(specialists) / n_engine * 100) if n_engine else 0
    if specialist_pct < 5.0:
        flags.append(
            f"NO SPECIALISTS  {len(specialists)}/{n_engine} engine tokens "
            f"({specialist_pct:.1f}%) have spread > 0.15"
        )

    return {
        "case_id": case_id,
        "title": spec.title,
        "mode": spec.mode,
        "vocab_size": spec.vocab_size,
        "n_dims": n_dims,
        "n_engine": n_engine,
        "dim_stats": dim_stats,
        "specialists": specialists,
        "specialist_pct": specialist_pct,
        "overlap": overlap,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _render_case(info: dict) -> None:
    """Print full diagnostic output for a single case."""
    console.rule(f"[bold]{info['case_id']}[/bold] - {info['title']}")
    console.print(
        f"mode={info['mode']}  vocab={info['vocab_size']}  "
        f"dims={info['n_dims']}  engine_tokens={info['n_engine']}"
    )
    console.print()

    # Per-dimension tables
    for ds in info["dim_stats"]:
        table = Table(
            title=f"Dim {ds['dim']}: {ds['label']}  (invariant: {ds['invariant']})",
            show_lines=False,
        )
        table.add_column("Rank", justify="right", style="dim", width=4)
        table.add_column("Token", style="cyan", min_width=30)
        table.add_column("Weight", justify="right", style="bold", width=8)

        for rank, (tok_id, w) in enumerate(ds["top10"], 1):
            table.add_row(str(rank), tok_id, f"{w:.4f}")

        table.caption = (
            f"mask_size={ds['mask_size']}  "
            f"max={ds['max_weight']:.4f}  "
            f"mean_nonzero={ds['mean_weight']:.4f}"
        )
        console.print(table)
        console.print()

    # Specialists
    console.print(
        f"[bold]Specialists[/bold] (spread > 0.15): "
        f"{len(info['specialists'])}/{info['n_engine']} engine tokens "
        f"({info['specialist_pct']:.1f}%)"
    )
    if info["specialists"]:
        spec_table = Table(show_header=True, show_lines=False)
        spec_table.add_column("Token", style="cyan", min_width=30)
        spec_table.add_column("Spread", justify="right", width=8)
        for tok_id, spread in info["specialists"][:15]:
            spec_table.add_row(tok_id, f"{spread:.4f}")
        if len(info["specialists"]) > 15:
            spec_table.add_row("...", f"({len(info['specialists'])} total)")
        console.print(spec_table)
    console.print()

    # Overlap matrix
    n_dims = info["n_dims"]
    overlap = info["overlap"]
    ov_table = Table(title="Head Mask Overlap", show_lines=True)
    ov_table.add_column("", width=6)
    for d in range(n_dims):
        ov_table.add_column(f"D{d}", justify="right", width=6)
    for a in range(n_dims):
        row = [f"D{a}"]
        for b in range(n_dims):
            row.append(str(overlap[a, b]))
        ov_table.add_row(*row)
    console.print(ov_table)
    console.print()

    # Health flags
    if info["flags"]:
        flag_text = Text()
        for flag in info["flags"]:
            flag_text.append(f"  !! {flag}\n", style="bold red")
        console.print(Panel(flag_text, title="Health Flags", border_style="red"))
    else:
        console.print(Panel("[green]No health flags - case looks good[/green]", title="Health Flags"))
    console.print()


def _render_compare(info_a: dict, info_b: dict) -> None:
    """Show two cases side by side."""
    table = Table(title="Case Comparison", show_lines=True)
    table.add_column("Metric", style="bold", min_width=25)
    table.add_column(info_a["case_id"], justify="right", min_width=20)
    table.add_column(info_b["case_id"], justify="right", min_width=20)

    table.add_row("Mode", info_a["mode"], info_b["mode"])
    table.add_row("Vocab size", str(info_a["vocab_size"]), str(info_b["vocab_size"]))
    table.add_row("Dimensions", str(info_a["n_dims"]), str(info_b["n_dims"]))
    table.add_row("Engine tokens", str(info_a["n_engine"]), str(info_b["n_engine"]))

    max_dims = max(info_a["n_dims"], info_b["n_dims"])
    for d in range(max_dims):
        ds_a = info_a["dim_stats"][d] if d < info_a["n_dims"] else None
        ds_b = info_b["dim_stats"][d] if d < info_b["n_dims"] else None
        label_a = ds_a["label"] if ds_a else "-"
        label_b = ds_b["label"] if ds_b else "-"
        table.add_row(
            f"Dim {d} label",
            label_a,
            label_b,
        )
        table.add_row(
            f"Dim {d} mask size",
            str(ds_a["mask_size"]) if ds_a else "-",
            str(ds_b["mask_size"]) if ds_b else "-",
        )
        table.add_row(
            f"Dim {d} max weight",
            f"{ds_a['max_weight']:.4f}" if ds_a else "-",
            f"{ds_b['max_weight']:.4f}" if ds_b else "-",
        )
        table.add_row(
            f"Dim {d} mean weight",
            f"{ds_a['mean_weight']:.4f}" if ds_a else "-",
            f"{ds_b['mean_weight']:.4f}" if ds_b else "-",
        )
        # Top token
        if ds_a and ds_a["top10"]:
            top_a = f"{ds_a['top10'][0][0]} ({ds_a['top10'][0][1]:.3f})"
        else:
            top_a = "-"
        if ds_b and ds_b["top10"]:
            top_b = f"{ds_b['top10'][0][0]} ({ds_b['top10'][0][1]:.3f})"
        else:
            top_b = "-"
        table.add_row(f"Dim {d} top token", top_a, top_b)

    table.add_row(
        "Specialists",
        f"{len(info_a['specialists'])} ({info_a['specialist_pct']:.1f}%)",
        f"{len(info_b['specialists'])} ({info_b['specialist_pct']:.1f}%)",
    )
    table.add_row(
        "Health flags",
        str(len(info_a["flags"])),
        str(len(info_b["flags"])),
    )

    console.print(table)
    console.print()

    # Show flags for both
    for info in (info_a, info_b):
        if info["flags"]:
            flag_text = Text()
            for flag in info["flags"]:
                flag_text.append(f"  !! {flag}\n", style="bold red")
            console.print(Panel(flag_text, title=f"Flags: {info['case_id']}", border_style="red"))
        else:
            console.print(Panel(
                f"[green]No flags[/green]",
                title=f"Flags: {info['case_id']}",
            ))
    console.print()


def _render_summary(results: list[dict]) -> None:
    """Show a summary table for all cases."""
    table = Table(title="Case Diagnostic Summary", show_lines=True)
    table.add_column("Case", style="cyan", min_width=20)
    table.add_column("Mode", width=12)
    table.add_column("Vocab", justify="right", width=6)
    table.add_column("Dims", justify="right", width=5)
    table.add_column("Engine", justify="right", width=7)
    table.add_column("Max W", justify="right", width=7)
    table.add_column("Specialists", justify="right", width=12)
    table.add_column("Flags", min_width=30)

    for info in results:
        max_w = max((ds["max_weight"] for ds in info["dim_stats"]), default=0)
        flag_str = ", ".join(f.split()[0] for f in info["flags"]) if info["flags"] else "[green]OK[/green]"
        max_style = "bold red" if max_w < 0.25 else ("yellow" if max_w < 0.36 else "green")

        table.add_row(
            info["case_id"],
            info["mode"],
            str(info["vocab_size"]),
            str(info["n_dims"]),
            str(info["n_engine"]),
            Text(f"{max_w:.3f}", style=max_style),
            f"{len(info['specialists'])} ({info['specialist_pct']:.1f}%)",
            flag_str,
        )

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for Living Tales case analysis",
    )
    parser.add_argument("case_id", nargs="?", help="Case ID to diagnose")
    parser.add_argument("--compare", metavar="CASE_ID", help="Second case ID for side-by-side comparison")
    parser.add_argument("--all", action="store_true", help="Scan all case JSONs and show summary")

    args = parser.parse_args()

    if args.all:
        jsons = sorted(_CASES_JSON.glob("*.json"))
        if not jsons:
            console.print(f"[red]No case JSON files found in {_CASES_JSON}[/red]")
            sys.exit(1)
        results = []
        for jp in jsons:
            case_id = jp.stem
            try:
                info = analyse_case(case_id)
                results.append(info)
            except Exception as e:
                console.print(f"[yellow]Skipping {case_id}: {e}[/yellow]")
        _render_summary(results)
        return

    if not args.case_id:
        parser.print_help()
        sys.exit(1)

    if args.compare:
        info_a = analyse_case(args.case_id)
        info_b = analyse_case(args.compare)
        _render_compare(info_a, info_b)
    else:
        info = analyse_case(args.case_id)
        _render_case(info)


if __name__ == "__main__":
    main()

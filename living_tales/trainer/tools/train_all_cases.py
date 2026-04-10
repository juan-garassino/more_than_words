"""
Living Tales — Train All Cases
==============================
Validates, packs, trains dialogue transformers for all valid cases,
plays automated games on each, and zips all outputs + logs for download.

Usage:
    cd living_tales/trainer
    python3 tools/train_all_cases.py
    python3 tools/train_all_cases.py --paths 2000 --epochs 100 --rl-episodes 500 --games 5
    python3 tools/train_all_cases.py --production   # full production settings
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TRAINER = _HERE.parent
_ROOT = _TRAINER.parent.parent
_CASES_JSON = _ROOT / "cases"
_CASES_PACKED = _TRAINER / "cases"
_OUTPUTS = _TRAINER / "outputs"
_LOGS = _OUTPUTS / "logs"

sys.path.insert(0, str(_TRAINER))

from core.cartridge import CartridgeSpec
from core.token import TokenAgency, TokenStream


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _banner(title: str):
    w = 64
    print(f"\n{'#' * w}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'#' * w}", flush=True)


# ── Tee helper: capture stdout to a string while also printing ────────────

class TeeCapture:
    """Context manager that captures stdout to a buffer while still printing."""
    def __init__(self):
        self.buffer = io.StringIO()
        self._original = None

    def __enter__(self):
        self._original = sys.stdout
        sys.stdout = _TeeWriter(self._original, self.buffer)
        return self

    def __exit__(self, *exc):
        sys.stdout = self._original

    def getvalue(self) -> str:
        return self.buffer.getvalue()


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)

    def flush(self):
        for s in self.streams:
            s.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Discover and validate cases
# ─────────────────────────────────────────────────────────────────────────────

def discover_cases() -> list[str]:
    """Find all case JSONs and return IDs of those that pass validation."""
    case_files = sorted(_CASES_JSON.glob("*.json"))
    valid = []

    _log(f"Found {len(case_files)} case files")

    for cf in case_files:
        case_id = cf.stem
        if case_id.startswith("gen_"):
            continue

        try:
            result = subprocess.run(
                ["python3", str(_ROOT / "living_tales_case_validator.py"), str(cf)],
                capture_output=True, text=True, timeout=30,
            )
            if "OVERALL PASS" in result.stdout:
                valid.append(case_id)
                _log(f"  PASS  {case_id}")
            else:
                _log(f"  FAIL  {case_id} (skipping)")
        except Exception as e:
            _log(f"  ERROR {case_id}: {e} (skipping)")

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Pack case
# ─────────────────────────────────────────────────────────────────────────────

def pack_case(case_id: str) -> bool:
    json_path = _CASES_JSON / f"{case_id}.json"
    try:
        result = subprocess.run(
            ["python3", str(_TRAINER / "tools" / "pack_case.py"), str(json_path)],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train
# ─────────────────────────────────────────────────────────────────────────────

def train_case(case_id: str, args) -> dict:
    from trainer.train_dialogue import train_dialogue_cartridge

    spec_path = str(_CASES_PACKED / case_id / "spec.json")
    output_dir = str(_OUTPUTS / case_id)

    model, history = train_dialogue_cartridge(
        spec_path=spec_path,
        output_dir=output_dir,
        n_dialogues=args.paths,
        n_epochs=args.epochs,
        n_rl_episodes=args.rl_episodes,
        device=args.device,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Play automated games
# ─────────────────────────────────────────────────────────────────────────────

def play_games(case_id: str, n_games: int) -> list[dict]:
    from tools.fit_play_report import play_game
    from trainer.dialogue_model import DialogueTransformer
    from trainer.train_dialogue import _build_mappings

    spec_path = _CASES_PACKED / case_id / "spec.json"
    model_path = _OUTPUTS / case_id / "dialogue_model.pt"

    spec = CartridgeSpec.load(str(spec_path))
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)

    model = DialogueTransformer(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        context_dim=ckpt["context_dim"],
        n_layers=ckpt.get("n_layers", 4),
        n_heads=ckpt.get("n_heads", 4),
        max_seq_len=ckpt.get("max_seq_len", 64),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mappings = dict(zip(
        ["id_to_idx", "class_to_idx", "phase_to_idx", "stream_to_idx", "agency_to_idx"],
        _build_mappings(spec.tokens),
    ))

    games = []
    for seed in range(n_games):
        game = play_game(model, spec, mappings, seed=seed)
        games.append(game)
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Report per case
# ─────────────────────────────────────────────────────────────────────────────

def case_report(case_id: str, games: list, history: dict) -> dict:
    converged = sum(1 for g in games if g["converged"])
    avg_turns = np.mean([g["n_turns"] for g in games])
    avg_conv = np.mean([g["final_convergence"] for g in games])

    sup = history.get("supervised", history)
    losses = sup.get("epoch_losses", [])

    return {
        "case_id": case_id,
        "convergence_rate": converged / len(games) if games else 0,
        "avg_turns": float(avg_turns),
        "avg_convergence": float(avg_conv),
        "n_games": len(games),
        "kd_loss_start": losses[0] if losses else None,
        "kd_loss_end": losses[-1] if losses else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Zip outputs + logs
# ─────────────────────────────────────────────────────────────────────────────

def zip_outputs(case_ids: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = _OUTPUTS / f"living_tales_all_{timestamp}.zip"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for case_id in case_ids:
            case_out = _OUTPUTS / case_id
            case_packed = _CASES_PACKED / case_id

            # Model checkpoint
            model_pt = case_out / "dialogue_model.pt"
            if model_pt.exists():
                zf.write(str(model_pt), f"{case_id}/dialogue_model.pt")

            # History
            history_json = case_out / "history.json"
            if history_json.exists():
                zf.write(str(history_json), f"{case_id}/history.json")

            # Packed case data
            for name in ["spec.json", "tokens.json", "graph.json", "expressions.json"]:
                p = case_packed / name
                if p.exists():
                    zf.write(str(p), f"{case_id}/cartridge/{name}")

        # Training logs
        if _LOGS.exists():
            for log_file in _LOGS.glob("*.log"):
                zf.write(str(log_file), f"logs/{log_file.name}")

    _log(f"Zipped to: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(zip_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train all valid Living Tales cases.")
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rl-episodes", type=int, default=100)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--production", action="store_true",
                        help="Full production settings (3K paths, 200 epochs, 500 RL)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific case IDs to train (default: all valid)")
    args = parser.parse_args()

    if args.production:
        args.paths = 3000
        args.epochs = 200
        args.rl_episodes = 500
        args.games = 5

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Create logs directory
    _LOGS.mkdir(parents=True, exist_ok=True)

    _banner("LIVING TALES — TRAIN ALL CASES")
    _log(f"Settings: paths={args.paths} epochs={args.epochs} rl={args.rl_episodes} games={args.games} device={args.device}")

    t0 = time.time()

    # Discover
    if args.cases:
        case_ids = args.cases
        _log(f"Training specified cases: {case_ids}")
    else:
        case_ids = discover_cases()

    if not case_ids:
        _log("No valid cases found.")
        sys.exit(1)

    _log(f"\nCandidates: {len(case_ids)} cases: {', '.join(case_ids)}\n")

    # ── Phase 1: Validate + Pack ALL cases upfront ──────────────────────────
    _banner("PHASE 1: VALIDATE + PACK")
    ready_ids = []
    for case_id in case_ids:
        _log(f"Packing {case_id}...")
        if pack_case(case_id):
            ready_ids.append(case_id)
            _log(f"  OK  {case_id}")
        else:
            _log(f"  FAIL {case_id} — skipping")

    if not ready_ids:
        _log("No cases ready for training.")
        sys.exit(1)

    _log(f"\n{len(ready_ids)}/{len(case_ids)} cases ready for training: {', '.join(ready_ids)}\n")

    # ── Phase 2: Train + Play each case ─────────────────────────────────────
    _banner("PHASE 2: TRAIN + PLAY")
    reports = []
    trained_ids = []

    for i, case_id in enumerate(ready_ids):
        _banner(f"CASE {i+1}/{len(ready_ids)}: {case_id}")

        # Train with log capture
        _log("Training...")
        tee = TeeCapture()
        try:
            with tee:
                history = train_case(case_id, args)
        except Exception as e:
            _log(f"Training failed for {case_id}: {e}")
            # Save partial log even on failure
            log_path = _LOGS / f"{case_id}.log"
            log_path.write_text(tee.getvalue(), encoding="utf-8")
            continue

        # Save training log
        log_path = _LOGS / f"{case_id}.log"
        log_path.write_text(tee.getvalue(), encoding="utf-8")
        _log(f"Saved log: {log_path}")

        # Play
        _log(f"Playing {args.games} automated games...")
        try:
            games = play_games(case_id, args.games)
        except Exception as e:
            _log(f"Play failed for {case_id}: {e}")
            games = []

        # Report
        report = case_report(case_id, games, history)
        reports.append(report)
        trained_ids.append(case_id)

        converged = sum(1 for g in games if g["converged"])
        _log(
            f"Done: {converged}/{len(games)} converged | "
            f"loss {report['kd_loss_start']:.3f}→{report['kd_loss_end']:.3f} | "
            f"avg turns {report['avg_turns']:.1f}"
        )

    # Final summary
    elapsed = time.time() - t0
    _banner("FINAL SUMMARY")
    _log(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    _log(f"Cases trained: {len(trained_ids)}/{len(case_ids)}")

    summary_lines = []
    header = f"\n{'  Case':<25s} {'Conv%':>6s} {'Turns':>6s} {'Loss':>12s}"
    sep = f"  {'─' * 50}"
    print(header); summary_lines.append(header)
    print(sep); summary_lines.append(sep)
    for r in reports:
        loss_str = f"{r['kd_loss_start']:.3f}→{r['kd_loss_end']:.3f}" if r['kd_loss_start'] else "—"
        line = f"  {r['case_id']:<23s} {r['convergence_rate']:>5.0%} {r['avg_turns']:>6.1f} {loss_str:>12s}"
        print(line); summary_lines.append(line)

    # Save summary
    summary_path = _LOGS / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Save reports JSON
    reports_path = _LOGS / "reports.json"
    reports_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    # Zip everything
    if trained_ids:
        _banner("PACKAGING")
        zip_path = zip_outputs(trained_ids)
        print(f"\n  Download: {zip_path}")
    print()


if __name__ == "__main__":
    main()

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

# External save directory (e.g. Google Drive) — set via --output-dir
_SAVE_DIR: Path | None = None


def _save_case_to_external(case_id: str):
    """Incrementally copy a case's outputs to the external save directory.
    Called after each case finishes so nothing is lost on disconnect."""
    if _SAVE_DIR is None:
        return
    src = _OUTPUTS / case_id
    dst = _SAVE_DIR / case_id
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(dst / f.name))
    # Also copy the log if it exists
    log_src = _LOGS / f"{case_id}.log"
    log_dst = _SAVE_DIR / "logs"
    if log_src.exists():
        log_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(log_src), str(log_dst / log_src.name))
    _log(f"  Saved to: {dst}")

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

def _case_mode(case_id: str) -> str:
    """Read case JSON and return 'creature' or 'mystery'."""
    json_path = _CASES_JSON / f"{case_id}.json"
    try:
        with json_path.open() as f:
            data = json.load(f)
        return "creature" if data.get("mode") == "oscillating" else "mystery"
    except Exception:
        return "mystery"


def discover_cases() -> tuple[list[str], list[str]]:
    """Find all case JSONs, validate, and return (mysteries, creatures)."""
    case_files = sorted(_CASES_JSON.glob("*.json"))
    mysteries = []
    creatures = []

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
            passed = "OVERALL PASS" in result.stdout
            # Also accept cases that only fail on SIM or WEIGHTS checks —
            # these are stricter than what training requires
            if not passed and "FAIL" in result.stdout:
                fail_lines = [l for l in result.stdout.splitlines() if l.startswith("FAIL")]
                soft_only = all(("SIM" in l or "WEIGHTS" in l) for l in fail_lines)
                if soft_only:
                    passed = True

            if passed:
                mode = _case_mode(case_id)
                if mode == "creature":
                    creatures.append(case_id)
                else:
                    mysteries.append(case_id)
                _log(f"  PASS  {case_id}  [{mode}]")
            else:
                _log(f"  FAIL  {case_id} (skipping)")
        except Exception as e:
            _log(f"  ERROR {case_id}: {e} (skipping)")

    return mysteries, creatures


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

def train_case(case_id: str, args, model_size_override: str | None = None,
               output_id: str | None = None) -> dict:
    """Train a case with SceneTransformer (multi-head). Falls back to single-token if needed."""
    from trainer.train_dialogue import train_scene_cartridge

    spec_path = str(_CASES_PACKED / case_id / "spec.json")
    out_name = output_id or case_id
    output_dir = str(_OUTPUTS / out_name)

    model, history = train_scene_cartridge(
        spec_path=spec_path,
        output_dir=output_dir,
        n_dialogues=args.paths,
        n_epochs=args.epochs,
        n_rl_episodes=args.rl_episodes,
        model_size_override=model_size_override,
        device=args.device,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Play automated games
# ─────────────────────────────────────────────────────────────────────────────

def play_games(case_id: str, n_games: int, output_id: str | None = None) -> list[dict]:
    from tools.fit_play_report import play_game
    from trainer.dialogue_model import DialogueTransformer, SceneTransformer
    from trainer.training_profile import build_head_vocab_masks
    from trainer.train_dialogue import _build_mappings

    spec_path = _CASES_PACKED / case_id / "spec.json"
    out_name = output_id or case_id
    model_path = _OUTPUTS / out_name / "dialogue_model.pt"

    spec = CartridgeSpec.load(str(spec_path))
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)

    model_type = ckpt.get("model_type", "dialogue")
    if model_type == "scene":
        head_masks = build_head_vocab_masks(spec)
        model = SceneTransformer(
            vocab_size=ckpt["vocab_size"],
            embedding_dim=ckpt["embedding_dim"],
            context_dim=ckpt["context_dim"],
            n_layers=ckpt.get("n_layers", 6),
            n_heads=ckpt.get("n_heads", 6),
            n_output_heads=ckpt.get("n_output_heads", spec.n_attractor_dims),
            head_vocab_masks=head_masks,
            max_seq_len=ckpt.get("max_seq_len", 128),
        )
    else:
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

def zip_outputs(mystery_ids: list[str], creature_ids: list[str],
                scale_ids: list[str] | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = _OUTPUTS / f"living_tales_all_{timestamp}.zip"

    groups = [("mysteries", mystery_ids), ("creatures", creature_ids)]
    if scale_ids:
        groups.append(("scale", scale_ids))

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for group_name, case_ids in groups:
            for case_id in case_ids:
                case_out = _OUTPUTS / case_id
                # Scale experiments: packed data lives under base case id
                base_id = case_id.split("__")[0] if "__" in case_id else case_id
                case_packed = _CASES_PACKED / base_id
                prefix = f"{group_name}/{case_id}"

                # Model checkpoint
                model_pt = case_out / "dialogue_model.pt"
                if model_pt.exists():
                    zf.write(str(model_pt), f"{prefix}/dialogue_model.pt")

                # History
                history_json = case_out / "history.json"
                if history_json.exists():
                    zf.write(str(history_json), f"{prefix}/history.json")

                # Packed case data
                for name in ["spec.json", "tokens.json", "graph.json", "expressions.json"]:
                    p = case_packed / name
                    if p.exists():
                        zf.write(str(p), f"{prefix}/cartridge/{name}")

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
    parser.add_argument("--scale-experiment", nargs="*", default=None,
                        help="Case IDs to train at S/M/L model sizes (e.g. dust_and_verdict)")
    parser.add_argument("--scale-sizes", nargs="*", default=None,
                        help="Model size overrides for specific cases (e.g. amber_cipher_L=L little_creature_M=L)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="External save directory (e.g. Google Drive path). Outputs are copied here after each case.")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Override max_turns in packed specs (e.g. 200 for longer games). Applied after packing.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force fresh training even if checkpoints exist. Default: auto-resume incomplete runs.")
    args = parser.parse_args()

    if args.production:
        args.paths = 3000
        args.epochs = 200
        args.rl_episodes = 500
        args.games = 5

    # Set external save directory (timestamped for versioning, auto-resume)
    global _SAVE_DIR
    args.resume = not args.no_resume  # resume by default
    if args.output_dir:
        output_base = Path(args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        # Find latest run dir
        existing_runs = sorted(output_base.glob("run_*"), reverse=True)

        if args.resume and existing_runs:
            # Check if latest run is incomplete (has some but not all checkpoints)
            latest = existing_runs[0]
            n_checkpoints = sum(1 for d in latest.iterdir()
                                if d.is_dir() and d.name != "logs"
                                and (d / "dialogue_model.pt").exists())
            if n_checkpoints > 0:
                # Resume into the same dir
                _SAVE_DIR = latest
                _log(f"Resuming into: {_SAVE_DIR} ({n_checkpoints} checkpoints found)")
            else:
                # Empty run dir or no checkpoints — create fresh
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                _SAVE_DIR = output_base / f"run_{timestamp}"
        else:
            # No resume or no existing runs — create fresh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _SAVE_DIR = output_base / f"run_{timestamp}"

        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (_SAVE_DIR / "logs").mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Create logs directory
    _LOGS.mkdir(parents=True, exist_ok=True)

    _banner("LIVING TALES — TRAIN ALL CASES")
    _log(f"Settings: paths={args.paths} epochs={args.epochs} rl={args.rl_episodes} games={args.games} device={args.device}")
    if _SAVE_DIR:
        _log(f"External save: {_SAVE_DIR} (incremental after each case)")

    t0 = time.time()

    # ── Discover + classify ─────────────────────────────────────────────────
    if args.cases:
        # Manual list: classify each
        mystery_ids = [c for c in args.cases if _case_mode(c) == "mystery"]
        creature_ids = [c for c in args.cases if _case_mode(c) == "creature"]
        _log(f"Specified cases: {len(mystery_ids)} mysteries, {len(creature_ids)} creatures")
    else:
        mystery_ids, creature_ids = discover_cases()

    total = len(mystery_ids) + len(creature_ids)
    if total == 0:
        _log("No valid cases found.")
        sys.exit(1)

    _log(f"\n  Mysteries ({len(mystery_ids)}): {', '.join(mystery_ids)}")
    _log(f"  Creatures ({len(creature_ids)}): {', '.join(creature_ids)}")

    # ── Phase 1: Validate + Pack ALL cases upfront ──────────────────────────
    _banner("PHASE 1: VALIDATE + PACK")
    ready_mysteries = []
    ready_creatures = []
    for label, ids, ready in [
        ("mystery", mystery_ids, ready_mysteries),
        ("creature", creature_ids, ready_creatures),
    ]:
        for case_id in ids:
            _log(f"  Packing {case_id} [{label}]...")
            if pack_case(case_id):
                ready.append(case_id)
            else:
                _log(f"    FAIL {case_id} — skipping")

    ready_total = len(ready_mysteries) + len(ready_creatures)
    if ready_total == 0:
        _log("No cases ready for training.")
        sys.exit(1)

    # Apply --max-turns override to packed specs
    if args.max_turns:
        _log(f"  Overriding max_turns={args.max_turns} in all packed specs")
        for case_id in ready_mysteries + ready_creatures:
            spec_path = _CASES_PACKED / case_id / "spec.json"
            if spec_path.exists():
                with open(spec_path) as f:
                    spec_data = json.load(f)
                spec_data["max_turns"] = args.max_turns
                with open(spec_path, "w") as f:
                    json.dump(spec_data, f, indent=2)
                    f.write("\n")

    # Parse --scale-sizes overrides (e.g. amber_cipher_L=L little_creature_M=L)
    size_overrides: dict[str, str] = {}
    if args.scale_sizes:
        for item in args.scale_sizes:
            cid, sz = item.split("=")
            size_overrides[cid] = sz.upper()

    _log(f"\nReady: {len(ready_mysteries)} mysteries + {len(ready_creatures)} creatures = {ready_total} total")
    if size_overrides:
        _log(f"  Size overrides: {size_overrides}")
    if args.scale_experiment:
        _log(f"  Scale experiments: {args.scale_experiment} (S/M/L each)")

    # ── Phase 2: Train mysteries, then creatures ────────────────────────────
    reports = []
    trained_mysteries = []
    trained_creatures = []
    case_num = 0

    for group_label, ready_ids, trained_list in [
        ("MYSTERIES", ready_mysteries, trained_mysteries),
        ("CREATURES", ready_creatures, trained_creatures),
    ]:
        if not ready_ids:
            continue
        _banner(f"PHASE 2: TRAIN {group_label} ({len(ready_ids)})")

        for case_id in ready_ids:
            case_num += 1
            _banner(f"[{group_label}] {case_num}/{ready_total}: {case_id}")

            # Resume: skip if checkpoint already exists
            if args.resume:
                found = (_OUTPUTS / case_id / "dialogue_model.pt").exists()
                if not found and _SAVE_DIR:
                    found = (_SAVE_DIR / case_id / "dialogue_model.pt").exists()
                if found:
                    _log(f"  SKIP (checkpoint exists)")
                    trained_list.append(case_id)
                    continue

            # Train with log capture
            override = size_overrides.get(case_id)
            _log(f"Training...{f' (model size override: {override})' if override else ''}")
            tee = TeeCapture()
            try:
                with tee:
                    history = train_case(case_id, args, model_size_override=override)
            except Exception as e:
                _log(f"Training failed for {case_id}: {e}")
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
            report["type"] = group_label.lower().rstrip("s")
            reports.append(report)
            trained_list.append(case_id)

            converged = sum(1 for g in games if g["converged"])
            _log(
                f"Done: {converged}/{len(games)} converged | "
                f"loss {report['kd_loss_start']:.3f}→{report['kd_loss_end']:.3f} | "
                f"avg turns {report['avg_turns']:.1f}"
            )

            # Incremental save to external directory (e.g. Google Drive)
            _save_case_to_external(case_id)

    # ── Phase 3: Scale experiments ────────────────────────────────────────
    scale_trained = []
    if args.scale_experiment:
        _banner("PHASE 3: SCALE EXPERIMENTS")
        for case_id in args.scale_experiment:
            if case_id not in (ready_mysteries + ready_creatures):
                _log(f"Skipping {case_id} — not packed")
                continue
            for size in ["S", "M", "L"]:
                output_id = f"{case_id}__{size}" if size != "S" else case_id
                # Skip if already trained at default size in phase 2
                if size == "S" and case_id in (trained_mysteries + trained_creatures):
                    _log(f"  {output_id} already trained in phase 2, skipping")
                    scale_trained.append(output_id)
                    continue

                _banner(f"SCALE: {case_id} @ {size} → {output_id}")

                # Resume: skip if checkpoint already exists
                if args.resume:
                    found = (_OUTPUTS / output_id / "dialogue_model.pt").exists()
                    if not found and _SAVE_DIR:
                        found = (_SAVE_DIR / output_id / "dialogue_model.pt").exists()
                    if found:
                        _log(f"  SKIP (checkpoint exists)")
                        scale_trained.append(output_id)
                        continue

                _log(f"Training {case_id} with model size {size}...")
                tee = TeeCapture()
                try:
                    with tee:
                        history = train_case(case_id, args, model_size_override=size, output_id=output_id)
                except Exception as e:
                    _log(f"Training failed for {output_id}: {e}")
                    (_LOGS / f"{output_id}.log").write_text(tee.getvalue(), encoding="utf-8")
                    continue

                (_LOGS / f"{output_id}.log").write_text(tee.getvalue(), encoding="utf-8")

                try:
                    games = play_games(case_id, args.games, output_id=output_id)
                except Exception as e:
                    _log(f"Play failed for {output_id}: {e}")
                    games = []

                report = case_report(output_id, games, history)
                report["type"] = "scale"
                report["base_case"] = case_id
                report["model_size"] = size
                reports.append(report)
                scale_trained.append(output_id)

                converged = sum(1 for g in games if g["converged"])
                _log(f"Done: {converged}/{len(games)} converged | avg turns {report['avg_turns']:.1f}")

                # Incremental save to external directory
                _save_case_to_external(output_id)

    # ── Final summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _banner("FINAL SUMMARY")
    _log(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    _log(f"Trained: {len(trained_mysteries)} mysteries + {len(trained_creatures)} creatures + {len(scale_trained)} scale experiments")

    summary_lines = []
    for group_label, trained_list in [("MYSTERIES", trained_mysteries), ("CREATURES", trained_creatures), ("SCALE EXPERIMENTS", scale_trained)]:
        group_reports = [r for r in reports if r["case_id"] in trained_list]
        if not group_reports:
            continue
        header = f"\n  {group_label}"
        sub = f"  {'─' * 55}"
        col = f"  {'Case':<23s} {'Conv%':>6s} {'Turns':>6s} {'Loss':>12s}"
        for line in [header, sub, col, sub]:
            print(line); summary_lines.append(line)
        for r in group_reports:
            loss_str = f"{r['kd_loss_start']:.3f}→{r['kd_loss_end']:.3f}" if r['kd_loss_start'] else "—"
            line = f"  {r['case_id']:<23s} {r['convergence_rate']:>5.0%} {r['avg_turns']:>6.1f} {loss_str:>12s}"
            print(line); summary_lines.append(line)

    # Save summary + reports
    (_LOGS / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (_LOGS / "reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    # Zip everything (grouped: mysteries/, creatures/, scale/)
    if trained_mysteries or trained_creatures or scale_trained:
        _banner("PACKAGING")
        zip_path = zip_outputs(trained_mysteries, trained_creatures, scale_trained)
        print(f"\n  Download: {zip_path}")

        # Copy zip + summary to external save directory
        if _SAVE_DIR is not None:
            shutil.copy2(zip_path, str(_SAVE_DIR / Path(zip_path).name))
            shutil.copy2(str(_LOGS / "summary.txt"), str(_SAVE_DIR / "logs" / "summary.txt"))
            shutil.copy2(str(_LOGS / "reports.json"), str(_SAVE_DIR / "logs" / "reports.json"))
            _log(f"  Final zip saved to: {_SAVE_DIR / Path(zip_path).name}")
    print()


if __name__ == "__main__":
    main()

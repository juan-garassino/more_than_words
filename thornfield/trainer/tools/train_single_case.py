import argparse
import json
import os
import sys
from pathlib import Path

# Avoid OpenMP SHM issues in restricted environments.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from core.cartridge import CartridgeSpec
from trainer.train_mystery import train_mystery_cartridge
from validator.convergence_proof import ConvergenceProof
from packager.export_mystery import export_mystery_cartridge

_CONNECTION_MODEL_TYPE = "connection"


def main() -> None:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description="Train and export a single mystery case.")
    parser.add_argument("case_id", help="Case id (folder name under cases)")
    parser.add_argument("--paths", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--proof-paths", type=int, default=200)
    parser.add_argument("--proof-max-attempts", type=int, default=2000)
    parser.add_argument("--skip-proof", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-type",
        choices=["triad", "connection"],
        default="triad",
        help="Model architecture: 'triad' (default) or 'connection'.",
    )
    args = parser.parse_args()

    case_dir = Path(__file__).resolve().parents[1] / "cases" / args.case_id
    spec_path = case_dir / "spec.json"
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_preview = CartridgeSpec.load(str(spec_path))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    from collections import Counter
    stream_counts = Counter(t.stream.value for t in spec_preview.tokens)
    agency_counts = Counter(t.agency.value for t in spec_preview.tokens)

    print("=" * 60)
    print(f"  THORNFIELD TRAINER — {spec_preview.title}")
    print("=" * 60)
    print(f"  case_id          : {args.case_id}")
    print(f"  device           : {device}")
    print(f"  paths            : {args.paths}")
    print(f"  epochs           : {args.epochs}")
    print(f"  proof_paths      : {args.proof_paths}")
    print(f"  skip_proof       : {args.skip_proof}")
    print(f"  model_type       : {args.model_type}")
    print("  --- spec ---")
    print(f"  vocab_size       : {spec_preview.vocab_size}")
    print(f"  embedding_dim    : {spec_preview.embedding_dim}")
    print(f"  context_dim      : {spec_preview.context_dim}")
    print(f"  attractor_dims   : {spec_preview.n_attractor_dims}")
    print(f"  convergence_rate : {spec_preview.convergence_rate}")
    print(f"  conv_threshold   : {spec_preview.convergence_threshold}")
    print(f"  turns            : {spec_preview.min_turns}–{spec_preview.max_turns}")
    print(f"  streams          : {dict(stream_counts)}")
    print(f"  agency           : {dict(agency_counts)}")
    print("=" * 60, flush=True)

    if args.model_type == _CONNECTION_MODEL_TYPE:
        from trainer.train_connection import train_connection_cartridge
        model, history = train_connection_cartridge(
            spec_path=str(spec_path),
            output_dir=str(output_dir),
            n_paths=args.paths,
            n_epochs=args.epochs,
            device=device,
        )
    else:
        model, history = train_mystery_cartridge(
            spec_path=str(spec_path),
            output_dir=str(output_dir),
            n_paths=args.paths,
            n_epochs=args.epochs,
            convergence_rate=0.25,
            min_turns=10,
            max_turns=18,
            device=device,
        )

    param_count = sum(p.numel() for p in model.parameters())
    print("─" * 60)
    print(f"  TRAINING DONE")
    print(f"  final_loss    : {history.get('loss', 0.0):.4f}")
    print(f"  model_params  : {param_count:,}")
    print("─" * 60, flush=True)

    # Persist trained model and history for local testing.
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec_path": str(spec_path),
            "case_id": args.case_id,
            "embedding_dim": spec_preview.embedding_dim,
            "context_dim": spec_preview.context_dim,
            "n_attractor_dims": spec_preview.n_attractor_dims,
            "model_type": args.model_type,
        },
        model_path,
    )
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved history: {history_path}")

    if args.skip_proof or args.model_type == _CONNECTION_MODEL_TYPE:
        if args.model_type == _CONNECTION_MODEL_TYPE:
            print("  Skipping convergence proof (not applicable for connection model).")
        else:
            print("  Skipping convergence proof by request.")
        print("=" * 60)
        print("  RUN SUMMARY")
        print("=" * 60)
        print(f"  case_id      : {args.case_id}")
        print(f"  device       : {device}")
        print(f"  final_loss   : {history.get('loss', 0.0):.4f}")
        print(f"  model_params : {param_count:,}")
        print(f"  proof        : SKIPPED")
        print(f"  export       : SKIPPED")
        print("=" * 60, flush=True)
        return

    spec = CartridgeSpec.load(str(spec_path))
    proof = ConvergenceProof().run(
        model,
        spec,
        n_test_paths=args.proof_paths,
        max_attempts=args.proof_max_attempts,
        verbose=True,
    )

    exported = export_mystery_cartridge(
        model=model,
        spec=spec,
        output_path=str(output_dir / f"{spec.title.replace(' ', '')}.cartridge"),
        proof_report=proof,
    )

    print("=" * 60)
    print("  RUN SUMMARY")
    print("=" * 60)
    print(f"  case_id          : {args.case_id}")
    print(f"  device           : {device}")
    print(f"  paths            : {args.paths}")
    print(f"  epochs           : {args.epochs}")
    print(f"  final_loss       : {history.get('loss', 0.0):.4f}")
    print(f"  model_params     : {param_count:,}")
    print(f"  proof            : {'PASSED' if proof['passed'] else 'FAILED'}")
    print(f"  convergence_rate : {proof['convergence_rate']:.1%}")
    print(f"  invariant_acc    : {proof['invariant_accuracy']:.1%}")
    print(f"  lyapunov         : {'PASS' if proof['lyapunov_passed'] else 'FAIL'} ({proof['lyapunov_monotone_rate']:.1%} monotone)")
    print(f"  basin_coverage   : {proof['basin_coverage']:.1%}")
    print(f"  spurious         : {proof['spurious_attractors']}")
    print(f"  turn_range       : {proof['min_turns']}–{proof['max_turns']}")
    print(f"  exported         : {'YES' if exported else 'NO (proof failed)'}")
    print("=" * 60, flush=True)

    if not exported:
        print(
            "\n[ERROR] Cartridge was not exported — proof did not pass.\n"
            "[ERROR] The game engine requires a valid cartridge. Fix the proof issues above and retrain.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

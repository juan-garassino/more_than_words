import argparse
import json
import os
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
    args = parser.parse_args()

    case_dir = Path(__file__).resolve().parents[1] / "cases" / args.case_id
    spec_path = case_dir / "spec.json"
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_preview = CartridgeSpec.load(str(spec_path))
    print("Spec summary:")
    print(f"  Title: {spec_preview.title}")
    print(f"  Vocab size: {spec_preview.vocab_size}")
    print(f"  Embedding dim: {spec_preview.embedding_dim}")
    print(f"  Context dim: {spec_preview.context_dim}")
    print(f"  Attractor dims: {spec_preview.n_attractor_dims}")
    print(f"  Convergence threshold: {spec_preview.convergence_threshold}")
    print(f"  Turns: {spec_preview.min_turns}-{spec_preview.max_turns}")

    print(f"Training case: {args.case_id}")
    print(f"Spec path: {spec_path}")
    print(f"Output dir: {output_dir}")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(
        f"Paths: {args.paths} | Epochs: {args.epochs} | Proof paths: {args.proof_paths} | Device: {device}"
    )

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
    print(f"Training complete. Final loss: {history.get('loss', 0.0):.4f}")
    print(f"Model parameters: {param_count:,}")

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
        },
        model_path,
    )
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved history: {history_path}")

    if args.skip_proof:
        print("Skipping convergence proof by request.")
        return

    spec = CartridgeSpec.load(str(spec_path))
    proof = ConvergenceProof().run(
        model,
        spec,
        n_test_paths=args.proof_paths,
        max_attempts=args.proof_max_attempts,
        verbose=True,
    )

    export_mystery_cartridge(
        model=model,
        spec=spec,
        output_path=str(output_dir / f"{spec.title.replace(' ', '')}.cartridge"),
        proof_report=proof,
    )


if __name__ == "__main__":
    main()

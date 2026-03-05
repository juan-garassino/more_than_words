import argparse
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
    print(f"Paths: {args.paths} | Epochs: {args.epochs} | Device: {args.device}")

    model, history = train_mystery_cartridge(
        spec_path=str(spec_path),
        output_dir=str(output_dir),
        n_paths=args.paths,
        n_epochs=args.epochs,
        convergence_rate=0.25,
        min_turns=10,
        max_turns=18,
        device=args.device,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Training complete. Final loss: {history.get('loss', 0.0):.4f}")
    print(f"Model parameters: {param_count:,}")

    spec = CartridgeSpec.load(str(spec_path))
    proof = ConvergenceProof().run(model, spec, n_test_paths=200)

    export_mystery_cartridge(
        model=model,
        spec=spec,
        output_path=str(output_dir / f"{spec.title.replace(' ', '')}.cartridge"),
        proof_report=proof,
    )


if __name__ == "__main__":
    main()

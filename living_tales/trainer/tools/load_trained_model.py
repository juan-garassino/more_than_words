import argparse
from pathlib import Path

import torch

from core.cartridge import CartridgeSpec
from trainer.energy_model import MysteryEnergyModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a trained mystery model for local testing.")
    parser.add_argument("model_path", help="Path to model.pt produced by train_single_case.py")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    checkpoint = torch.load(model_path, map_location=args.device)
    spec_path = checkpoint.get("spec_path")
    if spec_path is None:
        raise RuntimeError("Checkpoint missing spec_path. Re-train with updated train_single_case.py.")

    spec = CartridgeSpec.load(spec_path)
    model = MysteryEnergyModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_attractor_dims=spec.n_attractor_dims,
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {model_path}")
    print(f"Spec: {spec.title} ({spec.case_id})")
    print(f"Params: {param_count:,}")


if __name__ == "__main__":
    main()

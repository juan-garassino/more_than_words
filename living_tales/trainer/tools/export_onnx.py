"""
Living Tales — Export SceneTransformer to ONNX for Unity

Usage:
    cd living_tales/trainer
    PYTHONPATH=. python3 tools/export_onnx.py amber_cipher
    PYTHONPATH=. python3 tools/export_onnx.py little_creature_M

Loads a trained dialogue_model.pt and exports it as ONNX for Unity Sentis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cartridge import CartridgeSpec
from trainer.dialogue_model import DialogueTransformer, SceneTransformer
from trainer.training_profile import build_head_vocab_masks


def export_model(case_id: str, output_path: str | None = None):
    """Export a trained model to ONNX."""
    cases_dir = Path(__file__).resolve().parents[1] / "cases" / case_id
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs" / case_id

    spec_path = cases_dir / "spec.json"
    model_path = outputs_dir / "dialogue_model.pt"

    if not model_path.exists():
        print(f"Error: {model_path} not found")
        sys.exit(1)

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

    # Create dummy inputs
    seq_len = 16
    dummy_token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_class_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_phase_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_stream_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_agency_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_padding = torch.zeros(1, seq_len, dtype=torch.bool)

    dummy_inputs = (
        dummy_token_ids, dummy_class_ids, dummy_phase_ids,
        dummy_stream_ids, dummy_agency_ids, dummy_padding,
    )

    # Output path
    if output_path is None:
        output_path = str(outputs_dir / "model.onnx")

    # Determine output names
    if model_type == "scene":
        n_heads = ckpt.get("n_output_heads", spec.n_attractor_dims)
        output_names = [f"head_{d}" for d in range(n_heads)]
    else:
        output_names = ["logits"]

    # Export
    print(f"Exporting {case_id} ({model_type}) to ONNX...")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Output heads: {len(output_names)}")

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=[
            "token_ids", "class_ids", "phase_ids",
            "stream_ids", "agency_ids", "padding_mask",
        ],
        output_names=output_names,
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "class_ids": {0: "batch", 1: "seq_len"},
            "phase_ids": {0: "batch", 1: "seq_len"},
            "stream_ids": {0: "batch", 1: "seq_len"},
            "agency_ids": {0: "batch", 1: "seq_len"},
            "padding_mask": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 1: "seq_len"} for name in output_names},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Check file size
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Exported: {output_path} ({onnx_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("case_id", help="Case ID (e.g. amber_cipher, little_creature_M)")
    parser.add_argument("--output", default=None, help="Output ONNX path")
    args = parser.parse_args()

    export_model(args.case_id, args.output)


if __name__ == "__main__":
    main()

"""
probe_universal_transfer.py
============================

Holds out one case, trains a base model on the other two, and measures
universal-dim accuracy on the held-out case's trajectories. The eval-gate
calls this to verify the base+adapter generalization claim before merging
new training runs.

This script DOES NOT train end-to-end (training is Colab-only). It assumes
a base checkpoint already exists and probes its universal-dim accuracy on
held-out trajectories.

Usage:
    python tools/probe_universal_transfer.py \\
        --base outputs/_base/base_universal.pt \\
        --held-out venetian_mirror

Pass criteria: ≥80% universal-dim accuracy on held-out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
TRAINER_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(TRAINER_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from generator.trajectory_loader import TrajectoryLoader  # noqa: E402
from trainer.trajectory_dataset_v2 import TrajectoryDatasetV2  # noqa: E402
from trainer.structured_scene_model_v2 import (  # noqa: E402
    StructuredSceneTransformerV2, V2Config,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="path to base_universal.pt")
    p.add_argument("--held-out", required=True, help="case_id to evaluate on")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="minimum universal-dim accuracy to pass (default 0.80)")
    args = p.parse_args()

    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    dim_order = ckpt["dim_order"]
    dim_vocab = ckpt["dim_vocab"]
    full_vocab = ckpt["full_vocab"]
    universal = ckpt["universal_dims"]

    cfg_dict = ckpt["config"]
    cfg = V2Config(**{k: cfg_dict[k] for k in (
        "hidden_dim", "n_layers", "n_heads", "max_history", "lora_rank",
    ) if k in cfg_dict})
    cfg.universal_dims = universal

    model = StructuredSceneTransformerV2(dim_vocab, dim_order, full_vocab, cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    L = TrajectoryLoader(args.held_out, REPO_ROOT)
    trajs = L.load_all()
    fv2i = {t: i for i, t in enumerate(full_vocab)}
    dv2i = {d: {t: i for i, t in enumerate(toks)} for d, toks in dim_vocab.items()}

    ds = TrajectoryDatasetV2(
        trajs, dim_order, dim_vocab, fv2i, dv2i,
        max_history=cfg.max_history, universal_dims=universal,
        universal_only=False, truncate_history_p=0.0, include_branches=False,
    )
    print(f"[probe] held-out {args.held_out}: {len(ds)} examples; universal dims: {universal}")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=ds.collate)

    correct = {d: 0 for d in dim_order}
    total = {d: 0 for d in dim_order}
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["history_tokens"], batch["history_dims"],
                batch["player_card"], batch["target_scene"],
                target_scene_type=batch["scene_type"],
            )
            out.pop("_z_logits", None)
            for d in dim_order:
                pred = out[d].argmax(dim=-1)
                m = batch["active_mask"][d] > 0
                correct[d] += int((pred[m] == batch["target_scene"][d][m]).sum().item())
                total[d] += int(m.sum().item())

    print("\n[probe] per-dim accuracy on held-out:")
    for d in dim_order:
        acc = correct[d] / total[d] if total[d] else 0.0
        marker = "(universal)" if d in universal else "(case-specific)"
        print(f"     {d:14s} {acc:.3f}  {marker}")

    uni_total = sum(total[d] for d in universal)
    uni_correct = sum(correct[d] for d in universal)
    uni_acc = uni_correct / uni_total if uni_total else 0.0
    print(f"\n[probe] universal-only accuracy: {uni_acc:.3f}  (threshold {args.threshold:.2f})")

    if uni_acc < args.threshold:
        print(f"[probe] FAIL: {uni_acc:.3f} < {args.threshold:.2f}")
        sys.exit(1)
    print("[probe] PASS")


if __name__ == "__main__":
    main()

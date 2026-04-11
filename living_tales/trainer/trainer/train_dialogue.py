"""
Train a Living Tales dialogue transformer via supervised KD + REINFORCE.

Stage 1 — Supervised knowledge distillation:
    Generate interleaved (player, engine) dialogue trajectories using the
    Hopfield graph. Train the transformer on next-token prediction with
    soft targets from the Hopfield attractor weights.

    Loss = (1-α) * CE(logits, hard_target)
         + α * T² * KL(softmax(logits/T) ∥ soft_target)
         + λ * LyapunovReg(energies)

Stage 2 — REINFORCE fine-tuning:
    Play dialogue games with the model. Reward energy decreases,
    chronological compliance, and tag diversity. A KD anchor term
    prevents drift from the proven Hopfield structure.

Output: outputs/<case>/dialogue_model.pt
"""
from __future__ import annotations

import datetime
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.cartridge import CartridgeSpec
from core.creature_case import classify_creature_token_role
from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency
from generator.dialogue_sampler import (
    DialogueSampler, DialoguePath, SceneDialoguePath, ROLE_PLAYER, ROLE_ENGINE,
)
from trainer.dialogue_model import DialogueTransformer, SceneTransformer
from trainer.loss import LyapunovRegularization
from rl.dialogue_rewards import DialogueRewardConfig, DialogueRewardComputer
from trainer.training_profile import TrainingProfile, build_head_vocab_masks


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _banner(title: str, width: int = 62) -> None:
    bar = "=" * width
    pad = (width - len(title) - 2) // 2
    print(f"\n{bar}", flush=True)
    print(f"{'':>{pad}}  {title}", flush=True)
    print(bar, flush=True)


def _section(title: str, width: int = 62) -> None:
    print(f"\n{'─' * width}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'─' * width}", flush=True)


# ---------------------------------------------------------------------------
# Mapping helpers (shared with train_mystery / train_policy)
# ---------------------------------------------------------------------------

def _build_mappings(tokens: List[Token]):
    id_to_idx = {t.id: i for i, t in enumerate(tokens)}
    class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
    phase_to_idx = {p.value: i for i, p in enumerate(TokenPhase)}
    stream_to_idx = {s.value: i for i, s in enumerate(TokenStream)}
    agency_to_idx = {a.value: i for i, a in enumerate(TokenAgency)}
    return id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx


def _encode_token(token: Token, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx):
    return (
        id_to_idx[token.id],
        class_to_idx[token.token_class.value],
        phase_to_idx[token.phase.value],
        stream_to_idx[token.stream.value],
        agency_to_idx[token.agency.value],
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DialogueExample:
    """One training sequence: a full dialogue trajectory."""
    token_ids: List[int]
    class_ids: List[int]
    phase_ids: List[int]
    stream_ids: List[int]
    agency_ids: List[int]
    roles: List[str]
    energies: List[float]
    soft_targets: List[np.ndarray]  # per-step (V,)


def _dialogue_to_example(
    path: DialoguePath, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
) -> DialogueExample:
    token_ids, class_ids, phase_ids, stream_ids, agency_ids = [], [], [], [], []
    energies = []

    for turn in path.turns:
        tid, cid, pid, sid, aid = _encode_token(
            turn.token, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
        )
        token_ids.append(tid)
        class_ids.append(cid)
        phase_ids.append(pid)
        stream_ids.append(sid)
        agency_ids.append(aid)
        energies.append(turn.energy_at_step)

    return DialogueExample(
        token_ids=token_ids,
        class_ids=class_ids,
        phase_ids=phase_ids,
        stream_ids=stream_ids,
        agency_ids=agency_ids,
        roles=[turn.role for turn in path.turns],
        energies=energies,
        soft_targets=path.soft_targets,
    )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

PAD_IDX = 0


def _collate_dialogues(
    examples: List[DialogueExample],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Pad sequences and return batch tensors for teacher-forced training."""
    max_len = max(len(ex.token_ids) for ex in examples)
    B = len(examples)

    token_ids = np.full((B, max_len), PAD_IDX, dtype=np.int64)
    class_ids = np.full((B, max_len), 0, dtype=np.int64)
    phase_ids = np.full((B, max_len), 0, dtype=np.int64)
    stream_ids = np.full((B, max_len), 0, dtype=np.int64)
    agency_ids = np.full((B, max_len), 0, dtype=np.int64)
    padding_mask = np.ones((B, max_len), dtype=bool)
    targets = np.full((B, max_len), -100, dtype=np.int64)  # -100 = ignore in CE
    engine_targets = np.zeros((B, max_len), dtype=bool)
    energies = np.zeros((B, max_len), dtype=np.float32)

    V = examples[0].soft_targets[0].shape[0] if examples[0].soft_targets else 1
    soft_targets = np.zeros((B, max_len, V), dtype=np.float32)

    for i, ex in enumerate(examples):
        n = len(ex.token_ids)
        token_ids[i, :n] = ex.token_ids
        class_ids[i, :n] = ex.class_ids
        phase_ids[i, :n] = ex.phase_ids
        stream_ids[i, :n] = ex.stream_ids
        agency_ids[i, :n] = ex.agency_ids
        padding_mask[i, :n] = False
        energies[i, :n] = ex.energies

        # Teacher-forced targets: predict next token (shifted by 1)
        if n > 1:
            targets[i, :n - 1] = ex.token_ids[1:]
            engine_targets[i, :n - 1] = np.array(
                [role == ROLE_ENGINE for role in ex.roles[1:]],
                dtype=bool,
            )

        for t in range(min(n, len(ex.soft_targets))):
            soft_targets[i, t, :] = ex.soft_targets[t]

    return {
        "token_ids": torch.tensor(token_ids, device=device),
        "class_ids": torch.tensor(class_ids, device=device),
        "phase_ids": torch.tensor(phase_ids, device=device),
        "stream_ids": torch.tensor(stream_ids, device=device),
        "agency_ids": torch.tensor(agency_ids, device=device),
        "padding_mask": torch.tensor(padding_mask, device=device),
        "targets": torch.tensor(targets, device=device),
        "engine_targets": torch.tensor(engine_targets, device=device),
        "energies": torch.tensor(energies, device=device),
        "soft_targets": torch.tensor(soft_targets, device=device),
    }


# ---------------------------------------------------------------------------
# Soft target builder (vocabulary-wide, from Hopfield attractor weights)
# ---------------------------------------------------------------------------

def _build_vocab_soft_targets(
    spec: CartridgeSpec, temperature: float, device: str,
) -> torch.Tensor:
    """
    Build a single (V,) soft target from attractor weights, averaged across dims.
    Used as KD anchor for REINFORCE.
    """
    all_weights = np.stack([t.attractor_weights for t in spec.tokens])  # (V, D)
    logits = all_weights / max(temperature, 1e-8)
    # Softmax per dim, then average
    exp_l = np.exp(logits - logits.max(axis=0, keepdims=True))
    soft_per_dim = exp_l / exp_l.sum(axis=0, keepdims=True)  # (V, D)
    soft = soft_per_dim.mean(axis=1)  # (V,)
    soft /= soft.sum()
    return torch.tensor(soft, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Inference probe — run during training to detect collapse
# ---------------------------------------------------------------------------

def _inference_probe(model, spec, id_to_idx, class_to_idx, phase_to_idx,
                     stream_to_idx, agency_to_idx, device):
    """
    Realistic inference probe: builds game-like context (opening + several
    player/engine exchanges) then tests action→response predictions.

    Two probe modes:
      SHORT — opening + 1 action (4 tokens, tests raw action mapping)
      GAME  — opening + 6 context tokens + action (10 tokens, matches gameplay)
    """
    from core.token import TokenAgency, TokenStream

    idx_to_id = {v: k for k, v in id_to_idx.items()}

    # Build engine mask
    engine_mask = torch.zeros(spec.vocab_size, dtype=torch.bool, device=device)
    for t in spec.tokens:
        if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED) and not t.is_invariant:
            engine_mask[id_to_idx[t.id]] = True

    if not engine_mask.any():
        return

    def _enc(tok):
        return (
            id_to_idx[tok.id], class_to_idx[tok.token_class.value],
            phase_to_idx[tok.phase.value], stream_to_idx[tok.stream.value],
            agency_to_idx[tok.agency.value],
        )

    # Build opening context
    opening_seqs = {'t': [], 'c': [], 'p': [], 's': [], 'a': []}
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        enc = _enc(tok)
        opening_seqs['t'].append(enc[0]); opening_seqs['c'].append(enc[1])
        opening_seqs['p'].append(enc[2]); opening_seqs['s'].append(enc[3])
        opening_seqs['a'].append(enc[4])

    # Build a game-like context: opening + 6 random EARLY tokens (3 player + 3 engine)
    context_pool = [
        t for t in spec.tokens
        if not t.is_invariant and t.stream != TokenStream.OPENING
        and t.phase.value == "EARLY"
    ]
    np.random.seed(42)  # deterministic probe
    game_ctx_tokens = []
    if len(context_pool) >= 6:
        chosen_ctx = list(np.random.choice(len(context_pool), size=6, replace=False))
        game_ctx_tokens = [context_pool[i] for i in chosen_ctx]

    game_seqs = {k: list(v) for k, v in opening_seqs.items()}
    for tok in game_ctx_tokens:
        enc = _enc(tok)
        game_seqs['t'].append(enc[0]); game_seqs['c'].append(enc[1])
        game_seqs['p'].append(enc[2]); game_seqs['s'].append(enc[3])
        game_seqs['a'].append(enc[4])

    # Test actions — auto-detect from case tokens (works for mysteries AND creatures)
    # Prefer action: tokens, fall back to any player-playable non-opening tokens
    probe_actions = [t.id for t in spec.tokens
                     if t.id.startswith('action:') and t.id in id_to_idx]
    if not probe_actions:
        # Mystery fallback: pick player-agency EARLY tokens as probe inputs
        probe_actions = [t.id for t in spec.tokens
                         if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
                         and not t.is_invariant and t.stream != TokenStream.OPENING
                         and t.id in id_to_idx]
    probe_actions = probe_actions[:5]  # limit to 5
    if not probe_actions:
        _log("  [PROBE] no probe actions found, skipping")
        return {}

    model.eval()

    def _run_probe(label, base_seqs, temp):
        _log(f"  [{label}] action → prediction (temp={temp}, ctx={len(base_seqs['t'])} tokens):")
        chosen_ids_local: List[str] = []
        entropies_local: List[float] = []
        with torch.no_grad():
            for action_id in probe_actions:
                tok = spec.get_token(action_id)
                enc = _enc(tok)
                test_t = torch.tensor([base_seqs['t'] + [enc[0]]], dtype=torch.long, device=device)
                test_c = torch.tensor([base_seqs['c'] + [enc[1]]], dtype=torch.long, device=device)
                test_p = torch.tensor([base_seqs['p'] + [enc[2]]], dtype=torch.long, device=device)
                test_s = torch.tensor([base_seqs['s'] + [enc[3]]], dtype=torch.long, device=device)
                test_a = torch.tensor([base_seqs['a'] + [enc[4]]], dtype=torch.long, device=device)

                chosen_idx, probs = model.predict_next(
                    test_t, test_c, test_p, test_s, test_a,
                    valid_mask=engine_mask, temperature=temp,
                )
                chosen_id = idx_to_id[chosen_idx]
                chosen_ids_local.append(chosen_id)

                masked_probs = probs.clone()
                masked_probs[~engine_mask] = 0
                masked_probs = masked_probs / masked_probs.sum().clamp(min=1e-12)
                ent = float(-(masked_probs * masked_probs.clamp(min=1e-12).log()).sum().item())
                entropies_local.append(ent)
                top3_idx = masked_probs.topk(3).indices.tolist()
                top3_str = ", ".join(f"{idx_to_id[i].split(':')[1]}({masked_probs[i]:.0%})" for i in top3_idx)

                action_short = action_id.split(':')[1]
                chosen_short = chosen_id.split(':')[1]
                _log(f"    {action_short:>15s} → {chosen_short:<25s}  [{top3_str}]")

        counts: Dict[str, int] = {}
        for tid in chosen_ids_local:
            counts[tid] = counts.get(tid, 0) + 1
        dominant_id = max(counts, key=counts.get)
        dominance_rate = counts[dominant_id] / max(len(chosen_ids_local), 1)
        mean_ent = float(np.mean(entropies_local)) if entropies_local else 0.0
        _log(
            f"  [{label}] diversity: "
            f"unique={len(counts)}/{len(chosen_ids_local)}  "
            f"dominant={dominant_id.split(':')[-1]}({dominance_rate:.0%})  "
            f"entropy={mean_ent:.2f}"
        )
        return {
            "unique_predictions": len(counts),
            "dominant_prediction": dominant_id,
            "dominance_rate": dominance_rate,
            "mean_entropy": mean_ent,
        }

    # Run both probes
    short_metrics = _run_probe("SHORT", opening_seqs, temp=0.3)
    game_metrics = _run_probe("GAME", game_seqs, temp=1.0)

    model.train()

    # Return game probe metrics as the primary signal
    game_metrics["short_probe"] = short_metrics
    return game_metrics


# ---------------------------------------------------------------------------
# Stage 1 — Supervised KD training
# ---------------------------------------------------------------------------

def train_dialogue_supervised(
    spec_path: str,
    output_dir: str,
    n_dialogues: int = 2000,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float | None = None,
    kd_temperature: float = 2.0,
    kd_alpha: float | None = None,
    lyapunov_weight: float = 0.1,
    freeze_emb_epochs: int = 5,
    model_size_override: str | None = None,
    device: str = "cpu",
) -> Tuple[DialogueTransformer, Dict]:
    """
    Train a DialogueTransformer on sampled dialogue trajectories with KD.
    """
    _banner("DIALOGUE TRANSFORMER — SUPERVISED KD")

    spec = CartridgeSpec.load(spec_path)

    # --- Training profile: auto-derived from case spec ---
    profile = TrainingProfile.from_spec(spec, model_size_override=model_size_override)
    _log(f"Profile: {profile.log_summary()}")

    # Allow explicit overrides, otherwise use profile
    if lr is None:
        lr = profile.lr
    if kd_alpha is None:
        kd_alpha = profile.kd_alpha
    mappings = _build_mappings(spec.tokens)
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = mappings

    # --- Sample dialogues ---
    _section("Sampling dialogue trajectories")
    sampler = DialogueSampler(
        spec,
        player_temperature=1.2,
        engine_temperature=0.8,
        soft_target_temperature=kd_temperature,
    )
    t0 = time.time()
    paths = sampler.sample_batch(n_dialogues, verbose=True)
    _log(f"Sampled {len(paths)} dialogues in {time.time() - t0:.1f}s")

    if not paths:
        raise RuntimeError("No dialogue paths sampled — check case specification.")

    converged = sum(1 for p in paths if p.converged)
    avg_len = np.mean([len(p.turns) for p in paths])
    _log(f"  converged: {converged}/{len(paths)} ({converged/len(paths):.1%})")
    _log(f"  avg length: {avg_len:.1f} turns")

    # --- Convert to training examples ---
    examples = [
        _dialogue_to_example(p, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx)
        for p in paths
    ]

    # --- Create model (architecture from profile) ---
    model = DialogueTransformer(
        vocab_size=spec.vocab_size,
        embedding_dim=profile.embedding_dim,
        context_dim=profile.context_dim,
        n_layers=profile.n_layers,
        n_heads=profile.n_heads,
        max_seq_len=128,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    _log(f"Model params: {param_count:,}")

    # --- Optimizer + cosine LR schedule ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)
    lyapunov_reg = LyapunovRegularization()

    # --- Training loop ---
    _section("Training")
    history = {"epoch_losses": [], "loss": 0.0, "probe_metrics": []}
    latest_probe = None

    for epoch in range(n_epochs):
        model.train()

        # Freeze embeddings for early epochs
        freeze = epoch < freeze_emb_epochs
        for p in model.token_embedding.parameters():
            p.requires_grad = not freeze

        # Shuffle and batch
        np.random.shuffle(examples)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start : batch_start + batch_size]
            batch = _collate_dialogues(batch_examples, device)

            # Forward pass
            logits = model(
                batch["token_ids"],
                batch["class_ids"],
                batch["phase_ids"],
                batch["stream_ids"],
                batch["agency_ids"],
                batch["padding_mask"],
            )  # (B, S, V)

            B, S, V = logits.shape
            targets = batch["targets"]  # (B, S)
            engine_targets = batch["engine_targets"]  # (B, S)
            train_mask = (targets != -100) & engine_targets

            # --- Hard CE loss (focal loss: γ from profile) ---
            flat_mask = train_mask.reshape(B * S)
            if flat_mask.any():
                flat_logits = logits.reshape(B * S, V)[flat_mask]
                flat_targets = targets.reshape(B * S)[flat_mask]
                with torch.no_grad():
                    p_t = F.softmax(flat_logits, dim=-1).gather(1, flat_targets.unsqueeze(1)).squeeze(1)
                    focal_weight = (1 - p_t) ** profile.focal_gamma
                per_sample_ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                ce_loss = (focal_weight * per_sample_ce).mean()
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # --- Soft KD loss ---
            # Compute KL divergence between student and teacher distributions
            soft_t = batch["soft_targets"]  # (B, S, V)
            # Only compute KD on engine targets, not every next-token position.
            valid = train_mask.unsqueeze(-1).expand_as(logits)  # (B, S, V)
            if valid.any():
                student_log = F.log_softmax(logits / kd_temperature, dim=-1)
                teacher = soft_t.clamp(min=1e-12)
                kl = F.kl_div(student_log, teacher, reduction="none")
                kl = (kl * valid.float()).sum() / valid.float().sum()
                kd_loss = kd_temperature ** 2 * kl
            else:
                kd_loss = torch.tensor(0.0, device=device)

            # --- Lyapunov regularization ---
            energies = batch["energies"]  # (B, S)
            lya_loss = lyapunov_reg(energies)

            # --- Entropy + diversity: adaptive, profile-driven anti-collapse ---
            pred_log_probs = F.log_softmax(logits, dim=-1)
            pred_probs = F.softmax(logits, dim=-1)
            pred_entropy = -(pred_probs * pred_log_probs).sum(dim=-1)  # (B, S)
            entropy_mask = train_mask.float()
            mean_entropy = (pred_entropy * entropy_mask).sum() / entropy_mask.sum().clamp(min=1)

            entropy_bonus = torch.tensor(0.0, device=device)
            diversity_loss = torch.tensor(0.0, device=device)

            warmup_done = epoch >= profile.warmup_epochs
            if warmup_done and mean_entropy.item() < profile.collapse_threshold:
                entropy_bonus = profile.entropy_coef * mean_entropy
                if flat_mask.any():
                    avg_pred = pred_probs.reshape(B * S, V)[flat_mask].mean(dim=0)
                    batch_entropy = -(avg_pred * avg_pred.clamp(min=1e-12).log()).sum()
                    max_entropy = math.log(V)
                    diversity_loss = 1.0 - batch_entropy / max_entropy

            # --- Total loss ---
            total = (
                (1 - kd_alpha) * ce_loss
                + kd_alpha * kd_loss
                + lyapunov_weight * lya_loss
                - entropy_bonus
                + profile.diversity_coef * diversity_loss
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch_losses"].append(avg_loss)
        history["loss"] = avg_loss
        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            frozen_tag = " [emb frozen]" if freeze else ""
            cur_lr = scheduler.get_last_lr()[0]
            _log(f"Epoch {epoch:>3d}/{n_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}{frozen_tag}")
        elif epoch % 5 == 0 and epoch <= 30:
            # Extra early logging to catch collapse forming
            _log(f"Epoch {epoch:>3d}/{n_epochs}  loss={avg_loss:.4f}")

        # --- Inference probe: epoch 10, then every 25 epochs ---
        if (epoch == 10) or (epoch % 25 == 0 and epoch > 0):
            latest_probe = _inference_probe(
                model, spec, id_to_idx, class_to_idx, phase_to_idx,
                stream_to_idx, agency_to_idx, device,
            )
            if latest_probe:
                history["probe_metrics"].append({"epoch": epoch, **latest_probe})

    if latest_probe is None:
        latest_probe = _inference_probe(
            model, spec, id_to_idx, class_to_idx, phase_to_idx,
            stream_to_idx, agency_to_idx, device,
        )
        if latest_probe:
            history["probe_metrics"].append({"epoch": n_epochs - 1, **latest_probe})

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    # --- Save checkpoint ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / "dialogue_model.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "spec_path": str(spec_path),
        "case_id": spec.case_id,
        "embedding_dim": profile.embedding_dim,
        "context_dim": profile.context_dim,
        "n_layers": profile.n_layers,
        "n_heads": profile.n_heads,
        "max_seq_len": 128,
        "vocab_size": spec.vocab_size,
        "model_type": "dialogue",
        "id_to_idx": id_to_idx,
        "class_to_idx": class_to_idx,
        "phase_to_idx": phase_to_idx,
        "stream_to_idx": stream_to_idx,
        "agency_to_idx": agency_to_idx,
    }, ckpt_path)
    _log(f"Saved supervised checkpoint: {ckpt_path}")

    return model, history


# ---------------------------------------------------------------------------
# Stage 2 — REINFORCE fine-tuning
# ---------------------------------------------------------------------------

def _compute_dialogue_reward(
    graph,
    context_ids_before: List[str],
    context_ids_after: List[str],
    token: Token,
    turn: int,
    max_turns: int,
    new_affinity_tags: set,
    prev_affinity_tags: set,
) -> float:
    """Per-turn reward for dialogue RL."""
    # Energy decrease
    energy_before = graph.subgraph_energy(context_ids_before) if context_ids_before else 0.0
    energy_after = graph.subgraph_energy(context_ids_after)
    energy_reward = energy_before - energy_after

    # Chronology bonus: token phase matches narrative stage
    game_turn = turn // 2
    chrono_bonus = 0.0
    if token.is_available_at_turn(game_turn):
        chrono_bonus = 0.2

    # Diversity: new affinity tags
    new_tags = new_affinity_tags - prev_affinity_tags
    diversity_bonus = min(1.0, len(new_tags) / 3.0)

    # Speed penalty (fading)
    speed_penalty = math.exp(-turn / max(max_turns, 1))

    return energy_reward + chrono_bonus + 0.1 * diversity_bonus - 0.3 * speed_penalty


def train_dialogue_rl(
    model: DialogueTransformer,
    spec_path: str,
    output_dir: str,
    n_episodes: int = 500,
    lr: float = 3e-5,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    kd_anchor_weight: float = 0.05,
    kd_temperature: float = 2.0,
    max_turns: int | None = None,
    reward_config: DialogueRewardConfig | None = None,
    device: str = "cpu",
) -> Tuple[DialogueTransformer, Dict]:
    """
    REINFORCE fine-tuning of the dialogue transformer.
    """
    _banner("DIALOGUE TRANSFORMER — REINFORCE")

    spec = CartridgeSpec.load(spec_path)
    profile = TrainingProfile.from_spec(spec)
    mappings = _build_mappings(spec.tokens)
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = mappings
    idx_to_token = {i: t for t, i in id_to_idx.items()}

    max_t = max_turns if max_turns is not None else spec.max_turns * 2
    min_t = spec.min_turns

    # Reward computer
    if reward_config is None:
        reward_config = DialogueRewardConfig()
    reward_computer = DialogueRewardComputer(
        config=reward_config,
        graph=spec.token_graph,
        max_turns=max_t,
        convergence_threshold=spec.convergence_threshold,
    )

    # KD anchor (soft target distribution)
    soft_target = _build_vocab_soft_targets(spec, kd_temperature, device)
    if spec.mode == "oscillating":
        kd_anchor_weight *= 0.25

    # Token pools
    player_tokens = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]
    engine_tokens = [
        t for t in spec.tokens
        if t.agency in (TokenAgency.ENGINE, TokenAgency.SHARED)
        and not t.is_invariant and t.stream != TokenStream.OPENING
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Rolling stats
    returns_window = deque(maxlen=50)
    convergence_window = deque(maxlen=50)

    history = {"episode_returns": [], "convergence_rate": 0.0, "loss": 0.0}

    _section("REINFORCE training")

    for ep in range(n_episodes):
        model.eval()

        # --- Play one dialogue episode ---
        convergence_dims = np.zeros(spec.n_attractor_dims, dtype=np.float32)
        placed_ids: set = set()
        context_ids: List[str] = []
        prev_tags: set = set()

        # Sequences for model input
        seq_token, seq_class, seq_phase, seq_stream, seq_agency = [], [], [], [], []

        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        engine_logits_for_kd: List[torch.Tensor] = []
        recent_player: List = []  # last 3 player tokens for responsiveness
        signal_history: List[float] = []  # attractor weight norms for pacing
        recent_engine_roles: deque[str] = deque(maxlen=4)
        unresolved_problem_roles: deque[str] = deque(maxlen=6)

        # Opening tokens (no gradient)
        for tid in spec.opening_token_ids:
            tok = spec.get_token(tid)
            placed_ids.add(tok.id)
            context_ids.append(tok.id)
            convergence_dims = np.minimum(
                1.0, convergence_dims + tok.attractor_weights * spec.convergence_rate,
            )
            t_enc = _encode_token(tok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx)
            seq_token.append(t_enc[0])
            seq_class.append(t_enc[1])
            seq_phase.append(t_enc[2])
            seq_stream.append(t_enc[3])
            seq_agency.append(t_enc[4])

        is_player_turn = True
        converged = False

        for turn in range(len(seq_token), max_t):
            game_turn = turn // 2

            if is_player_turn:
                # Player: random valid token from pool (simulated player)
                pool = [
                    t for t in player_tokens
                    if t.id not in placed_ids and t.is_available_at_turn(game_turn)
                ]
                if not pool:
                    break
                chosen_tok = pool[np.random.randint(len(pool))]

                # Encode and add (no gradient for player)
                t_enc = _encode_token(
                    chosen_tok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
                )
                seq_token.append(t_enc[0])
                seq_class.append(t_enc[1])
                seq_phase.append(t_enc[2])
                seq_stream.append(t_enc[3])
                seq_agency.append(t_enc[4])

                ids_before = list(context_ids)
                placed_ids.add(chosen_tok.id)
                context_ids.append(chosen_tok.id)

                convergence_dims = np.minimum(
                    1.0, convergence_dims + chosen_tok.attractor_weights * spec.convergence_rate,
                )
                conv_at = float(convergence_dims.min())

                r = reward_computer.compute_turn_reward(
                    ids_before, context_ids, chosen_tok, turn,
                    prev_tags, recent_player, signal_history, conv_at,
                )
                prev_tags |= set(chosen_tok.affinity_tags)
                recent_player.append(chosen_tok)
                if len(recent_player) > 3:
                    recent_player.pop(0)
                signal_history.append(float(np.linalg.norm(chosen_tok.attractor_weights)))
                rewards.append(r)
                log_probs.append(torch.tensor(0.0, device=device))  # no gradient
            else:
                # Engine: model predicts next token (WITH gradient)
                model.train()

                inp_t = torch.tensor([seq_token], dtype=torch.long, device=device)
                inp_c = torch.tensor([seq_class], dtype=torch.long, device=device)
                inp_p = torch.tensor([seq_phase], dtype=torch.long, device=device)
                inp_s = torch.tensor([seq_stream], dtype=torch.long, device=device)
                inp_a = torch.tensor([seq_agency], dtype=torch.long, device=device)
                pad = torch.zeros(1, len(seq_token), dtype=torch.bool, device=device)

                logits = model(inp_t, inp_c, inp_p, inp_s, inp_a, pad)
                last_logits = logits[0, -1, :]  # (V,)

                # Phase mask: only allow valid engine tokens
                valid_mask = torch.zeros(spec.vocab_size, dtype=torch.bool, device=device)
                for t in engine_tokens:
                    if t.id not in placed_ids and t.is_available_at_turn(game_turn):
                        valid_mask[id_to_idx[t.id]] = True

                if not valid_mask.any():
                    break

                masked_logits = last_logits.masked_fill(~valid_mask, float("-inf"))
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()
                lp = dist.log_prob(action)

                chosen_idx = action.item()
                chosen_id = idx_to_token[chosen_idx]
                chosen_tok = spec.get_token(chosen_id)
                chosen_role = classify_creature_token_role(chosen_tok.id) if spec.mode == "oscillating" else ""

                t_enc = _encode_token(
                    chosen_tok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
                )
                seq_token.append(t_enc[0])
                seq_class.append(t_enc[1])
                seq_phase.append(t_enc[2])
                seq_stream.append(t_enc[3])
                seq_agency.append(t_enc[4])

                ids_before = list(context_ids)
                placed_ids.add(chosen_tok.id)
                context_ids.append(chosen_tok.id)

                convergence_dims = np.minimum(
                    1.0, convergence_dims + chosen_tok.attractor_weights * spec.convergence_rate,
                )
                conv_at = float(convergence_dims.min())

                r = reward_computer.compute_turn_reward(
                    ids_before, context_ids, chosen_tok, turn,
                    prev_tags, recent_player, signal_history, conv_at,
                )
                prev_tags |= set(chosen_tok.affinity_tags)
                signal_history.append(float(np.linalg.norm(chosen_tok.attractor_weights)))
                engine_logits_for_kd.append(masked_logits)
                if spec.mode == "oscillating":
                    if chosen_role in {"decay", "decline", "need", "mood", "combo"}:
                        unresolved_problem_roles.append(chosen_role)
                    elif chosen_role == "recovery":
                        unresolved_problem_roles.clear()
                    repeats = sum(1 for role in recent_engine_roles if role == chosen_role)
                    if chosen_role in {"context", "state"}:
                        repeats += 1
                    if unresolved_problem_roles and chosen_role not in {"action", "recovery"}:
                        r -= 0.08 * len(unresolved_problem_roles)
                    if unresolved_problem_roles and chosen_role in {"action", "recovery"}:
                        r += 0.12 * len(unresolved_problem_roles)
                    r -= 0.12 * repeats
                    recent_engine_roles.append(chosen_role)
                rewards.append(r)
                log_probs.append(lp)

                model.eval()

            is_player_turn = not is_player_turn

            conv_score = float(convergence_dims.min())
            if conv_score >= spec.convergence_threshold and game_turn >= min_t:
                converged = True
                # Terminal shaping
                if min_t <= game_turn <= min_t + 5:
                    rewards[-1] += 1.0  # good pacing
                break

        if not rewards:
            continue

        # Terminal penalty for non-convergence
        if not converged:
            rewards[-1] -= 1.0
        if spec.mode == "oscillating" and unresolved_problem_roles:
            rewards[-1] -= 0.2 * len(unresolved_problem_roles)

        # --- Compute returns ---
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # --- Policy gradient loss ---
        policy_loss = torch.tensor(0.0, device=device)
        n_engine_turns = 0

        for i, (lp, G_i) in enumerate(zip(log_probs, returns_t)):
            if lp.requires_grad:
                policy_loss -= lp * G_i
                n_engine_turns += 1

        if n_engine_turns > 0:
            policy_loss /= n_engine_turns

            # Entropy floor: prevent RL from collapsing to one token
            entropy_bonus = torch.tensor(0.0, device=device)
            if engine_logits_for_kd:
                stacked = torch.stack(engine_logits_for_kd, dim=0)  # (N, V)
                rl_probs = F.softmax(stacked, dim=-1)
                rl_log_probs = F.log_softmax(stacked, dim=-1)
                rl_entropy = -(rl_probs * rl_log_probs).sum(dim=-1).mean()
                if rl_entropy.item() < profile.collapse_threshold:
                    entropy_bonus = profile.entropy_coef * rl_entropy

            total_loss = policy_loss - entropy_bonus

            # KD anchor: push model distribution toward Hopfield soft targets
            if kd_anchor_weight > 0 and engine_logits_for_kd:
                student_log = F.log_softmax(
                    torch.stack(engine_logits_for_kd, dim=0) / kd_temperature,
                    dim=-1,
                )
                teacher = soft_target.unsqueeze(0).expand_as(student_log)
                kl = F.kl_div(student_log, teacher, reduction="batchmean")
                total_loss += kd_anchor_weight * kd_temperature ** 2 * kl

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        ep_return = sum(rewards)
        returns_window.append(ep_return)
        convergence_window.append(1.0 if converged else 0.0)

        if (ep + 1) % 50 == 0 or ep == 0:
            mean_ret = np.mean(list(returns_window))
            conv_rate = np.mean(list(convergence_window))
            n_decay = sum(1 for t in seq_token if idx_to_token.get(t, '').startswith('decay:'))
            n_recovery = sum(1 for t in seq_token if idx_to_token.get(t, '').startswith('recovery:'))
            _log(
                f"Episode {ep + 1:>4d}/{n_episodes}  "
                f"return={mean_ret:+.2f}  "
                f"conv={conv_rate:.1%}  "
                f"turns={len(rewards)}  "
                f"decay={n_decay} recov={n_recovery}"
            )

        # --- RL inference probe every 100 episodes ---
        if (ep + 1) % 100 == 0:
            _log(f"  [RL PROBE] episode {ep+1}:")
            _inference_probe(
                model, spec, id_to_idx, class_to_idx, phase_to_idx,
                stream_to_idx, agency_to_idx, device,
            )

    history["convergence_rate"] = float(np.mean(list(convergence_window))) if convergence_window else 0.0
    history["loss"] = float(np.mean(list(returns_window))) if returns_window else 0.0

    # --- Save final checkpoint ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / "dialogue_model.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "spec_path": str(spec_path),
        "case_id": spec.case_id,
        "embedding_dim": model.embedding_dim,
        "context_dim": model.context_dim,
        "n_layers": len(model.transformer.layers),
        "n_heads": model.transformer.layers[0].self_attn.num_heads,
        "max_seq_len": model.max_seq_len,
        "vocab_size": spec.vocab_size,
        "model_type": "dialogue",
        "id_to_idx": id_to_idx,
        "class_to_idx": class_to_idx,
        "phase_to_idx": phase_to_idx,
        "stream_to_idx": stream_to_idx,
        "agency_to_idx": agency_to_idx,
    }, ckpt_path)
    _log(f"Saved RL checkpoint: {ckpt_path}")

    return model, history


# ---------------------------------------------------------------------------
# Combined: supervised KD then REINFORCE
# ---------------------------------------------------------------------------

def train_dialogue_cartridge(
    spec_path: str,
    output_dir: str,
    n_dialogues: int = 2000,
    n_epochs: int = 100,
    n_rl_episodes: int = 500,
    batch_size: int = 32,
    lr: float | None = None,
    rl_lr: float = 3e-5,
    kd_temperature: float = 2.0,
    kd_alpha: float | None = None,
    model_size_override: str | None = None,
    lyapunov_weight: float = 0.1,
    device: str = "cpu",
) -> Tuple[DialogueTransformer, Dict]:
    """
    Full dialogue training pipeline: supervised KD → REINFORCE.
    """
    model, sup_history = train_dialogue_supervised(
        spec_path=spec_path,
        output_dir=output_dir,
        n_dialogues=n_dialogues,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        kd_temperature=kd_temperature,
        kd_alpha=kd_alpha,
        lyapunov_weight=lyapunov_weight,
        model_size_override=model_size_override,
        device=device,
    )

    model, rl_history = train_dialogue_rl(
        model=model,
        spec_path=spec_path,
        output_dir=output_dir,
        n_episodes=n_rl_episodes,
        lr=rl_lr,
        kd_temperature=kd_temperature,
        device=device,
    )

    combined_history = {
        "supervised": sup_history,
        "rl": rl_history,
        "loss": rl_history.get("loss", sup_history.get("loss", 0.0)),
    }

    # Save combined history for visualization
    import json
    history_path = Path(output_dir) / "history.json"
    history_path.write_text(json.dumps(combined_history, indent=2, default=str))
    _log(f"Saved history: {history_path}")

    return model, combined_history


# ---------------------------------------------------------------------------
# Scene training — multi-head parallel prediction
# ---------------------------------------------------------------------------

def _scene_to_tensors(
    path: SceneDialoguePath,
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
    n_dims: int,
) -> dict:
    """Convert a SceneDialoguePath to training tensors."""
    token_ids, class_ids, phase_ids, stream_ids, agency_ids = [], [], [], [], []
    roles = []

    for tok in path.all_tokens:
        tid, cid, pid, sid, aid = _encode_token(
            tok, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
        )
        token_ids.append(tid)
        class_ids.append(cid)
        phase_ids.append(pid)
        stream_ids.append(sid)
        agency_ids.append(aid)

    return {
        "token_ids": token_ids,
        "class_ids": class_ids,
        "phase_ids": phase_ids,
        "stream_ids": stream_ids,
        "agency_ids": agency_ids,
        "roles": path.all_roles,
        "soft_targets": path.soft_targets,
    }


def _collate_scene_batch(examples: list, device: str, n_dims: int, vocab_size: int,
                         head_vocab_masks: torch.Tensor = None) -> dict:
    """Pad and batch scene examples."""
    max_len = max(len(ex["token_ids"]) for ex in examples)
    B = len(examples)

    token_ids = np.full((B, max_len), 0, dtype=np.int64)
    class_ids = np.full((B, max_len), 0, dtype=np.int64)
    phase_ids = np.full((B, max_len), 0, dtype=np.int64)
    stream_ids = np.full((B, max_len), 0, dtype=np.int64)
    agency_ids = np.full((B, max_len), 0, dtype=np.int64)
    padding_mask = np.ones((B, max_len), dtype=bool)

    # Per-head targets: for each position, which token should each head predict?
    # Shape: (B, max_len, n_dims) — target token index per head, -100 = ignore
    head_targets = np.full((B, max_len, n_dims), -100, dtype=np.int64)

    V = vocab_size
    soft_targets = np.zeros((B, max_len, V), dtype=np.float32)

    for i, ex in enumerate(examples):
        n = len(ex["token_ids"])
        token_ids[i, :n] = ex["token_ids"]
        class_ids[i, :n] = ex["class_ids"]
        phase_ids[i, :n] = ex["phase_ids"]
        stream_ids[i, :n] = ex["stream_ids"]
        agency_ids[i, :n] = ex["agency_ids"]
        padding_mask[i, :n] = False

        # Build head targets: look ahead from each position to find the next
        # engine token for each dimension
        # For simplicity: at each engine position, that token IS the target for its head
        # We use shifted targets: position t predicts what comes at t+1
        if n > 1:
            for t in range(n - 1):
                next_tok_idx = ex["token_ids"][t + 1]
                next_role = ex["roles"][t + 1]
                if next_role == ROLE_ENGINE:
                    # Only set target for heads where this token is in their vocabulary
                    for d in range(n_dims):
                        if head_vocab_masks is not None and not head_vocab_masks[d, next_tok_idx]:
                            continue  # token not in this head's vocab
                        head_targets[i, t, d] = next_tok_idx

        for t in range(min(n, len(ex["soft_targets"]))):
            soft_targets[i, t, :] = ex["soft_targets"][t]

    return {
        "token_ids": torch.tensor(token_ids, device=device),
        "class_ids": torch.tensor(class_ids, device=device),
        "phase_ids": torch.tensor(phase_ids, device=device),
        "stream_ids": torch.tensor(stream_ids, device=device),
        "agency_ids": torch.tensor(agency_ids, device=device),
        "padding_mask": torch.tensor(padding_mask, device=device),
        "head_targets": torch.tensor(head_targets, device=device),
        "soft_targets": torch.tensor(soft_targets, device=device),
    }


def train_scene_supervised(
    spec_path: str,
    output_dir: str,
    n_dialogues: int = 2000,
    n_epochs: int = 100,
    batch_size: int = 32,
    kd_temperature: float = 2.0,
    lyapunov_weight: float = 0.1,
    freeze_emb_epochs: int = 5,
    model_size_override: str | None = None,
    device: str = "cpu",
) -> Tuple[SceneTransformer, Dict]:
    """Train a SceneTransformer on scene dialogue trajectories."""
    _banner("SCENE TRANSFORMER — SUPERVISED KD")

    spec = CartridgeSpec.load(spec_path)
    profile = TrainingProfile.from_spec(spec, model_size_override=model_size_override)
    _log(f"Profile: {profile.log_summary()}")

    mappings = _build_mappings(spec.tokens)
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = mappings
    n_dims = spec.n_attractor_dims

    lr = profile.lr
    kd_alpha = profile.kd_alpha

    # --- Build head vocab masks ---
    head_masks = build_head_vocab_masks(spec)
    _log(f"Head masks: {n_dims} heads")
    for d in range(n_dims):
        n_tok = head_masks[d].sum().item()
        _log(f"  Head {d}: {n_tok} tokens")

    # --- Sample scene dialogues ---
    _section("Sampling scene dialogues")
    sampler = DialogueSampler(
        spec,
        player_temperature=1.2,
        engine_temperature=0.8,
        soft_target_temperature=kd_temperature,
    )
    t0 = time.time()
    paths = sampler.sample_scene_batch(n_dialogues, verbose=True)
    _log(f"Sampled {len(paths)} scene dialogues in {time.time() - t0:.1f}s")

    if not paths:
        raise RuntimeError("No scene dialogues sampled.")

    converged = sum(1 for p in paths if p.converged)
    avg_len = np.mean([len(p.all_tokens) for p in paths])
    _log(f"  converged: {converged}/{len(paths)} ({converged/len(paths):.1%})")
    _log(f"  avg length: {avg_len:.1f} tokens")

    # --- Convert to training examples ---
    examples = [
        _scene_to_tensors(p, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx, n_dims)
        for p in paths
    ]

    # --- Create model ---
    model = SceneTransformer(
        vocab_size=spec.vocab_size,
        embedding_dim=profile.embedding_dim,
        context_dim=profile.context_dim,
        n_heads=profile.n_heads,
        n_layers=profile.n_layers,
        n_output_heads=n_dims,
        head_vocab_masks=head_masks,
        max_seq_len=128,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    _log(f"Model params: {param_count:,}")

    # --- Optimizer + cosine LR ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)

    # --- Training loop ---
    _section("Training")
    history = {"epoch_losses": [], "loss": 0.0}
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()

        freeze = epoch < freeze_emb_epochs
        for p in model.token_embedding.parameters():
            p.requires_grad = not freeze

        np.random.shuffle(examples)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start:batch_start + batch_size]
            batch = _collate_scene_batch(batch_examples, device, n_dims, spec.vocab_size, head_masks)

            # Forward: get logits per head
            all_logits = model(
                batch["token_ids"], batch["class_ids"], batch["phase_ids"],
                batch["stream_ids"], batch["agency_ids"], batch["padding_mask"],
            )  # list of n_dims tensors, each (B, S, V)

            # --- Per-head CE loss ---
            total_ce = torch.tensor(0.0, device=device)
            n_valid_heads = 0

            for d in range(n_dims):
                logits_d = all_logits[d]  # (B, S, V)
                targets_d = batch["head_targets"][:, :, d]  # (B, S)
                B, S, V = logits_d.shape

                # Only compute loss where we have targets
                valid = targets_d != -100
                if not valid.any():
                    continue

                flat_logits = logits_d.reshape(B * S, V)[valid.reshape(B * S)]
                flat_targets = targets_d.reshape(B * S)[valid.reshape(B * S)]

                # Focal loss
                with torch.no_grad():
                    p_t = F.softmax(flat_logits, dim=-1).gather(1, flat_targets.unsqueeze(1)).squeeze(1)
                    focal_w = (1 - p_t) ** profile.focal_gamma
                per_sample = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                head_loss = (focal_w * per_sample).mean()
                total_ce = total_ce + head_loss
                n_valid_heads += 1

            if n_valid_heads > 0:
                total_ce = total_ce / n_valid_heads

            # --- Entropy bonus (per head, adaptive) ---
            entropy_bonus = torch.tensor(0.0, device=device)
            warmup_done = epoch >= profile.warmup_epochs

            if warmup_done:
                for d in range(n_dims):
                    logits_d = all_logits[d]
                    probs_d = F.softmax(logits_d, dim=-1)
                    log_probs_d = F.log_softmax(logits_d, dim=-1)
                    ent_d = -(probs_d * log_probs_d).sum(dim=-1).mean()
                    if ent_d.item() < profile.collapse_threshold:
                        entropy_bonus = entropy_bonus + profile.entropy_coef * ent_d

                if n_dims > 0:
                    entropy_bonus = entropy_bonus / n_dims

            # Batch diversity loss (per-head) — penalize all heads predicting same thing
            diversity_loss = torch.tensor(0.0, device=device)
            if warmup_done:
                for d in range(n_dims):
                    logits_d = all_logits[d]
                    probs_d = F.softmax(logits_d, dim=-1)
                    avg_pred = probs_d.mean(dim=(0, 1))  # (V,) batch-average prediction
                    batch_ent = -(avg_pred * avg_pred.clamp(min=1e-12).log()).sum()
                    n_valid = float(model.head_vocab_masks[d].sum()) if hasattr(model, 'head_vocab_masks') else float(logits_d.shape[-1])
                    max_ent = math.log(max(n_valid, 2.0))
                    diversity_loss = diversity_loss + (1.0 - batch_ent / max_ent)
                diversity_loss = diversity_loss / max(n_dims, 1)

            # Per-head collapse penalty — penalize heads with >80% top-1 probability
            collapse_penalty = torch.tensor(0.0, device=device)
            if warmup_done:
                for d in range(n_dims):
                    probs_d = F.softmax(all_logits[d], dim=-1)
                    top_prob = probs_d.max(dim=-1).values.mean()
                    if top_prob.item() > 0.80:
                        collapse_penalty = collapse_penalty + (top_prob - 0.80)
                collapse_penalty = collapse_penalty / max(n_dims, 1)

            # --- Total loss ---
            total = (1 - profile.kd_alpha) * total_ce - entropy_bonus + profile.diversity_coef * diversity_loss + 0.5 * collapse_penalty

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch_losses"].append(avg_loss)
        history["loss"] = avg_loss
        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            frozen_tag = " [emb frozen]" if freeze else ""
            cur_lr = scheduler.get_last_lr()[0]
            _log(f"Epoch {epoch:>3d}/{n_epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  div={diversity_loss.item():.4f}  col={collapse_penalty.item():.4f}{frozen_tag}")

        # --- Scene probe every 25 epochs ---
        if (epoch == 10) or (epoch % 25 == 0 and epoch > 0):
            _scene_probe(model, spec, id_to_idx, class_to_idx, phase_to_idx,
                        stream_to_idx, agency_to_idx, device)

        # --- Early stopping: patience 50 epochs after warmup ---
        if epoch > profile.warmup_epochs:
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 50:
                _log(f"Early stopping at epoch {epoch} (no improvement for 50 epochs, best={best_loss:.4f})")
                history["early_stopped"] = epoch
                break

    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True

    # --- Save checkpoint ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path / "dialogue_model.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "spec_path": str(spec_path),
        "case_id": spec.case_id,
        "model_type": "scene",
        "embedding_dim": profile.embedding_dim,
        "context_dim": profile.context_dim,
        "n_layers": profile.n_layers,
        "n_heads": profile.n_heads,
        "n_output_heads": n_dims,
        "max_seq_len": 128,
        "vocab_size": spec.vocab_size,
        "id_to_idx": id_to_idx,
        "class_to_idx": class_to_idx,
        "phase_to_idx": phase_to_idx,
        "stream_to_idx": stream_to_idx,
        "agency_to_idx": agency_to_idx,
    }, ckpt_path)
    _log(f"Saved scene checkpoint: {ckpt_path}")

    return model, history


def _scene_probe(model, spec, id_to_idx, class_to_idx, phase_to_idx,
                 stream_to_idx, agency_to_idx, device):
    """Probe: play one action and show what each head predicts."""
    from core.token import TokenAgency, TokenStream

    idx_to_id = {v: k for k, v in id_to_idx.items()}

    def _enc(tok):
        return (
            id_to_idx[tok.id], class_to_idx[tok.token_class.value],
            phase_to_idx[tok.phase.value], stream_to_idx[tok.stream.value],
            agency_to_idx[tok.agency.value],
        )

    # Build opening context
    seq_t, seq_c, seq_p, seq_s, seq_a = [], [], [], [], []
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        enc = _enc(tok)
        seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
        seq_s.append(enc[3]); seq_a.append(enc[4])

    # Find probe actions
    probe_actions = [t.id for t in spec.tokens
                     if t.id.startswith('action:') and t.id in id_to_idx]
    if not probe_actions:
        probe_actions = [t.id for t in spec.tokens
                         if t.agency in (TokenAgency.PLAYER, TokenAgency.SHARED)
                         and not t.is_invariant and t.stream != TokenStream.OPENING
                         and t.id in id_to_idx]
    probe_actions = probe_actions[:3]
    if not probe_actions:
        return

    model.eval()
    _log(f"  [SCENE PROBE] action → scene prediction ({spec.n_attractor_dims} heads):")

    with torch.no_grad():
        for action_id in probe_actions:
            tok = spec.get_token(action_id)
            enc = _enc(tok)
            test_t = torch.tensor([seq_t + [enc[0]]], dtype=torch.long, device=device)
            test_c = torch.tensor([seq_c + [enc[1]]], dtype=torch.long, device=device)
            test_p = torch.tensor([seq_p + [enc[2]]], dtype=torch.long, device=device)
            test_s = torch.tensor([seq_s + [enc[3]]], dtype=torch.long, device=device)
            test_a = torch.tensor([seq_a + [enc[4]]], dtype=torch.long, device=device)

            results = model.predict_scene(test_t, test_c, test_p, test_s, test_a, temperature=0.8)

            action_short = action_id.split(':')[1]
            scene_parts = []
            for d, (chosen_idx, probs) in enumerate(results):
                if chosen_idx >= 0:
                    tok_name = idx_to_id[chosen_idx].split(':')[1]
                    prob = probs[chosen_idx].item()
                    scene_parts.append(f"d{d}:{tok_name}({prob:.0%})")
                else:
                    scene_parts.append(f"d{d}:NONE")

            _log(f"    {action_short:>15s} → [{', '.join(scene_parts)}]")

    model.train()


def train_scene_cartridge(
    spec_path: str,
    output_dir: str,
    n_dialogues: int = 2000,
    n_epochs: int = 100,
    n_rl_episodes: int = 500,
    batch_size: int = 32,
    kd_temperature: float = 2.0,
    model_size_override: str | None = None,
    device: str = "cpu",
) -> Tuple[SceneTransformer, Dict]:
    """Full scene training pipeline: supervised KD (RL TBD in next iteration)."""
    model, sup_history = train_scene_supervised(
        spec_path=spec_path,
        output_dir=output_dir,
        n_dialogues=n_dialogues,
        n_epochs=n_epochs,
        batch_size=batch_size,
        kd_temperature=kd_temperature,
        model_size_override=model_size_override,
        device=device,
    )

    # TODO: Scene RL fine-tuning (play N-token scenes, reward per scene)
    # For now, supervised-only is enough to test the multi-head architecture

    combined_history = {
        "supervised": sup_history,
        "loss": sup_history.get("loss", 0.0),
    }

    import json
    history_path = Path(output_dir) / "history.json"
    history_path.write_text(json.dumps(combined_history, indent=2, default=str))
    _log(f"Saved history: {history_path}")

    return model, combined_history

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
from generator.dialogue_sampler import DialogueSampler, DialoguePath, ROLE_PLAYER, ROLE_ENGINE
from trainer.dialogue_model import DialogueTransformer
from trainer.loss import LyapunovRegularization
from rl.dialogue_rewards import DialogueRewardConfig, DialogueRewardComputer


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
    """Sample action→response predictions to monitor training quality."""
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
    seq_t, seq_c, seq_p, seq_s, seq_a = [], [], [], [], []
    for tid in spec.opening_token_ids:
        tok = spec.get_token(tid)
        enc = _enc(tok)
        seq_t.append(enc[0]); seq_c.append(enc[1]); seq_p.append(enc[2])
        seq_s.append(enc[3]); seq_a.append(enc[4])

    # Test a few actions
    probe_actions = ['action:fill_bowl', 'action:toss_ball', 'action:scratch_chin',
                     'action:brush_coat', 'action:dim_lamp']
    probe_actions = [a for a in probe_actions if a in id_to_idx]

    if not probe_actions:
        return

    model.eval()
    _log("  [PROBE] action → model prediction:")
    chosen_ids: List[str] = []
    entropies: List[float] = []
    with torch.no_grad():
        for action_id in probe_actions:
            tok = spec.get_token(action_id)
            enc = _enc(tok)
            test_t = torch.tensor([seq_t + [enc[0]]], dtype=torch.long, device=device)
            test_c = torch.tensor([seq_c + [enc[1]]], dtype=torch.long, device=device)
            test_p = torch.tensor([seq_p + [enc[2]]], dtype=torch.long, device=device)
            test_s = torch.tensor([seq_s + [enc[3]]], dtype=torch.long, device=device)
            test_a = torch.tensor([seq_a + [enc[4]]], dtype=torch.long, device=device)

            chosen_idx, probs = model.predict_next(
                test_t, test_c, test_p, test_s, test_a,
                valid_mask=engine_mask, temperature=0.3,
            )
            chosen_id = idx_to_id[chosen_idx]
            chosen_ids.append(chosen_id)

            # Top 3
            masked_probs = probs.clone()
            masked_probs[~engine_mask] = 0
            masked_probs = masked_probs / masked_probs.sum().clamp(min=1e-12)
            entropies.append(float(-(masked_probs * masked_probs.clamp(min=1e-12).log()).sum().item()))
            top3_idx = masked_probs.topk(3).indices.tolist()
            top3_str = ", ".join(f"{idx_to_id[i].split(':')[1]}({masked_probs[i]:.0%})" for i in top3_idx)

            action_short = action_id.split(':')[1]
            chosen_short = chosen_id.split(':')[1]
            _log(f"    {action_short:>15s} → {chosen_short:<25s}  [{top3_str}]")
    model.train()
    counts: Dict[str, int] = {}
    for token_id in chosen_ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    dominant_id = max(counts, key=counts.get)
    dominance_rate = counts[dominant_id] / max(len(chosen_ids), 1)
    metrics = {
        "unique_predictions": len(counts),
        "dominant_prediction": dominant_id,
        "dominance_rate": dominance_rate,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }
    _log(
        "  [PROBE] diversity: "
        f"unique={metrics['unique_predictions']}/{len(chosen_ids)}  "
        f"dominant={dominant_id.split(':')[-1]}({dominance_rate:.0%})  "
        f"entropy={metrics['mean_entropy']:.2f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage 1 — Supervised KD training
# ---------------------------------------------------------------------------

def train_dialogue_supervised(
    spec_path: str,
    output_dir: str,
    n_dialogues: int = 2000,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.3,
    lyapunov_weight: float = 0.1,
    freeze_emb_epochs: int = 5,
    device: str = "cpu",
) -> Tuple[DialogueTransformer, Dict]:
    """
    Train a DialogueTransformer on sampled dialogue trajectories with KD.
    """
    _banner("DIALOGUE TRANSFORMER — SUPERVISED KD")

    spec = CartridgeSpec.load(spec_path)
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

    # --- Create model ---
    # Use 4 layers for better action→response learning
    model = DialogueTransformer(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_layers=4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    _log(f"Model params: {param_count:,}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

            # --- Hard CE loss ---
            flat_mask = train_mask.reshape(B * S)
            if flat_mask.any():
                flat_logits = logits.reshape(B * S, V)[flat_mask]
                flat_targets = targets.reshape(B * S)[flat_mask]
                ce_loss = F.cross_entropy(flat_logits, flat_targets)
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

            # --- Entropy bonus: penalize collapsed predictions ---
            pred_log_probs = F.log_softmax(logits, dim=-1)
            pred_probs = F.softmax(logits, dim=-1)
            pred_entropy = -(pred_probs * pred_log_probs).sum(dim=-1)  # (B, S)
            entropy_mask = train_mask.float()
            mean_entropy = (pred_entropy * entropy_mask).sum() / entropy_mask.sum().clamp(min=1)

            # --- Total loss ---
            total = (
                (1 - kd_alpha) * ce_loss
                + kd_alpha * kd_loss
                + lyapunov_weight * lya_loss
                - 0.05 * mean_entropy  # reward diverse predictions
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

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            frozen_tag = " [emb frozen]" if freeze else ""
            _log(f"Epoch {epoch:>3d}/{n_epochs}  loss={avg_loss:.4f}{frozen_tag}")

        # --- Inference probe every 25 epochs ---
        if epoch % 25 == 0 and epoch > 0:
            latest_probe = _inference_probe(
                model, spec, id_to_idx, class_to_idx, phase_to_idx,
                stream_to_idx, agency_to_idx, device,
            )
            history["probe_metrics"].append({"epoch": epoch, **latest_probe})

    if latest_probe is None:
        latest_probe = _inference_probe(
            model, spec, id_to_idx, class_to_idx, phase_to_idx,
            stream_to_idx, agency_to_idx, device,
        )
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
        "embedding_dim": spec.embedding_dim,
        "context_dim": spec.context_dim,
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
        entropy_loss = torch.tensor(0.0, device=device)
        n_engine_turns = 0

        for i, (lp, G_i) in enumerate(zip(log_probs, returns_t)):
            if lp.requires_grad:
                policy_loss -= lp * G_i
                n_engine_turns += 1

        if n_engine_turns > 0:
            policy_loss /= n_engine_turns

            # Entropy bonus (encourage exploration)
            # Recompute last engine logits for entropy
            # (simplified: use mean log_prob as proxy)

            total_loss = policy_loss

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
            _log(
                f"Episode {ep + 1:>4d}/{n_episodes}  "
                f"return={mean_ret:+.2f}  "
                f"conv={conv_rate:.1%}  "
                f"turns={len(rewards)}"
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
        "embedding_dim": spec.embedding_dim,
        "context_dim": spec.context_dim,
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
    lr: float = 1e-3,
    rl_lr: float = 3e-5,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.3,
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

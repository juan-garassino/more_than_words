"""
Train a Thornfield transformer policy via supervised pretraining + REINFORCE.

Stage 1 — Knowledge distillation from Hopfield attractor weights:
    The Hopfield graph encodes the solution as energy minima. Each token carries
    attractor_weights[d] ∈ [0,1] for each dimension d, expressing how strongly
    that token points toward the solution in that dimension.

    Strict KD uses these weights as SOFT TARGETS — a probability distribution
    over the full vocabulary for each dimension — rather than a hard one-hot on
    the single invariant token.

        soft_targets[d] = softmax(attractor_weights[:, d] / T)

    This tells the policy not just "the answer is renard_voss" but "renard_voss
    is most likely (0.72), followed by tokens that partially indicate dim 0
    (0.08, 0.05, ...)". The policy learns the full energy gradient, not just
    the minimum.

    Loss = α * CrossEntropy(logits, hard_target)          ← correctness anchor
         + (1-α) * T² * KL(softmax(logits/T) ∥ soft_targets)  ← gradient shape

    Default: α=0.3, T=2.0  (mostly soft targets)

Stage 2 — REINFORCE fine-tuning with KD anchor:
    RL explores beyond the demonstrations with game-feel rewards (energy, timing,
    diversity). A small KD regularisation term (kd_coef * KL) keeps the policy
    anchored to the Hopfield energy landscape — prevents drifting to low-energy
    but narratively incoherent triads.

Output: outputs/<case>/policy.pt
"""
from __future__ import annotations

import argparse
import datetime
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.cartridge import CartridgeSpec
from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency
from generator.path_sampler import PathSampler
from rl.casebook_env import CasebookEnv
from trainer.energy_model import MysteryEnergyModel


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
# Mapping helpers
# ---------------------------------------------------------------------------

def _build_mappings(tokens: List[Token]):
    id_to_idx = {t.id: i for i, t in enumerate(tokens)}
    class_to_idx = {c.value: i for i, c in enumerate(TokenClass)}
    phase_to_idx = {p.value: i for i, p in enumerate(TokenPhase)}
    stream_to_idx = {s.value: i for i, s in enumerate(TokenStream)}
    agency_to_idx = {a.value: i for i, a in enumerate(TokenAgency)}
    return id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx


def _get_retrieval_logits(
    model: MysteryEnergyModel,
    placed_tokens: List[Token],
    placed_positions: List[Tuple[int, int]],
    id_to_idx: Dict[str, int],
    class_to_idx: Dict[str, int],
    phase_to_idx: Dict[str, int],
    stream_to_idx: Dict[str, int],
    agency_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Compute retrieval logits (1, n_dims, vocab_size) via model sub-modules."""
    n = len(placed_tokens)
    token_ids = torch.tensor(
        [id_to_idx[t.id] for t in placed_tokens], dtype=torch.long, device=device
    )
    class_ids = torch.tensor(
        [class_to_idx[t.token_class.value] for t in placed_tokens],
        dtype=torch.long, device=device,
    )
    phase_ids = torch.tensor(
        [phase_to_idx[t.phase.value] for t in placed_tokens],
        dtype=torch.long, device=device,
    )
    stream_ids = torch.tensor(
        [stream_to_idx[t.stream.value] for t in placed_tokens],
        dtype=torch.long, device=device,
    )
    agency_ids = torch.tensor(
        [agency_to_idx[t.agency.value] for t in placed_tokens],
        dtype=torch.long, device=device,
    )
    pos_t = torch.tensor(placed_positions, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.zeros(1, n, dtype=torch.bool, device=device)

    placed_emb = model.token_embedding(token_ids, class_ids, phase_ids, stream_ids, agency_ids)
    placed_emb = placed_emb.unsqueeze(0)
    context = model.casebook_encoder(placed_emb, pos_t, mask)

    all_ids = torch.arange(model.vocab_size, device=device)
    all_embs = model.token_embedding.token_emb(all_ids)
    return model.retrieval_head(context, all_embs)  # (1, n_dims, V)


def _obs_to_tokens_positions(
    obs: dict,
    spec: CartridgeSpec,
) -> Tuple[List[Token], List[Tuple[int, int]]]:
    placed_ids = obs["placed_token_ids"]
    tokens = [spec.get_token(tid) for tid in placed_ids]
    positions = [(min(i // 3, 7), i % 3) for i in range(len(tokens))]
    return tokens, positions


# ---------------------------------------------------------------------------
# Triad sampling from logits (used in REINFORCE)
# ---------------------------------------------------------------------------

def _sample_triad_from_logits(
    logits: torch.Tensor,
    spec: CartridgeSpec,
    id_to_idx: Dict[str, int],
    placed_ids: set,
) -> Tuple[Optional[List[str]], Optional[torch.Tensor]]:
    """
    Sample a valid triad (3 tokens of distinct classes, at least one graph edge).
    Returns (token_id_list, summed_log_prob) or (None, None).
    """
    idx_to_token = {i: t for i, t in enumerate(spec.tokens)}
    n_dims = logits.size(1)
    non_inv_ids = {t.id for t in spec.tokens if not t.is_invariant}
    available_ids = non_inv_ids - placed_ids

    for _ in range(20):
        selected_tokens: List[Token] = []
        log_probs: List[torch.Tensor] = []
        selected_ids_this: set = set()
        selected_classes: set = set()
        valid = True

        for d in range(n_dims):
            dim_logits = logits[0, d, :]
            inf_mask = torch.full_like(dim_logits, float("-inf"))
            valid_indices = [
                id_to_idx[tid]
                for tid in available_ids
                if tid not in selected_ids_this
                and idx_to_token[id_to_idx[tid]].token_class.value not in selected_classes
            ]
            if not valid_indices:
                valid = False
                break
            inf_mask[valid_indices] = 0.0
            probs = F.softmax(dim_logits + inf_mask, dim=0)
            cat = torch.distributions.Categorical(probs)
            choice = cat.sample()
            log_probs.append(cat.log_prob(choice))
            tok = idx_to_token[choice.item()]
            selected_tokens.append(tok)
            selected_ids_this.add(tok.id)
            selected_classes.add(tok.token_class.value)

        if not valid or len(selected_tokens) != n_dims:
            continue

        ids = [t.id for t in selected_tokens]
        graph = spec.token_graph
        has_edge = any(
            graph.weight(ids[i], ids[j]) > 0.05
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        )
        if has_edge:
            return ids, torch.stack(log_probs).sum()

    # Fallback: first valid triad of different classes (no gradient)
    available = [t for t in spec.tokens if not t.is_invariant and t.id not in placed_ids]
    seen_classes: set = set()
    fallback: List[str] = []
    for t in available:
        if t.token_class.value not in seen_classes:
            fallback.append(t.id)
            seen_classes.add(t.token_class.value)
            if len(fallback) == 3:
                return fallback, None
    return None, None


def _mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits[0], dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Strict knowledge distillation — soft targets from Hopfield attractor weights
# ---------------------------------------------------------------------------

def _build_soft_targets(
    spec: CartridgeSpec,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Build soft target distributions directly from the Hopfield attractor weights.

    For each dimension d:
        soft_targets[d] = softmax(attractor_weights_all_tokens[:, d] / temperature)

    This is the Hopfield energy gradient expressed as a probability distribution.
    Every token that points toward dimension d receives proportional credit —
    not just the invariant (argmax), but all supporting tokens.

    Temperature controls sharpness:
        T → 0  : approaches hard one-hot (same as cross-entropy)
        T = 2  : peaked but distributes mass to supporters (recommended)
        T → ∞  : uniform over all tokens (no signal)

    Returns: (n_dims, vocab_size) float tensor
    """
    all_weights = torch.tensor(
        np.stack([t.attractor_weights for t in spec.tokens]),
        dtype=torch.float32, device=device,
    )  # (V, n_dims)
    soft = F.softmax(all_weights.T / temperature, dim=-1)  # (n_dims, V)

    # Log teacher distribution statistics once
    entropies = -(soft * soft.clamp(min=1e-12).log()).sum(dim=-1)
    for d in range(spec.n_attractor_dims):
        inv_id = spec.invariant_token_ids[d]
        inv_idx = next(i for i, t in enumerate(spec.tokens) if t.id == inv_id)
        inv_prob = soft[d, inv_idx].item()
        _log(
            f"[KD] dim {d}  teacher entropy={entropies[d].item():.3f}  "
            f"invariant_prob={inv_prob:.3f}  "
            f"(T={temperature:.1f})"
        )
    return soft


# ---------------------------------------------------------------------------
# Stage 1 — Supervised pretraining
# ---------------------------------------------------------------------------

def _build_supervised_examples(spec: CartridgeSpec, n_paths: int) -> list:
    _log(f"[SUPERVISED] Sampling {n_paths} training paths (allow_partial=True)...")
    sampler = PathSampler(spec, sampling_temperature=1.4, min_affinity=0.05, allow_partial=True)
    t0 = time.time()
    paths = sampler.sample_batch(n_paths, verbose=True)
    elapsed = time.time() - t0
    _log(f"[SUPERVISED] {len(paths)} paths sampled in {elapsed:.1f}s")

    id_to_idx = {t.id: i for i, t in enumerate(spec.tokens)}
    n_dims = spec.n_attractor_dims

    invariant_indices = np.zeros(n_dims, dtype=np.int64)
    for inv_id in spec.invariant_token_ids:
        tok = spec.get_token(inv_id)
        for d in range(n_dims):
            if tok.attractor_weights[d] > 0.5:
                invariant_indices[d] = id_to_idx[inv_id]

    examples = []
    for path in paths:
        context_tokens: List[Token] = []
        context_positions: List[Tuple[int, int]] = []
        for turn, triad in enumerate(path):
            if all(t.is_invariant for t in triad):
                break
            if context_tokens:
                examples.append({
                    "context_tokens": list(context_tokens),
                    "context_positions": list(context_positions),
                    "invariant_indices": invariant_indices.copy(),
                })
            row = min(turn, 7)
            for col, token in enumerate(triad):
                context_tokens.append(token)
                context_positions.append((row, col))

    _log(f"[SUPERVISED] Built {len(examples)} training examples from {len(paths)} paths")
    return examples


def _collate_batch(
    examples: list,
    id_to_idx: Dict[str, int],
    class_to_idx: Dict[str, int],
    phase_to_idx: Dict[str, int],
    stream_to_idx: Dict[str, int],
    agency_to_idx: Dict[str, int],
    n_dims: int,
    device: torch.device,
):
    """Collate variable-length examples into padded GPU tensors for a batched forward pass."""
    B = len(examples)
    max_len = max(len(ex["context_tokens"]) for ex in examples)

    tok_t   = torch.zeros(B, max_len, dtype=torch.long)
    cls_t   = torch.zeros(B, max_len, dtype=torch.long)
    phase_t = torch.zeros(B, max_len, dtype=torch.long)
    strm_t  = torch.zeros(B, max_len, dtype=torch.long)
    agcy_t  = torch.zeros(B, max_len, dtype=torch.long)
    pos_t   = torch.zeros(B, max_len, 2, dtype=torch.float32)
    mask_t  = torch.ones(B, max_len, dtype=torch.bool)   # True = padding (ignored by encoder)
    inv_t   = torch.zeros(B, n_dims, dtype=torch.long)

    for i, ex in enumerate(examples):
        toks = ex["context_tokens"]
        pos  = ex["context_positions"]
        n    = len(toks)
        tok_t[i, :n]   = torch.tensor([id_to_idx[t.id]                   for t in toks], dtype=torch.long)
        cls_t[i, :n]   = torch.tensor([class_to_idx[t.token_class.value] for t in toks], dtype=torch.long)
        phase_t[i, :n] = torch.tensor([phase_to_idx[t.phase.value]       for t in toks], dtype=torch.long)
        strm_t[i, :n]  = torch.tensor([stream_to_idx[t.stream.value]     for t in toks], dtype=torch.long)
        agcy_t[i, :n]  = torch.tensor([agency_to_idx[t.agency.value]     for t in toks], dtype=torch.long)
        pos_t[i, :n]   = torch.tensor(pos, dtype=torch.float32)
        mask_t[i, :n]  = False
        inv_t[i]       = torch.tensor(ex["invariant_indices"], dtype=torch.long)

    return (
        tok_t.to(device), cls_t.to(device), phase_t.to(device),
        strm_t.to(device), agcy_t.to(device), pos_t.to(device),
        mask_t.to(device), inv_t.to(device), max_len,
    )


def train_supervised(
    model: MysteryEnergyModel,
    spec: CartridgeSpec,
    n_paths: int,
    n_epochs: int,
    device: torch.device,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.3,
) -> None:
    _banner("STAGE 1 — KNOWLEDGE DISTILLATION (Hopfield → Transformer)")
    _log(
        f"[SUPERVISED] paths={n_paths}  epochs={n_epochs}  device={device}  "
        f"kd_temperature={kd_temperature}  kd_alpha={kd_alpha}"
    )
    _log(
        f"[SUPERVISED] loss = {kd_alpha:.1f} * hard_CE "
        f"+ {1-kd_alpha:.1f} * T² * KL(policy/T ∥ attractor_softmax/T)"
    )

    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = _build_mappings(spec.tokens)
    n_dims = spec.n_attractor_dims

    # Build soft targets from Hopfield attractor weights — the teacher distribution
    soft_targets = _build_soft_targets(spec, kd_temperature, device)  # (n_dims, V)

    examples = _build_supervised_examples(spec, n_paths)
    if not examples:
        _log("[SUPERVISED] ERROR — no examples generated, skipping supervised stage")
        return

    # Freeze TokenEmbedding for first 5 epochs
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    _log("[SUPERVISED] TokenEmbedding frozen for epochs 1-5")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    stage_t0 = time.time()
    loss_history: List[float] = []
    hard_history: List[float] = []
    soft_history: List[float] = []

    batch_size = 64
    for epoch in range(n_epochs):
        if epoch == 5:
            _log("[SUPERVISED] Unfreezing TokenEmbedding (epoch 6+)")
            for param in model.token_embedding.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        epoch_t0 = time.time()
        random.shuffle(examples)
        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        n_batches = 0

        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            tok_t, cls_t, phase_t, strm_t, agcy_t, pos_t, mask_t, inv_t, max_len = _collate_batch(
                batch, id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
                n_dims, device,
            )
            B = tok_t.size(0)

            # Batched forward pass — all on GPU
            placed_emb = model.token_embedding(
                tok_t.view(-1), cls_t.view(-1), phase_t.view(-1),
                strm_t.view(-1), agcy_t.view(-1),
            ).view(B, max_len, -1)                                        # (B, max_len, emb_dim)
            context  = model.casebook_encoder(placed_emb, pos_t, mask_t) # (B, context_dim)
            all_embs = model.token_embedding.token_emb(
                torch.arange(model.vocab_size, device=device)
            )                                                              # (V, emb_dim)
            logits   = model.retrieval_head(context, all_embs)            # (B, n_dims, V)

            loss = torch.tensor(0.0, device=device)
            batch_hard = 0.0
            batch_soft = 0.0
            for d in range(n_dims):
                hard_ce = F.cross_entropy(logits[:, d, :], inv_t[:, d])
                student_log_soft = F.log_softmax(logits[:, d, :] / kd_temperature, dim=-1)
                kl_soft = F.kl_div(
                    student_log_soft,
                    soft_targets[d].unsqueeze(0).expand(B, -1),
                    reduction="batchmean",
                    log_target=False,
                ) * (kd_temperature ** 2)
                loss_d = kd_alpha * hard_ce + (1.0 - kd_alpha) * kl_soft
                loss = loss + loss_d
                batch_hard += hard_ce.item()
                batch_soft += kl_soft.item()

            loss = loss / n_dims
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_hard += batch_hard / n_dims
            total_soft += batch_soft / n_dims
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        avg_hard = total_hard / max(n_batches, 1)
        avg_soft = total_soft / max(n_batches, 1)
        loss_history.append(avg)
        hard_history.append(avg_hard)
        soft_history.append(avg_soft)
        epoch_sec = time.time() - epoch_t0
        frozen_str = "emb=frozen" if epoch < 5 else "emb=train"
        _log(
            f"[SUPERVISED] Epoch {epoch+1:>2}/{n_epochs}  "
            f"loss={avg:.4f}  hard={avg_hard:.4f}  soft={avg_soft:.4f}  "
            f"{frozen_str}  ({epoch_sec:.1f}s)"
        )

    total_sec = time.time() - stage_t0
    _log(
        f"[SUPERVISED] Done — {n_epochs} epochs in {total_sec:.0f}s  "
        f"total {loss_history[0]:.4f}→{loss_history[-1]:.4f}  "
        f"hard {hard_history[0]:.4f}→{hard_history[-1]:.4f}  "
        f"soft {soft_history[0]:.4f}→{soft_history[-1]:.4f}"
    )

    # Return soft_targets so train_rl can use them as anchor
    return soft_targets


# ---------------------------------------------------------------------------
# Stage 2 — REINFORCE
# ---------------------------------------------------------------------------

def train_rl(
    model: MysteryEnergyModel,
    spec: CartridgeSpec,
    n_episodes: int,
    device: torch.device,
    soft_targets: Optional[torch.Tensor] = None,
    kd_coef: float = 0.05,
) -> None:
    _banner("STAGE 2 — REINFORCE FINE-TUNING")
    anchor_str = f"  kd_anchor={kd_coef}" if soft_targets is not None else "  kd_anchor=off"
    _log(f"[RL] episodes={n_episodes}  device={device}  γ=0.99  lr=3e-5{anchor_str}")

    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = _build_mappings(spec.tokens)
    env = CasebookEnv(spec, spec.token_graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    gamma = 0.99
    entropy_coef = 0.01
    min_turns = spec.min_turns

    # Rolling stats over last 50 episodes
    recent_conv = deque(maxlen=50)
    recent_turns = deque(maxlen=50)
    recent_loss = deque(maxlen=50)

    stage_t0 = time.time()
    log_every = max(1, n_episodes // 20)  # ~20 progress lines total

    for episode in range(n_episodes):
        obs = env.reset()
        log_probs_ep: List[torch.Tensor] = []
        rewards_ep: List[float] = []
        last_logits: Optional[torch.Tensor] = None
        info: dict = {}

        for _ in range(spec.max_turns - 1):
            tokens_ctx, positions_ctx = _obs_to_tokens_positions(obs, spec)
            logits = _get_retrieval_logits(
                model, tokens_ctx, positions_ctx,
                id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
                device,
            )
            last_logits = logits

            placed_ids = set(obs["placed_token_ids"])
            token_ids, lp = _sample_triad_from_logits(logits, spec, id_to_idx, placed_ids)
            if token_ids is None:
                break

            obs, reward, done, info = env.step(token_ids)
            if lp is not None:
                log_probs_ep.append(lp)
                rewards_ep.append(reward)
            if done:
                break

        if not log_probs_ep:
            continue

        turn = info.get("turn", 0)
        converged = info.get("converged", False)

        # Terminal shaping
        if converged and turn < min_turns:
            rewards_ep[-1] -= 2.0
        if converged and turn >= 13:
            rewards_ep[-1] += 1.0

        # Discounted returns
        G = 0.0
        returns: List[float] = []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = torch.tensor(0.0, device=device)
        for lp, G_val in zip(log_probs_ep, returns_t):
            policy_loss = policy_loss - lp * G_val

        entropy = _mean_entropy(last_logits) if last_logits is not None else torch.tensor(0.0)

        # KD anchor: keep policy close to Hopfield energy landscape
        kd_anchor = torch.tensor(0.0, device=device)
        if soft_targets is not None and last_logits is not None:
            n_dims = last_logits.size(1)
            for d in range(n_dims):
                kd_anchor = kd_anchor + F.kl_div(
                    F.log_softmax(last_logits[0, d, :], dim=0).unsqueeze(0),
                    soft_targets[d].unsqueeze(0),
                    reduction="batchmean",
                    log_target=False,
                )
            kd_anchor = kd_anchor / n_dims

        total_loss = policy_loss - entropy_coef * entropy + kd_coef * kd_anchor

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        recent_conv.append(int(converged))
        recent_turns.append(turn if converged else 0)
        recent_loss.append(total_loss.item())

        if (episode + 1) % log_every == 0 or episode == n_episodes - 1:
            conv_rate = sum(recent_conv) / max(len(recent_conv), 1)
            mean_t = (
                sum(recent_turns) / max(sum(recent_conv), 1)
                if sum(recent_conv) > 0 else 0.0
            )
            mean_loss = sum(recent_loss) / max(len(recent_loss), 1)
            elapsed = time.time() - stage_t0
            _log(
                f"[RL] ep {episode+1:>4}/{n_episodes}  "
                f"conv={conv_rate:.0%} (last {len(recent_conv)})  "
                f"mean_turns={mean_t:.1f}  "
                f"loss={mean_loss:.3f}  "
                f"elapsed={elapsed:.0f}s"
            )

    total_sec = time.time() - stage_t0
    _log(f"[RL] Done — {n_episodes} episodes in {total_sec:.0f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Thornfield transformer policy")
    parser.add_argument("case_id", help="Case identifier (e.g. amber_cipher)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--supervised-paths", type=int, default=2000)
    parser.add_argument("--supervised-epochs", type=int, default=20)
    parser.add_argument("--rl-episodes", type=int, default=500)
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--kd-temperature", type=float, default=0.20,
                        help="Softmax temperature for teacher distribution. "
                             "amber_cipher attractor weights: invariant=1.0, non-inv mean~0.13. "
                             "T=0.50 → P(invariant)~7%% (near-uniform, fights hard CE). "
                             "T=0.20 → P(invariant)~50%% (teacher is meaningful). "
                             "T=0.10 → P(invariant)~99%% (same as hard CE). (default 0.20)")
    parser.add_argument("--kd-alpha", type=float, default=0.3,
                        help="Weight of hard CE vs soft KD (0=all soft, 1=all hard, default 0.3)")
    parser.add_argument("--kd-coef", type=float, default=0.05,
                        help="KD anchor coefficient in REINFORCE loss (default 0.05)")
    args = parser.parse_args()

    _device_str = args.device
    if _device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        _device_str = "cpu"
    device = torch.device(_device_str)
    case_id = args.case_id
    run_t0 = time.time()

    _banner(f"THORNFIELD — TRAIN POLICY  ({case_id})")
    _log(f"[POLICY] device={args.device}  supervised_paths={args.supervised_paths}"
         f"  supervised_epochs={args.supervised_epochs}  rl_episodes={args.rl_episodes}"
         f"  kd_temperature={args.kd_temperature}  kd_alpha={args.kd_alpha}"
         f"  kd_coef={args.kd_coef}")

    spec_path = f"cases/{case_id}/spec.json"
    _log(f"[POLICY] Loading spec from {spec_path}")
    spec = CartridgeSpec.load(spec_path)
    _log(
        f"[POLICY] Spec loaded — vocab={spec.vocab_size}  "
        f"dims={spec.n_attractor_dims}  "
        f"threshold={spec.convergence_threshold}  "
        f"turns={spec.min_turns}–{spec.max_turns}"
    )

    model_path = f"outputs/{case_id}/model.pt"
    _log(f"[POLICY] Loading base model from {model_path}")
    model = MysteryEnergyModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_attractor_dims=spec.n_attractor_dims,
        token_graph=spec.token_graph,
    )
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    _log(f"[POLICY] Model ready — {param_count:,} parameters")

    soft_targets = train_supervised(
        model, spec,
        args.supervised_paths, args.supervised_epochs, device,
        kd_temperature=args.kd_temperature,
        kd_alpha=args.kd_alpha,
    )

    if not args.skip_rl:
        train_rl(
            model, spec, args.rl_episodes, device,
            soft_targets=soft_targets,
            kd_coef=args.kd_coef,
        )

    output_path = f"outputs/{case_id}/policy.pt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec_path": spec_path,
            "case_id": case_id,
            "embedding_dim": spec.embedding_dim,
            "context_dim": spec.context_dim,
            "n_attractor_dims": spec.n_attractor_dims,
            "model_type": "triad_policy",
        },
        output_path,
    )

    total_sec = time.time() - run_t0
    abs_path = Path(output_path).resolve()

    _banner("TRAINING COMPLETE")
    print(f"  Total time : {total_sec:.0f}s", flush=True)
    print(f"  Output     : {output_path}", flush=True)
    print(f"  Full path  : {abs_path}", flush=True)
    print(f"", flush=True)
    print(f"  Next step  : make ac-s05-benchmark", flush=True)
    print("=" * 62, flush=True)


if __name__ == "__main__":
    main()

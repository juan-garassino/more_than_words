"""
Benchmark Hopfield baseline vs Transformer policy on a Thornfield case.

Usage:
    python3 tools/benchmark_models.py amber_cipher [--n-episodes 200] [--plot]

Runs:
    1. Hopfield baseline  : PathSampler(allow_partial=False) × n_episodes
    2. Transformer policy : CasebookEnv + greedy policy × n_episodes
    3. Proof gate         : full ConvergenceProof on the transformer model

Reports a comparison table and a clear XCODE RECOMMENDATION block
showing exactly which .pt file to ship in the iOS game.
"""
from __future__ import annotations

import argparse
import datetime
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from core.cartridge import CartridgeSpec
from core.token import Token, TokenClass, TokenPhase, TokenStream, TokenAgency
from generator.path_sampler import PathSampler
from rl.casebook_env import CasebookEnv
from trainer.energy_model import MysteryEnergyModel
from trainer.hopfield_analyzer import HopfieldAnalyzer
from validator.convergence_proof import ConvergenceProof


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
# Shared helpers
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
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
    device: torch.device,
) -> torch.Tensor:
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
    return model.retrieval_head(context, all_embs)


def _select_triad(
    logits: torch.Tensor,
    spec: CartridgeSpec,
    placed_ids: set,
    id_to_idx: Dict[str, int],
    greedy: bool = True,
) -> List[str] | None:
    idx_to_token = {i: t for i, t in enumerate(spec.tokens)}
    vocab_size = logits.size(2)
    non_inv_ids = {t.id for t in spec.tokens if not t.is_invariant}
    available_ids = non_inv_ids - placed_ids

    for attempt in range(30):
        selected_tokens: List[Token] = []
        selected_ids_this: set = set()
        selected_classes: set = set()
        valid = True

        for d in range(logits.size(1)):
            dim_logits = logits[0, d, :].clone()
            for idx in range(vocab_size):
                tok = idx_to_token[idx]
                if (
                    tok.id not in available_ids
                    or tok.id in selected_ids_this
                    or tok.token_class.value in selected_classes
                ):
                    dim_logits[idx] = float("-inf")

            if (dim_logits != float("-inf")).sum() == 0:
                valid = False
                break

            if greedy and attempt == 0:
                chosen_idx = int(dim_logits.argmax().item())
            else:
                probs = F.softmax(dim_logits, dim=0)
                chosen_idx = int(torch.multinomial(probs, 1).item())

            tok = idx_to_token[chosen_idx]
            selected_tokens.append(tok)
            selected_ids_this.add(tok.id)
            selected_classes.add(tok.token_class.value)

        if not valid:
            continue

        ids = [t.id for t in selected_tokens]
        graph = spec.token_graph
        if any(
            graph.weight(ids[i], ids[j]) > 0.05
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        ):
            return ids

    # Absolute fallback
    avail = [t for t in spec.tokens if not t.is_invariant and t.id not in placed_ids]
    seen_classes: set = set()
    fallback: List[str] = []
    for t in avail:
        if t.token_class.value not in seen_classes:
            fallback.append(t.id)
            seen_classes.add(t.token_class.value)
            if len(fallback) == 3:
                return fallback
    return None


# ---------------------------------------------------------------------------
# Hopfield baseline
# ---------------------------------------------------------------------------

def run_hopfield_baseline(spec: CartridgeSpec, n_episodes: int) -> Dict:
    _section(f"HOPFIELD BASELINE  ({n_episodes} episodes)")
    _log(f"[HOPFIELD] Sampling convergent paths (allow_partial=False)...")
    t0 = time.time()
    sampler = PathSampler(spec, sampling_temperature=1.4, allow_partial=False)
    paths = sampler.sample_batch(n_episodes, verbose=True, max_attempts=n_episodes * 10)
    elapsed = time.time() - t0

    inv_ids = set(spec.invariant_token_ids)
    conv_count = 0
    correct_count = 0
    turns_list: List[int] = []

    for path in paths:
        if path and all(t.is_invariant for t in path[-1]):
            conv_count += 1
            turns_list.append(len(path))
            if set(t.id for t in path[-1]) == inv_ids:
                correct_count += 1

    n = n_episodes
    results = {
        "convergence_rate": conv_count / max(n, 1),
        "solution_accuracy": correct_count / max(n, 1),
        "mean_turns": sum(turns_list) / max(len(turns_list), 1),
        "turns_list": turns_list,
        "turns_ge_13": sum(1 for t in turns_list if t >= 13) / max(len(turns_list), 1),
    }
    _log(
        f"[HOPFIELD] Done ({elapsed:.0f}s) — "
        f"conv={results['convergence_rate']:.1%}  "
        f"mean_turns={results['mean_turns']:.1f}  "
        f"accuracy={results['solution_accuracy']:.1%}"
    )
    return results


# ---------------------------------------------------------------------------
# Transformer policy rollouts
# ---------------------------------------------------------------------------

def run_transformer_policy(
    spec: CartridgeSpec,
    model: MysteryEnergyModel,
    n_episodes: int,
    device: torch.device,
) -> Dict:
    _section(f"TRANSFORMER POLICY  ({n_episodes} episodes)")
    _log(f"[TRANSFORMER] Greedy rollouts with policy.pt...")
    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx = _build_mappings(spec.tokens)
    env = CasebookEnv(spec, spec.token_graph)
    model.eval()

    conv_count = 0
    correct_count = 0
    turns_list: List[int] = []
    t0 = time.time()
    log_every = max(1, n_episodes // 8)

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset()
            info: dict = {}

            for _ in range(spec.max_turns - 1):
                placed_ids_list = obs["placed_token_ids"]
                placed_tokens = [spec.get_token(tid) for tid in placed_ids_list]
                positions = [(min(i // 3, 7), i % 3) for i in range(len(placed_tokens))]

                logits = _get_retrieval_logits(
                    model, placed_tokens, positions,
                    id_to_idx, class_to_idx, phase_to_idx, stream_to_idx, agency_to_idx,
                    device,
                )
                token_ids = _select_triad(logits, spec, set(placed_ids_list), id_to_idx, greedy=True)
                if token_ids is None:
                    break
                obs, _, done, info = env.step(token_ids)
                if done:
                    break

            if info.get("converged", False):
                conv_count += 1
                turns_list.append(info["turn"])
            if info.get("correct_invariants", False):
                correct_count += 1

            if (ep + 1) % log_every == 0:
                _log(
                    f"[TRANSFORMER] {ep+1}/{n_episodes} episodes  "
                    f"conv so far: {conv_count/(ep+1):.1%}"
                )

    elapsed = time.time() - t0
    results = {
        "convergence_rate": conv_count / max(n_episodes, 1),
        "solution_accuracy": correct_count / max(n_episodes, 1),
        "mean_turns": sum(turns_list) / max(len(turns_list), 1),
        "turns_list": turns_list,
        "turns_ge_13": sum(1 for t in turns_list if t >= 13) / max(len(turns_list), 1),
    }
    _log(
        f"[TRANSFORMER] Done ({elapsed:.0f}s) — "
        f"conv={results['convergence_rate']:.1%}  "
        f"mean_turns={results['mean_turns']:.1f}  "
        f"accuracy={results['solution_accuracy']:.1%}"
    )
    return results


# ---------------------------------------------------------------------------
# Proof gate + Lyapunov
# ---------------------------------------------------------------------------

def run_transformer_proof(spec: CartridgeSpec, model: MysteryEnergyModel) -> Dict:
    _section("PROOF GATE  (transformer)")
    _log("[PROOF] Running ConvergenceProof on policy.pt...")
    return ConvergenceProof().run(model, spec, n_test_paths=200, max_attempts=2000, verbose=True)


def run_lyapunov_check(spec: CartridgeSpec, model: MysteryEnergyModel) -> float:
    _log("[LYAPUNOV] Sampling 100 paths for Lyapunov check...")
    sampler = PathSampler(spec, sampling_temperature=1.4, allow_partial=False)
    paths = sampler.sample_batch(100, verbose=False, max_attempts=500)
    if not paths:
        _log("[LYAPUNOV] WARNING — 0 paths sampled, returning 0.0")
        return 0.0
    result = HopfieldAnalyzer().lyapunov_check(model, paths[:100])
    _log(f"[LYAPUNOV] monotone_rate={result['monotone_rate']:.1%}  violations={len(result['violations'])}")
    return result["monotone_rate"]


# ---------------------------------------------------------------------------
# Comparison table + Xcode recommendation
# ---------------------------------------------------------------------------

def print_results(
    case_id: str,
    hopfield: Dict,
    transformer: Dict,
    proof: Dict,
    lyapunov_rate: float,
    policy_path: str,
    model_path: str,
) -> None:
    _banner("BENCHMARK RESULTS")

    def pct(v: float) -> str:
        return f"{v:.1%}"

    def fmt(v: float) -> str:
        return f"{v:.1f}"

    t_proof = "PASS" if proof.get("passed") else "FAIL"

    rows = [
        ("Proof gate",        "N/A (symbolic)",                  t_proof,                              "PASS"),
        ("Convergence rate",  pct(hopfield["convergence_rate"]),  pct(transformer["convergence_rate"]),  "≥ 90%"),
        ("Mean turns",        fmt(hopfield["mean_turns"]),         fmt(transformer["mean_turns"]),         "≥ 13"),
        ("Turns ≥ 13 (%)",   pct(hopfield["turns_ge_13"]),        pct(transformer["turns_ge_13"]),        "≥ 70%"),
        ("Solution accuracy", pct(hopfield["solution_accuracy"]),  pct(transformer["solution_accuracy"]),  "100%"),
        ("Lyapunov monotone", pct(lyapunov_rate),                  pct(lyapunov_rate),                     "≥ 90%"),
    ]

    col_w = [22, 22, 22, 14]
    sep = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"
    header = "| {:<{}} | {:<{}} | {:<{}} | {:<{}} |".format(
        "Metric", col_w[0],
        "Hopfield", col_w[1],
        "Transformer", col_w[2],
        "Pass bar", col_w[3],
    )
    print(f"\n{header}", flush=True)
    print(sep, flush=True)
    for metric, h_val, t_val, bar in rows:
        print("| {:<{}} | {:<{}} | {:<{}} | {:<{}} |".format(
            metric, col_w[0], h_val, col_w[1], t_val, col_w[2], bar, col_w[3]
        ), flush=True)

    passes = [
        proof.get("passed", False),
        transformer["convergence_rate"] >= 0.90,
        transformer["mean_turns"] >= 13,
        transformer["turns_ge_13"] >= 0.70,
        transformer["solution_accuracy"] >= 1.0,
        lyapunov_rate >= 0.90,
    ]
    ship_transformer = all(passes)
    decision = "SHIP TRANSFORMER" if ship_transformer else "KEEP HOPFIELD"

    # Resolve which file to use
    if ship_transformer:
        xcode_file = policy_path
        xcode_label = "policy.pt  (transformer — trained policy)"
    else:
        xcode_file = model_path
        xcode_label = "model.pt   (hopfield — symbolic baseline)"

    xcode_abs = str(Path(xcode_file).resolve())

    # Xcode recommendation block
    width = 62
    bar = "=" * width
    _banner("XCODE MODEL RECOMMENDATION")
    print(f"  DECISION  : {decision}", flush=True)
    print(f"", flush=True)
    print(f"  USE FILE  : {xcode_file}", flush=True)
    print(f"  LABEL     : {xcode_label}", flush=True)
    print(f"  FULL PATH : {xcode_abs}", flush=True)
    print(f"", flush=True)

    if not ship_transformer:
        failing = []
        checks = [
            ("Proof gate",       proof.get("passed", False),                    True),
            ("Convergence ≥90%", transformer["convergence_rate"] >= 0.90,       True),
            ("Mean turns ≥13",   transformer["mean_turns"] >= 13,               True),
            ("Turns≥13 ≥70%",   transformer["turns_ge_13"] >= 0.70,            True),
            ("Accuracy 100%",    transformer["solution_accuracy"] >= 1.0,       True),
            ("Lyapunov ≥90%",   lyapunov_rate >= 0.90,                         True),
        ]
        for label, passed, _ in checks:
            if not passed:
                failing.append(label)
        print(f"  FAILING   : {', '.join(failing)}", flush=True)
        print(f"  ACTION    : retrain policy (ac-s04-train-policy) or adjust hyperparams", flush=True)

    print("=" * width, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hopfield vs Transformer")
    parser.add_argument("case_id", help="Case identifier (e.g. amber_cipher)")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--plot", action="store_true", help="Save turn-distribution plot")
    args = parser.parse_args()

    case_id = args.case_id
    run_t0 = time.time()

    _banner(f"THORNFIELD — BENCHMARK  ({case_id})")
    _log(f"[BENCHMARK] n_episodes={args.n_episodes}")

    spec = CartridgeSpec.load(f"cases/{case_id}/spec.json")
    _log(
        f"[BENCHMARK] Spec loaded — vocab={spec.vocab_size}  "
        f"dims={spec.n_attractor_dims}  turns={spec.min_turns}–{spec.max_turns}"
    )

    policy_path = f"outputs/{case_id}/policy.pt"
    model_path = f"outputs/{case_id}/model.pt"
    _log(f"[BENCHMARK] Loading policy from {policy_path}")

    device = torch.device("cpu")
    model = MysteryEnergyModel(
        vocab_size=spec.vocab_size,
        embedding_dim=spec.embedding_dim,
        context_dim=spec.context_dim,
        n_attractor_dims=spec.n_attractor_dims,
        token_graph=spec.token_graph,
    )
    ckpt = torch.load(policy_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    _log(f"[BENCHMARK] Policy loaded")

    hopfield_results = run_hopfield_baseline(spec, args.n_episodes)
    transformer_results = run_transformer_policy(spec, model, args.n_episodes, device)
    proof_results = run_transformer_proof(spec, model)
    lyapunov = run_lyapunov_check(spec, model)

    print_results(
        case_id,
        hopfield_results,
        transformer_results,
        proof_results,
        lyapunov,
        policy_path,
        model_path,
    )

    total_sec = time.time() - run_t0
    _log(f"[BENCHMARK] Total time: {total_sec:.0f}s")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            bins = list(range(1, spec.max_turns + 2))
            ax.hist(hopfield_results["turns_list"], bins=bins, alpha=0.6,
                    label="Hopfield", color="steelblue")
            ax.hist(transformer_results["turns_list"], bins=bins, alpha=0.6,
                    label="Transformer", color="coral")
            ax.axvline(13, color="green", linestyle="--", label="Target min (13 turns)")
            ax.set_xlabel("Turns to convergence")
            ax.set_ylabel("Count")
            ax.set_title(f"Turn distribution — {case_id}")
            ax.legend()
            plt.tight_layout()
            plot_path = f"outputs/{case_id}/benchmark_turns.png"
            plt.savefig(plot_path)
            _log(f"[BENCHMARK] Plot saved to {plot_path}")
        except ImportError:
            _log("[BENCHMARK] matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()

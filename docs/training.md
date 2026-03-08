# Thornfield — Training Guide

## Are we ready to train?

**Yes.** For `amber_cipher`:

| Artifact | Status |
|---|---|
| `amber_cipher.json` | exists, validated |
| `cases/amber_cipher/spec.json` | packed |
| `outputs/amber_cipher/model.pt` | **trained, proof passed** |
| `outputs/amber_cipher/policy.pt` | not yet — run `make ac-s04-train-policy` |

The Hopfield model is trained. The next step is the transformer policy:

```bash
# CPU (~40 min for 500 RL episodes)
make ac-s04-train-policy

# GPU on Colab (~5-8 min)
make ac-s04-train-policy-gpu

# Then benchmark and get the Xcode recommendation:
make ac-s05-benchmark
```

For `amber_cipher_M` (5-dim, 152 tokens), `model.pt` does not exist yet:

```bash
make acm-pipeline        # full pipeline CPU (~2-3 hrs)
make acm-pipeline-gpu    # full pipeline GPU (~20 min)
```

---

## The training philosophy

### Why two stages?

The Hopfield graph engine is **symbolically correct** — it provably converges to the stored solution within the turn window (proof gate). But it is a fixed rule: it samples paths according to graph energy, with no concept of game-feel, pacing, or the diversity a real player would want.

The transformer policy learns from Hopfield expertise, then improves beyond it via reinforcement learning. The goal is a model that passes the same proof gate as the Hopfield engine but also plays better games — more exploratory, better paced, hitting the target turn window of 13–17 turns.

---

## Stage 1 — Strict knowledge distillation from Hopfield

```
PathSampler(allow_partial=True)
    generates N convergent paths
         │
         ▼
For each path, at each turn t:
    context = all tokens placed at turns 0..t-1
         │
         ▼
    Teacher: soft_targets[d] = softmax(attractor_weights[:, d] / T)
             — a probability distribution over all V tokens for each dim,
               derived directly from the Hopfield energy landscape

    Student: retrieval_logits[:, d, :] from HopfieldRetrievalHead

    loss = (1/n_dims) Σ_d [
        α * CrossEntropy(logits_d, hard_invariant_index)
      + (1-α) * T² * KL(softmax(logits_d/T) ∥ soft_targets[d])
    ]

    Default: α=0.3, T=2.0  (70% weight on soft KD)
```

This is **strict knowledge distillation** (Hinton et al., 2015): the teacher is the Hopfield attractor weight matrix expressed as a probability distribution, not just the argmax. Every token that partially points toward dimension `d` carries proportional credit. The T² scaling preserves gradient magnitude at higher temperatures.

**Why soft targets beat hard targets here:**
- Hard CE: tells the policy only "the answer is `renard_voss`"
- Soft KD: tells the policy "renard_voss (0.72), but also these tokens support dim 0 (0.08, 0.05, ...)"
- The full gradient shape mirrors the Hopfield energy landscape — the policy learns which supporting tokens co-activate the correct attractor

**What the transformer learns:**
- The full Hopfield energy gradient per dimension (not just argmax)
- Which tokens in context signal which invariant dimension
- Relative plausibility of all tokens as attractor pointers

**What it does NOT learn (yet):**
- How many turns to take
- How to explore diverse paths
- Game-feel pacing

**Training schedule:**
- Epochs 1–5: `TokenEmbedding` frozen (learn retrieval head only)
- Epochs 6–20: full model trainable
- Adam, lr=1e-4, gradient clipping at 1.0

---

## Stage 2 — REINFORCE fine-tuning

The policy is now deployed in `CasebookEnv` and rewarded for playing well.

### Reward function

```python
reward = energy_drop - 0.5 * speed_penalty + 0.1 * diversity_bonus

# energy_drop:    subgraph_energy(before) - subgraph_energy(after)
#                 positive when well-connected tokens are placed
#                 this aligns with the Hopfield Lyapunov function

# speed_penalty:  exp(-turn / max_turns)
#                 fades from 1.0 at turn 1 → ~0.4 at turn 12
#                 discourages rushing to convergence

# diversity_bonus: len(new_affinity_tags) / 3
#                  rewards placing tokens that open new narrative threads
```

### Terminal shaping

```python
if converged and turn < min_turns:
    reward[-1] -= 2.0   # penalise premature convergence

if converged and turn >= 13:
    reward[-1] += 1.0   # bonus for hitting target window
```

### REINFORCE update

```python
# Discounted returns (γ = 0.99)
G_t = r_t + 0.99 * G_{t+1}

# Normalise
returns = (returns - mean) / (std + 1e-8)

# Policy gradient + entropy bonus + KD anchor
loss = -Σ log_π(a_t|s_t) * G_t
     - 0.01 * entropy(logits)
     + 0.05 * KL(policy ∥ soft_targets)   ← Hopfield anchor
```

The entropy bonus prevents the policy from collapsing to a single deterministic path. The **KD anchor** (`kd_coef * KL`) keeps the RL policy anchored to the Hopfield energy landscape — it prevents the policy from drifting to low-energy but narratively incoherent triads that happen to get high game rewards. The `soft_targets` tensor is carried over from Stage 1 (no recomputation needed).

**Training settings:** Adam lr=3e-5, gradient clipping 1.0, 500 episodes

---

## Knowledge distillation — what we actually do

| Term | What it means here |
|---|---|
| **Behavioral cloning** | Imitate expert action sequences with hard CE on argmax |
| **Knowledge distillation** (strict) | Match teacher's full output distribution as soft targets |
| **Policy distillation** | Train student policy to match teacher policy distribution |

Stage 1 is **strict knowledge distillation**: the soft targets are `softmax(attractor_weights[:, d] / T)` — the Hopfield energy landscape expressed as a probability distribution over the full vocabulary. This is strictly better than behavioral cloning because:

- Hard CE treats all non-invariant tokens as equally wrong
- Soft KD assigns proportional credit to tokens that partially point toward each attractor dimension
- The gradient shape mirrors the Hopfield Lyapunov function, not just its minimum

The teacher is not a second neural model — it is the attractor weight matrix directly. This avoids the cost of running a teacher forward pass at every training step while still capturing the full energy gradient.

**Summary of the Hopfield → Transformer transfer:**

```
Hopfield graph (symbolic, deterministic)
    │
    │  PathSampler generates N demonstration paths
    │  Each path: a sequence of triads that converges to the solution
    │
    ▼
Stage 1: Behavioral cloning
    │  Transformer learns Q·K^T retrieval from partial-path context
    │  Knows WHAT the solution is given any partial state
    │
    ▼
Stage 2: REINFORCE
    │  Transformer explores beyond demonstrations
    │  Optimises HOW to reach the solution (timing, diversity, energy flow)
    │
    ▼
Transformer policy (neural, stochastic)
    │
    ▼
Proof gate (same as Hopfield)
    convergence ≥ 90%, Lyapunov ≥ 90%, invariant accuracy = 100%
    → SHIP TRANSFORMER
```

---

## Convergence spec — amber_cipher

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 72 | 69 non-invariant + 3 invariant |
| `n_attractor_dims` | 3 | suspect, mechanism, motive |
| `convergence_rate` | 0.40 | accumulation per triad per dim |
| `convergence_threshold` | 0.75 | min-dim gate |
| `min_turns` | 10 | earliest legal convergence |
| `max_turns` | 18 | timeout |
| `target_turns` | 13–17 | game-feel target (RL reward) |
| `embedding_dim` | 64 | token embedding size |
| `context_dim` | 128 | casebook encoder output |

With `rate=0.40` and mean attractor contribution ~0.133/dim, expected convergence at turn ~15 (within the 10–18 window).

---

## Convergence spec — amber_cipher_M

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 152 | 147 non-invariant + 5 invariant |
| `n_attractor_dims` | 5 | suspect + location + mechanism + motive + accomplice |
| `convergence_rate` | 0.38 | slightly lower for harder case |
| `convergence_threshold` | 0.75 | same gate |
| `min_turns` | 15 | more exploration required |
| `max_turns` | 28 | longer game |

---

## Debugging convergence failures

### PathSampler returns 0 paths (`allow_partial=False`)

The convergence threshold is unreachable. Check:
1. `convergence_rate` — raise if too low (was 0.25, now 0.40 for amber_cipher)
2. Mean attractor weights — if most tokens contribute <0.05/dim, no path can reach 0.75
3. `max_turns` — may need to be increased

### Lyapunov violations > 10%

The graph has negative-weight edges. Check:
1. All edge weights in `graph.json` should be > 0
2. The Lyapunov check uses cumulative `subgraph_energy`, which is guaranteed monotone for non-negative weights

### Transformer DECISION: KEEP HOPFIELD

Check which metrics are failing in the output:
- `convergence_rate < 90%` → increase supervised epochs or RL episodes
- `mean_turns < 13` → the `speed_penalty` in the reward is insufficient; increase its coefficient (0.5 → 0.7)
- `solution_accuracy < 100%` → supervised pretraining is not converging; check loss curve in Stage 1
- `proof gate FAIL` → run `make ac-s03-train-hopfield` again, then retrain policy

---

## Extending to a new case

1. Author `my_case.json` following `amber_cipher.json` structure
2. Run `python3 thornfield_case_validator.py my_case.json`
3. Add Makefile targets following the `ac-` or `acm-` pattern
4. Pack, train, benchmark: same pipeline

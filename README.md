# Thornfield

A symbolic mystery game engine built on a **Modern Hopfield Network**, with a Python training pipeline and an iOS runtime. There is no LLM. Inference operates entirely on token IDs and float weights — no text generation, no embeddings from language models.

The game is **retrieval**: the player places triads of tokens onto a casebook. Each placement updates a convergence state across attractor dimensions. When the energy landscape converges to the stored invariants (killer, mechanism, motive), the case is solved.

---

## Core idea

The Hopfield network stores a mystery solution as an energy minimum. Every token in the vocabulary has a weight vector (attractor weights) and edges to related tokens (affinity graph). Placing tokens decreases system energy. The solution — three invariant tokens — sits at the global minimum.

The training pipeline validates this mathematically via a **proof gate**: a Lyapunov monotonicity check, basin coverage test, and convergence proof that the network reliably reaches the correct attractors within the allowed turn window.

A **transformer policy** is then trained on top: first by imitating Hopfield-generated paths (behavioral cloning), then fine-tuned with REINFORCE to optimise for game-feel metrics like mean turn count and solution diversity. The transformer must pass the same proof gate before it can ship.

---

## Cases

| File | ID | Dims | Vocab | Difficulty | Status |
|---|---|---|---|---|---|
| `amber_cipher.json` | `amber_cipher` | 3 | 72 tokens | medium | trained, proof passed |
| `amber_cipher_M.json` | `amber_cipher_M` | 5 | 152 tokens | hard | packed, ready to train |
| `attended_hour.json` | — | — | — | — | draft |
| `fog_over_brussels.json` | — | — | — | — | draft |
| `hollow_season.json` | — | — | — | — | draft |

---

## Quick start

```bash
# Full pipeline for amber_cipher (CPU, ~40 min)
make ac-pipeline

# Full pipeline on Colab GPU (~8 min)
make ac-pipeline-gpu

# Or run stages individually:
make ac-s01-validate        # check amber_cipher.json structure
make ac-s02-pack            # pack → thornfield/trainer/cases/amber_cipher/
make ac-s03-train-hopfield  # train Hopfield model → outputs/amber_cipher/model.pt
make ac-s04-train-policy    # supervised + REINFORCE → outputs/amber_cipher/policy.pt
make ac-s05-benchmark       # compare both, print XCODE RECOMMENDATION
```

At the end of `ac-s05-benchmark`, the output includes:

```
================================================================
  XCODE MODEL RECOMMENDATION
================================================================
  DECISION  : SHIP TRANSFORMER
  USE FILE  : outputs/amber_cipher/policy.pt
  FULL PATH : /path/to/.../outputs/amber_cipher/policy.pt
================================================================
```

---

## Pipeline overview

```
amber_cipher.json
      │
      ▼ s01  thornfield_case_validator.py
  validated JSON
      │
      ▼ s02  tools/pack_case.py
  cases/amber_cipher/
    spec.json, tokens.json, graph.json, ...
      │
      ▼ s03  tools/train_single_case.py
  outputs/amber_cipher/model.pt          ← Hopfield model + proof gate
      │
      ├─ Hopfield paths (PathSampler, allow_partial=True)
      │       2000 convergent demonstrations
      │
      ▼ s04  trainer/train_policy.py
  [Stage 1] Supervised pretraining       ← behavioral cloning from Hopfield
  [Stage 2] REINFORCE fine-tuning        ← RL on CasebookEnv
  outputs/amber_cipher/policy.pt
      │
      ▼ s05  tools/benchmark_models.py
  comparison table + proof gate on transformer
  → DECISION: SHIP TRANSFORMER / KEEP HOPFIELD
```

See [`docs/training.md`](docs/training.md) for the full training philosophy.

---

## Architecture

```
MysteryEnergyModel
├── TokenEmbedding          token_id + class + phase + stream + agency → (B, 64)
├── CasebookEncoder         attention over placed tokens → context (B, 128)
├── TriadEnergyHead         energy score for a candidate triad (B, 1)
├── ConvergenceHead         predicted convergence delta per dim (B, n_dims)
├── HopfieldRetrievalHead   Q·K^T retrieval → invariant logits (B, n_dims, V)
└── TokenResonanceHead      field resonance → next-token suggestions (B, V)
```

The **HopfieldRetrievalHead** is the game policy. It implements transformer self-attention where:
- `Q` = query vectors projected from context (one per attractor dimension)
- `K = V` = token embedding matrix for the full vocabulary

This is the Modern Hopfield / transformer-attention equivalence: `scores = QK^T / sqrt(d)`.

See [`docs/architecture.md`](docs/architecture.md) for the full model breakdown.

---

## Makefile reference

### amber_cipher  (`ac-`)

| Target | What it does |
|---|---|
| `ac-s01-validate` | Validate `amber_cipher.json` |
| `ac-s02-pack` | Pack case → `trainer/cases/amber_cipher/` |
| `ac-s03-train-hopfield` | Train Hopfield model, CPU |
| `ac-s03-train-hopfield-gpu` | Train Hopfield model, GPU |
| `ac-s03-train-hopfield-fastproof` | Train with reduced proof (dev) |
| `ac-s04-train-policy` | Supervised + REINFORCE policy, CPU |
| `ac-s04-train-policy-gpu` | Supervised + REINFORCE policy, GPU |
| `ac-s05-benchmark` | Benchmark + Xcode recommendation |
| `ac-pipeline` | Full pipeline, CPU |
| `ac-pipeline-gpu` | Full pipeline, GPU |

### amber_cipher_M  (`acm-`)  — 5-dim, 152 tokens

| Target | What it does |
|---|---|
| `acm-s01-validate` | Validate `amber_cipher_M.json` |
| `acm-s02-pack` | Pack case → `trainer/cases/amber_cipher_M/` |
| `acm-s03-train-hopfield` | Train Hopfield model, CPU |
| `acm-s03-train-hopfield-gpu` | Train Hopfield model, GPU |
| `acm-s04-train-policy` | Supervised + REINFORCE policy, CPU |
| `acm-s04-train-policy-gpu` | Supervised + REINFORCE policy, GPU |
| `acm-s05-benchmark` | Benchmark + Xcode recommendation |
| `acm-pipeline` | Full pipeline, CPU |
| `acm-pipeline-gpu` | Full pipeline, GPU |

---

## Key files

```
more_than_words/
├── amber_cipher.json              case definition (small, 3-dim)
├── amber_cipher_M.json            case definition (medium, 5-dim)
├── thornfield_case_validator.py   validates case JSON before packing
├── Makefile                       all pipeline commands
│
├── docs/
│   ├── architecture.md            model components deep-dive
│   └── training.md                training philosophy + distillation
│
└── thornfield/trainer/
    ├── core/
    │   ├── token.py               Token, TokenClass, TokenPhase, TokenStream
    │   ├── hopfield.py            TokenGraph — energy, subgraph_energy (Lyapunov)
    │   ├── casebook.py            CasebookState — convergence tracking
    │   └── cartridge.py           CartridgeSpec — full case spec loader
    │
    ├── generator/
    │   └── path_sampler.py        PathSampler — Monte Carlo path generator
    │
    ├── trainer/
    │   ├── energy_model.py        MysteryEnergyModel (all heads)
    │   ├── loss.py                EnergyMargin + Attractor + Lyapunov + Retrieval
    │   ├── train_mystery.py       Hopfield training loop
    │   └── train_policy.py        Supervised pretraining + REINFORCE loop
    │
    ├── rl/
    │   ├── casebook_env.py        Gym-style env wrapping CasebookState
    │   └── rewards.py             compute_reward — energy + timing + diversity
    │
    ├── validator/
    │   └── convergence_proof.py   Lyapunov + basin + invariant proof gate
    │
    ├── tools/
    │   ├── pack_case.py           JSON → cases/<id>/
    │   ├── train_single_case.py   Hopfield training entry point
    │   └── benchmark_models.py    Hopfield vs Transformer comparison
    │
    └── outputs/
        ├── amber_cipher/
        │   ├── model.pt           trained Hopfield weights
        │   ├── policy.pt          trained transformer policy (after s04)
        │   └── TheAmberCipher.cartridge   iOS export
        └── amber_cipher_M/
            └── model.pt           (after acm-s03)
```

---

## Invariants (non-negotiable)

- The engine never reads surface expressions (token labels are UI-only).
- Convergence score is `min(convergence_dimensions)` — the weakest attractor dimension gates the solution.
- Cartridges cannot export without a passed proof gate.
- `subgraph_energy` is the Lyapunov function. For non-negative weights it is guaranteed monotone decreasing as tokens are added.

---

## Requirements

```bash
pip install -r thornfield/trainer/requirements.txt
# or:
make colab-install
```

Python 3.10+, PyTorch 2.x. No GPU required for `amber_cipher` (CPU ~40 min full pipeline).

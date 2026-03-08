# Thornfield

A mystery game **powered by a transformer**, trained via knowledge distillation and reinforcement learning. Each case ships as a self-contained trained model — no server, no LLM, no text generation. Inference runs entirely on-device from token IDs and float weights.

The game is **retrieval**: the player places triads of tokens onto a casebook. Each placement updates a convergence state across attractor dimensions. When the field converges to the stored solution (killer, mechanism, motive), the case is solved.

---

## What ships

**The transformer ships. The Hopfield never does.**

| Component | Role | Ships? |
|---|---|---|
| Hopfield network | Training scaffold and proof oracle | No |
| `model.pt` | Trained Hopfield weights | No |
| `policy.pt` | Transformer trained via KD + RL | **Yes — runs the iOS game** |
| `.cartridge` | Packed case spec (vocabulary, graph, convergence rules) | **Yes** |

Each case is a DLC: one `.cartridge` + one `policy.pt`. The transformer runs on-device. No backend required.

---

## The cases

Twenty-four cases across mystery, naval, and adventure formats. See [`docs/cases/`](docs/cases/) for the full narrative specifications.

### Mystery (01–20)

| # | Title | Period | Setting | Difficulty |
|---|---|---|---|---|
| 01 | The Amber Cipher | 1887 | Railway junction, Essex | medium |
| 02 | The Venetian Mirror | 1931 | Palazzo, Venice carnival | hard |
| 03 | Fog Over Brussels | 1961 | Belgian embassy | hard |
| 04 | The Hollow Season | 1907 | Edwardian country house | medium |
| 05 | The Resonance Test | 1974 | Music conservatory, London | easy |
| 06 | The Tidal Interval | Present | Island research station | medium |
| 07 | The Third Signature | 1935 | London literary club | hard |
| 08 | The Sulphur Line | 1889 | Victorian chemical works | medium |
| 09 | The Orchard at Dusk | 1903 | Rural England, harvest | easy |
| 10 | The Attended Hour | Present | Hospital cardiac ward | hard |
| 11 | The Winter Station | 1912 | Antarctic research depot | medium |
| 12 | The Monsoon Ledger | 1905 | Calcutta, Bengal Partition | medium |
| 13 | The Observatory Clock | 1900 | Paris Observatory | medium |
| 14 | The Endgame | 1972 | Reykjavik chess championship | hard |
| 15 | The Amber Silence | 1943 | Occupied Normandy | hard |
| 16 | The Signal Fire | 1943 | Pacific island, WWII | hard |
| 17 | The Covenant Garden | 1349 | Yorkshire monastery | medium |
| 18 | The Mountain Exchange | 1938 | Swiss Alps, pre-war | hard |
| 19 | The Instrument Landing | 1954 | Post-war London | medium |
| 20 | The Burning Glass | 1909 | Istanbul, Young Turk era | medium |

### Naval (21)

| # | Title | Period | Setting | Difficulty |
|---|---|---|---|---|
| 21 | The Dead Calm | 1698 | Pirate brigantine, Caribbean | hard |

### Adventure (A01–A03)

Adventure cases converge toward a **chosen state** rather than a fixed truth. The player's choices determine what becomes possible. The field records them.

| # | Title | Period | Protagonist |
|---|---|---|---|
| A01 | The Thirteenth Tide | 1697 | Sera Vane, cartographer's daughter, Nassau |
| A02 | The Glass Cartographer | 1627 | Lena Faber, glassmaker's daughter, Bohemia |
| A03 | The Iron Cartridge | 1876 | Elias Drum, interpreter, Dakota Territory |

---

## Quick start

```bash
# Full pipeline for amber_cipher (CPU, ~40 min)
make ac-pipeline

# Full pipeline on Colab GPU (~8 min)
make ac-pipeline-gpu

# Or run stages individually:
make ac-s01-validate        # check cases/amber_cipher.json
make ac-s02-pack            # pack → thornfield/trainer/cases/amber_cipher/
make ac-s03-train-hopfield  # train Hopfield model → outputs/amber_cipher/model.pt
make ac-s04-train-policy    # supervised + REINFORCE → outputs/amber_cipher/policy.pt
make ac-s05-benchmark       # compare both, print XCODE RECOMMENDATION
```

At the end of `ac-s05-benchmark`:

```
================================================================
  XCODE MODEL RECOMMENDATION
================================================================
  DECISION  : SHIP TRANSFORMER
  USE FILE  : outputs/amber_cipher/policy.pt
================================================================
```

---

## Architecture

```
MysteryEnergyModel
├── TokenEmbedding          token_id + class + phase + stream + agency → (B, 64)
├── CasebookEncoder         attention over placed tokens → context (B, 128)
├── HopfieldRetrievalHead   Q·K^T retrieval → invariant logits (B, n_dims, V)
├── TriadEnergyHead         energy score for a candidate triad (B, 1)
├── ConvergenceHead         predicted convergence delta per dim (B, n_dims)
└── TokenResonanceHead      field resonance → next-token suggestions (B, V)
```

The `HopfieldRetrievalHead` is the game policy. It implements transformer self-attention where Q = query vectors projected from context (one per attractor dimension) and K = V = token embedding matrix for the full vocabulary. This is the Modern Hopfield / transformer-attention equivalence: `scores = QK^T / sqrt(d)`.

See [`docs/architecture.md`](docs/architecture.md) for the full model breakdown.

---

## Repository layout

```
more_than_words/
├── CLAUDE.md                     project instructions
├── Makefile                      all pipeline commands
├── thornfield_case_validator.py  validates case JSON before packing
│
├── cases/                        case definitions (JSON)
│   ├── amber_cipher.json         trained, proof passed
│   ├── amber_cipher_M.json       large variant (5-dim, 152 tokens)
│   └── *.json                    draft cases
│
├── docs/
│   ├── architecture.md           model components deep-dive
│   ├── training.md               training philosophy + distillation
│   └── cases/                    narrative specifications for all 24 cases
│       ├── index.md
│       ├── 01_amber_cipher.md … 21_dead_calm.md
│       └── A01_thirteenth_tide.md … A03_iron_cartridge.md
│
├── notebooks/                    Colab and development notebooks
│
└── thornfield/trainer/
    ├── core/                     Token, CasebookState, TokenGraph, CartridgeSpec
    ├── generator/                PathSampler
    ├── trainer/                  MysteryEnergyModel, train_mystery.py, train_policy.py
    ├── rl/                       CasebookEnv, rewards.py
    ├── validator/                convergence_proof.py
    ├── tools/                    pack_case.py, benchmark_models.py
    └── outputs/                  model.pt + policy.pt per case
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

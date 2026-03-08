# Thornfield — Architecture

## The core abstraction

A Thornfield mystery is a **Modern Hopfield Network** expressed as a game.

The network stores one solution: three (or more, for multi-dim cases) invariant tokens representing killer, mechanism, and motive. Every other token in the vocabulary is a distractor, a clue, or an atmospheric element. The player's goal is to guide the network to its stored attractor by placing triads.

The engine never interprets text. It operates on:
- **Token IDs** — integer indices into the vocabulary
- **Attractor weights** — per-token float vectors, one element per solution dimension
- **Affinity graph** — symmetric weighted edges between token pairs (the Hopfield weight matrix)

---

## TokenGraph — the Hopfield weight matrix

```python
# core/hopfield.py

class TokenGraph:
    nodes: List[str]
    edges: Dict[Tuple[str, str], float]   # (a, b) → weight ≥ 0

    def subgraph_energy(self, token_ids: List[str]) -> float:
        # E = -Σ w(i,j) for all pairs (i,j) in placed tokens
        # Equivalent to: E = -½ x^T W x  (standard Hopfield energy)
```

**Key property**: `subgraph_energy` is the Lyapunov function. For any graph with non-negative weights, total energy decreases monotonically as tokens are added. This guarantees proof gate passage.

`induced_subgraph_energy` adds a cross-term (0.75× weight) between candidate tokens and already-placed context, used by `PathSampler` for trajectory scoring.

---

## CasebookState — convergence tracker

```python
# core/casebook.py

@dataclass
class CasebookState:
    convergence_dimensions: np.ndarray  # (n_dims,) floats in [0, 1]
    convergence_rate: float             # 0.40 for amber_cipher

    def place_triad(self, tokens, position):
        contribution = mean(attractor_weights for token in tokens)
        convergence_dimensions = min(1.0, convergence_dimensions + contribution * rate)

    @property
    def convergence_score(self) -> float:
        return convergence_dimensions.min()  # gate is the weakest dimension
```

The score gates on the **minimum** dimension — the player cannot ignore any dimension of the solution. A triad of three red herrings contributes near-zero to all dimensions. Only well-chosen triads push all dimensions toward 1.0.

---

## MysteryEnergyModel

```
Input tokens (placed)
        │
        ▼
  TokenEmbedding
  ┌────────────────────────────────────────────────────────┐
  │  token_emb  (V, 64)                                    │
  │  class_emb  (14, 16)   TokenClass enum                 │
  │  phase_emb  (8, 8)     EARLY / MID / LATE / INVARIANT  │
  │  stream_emb (4, 8)     EVIDENCE / ATMOSPHERE / ...     │
  │  agency_emb (3, 8)     PLAYER / ENGINE / SHARED        │
  │  proj       → (B, 64)                                  │
  └────────────────────────────────────────────────────────┘
        │
        ▼
  CasebookEncoder
  ┌────────────────────────────────────────────────────────┐
  │  spatial_enc  (row, col) → (16,)                       │
  │  proj         (64+16) → (128,)                         │
  │  MultiheadAttention  4 heads, batch_first              │
  │  pool         mean of attended tokens                  │
  │  → context (B, 128)                                    │
  └────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
  TriadEnergyHead                             HopfieldRetrievalHead  ← THE POLICY
  ┌──────────────────────┐                   ┌───────────────────────────────────┐
  │  joint(emb×3) → 128  │                   │  queries = Linear(ctx, n_dims×64) │
  │  combiner + sigmoid  │                   │  logits  = queries @ token_emb.T  │
  │  → (B, 1)  energy    │                   │  → (B, n_dims, V)                 │
  └──────────────────────┘                   │                                   │
                                             │  This IS transformer attention:   │
        ▼                                    │  Q = context projections          │
  ConvergenceHead                            │  K = V = vocabulary embeddings    │
  ┌──────────────────────┐                   └───────────────────────────────────┘
  │  (emb×3 + ctx) → 128 │
  │  Linear → n_dims     │                   TokenResonanceHead
  │  sigmoid             │                   ┌──────────────────────────────────┐
  │  → (B, n_dims)       │                   │  ctx_proj(128→64)                │
  └──────────────────────┘                   │  logits = proj @ token_emb.T     │
                                             │  → (B, V)  hint suggestions      │
                                             └──────────────────────────────────┘
```

### HopfieldRetrievalHead in detail

This head implements the **Modern Hopfield Network retrieval** as a single attention operation:

```
scores = Q · K^T / sqrt(d)
```

where:
- `Q = queries(context)` — shape `(B, n_dims, 64)`, one query per attractor dimension
- `K = token_embedding_matrix` — shape `(V, 64)`, fixed vocabulary

At the energy minimum, the queries align with the invariant token embeddings. The argmax per dimension gives the predicted solution.

The **transformer-attention = Modern Hopfield** equivalence (Ramsauer et al., 2020) means this head can, in principle, store and retrieve exponentially many patterns — far more than the classical Hopfield network.

---

## PathSampler — trajectory generator

`PathSampler` generates training paths via Monte Carlo sampling over the affinity graph.

```
1. Place opening triad (fixed, from spec.opening_token_ids)
2. For each turn:
   a. Collect candidate triads:
      - Must have tokens from 3 different TokenClasses
      - Must have at least one graph edge (weight > 0.05)
      - No repulsion tag conflicts
      - Tag-indexed lookup (fast), fallback to full scan
   b. Score each candidate:
      score = -induced_subgraph_energy + narrative_gradient * 0.2
   c. Softmax sample with temperature
3. Stop when:
   - convergence_score ≥ 0.75 AND turn ≥ min_turns (converged)
   - turn ≥ max_turns (timed out)
   - allow_partial=False AND not converged → return None
```

**Two modes:**
- `allow_partial=True` — training mode: accepts partial paths
- `allow_partial=False` — proof mode: only returns fully converging paths

---

## CasebookEnv — RL environment

Gym-style wrapper around `CasebookState` for policy training.

```
reset() → places opening triad, returns {"placed_token_ids": [...], "turn": 1}
step(token_ids) → (obs, reward, done, info)

reward = energy_drop - 0.5 * speed_penalty + 0.1 * diversity_bonus
       where:
         energy_drop    = subgraph_energy(before) - subgraph_energy(after)
         speed_penalty  = exp(-turn / max_turns)   ← penalises early convergence
         diversity_bonus = new_affinity_tags / 3.0  ← rewards exploration
```

---

## Loss functions (Hopfield training)

| Loss | Formula | Purpose |
|---|---|---|
| `EnergyMarginLoss` | `relu(0.4 - (E_neg - E_pos)).mean()` | Positive triads have lower energy than negatives |
| `AttractorConvergenceLoss` | `relu(0.75 - cumulative_dims).mean()` | Push all dimensions toward threshold |
| `LyapunovRegularization` | `relu(E_t+1 - E_t).mean()` | Penalise energy increases along paths |
| `HopfieldRetrievalLoss` | cross-entropy per dim | Retrieval head predicts invariant tokens |
| `ConvergenceHead MSE` | `MSE(predicted_delta, target_delta)` | Accurate convergence delta prediction |

---

## Proof gate

Four checks must all pass before a cartridge can export:

| Check | Threshold | Method |
|---|---|---|
| Convergence rate | 100% of sampled paths | `PathSampler(allow_partial=False)` |
| Invariant accuracy | 100% | last triad of each path == invariant set |
| Lyapunov monotone | ≥ 90% steps | `subgraph_energy` cumulative decrease |
| Basin coverage | ≥ 90% | `PathSampler(allow_partial=False)` × 200 |

The Lyapunov check uses **cumulative total system energy** — the sum of all edge weights between all placed tokens so far. This is the correct Lyapunov function for a Hopfield network. For non-negative edge weights, it is guaranteed monotone decreasing.

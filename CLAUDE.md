# Thornfield (Concept + What to Validate)

Thornfield is a symbolic Modern Hopfield network expressed as an iOS mystery game and a Python training pipeline. There is no LLM. The engine operates only on token IDs and float weights. Surface expressions are UI-only and must never influence inference.

The game is retrieval. Players place triads (three tokens) as synchronous Hopfield updates. The network converges to stored invariants (killer, mechanism, motive) via an energy landscape defined by an affinity graph.

Each case is a fully specified symbolic field:
- A fixed token vocabulary with phases, attractor weights, tags, and invariants.
- A sparse, story-justified affinity graph (Hopfield weight matrix).
- A convergence rule using the minimum over attractor dimensions.
- A proof gate that must pass before export.

## What to Validate
- Structure rules: token counts, class distribution, invariant purity, phase counts.
- Attractor gradients: early/mid/late weight bands, red herring cap, balanced convergence.
- Graph correctness: symmetric edges, no self-loops, invariant isolation, enabler bridge.
- Training viability: path sampling succeeds, loss is finite, proof runs to completion.
- Convergence proof: Lyapunov monotonicity, basin coverage, no spurious attractors.

## What to Optimize
- Proof reliability and runtime (avoid stalls in path sampling).
- Training stability (avoid NaNs, ensure monotone energy trends).
- Signal-to-noise in graph design (story-justified edges only).
- Progress visibility (clear batch/proof progress output).

## Key Commands
- Validate case: `python3 thornfield_case_validator.py amber_cipher.json`
- Pack case: `python3 thornfield/trainer/tools/pack_case.py amber_cipher.json`
- Train + proof: `make train-amber-cipher-colab-cpu`
- Train without proof: add `--skip-proof`
- Load trained model: `python3 thornfield/trainer/tools/load_trained_model.py thornfield/trainer/outputs/amber_cipher/model.pt`

## Invariants (Non‑Negotiable)
- The engine never reads surface expressions.
- Convergence score is the minimum across dimensions.
- Cartridges cannot export without a passed proof.

# Living Tales

Living Tales is a symbolic mystery game where a tiny overfitted transformer co-narrates with the player through token dialogue. The Hopfield network is the story compiler (training + proof); the transformer is the story runtime (shipped to device).

There is no LLM. The engine operates only on token IDs and float weights. Surface expressions are UI-only labels and must never influence inference. Tokens are symbolic: suspects, events, locations, objects, motives — not natural language.

## Game Loop

Player plays a symbolic token → model responds with a symbolic token (a clue) → repeat. This is a token-level dialogue, not text generation. The model is intentionally overfitted to one case — it knows one mystery perfectly across multiple narrative paths. Replayability comes from structural diversity: same solution, different journeys.

Every game starts from the story's origin. Tokens unfold in strict chronological order: EARLY → MID → LATE. Phase gating is a hard constraint at every level (sampling, training, inference, TUI).

## Training Pipeline

```
1. Validate case JSON           living_tales_case_validator.py
2. Pack case                    tools/pack_case.py
3. Train Hopfield (existing)    tools/train_single_case.py --model-type triad
4. Sample dialogue trajectories generator/dialogue_sampler.py
5. Supervised KD (Hopfield →    trainer/train_dialogue.py
   transformer on dialogues)
6. REINFORCE fine-tuning        trainer/train_dialogue.py
7. Convergence proof            validator/convergence_proof.py
8. Export cartridge              packager/export_mystery.py
```

## Key Commands

- Validate case: `python3 living_tales_case_validator.py cases/amber_cipher.json`
- Pack case: `python3 living_tales/trainer/tools/pack_case.py cases/amber_cipher.json`
- Train Hopfield: `make ac-s03-train-hopfield`
- Train dialogue transformer: `make ac-s04-train-dialogue`
- Train dialogue (fast): `make ac-s04-train-dialogue-fast`
- Play interactively: `cd living_tales/trainer && python3 tools/play_dialogue.py amber_cipher`
- Run evals: `cd evals && make eval-all`
- Load trained model: `python3 living_tales/trainer/tools/load_trained_model.py living_tales/trainer/outputs/amber_cipher/dialogue_model.pt`

## Invariants (Non-Negotiable)

- The engine never reads surface expressions.
- Convergence score is the minimum across dimensions.
- Cartridges cannot export without a passed proof.
- Phase order (EARLY → MID → LATE) is inviolable.
- KD anchor prevents RL drift from proven Hopfield structure.
- Tokens are always symbolic — this is not a language model.

## What to Validate

- Structure rules: token counts, class distribution, invariant purity, phase counts.
- Attractor gradients: early/mid/late weight bands, red herring cap, balanced convergence.
- Graph correctness: symmetric edges, no self-loops, invariant isolation, enabler bridge.
- Training viability: dialogue sampling succeeds, KD loss decreases, RL reward increases.
- Convergence proof: Lyapunov monotonicity, basin coverage, no spurious attractors.
- Chronology: phase compliance >= 90% across all eval games.

## What to Optimize

- Dialogue trajectory quality (diverse, convergent paths).
- KD temperature tuning (start T=2.0, adjust per eval perplexity).
- RL reward shaping (energy + chronology + diversity).
- Overfitting depth (perplexity < 10, accuracy > 50% on held-out dialogues).
- Progress visibility (clear training/eval output).

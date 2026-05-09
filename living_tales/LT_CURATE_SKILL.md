# `lt-curate` — Living Tales Trajectory Curator Skill

A Claude Code skill that automates trajectory dataset authoring, review, and validation for Living Tales cases using parallel subagents.

The Living Tales engine is trained on hand-authored trajectories — sequences of `(player_card → 11-dim scene_tuple)` pairs that demonstrate complete playthroughs of a case. A polished case ships with ~25 trajectories totaling ~1,000 turn pairs. That volume is too large for the human author to produce alone, so this skill orchestrates parallel subagents that draft, refine, and validate trajectories under tight schema constraints.

The skill never trains the model — training runs on Colab. The skill never writes the case foundation (`dimensions.json`, `constraints.json`, `phrases.json`, `VOICES.md`, `snippets.json`) — those remain the human author's hand-authored work. The skill produces only **trajectory data**: the dataset the model overfits on.

## When to use it

| Mode | Trigger phrases | What it does |
|---|---|---|
| **bootstrap** | *"author the trajectories for X"*, *"build the dataset for X"*, *"bootstrap X"* | Verifies the case foundation exists, writes a flagship trajectory in chunks, then fans out 5 parallel authoring agents producing 5–7 trajectories each, then runs a parallel review pass. Final state: ~25 validated trajectories. |
| **expand** | *"add N more wrong-accusation trajectories"*, *"expand the portfolio"* | Reads `manifest.json`, identifies coverage gaps, fans out targeted authoring agents to fill the gaps. |
| **review** | *"review and refine the trajectories"*, *"polish the dataset"* | 4 parallel review agents, each refining a bucket: correct accusations, wrong accusations, cold trails + red-herring traps, and a `phrases.json` polish pass for gender neutrality and awkward-seam fixes. |
| **validate** | *"validate the dataset"*, *"check coverage"* | Runs `validate_trajectories.py --all --coverage`. Reports per-trajectory pass/fail and the coverage checklist. |
| **compose** | *"render all the prose"*, *"compose preview"* | Renders every trajectory's prose end-to-end to a single text file for spot-checking. |

## Invocation

The skill is registered at:

```
/Users/juan-garassino/Code/005-products/010-more-than-words/.claude/skills/lt-curate/SKILL.md
```

It is invoked by:

- The slash command `/lt-curate`
- Any natural-language phrasing that matches the trigger list above (Claude detects intent and selects the mode)
- An explicit mode/argument string, e.g. `/lt-curate review amber_cipher`, `/lt-curate expand amber_cipher --bucket cold --count 3`

The skill activates automatically when Claude is in the project workspace and the user's message includes the trigger phrases.

## What the skill expects

Before invocation, the case must have:

```
living_tales/trainer/cases/<case_id>/
├── dimensions.json          # 11-dim schema + token vocabularies (hand-authored)
├── constraints.json         # hard-mask rules (hand-authored)
├── phrases.json             # slot-fill phrases en/es/fr (hand-authored)
├── snippets.json            # reusable scene fragments (hand-authored)
├── VOICES.md                # NPC voice specs (hand-authored)
├── trajectories/
│   ├── manifest.json        # index file
│   └── <flagship>.json      # at least one polished trajectory as voice template
```

If any of these is missing, the skill halts and tells the user what to author first. It will not author them itself.

It also expects:

- Validator at `living_tales/trainer/tools/validate_trajectories.py` is functional
- Composer at `living_tales/trainer/generator/structured_scene_composer.py` is functional
- Python interpreter `/Users/juan-garassino/.pyenv/versions/3.12.9/envs/mySandbox/bin/python` has the required deps (numpy, pygame, etc.)

## Portfolio recipe

The default 25-trajectory portfolio for any case:

| Bucket | Count | Purpose | Example outcomes |
|---|---|---|---|
| Correct accusations | 10 | Different evidence chains converging on the culprit | `correct_<culprit>` |
| Wrong accusations | 8 | Each plausible wrong suspect gets at least one | `wrong_<suspect>` |
| Cold trails | 4 | Player exhausts cards or quits without solving | `cold_trail` |
| Red-herring traps | 3 | High apparent convergence pointing at the wrong target | `red_herring_trap` |

Trajectory lengths: 40–55 turns, except speedruns (~32) and partnership-recovery arcs (~47).

For a new case, the bucket fill is derived from the case's `briefing.suspects` list — every plausible wrong suspect should receive a wrong-accusation trajectory.

## Workflow internals

### Authoring round

The skill fans out 5 parallel authoring subagents. Each receives:

1. The required-reading list (SCHEMA, AUTHORING_TRAJECTORIES, dimensions, constraints, phrases, VOICES, snippets, briefing, flagship trajectory).
2. A precise brief per trajectory (path, length, narrative arc).
3. The full hard-mask constraint reference.
4. A voice checklist citing VOICES.md.
5. The verification command for the validator.

Each subagent produces 1–7 trajectory JSONs, validates each before declaring complete, and updates `manifest.json` (with read-current-state-first to avoid clobbering parallel agents).

Typical bucket assignment for parallel authoring:

| Agent | Trajectories | Approx length |
|---|---|---|
| A1 | 5 correct accusations (paths A–E) | 42 turns each |
| A2 | 4 correct accusations (paths F–I) + 1 speedrun | 42 / 32 turns |
| A3 | 4 wrong accusations (suspects 1–4) | 41 turns each |
| A4 | 4 wrong accusations (suspects 5–7 + partial-correctness) | 42 turns each |
| A5 | 4 cold trails + 3 red-herring traps | 43 / 43 turns |

### Review round

After authoring, the trajectories validate logically but voice/pacing is first-pass. The review round fans out 4 parallel agents, each refining a bucket:

| Agent | Bucket |
|---|---|
| R1 | All correct-accusation trajectories (≈10 files) |
| R2 | All wrong-accusation trajectories (≈8 files) |
| R3 | Cold trails + red-herring traps (≈7 files) |
| R4 | `phrases.json` polish — gender neutrality, dangling prepositions, transition variant authoring |

Each review agent reads every trajectory in its bucket, composes the prose using the SceneComposer, identifies issues (voice slips, awkward seams, pacing flatness, repetition, wrong CAUSE for player-card class, Lyapunov violations), applies surgical fixes via the Edit tool, and re-validates.

### Validation + coverage report

After authoring or review, the skill runs:

```bash
python tools/validate_trajectories.py <case_id> --all --coverage
```

The coverage checklist reports gaps against:

- Each LOCATION token appears in ≥ 4 trajectories
- Each PRESENCE NPC appears in ≥ 3 trajectories with `STANCE != unaware`
- Each OBJECT_FOCUS appears with `REVELATION != none` in ≥ 2 trajectories
- Every CAUSE token appears in ≥ 3 trajectories
- Every REVELATION token (except `none`) appears in ≥ 2 trajectories
- Every TRANSITION token (except `none`) appears in ≥ 5 trajectories
- All 4 BEAT phases represented across the portfolio
- Each suspect has ≥ 1 correct + 1 wrong accusation trajectory

Gaps come back as concrete suggestions — specific trajectories to author next time. The user can then re-invoke `/lt-curate expand` to fill them.

### Compose preview

For human spot-checking, the skill can render every trajectory's prose to a single text file:

```
cases/<case_id>/trajectories/_compose_preview.txt
```

The user can read this in any editor without invoking the live game.

## Hard-mask constraint reference

All authoring agents receive this constraint reference. The skill regenerates it from `constraints.json` per case, but the universal core rules are:

```
- transition:stayed              → LOCATION must equal previous scene's LOCATION
- transition:crossed_to          → LOCATION must differ from previous
- presence:alone                 → STANCE = none, TELL = none
- stance:hostile                 → non-none TELL from physical-tell set
- revelation:contradiction_surfaces → cause:noticed_inconsistency
- revelation:name_uncovered      → convergence_min ≥ 0.5 + OBJECT_FOCUS != none
- revelation:motive_emerges      → convergence_min ≥ 0.5
- beat:closing_in                → convergence_min ≥ 0.5
- beat:verdict_ready             → convergence_min ≥ 0.75
- atmosphere:dawn_approaches     → turn_gte 40
- cause:following_witness_lead   → last player card class WITNESS or SUSPECT
- cause:recognized_object        → last player card class OBJECT or MODIFIER
- cause:summoned                 → PRESENCE != alone
- action:examines                → OBJECT_FOCUS != none
- action:questions / action:confronts → PRESENCE != alone
- action:arrives                 → TRANSITION in {crossed_to, entered, returned, pursued_to, descended_to}
```

Any trajectory turn violating these rules is rejected by the validator before being declared complete.

## Rate-limit awareness

Subagent calls consume the user's rate limit. If the skill hits limits mid-run:

1. It stops launching new agents.
2. It tells the user the reset time and which trajectories remain unfinished.
3. It updates `manifest.json` to reflect what was actually authored, never what was attempted.
4. The user can retry the skill in a narrower bucket once the reset arrives.

This pattern was observed during initial bootstrap of `amber_cipher` — the four review agents launched simultaneously hit the limit, returned no work, and the skill flagged it cleanly.

## What this skill does NOT do

- **It does not train the model.** Training runs on Colab per the project's `feedback_train_on_colab.md` memory. The skill produces only the dataset.
- **It does not author the case foundation.** `dimensions.json`, `constraints.json`, `phrases.json`, `VOICES.md`, and `snippets.json` are hand-authored by the human; the skill assumes they exist. If they don't, the skill halts.
- **It does not generate trajectories without subagents.** The user explicitly cannot author at this scale; the skill never asks them to.
- **It does not run anything beyond the validator and composer.** Output goes to disk; the user takes the rest of the loop manually (Colab push, training, checkpoint download, pygame playtest).

## Output format

Every skill invocation produces a structured summary:

```
Mode: <mode>
Case: <case_id>
Trajectories before: <N>
Trajectories after: <M>
Validation: <X>/<Y> PASS, <Z> violations
Coverage gaps: [list of missing items, or "none"]
Files created: [list]
Files refined: [list]
Notable findings: [up to 3 issues caught during review]
Ready for Colab training: yes / no — <reason>
Next step: <recommended action>
```

This summary is the contract — the user knows exactly what state the dataset is in after every invocation.

## Recommended call patterns

### Bootstrap a new case

```
/lt-curate bootstrap <case_id>
```

Prerequisite: the case foundation files exist. The skill writes the flagship in two chunks (15 turns, then extend to 45), then fans out 5 authoring agents (parallel), then 4 review agents (parallel). End state: ~25 validated trajectories ready for Colab.

### Expand an existing portfolio

```
/lt-curate expand <case_id> [--bucket correct|wrong|cold|red_herring|mixed] [--count N]
```

Identifies coverage gaps, authors only the gaps, runs review only on the new files.

### Review-only refinement pass

```
/lt-curate review <case_id>
```

For dataset that validates clean but needs voice/pacing polish. 4 parallel review agents.

### Validate + report

```
/lt-curate validate <case_id>
```

Read-only — runs the validator + coverage check, reports gaps.

### Compose preview

```
/lt-curate compose <case_id> [--lang en|es|fr]
```

Renders every trajectory's prose to `_compose_preview.txt` for spot-checking.

## Relationship to other Living Tales documents

- **`SCHEMA.md`** — defines the 11-dim engine architecture. The skill operates within this schema; never modifies it.
- **`AUTHORING_TRAJECTORIES.md`** — the human-readable authoring guide. The skill's subagent prompts cite it. If the skill produces awkward outputs, the fix is usually in this guide, not the skill.
- **`IMAGES.md`** — orthogonal track for diffusion-generated UI ornaments. Not used by the skill.
- **`feedback_train_on_colab.md`** (memory) — explains why the skill never trains. Training is a separate manual loop.
- **`feedback_case_authoring.md`** (memory) — the foundational principle that the dataset is hand-authored. The skill respects this by orchestrating subagents (which act as the human's collaborators) rather than running an algorithmic sampler.

## Versioning

Skill version 1.0 — designed against the 11-dim schema (LOCATION, TRANSITION, CAUSE, PRESENCE, STANCE, ACTION, OBJECT_FOCUS, TELL, ATMOSPHERE, REVELATION, BEAT) used by amber_cipher. Future schema versions may require the skill's prompt templates to be updated; the skill itself is data-driven (reads the case's actual `dimensions.json` and `constraints.json` at invocation time) so most schema evolutions will be transparent.

---
name: lt-curate
description: Curate and author Living Tales trajectory training datasets using parallel subagents. Bootstraps full 25-trajectory portfolios for a new case, expands existing portfolios, runs parallel review/refinement passes over agent drafts, validates against hard-mask constraints, runs coverage reports, and composes prose previews. Use when the user asks to "curate trajectories", "author the dataset", "build the trajectory portfolio", "review the trajectories", "expand the dataset for case X", "add more trajectories", "validate the trajectory dataset", or any work involving the hand-authored training data for a Living Tales case. Designed specifically for Living Tales' multidimensional structured-scene engine where the dataset is hand-written (no Hopfield sampling, no LLM at runtime).
---

# Living Tales Trajectory Curator

This skill packages the parallel-subagent workflow for authoring, reviewing, and validating Living Tales trajectory datasets. The user cannot manually author 25 trajectories of 40+ turns each, and a single agent doing it sequentially is slow and rate-limit-prone. The skill fans out subagents for both authoring and review, then synthesizes the results.

## Architectural assumptions

Read these once before invoking the skill — the workflow assumes them:

- The case has a complete `dimensions.json`, `constraints.json`, `phrases.json`, `VOICES.md`, and `snippets.json` already authored at `living_tales/trainer/cases/<case_id>/`.
- The case has at least one polished flagship trajectory (the "gold standard") at `cases/<case_id>/trajectories/<flagship>.json`. New trajectories use this as voice/structure template.
- The validator at `living_tales/trainer/tools/validate_trajectories.py` works.
- The composer at `living_tales/trainer/generator/structured_scene_composer.py` works.
- Python interpreter: `/Users/juan-garassino/.pyenv/versions/3.12.9/envs/mySandbox/bin/python`.

If any of these is missing, halt and tell the user what's needed before proceeding.

## Modes

The skill supports five modes. Detect the user's intent and pick one:

| Mode | When | What happens |
|---|---|---|
| `bootstrap` | New case — no trajectory directory yet, or only flagship | Generate full portfolio plan, fan out ~4-5 authoring agents in parallel, then ~4 review agents |
| `expand` | Existing portfolio + user wants more variants | User specifies count or types ("3 more wrong-accusation trajectories"); fan out matching authoring agents |
| `review` | Trajectories validate but voice/pacing needs polish | Fan out 4 review agents, each refining a bucket |
| `validate` | User wants the coverage report + violation summary | Run `tools/validate_trajectories.py <case_id> --all --coverage`, summarize |
| `compose` | User wants to read all prose end-to-end | Run composer over every trajectory; render to a single text file |

## Portfolio recipe (default for `bootstrap`)

For a 25-trajectory portfolio:

| Bucket | Count | Outcomes |
|---|---|---|
| Correct accusations of the culprit (different evidence chains) | 10 | `correct_<culprit>` |
| Wrong accusations | 8 | `wrong_<each plausible wrong suspect>` |
| Cold trails | 4 | `cold_trail` |
| Red-herring traps | 3 | `red_herring_trap` |

Each trajectory is **40-50 turns** (cold trails up to 55, speedruns may be 30-32, partnership-recovery may be 47).

For new cases, derive the bucket-fill from the case's suspect list (`cases/<case_id>.json` `briefing.suspects`) — every plausible wrong suspect should have at least one wrong-accusation trajectory.

## Authoring round (parallel)

Fan out subagents. Each subagent authors 1-5 trajectories. Use 4-5 parallel agents to keep within rate limits.

**Subagent prompt template (authoring):**

```
Author <N> trajectory drafts for Living Tales case `<case_id>`.

Project root: /Users/juan-garassino/Code/005-products/010-more-than-words

REQUIRED READING (must read in order):
1. living_tales/SCHEMA.md
2. living_tales/AUTHORING_TRAJECTORIES.md
3. living_tales/trainer/cases/<case_id>/dimensions.json
4. living_tales/trainer/cases/<case_id>/constraints.json
5. living_tales/trainer/cases/<case_id>/phrases.json
6. living_tales/trainer/cases/<case_id>/VOICES.md
7. living_tales/trainer/cases/<case_id>/snippets.json
8. living_tales/trainer/cases/<case_id>.json (briefing)
9. living_tales/trainer/cases/<case_id>/trajectories/<flagship>.json (template)

DELIVER <N> trajectory files:
[for each: full path, target length in turns, brief paragraph describing the path]

CONSTRAINTS (validator enforces):
[the full constraint list — see "Constraint reference" section below]

VOICE CHECKLIST per NPC: [reference VOICES.md spec]

VERIFY each:
  cd /Users/juan-garassino/Code/005-products/010-more-than-words/living_tales/trainer
  /Users/juan-garassino/.pyenv/versions/3.12.9/envs/mySandbox/bin/python tools/validate_trajectories.py <case_id> --traj <traj_id>

All must PASS: no violations.

UPDATE manifest.json — append entries (read current state first; do not overwrite parallel agents' entries).

REPORT BACK (under 350 words):
- N files created
- Per-trajectory validation status
- Highlights: 3 turns of composed prose from one trajectory
```

**Bucket assignments for parallel authoring (example for 24-trajectory expansion beyond flagship):**

| Agent | Trajectories | Avg length |
|---|---|---|
| A1 | 5 correct accusations (path A-E) | 42 |
| A2 | 4 correct accusations (path F-I) + 1 speedrun | 42 / 32 |
| A3 | 4 wrong accusations (suspects 1-4) | 41 |
| A4 | 4 wrong accusations (suspects 5-7 + partial-correctness Voss-wrong-motive) | 42 |
| A5 | 4 cold trails + 3 red-herring traps | 43 / 43 |

Five parallel agents. Each reads the same case data. Each writes 5-7 distinct files. No file conflicts.

## Review round (parallel, optional but recommended)

After authoring, the trajectories validate logically but voice/pacing is first-pass. Fan out review agents.

**Subagent prompt template (review):**

```
Review and refine the following N trajectory drafts for Living Tales case `<case_id>`.

REQUIRED READING:
[same as authoring + the trajectories themselves]

FILES TO REVIEW (refine in place via Edit tool):
[list of trajectory paths]

WORKFLOW per trajectory:
1. Compose every turn's prose using SceneComposer (en).
2. Read end-to-end as continuous detective story.
3. Flag issues:
   - Voice slips (NPC stance/tell doesn't match VOICES.md)
   - Awkward seams (action+object grammar doesn't compose)
   - Pacing flatness (too many partial_match in a row, premature closing_in)
   - Repetition (same scene tuple 3+ turns)
   - Wrong CAUSE for player_card class
   - Convergence jumps that violate Lyapunov
4. Apply fixes via Edit tool — surgical, dim-level.
5. Validate: must PASS no violations.

DELIVERABLES:
- All files refined in place
- Brief report (<400 words): most common issue, total fixes, 3 before/after composed-prose examples, final validation status
```

**Bucket assignments for review (4 parallel agents typically):**

| Agent | Bucket | Count |
|---|---|---|
| R1 | All correct-accusation trajectories | ~10 |
| R2 | All wrong-accusation trajectories | ~8 |
| R3 | Cold trails + red-herring traps | ~7 |
| R4 | phrases.json polish (voice + gender neutrality) | 1 file |

R4 is special — it polishes the slot-fill phrases themselves. Common needs: gender-neutral STANCE/TELL phrases (the composer doesn't know NPC gender; phrases authored as "his X" mismatch when PRESENCE is a female NPC), tightening dangling prepositions in ACTION phrases, varying TRANSITION phrases to reduce "Still at X" repetition.

## Validation mode

Run:
```bash
cd /Users/juan-garassino/Code/005-products/010-more-than-words/living_tales/trainer
/Users/juan-garassino/.pyenv/versions/3.12.9/envs/mySandbox/bin/python tools/validate_trajectories.py <case_id> --all --coverage
```

Report:
- Total trajectories
- Per-bucket counts (correct / wrong / cold / red_herring)
- Per-trajectory pass/fail
- Coverage checklist:
  - Each LOCATION ≥ 4 trajectories
  - Each PRESENCE NPC ≥ 3 trajectories with `STANCE != unaware`
  - Each OBJECT_FOCUS appears with `REVELATION != none` ≥ 2 trajectories
  - Each CAUSE ≥ 3 trajectories
  - Each REVELATION (except `none`) ≥ 2 trajectories
  - Each TRANSITION (except `none`) ≥ 5 trajectories
  - All 4 BEAT phases represented
  - Each suspect has ≥ 1 correct + 1 wrong accusation trajectory

If coverage gaps exist, suggest specific trajectories to author (then user can re-invoke skill in `expand` mode).

## Compose mode

Render all trajectories' prose to a single text file:
```bash
cd /Users/juan-garassino/Code/005-products/010-more-than-words/living_tales/trainer
/Users/juan-garassino/.pyenv/versions/3.12.9/envs/mySandbox/bin/python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from generator.structured_scene_composer import SceneComposer
from generator.trajectory_loader import TrajectoryLoader

case_id = '<case_id>'
lang = 'en'
loader = TrajectoryLoader(case_id, Path('../..').resolve())
composer = SceneComposer.load(case_id, lang=lang)

with open(f'cases/{case_id}/trajectories/_compose_preview.txt', 'w') as out:
    for t in loader.load_all():
        out.write(f'\\n=== {t.trajectory_id} ({t.outcome}) — {len(t.turns)} turns ===\\n\\n')
        for turn in t.turns:
            out.write(f'[{turn.turn:2d}] {composer.compose(turn.scene)}\\n')
print('preview written to cases/<case_id>/trajectories/_compose_preview.txt')
"
```

Tell the user the path. They can read it in any editor.

## Constraint reference

Hard-mask rules to communicate to authoring agents (varies by case — read `constraints.json` for the case's actual rules):

```
- transition:stayed → LOCATION must equal previous scene's LOCATION
- transition:crossed_to → LOCATION must differ from previous
- presence:alone → STANCE=none, TELL=none
- stance:hostile → non-none TELL from physical-tell set
- revelation:contradiction_surfaces → cause:noticed_inconsistency
- revelation:name_uncovered → convergence_min ≥ 0.5 + OBJECT_FOCUS != none
- revelation:motive_emerges → convergence_min ≥ 0.5
- beat:closing_in → convergence_min ≥ 0.5
- beat:verdict_ready → convergence_min ≥ 0.75
- atmosphere:dawn_approaches → turn_gte 40
- cause:following_witness_lead → last player card class WITNESS or SUSPECT
- cause:recognized_object → last player card class OBJECT or MODIFIER
- cause:summoned → PRESENCE != alone
- action:examines → OBJECT_FOCUS != none
- action:questions/confronts → PRESENCE != alone
- action:arrives → TRANSITION in {crossed_to, entered, returned, pursued_to, descended_to}
```

## Rate-limit awareness

Subagent calls consume the user's rate limit. If you hit limits during a curate run:

1. Stop launching new agents.
2. Tell the user clearly: "Rate limit reached at <time>. Reset at <reset time per your env>. <N>/<total> trajectories complete. Remaining: <list>."
3. Save state — make sure manifest.json reflects what was actually authored, not what was attempted.
4. Suggest the user retry the skill with `--continue` or with a narrower bucket once rate limit resets.

## Recommended call patterns

**Bootstrap a new case** (25 trajectories from scratch, given case data is ready):
1. Verify all required case files exist.
2. Author the flagship trajectory yourself or with one focused agent (15-25 turns first, then extend to 45 with another agent — chunked authoring is more reliable than 45-turn single-shot).
3. Once flagship validates clean, fan out 5 parallel authoring agents (5-7 trajectories each).
4. After authoring round completes, fan out 4 review agents in parallel.
5. Run validation + coverage; report any gaps.

**Expand existing portfolio**:
1. Read manifest.json to see what exists.
2. Identify gaps via coverage report.
3. Fan out authoring agents targeting only the gaps.
4. Fan out review agents over only the new files.
5. Re-validate.

**Review pass**:
1. Read manifest.json.
2. Bucket by outcome.
3. Fan out 4 review agents (one per bucket + phrases.json).
4. Synthesize report.

## Output format

After every skill invocation, give the user a structured summary:

```
Mode: <mode>
Trajectories before: <N>
Trajectories after: <M>
Validation: <X>/<Y> PASS, <Z> violations
Coverage gaps: [list of missing items]
Files created: [list]
Files refined: [list]
Notable findings: [up to 3 voice/pacing issues caught by review agents]
Next step: [recommended user action]
```

Always tell the user whether the dataset is ready for Colab training (≥ 80% coverage thresholds met, all PASS). If not, say what's still needed.

## Inputs the skill expects

When invoked, parse the user's request for:

- **case_id** — required. Defaults to `amber_cipher` if not specified.
- **mode** — `bootstrap` | `expand` | `review` | `validate` | `compose`. Inferred from phrasing if not explicit.
- **count** (for `expand`) — how many new trajectories. Default 5.
- **bucket** (for `expand`) — `correct` / `wrong` / `cold` / `red_herring` / `mixed`. Default `mixed`.
- **lang** (for `compose`) — `en` | `es` | `fr`. Default `en`.

## What this skill does NOT do

- Train the model. Training runs on Colab per the project's `feedback_train_on_colab.md` memory. The skill only handles dataset.
- Author SCHEMA / dimensions / constraints / phrases / VOICES files. Those are the case foundation; the skill assumes they exist. If they don't, the skill halts and tells the user.
- Generate trajectories without subagents. The user explicitly cannot manually author at this scale; the skill never asks them to.
- Run any tools or commands beyond the validator and composer.

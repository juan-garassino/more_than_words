# Authoring Trajectories — Hand-Written Training Data

This document is the reference for hand-writing the training dataset for a Living Tales case. Read it once before writing your first trajectory; refer back to specific sections as questions arise.

The training data is **hand-authored**, not algorithmically sampled. This is a foundational design choice: the engine learns from playthroughs you wrote yourself, not from generated approximations. The trade-off is volume (~500 turn pairs hand-written vs. millions algorithmically sampled), but every example is a deliberate, intended slice of how the case should play.

The transformer is a **tiny per-case overfitter**. It learns to reproduce your trajectories. At inference, given a player choice + current state, it emits the dim tuple closest to what your authored trajectories show in similar situations. Hard-mask constraints filter logical impossibilities; the graph contributes a soft preference. Everything else comes from your writing.

---

## What a trajectory is

A trajectory is **one complete playthrough of the case from opening to ending**. A typical amber_cipher trajectory is 30–60 turns long.

Each trajectory file represents one possible path through the case: one set of player choices, one set of engine responses, one ending. To give the model coverage, you author **multiple distinct trajectories** per case — different starting hypotheses, different paths to the truth, different failure modes.

Recommended per-case trajectory portfolio (15–25 trajectories total):

| Category | Count | Purpose |
|---|---|---|
| Correct accusations via different paths | 5–8 | Player reaches the truth through different chains of evidence |
| Wrong accusations | 3–5 | Player names the wrong suspect; covers each plausible wrong-suspect once |
| Cold trails | 2–3 | Player exhausts cards or quits without accusing |
| Red-herring traps | 2–3 | Player reaches high apparent convergence on red-herring tokens, then accuses wrong |
| Edge paths | 2–4 | Unusual orderings (revisiting locations, late witness reveals, etc.) |

**Trajectories share scenes but in different orders and contexts.** A trajectory authored as "Voss via cufflink" will share many scene tuples with "Voss via telegraph" — what differs is which token the player plays first, which alters CAUSE, REVELATION, and BEAT trajectories. That overlap teaches the model that the same evidence can surface in different orders.

---

## File layout

```
living_tales/trainer/cases/<case_id>/trajectories/
├── manifest.json
├── voss_via_cufflink.json
├── voss_via_cipher_sheet.json
├── webb_via_boots.json
├── cold_trail_lampgame.json
├── red_herring_partnership.json
└── ...
```

`manifest.json` is the index — it tells the loader which trajectories exist and what outcome each represents.

```json
{
  "case_id": "amber_cipher",
  "schema_version": 1,
  "trajectories": [
    {
      "id": "voss_via_cufflink",
      "outcome": "correct_voss",
      "length": 38,
      "starting_hypothesis": "object_first",
      "tags": ["cufflink_chain", "RV_initials"]
    },
    {
      "id": "cold_trail_lampgame",
      "outcome": "cold_trail",
      "length": 60,
      "starting_hypothesis": "atmospheric_red_herrings",
      "tags": ["lamp_focus", "no_witness_engaged"]
    }
  ]
}
```

Outcomes (controlled vocabulary):
- `correct_<suspect_id>` — player accused correctly
- `wrong_<suspect_id>_via_<reason>` — player accused wrong (specific to which wrong suspect and why)
- `cold_trail` — player ran out of cards or quit
- `red_herring_trap` — player reached high apparent convergence on red-herring tokens, then accused wrong

---

## Single-trajectory schema

Each `<traj_id>.json` file:

```json
{
  "schema_version": 1,
  "trajectory_id": "voss_via_cufflink",
  "case_id": "amber_cipher",
  "description": "Player suspects Voss early via the cufflink, confronts him in the office, reaches verdict at dawn",
  "outcome": "correct_voss",
  "starting_hypothesis": "object_first",
  "tags": ["cufflink_chain", "RV_initials"],

  "opening": {
    "_comment": "Tokens placed by the engine before turn 1. Mirrors case spec opening_token_ids.",
    "tokens": [
      "location:thornfield_crossing",
      "event:aldous_verne_discovered",
      "time:between_trains"
    ]
  },

  "turns": [
    {
      "turn": 1,
      "_note": "Player picks up the cufflink at the platform — first object examination",
      "player_card": "object:initialed_cufflink",
      "scene": {
        "LOCATION":     "location:thornfield_crossing",
        "TRANSITION":   "transition:stayed",
        "CAUSE":        "cause:examining_evidence",
        "PRESENCE":     "presence:alone",
        "STANCE":       "stance:none",
        "ACTION":       "action:examines",
        "OBJECT_FOCUS": "object:initialed_cufflink",
        "TELL":         "tell:none",
        "ATMOSPHERE":   "atmosphere:fog_thickens",
        "REVELATION":   "revelation:partial_match",
        "BEAT":         "beat:orientation"
      },
      "convergence_after": [0.05, 0.05, 0.42]
    },
    {
      "turn": 2,
      "_note": "Player questions ticket clerk about the cufflink owner",
      "player_card": "witness:ticket_clerk",
      "scene": {
        "LOCATION":     "location:platform_two",
        "TRANSITION":   "transition:crossed_to",
        "CAUSE":        "cause:following_witness_lead",
        "PRESENCE":     "presence:with_ticket_clerk",
        "STANCE":       "stance:cooperative",
        "ACTION":       "action:questions",
        "OBJECT_FOCUS": "object:initialed_cufflink",
        "TELL":         "emotion:steady_hands",
        "ATMOSPHERE":   "atmosphere:lamp_steady",
        "REVELATION":   "revelation:partial_match",
        "BEAT":         "beat:orientation"
      },
      "convergence_after": [0.10, 0.15, 0.50]
    }
    // ... continue for ~30–60 turns total ...
  ],

  "ending": {
    "type": "all_strong",
    "accused": "suspect:renard_voss",
    "final_convergence": [0.85, 0.78, 0.92]
  }
}
```

Required fields per turn:

- `turn` — 1-indexed turn number (sanity check against ordering bugs).
- `player_card` — the token the player played that turn. Must be a valid player card per the case's `player_cards` declaration. Travel cards (`travel:to_office`) are valid here too.
- `scene` — the 11-dim engine response tuple. Every dim slot must be filled (see "all dims always fire" decision in the plan).

Optional fields:

- `_note` — a free-text comment for your own reference; the loader ignores it.
- `convergence_after` — the convergence vector after this turn. Helpful for sanity-checking Lyapunov monotonicity. The validator can recompute it from `attractor_weights`; if you write it explicitly, it's checked against the computed value.

---

## How to write a trajectory

Recommended workflow:

### 1. Sketch the path before writing JSON

In a scratch buffer, write the trajectory's *intended player journey* in plain prose. Example:

> *"Voss-via-cufflink: Player examines the cufflink first (turn 1). Engine: alone at the crossing, partial match. Player questions the ticket clerk about it (turn 2). Clerk recognizes initials but plays it cool. Player checks the telegraph form (turn 3). Coal dust on the form — connects to Voss's industrial business. Player visits the goods shed (turn 4). Voss is there, evasive, won't make eye contact. Player notices coal dust on his coat (turn 5). Confrontation, name uncovered. ..."*

This sketch is your authoring outline. Each paragraph is a turn or two.

### 2. Translate sketch → 11-dim tuples

For each turn, fill the 11 dims:

- **LOCATION**: where is this scene? (One of the case's location tokens.)
- **TRANSITION**: how did we arrive here from the previous LOCATION? (`stayed`, `crossed_to`, `entered`, `returned`, `pursued_to`, `called_away_to`, `descended_to`. First turn always `none`.)
- **CAUSE**: what triggered this scene? Usually maps directly from the player's card type (witness card → `following_witness_lead`; object card → `examining_evidence`; etc.).
- **PRESENCE**: who is there? (`with_<id>` or `alone`.)
- **STANCE**: their disposition. `none` if alone.
- **ACTION**: what is the detective doing? (`examines`, `questions`, `notices`, `discovers`, `confronts`, `follows`, `waits`, `leaves`, `arrives`, `recalls`.)
- **OBJECT_FOCUS**: physical object in focus, or `none`.
- **TELL**: observed emotional cue, or `none`.
- **ATMOSPHERE**: environmental beat. Choose to support the scene's mood.
- **REVELATION**: what does this scene yield? Most early turns are `partial_match` or `none`. `name_uncovered` and `motive_emerges` only after convergence ≥ 0.5.
- **BEAT**: `orientation` early, `investigation` middle, `closing_in` after convergence ≥ 0.5, `verdict_ready` after ≥ 0.75.

### 3. Validate as you go

Run the validator after every 5–10 turns:

```bash
cd living_tales/trainer
python tools/validate_trajectories.py amber_cipher --traj voss_via_cufflink
```

The validator checks:
- All dim slots are present and use legal token IDs.
- All hard-mask constraints in `constraints.json` hold for each turn.
- Lyapunov monotonicity holds (convergence is non-decreasing modulo decay).
- The trajectory length is within the case's `min_turns`–`max_turns` bounds.
- The ending is consistent with `final_convergence` and `outcome`.

### 4. Final pass: read the prose

Run the composer over each turn:

```bash
python tools/compose_trajectory.py amber_cipher --traj voss_via_cufflink --lang en
```

Read the resulting prose end to end. Does the case feel like a real Obra-Dinn-style detective playthrough? If a turn reads awkward or tonally off, the dim tuple is probably the wrong choice. Re-author that turn.

---

## What makes a good trajectory

- **A clear chain of evidence.** Each turn either (a) examines something new, (b) connects to something already seen, or (c) provides a transition between locations. Avoid "filler" turns where nothing happens.
- **Stance shifts as a curve.** Early NPC scenes are often `cooperative` or `unaware`; mid-game shifts to `defensive` or `evasive`; late game can hit `hostile`. A trajectory that stays at a single stance from turn 1 to 50 reads flat.
- **Atmospheres mark act breaks.** `atmosphere:fog_thickens` works for early; `atmosphere:silence_holds` for mid-game tension; `atmosphere:dawn_approaches` only late. Use them to mark time passing.
- **Revelations are rare and earned.** Most turns: `revelation:none` or `revelation:partial_match`. Big revelations (`name_uncovered`, `motive_emerges`) only at convergence ≥ 0.5, and they should land *because* of the player's recent move, not arbitrarily.
- **Convergence as gate, not goal.** A good trajectory reaches `verdict_ready` (convergence min ≥ 0.75) at turn ~40–50, not turn 10. Lyapunov-monotone progress through the dims, not a sprint.
- **Cold trails feel different.** A `cold_trail` trajectory has more `dead_end` revelations, more `none` tells, more `lamp_flickers` atmospheres. The fog never lifts.
- **Red-herring traps look convincing on the surface.** A `red_herring_trap` trajectory has high apparent convergence (because red-herring tokens carry inflated attractor weights) but the BEAT never reaches `verdict_ready` cleanly, or it does but for the wrong suspect.

---

## Coverage checklist

Before declaring the trajectory portfolio complete for a case, verify:

- Every LOCATION token appears in at least 2 trajectories.
- Every PRESENCE NPC (suspect + witness) appears in at least 2 trajectories with `STANCE != unaware`.
- Every OBJECT_FOCUS token appears as the focus of at least 1 turn.
- Every CAUSE token appears in at least 1 turn.
- Every REVELATION token (except `none`) appears in at least 1 turn.
- Every TRANSITION token (except `none`) appears in at least 2 trajectories.
- The trajectory portfolio covers all 4 BEAT phases (orientation → investigation → closing_in → verdict_ready).
- Each suspect has at least 1 `correct_<suspect>` accusation trajectory and at least 1 `wrong_<suspect>` trajectory (so the model sees both).

The validator's `--coverage` flag prints this checklist with pass/fail per item.

---

## Worked example: a 5-turn opening

Here's the first 5 turns of `voss_via_cufflink` written out fully, with the resulting composed prose underneath each turn.

```json
"turns": [
  {
    "turn": 1,
    "player_card": "object:initialed_cufflink",
    "scene": {
      "LOCATION":     "location:thornfield_crossing",
      "TRANSITION":   "transition:stayed",
      "CAUSE":        "cause:examining_evidence",
      "PRESENCE":     "presence:alone",
      "STANCE":       "stance:none",
      "ACTION":       "action:examines",
      "OBJECT_FOCUS": "object:initialed_cufflink",
      "TELL":         "tell:none",
      "ATMOSPHERE":   "atmosphere:fog_thickens",
      "REVELATION":   "revelation:partial_match",
      "BEAT":         "beat:orientation"
    }
  }
]
```

Composed:

> *Tracing the evidence, still at Thornfield Crossing. The detective examined the initialed cufflink. Outside, the fog thickened around the lamps. Close — but not certain.*

```json
{
  "turn": 2,
  "player_card": "witness:ticket_clerk",
  "scene": {
    "LOCATION":     "location:platform_two",
    "TRANSITION":   "transition:crossed_to",
    "CAUSE":        "cause:following_witness_lead",
    "PRESENCE":     "presence:with_ticket_clerk",
    "STANCE":       "stance:cooperative",
    "ACTION":       "action:questions",
    "OBJECT_FOCUS": "object:initialed_cufflink",
    "TELL":         "emotion:steady_hands",
    "ATMOSPHERE":   "atmosphere:lamp_steady",
    "REVELATION":   "revelation:partial_match",
    "BEAT":         "beat:orientation"
  }
}
```

Composed:

> *On the witness's word, crossed to Platform Two. The ticket clerk peered through the window, open, willing to speak. The detective questioned the initialed cufflink. His hands stayed steady. The lamps held steady. Close — but not certain.*

(*Note: "questioned the initialed cufflink" is an awkward seam — `action:questions` reads better with a person object. This is the kind of authoring detail to refine in a final pass. Either change ACTION to `examines` here, or accept that for a witness scene the OBJECT_FOCUS becomes the topic of questioning rather than the literal grammatical object. Author judgment.*)

Continue this way for ~35 more turns until the trajectory reaches an ending.

---

## How the trainer consumes trajectories

The training pipeline:

1. `trajectory_loader.py` reads every `.json` in `cases/<case>/trajectories/`.
2. Each turn becomes a training example: `(history_so_far, player_card_at_t) → scene_tuple_at_t`.
3. History is encoded as the prior turns' player cards + scene tuples.
4. A tiny `StructuredSceneTransformer` (2–3 layers, ~500K params) is trained with per-head NLL/CE loss against the authored scene tuples.
5. Heavy regularization keeps the model from drifting outside the authored distribution.
6. Hard masks (from `constraints.json`) are applied at inference; the model literally cannot emit logically impossible combinations.

Training is **fast** — minutes on a laptop, not hours on Colab. Overfitting is the goal: the model memorizes the authored playthroughs and interpolates between them when the player makes choices not exactly matching any trajectory.

The cartridge ships as `outputs/<case>/dialogue_model.pt` (the trained tiny model) plus the case JSONs (dimensions, phrases, constraints, trajectories). Total size: ~5MB per case.

---

## What you do NOT need to author

These are derived from the case spec and don't need separate hand-writing:

- The token graph edges (already in `cases/<case>/graph.json`).
- The convergence math (`attractor_weights` per token, `convergence_rate`, `min_turns`).
- The opening tokens (already in `opening_token_ids`).
- The endings prose (already in `case_data["endings"]` and `ending_fragments`).
- Multi-language phrases (in `phrases.json` — authored once, not per-trajectory).

You only author:
- The sequences of turns: `(player_card, scene_tuple)` pairs.
- The path metadata (description, outcome, tags) per trajectory.

---

## Volume guidance

For a flagship case like amber_cipher, target:

- **Minimum viable**: 5 trajectories, ~200 turn pairs total. Model will overfit hard; gameplay variety limited.
- **Solid**: 15 trajectories, ~600 turn pairs total. Good coverage; engine produces convincing scenes for most player choices.
- **Strong**: 25 trajectories, ~1000 turn pairs total. Each suspect has ≥3 paths; cold trails and red-herring traps well represented; replay value is real.

Each trajectory takes 1–3 hours to author once you're fluent with the dim schema. Total authoring budget per case: 5–10 days.

For subsequent cases (after amber_cipher proves the workflow), expect 4–6 days per case — your first case teaches you the schema; later cases are faster.

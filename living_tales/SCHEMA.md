# Living Tales — Multidimensional Situational-Token Schema

This document specifies the dimensional output architecture for the Living Tales engine. It is the authoritative replication artifact: any case that follows this schema becomes playable on the new structured engine.

The engine still operates only on token IDs and float weights. There is no LLM. The novelty is in *what kinds* of tokens the model emits per turn — a complete situational tableau across multiple dimensions, rather than a flat tuple of attractor predictions.

The aesthetic target is **Obra Dinn**: methodical, cold, reconstructive. Third-person past tense. The player is a detective rebuilding what happened, not a participant living through it.

---

## Why dimensions

In the legacy engine, each turn the model emits one token per attractor head (typically 3: who/how/why). The heads are independent and the result is a list of facts, not a scene. There is no architectural place for *flow* (movement between locations), *cause* (why the scene shifted), *atmosphere* (environmental beats), or *story rhythm* (act breaks). All of that must be coaxed out of the surface prose layer.

The dimensional schema makes these first-class. Every turn the engine emits exactly one token per dimension. The dimensions taken together describe a complete *moment*: where, when, with whom, doing what, with what tell, in what mood, yielding what discovery. The renderer slot-fills authored phrase fragments deterministically. The result reads as a scene because the structure of a scene is built into the architecture.

Coherence is guaranteed by **hard masks** (declarative rules that forbid logical impossibilities) and **graph weights** (soft preferences for plausible combinations). The model learns the rest from training trajectories.

---

## Universal core dimensions

Every case in the new engine emits these ten dimensions every turn. They are shared across all cases. Each case authors a small case-specific vocabulary per dimension.

### 1. `LOCATION`

The place where the scene is set.

- Vocab: case-specific places + `none`
- Cardinality: typically 8–12 per case
- Drives: backdrop swap in pygame; spatial logic for transitions
- Example tokens (amber_cipher): `platform_two`, `goods_shed`, `signal_box`, `station_office`, `none`

### 2. `TRANSITION`

The relation of this scene's location to the previous scene's location.

- Vocab: shared across cases + `none`
- Cardinality: 8 universal tokens
- Drives: motion narration ("Crossed to the office")
- Universal tokens:
  - `stayed` — same location as previous
  - `crossed_to` — moved to adjacent space within complex
  - `entered` — moved into an enclosed space
  - `exited` — moved out of an enclosed space
  - `returned` — back to a previously visited place
  - `pursued_to` — followed a lead/person to a new place
  - `called_away_to` — summoned by another character
  - `descended_to` — moved to a lower/hidden level
  - `none` — first scene, no transition

### 3. `CAUSE`

What triggered this scene — the link to the player's just-played token. Always answers "why are we now seeing this?"

- Vocab: shared core + case-specific extensions + `none`
- Cardinality: ~12 per case
- Drives: causal narration ("Following the porter's hint, ...")
- Universal core tokens:
  - `following_witness_lead` — a witness pointed somewhere
  - `examining_evidence` — player's action focused on physical clue
  - `recognized_object` — something just clicked about an object
  - `noticed_inconsistency` — contradiction surfaced
  - `pursued_suspicion` — followed a hunch about a suspect
  - `revisiting_scene` — went back to confirm
  - `summoned` — engaged by another character
  - `chronological_drift` — time passed, scene shifted naturally
  - `none` — first scene, no cause

### 4. `PRESENCE`

Who is physically present with the detective in this scene.

- Vocab: case-specific (suspects + witnesses) + `alone` + `none`
- Cardinality: ~13 per case
- Drives: who's in the scene; who can speak/react
- Tokens like: `with_price`, `with_voss`, `alone`, `observed_by_porter`

### 5. `STANCE`

The disposition of the present NPC toward the detective. Only meaningful when `PRESENCE != alone`.

- Vocab: shared, ~6 tokens
- Drives: how the NPC behaves in slot-fill
- Universal tokens:
  - `cooperative` — open, helpful
  - `defensive` — guarded, wary
  - `evasive` — actively avoiding subject
  - `hostile` — confrontational
  - `unaware` — doesn't know they're being observed
  - `none` — alone or not applicable

### 6. `ACTION`

What the detective is physically doing in this scene.

- Vocab: shared core + case-specific extensions
- Cardinality: ~12 per case
- Drives: verb in scene's main clause
- Universal core tokens (verbs in third-person past, infinitive form here):
  - `examines` — close inspection
  - `questions` — direct interrogation
  - `notices` — passive observation
  - `discovers` — sudden recognition
  - `confronts` — accusation/pressure
  - `follows` — pursues a lead
  - `waits` — passive presence
  - `leaves` — departs scene
  - `arrives` — enters a new place
  - `recalls` — connects to prior knowledge
  - `none` — no defining action

### 7. `OBJECT_FOCUS`

The physical object in focus this scene, or `none`.

- Vocab: case-specific objects + `none`
- Cardinality: ~14 per case
- Drives: what the action is operating on
- Tokens like: `cufflink`, `telegram`, `boot_print`, `lamp`, `none`

### 8. `TELL`

An observable affective signal — the scene's "tell" that something hidden is surfacing. Only meaningful with `PRESENCE != alone`, often `none`.

- Vocab: shared, ~8 tokens
- Universal tokens:
  - `paled` — visible color drained
  - `tightened` — body went rigid
  - `glanced_aside` — broke eye contact
  - `silent` — refused to speak
  - `hesitated` — paused before answering
  - `over_explained` — unprompted volume of detail
  - `softened` — stance eased
  - `none` — no tell

### 9. `ATMOSPHERE`

The environmental beat. The world doing something around the detective.

- Vocab: shared core + case-specific extensions
- Cardinality: ~6 per case
- Drives: the closing or framing sentence of the slot-fill
- Universal core tokens:
  - `fog_thickens`, `fog_lifts`
  - `lamp_flickers`, `lamp_steady`
  - `silence_holds`, `silence_breaks`
  - `train_passes`, `dawn_approaches`
  - `wind_rises`
  - `none`

### 10. `REVELATION`

What the scene yields — the cognitive/narrative output. Critical for player progress.

- Vocab: shared core + case-specific extensions
- Cardinality: ~8 per case
- Drives: discovery-beat language; journal updates
- Universal core tokens:
  - `contradiction_surfaces` — two facts no longer agree
  - `alibi_holds` — a defence checks out
  - `name_uncovered` — someone's identity becomes legible
  - `motive_emerges` — a why becomes plausible
  - `dead_end` — the line goes nowhere
  - `partial_match` — close but uncertain
  - `confirmation` — an earlier hypothesis verified
  - `none` — no revelation this turn

### 11. `BEAT` (story rhythm)

A coarse signal of narrative phase. Used by the renderer to vary tone (early scenes are orientational, late scenes are tense).

- Vocab: shared, fixed 4 tokens
- Cardinality: 4
- Universal tokens:
  - `orientation` — opening, taking stock
  - `investigation` — main body
  - `closing_in` — pieces converging
  - `verdict_ready` — convergence min ≥ 0.75; player should accuse

`BEAT` is gated by hard masks against convergence state.

---

## Case-specific extensions

A case may declare additional dimensions on top of the universal 10. Examples:

- `attended_hour` (hospital): adds `MEDICAL_TELL` (e.g. `pulse_irregular`, `pupil_response`).
- `venetian_mirror` (haunted): adds `OMINOUS_SIGN` (e.g. `mirror_fogged`, `unfamiliar_face_appears`).

Case-specific dimensions follow the same authoring rules: phrase fields per token in en/es/fr; hard-mask integration; cardinality 4–8 tokens.

---

## Token annotation rules

Every token in a case JSON gains a `dim` field naming its dimension:

```json
{
  "id": "object:cufflink",
  "class": "OBJECT",
  "dim": "OBJECT_FOCUS",
  "phase": "MID",
  "agency": "PLAYER",
  "attractor_weights": [0.05, 0.08, 0.42],
  ...
}
```

The legacy `class` field stays (used by some validators), but `dim` is now authoritative for emission slots. Tokens that don't fit a dim get `dim: null` and are not emitted; they remain as graph anchors / convergence weights only.

---

## Phrase fields

Each dim-bearing token gets a `phrase` block in `cases/<case>/phrases.json`:

```json
{
  "object:cufflink": {
    "phrase": {
      "en": "the cufflink lay where it had been left",
      "es": "el gemelo descansaba donde había sido dejado"
    }
  },
  "transition:crossed_to": {
    "phrase": {
      "en": "crossed to",
      "es": "cruzó hasta"
    }
  }
}
```

### Voice rules (Obra Dinn)

- **Third-person past tense.** Subject is "the detective" (or `she`/`he` once established).
- **Cold, observational, terse.** No interior monologue. No "I felt", no "she wondered". The world acts on the detective; the detective acts on the world. Period.
- **Sensory grounding required.** At least one sensory anchor per scene (light, sound, texture, temperature). Atmosphere dim usually carries it.
- **No editorializing.** "Price's shoulders tightened" — yes. "Price was clearly nervous" — never.
- **Length budget.** Each dim's phrase is 2–8 words. The composer assembles them into a paragraph of 30–60 words.

### Slot-fill grammar

The composer assembles phrases into sentences via a grammar. The default scene template:

```
[ Sentence 1: motion ]
   {transition.phrase} {location.phrase}.
[ Sentence 2: presence + stance ]
   {presence.phrase}, {stance.phrase}.
[ Sentence 3: action + object ]
   The detective {action.phrase} {object_focus.phrase}.
[ Sentence 4: tell — only if PRESENCE != alone and TELL != none ]
   {presence.subject_phrase} {tell.phrase}.
[ Sentence 5: atmosphere ]
   {atmosphere.phrase}.
[ Sentence 6: revelation — only if REVELATION != none ]
   {revelation.phrase}.
```

Phrase fields are authored to compose grammatically when slot-filled. Example:

```
TRANSITION: crossed_to        → "crossed to"
LOCATION:   station_office    → "the station office"
PRESENCE:   with_price        → "Price was waiting at the desk"
STANCE:     defensive         → "his hands flat on the leather"
ACTION:     examines          → "examined"
OBJECT:     cufflink          → "the cufflink"
TELL:       tightened         → "His jaw tightened when she picked it up"
ATMOSPHERE: fog_thickens      → "Outside, the fog thickened around the lamps"
REVELATION: name_uncovered    → "The initials read R.V."
```

Composes to:

> *"Crossed to the station office. Price was waiting at the desk, his hands flat on the leather. The detective examined the cufflink. His jaw tightened when she picked it up. Outside, the fog thickened around the lamps. The initials read R.V."*

---

## Hard-mask grammar

Logical impossibilities are declared in `cases/<case>/constraints.json` and enforced at inference. Each rule is a conditional implication on emitted dim values. Format:

```json
{
  "rules": [
    {
      "if": {"TRANSITION": "stayed"},
      "then": {"LOCATION": "@equals_previous_scene_location"}
    },
    {
      "if": {"TRANSITION": "crossed_to"},
      "then": {"LOCATION": "@differs_from_previous_scene_location"}
    },
    {
      "if": {"PRESENCE": "alone"},
      "then": {"STANCE": "none", "TELL": "none"}
    },
    {
      "if": {"STANCE": "hostile"},
      "then": {"TELL": "@in", "values": ["paled", "tightened", "silent", "glanced_aside"]}
    },
    {
      "if": {"BEAT": "verdict_ready"},
      "then": {"@convergence_min_gte": 0.75}
    },
    {
      "if": {"REVELATION": "name_uncovered"},
      "then": {"@convergence_min_gte": 0.5}
    }
  ]
}
```

Special predicates (`@equals_previous_scene_location`, `@convergence_min_gte`, etc.) are evaluated at runtime against game state. The constraint compiler converts each rule into a per-head boolean mask that zeros logits violating the rule.

### Hard-mask layer semantics

At inference, after the model produces raw logits per head:

1. Apply per-head valid mask (the dim's vocabulary subset, intersected with phase availability, intersected with `placed_ids` exclusion).
2. Apply hard-mask rules (zero out impossible combinations given dims emitted earlier in this scene + game state).
3. Apply graph-weight logit bias (current mechanism — soft preferences).
4. Sample.

Hard masks come *before* graph bias; the model literally cannot emit logically impossible tokens.

---

## Graph edges in dimensional space

The token graph stays. Edges now carry a `type` field:

```json
{
  "from": "object:cufflink",
  "to": "suspect:voss",
  "weight": 0.50,
  "type": "identifier"
}
```

Edge types (used to vary slot-fill voice):

- `physical` — co-location ("X was beside Y")
- `identifier` — possession or marking ("X belonged to Y")
- `tell` — emotional cue ("X gave Y away")
- `causal` — direct chain ("X led to Y")
- `contradiction` — incompatibility ("X did not match Y")
- `thematic` — narrative resonance, no logical claim

Edges can be within-dim (location ↔ location for transition logic) or cross-dim (location ↔ presence for "who's here"). Both are stored uniformly.

---

## Convergence as gate

Convergence math stays unchanged for the original 3 attractor dims (who/how/why). What changes is what convergence *unlocks*:

- Convergence < 0.25 → BEAT is `orientation`. `REVELATION:name_uncovered` and `REVELATION:motive_emerges` are masked off.
- 0.25 ≤ convergence < 0.5 → BEAT is `investigation`. Most revelations available except final-name ones.
- 0.5 ≤ convergence < 0.75 → BEAT is `closing_in`. All revelations available. `verdict_ready` masked.
- convergence ≥ 0.75 → BEAT is `verdict_ready`. Player should accuse.

The player can accuse at any time. The accusation outcome (correct/incorrect/cold) depends on which suspect they choose, not on convergence. Convergence is purely a gate on what kinds of tokens the engine can emit.

Failure modes:

- **Wrong accusation**: player picks the wrong suspect. Renders `wrong_accusation` ending.
- **Cold trail**: player exhausts cards, never reaches verdict_ready, walks away. Renders `cold_case` ending.
- **Red herring trap**: red-herring tokens have inflated attractor weights. Playing them raises convergence_min falsely. The model can emit `verdict_ready` BEAT and the player feels confident — but the suspect they identify will be wrong. Hidden trap.

---

## Player input

Player cards now include both **inquiry** tokens (witnesses, objects, motives, emotions — current model) and **travel** tokens (e.g. `travel:visit_office`, `travel:return_to_platform`).

- Inquiry cards play as today: bias the engine toward responses connected to the played token via graph weights.
- Travel cards force `LOCATION` and `TRANSITION` — the engine emits the requested place and a sensible transition. Other dims are filled normally.

Initial hand mix: ~5 inquiry + 2 travel cards. Refilled phase-aware (already implemented in `pygame_play.py:GameState.refill_hand`).

---

## Journal (J key)

The journal is a runtime data structure built from the dialogue history. It auto-fills on every emitted scene tuple. The pygame UI renders it on demand (J key toggle). Sections:

1. **Locations visited** — every unique `LOCATION` token emitted, with its backdrop thumbnail, the turn number when first reached, and a one-line note (`location.summary` field per token).
2. **People met** — every `PRESENCE` token (suspects + witnesses), with their briefing intro, the turn first met, and the most recent meaningful line they spoke.
3. **Evidence** — every `OBJECT_FOCUS` token emitted with `REVELATION != none` in the same scene (i.e., evidence that yielded something). Includes the revelation it triggered.
4. **Timeline** — chronological list of `REVELATION` tokens, in turn order, with the scene paragraph that produced each.

The journal is purely a view on `dialogue_history`. No persistent state beyond the existing game state.

---

## Cartridge spec extensions

`CartridgeSpec` (in `core/cartridge.py`) gains:

- `dimensions: List[Dim]` — declared dim schema for this case.
- `dim_vocab: Dict[str, List[str]]` — token IDs per dim.
- `constraints: List[ConstraintRule]` — compiled hard-mask rules.
- `phrases: Dict[str, Dict[str, str]]` — token_id → lang → phrase.

Backward-compat: cases without these fields fall back to the legacy 3-head architecture. Migration is strictly opt-in per case.

---

## Authoring checklist (per case)

To migrate a case to the new schema:

1. **Tag tokens** — every existing token gets a `dim` field. ~3 hours.
2. **Add dim-specific tokens** — transitions, causes, atmospheres, beats not in the original case. ~1 day of design + naming.
3. **Author phrases** — phrase per token in en + es (+ fr if supported). ~270 short strings, 1.5 days of writing.
4. **Author constraints** — ~20 hard-mask rules. Half a day.
5. **Author cross-dim graph edges** — ~30 new edges (location ↔ presence, action ↔ object, etc.). Half a day.
6. **Validate** — `python tools/validate_dim_schema.py <case>` checks coverage, runs the composer on synthetic tuples, asserts no impossible combinations sample.
7. **Sample trajectories** — `python tools/sample_dialogues.py <case> --n 5000` produces training data.
8. **Train** — `make ac-s04-train-dialogue` retrains.
9. **Playtest** — open in pygame; verify scenes read as intended.

Total per-case authoring: ~5 days. Architecture/training infra is not re-implemented per case.

---

## Versioning

This schema is **v1**. Future versions may:

- Add new universal dimensions (e.g. `TIME_OF_DAY` if needed for chronology).
- Tighten constraint grammar.
- Add new edge types.

Cases declare their schema version in `dimensions.json`. The loader supports backward compatibility: a v1 case loaded by a v2 engine works; a v2 case loaded by a v1 engine errors gracefully.

---

## Non-goals (explicit)

- **No prose generation by the model.** Phrases are fully authored. The model only chooses token IDs.
- **No LLM at any layer.** Slot-fill is deterministic.
- **No per-case neural prose model.** Phrases are flat strings.
- **No abandonment of overfitting per case.** Each case has its own trained `StructuredSceneTransformer`.
- **No rewrite of convergence math.** Lyapunov / attractor mechanics unchanged.

---

## Reference: amber_cipher dimension table

| Dim | Vocab size | Notes |
|---|---|---|
| LOCATION | 10 | All 8 case locations + 2 transitional gates |
| TRANSITION | 8 | Universal |
| CAUSE | 12 | 8 universal + 4 case-specific |
| PRESENCE | 13 | 7 suspects + 5 witnesses + alone |
| STANCE | 6 | Universal |
| ACTION | 12 | 10 universal + 2 case-specific |
| OBJECT_FOCUS | 14 | All case objects + none |
| TELL | 8 | Universal |
| ATMOSPHERE | 6 | 4 universal + 2 case-specific (fog, train) |
| REVELATION | 8 | 7 universal + 1 case-specific |
| BEAT | 4 | Universal |

**Total tokens for amber_cipher: ~89.** Up from 72 in legacy schema. About 25 tokens are new (transitions, causes, beats, dim-specific atmospheres, etc.); the rest are existing tokens tagged with their dim.

Phrase authoring volume: 89 tokens × 2 langs × 1 phrase = 178 short strings. Plus ~20 constraint rules. Plus ~30 cross-dim graph edges. ~5 days of focused authoring.

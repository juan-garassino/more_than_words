---
name: lt-art
description: Generate diffusion-ready prompts for the pixel art and cinematic assets a Living Tales case needs — per-location backdrops, suspect/witness portraits, briefing establishing shot, verdict ending cards, and case-specific UI ornaments. Reads the case foundation files (dimensions.json, briefing, VOICES.md) and the existing IMAGES.md catalogue, then produces a structured art_prompts.md ready to paste into SDXL / Flux / Midjourney. Use when the user asks to "generate the art prompts", "create the diffusion prompts", "prompt the case art", "make the pixel art prompt sheet", "generate prompts for case X", "write prompts for the locations / portraits / endings", or any work involving the visual asset commission for a Living Tales case. Designed for the project's Victorian-pixel-art aesthetic; baked-in style spine ensures consistency across all assets in a case.
---

# Living Tales Art Prompt Generator

This skill packages the workflow for producing diffusion-ready prompts for the visual assets a Living Tales case needs. It does not generate images — it generates the prompts the user (or their diffusion service) feeds to SDXL / Flux / Midjourney to produce them.

The skill respects the project's aesthetic invariants (Victorian-pixel-art, gas-lamp lighting, deep ink-blue night, painterly-to-pixelated treatment) and bakes them into every prompt as a "style spine" so all assets in a case look like they belong to the same world.

The output is a single markdown file per case at `cases/<case_id>/art_prompts.md` — copy-pasteable, organised by asset category, ready to drop into any diffusion tool's prompt field.

## When to use it

| Mode | Trigger phrases | What it does |
|---|---|---|
| **case** | *"generate art prompts for X"*, *"prompt the case art"*, *"make the prompt sheet"* | Reads case data, generates prompts for all per-case assets: location backdrops, NPC portraits, briefing card, verdict cards. Output: `cases/<case_id>/art_prompts.md`. |
| **ui** | *"generate UI ornament prompts"*, *"prompt the universal art"* | Generates prompts for shared UI assets (parchment textures, class icons, dividers, frames, vignette, title screen) — one-time, not per-case. Output: `living_tales/art_prompts_ui.md`. |
| **inventory** | *"what art is missing for X"*, *"art inventory for X"* | Compares existing files in `art/<case_id>/` to what the case needs (per dimensions.json + briefing). Reports missing files. |
| **style-pack** | *"create a style pack for X"*, *"generate the style guide"* | Generates a consolidated 1-page style guide for the case — palette swatches, lighting reference, era/mood description. For sharing with a diffusion artist or pinning at the top of every prompt. |
| **refine** | *"refine the prompt for X"*, *"the cufflink art looks wrong"* | Given a specific asset the user is unhappy with, generate 3-5 prompt variants targeting the issue (lighting, composition, palette, era detail). |

## Architectural assumptions

Read these once before invoking:

- The case has `living_tales/trainer/cases/<case_id>/dimensions.json` (drives the LOCATION list).
- The case has `living_tales/trainer/cases/<case_id>.json` with a `briefing` block listing suspects.
- The case has `living_tales/trainer/cases/<case_id>/VOICES.md` (used for portrait prompts — the voice description informs visual character).
- Existing art lives at `art/<case_id>/direct_pixel_art_v1/pixel_320x320/` (and 240×240).
- Universal UI ornament prompts have a separate output (`living_tales/art_prompts_ui.md`) since they are case-agnostic.
- The skill reads `living_tales/IMAGES.md` to understand the asset catalogue conventions (filenames, dimensions, priority tiers).

If any of these is missing, the skill halts and reports what's needed.

## The style spine

Every per-case prompt is prefixed with a "style spine" that locks the visual aesthetic. Example for amber_cipher (1879 Yorkshire railway):

```
STYLE SPINE
Pixel art, 16-bit era, 320×320 native resolution.
1879 Yorkshire railway station, late November, 11pm-to-dawn.
Deep ink-blue night sky (#0E1420 → #1A1F2C), amber gas-lamp pools (#C49640 → #B08A3F), wet cobblestones, fog drifting at lamp-height.
Painterly source rendered down to crisp pixel art with limited 16-color palette.
Camera: medium-wide, slightly low angle, suggesting the detective's POV.
Mood: methodical, cold, observational — Obra Dinn meets Kentucky Route Zero.
NEGATIVE: text, watermarks, modern signage, color photographs, anime, anti-aliased smooth edges, overly bright, neon.
```

The style spine is generated from the case's setting (extracted from briefing + VOICES.md). For a hospital case (`attended_hour`) the spine swaps to clinical fluorescent + sterile palette. For a Victorian seance case the spine swaps to candle-lit interiors + heavy velvet.

The spine is the first thing in `art_prompts.md` and gets referenced by every individual prompt.

## Asset categories (case mode)

The skill generates prompts for these asset categories per case:

### 1. Location backdrops (highest priority — the gameplay surface)

For each `LOCATION` token in `dimensions.json`:
- One prompt at 320×320 (primary) and one at 240×240 (compact mode).
- Filename convention: `<case_id>_<location_id>_pixel_320x320_v01.png`.
- Style spine + scene-specific subject + camera + mood.

Example (amber_cipher, `location:goods_shed`):

```
**Filename:** amber_cipher_goods_shed_pixel_320x320_v01.png
**Dimensions:** 320×320 pixel art

**Prompt:**
[STYLE SPINE]

Subject: interior of a Victorian railway goods shed at night. Coal dust on the wooden floor. Stacked wooden crates marked with railway company stencils, partly tarped. Iron coal scuttle in the foreground. A single oil lantern hanging from a beam, casting a small amber pool against the deep blue surrounding shadows. The wide sliding door is half open showing fog at the platform beyond. No people in frame.

Composition: medium-wide, three-quarter perspective looking diagonally across the shed interior. Focal point: the lantern's pool of light catching the edge of a crate.

Mood: oppressive silence, industrial cold, evidence-laden.

Render at 4× target resolution (1280×1280) then nearest-neighbor downscale to 320×320 to preserve crisp pixel edges.
```

### 2. Suspect & witness portraits

For each `PRESENCE` NPC (suspects + witnesses) in `dimensions.json`:
- One portrait at 96×96 (small overlay) and one at 192×192 (case-file recall view).
- Filename: `art/ui/portraits/<case_id>/<role>_<id>.png` (per IMAGES.md P1.1).
- The prompt incorporates VOICES.md description.

Example (amber_cipher, `suspect:renard_voss`):

```
**Filename:** voss_portrait_96x96.png and voss_portrait_192x192.png
**Dimensions:** 96×96 (small) + 192×192 (large) pixel art

**Prompt:**
[STYLE SPINE]

Subject: tintype-style portrait of Renard Voss — a Continental man of business in his late forties. Damp wool coat with the collar slightly turned up. Fine but slightly disheveled hair. Bookkeeper's eyes. Carries a leather satchel resting against his knee in the lower frame. Dark brown coat, ivory shirt, narrow black tie.

Pose: head and shoulders, three-quarter angle, gaze directed slightly off-camera (avoiding direct eye contact — speaks to his "closes under pressure" voice).

Lighting: gas-lamp from upper-right, leaving the left side of the face in shadow.

Mood: closed, transactional, the kind of man who answers each question once and then no more.

Background: undefined dark — let the figure carry the frame.
```

### 3. Briefing card (cinematic establishing shot)

The pre-game briefing screen — one painterly oil-painting-style image (NOT pixel art) at 480×480 to set the cinematic tone before pixel-art gameplay starts.

Filename: `art/ui/cinematics/briefing_<case_id>.png`. Per IMAGES.md P2.1.

Prompt structure: subject is the case's opening location, but rendered painterly (oil paint) rather than pixel.

### 4. Verdict ending cards (5 per case)

Five painterly cards keyed to the ending types — `all_strong`, `lucky_guess`, `wrong_accusation`, `cold_case`, `partial`. Each is the dimmed visual context the player sees on the ending screen.

Filenames: `art/ui/cinematics/verdict_<TYPE>.png` (universal, not case-specific) OR `art/ui/cinematics/<case_id>_verdict_<TYPE>.png` (case-specific). Default: case-specific so each case's ending feels grounded.

### 5. Case-specific objects (optional)

For "hero" objects (the cufflink, cipher_sheet, ledger_book in amber_cipher), generate close-up pixel art prompts at 64×64 used as inline icons in the journal "Evidence" section.

Filename: `art/ui/objects/<case_id>/<object_id>.png`.

## Asset categories (ui mode)

For the universal UI ornaments per IMAGES.md:

| Asset | Path | Dimensions | Notes |
|---|---|---|---|
| Parchment dark tile | `art/ui/textures/parchment_dark_tile.png` | 256×256 tileable | Tiled background |
| Class icons | `art/ui/icons/class_<NAME>.png` | 32×32 | 11 icons (suspect, witness, object, etc.) |
| Decorative dividers | `art/ui/ornaments/divider_*.png` | 480×16 | 3 styles |
| Card corners | `art/ui/ornaments/corner_*.png` | 16×16 | 4 corners |
| Title-strip emblem | `art/ui/ornaments/emblem.png` | 24×24 | Wax-seal magnifier |
| Vignette overlay | `art/ui/textures/vignette.png` | 480×800 | Soft black corners |
| Title screen | `art/ui/cinematics/title_screen.png` | 480×800 | Detective's desk painterly |
| Case thumbnails | `art/ui/thumbnails/<case_id>.png` | 64×64 | Per-case emblem |

Each gets a prompt grounded in the same style language but case-agnostic (palette + era only, no case-specific imagery).

## Output format (art_prompts.md)

The skill writes a single markdown file per case. Structure:

```markdown
# Art Prompts: <Case Title>

> Generated by lt-art skill on <date>. Drop each prompt into your diffusion tool of choice (SDXL, Flux, Midjourney). The style spine at the top is canonical for this case — every individual prompt assumes it.

## Style Spine

[the locked style block]

## Palette

| Hex | Name | Usage |
|---|---|---|
| #0E1420 | ink_night | Deepest shadows, sky |
| #1A1F2C | charcoal | Mid shadows |
| #C49640 | gas_amber | Lamp pools, accents |
| #B08A3F | brass | Hardware, fixtures |
| #E8DCC0 | cream | Highlights, paper |
| #2A1F12 | sepia | Text, period documents |
| ... | ... | ... |

## 1. Location Backdrops

### location:thornfield_crossing
[prompt block]

### location:platform_two
[prompt block]

...

## 2. NPC Portraits

### suspect:renard_voss
[prompt block]

### suspect:stationmaster
[prompt block]

...

### witness:ticket_clerk
[prompt block]

...

## 3. Briefing Card
[prompt block]

## 4. Verdict Cards

### verdict_all_strong (justice)
[prompt block]

### verdict_lucky_guess
[prompt block]

### verdict_wrong_accusation
[prompt block]

### verdict_cold_case
[prompt block]

### verdict_partial
[prompt block]

## 5. Hero Objects (optional)

### object:initialed_cufflink
[prompt block]

...

## Generation Notes

- Render at 4× the target resolution then nearest-neighbor downscale to preserve crisp pixel edges.
- Use a fixed seed per asset family (locations: 1000-1099, NPCs: 2000-2099, etc.) so re-generations are reproducible.
- Capture the seed in `art/<case_id>/PROVENANCE.txt` for each accepted version.
- For SDXL: use a pixel-art LoRA (e.g. PixelArtSDXL) if available.
- For Flux: native text adherence is good enough without LoRA.
- For Midjourney: append `--niji` for stylized pixel-art mode + `--ar 1:1` for square assets.
```

## Workflow internals

### Case mode (the most common)

1. **Read inputs:**
   - `cases/<case_id>/dimensions.json` → list of LOCATION + PRESENCE tokens
   - `cases/<case_id>.json` → briefing.suspects (with descriptions) + briefing.crime + briefing.setting
   - `cases/<case_id>/VOICES.md` → per-NPC voice descriptions (used in portrait prompts)
   - `cases/<case_id>/phrases.json` → object descriptions
   - `living_tales/IMAGES.md` → catalogue conventions

2. **Derive style spine** from briefing.setting + briefing.crime. Consult these for era/lighting/mood. Lock the palette per case.

3. **Generate prompts in parallel via 3 subagents** (this is where the skill leverages parallelism):

   | Agent | Bucket | Output sections |
   |---|---|---|
   | **P1** | Location backdrops | Section 1 of art_prompts.md |
   | **P2** | NPC portraits | Section 2 |
   | **P3** | Cinematics + objects | Sections 3, 4, 5 |

   Each subagent gets the style spine, the palette, the relevant case data, and the prompt template structure. They produce their section of the markdown.

4. **Synthesize** — concatenate sections, prepend style spine + palette, write to `cases/<case_id>/art_prompts.md`.

5. **Inventory check** — compare prompts generated vs files already in `art/<case_id>/`. Mark each prompt with `EXISTS / MISSING` so the user knows what to commission first.

### UI mode

1. Read `living_tales/IMAGES.md` for the universal asset list.
2. Single agent generates all UI prompts (no per-case data needed).
3. Output: `living_tales/art_prompts_ui.md`.

### Inventory mode

1. Walk `art/<case_id>/` for existing files.
2. Compare to expected per dimensions.json (locations) + briefing.suspects (portraits) + ending types (verdicts).
3. Report:
   ```
   Case: amber_cipher
   Locations: 8 expected, 7 present (signal_box missing)
   NPC portraits: 12 expected, 0 present
   Verdict cards: 5 expected, 0 present
   Briefing card: 1 expected, 0 present
   Total missing: 19 assets
   ```
4. Suggest invocation: `/lt-art case amber_cipher` to generate the missing prompts.

### Refine mode

When the user has a specific asset that came back wrong:

1. Read the existing prompt from art_prompts.md.
2. Read the user's complaint (e.g. "the cufflink looks plastic").
3. Generate 3-5 variant prompts adjusting the issue:
   - More specific material description ("solid silver, oxidized in the recesses")
   - Different lighting angle ("backlit by the lantern, casting long shadows")
   - Closer composition crop
   - Different period reference ("Victorian late-1870s, not Edwardian")
4. Output the variants in a small markdown patch the user appends to art_prompts.md.

### Style-pack mode

A consolidated single-page style guide for the case. Useful when:
- Sharing with a contracted diffusion artist
- Pinning at the top of every prompt for consistency
- Giving a Midjourney user a `--ref` URL set

Output: `cases/<case_id>/STYLE_PACK.md` with:
- Era/setting (1-paragraph description)
- Palette swatches (hex codes + named usage)
- Lighting reference (reference paintings/films/games)
- Mood reference (3-5 adjectives + a music cue)
- Composition rules (camera angle conventions)
- Negative prompt block

## Constraints respected

- The skill does NOT generate images. It only writes prompt text.
- The skill does NOT modify trained models or the engine.
- The skill does NOT alter case data — it only reads it.
- The skill does NOT replace existing prompts unless explicitly invoked in `refine` mode.

## Output is human-readable

The output is markdown. The user (or their commissioned artist) reads it with no Claude assistance. Each prompt is a complete, self-contained block they can paste into any diffusion tool.

## Recommended call patterns

**First time on a case (full prompt sheet):**
```
/lt-art case amber_cipher
```
Output: `cases/amber_cipher/art_prompts.md` with all per-case prompts.

**Universal UI ornaments (one-time, project-wide):**
```
/lt-art ui
```
Output: `living_tales/art_prompts_ui.md`.

**Audit existing art:**
```
/lt-art inventory amber_cipher
```
Output: stdout report of present vs missing.

**Refine a specific asset:**
```
/lt-art refine amber_cipher --asset object:initialed_cufflink --issue "looks plastic, want more period-correct silver"
```
Output: 3-5 prompt variants appended to art_prompts.md.

**Generate a style pack to share:**
```
/lt-art style-pack amber_cipher
```
Output: `cases/amber_cipher/STYLE_PACK.md`.

## Output summary format

After every invocation, the skill prints:

```
Mode: <mode>
Case: <case_id>
Style spine: <one-line summary>
Prompts generated: <N>
Output file: <path>
Inventory:
  Locations: <generated>/<expected>
  Portraits: <generated>/<expected>
  Cinematics: <generated>/<expected>
Already in art/: <existing count>
Missing (to commission first): <list of top 5>
Next step: paste the prompts into <recommended diffusion tool> at <recommended resolution>
```

## What this skill does NOT do

- Generate images (only prompts).
- Decide which diffusion tool to use (it produces tool-agnostic prompts).
- Write SCHEMA / dimensions / case data — those are foundation files authored elsewhere.
- Run any pipeline beyond reading case data and writing markdown files.
- Modify trained models, the engine, the trajectory dataset, or anything else.

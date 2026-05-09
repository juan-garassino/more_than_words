# Living Tales — Visual Asset Catalogue

This document inventories the diffusion-generated images we want to commission
to push the pygame UI from "atmospheric" to "beautiful". Each asset is scoped
with an exact filename, size, count, style direction, and a starter diffusion
prompt you can adapt for SDXL/Flux/Midjourney.

The ground rule is the same as the engine's: the **picture is sacred**. The
backdrop image is never overlaid. All UI ornaments must compose around it,
not on it. Every asset listed here is meant to live in the chrome of the
window — the title strip, the location subtitle, the narration panel, the
evidence cards, the briefing/ending screens — never on the active scene.

Existing assets are listed in the "Already have" sections so you don't
re-commission them.

Folder convention:

```
art/
├── <case_id>/                      # per-case scene art (existing)
│   ├── direct_pixel_art_v1/
│   │   ├── pixel_320x320/          # primary scene backdrops (current default)
│   │   ├── pixel_240x240/          # smaller-screen alternative
│   │   └── masters_square_320/     # painterly hi-res sources for cinematics
│   └── scene_backdrops_v2/
└── ui/                             # NEW — shared UI ornaments (this doc's scope)
    ├── frames/
    ├── ornaments/
    ├── icons/
    ├── textures/
    ├── portraits/
    └── cinematics/
```

## Priorities

- **P0** — Massive aesthetic lift, small asset count. Ship first.
- **P1** — Per-case visual identity (portraits, missing scenes).
- **P2** — Atmospheric cinematics (briefings, endings, transitions).
- **P3** — Animated effects (particle sprites, scanlines, rain).

---

## P0 — UI ornaments (ship first)

These transform the chrome immediately. All transparent PNG, pixel art style,
limited palette: warm sepia + brass + ink-blue + cream.

### P0.1 — Parchment texture

| | |
|---|---|
| Path | `art/ui/textures/parchment_dark_tile.png` |
| Size | 256×256, **tileable** (seamless edges) |
| Count | 1 |
| Use | Background fill for narration panel and evidence cards. |

Style: dark warm parchment, subtle grain and water-stains, tobacco-stained
edges. Must read as "old paper" but stay dark enough that cream text pops.
Avoid heavy text/sigils — it tiles, so any feature repeats.

> **Prompt seed:** *"seamless tileable texture, aged parchment, deep tobacco
> brown, faint ink stains, subtle paper grain, gas-lamp lit, no text, no
> drawings, pixel art, 16-bit era, 256x256, edges seamless"*

### P0.2 — Class icons

| | |
|---|---|
| Path | `art/ui/icons/class_<NAME>.png` |
| Size | 32×32, transparent background |
| Count | 11 (one per token class) |
| Filenames | `class_suspect.png`, `class_witness.png`, `class_object.png`, `class_motive.png`, `class_event.png`, `class_location.png`, `class_action.png`, `class_emotion.png`, `class_modifier.png`, `class_time.png`, `class_accomplice.png` |

Each is a single-color (cream `#E8DCC0` over transparent) ink illustration
in the style of a Victorian field-notebook glyph. They replace the current
Unicode symbols on evidence cards.

Suggested glyphs:
- **suspect** — silhouetted bust in profile
- **witness** — single eye, deco frame
- **object** — magnifying glass over a tag
- **motive** — heart pierced by a coin
- **event** — pocket-watch face
- **location** — compass rose / signpost
- **action** — running boot or hand
- **emotion** — theatrical mask
- **modifier** — small wax seal
- **time** — clock pendulum
- **accomplice** — two silhouettes overlapping

> **Prompt seed:** *"32x32 pixel art icon, single cream color on transparent,
> Victorian engraving style, [GLYPH], minimalist line work, period-appropriate,
> clean silhouette, no shading, sharp 1-px line work"*

### P0.3 — Decorative dividers

| | |
|---|---|
| Path | `art/ui/ornaments/divider_<NAME>.png` |
| Size | 480×16, transparent background |
| Count | 3 (`divider_thin.png`, `divider_fleuron.png`, `divider_double.png`) |

Horizontal rule replacements for the location subtitle bar and panel breaks.
`fleuron` has a centered ornamental flourish (Art Nouveau leaf or wax seal),
the others are stylized rules. Brass/gold tone (`#B08A3F`).

> **Prompt seed:** *"horizontal divider ornament, brass color on transparent
> background, Art Nouveau, single fleuron centered, 480x16 pixel art, 16-bit
> era, fine engraving, period decorative element, no text"*

### P0.4 — Card corner ornaments

| | |
|---|---|
| Path | `art/ui/ornaments/corner_tl.png` (+ tr, bl, br) |
| Size | 16×16 each, transparent |
| Count | 4 (one per corner — they mirror) |

Replace the current 2×2 gold pixel ticks at evidence-card corners with
proper ornamental brackets — small Art Nouveau flourishes that frame the card.
Brass `#B08A3F`.

> **Prompt seed:** *"16x16 pixel art corner ornament, brass on transparent,
> Art Nouveau frame bracket, single corner only, top-left orientation,
> minimalist Victorian filigree, sharp pixel edges"*

### P0.5 — Title strip emblem

| | |
|---|---|
| Path | `art/ui/ornaments/emblem.png` |
| Size | 24×24, transparent |
| Count | 1 |

A small wax-seal / monogram-style emblem to sit beside the case title in the
top strip. Reads as the imprint of the detective agency. Deep crimson over
brass.

> **Prompt seed:** *"24x24 pixel art wax seal, deep red wax with brass
> impression of a magnifying glass over an eye, Victorian detective agency
> sigil, transparent background, single object centered, no shadow"*

### P0.6 — Vignette overlay

| | |
|---|---|
| Path | `art/ui/textures/vignette.png` |
| Size | 480×800, partially transparent |
| Count | 1 |

A full-window overlay with a soft black vignette in the corners and slight
chromatic aberration that gives the entire window a "lantern-lit" feel.
Centre is fully transparent, corners fade to black at ~30% alpha.

> **Prompt seed:** *"480x800 vignette overlay, transparent center fading to
> 30% black at corners, slight blue-cyan chromatic aberration at edges, gas
> lamp atmosphere, pure overlay layer, no detail in center"*

### P0.7 — Title-screen background

| | |
|---|---|
| Path | `art/ui/cinematics/title_screen.png` |
| Size | 480×800 |
| Count | 1 |

The image behind the case-select menu. A painterly Victorian detective's
desk: pocket watch, magnifying glass, cipher pages, tobacco pipe, brass
oil lamp. Warm, dim, filling the entire portrait frame. Used as backdrop
when no case is loaded.

> **Prompt seed:** *"painterly Victorian detective's desk overhead view,
> pocket watch, magnifying glass over cipher page, tobacco pipe, brass oil
> lamp, scattered evidence photographs, leather-bound notebook, deep amber
> chiaroscuro lighting, painterly oil-paint look, 480x800 portrait, no text"*

### P0.8 — Case-select thumbnails

| | |
|---|---|
| Path | `art/ui/thumbnails/<case_id>.png` |
| Size | 64×64 |
| Count | 20 (one per case) |

Small icon for each case in the case-select list. A miniature crest or
emblem keyed to the case — e.g. *amber_cipher* = stopped clock, *attended_hour*
= a heart-rate trace, *orchard_at_dusk* = boot prints. Distinguishable at
a glance even when the model is not yet trained (then rendered desaturated).

> **Prompt seed:** *"64x64 pixel art emblem, [CASE-SPECIFIC OBJECT], on
> transparent background, brass + cream colors, single iconic object centered,
> period engraving style, no text, no border"*

---

## P1 — Per-case identity

### P1.1 — Suspect & witness portraits

| | |
|---|---|
| Path | `art/ui/portraits/<case_id>/<role>_<name>.png` |
| Size | 96×96 |
| Count | ~12 per case (7 suspects + 5 witnesses for amber_cipher) |

Square pixel-art portraits framed like a tintype photograph. Used as a small
overlay beside the clue panel when the model emits a SUSPECT or WITNESS
token. Should match the case's pixel-art style (existing scene backdrops
in `art/<case_id>/direct_pixel_art_v1/`).

For amber_cipher, the IDs are:
- suspects: `stationmaster, railway_clerk, estranged_daughter, renard_voss,
  night_porter, platform_guard, travelling_broker`
- witnesses: `ticket_clerk, porter, signalman, carriage_cleaner,
  telegraph_operator`

> **Prompt seed:** *"96x96 pixel art portrait, [CHARACTER DESCRIPTION FROM
> CASE BRIEFING], tintype photograph framing, sepia tones, period costume
> 1890s railway station, dim gas-lamp lighting, 16-bit era, head and
> shoulders, looking slightly off camera, no text"*

The 8 generic UUID-named PNGs already in `living_tales_sprites/final_set/`
need a manual mapping to suspect/witness IDs before they can be used. Until
that's done, treat this as net-new commissions per case.

### P1.2 — Missing scene backdrops

13 cases currently ship with only **1 backdrop** — usually a placeholder/concept
piece. Each needs N backdrops where N = number of LOCATION tokens in that case.

| Case | Locations needed | Existing backdrops |
|---|---|---|
| amber_silence | TBD | 1 (concept only) |
| burning_glass | TBD | 1 |
| covenant_garden | TBD | 1 |
| dead_calm | TBD | 1 |
| endgame | TBD | 1 |
| glass_cartographer | TBD | 1 |
| instrument_landing | TBD | 1 |
| iron_cartridge | TBD | 1 |
| monsoon_ledger | TBD | 1 |
| mountain_exchange | TBD | 1 |
| observatory_clock | TBD | 1 |
| signal_fire | TBD | 1 |
| thirteenth_tide | TBD | 1 |

For each missing case, run `living_tales/trainer/cases/<case_id>/spec.json`
to extract the LOCATION tokens, then commission one 320×320 PNG per token at
`art/<case_id>/direct_pixel_art_v1/pixel_320x320/<case_id>_<location_id>_pixel_320x320_v01.png`.

Style must match the case's existing concept piece (palette, era, mood).

### P1.3 — amber_cipher — `signal_box`

The only LOCATION in a "complete" case that has no art (we currently proxy
to `station_office_doorway`). One 320×320 PNG named
`amber_cipher_signal_box_pixel_320x320_v01.png` would close the gap.

> **Prompt seed:** *"signal box interior at night, brass levers, oil lamp,
> rain-streaked window, Victorian railway, pixel art 320x320 16-bit era,
> matches Living Tales amber_cipher palette: deep blue night, amber lamps,
> wet cobblestones outside"*

---

## P2 — Cinematics

### P2.1 — Briefing card

| | |
|---|---|
| Path | `art/ui/cinematics/briefing_<case_id>.png` |
| Size | 480×480 |
| Count | 20 |

A wide painterly establishing shot for the briefing screen — same scene as
the opening location, but rendered painterly (oil-paint look) rather than
pixel-art. Sets a cinematic tone before the pixel-art game proper starts.

For amber_cipher this would be Thornfield Crossing rendered as an oil
painting at twilight, foggy, atmospheric.

> **Prompt seed:** *"painterly oil painting, [CASE'S OPENING LOCATION],
> dramatic chiaroscuro, narrative cinematic, 480x480, atmospheric,
> period-appropriate, no text overlay"*

### P2.2 — Verdict cards

| | |
|---|---|
| Path | `art/ui/cinematics/verdict_<TYPE>.png` |
| Size | 480×480 |
| Count | 5 (one per ending type) |

Painterly cards for each ending category, used as the ending-screen
backdrop:
- `verdict_all_strong.png` — courtroom, gavel, light streaming through
- `verdict_lucky_guess.png` — accusation, half-shadowed face, "is it him?"
- `verdict_wrong_accusation.png` — empty prison cell, key on floor
- `verdict_cold_case.png` — fog, abandoned street, file folder closing
- `verdict_partial.png` — newspaper clipping, words half-covered

These replace the per-case ending backdrop currently keyed by `_ending` in
`scene_map.json`.

> **Prompt seed:** *"[VERDICT THEME], painterly Victorian noir, dramatic
> lighting, no text, 480x480, oil painting, atmospheric"*

### P2.3 — Accusation screen

| | |
|---|---|
| Path | `art/ui/cinematics/accusation_board.png` |
| Size | 480×640 |
| Count | 1 |

A "detective's evidence board" backdrop — corkboard with photos, red string,
pinned notes — used during the accusation modal. Suspects' portraits will
be overlaid programmatically.

> **Prompt seed:** *"detective's evidence board, corkboard with red string
> connecting photographs, pinned notes, magnifying glass, dim gas lamp,
> Victorian noir, painterly, 480x640 portrait, no text on notes"*

---

## P3 — Atmospheric effects (lowest priority)

### P3.1 — Particle sprite sheets

| | |
|---|---|
| Path | `art/ui/effects/<TYPE>_sheet.png` |
| Size | 256×256, 4×4 grid of 64×64 frames |
| Count | 4 (`rain`, `fog`, `dust_motes`, `embers`) |

For optional weather/atmosphere overlays drawn on top of the backdrop
(NOT on the picture — we'd composite them as faint translucent layers
inside the backdrop area only, never over text).

### P3.2 — CRT scanline overlay

| | |
|---|---|
| Path | `art/ui/textures/scanlines.png` |
| Size | 480×480, alpha at ~10% |
| Count | 1 |

Subtle horizontal scanlines to give the backdrop a "memory recalled" feel.
Toggle via a `--crt` flag.

---

## Implementation hooks

When a P0 asset lands, the corresponding hook in `pygame_play.py` is:

| Asset | Wire-up location |
|---|---|
| Parchment texture | `Renderer._draw_narration` — replace `pygame.draw.rect(... PARCHMENT ...)` with `screen.blit(self._parchment_tile, …)` looped to fill `NARRATION_RECT`. Same in `_draw_hand` for card backgrounds. |
| Class icons | `Renderer._draw_hand` — replace the `CLASS` text with `screen.blit(self._class_icons[cls], …)`. |
| Dividers | `Renderer._draw_location_bar` — replace the two `pygame.draw.line` calls with `screen.blit(self._divider, …)`. |
| Card corners | `Renderer._draw_hand` — replace the 4 `pygame.draw.rect` corner ticks with `screen.blit(self._corner_tl, …)` etc. |
| Vignette | `Renderer._draw_frame` — final blit after everything else. |
| Title-screen bg | New `case_select_screen()` — `screen.blit(self._title_bg, (0, 0))` before drawing the menu. |
| Thumbnails | `case_select_screen()` — small thumbnail next to each case row. |
| Suspect portraits | `Renderer._draw_narration` — when last clue is SUSPECT/WITNESS, blit a 96×96 portrait in the panel's right gutter. |

Each P0 asset should be loaded once in `Renderer.__init__` with graceful
fallback to the current programmatic rendering when the file is absent —
that way the catalogue can be filled incrementally without breaking the
build.

## Generation logistics

- **Resolution:** generate at 4× the target, then nearest-neighbor downscale
  to keep crisp pixel edges. SDXL/Flux at native pixel-art resolutions
  produces softer results than scale-down.
- **Palette lock:** export the project palette (the `INK / GOLD / BRASS /
  CREAM / PARCHMENT / SEPIA` constants in `pygame_play.py:54-78`) as a
  16-color palette and run an indexed-color pass after generation. This
  enforces visual consistency across cases.
- **Naming:** stick to `lowercase_with_underscores.png`. Generate v01, v02,
  v03 candidates per asset; pick the best, drop into the canonical name.
- **License:** if using a hosted diffusion service, capture each prompt and
  seed in `art/ui/<asset>/PROVENANCE.txt` so we can re-roll deterministically.

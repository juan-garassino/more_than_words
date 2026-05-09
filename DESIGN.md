# Living Tales Pixel-Art Style Guide

## Purpose

This document locks the visual language for Living Tales background generation so future image work can recreate the same style even before adding case-specific reference pictures.

Use this together with:

- the case visual brief
- the case prompt file
- the selected reference images from approved generations

If a reference image conflicts with this document, follow this document.

## Core Art Direction

The target is not painted concept art.

The target is:

- direct native pixel-art game background
- square-native or gameplay-native composition from the start
- then terminal-based reduction into final pixel outputs
- cozy, readable, explorable adventure-game spaces
- human-eye viewpoint, not elevated overview
- warm protected light pockets inside colder or harsher worlds
- chunky miniature-diorama readability
- selective low-fidelity ambiguity instead of dense rendered detail

Closest shorthand:

- cozy handheld-JRPG warmth
- in the spirit of `Golden Sun`
- with point-and-click adventure readability
- but always as a background, never as character portrait art

## Camera And Composition

Always prefer:

- human-height eye level
- a standing person’s viewpoint
- asymmetrical compositions
- clear route readability
- clear foreground, midground, background separation
- one obvious safe or inviting pocket in the scene

Avoid:

- elevated bird’s-eye views
- tactical overview angles
- centered poster framing
- postcard symmetry
- giant hero staging
- empty establishing plates with no route logic

## Mood And Readability

Every scene should feel like a small playable world.

The desired effect is:

- a harsh, mysterious, or pressured environment
- with one pocket of warmth, shelter, or invitation
- so the player feels they could step into the scene and stand there safely for a moment

The scene should feel:

- cozy
- explorable
- slightly miniature
- emotionally inviting
- readable at small size

## Pixel-Art Handling

The image model will generate at high resolution.

We still want the image to feel low-resolution in spirit before reduction. That means prompts should bias toward:

- large clustered shapes
- simplified prop groupings
- strong silhouette blocks
- restrained texture detail
- readable light masses
- no fine realism

Workflow:

1. Generate a high-resolution source with the built-in `imagegen` skill.
2. Author separate source art for `320` and `240` when comparing density.
3. Reduce with the terminal pixel-resizing command.

Do not treat one source resized two ways as two different art directions.

## Protagonist Lock

Every approved scene should include the recurring main character.

The protagonist must be:

- small in frame
- readable by silhouette
- treated like a playable adventure-game avatar
- integrated into the route or warm-light pocket
- never portrait-framed
- never the dominant visual mass

### Hard Rule: Not A Child

The protagonist is **not** a kid.

Never let the silhouette drift into:

- child proportions
- oversized head with short body
- toy-like toddler stance
- schoolchild energy
- cute chibi proportions
- tiny child next to adult NPCs

The protagonist should read as:

- an adult
- slim or average adult build
- compact because of scale, not because of age
- calm, observant, mobile, self-possessed

### Silhouette Rules

Keep the protagonist consistent across cases:

- adult-height proportions relative to the world
- head not oversized
- torso and leg length balanced like an adult
- stable travel silhouette
- coat, cloak, jacket, or practical outerwear shape is preferred
- hat, hair, collar, satchel, or coat hem can be used as the repeatable identifier

Expression should come from:

- stance
- direction of movement
- pause-at-threshold behavior

Not from:

- facial detail
- close-up gestures
- exaggerated cartoon posing

## Supporting Characters

Background figures should feel like idle NPCs.

Use:

- tiny background people
- restrained body language
- waiting, watching, crossing, working, or lingering poses

Avoid:

- action tableau posing
- cinematic fight motion
- melodramatic gestures
- crowd scenes that overpower the environment

## Density Split For `320` Vs `240`

When creating both variants, the art should change.

### `320`-authored pass

Use:

- slightly richer environmental grouping
- more secondary props
- fuller light clustering
- more medium-detail architectural rhythm

Still avoid dense realism.

### `240`-authored pass

Use:

- fewer props
- chunkier masses
- simpler route read
- larger silhouette blocks
- more aggressive simplification

The `240` version should feel more iconic, not merely smaller.

## Prompt Language To Keep

Useful shared phrases:

- `pixel-art game background`
- `direct native pixel art`
- `readable at mobile game scale`
- `human-height eye-level view`
- `warm protected light pocket inside a colder world`
- `miniature diorama readability`
- `small recurring protagonist with stable adult silhouette`
- `small caricatured background people only`
- `not a painted image converted later`

## Prompt Language To Avoid

Do not use wording that pushes the model toward the wrong medium or wrong staging:

- `concept art`
- `cinematic realism`
- `painted keyframe`
- `matte painting`
- `hero shot`
- `epic wide aerial`
- `top-down`
- `child protagonist`
- `cute kid adventurer`
- `chibi hero`

## Negative Drift Checklist

Reject or regenerate if the scene drifts into any of these:

- elevated perspective
- too much realism
- painterly look
- muddy unreadable lighting
- empty scene with no route
- centered poster composition
- protagonist missing
- protagonist reading as a child
- protagonist too large
- text or readable signage
- close-up faces
- too many figures
- loss of the cozy warm-pocket effect

## Final Review Questions

Before accepting a scene, check:

1. Does it read as direct pixel-art game background art rather than painted concept art?
2. Is the camera at human eye height?
3. Is there a warm or safe pocket inside the larger environment?
4. Does the world feel miniature, explorable, and readable?
5. Is the protagonist visible?
6. Does the protagonist read clearly as an adult, not a child?
7. Does the scene remain strong after pixel reduction?

## Default Master Prompt Add-On

Use this block in future prompts unless a case brief overrides it:

```text
Style bias: cozy handheld-JRPG warmth in the spirit of Golden Sun, with inviting light pockets and readable adventurous charm
Composition/framing: square-native or gameplay-native composition, human-height eye-level view, asymmetrical layout, clear route readability, layered depth
Constraints: direct native pixel art, small recurring protagonist with stable adult silhouette, tiny background people only, readable at small size, no readable text or signage
Avoid: painterly concept art, cinematic realism, elevated overview, centered poster framing, child-like protagonist proportions, chibi styling, close-up faces
```

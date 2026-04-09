# Creature Track

The creature cartridges reuse the Living Tales token/graph engine in `oscillating` mode.

## Core loop

The loop is:

`context -> decay/decline -> action/combo -> recovery -> context`

Unlike mystery cases, creature sessions do not aim for a final solved state. Dimensions begin at a midpoint and move up or down within fixed bounds.

## Canonical slice

`little_creature_M` is the current balancing target.

Its dimensions are:

1. `well_fed`
2. `content`
3. `rested`
4. `bonded`
5. `groomed`
6. `healthy`
7. `in_season`

## Stable token roles

Creature tooling groups tokens into these roles:

- `context`
- `decay`
- `decline`
- `combo`
- `action`
- `recovery`
- `state`

The current implementation derives these roles from token ID prefixes so authored content can be validated and reported consistently.

## What gets measured

Creature evaluation is not proof-gated like mystery.

The current balancing metrics are:

- decay-to-recovery closure rate
- combo frequency
- repetition rate
- dead-turn rate
- arc diversity
- per-dimension volatility

## Tooling

```bash
python3 living_tales_case_validator.py cases/little_creature_M.json
python3 -m evals.utils.baseline_runner little_creature_M --n-games 100
python3 living_tales/trainer/tools/report_creature_case.py little_creature_M --runs 100
```

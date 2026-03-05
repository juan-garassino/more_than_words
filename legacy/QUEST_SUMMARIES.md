# More Than Words: Quest Data Summaries

This document provides natural language summaries for the synthetic dataset used to train the "More Than Words" Quest Engine. These stories demonstrate how the **Symbolic Transformer** learns that the same starting conditions can lead to vastly different outcomes based on player choice and sequence history.

## 🌲 The Forest Branch 
**Starting Condition**: `LOC_FOREST` + `NPC_ENEMY` (High Danger)

### Story 1A: The Diplomat
*   **Narrative**: The hero encounters an enemy in the forest but chooses to **PERSUADE** them. The enemy calms down, allowing the hero to **TRADE** for a map. Finally, the hero **READS A TOME** in safety, revealing a secret.
*   **Theme**: Charisma & Knowledge.
*   **Key Tokens**: `PERSUADE`, `SUSPECT_CALM`, `TRADE`, `ITEM_MAP`.

### Story 1B: The Warrior
*   **Narrative**: The hero encounters the same enemy but chooses to **FIGHT**. An ambush triggers! Detailed magic combat ensues (`CAST_SPELL`), but it's too much. The hero is forced to **FLEE** into a dark cave, injured and stressed.
*   **Theme**: Violence & Consequence.
*   **Key Tokens**: `FIGHT`, `EVENT_AMBUSH`, `CAST_SPELL`, `FLEE`, `LOC_CAVE`.

### Story 1C: The Rogue
*   **Narrative**: The hero sees the enemy but chooses to **SNEAK**. Passing unseen in the darkness, they **PICK UP** a key. They travel East to a locked Tower, use the key, and find **TREASURE**.
*   **Theme**: Stealth & Reward.
*   **Key Tokens**: `SNEAK`, `DARKNESS`, `PICK_UP`, `ITEM_KEY`, `USE_ITEM`, `EVENT_TREASURE`.

---

## 🦇 The Cave Branch
**Starting Condition**: `LOC_CAVE` + `DARKNESS` + `MAGIC_AURA`

### Story 2A: The Mage
*   **Narrative**: The hero lights up the cave with **CAST_SPELL**. They find a Cursed Object but handle it safely, reading a Tome to understand its secrets.
*   **Theme**: Magic Mastery.

### Story 2B: The Fool
*   **Narrative**: The hero stumbles forward blindly (`MOVE_N`) into a **TRAP**. They try to fight the trap but get **SICK** and end up waiting in the dark, lethargic and defeated.
*   **Theme**: Blind Action vs. Caution.

---

## 💰 The Market Branch
**Starting Condition**: `LOC_MARKET` + `NPC_MERCHANT`

### Story 3A: The Customer
*   **Narrative**: The hero **TRADES** with the merchant for a Potion. Drinking it restores full energy (`ENERGETIC`, `HEALTHY`). They end the day gossiping with an Ally.
*   **Theme**: Commerce & Socializing.

### Story 3B: The Thief
*   **Narrative**: The hero tries to **SNEAK** and rob the merchant. They get caught (`NPC_ENEMY`), grab the item anyway, and are forced to **FLEE** to a dirty alley, tired and branded a criminal.
*   **Theme**: Crime & Punishment.

---

## 👻 The Mansion Mystery
**Starting Condition**: `LOC_MANSION` + `DARKNESS` (Haunted)

### Story 4A: Ghost Whisperer
*   **Narrative**: The hero **SPEAKS** to the angry ghost. Through **PERSUASION**, the ghost becomes an Ally and opens a secret door to the Lab.
*   **Theme**: Diplomacy.

### Story 4B: Ghost Buster
*   **Narrative**: The hero attacks with **CAST_SPELL** and **FIGHT**. They take damage, have to use a Potion, and the encounter ends in chaos.
*   **Theme**: Combat.

---

## 🕵️‍♀️ The Grand Mystery (Complex Loop)
**Starting Condition**: `LOC_OFFICE`

### Story 5: The Time Loop Detective
*   **Narrative**: Starts in an Office with a Clue. The hero **READS A TOME** which teleports/links them to a Forest Ambush. They **SNEAK** past, **HACK** an artifact, and use it to return to the Office where a body is suddenly revealed (`BODY_FOUND`). The hero immediately **INTERROGATES** a nervous suspect and ends with a dramatic **ACCUSE**.
*   **Theme**: Complex causal chains and "butterfly effect" logic.

---

## 🌌 The Epic Multi-Genre Adventure
**Theme**: Dimension Hopping & Glitch Logic.

### Story 1: Cyber-Start
*   **Narrative**: Starts in `LOC_NEON_CITY` with a `HACKER`. The hero **HACKS** a system causing `SYSTEM_FAILURE` and grabs a Data Chip. Fleeing to a Cyber Cafe, they use a Neural Link which triggers a `MAGIC_AURA`...
*   **Key Tokens**: `HACK`, `SYSTEM_FAILURE`, `LOC_NEON_CITY`.

### Story 2: The Clint Eastwood Isekai
*   **Narrative**: ...The Neural Link glitches (`EVENT_GLITCH`) and transports the user to a `LOC_DESERT` with a `TUMBLEWEED`. They meet an Outlaw, engage in a **DUEL** using a `REVOLVER` in a Ghost Town.
*   **Key Tokens**: `LOC_DESERT`, `EVENT_DUEL`, `ITEM_REVOLVER`.

### Story 3: The Whiskey Wizard
*   **Narrative**: Drinking `WHISKEY` in a Saloon acts as a Potion (`DRINK_POTION`), warping reality again. The hero wakes up in a Fantasy `FOREST`, trades with an Ally, and casts a spell to find an Artifact in a Cave.
*   **Key Tokens**: `DRINK_POTION`, `LOC_SALOON` -> `LOC_FOREST`.

### Story 4: The Glitch Loop
*   **Narrative**: Using the Artifact in the Fantasy Tower reveals it's actually a `ITEM_DATA_CHIP`. The simulation breaks (`SYSTEM_FAILURE`). A **SHERIFF** appears in the **NETRUNNER DEN**. The genres collapse into a final raid.
*   **Key Tokens**: `NPC_SHERIFF` in `LOC_NETRUNNER_DEN`, `WEAPON_LASER`.

---

## ♾️ Typically Generated "Universe" Stories
**Source**: `epic_universe_10k.jsonl` (Procedurally Generated)
**Scale**: 5,000+ Stories
**Logic**: These stories are generated by a **Narrative Graph Engine** that simulates the "Glitch Logic" above. It creates long-form adventures (50-200 steps) where the genre shifts dynamically based on "Portal Events" (e.g., finding a Neural Link in a Saloon). This provides the **Volume** needed for the Transformer to learn the deep correlations between tokens like `WHISKEY` and `POTION`.

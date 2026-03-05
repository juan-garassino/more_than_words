# More Than Words: Vision & Roadmap

## 🌟 From Tamagotchi to Odyssey

"More Than Words" is currently a symbolic pet simulator, but its engine is built for something far greater. The core technology—a **Symbolic Transformer** capable of modeling complex, multi-modal states (Action, World, Emotion) with ambiguous latency (λ)—is the foundation for a **Generative Quest Game**.

### The Shift: Quest Mechanics
The current "Pet State" (Hunger, Dirt, etc.) will evolve into **Hero State** (Stamina, Mana, Courage, Inventory).
The "World Tokens" (LOC_KITCHEN, FOOD_PRESENT) will evolve into **Narrative Tokens** (CAVE_ENTRANCE, ENEMY_GOBLIN, MYSTERIOUS_RUNE).

#### How the Transformer Becomes the Dungeon Master
Instead of just reacting to "Feed", the model will predict the unfolding story:
1.  **Input**: Player Action (`CAST_SPELL`) + Recent History (`[ENTERED_CAVE, SAW_GLOW]`).
2.  **Output**:
    *   **Action Suggestion**: "The AI suggests: `RUN` or `NEGOTIATE`."
    *   **World State**: Emits tokens like `ENEMY_DEFEATED`, `LOOT_FOUND`, `DOOR_OPENED`.
    *   **Lambda (λ)**: Controls the *Atmosphere*.
        *   **Low λ**: Clear description ("You see a rusty key").
        *   **High λ**: Lovecraftian horror / Dream sequence ("The walls breathe with a rhythmic pulse", "You feel a color you've never seen").

### The Latent Multiverse: "Dreaming" Between Stories
The 5 handcrafted stories are not rigid scripts; they are **gravity wells** in the model's latent space. The model learns to **blend** them:
*   **Narrative Wormholes**: Common tokens like `CAST_SPELL` or `SNEAK` exist in multiple stories. They act as bridges. You might start **SNEAKING** in the Forest (Story 1) and the model might seamlessly transition you to the Alley (Story 3) because "Sneaking" conceptually links them.
*   **High-Lambda Drift**: When ambiguity (λ) is high, the model is less constrained by the immediate logic. It is more likely to hallucinate a "dream logic" transition, warping the player from the Market to the Mansion instantly.

---

## 🧠 Brainstorming: Leveraging the Transformer Engine

The **Symbolic Transformer** (SimFormer) is versatile. Here are other ways we can leverage this engine beyond simple pet simulation:

### 1. The "Infinite Fable" (Procedural Narratives)
*   **Concept**: A text-less adventure game.
*   **Mechanic**: The story is told entirely through sequences of icons (Hieroglyphics). The player must interpret the sequence to understand the quest.
*   **Lambda Role**: Complexity of the puzzle. Higher difficulty = more abstract symbols.

### 2. "Neuro-Symbolic" NPC Brains
*   **Concept**: NPCs in a standard RPG that don't use canned dialogue trees.
*   **Mechanic**: Each NPC has a small SimFormer model. They have internal states (Fear, Greed, Loyalty).
*   **Interaction**: When you talk to them, you exchange *Intent Tokens* (THREATEN, BRIBE, FLIRT). The model predicts their reaction token (ATTACK, ACCEPT, BLUSH) based on their personality embedding.

### 3. Generative Music / Soundscapes
*   **Concept**: The game soundtrack reacts to the narrative state.
*   **Mechanic**:
    *   **Tokens**: Musical motifs (CHORD_C_MAJ, RHYTHM_FAST, INST_FLUTE).
    *   **Lambda**: Controls dissonance and reverb. High stress (High λ) = Chaotic, dissonant, reverb-heavy audio.

### 4. "Living Lock Screen" (iOS 16+ Live Activities)
*   **Concept**: A persistent background quest that plays out on your lock screen.
*   **Mechanic**: The model runs on-device (Core ML). Every time you wake your phone, the "Hero" has made a move or encountered an event based on the time passed. You only intervene when they are in danger (detected by the model predicting a `DANGER` token).

---

## 🗺 Roadmap

### Phase 1: Foundation (Current)
- [x] Symbolic Transformer Architecture
- [x] Core ML Pipeline
- [x] Basic "Pet" Loop (Feed/Clean/Sleep)
- [x] Visual Ambiguity (λ-fog)

### Phase 2: The Explorer (Next)
- [ ] **Expand Vocabulary**: Add `QUEST`, `ITEM`, `NPC`, `BIOME` tokens.
- [ ] **State Expansion**: Add `Inventory` and `Skills` to the Simulator.
- [ ] **New Training Data**: Generate "Quest Trajectories" (Hero leaves home, finds item, returns).

### Phase 3: The Dungeon Master
- [ ] **World Generation**: Model predicts the layout of the next room (`ROOM_TYPE`, `DOOR_N`, `TRAP_E`).
- [ ] **Narrative Arcs**: Train on longer sequences where actions have consequences 100 steps later.
- [ ] **Multi-Agent**: Two models interacting (Hero vs Villain).

import json
import random
from typing import List, Dict
import copy

# --- HANDCRAFTED NARRATIVE ARCHETYPES ---
# I am writing these stories explicitly to ensure high quality and coherence.
# The script will then amplify them by adding variation, ensuring volume.

ARCHETYPES = [
    # STORY 1: THE CYBERPUNK ASCENSION (Rags to Creator)
    {
        "base_id": "cyber_ascension",
        "steps": [
            # Scene 1: The Gutter
            {"action": "START", "world": ["LOC_ALLEY", "ATMOS_RAIN", "VIBE_DESPAIR"], "emotions": {"stress": 0.8, "loneliness": 0.9}, "lambda": 0.2},
            {"action": "WAIT", "world": ["LOC_ALLEY", "SENSE_SMELL_OZONE", "NPC_DROID"], "emotions": {"stress": 0.7, "loneliness": 0.9}, "lambda": 0.3},
            {"action": "INSPECT", "world": ["LOC_ALLEY", "ITEM_DATA_CHIP", "VISUAL_NEON_GLOW"], "emotions": {"stress": 0.6, "energy": 0.4}, "lambda": 0.4},
            {"action": "PICK_UP", "world": ["LOC_ALLEY", "ITEM_DATA_CHIP", "SECRET_REVEALED"], "emotions": {"stress": 0.5, "energy": 0.6}, "lambda": 0.3},
            # Scene 2: The Heist Setup
            {"action": "MOVE_N", "world": ["LOC_CYBER_CAFE", "NPC_HACKER", "SENSE_SMELL_SMOKE"], "emotions": {"stress": 0.4, "energy": 0.5}, "lambda": 0.2},
            {"action": "TRADE", "world": ["LOC_CYBER_CAFE", "ITEM_NEURAL_LINK", "NPC_HACKER"], "emotions": {"stress": 0.3, "happiness": 0.6}, "lambda": 0.2},
            {"action": "USE_ITEM", "world": ["LOC_NETRUNNER_DEN", "SYSTEM_FAILURE", "VISUAL_NEON_GLOW"], "emotions": {"stress": 0.8, "energy": 0.9}, "lambda": 0.8},
            # Scene 3: The Glitch / Ascension
            {"action": "HACK", "world": ["LOC_NETRUNNER_DEN", "EVENT_GLITCH", "MAGIC_AURA"], "emotions": {"stress": 0.9, "energy": 1.0}, "lambda": 0.9},
            {"action": "WAIT", "world": ["LOC_FOREST", "ATMOS_FOG", "VIBE_TENSION"], "emotions": {"stress": 0.5, "energy": 0.3}, "lambda": 0.9}, # Dimension Hop
            {"action": "INSPECT", "world": ["LOC_FOREST", "ITEM_ARTIFACT", "SENSE_SOUND_SIREN"], "emotions": {"stress": 0.4, "happiness": 0.1}, "lambda": 0.7}
        ]
    },
    # STORY 2: THE WESTERN REVENGE (Tragedy Loop)
    {
        "base_id": "western_revenge",
        "steps": [
            # Scene 1: The Discovery
            {"action": "START", "world": ["LOC_GHOST_TOWN", "ATMOS_HEATWAVE", "TUMBLEWEED"], "emotions": {"stress": 0.2, "energy": 0.4}, "lambda": 0.2},
            {"action": "MOVE_E", "world": ["LOC_GHOST_TOWN", "VISUAL_BLOOD_STAIN", "VIBE_TENSION"], "emotions": {"stress": 0.6, "energy": 0.6}, "lambda": 0.4},
            {"action": "INSPECT", "world": ["LOC_GHOST_TOWN", "BODY_FOUND", "NPC_SHERIFF"], "emotions": {"stress": 0.9, "happiness": 0.0}, "lambda": 0.3},
            # Scene 2: The Hunt
            {"action": "PICK_UP", "world": ["LOC_GHOST_TOWN", "ITEM_REVOLVER", "VIBE_DESPAIR"], "emotions": {"stress": 0.8, "energy": 0.8}, "lambda": 0.3},
            {"action": "MOVE_S", "world": ["LOC_DESERT", "ATMOS_HEATWAVE", "NPC_OUTLAW"], "emotions": {"stress": 0.7, "energy": 0.7}, "lambda": 0.4},
            {"action": "SPEAK", "world": ["LOC_DESERT", "NPC_OUTLAW", "SUSPECT_ANGRY"], "emotions": {"stress": 0.8, "energy": 0.8}, "lambda": 0.5},
            # Scene 3: The Duel
            {"action": "FIGHT", "world": ["LOC_DESERT", "EVENT_DUEL", "SENSE_SMELL_SMOKE"], "emotions": {"stress": 1.0, "energy": 1.0}, "lambda": 0.6},
            {"action": "WAIT", "world": ["LOC_DESERT", "BODY_FOUND", "VIBE_TRIUMPH"], "emotions": {"stress": 0.4, "happiness": 0.2}, "lambda": 0.2}
        ]
    },
    # STORY 3: THE FANTASY MERCHANT (Peaceful but Weird)
    {
        "base_id": "fantasy_merchant",
        "steps": [
            {"action": "START", "world": ["LOC_MARKET", "NPC_MERCHANT", "HAPPY_M"], "emotions": {"stress": 0.1, "happiness": 0.8}, "lambda": 0.1},
            {"action": "TRADE", "world": ["LOC_MARKET", "ITEM_POTION", "NPC_MERCHANT"], "emotions": {"stress": 0.1, "happiness": 0.9}, "lambda": 0.1},
            {"action": "DRINK_POTION", "world": ["LOC_MARKET", "MAGIC_AURA", "VISUAL_NEON_GLOW"], "emotions": {"stress": 0.0, "energy": 1.0}, "lambda": 0.5},
            {"action": "SPEAK", "world": ["LOC_MARKET", "NPC_DROID", "VIBE_TENSION"], "emotions": {"stress": 0.4, "happiness": 0.5}, "lambda": 0.7}, # Glitch encounter
            {"action": "PERSUADE", "world": ["LOC_MARKET", "NPC_ALLY", "ITEM_DATA_CHIP"], "emotions": {"stress": 0.2, "happiness": 0.7}, "lambda": 0.4}
        ]
    }
]

def add_variation(story_template: Dict, variation_idx: int) -> Dict:
    """
    Takes a hand-written story and adds subtle, realistic variations
    to create a massive dataset without losing the narrative logic.
    """
    new_story = copy.deepcopy(story_template)
    new_story["story_id"] = f"{story_template['base_id']}_{variation_idx}"
    
    # 1. Atmospheric Noise
    # Sometimes it rains, sometimes it's foggy.
    ATMOSPHERES = ["ATMOS_RAIN", "ATMOS_FOG", "ATMOS_STORM", "ATMOS_HEATWAVE"]
    SENSORY = ["SENSE_SMELL_OZONE", "SENSE_SMELL_SMOKE", "SENSE_SOUND_SIREN", "VISUAL_NEON_GLOW"]
    
    for step in new_story["steps"]:
        # 10% chance to swap atmosphere
        if random.random() < 0.2:
            # Find if there is an atmos token
            for i, token in enumerate(step["world"]):
                if "ATMOS" in token:
                    step["world"][i] = random.choice(ATMOSPHERES)
                if "SENSE" in token and random.random() < 0.3:
                     step["world"][i] = random.choice(SENSORY)
                     
        # 2. Emotional Fluctuation
        # Emotions are rarely static. Add jitter.
        for dim, val in step["emotions"].items():
            jitter = random.uniform(-0.1, 0.1)
            step["emotions"][dim] = max(0.0, min(1.0, val + jitter))
            
        # 3. Lambda Fluctuation
        step["lambda"] = max(0.0, min(1.0, step["lambda"] + random.uniform(-0.05, 0.05)))
        
    return new_story

def generate_full_dataset(output_path, target_stories=2000):
    print(f"Amplifying {len(ARCHETYPES)} narrative archetypes into {target_stories} unique stories...")
    
    all_stories = []
    
    # We loop through archetypes and generate variations
    count = 0
    while count < target_stories:
        template = random.choice(ARCHETYPES)
        story = add_variation(template, count)
        all_stories.append(story)
        count += 1
        
    with open(output_path, 'w') as f:
        for story in all_stories:
            f.write(json.dumps(story) + "\n")
            
    print(f"Success! Saved {count} rich stories/lines to {output_path}")

if __name__ == "__main__":
    generate_full_dataset("ML_Training/data/epic_quest_rich.jsonl")

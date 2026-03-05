import json
import random
from typing import List, Dict
from tqdm import tqdm

# --- CONFIGURATION ---
NUM_STORIES = 5000
MIN_LENGTH = 50
MAX_LENGTH = 200
OUTPUT_PATH = "ML_Training/data/epic_universe_10k.jsonl"

# --- KNOWLEDGE BASE ---
# This is the "Soul" of the generation. Handcrafted rules and transitions.

GENRES = {
    "CYBERPUNK": {
        "locations": ["LOC_NEON_CITY", "LOC_CYBER_CAFE", "LOC_NETRUNNER_DEN", "LOC_LAB"],
        "npcs": ["NPC_HACKER", "NPC_CORPO", "NPC_DROID"],
        "items": ["ITEM_DATA_CHIP", "ITEM_NEURAL_LINK", "WEAPON_LASER"],
        "events": ["EVENT_GLITCH", "SYSTEM_FAILURE", "EVENT_RAID"],
        "actions": ["HACK", "USE_ITEM", "FIGHT", "FLEE", "SNEAK"],
        "vibes": ["DANGER_HIGH", "DARKNESS", "LIGHT"]
    },
    "WESTERN": {
        "locations": ["LOC_SALOON", "LOC_DESERT", "LOC_GHOST_TOWN"],
        "npcs": ["NPC_SHERIFF", "NPC_OUTLAW", "NPC_BARTENDER"],
        "items": ["ITEM_REVOLVER", "ITEM_WHISKEY", "ITEM_BADGE"],
        "events": ["EVENT_DUEL", "EVENT_POKER_GAME", "TUMBLEWEED"],
        "actions": ["FIGHT", "DRINK_POTION", "TRADE", "SPEAK", "WAIT"],
        "vibes": ["DANGER_MED", "LIGHT", "DIRTY_H"]
    },
    "FANTASY": {
        "locations": ["LOC_FOREST", "LOC_CAVE", "LOC_TOWER", "LOC_MARKET"],
        "npcs": ["NPC_ALLY", "NPC_ENEMY", "NPC_MERCHANT"],
        "items": ["ITEM_KEY", "ITEM_MAP", "ITEM_POTION", "ITEM_ARTIFACT", "CURSED_OBJECT"],
        "events": ["MAGIC_AURA", "SECRET_REVEALED", "EVENT_AMBUSH", "EVENT_TREASURE"],
        "actions": ["CAST_SPELL", "READ_TOME", "OPEN_DOOR", "PICK_UP", "PERSUADE"],
        "vibes": ["MAGIC_AURA", "DARKNESS", "HAPPY_M"]
    }
}

# "Wormholes" - Specific actions/items that trigger genre shifts
PORTALS = {
    ("CYBERPUNK", "WESTERN"): {
        "trigger_action": "USE_ITEM",
        "trigger_item": "ITEM_NEURAL_LINK",
        "narrative_token": "EVENT_GLITCH" # The simulation breaks, you wake up in desert
    },
    ("WESTERN", "FANTASY"): {
        "trigger_action": "DRINK_POTION",
        "trigger_item": "ITEM_WHISKEY",
        "narrative_token": "MAGIC_AURA" # The whiskey was a potion
    },
    ("FANTASY", "CYBERPUNK"): {
        "trigger_action": "READ_TOME", # Or using an artifact
        "trigger_item": "ITEM_ARTIFACT",
        "narrative_token": "SYSTEM_FAILURE" # The magic artifact is tech
    }
}

class UniverseGenerator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.current_genre = random.choice(list(GENRES.keys()))
        self.inventory = []
        self.health = 1.0
        self.lambda_val = 0.5
        self.steps_in_genre = 0
        
    def get_genre_data(self):
        return GENRES[self.current_genre]
        
    def step(self):
        data = self.get_genre_data()
        
        # 1. Decide if we switch genres (Portal Event)
        # Only switch if we've been here a while to establish context
        if self.steps_in_genre > 10 and random.random() < 0.15:
            # Find a valid portal
            for (src, dst), rule in PORTALS.items():
                if src == self.current_genre:
                    # Execute Portal Sequence
                    action = rule["trigger_action"]
                    # If we need an item, pretend we found it or have it
                    world_tokens = [data["locations"][0], rule["trigger_item"], rule["narrative_token"]]
                    
                    self.current_genre = dst
                    self.steps_in_genre = 0
                    self.lambda_val = 0.9 # High ambiguity during transition
                    
                    return {
                        "action": action,
                        "world": world_tokens,
                        "emotions": self._gen_emotions(high_stress=True),
                        "lambda": self.lambda_val
                    }
        
        # 2. Normal Step
        self.steps_in_genre += 1
        
        # Pick a narrative beat
        beat_type = random.choice(["EXPLORE", "COMBAT", "SOCIAL", "LOOT"])
        
        action = "WAIT"
        world = []
        
        if beat_type == "EXPLORE":
            action = random.choice(["MOVE_N", "MOVE_E", "SEARCH_AREA", "SNEAK"])
            loc = random.choice(data["locations"])
            vibe = random.choice(data["vibes"])
            world = [loc, vibe]
            self.lambda_val = max(0.2, self.lambda_val - 0.1)
            
        elif beat_type == "COMBAT":
            action = "FIGHT"
            if "CAST_SPELL" in data["actions"]: action = random.choice(["FIGHT", "CAST_SPELL"])
            if "HACK" in data["actions"]: action = random.choice(["FIGHT", "HACK"])
            
            enemy = random.choice([n for n in data["npcs"] if "ENEMY" in n or "OUTLAW" in n or "DROID" in n])
            event = random.choice(data["events"])
            world = [data["locations"][0], enemy, event, "DANGER_HIGH"]
            self.lambda_val = 0.3 # Focused
            
        elif beat_type == "SOCIAL":
            action = "SPEAK"
            if "PERSUADE" in data["actions"]: action = random.choice(["SPEAK", "PERSUADE"])
            if "TRADE" in data["actions"]: action = random.choice(["SPEAK", "TRADE"])
            
            npc = random.choice(data["npcs"])
            world = [data["locations"][0], npc, "HAPPY_M"]
            self.lambda_val = 0.2
            
        elif beat_type == "LOOT":
            action = "PICK_UP"
            item = random.choice(data["items"])
            world = [data["locations"][0], item, "EVENT_TREASURE"]
            self.lambda_val = 0.4
            
        # Add random randomness to world
        if random.random() < 0.3:
            world.append(random.choice(data["events"]))
            
        return {
            "action": action,
            "world": world,
            "emotions": self._gen_emotions(),
            "lambda": self.lambda_val
        }

    def _gen_emotions(self, high_stress=False):
        base = random.random()
        return {
            "happiness": 0.1 if high_stress else base,
            "stress": 0.9 if high_stress else 1.0 - base,
            "energy": random.random(),
            "hunger": random.random() * 0.5,
            "loneliness": random.random() * 0.5
        }

    def generate_story(self, idx):
        self.reset()
        length = random.randint(MIN_LENGTH, MAX_LENGTH)
        steps = []
        
        for _ in range(length):
            steps.append(self.step())
            
        return {
            "story_id": f"epic_universe_{idx}",
            "genre_mix": "CYBER_WESTERN_FANTASY",
            "steps": steps
        }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Igniting Universe Generator...")
    print(f"Goal: {NUM_STORIES} stories with Glitch Logic.")
    
    gen = UniverseGenerator()
    
    with open(OUTPUT_PATH, 'w') as f:
        for i in tqdm(range(NUM_STORIES)):
            story = gen.generate_story(i)
            f.write(json.dumps(story) + "\n")
            
    print(f"Universe created at {OUTPUT_PATH}")

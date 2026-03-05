from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Set
import json
import os

@dataclass
class Token:
    """Represents a symbolic token with type and optional parameters"""
    symbol: str
    token_type: str  # 'ACTION', 'WORLD', 'EMOTION'
    param: Optional[float] = None

    def __repr__(self):
        if self.param is not None:
            return f"<{self.symbol}:{self.param:.1f}>"
        return f"<{self.symbol}>"

    def __hash__(self):
        return hash((self.symbol, self.token_type, self.param))

    def __eq__(self, other):
        return (self.symbol == other.symbol and
                self.token_type == other.token_type and
                self.param == other.param)


class SymbolicTokenizer:
    """
    Tokenizer for symbolic game tokens.
    Supports dynamic vocabulary building from data.
    """

    def __init__(self, vocab_path: str = None):
        # Base special tokens
        self.special_tokens = ['PAD', 'START', 'END', 'UNKNOWN']
        
        # Default vocabulary (Tamagotchi base) - will be extended if loading from file or scanning data
        self.action_vocab = [
             'FEED', 'CLEAN', 'PLAY', 'SLEEP_CMD', 'INSPECT', 'WAIT',
             'MOVE_N', 'MOVE_S', 'MOVE_E', 'MOVE_W', 'SPEAK', 'IGNORE'
        ]

        self.world_vocab = [
             'HUNGRY_L', 'HUNGRY_M', 'HUNGRY_H',
             'DIRTY_L', 'DIRTY_M', 'DIRTY_H',
             'TIRED_L', 'TIRED_M', 'TIRED_H',
             'HAPPY_L', 'HAPPY_M', 'HAPPY_H',
             'SICK', 'HEALTHY', 'ENERGETIC', 'LETHARGIC',
             'FOOD_PRESENT', 'FOOD_ABSENT',
             'LOC_0', 'LOC_1', 'LOC_2', 'LOC_3',
             'ROOM_KITCHEN', 'ROOM_BEDROOM', 'ROOM_GARDEN'
        ]

        # Emotion tokens are handled differently (continuous values)
        self.emotion_dims = ['happiness', 'energy', 'stress', 'hunger', 'loneliness']

        # If vocab_path is provided and exists, load it
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            # Otherwise build default
            self.build_vocab()

    def build_vocab(self):
        """Build token-to-index mappings from current lists"""
        # Ensure unique and sorted for determinism
        self.action_vocab = sorted(list(set(self.action_vocab)))
        self.world_vocab = sorted(list(set(self.world_vocab)))
        
        all_tokens = self.special_tokens + self.action_vocab + self.world_vocab

        self.token2idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.vocab_size = len(self.token2idx)

        # Type boundaries for embeddings
        self.type_ranges = {
            'SPECIAL': (0, len(self.special_tokens)),
            'ACTION': (len(self.special_tokens),
                      len(self.special_tokens) + len(self.action_vocab)),
            'WORLD': (len(self.special_tokens) + len(self.action_vocab),
                     len(self.special_tokens) + len(self.action_vocab) + len(self.world_vocab))
        }
        
    def fit_on_instruction_data(self, file_paths: List[str]):
        """
        Scans JSONL files and adds new tokens to vocabulary.
        Expects keys: "action" (string) and "world" (list of strings).
        """
        new_actions = set(self.action_vocab)
        new_world = set(self.world_vocab)
        
        print(f"Scanning {len(file_paths)} files to build vocabulary...")
        
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: File not found {path}")
                continue
                
            with open(path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        steps = entry.get("steps", [])
                        for step in steps:
                            # Add action
                            if "action" in step:
                                new_actions.add(step["action"])
                            
                            # Add world tokens
                            if "world" in step:
                                for w_token in step["world"]:
                                    new_world.add(w_token)
                    except json.JSONDecodeError:
                        continue
        
        # Remove special tokens if they accidentally got in
        for st in self.special_tokens:
            if st in new_actions: new_actions.remove(st)
            if st in new_world: new_world.remove(st)
            
        self.action_vocab = sorted(list(new_actions))
        self.world_vocab = sorted(list(new_world))
        
        self.build_vocab()
        print(f"Vocabulary updated. Size: {self.vocab_size} (Actions: {len(self.action_vocab)}, World: {len(self.world_vocab)})")

    def save_vocab(self, path: str):
        """Save vocabulary to JSON"""
        data = {
            "action_vocab": self.action_vocab,
            "world_vocab": self.world_vocab,
            "special_tokens": self.special_tokens,
            "emotion_dims": self.emotion_dims
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary saved to {path}")

    def load_vocab(self, path: str):
        """Load vocabulary from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.action_vocab = data.get("action_vocab", [])
        self.world_vocab = data.get("world_vocab", [])
        self.special_tokens = data.get("special_tokens", ['PAD', 'START', 'END', 'UNKNOWN'])
        self.emotion_dims = data.get("emotion_dims", ['happiness', 'energy', 'stress', 'hunger', 'loneliness'])
        
        self.build_vocab()
        print(f"Loaded vocabulary from {path}. Size: {self.vocab_size}")

    def encode_token(self, token: Token) -> int:
        """Convert Token to index"""
        return self.token2idx.get(token.symbol, self.token2idx['UNKNOWN'])

    def decode_token(self, idx: int, token_type: str) -> Token:
        """Convert index to Token"""
        symbol = self.idx2token.get(idx, 'UNKNOWN')
        return Token(symbol, token_type)

    def get_type_id(self, token_type: str) -> int:
        """Get type embedding ID"""
        type_map = {'ACTION': 0, 'WORLD': 1, 'EMOTION': 2, 'SPECIAL': 3}
        return type_map.get(token_type, 3)

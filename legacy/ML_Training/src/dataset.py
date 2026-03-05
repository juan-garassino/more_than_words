import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from .tokenizer import Token, SymbolicTokenizer

class TamagotchiDataset(Dataset):
    """
    Dataset that uses fixed-length blocks (Karpathy style).
    Concatenates all trajectories into one long sequence, then chunks into blocks.
    """

    def __init__(
        self,
        trajectories: List[List[Tuple[Token, List[Token], List[float]]]],
        tokenizer: SymbolicTokenizer,
        block_size: int = 128  # Fixed sequence length
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Flatten all trajectories into one long token stream
        all_tokens = []
        all_type_ids = []
        all_action_targets = []
        all_world_targets = []
        all_emotion_targets = []

        print(f"Flattening {len(trajectories)} trajectories into token stream...")

        for traj in trajectories:
            for action, world_tokens, emotions in traj:
                # Add action token
                all_tokens.append(self.tokenizer.encode_token(action))
                all_type_ids.append(self.tokenizer.get_type_id('ACTION'))

                # Store targets (we'll use these for prediction)
                all_action_targets.append(self.tokenizer.encode_token(action))

                # World target as multi-hot
                world_target = torch.zeros(len(self.tokenizer.world_vocab))
                for wt in world_tokens:
                    if wt.symbol in self.tokenizer.world_vocab:
                        idx = self.tokenizer.world_vocab.index(wt.symbol)
                        world_target[idx] = 1.0
                all_world_targets.append(world_target)

                # Emotion target
                all_emotion_targets.append(torch.tensor(emotions, dtype=torch.float32))

        # Convert to tensors
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.type_ids = torch.tensor(all_type_ids, dtype=torch.long)
        self.action_targets = torch.tensor(all_action_targets, dtype=torch.long)
        self.world_targets = torch.stack(all_world_targets)
        self.emotion_targets = torch.stack(all_emotion_targets)

        print(f"Total tokens: {len(self.tokens)}")
        print(f"Number of {block_size}-token blocks: {len(self.tokens) // block_size}")

    def __len__(self):
        # Number of complete blocks we can make
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        """
        Returns a block of tokens and corresponding targets.
        Input: tokens[idx*block_size : (idx+1)*block_size]
        Target: targets at position (idx+1)*block_size - 1 (predict next)
        """
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        # Get block of tokens
        block_tokens = self.tokens[start_idx:end_idx]
        block_type_ids = self.type_ids[start_idx:end_idx]

        # Target is the NEXT action/world/emotion after this block
        # Use last position's target (clamp to valid range)
        target_idx = min(end_idx - 1, len(self.action_targets) - 1)

        # Get action target as int (will be converted to tensor in collate_fn)
        action_target = self.action_targets[target_idx].item()
        if action_target >= len(self.tokenizer.action_vocab):
            action_target = 0  # Default to first action if invalid

        return {
            'tokens': block_tokens,
            'type_ids': block_type_ids,
            'target_action': action_target,  # Return as int
            'target_world': self.world_targets[target_idx],
            'target_emotions': self.emotion_targets[target_idx]
        }


def collate_fn(batch):
    """Collate function - stacks tensors and converts scalars to tensors"""
    return {
        'tokens': torch.stack([item['tokens'] for item in batch]),
        'type_ids': torch.stack([item['type_ids'] for item in batch]),
        'target_action': torch.tensor([item['target_action'] for item in batch], dtype=torch.long),
        'target_world': torch.stack([item['target_world'] for item in batch]),
        'target_emotions': torch.stack([item['target_emotions'] for item in batch])
    }

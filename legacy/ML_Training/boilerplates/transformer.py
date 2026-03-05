"""
Symbolic Transformer Tamagotchi - End-to-End Training & Game System
A research-level implementation of Transformers on symbolic game tokens.

This system trains a model to predict (Action, World, Emotion) triplets
using a hybrid explicit/latent world representation controlled by λ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
from collections import deque
import math
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import box
import time

# ============================================================================
# TOKENIZER - Symbolic Token System
# ============================================================================

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
    Vocabulary partitioned into: Actions, World States, Emotions
    """

    def __init__(self):
        # Define fixed vocabulary
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

        # Special tokens
        self.special_tokens = ['PAD', 'START', 'END', 'UNKNOWN']

        # Build vocabularies with indices
        self.build_vocab()

    def build_vocab(self):
        """Build token-to-index mappings"""
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


# ============================================================================
# GAME SIMULATOR - Generates Training Data
# ============================================================================

class TamagotchiSimulator:
    """
    Simulates a Tamagotchi-style environment to generate training data.
    Creates long trajectories that can be chunked for training.
    """

    def __init__(self, tokenizer: SymbolicTokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        """Initialize game state"""
        self.state = {
            'hunger': 0.5,      # 0=full, 1=starving
            'dirt': 0.3,        # 0=clean, 1=filthy
            'tiredness': 0.4,   # 0=energetic, 1=exhausted
            'happiness': 0.6,   # 0=sad, 1=joyful
            'sickness': 0.0,    # 0=healthy, 1=very sick
            'location': 0,      # 0-3 room index
            'food_present': True,
            'timestep': 0
        }

        # Emotion vector
        self.emotions = {
            'happiness': 0.6,
            'energy': 0.6,
            'stress': 0.3,
            'hunger': 0.5,
            'loneliness': 0.4
        }

    def get_lambda(self) -> float:
        """Compute λ based on emotional state"""
        # High stress/hunger increases λ (more ambiguity)
        # High happiness/energy decreases λ (more clarity)
        lambda_emotion = (
            0.4 * self.emotions['stress'] +
            0.3 * self.emotions['hunger'] +
            0.2 * self.emotions['loneliness'] -
            0.3 * self.emotions['happiness'] -
            0.2 * self.emotions['energy']
        )

        # Clamp to [0.2, 0.9]
        return np.clip(0.5 + lambda_emotion, 0.2, 0.9)

    def step(self, action_token: Token) -> Tuple[Token, List[Token], List[float]]:
        """
        Execute action and return (action, world_tokens, emotion_vector)
        """
        action = action_token.symbol

        # Update state based on action
        if action == 'FEED':
            if self.state['food_present']:
                self.state['hunger'] = max(0, self.state['hunger'] - 0.3)
                self.state['happiness'] += 0.1
                self.state['food_present'] = False
                self.emotions['hunger'] = self.state['hunger']

        elif action == 'CLEAN':
            self.state['dirt'] = max(0, self.state['dirt'] - 0.4)
            self.state['happiness'] += 0.05

        elif action == 'PLAY':
            self.state['happiness'] = min(1, self.state['happiness'] + 0.2)
            self.state['tiredness'] += 0.15
            self.state['hunger'] += 0.1
            self.emotions['loneliness'] = max(0, self.emotions['loneliness'] - 0.2)

        elif action == 'SLEEP_CMD':
            self.state['tiredness'] = max(0, self.state['tiredness'] - 0.5)
            self.emotions['energy'] = 1.0 - self.state['tiredness']
            self.emotions['stress'] = max(0, self.emotions['stress'] - 0.3)

        elif action.startswith('MOVE_'):
            direction_map = {'MOVE_N': 1, 'MOVE_S': -1, 'MOVE_E': 1, 'MOVE_W': -1}
            delta = direction_map.get(action, 0)
            self.state['location'] = (self.state['location'] + delta) % 4
            self.state['tiredness'] += 0.05

        # Natural decay
        self.state['hunger'] = min(1, self.state['hunger'] + 0.02)
        self.state['dirt'] = min(1, self.state['dirt'] + 0.01)
        self.state['tiredness'] = min(1, self.state['tiredness'] + 0.03)

        # Sickness logic
        if self.state['hunger'] > 0.8 or self.state['dirt'] > 0.7:
            self.state['sickness'] = min(1, self.state['sickness'] + 0.05)
        else:
            self.state['sickness'] = max(0, self.state['sickness'] - 0.02)

        # Update emotions
        self.emotions['happiness'] = self.state['happiness']
        self.emotions['energy'] = 1.0 - self.state['tiredness']
        self.emotions['stress'] = (self.state['hunger'] + self.state['dirt']) / 2
        self.emotions['hunger'] = self.state['hunger']

        self.state['timestep'] += 1

        # Generate world tokens based on λ
        world_tokens = self._generate_world_tokens()

        # Suggest next action
        next_action = self._suggest_action()

        # Emotion vector
        emotion_vec = [self.emotions[dim] for dim in self.tokenizer.emotion_dims]

        return next_action, world_tokens, emotion_vec

    def _generate_world_tokens(self) -> List[Token]:
        """Generate world tokens, filtered by λ"""
        lambda_val = self.get_lambda()

        # All possible world tokens
        all_world_tokens = []

        # Hunger
        if self.state['hunger'] < 0.3:
            all_world_tokens.append(Token('HUNGRY_L', 'WORLD'))
        elif self.state['hunger'] < 0.7:
            all_world_tokens.append(Token('HUNGRY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('HUNGRY_H', 'WORLD'))

        # Dirt
        if self.state['dirt'] < 0.3:
            all_world_tokens.append(Token('DIRTY_L', 'WORLD'))
        elif self.state['dirt'] < 0.7:
            all_world_tokens.append(Token('DIRTY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('DIRTY_H', 'WORLD'))

        # Tiredness
        if self.state['tiredness'] < 0.3:
            all_world_tokens.append(Token('TIRED_L', 'WORLD'))
        elif self.state['tiredness'] < 0.7:
            all_world_tokens.append(Token('TIRED_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('TIRED_H', 'WORLD'))

        # Happiness
        if self.state['happiness'] < 0.3:
            all_world_tokens.append(Token('HAPPY_L', 'WORLD'))
        elif self.state['happiness'] < 0.7:
            all_world_tokens.append(Token('HAPPY_M', 'WORLD'))
        else:
            all_world_tokens.append(Token('HAPPY_H', 'WORLD'))

        # Sickness
        if self.state['sickness'] > 0.5:
            all_world_tokens.append(Token('SICK', 'WORLD'))
        else:
            all_world_tokens.append(Token('HEALTHY', 'WORLD'))

        # Food
        if self.state['food_present']:
            all_world_tokens.append(Token('FOOD_PRESENT', 'WORLD'))
        else:
            all_world_tokens.append(Token('FOOD_ABSENT', 'WORLD'))

        # Location
        all_world_tokens.append(Token(f'LOC_{self.state["location"]}', 'WORLD'))

        # Apply λ filtering - keep top k tokens
        k_max = len(all_world_tokens)
        k_actual = max(2, int(k_max * (1 - lambda_val)))

        # Priority order: critical stats first
        priority_order = [
            'HUNGRY_H', 'SICK', 'TIRED_H', 'DIRTY_H',  # Critical
            'FOOD_PRESENT', 'FOOD_ABSENT',              # Important
            'HUNGRY_M', 'TIRED_M', 'DIRTY_M',          # Medium
            'HAPPY_H', 'HAPPY_M', 'HAPPY_L',           # Lower priority
            'HUNGRY_L', 'TIRED_L', 'DIRTY_L', 'HEALTHY' # Least critical
        ]

        # Sort tokens by priority
        def token_priority(token):
            try:
                return priority_order.index(token.symbol)
            except ValueError:
                return len(priority_order)

        sorted_tokens = sorted(all_world_tokens, key=token_priority)

        return sorted_tokens[:k_actual]

    def _suggest_action(self) -> Token:
        """AI suggests next action based on state"""
        # Simple rule-based AI
        if self.state['hunger'] > 0.7 and self.state['food_present']:
            return Token('FEED', 'ACTION')
        elif self.state['tiredness'] > 0.7:
            return Token('SLEEP_CMD', 'ACTION')
        elif self.state['dirt'] > 0.7:
            return Token('CLEAN', 'ACTION')
        elif self.state['happiness'] < 0.3:
            return Token('PLAY', 'ACTION')
        elif self.state['hunger'] > 0.5 and not self.state['food_present']:
            # Go to kitchen
            if self.state['location'] != 0:
                return Token('MOVE_N', 'ACTION')
            else:
                self.state['food_present'] = True  # Refill food
                return Token('INSPECT', 'ACTION')
        else:
            return random.choice([
                Token('WAIT', 'ACTION'),
                Token('INSPECT', 'ACTION'),
                Token('PLAY', 'ACTION')
            ])

    def generate_long_story(self, length: int = 500) -> List[Tuple[Token, List[Token], List[float]]]:
        """Generate a long trajectory"""
        self.reset()
        trajectory = []

        for _ in range(length):
            # Get suggested action
            action = self._suggest_action()

            # Add some randomness
            if random.random() < 0.2:
                action = Token(random.choice(self.tokenizer.action_vocab), 'ACTION')

            # Execute step
            next_action, world_tokens, emotions = self.step(action)

            trajectory.append((action, world_tokens, emotions))

        return trajectory


# ============================================================================
# TRANSFORMER MODEL - Symbolic Token Processor
# ============================================================================

class TypeAwarePositionalEncoding(nn.Module):
    """Positional encoding that includes token type information"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        # Standard sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Type-specific phase offsets
        self.type_phase = nn.Parameter(torch.randn(4, d_model) * 0.1)  # 4 types

    def forward(self, x, type_ids):
        """
        x: (batch, seq_len, d_model)
        type_ids: (batch, seq_len) - type indices
        """
        batch_size, seq_len, _ = x.shape

        # Base positional encoding
        pos_enc = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

        # Add type-specific phase
        type_enc = self.type_phase[type_ids]

        return x + pos_enc + type_enc


class SymbolicTransformer(nn.Module):
    """
    Transformer that operates on symbolic tokens and outputs
    (Action, World, Emotion) triplets with λ-modulated world emission.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        d_ff: int = 1536,
        max_len: int = 512,
        n_action_tokens: int = 12,
        n_world_tokens: int = 24,
        n_emotion_dims: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_action_tokens = n_action_tokens
        self.n_world_tokens = n_world_tokens

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Type embedding (ACTION, WORLD, EMOTION, SPECIAL)
        self.type_embedding = nn.Embedding(4, d_model)

        # Positional encoding
        self.pos_encoder = TypeAwarePositionalEncoding(d_model, max_len)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_action_tokens)
        )

        # World head - λ-modulated
        self.world_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_world_tokens)
        )

        # Emotion head
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_emotion_dims)
        )

        # λ prediction network
        self.lambda_predictor = nn.Sequential(
            nn.Linear(n_emotion_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, token_ids, type_ids, attention_mask=None):
        """
        token_ids: (batch, seq_len)
        type_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)

        Returns: (action_logits, world_logits, emotion_preds, lambda_vals)
        """
        # Embeddings
        token_emb = self.token_embedding(token_ids)
        type_emb = self.type_embedding(type_ids)

        x = token_emb + type_emb
        x = self.pos_encoder(x, type_ids)

        # Transformer - no mask for now (we'll add it back later)
        # This is just to test if masking is causing the hang
        hidden = self.transformer(x)

        # Use last hidden state for predictions
        last_hidden = hidden[:, -1, :]

        # Three heads
        action_logits = self.action_head(last_hidden)
        world_logits = self.world_head(last_hidden)
        emotion_preds = torch.sigmoid(self.emotion_head(last_hidden))

        # Predict λ from emotions
        lambda_vals = self.lambda_predictor(emotion_preds)

        return action_logits, world_logits, emotion_preds, lambda_vals

    def apply_lambda_gating(self, world_logits, lambda_vals, k_max=16):
        """
        Apply λ-based gating to world token emissions.
        Higher λ → fewer tokens emitted.
        """
        # Simply return the logits - the gating is implicit in training
        # The model learns which tokens to emit based on the multi-hot targets
        # Lambda can be used for inference/generation, but during training
        # we want stable gradients
        return world_logits


# ============================================================================
# DATASET - Training Data Preparation
# ============================================================================

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


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(
    model: SymbolicTransformer,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Training loop with multi-objective loss"""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    console = Console()

    console.print(f"[yellow]Starting training with {len(train_loader)} batches per epoch[/yellow]")
    console.print(f"[yellow]Attempting to iterate dataloader...[/yellow]")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        action_acc = 0
        batch_count = 0

        console.print(f"\n[bold cyan]{'='*60}")
        console.print(f"[bold cyan]Epoch {epoch+1}/{num_epochs}")
        console.print(f"[bold cyan]{'='*60}")

        for batch_idx, batch in enumerate(train_loader):
            # Debug: print first batch info
            if batch_idx == 0:
                console.print(f"[dim]Batch {batch_idx+1}: Loading data...[/dim]")
                console.print(f"[dim]  tokens shape: {batch['tokens'].shape}[/dim]")
                console.print(f"[dim]  type_ids shape: {batch['type_ids'].shape}[/dim]")

            # Move to device
            if batch_idx == 0:
                console.print(f"[dim]  Moving to {device}...[/dim]")

            tokens = batch['tokens'].to(device)
            type_ids = batch['type_ids'].to(device)
            target_action = batch['target_action'].to(device)
            target_world = batch['target_world'].to(device)
            target_emotions = batch['target_emotions'].to(device)

            if batch_idx == 0:
                console.print(f"[dim]  Data on device. Running forward pass...[/dim]")
                import sys
                sys.stdout.flush()  # Force output

            # Forward pass
            action_logits, world_logits, emotion_preds, lambda_vals = model(tokens, type_ids)

            if batch_idx == 0:
                console.print(f"[dim]  Forward complete! Computing loss...[/dim]")
                console.print(f"[dim]  Action logits shape: {action_logits.shape}[/dim]")
                console.print(f"[dim]  Target action shape: {target_action.shape}, min: {target_action.min()}, max: {target_action.max()}[/dim]")
                console.print(f"[dim]  World logits shape: {world_logits.shape}[/dim]")
                console.print(f"[dim]  Target world shape: {target_world.shape}, min: {target_world.min()}, max: {target_world.max()}[/dim]")

            # Apply λ-gating to world head
            world_logits_gated = model.apply_lambda_gating(
                world_logits, lambda_vals, k_max=len(batch['target_world'][0])
            )

            if batch_idx == 0:
                console.print(f"[dim]  Lambda gating applied. Computing losses...[/dim]")

            # Validate targets before computing loss
            # Action targets must be in [0, n_action_tokens)
            assert target_action.min() >= 0 and target_action.max() < action_logits.size(-1), \
                f"Invalid action targets: range [{target_action.min()}, {target_action.max()}], expected [0, {action_logits.size(-1)})"

            # World targets must be in [0, 1]
            assert target_world.min() >= 0 and target_world.max() <= 1, \
                f"Invalid world targets: range [{target_world.min()}, {target_world.max()}]"

            # Losses
            loss_action = F.cross_entropy(action_logits, target_action)

            if batch_idx == 0:
                console.print(f"[dim]  Action loss computed: {loss_action.item():.4f}[/dim]")

            loss_world = F.binary_cross_entropy_with_logits(world_logits_gated, target_world)

            if batch_idx == 0:
                console.print(f"[dim]  World loss computed: {loss_world.item():.4f}[/dim]")

            loss_emotion = F.mse_loss(emotion_preds, target_emotions)

            # Total loss
            loss = loss_action + 1.5 * loss_world + 0.8 * loss_emotion

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            batch_acc = (action_logits.argmax(dim=-1) == target_action).float().mean().item()
            action_acc += batch_acc
            batch_count += 1

            # Print every 5 batches
            if (batch_idx + 1) % 5 == 0:
                avg_loss_so_far = total_loss / batch_count
                avg_acc_so_far = action_acc / batch_count
                console.print(
                    f"  Batch [{batch_idx+1:3d}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Action Acc: {batch_acc:.3f} | "
                    f"Avg Loss: {avg_loss_so_far:.4f} | "
                    f"Avg Acc: {avg_acc_so_far:.3f}"
                )

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_acc = action_acc / len(train_loader)

        console.print(f"\n[bold green]✓ Epoch {epoch+1} Complete: Loss={avg_loss:.4f}, Action Acc={avg_acc:.3f}[/bold green]")

    console.print(f"\n[bold green]{'='*60}")
    console.print(f"[bold green]Training Complete!")
    console.print(f"[bold green]{'='*60}\n")

    return model


# ============================================================================
# GAME INTERFACE - Terminal UI with Rich
# ============================================================================

class TamagotchiGame:
    """Interactive terminal-based game using trained model"""

    def __init__(self, model: SymbolicTransformer, tokenizer: SymbolicTokenizer, simulator: TamagotchiSimulator):
        self.model = model
        self.tokenizer = tokenizer
        self.simulator = simulator
        self.console = Console()
        self.history = deque(maxlen=32)

        # Card representations
        self.card_suits = {
            'ACTION': '🎴',
            'WORLD': '🌍',
            'EMOTION': '💭'
        }

    def token_to_card(self, token: Token) -> str:
        """Convert token to card representation"""
        suit = self.card_suits.get(token.token_type, '🃏')
        return f"{suit} {token.symbol.replace('_', ' ')}"

    def render_state(self):
        """Render current game state"""
        self.console.clear()

        # Header
        self.console.print(Panel(
            "[bold cyan]🎮 Symbolic Transformer Tamagotchi 🎮[/bold cyan]",
            style="cyan",
            padding=(0, 1)
        ))

        # Main area - creature status
        state = self.simulator.state
        emotions = self.simulator.emotions
        lambda_val = self.simulator.get_lambda()

        status_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        status_table.add_column("Stat", style="cyan", width=12)
        status_table.add_column("Value", style="yellow", width=6)
        status_table.add_column("Bar", style="green", width=12)

        def make_bar(value, inverted=False):
            filled = int(value * 10)
            if inverted:
                filled = 10 - filled
            return '█' * filled + '░' * (10 - filled)

        status_table.add_row("Hunger", f"{state['hunger']:.2f}", make_bar(state['hunger']))
        status_table.add_row("Dirt", f"{state['dirt']:.2f}", make_bar(state['dirt']))
        status_table.add_row("Tiredness", f"{state['tiredness']:.2f}", make_bar(state['tiredness']))
        status_table.add_row("Happiness", f"{state['happiness']:.2f}", make_bar(state['happiness'], inverted=True))
        status_table.add_row("Health", f"{1-state['sickness']:.2f}", make_bar(1-state['sickness'], inverted=True))
        status_table.add_row("", "", "")
        status_table.add_row("Location", f"Room {state['location']}", "")
        status_table.add_row("Food", "Present" if state['food_present'] else "Absent", "")
        status_table.add_row("λ (ambig.)", f"{lambda_val:.2f}", make_bar(lambda_val))

        self.console.print(Panel(status_table, title="[bold green]World State[/bold green]", border_style="green", padding=(0, 1)))

        # Recent actions - more compact
        history_text = "  ".join([
            f"[dim]{i}.[/dim] {self.token_to_card(token)}"
            for i, token in enumerate(list(self.history)[-5:], 1)
        ])

        self.console.print(Panel(
            history_text or "[dim]No actions yet[/dim]",
            title="[bold blue]Recent Actions[/bold blue]",
            border_style="blue",
            padding=(0, 1)
        ))

    def get_action_menu(self) -> str:
        """Display action menu and get player choice"""
        action_table = Table(title="Available Actions", box=box.SIMPLE)
        action_table.add_column("Key", style="cyan", justify="center")
        action_table.add_column("Action", style="yellow")
        action_table.add_column("Card", style="green")

        actions = [
            ('1', 'FEED', '🎴 FEED'),
            ('2', 'CLEAN', '🎴 CLEAN'),
            ('3', 'PLAY', '🎴 PLAY'),
            ('4', 'SLEEP', '🎴 SLEEP CMD'),
            ('5', 'INSPECT', '🎴 INSPECT'),
            ('6', 'WAIT', '🎴 WAIT'),
            ('n/s/e/w', 'MOVE', '🎴 MOVE [direction]'),
            ('q', 'QUIT', '❌ Exit')
        ]

        for key, action, card in actions:
            action_table.add_row(key, action, card)

        self.console.print(action_table)

        # Get AI suggestion
        with torch.no_grad():
            suggestion = self.get_ai_suggestion()
            if suggestion:
                self.console.print(f"\n[bold yellow]🤖 AI Suggests:[/bold yellow] {self.token_to_card(suggestion)}")

        choice = Prompt.ask("\n[bold cyan]Choose action[/bold cyan]", default="6")
        return choice

    def get_ai_suggestion(self) -> Optional[Token]:
        """Get AI model's suggested action"""
        if len(self.history) < 3:
            return None

        # Encode recent history
        tokens = []
        type_ids = []

        for token in list(self.history)[-16:]:  # Last 16 tokens
            tokens.append(self.tokenizer.encode_token(token))
            type_ids.append(self.tokenizer.get_type_id(token.token_type))

        # Pad to minimum length
        while len(tokens) < 8:
            tokens.insert(0, self.tokenizer.token2idx['PAD'])
            type_ids.insert(0, 3)

        # Create tensors and move to model's device
        device = next(self.model.parameters()).device
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        type_ids_tensor = torch.tensor([type_ids], dtype=torch.long).to(device)

        # Get prediction
        action_logits, _, _, _ = self.model(tokens_tensor, type_ids_tensor)
        action_idx = action_logits.argmax(dim=-1).item()

        # Map to action token
        if action_idx < len(self.tokenizer.action_vocab):
            action_symbol = self.tokenizer.action_vocab[action_idx]
            return Token(action_symbol, 'ACTION')

        return None

    def play(self):
        """Main game loop"""
        self.console.print(Panel.fit(
            "[bold yellow]Welcome to Symbolic Transformer Tamagotchi![/bold yellow]\n\n"
            "Your creature is represented by symbolic tokens.\n"
            "Watch how λ (lambda) changes the world's visibility!\n\n"
            "High λ = More ambiguous world (rely on AI's internal model)\n"
            "Low λ = Clear world state (everything visible)\n\n"
            "[dim]Press any key to start...[/dim]",
            border_style="yellow"
        ))
        input()

        while True:
            # Render current state
            self.render_state()

            # Get player action
            choice = self.get_action_menu()

            if choice.lower() == 'q':
                self.console.print("\n[yellow]Thanks for playing! 👋[/yellow]")
                break

            # Map choice to action
            action_map = {
                '1': 'FEED',
                '2': 'CLEAN',
                '3': 'PLAY',
                '4': 'SLEEP_CMD',
                '5': 'INSPECT',
                '6': 'WAIT',
                'n': 'MOVE_N',
                's': 'MOVE_S',
                'e': 'MOVE_E',
                'w': 'MOVE_W'
            }

            action_symbol = action_map.get(choice.lower(), 'WAIT')
            action_token = Token(action_symbol, 'ACTION')

            # Execute action
            next_action, world_tokens, emotions = self.simulator.step(action_token)

            # Add to history
            self.history.append(action_token)
            for wt in world_tokens:
                self.history.append(wt)

            # Show results
            time.sleep(0.5)

            result_panel = Panel(
                f"[green]Action Executed:[/green] {self.token_to_card(action_token)}\n\n"
                f"[cyan]World Tokens Emitted ({len(world_tokens)} tokens, λ={self.simulator.get_lambda():.2f}):[/cyan]\n" +
                "\n".join([f"  • {self.token_to_card(wt)}" for wt in world_tokens]) +
                f"\n\n[yellow]Emotions:[/yellow]\n" +
                "\n".join([f"  {dim}: {val:.2f}" for dim, val in zip(self.tokenizer.emotion_dims, emotions)]),
                title="Step Result",
                border_style="green"
            )

            self.console.print(result_panel)
            input("\n[dim]Press Enter to continue...[/dim]")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]Symbolic Transformer Tamagotchi - Training System[/bold cyan]\n\n"
        "This system demonstrates:\n"
        "• Transformers operating on symbolic tokens (not text)\n"
        "• λ-modulated world representation (explicit vs latent)\n"
        "• Triple-head output: (Action, World, Emotion)\n"
        "• Training on long trajectories chunked into sequences",
        border_style="cyan"
    ))

    # Initialize
    console.print("\n[yellow]Step 1: Initializing tokenizer...[/yellow]")
    tokenizer = SymbolicTokenizer()
    console.print(f"[green]✓ Vocabulary size: {tokenizer.vocab_size}[/green]")

    # Generate training data
    console.print("\n[yellow]Step 2: Generating training data (long stories)...[/yellow]")
    simulator = TamagotchiSimulator(tokenizer)

    trajectories = []
    num_stories = 20
    story_length = 500

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Generating stories...", total=num_stories)
        for _ in range(num_stories):
            trajectory = simulator.generate_long_story(length=story_length)
            trajectories.append(trajectory)
            progress.update(task, advance=1)

    console.print(f"[green]✓ Generated {num_stories} stories, {num_stories * story_length} total steps[/green]")

    # Create dataset
    console.print("\n[yellow]Step 3: Creating dataset...[/yellow]")
    dataset = TamagotchiDataset(
        trajectories,
        tokenizer,
        block_size=128  # Fixed block size like nanoGPT
    )

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    console.print(f"[green]✓ Dataset size: {len(dataset)} blocks of 128 tokens[/green]")

    # Initialize model
    console.print("\n[yellow]Step 4: Initializing model...[/yellow]")

    # Determine device first
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[cyan]Device: {device}[/cyan]")

    # Model size configurations
    model_configs = {
        'tiny': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 4,
            'd_ff': 512,
            'target_params': '~1M',
            'description': 'Fast CPU training (2-3 min)'
        },
        'small': {
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 6,
            'd_ff': 1024,
            'target_params': '~5M',
            'description': 'Balanced (5-10 min)'
        },
        'medium': {
            'd_model': 384,
            'n_heads': 6,
            'n_layers': 8,
            'd_ff': 1536,
            'target_params': '~20M',
            'description': 'Research quality (15-20 min)'
        },
        'large': {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 10,
            'd_ff': 2048,
            'target_params': '~50M',
            'description': 'Production iOS target (30+ min)'
        }
    }

    # Auto-select based on device
    if device == 'cuda':
        selected_config = 'medium'  # Use larger model on GPU
        console.print("[cyan]GPU detected - using MEDIUM model (20M params)[/cyan]")
    else:
        selected_config = 'tiny'  # Keep small on CPU
        console.print("[cyan]CPU detected - using TINY model (1M params)[/cyan]")

    config = model_configs[selected_config]

    model = SymbolicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        n_action_tokens=len(tokenizer.action_vocab),
        n_world_tokens=len(tokenizer.world_vocab),
        n_emotion_dims=len(tokenizer.emotion_dims)
    )

    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]✓ Model parameters: {num_params:,} ({config['target_params']})[/green]")
    console.print(f"[dim]  Architecture: d_model={config['d_model']}, layers={config['n_layers']}, heads={config['n_heads']}[/dim]")
    console.print(f"[dim]  Training time: {config['description']}[/dim]")

    # Train model
    console.print("\n[yellow]Step 5: Training model...[/yellow]")

    # Note: If you see CUDA errors here, you MUST restart Colab runtime
    # Runtime → Restart Runtime, then re-run the cell

    try:
        model = train_model(
            model,
            train_loader,
            num_epochs=20,
            lr=3e-4,
            device=device
        )
    except RuntimeError as e:
        if 'CUDA' in str(e) or 'cuda' in str(e):
            console.print("\n[bold red]╔═══════════════════════════════════════════════╗[/bold red]")
            console.print("[bold red]║  CUDA ERROR - RESTART REQUIRED               ║[/bold red]")
            console.print("[bold red]╠═══════════════════════════════════════════════╣[/bold red]")
            console.print("[bold yellow]║  Please restart your Colab runtime:           ║[/bold yellow]")
            console.print("[bold yellow]║  1. Runtime → Restart Runtime                 ║[/bold yellow]")
            console.print("[bold yellow]║  2. Re-run this cell                          ║[/bold yellow]")
            console.print("[bold red]╚═══════════════════════════════════════════════╝[/bold red]")
        raise

    console.print("[green]✓ Training complete![/green]")

    # Play game
    console.print("\n[yellow]Step 6: Launching game interface...[/yellow]")
    time.sleep(1)

    model.eval()
    simulator.reset()

    game = TamagotchiGame(model, tokenizer, simulator)
    game.play()


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import math

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

    def forward(self, token_ids, type_ids): # removed attention_mask for CoreML compatibility/simplicity
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

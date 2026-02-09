import torch.nn as nn
# Import necessary base classes and functions from the core utility file
from attention_core import BaseMultiHeadAttention, scaled_dot_product_attention, LayerNormalization, PositionwiseFeedForward

# --- 1. Encoder's Multi-Head Self-Attention (Self-Contained Logic) ---

class EncoderSelfAttention(BaseMultiHeadAttention):
    """
    Multi-Head Self-Attention for the Encoder. Q=K=V=Input.
    The Encoder attends to ALL tokens in the input sequence (no look-ahead mask).
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__(h, d_model, dropout)
        # Linear projections for Q, K, V, and final Output (W_o)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        query, key, value = x, x, x

        # 1. Project and split heads
        query = self.split_heads(self.w_q(query))
        key = self.split_heads(self.w_k(key))
        value = self.split_heads(self.w_v(value))

        # 2. Apply Scaled Dot-Product Attention (mask handles padding only)
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. Combine heads and apply final linear projection (W_o)
        return self.w_o(self.combine_heads(x))

# --- 2. Full Encoder Layer (Single Block) ---

class EncoderLayer(nn.Module):
    """
    A single layer of the Encoder stack.
    [Self-Attention -> Add & Norm] -> [FFN -> Add & Norm]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = EncoderSelfAttention(num_heads, d_model, dropout)
        self.norm1 = LayerNormalization(d_model) # Norm after Self-Attention sub-layer
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNormalization(d_model) # Norm after FFN sub-layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source_mask=None):
        # 1. Multi-Head Self-Attention sub-layer
        attn_output = self.self_attn(x, source_mask)
        x = self.norm1(x + self.dropout(attn_output)) # Add Residual + Layer Norm (first sub-layer)

        # 2. Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # Add Residual + Layer Norm (second sub-layer)

        return x

# --- 3. Full Encoder Stack (N Layers) ---

class Encoder(nn.Module):
    """
    The full Encoder unit, composed of N identical EncoderLayers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model) # Final normalization after the last layer

    def forward(self, x, source_mask):
        # Pass the input sequentially through all N layers
        for layer in self.layers:
            x = layer(x, source_mask)
        return self.norm(x)

import torch.nn as nn
from attention_core import BaseMultiHeadAttention, scaled_dot_product_attention, LayerNormalization, PositionwiseFeedForward

# --- 1. Masked Self-Attention (Causal Logic) ---

class DecoderSelfAttention(BaseMultiHeadAttention):
    """
    Multi-Head Self-Attention for the Decoder.
    Requires a 'look-ahead mask' (target_mask) to prevent attending to future tokens.
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__(h, d_model, dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, target_mask=None):
        query, key, value = x, x, x
        query = self.split_heads(self.w_q(query))
        key = self.split_heads(self.w_k(key))
        value = self.split_heads(self.w_v(value))
        # CRITICAL: mask is applied here
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=target_mask, dropout=self.dropout)
        return self.w_o(self.combine_heads(x))

# --- 2. Multi-Head Cross-Attention (Encoder-Decoder Bridge) ---

class CrossAttention(BaseMultiHeadAttention):
    """
    Multi-Head Attention allowing the Decoder to look at the Encoder's output.
    Query (Q) from Decoder, Key/Value (K/V) from Encoder Output.
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__(h, d_model, dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, decoder_input, encoder_output, source_mask=None):
        query = decoder_input
        key, value = encoder_output, encoder_output

        query = self.split_heads(self.w_q(query))
        key = self.split_heads(self.w_k(key))
        value = self.split_heads(self.w_v(value))

        # mask handles padding in encoder output
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=source_mask, dropout=self.dropout)
        return self.w_o(self.combine_heads(x))

# --- 3. Full Decoder Layer (Single Block) ---

class DecoderLayer(nn.Module):
    """
    A single layer of the Decoder stack.
    [Masked Self-Attention -> Add & Norm] -> [Cross-Attention -> Add & Norm] -> [FFN -> Add & Norm]
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = DecoderSelfAttention(num_heads, d_model, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.cross_attn = CrossAttention(num_heads, d_model, dropout)
        self.norm2 = LayerNormalization(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        # 1. Masked Self-Attention (Causal)
        attn_output = self.self_attn(x, target_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention (Encoder-Decoder interaction)
        cross_attn_output = self.cross_attn(x, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. Feed-Forward Network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# --- 4. Full Decoder Stack (N Layers) ---

class Decoder(nn.Module):
    """
    The full Decoder unit, composed of N identical DecoderLayers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

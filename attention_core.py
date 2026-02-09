import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- A. Reusable Utility Components ---

class LayerNormalization(nn.Module):
    """Applies Layer Normalization to the input."""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN).
    This is applied to every position separately and identically.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply ReLU(x * W1 + B1) * W2 + B2
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# --- B. Scaled Dot-Product Attention (The Core Logic) ---

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Computes the attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

# --- C. Multi-Head Attention Base Class ---

class BaseMultiHeadAttention(nn.Module):
    """Handles the common mechanics: splitting, combining heads, and linear projections."""
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by num_heads (h)"

        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        """Reshapes and transposes for parallel attention calculation."""
        nbatches, seq_len, _ = x.size()
        return x.view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Reverses the split and transpose."""
        nbatches, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(nbatches, seq_len, self.d_model)

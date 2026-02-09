import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# --- 1. Import necessary stack modules ---
from input_modules import EncoderInputLayer 
from encoder_stack import Encoder
from decoder_stack import Decoder

# --- 2. Masking Utilities (Crucial for Training) ---

def subsequent_mask(size: int):
    """
    Creates a square look-ahead mask for the decoder.
    """
    subsequent_mask = torch.triu(torch.ones((1, size, size), dtype=torch.uint8), diagonal=1)
    return subsequent_mask == 0 

def make_source_mask(src: torch.Tensor, pad_idx: int):
    """
    Creates a padding mask for the encoder input and decoder cross-attention.
    Fix: Ensures shape is (batch, 1, 1, seq_len) for broadcast.
    """
    return (src != pad_idx).unsqueeze(-2).unsqueeze(-2)

def make_std_mask(tgt: torch.Tensor, pad_idx: int):
    """
    Creates the combined mask for the decoder's self-attention.
    Fix: Ensures the padding mask dimension is suitable for broadcasting.
    """
    # 1. Padding mask: Mask out padding tokens
    # Shape: (batch, 1, 1, seq_len)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(-2).unsqueeze(-2) 
    
    # 2. Look-ahead mask: Prevents attending to future tokens
    sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_pad_mask.data)
    
    # 3. Combine the two masks (logical AND)
    return tgt_pad_mask & sub_mask.bool()

# --- 3. The Master Transformer Model ---

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, pad_idx: int, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        # --- CRITICAL FIX: Store pad_idx as an instance attribute ---
        self.pad_idx = pad_idx 
        # -----------------------------------------------------------
        self.d_model = d_model

        # --- Input/Output Embedding Layers (from input_modules.py) ---
        self.src_embed_layer = EncoderInputLayer(d_model, src_vocab_size, max_len, dropout) 
        self.tgt_embed_layer = EncoderInputLayer(d_model, tgt_vocab_size, max_len, dropout)

        # --- Encoder and Decoder Stacks (from encoder_stack.py, decoder_stack.py) ---
        self.encoder = Encoder(d_model, h, d_ff, dropout, N)
        self.decoder = Decoder(d_model, h, d_ff, dropout, N)

        # --- Final Prediction Head ---
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters with Xavier initialization for stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # The pad_idx is now correctly accessible via self.pad_idx
        src_mask = make_source_mask(src, self.pad_idx)
        tgt_mask = make_std_mask(tgt, self.pad_idx)

        # 2. Encode
        src_embed = self.src_embed_layer(src)
        encoder_output = self.encoder(src_embed, src_mask) 
        
        # 3. Decode
        tgt_embed = self.tgt_embed_layer(tgt)
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
        
        # 4. Final Projection
        return self.generator(decoder_output)
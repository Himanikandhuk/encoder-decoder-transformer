import torch
import torch.nn as nn
import math

# --- 1. Input/Output Embedding Layer ---

class InputEmbeddings(nn.Module):
    """
    Converts integer token IDs into dense, fixed-size numerical vectors (embeddings).
    This layer is used for both the Encoder's input (NL) and the Decoder's input (Code).
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model (int): The dimensionality of the embedding vectors (e.g., 512).
        vocab_size (int): The total number of unique tokens in the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # The core embedding lookup table
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Performs the embedding lookup and scales the output as per the original paper.
        """
        # Scaling by sqrt(d_model) improves stability before adding positional encoding.
        return self.embedding(x) * math.sqrt(self.d_model)

# --- 2. Positional Encoding Layer ---

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings since the Transformer lacks
    recurrent or convolutional layers to inherently track token order.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        d_model (int): The embedding dimensionality.
        seq_len (int): The maximum expected sequence length (e.g., 512 tokens).
        dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create a non-learnable positional encoding matrix (pe)
        # Shape: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a tensor representing positions (0, 1, 2, ..., seq_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Shape: (seq_len, 1)

        # Create the division term (the denominator in the sin/cos formula)
        # div_term = 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sine function to even indices (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply the cosine function to odd indices (2i + 1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register as a buffer (not a learnable parameter)
        pe = pe.unsqueeze(0) # Shape: (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encoding matrix to the input embeddings (x).
        """
        # Slice the positional encoding matrix to match the length of the input tensor x
        # requires_grad_(False) ensures no gradients are calculated for 'pe'
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

# --- 3. Combined Initial Layer (Demonstration) ---

class EncoderInputLayer(nn.Module):
    """
    Combines embedding and positional encoding, representing the full input layer.
    """
    def __init__(self, d_model: int, vocab_size: int, seq_len: int, dropout: float):
        super().__init__()
        self.word_embedding = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

    def forward(self, x):
        x = self.word_embedding(x)
        x = self.positional_encoding(x)
        return x

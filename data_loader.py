import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple
import numpy as np

from tokenizer_util import CustomTokenizer 

# --- Configuration (MUST match hyperparameters) ---
MAX_SEQ_LEN = 300
PAD_IDX = 0

# --- NEW: Define Column Names (Based on uploaded CSV format) ---
NL_COLUMN_NAME = 'Problem' 
CODE_COLUMN_NAME = 'Python Code' 

# --- 1. PyTorch Dataset Implementation ---

class CodeDataset(Dataset):
    """
    A custom PyTorch Dataset to handle the Natural Language (NL) to Code pairs.
    """
    def __init__(self, df: pd.DataFrame, tokenizer: CustomTokenizer):
        self.df = df
        self.tokenizer = tokenizer
        # Access data using the defined column names
        self.nl_prompts = df[NL_COLUMN_NAME].astype(str).tolist()
        self.code_solutions = df[CODE_COLUMN_NAME].astype(str).tolist()

        self.src_sos_id = tokenizer.token_to_id['<SOS>']
        self.src_eos_id = tokenizer.token_to_id['<EOS>']
        self.tgt_sos_id = tokenizer.token_to_id['<SOS>']
        self.tgt_eos_id = tokenizer.token_to_id['<EOS>']

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tokenized, raw source (NL) and target (Code) sequences.
        """
        src_text = self.nl_prompts[idx]
        tgt_text = self.code_solutions[idx]
        
        # --- 1. Process Source (NL Prompt) ---
        src_tokens = self.tokenizer.tokenize(src_text)
        src_sequence = [self.src_sos_id] + self.tokenizer.encode(src_tokens) + [self.src_eos_id]
        src_tensor = torch.tensor(src_sequence, dtype=torch.long)

        # --- 2. Process Target (Code Solution) ---
        tgt_tokens = self.tokenizer.tokenize(tgt_text)
        tgt_sequence = [self.tgt_sos_id] + self.tokenizer.encode(tgt_tokens) + [self.tgt_eos_id]
        tgt_tensor = torch.tensor(tgt_sequence, dtype=torch.long)

        return src_tensor, tgt_tensor

# --- 2. Custom Collation Function (collate_fn) ---

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads sequences within a batch to uniform length and separates the 
    Encoder input, Decoder input (tgt_in), and Decoder target (tgt_y).
    """
    src_sequences, tgt_sequences = zip(*batch)

    # Find the maximum length in the current batch for padding
    max_src_len = min(MAX_SEQ_LEN, max(len(s) for s in src_sequences))
    max_tgt_len = min(MAX_SEQ_LEN, max(len(t) for t in tgt_sequences))

    # --- Pad Source Sequences (for Encoder Input) ---
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_sequences, 
        batch_first=True, 
        padding_value=PAD_IDX
    )[:, :max_src_len] # Apply MAX_SEQ_LEN truncation/padding

    # --- Pad Target Sequences (for Decoder Input and Output) ---
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_sequences, 
        batch_first=True, 
        padding_value=PAD_IDX
    )[:, :max_tgt_len] # Apply MAX_SEQ_LEN truncation/padding

    # --- Critical Transformation for Sequence-to-Sequence Training ---
    # tgt_in: Includes SOS, excludes EOS. Used as the Decoder's input.
    tgt_in = tgt_padded[:, :-1]
    
    # tgt_y: Excludes SOS, includes EOS. Used as the CrossEntropyLoss ground truth target.
    tgt_y = tgt_padded[:, 1:]

    # Return source, decoder input, true target, and original sequence lengths
    src_lengths = torch.tensor([len(s) for s in src_sequences])
    tgt_lengths = torch.tensor([len(t) for t in tgt_sequences])

    return src_padded, tgt_in, tgt_y, src_lengths, tgt_lengths

# --- 3. Orchestrator Function ---

def create_data_loaders(data_path: str, tokenizer: CustomTokenizer, batch_size: int = 32, validation_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """
    Orchestrates the entire data loading and splitting process using Pandas.
    """
    print(f"Loading data from: {data_path}")
    
    # Load the data and handle missing values by dropping rows
    data = pd.read_csv(data_path).dropna(subset=[NL_COLUMN_NAME, CODE_COLUMN_NAME]).reset_index(drop=True)
    print(f"Loaded and cleaned {len(data)} samples.")

    # --- Splitting Data ---
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int((1 - validation_split) * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    # --- Creating Datasets and Loaders ---
    train_dataset = CodeDataset(train_data, tokenizer)
    val_dataset = CodeDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=True 
    )

    print(f"Data ready: Train samples={len(train_data)}, Validation samples={len(val_data)}")
    return train_loader, val_loader

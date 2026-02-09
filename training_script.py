import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time

# Import your custom modules
from transformer_model import Transformer
from tokenizer_util import CustomTokenizer, PAD_TOKEN 
from data_loader import create_data_loaders, NL_COLUMN_NAME, CODE_COLUMN_NAME

# --- 1. Global Configuration (The "Sprint" Settings) ---
D_MODEL = 512       
D_FF = 2048         
NUM_HEADS = 8       
NUM_LAYERS = 6      
MAX_SEQ_LEN = 300   
DROPOUT = 0.1       

# Iterative Training Parameters
BATCH_SIZE = 32     
LEARNING_RATE = 1e-4

# --- SET THESE FOR YOUR SPRINT ---
NUM_EPOCHS_TO_RUN = 10     # How many epochs to run in this session
START_EPOCH = 1            # If resuming, set this to the NEXT epoch (e.g., 2 if 1 is done)
RESUME_TRAINING = True     # Set to True to load weights from previous checkpoint
# ---------------------------------

DATA_FILE_PATH = 'dataset/ProblemSolutionPythonV3_TRAIN.csv' 
CHECKPOINT_DIR = './checkpoints'

def run_full_training_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training Sprint on {device} ---")
    
    # 1. Load Data and Initialize Tokenizer
    try:
        raw_data = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {DATA_FILE_PATH}.")
        return

    tokenizer = CustomTokenizer(raw_data)
    PAD_IDX = tokenizer.token_to_id[PAD_TOKEN]
    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = tokenizer.vocab_size

    # 2. Create Data Loaders
    train_loader, val_loader = create_data_loaders(DATA_FILE_PATH, tokenizer, batch_size=BATCH_SIZE)

    # 3. Initialize the Transformer Model
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        pad_idx=PAD_IDX,
        N=NUM_LAYERS,
        d_model=D_MODEL,
        d_ff=D_FF,
        h=NUM_HEADS,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN
    ).to(device)
    
    # 4. Define Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) 

    # --- 5. LOGIC TO RESUME FROM CHECKPOINT ---
    if RESUME_TRAINING and START_EPOCH > 1:
        last_checkpoint = os.path.join(CHECKPOINT_DIR, f'transformer_checkpoint_epoch_{START_EPOCH - 1}.pth')
        if os.path.exists(last_checkpoint):
            print(f"Loading previous weights from: {last_checkpoint}")
            model.load_state_dict(torch.load(last_checkpoint))
        else:
            print(f"Warning: Checkpoint {last_checkpoint} not found. Starting from scratch.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # --- 6. Training Loop ---
    model.train()
    
    # Range adjusted to support iterative numbering
    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS_TO_RUN):
        epoch_start_time = time.time()
        total_epoch_loss = 0
        
        for batch_idx, (src, tgt_in, tgt_y, src_len, tgt_len) in enumerate(train_loader):
            src, tgt_in, tgt_y = src.to(device), tgt_in.to(device), tgt_y.to(device)

            optimizer.zero_grad() 

            # Forward Pass
            output_logits = model(src, tgt_in) 
            
            # Loss Calculation
            loss = criterion(output_logits.contiguous().view(-1, TGT_VOCAB_SIZE),
                             tgt_y.contiguous().view(-1))
            
            # Backward Pass
            loss.backward()

            # --- Safety: Gradient Clipping ---
            # Prevents the "Exploding Gradient" problem common in Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- End of Epoch Analysis ---
        avg_loss = total_epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print("-" * 50)
        print(f"Epoch {epoch} Finished | Time: {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'transformer_checkpoint_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved: {checkpoint_path}")
    
    print("\nTraining Sprint Complete. Run inference cell to analyze learning.")

if __name__ == "__main__":
    run_full_training_pipeline()
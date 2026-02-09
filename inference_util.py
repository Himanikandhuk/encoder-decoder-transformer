import torch
import torch.nn.functional as F

def generate_code(model, tokenizer, prompt, max_len=300, device='cuda', temperature=0.7):
    """
    Generates Python code from a prompt using temperature sampling 
    to ensure structural variety and prevent logical loops.
    """
    model.eval()
    
    # 1. Prepare input
    tokens = tokenizer.tokenize(prompt)
    src_ids = [tokenizer.token_to_id['<SOS>']] + tokenizer.encode(tokens) + [tokenizer.token_to_id['<EOS>']]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    # 2. Start decoding
    tgt_ids = [tokenizer.token_to_id['<SOS>']]
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_ids).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
            
            # Select the last token prediction
            logits = output[0, -1, :]
            
            # Apply temperature sampling
            # Low Temp (0.1) = Very predictable/Repetitive
            # High Temp (1.0) = Creative/Unstable
            # 0.7 is the "Sweet Spot" for code logic
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
        tgt_ids.append(next_token)
        
        if next_token == tokenizer.token_to_id['<EOS>']:
            break
            
    # 3. Use the new formatting-aware decoder
    return tokenizer.decode(tgt_ids)
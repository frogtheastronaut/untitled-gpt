import torch
import argparse
import os
import sys
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTModel
from config import GPTConfig
from dataset import TiktokenTokenizer

def generate(model, context, max_length, temperature=1.0, top_k=None):
    model.eval()
    
    for i in range(max_length):
        idx_cond = context[:, -model.config.n_ctx:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Stop if it's the end token
        if idx_next.item() == 50256:  # <|endoftext|>
            break
            
        context = torch.cat((context, idx_next), dim=1)
    
    return context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='Directory containing checkpoint')
    parser.add_argument('--prompt', type=str, default="Hello", help='Input prompt')
    parser.add_argument('--length', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = TiktokenTokenizer()
    print(f"Loaded tokenizer with {tokenizer.vocab_size} tokens.")

    # Locate Checkpoint
    ckpt_path = os.path.join(args.work_dir, 'final_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.work_dir, 'checkpoint_last.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"Error: No checkpoint found in {args.work_dir}")
        return
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load Config
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Loaded configuration from checkpoint.")
    else:
        print("Error: Checkpoint does not contain configuration. Cannot reconstruct model.")
        return

    # Initialize Model
    model = GPTModel(config)
    
    # Load State Dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    
    # Generate
    context_tokens = tokenizer.encode(args.prompt)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\nGenerating from prompt: '{args.prompt}'\n" + "-"*50)
    out_tokens = generate(model, context, args.length, args.temperature, args.top_k)
    out_text = tokenizer.decode(out_tokens[0].tolist())
    print(out_text)
    print("-" * 50)

if __name__ == '__main__':
    main()
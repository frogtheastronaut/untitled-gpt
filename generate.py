import torch
import argparse
import os
import sys
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPTModel
from config import GPTConfig
from dataset import CharTokenizer

def generate(model, context, length, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(length):
        # crop context if needed
        idx_cond = context[:, -model.config.n_ctx:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)
        
    return context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True, help='Directory containing checkpoint and vocab.json')
    parser.add_argument('--prompt', type=str, default="Hello", help='Input prompt')
    parser.add_argument('--length', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--column_name', type=str, default='text', help='Name of the text column in the dataset')
    
    # Model architecture flags (must match training)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=128)
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load Vocab
    vocab_path = os.path.join(args.work_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"Error: vocab.json not found in {args.work_dir}")
        return
        
    with open(vocab_path, 'r') as f:
        vocab_chars = json.load(f)
    
    tokenizer = CharTokenizer(vocab=vocab_chars)
    print(f"Loaded vocab with {tokenizer.vocab_size} characters.")

    # Load Model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head
    )
    
    model = GPTModel(config)
    
    ckpt_path = os.path.join(args.work_dir, 'final_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.work_dir, 'checkpoint_last.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"Error: No checkpoint found in {args.work_dir}")
        return
        
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Use strict=False to handle architecture changes (e.g. bias buffer removal)
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

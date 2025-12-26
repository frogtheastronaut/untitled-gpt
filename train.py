import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import glob

from model import GPTModel
from config import GPTConfig
from dataset import TextDataset, CharTokenizer

import json

def save_checkpoint(model, optimizer, step, filepath):
    print(f"Saving checkpoint to {filepath}")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def save_vocab(vocab, filepath):
    with open(filepath, 'w') as f:
        json.dump(vocab, f)

def load_checkpoint(filepath, model, optimizer):
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def main():
    parser = argparse.ArgumentParser(description="Train GPT")
    
    # Data params
    parser.add_argument('--data_files', type=str, nargs='+', required=True, help='List of dataset files (csv, tsv, txt)')
    parser.add_argument('--work_dir', type=str, default='out', help='Directory to save checkpoints')
    
    # Model params
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=128, help='Context size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=10000, help='Total training iterations')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Dataset
    print("Loading dataset...")
    dataset = TextDataset(args.data_files, args.block_size)
    print(f"Vocab size: {dataset.vocab_size}")
    
    # Save vocab
    save_vocab(dataset.tokenizer.chars, os.path.join(args.work_dir, 'vocab.json'))
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Config & Model
    config = GPTConfig(
        vocab_size=dataset.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout
    )
    
    model = GPTModel(config)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    start_step = 0
    
    # Auto-resume logic
    ckpt_path = os.path.join(args.work_dir, 'checkpoint_last.pt')
    if args.resume and os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    model.train()
    data_iter = iter(dataloader)
    
    pbar = tqdm(range(start_step, args.max_iters))
    for step in pbar:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            
    # Save final model
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.work_dir, 'final_model.pt'))
    print("Training complete!")

if __name__ == '__main__':
    main()

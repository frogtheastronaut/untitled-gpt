import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import GPTModel
from config import GPTConfig
from dataset import TextDataset, TiktokenTokenizer

def save_checkpoint(model, optimizer, step, filepath, config=None):
    print(f"Saving checkpoint to {filepath}")
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if config:
        state['config'] = config
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer):
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def estimate_total_batches(data_files, tokenizer, block_size, batch_size, max_lines=None):
    """
    Estimate the total number of batches for a single epoch.
    This is an approximation based on file size and a sample of the data.
    """
    total_bytes = 0
    for f in data_files:
        if os.path.exists(f):
            total_bytes += os.path.getsize(f)
    
    if total_bytes == 0:
        return None
        
    # Sample first file to get stats
    sample_size = 1024 * 1024 # 1MB
    try:
        with open(data_files[0], 'r', encoding='utf-8', errors='ignore') as f:
            sample_text = f.read(sample_size)
    except Exception:
        return None
    
    if not sample_text:
        return None
        
    sample_bytes = len(sample_text.encode('utf-8'))
    sample_lines = sample_text.count('\n')
    # Use the tokenizer to count tokens in the sample
    sample_tokens = len(tokenizer.encode(sample_text))
    
    tokens_per_byte = sample_tokens / sample_bytes
    lines_per_byte = sample_lines / sample_bytes if sample_lines > 0 else 0
    
    est_total_tokens = total_bytes * tokens_per_byte
    
    if max_lines is not None and lines_per_byte > 0:
        est_total_lines = total_bytes * lines_per_byte
        if max_lines < est_total_lines:
            est_total_tokens = est_total_tokens * (max_lines / est_total_lines)
            
    # Total batches = tokens / block_size / batch_size
    est_batches = int(est_total_tokens / (block_size * batch_size))
    return est_batches

def evaluate(model, dataloader, device, max_batches=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')

def main():
    parser = argparse.ArgumentParser(description="Train GPT")
    
    # Data params
    parser.add_argument('--data_files', type=str, nargs='+', required=True, help='List of dataset files (csv, tsv, txt)')
    parser.add_argument('--work_dir', type=str, default='out', help='Directory to save checkpoints')
    parser.add_argument('--separator', type=str, default=None, help='Custom separator for plain text files')
    parser.add_argument('--max_lines', type=int, default=None, help='Maximum number of lines/records to read')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Validation split ratio')
    
    # Model params
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=128, help='Context size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_iters', type=int, default=10000, help='Total training iterations')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (overrides max_iters)')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
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
    train_dataset = TextDataset(args.data_files, args.block_size, separator=args.separator, max_lines=args.max_lines, split='train', val_ratio=args.val_ratio)
    val_dataset = TextDataset(args.data_files, args.block_size, separator=args.separator, max_lines=args.max_lines, split='val', val_ratio=args.val_ratio)
    
    print(f"Vocab size: {train_dataset.vocab_size}")
    
    # IterableDataset cannot be shuffled by DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Config & Model
    config = GPTConfig(
        vocab_size=train_dataset.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout
    )
    
    model = GPTModel(config)
    model.to(device)
    
    # Estimate total steps for scheduler
    total_steps = args.max_iters
    if args.epochs is not None:
        print("Estimating dataset size...")
        est_batches = estimate_total_batches(args.data_files, train_dataset.tokenizer, args.block_size, args.batch_size, args.max_lines)
        if est_batches:
            est_batches = int(est_batches * (1 - args.val_ratio))
            print(f"Estimated batches per epoch: {est_batches}")
            total_steps = est_batches * args.epochs
        else:
            print("Could not estimate dataset size. Using default T_max=10000")
            total_steps = 10000

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Only create scaler for CUDA (MPS doesn't support mixed precision)
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    start_step = 0
    
    # Auto-resume logic
    ckpt_path = os.path.join(args.work_dir, 'checkpoint_last.pt')
    if args.resume and os.path.exists(ckpt_path):
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    model.train()
    
    # Training statistics
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5  # Stop if no improvement for 5 validations
    
    if args.epochs is not None:
        step = start_step
        for epoch in range(args.epochs):
            print(f"Starting epoch {epoch+1}/{args.epochs}")
            # Use estimated batches for pbar if available
            pbar = tqdm(train_loader, total=est_batches if 'est_batches' in locals() else None)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                # CUDA: Use autocast for mixed precision
                if device == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        logits, loss = model(x, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # MPS or CPU - no autocast
                    logits, loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    optimizer.step()
                
                scheduler.step()
                step += 1
                pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
                
                if step % args.save_every == 0:
                    save_checkpoint(model, optimizer, step, ckpt_path, config)
                    val_loss = evaluate(model, val_loader, device)
                    print(f"Step {step}, Val Loss: {val_loss:.4f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        save_checkpoint(model, optimizer, step, 
                                      os.path.join(args.work_dir, 'checkpoint_best.pt'), 
                                      config)
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience:
                            print(f"Early stopping at step {step} - no improvement for {max_patience} validations")
                            break
        
        # Save final model
        save_checkpoint(model, optimizer, step, os.path.join(args.work_dir, 'final_model.pt'), config)
    else:
        data_iter = iter(train_loader)
        
        pbar = tqdm(range(start_step, args.max_iters))
        for step in pbar:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
                
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # CUDA: Use autocast for mixed precision
            if device == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS or CPU - no autocast
                logits, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            if (step + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, ckpt_path, config)
                val_loss = evaluate(model, val_loader, device)
                print(f"Step {step+1}, Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, step + 1, 
                                  os.path.join(args.work_dir, 'checkpoint_best.pt'), 
                                  config)
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"Early stopping at step {step+1} - no improvement for {max_patience} validations")
                        break
                
        # Save final model
        save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.work_dir, 'final_model.pt'), config)
    print("Training complete!")

if __name__ == '__main__':
    main()
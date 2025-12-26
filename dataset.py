import torch
from torch.utils.data import IterableDataset
import pandas as pd
import os
import argparse
import tiktoken

class TiktokenTokenizer:
    def __init__(self, model_name="gpt2"):
        self.enc = tiktoken.get_encoding(model_name)
        self.vocab_size = self.enc.n_vocab
        # gpt2 encoding uses 50256 as <|endoftext|>
        self.eot = self.enc.eot_token 

    def encode(self, s):
        # allowed_special="all" allows <|endoftext|> to be encoded properly
        return self.enc.encode(s, allowed_special="all")

    def decode(self, l):
        return self.enc.decode(l)

class TextDataset(IterableDataset):
    def __init__(self, file_paths, block_size, tokenizer=None, column_name='text', separator=None, max_lines=None, split='train', val_ratio=0.1, seed=42):
        self.file_paths = file_paths
        self.block_size = block_size
        self.column_name = column_name
        self.separator = separator
        self.max_lines = max_lines
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        
        if tokenizer is None:
            self.tokenizer = TiktokenTokenizer()
        else:
            self.tokenizer = tokenizer
            
        self.vocab_size = self.tokenizer.vocab_size

    def _iterate_text(self):
        lines_count = 0
        # Simple deterministic hashing for split
        # We use a counter to decide if a record belongs to train or val
        # This assumes records are somewhat shuffled or independent
        
        total_processed = 0
        
        for file_path in self.file_paths:
            if self.max_lines is not None and lines_count >= self.max_lines:
                break
                
            if file_path.endswith('.tsv') or file_path.endswith('.csv'):
                sep = '\t' if file_path.endswith('.tsv') else ','
                # Use chunksize for lazy loading
                chunk_iter = pd.read_csv(file_path, sep=sep, chunksize=1000)
                for chunk in chunk_iter:
                    if self.column_name in chunk.columns:
                        texts = chunk[self.column_name].astype(str).tolist()
                    else:
                        texts = chunk.iloc[:, 0].astype(str).tolist()
                    
                    for text in texts:
                        # Split logic
                        is_val = (total_processed % 100) < (self.val_ratio * 100)
                        total_processed += 1
                        
                        if self.split == 'train' and is_val:
                            continue
                        if self.split == 'val' and not is_val:
                            continue
                            
                        yield text
                        lines_count += 1
                        if self.max_lines is not None and lines_count >= self.max_lines:
                            return
            else:
                # Plain text
                sep = self.separator if self.separator else '\n'
                
                if sep == '\n':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            # Split logic
                            is_val = (total_processed % 100) < (self.val_ratio * 100)
                            total_processed += 1
                            
                            if self.split == 'train' and is_val:
                                continue
                            if self.split == 'val' and not is_val:
                                continue

                            yield line
                            lines_count += 1
                            if self.max_lines is not None and lines_count >= self.max_lines:
                                return
                else:
                    # Custom separator logic
                    with open(file_path, 'r', encoding='utf-8') as f:
                        buffer = ""
                        while True:
                            chunk = f.read(4096)
                            if not chunk:
                                if buffer:
                                    # Split logic
                                    is_val = (total_processed % 100) < (self.val_ratio * 100)
                                    total_processed += 1
                                    
                                    if (self.split == 'train' and not is_val) or (self.split == 'val' and is_val):
                                        yield buffer
                                        lines_count += 1
                                break
                            buffer += chunk
                            while sep in buffer:
                                part, buffer = buffer.split(sep, 1)
                                
                                # Split logic
                                is_val = (total_processed % 100) < (self.val_ratio * 100)
                                total_processed += 1
                                
                                if (self.split == 'train' and not is_val) or (self.split == 'val' and is_val):
                                    yield part
                                    lines_count += 1
                                    
                                if self.max_lines is not None and lines_count >= self.max_lines:
                                    return

    def __iter__(self):
        buffer = []
        for text in self._iterate_text():
            encoded = self.tokenizer.encode(text)
            buffer.extend(encoded)
            
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:] # Non-overlapping chunks
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

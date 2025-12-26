import torch
from torch.utils.data import IterableDataset
import pandas as pd
import os
import argparse

class CharTokenizer:
    def __init__(self, data=None, vocab=None):
        if vocab:
            self.chars = vocab
        elif data:
            self.chars = sorted(list(set(data)))
        else:
            raise ValueError("Either data or vocab must be provided")
            
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class TextDataset(IterableDataset):
    def __init__(self, file_paths, block_size, tokenizer=None, column_name='text', separator=None, max_lines=None):
        self.file_paths = file_paths
        self.block_size = block_size
        self.column_name = column_name
        self.separator = separator
        self.max_lines = max_lines
        
        if tokenizer is None:
            # Build vocab by scanning data
            chars = set()
            print("Building vocabulary from data...")
            for text in self._iterate_text():
                chars.update(text)
            self.tokenizer = CharTokenizer(vocab=sorted(list(chars)))
        else:
            self.tokenizer = tokenizer
            
        self.vocab_size = self.tokenizer.vocab_size

    def _iterate_text(self):
        lines_count = 0
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
                                    yield buffer
                                    lines_count += 1
                                break
                            buffer += chunk
                            while sep in buffer:
                                part, buffer = buffer.split(sep, 1)
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

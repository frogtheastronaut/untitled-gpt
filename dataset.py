import torch
from torch.utils.data import Dataset
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

class TextDataset(Dataset):
    def __init__(self, file_paths, block_size, tokenizer=None, column_name='text'):
        self.data = ""
        for file_path in file_paths:
            if file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # Try reading as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data += f.read()
                    continue
            
            # Assume the text is in the first column if not specified, or look for 'text' column
            if column_name in df.columns:
                text_col = column_name
            else:
                text_col = df.columns[0]
            
            self.data += "\n".join(df[text_col].astype(str).tolist())

        if tokenizer is None:
            self.tokenizer = CharTokenizer(data=self.data)
        else:
            self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size
        self.ids = self.tokenizer.encode(self.data)
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.ids[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

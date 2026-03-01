import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json


class CharDataset(Dataset):
    def __init__(self, data_dir, split, block_size):

        assert split in ["train", "val", "test"], "Invalid split name"

        self.block_size = block_size

        # Load tensor with PyTorch
        data_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")

        self.data = torch.load(data_path)

        # Load vocabulary
        vocab_path = os.path.join(data_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)

        self.itos = {int(v): k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        if len(self.data) < block_size + 1:
            raise ValueError("Dataset too small for given block_size")

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1].long()  # ensure torch.long
        y = chunk[1:].long()
        return x, y
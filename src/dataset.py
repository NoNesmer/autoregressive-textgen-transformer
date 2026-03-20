import torch
from torch.utils.data import Dataset
import os
import json


class CharDataset(Dataset):
    """
    Character-level dataset for autoregressive language modeling.

    With stride=block_size (default), sequences are non-overlapping, making
    full-dataset training feasible on CPU (~27 min/epoch for the baseline model).
    Set stride=1 for maximum data utilization (very slow on CPU).
    """

    def __init__(self, data_dir, split, block_size, stride=None):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"

        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

        data_path = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Run src/prepare_data.py first.")

        self.data = torch.load(data_path, weights_only=True)

        vocab_path = os.path.join(data_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)
        self.itos = {int(v): k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        if len(self.data) < block_size + 1:
            raise ValueError("Dataset too small for given block_size")

    def __len__(self):
        return max(1, (len(self.data) - self.block_size - 1) // self.stride)

    def __getitem__(self, idx):
        offset = idx * self.stride
        chunk = self.data[offset: offset + self.block_size + 1]
        x = chunk[:-1].long()
        y = chunk[1:].long()
        return x, y

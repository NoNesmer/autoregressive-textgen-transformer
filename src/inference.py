"""Shared helpers for loading config, vocab, and checkpoints (evaluation & experiments)."""

import json
import os
import yaml
import torch

from src.model import CharTransformer

DATA_DIR = "data/processed"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_vocab(data_dir=DATA_DIR):
    vocab_path = os.path.join(data_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    itos = {int(v): k for k, v in stoi.items()}
    return stoi, itos


def load_model(config, vocab_size, checkpoint_path, device):
    model = CharTransformer(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        embed_dim=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

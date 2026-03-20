"""
prepare_data.py

Preprocess the Shakespeare dataset for character-level autoregressive modeling.

Pipeline:
1. Download raw text from Project Gutenberg (if not already present)
2. Strip Gutenberg boilerplate (header/footer)
3. Normalize text (case, whitespace, punctuation)
4. Build character vocabulary
5. Encode text to integer tokens
6. Create contiguous train/val/test split (80/10/10)
7. Save processed binary files + vocabulary
"""

import os
import json
import urllib.request
import numpy as np
import torch

RAW_DIR = "data/raw"
RAW_PATH = os.path.join(RAW_DIR, "shakespeare.txt")
PROCESSED_DIR = "data/processed"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


def download_data():
    """Download Shakespeare text from Project Gutenberg if not present."""
    if os.path.exists(RAW_PATH):
        print(f"Raw data already exists at {RAW_PATH}, skipping download.")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading Shakespeare from {GUTENBERG_URL} ...")
    urllib.request.urlretrieve(GUTENBERG_URL, RAW_PATH)
    print(f"Saved to {RAW_PATH}")


def load_raw_text(path):
    """Load raw UTF-8 text file."""
    print("Loading raw text...")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Raw text length: {len(text):,} characters")
    return text


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg header and footer to keep only literary content."""
    print("Stripping Gutenberg boilerplate...")

    start_marker = "*** START OF"
    end_marker = "*** END OF"

    start_idx = text.find(start_marker)
    if start_idx != -1:
        start_idx = text.find("\n", start_idx) + 1
    else:
        start_idx = 0

    end_idx = text.find(end_marker)
    if end_idx == -1:
        end_idx = len(text)

    text = text[start_idx:end_idx].strip()
    print(f"Text length after stripping: {len(text):,} characters")
    return text


def normalize_text(text):
    """
    Normalize text:
    - Remove BOM
    - Lowercase
    - Normalize newlines and whitespace
    - Normalize fancy quotes/dashes
    """
    print("Normalizing text...")

    text = text.replace("\ufeff", "")
    text = text.lower()
    text = text.replace("\r\n", "\n")
    text = text.replace("\t", " ")

    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2014", "-")
    text = text.replace("\u2026", "...")

    print("Normalization complete.")
    return text


def build_vocab(text):
    """Build character vocabulary from text."""
    print("Building character vocabulary...")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"Vocabulary size: {len(chars)}")
    print(f"Vocabulary preview: {chars[:20]} ...")
    return chars, stoi, itos


def encode_text(text, stoi):
    """Convert text into numpy array of integer token IDs."""
    print("Encoding text to integer tokens...")
    encoded = np.array([stoi[c] for c in text], dtype=np.int64)
    print(f"Encoded data shape: {encoded.shape}")
    return encoded


def split_data(encoded):
    """Create contiguous 80/10/10 split (no shuffling — important for LM)."""
    print("Splitting dataset into train/val/test...")
    n = len(encoded)
    train_end = int(TRAIN_SPLIT * n)
    val_end = int((TRAIN_SPLIT + VAL_SPLIT) * n)

    train_data = encoded[:train_end]
    val_data = encoded[train_end:val_end]
    test_data = encoded[val_end:]

    print(f"Train size: {len(train_data):,}")
    print(f"Val size:   {len(val_data):,}")
    print(f"Test size:  {len(test_data):,}")
    return train_data, val_data, test_data


def save_processed_data(train_data, val_data, test_data, stoi):
    """Save datasets as PyTorch tensors and vocabulary as JSON."""
    print("Saving processed data...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    torch.save(torch.from_numpy(train_data), os.path.join(PROCESSED_DIR, "train.bin"))
    torch.save(torch.from_numpy(val_data), os.path.join(PROCESSED_DIR, "val.bin"))
    torch.save(torch.from_numpy(test_data), os.path.join(PROCESSED_DIR, "test.bin"))

    with open(os.path.join(PROCESSED_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)

    print("Processed files saved to:", PROCESSED_DIR)


def main():
    download_data()
    text = load_raw_text(RAW_PATH)
    text = strip_gutenberg_boilerplate(text)
    text = normalize_text(text)
    chars, stoi, itos = build_vocab(text)
    encoded = encode_text(text, stoi)
    train_data, val_data, test_data = split_data(encoded)
    save_processed_data(train_data, val_data, test_data, stoi)
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

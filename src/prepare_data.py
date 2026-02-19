"""
prepare_data.py

Preprocess the Shakespeare dataset for character-level autoregressive modeling.

Pipeline:
1. Load raw Project Gutenberg text
2. Normalize text (case, whitespace, punctuation)
3. Build character vocabulary
4. Encode text to integer tokens
5. Create contiguous train/val/test split (80/10/10)
6. Save processed binary files + vocabulary

"""

import os
import json
import numpy as np

RAW_PATH = "data/raw/shakespeare.txt"
PROCESSED_DIR = "data/processed"

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # remaining 0.1 goes to test


def load_raw_text(path):
    """Load raw UTF-8 text file."""
    print("Loading raw text...")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Raw text length: {len(text):,} characters")
    return text


def normalize_text(text):
    """
    Normalize text:
    - Lowercase
    - Normalize newlines
    - Replace tabs
    - Normalize quotes/dashes
    - Remove Byte Order Mark
    """

    print("Normalizing text...")

    # Remove BOM (Byte Order Mark)
    text = text.replace("\ufeff", "")

    # Lowercase
    text = text.lower()

    # Normalize Windows newlines to Unix
    text = text.replace("\r\n", "\n")

    # Replace tabs with space
    text = text.replace("\t", " ")

    # Normalize fancy quotes/dashes
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", "-")
    text = text.replace("…", "...")

    print("Normalization complete.")
    return text


def build_vocab(text):
    """
    Build character vocabulary from text.
    Returns:
        chars: sorted list of unique characters
        stoi: dict mapping char -> int
        itos: dict mapping int -> char
    """
    print("Building character vocabulary...")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary preview: {chars[:20]} ...")

    return chars, stoi, itos


def encode_text(text, stoi):
    """
    Convert text into numpy array of integer token IDs.
    """
    print("Encoding text to integer tokens...")
    encoded = np.array([stoi[c] for c in text], dtype=np.uint16)
    print(f"Encoded data shape: {encoded.shape}")
    return encoded


def split_data(encoded): # need to be improved further
    """
    Create contiguous 80/10/10 split.
    No shuffling (important for language modeling).
    """
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
    """
    Save binary datasets and vocabulary to disk.
    """

    print("Saving processed data...")

    # Save binary token arrays
    train_data.tofile(os.path.join(PROCESSED_DIR, "train.bin"))
    val_data.tofile(os.path.join(PROCESSED_DIR, "val.bin"))
    test_data.tofile(os.path.join(PROCESSED_DIR, "test.bin"))

    # Save vocabulary
    with open(os.path.join(PROCESSED_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)

    print("Processed files saved to:", PROCESSED_DIR)


def main():

    # 1. Load raw text
    text = load_raw_text(RAW_PATH)

    # 2. Normalize text
    text = normalize_text(text)

    # 3. Build vocabulary
    chars, stoi, itos = build_vocab(text)

    # 4. Encode text
    encoded = encode_text(text, stoi)

    # 5. Split dataset
    train_data, val_data, test_data = split_data(encoded)

    # 6. Save processed data
    save_processed_data(train_data, val_data, test_data, stoi)

    print("\nPreprocessing complete")

if __name__ == "__main__":
    main()

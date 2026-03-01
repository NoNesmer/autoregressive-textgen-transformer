import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CharDataset
from src.model import CharTransformer

DATA_DIR = "data/processed"


# Utility: Load Config
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Utility: Evaluate
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x)
        B, T, C = logits.shape

        loss = criterion(
            logits.view(B * T, C),
            y.view(B * T)
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Main Training
def main(args):
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load Data
    train_data = torch.load("data/processed/train.bin")
    val_data   = torch.load("data/processed/val.bin")

    train_dataset = CharDataset(DATA_DIR, split="train", block_size=config["block_size"])
    val_dataset   = CharDataset(DATA_DIR, split="val",   block_size=config["block_size"])

    if "subset_size" in config:
        train_data = train_data[: config["subset_size"]]
        val_data = val_data[: config["subset_size"] // 10]  # smaller val set

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    # Model
    model = CharTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        block_size=config["block_size"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=config["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    # OVERFIT TINY MODE
    if args.overfit_tiny:
        print("⚠ Running Overfit Tiny Sanity Check")

        tiny_batch = next(iter(train_loader))
        x_tiny, y_tiny = tiny_batch
        x_tiny = x_tiny.to(device)
        y_tiny = y_tiny.to(device)

        for step in range(500):
            optimizer.zero_grad()

            logits, loss = model(x_tiny, targets=y_tiny)
            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                y_tiny.view(B * T)
            )

            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

        print("Final tiny loss:", loss.item())
        return

    # Normal Training
    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader)

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, loss = model(x, targets=y)
            B, T, C = logits.shape

            loss = criterion(
                logits.view(B * T, C),
                y.view(B * T)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_train_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

        # Save checkpoint
        os.makedirs("results/checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"results/checkpoints/model_epoch_{epoch}.pt"
        )

    # Save Learning Curves
    os.makedirs("results/plots", exist_ok=True)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig("results/plots/training_curve.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml"
    )
    parser.add_argument(
        "--overfit_tiny",
        action="store_true"
    )

    args = parser.parse_args()
    main(args)
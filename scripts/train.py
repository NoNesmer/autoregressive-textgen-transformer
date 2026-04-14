import os
import sys
import argparse
import json
import csv
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CharDataset
from src.model import CharTransformer

DATA_DIR = "data/processed"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vocab_size():
    """Read actual vocab size from vocab.json to avoid config mismatch."""
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    return len(stoi)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer determinism (important for small-model ablations).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, C), y.view(B * T))
        total_loss += loss.item() * (B * T)
        total_tokens += B * T
    return total_loss / total_tokens


def save_checkpoint(path, epoch, model, optimizer, train_loss, val_loss, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }, path)


def main(args):
    config = load_config(args.config)
    if args.epochs_override is not None:
        config["epochs"] = int(args.epochs_override)
    if args.patience_override is not None:
        config["patience"] = int(args.patience_override)
    vocab_size = load_vocab_size()
    config["vocab_size"] = vocab_size
    print(f"Vocab size (from vocab.json): {vocab_size}")

    seed = int(config.get("seed", 42))
    config["seed"] = seed
    seed_everything(seed)
    print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = config.get("run_name", "default")
    ckpt_dir = f"results/checkpoints/{run_name}"
    log_dir = "results/logs"
    plot_dir = "results/plots"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    stride = config.get("stride", None)
    train_dataset = CharDataset(DATA_DIR, "train", config["block_size"], stride=stride)
    val_dataset = CharDataset(DATA_DIR, "val", config["block_size"], stride=stride)

    if "subset_size" in config:
        max_train = min(config["subset_size"], len(train_dataset))
        max_val = min(config["subset_size"] // 10, len(val_dataset))
        train_dataset = Subset(train_dataset, range(max_train))
        val_dataset = Subset(val_dataset, range(max_val))

    # Deterministic DataLoader shuffling across runs.
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=dl_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch:   {len(val_loader)}")

    dropout = config.get("dropout", 0.1)
    model = CharTransformer(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        embed_dim=config["d_model"],
        num_heads=config["n_heads"],
        num_layers=config["n_layers"],
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    patience = config.get("patience", 5)
    epochs = config["epochs"]

    # --- Overfit tiny sanity check ---
    if args.overfit_tiny:
        print("\n=== Overfit Tiny Sanity Check ===")
        tiny_batch = next(iter(train_loader))
        x_tiny, y_tiny = tiny_batch[0].to(device), tiny_batch[1].to(device)
        model.train()
        for step in range(500):
            optimizer.zero_grad()
            logits = model(x_tiny)
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, C), y_tiny.view(B * T))
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")
        print(f"  Final loss: {loss.item():.4f}")
        return

    # --- Normal training loop ---
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = -1

    csv_path = os.path.join(log_dir, f"training_log_{run_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "time_seconds"])

    print(f"\nStarting training: {epochs} epochs, patience={patience}, run='{run_name}'")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_tokens = 0
        epoch_start = time.time()

        lr = optimizer.param_groups[0]["lr"]

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}",
            postfix="-.----",
        )

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, C), y.view(B * T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_tokens = B * T
            total_train_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            loop.postfix = f"{loss.item():.4f}"

        scheduler.step()

        avg_train_loss = total_train_loss / total_tokens
        avg_val_loss = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"\n  Epoch {epoch+1}/{epochs}  |  Train: {avg_train_loss:.4f}  |  "
              f"Val: {avg_val_loss:.4f}  |  LR: {lr:.2e}  |  Time: {elapsed:.0f}s")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{lr:.2e}", f"{elapsed:.1f}"])

        save_checkpoint(
            os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"),
            epoch, model, optimizer, avg_train_loss, avg_val_loss, config,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pt"),
                epoch, model, optimizer, avg_train_loss, avg_val_loss, config,
            )
            print(f"  >> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  >> No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {os.path.join(ckpt_dir, 'best_model.pt')}")

    # Save learning curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title(f"Training Curve — {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(plot_dir, f"training_curve_{run_name}.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CharTransformer")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--overfit_tiny", action="store_true",
                        help="Run sanity check: overfit a single batch")
    parser.add_argument("--epochs_override", type=int, default=None,
                        help="Override config epochs (useful for calibration runs)")
    parser.add_argument("--patience_override", type=int, default=None,
                        help="Override config patience (useful for calibration runs)")
    args = parser.parse_args()
    main(args)

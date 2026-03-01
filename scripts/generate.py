import torch
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CharTransformer

# Paths
DATA_DIR = "data/processed"
CHECKPOINT = "results/checkpoints/model_epoch_2.pt"

# Load vocabulary
with open(os.path.join(DATA_DIR, "vocab.json"), "r", encoding="utf-8") as f:
    stoi = json.load(f)
itos = {int(v): k for k, v in stoi.items()}
vocab_size = len(stoi)
print("Vocabulary size:", vocab_size)

# Initialize model
model = CharTransformer(
    vocab_size=65,  # Must match checkpoint
    embed_dim=64,           # Small model embedding dim
    num_heads=2,
    num_layers=1,
    block_size=128          # Small model block size
)

# Load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()

# Text generation
def generate(prompt, max_new_tokens=100):
    context = [stoi.get(ch, 0) for ch in prompt.lower()]
    context = torch.tensor([context], dtype=torch.long, device=device)
    generated = context

    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[0, -1]
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    return "".join([itos[int(i)] for i in generated[0]])

# Example usage
if __name__ == "__main__":
    prompt = "to"
    generated_text = generate(prompt, max_new_tokens=100)
    print("Generated text:\n", generated_text)
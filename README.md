# Autoregressive Character-Level Transformer

## Project Overview

This repository implements a **character-level autoregressive transformer** for text generation, using the Shakespeare dataset as an example.  
The goal of this baseline implementation is to provide a **working, reproducible pipeline** for training, evaluating, and generating text.

---

## Project Structure


autoregressive-textgen-transformer/
├── configs/
│ └── baseline.yaml # Default configuration
├── data/
│ ├── raw/
│ │ └── shakespeare.txt # Raw text data
│ └── processed/ # Preprocessed binary datasets and vocab
├── results/
│ ├── checkpoints/ # Saved model checkpoints
│ └── plots/ # Training curves
├── scripts/
│ ├── train.py # Training script
│ └── generate.py # Inference / text generation script
├── src/
│ ├── dataset.py # Dataset utilities
│ └── model.py # CharTransformer model implementation
└── README.md


---

## Setup Instructions

1. **Install dependencies** (Python 3.10+ recommended):

```bash
pip install torch numpy matplotlib tqdm pyyaml

Prepare data:

python scripts/prepare_data.py

This script downloads or loads the raw Shakespeare text.

Normalizes and tokenizes the text.

Creates a character-level vocabulary.

Saves processed datasets as binary .bin files and the vocabulary as vocab.json.

Training
Standard Training
python scripts/train.py --config configs/baseline.yaml

Trains the model according to the configuration file.

Saves checkpoints after each epoch in results/checkpoints/.

Computes validation loss at the end of each epoch.

Saves learning curves in results/plots/training_curve.png.

Sanity Check (Overfit Tiny)
python scripts/train.py --config configs/baseline.yaml --overfit_tiny

Runs a small “sanity check” on a tiny batch.

Demonstrates that the model can overfit a small subset.

Useful for testing your pipeline and hyperparameters.

Configuration

Configuration is loaded from a YAML file (configs/baseline.yaml).

Key hyperparameters include:

vocab_size: size of character vocabulary

d_model: embedding dimension

n_heads: number of attention heads

n_layers: number of transformer layers

block_size: sequence length

batch_size: batch size

learning_rate: optimizer learning rate

weight_decay: optimizer weight decay

epochs: number of training epochs

Inference / Generation
python scripts/generate.py

Loads a trained checkpoint (e.g., results/checkpoints/model_epoch_0.pt).

Loads vocabulary from data/processed/vocab.json.

Generates character-level text from a prompt:

prompt = "to be or not to be"
generated_text = generate(prompt, max_new_tokens=200)

Default generation uses greedy decoding.

Can be extended to top-k or temperature sampling.

Evaluation

Training and validation losses are computed after each epoch.

Loss curves are saved as plots for reproducibility.

Sanity checks provide quantitative metrics (loss) and qualitative samples (generated text).

Example sanity check output:

Step 0 | Loss: 4.3397
Step 250 | Loss: 1.1557
Step 450 | Loss: 0.0927
Final tiny loss: 0.0530
Reproducibility Notes

The repository provides clean scripts:

train.py – fully reproducible training pipeline with checkpointing.

generate.py – independent inference pipeline for generating text.

Baseline implementation supports:

CPU-friendly options.

Mini-model training for faster experimentation (~10 min).

Overfit-tiny sanity check for verifying correctness.

All dependencies, data preprocessing, and configuration are documented for reproducibility.

References

GPT-style transformer for character-level language modeling.

PyTorch documentation: https://pytorch.org/docs/stable/index.html

Shakespeare dataset: Project Gutenberg


---
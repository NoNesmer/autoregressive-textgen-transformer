# Autoregressive Character-Level Transformer

## Project Overview

This repository implements a **character-level autoregressive transformer** for text generation, using the Shakespeare dataset as an example.  
The goal of this baseline implementation is to provide a **working, reproducible pipeline** for training, evaluating, and generating text.

---

## Setup Instructions

1. **Install dependencies**

```bash
pip install torch numpy matplotlib tqdm pyyaml
```
2. **Prepare data:**

```bash
python scripts/prepare_data.py
```
This script loads the raw Shakespeare text.

Normalizes and tokenizes the text.

Creates a character-level vocabulary.

Saves processed datasets as binary .bin files and the vocabulary as vocab.json.

3. **Training**
Standard Training

```bash
python scripts/train.py --config configs/baseline.yaml
```
Trains the model according to the configuration file.

Saves checkpoints after each epoch in results/checkpoints/.

Computes validation loss at the end of each epoch.

Saves learning curves in results/plots/training_curve.png.

4. **Baseline model training**

```bash
python scripts/train.py --config configs/mini.yaml
```
mini.yaml configuration for a small model for a baseline

ensure faster training, shows model functionality

5. **Sanity Check (Overfit Tiny)**
```bash
python scripts/train.py --config configs/baseline.yaml --overfit_tiny
```
Runs a small “sanity check” on a tiny batch.

Demonstrates that the model can overfit a small subset.

Useful for testing of pipeline and hyperparameters.

6. **Configuration**

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

7. **Inference / Generation**
```bash
python scripts/generate.py
```

Loads a trained checkpoint (e.g., results/checkpoints/model_epoch_0.pt).

Loads vocabulary from data/processed/vocab.json.

Generates character-level text from a prompt:

prompt = "to be or not to be"
generated_text = generate(prompt, max_new_tokens=100)

Default generation uses greedy decoding.

Will be extended to top-k or temperature sampling.

8. **Evaluation**

Training and validation losses are computed after each epoch.

Loss curves are saved as plots for reproducibility.

Sanity checks provide quantitative metrics (loss) and qualitative samples (generated text).

9. **Conclusion**

The repository provides clean scripts:

train.py – fully reproducible training pipeline with checkpointing.

generate.py – independent inference pipeline for generating text.

Baseline implementation supports:

-CPU-friendly options.

-Mini-model training for faster experimentation (~20 min).

-Overfit-tiny sanity check for verifying correctness.

-All dependencies, data preprocessing, and configuration are documented for reproducibility.


---
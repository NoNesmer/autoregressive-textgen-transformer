import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Self Attention (Causal)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # apply causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


# Feedforward MLP

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, block_size)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# Full Model

class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, block_size)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        assert T <= self.block_size, "Sequence too long"

        positions = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(positions)

        tok_emb = self.token_embedding(x)

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
"""
Jour 6 — Transformer Encoder Block from scratch
=================================================
PyTorch implementation of a single Transformer encoder block:
  MHA -> Add & Norm -> FFN -> Add & Norm

Also includes:
  - Sinusoidal positional encoding
  - A mini "stack" of N blocks
  - Shape tracing through every step

If PyTorch is not installed, the script prints an informative message.
A NumPy fallback is provided for the positional encoding.

Run: python 02-code/06-transformer-architecture.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# PART 1: Sinusoidal Positional Encoding (NumPy — always runnable)
# ============================================================================

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Compute the (max_len, d_model) sinusoidal positional encoding.

      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Intuition: each dim is a sin/cos at a different frequency, so the
    vector at position `pos` is a unique multi-scale "clock reading".
    """
    PE = np.zeros((max_len, d_model))
    # Build positions as a column vector and dim-indices as a row vector
    position = np.arange(max_len)[:, None]  # (max_len, 1)
    # Divisors differ for each pair of (sin, cos) dimensions
    # 2i/d_model for i=0..d_model/2-1
    i = np.arange(d_model // 2)[None, :]    # (1, d_model/2)
    div_term = np.exp(-math.log(10000.0) * (2 * i / d_model))  # (1, d_model/2)

    # angle = pos * 1/10000^(2i/d)
    angle = position * div_term  # (max_len, d_model/2)

    # Even dims = sin, odd dims = cos
    PE[:, 0::2] = np.sin(angle)
    PE[:, 1::2] = np.cos(angle)
    return PE


# Run this immediately — does not need PyTorch
print("=" * 70)
print("PART 1: Sinusoidal Positional Encoding (NumPy, always runs)")
print("=" * 70)

PE = sinusoidal_positional_encoding(max_len=10, d_model=8)
print(f"\nShape: {PE.shape}")
print(f"\nPE[:4, :8] (first 4 positions, all 8 dims):")
for p in range(4):
    row = " ".join(f"{v:+.3f}" for v in PE[p])
    print(f"  pos {p}: {row}")
print("\nObservation: low dims vary fast (high freq), high dims vary slowly.")
print("PE[0] contains sin(0)=0 for even dims and cos(0)=1 for odd dims.")

# Verify the magic property: PE rows have roughly constant norm
norms = np.linalg.norm(PE, axis=-1)
print(f"\nRow norms: min={norms.min():.3f}, max={norms.max():.3f}")
print("(Roughly constant because sin^2 + cos^2 = 1 per pair of dims.)")


# ============================================================================
# PART 2: PyTorch Transformer encoder block
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("\n" + "=" * 70)
    print("PyTorch not installed — skipping PyTorch block.")
    print("Run: pip install torch  to enable the Transformer block demo.")
    print("=" * 70)


if HAS_TORCH:

    class MultiHeadSelfAttention(nn.Module):
        """
        Multi-head self-attention with optional causal mask.
        Using torch's F.scaled_dot_product_attention for the core operation.
        """

        def __init__(self, d_model, n_heads, dropout=0.0):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must divide n_heads"
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            # Three separate projections for Q, K, V
            # WHY separate: lets the model learn distinct roles for each
            self.W_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_K = nn.Linear(d_model, d_model, bias=False)
            self.W_V = nn.Linear(d_model, d_model, bias=False)
            # Output projection mixes information across heads
            self.W_O = nn.Linear(d_model, d_model, bias=False)
            self.dropout = dropout

        def forward(self, x, mask=None):
            """
            x: (batch, seq_len, d_model)
            mask: optional (batch, seq_len, seq_len) or broadcastable
            returns: (batch, seq_len, d_model)
            """
            B, T, D = x.shape

            # Project to Q, K, V then split into heads
            # (B, T, D) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
            q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

            # Let PyTorch's kernel compute attention efficiently
            # is_causal=True applies the triangular mask automatically
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(mask is None and False),  # we handle causal via mask
            )
            # out shape: (B, n_heads, T, d_head)

            # Concat heads back: (B, T, n_heads * d_head) = (B, T, D)
            out = out.transpose(1, 2).contiguous().view(B, T, D)

            return self.W_O(out)

    class FeedForward(nn.Module):
        """
        Position-wise feed-forward network.
          x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
        Applied to each position independently (same weights everywhere).
        """

        def __init__(self, d_model, d_ff, dropout=0.0):
            super().__init__()
            # WHY d_ff = 4 * d_model: standard ratio from "Attention is All You Need".
            # Expands the representation, applies non-linearity, then contracts.
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # GELU is used in GPT/BERT; original Transformer used ReLU. Similar story.
            return self.fc2(self.dropout(F.gelu(self.fc1(x))))

    class TransformerBlock(nn.Module):
        """
        One Transformer encoder block (pre-norm variant, as in GPT-2+).

          x = x + Attention(LN(x))
          x = x + FFN(LN(x))
        """

        def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Pre-norm: normalize BEFORE the sublayer, add residual after
            # This keeps the residual path "pure" and helps train deeper stacks
            x = x + self.drop(self.attn(self.ln1(x), mask=mask))
            x = x + self.drop(self.ffn(self.ln2(x)))
            return x

    class TinyTransformer(nn.Module):
        """
        A minimal stack of N Transformer blocks with token + positional embeddings.
        Outputs the final hidden states (not logits yet — that comes tomorrow).
        """

        def __init__(self, vocab_size, max_len, d_model=64, n_heads=4,
                     d_ff=256, n_layers=2, dropout=0.0):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)

            # Precompute positional encoding as a non-trainable buffer
            pe = torch.tensor(
                sinusoidal_positional_encoding(max_len, d_model),
                dtype=torch.float32
            )
            self.register_buffer('pos_emb', pe)

            # Stack of Transformer blocks
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)

        def forward(self, token_ids, mask=None):
            """token_ids: (B, T)"""
            B, T = token_ids.shape
            # Token embeddings + positional encoding (broadcasted over batch)
            x = self.token_emb(token_ids) + self.pos_emb[:T].unsqueeze(0)

            for block in self.blocks:
                x = block(x, mask=mask)

            return self.ln_f(x)


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":

    if HAS_TORCH:
        print("\n" + "=" * 70)
        print("PART 2: Full Transformer block in PyTorch — shape trace")
        print("=" * 70)

        torch.manual_seed(42)

        # Hyperparameters
        vocab_size = 100
        max_len = 16
        d_model = 64
        n_heads = 4
        d_ff = 256
        n_layers = 2

        print(f"\nConfig:")
        print(f"  vocab_size = {vocab_size}")
        print(f"  max_len    = {max_len}")
        print(f"  d_model    = {d_model}")
        print(f"  n_heads    = {n_heads}")
        print(f"  d_ff       = {d_ff}  (= 4 * d_model)")
        print(f"  n_layers   = {n_layers}")

        model = TinyTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {n_params:,}")

        # Break down by component — eyeball the attention vs FFN split
        attn_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'attn' in name
        )
        ffn_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'ffn' in name
        )
        print(f"  Attention params (across all blocks): {attn_params:,}")
        print(f"  FFN params (across all blocks):       {ffn_params:,}")
        print(f"  FFN/Attention ratio: {ffn_params / attn_params:.2f}x")
        print(f"  Embedding params: {model.token_emb.weight.numel():,}")

        # Dummy input: a batch of 2 sequences, 10 tokens each
        dummy_input = torch.randint(0, vocab_size, (2, 10))
        print(f"\nDummy input shape : {tuple(dummy_input.shape)}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape      : {tuple(output.shape)}")
        print("(batch=2, seq_len=10, d_model=64) — same shape as embedded input")

        # Trace shapes through ONE block for the dummy input
        print("\n--- Shape trace through the first block ---")
        with torch.no_grad():
            x = model.token_emb(dummy_input) + model.pos_emb[:dummy_input.size(1)]
            print(f"  After token + pos embed : {tuple(x.shape)}")
            block = model.blocks[0]
            x1 = block.ln1(x)
            print(f"  After ln1               : {tuple(x1.shape)}")
            a = block.attn(x1)
            print(f"  After self-attention    : {tuple(a.shape)}")
            x = x + a
            print(f"  After residual + attn   : {tuple(x.shape)}")
            x2 = block.ln2(x)
            print(f"  After ln2               : {tuple(x2.shape)}")
            f_out = block.ffn(x2)
            print(f"  After FFN               : {tuple(f_out.shape)}")
            x = x + f_out
            print(f"  After residual + FFN    : {tuple(x.shape)}")
            print("  -> Shape is preserved: we can stack this block N times.")

        # Test causal mask behavior
        print("\n--- Causal mask test ---")
        seq_len = 6
        # Build a boolean causal mask: True where attention is FORBIDDEN
        # F.scaled_dot_product_attention uses -inf additive mask
        causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        print(f"  Causal mask (True = blocked):\n{causal.int().tolist()}")
        tokens = torch.randint(0, vocab_size, (1, seq_len))
        # For demo, convert bool mask to additive
        additive_mask = torch.zeros(seq_len, seq_len)
        additive_mask.masked_fill_(causal, float('-inf'))
        out_causal = model(tokens, mask=additive_mask)
        print(f"  Output with causal mask shape: {tuple(out_causal.shape)}")
        print("  (Each position only depends on positions <= itself)")

        # Print a few weights to prove the model is alive
        print("\n--- Model is alive: sample stats ---")
        print(f"  Output mean : {output.mean().item():+.4f}")
        print(f"  Output std  : {output.std().item():+.4f}")
        print(f"  Output min  : {output.min().item():+.4f}")
        print(f"  Output max  : {output.max().item():+.4f}")

    else:
        # PyTorch not available — still run the NumPy positional encoding demo
        print("\nNote: Transformer block requires PyTorch. NumPy positional")
        print("encoding works — that covered PART 1 above.")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
  A Transformer block is:
    1. MHA → residual → LayerNorm
    2. FFN → residual → LayerNorm

  The shape is preserved (seq_len, d_model) → we can stack N of them.

  Key facts from this demo:
    - FFN has ~2x more parameters than attention (because d_ff = 4*d_model).
    - Pre-norm (LayerNorm BEFORE sublayer) is used in modern models (GPT-2+).
    - The same block is reused for encoder and decoder; the only difference
      is the causal mask applied in decoder self-attention.

  Tomorrow (J7): we put it all together to build a tiny GPT that generates text.
""")

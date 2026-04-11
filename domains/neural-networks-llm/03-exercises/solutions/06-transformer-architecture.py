"""
Solutions — Jour 6 : Transformer Architecture
==============================================
Solutions for the 3 easy exercises.

Run: python 03-exercises/solutions/06-transformer-architecture.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.set_printoptions(precision=4, suppress=True)


# ============================================================================
# EXERCISE 1: Shape tracing through a Transformer block
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Shape trace through a Transformer block")
print("=" * 70)

batch_size = 4
seq_len = 10
d_model = 128
n_heads = 8
d_ff = 512
d_head = d_model // n_heads  # = 16

print(f"""
  Config:
    batch_size = {batch_size}
    seq_len    = {seq_len}
    d_model    = {d_model}
    n_heads    = {n_heads}
    d_ff       = {d_ff}
    d_head     = d_model / n_heads = {d_head}
""")

steps = [
    ("1. Input                         ", (batch_size, seq_len, d_model)),
    ("2. LayerNorm(input)              ", (batch_size, seq_len, d_model)),
    ("3. Linear d_model -> d_model (Q) ", (batch_size, seq_len, d_model)),
    ("   (same for K and V)            ", None),
    ("4. Reshape into heads            ", (batch_size, seq_len, n_heads, d_head)),
    ("5. Transpose (batch, heads, ...) ", (batch_size, n_heads, seq_len, d_head)),
    ("6. Q @ K^T (attention matrix)    ", (batch_size, n_heads, seq_len, seq_len)),
    ("7. softmax(...) @ V              ", (batch_size, n_heads, seq_len, d_head)),
    ("8. Concat heads                  ", (batch_size, seq_len, d_model)),
    ("9. Output projection W_O         ", (batch_size, seq_len, d_model)),
    ("10. Residual + LayerNorm         ", (batch_size, seq_len, d_model)),
    ("11a. FFN Linear d_model -> d_ff  ", (batch_size, seq_len, d_ff)),
    ("11b. GELU                        ", (batch_size, seq_len, d_ff)),
    ("11c. FFN Linear d_ff -> d_model  ", (batch_size, seq_len, d_model)),
    ("12. Residual + LayerNorm final   ", (batch_size, seq_len, d_model)),
]

for label, shape in steps:
    if shape is None:
        print(f"  {label}")
    else:
        print(f"  {label} -> {shape}")

print("""
--- Q13: Why is the final shape identical to the input? ---

  Because we want to STACK multiple blocks. If block N outputs (B, T, d_model),
  then block N+1 can consume it without any adapter.

  This shape-preserving property is what makes the Transformer a "drop-in"
  modular component: you add more layers for more capacity, and nothing else
  in the architecture needs to change.

  It's the same reason ResNet uses shape-preserving residual blocks: stacking
  depth is the main lever for scaling.
""")


# ============================================================================
# EXERCISE 2: Residual connections and gradient flow
# ============================================================================

print("=" * 70)
print("EXERCISE 2: Residual connection — manual calculations")
print("=" * 70)

# Q1: 3 layers of f(x) = 0.5 * x, without residual
print("\n--- Q1: 3 layers, f_i(x) = 0.5 * x, NO residual ---")
x = 1.0
y = x
for _ in range(3):
    y = 0.5 * y
print(f"  y = {y}")
print(f"  y/x = {y/x}")
print(f"  dy/dx = 0.5^3 = {0.5**3}")

# Q2: 100 layers
print("\n--- Q2: 100 layers, NO residual ---")
y100 = 0.5 ** 100
print(f"  y/x = 0.5^100 = {y100:.6e}")
print(f"  dy/dx = {y100:.6e}")
print("  -> VANISHING gradient: signal is ~10^-30, no learning possible.")

# Q3: with residual, 1 layer
print("\n--- Q3: 1 layer WITH residual: y = x + 0.5*x = 1.5*x ---")
print(f"  dy/dx = 1.5")

# Q4: 100 layers with residual
print("\n--- Q4: 100 layers WITH residual (each: x -> x + 0.5*x) ---")
y_res = 1.5 ** 100
print(f"  y/x = 1.5^100 = {y_res:.6e}")
print("  -> Grows (artificial case). With real random weights it's closer to 1.")

# Q5: f_i(x) = 0.001 * x
print("\n--- Q5: f_i(x) = 0.001 * x, 100 layers ---")
no_res = 0.001 ** 100
with_res = (1.001) ** 100
print(f"  No residual : y/x = 0.001^100 = {no_res:.6e}   (completely dead)")
print(f"  Residual    : y/x = 1.001^100 = {with_res:.6f}  (~1.1x, identity preserved)")
print("  -> Residuals let the network learn small corrections without losing signal.")

print("""
--- Q6: Why is "identity as default" useful? ---

  At the start of training, weights are random. The sublayers (attention, FFN)
  produce essentially random noise. Without residuals, this noise is all the
  output has — the input is LOST after one layer.

  With residuals, the output is input + noise. As training progresses, the
  sublayers learn to refine this "input passthrough" with useful corrections.
  The model starts from a usable baseline (identity) and improves from there.

  This is why very deep Transformers (96+ layers) are trainable AT ALL. Without
  residuals, stacking more than ~6 layers quickly becomes unstable.
""")


# ============================================================================
# EXERCISE 3: LayerNorm vs BatchNorm
# ============================================================================

print("=" * 70)
print("EXERCISE 3: LayerNorm vs BatchNorm")
print("=" * 70)

# Build the tensor
X = np.array([
    [[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]],       # batch 0
    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],       # batch 1
    [[10, 20, 30], [0, 0, 0], [0, 0, 0], [0, 0, 0]],    # batch 2
], dtype=float)

print(f"\nInput shape: {X.shape}  (batch, seq, dim)")


def layer_norm_vec(v, eps=1e-5):
    """LayerNorm a single feature vector."""
    mu = np.mean(v)
    sigma = np.std(v)
    return (v - mu) / (sigma + eps), mu, sigma


# Q1: LayerNorm on two different tokens
print("\n--- Q1: LayerNorm ---")
v1 = X[0, 0]  # [1, 2, 3]
normed1, mu1, s1 = layer_norm_vec(v1)
print(f"  Token (batch=0, seq=0) = {v1}")
print(f"    mu    = {mu1:.4f}")
print(f"    sigma = {s1:.4f}")
print(f"    normalized = {normed1}")

v2 = X[2, 0]  # [10, 20, 30]
normed2, mu2, s2 = layer_norm_vec(v2)
print(f"\n  Token (batch=2, seq=0) = {v2}")
print(f"    mu    = {mu2:.4f}")
print(f"    sigma = {s2:.4f}")
print(f"    normalized = {normed2}")

print("\n  Observation: both vectors normalize to the SAME values despite")
print("  very different scales. LayerNorm is scale-invariant per token.")

# Q2: BatchNorm over batch axis for a single position/dim
print("\n--- Q2: BatchNorm ---")
# For (seq=0, dim=0): values across batches = [1.0, 1.0, 10.0]
vals = X[:, 0, 0]  # [1, 1, 10]
mu_bn = np.mean(vals)
sigma_bn = np.std(vals)
print(f"  Position (seq=0, dim=0) across batches: {vals}")
print(f"    mu    = {mu_bn:.4f}")
print(f"    sigma = {sigma_bn:.4f}")
print("  -> The normalized value for batch 2 depends on batches 0 and 1.")
print("     Two batches that happen to be neighbors influence each other.")

# Padding contamination
vals_pad = X[:, 1, 0]  # [4.0, 2.0, 0.0] — includes padding!
print(f"\n  Position (seq=1, dim=0) across batches: {vals_pad}")
print(f"    The 0.0 is a padding token — it SHOULD be ignored but BatchNorm")
print(f"    includes it in the statistics. This corrupts the normalization.")

# Q3: batch_size = 1 at inference
print("\n--- Q3: Inference with batch_size = 1 ---")
print("  BatchNorm computes mu = x and sigma = 0, giving division by zero.")
print("  The standard workaround: during training, BN tracks running averages")
print("  of mu and sigma. At inference, it uses those running stats instead")
print("  of batch stats. This works but introduces a mismatch between")
print("  training and inference behavior — a known source of subtle bugs.")
print("  LayerNorm has no such issue: mu and sigma are computed per token,")
print("  so batch_size doesn't matter.")

# Q4: verdict
print("""
--- Q4: Why LayerNorm wins for Transformers ---

  1. Variable-length sequences + padding: BatchNorm statistics get corrupted
     by padding tokens. LayerNorm normalizes each token independently,
     unaffected by its neighbors or paddings.

  2. batch_size = 1 at inference: autoregressive generation produces one
     token at a time. BatchNorm requires running stats — LayerNorm works
     immediately with any batch size.

  3. No cross-example contamination: in BatchNorm, example A's normalization
     depends on example B. In LayerNorm, each example is self-contained —
     cleaner, more predictable, easier to reason about.

  (Bonus) Modern LLMs (LLaMA, GPT) use RMSNorm, an even simpler variant:
     RMSNorm(x) = x / sqrt(mean(x^2)) * gamma   (no mu, no beta)
  Fewer ops, works just as well.
""")

print("=" * 70)
print("All 3 exercises completed.")
print("=" * 70)

"""
Jour 9 — LLMs modernes: RoPE, RMSNorm, SwiGLU from scratch
===========================================================
PyTorch if available, NumPy fallback otherwise.

Covers:
  1. RoPE — rotary positional embedding, rotation of q/k
  2. RMSNorm — root mean square normalization
  3. SwiGLU FFN — gated linear unit with Swish
  4. Side-by-side comparison with vanilla alternatives

Run: python 02-code/09-llms-modernes-architectures.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)

# Try to import torch. If unavailable, we use NumPy-only implementations.
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[info] PyTorch not available — using NumPy-only fallback.")


# ============================================================================
# PART 1: RoPE — Rotary Positional Embedding (NumPy version)
# ============================================================================

print("=" * 70)
print("PART 1: RoPE — Rotary Positional Embedding")
print("=" * 70)


def precompute_rope_frequencies(head_dim, max_seq_len, base=10000.0):
    """
    Precompute the rotation angles for each (position, frequency pair).

    WHY precompute: the angles depend only on position and head_dim, not on
    the actual query/key values. We can cache them once at model init.

    Returns:
      cos: (max_seq_len, head_dim/2) — cosines of the rotation angles
      sin: (max_seq_len, head_dim/2) — sines of the rotation angles
    """
    # WHY head_dim/2: we rotate 2D pairs (x_2j, x_2j+1), so we need half as
    # many frequencies as dimensions.
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Different frequencies for each pair of dimensions.
    # Pair j has frequency theta_j = 1 / base^(2j / head_dim).
    # Small j -> slow rotation (captures long-range dependencies).
    # Large j -> fast rotation (captures short-range dependencies).
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    # shape: (head_dim / 2,)

    # For each position m, the angle for pair j is m * theta_j.
    positions = np.arange(max_seq_len)  # (max_seq_len,)
    angles = np.outer(positions, freqs)  # (max_seq_len, head_dim/2)

    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply rotary positional embedding to a tensor x.

    Args:
      x: (seq_len, head_dim) — the query or key vector
      cos, sin: (seq_len, head_dim/2) — precomputed rotation angles

    WHY this exact formula: we treat each pair (x_2j, x_2j+1) as a 2D
    vector and rotate it by the angle (m * theta_j). The 2D rotation
    matrix is:
        [cos -sin]
        [sin  cos]
    Applied to (x_a, x_b) gives:
        x_a' = x_a * cos - x_b * sin
        x_b' = x_a * sin + x_b * cos
    """
    # Split x into even-indexed (x_a) and odd-indexed (x_b) pairs.
    x_a = x[:, 0::2]  # even indices: 0, 2, 4, ...
    x_b = x[:, 1::2]  # odd indices: 1, 3, 5, ...

    # Apply the rotation element-wise
    x_a_new = x_a * cos - x_b * sin
    x_b_new = x_a * sin + x_b * cos

    # Interleave them back into the original shape
    result = np.empty_like(x)
    result[:, 0::2] = x_a_new
    result[:, 1::2] = x_b_new
    return result


# Demo: see that q_m . k_n depends only on (n - m)
print("\nDemo RoPE: produit scalaire depend de la distance, pas des positions.")

head_dim = 8
max_seq_len = 16
cos, sin = precompute_rope_frequencies(head_dim, max_seq_len)

# A fixed query and key vector (same content, different positions)
q_base = np.array([1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7])
k_base = np.array([0.6, -0.2, 0.9, 0.1, -0.4, 0.3, 0.5, -0.6])

# Test: positions (m=2, n=5) vs (m=7, n=10) — both have distance 3
for (m, n) in [(2, 5), (7, 10), (0, 3), (12, 15)]:
    q_m = apply_rope(q_base.reshape(1, -1), cos[m:m+1], sin[m:m+1])[0]
    k_n = apply_rope(k_base.reshape(1, -1), cos[n:n+1], sin[n:n+1])[0]
    score = np.dot(q_m, k_n)
    print(f"  m={m:2d}, n={n:2d} (distance {n - m}): q_m . k_n = {score:+.6f}")

print("""
Observation: pour la meme distance (n-m), le produit scalaire est IDENTIQUE.
C'est la propriete cle de RoPE: l'attention encode position RELATIVE,
pas absolue.
""")


# ============================================================================
# PART 2: RMSNorm
# ============================================================================

print("=" * 70)
print("PART 2: RMSNorm vs LayerNorm")
print("=" * 70)


def layer_norm(x, eps=1e-6):
    """
    Classic LayerNorm: (x - mean) / std.
    Computed over the last axis.
    """
    mean = x.mean(axis=-1, keepdims=True)
    # WHY population std (ddof=0): matches PyTorch's default LayerNorm
    std = x.std(axis=-1, keepdims=True, ddof=0)
    return (x - mean) / (std + eps)


def rms_norm(x, eps=1e-6):
    """
    RMSNorm: x / sqrt(mean(x**2)).

    WHY simpler: we drop the centering (mean subtraction). Empirically,
    centering adds no value for language modeling; skipping it is faster,
    uses less memory, and is more stable in float16.
    """
    # Root Mean Square = sqrt of the average of squared values
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True))
    return x / (rms + eps)


# Demo: the output statistics
x = np.random.randn(4, 8) * 3.0 + 1.5  # shifted and scaled input
print(f"\nInput shape: {x.shape}")
print(f"Input mean:  {x.mean():+.3f}, std: {x.std():.3f}")

x_ln = layer_norm(x)
x_rms = rms_norm(x)

print(f"\nAfter LayerNorm:")
print(f"  mean per row: {x_ln.mean(axis=-1).round(4)}")
print(f"  std per row:  {x_ln.std(axis=-1).round(4)}")

print(f"\nAfter RMSNorm:")
print(f"  mean per row: {x_rms.mean(axis=-1).round(4)}")
print(f"  RMS per row:  {np.sqrt((x_rms ** 2).mean(axis=-1)).round(4)}")

print("""
Observation:
- LayerNorm forces mean=0 and std=1 per row.
- RMSNorm forces RMS=1 per row, but the mean can be non-zero.
Both normalize the magnitude; only LayerNorm centers. Empirically,
centering does not help language models.
""")


# ============================================================================
# PART 3: SwiGLU FFN
# ============================================================================

print("=" * 70)
print("PART 3: SwiGLU FFN vs GeLU FFN")
print("=" * 70)


def swish(x):
    """
    Swish activation (also called SiLU): x * sigmoid(x).

    WHY: smoother than ReLU, differentiable everywhere, empirically better
    than ReLU and GeLU when combined with a GLU gate.
    """
    # Use a numerically stable sigmoid to avoid overflow for very negative x
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu(x):
    """GeLU approximation used by GPT-2."""
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def ffn_gelu(x, W1, W2):
    """
    Classic FFN: Linear -> GeLU -> Linear.
    2 weight matrices: W1 (d_model x d_ff), W2 (d_ff x d_model).
    """
    h = x @ W1       # (seq_len, d_ff)
    h = gelu(h)      # activation
    out = h @ W2     # (seq_len, d_model)
    return out


def ffn_swiglu(x, W_gate, W_up, W_down):
    """
    SwiGLU FFN: two input linears (one acts as a gate), then a down projection.

    Formula:
      h = Swish(W_gate @ x) * (W_up @ x)
      out = W_down @ h

    WHY 3 matrices instead of 2: we need two parallel projections
    (gate and value) that we multiply element-wise.

    WHY d_ff = 8/3 * d_model (not 4 * d_model): to keep the same parameter
    count as a classic 2-matrix FFN. 2 * 4 = 8 / 3 * 3 ≈ equivalent params.
    """
    gate = swish(x @ W_gate)  # (seq_len, d_ff)
    up = x @ W_up             # (seq_len, d_ff)
    h = gate * up             # gated hidden — element-wise product
    out = h @ W_down          # (seq_len, d_model)
    return out


# Build both FFNs with matched parameter counts
d_model = 16
d_ff_gelu = 4 * d_model           # classic ratio
d_ff_swiglu = int((8 / 3) * d_model)  # match param count for SwiGLU

W1 = np.random.randn(d_model, d_ff_gelu) * 0.02
W2 = np.random.randn(d_ff_gelu, d_model) * 0.02

W_gate = np.random.randn(d_model, d_ff_swiglu) * 0.02
W_up = np.random.randn(d_model, d_ff_swiglu) * 0.02
W_down = np.random.randn(d_ff_swiglu, d_model) * 0.02

# Parameter counts
params_gelu = W1.size + W2.size
params_swiglu = W_gate.size + W_up.size + W_down.size
print(f"\nGeLU FFN params:   {params_gelu}")
print(f"SwiGLU FFN params: {params_swiglu}")
print(f"Ratio: {params_swiglu / params_gelu:.2f}x")

# Forward pass on a dummy input
x_in = np.random.randn(3, d_model)  # batch of 3 tokens
out_gelu = ffn_gelu(x_in, W1, W2)
out_swiglu = ffn_swiglu(x_in, W_gate, W_up, W_down)

print(f"\nGeLU output shape:   {out_gelu.shape}")
print(f"SwiGLU output shape: {out_swiglu.shape}")
print(f"GeLU output norm:    {np.linalg.norm(out_gelu):.4f}")
print(f"SwiGLU output norm:  {np.linalg.norm(out_swiglu):.4f}")


# ============================================================================
# PART 4: PyTorch versions (if available)
# ============================================================================

if HAS_TORCH:
    print("\n" + "=" * 70)
    print("PART 4: PyTorch implementations")
    print("=" * 70)

    class RMSNormTorch(nn.Module):
        """Production-style RMSNorm, matching LLaMA."""

        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            # Learnable scale gamma, initialized to 1
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            # Cast to float32 for numerical stability, then back
            dtype = x.dtype
            x = x.float()
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            out = (x / rms) * self.weight.float()
            return out.to(dtype)

    class SwiGLUFFN(nn.Module):
        """SwiGLU FFN as used in LLaMA."""

        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            # LLaMA uses 8/3 * dim, rounded to a multiple of 256
            if hidden_dim is None:
                hidden_dim = int(8 / 3 * dim)
            self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
            self.w_up = nn.Linear(dim, hidden_dim, bias=False)
            self.w_down = nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x):
            # SiLU(x) = x * sigmoid(x) — equivalent to Swish
            return self.w_down(torch.nn.functional.silu(self.w_gate(x))
                               * self.w_up(x))

    # Instantiate and test
    dim = 32
    rms = RMSNormTorch(dim)
    ffn = SwiGLUFFN(dim)

    x_t = torch.randn(2, 5, dim)  # (batch, seq, dim)
    print(f"\nInput: {tuple(x_t.shape)}")

    rms_out = rms(x_t)
    ffn_out = ffn(x_t)

    print(f"RMSNorm output: {tuple(rms_out.shape)}")
    print(f"SwiGLU output:  {tuple(ffn_out.shape)}")
    print(f"RMS per token (should be ~1 before scale): "
          f"{rms_out.pow(2).mean(-1).sqrt().mean().item():.4f}")

    # Count parameters
    total_rms = sum(p.numel() for p in rms.parameters())
    total_ffn = sum(p.numel() for p in ffn.parameters())
    print(f"\nRMSNorm params: {total_rms}")
    print(f"SwiGLU params:  {total_ffn}")
else:
    print("\n[info] Skipping PART 4 (no PyTorch).")


print("\n" + "=" * 70)
print("Fin — tu as maintenant implemente RoPE, RMSNorm et SwiGLU.")
print("=" * 70)

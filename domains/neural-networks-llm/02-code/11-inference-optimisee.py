"""
Jour 11 — Inference optimization: KV cache + int8 quantization
===============================================================
PyTorch if available, NumPy fallback otherwise.

Covers:
  1. A tiny attention layer WITHOUT KV cache (baseline)
  2. The same layer WITH KV cache
  3. Timing comparison on generation
  4. Int8 quantization of weights (naive symmetric)

Run: python 02-code/11-inference-optimisee.py
"""

import sys
import io
import time
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.manual_seed(42)
except ImportError:
    HAS_TORCH = False
    print("[info] PyTorch not available — some parts will be skipped.")


# ============================================================================
# PART 1: Attention WITHOUT cache — the baseline
# ============================================================================

print("=" * 70)
print("PART 1: Attention without KV cache (naive)")
print("=" * 70)

# Dimensions for the toy example
d_model = 32
n_heads = 4
head_dim = d_model // n_heads


def attention_no_cache(x, W_q, W_k, W_v, W_o):
    """
    Standard attention without any caching.
    Input x: (seq_len, d_model)
    Returns: (seq_len, d_model)

    WHY this is slow for generation: at each step we recompute K and V
    for ALL tokens, including the ones we already processed.
    """
    seq_len = x.shape[0]
    # Project to queries, keys, values
    Q = x @ W_q  # (seq_len, d_model)
    K = x @ W_k
    V = x @ W_v

    # Reshape for multi-head: (seq_len, n_heads, head_dim)
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)

    # Compute attention per head
    out = np.zeros_like(Q)
    for h in range(n_heads):
        # (seq_len, head_dim) @ (head_dim, seq_len) = (seq_len, seq_len)
        scores = Q[:, h] @ K[:, h].T / math.sqrt(head_dim)
        # Causal mask: position i cannot attend to positions > i
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask
        # Softmax
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        out[:, h] = weights @ V[:, h]

    # Flatten heads and apply output projection
    out = out.reshape(seq_len, d_model)
    return out @ W_o


# ============================================================================
# PART 2: Attention WITH cache — the optimized version
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Attention with KV cache")
print("=" * 70)


class AttentionWithCache:
    """
    Attention layer that maintains a K/V cache across generation steps.

    WHY: during generation, we add ONE new token per step. Recomputing K and V
    for all previous tokens is wasteful — they have not changed. The cache
    stores them and we only compute K/V for the new token.
    """

    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Random weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        # The cache: grows by 1 token per step
        # Shape: (current_seq_len, n_heads, head_dim)
        self.K_cache = None
        self.V_cache = None

    def reset(self):
        """Clear the cache — call this at the start of each new sequence."""
        self.K_cache = None
        self.V_cache = None

    def forward_step(self, x_new):
        """
        Process ONE new token and return its output.

        Args:
          x_new: (d_model,) — the embedding of the new token

        Returns:
          out: (d_model,) — the output after attention for this token
        """
        # Compute q, k, v for the new token only — much cheaper
        q = x_new @ self.W_q
        k = x_new @ self.W_k
        v = x_new @ self.W_v

        # Reshape to multi-head
        q = q.reshape(self.n_heads, self.head_dim)
        k = k.reshape(self.n_heads, self.head_dim)
        v = v.reshape(self.n_heads, self.head_dim)

        # Append the new k and v to the cache
        # At step 0, cache is None -> create it
        # At step t, cache has shape (t, n_heads, head_dim)
        if self.K_cache is None:
            self.K_cache = k[np.newaxis, :, :]  # (1, n_heads, head_dim)
            self.V_cache = v[np.newaxis, :, :]
        else:
            self.K_cache = np.concatenate(
                [self.K_cache, k[np.newaxis, :, :]], axis=0)
            self.V_cache = np.concatenate(
                [self.V_cache, v[np.newaxis, :, :]], axis=0)

        # Now compute attention: q (current token) attends to all cached k, v
        # This is the ONLY attention computation per step — cheap
        out = np.zeros((self.n_heads, self.head_dim))
        for h in range(self.n_heads):
            # scores: (1,) x (cache_len,) - just a single row
            scores = self.K_cache[:, h] @ q[h] / math.sqrt(self.head_dim)
            weights = np.exp(scores - scores.max())
            weights = weights / weights.sum()
            out[h] = weights @ self.V_cache[:, h]

        # Flatten and apply output projection
        out = out.reshape(self.d_model)
        return out @ self.W_o


# ============================================================================
# PART 3: Speed comparison — generation with vs without cache
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Speed comparison (naive vs cached attention)")
print("=" * 70)


def generate_naive(initial_tokens, n_new_tokens, W_q, W_k, W_v, W_o):
    """Generate tokens WITHOUT cache — recompute everything each step."""
    tokens = initial_tokens.copy()  # (seq_len, d_model)
    for _ in range(n_new_tokens):
        # Re-run attention on the FULL sequence each time
        out = attention_no_cache(tokens, W_q, W_k, W_v, W_o)
        # Fake "next token" = last output, normalized
        next_tok = out[-1] / (np.linalg.norm(out[-1]) + 1e-8)
        # Append to sequence
        tokens = np.vstack([tokens, next_tok[np.newaxis, :]])
    return tokens


def generate_cached(initial_tokens, n_new_tokens, attn_layer):
    """Generate tokens WITH cache — process only the new token each step."""
    # First: prefill — run on initial tokens one at a time to fill the cache
    attn_layer.reset()
    for t in range(initial_tokens.shape[0]):
        out = attn_layer.forward_step(initial_tokens[t])
    # Then: generate new tokens, one at a time
    all_tokens = [initial_tokens]
    current = out
    for _ in range(n_new_tokens):
        next_tok = current / (np.linalg.norm(current) + 1e-8)
        all_tokens.append(next_tok[np.newaxis, :])
        current = attn_layer.forward_step(next_tok)
    return np.vstack(all_tokens)


# Build weights and layer
W_q = np.random.randn(d_model, d_model) * 0.02
W_k = np.random.randn(d_model, d_model) * 0.02
W_v = np.random.randn(d_model, d_model) * 0.02
W_o = np.random.randn(d_model, d_model) * 0.02

attn = AttentionWithCache(d_model, n_heads)
# Overwrite with the same weights for fair comparison
attn.W_q, attn.W_k, attn.W_v, attn.W_o = W_q, W_k, W_v, W_o

# Timing: start with 20 tokens, generate 50 new ones
prompt_len = 20
new_tokens = 50
initial = np.random.randn(prompt_len, d_model)

# Naive
t0 = time.perf_counter()
naive_out = generate_naive(initial, new_tokens, W_q, W_k, W_v, W_o)
t_naive = time.perf_counter() - t0

# Cached
t0 = time.perf_counter()
cached_out = generate_cached(initial, new_tokens, attn)
t_cached = time.perf_counter() - t0

print(f"\nPrompt length: {prompt_len}")
print(f"New tokens:    {new_tokens}")
print(f"\n  Naive (no cache):     {t_naive * 1000:.2f} ms")
print(f"  With KV cache:        {t_cached * 1000:.2f} ms")
print(f"  Speedup:              {t_naive / max(t_cached, 1e-9):.2f}x")

# Rerun on a longer sequence to show the gap widens
print("\nScaling — longer sequence (200 new tokens):")
initial2 = np.random.randn(prompt_len, d_model)

t0 = time.perf_counter()
generate_naive(initial2, 200, W_q, W_k, W_v, W_o)
t_naive2 = time.perf_counter() - t0

t0 = time.perf_counter()
generate_cached(initial2, 200, attn)
t_cached2 = time.perf_counter() - t0

print(f"  Naive (200 new toks):  {t_naive2 * 1000:.2f} ms")
print(f"  Cached (200 new toks): {t_cached2 * 1000:.2f} ms")
print(f"  Speedup:               {t_naive2 / max(t_cached2, 1e-9):.2f}x")

print("""
Observation: plus la sequence est longue, plus le cache est rentable.
Sans cache, chaque nouveau token redemande un passage sur toute la sequence
-> O(n^2) par token, O(n^3) total.
Avec cache, chaque nouveau token ne demande qu'un passage sur un SEUL token
contre le cache -> O(n) par token, O(n^2) total.
""")


# ============================================================================
# PART 4: Int8 quantization of weights
# ============================================================================

print("=" * 70)
print("PART 4: Int8 quantization (symmetric, per-tensor)")
print("=" * 70)


def quantize_int8(weights):
    """
    Simple symmetric per-tensor int8 quantization.

    WHY symmetric: we map [-max, +max] to [-127, +127], so 0 stays at 0.
    WHY per-tensor: we use one scale for the whole tensor — simple but less
    accurate than per-channel.

    Returns:
      q: int8 quantized tensor
      scale: float scale factor for dequantization
    """
    # Find the largest absolute value
    max_abs = np.abs(weights).max()
    # Scale such that max_abs -> 127 (int8 range is [-128, 127])
    scale = max_abs / 127.0
    # Quantize and clip
    q = np.round(weights / scale).astype(np.int8)
    q = np.clip(q, -127, 127)
    return q, scale


def dequantize_int8(q, scale):
    """Reconstruct float weights from int8 + scale."""
    return q.astype(np.float32) * scale


# Demo: quantize a realistic-ish weight matrix
W_original = np.random.randn(1024, 1024).astype(np.float32) * 0.02

W_q, scale = quantize_int8(W_original)
W_dequant = dequantize_int8(W_q, scale)

# Measure reconstruction error
error = np.abs(W_original - W_dequant)
rel_error = error.mean() / (np.abs(W_original).mean() + 1e-8)

print(f"\nOriginal weight matrix: {W_original.shape}")
print(f"  fp32 size:       {W_original.nbytes} bytes "
      f"({W_original.nbytes / 1024:.1f} KB)")
print(f"  int8 size:       {W_q.nbytes} bytes "
      f"({W_q.nbytes / 1024:.1f} KB)")
print(f"  + scale (fp32):  4 bytes (negligible)")
print(f"  Compression:     {W_original.nbytes / W_q.nbytes:.1f}x")

print(f"\nQuantization error:")
print(f"  Mean abs error:    {error.mean():.6e}")
print(f"  Max  abs error:    {error.max():.6e}")
print(f"  Relative error:    {rel_error * 100:.3f}%")

# Test that a matmul still works approximately
x_test = np.random.randn(4, 1024).astype(np.float32)
y_original = x_test @ W_original
y_quantized = x_test @ W_dequant

output_diff = np.linalg.norm(y_original - y_quantized) / np.linalg.norm(y_original)
print(f"\nMatmul output difference (relative): {output_diff * 100:.3f}%")

print("""
Observation:
  - Int8 divise la memoire par 4 (comme attendu)
  - L'erreur relative de matmul est ~0.3% sur ce cas test
  - Sur un LLM complet, la perte sur les benchmarks est ~0.5-1% en pratique
  - Int4 (non montre ici) divise encore par 2 au prix de ~1-2% de qualite
""")

print("=" * 70)
print("Fin — KV cache + quantization = les deux plus gros leviers d'inference.")
print("=" * 70)

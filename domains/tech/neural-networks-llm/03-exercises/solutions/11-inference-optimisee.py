"""
Solutions — Jour 11 : Inference optimisee

Run: python 03-exercises/solutions/11-inference-optimisee.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — KV cache speedup
# ============================================================================

print("=" * 70)
print("Exercice 1: KV cache speedup")
print("=" * 70)

d_model = 4096
n_layers = 32


def flops_naive_per_token(t, d, n_layers):
    """Roughly (t+1)^2 * d per layer — recomputes all K, V at each step."""
    return (t + 1) ** 2 * d * n_layers


def flops_cached_per_token(t, d, n_layers):
    """Only (t+1) * d per layer — just one new q, k, v plus attention."""
    return (t + 1) * d * n_layers


print(f"\nConfig: d_model={d_model}, n_layers={n_layers}")
print(f"\n{'seq_len':>10s} {'naive FLOPs':>18s} {'cached FLOPs':>18s} "
      f"{'speedup':>10s}")
print("-" * 62)

for seq in [100, 1000, 10_000]:
    naive = flops_naive_per_token(seq, d_model, n_layers)
    cached = flops_cached_per_token(seq, d_model, n_layers)
    speedup = naive / cached
    print(f"{seq:>10d} {naive:>18.3e} {cached:>18.3e} {speedup:>9.1f}x")

# 4) Cumulative cost over 1000 tokens
# naive: sum(t^2) for t=1..n = n*(n+1)*(2n+1)/6 ≈ n^3/3
# cached: sum(t) for t=1..n = n*(n+1)/2 ≈ n^2/2
n = 1000
naive_total = n * (n + 1) * (2 * n + 1) / 6 * d_model * n_layers
cached_total = n * (n + 1) / 2 * d_model * n_layers
print(f"\n4) Cumulative for {n} tokens:")
print(f"   Naive total:  {naive_total:.3e} FLOPs")
print(f"   Cached total: {cached_total:.3e} FLOPs")
print(f"   Ratio:        {naive_total / cached_total:.1f}x "
      f"(≈ n/1.5 = {n / 1.5:.0f})")

print("""
5) Tradeoff:
   Le cache coute 2 GB de VRAM (pour seq=4k) mais economise ~330x de compute.
   Le compute economise vaut des dizaines de dollars, la memoire coute des
   centimes. Le trade-off est trivialement favorable au cache.
""")


# ============================================================================
# Exercice 2 — Quantization by hand
# ============================================================================

print("=" * 70)
print("Exercice 2: Quantization by hand")
print("=" * 70)

W = np.array([0.1, -0.3, 1.5, -2.0, 0.05, 0.8])

# Int8 quantization
max_abs = np.abs(W).max()
scale_int8 = max_abs / 127
q_int8 = np.round(W / scale_int8).astype(np.int32)  # int32 for display
q_int8 = np.clip(q_int8, -127, 127)
W_dequant_int8 = q_int8.astype(np.float32) * scale_int8

print(f"\nOriginal W: {W}")
print(f"max_abs = {max_abs}")
print(f"scale (int8) = {scale_int8:.6f}")
print(f"Quantized (int8): {q_int8.tolist()}")
print(f"Dequantized: {W_dequant_int8.round(4).tolist()}")

error = np.abs(W - W_dequant_int8)
print(f"\nAbs error: {error.round(6).tolist()}")
print(f"Mean error: {error.mean():.6f}")
print(f"Max error:  {error.max():.6f}")

# 4) Small values precision
print(f"\n4) Element 0.05:")
print(f"   q = round(0.05 / {scale_int8:.4f}) = {round(0.05 / scale_int8)}")
print(f"   W_dequant = {round(0.05 / scale_int8) * scale_int8:.6f}")
print(f"   Relative error: "
      f"{abs(0.05 - round(0.05 / scale_int8) * scale_int8) / 0.05 * 100:.1f}%")
print("   -> les petites valeurs sont tres affectees par une scale globale.")

# 6) Int4
max_int4 = 7
scale_int4 = max_abs / max_int4
q_int4 = np.round(W / scale_int4).astype(np.int32)
q_int4 = np.clip(q_int4, -max_int4, max_int4)
W_dequant_int4 = q_int4.astype(np.float32) * scale_int4
error_int4 = np.abs(W - W_dequant_int4)
print(f"\n6) Int4 (range [-7, 7]):")
print(f"   scale = {scale_int4:.6f}")
print(f"   Quantized: {q_int4.tolist()}")
print(f"   Dequant: {W_dequant_int4.round(4).tolist()}")
print(f"   Mean error: {error_int4.mean():.6f}")
print(f"   -> erreur {error_int4.mean() / error.mean():.1f}x plus grande "
      f"qu'int8.")


# ============================================================================
# Exercice 3 — Inference time estimation
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3: Inference time estimation (LLaMA 2 7B on H100)")
print("=" * 70)

# Constants
H100_FLOPS = 900e12  # 900 TFLOPS fp16
H100_BW = 3e12  # 3 TB/s memory bandwidth

# Model: LLaMA 2 7B fp16
N_params = 7e9
W_bytes = 14e9  # 14 GB
KV_cache_bytes = 1e9  # 1 GB

# 1) Prefill: compute-bound
flops_per_prefill_token = 6 * N_params
prefill_time_per_token = flops_per_prefill_token / H100_FLOPS
prefill_tok_per_s = 1 / prefill_time_per_token
print(f"\n1) Prefill (compute-bound):")
print(f"   FLOPs per token: 6 * 7e9 = {flops_per_prefill_token:.2e}")
print(f"   Time per token:  {prefill_time_per_token * 1e9:.1f} ns")
print(f"   Throughput:      {prefill_tok_per_s:.0f} tokens/sec (theorique)")

# 2) Decode: memory-bound
bytes_per_decode_token = W_bytes + KV_cache_bytes
decode_time_per_token = bytes_per_decode_token / H100_BW
decode_tok_per_s = 1 / decode_time_per_token
print(f"\n2) Decode (memory-bound):")
print(f"   Bytes read per token: {bytes_per_decode_token / 1e9:.1f} GB")
print(f"   Time per token:  {decode_time_per_token * 1000:.2f} ms")
print(f"   Throughput:      {decode_tok_per_s:.0f} tokens/sec")

# 3) Ratio
print(f"\n3) Ratio prefill/decode = "
      f"{prefill_tok_per_s / decode_tok_per_s:.0f}x")
print("   Le prefill est ~100x plus rapide que le decode en tokens/sec.")

# 4) 70B MHA vs GQA
print("\n4) LLaMA 2 70B:")
W_70B = 140e9
cache_70B_mha = 10e9  # MHA
cache_70B_gqa = 1.25e9  # GQA 8x

t_mha = (W_70B + cache_70B_mha) / H100_BW
t_gqa = (W_70B + cache_70B_gqa) / H100_BW

print(f"   MHA: cache {cache_70B_mha / 1e9:.1f} GB + poids 140 GB = "
      f"{(W_70B + cache_70B_mha) / 1e9:.1f} GB")
print(f"     -> {t_mha * 1000:.1f} ms/token, {1 / t_mha:.0f} tok/s")
print(f"   GQA: cache {cache_70B_gqa / 1e9:.2f} GB + poids 140 GB = "
      f"{(W_70B + cache_70B_gqa) / 1e9:.2f} GB")
print(f"     -> {t_gqa * 1000:.1f} ms/token, {1 / t_gqa:.0f} tok/s")
print(f"   Gain GQA: {(t_mha - t_gqa) / t_mha * 100:.0f}%")
print("   (pas enorme sur 70B car les poids dominent; mais GQA permet")
print("    surtout de batcher davantage, ce qui est le vrai gain)")

# 5) Batching
print("\n5) Batching 32 requetes simultanees:")
print("   Les poids (14 GB) ne sont lus QU'UNE FOIS pour tout le batch")
print("   Le cache est batch_size fois plus gros mais reste petit devant")
print(f"   Bytes per step batch=32: 14 GB + 32 * 1 GB = 46 GB")
time_b32 = 46e9 / H100_BW
print(f"   Time per step: {time_b32 * 1000:.1f} ms")
print(f"   Total throughput: 32 tokens / {time_b32 * 1000:.1f} ms = "
      f"{32 / time_b32:.0f} tok/s")
print(f"   -> {(32 / time_b32) / decode_tok_per_s:.1f}x meilleure "
      f"throughput qu'en batch=1")

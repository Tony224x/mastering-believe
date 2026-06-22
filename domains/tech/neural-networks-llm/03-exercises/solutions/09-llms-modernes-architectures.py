"""
Solutions — Jour 9 : Architectures modernes
============================================

Run: python 03-exercises/solutions/09-llms-modernes-architectures.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — RoPE a la main
# ============================================================================

print("=" * 70)
print("Exercice 1: RoPE rotation")
print("=" * 70)


def rotation_matrix(theta):
    """2D rotation matrix for angle theta."""
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta),  math.cos(theta)]])


q = np.array([1.0, 0.0])
k = np.array([1.0, 0.0])

# 1) No rotation
print(f"\n1) q . k sans rotation = {np.dot(q, k):.4f}")

# 2) alpha = beta = pi/4
alpha, beta = math.pi / 4, math.pi / 4
q_prime = rotation_matrix(alpha) @ q
k_prime = rotation_matrix(beta) @ k
dot_1 = np.dot(q_prime, k_prime)
print(f"\n2) alpha = beta = pi/4, difference = 0:")
print(f"   q'  = {q_prime.round(4)}")
print(f"   k'  = {k_prime.round(4)}")
print(f"   q' . k' = {dot_1:.4f}")

# 3) alpha = pi/3, beta = 2pi/3, difference = pi/3
alpha2, beta2 = math.pi / 3, 2 * math.pi / 3
q_second = rotation_matrix(alpha2) @ q
k_second = rotation_matrix(beta2) @ k
dot_2 = np.dot(q_second, k_second)
print(f"\n3) alpha = pi/3, beta = 2pi/3, difference = pi/3:")
print(f"   q'' = {q_second.round(4)}")
print(f"   k'' = {k_second.round(4)}")
print(f"   q'' . k'' = {dot_2:.4f}")

# 4) Compare
print(f"\n4) Comparaison:")
print(f"   q' . k' = {dot_1:.4f} (difference = 0, cos(0) = 1)")
print(f"   q'' . k'' = {dot_2:.4f} (difference = pi/3, cos(pi/3) = 0.5)")
print("   -> le produit scalaire depend SEULEMENT de la difference d'angle.")
print("   Formule: (R(a) q) . (R(b) k) = q . R(b-a) . k")

# 5) Analogy with RoPE
print("\n5) Analogie avec RoPE:")
print("   Si alpha = m*theta_j et beta = n*theta_j,")
print("   alors q_m . k_n depend seulement de (n-m)*theta_j.")
print("   -> L'attention encode la position RELATIVE automatiquement.")


# ============================================================================
# Exercice 2 — KV cache size calculation
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 2: Taille du KV cache")
print("=" * 70)


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_elem=2):
    """
    KV cache size in bytes.

    Formula: 2 (K+V) * n_layers * n_kv_heads * head_dim * seq_len * batch * bpe
    """
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


def fmt_bytes(b):
    """Human-readable byte count."""
    if b < 1024:
        return f"{b} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.2f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024 ** 2:.2f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


models = [
    ("GPT-2 small (MHA)", 12, 12, 12, 64),
    ("LLaMA 2 7B (MHA)", 32, 32, 32, 128),
    ("LLaMA 2 70B (GQA)", 80, 64, 8, 128),
    ("LLaMA 3 70B (GQA)", 80, 64, 8, 128),
]

seq_len = 4096
print(f"\nConfiguration: seq_len={seq_len}, bf16 (2 bytes)")
print(f"{'Model':<25s} {'n_lay':>6s} {'n_h':>5s} {'n_kv':>5s} "
      f"{'hd':>4s}  {'cache b=1':>12s} {'cache b=8':>12s}")
print("-" * 78)

for name, n_layers, n_heads, n_kv_heads, head_dim in models:
    c1 = kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, 1)
    c8 = kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, 8)
    print(f"{name:<25s} {n_layers:>6d} {n_heads:>5d} {n_kv_heads:>5d} "
          f"{head_dim:>4d}  {fmt_bytes(c1):>12s} {fmt_bytes(c8):>12s}")

# Bonus: LLaMA 2 70B MHA vs GQA
print("\n5) Bonus — LLaMA 2 70B: MHA vs GQA (batch=1):")
gqa_cache = kv_cache_bytes(80, 8, 128, 4096, 1)
mha_cache = kv_cache_bytes(80, 64, 128, 4096, 1)
print(f"   GQA (n_kv=8):  {fmt_bytes(gqa_cache)}")
print(f"   MHA (n_kv=64): {fmt_bytes(mha_cache)}")
print(f"   Ratio: {mha_cache / gqa_cache:.0f}x plus avec MHA")

# 6) GPU fit
print("\n6) Peut-on faire tenir LLaMA 2 70B sur une GPU 80 GB ?")
print("   Poids: 70B params * 2 bytes = 140 GB")
print("   Non — une seule GPU ne peut pas. Il faut au moins 2 GPUs (tensor parallel).")
print("   Le KV cache (1.3 GB a seq=4k b=1) est petit devant les poids.")
print("   Sans GQA (cache = 10.7 GB), ca aurait ete impraticable a batch eleve.")


# ============================================================================
# Exercice 3 — SwiGLU parameter matching
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3: SwiGLU parameters")
print("=" * 70)

d_model = 4096

# 1) Classic FFN with GeLU
d_ff_gelu = 4 * d_model
params_gelu = 2 * d_model * d_ff_gelu
print(f"\n1) FFN GeLU (d_ff = 4*d_model = {d_ff_gelu}):")
print(f"   W1: {d_model} x {d_ff_gelu} = {d_model * d_ff_gelu:,} params")
print(f"   W2: {d_ff_gelu} x {d_model} = {d_ff_gelu * d_model:,} params")
print(f"   Total: {params_gelu:,} params ({params_gelu / 1e6:.1f}M)")

# 2) Naive SwiGLU with same d_ff
params_swiglu_naive = 3 * d_model * d_ff_gelu
print(f"\n2) SwiGLU naif (meme d_ff = {d_ff_gelu}):")
print(f"   3 matrices: {params_swiglu_naive:,} params "
      f"({params_swiglu_naive / 1e6:.1f}M)")
print(f"   Ratio: {params_swiglu_naive / params_gelu:.2f}x (50% en plus)")

# 3) Required d_ff to match
d_ff_swiglu_exact = int(round(2 * d_ff_gelu / 3))
params_swiglu_matched = 3 * d_model * d_ff_swiglu_exact
print(f"\n3) Pour matcher, d_ff_swiglu = 2/3 * d_ff_gelu = 8/3 * d_model:")
print(f"   = {d_ff_swiglu_exact} (theorique: {8 * d_model / 3:.1f})")
print(f"   Params avec d_ff = {d_ff_swiglu_exact}: {params_swiglu_matched:,}")

# 4) LLaMA 7B actual value
d_ff_llama = 11008
print(f"\n4) LLaMA 7B utilise d_ff = {d_ff_llama}")
print(f"   8/3 * 4096 = {8 * 4096 / 3:.1f}, arrondi au multiple de 256 "
      f"= {d_ff_llama}")
print("   Pourquoi 256 ? Pour l'alignement memoire GPU (tensor cores aiment)")

# 5) LLaMA 7B layer parameters total
print("\n5) Parametres d'une couche LLaMA 7B:")
# Attention: q, k, v, o projections. LLaMA 7B uses MHA so n_kv = n_heads
n_heads = 32
head_dim = 128
d_q = d_v = d_model
# q, k, v, o are each (d_model, d_model) for MHA
attn_params = 4 * d_model * d_model  # Q, K, V, O
ffn_params = 3 * d_model * d_ff_llama  # SwiGLU
norm_params = 2 * d_model  # 2 RMSNorm gammas
layer_params = attn_params + ffn_params + norm_params
print(f"   Attention (Q, K, V, O): {attn_params:,}")
print(f"   FFN SwiGLU:            {ffn_params:,}")
print(f"   2 RMSNorm:             {norm_params:,}")
print(f"   Total par couche:      {layer_params:,}")

# Total stack
n_layers = 32
total = n_layers * layer_params
print(f"\n   Stack de {n_layers} couches: {total:,} ({total / 1e9:.2f}B)")
print("   (le 7B total = stack + embeddings 32k*4096*2 ~262M + norm final)")

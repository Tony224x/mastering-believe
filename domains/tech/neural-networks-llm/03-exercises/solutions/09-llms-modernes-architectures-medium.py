"""
Solutions MEDIUM — Jour 9 : LLMs modernes (RoPE, RMSNorm, SwiGLU, GQA)
=====================================================================
Exercices 4, 5, 6 (medium). Pur NumPy (miroir de 02-code/09).

4. RoPE : invariance par translation du produit scalaire.
5. RMSNorm vs LayerNorm : stats, cout, gradient.
6. KV-cache GQA/MQA : taille memoire + equivalence GQA(n_kv=n_heads)==MHA.

Run: python 03-exercises/solutions/09-llms-modernes-architectures-medium.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# RoPE (repris de 02-code/09)
# ----------------------------------------------------------------------------

def precompute_rope_frequencies(head_dim, max_seq_len, base=10000.0):
    assert head_dim % 2 == 0
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    positions = np.arange(max_seq_len)
    angles = np.outer(positions, freqs)         # (seq, head_dim/2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """x (seq, head_dim) ; cos/sin (seq, head_dim/2). Rotation des paires (2j,2j+1)."""
    x_a, x_b = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_a * cos - x_b * sin
    out[:, 1::2] = x_a * sin + x_b * cos
    return out


# ============================================================================
# EXERCISE 4: RoPE — invariance par translation
# ============================================================================

print("=" * 70)
print("EXERCISE 4: RoPE — invariance par translation du produit scalaire")
print("=" * 70)

head_dim, max_seq = 8, 64
cos, sin = precompute_rope_frequencies(head_dim, max_seq)
q = np.array([1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.7])
k = np.array([0.6, -0.2, 0.9, 0.1, -0.4, 0.3, 0.5, -0.6])


def rope_score(q, k, m, n):
    qm = apply_rope(q.reshape(1, -1), cos[m:m + 1], sin[m:m + 1])[0]
    kn = apply_rope(k.reshape(1, -1), cos[n:n + 1], sin[n:n + 1])[0]
    return float(qm @ kn)


print("\n  Distance d=3 (positions absolues differentes):")
scores_d3 = [rope_score(q, k, m, n) for (m, n) in [(2, 5), (7, 10), (0, 3), (12, 15)]]
for (m, n), s in zip([(2, 5), (7, 10), (0, 3), (12, 15)], scores_d3):
    print(f"    m={m:2d} n={n:2d}: score = {s:+.6f}")
print(f"  ecart max entre ces scores : {max(scores_d3) - min(scores_d3):.2e}  (< 1e-9)")

# Decroissance moyenne avec la distance (sur vecteurs aleatoires, q==k).
print("\n  Decroissance moyenne du score d'auto-attention avec la distance:")
for d in [0, 1, 4, 16, 32]:
    scs = []
    for _ in range(2000):
        v = np.random.randn(head_dim)
        scs.append(rope_score(v, v, 0, d))
    print(f"    distance {d:>2}: score moyen = {np.mean(scs):+.4f}")

# Equivalence rotation 2D pour la paire 0.
freq0 = 1.0 / (10000.0 ** (0 / head_dim))
m = 5
theta = m * freq0
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
pair = q[:2]
via_matrix = R @ pair
via_rope = apply_rope(q.reshape(1, -1), cos[m:m + 1], sin[m:m + 1])[0][:2]
print(f"\n  Equivalence rotation 2D (paire 0, pos {m}): max diff = {np.max(np.abs(via_matrix - via_rope)):.2e}")

# Effet de la base.
print("\n  Effet de la base sur la longueur d'onde de la frequence la plus lente:")
for base in [10000.0, 500000.0]:
    slow_freq = 1.0 / (base ** ((head_dim - 2) / head_dim))
    period = 2 * np.pi / slow_freq
    print(f"    base={base:>9.0f}: periode (frequence lente) = {period:.1f} positions")
print("  -> base plus grande = longueur d'onde plus longue = couvre un contexte plus large.")


# ============================================================================
# EXERCISE 5: RMSNorm vs LayerNorm — stats, cout, gradient
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: RMSNorm vs LayerNorm")
print("=" * 70)


def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True, ddof=0)
    return (x - mu) / (std + eps)


def rms_norm(x, eps=1e-6):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms


x = np.random.randn(4, 8) * 3.0 + 1.5           # decentre
x_ln, x_rms = layer_norm(x), rms_norm(x)
print(f"\n  Input mean={x.mean():+.3f} std={x.std():.3f}")
print(f"  LayerNorm : mean/ligne ~{np.round(x_ln.mean(-1), 3)}  std/ligne ~{np.round(x_ln.std(-1), 3)}")
print(f"  RMSNorm   : mean/ligne ~{np.round(x_rms.mean(-1), 3)}  RMS/ligne ~{np.round(np.sqrt((x_rms ** 2).mean(-1)), 3)}")
print("  -> LayerNorm centre (mean=0) ; RMSNorm normalise l'amplitude SANS centrer.")

# Cout : LayerNorm fait moyenne + variance + soustraction ; RMSNorm 1 passe.
D = 8
ops_ln = 2 * D + D + D       # moyenne, variance (incl. soustractions), normalisation
ops_rms = D + D             # carres+somme, division
print(f"\n  Cout approx (D={D}): LayerNorm ~{ops_ln} ops, RMSNorm ~{ops_rms} ops")
print(f"  RMSNorm economise ~{(1 - ops_rms / ops_ln) * 100:.0f}% (pas de centrage, 1 passe).")


def rmsnorm_backward(dy, x, eps=1e-6):
    """
    Backward de y = x / r, r = sqrt(mean(x^2)+eps), par ligne (dim D).
      dx = (1/r) * (dy - (x / (D r^2)) * sum(dy*x))
    """
    D = x.shape[-1]
    r = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    sum_dy_x = np.sum(dy * x, axis=-1, keepdims=True)
    return (1.0 / r) * (dy - (x / (D * r ** 2)) * sum_dy_x)


# Gradient check de RMSNorm.
x = np.random.randn(3, 6)
dy = np.random.randn(3, 6)
dx = rmsnorm_backward(dy, x)
eps_fd = 1e-5
gnum = np.zeros_like(x)
it = np.nditer(x, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index; o = x[idx]
    x[idx] = o + eps_fd; lp = np.sum(rms_norm(x) * dy)
    x[idx] = o - eps_fd; lm = np.sum(rms_norm(x) * dy)
    x[idx] = o
    gnum[idx] = (lp - lm) / (2 * eps_fd)
    it.iternext()
rel = np.max(np.abs(dx - gnum) / (np.abs(dx) + np.abs(gnum) + 1e-8))
print(f"\n  Gradient check RMSNorm : max rel err = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")
print("  -> LLaMA/Mistral/Qwen utilisent RMSNorm : meme qualite, moins de calcul, plus stable en fp16.")


# ============================================================================
# EXERCISE 6: KV-cache GQA/MQA — taille memoire + equivalence
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: KV-cache GQA/MQA — taille memoire")
print("=" * 70)


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=2):
    """Taille du KV-cache. Facteur 2 pour K et V."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


# Config LLaMA-2-70B.
n_layers, n_heads, head_dim = 80, 64, 128
seq_len = 4096
print(f"\n  LLaMA-2-70B (n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}, "
      f"seq={seq_len}, fp16):")
mha = kv_cache_bytes(n_layers, n_heads, head_dim, seq_len)
gqa = kv_cache_bytes(n_layers, 8, head_dim, seq_len)
mqa = kv_cache_bytes(n_layers, 1, head_dim, seq_len)
print(f"    MHA (n_kv=64) : {mha / 1e9:7.2f} GB")
print(f"    GQA (n_kv=8)  : {gqa / 1e9:7.2f} GB  (reduction x{mha / gqa:.0f})")
print(f"    MQA (n_kv=1)  : {mqa / 1e9:7.2f} GB  (reduction x{mha / mqa:.0f})")

print("\n  Croissance du cache GQA avec le contexte (lineaire en seq_len):")
for L in [4096, 32768, 131072]:
    sz = kv_cache_bytes(n_layers, 8, head_dim, L)
    print(f"    seq={L:>7}: {sz / 1e9:7.2f} GB")
print("  -> en long contexte, le cache (croit avec seq) depasse les poids (fixes) -> goulot memoire.")


# Equivalence GQA(n_kv=n_heads) == MHA.
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_heads(q, k, v):
    """q,k,v : (n_heads, seq, hd). Attention causale par tete -> (n_heads, seq, hd)."""
    n_h, seq, hd = q.shape
    mask = np.triu(np.ones((seq, seq)), k=1)
    out = np.zeros_like(q)
    for h in range(n_h):
        s = q[h] @ k[h].T / np.sqrt(hd)
        s = np.where(mask == 1, -np.inf, s)
        out[h] = softmax(s, -1) @ v[h]
    return out


seq, hd = 5, 4
n_h, n_kv = 4, 4
q = np.random.randn(n_h, seq, hd)
k = np.random.randn(n_kv, seq, hd)
v = np.random.randn(n_kv, seq, hd)
# GQA repeat : chaque K/V head sert n_rep queries.
n_rep = n_h // n_kv
k_rep = np.repeat(k, n_rep, axis=0)
v_rep = np.repeat(v, n_rep, axis=0)
out_gqa = attention_heads(q, k_rep, v_rep)
out_mha = attention_heads(q, k, v)              # n_kv==n_heads -> deja MHA
print(f"\n  GQA(n_kv=n_heads) == MHA : max diff = {np.max(np.abs(out_gqa - out_mha)):.2e}  (< 1e-12)")
print("  Compromis GQA : moins de K/V distinctes -> moins de cache + plus rapide,")
print("  au prix d'une legere perte de capacite. MQA (n_kv=1) est l'extreme.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 9)")
print("=" * 70)

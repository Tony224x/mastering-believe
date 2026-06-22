"""
Solutions HARD — Jour 11 : Inference optimisee
===============================================
Exercices 7, 8 (hard).

Pur NumPy. Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/11-inference-optimisee-hard.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# EXERCICE 7 : Flash Attention — softmax online (streaming)
# ============================================================================

print("=" * 70)
print("EXERCICE 7 : Flash Attention — softmax online from scratch")
print("=" * 70)


def online_softmax(scores, block_size):
    """
    Softmax 1D calcule par blocs avec running max (m) et running sum (l).
    Renvoie le vecteur softmax SANS jamais materialiser exp de tout le vecteur.
    """
    n = len(scores)
    m = -np.inf       # running max
    l = 0.0           # running denominateur (somme des exp rescalees)
    # On a besoin de deux passes ici car on veut renvoyer le vecteur complet ;
    # le point pedagogique est la mise a jour incrementale de (m, l).
    for start in range(0, n, block_size):
        block = scores[start:start + block_size]
        m_block = block.max()
        m_new = max(m, m_block)
        # WHY exp(m - m_new): rescale l'ancienne somme dans la nouvelle echelle
        l = l * math.exp(m - m_new) + np.exp(block - m_new).sum()
        m = m_new
    # Deuxieme passe : normaliser
    return np.exp(scores - m) / l


s = np.random.randn(50) * 3
ref = np.exp(s - s.max()) / np.exp(s - s.max()).sum()
for bs in [1, 7, 50]:
    online = online_softmax(s, bs)
    print(f"  softmax online (block={bs:>2d}) vs classique : "
          f"ecart={np.max(np.abs(online - ref)):.2e}")


def flash_attention(Q, K, V, block_size, causal=True):
    """
    Flash Attention (un head) par blocs de K/V, avec rescaling online de
    l'accumulateur de sortie O. Memoire de scores: O(block_size) au lieu de O(n).

    Pour chaque ligne i de Q on maintient (m_i, l_i, O_i) et on iterre sur les
    blocs de K/V :
       scores = q_i . K_block / sqrt(d)
       m_new  = max(m_i, max(scores))
       O_i    = O_i * exp(m_i - m_new) + exp(scores - m_new) @ V_block
       l_i    = l_i * exp(m_i - m_new) + sum(exp(scores - m_new))
    A la fin: O_i /= l_i.
    """
    n, d = Q.shape
    nk = K.shape[0]
    O = np.zeros((n, d))
    m = np.full(n, -np.inf)
    l = np.zeros(n)
    scale = 1.0 / math.sqrt(d)

    for ks in range(0, nk, block_size):
        Kb = K[ks:ks + block_size]
        Vb = V[ks:ks + block_size]
        # scores: (n, block) = Q @ Kb.T
        scores = (Q @ Kb.T) * scale
        if causal:
            # ligne i (position i), colonne j (position ks+j) : interdit si ks+j > i
            cols = np.arange(ks, ks + Kb.shape[0])[None, :]
            rows = np.arange(n)[:, None]
            scores = np.where(cols <= rows, scores, -np.inf)
        m_block = scores.max(axis=1)                  # (n,)
        m_new = np.maximum(m, m_block)
        # facteur de rescaling de l'ancien etat
        alpha = np.exp(m - m_new)                      # (n,)
        p = np.exp(scores - m_new[:, None])           # (n, block)
        O = O * alpha[:, None] + p @ Vb
        l = l * alpha + p.sum(axis=1)
        m = m_new
    # eviter division par 0 sur lignes entierement masquees (n'arrive pas ici)
    return O / l[:, None]


def standard_attention(Q, K, V, causal=True):
    n, d = Q.shape
    scores = (Q @ K.T) / math.sqrt(d)
    if causal:
        mask = np.triu(np.full((n, K.shape[0]), -np.inf), k=1)
        scores = scores + mask
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return w @ V


n, d = 16, 8
Q = np.random.randn(n, d)
K = np.random.randn(n, d)
V = np.random.randn(n, d)
ref_attn = standard_attention(Q, K, V)
print("\nFlash Attention vs standard (causale):")
for bs in [1, 4, 16]:
    fa = flash_attention(Q, K, V, bs)
    print(f"  block={bs:>2d} : ecart max = {np.max(np.abs(fa - ref_attn)):.2e}")

# Empreinte memoire (pic) en fonction de n
print("\nMemoire de pic (elements stockes) standard O(n^2) vs flash O(n + block*d):")
block = 4
for nn in [128, 1024, 8192]:
    std = nn * nn               # matrice de scores complete
    flash = nn + block * d + nn  # accumulateurs (O, m, l) + un bloc de scores
    print(f"  n={nn:>5d} : standard={std:.2e}  flash={flash:.2e}  "
          f"ratio={std/flash:.0f}x")

print("""
Analyse:
- Rescaling exp(m_old - m_new) : l'argument est toujours <= 0 (m_new >= m_old),
  donc exp(...) in (0,1] -> jamais d'overflow. C'est ce qui rend l'online softmax stable.
- Flash fait le MEME nombre de FLOPs que l'attention standard (memes produits q.k et p.v).
  Il est plus rapide sur GPU car il ne materialise jamais la matrice (n,n) en HBM : tout
  reste en SRAM (tiling). L'attention etant IO-bound, reduire les transferts HBM = win.
""")


# ============================================================================
# EXERCICE 8 : Roofline de l'inference LLM + KV cache quantize
# ============================================================================

print("=" * 70)
print("EXERCICE 8 : Roofline inference LLM (prefill/decode, GQA, batching)")
print("=" * 70)

PEAK_FLOPS = 900e12   # H100 fp16
MEM_BW = 3e12         # 3 TB/s
RIDGE = PEAK_FLOPS / MEM_BW  # FLOPs/byte au point de bascule
print(f"\nRidge point (FLOPs/byte) = {RIDGE:.0f}")


def regime(intensity):
    return "compute-bound" if intensity > RIDGE else "memory-bound"


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq, batch, bpp):
    return 2 * n_layers * n_kv_heads * head_dim * seq * batch * bpp


# LLaMA 2 7B
N = 7e9
bpp = 2          # fp16
n_layers, n_heads, d_model = 32, 32, 4096
head_dim = d_model // n_heads

# Prefill (P tokens)
P = 512
flops_prefill = 2 * N * P
bytes_prefill = N * bpp           # poids lus une fois
intensity_prefill = flops_prefill / bytes_prefill
tput_prefill = PEAK_FLOPS / (2 * N)
print(f"\nPREFILL (P={P}): intensite={intensity_prefill:.0f} FLOPs/byte "
      f"-> {regime(intensity_prefill)}")
print(f"  throughput ≈ {tput_prefill:.0f} tokens/s")

# Decode (1 token, batch 1)
seq = 4096
kv = kv_cache_bytes(n_layers, n_heads, head_dim, seq, batch=1, bpp=bpp)  # MHA
flops_decode = 2 * N
bytes_decode = N * bpp + kv
intensity_decode = flops_decode / bytes_decode
tput_decode = MEM_BW / bytes_decode
print(f"\nDECODE (seq={seq}, batch=1, MHA): KV cache = {kv/1e9:.2f} GB")
print(f"  intensite={intensity_decode:.1f} FLOPs/byte -> {regime(intensity_decode)}")
print(f"  throughput ≈ {tput_decode:.0f} tokens/s")
print(f"  ratio prefill/decode par token ≈ {tput_prefill/tput_decode:.0f}x")

# Leviers
print("\nLeviers (throughput decode):")
# GQA sur 7B
kv_gqa = kv_cache_bytes(n_layers, 8, head_dim, seq, 1, bpp)  # 8 kv heads
tput_gqa = MEM_BW / (N * bpp + kv_gqa)
print(f"  GQA (8 kv heads) 7B : cache {kv_gqa/1e9:.2f} GB -> "
      f"{tput_gqa:.0f} tok/s (gain {tput_gqa/tput_decode:.2f}x)")
print("    -> faible gain sur 7B : les poids (14 GB) dominent le cache (2 GB).")

# Meme exercice sur 70B (poids 140 GB, cache plus gros)
N70 = 70e9
n_layers70, n_heads70, d70 = 80, 64, 8192
hd70 = d70 // n_heads70
kv70_mha = kv_cache_bytes(n_layers70, n_heads70, hd70, seq, 1, bpp)
kv70_gqa = kv_cache_bytes(n_layers70, 8, hd70, seq, 1, bpp)
tput70_mha = MEM_BW / (N70 * bpp + kv70_mha)
tput70_gqa = MEM_BW / (N70 * bpp + kv70_gqa)
print(f"  70B MHA: cache {kv70_mha/1e9:.1f} GB -> {tput70_mha:.1f} tok/s")
print(f"  70B GQA: cache {kv70_gqa/1e9:.1f} GB -> {tput70_gqa:.1f} tok/s "
      f"(gain {tput70_gqa/tput70_mha:.2f}x) -> GQA aide plus sur les gros modeles/contextes")

# KV cache int8
kv_int8 = kv / 2
tput_int8 = MEM_BW / (N * bpp + kv_int8)
print(f"  KV cache int8 (7B): cache {kv_int8/1e9:.2f} GB -> {tput_int8:.0f} tok/s")

# Batching
print("\nBatching (7B MHA), throughput TOTAL vs batch B:")
for B in [1, 8, 32, 128]:
    kvB = kv_cache_bytes(n_layers, n_heads, head_dim, seq, B, bpp)
    tput_total = B * MEM_BW / (N * bpp + kvB)
    print(f"  B={B:>4d} : {tput_total:>7.0f} tok/s total "
          f"({tput_total/B:.0f} tok/s/requete)")

print("""
Analyse:
- Le batching est le levier #1 du THROUGHPUT serveur : les poids (14 GB) sont lus une
  seule fois et amortis sur B tokens generes. Mais il n'ameliore PAS la latence d'UNE
  requete (chaque requete attend toujours un pass complet). Quand B*kv_bytes devient
  comparable a N*bpp, le throughput total sature (le cache redevient le goulot).
- GQA aide surtout quand le cache pese lourd devant les poids : gros contexte (seq long)
  et/ou gros modele (70B). Sur un 7B a seq=4k, le cache est petit -> gain marginal.
""")

print("=" * 70)
print("Fin solutions hard Jour 11.")
print("=" * 70)

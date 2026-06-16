"""
Solutions MEDIUM — Jour 18 : Long context (Flash Attention, RoPE scaling)
========================================================================
Exercices 4, 5, 6 (medium). Pur NumPy, comme 02-code/18-long-context-attention-scaling.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/18-long-context-attention-scaling-medium.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def softmax(x, axis=-1):
    """Softmax numeriquement stable."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# EXERCISE 4 — Tiled (Flash) attention from scratch
# ============================================================================

print("=" * 70)
print("EXERCISE 4 : tiled (Flash) attention == naive attention")
print("=" * 70)


def naive_attention(Q, K, V):
    """Attention vanilla : materialise la matrice S (N x N). O(N^2) memoire."""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)       # le bottleneck memoire
    P = softmax(S, axis=-1)
    return P @ V


def tiled_attention(Q, K, V, block_size=64):
    """
    Flash Attention v1 simulee en numpy (logique identique a 02-code/18).
    Ne materialise JAMAIS la matrice N x N : on traite Q par blocs (boucle
    externe), K/V par blocs (boucle interne), et on maintient un online softmax.
    Resultat numeriquement EGAL au naive, avec O(N) memoire au lieu de O(N^2).
    """
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(0, N, block_size):
        Qi = Q[i:i + block_size]                  # (Bq, d) bloc de queries
        Oi = np.zeros_like(Qi)                     # accumulateur de sortie
        mi = np.full((Qi.shape[0],), -np.inf)      # running max par query
        li = np.zeros((Qi.shape[0],))              # running denom par query
        for j in range(0, N, block_size):
            Kj = K[j:j + block_size]               # (Bk, d)
            Vj = V[j:j + block_size]               # (Bk, d)
            Sij = Qi @ Kj.T / math.sqrt(d)         # (Bq, Bk) — la tuile, seule en SRAM
            # Online softmax (Tri Dao 2022).
            mij = np.max(Sij, axis=-1)             # max du bloc courant
            mi_new = np.maximum(mi, mij)           # nouveau running max
            # POURQUOI rescale : Oi et li ont ete accumules avec l'ancien max mi.
            # Quand le max change, on corrige l'historique par alpha=exp(mi-mi_new).
            alpha = np.exp(mi - mi_new)
            Pij = np.exp(Sij - mi_new[:, None])    # exp relatif au nouveau max
            li = alpha * li + np.sum(Pij, axis=-1)
            Oi = alpha[:, None] * Oi + Pij @ Vj
            mi = mi_new
        O[i:i + block_size] = Oi / li[:, None]     # normalisation finale du bloc
    return O


N, d = 256, 32
Q = np.random.randn(N, d).astype(np.float64)
K = np.random.randn(N, d).astype(np.float64)
V = np.random.randn(N, d).astype(np.float64)

O_naive = naive_attention(Q, K, V)
O_tiled = tiled_attention(Q, K, V, block_size=64)
max_diff = float(np.max(np.abs(O_naive - O_tiled)))
print(f"\n  N={N}, d={d}")
print(f"  max|O_naive - O_tiled| = {max_diff:.2e}  ({'OK' if max_diff < 1e-4 else 'FAIL'})")
assert max_diff < 1e-4, "Tiled doit egaler naive numeriquement"

# Invariance au block_size : le tiling ne change JAMAIS le resultat.
print("\n  Invariance au block_size :")
for bs in [32, 64, 128]:
    diff = float(np.max(np.abs(O_naive - tiled_attention(Q, K, V, block_size=bs))))
    print(f"    block_size={bs:<4} max|diff|={diff:.2e}  ({'OK' if diff < 1e-4 else 'FAIL'})")
    assert diff < 1e-4

# Pic memoire : vanilla materialise N*N, tiled seulement block^2.
bs = 64
print(f"\n  Pic matrice d'attention (FP16) :")
print(f"    vanilla : N*N*2     = {N * N * 2:>10} octets")
print(f"    tiled   : B*B*2     = {bs * bs * 2:>10} octets (B={bs})")
print("  -> meme sortie, memoire O(N) au lieu de O(N^2). Claim porteur de Flash Attention.")


# ============================================================================
# EXERCISE 5 — Sliding window attention + receptive field
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5 : sliding window attention + receptive field")
print("=" * 70)


def make_sliding_mask(N, W):
    """
    Masque causal a fenetre glissante.
    La position i attend a [max(0, i-W+1), i]. 1 = visible, 0 = masque.
    """
    mask = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        lo = max(0, i - W + 1)
        mask[i, lo:i + 1] = 1.0
    return mask


def attention_with_mask(Q, K, V, mask):
    """Attention naive avec masque additif (-1e9 la ou masque)."""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    S = np.where(mask > 0, S, -1e9)
    P = softmax(S, axis=-1)
    return P, P @ V


N, d, W = 64, 16, 8
Q = np.random.randn(N, d).astype(np.float64)
K = np.random.randn(N, d).astype(np.float64)
V = np.random.randn(N, d).astype(np.float64)

full_mask = np.tril(np.ones((N, N), dtype=np.float64))   # causal full
sw_mask = make_sliding_mask(N, W)

P_full, _ = attention_with_mask(Q, K, V, full_mask)
P_sw, _ = attention_with_mask(Q, K, V, sw_mask)

print(f"\n  N={N}, fenetre W={W}")
print(f"  Full causal : tokens attendus / ligne (moyenne) = {full_mask.sum(axis=1).mean():.1f}")
print(f"  Sliding W={W} : tokens attendus / ligne (moyenne) = {sw_mask.sum(axis=1).mean():.1f}")

# Attention hors-fenetre du dernier token : colonnes 0 .. N-W-1.
last = N - 1
far_full = float(P_full[last, :N - W].sum())
far_sw = float(P_sw[last, :N - W].sum())
print(f"\n  Attention du dernier token (i={last}) vers tokens a > W positions en arriere :")
print(f"    full causal : {far_full:.4f}  (non nul : attend aux tokens anciens)")
print(f"    sliding W   : {far_sw:.2e}  ({'OK' if far_sw < 1e-7 else 'FAIL'} : exactement masque)")
assert far_sw < 1e-7, "L'attention hors-fenetre doit etre nulle en sliding"

# Receptive field cumule a travers L couches.
L_layers, W_mistral = 32, 4096
receptive = L_layers * W_mistral
print(f"\n  Receptive field theorique apres L={L_layers} couches sliding (W={W_mistral}) :")
print(f"    L * W = {L_layers} * {W_mistral} = {receptive} tokens (~{receptive//1000}K)")
print("  -> THEORIQUE : chaque hop de couche dilue l'info. Le receptive field cumule")
print("     n'est PAS equivalent a une attention long-range exacte (full attention).")


# ============================================================================
# EXERCISE 6 — RoPE scaling : NTK-aware et YaRN par bande
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6 : RoPE scaling — PI vs NTK-aware vs YaRN")
print("=" * 70)


def rope_frequencies(d, base=10000.0):
    """theta_i = 1 / base^(2i/d) pour d/2 paires (identique a 02-code/18)."""
    i = np.arange(0, d, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d))


def rope_pi_frequencies(d, base, scale):
    """Position Interpolation : diviser toutes les freqs par scale."""
    return rope_frequencies(d, base) / scale


def rope_ntk_frequencies(d, base, scale):
    """
    NTK-aware : augmenter la BASE au lieu de comprimer les positions.
      new_base = base * scale^(d/(d-2))
    Preserve les hautes freqs (resolution locale), n'etale que les basses.
    """
    new_base = base * (scale ** (d / (d - 2)))
    return rope_frequencies(d, new_base)


def rope_yarn_frequencies(d, base, scale, alpha=1.0, beta=32.0, L_train=4096):
    """
    YaRN : scaling PAR BANDE selon la longueur d'onde.
      haute freq (lambda << L_train) -> garde theta_i (intact)
      basse freq (lambda >> L_train) -> PI (theta_i / scale)
      bande transition              -> rampe lineaire (NTK doux)
    """
    inv_freq = rope_frequencies(d, base)
    inv_freq_pi = inv_freq / scale
    wavelen = 2 * math.pi / inv_freq            # longueur d'onde par dimension
    ratio = L_train / wavelen                    # >1 = haute freq, <1 = basse freq
    ramp = np.clip((ratio - alpha) / (beta - alpha), 0.0, 1.0)
    # ramp=0 (basse freq) -> PI ; ramp=1 (haute freq) -> original.
    return (1 - ramp) * inv_freq_pi + ramp * inv_freq


d, base = 64, 10000.0
L_train, L_target = 4096, 32768
scale = L_target / L_train  # 8.0

f_orig = rope_frequencies(d, base)
f_pi = rope_pi_frequencies(d, base, scale)
f_ntk = rope_ntk_frequencies(d, base, scale)
f_yarn = rope_yarn_frequencies(d, base, scale, L_train=L_train)

new_base = base * (scale ** (d / (d - 2)))
print(f"\n  d={d}, base={base}, L_train={L_train}, L_target={L_target}, scale={scale:.1f}")
print(f"  NTK new_base = base * scale^(d/(d-2)) = {new_base:.1f}")

print(f"\n  {'paire':>6} | {'orig':>11} | {'PI':>11} | {'NTK':>11} | {'YaRN':>11}")
print("  " + "-" * 62)
for k in [0, 4, 8, 16, 24, 30]:
    print(f"  {k:>6} | {f_orig[k]:>11.4e} | {f_pi[k]:>11.4e} | "
          f"{f_ntk[k]:>11.4e} | {f_yarn[k]:>11.4e}")

# Preservation haute freq (k=0) : ratio new/orig (1.0 = intact, 0.125 = ecrase).
print("\n  Preservation HAUTE freq (k=0, ratio new/orig — 1.0 = intact) :")
print(f"    PI   : {f_pi[0]/f_orig[0]:.4f}  (= 1/scale, ecrase la resolution locale)")
print(f"    NTK  : {f_ntk[0]/f_orig[0]:.4f}  (quasi intact)")
print(f"    YaRN : {f_yarn[0]/f_orig[0]:.4f}  (intact en haute freq)")

# Preservation basse freq (derniere paire) : YaRN doit etre PI-like.
print("\n  Comportement BASSE freq (derniere paire, gere le long-range) :")
print(f"    PI   : {f_pi[-1]/f_orig[-1]:.4f}  (= 1/scale = {1/scale:.4f})")
print(f"    NTK  : {f_ntk[-1]/f_orig[-1]:.4f}")
print(f"    YaRN : {f_yarn[-1]/f_orig[-1]:.4f}  (PI-like : comprime le long-range)")

print("\n  -> YaRN garde les hautes freqs intactes (local) ET comprime les basses")
print("     (range) : meilleur compromis qualite long-context (standard 2024-2026).")

print("\nDone (MEDIUM).")

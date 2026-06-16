"""
Solutions EASY — Jour 18 : Long context (Flash Attention, RoPE scaling)
======================================================================
Exercices 1, 2, 3 (faciles). Pur NumPy, comme 02-code/18-long-context-attention-scaling.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/18-long-context-attention-scaling.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


# ============================================================================
# Exercice 1 — Le mur memoire : naive O(N^2) vs tiled O(B^2)
# ============================================================================

print("=" * 70)
print("Exercice 1 : Le mur memoire (naive O(N^2) vs tiled O(B^2))")
print("=" * 70)

# La matrice de scores S est de taille N x N. En FP16 chaque element = 2 octets.
# C'est CETTE matrice que l'attention vanilla doit materialiser en VRAM.
BYTES_FP16 = 2
B = 128  # taille de tuile Flash Attention (tient en SRAM)


def fmt_bytes(n):
    """Formatte des octets en MB/GB/TB de facon lisible."""
    mb = n / (1024 ** 2)
    if mb < 1024:
        return f"{mb:.3f} MB" if mb < 1 else f"{mb:.1f} MB"
    gb = mb / 1024
    if gb < 1024:
        return f"{gb:.2f} GB"
    return f"{gb / 1024:.2f} TB"


print(f"\n  Pic memoire de la matrice d'attention (FP16, 1 head) :\n")
print(f"  {'N':>10} | {'Naive O(N^2)':>14} | {'Tiled O(B^2), B=128':>20}")
print("  " + "-" * 52)
tiled_bytes = B * B * BYTES_FP16  # CONSTANT, independant de N
for N in [1024, 4096, 16384, 100_000, 1_000_000]:
    naive_bytes = N * N * BYTES_FP16
    print(f"  {N:>10} | {fmt_bytes(naive_bytes):>14} | {fmt_bytes(tiled_bytes):>20}")

print(f"\n  Pic tiled (constant) = {tiled_bytes} octets = {fmt_bytes(tiled_bytes)}")
print("  -> independant de N : le tiling change la COMPLEXITE memoire (O(N^2)->O(N)).")

# Ratio a N=100_000 : (N/B)^2.
N = 100_000
ratio = (N / B) ** 2
print(f"\n  Ratio vanilla/tiled a N={N} : (N/B)^2 = ({N}/{B})^2 = {ratio:,.0f}x")

# Cumul heads x couches a N=1M (Llama 70B : ~64 heads, 80 couches).
N = 1_000_000
heads, layers = 64, 80
one_head = N * N * BYTES_FP16
cumul = one_head * heads * layers
print(f"\n  A N={N} (1 head)      : {fmt_bytes(one_head)}")
print(f"  x {heads} heads x {layers} couches : {fmt_bytes(cumul)}")
print("  -> des PETAOCTETS : impossible sur un GPU (H100 = 80 GB). D'ou Flash Attention.")
print("  Note : les FLOPs restent O(N^2*d) — Flash NE reduit PAS le compute, mais")
print("  les transferts memoire HBM<->SRAM (le vrai bottleneck, insight Tri Dao 2022).")


# ============================================================================
# Exercice 2 — Online softmax : fusionner deux blocs
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 2 : Online softmax (fusion streaming de deux blocs)")
print("=" * 70)

S1 = np.array([1.0, 3.0, 2.0], dtype=np.float64)
S2 = np.array([0.5, 4.0, 2.5], dtype=np.float64)

# --- One-shot (reference) ---
# Softmax stable : on soustrait le max global avant l'exponentielle.
S = np.concatenate([S1, S2])
m = float(np.max(S))
p = np.exp(S - m)
l = float(np.sum(p))
softmax_ref = p / l
print(f"\n  One-shot : m={m:.4f}, l={l:.6f}")
print(f"  softmax(S) = {np.round(softmax_ref, 5).tolist()}")

# --- Streaming (Flash-style) ---
# Etape 1 : voir S1 seul.
m1 = float(np.max(S1))
l1 = float(np.sum(np.exp(S1 - m1)))

# Etape 2 : S2 arrive. On fusionne SANS jamais concatener.
m2 = float(np.max(S2))
m_new = max(m1, m2)  # nouveau max global
# POURQUOI le rescale : les exp de S1 avaient ete calculees relativement a m1.
# Pour les rendre coherentes avec le nouveau max m_new, on multiplie la somme
# accumulee par exp(m1 - m_new). Sans ca, on additionnerait des exponentielles
# calculees avec des references de max differentes -> resultat faux.
alpha = math.exp(m1 - m_new)         # facteur de correction de l'ancien bloc
l_new = alpha * l1 + float(np.sum(np.exp(S2 - m_new)))

print(f"\n  Streaming :")
print(f"    apres S1 : m1={m1:.4f}, l1={l1:.6f}")
print(f"    S2 arrive: m2={m2:.4f}, m_new=max(m1,m2)={m_new:.4f}")
print(f"    rescale  : alpha=exp(m1-m_new)={alpha:.6f}")
print(f"    l_new = alpha*l1 + sum(exp(S2-m_new)) = {l_new:.6f}")

# --- Verification d'equivalence exacte ---
err_m = abs(m_new - m)
err_l = abs(l_new - l)
print(f"\n  |m_new - m| = {err_m:.2e}  ({'OK' if err_m < 1e-9 else 'FAIL'})")
print(f"  |l_new - l| = {err_l:.2e}  ({'OK' if err_l < 1e-9 else 'FAIL'})")
assert err_m < 1e-9 and err_l < 1e-9, "L'online softmax doit egaler le one-shot"
print("  -> la fusion streaming reconstruit exactement les statistiques globales.")
print("  C'est le coeur de Flash Attention : softmax exact sans materialiser S.")


# ============================================================================
# Exercice 3 — RoPE inverse frequencies + Position Interpolation
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3 : RoPE frequencies + Position Interpolation (PI)")
print("=" * 70)


def rope_frequencies(d, base=10000.0):
    """
    Frequences inverses RoPE pour d/2 paires.
      theta_i = 1 / base^(2i/d)   pour i = 0, 2, 4, ..., d-2
    (identique a 02-code/18). Haute freq a i=0 (theta=1), basse freq a i=d-2.
    """
    i = np.arange(0, d, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d))


d, base = 64, 10000.0
f_orig = rope_frequencies(d, base)

print(f"\n  d={d}, base={base}, {len(f_orig)} paires de frequences")
print(f"  f_orig[0]  (haute freq) = {f_orig[0]:.6e}")
print(f"  f_orig[-1] (basse freq) = {f_orig[-1]:.6e}")

# Position Interpolation : diviser la position effective par scale
# == diviser toutes les frequences par scale.
L_train, L_target = 4096, 32768
scale = L_target / L_train  # = 8.0
f_pi = f_orig / scale
print(f"\n  PI : L_train={L_train}, L_target={L_target}, scale={scale:.1f}")
print(f"  f_pi[0]  = {f_pi[0]:.6e}")
print(f"  f_pi[-1] = {f_pi[-1]:.6e}")

# Ratio f_pi / f_orig : DOIT etre 1/scale partout (PI est uniforme).
ratio_high = f_pi[0] / f_orig[0]
ratio_low = f_pi[-1] / f_orig[-1]
print(f"\n  Ratio f_pi/f_orig :")
print(f"    haute freq (k=0)   = {ratio_high:.4f}  (= 1/scale = {1/scale:.4f})")
print(f"    basse freq (k=-1)  = {ratio_low:.4f}  (= 1/scale = {1/scale:.4f})")
print("  -> PI comprime TOUTES les freqs du meme facteur, y compris les hautes")
print("     (resolution locale fine) : c'est ca qui degrade la precision locale.")

# Longueurs d'onde : lambda_i = 2*pi / theta_i.
lam_high = 2 * math.pi / f_orig[0]
lam_low = 2 * math.pi / f_orig[-1]
print(f"\n  Longueurs d'onde (orig) :")
print(f"    haute freq : lambda = 2*pi/theta = {lam_high:.2f} tokens (varie vite -> local)")
print(f"    basse freq : lambda = {lam_low:.0f} tokens (varie lentement -> long-range)")
print("  -> hautes freqs = position locale fine ; basses freqs = position globale.")
print("  PI ecrase les deux ; NTK/YaRN (medium) preservent les hautes freqs.")

print("\nDone (EASY).")

"""
Solutions HARD — Jour 8 : Pretraining & Tokenization
====================================================
Exercices 7, 8 (hard). NumPy + stdlib.

7. Entropie de Shannon (ordre 0/1), Huffman, lien LM <-> compression.
8. Estimation des coefficients d'une scaling law (loi de puissance) par regression.

Run: python 03-exercises/solutions/08-pretraining-tokenization-hard.py
"""

import numpy as np
import heapq
from collections import Counter

np.random.seed(42)


# ============================================================================
# EXERCISE 7: Entropie, Huffman, compression
# ============================================================================

print("=" * 70)
print("EXERCISE 7: Entropie de Shannon, Huffman, lien LM <-> compression")
print("=" * 70)

text = ("la pluie tombe sur la ville la nuit la lune brille la mer est calme "
        "la terre tourne la vie continue la pluie revient ") * 12

# --- Entropie d'ordre 0 ---
freqs = Counter(text)
N = len(text)
probs = {c: n / N for c, n in freqs.items()}
H0 = -sum(p * np.log2(p) for p in probs.values())
print(f"\n  Entropie d'ordre 0 : H_0 = {H0:.4f} bits/symbole (vocab = {len(freqs)})")


# --- Huffman from scratch ---
def huffman_code_lengths(freqs):
    """Construit l'arbre de Huffman et renvoie {symbole: longueur de code}."""
    # Tas de (frequence, id_unique, sous-arbre). id_unique departage les egalites.
    heap = [[f, i, c] for i, (c, f) in enumerate(freqs.items())]
    heapq.heapify(heap)
    counter = len(heap)
    # Cas degenere : un seul symbole -> code de longueur 1.
    if len(heap) == 1:
        return {heap[0][2]: 1}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        merged = [lo[0] + hi[0], counter, [lo[2], hi[2]]]
        counter += 1
        heapq.heappush(heap, merged)
    root = heap[0][2]
    lengths = {}

    def walk(node, depth):
        if isinstance(node, list):
            walk(node[0], depth + 1)
            walk(node[1], depth + 1)
        else:
            lengths[node] = depth
    walk(root, 0)
    return lengths


lengths = huffman_code_lengths(freqs)
L_huffman = sum(probs[c] * lengths[c] for c in freqs)
print(f"  Huffman : longueur moyenne L = {L_huffman:.4f} bits/symbole")
print(f"  Encadrement de Shannon : H_0={H0:.4f} <= L={L_huffman:.4f} < H_0+1={H0 + 1:.4f}")
assert H0 - 1e-9 <= L_huffman < H0 + 1 + 1e-9, "encadrement Shannon viole"
print("  -> encadrement verifie : Huffman est optimal a moins d'1 bit pres.")


# --- Entropie d'ordre 1 (conditionnelle) ---
bigram = Counter(zip(text[:-1], text[1:]))
total_bg = sum(bigram.values())
# H(c_t | c_{t-1}) = -sum p(a,b) log2 p(b|a)
ctx_counts = Counter(text[:-1])
H1 = 0.0
for (a, b), n in bigram.items():
    p_ab = n / total_bg
    p_b_given_a = n / ctx_counts[a]
    H1 -= p_ab * np.log2(p_b_given_a)
print(f"\n  Entropie d'ordre 1 : H_1 = {H1:.4f} bits/symbole")
print(f"  H_1 <= H_0 : {H1:.4f} <= {H0:.4f} -> {H1 <= H0 + 1e-9} (gain = {H0 - H1:.4f} bits)")
print("  -> conditionner sur le contexte reduit l'incertitude.")

# Hierarchie uniforme >= H_0 >= bigramme (bits/token).
V = len(freqs)
uniform_bits = np.log2(V)
print(f"\n  Hierarchie (bits/token):")
print(f"    uniforme = log2(V) = {uniform_bits:.4f}")
print(f"    ordre 0  = H_0     = {H0:.4f}")
print(f"    ordre 1  = H_1     = {H1:.4f}")
assert uniform_bits >= H0 >= H1 - 1e-9
print("    -> uniforme >= H_0 >= H_1 : chaque niveau de modele compresse mieux.")

print("\n  LM = compression : la longueur de code optimale d'un symbole de proba p")
print("  est -log2(p) (Shannon). Un LM qui atteint X bits/token peut compresser le")
print("  texte a ~X bits/token (codage arithmetique). Mieux predire = mieux compresser.")


# ============================================================================
# EXERCISE 8: Estimation d'une scaling law par regression log-log
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Estimation d'une scaling law (loi de puissance)")
print("=" * 70)

# Verite terrain : L(N) = E + A / N^alpha
E_true, A_true, alpha_true = 1.7, 400.0, 0.34

# Donnees synthetiques (N en echelle log, bruit multiplicatif ~2%).
N_pts = np.logspace(6, 10, 12)                  # 1e6 .. 1e10
L_clean = E_true + A_true / N_pts ** alpha_true
noise = 1.0 + np.random.randn(len(N_pts)) * 0.02
L_pts = L_clean * noise
print(f"\n  {len(N_pts)} points generes de N=1e6 a N=1e10 (bruit ~2%).")


def linfit(x, y):
    """Regression lineaire moindres carres (formules fermees). Renvoie slope, intercept, R^2."""
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


# Balayage de E : log(L - E) = log(A) - alpha * log(N) est une droite.
logN = np.log(N_pts)
best = None
for E_cand in np.linspace(1.0, 2.0, 101):
    resid = L_pts - E_cand
    if np.any(resid <= 0):
        continue                                # log impossible -> E trop grand
    slope, intercept, r2 = linfit(logN, np.log(resid))
    if best is None or r2 > best['r2']:
        alpha_hat = -slope                      # pente = -alpha
        A_hat = np.exp(intercept)
        best = dict(E=E_cand, alpha=alpha_hat, A=A_hat, r2=r2)

print(f"\n  Coefficients retrouves (vs verite terrain):")
print(f"    E     : {best['E']:.3f}  (vrai {E_true})    err {abs(best['E'] - E_true) / E_true * 100:.1f}%")
print(f"    alpha : {best['alpha']:.3f}  (vrai {alpha_true})   err {abs(best['alpha'] - alpha_true) / alpha_true * 100:.1f}%")
print(f"    A     : {best['A']:.1f}  (vrai {A_true})   err {abs(best['A'] - A_true) / A_true * 100:.1f}%")
print(f"    R^2 du fit log-log : {best['r2']:.4f}")

# Extrapolation a 10x le plus grand point.
N_extra = N_pts.max() * 10
L_pred = best['E'] + best['A'] / N_extra ** best['alpha']
L_true_extra = E_true + A_true / N_extra ** alpha_true
print(f"\n  Extrapolation a N = {N_extra:.2e} (10x le plus gros point d'entrainement):")
print(f"    loss predite : {L_pred:.4f}")
print(f"    loss vraie   : {L_true_extra:.4f}")
print(f"    erreur       : {abs(L_pred - L_true_extra):.4f}")
print("  -> on ajuste sur de PETITS modeles et on extrapole le comportement des GROS.")

print("\n  Loi de PUISSANCE (droite en log-log) et non exponentielle : la loss decroit")
print("  de plus en plus lentement (rendements decroissants). alpha plus grand =")
print("  chaque doublement de N rapporte davantage de reduction de loss.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 8)")
print("=" * 70)

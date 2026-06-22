"""
Solutions EASY — Jour 19 : Quantization
=======================================
Exercices 1, 2, 3 (faciles). Pur NumPy, comme 02-code/19-quantization.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/19-quantization.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def report(name, x, x_hat):
    """MSE + erreur max + MSE relative (lisible a travers les echelles)."""
    err = np.asarray(x) - np.asarray(x_hat)
    mse = float(np.mean(err ** 2))
    max_abs = float(np.max(np.abs(err)))
    rel = mse / float(np.mean(np.asarray(x) ** 2) + 1e-12)
    print(f"  {name:<38s} MSE={mse:.6e}  max|err|={max_abs:.5f}  relMSE={rel:.3%}")


# ============================================================================
# Exercice 1 — Symmetric INT8 / INT4 a la main
# ============================================================================

print("=" * 70)
print("Exercice 1 : Symmetric INT8 / INT4 sur un petit vecteur")
print("=" * 70)

W = np.array([0.12, -0.45, 2.10, -3.00, 0.03, 1.20], dtype=np.float64)


def quantize_symmetric(x, n_bits):
    """
    Quantization symetrique signee.
      qmax = 2^(n_bits-1) - 1   (127 pour INT8, 7 pour INT4)
      scale = max(|x|) / qmax   -> un seul parametre, distribution centree sur 0
      q = clip(round(x / scale), -qmax, qmax)
      x_hat = q * scale
    """
    qmax = (1 << (n_bits - 1)) - 1
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)
    return q, scale


# --- INT8 ---
q8, s8 = quantize_symmetric(W, 8)
W_hat8 = q8 * s8
print(f"\n  W           = {W}")
print(f"  max_abs     = {np.max(np.abs(W)):.4f}")
print(f"  scale INT8  = {s8:.6f}  (= 3.0 / 127)")
print(f"  q (INT8)    = {q8.tolist()}")
print(f"  W_hat (INT8)= {np.round(W_hat8, 4).tolist()}")
report("INT8 symmetric", W, W_hat8)

# Le petit element 0.03 : erreur relative enorme.
# POURQUOI : avec un scale dimensionne sur max_abs=3.0, un pas vaut ~0.0236.
# La valeur 0.03 tombe sur q=1 -> 0.0236, soit ~21% d'erreur relative.
small = W[4]  # 0.03
small_hat = W_hat8[4]
print(f"\n  Petit element 0.03 -> q={q8[4]} -> {small_hat:.4f}, "
      f"erreur relative = {abs(small - small_hat) / abs(small):.1%}")
print("  -> Les petites valeurs souffrent : le scale unique est dimensionne")
print("     sur le max ; il ne reste que quelques codes pour les petites valeurs.")

# --- INT4 ---
q4, s4 = quantize_symmetric(W, 4)
W_hat4 = q4 * s4
print(f"\n  scale INT4  = {s4:.6f}  (= 3.0 / 7)")
print(f"  q (INT4)    = {q4.tolist()}")
report("INT4 symmetric", W, W_hat4)
print("  -> INT4 : chaque pas vaut ~16x plus qu'en INT8 (7 niveaux vs 127),")
print("     donc l'erreur par valeur est ~16x plus grande. C'est le prix des bits.")


# ============================================================================
# Exercice 2 — Symmetric vs asymmetric sur une distribution skewed
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 2 : Symmetric vs asymmetric INT4 (activations post-ReLU)")
print("=" * 70)

A = np.array([0.0, 0.5, 1.2, 2.4, 3.8, 5.0], dtype=np.float64)  # range [0, 5]


def quantize_asymmetric_unsigned(x, n_bits):
    """
    Asymetrique non-signe [0, 2^n - 1] (convention PyTorch quint8 / code J19).
      scale = (max - min) / qmax
      zero_point = clip(round(qmin - min/scale), qmin, qmax)
      q = clip(round(x/scale + zero_point), qmin, qmax)
      x_hat = (q - zero_point) * scale
    """
    qmin, qmax = 0, (1 << n_bits) - 1
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.int64), 1.0, 0
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int64)
    return q, scale, zero_point


# Symmetric INT4 : gaspille toute la moitie negative car A >= 0.
q_sym, s_sym = quantize_symmetric(A, 4)
A_sym_hat = q_sym * s_sym
n_neg_codes = 7  # -7..-1 jamais atteints par des donnees positives
print(f"\n  A             = {A}")
print(f"  Symmetric INT4: scale={s_sym:.4f} (=5/7), q={q_sym.tolist()}")
print(f"    codes negatifs gaspilles : {n_neg_codes} sur 15 (la moitie negative)")
report("symmetric INT4 (skewed)", A, A_sym_hat)

# Asymmetric INT4 : utilise toute la plage [0, 5].
q_asym, s_asym, zp = quantize_asymmetric_unsigned(A, 4)
A_asym_hat = (q_asym - zp) * s_asym
print(f"\n  Asymmetric INT4: scale={s_asym:.4f} (=5/15), zero_point={zp}, "
      f"q={q_asym.tolist()}")
report("asymmetric INT4 (skewed)", A, A_asym_hat)

mse_sym = float(np.mean((A - A_sym_hat) ** 2))
mse_asym = float(np.mean((A - A_asym_hat) ** 2)) + 1e-12
print(f"\n  Gain asymmetric : MSE divisee par ~{mse_sym / mse_asym:.1f}x")
print("  POURQUOI : l'asymetrique mappe le min reel au code 0 et le max au code")
print("  15 -> 16 niveaux sur [0, 5]. Le symetrique n'en utilise que ~8 (cote +).")
print("\n  Pour des POIDS centres sur 0 : symmetric ~= asymmetric, et symmetric")
print("  coute 1 parametre (scale) au lieu de 2 (scale + zero_point) -> on le garde.")


# ============================================================================
# Exercice 3 — Per-tensor vs per-channel
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3 : Per-tensor vs per-channel sur colonnes a echelles mixtes")
print("=" * 70)

M = np.array([
    [0.10, 8.0, 0.02],
    [-0.08, -7.5, 0.03],
], dtype=np.float64)
print(f"\n  M =\n{M}")


def quantize_symmetric_int8_global(x):
    qmax = 127
    scale = float(np.max(np.abs(x))) / qmax
    scale = scale if scale > 0 else 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)
    return q, scale


def quantize_symmetric_int8_per_column(x):
    """Un scale par colonne : max(|colonne|) / 127."""
    qmax = 127
    abs_max = np.max(np.abs(x), axis=0, keepdims=True)  # (1, n_cols)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)
    return q, scale


# Per-tensor : un seul scale dimensionne sur le 8.0 global.
q_pt, s_pt = quantize_symmetric_int8_global(M)
M_pt_hat = q_pt * s_pt
print(f"\n  Per-tensor  : scale global = {s_pt:.5f} (= 8.0/127)")
print(f"    q =\n{q_pt}")
print("    -> col_0 (~0.1) et col_2 (~0.02-0.03) arrondies vers q=0 ou ±1 : ECRASEES")
report("per-tensor INT8", M, M_pt_hat)

# Per-channel : un scale par colonne.
q_pc, s_pc = quantize_symmetric_int8_per_column(M)
M_pc_hat = q_pc * s_pc
print(f"\n  Per-channel : scales = {np.round(s_pc.ravel(), 6).tolist()}")
report("per-channel INT8", M, M_pc_hat)

# Erreur relative sur col_2 (la plus petite).
col2 = M[:, 2]
rel_pt = float(np.mean(np.abs(col2 - M_pt_hat[:, 2]) / (np.abs(col2) + 1e-12)))
rel_pc = float(np.mean(np.abs(col2 - M_pc_hat[:, 2]) / (np.abs(col2) + 1e-12)))
print(f"\n  Erreur relative moyenne sur col_2 : "
      f"per-tensor={rel_pt:.0%}, per-channel={rel_pc:.2%}")

print("\n  Cout memoire (matrice reelle 4096x4096, INT4) :")
print("    per-channel (per-row) : 4096 scales FP16 -> 4096*16 / (4096*4096*4) "
      f"= {4096 * 16 / (4096 * 4096 * 4):.2%} overhead")
print("    per-group g=128       : 4096*32 scales FP16 -> 16 bits / (128*4 bits) "
      f"= {16 / (128 * 4):.2%} overhead")

print("\nDone (EASY).")

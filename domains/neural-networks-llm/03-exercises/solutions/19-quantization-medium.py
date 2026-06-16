"""
Solutions MEDIUM — Jour 19 : Quantization
=========================================
Exercices 4, 5, 6 (medium). Pur NumPy, comme 02-code/19-quantization.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/19-quantization-medium.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def rel_mse(x, x_hat):
    return mse(x, x_hat) / (float(np.mean(np.asarray(x) ** 2)) + 1e-12)


# ============================================================================
# EXERCISE 4 — Quantization par groupe (group-wise) from scratch
# ============================================================================

print("=" * 70)
print("EXERCISE 4 : per-tensor vs per-channel vs per-group INT4")
print("=" * 70)

# Matrice "realiste" : gaussienne, avec 3 colonnes outliers 30x plus grandes.
d_out, d_in = 512, 512
W = np.random.randn(d_out, d_in).astype(np.float64) * 0.5
outlier_cols = np.random.choice(d_in, size=3, replace=False)
W[:, outlier_cols] *= 30.0  # heterogeneite forte (typique d'un LLM)


def quantize_per_tensor(w, n_bits=4):
    """Un seul scale global (symetrique signe)."""
    qmax = (1 << (n_bits - 1)) - 1
    scale = float(np.max(np.abs(w))) / qmax
    scale = scale if scale > 0 else 1.0
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale, 1  # 1 scale


def quantize_per_channel(w, n_bits=4):
    """Un scale par ligne (axis=1 reduit -> scale shape (d_out, 1))."""
    qmax = (1 << (n_bits - 1)) - 1
    abs_max = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale, w.shape[0]  # d_out scales


def quantize_per_group(w, group_size=128, n_bits=4):
    """
    Decoupe CHAQUE ligne en blocs de group_size, un scale par bloc.
    Gere le padding si d_in n'est pas divisible.
    POURQUOI per-group : un scale local par bloc empeche un outlier d'une
    colonne d'ecraser tout le reste de la ligne.
    """
    qmax = (1 << (n_bits - 1)) - 1
    n_rows, n_cols = w.shape
    pad = (-n_cols) % group_size
    if pad:
        w_p = np.concatenate([w, np.zeros((n_rows, pad))], axis=1)
    else:
        w_p = w
    n_groups = w_p.shape[1] // group_size
    # Reshape en (n_rows, n_groups, group_size) -> scale par (row, group).
    blocks = w_p.reshape(n_rows, n_groups, group_size)
    abs_max = np.max(np.abs(blocks), axis=2, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)
    q = np.clip(np.round(blocks / scale), -qmax, qmax)
    deq = (q * scale).reshape(n_rows, -1)[:, :n_cols]
    n_scales = n_rows * n_groups
    return deq, n_scales


def overhead_bits_per_weight(n_scales, n_weights, scale_bits=16):
    """Bits/poids ajoutes par les scales FP16."""
    return n_scales * scale_bits / n_weights


W_pt, ns_pt = quantize_per_tensor(W)
W_pc, ns_pc = quantize_per_channel(W)
W_pg, ns_pg = quantize_per_group(W, group_size=128)

n_w = W.size
print(f"\n  Matrice {W.shape}, 3 colonnes outliers x30")
print(f"  {'granularite':<22} {'relMSE':<12} {'overhead bits/poids':<20}")
print("  " + "-" * 54)
print(f"  {'per-tensor':<22} {rel_mse(W, W_pt):<12.4%} "
      f"{overhead_bits_per_weight(ns_pt, n_w):<20.5f}")
print(f"  {'per-channel (per-row)':<22} {rel_mse(W, W_pc):<12.4%} "
      f"{overhead_bits_per_weight(ns_pc, n_w):<20.5f}")
print(f"  {'per-group g=128':<22} {rel_mse(W, W_pg):<12.4%} "
      f"{overhead_bits_per_weight(ns_pg, n_w):<20.5f}")
print("\n  -> per-group bat per-channel : les outliers sont confines a leur bloc")
print("     de 128 au lieu de polluer toute la ligne.")

print("\n  Balayage group_size (relMSE vs overhead) :")
for gs in [32, 64, 128, 256]:
    deq, ns = quantize_per_group(W, group_size=gs)
    print(f"    g={gs:<4} relMSE={rel_mse(W, deq):.4%}  "
          f"overhead={overhead_bits_per_weight(ns, n_w):.4f} bits/poids")
print("  -> coude vers g=128 : en dessous, overhead grimpe sans gain MSE majeur.")


# ============================================================================
# EXERCISE 5 — Outliers + migration SmoothQuant-style
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5 : outliers et migration SmoothQuant (X@W = (X/s)@(s*W))")
print("=" * 70)

n = 256
X = np.random.randn(n, n).astype(np.float64) * 0.5
# Injecter 0.1% d'outliers de magnitude 50.
n_out = max(1, int(0.001 * X.size))
idx = np.random.choice(X.size, size=n_out, replace=False)
flat = X.flatten()
outlier_mask_flat = np.zeros(X.size, dtype=bool)
outlier_mask_flat[idx] = True
flat[idx] = np.sign(np.random.randn(n_out)) * 50.0
X = flat.reshape(n, n)
normal_mask = ~outlier_mask_flat.reshape(n, n)  # True = entree normale


def quantize_sym_per_tensor_int8(x):
    qmax = 127
    scale = float(np.max(np.abs(x))) / qmax
    scale = scale if scale > 0 else 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax)
    return q * scale


def mse_normal_only(x, x_hat, mask):
    """MSE sur les entrees NORMALES seulement (la qualite qui compte)."""
    diff = (x - x_hat)[mask]
    return float(np.mean(diff ** 2))


# Baseline naif.
X_naive = quantize_sym_per_tensor_int8(X)
mse_naive = mse_normal_only(X, X_naive, normal_mask)

# SmoothQuant-like : s_j = max(|X[:,j]|)^alpha, X_smoothed = X / s.
alpha = 0.5
col_max = np.max(np.abs(X), axis=0, keepdims=True)
col_max = np.where(col_max == 0, 1.0, col_max)
s = np.power(col_max, alpha)
X_smoothed = X / s
X_sm_hat = quantize_sym_per_tensor_int8(X_smoothed) * s  # un-do migration
mse_sm = mse_normal_only(X, X_sm_hat, normal_mask)

print(f"\n  #outliers = {n_out} (~0.1%), magnitude 50, std normal ~0.5")
print(f"  MSE (entrees normales) naif         = {mse_naive:.6e}")
print(f"  MSE (entrees normales) SmoothQuant  = {mse_sm:.6e}")
print(f"  -> reduction x{mse_naive / (mse_sm + 1e-18):.0f} sur les 99.9% normaux")

# Verification d'invariance : (X/s) @ (s*W) == X @ W.
Wmat = np.random.randn(n, n)
lhs = (X / s.T) @ (s.T * Wmat)  # s broadcast sur les lignes de W (canaux d'entree)
# Attention : s est per-colonne de X = per-ligne de W. On verifie proprement :
s_col = s.ravel()  # (n,) un scale par canal d'entree
lhs = (X / s_col[None, :]) @ (s_col[:, None] * Wmat)
rhs = X @ Wmat
inv_err = float(np.max(np.abs(lhs - rhs)))
print(f"\n  Invariance |(X/s)@(s*W) - X@W| max = {inv_err:.2e}  "
      f"({'OK' if inv_err < 1e-4 else 'FAIL'})")
print("  -> la migration est gratuite : le scale se compense exactement.")

# Balayage alpha : erreur conjointe quand X ET W sont quantizes.
print("\n  Balayage alpha (erreur conjointe X et W quantizes) :")
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    s_a = np.power(col_max, a).ravel()
    s_a = np.where(s_a == 0, 1.0, s_a)
    Xs = X / s_a[None, :]
    Ws = s_a[:, None] * Wmat
    Xs_q = quantize_sym_per_tensor_int8(Xs)
    Ws_q = quantize_sym_per_tensor_int8(Ws)
    y_true = X @ Wmat
    y_q = (Xs_q) @ (Ws_q)
    print(f"    alpha={a:<4} MSE(sortie) = {mse(y_true, y_q):.4e}")
print("  -> alpha=0 (naif) et alpha=1 (tout sur W) sont sous-optimaux ;")
print("     un midpoint (~0.5) equilibre la difficulte entre X et W.")


# ============================================================================
# EXERCISE 6 — NF4 from scratch vs INT4 lineaire
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6 : NF4 (NormalFloat 4-bit) vs INT4 lineaire per-block")
print("=" * 70)


def erfinv_approx(y):
    """
    Inverse de erf (Winitzki 2008). Precis a ~1e-3, suffisant pour placer
    16 niveaux de codebook. POURQUOI cette approx : ni la stdlib ni numpy
    n'exposent erfinv (scipy.special.erfinv le ferait exactement, mais on
    reste dependency-free).
    """
    y = np.asarray(y, dtype=np.float64)
    a = 0.147
    ln = np.log(1.0 - y * y)
    first = 2.0 / (np.pi * a) + ln / 2.0
    inside = first * first - ln / a
    return np.sign(y) * np.sqrt(np.sqrt(inside) - first)


def build_nf4_codebook():
    """16 quantiles de N(0,1) : x = sqrt(2)*erfinv(2p-1), p = (i+0.5)/16."""
    probs = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    levels = np.sqrt(2.0) * erfinv_approx(2.0 * probs - 1.0)
    levels = levels / np.max(np.abs(levels))  # normaliser a [-1, 1]
    return levels


NF4 = build_nf4_codebook()
print(f"\n  Codebook NF4 (16 niveaux) :")
print("   ", np.round(NF4, 3).tolist())
# Densite : ecart entre niveaux centraux vs niveaux de queue.
gap_center = NF4[8] - NF4[7]
gap_tail = NF4[-1] - NF4[-2]
print(f"  Ecart central |NF4[8]-NF4[7]| = {abs(gap_center):.3f}, "
      f"ecart queue |NF4[-1]-NF4[-2]| = {abs(gap_tail):.3f}")
print("  -> niveaux DENSES pres de 0, ESPACES dans les queues (= optimal gaussien).")


def quantize_codebook_perblock(x, codebook, block=64):
    """Quantize x par bloc en snappant au niveau le plus proche du codebook."""
    flat = x.flatten()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    abs_max = np.max(np.abs(blocks), axis=1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scaled = blocks / abs_max  # dans [-1, 1]
    # argmin de |scaled - level| sur le codebook.
    diffs = np.abs(scaled[..., None] - codebook[None, None, :])
    codes = np.argmin(diffs, axis=-1)
    deq = (codebook[codes] * abs_max).flatten()
    if pad:
        deq = deq[:-pad]
    return deq.reshape(x.shape)


# INT4 lineaire = codebook {-7..7}/7.
INT4_LINEAR = (np.arange(-7, 8, dtype=np.float64)) / 7.0  # 15 niveaux symetriques

# Poids gaussiens.
Wg = np.random.randn(1024, 1024).astype(np.float64)
Wg_nf4 = quantize_codebook_perblock(Wg, NF4, block=64)
Wg_int4 = quantize_codebook_perblock(Wg, INT4_LINEAR, block=64)
print(f"\n  Poids GAUSSIENS (1024x1024), per-block b=64 :")
print(f"    MSE INT4 lineaire = {mse(Wg, Wg_int4):.6e}")
print(f"    MSE NF4           = {mse(Wg, Wg_nf4):.6e}")
print(f"    -> NF4 reduit la MSE de {1 - mse(Wg, Wg_nf4) / mse(Wg, Wg_int4):.0%}")

# Distribution UNIFORME : NF4 perd son avantage.
Wu = np.random.uniform(-1, 1, size=(1024, 1024))
Wu_nf4 = quantize_codebook_perblock(Wu, NF4, block=64)
Wu_int4 = quantize_codebook_perblock(Wu, INT4_LINEAR, block=64)
print(f"\n  Poids UNIFORMES U(-1,1) :")
print(f"    MSE INT4 lineaire = {mse(Wu, Wu_int4):.6e}")
print(f"    MSE NF4           = {mse(Wu, Wu_nf4):.6e}")
print("    -> sur de l'uniforme, INT4 lineaire (niveaux equirepartis) gagne :")
print("       NF4 est optimise pour le gaussien, pas pour l'uniforme.")

print("\nDone (MEDIUM).")

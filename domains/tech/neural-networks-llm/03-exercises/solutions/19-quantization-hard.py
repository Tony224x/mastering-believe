"""
Solutions HARD — Jour 19 : Quantization
=======================================
Exercices 7, 8, 9 (hard). Pur NumPy, comme 02-code/19-quantization.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/19-quantization-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ============================================================================
# EXERCISE 7 — GPTQ-lite (error compensation via Hessienne)
# ============================================================================

print("=" * 70)
print("EXERCISE 7 : GPTQ-lite vs RTN (compensation d'erreur via Hessienne)")
print("=" * 70)

d_out, d_in = 64, 256
n_calib = 512
W_true = np.random.randn(d_out, d_in).astype(np.float64) * 0.3
X = np.random.randn(n_calib, d_in).astype(np.float64)
Y = X @ W_true.T  # sortie de reference (FP)


def rtn_quantize_int4_per_channel(w):
    """Round-to-nearest INT4 symetrique per-channel (par ligne de w)."""
    qmax = 7
    abs_max = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale


def quantize_scalar_to_grid(value, scale, qmax=7):
    """Quantize une seule valeur sur la grille INT4 definie par scale."""
    q = np.clip(np.round(value / scale), -qmax, qmax)
    return q * scale


def gptq_lite(w, H_inv, act_order=False):
    """
    GPTQ-lite : quantize colonne par colonne, compense l'erreur sur les
    colonnes restantes via H_inv (regle OBS).

    POURQUOI : la sortie Y = X @ W.T depend de la COMBINAISON des colonnes
    ponderee par X. Quand on quantize la colonne j (erreur e_j), on peut
    ajuster les colonnes suivantes pour annuler une partie de l'effet sur Y.
    La direction optimale (OBS/OBQ) est donnee par H_inv[j, k]/H_inv[j, j].
    """
    w = w.copy()
    n = w.shape[1]
    # Scale per-channel fige (calcule une fois sur w original, comme GPTQ).
    qmax = 7
    abs_max = np.max(np.abs(w), axis=1, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)  # (d_out, 1)

    order = np.arange(n)
    if act_order:
        # act-order : traiter les colonnes par importance decroissante.
        order = np.argsort(-np.diag(np.linalg.inv(H_inv)))  # diag(H) decroissant
    processed = []
    for j in order:
        # Quantize la colonne j.
        col_q = quantize_scalar_to_grid(w[:, j], scale[:, 0], qmax)
        err = w[:, j] - col_q  # (d_out,) erreur de quantization de la colonne
        w[:, j] = col_q
        # Propager l'erreur aux colonnes NON encore traitees.
        denom = H_inv[j, j] + 1e-12
        for k in order:
            if k == j or k in processed:
                continue
            w[:, k] += err * (H_inv[j, k] / denom)
        processed.append(j)
    return w


# Hessienne + dampening.
H = 2.0 * (X.T @ X)
damp = 0.01 * np.mean(np.diag(H))
H_damped = H + damp * np.eye(d_in)
H_inv = np.linalg.inv(H_damped)

# RTN baseline.
W_rtn = rtn_quantize_int4_per_channel(W_true)
err_rtn = mse(Y, X @ W_rtn.T)

# GPTQ-lite (ordre naturel).
W_gptq = gptq_lite(W_true, H_inv, act_order=False)
err_gptq = mse(Y, X @ W_gptq.T)

# GPTQ-lite act-order.
W_gptq_ao = gptq_lite(W_true, H_inv, act_order=True)
err_gptq_ao = mse(Y, X @ W_gptq_ao.T)

print(f"\n  Couche {W_true.shape}, calib {X.shape}, INT4 per-channel")
print(f"  Erreur de sortie ||Y - Y_q||^2 :")
print(f"    RTN baseline           = {err_rtn:.6e}")
print(f"    GPTQ-lite (naturel)    = {err_gptq:.6e}  (x{err_rtn / err_gptq:.1f} mieux)")
print(f"    GPTQ-lite (act-order)  = {err_gptq_ao:.6e}")
print("  -> GPTQ-lite bat RTN a memes bits : la compensation Hessienne ramene")
print("     une partie de l'erreur en ajustant les colonnes suivantes.")

# Effet du dampening : sans, H peut etre mal conditionnee.
cond_no_damp = np.linalg.cond(H)
cond_damp = np.linalg.cond(H_damped)
print(f"\n  Conditionnement de H : sans damp = {cond_no_damp:.2e}, "
      f"avec damp = {cond_damp:.2e}")
print("  -> le dampening reduit le conditionnement et stabilise l'inversion.")


# ============================================================================
# EXERCISE 8 — Courbe perplexite-proxy vs bits/poids
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 : courbe qualite (cross-entropy) vs bits/poids")
print("=" * 70)


def make_toy_model(d_in, d_h, vocab, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0 / np.sqrt(d_in), size=(d_in, d_h))
    W2 = rng.normal(0, 1.0 / np.sqrt(d_h), size=(d_h, vocab))
    return W1, W2


def forward_ce(W1, W2, Xin, targets):
    """Cross-entropy moyenne du mini-MLP sur (Xin, targets)."""
    h = np.maximum(0, Xin @ W1)
    logits = h @ W2
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-12))
    return ce


def quantize_per_group_nbits(w, n_bits, group_size=64):
    """Quantize symetrique per-group, parametre par n_bits."""
    qmax = (1 << (n_bits - 1)) - 1
    n_rows, n_cols = w.shape
    pad = (-n_cols) % group_size
    w_p = np.concatenate([w, np.zeros((n_rows, pad))], axis=1) if pad else w
    n_groups = w_p.shape[1] // group_size
    blocks = w_p.reshape(n_rows, n_groups, group_size)
    abs_max = np.max(np.abs(blocks), axis=2, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax)
    q = np.clip(np.round(blocks / scale), -qmax, qmax)
    return (q * scale).reshape(n_rows, -1)[:, :n_cols]


def quantize_per_tensor_nbits(w, n_bits):
    qmax = (1 << (n_bits - 1)) - 1
    scale = float(np.max(np.abs(w))) / qmax
    scale = scale if scale > 0 else 1.0
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return q * scale


def eval_quantization_curve(d_h, label, seed=0):
    rng = np.random.default_rng(seed + 1)
    d_in, vocab, n = 32, 16, 1000
    W1, W2 = make_toy_model(d_in, d_h, vocab, seed=seed)
    Xin = rng.normal(0, 1, size=(n, d_in))
    # Targets = argmax du modele FP perturbe (pour des labels coherents/non triviaux).
    h = np.maximum(0, Xin @ W1)
    logits = h @ W2 + rng.normal(0, 0.1, size=(n, vocab))
    targets = np.argmax(logits, axis=1)
    ce_fp = forward_ce(W1, W2, Xin, targets)
    print(f"\n  [{label}] d_h={d_h}, CE(FP32) = {ce_fp:.4f}, "
          f"perplexite = {np.exp(ce_fp):.3f}")
    print(f"    {'bits':<8} {'CE':<10} {'ppl':<10} {'delta vs FP':<12}")
    print("    " + "-" * 42)
    for nb in [8, 6, 5, 4, 3, 2]:
        W1q = quantize_per_group_nbits(W1, nb)
        W2q = quantize_per_group_nbits(W2, nb)
        ce = forward_ce(W1q, W2q, Xin, targets)
        print(f"    {nb:<8} {ce:<10.4f} {np.exp(ce):<10.3f} "
              f"{(ce - ce_fp) / ce_fp:<12.2%}")
    # Q2 sans groupes (per-tensor) : chute supplementaire.
    W1q = quantize_per_tensor_nbits(W1, 2)
    W2q = quantize_per_tensor_nbits(W2, 2)
    ce2 = forward_ce(W1q, W2q, Xin, targets)
    print(f"    {'2 (PT)':<8} {ce2:<10.4f} {np.exp(ce2):<10.3f} "
          f"{(ce2 - ce_fp) / ce_fp:<12.2%}  (per-tensor, pire)")
    return ce_fp


print("\n  Phenomenes attendus : plateau 8->4, coude 4->3, cliff a 2.")
ce_small = eval_quantization_curve(d_h=64, label="petit modele")
ce_big = eval_quantization_curve(d_h=256, label="gros modele (x4)")
print("\n  -> Le gros modele encaisse generalement mieux la quantization agressive")
print("     (delta relatif plus faible a 3-4 bits) : la redondance interne")
print("     absorbe une partie de l'erreur de rounding.")


# ============================================================================
# EXERCISE 9 — Double quantization (QLoRA)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9 : double quantization (quantizer les scales)")
print("=" * 70)


def erfinv_approx(y):
    y = np.asarray(y, dtype=np.float64)
    a = 0.147
    ln = np.log(1.0 - y * y)
    first = 2.0 / (np.pi * a) + ln / 2.0
    inside = first * first - ln / a
    return np.sign(y) * np.sqrt(np.sqrt(inside) - first)


def build_nf4_codebook():
    probs = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    levels = np.sqrt(2.0) * erfinv_approx(2.0 * probs - 1.0)
    return levels / np.max(np.abs(levels))


NF4 = build_nf4_codebook()


def nf4_quantize_blocks(x, block=64):
    """Retourne (codes, scales_par_bloc). Les scales = abs_max par bloc."""
    flat = x.flatten()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    abs_max = np.max(np.abs(blocks), axis=1)  # (n_blocks,) = scales FP32
    abs_max_safe = np.where(abs_max == 0, 1.0, abs_max)
    scaled = blocks / abs_max_safe[:, None]
    diffs = np.abs(scaled[..., None] - NF4[None, None, :])
    codes = np.argmin(diffs, axis=-1)
    return codes, abs_max, pad, x.shape, block


def nf4_dequantize(codes, scales, pad, shape, block):
    deq = (NF4[codes] * scales[:, None]).flatten()
    if pad:
        deq = deq[:-pad]
    return deq.reshape(shape)


W = np.random.randn(1024, 1024).astype(np.float64)
codes, scales, pad, shape, block = nf4_quantize_blocks(W, block=64)
n_blocks = scales.size
n_weights = W.size

# --- Cout SANS double quant ---
bits_codes = 4
bits_scale_simple = 32 / block  # 1 FP32 par bloc de 64 poids
total_simple = bits_codes + bits_scale_simple
print(f"\n  W {W.shape}, NF4 block=64, n_blocks={n_blocks}")
print(f"  Sans double quant : {bits_codes} (codes) + {bits_scale_simple:.3f} "
      f"(scale FP32 / 64) = {total_simple:.3f} bits/poids")

# --- Double quant : quantizer les scales (INT8 per super-bloc de 256) ---
super_block = 256
pad_s = (-scales.size) % super_block
scales_p = np.concatenate([scales, np.zeros(pad_s)]) if pad_s else scales
super_blocks = scales_p.reshape(-1, super_block)
# Quantize chaque scale en INT8 (un scale-de-scale FP32 par super-bloc).
ss_absmax = np.max(np.abs(super_blocks), axis=1, keepdims=True)
ss_absmax = np.where(ss_absmax == 0, 1.0, ss_absmax)
ss_scale = ss_absmax / 127.0
scales_q = np.clip(np.round(super_blocks / ss_scale), -127, 127) * ss_scale
scales_dq = scales_q.flatten()
if pad_s:
    scales_dq = scales_dq[:-pad_s]

bits_scale_double = 8 / block + 32 / (super_block * block)
total_double = bits_codes + bits_scale_double
print(f"  Avec double quant : {bits_codes} + 8/64 + 32/(256*64) = "
      f"{total_double:.4f} bits/poids")
print(f"  Economie : {total_simple - total_double:.3f} bits/poids "
      f"(QLoRA paper : ~0.373)")

# --- Degradation supplementaire due a la double quant ---
W_simple = nf4_dequantize(codes, scales, pad, shape, block)
W_double = nf4_dequantize(codes, scales_dq, pad, shape, block)
mse_simple = mse(W, W_simple)
mse_double = mse(W, W_double)
print(f"\n  MSE NF4 simple       = {mse_simple:.6e}")
print(f"  MSE NF4 double quant = {mse_double:.6e}")
print(f"  Surcout d'erreur     = {(mse_double - mse_simple) / mse_simple:+.2%}")
print("  -> negligeable : les scales sont peu nombreux et lisses, donc leur")
print("     quantization (8 bits) n'ajoute presque pas d'erreur sur W.")
print("  Danger : si block tres petit -> beaucoup de scales -> leur quantization")
print("     compterait davantage dans l'erreur totale.")

print("\nDone (HARD).")

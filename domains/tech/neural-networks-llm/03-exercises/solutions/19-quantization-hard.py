"""
Solutions HARD — Jour 19 : Quantization
=======================================
Exercices 7, 8, 9, 10 (hard). Pur NumPy, comme 02-code/19-quantization.py.
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

# ===========================================================================
# EXERCICE 10 - MXFP4, carte de precision mixte, et le vrai argument du QAT
# ===========================================================================
print("\n" + "=" * 70)
print("EXERCICE 10 - MX formats et QAT (Kimi K3, 2026)")
print("=" * 70)

MX_BLOCK = 32
FP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0


def round_to_fp4_e2m1(x):
    """Arrondi au plus proche des 8 magnitudes representables en FP4 E2M1."""
    sign = np.sign(x)
    idx = np.argmin(np.abs(np.abs(x)[..., None] - FP4_MAGNITUDES), axis=-1)
    return sign * FP4_MAGNITUDES[idx]


def quantize_mx(x, elem_max=FP4_MAX, block=MX_BLOCK, round_fn=round_to_fp4_e2m1):
    """Format MX : bloc de 32, UNE echelle partagee en puissance de deux."""
    flat = x.flatten().astype(np.float64)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)

    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    # E8M0 : l'echelle est 2^e avec e entier, aucune mantisse. On FLOOR la
    # difference d'exposants. Un arrondi vers le haut donnerait une echelle
    # trop grande : le maximum du bloc, divise par elle, tomberait au-dessus
    # de elem_max et serait CLIPPE - on perdrait la plus grande valeur du
    # bloc, exactement celle qu'il fallait preserver.
    shared_exp = np.floor(np.log2(amax)) - np.floor(np.log2(elem_max))
    scale = np.exp2(shared_exp)

    q = round_fn(np.clip(blocks / scale, -elem_max, elem_max))
    out = (q * scale).flatten()
    return (out[:-pad] if pad else out).reshape(x.shape)


def _perblock(x, block, quant_fn):
    flat = x.flatten().astype(np.float64)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    out = quant_fn(blocks, amax).flatten()
    return (out[:-pad] if pad else out).reshape(x.shape)


def int4_linear(blocks, amax):
    scale = amax / 7.0
    return np.clip(np.round(blocks / scale), -7, 7) * scale


def rel_mse(x, x_hat):
    return float(np.mean((x - x_hat) ** 2) / np.mean(x ** 2))


# --- A. MXFP4 vs INT4 vs NF4, a bloc EGAL ----------------------------------
print("\n  [A] Trois formats 4 bits, tous avec block=32")
rng10 = np.random.default_rng(42)
w = rng10.standard_normal((1024, 1024))

# NF4 : les 16 niveaux places aux quantiles d'une gaussienne (voir exercice 6).
nf4_codebook = np.array([
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844175, -0.1848030,
    -0.09105003, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261700, 0.72295684, 1.0])


def nf4(blocks, amax):
    norm = blocks / amax
    idx = np.argmin(np.abs(norm[..., None] - nf4_codebook), axis=-1)
    return nf4_codebook[idx] * amax


results = {
    "MXFP4 (echelle 2^e)": rel_mse(w, quantize_mx(w)),
    "INT4 lineaire (echelle fp32)": rel_mse(w, _perblock(w, MX_BLOCK, int4_linear)),
    "NF4 (echelle fp32)": rel_mse(w, _perblock(w, MX_BLOCK, nf4)),
}
for name, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"      {name:<32} MSE relative = {v:.4%}")

print("\n      MXFP4 est le MOINS precis des trois. Deux causes distinctes :")
print("      (a) l'echelle est une puissance de deux : si amax tombe juste")
print("          au-dessus d'une puissance de deux, on gaspille jusqu'a un")
print("          facteur 2 de dynamique utile ;")
print("      (b) NF4 place en plus ses 16 niveaux aux quantiles d'une")
print("          gaussienne, la ou les poids sont effectivement denses.")
print("      Sur quel axe MX gagne-t-il alors ? La dequantification. Une")
print("      echelle 2^e est un DECALAGE D'EXPOSANT, pas une multiplication ;")
print("      le bloc de 32 et le format sont figes par la spec OCP. Le")
print("      dequantize tient donc dans le datapath du Tensor Core au lieu")
print("      de couter un kernel separe. On echange de la precision par bit")
print("      contre du debit - et l'ecart de precision se rattrape par le")
print("      QAT, pas par le format.")

# --- B. La carte de precision ---------------------------------------------
print("\n  [B] Budget memoire d'une couche MoE Kimi K3")
D, LATENT, D_FF = 7168, 3584, 3072
N_ROUTED, N_SHARED = 896, 2
MXFP4_BITS = 4 + 8 / MX_BLOCK          # element + echelle partagee amortie

composants = [
    ("experts routes (MXFP4)", N_ROUTED * 2 * LATENT * D_FF, MXFP4_BITS),
    ("experts partages (BF16)", N_SHARED * 2 * D * D_FF, 16.0),
    ("projections latentes (BF16)", 2 * D * LATENT, 16.0),
    ("routeur (BF16)", D * N_ROUTED, 16.0),
]
print(f"      bits effectifs MXFP4 = 4 + 8/{MX_BLOCK} = {MXFP4_BITS:.2f}")
print()
print(f"      {'composant':<30}{'params':>16}{'bits':>7}{'Go':>8}{'Go BF16':>10}")
tot, tot16, n_all = 0.0, 0.0, 0
for name, n, bits in composants:
    tot += n * bits / 8 / 1e9
    tot16 += n * 16 / 8 / 1e9
    n_all += n
    print(f"      {name:<30}{n:>16,}{bits:>7.2f}"
          f"{n * bits / 8 / 1e9:>8.2f}{n * 16 / 8 / 1e9:>10.2f}")
print(f"      {'TOTAL':<30}{n_all:>16,}{'':>7}{tot:>8.2f}{tot16:>10.2f}"
      f"   ({tot16 / tot:.2f}x)")

frac = composants[0][1] / n_all
print(f"\n      Les experts routes pesent {frac:.1%} des parametres.")
print("      Consequence directe : quantifier EUX est tout le gain, et")
print("      laisser routeur, projections latentes et experts partages en")
print("      haute precision ne coute presque rien. Or ce sont precisement")
print("      les composants a proteger : le routeur decide de la SELECTION")
print("      (une erreur y change d'expert, pas juste de quelques chiffres")
print("      apres la virgule), et les experts partages sont sur le chemin")
print("      de CHAQUE token, donc leur erreur ne se moyenne jamais.")
print(f"\n      Les activations, elles, sont en MXFP8 E4M3 (max "
      f"{FP8_E4M3_MAX:.0f}). C'est")
print("      exactement la contrainte qui a impose SiTU-GLU au jour 9 : une")
print("      activation non bornee produit des valeurs qui deviennent des inf")
print("      dans ce format. Le format d'activation a dicte la fonction")
print("      d'activation.")

# --- C. L'ecart que le QAT supprime ---------------------------------------
print("\n  [C] L'ecart train/inference, mesure en KL")
D_H, V = 512, 4096
h_state = rng10.standard_normal((256, D_H))
W_head = rng10.standard_normal((D_H, V)) / np.sqrt(D_H)
W_head_q = quantize_mx(W_head)


def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


base_hi = h_state @ W_head
base_lo = h_state @ W_head_q

print("      Des poids aleatoires donnent un softmax presque plat, ou deux")
print("      politiques quelconques se ressemblent. On balaye donc une")
print("      echelle de logits pour couvrir plusieurs niveaux de confiance.")
print()
print(f"      {'echelle':>9}{'proba top-1':>14}{'KL(train||inf)':>17}"
      f"{'accord top-1':>15}{'recouv. top-5':>16}")
for s in [1.0, 3.0, 6.0, 10.0]:
    p_hi, p_lo = softmax_rows(base_hi * s), softmax_rows(base_lo * s)
    kl = float(np.mean(np.sum(
        p_hi * np.log((p_hi + 1e-12) / (p_lo + 1e-12)), axis=-1)))
    acc1 = float(np.mean(p_hi.argmax(-1) == p_lo.argmax(-1)))
    t5h = np.argsort(-base_hi, axis=-1)[:, :5]
    t5l = np.argsort(-base_lo, axis=-1)[:, :5]
    ov = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(t5h, t5l)]))
    print(f"      {s:>9.1f}{float(p_hi.max(-1).mean()):>14.1%}{kl:>17.4f}"
          f"{acc1:>15.1%}{ov:>16.1%}")

print("\n      L'accord top-1 et le recouvrement top-5 NE BOUGENT PAS avec")
print("      l'echelle : multiplier tous les logits par une constante ne peut")
print("      pas les reordonner. Ce qui bouge, c'est la KL - le meme desaccord")
print("      de classement coute d'autant plus cher que la politique est")
print("      confiante. Morale : ces trois metriques ne mesurent pas la meme")
print("      chose, et seule la KL voit la confiance.")
print("      (Ne pas lire l'accord top-1 comme un benchmark : ces poids sont")
print("       aleatoires, donc les premiers logits sont quasi ex aequo et")
print("       basculent facilement. Une tete entrainee separe bien mieux.)")

print("\n      Pourquoi cet ecart est une question de CORRECTION en RL :")
print("      les rollouts sont echantillonnes depuis le moteur d'inference")
print("      (poids quantifies), mais le gradient est calcule contre les poids")
print("      d'entrainement (pleine precision). L'algorithme croit corriger la")
print("      politique qui a genere les donnees ; ce n'est pas la meme. Il est")
print("      donc silencieusement off-policy, avec un biais que personne ne")
print("      mesure et que la KL ci-dessus quantifie.")
print("      Le QAT fait passer le forward d'entrainement par les MEMES poids")
print("      quantifies : l'ecart devient nul PAR CONSTRUCTION. C'est cet")
print("      argument-la - pas l'economie de memoire - qui justifie de garder")
print("      le QAT actif pendant tout le post-training, SFT et RL compris.")
print("\n      Honnetete : cette experience MESURE l'ecart. Elle n'entraine")
print("      rien, donc elle ne demontre pas que le QAT recupere la qualite")
print("      perdue - cela reste une affirmation du papier, a lire comme telle.")


print("\nDone (HARD).")

"""
Solutions HARD — Jour 10 : Fine-tuning & Alignment
===================================================
Exercices 7, 8 (hard).

Pur NumPy. Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/10-fine-tuning-alignment-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ============================================================================
# EXERCICE 7 : Deriver DPO depuis l'objectif RLHF contraint en KL
# ============================================================================
#
# PARTIE A — derivation (papier).
#
# Objectif RLHF (pour un x fixe), maximiser sur la distribution pi:
#     J(pi) = sum_y pi(y) r(y)  -  beta * KL(pi || pi_ref)
#           = sum_y pi(y) r(y)  -  beta * sum_y pi(y) log(pi(y)/pi_ref(y))
# sous contrainte sum_y pi(y) = 1.
#
# Lagrangien : L = J(pi) + lambda (1 - sum_y pi(y)).
# dL/dpi(y) = r(y) - beta (log(pi(y)/pi_ref(y)) + 1) - lambda = 0
#  => log(pi(y)/pi_ref(y)) = (r(y) - lambda)/beta - 1
#  => pi(y) = pi_ref(y) * exp(r(y)/beta) * exp(-lambda/beta - 1)
# Le dernier facteur est une constante de normalisation => 1/Z(x) avec
#     Z(x) = sum_y pi_ref(y) exp(r(y)/beta).
# DONC :   pi*(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(y)/beta).        (etape 1)
#
# Inversion (etape 2) :
#     r(y) = beta * log(pi*(y)/pi_ref(y)) + beta * log Z(x).
#
# Bradley-Terry (etape 3) :
#     P(y_w > y_l) = sigmoid(r(y_w) - r(y_l))
# Or r(y_w) - r(y_l) = beta[log(pi*(y_w)/pi_ref(y_w)) - log(pi*(y_l)/pi_ref(y_l))]
#   + beta log Z(x) - beta log Z(x)   <-- le terme log Z(x) S'ANNULE.
# => On peut entrainer pi_theta a maximiser sigmoid(beta[logratio_w - logratio_l])
#    SANS estimer Z(x) ni entrainer de reward model. C'est la loss DPO.
#
# ----------------------------------------------------------------------------
# PARTIE B — validation numerique.
# ============================================================================

print("=" * 70)
print("EXERCICE 7 : Derivation DPO depuis RLHF + KL — validation numerique")
print("=" * 70)

K = 6  # taille de l'espace d'actions (un seul prompt)
beta = 0.5

# pi_ref arbitraire (distribution valide) et reward arbitraire
ref_logits = np.random.randn(K)
pi_ref = np.exp(ref_logits) / np.sum(np.exp(ref_logits))
r = np.random.randn(K) * 2.0  # reward arbitraire

# Etape 1 : solution fermee pi*
Z = np.sum(pi_ref * np.exp(r / beta))
pi_star = (pi_ref * np.exp(r / beta)) / Z
print(f"\npi* somme = {pi_star.sum():.6f} (doit valoir 1)")


def rlhf_objective(pi, pi_ref, r, beta):
    """J(pi) = E_pi[r] - beta KL(pi||pi_ref)."""
    pi = np.clip(pi, 1e-12, None)
    kl = np.sum(pi * np.log(pi / pi_ref))
    return np.sum(pi * r) - beta * kl


# Verifier que pi* maximise l'objectif vs perturbations aleatoires
J_star = rlhf_objective(pi_star, pi_ref, r, beta)
better = 0
for _ in range(1000):
    noise = np.random.randn(K) * 0.1
    pert = pi_star + noise
    pert = np.clip(pert, 1e-6, None)
    pert /= pert.sum()  # reprojeter sur le simplexe
    if rlhf_objective(pert, pi_ref, r, beta) > J_star + 1e-9:
        better += 1
print(f"J(pi*) = {J_star:.4f}")
print(f"Perturbations qui battent pi* : {better}/1000 (doit etre 0 -> pi* est optimal)")

# Etape 2 : reconstruire r depuis les policies (a la constante beta log Z pres)
r_reconstruct = beta * np.log(pi_star / pi_ref)  # = r - beta log Z (constante)
const = r - r_reconstruct
print(f"\nr - r_reconstruct : {np.round(const, 6)} (constante = beta*logZ = {beta*np.log(Z):.4f})")
# Les DIFFERENCES de reward sont exactes (la constante s'annule)
i, j = 0, 3
diff_true = r[i] - r[j]
diff_rec = r_reconstruct[i] - r_reconstruct[j]
print(f"r(y_i)-r(y_j) vrai={diff_true:.6f}  reconstruit={diff_rec:.6f}  "
      f"ecart={abs(diff_true-diff_rec):.2e}")

# Etape 3 : Bradley-Terry identique avec les deux jeux de rewards
p_true = sigmoid(r[i] - r[j])
p_rec = sigmoid(r_reconstruct[i] - r_reconstruct[j])
print(f"P(y_i>y_j) avec r vrai={p_true:.6f}  avec r reconstruit={p_rec:.6f}  "
      f"ecart={abs(p_true-p_rec):.2e}")
print("  -> identiques : le terme beta logZ s'annule dans la difference. CQFD.")


# ============================================================================
# EXERCICE 8 : Memoire d'entrainement — full FT vs LoRA vs QLoRA
# ============================================================================

print("\n" + "=" * 70)
print("EXERCICE 8 : Empreinte memoire full FT vs LoRA vs QLoRA")
print("=" * 70)

GB = 1024 ** 3


def memory_breakdown(n_params_base, n_params_trainable, method):
    """
    Renvoie un dict (octets) : weights / gradients / optimizer / master.

    Conventions (mixed precision typique) :
      - poids stockes en fp16 (2 bytes) sauf QLoRA (4-bit = 0.5 byte) pour la base gelee.
      - gradients : fp16 (2 bytes), UNIQUEMENT sur les params entrainables.
      - Adam : 2 etats (m, v) en fp32 (4 bytes) -> 8 bytes/param entrainable.
      - master weights : copie fp32 (4 bytes) des params entrainables (mixed precision).
    """
    if method == "full":
        w_bytes = n_params_base * 2          # tout en fp16
        trainable = n_params_base            # tout entrainable
    elif method == "lora":
        w_bytes = n_params_base * 2          # base fp16 gelee
        trainable = n_params_trainable       # seuls les adaptateurs
    elif method == "qlora":
        w_bytes = n_params_base * 0.5        # base 4-bit gelee
        trainable = n_params_trainable
    else:
        raise ValueError(method)

    grad_bytes = trainable * 2               # gradients fp16
    adam_bytes = trainable * 8               # m + v en fp32
    master_bytes = trainable * 4             # master fp32 des params entraines
    return {
        "weights": w_bytes,
        "gradients": grad_bytes,
        "optimizer": adam_bytes,
        "master": master_bytes,
    }


def activations_bytes(batch, seq_len, d_model, n_layers, c=8, checkpointing=False):
    """
    Estimation grossiere des activations stockees pour le backward.
      ~ batch * seq_len * d_model * n_layers * c bytes.
    Gradient checkpointing : on ne garde que ~sqrt(n_layers) "checkpoints"
    et on recalcule le reste -> divise le stockage par ~sqrt(n_layers).
    """
    raw = batch * seq_len * d_model * n_layers * c
    return raw / np.sqrt(n_layers) if checkpointing else raw


# --- LLaMA 7B ---
n_base = 7e9
n_layers, d_model = 32, 4096
r = 16
# LoRA sur Q,K,V,O : 4 projections, chaque adaptateur = 2*d*r params
n_trainable = n_layers * 4 * (2 * d_model * r)
print(f"\nLLaMA 7B : base={n_base:.2e} params, "
      f"adaptateurs LoRA r={r} = {n_trainable:.2e} params")

print(f"\n{'method':>7s} {'weights':>9s} {'grad':>8s} {'adam':>8s} {'master':>8s} {'TOTAL':>9s}")
totals = {}
for m in ["full", "lora", "qlora"]:
    bd = memory_breakdown(n_base, n_trainable, m)
    total = sum(bd.values())
    totals[m] = total
    print(f"{m:>7s} {bd['weights']/GB:>8.1f}G {bd['gradients']/GB:>7.2f}G "
          f"{bd['optimizer']/GB:>7.2f}G {bd['master']/GB:>7.2f}G {total/GB:>8.1f}G")

print("""
Ordres de grandeur attendus (hors activations) :
  full  ~ 14(w)+14(grad)+56(adam)+28(master) ~ 112 GB de states (+activations)
  lora  ~ 14(w base) + qq centaines de MB pour grad/adam/master ~ 14-15 GB
  qlora ~ 3.5(w 4bit) + qq centaines de MB ~ 4-5 GB
(le full FT explose a cause d'Adam fp32 sur 7B params : 8 bytes * 7e9 = 56 GB.)
""")

# --- Activations + gradient checkpointing ---
batch, seq_len = 1, 2048
act = activations_bytes(batch, seq_len, d_model, n_layers, checkpointing=False)
act_ckpt = activations_bytes(batch, seq_len, d_model, n_layers, checkpointing=True)
print(f"Activations (batch={batch}, seq={seq_len}):")
print(f"  sans checkpointing : {act/GB:.2f} GB")
print(f"  avec checkpointing : {act_ckpt/GB:.2f} GB "
      f"(divise par ~sqrt({n_layers})={np.sqrt(n_layers):.1f})")
full_with_act = totals["full"] + act
full_with_ckpt = totals["full"] + act_ckpt
print(f"  full FT + activations brutes      : {full_with_act/GB:.1f} GB")
print(f"  full FT + activations checkpointed: {full_with_ckpt/GB:.1f} GB")


# --- Decision : fits sur quelle GPU ? ---
def fits(method, gpu_gb, include_act=True):
    total = totals[method]
    if include_act:
        total += act_ckpt  # on suppose le checkpointing active
    return total <= gpu_gb * GB


gpus = {"RTX 4090 24GB": 24, "A100 40GB": 40, "A100 80GB": 80, "H100 80GB": 80}
print(f"\n{'method':>7s} | " + " | ".join(f"{g:>14s}" for g in gpus))
print("-" * 70)
for m in ["full", "lora", "qlora"]:
    cells = []
    for g, mem in gpus.items():
        cells.append(f"{'OUI' if fits(m, mem) else 'NON':>14s}")
    print(f"{m:>7s} | " + " | ".join(cells))

print("""
Decision : QLoRA tient sur une RTX 4090 (24 GB) grand public : base 4-bit (~3.5 GB)
+ adaptateurs/Adam (qq centaines de MB) + activations checkpointed. C'est exactement
la promesse du paper QLoRA (fine-tuner un 65B sur une seule GPU 48 GB).

Trade-off qualite QLoRA : malgre le 4-bit, la qualite reste proche du full FT car
(1) la quantization NF4 (NormalFloat4) + double quantization preservent l'essentiel
de l'information des poids geles, et (2) les adaptateurs LoRA, eux, restent en pleine
precision et apprennent les corrections necessaires. La degradation typique est < 1%.
""")

print("=" * 70)
print("Fin solutions hard Jour 10.")
print("=" * 70)

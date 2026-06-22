"""
Solutions MEDIUM — Jour 10 : Fine-tuning & Alignment
=====================================================
Exercices 4, 5, 6 (medium).

Pur NumPy (comme 02-code/10-fine-tuning-alignment.py). Aucun framework requis.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/10-fine-tuning-alignment-medium.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# HELPERS
# ============================================================================

def sigmoid(z):
    """Sigmoid numeriquement stable (clip pour eviter overflow de exp)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def log_sigmoid(z):
    """
    -log(1+exp(-z)) stable.
    WHY: -log(sigmoid(z)) = log(1+exp(-z)) deborde pour z tres negatif.
    On utilise l'identite log_sigmoid(z) = -softplus(-z) avec softplus stable.
    """
    # softplus(x) = log(1+exp(x)) stable = max(x,0) + log(1+exp(-|x|))
    x = -z
    softplus = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    return -softplus


def log_softmax(z):
    """log_softmax stable sur un vecteur 1D."""
    z = z - np.max(z)
    return z - np.log(np.sum(np.exp(z)))


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ============================================================================
# EXERCICE 4 : DPO from scratch — forward, gradient analytique, training
# ============================================================================

print("=" * 70)
print("EXERCICE 4 : DPO from scratch")
print("=" * 70)


def dpo_loss_and_grad(z_policy, z_ref, chosen_id, rejected_id, beta=0.1):
    """
    Calcule la loss DPO ET son gradient analytique par rapport a z_policy
    (les logits de la policy). z_ref est fige.

    Forward:
      logp_w = log_softmax(z_policy)[chosen]
      logp_l = log_softmax(z_policy)[rejected]
      margin = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
      L      = -log sigmoid(margin)

    Gradient (chaine de derivees):
      dL/dmargin = -(1 - sigmoid(margin)) = sigmoid(margin) - 1
        WHY: d/dm[-log sigmoid(m)] = -(1 - sigmoid(m)).
      dmargin/dlogp_w = +beta  ;  dmargin/dlogp_l = -beta
      dlogp_i/dz[j]   = delta_ij - softmax(z)[j]   (jacobien du log_softmax)
    """
    lp_policy = log_softmax(z_policy)
    lp_ref = log_softmax(z_ref)

    logp_w = lp_policy[chosen_id]
    logp_l = lp_policy[rejected_id]
    ref_w = lp_ref[chosen_id]
    ref_l = lp_ref[rejected_id]

    margin = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    loss = -log_sigmoid(margin)  # = -log(sigmoid(margin))

    # --- gradient ---
    s = sigmoid(margin)
    dL_dmargin = s - 1.0  # negatif tant que margin n'est pas +inf -> pousse margin a monter

    p = softmax(z_policy)  # necessaire pour le jacobien du log_softmax
    grad = np.zeros_like(z_policy)

    # Contribution de logp_w : dmargin/dlogp_w = +beta
    # dlogp_w/dz = e_chosen - p
    grad_logp_w = -p.copy()
    grad_logp_w[chosen_id] += 1.0
    grad += dL_dmargin * beta * grad_logp_w

    # Contribution de logp_l : dmargin/dlogp_l = -beta
    grad_logp_l = -p.copy()
    grad_logp_l[rejected_id] += 1.0
    grad += dL_dmargin * (-beta) * grad_logp_l

    return loss, grad, margin


# --- Verification du gradient par difference finie ---
vocab = 5
z_policy0 = np.random.randn(vocab) * 0.3
z_ref = z_policy0.copy()  # reference = copie figee de l'init (WHY: au step 0 margin=0)
chosen_id, rejected_id = 0, 4
beta = 0.1

# On perturbe legerement la policy pour que le gradient soit non trivial
z_test = z_policy0 + np.random.randn(vocab) * 0.5
loss0, grad_analytic, _ = dpo_loss_and_grad(z_test, z_ref, chosen_id, rejected_id, beta)

eps = 1e-6
grad_numeric = np.zeros_like(z_test)
for i in range(vocab):
    zp = z_test.copy(); zp[i] += eps
    zm = z_test.copy(); zm[i] -= eps
    lp, _, _ = dpo_loss_and_grad(zp, z_ref, chosen_id, rejected_id, beta)
    lm, _, _ = dpo_loss_and_grad(zm, z_ref, chosen_id, rejected_id, beta)
    grad_numeric[i] = (lp - lm) / (2 * eps)

max_err = np.max(np.abs(grad_analytic - grad_numeric))
print(f"\nTest gradient (difference finie):")
print(f"  grad analytique : {np.round(grad_analytic, 5)}")
print(f"  grad numerique  : {np.round(grad_numeric, 5)}")
print(f"  ecart max       : {max_err:.2e}  ({'OK' if max_err < 1e-5 else 'FAIL'})")

# --- Mini training loop (SGD) ---
print("\nMini DPO training (prefere token 0 sur token 4), lr=0.5:")
print(f"{'step':>5s} {'p(chosen)':>11s} {'p(rejected)':>12s} {'margin':>8s} {'loss':>8s}")
z = z_policy0.copy()
lr = 0.5
for step in range(50):
    loss, grad, margin = dpo_loss_and_grad(z, z_ref, chosen_id, rejected_id, beta)
    z = z - lr * grad  # SGD: descente -> minimise la loss
    if step % 10 == 0 or step == 49:
        p = softmax(z)
        print(f"{step:>5d} {p[chosen_id]:>11.4f} {p[rejected_id]:>12.4f} "
              f"{margin:>8.4f} {loss:>8.4f}")

print("""
Analyse:
- p(chosen) monte, p(rejected) baisse : DPO pousse la policy vers la preference.
- Le margin grandit mais la loss ne tend jamais EXACTEMENT vers 0 : -log sigma(m)
  -> 0 seulement quand m -> +inf, ce qui demanderait des logits infinis.
- beta plus grand = gradient plus ample (le facteur beta multiplie le grad) :
  convergence plus rapide mais risque d'instabilite / d'overfitting des preferences.
""")


# ============================================================================
# EXERCICE 5 : LoRA — forward, backward, fusion, rang effectif
# ============================================================================

print("=" * 70)
print("EXERCICE 5 : LoRA forward/backward/fusion")
print("=" * 70)


class LoRALinearNumpy:
    """
    LoRA en NumPy avec forward ET backward.

      y = x @ W.T + (x @ A.T) @ B.T * scale
      W: (out, in) gele
      A: (r, in)  entrainable, init petit random
      B: (out, r) entrainable, init ZERO  (WHY: B@A=0 au depart -> y = base)
    """

    def __init__(self, in_features, out_features, r=4, alpha=8, W=None):
        self.W = (np.random.randn(out_features, in_features) * 0.02
                  if W is None else W.copy())
        self.A = np.random.randn(r, in_features) * 0.01
        self.B = np.zeros((out_features, r))  # init zero, crucial
        self.scale = alpha / r
        self.in_f, self.out_f, self.r = in_features, out_features, r

    def forward(self, x):
        self._x = x                       # cache pour backward
        self._h = x @ self.A.T            # (batch, r)
        base = x @ self.W.T
        lora = self._h @ self.B.T * self.scale
        return base + lora

    def backward(self, dY):
        """
        dY: (batch, out) = dL/dy.
        Retourne (dA, dB). W n'a PAS de gradient (gele).
          dL/dB = scale * dY.T @ h           -> (out, r)
          dL/dA = scale * (dY @ B).T @ x     -> (r, in)
        """
        dB = self.scale * (dY.T @ self._h)            # (out, r)
        dA = self.scale * ((dY @ self.B).T @ self._x)  # (r, in)
        # dL/dW serait dY.T @ x mais W est gele -> on ne le calcule/applique pas.
        return dA, dB

    def merge(self):
        """Renvoie W fusionne: W + scale * (B @ A). Cout inference = 0."""
        return self.W + self.scale * (self.B @ self.A)


# Loss bidon pour tester les gradients : L = 0.5 * ||y - target||^2  -> dL/dy = y - target
in_f, out_f, r = 6, 4, 2
layer = LoRALinearNumpy(in_f, out_f, r=r, alpha=4)
layer.B = np.random.randn(out_f, r) * 0.1  # on sort de l'init zero pour tester le grad
x = np.random.randn(3, in_f)
target = np.random.randn(3, out_f)


def loss_fn(layer, x, target):
    y = layer.forward(x)
    return 0.5 * np.sum((y - target) ** 2)


y = layer.forward(x)
dY = y - target
dA, dB = layer.backward(dY)

# Verification difference finie sur A
eps = 1e-6
dA_num = np.zeros_like(layer.A)
for i in range(r):
    for j in range(in_f):
        layer.A[i, j] += eps; lp = loss_fn(layer, x, target)
        layer.A[i, j] -= 2 * eps; lm = loss_fn(layer, x, target)
        layer.A[i, j] += eps
        dA_num[i, j] = (lp - lm) / (2 * eps)

dB_num = np.zeros_like(layer.B)
for i in range(out_f):
    for j in range(r):
        layer.B[i, j] += eps; lp = loss_fn(layer, x, target)
        layer.B[i, j] -= 2 * eps; lm = loss_fn(layer, x, target)
        layer.B[i, j] += eps
        dB_num[i, j] = (lp - lm) / (2 * eps)

print(f"\nGradient A: ecart max analytique/numerique = {np.max(np.abs(dA - dA_num)):.2e}")
print(f"Gradient B: ecart max analytique/numerique = {np.max(np.abs(dB - dB_num)):.2e}")

# Fusion : forward avec adaptateur == forward avec W fusionne sans adaptateur
W_merged = layer.merge()
y_lora = layer.forward(x)
y_merged = x @ W_merged.T
print(f"Fusion: ecart max forward(adapter) vs forward(W_merged) = "
      f"{np.max(np.abs(y_lora - y_merged)):.2e}")
print("  -> la fusion est exacte ; a l'inference, W_merged remplace W (aucun cout).")

# Rang effectif : LoRA capture un Delta de rang <= r parfaitement
print("\nRang effectif:")
np.random.seed(0)
d = 8
W_base = np.random.randn(d, d) * 0.1
# Delta de rang 2 (= r) : produit de deux matrices fines
U = np.random.randn(d, 2); Vt = np.random.randn(2, d)
Delta_low = U @ Vt
W_target_low = W_base + Delta_low

lora2 = LoRALinearNumpy(d, d, r=2, alpha=2, W=W_base)
# On entraine A,B pour matcher W_target via la "loss" ||(W + scale*B@A) - W_target||^2
lr = 0.05
for _ in range(4000):
    Wcur = lora2.W + lora2.scale * (lora2.B @ lora2.A)
    G = Wcur - W_target_low  # gradient de 0.5*||.||^2 par rapport a Wcur
    # backprop vers A et B : Wcur = W + scale*(B@A)
    lora2.B -= lr * lora2.scale * (G @ lora2.A.T)
    lora2.A -= lr * lora2.scale * (lora2.B.T @ G)
err_low = np.linalg.norm((lora2.W + lora2.scale * (lora2.B @ lora2.A)) - W_target_low)
print(f"  Delta rang 2, r=2 : erreur residuelle = {err_low:.4f} (≈0 -> capture exacte)")

# Delta de rang 5 > r=2 : LoRA ne capture que les 2 premieres composantes (SVD tronquee)
U5 = np.random.randn(d, 5); V5 = np.random.randn(5, d)
Delta_high = U5 @ V5
W_target_high = W_base + Delta_high
lora2b = LoRALinearNumpy(d, d, r=2, alpha=2, W=W_base)
for _ in range(4000):
    Wcur = lora2b.W + lora2b.scale * (lora2b.B @ lora2b.A)
    G = Wcur - W_target_high
    lora2b.B -= lr * lora2b.scale * (G @ lora2b.A.T)
    lora2b.A -= lr * lora2b.scale * (lora2b.B.T @ G)
err_high = np.linalg.norm((lora2b.W + lora2b.scale * (lora2b.B @ lora2b.A)) - W_target_high)
# Borne theorique: meilleure approx de rang 2 = somme des valeurs singulieres restantes
sv = np.linalg.svd(Delta_high, compute_uv=False)
best_rank2_err = np.sqrt(np.sum(sv[2:] ** 2))
print(f"  Delta rang 5, r=2 : erreur residuelle = {err_high:.4f}")
print(f"    borne SVD (meilleure approx rang 2) = {best_rank2_err:.4f} (LoRA s'en approche)")

print("""
Analyse B=0 vs A=0:
- B=0 -> au step 0, l'adaptateur ajoute scale*(x@A.T)@0 = 0 : le modele LoRA == modele base.
  Aucune regression au demarrage, puis l'adaptateur "s'allume" via le gradient.
- Si A et B etaient random, l'adaptateur ajouterait du bruit des le step 0 : le modele
  partirait deja biaise/degrade. (A=0 marcherait aussi mais la convention LoRA est B=0.)
""")


# ============================================================================
# EXERCICE 6 : DPO vs IPO vs SimPO
# ============================================================================

print("=" * 70)
print("EXERCICE 6 : DPO vs IPO vs SimPO")
print("=" * 70)


def dpo_loss(logr_w, logr_l, beta=0.1):
    """logr_* = logp_theta - logp_ref. Classification binaire."""
    return -log_sigmoid(beta * (logr_w - logr_l))


def ipo_loss(logr_w, logr_l, beta=0.1):
    """Regression: (logr_w - logr_l - 1/(2 beta))^2."""
    return (logr_w - logr_l - 1.0 / (2.0 * beta)) ** 2


def simpo_loss(logp_w, logp_l, len_w, len_l, beta=2.0, gamma=1.0):
    """
    Pas de reference. Reward = beta/|y| * logp_theta(y) (longueur-normalise),
    plus une marge gamma.
    """
    r_w = beta / len_w * logp_w
    r_l = beta / len_l * logp_l
    return -log_sigmoid(r_w - r_l - gamma)


# Batch synthetique
np.random.seed(7)
n = 8
logp_w = -np.abs(np.random.randn(n)) * 2 - 1     # log-probs (negatifs)
logp_l = -np.abs(np.random.randn(n)) * 2 - 1.5
ref_w = -np.abs(np.random.randn(n)) * 2 - 1
ref_l = -np.abs(np.random.randn(n)) * 2 - 1.5
logr_w = logp_w - ref_w
logr_l = logp_l - ref_l
len_w = np.random.randint(10, 40, size=n).astype(float)
len_l = np.random.randint(10, 40, size=n).astype(float)

print(f"\nLoss moyenne sur {n} paires:")
print(f"  DPO   : {np.mean(dpo_loss(logr_w, logr_l)):.4f}")
print(f"  IPO   : {np.mean(ipo_loss(logr_w, logr_l)):.4f}")
print(f"  SimPO : {np.mean(simpo_loss(logp_w, logp_l, len_w, len_l)):.4f}")

# Biais de longueur : y_w correct mais TRES long (somme de log-probs tres negative)
print("\nBiais de longueur (y_w correct mais tres long):")
# y_w: 100 tokens a -0.1 chacun -> logp = -10 ; y_l: 5 tokens a -0.1 -> logp = -0.5
logp_w_long, len_w_long = -10.0, 100.0
logp_l_short, len_l_short = -0.5, 5.0
# Avec une reference neutre (0), logr = logp
dpo_biased = dpo_loss(np.array([logp_w_long]), np.array([logp_l_short]))[0]
simpo_fair = simpo_loss(np.array([logp_w_long]), np.array([logp_l_short]),
                        np.array([len_w_long]), np.array([len_l_short]))[0]
print(f"  DPO (somme brute)         : loss={dpo_biased:.3f}  "
      f"-> penalise y_w juste parce qu'il est long (logp tres negatif)")
print(f"  SimPO (normalise longueur): per-token w={logp_w_long/len_w_long:.3f} "
      f"vs l={logp_l_short/len_l_short:.3f}, loss={simpo_fair:.3f}")
print("  -> SimPO juge la qualite PAR TOKEN, pas la longueur brute.")

# Saturation du gradient DPO vs IPO quand l'ecart devient grand
print("\nSaturation du gradient (ecart logr_w - logr_l croissant):")
print(f"{'ecart':>8s} {'|dDPO/dx|':>12s} {'|dIPO/dx|':>12s}")
beta = 0.1
for gap in [0.0, 2.0, 5.0, 10.0, 30.0]:
    # gradient DPO par rapport a x=(logr_w - logr_l): -beta*(1 - sigmoid(beta*x))
    g_dpo = abs(-beta * (1 - sigmoid(beta * gap)))
    # gradient IPO: 2*(x - 1/(2beta))
    g_ipo = abs(2 * (gap - 1.0 / (2 * beta)))
    print(f"{gap:>8.1f} {g_dpo:>12.5f} {g_ipo:>12.5f}")
print("  -> DPO sature (grad -> 0 quand x grand) : il arrete de pousser.")
print("     IPO vise une cible fixe -> il ne sur-pousse pas (plus robuste a l'overfit).")

print("""
Tableau recap:
  Methode | Reference ? | Type           | Longueur          | + / -
  --------|-------------|----------------|-------------------|---------------------------
  DPO     | OUI         | classification | somme brute       | +simple / -biais longueur, sur-confiance
  IPO     | OUI         | regression     | somme brute       | +pas d'overfit / -toujours besoin ref
  SimPO   | NON         | classification | normalise+marge   | +pas de ref, anti-longueur / -tuning gamma
""")

print("=" * 70)
print("Fin solutions medium Jour 10.")
print("=" * 70)

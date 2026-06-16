"""
Solutions HARD — Jour 12 : Multimodalite
=========================================
Exercices 7, 8 (hard).

Pur NumPy. Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/12-multimodalite-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


def l2_normalize(x, axis=-1, eps=1e-8):
    return x / np.sqrt((x ** 2).sum(axis=axis, keepdims=True) + eps)


def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================================
# EXERCICE 7 : Mini-CLIP complet avec backprop entiere a la main
# ============================================================================

print("=" * 70)
print("EXERCICE 7 : Mini-CLIP complet (backprop manuelle + Adam + zero-shot)")
print("=" * 70)


class MiniCLIP:
    """CLIP en NumPy : deux projections lineaires + temperature apprise."""

    def __init__(self, d_img, d_txt, d, seed=0):
        rng = np.random.default_rng(seed)
        self.W_img = rng.standard_normal((d_img, d)) * 0.1
        self.b_img = np.zeros(d)
        self.W_txt = rng.standard_normal((d_txt, d)) * 0.1
        self.b_txt = np.zeros(d)
        self.log_logit_scale = np.array(np.log(10.0))  # temperature apprise (log)

    def forward(self, img_feats, txt_feats):
        """Renvoie loss + un cache pour le backward."""
        z_img = img_feats @ self.W_img + self.b_img
        z_txt = txt_feats @ self.W_txt + self.b_txt
        img_n = l2_normalize(z_img)
        txt_n = l2_normalize(z_txt)
        scale = np.exp(np.clip(self.log_logit_scale, None, np.log(100.0)))
        S = scale * (img_n @ txt_n.T)
        N = S.shape[0]
        P_row = softmax_rows(S)
        P_col = softmax_rows(S.T)
        li = -np.mean(np.log(P_row[np.arange(N), np.arange(N)] + 1e-12))
        lt = -np.mean(np.log(P_col[np.arange(N), np.arange(N)] + 1e-12))
        loss = (li + lt) / 2
        cache = (img_feats, txt_feats, z_img, z_txt, img_n, txt_n,
                 scale, S, P_row, P_col, N)
        return loss, cache

    def backward(self, cache):
        (img_feats, txt_feats, z_img, z_txt, img_n, txt_n,
         scale, S, P_row, P_col, N) = cache
        onehot = np.eye(N)
        dS = ((P_row - onehot) / N + ((P_col - onehot) / N).T) / 2

        # gradient temperature : S = scale * (img_n @ txt_n.T), scale=exp(log_ls)
        sim = img_n @ txt_n.T
        d_scale = np.sum(dS * sim)
        d_log_ls = d_scale * scale  # dscale/dlog_ls = scale

        # backprop vers img_n, txt_n
        d_img_n = scale * (dS @ txt_n)
        d_txt_n = scale * (dS.T @ img_n)

        # backprop normalisation L2
        def bp_l2(z, dn):
            norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
            zhat = z / norm
            proj = (dn * zhat).sum(axis=1, keepdims=True) * zhat
            return (dn - proj) / norm

        d_z_img = bp_l2(z_img, d_img_n)
        d_z_txt = bp_l2(z_txt, d_txt_n)

        # backprop Linear
        grads = {
            "W_img": img_feats.T @ d_z_img,
            "b_img": d_z_img.sum(axis=0),
            "W_txt": txt_feats.T @ d_z_txt,
            "b_txt": d_z_txt.sum(axis=0),
            "log_logit_scale": np.array(d_log_ls),
        }
        return grads


# --- Verification des gradients par difference finie ---
N, d_img, d_txt, d = 8, 10, 12, 6
rng = np.random.default_rng(1)
base = rng.standard_normal((N, max(d_img, d_txt)))
img_feats = base[:, :d_img] + rng.standard_normal((N, d_img)) * 0.2
txt_feats = base[:, :d_txt] + rng.standard_normal((N, d_txt)) * 0.2

model = MiniCLIP(d_img, d_txt, d, seed=2)
loss, cache = model.forward(img_feats, txt_feats)
grads = model.backward(cache)

eps = 1e-5
print("\nVerification gradients (difference finie):")
for name in ["W_img", "b_img", "W_txt", "b_txt", "log_logit_scale"]:
    param = getattr(model, name)
    g_num = np.zeros_like(np.atleast_1d(param), dtype=float)
    flat = np.atleast_1d(param).ravel()
    g_flat = g_num.ravel()
    for k in range(flat.size):
        orig = flat[k]
        flat[k] = orig + eps
        setattr(model, name, np.atleast_1d(param).reshape(np.shape(param)))
        lp, _ = model.forward(img_feats, txt_feats)
        flat[k] = orig - eps
        lm, _ = model.forward(img_feats, txt_feats)
        flat[k] = orig
        g_flat[k] = (lp - lm) / (2 * eps)
    err = np.max(np.abs(np.atleast_1d(grads[name]).ravel() - g_flat))
    print(f"  {name:>16s} : ecart max = {err:.2e}  ({'OK' if err < 1e-4 else 'FAIL'})")


# --- Adam en NumPy + training ---
class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.atleast_1d(v), dtype=float) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.atleast_1d(v), dtype=float) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = np.atleast_1d(grads[k]).astype(float)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)  # biais-correction
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            new = np.atleast_1d(params[k]).astype(float) - upd
            params[k] = new.reshape(np.shape(params[k]))
        return params


print("\nTraining mini-CLIP (Adam, 300 steps):")
N = 32
base = rng.standard_normal((N, 16))
img_feats = base + rng.standard_normal((N, 16)) * 0.3
txt_feats = base + rng.standard_normal((N, 16)) * 0.3
model = MiniCLIP(16, 16, 16, seed=3)
params = {k: getattr(model, k) for k in
          ["W_img", "b_img", "W_txt", "b_txt", "log_logit_scale"]}
opt = Adam(params, lr=0.05)
for step in range(300):
    for k in params:
        setattr(model, k, params[k])
    loss, cache = model.forward(img_feats, txt_feats)
    grads = model.backward(cache)
    params = opt.step(params, grads)
    if step % 60 == 0 or step == 299:
        _, c = model.forward(img_feats, txt_feats)
        S = c[7]
        acc = np.mean(S.argmax(axis=1) == np.arange(N))
        print(f"  step {step:3d}  loss={loss:.4f}  top1={acc*100:.0f}%")

# --- Zero-shot classification ---
print("\nZero-shot classification (3 classes):")
for k in params:
    setattr(model, k, params[k])
# 3 prompts de classe (vecteurs texte) + images proches d'une classe
class_txt = rng.standard_normal((3, 16))
class_emb = l2_normalize(class_txt @ model.W_txt + model.b_txt)
n_test = 30
true_cls = rng.integers(0, 3, size=n_test)
test_img_feats = class_txt[true_cls] + rng.standard_normal((n_test, 16)) * 0.3
test_emb = l2_normalize(test_img_feats @ model.W_img + model.b_img)
sims = test_emb @ class_emb.T
pred_cls = sims.argmax(axis=1)
acc_zs = np.mean(pred_cls == true_cls)
print(f"  accuracy zero-shot = {acc_zs*100:.0f}% (hasard = 33%)")

print("""
Analyse:
- Normalisation L2 indispensable : sans elle, le modele peut minimiser la loss en
  gonflant les normes (augmenter tous les scores) sans rien apprendre de semantique.
  La normalisation force a apprendre la DIRECTION (cosinus), pas la magnitude.
- On clampe logit_scale (exp <= 100) : sinon la temperature explose, le softmax sature
  et les gradients meurent (un seul logit a ~1, les autres a ~0).
""")


# ============================================================================
# EXERCICE 8 : VQ-VAE — quantization vectorielle from scratch
# ============================================================================

print("=" * 70)
print("EXERCICE 8 : VQ-VAE — vector quantization from scratch")
print("=" * 70)


def vector_quantize(z_e, codebook):
    """
    Trouve pour chaque vecteur de z_e (n, d) le code le plus proche (L2).
    Renvoie (indices (n,), z_q (n, d)).
    Distance vectorisee: ||z-e||^2 = ||z||^2 - 2 z.e + ||e||^2.
    """
    z2 = (z_e ** 2).sum(axis=1, keepdims=True)          # (n,1)
    e2 = (codebook ** 2).sum(axis=1)[None, :]           # (1,K)
    cross = z_e @ codebook.T                             # (n,K)
    dist = z2 - 2 * cross + e2                           # (n,K)
    indices = dist.argmin(axis=1)
    z_q = codebook[indices]
    return indices, z_q


# Encodeur / decodeur lineaires (demo)
K, d = 16, 8           # K codes de dimension d
d_pixel = 12           # dimension "image aplatie" d'un patch
n_patches = 16         # grille 4x4

rng = np.random.default_rng(0)
codebook = rng.standard_normal((K, d)) * 0.5
W_enc = rng.standard_normal((d_pixel, d)) * 0.3
W_dec = rng.standard_normal((d, d_pixel)) * 0.3

# Une "image" = n_patches patches
image = rng.standard_normal((n_patches, d_pixel))


def encode(x):
    return x @ W_enc          # z_e (n, d)


def decode(zq):
    return zq @ W_dec         # reconstruction (n, d_pixel)


z_e = encode(image)
indices, z_q = vector_quantize(z_e, codebook)

# Straight-through estimator (forward identique a z_q)
z_q_st = z_e + (z_q - z_e)  # numeriquement = z_q ; en autograd le grad passerait par z_e
print(f"\nSTE forward == z_q ? ecart = {np.max(np.abs(z_q_st - z_q)):.2e}")
print("  (en backward, le STE copie dL/dz_q vers z_e car argmin n'est pas differentiable)")

# Loss VQ-VAE (3 termes)
beta = 0.25
recon = decode(z_q_st)
loss_recon = np.mean((recon - image) ** 2)
loss_codebook = np.mean((z_e.copy() - z_q) ** 2)      # stop_grad(z_e) -> rapproche codes des features
loss_commit = beta * np.mean((z_e - z_q.copy()) ** 2)  # stop_grad(e) -> rapproche encodeur du code
print(f"\nLoss VQ-VAE: recon={loss_recon:.4f}  codebook={loss_codebook:.4f}  "
      f"commit={loss_commit:.4f}")

# Grille de tokens discrets
grid = indices.reshape(4, 4)
print(f"\nGrille de tokens discrets (4x4) :\n{grid}")
usage = len(np.unique(indices)) / K
print(f"Utilisation codebook : {len(np.unique(indices))}/{K} codes ({usage*100:.0f}%)")
print("  (un faible taux = 'codebook collapse' : peu de codes utilises -> tokenizer pauvre)")

# Mini training : encodeur + decodeur + codebook
print("\nMini training VQ-VAE (loss reconstruction):")
# Jeu de plusieurs images synthetiques structurees
n_imgs = 8
imgs = rng.standard_normal((n_imgs, n_patches, d_pixel))
lr = 0.02
for step in range(400):
    total_recon = 0.0
    dW_enc = np.zeros_like(W_enc)
    dW_dec = np.zeros_like(W_dec)
    dcode = np.zeros_like(codebook)
    for img in imgs:
        z_e = img @ W_enc
        idx, z_q = vector_quantize(z_e, codebook)
        recon = z_q @ W_dec
        diff = recon - img
        total_recon += np.mean(diff ** 2)
        # gradient reconstruction (STE: grad de z_q copie vers z_e)
        g_recon = (2.0 / diff.size) * diff
        dW_dec += z_q.T @ g_recon
        g_zq = g_recon @ W_dec.T
        dW_enc += img.T @ g_zq          # STE : passe a travers z_e
        # codebook loss : rapproche les codes choisis de z_e
        for n in range(n_patches):
            dcode[idx[n]] += 2 * (codebook[idx[n]] - z_e[n]) / n_patches
        # commitment : rapproche l'encodeur du code (gradient vers W_enc)
        dW_enc += img.T @ (2 * beta * (z_e - z_q) / z_e.size)
    W_enc -= lr * dW_enc / n_imgs
    W_dec -= lr * dW_dec / n_imgs
    codebook -= lr * dcode / n_imgs
    if step % 80 == 0 or step == 399:
        print(f"  step {step:3d}  recon loss = {total_recon/n_imgs:.4f}")

# Images similaires -> tokens similaires
imgA = imgs[0]
imgA_noisy = imgA + rng.standard_normal(imgA.shape) * 0.05
idxA, _ = vector_quantize(imgA @ W_enc, codebook)
idxA2, _ = vector_quantize(imgA_noisy @ W_enc, codebook)
agreement = np.mean(idxA == idxA2)
print(f"\nImage vs version bruitee : {agreement*100:.0f}% de tokens identiques "
      f"(images similaires -> sequences similaires)")

print("=" * 70)
print("Fin solutions hard Jour 12.")
print("=" * 70)

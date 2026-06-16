"""
Solutions MEDIUM — Jour 12 : Multimodalite
===========================================
Exercices 4, 5, 6 (medium).

Pur NumPy (comme 02-code/12-multimodalite.py). Aucun framework requis.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/12-multimodalite-medium.py
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
# EXERCICE 4 : ViT patch embedding == Conv2d
# ============================================================================

print("=" * 70)
print("EXERCICE 4 : ViT patchify == Conv2d")
print("=" * 70)


def patchify(image, patch_size):
    """Decoupe (H,W,C) en patches aplatis (n_patches, patch*patch*C), row-major."""
    H, W, C = image.shape
    n_h, n_w = H // patch_size, W // patch_size
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :]
            # flatten en ordre (row, col, channel) -> meme ordre que le reshape du kernel
            patches.append(patch.flatten())
    return np.stack(patches, axis=0)


def conv2d_patch(image, kernel, patch_size):
    """
    Conv2d avec stride=kernel=patch_size, from scratch.
    kernel: (d_model, patch*patch*C) deja aplatie dans le MEME ordre que patchify.
    Renvoie (n_patches, d_model).
    WHY: une conv stride=patch sans chevauchement applique exactement le meme
    produit scalaire qu'un Linear sur le patch aplati.
    """
    H, W, C = image.shape
    n_h, n_w = H // patch_size, W // patch_size
    out = []
    for i in range(n_h):
        for j in range(n_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :].flatten()
            out.append(kernel @ patch)  # (d_model,)
    return np.stack(out, axis=0)


H = W = 32
C = 3
patch_size = 8
d_model = 64
image = np.random.randn(H, W, C)
patch_dim = patch_size * patch_size * C

W_proj = np.random.randn(patch_dim, d_model) * 0.02  # (patch_dim, d_model)

# Approche A : patchify + Linear
patches = patchify(image, patch_size)
embA = patches @ W_proj  # (n_patches, d_model)

# Approche B : conv2d avec kernel = W_proj.T (d_model, patch_dim)
embB = conv2d_patch(image, W_proj.T, patch_size)

print(f"\nn_patches = {patches.shape[0]} (attendu {(H//patch_size)**2})")
print(f"patch_dim = {patch_dim}")
print(f"Ecart max patchify+Linear vs Conv2d : {np.max(np.abs(embA - embB)):.2e}")
print(f"Params projection (les deux) : {W_proj.size} = patch_dim*d_model = "
      f"{patch_dim*d_model}")

# ViT-B/16
ps16, d768 = 16, 768
n_p = (224 // ps16) ** 2
pd = ps16 * ps16 * 3
print(f"\nViT-B/16 : n_patches={n_p}, patch_dim={pd}, "
      f"proj params={pd*d768:,}")
print(f"  sequence finale (CLS + patches) : ({n_p+1}, {d768})")

print("""
Analyse: les frameworks (timm, HF) implementent la patchification comme une Conv2d
car les kernels GPU de convolution sont ultra-optimises -> meme calcul, meme params,
mais throughput bien superieur a une boucle Python. La convention de flatten doit
matcher entre patch et kernel (ici row,col,channel).
""")


# ============================================================================
# EXERCICE 5 : CLIP contrastive loss + gradient + retrieval accuracy
# ============================================================================

print("=" * 70)
print("EXERCICE 5 : CLIP loss + gradient + retrieval")
print("=" * 70)


def clip_loss_and_grad_S(img_n, txt_n, logit_scale):
    """
    Loss CLIP symetrique et gradient par rapport a la matrice de scores S.
    S = logit_scale * img_n @ txt_n.T  (img_n, txt_n deja normalises)
    """
    N = img_n.shape[0]
    S = logit_scale * (img_n @ txt_n.T)
    # image -> texte : cross-entropy par ligne, cible = diagonale
    P_row = softmax_rows(S)
    loss_i2t = -np.mean(np.log(P_row[np.arange(N), np.arange(N)] + 1e-12))
    # texte -> image : par colonne == lignes de S.T
    P_col = softmax_rows(S.T)
    loss_t2i = -np.mean(np.log(P_col[np.arange(N), np.arange(N)] + 1e-12))
    loss = (loss_i2t + loss_t2i) / 2

    # gradient par rapport a S
    onehot = np.eye(N)
    dS_i2t = (P_row - onehot) / N
    dS_t2i = (P_col - onehot) / N  # par rapport a S.T
    dS = (dS_i2t + dS_t2i.T) / 2   # .T pour ramener dans l'espace de S
    return loss, dS, S


# Donnees alignees
N, d = 12, 8
base = np.random.randn(N, d)
img = l2_normalize(base + np.random.randn(N, d) * 0.2)
txt = l2_normalize(base + np.random.randn(N, d) * 0.2)
logit_scale = 10.0

loss, dS, S = clip_loss_and_grad_S(img, txt, logit_scale)

# Verif difference finie du gradient par rapport a S : on perturbe S directement.
def loss_from_S(S):
    N = S.shape[0]
    P_row = softmax_rows(S)
    P_col = softmax_rows(S.T)
    li = -np.mean(np.log(P_row[np.arange(N), np.arange(N)] + 1e-12))
    lt = -np.mean(np.log(P_col[np.arange(N), np.arange(N)] + 1e-12))
    return (li + lt) / 2


eps = 1e-6
dS_num = np.zeros_like(S)
for i in range(N):
    for j in range(N):
        Sp = S.copy(); Sp[i, j] += eps
        Sm = S.copy(); Sm[i, j] -= eps
        dS_num[i, j] = (loss_from_S(Sp) - loss_from_S(Sm)) / (2 * eps)
print(f"\nGradient dL/dS : ecart max analytique/numerique = "
      f"{np.max(np.abs(dS - dS_num)):.2e}")

# Retrieval accuracy
preds_i2t = S.argmax(axis=1)
top1_i2t = np.mean(preds_i2t == np.arange(N))
top5_i2t = np.mean([(i in np.argsort(-S[i])[:5]) for i in range(N)])
preds_t2i = S.T.argmax(axis=1)
top1_t2i = np.mean(preds_t2i == np.arange(N))
print(f"Retrieval i2t : top1={top1_i2t*100:.0f}%  top5={top5_i2t*100:.0f}%")
print(f"Retrieval t2i : top1={top1_t2i*100:.0f}%")

# Mini training (gradient analytique) sur projections
print("\nMini training CLIP (projections lineaires, gradient analytique):")
d_in, d_out = 16, 8
base = np.random.randn(N, d_in)
raw_img = base + np.random.randn(N, d_in) * 0.3
raw_txt = base + np.random.randn(N, d_in) * 0.3
W_img = np.random.randn(d_in, d_out) * 0.1
W_txt = np.random.randn(d_in, d_out) * 0.1
ls = 10.0
lr = 0.5
for step in range(300):
    z_img = raw_img @ W_img
    z_txt = raw_txt @ W_txt
    img_n = l2_normalize(z_img)
    txt_n = l2_normalize(z_txt)
    loss, dS, S = clip_loss_and_grad_S(img_n, txt_n, ls)

    # backprop S -> img_n, txt_n
    d_img_n = ls * (dS @ txt_n)
    d_txt_n = ls * (dS.T @ img_n)

    # backprop normalisation L2 : d(x/||x||)/dx = (I - x_hat x_hat^T)/||x||
    def backprop_l2(z, dn):
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        zhat = z / norm
        # par ligne : dn - (dn.zhat) zhat, le tout / norm
        proj = (dn * zhat).sum(axis=1, keepdims=True) * zhat
        return (dn - proj) / norm

    d_z_img = backprop_l2(z_img, d_img_n)
    d_z_txt = backprop_l2(z_txt, d_txt_n)
    dW_img = raw_img.T @ d_z_img
    dW_txt = raw_txt.T @ d_z_txt
    W_img -= lr * dW_img
    W_txt -= lr * dW_txt

    if step % 60 == 0 or step == 299:
        acc = np.mean(S.argmax(axis=1) == np.arange(N))
        print(f"  step {step:3d}  loss={loss:.4f}  top1={acc*100:.0f}%")

# Effet temperature
print("\nEffet temperature (logit_scale) sur la nettete du softmax (ligne 0):")
S_fixed = img_n @ txt_n.T
for ls_test in [1.0, 5.0, 20.0]:
    p = softmax_rows(ls_test * S_fixed)[0]
    print(f"  logit_scale={ls_test:>4.0f} : softmax[0]={np.round(p, 3)}  "
          f"max={p.max():.3f}")
print("""
-> logit_scale grand = softmax plus pique. CLIP APPREND logit_scale pour ajuster
   automatiquement la "confiance" : trop bas = pas discriminant, trop haut = sature
   les gradients. On le clampe (exp <= 100) pour eviter l'explosion.
""")


# ============================================================================
# EXERCICE 6 : LLaVA-style — features visuelles en tokens texte
# ============================================================================

print("=" * 70)
print("EXERCICE 6 : LLaVA-style projector")
print("=" * 70)


def gelu(x):
    """GELU approx (tanh)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


n_patches, d_vision, d_model = 49, 512, 768
vision_feats = np.random.randn(n_patches, d_vision)  # sortie encodeur visuel

# Projecteur simple (Linear)
W_lin = np.random.randn(d_vision, d_model) * 0.02
vis_tokens_lin = vision_feats @ W_lin
print(f"\nProjector Linear : {vision_feats.shape} -> {vis_tokens_lin.shape}")

# Projecteur MLP (LLaVA 1.5) : Linear -> GELU -> Linear
W1 = np.random.randn(d_vision, d_model) * 0.02
W2 = np.random.randn(d_model, d_model) * 0.02
vis_tokens_mlp = gelu(vision_feats @ W1) @ W2
print(f"Projector MLP    : {vision_feats.shape} -> {vis_tokens_mlp.shape}")

# Assemblage de la sequence multimodale
vocab, n_text = 1000, 6
text_table = np.random.randn(vocab, d_model) * 0.02
text_ids = np.random.randint(0, vocab, size=n_text)
text_emb = text_table[text_ids]
img_open = np.random.randn(1, d_model) * 0.02   # token <image>
img_close = np.random.randn(1, d_model) * 0.02  # token </image>

sequence = np.concatenate([img_open, vis_tokens_mlp, img_close, text_emb], axis=0)
expected_len = 1 + n_patches + 1 + n_text
print(f"\nSequence multimodale : {sequence.shape} "
      f"(attendu ({expected_len}, {d_model}))")
print("  -> prete pour un Transformer causal standard.")

# Compte de parametres du connector
p_lin = W_lin.size
p_mlp = W1.size + W2.size
print(f"\nParams connector : Linear={p_lin:,}  MLP={p_mlp:,}")
print("""
LLaVA gele le LLM et l'encodeur visuel et n'entraine (phase 1) que le connector :
peu de params a apprendre, peu de donnees, et on preserve les capacites du LLM et
de l'encodeur. Avantage vs Flamingo (cross-attention) : architecture ultra-simple,
on reutilise un LLM tel quel. Cout : les tokens visuels OCCUPENT le contexte
(49 tokens par image ici) -> sequence plus longue, attention plus chere.
""")

print("=" * 70)
print("Fin solutions medium Jour 12.")
print("=" * 70)

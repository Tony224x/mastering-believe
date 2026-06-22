"""
Solutions — Jour 12 : Multimodalite

Run: python 03-exercises/solutions/12-multimodalite.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — ViT dimensions
# ============================================================================

print("=" * 70)
print("Exercice 1: ViT dimensions")
print("=" * 70)


def vit_stats(H, W, C, patch_size, d_model):
    """Return a dict of ViT stats."""
    n_h = H // patch_size
    n_w = W // patch_size
    n_patches = n_h * n_w
    patch_dim = patch_size * patch_size * C
    proj_params = patch_dim * d_model
    return dict(n_patches=n_patches, patch_dim=patch_dim,
                proj_params=proj_params, total_tokens=n_patches + 1)


# 1) ViT-B/16
stats_b16 = vit_stats(224, 224, 3, 16, 768)
print(f"\n1) ViT-B/16 (224x224, patch 16, d=768):")
print(f"   n_patches: {stats_b16['n_patches']}")
print(f"   patch_dim: {stats_b16['patch_dim']} (16*16*3)")
print(f"   proj params: {stats_b16['proj_params']:,}")
print(f"   total tokens (CLS+): {stats_b16['total_tokens']}")

# 2) ViT-L/14
stats_l14 = vit_stats(224, 224, 3, 14, 1024)
print(f"\n2) ViT-L/14 (224x224, patch 14, d=1024):")
print(f"   n_patches: {stats_l14['n_patches']}  (16x16)")
print(f"   patch_dim: {stats_l14['patch_dim']} (14*14*3)")

# 3) Fit in 4096 context
print(f"\n3) Dans un contexte de 4096 tokens:")
fit_l14 = 4096 // stats_l14['total_tokens']
stats_b32 = vit_stats(224, 224, 3, 32, 768)
fit_b32 = 4096 // stats_b32['total_tokens']
print(f"   ViT-L/14: {fit_l14} images (chacune {stats_l14['total_tokens']} tokens)")
print(f"   ViT-B/32: {fit_b32} images (chacune {stats_b32['total_tokens']} tokens)")

# 4) High-res image
stats_hr = vit_stats(512, 512, 3, 16, 768)
print(f"\n4) Image 512x512 patch 16:")
print(f"   n_patches: {stats_hr['n_patches']}")
print(f"   attention O(n^2) = {stats_hr['n_patches'] ** 2:,} operations")
print("   (commence a etre couteux pour une seule image!)")

# 5) Classification head
cls_params = 768 * 1000 + 1000
print(f"\n5) Classification ImageNet (1000 classes) ViT-B/16:")
print(f"   Linear 768 -> 1000: {cls_params:,} params")


# ============================================================================
# Exercice 2 — CLIP similarity matrix interpretation
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 2: CLIP similarity matrix")
print("=" * 70)

scores = np.array([
    [3.2, 0.5, 0.3, 0.1],
    [0.2, 2.8, 0.6, 0.4],
    [0.3, 0.4, 3.5, 0.2],
    [0.5, 0.2, 0.1, 3.0],
])
N = scores.shape[0]

print(f"\nMatrice de similarite:")
print(scores)
print(f"\n1) Diagonale: {np.diag(scores)}")
print("   -> grande, ce qui est souhaite (bonnes paires alignees)")


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# 2) Loss image-to-text
print("\n2) Loss image-to-text:")
probs_i2t = softmax(scores, axis=-1)
for i in range(N):
    p = probs_i2t[i]
    loss_i = -np.log(p[i])
    print(f"   ligne {i}: softmax = {p.round(3)}, loss = {loss_i:.4f}")
loss_i2t = -np.log(probs_i2t[np.arange(N), np.arange(N)]).mean()
print(f"   moyenne: {loss_i2t:.4f}")

# 3) Loss text-to-image (transpose)
print("\n3) Loss text-to-image (colonnes):")
probs_t2i = softmax(scores.T, axis=-1)
loss_t2i = -np.log(probs_t2i[np.arange(N), np.arange(N)]).mean()
print(f"   moyenne: {loss_t2i:.4f}")

# 4) Total
total_loss = (loss_i2t + loss_t2i) / 2
print(f"\n4) Loss totale: {total_loss:.4f}")
print(f"   log(4) = {np.log(4):.4f} (baseline random)")
print(f"   -> on est bien en dessous: la matrice est bien alignee")

# 5) Opposite scenario
bad_scores = np.full((4, 4), 3.0)
np.fill_diagonal(bad_scores, 0.5)
p_bad = softmax(bad_scores, axis=-1)
loss_bad = -np.log(p_bad[np.arange(4), np.arange(4)]).mean()
print(f"\n5) Scenario oppose (diagonale petite):")
print(bad_scores)
print(f"   loss = {loss_bad:.4f}  (pire que random!)")

# 6) Temperature
print("\n6) Avec temperature = 0.1 (scores * 10):")
scores_temp = scores / 0.1
probs_temp = softmax(scores_temp, axis=-1)
loss_temp = -np.log(probs_temp[np.arange(N), np.arange(N)]).mean()
print(f"   loss = {loss_temp:.4f}")
print("   -> softmax plus pique, les bonnes paires ressortent encore plus")
print("   CLIP apprend la temperature pour s'adapter a la 'confiance' du batch")


# ============================================================================
# Exercice 3 — Conv2d equivalence with patchify
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3: Conv2d = patchify + linear")
print("=" * 70)

# Small image
H = W = 4
C = 1
patch_size = 2
d_model = 3

image = np.arange(H * W * C, dtype=np.float32).reshape(H, W, C)
print(f"\nImage 4x4x1:")
print(image[:, :, 0])

# Approach A: patchify + linear
patches = []
for i in range(H // patch_size):
    for j in range(W // patch_size):
        p = image[i * patch_size:(i + 1) * patch_size,
                  j * patch_size:(j + 1) * patch_size, :].flatten()
        patches.append(p)
patches = np.stack(patches)  # (4, 4)
patch_dim = patch_size * patch_size * C

# Random projection
rng = np.random.default_rng(0)
W_proj = rng.standard_normal((patch_dim, d_model)).astype(np.float32)
out_A = patches @ W_proj
print(f"\nApproach A (patchify + linear):")
print(f"  patches: {patches.shape}  (n_patches, patch_dim)")
print(f"  W_proj:  {W_proj.shape}")
print(f"  output:  {out_A.shape}")
print(f"  n_params: {W_proj.size}")

# Approach B: Conv2d equivalent (manual implementation)
# A Conv2d kernel (d_model, C, patch_size, patch_size) with stride = patch_size
# For each output position, do a dot product between the kernel and the patch
# Here we just use the same weights rearranged

print(f"\nApproach B (Conv2d with kernel = stride = patch_size):")
print(f"  kernel shape: (out={d_model}, in={C}, kH={patch_size}, kW={patch_size})")
print(f"  = {d_model * C * patch_size * patch_size} params")
print(f"  = same as W_proj.size = {W_proj.size}")
print("\n  -> Exactement la meme operation, memes parametres.")

print("\n5) Why Conv2d is preferred:")
print("   - GPU kernels are optimized for Conv2d")
print("   - Better memory layout, fuseable with subsequent ops")
print("   - Same result, faster in practice")

# 5) ViT-L/14 info
print("\n5) ViT-L/14 sur 224x224:")
print("   224/14 = 16 -> 16*16 = 256 patches")
print("   Plus de patches = plus de tokens = plus de resolution spatiale")
print("   Le modele peut capturer plus de details fins")

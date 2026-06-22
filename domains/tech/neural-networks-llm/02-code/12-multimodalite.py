"""
Jour 12 — Multimodality: ViT patch embedding + CLIP contrastive loss
=====================================================================
PyTorch if available, NumPy fallback otherwise.

Covers:
  1. ViT patch embedding: image -> patch tokens
  2. CLIP-style contrastive loss on a tiny synthetic dataset
  3. Demonstration of how the loss pushes good pairs up and bad pairs down

Run: python 02-code/12-multimodalite.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.manual_seed(42)
except ImportError:
    HAS_TORCH = False
    print("[info] PyTorch not available — using NumPy-only fallback.")


# ============================================================================
# PART 1: ViT patch embedding (NumPy)
# ============================================================================

print("=" * 70)
print("PART 1: ViT patch embedding — image as tokens")
print("=" * 70)


def patchify(image, patch_size):
    """
    Split an image into patches and flatten each patch.

    Args:
      image: (H, W, C) — HxW pixels, C channels (3 for RGB)
      patch_size: int — side length of a square patch (e.g. 16)

    Returns: (n_patches, patch_size * patch_size * C)

    WHY we flatten: a Transformer expects 1D tokens. Each patch becomes
    a single vector that we will project to d_model.
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image dims must be divisible by patch_size"

    # Number of patches along each axis
    n_h = H // patch_size
    n_w = W // patch_size

    patches = []
    # Iterate in row-major order — top-left to bottom-right
    for i in range(n_h):
        for j in range(n_w):
            patch = image[i * patch_size:(i + 1) * patch_size,
                          j * patch_size:(j + 1) * patch_size, :]
            # Flatten the (patch_size, patch_size, C) cube into a single vector
            patches.append(patch.flatten())

    return np.stack(patches, axis=0)


def patch_embed(patches, W_proj, cls_token, pos_embed):
    """
    Apply the linear projection, prepend a CLS token, add positional embeddings.

    Args:
      patches: (n_patches, patch_dim)
      W_proj:  (patch_dim, d_model)
      cls_token: (d_model,) — learnable CLS vector
      pos_embed: (n_patches + 1, d_model) — learnable positional embeddings

    Returns: (n_patches + 1, d_model) — the sequence fed to the Transformer
    """
    # Linear projection: each patch becomes a d_model vector
    embeddings = patches @ W_proj  # (n_patches, d_model)

    # Prepend the CLS token at position 0
    # The CLS is used for classification: its final representation summarizes
    # the whole image.
    cls_expanded = cls_token[np.newaxis, :]  # (1, d_model)
    embeddings = np.concatenate([cls_expanded, embeddings], axis=0)

    # Add positional embeddings (broadcast)
    embeddings = embeddings + pos_embed
    return embeddings


# Fake image: 32x32 RGB
H = W = 32
C = 3
patch_size = 8
d_model = 64

image = np.random.randn(H, W, C).astype(np.float32)
print(f"\nImage shape: {image.shape}  ({H}x{W}x{C})")

# Patchify
patches = patchify(image, patch_size)
n_patches = patches.shape[0]
patch_dim = patches.shape[1]
print(f"Patches:  {patches.shape}  ({n_patches} patches of {patch_dim}-dim)")
print(f"Nombre de patches attendus: ({H}/{patch_size})^2 = "
      f"{(H // patch_size) ** 2}")

# Build learnable-like weights
W_proj = np.random.randn(patch_dim, d_model).astype(np.float32) * 0.02
cls_token = np.random.randn(d_model).astype(np.float32) * 0.02
pos_embed = np.random.randn(n_patches + 1, d_model).astype(np.float32) * 0.02

# Full patch embedding
tokens = patch_embed(patches, W_proj, cls_token, pos_embed)
print(f"\nSequence finale (CLS + patches): {tokens.shape}")
print("  -> prete a etre passee au Transformer standard")


# ============================================================================
# PART 2: CLIP-style contrastive loss on a tiny synthetic dataset
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: CLIP-style contrastive loss")
print("=" * 70)


def l2_normalize(x, axis=-1, eps=1e-8):
    """
    Normalize vectors to unit L2 norm along the given axis.

    WHY: CLIP compares embeddings with cosine similarity. Pre-normalizing
    lets us use a simple dot product instead of computing norms every time.
    """
    norm = np.sqrt((x ** 2).sum(axis=axis, keepdims=True) + eps)
    return x / norm


def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    CLIP contrastive loss.

    Args:
      image_embeds: (N, d) — image embeddings (already normalized)
      text_embeds:  (N, d) — text embeddings  (already normalized)
      temperature: scalar, controls softmax sharpness (learned in real CLIP)

    Returns:
      loss: scalar — average of image->text and text->image cross-entropy

    WHY symmetric: we want both directions. An image should retrieve its
    text and a text should retrieve its image.
    """
    # Compute the full (N, N) similarity matrix via dot product
    # scores[i, j] = <image_i, text_j>
    scores = image_embeds @ text_embeds.T / temperature

    N = scores.shape[0]
    # The "correct" class for row i is i (diagonal)
    targets = np.arange(N)

    # Compute cross-entropy row-wise (image -> text direction)
    # log-softmax stabilized
    max_scores = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    log_probs_i2t = scores - max_scores - np.log(exp_scores.sum(axis=-1, keepdims=True))
    loss_i2t = -log_probs_i2t[np.arange(N), targets].mean()

    # Same in the text -> image direction (transpose)
    scores_t = scores.T
    max_scores_t = scores_t.max(axis=-1, keepdims=True)
    exp_scores_t = np.exp(scores_t - max_scores_t)
    log_probs_t2i = scores_t - max_scores_t - np.log(exp_scores_t.sum(axis=-1, keepdims=True))
    loss_t2i = -log_probs_t2i[np.arange(N), targets].mean()

    return (loss_i2t + loss_t2i) / 2, scores


# Build a synthetic "dataset": 4 pairs with some structure
# Each pair shares an underlying vector — CLIP should learn to bring them together
N = 4
latent_dim = 16
base = np.random.randn(N, latent_dim).astype(np.float32)

# Image embeddings = base + noise (so pair i has same underlying signal)
image_embeds_raw = base + np.random.randn(N, latent_dim).astype(np.float32) * 0.1
# Text embeddings = base + different noise
text_embeds_raw = base + np.random.randn(N, latent_dim).astype(np.float32) * 0.1

# Normalize
image_embeds = l2_normalize(image_embeds_raw)
text_embeds = l2_normalize(text_embeds_raw)

loss, scores = contrastive_loss(image_embeds, text_embeds, temperature=0.1)
print(f"\nContrastive loss (synth aligned): {loss:.4f}")
print(f"\nSimilarity matrix (diagonal = vraies paires):")
print("       " + " ".join(f"txt_{j}" for j in range(N)))
for i in range(N):
    row = " ".join(f"{scores[i, j]:6.2f}" for j in range(N))
    print(f"  img_{i} [{row} ]")

# Compare with RANDOM pairs (no alignment at all)
random_image = l2_normalize(np.random.randn(N, latent_dim).astype(np.float32))
random_text = l2_normalize(np.random.randn(N, latent_dim).astype(np.float32))
loss_random, _ = contrastive_loss(random_image, random_text, temperature=0.1)
print(f"\nContrastive loss (random non-alignes): {loss_random:.4f}")

print("""
Observation: la loss contrastive est BEAUCOUP plus basse quand les paires
sont alignees. Pendant l'entrainement, le modele est pousse a maximiser
la diagonale de la matrice de similarite et a minimiser le reste.
""")


# ============================================================================
# PART 3: Mini training — show the loss decreases
# ============================================================================

print("=" * 70)
print("PART 3: Mini training of a projection to learn alignment")
print("=" * 70)


def simulate_training(n_pairs=8, n_steps=200, lr=0.05, temperature=0.1):
    """
    Train TWO linear projections (image + text) so the matching pairs
    come together. Pure NumPy, gradient descent by hand.
    """
    d_input = 16
    d_out = 8

    # Simulate "raw features" coming from a vision encoder and a text encoder
    # Each pair shares a hidden underlying vector
    base = np.random.randn(n_pairs, d_input) * 0.5
    raw_image = base + np.random.randn(n_pairs, d_input) * 0.2
    raw_text = base + np.random.randn(n_pairs, d_input) * 0.2

    # Learnable projections
    W_img = np.random.randn(d_input, d_out) * 0.1
    W_txt = np.random.randn(d_input, d_out) * 0.1

    losses = []
    for step in range(n_steps):
        # Forward
        img_emb = raw_image @ W_img
        txt_emb = raw_text @ W_txt
        img_emb = l2_normalize(img_emb)
        txt_emb = l2_normalize(txt_emb)

        loss, scores = contrastive_loss(img_emb, txt_emb, temperature)
        losses.append(loss)

        # Approximate gradient: push the diagonal up, off-diagonal down
        # For a didactic demo, we use a very simple numerical gradient
        eps = 1e-4
        grad_img = np.zeros_like(W_img)
        grad_txt = np.zeros_like(W_txt)

        # Since computing the exact gradient by hand is tedious, we use a
        # simple approach: compute the gradient of the dot-product loss
        # analytically.
        # loss ~ -trace(scores) + log(sum exp scores)
        # For the educational point, we'll do a finite-difference step
        # on a few random directions. This is SLOW but keeps the code clear.
        # (In real code you'd use torch.autograd.)
        if step % 40 == 0:
            print(f"  step {step:3d}  loss = {loss:.4f}  "
                  f"diag mean = {np.diag(scores).mean():.3f}  "
                  f"offdiag mean = "
                  f"{(scores.sum() - np.trace(scores)) / (n_pairs * (n_pairs - 1)):.3f}")

        # Finite-difference gradient approximation (slow but clear)
        for _ in range(10):
            direction_img = np.random.randn(*W_img.shape) * eps
            direction_txt = np.random.randn(*W_txt.shape) * eps

            W_img_new = W_img + direction_img
            W_txt_new = W_txt + direction_txt
            img_emb_new = l2_normalize(raw_image @ W_img_new)
            txt_emb_new = l2_normalize(raw_text @ W_txt_new)
            new_loss, _ = contrastive_loss(img_emb_new, txt_emb_new, temperature)

            # If the random direction improves the loss, step in that direction
            if new_loss < loss:
                W_img = W_img_new
                W_txt = W_txt_new
                loss = new_loss

        # Also apply a lr * (something) update to move faster
        # Simple: shrink W so normalization focuses on direction
        # This is a toy loop — the key point is that the loss decreases

    return losses


losses = simulate_training(n_pairs=8, n_steps=200)
print(f"\nLoss initiale: {losses[0]:.4f}")
print(f"Loss finale:   {losses[-1]:.4f}")
print(f"Reduction:     {(1 - losses[-1] / losses[0]) * 100:.1f}%")

print("""
Observation: avec un simple apprentissage random-search (tres lent), la
diagonale de la matrice de similarite monte et le off-diagonal baisse.
En vrai CLIP, on utilise AdamW sur 400M paires pendant plusieurs jours
sur 256 A100s — les meme equations mais a echelle massive.
""")


# ============================================================================
# PART 4: PyTorch version (if available)
# ============================================================================

if HAS_TORCH:
    print("=" * 70)
    print("PART 4: PyTorch CLIP loss (real gradient descent)")
    print("=" * 70)

    class TinyCLIP(nn.Module):
        """Minimal CLIP with learnable linear projections."""

        def __init__(self, d_input=16, d_out=8):
            super().__init__()
            self.img_proj = nn.Linear(d_input, d_out)
            self.txt_proj = nn.Linear(d_input, d_out)
            # Learnable temperature (logit scale), init log(1/0.07) ~= 2.66
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

        def forward(self, img_feats, txt_feats):
            img = F.normalize(self.img_proj(img_feats), dim=-1)
            txt = F.normalize(self.txt_proj(txt_feats), dim=-1)
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            scores = logit_scale * img @ txt.T
            return scores

    # Build aligned data
    torch.manual_seed(0)
    N = 16
    d_input = 32
    base = torch.randn(N, d_input)
    img_feats = base + torch.randn(N, d_input) * 0.3
    txt_feats = base + torch.randn(N, d_input) * 0.3

    model = TinyCLIP(d_input=d_input, d_out=16)
    optim = torch.optim.AdamW(model.parameters(), lr=0.05)

    print("\nTraining CLIP PyTorch (200 steps):")
    for step in range(200):
        scores = model(img_feats, txt_feats)
        targets = torch.arange(N)
        loss_i = F.cross_entropy(scores, targets)
        loss_t = F.cross_entropy(scores.T, targets)
        loss = (loss_i + loss_t) / 2

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 50 == 0 or step == 199:
            # Accuracy: fraction of rows where argmax is the correct column
            preds = scores.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            print(f"  step {step:3d}  loss = {loss.item():.4f}  "
                  f"top-1 acc = {acc * 100:.1f}%")
else:
    print("\n[info] Skipping PART 4 (no PyTorch).")

print("\n" + "=" * 70)
print("Fin — ViT + CLIP = la fondation des LLMs multimodaux.")
print("=" * 70)

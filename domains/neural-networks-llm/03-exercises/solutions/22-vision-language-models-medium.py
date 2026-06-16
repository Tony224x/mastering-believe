"""
Solutions MEDIUM — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)
=====================================================================
Exercices 4, 5, 6 (medium). Pur NumPy, comme 02-code/22-vision-language-models.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/22-vision-language-models-medium.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


# ----------------------------------------------------------------------------
# Helpers communs (memes definitions que 02-code/22)
# ----------------------------------------------------------------------------
def l2norm(x):
    # Normalisation L2 par ligne : rend le produit scalaire = cosinus.
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def softmax(x, axis=-1):
    # Softmax stable : soustraire le max avant exp pour eviter l'overflow.
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_sigmoid(x):
    # log(1 / (1 + exp(-x))) numeriquement stable via logaddexp.
    return -np.logaddexp(0.0, -x)


# ============================================================================
# EXERCISE 4 — ViT patch embedding from scratch (+ patchify inversible)
# ============================================================================

print("=" * 70)
print("EXERCISE 4 : ViT patch embedding (32x32, patch 8, D=64) + invariance")
print("=" * 70)

H, W, C = 32, 32, 3          # image dims
P = 8                         # patch size (carre)
D = 64                        # token embedding dim

# Image synthetique : pixels RGB aleatoires dans [0, 1].
image = np.random.rand(H, W, C).astype(np.float32)
n_h, n_w = H // P, W // P     # grille 4x4 = 16 patches


def patchify(img, P):
    """
    Decoupe img (H, W, C) en (num_patches, P*P*C) via reshape + transpose.
    POURQUOI le transpose : apres reshape(n_h, P, n_w, P, C), l'axe 1 (lignes
    DANS un patch) et l'axe 2 (colonne de patches) sont entremeles. On ramene
    les deux index de patch (n_h, n_w) en tete AVANT d'aplatir, sinon une ligne
    de la matrice melangerait des pixels de patches voisins.
    """
    H, W, C = img.shape
    n_h, n_w = H // P, W // P
    x = img.reshape(n_h, P, n_w, P, C)        # (n_h, P, n_w, P, C)
    x = x.transpose(0, 2, 1, 3, 4)            # (n_h, n_w, P, P, C)
    return x.reshape(n_h * n_w, P * P * C)    # (num_patches, P*P*C)


def unpatchify(patches, n_h, n_w, P, C):
    """Inverse EXACT de patchify : reconstruit l'image (H, W, C)."""
    x = patches.reshape(n_h, n_w, P, P, C)    # defait l'aplatissement
    x = x.transpose(0, 2, 1, 3, 4)            # (n_h, P, n_w, P, C) = inverse du transpose
    return x.reshape(n_h * P, n_w * P, C)     # (H, W, C)


patches = patchify(image, P)
print(f"\n  image.shape   = {image.shape}")
print(f"  patches.shape = {patches.shape}   (num_patches, P*P*C)")
assert patches.shape == (n_h * n_w, P * P * C) == (16, 192)

# Projection lineaire patch_dim -> D (la "patch embedding" de ViT).
W_proj = np.random.randn(P * P * C, D).astype(np.float32) * 0.02
b_proj = np.zeros(D, dtype=np.float32)
tokens = patches @ W_proj + b_proj
print(f"  tokens.shape  = {tokens.shape}   (num_patches, D)")
assert tokens.shape == (16, D)

# Token [CLS] appris (ici aleatoire) prepend en tete de la sequence.
cls_token = np.random.randn(1, D).astype(np.float32) * 0.02
seq = np.concatenate([cls_token, tokens], axis=0)
print(f"  seq.shape (avec CLS) = {seq.shape}   (num_patches+1, D)")
assert seq.shape == (n_h * n_w + 1, D) == (17, 64)

# Patchify inversible : on reconstruit l'image et on verifie l'egalite exacte.
recon = unpatchify(patches, n_h, n_w, P, C)
ok_inv = np.allclose(recon, image)
print(f"\n  unpatchify(patches) == image ? {ok_inv}")
assert ok_inv, "patchify doit etre inversible (aucune info spatiale perdue)"
print("  -> le decoupage en patches ne perd ni ne melange aucune info spatiale.")


# ============================================================================
# EXERCISE 5 — CLIP InfoNCE contrastive loss from scratch
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5 : CLIP InfoNCE (softmax cross-entropy symetrique)")
print("=" * 70)

N = 8        # batch size
EMB = 32     # dim de l'espace partage

# Embeddings image/texte aleatoires, L2-normalises -> produit scalaire = cosinus.
img = l2norm(np.random.randn(N, EMB).astype(np.float32))
txt = l2norm(np.random.randn(N, EMB).astype(np.float32))
# Sanity : norme ~1.
assert np.allclose(np.linalg.norm(img, axis=1), 1.0, atol=1e-5)

S_rand = img @ txt.T   # (N, N), S[i, j] = cosinus(image_i, texte_j)
assert S_rand.shape == (N, N)


def infonce_loss(S, T=0.07):
    """
    InfoNCE symetrique de CLIP.
    POURQUOI : c'est une classification N-way ou la bonne classe de la ligne i
    est l'indice i (diagonale). Les N-1 autres colonnes sont les negatifs
    (les autres exemples du batch). On moyenne les deux sens i->t et t->i.
    """
    logits = S / T
    # i->t : softmax par ligne, cible = diagonale.
    p_i2t = softmax(logits, axis=-1)
    loss_i2t = -np.mean(np.log(np.diag(p_i2t) + 1e-12))
    # t->i : softmax par colonne (== softmax par ligne de la transposee).
    p_t2i = softmax(logits.T, axis=-1)
    loss_t2i = -np.mean(np.log(np.diag(p_t2i) + 1e-12))
    return 0.5 * (loss_i2t + loss_t2i)


def topk1_matching_acc(S):
    """Top-1 : pour chaque image i, argmax_j S[i,j] doit valoir i."""
    pred = np.argmax(S, axis=1)
    return float(np.mean(pred == np.arange(S.shape[0])))


# Batch "boosted" : chaque texte = copie bruitee de son image pairee -> diagonale haute.
txt_boost = l2norm(0.3 * txt + 0.7 * img)
S_boost = img @ txt_boost.T

loss_rand = infonce_loss(S_rand)
loss_boost = infonce_loss(S_boost)
acc_rand = topk1_matching_acc(S_rand)
acc_boost = topk1_matching_acc(S_boost)

print(f"\n  diag mean (random) = {np.diag(S_rand).mean():.3f}, "
      f"(boosted) = {np.diag(S_boost).mean():.3f}")
print(f"  InfoNCE loss : random = {loss_rand:.4f}, boosted = {loss_boost:.4f}")
print(f"  Top-1 match  : random = {acc_rand:.2%}, boosted = {acc_boost:.2%}")
assert loss_boost < loss_rand, "remonter la diagonale doit BAISSER la loss"
assert acc_boost >= acc_rand, "diagonale plus haute -> meilleur matching"
print("  -> remonter la similarite des vraies paires baisse la loss ET")
print("     ameliore la top-1 accuracy (= zero-shot matching).")


# ============================================================================
# EXERCISE 6 — SigLIP sigmoid pairwise loss + propriete de sharding
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6 : SigLIP sigmoid loss (par paire) + shardability")
print("=" * 70)

# On reutilise le meme batch L2-normalise et la matrice de similarite.
S = S_rand
T_sig = 10.0      # temperature SigLIP (convention scale-up)
b_sig = -10.0     # biais initialise tres negatif (1 positif vs N-1 negatifs)

# labels : +1 sur la diagonale (vraie paire), -1 ailleurs (convention papier SigLIP).
labels = 2 * np.eye(N, dtype=np.float32) - 1
assert labels[0, 0] == 1.0 and labels[0, 1] == -1.0

logits_sig = S * T_sig + b_sig                    # (N, N)
# SigLIP : -mean log sigmoid(label * logit) sur toutes les N*N paires.
siglip_loss = -np.mean(log_sigmoid(labels * logits_sig))
print(f"\n  SigLIP loss (T={T_sig}, b={b_sig}) = {siglip_loss:.4f}")

# --- Shardability : la SOMME des -log_sigmoid est additive sur des morceaux ---
# POURQUOI : chaque paire (i, j) est une classification binaire INDEPENDANTE.
# Il n'y a pas de denominateur global (contrairement au softmax) -> on peut
# decouper le batch, sommer les contributions, et recombiner exactement.
per_pair = -log_sigmoid(labels * logits_sig)      # (N, N) contributions par paire
total_sum = per_pair.sum()

half = N // 2
sum_top = per_pair[:half].sum()                   # moitie haute des lignes
sum_bot = per_pair[half:].sum()                   # moitie basse
sum_sharded = sum_top + sum_bot
print(f"  Somme totale        = {total_sum:.6f}")
print(f"  Somme par moities   = {sum_sharded:.6f}  (top {sum_top:.4f} + bot {sum_bot:.4f})")
assert np.allclose(sum_sharded, total_sum), "la SigLIP loss doit etre shardable"
# Retrouver la moyenne a partir de la somme totale.
assert np.allclose(total_sum / (N * N), siglip_loss)
print("  -> identite verifiee : on peut shard le batch sans synchronisation.")

# --- Contraste : la softmax de CLIP N'EST PAS shardable de cette facon ---
# Si on enleve des colonnes, le denominateur du softmax change -> les
# probabilites de la diagonale changent pour les lignes presentes.
logits_full = S / 0.07
p_full = softmax(logits_full, axis=-1)            # softmax sur TOUTES les colonnes
diag_p_full = np.diag(p_full)

p_half_cols = softmax(logits_full[:, :half], axis=-1)   # softmax sur la moitie des colonnes
# Comparer la proba de la "bonne" colonne pour les lignes < half (diagonale presente).
diag_p_half = np.array([p_half_cols[i, i] for i in range(half)])
max_drift = float(np.max(np.abs(diag_p_full[:half] - diag_p_half)))
print(f"\n  CLIP softmax : drift max des proba diagonales "
      f"(tout vs demi-colonnes) = {max_drift:.4f}")
assert max_drift > 1e-6, "le softmax DOIT changer quand on retire des colonnes"
print("  -> la softmax a une normalisation GLOBALE : retirer des colonnes")
print("     change les probabilites -> non shardable sans tout le batch.")
print("  C'est pourquoi SigLIP scale a des batches de 1M+ (Zhai 2023).")

print("\nDone (MEDIUM).")

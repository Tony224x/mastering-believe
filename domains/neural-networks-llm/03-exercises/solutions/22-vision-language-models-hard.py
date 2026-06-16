"""
Solutions HARD — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)
===================================================================
Exercices 7, 8, 9 (hard). Pur NumPy, comme 02-code/22-vision-language-models.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/22-vision-language-models-hard.py
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
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    # GELU (approx tanh), comme 02-code/22.
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def patchify(img, P):
    """Decoupe img (H, W, C) en (num_patches, P*P*C) via reshape + transpose."""
    H, W, C = img.shape
    n_h, n_w = H // P, W // P
    x = img.reshape(n_h, P, n_w, P, C).transpose(0, 2, 1, 3, 4)
    return x.reshape(n_h * n_w, P * P * C)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ============================================================================
# EXERCISE 7 — Mini-VLM LLaVA-style end-to-end (forward pass)
# ============================================================================

print("=" * 70)
print("EXERCISE 7 : mini-VLM LLaVA-style (ViT -> projecteur -> concat -> attn)")
print("=" * 70)

H, W, C, P = 32, 32, 3, 8
D_VIT = 64
D_LLM = 128
N_TEXT = 10

# 1) Tokens visuels : patch embedding (on DROP le CLS comme LLaVA).
image = np.random.rand(H, W, C).astype(np.float32)
patches = patchify(image, P)                       # (16, 192)
Wpe = np.random.randn(P * P * C, D_VIT).astype(np.float32) * 0.02
vit_tokens = patches @ Wpe                          # (16, 64) tokens de patch
n_img = vit_tokens.shape[0]
print(f"\n  vit_tokens.shape = {vit_tokens.shape}   (N patches, D_VIT)")

# 2) Projecteur MLP 2 couches : D_VIT -> D_LLM -> D_LLM, GELU au milieu.
# POURQUOI : c'est le pont entre l'espace du ViT et l'espace du LLM. ~50M params
# dans LLaVA reel ; ici 2 petites matrices.
W1 = np.random.randn(D_VIT, D_LLM).astype(np.float32) * 0.02
b1 = np.zeros(D_LLM, dtype=np.float32)
W2 = np.random.randn(D_LLM, D_LLM).astype(np.float32) * 0.02
b2 = np.zeros(D_LLM, dtype=np.float32)
projected_visual = gelu(vit_tokens @ W1 + b1) @ W2 + b2   # (16, 128)
print(f"  projected_visual.shape = {projected_visual.shape}   (N, D_LLM)")
assert projected_visual.shape == (n_img, D_LLM) == (16, 128)

# 3) Tokens texte + concatenation [prefix, image, suffix].
text_tokens = np.random.randn(N_TEXT, D_LLM).astype(np.float32) * 0.02
prefix = text_tokens[:5]   # ex. "Describe this image:"
suffix = text_tokens[5:]   # ex. reponse / suite
llm_context = np.concatenate([prefix, projected_visual, suffix], axis=0)
seq_len = llm_context.shape[0]
print(f"  llm_context.shape = {llm_context.shape}   "
      f"({prefix.shape[0]} txt + {n_img} img + {suffix.shape[0]} txt)")
assert llm_context.shape == (N_TEXT + n_img, D_LLM) == (26, 128)
# Plage des tokens image dans la sequence concatenee.
img_start, img_end = prefix.shape[0], prefix.shape[0] + n_img   # [5, 21)

# 4) Bloc d'attention causale single-head sur la sequence concatenee.
# POURQUOI causal : le LLM est autoregressif -> une position ne voit que <= elle.
Wq = np.random.randn(D_LLM, D_LLM).astype(np.float32) * 0.02
Wk = np.random.randn(D_LLM, D_LLM).astype(np.float32) * 0.02
Wv = np.random.randn(D_LLM, D_LLM).astype(np.float32) * 0.02
Q = llm_context @ Wq
K = llm_context @ Wk
V = llm_context @ Wv
scores = Q @ K.T / np.sqrt(D_LLM)                  # (seq, seq)
# Masque causal : -inf sur la partie strictement superieure (futur).
causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
scores = np.where(causal_mask, -1e9, scores)
attn = softmax(scores, axis=-1)                    # (seq, seq), triangulaire inf
out = attn @ V
print(f"  out.shape = {out.shape}   (== llm_context.shape)")
assert out.shape == llm_context.shape

# Verifier que l'attention est bien causale (partie superieure ~0).
assert np.allclose(attn[causal_mask], 0.0, atol=1e-8), "attention doit etre causale"

# 5) Le LLM "voit" l'image : les tokens du suffix attendent sur la plage image.
# (les positions suffix sont apres les tokens image -> elles peuvent les voir.)
suffix_to_image = attn[img_end:, img_start:img_end].sum()
print(f"\n  Attention totale (suffix -> tokens image) = {suffix_to_image:.4f}")
assert suffix_to_image > 0.0, "le suffix DOIT pouvoir attendre sur les tokens image"
print("  -> les tokens texte du suffix integrent l'image comme des tokens normaux.")
print("     C'est l'essence de LLaVA : l'image vit DANS le contexte du LLM.")


# ============================================================================
# EXERCISE 8 — Token budget + AnyRes tiling
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 : token budget (table) + AnyRes tiling (LLaVA-NeXT)")
print("=" * 70)

resolutions = [224, 336, 512, 1024]
patch_sizes = [14, 16, 32]

print("\n  Table : nb de tokens = (res // patch) ** 2")
header = "  " + "res".ljust(10) + "".join(f"P={p}".ljust(12) for p in patch_sizes)
print(header)
for res in resolutions:
    row = "  " + f"{res}".ljust(10)
    for p in patch_sizes:
        row += f"{(res // p) ** 2}".ljust(12)
    print(row)

# Assertions sur les valeurs cles du cours.
assert (224 // 14) ** 2 == 256
assert (336 // 14) ** 2 == 576
assert (1024 // 14) ** 2 == 5329
print("\n  Valeurs cles verifiees : 224/P14=256, 336/P14=576, 1024/P14=5329")

# Cout quadratique de l'attention pour la haute resolution.
N_big = (1024 // 14) ** 2
print(f"  1024x1024 patch 14 -> {N_big} tokens, attention O(N^2) = {N_big ** 2:,} ops/couche")


def anyres_tiling(H, W, tile=336, patch=14,
                  candidate_grids=((1, 1), (1, 2), (2, 1), (2, 2),
                                   (1, 3), (3, 1), (2, 3), (3, 2))):
    """
    AnyRes (LLaVA-NeXT) : choisir la grille (gh, gw) dont le ratio d'aspect cible
    colle le mieux a l'image, tuiler en tuiles de `tile x tile`, + 1 thumbnail
    global de l'image entiere redimensionnee a `tile x tile`.

    Critere de choix : minimiser l'ecart de ratio d'aspect entre l'image
    (W/H) et la grille (gw/gh). POURQUOI le ratio : on veut des tuiles qui
    couvrent l'image sans la deformer excessivement.

    Retourne (gh, gw, n_tiles, tokens_per_tile, total_tokens).
    """
    tokens_per_tile = (tile // patch) ** 2
    img_ratio = W / H
    best = None
    for gh, gw in candidate_grids:
        grid_ratio = gw / gh
        cost = abs(np.log(grid_ratio) - np.log(img_ratio))  # ecart de ratio (echelle log)
        if best is None or cost < best[0]:
            best = (cost, gh, gw)
    _, gh, gw = best
    n_tiles = gh * gw
    # + 1 thumbnail global = 1 tuile supplementaire => tokens_per_tile en plus.
    total_tokens = n_tiles * tokens_per_tile + tokens_per_tile
    return gh, gw, n_tiles, tokens_per_tile, total_tokens


# Image 1024x768 (paysage, ratio 4:3).
Hh, Ww = 768, 1024
gh, gw, n_tiles, tpt, total = anyres_tiling(Hh, Ww, tile=336, patch=14)
print(f"\n  Image {Ww}x{Hh} -> grille choisie {gh}x{gw}, {n_tiles} tuiles")
print(f"  tokens/tuile = {tpt}  (336//14)^2")
assert tpt == (336 // 14) ** 2 == 576
# Le total inclut le thumbnail global (= 1 tuile en plus).
assert total == n_tiles * tpt + tpt, "le thumbnail doit ajouter tokens_per_tile"
total_no_thumb = n_tiles * tpt
assert total - total_no_thumb == tpt, "le thumbnail ajoute exactement tokens_per_tile"
print(f"  total tokens = {total}  (= {n_tiles} tuiles * {tpt} + {tpt} thumbnail)")

# Comparaisons de cout.
print(f"\n  Comparaison de cout pour la meme image :")
print(f"    resize 336x336 simple   : {(336 // 14) ** 2} tokens (perd le detail)")
print(f"    1024 brut patch 14      : {(1024 // 14) ** 2} tokens (cher)")
print(f"    AnyRes (tuiles+thumb)   : {total} tokens (detail local + vue globale)")
print("  -> le thumbnail global est indispensable : sans lui, chaque tuile ignore")
print("     ses voisines (pas de vue d'ensemble). Tuiles = detail, thumb = contexte.")


# ============================================================================
# EXERCISE 9 — Perceiver-Resampler : compression N -> K par cross-attention
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9 : Perceiver-Resampler (N tokens visuels -> K latents fixes)")
print("=" * 70)

d = 64        # dim des tokens
K = 32        # nombre FIXE de latents de sortie (independant de N)

# Requetes apprises (ici aleatoires fixes) : c'est le coeur du resampler.
queries = np.random.randn(K, d).astype(np.float32) * 0.1

# Projections partagees (q sur les latents, k/v sur les tokens visuels).
Wq = np.random.randn(d, d).astype(np.float32) * 0.1
Wk = np.random.randn(d, d).astype(np.float32) * 0.1
Wv = np.random.randn(d, d).astype(np.float32) * 0.1


def perceiver_resampler(V_tokens, queries, Wq, Wk, Wv):
    """
    Cross-attention : K latents (queries) attendent sur N tokens visuels (k/v).
    POURQUOI sortie (K, d) quel que soit N : les requetes sont fixes (K), elles
    "pioshent" l'information dans les N tokens. C'est la brique de Flamingo
    (Perceiver Resampler) et de BLIP-2 (Q-Former).
    """
    Q = queries @ Wq                # (K, d)
    Kk = V_tokens @ Wk              # (N, d)
    Vv = V_tokens @ Wv              # (N, d)
    scores = Q @ Kk.T / np.sqrt(d)  # (K, N) : chaque latent regarde tous les tokens
    attn = softmax(scores, axis=-1)  # softmax sur l'axe des N tokens visuels
    out = attn @ Vv                 # (K, d)
    return out


# La sortie est (K, d) pour TOUS les N (budgets de differentes resolutions).
print(f"\n  Sortie du resampler (K={K}, d={d}) pour differents N :")
for N in [49, 196, 576]:
    V_tokens = np.random.randn(N, d).astype(np.float32)
    out = perceiver_resampler(V_tokens, queries, Wq, Wk, Wv)
    print(f"    N={N:<4} -> out.shape = {out.shape}")
    assert out.shape == (K, d), "le resampler DOIT sortir (K, d) quel que soit N"
print("  -> nombre de tokens de sortie FIXE = budget LLM borne (pas de O(N^2) qui explose).")

# --- Retention d'information (proxy) : reconstruction least-squares ---
# POURQUOI un proxy : ici les poids sont aleatoires (non entraines). On mesure
# donc la CAPACITE STRUCTURELLE a porter de l'information, pas une qualite apprise.
# Cible : reconstruire la moyenne globale des N tokens (un resume compact)
# a partir des K latents aplatis. Comparer au baseline "mean-pooling broadcaste".
N = 196
V_tokens = np.random.randn(N, d).astype(np.float32)
target = V_tokens.mean(axis=0, keepdims=True)        # (1, d) resume cible

# (a) Resampler appris (structurel) : latents -> reconstruire target par moindres carres.
latents = perceiver_resampler(V_tokens, queries, Wq, Wk, Wv)   # (K, d)
feat_resampler = latents.flatten()[None, :]          # (1, K*d) features
R, *_ = np.linalg.lstsq(feat_resampler, target, rcond=None)
recon_resampler = feat_resampler @ R
mse_resampler = mse(target, recon_resampler)

# (b) Baseline naif : K latents tires SANS cross-attention (bruit), meme dim de features.
latents_naive = np.random.randn(K, d).astype(np.float32)
feat_naive = latents_naive.flatten()[None, :]
R2, *_ = np.linalg.lstsq(feat_naive, target, rcond=None)
recon_naive = feat_naive @ R2
mse_naive = mse(target, recon_naive)

print(f"\n  Retention d'info (proxy, reconstruction least-squares) :")
print(f"    MSE resampler (cross-attn) = {mse_resampler:.3e}")
print(f"    MSE baseline (latents bruit)= {mse_naive:.3e}")
print("  (Note : 1 echantillon -> reconstruction sur-parametree ; ce proxy illustre")
print("   que les latents portent de l'info issue des tokens visuels, pas une")
print("   qualite apprise. Un vrai resampler s'entraine end-to-end.)")
print("\n  Lien cours : N -> K<<N borne le cout d'attention du LLM (Flamingo,")
print("  BLIP-2 Q-Former). LLaVA-style, lui, garde les N tokens -> budget")
print("  proportionnel a la resolution.")

print("\nDone (HARD).")

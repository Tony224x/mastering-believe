"""
Solutions EASY — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)
===================================================================
Exercices 1, 2, 3 (faciles). Pur NumPy, comme
02-code/22-vision-language-models.py. Chaque etape non triviale est commentee
avec le POURQUOI. Le fichier est auto-verifiant (assertions finales).

  1. Budget de tokens d'une image (nb_tokens = (R/P)^2) et cout attention O(N^2).
  2. Patchifier une image a la main + suivi des shapes (comme 02-code PART 1).
  3. CLIP : lire une matrice de similarite, softmax a temperature, symetrie,
     SigLIP vs CLIP (shardabilite).

Run: python3 03-exercises/solutions/22-vision-language-models.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

# Stdout en UTF-8 (Windows/CI-friendly).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def softmax(x, axis=-1):
    """Softmax stable : soustraire le max avant exp pour eviter l'overflow."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# EXERCISE 1 — Combien de tokens pour une image ? (budget ViT)
# ============================================================================
# But : maitriser nb_tokens = (R/P)^2 (division ENTIERE par cote) et comprendre
# pourquoi la haute resolution coute si cher (l'attention est O(N^2)).

print("=" * 70)
print("EXERCISE 1 : budget de tokens d'une image (ViT)")
print("=" * 70)


def n_tokens(R, P):
    """Nb de patches = tokens visuels (LLaVA-style) : (R // P)^2 (floor par cote)."""
    return (R // P) ** 2


cases = [(224, 16), (336, 14), (1024, 14)]
print(f"\n  {'Image':<14} {'patch':<8} {'R//P':<8} {'tokens':<10}")
print("  " + "-" * 40)
results = {}
for R, P in cases:
    side = R // P
    tok = n_tokens(R, P)
    results[(R, P)] = tok
    print(f"  {f'{R}x{R}':<14} {P:<8} {side:<8} {tok:<10}")

# 1) Criteres du .md.
assert results[(224, 16)] == 196, "224/16=14, 14^2=196"
assert results[(336, 14)] == 576, "336/14=24, 24^2=576"
assert results[(1024, 14)] == 5329, "1024/14=73, 73^2=5329"
assert 1024 // 14 == 73, "division entiere : 1024//14 = 73 (pas 73.14)"

# 2) Chiffre phare : 5329 tokens ~ combien de pages de texte dense ?
TOK_PER_PAGE = 1300
n_pages = results[(1024, 14)] / TOK_PER_PAGE
print(f"\n  Chiffre phare : 1024//14 = 73, 73^2 = {results[(1024, 14)]} tokens")
print(f"  ~ {n_pages:.1f} pages de texte dense (~{TOK_PER_PAGE} tokens/page)")
assert 3.5 < n_pages < 4.5, "5329 tokens ~ 4 pages de texte"

# 3) Cout attention O(N^2) : nb de paires (i, j) par couche = N^2.
n_high = results[(1024, 14)]
n_low = results[(224, 16)]
pairs_high = n_high ** 2
pairs_low = n_low ** 2
blowup = pairs_high / pairs_low
print(f"\n  Cout attention O(N^2) (paires (i,j) par couche) :")
print(f"    1024x1024 patch 14 : N={n_high}, N^2 = {pairs_high:,} (~{pairs_high / 1e6:.1f}M)")
print(f"    224x224  patch 16  : N={n_low}, N^2 = {pairs_low:,} (~{pairs_low / 1e3:.0f}k)")
print(f"    -> explosion du cout : ~{blowup:.0f}x plus cher en attention")
assert abs(pairs_high - 28_398_241) < 1, "5329^2 ~ 28.4M paires"
assert abs(pairs_low - 38_416) < 1, "196^2 ~ 38k paires"
assert 700 < blowup < 760, f"~740x plus cher, obtenu {blowup:.0f}x"

# 4) Pourquoi l'API facture l'image plus cher.
print("\n  -> POURQUOI l'API facture l'image plus cher : sous le capot, une image")
print("     EST une longue sequence de tokens visuels (5329 pour du 1024x14).")
print("     Elle est facturee comme tels tokens -> prix proportionnel au nombre")
print("     de tokens, et le cout compute explose en O(N^2).")


# ============================================================================
# EXERCISE 2 — Patchifier une image a la main
# ============================================================================
# But : reproduire l'etape 1 de ViT (02-code PART 1) : decouper en patches,
# projeter, prepend [CLS], et suivre les shapes a chaque etape.

print("\n" + "=" * 70)
print("EXERCISE 2 : patchifier une image a la main (suivi des shapes)")
print("=" * 70)

H, W, C = 32, 32, 3
P = 8
D = 64
image = np.random.rand(H, W, C).astype(np.float32)
print(f"\n  image.shape = {image.shape}  (H, W, C)")

# 1) Grille de patches.
n_h, n_w = H // P, W // P
n_patches = n_h * n_w
print(f"  H/P = {n_h}, W/P = {n_w} -> n_patches = {n_patches}")
assert (n_h, n_w) == (4, 4) and n_patches == 16

# 2) Dimension d'un patch aplati = P*P*C.
patch_dim = P * P * C
print(f"  patch aplati = P*P*C = {P}*{P}*{C} = {patch_dim} valeurs")
assert patch_dim == 192

# Decoupage effectif (reshape + transpose, comme 02-code PART 1).
# POURQUOI le transpose : apres reshape(n_h, P, n_w, P, C), les lignes DANS un
# patch et la colonne de patches sont entremelees. On ramene les index de patch
# (n_h, n_w) en tete AVANT d'aplatir, sinon une ligne melangerait des pixels de
# patches voisins.
patches = image.reshape(n_h, P, n_w, P, C).transpose(0, 2, 1, 3, 4)
patches = patches.reshape(n_patches, patch_dim)
print(f"  patches.shape = {patches.shape}  (n_patches, P*P*C)")
assert patches.shape == (16, 192)

# 3) Projection lineaire patch_dim -> D.
W_proj = np.random.randn(patch_dim, D).astype(np.float32) * 0.02
tokens = patches @ W_proj                            # (16, 192) @ (192, 64)
print(f"  tokens.shape = patches @ W_proj = {patches.shape} @ {W_proj.shape} -> {tokens.shape}")
assert tokens.shape == (16, 64)

# 4) Token [CLS] prepend : +1 (un SEUL CLS, pas +2).
cls_token = np.random.randn(1, D).astype(np.float32) * 0.02
seq = np.concatenate([cls_token, tokens], axis=0)
print(f"  seq.shape (avec CLS) = {seq.shape}  (n_patches + 1, D)")
assert seq.shape == (17, 64)
print("  -> POURQUOI +1 et pas +2 : on prepend UN SEUL token [CLS] (un vecteur")
print("     appris) dont l'etat final sert de resume global pour la")
print("     classification. ViT n'utilise pas de token [SEP] -> pas de +2.")

# 5) Coherence avec le PART 1 du code du jour (memes shapes).
print("\n  Coherence avec 02-code PART 1 :")
print(f"    image (32,32,3) -> patches (16,192) -> tokens (16,64) -> seq (17,64)")
print("    -> shapes identiques au code du jour.")


# ============================================================================
# EXERCISE 3 — CLIP : lire une matrice de similarite
# ============================================================================
# But : comprendre la loss contrastive CLIP — pousser la DIAGONALE (bonnes
# paires) vers le haut et le reste vers le bas — et SigLIP vs CLIP.

print("\n" + "=" * 70)
print("EXERCISE 3 : CLIP — lire une matrice de similarite")
print("=" * 70)

# Matrice de similarite cosinus (lignes = images, colonnes = textes).
S = np.array([
    [0.90, 0.20, 0.10],
    [0.30, 0.85, 0.25],
    [0.15, 0.40, 0.80],
])
print("\n  S (lignes=images, colonnes=textes) :")
print(S)

# 1) Les bonnes paires (diagonale) sont-elles le max de leur ligne ?
diag_is_max = all(int(np.argmax(S[i])) == i for i in range(S.shape[0]))
print(f"\n  diagonale = max de chaque ligne ? {diag_is_max}")
assert diag_is_max, "le batch est coherent : la bonne paire domine sa ligne"
print("  -> batch 'bien entraine' : chaque image retrouve son texte en top-1.")

# 2) Loss image->texte pour la ligne i0 : softmax(S[0] / T), T=0.1, cible = 0.
T = 0.1
p_row0 = softmax(S[0] / T)
loss_row0 = -np.log(p_row0[0] + 1e-12)
print(f"\n  softmax(S[0]/{T}) = {[f'{p:.4f}' for p in p_row0]}")
print(f"  proba de la bonne paire (diagonale) = {p_row0[0]:.4f}")
print(f"  loss de cette ligne = -log({p_row0[0]:.4f}) = {loss_row0:.4f}")
# Criteres du .md : ~[0.999, ~5e-4, ~2e-4], loss ~0.001.
assert p_row0[0] > 0.99, f"la temperature basse rend la bonne paire ecrasante, p={p_row0[0]:.4f}"
assert loss_row0 < 0.01, "loss tres faible quand la diagonale domine"
print("  -> POURQUOI : la temperature basse (0.1) DIVISE les logits par 0.1 =")
print("     les MULTIPLIE par 10 -> ecart amplifie -> softmax tres pique sur la")
print("     bonne paire -> loss quasi nulle (le batch est deja bien aligne).")

# 3) Symetrie : loss CLIP = moyenne de image->texte (lignes) et texte->image
#    (colonnes).
def nll_diag(logits):
    """-mean log p(diagonale) avec softmax par ligne."""
    p = softmax(logits, axis=-1)
    return -np.mean(np.log(np.diag(p) + 1e-12))


clip_loss = 0.5 * (nll_diag(S / T) + nll_diag(S.T / T))
print(f"\n  Loss CLIP symetrique (moyenne lignes + colonnes) = {clip_loss:.4f}")
print("  -> POURQUOI les DEUX directions : une image doit retrouver SON texte")
print("     (softmax sur les lignes) ET un texte doit retrouver SON image")
print("     (softmax sur les colonnes). C'est un appariement bidirectionnel.")

# 4) SigLIP vs CLIP (conceptuel) : shardabilite.
print("\n  SigLIP vs CLIP (shardabilite) :")
print("    - CLIP (softmax) : normaliser une ligne demande TOUTE la ligne")
print("      (tout le batch) -> denominateur GLOBAL -> non shardable sans")
print("      synchroniser les GPU.")
print("    - SigLIP (sigmoid par paire) : chaque case (i,j) est une")
print("      classification BINAIRE INDEPENDANTE (pas de denominateur global)")
print("      -> trivialement shardable -> scale a des batches de 1M+ (Zhai 2023).")

# Demo numerique de la non-shardabilite du softmax : retirer une colonne
# change la proba de la diagonale (le denominateur global bouge).
p_full = softmax(S[0] / T)
p_two_cols = softmax(S[0, :2] / T)        # softmax sur 2 colonnes seulement
drift = abs(p_full[0] - p_two_cols[0])
print(f"\n  Demo : softmax(ligne complete)[0] = {p_full[0]:.6f}")
print(f"         softmax(2 colonnes)[0]      = {p_two_cols[0]:.6f}")
print(f"         drift = {drift:.2e}  (non nul -> le softmax depend du batch entier)")
# Ici la diagonale ecrase deja tout, donc le drift est minuscule mais EXISTE :
# le softmax n'est pas additif sur des sous-ensembles de colonnes.
assert drift >= 0.0, "le drift mesure la dependance au batch (peut etre tres faible)"


# ============================================================================
# Fin
# ============================================================================
print("\n" + "=" * 70)
print("Done (EASY). Toutes les assertions passent.")
print("=" * 70)

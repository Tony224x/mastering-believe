# Exercices Medium — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)

---

## Exercice 4 : ViT patch embedding from scratch (et patchify inversible)

### Objectif

Reproduire le tout debut du pipeline ViT : transformer une image en sequence de tokens (patchify -> projection lineaire -> token `[CLS]`), et prouver que le decoupage en patches est **inversible** (on peut reconstruire l'image pixel pour pixel a partir des patches).

### Consigne

1. **Image synthetique** : creer `image` de shape `(H, W, C)` avec `H = W = 32`, `C = 3`, pixels aleatoires dans `[0, 1]`. Choisir un patch carre `P = 8` (donc une grille `4x4 = 16` patches).

2. **Patchify** : decouper l'image en `num_patches = (H/P) * (W/P)` patches via reshape + transpose (pas de boucle Python sur les pixels). Cible : une matrice `patches` de shape `(num_patches, P*P*C)`.
   - Reutiliser le pattern du `02-code/22` : `image.reshape(n_h, P, n_w, P, C).transpose(0, 2, 1, 3, 4).reshape(n_h*n_w, P*P*C)`.
   - Expliquer en commentaire **pourquoi** le transpose est necessaire (sinon les lignes melangent les patches voisins).

3. **Projection lineaire** : une matrice `W_proj` de shape `(P*P*C, D)` avec `D = 64`, plus un biais `b_proj`. Calculer `tokens = patches @ W_proj + b_proj` de shape `(num_patches, D)`.

4. **Token `[CLS]`** : prepend un vecteur appris (ici aleatoire) de shape `(1, D)` en tete de la sequence. Asserter que la sequence finale a la shape `(num_patches + 1, D)`.

5. **Patchify inversible** : ecrire `unpatchify(patches, n_h, n_w, P, C)` qui reconstruit l'image originale a partir de la matrice de patches (inverse exact du reshape/transpose). Asserter `np.allclose(unpatchify(...), image)`.
   - C'est la garantie qu'aucune information spatiale n'est perdue ou melangee par le decoupage.

### Criteres de reussite

- [ ] `patches.shape == (16, 192)` (P*P*C = 8*8*3 = 192) et `tokens.shape == (16, 64)`
- [ ] La sequence avec CLS a la shape `(17, 64)` (assertion explicite)
- [ ] `unpatchify` reconstruit l'image a l'identique : `np.allclose(unpatchify(patches, ...), image)` est `True`
- [ ] Le patchify n'utilise pas de boucle Python sur les pixels (reshape + transpose)
- [ ] Chaque etape non triviale est commentee avec le POURQUOI

---

## Exercice 5 : CLIP InfoNCE contrastive loss from scratch

### Objectif

Implementer la loss contrastive de CLIP (softmax cross-entropy symetrique sur la diagonale d'une matrice de similarite) et verifier deux proprietes : (a) la loss **baisse** quand on remonte la similarite diagonale, (b) la top-1 accuracy de matching (style zero-shot) augmente.

### Consigne

1. **Batch synthetique** : `N = 8` paires (image, texte) dans un espace partage de dim `EMB = 32`. Generer `img` et `txt` aleatoires, puis **L2-normaliser** chaque vecteur (reutiliser `l2norm` du `02-code/22`). Les embeddings normalises rendent le produit scalaire egal au cosinus.

2. **Matrice de similarite** : `S = img @ txt.T` de shape `(N, N)`. `S[i, j]` = cosinus(image_i, texte_j). La diagonale = les vraies paires.

3. **InfoNCE** :
   - Temperature `T = 0.07` (defaut CLIP), `logits = S / T`.
   - `softmax` stable (soustraire le max par ligne avant `exp`).
   - Loss image->texte = cross-entropy ou la cible de la ligne `i` est l'indice `i` (diagonale) : `-mean(log(softmax(logits)[i, i]))`.
   - Loss texte->image = la meme chose sur `logits.T` (par colonne).
   - `clip_loss = 0.5 * (loss_i2t + loss_t2i)`.

4. **La loss baisse quand la diagonale monte** : construire un second batch ou chaque texte est une copie bruitee de son image pairee (`txt2 = l2norm(0.3*txt + 0.7*img)`), donc la diagonale de `S` est plus haute. Asserter `clip_loss(S_boosted) < clip_loss(S_random)`.

5. **Top-1 accuracy (style zero-shot)** : pour chaque image `i`, predire `argmax_j S[i, j]`. La prediction est correcte si elle vaut `i`. Calculer l'accuracy sur le batch random et sur le batch boosted, et montrer qu'elle est plus haute sur le batch boosted.

### Criteres de reussite

- [ ] `S.shape == (N, N)` et les embeddings sont bien L2-normalises (`norm ≈ 1`)
- [ ] La loss InfoNCE est symetrique (moyenne des deux sens i2t et t2i)
- [ ] Assertion `clip_loss(boosted) < clip_loss(random)` verifiee
- [ ] La top-1 accuracy de matching est calculee et plus elevee sur le batch boosted
- [ ] Le softmax est numeriquement stable (max-subtraction)
- [ ] Code commente avec le POURQUOI (negatifs = autres exemples du batch)

---

## Exercice 6 : SigLIP sigmoid pairwise loss + propriete de sharding

### Objectif

Implementer la loss sigmoid par paire de SigLIP (avec biais appris) et demontrer numeriquement sa propriete cle : elle est **shardable** (calculable par morceaux du batch puis recombinee), contrairement a la softmax globale de CLIP.

### Consigne

1. **Reutiliser le batch** : meme `img` / `txt` L2-normalises que l'exercice 5, meme matrice `S = img @ txt.T`.

2. **Loss SigLIP** :
   - `labels = 2 * np.eye(N) - 1` (`+1` sur la diagonale = vraie paire, `-1` ailleurs — convention du papier SigLIP).
   - `logits = S * T_sig + b_sig` avec `T_sig = 10.0` (temperature scale-up) et `b_sig = -10.0` (biais initialise tres negatif pour compenser le desequilibre 1 positif vs N-1 negatifs).
   - `log_sigmoid(x) = -np.logaddexp(0.0, -x)` (version stable, comme `02-code/22`).
   - `siglip_loss = -mean(log_sigmoid(labels * logits))` sur les `N*N` paires.

3. **Shardability (l'identite cle)** : la somme `-sum(log_sigmoid(labels * logits))` peut etre calculee en additionnant deux moities du batch, car il n'y a **aucune normalisation globale** (chaque paire est independante).
   - Decouper les lignes de `S` en deux moities, calculer la somme des `-log_sigmoid(...)` sur chaque moitie, puis additionner.
   - Asserter que `somme_moitie_haute + somme_moitie_basse == somme_totale` (au float-error pres, `np.allclose`).
   - Retrouver la loss moyenne en divisant la somme totale par `N*N`.

4. **Contraste conceptuel avec CLIP** : montrer numeriquement que la **softmax de CLIP n'est PAS shardable** ainsi : recalculer le softmax sur une demi-matrice (moins de colonnes) change les probabilites des lignes presentes (le denominateur depend des colonnes presentes). Imprimer la difference des probabilites de la diagonale entre "softmax sur tout" et "softmax sur la moitie des colonnes" pour rendre l'argument concret.

### Criteres de reussite

- [ ] `labels` vaut `+1` sur la diagonale, `-1` ailleurs (assertion sur quelques entrees)
- [ ] `log_sigmoid` est implementee de facon stable via `logaddexp`
- [ ] La loss SigLIP est calculee comme `-mean(log_sigmoid(labels*logits))`
- [ ] Identite de sharding verifiee : somme par moities == somme totale (`np.allclose`)
- [ ] Demonstration que la softmax CLIP change quand on enleve des colonnes (non-shardable)
- [ ] Code commente : sigmoid = N*N classifications binaires independantes -> pas de sync cross-batch

# Exercices Medium — Jour 12 : Multimodalite

---

## Exercice 4 : ViT patch embedding == Conv2d (prouver l'equivalence)

### Objectif

Implementer la patchification de ViT de deux manieres (decoupe+projection lineaire vs convolution stride=patch) et prouver numeriquement qu'elles sont identiques.

### Consigne

1. Reprendre `patchify` et `patch_embed` du code (`02-code/12-multimodalite.py`).

2. Implementer une **Conv2d "from scratch"** (NumPy) avec `kernel_size = stride = patch_size` et `out_channels = d_model`, en row-major (top-left → bottom-right).

3. **Equivalence** : construire une image `(H, W, C)` et une matrice de projection `W_proj (patch_dim, d_model)`. Montrer que :
   - Approche A : `patchify` puis `patches @ W_proj` → `(n_patches, d_model)`
   - Approche B : `conv2d(image, kernel)` ou `kernel` est `W_proj` reshapee en `(d_model, C, patch, patch)`, puis aplatie en `(n_patches, d_model)`
   - Les deux donnent EXACTEMENT le meme resultat (ecart < 1e-10)
   - **Attention a l'ordre de flatten** : il faut que l'ordre des elements dans le patch corresponde a l'ordre dans le kernel. Documenter la convention choisie.

4. **Compte de parametres** : verifier que les deux approches ont exactement le meme nombre de parametres (`patch_dim * d_model`).

5. Calculer pour ViT-B/16 (224x224, patch 16, d=768) : n_patches, patch_dim, params de la projection. Puis ajouter CLS + positional embeddings et donner la shape de la sequence finale.

6. Analyser : pourquoi tous les frameworks (timm, HuggingFace) implementent la patchification comme une Conv2d plutot qu'avec une boucle Python ?

### Criteres de reussite

- [ ] La Conv2d from scratch est correcte (stride=kernel, pas de chevauchement)
- [ ] Les deux approches donnent un resultat identique (ecart < 1e-10)
- [ ] La convention de flatten (ordre des dims) est documentee et coherente
- [ ] Le compte de parametres est identique dans les deux cas
- [ ] Les chiffres ViT-B/16 sont corrects (196 patches, patch_dim 768)

---

## Exercice 5 : CLIP contrastive loss + gradient + accuracy retrieval

### Objectif

Implementer la loss contrastive symetrique de CLIP avec son gradient analytique, et mesurer la qualite de l'alignement via une accuracy de retrieval (top-1 et top-k).

### Consigne

1. Reprendre `contrastive_loss` du code. Implementer en plus la **temperature apprise** : `logit_scale` (un scalaire) tel que `scores = logit_scale * (img @ txt.T)`.

2. Implementer le **gradient de la loss CLIP** par rapport a la matrice de scores `S` :
   - Pour la direction image→texte (cross-entropy par ligne) : `dL_i2t/dS = (softmax_row(S) - onehot_diag) / N`
   - Pour texte→image (par colonne) : symetrique sur `S.T`
   - La loss totale est la moyenne des deux → gradient = moyenne des deux gradients
   - Verifier ce gradient par difference finie (ecart < 1e-5)

3. **Retrieval accuracy** : sur un batch de N paires alignees + bruit, mesurer :
   - top-1 image→texte : fraction des lignes ou `argmax(S[i]) == i`
   - top-5 image→texte
   - Faire de meme texte→image

4. **Mini training** : entrainer deux projections lineaires (image et texte) par descente de gradient (gradient analytique, pas finite-difference !) pour aligner des features synthetiques partageant un signal latent. Afficher loss + top-1 accuracy au fil des steps. Montrer que l'accuracy monte vers ~100%.

5. **Effet de la temperature** : pour une matrice de scores fixee, montrer comment baisser la temperature (= augmenter logit_scale) rend le softmax plus pique. Pourquoi CLIP APPREND la temperature au lieu de la fixer ?

### Criteres de reussite

- [ ] La loss CLIP symetrique est correcte (moyenne i2t + t2i)
- [ ] Le gradient analytique de la loss passe le test de difference finie (< 1e-5)
- [ ] L'accuracy retrieval top-1 et top-5 est calculee dans les deux directions
- [ ] Le mini training (gradient analytique) fait monter l'accuracy vers ~100%
- [ ] L'effet de la temperature sur la nettete du softmax est demontre et explique

---

## Exercice 6 : LLaVA-style — projeter des features visuelles en tokens texte

### Objectif

Implementer l'approche LLaVA (option 1 du cours) : un projecteur qui transforme des features visuelles (sortie d'un encoder type CLIP) en "tokens visuels" injectables dans la sequence d'un LLM.

### Consigne

1. Simuler la sortie d'un encoder visuel : un tenseur `(n_patches, d_vision)` (ex: 49 patches, d_vision=512) representant une image.

2. Implementer un **projecteur** (l'equivalent du connector de LLaVA) :
   - Version simple : un `Linear(d_vision, d_model)` → `(n_patches, d_model)`
   - Version MLP (LLaVA 1.5) : `Linear → GELU → Linear` → `(n_patches, d_model)`
   - Ces `n_patches` vecteurs deviennent des "tokens visuels" de dimension `d_model`, comme s'ils etaient des embeddings de mots.

3. **Assembler la sequence multimodale** : construire la sequence d'entree du LLM en concatenant :
   ```
   [emb(<image>), tok_vis_1, ..., tok_vis_49, emb(<\image>), emb(mot_1), ..., emb(mot_k)]
   ```
   ou les `tok_vis` sont les tokens visuels projetes et les `emb(mot)` viennent d'une table d'embeddings texte (table aleatoire pour la demo).

4. Verifier les shapes : la sequence finale doit etre `(n_patches + n_text_tokens + 2, d_model)` et etre prete a passer dans un Transformer standard (causal).

5. **Compte de parametres du connector** : combien de params pour le Linear simple vs le MLP ? Pourquoi LLaVA gele le LLM et l'encoder visuel et n'entraine (au debut) QUE le connector ?

6. Analyser : quel est l'avantage de l'approche LLaVA (tokens texte) par rapport a la cross-attention de Flamingo ? Quel est le cout (sequence plus longue) ?

### Criteres de reussite

- [ ] Le projecteur Linear et le projecteur MLP sont implementes correctement
- [ ] Les tokens visuels ont la bonne dimension `d_model`
- [ ] La sequence multimodale est assemblee avec les bonnes shapes (image tokens + texte)
- [ ] Le compte de parametres du connector est correct
- [ ] L'analyse LLaVA vs Flamingo est juste (simplicite vs cout en longueur de sequence)

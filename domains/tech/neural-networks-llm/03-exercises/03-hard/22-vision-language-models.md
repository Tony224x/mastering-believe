# Exercices Hard — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)

---

## Exercice 7 : Mini-VLM LLaVA-style end-to-end (forward pass complet)

### Objectif

Assembler le pipeline complet d'un VLM LLaVA-style en numpy : ViT -> tokens visuels -> projecteur MLP 2 couches (GELU) vers l'espace du LLM -> concatenation `[texte_prefix, image_tokens, texte_suffix]` -> un bloc d'**attention causale** sur la sequence concatenee, pour montrer que le LLM "voit" les tokens image comme des tokens normaux. Verifier la comptabilite des tokens par assertions.

### Consigne

1. **Tokens visuels** : reprendre le patch embedding de l'exercice 4 (image `32x32`, patch `8`, `D_VIT = 64`) pour obtenir `vit_tokens` de shape `(16, 64)`. Comme LLaVA, on **drop le `[CLS]`** et on garde les 16 tokens de patch.

2. **Projecteur MLP 2 couches** : `D_VIT -> D_LLM -> D_LLM` avec un `D_LLM = 128`, GELU au milieu (reutiliser le `gelu` du `02-code/22`).
   - `projected_visual = gelu(vit_tokens @ W1 + b1) @ W2 + b2`, shape `(16, 128)`.
   - Asserter `projected_visual.shape == (16, D_LLM)`.

3. **Tokens texte + concatenation** : `N_TEXT = 10` tokens texte (embeddings synthetiques dans `D_LLM`). Decouper en `prefix` (ex. 5 tokens, le prompt "Describe this image:") et `suffix` (5 tokens). Construire `llm_context = concat([prefix, projected_visual, suffix])`.
   - **Comptabilite des tokens** : asserter `llm_context.shape == (N_TEXT + 16, D_LLM)` et imprimer le decompte `prefix + image + suffix`.

4. **Bloc d'attention causale** sur `llm_context` (single-head, from scratch) :
   - `Q = X @ Wq`, `K = X @ Wk`, `V = X @ Wv` (toutes `D_LLM -> D_LLM`).
   - `scores = Q @ K.T / sqrt(D_LLM)`, masque **causal** (triangulaire inferieure : une position ne regarde que les positions <= elle).
   - `attn = softmax(scores masque)`, `out = attn @ V`, shape `(seq_len, D_LLM)`.
   - Asserter `out.shape == llm_context.shape`.

5. **Le LLM voit l'image comme du texte** : verifier que les positions **texte du suffix** (apres les tokens image) ont une attention **non nulle** sur les positions des tokens image (somme des poids d'attention vers la plage image > 0). C'est la preuve operationnelle que l'image est integree dans le contexte autoregressif comme n'importe quel token.

### Criteres de reussite

- [ ] `projected_visual.shape == (16, 128)` (assertion)
- [ ] `llm_context.shape == (26, 128)` avec le decompte 5 + 16 + 5 imprime (assertion sur le total)
- [ ] L'attention est **causale** (matrice de poids strictement triangulaire inferieure non nulle ; partie superieure = 0)
- [ ] `out.shape == llm_context.shape` (assertion)
- [ ] Les tokens du suffix attendent (poids > 0) sur la plage des tokens image
- [ ] Code commente avec le POURQUOI (projecteur = pont entre espaces, concat = recette LLaVA)

---

## Exercice 8 : Token budget + AnyRes tiling (reproduction de la table + tuilage)

### Objectif

Reproduire le calcul du budget de tokens visuels (la table du cours : resolution x patch), puis implementer le tuilage AnyRes de LLaVA-NeXT (choix de la grille la plus proche, tuiles de resolution fixe + thumbnail global) et verifier le compte total de tokens par assertion.

### Consigne

1. **Table du budget** : pour `resolutions = [224, 336, 512, 1024]` et `patches = [14, 16, 32]`, calculer `n_tokens = (res // p) ** 2` et afficher la table.
   - Verifier par assertion les valeurs cles du cours : `(224 // 14)**2 == 256`, `(336 // 14)**2 == 576`, `(1024 // 14)**2 == 5329`.

2. **Cout quadratique** : pour l'image `1024x1024` patch `14` (5329 tokens), imprimer le cout d'attention `O(N^2)` par couche (`5329 ** 2`) pour rendre concret pourquoi la haute resolution coute cher.

3. **AnyRes tiling** : implementer `anyres_tiling(H, W, tile=336, patch=14, candidate_grids=[(1,1),(1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2)])` qui :
   - choisit la grille `(gh, gw)` dont la resolution cible `(gh*tile, gw*tile)` a le ratio d'aspect le plus proche de l'image (minimiser une distance sur le ratio, ou sur l'aire effective utile — documenter le critere choisi) ;
   - calcule `n_tiles = gh * gw` ;
   - calcule `tokens_per_tile = (tile // patch) ** 2` ;
   - ajoute **un thumbnail global** = 1 tuile `tile x tile` de l'image entiere redimensionnee, soit `tokens_per_tile` tokens supplementaires ;
   - retourne `(gh, gw, total_tokens)` avec `total_tokens = n_tiles * tokens_per_tile + tokens_per_tile`.

4. **Assertion sur le compte** : pour une image `1024x768`, asserter que `total_tokens == n_tiles * tokens_per_tile + tokens_per_tile` (le thumbnail est bien compte), et imprimer la grille choisie + le total.
   - Verifier aussi que le thumbnail ajoute exactement `tokens_per_tile` tokens (compte sans thumbnail == compte avec thumbnail moins `tokens_per_tile`).

5. **Analyse** : pourquoi le thumbnail global est indispensable (chaque tuile ignore ses voisines -> sans thumbnail, pas de vue d'ensemble) ? Comparer le total AnyRes au cout d'un simple redimensionnement `336x336` (576 tokens) et au cout brut `1024x1024` patch 14 (5329 tokens).

### Criteres de reussite

- [ ] La table reproduit `256` / `576` / `5329` (assertions sur ces 3 valeurs)
- [ ] `anyres_tiling` choisit une grille coherente avec l'aspect ratio de l'image
- [ ] `tokens_per_tile == (336 // 14) ** 2 == 576`
- [ ] Assertion : `total_tokens == n_tiles * tokens_per_tile + tokens_per_tile` (thumbnail inclus)
- [ ] Le total AnyRes est compare au `336x336` simple et au `1024` brut
- [ ] L'analyse explique le role du thumbnail global

---

## Exercice 9 : Perceiver-Resampler — compression N -> K tokens par cross-attention

### Objectif

Implementer un resampler a la Perceiver/Q-Former : `K` requetes apprises (latents) cross-attendent sur les `N` tokens visuels (cles/valeurs) pour produire **exactement `K` tokens** quel que soit `N`. Asserter que la sortie est `(K, d)` independamment de `N`, et mesurer la retention d'information comme proxy (reconstruction lineaire des tokens d'origine depuis les `K` latents).

### Consigne

1. **Tokens visuels variables** : generer des tokens visuels `V` de shape `(N, d)` avec `d = 64`, pour plusieurs `N` (ex. `49`, `196`, `576` — les budgets de differentes resolutions). Ce sont les cles/valeurs.

2. **Requetes apprises** : `K = 32` latents `queries` de shape `(K, d)` (parametre appris, ici aleatoire fixe). C'est le nombre **fixe** de tokens de sortie, independant de `N`.

3. **Cross-attention resampler** (single-head, from scratch) :
   - Projections `Wq, Wk, Wv` (`d -> d`).
   - `Q = queries @ Wq` (shape `(K, d)`), `Kk = V @ Wk` (shape `(N, d)`), `Vv = V @ Wv` (shape `(N, d)`).
   - `scores = Q @ Kk.T / sqrt(d)` (shape `(K, N)`) — chaque latent regarde tous les tokens visuels (pas de masque causal : compression globale).
   - `attn = softmax(scores, axis=-1)`, `out = attn @ Vv` (shape `(K, d)`).
   - **Assertion cle** : `out.shape == (K, d)` pour TOUS les `N` testes (49, 196, 576). C'est la propriete "nombre de tokens fixe" de Flamingo/BLIP-2.

4. **Retention d'information (proxy)** : on veut savoir si les `K` latents preservent assez d'information sur les `N` tokens. Mesurer par **reconstruction lineaire least-squares** :
   - resoudre `R` minimisant `|| (out_flat) @ R - V_target ||` ou l'on reconstruit une cible compacte des tokens visuels (ex. la moyenne des tokens, ou un pooling) a partir des latents aplatis.
   - Comparer la MSE de reconstruction du resampler appris a un baseline naif (mean-pooling des N tokens broadcaste sur K, ou K latents tires sans cross-attention). Imprimer le ratio.
   - Documenter que c'est un proxy pedagogique (un vrai resampler est entraine end-to-end ; ici les poids sont aleatoires, donc on mesure la capacite structurelle, pas la qualite apprise).

5. **Lien avec le cours** : expliquer pourquoi cette compression `N -> K << N` (a) borne le cout d'attention du LLM en aval (K fixe, pas O(N^2) qui explose), (b) est la brique de Flamingo (Perceiver Resampler) et BLIP-2 (Q-Former). Contraster avec LLaVA-style (qui garde les N tokens, donc budget proportionnel a la resolution).

### Criteres de reussite

- [ ] `out.shape == (K, d)` pour `N in {49, 196, 576}` (assertion pour chaque `N`)
- [ ] La cross-attention est correcte : `scores` shape `(K, N)`, softmax sur l'axe des tokens visuels
- [ ] La retention d'information est mesuree par reconstruction least-squares et comparee a un baseline
- [ ] Le caractere "proxy pedagogique" (poids aleatoires non entraines) est explicite
- [ ] L'analyse relie le resampler a Flamingo/BLIP-2 et au controle du budget tokens
- [ ] Code commente avec le POURQUOI (queries fixes -> sortie de taille fixe quel que soit N)

# Exercices Faciles — Jour 22 : Vision-language models (ViT, CLIP, LLaVA)

---

## Exercice 1 : Combien de tokens pour une image ? (budget ViT)

### Objectif

Maitriser le calcul de base du cout d'une image dans un VLM : `nb_tokens = (H/P) * (W/P)`, et comprendre pourquoi la haute resolution coute si cher.

### Consigne

Pour une image carree `R x R` decoupee en patches `P x P`, le nombre de patches (= tokens visuels en LLaVA-style) est `(R/P)^2`.

1. **Calculer** le nombre de tokens pour :
   - `224 x 224`, patch `16` -> ?
   - `336 x 336`, patch `14` -> ?
   - `1024 x 1024`, patch `14` -> ?

2. **Verifier le chiffre phare du cours** : `1024 / 14 = 73` (division entiere), `73^2 = 5329` tokens. Combien de "pages de texte dense" cela represente-t-il (~1300 tokens/page) ?

3. **Cout attention** : l'attention est `O(N^2)`. Pour `N = 5329` tokens d'image, combien de paires `(i, j)` la self-attention calcule-t-elle par couche (`N^2`) ? Comparer a une image `224x224` patch `16` (196 tokens). De combien de fois le cout explose-t-il ?

4. **Pourquoi l'API facture l'image plus cher** : en une phrase, relier le nombre de tokens visuels au prix.

### Criteres de reussite

- [ ] `224/16 = 14`, `14^2 = 196` tokens
- [ ] `336/14 = 24`, `24^2 = 576` tokens
- [ ] `1024/14 = 73`, `73^2 = 5329` tokens (≈ 4 pages de texte)
- [ ] `N^2` : `5329^2 ≈ 28.4M` paires vs `196^2 ≈ 38k` -> ~740x plus cher en attention
- [ ] Comprehension : sous le capot une image = beaucoup de tokens -> facturee comme tels

---

## Exercice 2 : Patchifier une image a la main

### Objectif

Reproduire l'etape 1 de ViT (PART 1 du code du jour) : decouper une image en patches et compter les dimensions a chaque etape.

### Consigne

Image `H x W x C = 32 x 32 x 3`, patch `P = 8`, dim de projection `D = 64`.

1. **Grille de patches** : combien de patches par cote (`H/P`, `W/P`) ? Combien de patches au total `n_patches` ?

2. **Dimension d'un patch aplati** : un patch `P x P x C`, combien de valeurs contient-il (`P*P*C`) ?

3. **Projection lineaire** : on multiplie la matrice `patches (n_patches, P*P*C)` par `W_proj (P*P*C, D)`. Quelle est la shape de `tokens` apres projection ?

4. **Token [CLS]** : on prepend un token appris. Quelle est la shape de la sequence finale `seq` ? Pourquoi `+1` et pas `+2` ?

5. **Coherence** : verifier que tes shapes correspondent a la sortie du code `02-code/22-vision-language-models.py` (PART 1).

### Criteres de reussite

- [ ] `H/P = 4`, `W/P = 4` -> `n_patches = 16`
- [ ] Patch aplati = `8*8*3 = 192` valeurs
- [ ] `patches (16, 192) @ W_proj (192, 64)` -> `tokens (16, 64)`
- [ ] `seq` avec CLS = `(17, 64)` (un seul CLS prepend)
- [ ] Les shapes matchent le PART 1 du code du jour

---

## Exercice 3 : CLIP — lire une matrice de similarite

### Objectif

Comprendre la structure de la loss contrastive CLIP : pousser la diagonale (bonnes paires) vers le haut et le reste vers le bas.

### Consigne

Batch de 3 paires (image_i, texte_i). Matrice de similarite cosinus `S` (lignes = images, colonnes = textes) :

```
        t0     t1     t2
i0 [  0.90   0.20   0.10 ]
i1 [  0.30   0.85   0.25 ]
i2 [  0.15   0.40   0.80 ]
```

1. **Les bonnes paires** sont sur la **diagonale** (`i0-t0`, `i1-t1`, `i2-t2`). Verifier qu'elles ont bien la similarite la plus haute de leur ligne. Le batch est-il "bien entraine" ?

2. **Loss image->texte** : pour la ligne `i0`, on applique `softmax(S[0] / T)` avec `T = 0.1` et la cible est l'index 0 (la diagonale). Calculer `softmax(S[0] / 0.1)` puis la proba de la bonne paire. La loss de cette ligne est `-log(p_diagonale)`.

3. **Symetrie** : la loss CLIP est la moyenne de la loss image->texte (softmax sur les **lignes**) et texte->image (softmax sur les **colonnes**). Pourquoi les deux directions ?

4. **SigLIP vs CLIP (conceptuel)** : la loss CLIP a besoin de **toute la ligne** (donc tout le batch) pour normaliser le softmax. La loss SigLIP traite chaque case `(i,j)` comme une classification binaire independante. Laquelle est triviale a sharder sur plusieurs GPU ? Pourquoi ?

### Criteres de reussite

- [ ] La diagonale est bien le max de chaque ligne -> batch coherent
- [ ] `softmax(S[0]/0.1)` ≈ `[0.999, ~5e-4, ~2e-4]` : la temperature basse (0.1) rend la bonne paire ecrasante, loss `-log(0.999) ≈ 0.001`
- [ ] Les deux directions (lignes + colonnes) car une image doit retrouver son texte ET inversement
- [ ] Comprehension : SigLIP (sigmoid par paire, pas de normalisation globale) est shardable ; CLIP (softmax global) demande tout le batch -> SigLIP scale a 1M+

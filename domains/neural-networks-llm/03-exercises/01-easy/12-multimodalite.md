# Exercices Faciles — Jour 12 : Multimodalite

---

## Exercice 1 : ViT — dimensions et comptes

### Objectif

Savoir calculer le nombre de patches et la taille du vocabulaire visuel d'un ViT pour differentes configurations.

### Consigne

1. **ViT-B/16** (base, patch size 16) :
   - Image 224x224 RGB
   - Patch size 16x16
   - Combien de patches ?
   - Dimension de chaque patch avant projection : ?
   - Si d_model = 768, combien de parametres dans la projection lineaire ?

2. **ViT-L/14** (large, patch size 14) :
   - Image 224x224 RGB
   - Patch size 14x14 (224 n'est pas divisible par 14, donc on utilise 196 → ajuster a 238 ou 210)
   - Prendre 224/14 = 16, donc grille 16x16 = 256 patches
   - Dimension par patch : 14*14*3 = ?

3. **Comparer avec texte** : si un LLM a un context length de 4096 tokens, combien d'images de ViT-L/14 peux-tu y faire tenir (apres avoir ajoute le CLS) ? Et en ViT-B/32 (patch 32) ?

4. **Sequence length scaling** : pour une image de 512x512 (haute resolution), combien de patches avec patch size 16 ? Quelle est la complexite de l'attention sur cette sequence ?

5. **Classification head** : combien de parametres pour une tete de classification ImageNet (1000 classes) sur ViT-B/16 ? (Indice : juste un Linear de d_model vers 1000)

### Criteres de reussite

- [ ] ViT-B/16 : (224/16)² = 196 patches, patch_dim = 16*16*3 = 768
- [ ] Projection lineaire : 768 * 768 = 589 824 params
- [ ] ViT-L/14 : patch_dim = 14*14*3 = 588
- [ ] Dans 4096 tokens : ~15 images de ViT-L/14 (256+1 each), ~55 images de ViT-B/32 (49+1 each)
- [ ] 512x512 avec patch 16 : 1024 patches, attention O(n²) = ~1M operations — commence a etre cher
- [ ] Classification head : 768 * 1000 + 1000 = 769 000 params

---

## Exercice 2 : CLIP contrastive loss — interpreter la matrice

### Objectif

Comprendre la matrice de similarite de CLIP et savoir lire les signes de bonne ou mauvaise convergence.

### Consigne

Soit un batch de 4 paires (image, text) avec la matrice de similarite `(N, N)` suivante (scores avant softmax, temperature = 1.0) :

```
         txt_1  txt_2  txt_3  txt_4
img_1  [  3.2    0.5    0.3    0.1 ]
img_2  [  0.2    2.8    0.6    0.4 ]
img_3  [  0.3    0.4    3.5    0.2 ]
img_4  [  0.5    0.2    0.1    3.0 ]
```

1. **Observation** : les valeurs de la diagonale sont-elles grandes ou petites comparees au reste ? Est-ce ce qu'on veut ?

2. **Loss image-to-text** : pour chaque ligne i, calculer `softmax(row_i)` et prendre `-log(softmax[i][i])`.
   - Faire le calcul pour la ligne 1 : softmax = [?, ?, ?, ?], loss_1 = ?
   - Additionner les 4 loss et diviser par 4.

3. **Loss text-to-image** : meme chose mais sur les colonnes.

4. **Loss totale** : moyenne des deux. Est-ce une bonne loss (proche de 0) ou mauvaise (proche de log(4) ≈ 1.39) ?

5. **Scenario oppose** : maintenant, imagine que la diagonale contient des petites valeurs (0.5) et que toutes les autres sont grandes (3.0). Quelle serait la loss ? Le modele est-il "bon" ou "mauvais" ?

6. **Temperature** : si on passait la matrice par `scores / 0.1` (temperature basse), comment le softmax change-t-il ? Pourquoi CLIP apprend la temperature ?

### Criteres de reussite

- [ ] Diagonale grande → les images et textes de meme paire sont proches (bon signe)
- [ ] Ligne 1 : softmax ≈ [0.93, 0.02, 0.02, 0.01], loss_1 ≈ 0.07
- [ ] Loss totale ≈ 0.1-0.15 (proche de 0, tres bon)
- [ ] Scenario oppose : loss proche de log(4) = 1.39 (baseline uniforme)
- [ ] Temperature basse → softmax plus pique → differences amplifiees. CLIP apprend la temperature pour ajuster la "confiance" selon le batch.

---

## Exercice 3 : Tokeniser une image avec un CNN simple

### Objectif

Comprendre que la "patchification" de ViT n'est en fait qu'une convolution avec stride = patch_size.

### Consigne

Une convolution 2D avec :
- kernel_size = patch_size
- stride = patch_size
- out_channels = d_model

produit exactement la meme chose que "decouper en patches puis projeter lineairement".

1. **Equivalence** : prendre une image 32x32x3 et patch_size = 8.
   - Approche A (patchify + linear) : 16 patches de (8*8*3 = 192), projete vers d_model=64 → sequence (16, 64)
   - Approche B (Conv2d) : applique une conv 2D kernel=8 stride=8 out_channels=64 → sortie (64, 4, 4) qui aplatie donne (16, 64)

2. **Parametres equivalents** : combien de parametres dans chaque approche ?
   - Approche A : matrice de projection (192, 64) = 12 288 params
   - Approche B : conv 2D (3, 64, 8, 8) = 12 288 params
   - Pareil !

3. **Pourquoi Conv2d est prefere en pratique** : 
   - Les kernels GPU optimises pour les convolutions sont tres rapides
   - Meme memoire mais throughput plus eleve
   - Tous les frameworks ViT (timm, HuggingFace) utilisent Conv2d

4. **Calcul a la main** : sur une image 4x4x1 (pour simplifier) avec patch_size=2 :
   - Decouper en 4 patches
   - Imaginer une projection vers d=3
   - Combien de parametres ? Combien de patches tokens en sortie ?

5. **Generalisation** : si on utilise patch_size = 14 sur 224x224, combien de patches ? C'est ViT-L/14 : la version "large". Pourquoi plus de patches donne plus de "resolution" au modele ?

### Criteres de reussite

- [ ] Patchify + Linear equivaut exactement a Conv2d kernel=stride=patch
- [ ] Parametres : 192 * 64 = 12 288 dans les deux cas
- [ ] Image 4x4 patch 2 : 4 patches, Conv params = (1, 3, 2, 2) = 12 params
- [ ] ViT-L/14 : 16x16 = 256 patches (apres ajustement). Plus de patches = plus de "resolution spatiale" pour le modele, meilleur pour detecter les petits details
- [ ] Comprehension : Conv2d est plus rapide en GPU, c'est l'implementation standard

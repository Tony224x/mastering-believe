# Exercices Medium — Jour 12 : Multimodalite

---

## Exercice 4 : Patchify / unpatchify — la porte d'entree du ViT

### Objectif

Implementer le decoupage d'une image en patches (et son inverse) de maniere vectorisee — la premiere operation de tout Vision Transformer, et un piege a reshape classique.

### Consigne

1. Implementer :

```python
def patchify(img, patch_size):
    """img: (H, W, C) -> (n_patches, patch_size*patch_size*C)
    Ordre des patches: ligne par ligne (raster). Vectorise (reshape/transpose),
    PAS de double boucle Python."""

def unpatchify(patches, H, W, C, patch_size):
    """Inverse exact de patchify."""
```

2. Verifier sur une image test `(8, 8, 3)` remplie de `np.arange` :
   - shapes : 16 patches de dimension 192 pour patch_size=2... non : patch_size=2 → 16 patches de 12 ; verifier aussi patch_size=4 → 4 patches de 48
   - **round-trip exact** : `unpatchify(patchify(img)) == img` (egalite stricte, pas de tolerance)
   - **contenu du premier patch** : `patchify(img, 2)[0] == img[:2, :2, :].flatten()` — c'est CE test qui detecte le reshape naif `img.reshape(n_patches, -1)` (faux car il decoupe en bandes horizontales, pas en carres). Montrer explicitement que le reshape naif echoue ce test.

3. Calculer pour ViT-B/16 sur une image 224x224x3 :
   - nombre de patches (196), dimension d'un patch aplati (768), shape de la sequence avec [CLS] (197, 768 apres projection)
   - parametres de la projection lineaire patch→d_model (768*768 + 768) — remarquer la coincidence 16*16*3 = 768 = d_model
4. Verifier que `patchify` fonctionne aussi en batch `(B, H, W, C) -> (B, n_patches, d_patch)` sans boucle.

### Criteres de reussite

- [ ] Round-trip strictement exact pour patch_size ∈ {2, 4}
- [ ] Le test du premier patch passe ET le contre-exemple du reshape naif est montre
- [ ] Aucune boucle Python sur les pixels/patches (transpose + reshape uniquement)
- [ ] Les comptes ViT-B/16 (196, 768, 197, ~590k params) sont retrouves par le code
- [ ] La version batchee est correcte (comparee a la boucle sur les images, egalite exacte)

---

## Exercice 5 : La loss contrastive de CLIP from scratch

### Objectif

Implementer l'InfoNCE symetrique de CLIP et verifier ses valeurs sur des cas calculables a la main — comprendre ce que "aligner deux modalites" veut dire numeriquement.

### Consigne

1. Implementer :

```python
def clip_loss(img_emb, txt_emb, temperature):
    """img_emb, txt_emb: (B, d), NON normalises en entree.
    1. L2-normaliser chaque ligne
    2. logits = img_emb @ txt_emb.T / temperature
    3. targets = diagonale (la paire i va avec i)
    4. loss = (CE(logits, targets) + CE(logits.T, targets)) / 2"""
```

   avec une cross-entropy stable (log-sum-exp).

2. **Cas a la main 2x2** (verifier a 1e-6) : embeddings deja normalises tels que la matrice de similarite soit `[[1, 0], [0, 1]]`, temperature=1 :
   - logits = [[1, 0], [0, 1]] ; CE ligne 0 = -log(e/(e+1)) ≈ 0.3133 ; loss totale = 0.3133

3. Cas limites (B=8, d=16, seed fixe) :
   - **alignement parfait** : txt_emb = img_emb → avec temperature=0.01 la loss → ~0 (< 0.01) ; avec temperature=1 elle reste > 0 (calculer pourquoi : les negatifs ont une similarite non nulle... ici sim diagonale=1, hors diagonale aleatoire — verifier que loss(T=0.01) << loss(T=1))
   - **embeddings aleatoires independants** : loss ≈ ln(B) = ln(8) ≈ 2.079 (± 0.3 sur quelques seeds) — le modele ne sait rien, il predit ~uniforme
   - **pire que le hasard** : decaler les textes d'un cran (txt_emb = img_emb roule de 1) avec temperature=0.05 → la loss explose (> 10) car la bonne paire a une similarite plus faible que la "fausse" paire parfaite

4. **Symetrie** : verifier que `clip_loss(img, txt) == clip_loss(txt, img)` (1e-12) et expliquer pourquoi la version a une seule direction (image→texte seulement) est moins bonne (chaque direction fournit B negatifs par exemple a l'AUTRE modalite).

5. Gradient sanity check : par differences finies sur 5 coordonnees de img_emb, verifier le gradient d'une implementation analytique OU simplement verifier que descendre le gradient numerique 50 steps (lr=0.5) sur les embeddings du cas aleatoire fait baisser la loss sous 1.0.

### Criteres de reussite

- [ ] Le cas 2x2 a la main correspond a 1e-6
- [ ] La loss aleatoire ≈ ln(B) et l'explication "uniforme sur B candidats" est en commentaire
- [ ] Les cas alignement parfait et adversarial donnent les comportements attendus
- [ ] La symetrie est verifiee a 1e-12
- [ ] La normalisation L2 est appliquee AVANT le produit scalaire (teste : multiplier img_emb par 100 ne change pas la loss)

---

## Exercice 6 : Debugger un mini-ViT casse

### Objectif

Trouver 3 bugs dans un forward de ViT — dont le plus vicieux de la vision : l'oubli du positional embedding, indetectable sans le bon test.

### Consigne

Le code suivant contient **3 bugs** :

```python
def vit_forward_buggy(img, params, patch_size=2):
    p = img.reshape(-1, patch_size * patch_size * img.shape[-1])  # BUG 1 ?
    x = p @ params['W_patch']                  # (n_patches, d)
    cls = params['cls_token']                  # (1, d)
    x = np.concatenate([x, cls], axis=0)       # BUG 2 ?
    # ... pas de pos_emb ...                   # BUG 3 ?
    x = transformer_block(x, params)           # attention bidirectionnelle
    return x[0] @ params['W_head']             # lit la position 0
```

1. Identifier les 3 bugs :
   - le patchify naif (bandes au lieu de carres) — reutiliser le test "premier patch" de l'exercice 4
   - le [CLS] ajoute a la FIN alors que la head lit `x[0]` — la classification se fait sur un patch, pas sur le CLS
   - l'absence de positional embedding — le modele est aveugle a la disposition spatiale

2. Ecrire le test qui detecte le bug 3 : **le test de permutation**. Sans pos_emb, permuter les patches d'entree (image "melangee") ne change PAS le logit de classification (l'attention bidirectionnelle + CLS est invariante par permutation des autres tokens). Avec pos_emb, il change. Verifier les deux affirmations numeriquement (tolerance 1e-10 pour l'invariance).

3. Ecrire `vit_forward_fixed` (patchify correct, CLS prepend en position 0, pos_emb appris additionne) et verifier :
   - les 3 tests passent
   - une image et sa version "melangee" donnent desormais des sorties differentes (norme de la difference > 0.01)

4. Question (commentaire) : pourquoi le bug 3 est-il le plus dangereux des trois en pratique ? (Le modele apprend quand meme quelque chose — les statistiques de patches, comme un bag-of-words — les courbes d'entrainement descendent, et on ne voit rien sans test cible.)

### Criteres de reussite

- [ ] Les 3 bugs sont identifies avec leur symptome
- [ ] Le test de permutation est implemente et demontre l'invariance (buggy) puis sa rupture (fixed)
- [ ] Le test "premier patch carre" detecte le bug 1
- [ ] Un test verifie que la head lit bien le token CLS (modifier le pos 0 de l'input du bloc doit impacter la sortie via le CLS prepend correct)
- [ ] La reponse sur la dangerosite du bug 3 est correcte

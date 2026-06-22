# Exercices Hard — Jour 12 : Multimodalite

---

## Exercice 7 : CLIP de bout en bout — gradient complet a travers les projections

### Objectif

Implementer un mini-CLIP COMPLET en NumPy (deux encodeurs lineaires + loss contrastive + temperature apprise) avec la backpropagation ENTIERE faite a la main, et le faire converger sur des donnees synthetiques structurees (zero-shot classification).

### Consigne

1. **Architecture** :
   - `img_proj` : `Linear(d_img, d)` (W_img, b_img)
   - `txt_proj` : `Linear(d_txt, d)` (W_txt, b_txt)
   - normalisation L2 des deux sorties
   - `logit_scale` (scalaire appris, parametrise en log)
   - scores `S = exp(logit_scale) * img_norm @ txt_norm.T`
   - loss = moyenne(cross-entropy par ligne, cross-entropy par colonne)

2. **Backprop complete (a la main)** — c'est le coeur de l'exercice :
   - `dL/dS` (deja vu : softmax - onehot, moyenne des deux directions)
   - `dL/dlogit_scale` (S depend de logit_scale par un facteur multiplicatif)
   - retropropager a travers `img_norm @ txt_norm.T` vers `img_norm` et `txt_norm`
   - retropropager a travers la **normalisation L2** (jacobien : `d(x/||x||)/dx = (I - x̂x̂^T)/||x||`)
   - retropropager a travers les Linear vers `W_img, b_img, W_txt, b_txt`
   - Verifier CHAQUE gradient (W_img, W_txt, b_img, b_txt, logit_scale) par difference finie (< 1e-4)

3. **Training** : generer N=32 paires partageant un signal latent. Entrainer avec Adam (implementer Adam en NumPy) sur 300 steps. Tracer loss + top-1 retrieval accuracy → doit atteindre ~100%.

4. **Zero-shot classification** : apres entrainement, construire 3 "prompts de classe" (3 vecteurs texte synthetiques distincts). Pour de nouvelles "images" (features synthetiques proches d'une des classes), classer par similarite cosinus image↔prompt-de-classe. Mesurer l'accuracy zero-shot. C'est EXACTEMENT le mecanisme du zero-shot de CLIP.

5. Analyser :
   - Pourquoi la normalisation L2 est-elle indispensable (sans elle, le modele triche en gonflant les normes) ?
   - Pourquoi clampe-t-on `logit_scale` (en pratique a exp(4.6)≈100) ?

### Criteres de reussite

- [ ] Tous les gradients (W_img, W_txt, b_img, b_txt, logit_scale) passent le test de difference finie (< 1e-4)
- [ ] Le jacobien de la normalisation L2 est correct
- [ ] Adam est implemente correctement (biais-correction des moments)
- [ ] Le training atteint ~100% de top-1 retrieval
- [ ] La zero-shot classification fonctionne (accuracy nettement > hasard)
- [ ] L'analyse normalisation L2 + clamp logit_scale est correcte

---

## Exercice 8 : Tokenization d'image — VQ-VAE (quantization vectorielle) from scratch

### Objectif

Implementer le coeur d'un VQ-VAE : la quantization vectorielle (option "discrete tokens" du cours), qui transforme une image continue en une grille de tokens discrets — la fondation de la generation multimodale (DALL-E, Chameleon).

### Consigne

1. **Codebook** : un dictionnaire de `K` vecteurs d'embedding de dimension `d` (la "palette" de tokens visuels), initialise aleatoirement.

2. **Quantization vectorielle** `vq(z_e, codebook)` :
   - Entree : `z_e` de shape `(n, d)` = features continues d'un encodeur (n patches)
   - Pour chaque vecteur, trouver l'indice du **plus proche** vecteur du codebook (distance L2)
   - Sortie : `indices (n,)` (les tokens discrets) et `z_q (n, d)` (les vecteurs du codebook correspondants)
   - C'est un "argmin nearest neighbor" — implementer le calcul de distance de maniere vectorisee : `||z_e - e||^2 = ||z_e||^2 - 2 z_e·e + ||e||^2`

3. **Straight-through estimator** : la quantization (argmin) n'est pas differentiable. Le STE copie le gradient de `z_q` vers `z_e` : `z_q_st = z_e + stop_gradient(z_q - z_e)`. Expliquer (en commentaire) pourquoi en forward `z_q_st == z_q` mais en backward le gradient passe a travers `z_e`. (Simuler le forward ; le backward STE peut etre decrit/illustre numeriquement.)

4. **Loss VQ-VAE** (3 termes) :
   - reconstruction : `||decode(z_q) - x||^2`
   - codebook loss : `||stop_grad(z_e) - e||^2` (rapproche les vecteurs du codebook des features)
   - commitment loss : `beta * ||z_e - stop_grad(e)||^2` (rapproche l'encodeur du codebook choisi)
   - Implementer le calcul des trois termes pour un encodeur/decodeur lineaires.

5. **Demo** : encoder une petite image en `n` tokens discrets, afficher la grille d'indices (ex: une grille 4x4 d'entiers ∈ [0, K)). Mesurer le **taux d'utilisation du codebook** (combien des K codes sont reellement utilises) — le "codebook collapse" est un probleme classique.

6. **Mini training** : entrainer encodeur + decodeur + codebook (descente de gradient simple) sur un petit jeu d'images synthetiques et montrer que la loss de reconstruction baisse. Montrer qu'apres entrainement, des images similaires donnent des sequences de tokens similaires.

### Criteres de reussite

- [ ] La quantization vectorielle trouve le plus proche voisin correctement (calcul de distance vectorise)
- [ ] Le principe du straight-through estimator est correctement explique (forward == z_q, backward via z_e)
- [ ] Les 3 termes de la loss VQ-VAE sont implementes (reconstruction, codebook, commitment) avec les bons stop-gradient
- [ ] La grille de tokens discrets est affichee
- [ ] Le taux d'utilisation du codebook est mesure (et le collapse discute)
- [ ] Le mini training fait baisser la loss de reconstruction

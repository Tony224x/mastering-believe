# Exercices Hard — Jour 6 : Transformer Architecture

---

## Exercice 7 : Bloc Transformer complet from scratch + tests de proprietes

### Objectif

Assembler un bloc Transformer decoder complet (Pre-LN) en NumPy pur et le valider non pas "a l'oeil" mais par des proprietes mathematiques exactes.

### Consigne

1. Implementer un bloc decoder Pre-LN complet :

```
x = x + MultiHeadAttention(LayerNorm1(x))   # attention causale
x = x + FFN(LayerNorm2(x))                  # FFN: Linear -> GELU -> Linear
```

   Composants from scratch : LayerNorm (gamma/beta), MHA causale multi-tetes (W_q, W_k, W_v, W_o), FFN avec GELU (approximation tanh acceptee). Config de test : `d_model=16, n_heads=4, d_ff=64, T=8, batch=2`.

2. **Test de shapes** : assert sur chaque intermediaire (post-LN, Q/K/V splittes, weights, post-merge, post-FFN).

3. **Test de causalite de bout en bout** : perturber `x[:, t0+1:, :]` (tous les tokens apres t0) avec un bruit fort et verifier que `output[:, :t0+1, :]` est inchange a 1e-12 pres, pour chaque t0. C'est LE test qui detecte un masque mal applique n'importe ou dans le bloc.

4. **Test d'identite** : mettre `W_o = 0` et la 2e couche du FFN a 0 → le bloc doit etre exactement l'identite (`output == x`). Expliquer en commentaire pourquoi c'est la base du "zero-init residual" utilise pour stabiliser les gros modeles.

5. **Test de permutation** : sans positional encoding et SANS masque causal (variante bidirectionnelle du meme bloc), permuter les tokens d'entree doit permuter les sorties a l'identique (`block(x[perm]) == block(x)[perm]`, tolerance 1e-10). Avec masque causal, cette propriete doit etre PERDUE — verifier les deux.

### Criteres de reussite

- [ ] Le bloc est strictement Pre-LN (les LN sont sur les branches, pas sur le residual stream)
- [ ] Tous les asserts de shapes passent
- [ ] Causalite : difference max sur le prefixe < 1e-12 pour tous les t0 testes
- [ ] Test d'identite exact (difference nulle) avec les projections de sortie a zero
- [ ] Equivariance par permutation verifiee dans le cas bidirectionnel ET sa violation verifiee dans le cas causal
- [ ] Chaque composant (LN, GELU, MHA, FFN) est implemente from scratch et commente

---

## Exercice 8 : Positional encodings — propriete de shift relatif + ALiBi

### Objectif

Aller au-dela du "on additionne des sinus" : prouver numeriquement la propriete de translation du PE sinusoidal et implementer ALiBi, l'alternative par biais d'attention.

### Consigne

1. Implementer le PE sinusoidal standard `PE(pos, 2i) = sin(pos/10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)` pour `d=64, max_pos=128`.

2. **Propriete de shift lineaire** : pour chaque paire de dimensions `(2i, 2i+1)`, il existe une matrice de rotation 2x2 `R_k` (qui depend de k mais PAS de pos) telle que `[PE(pos+k, 2i), PE(pos+k, 2i+1)] = R_k @ [PE(pos, 2i), PE(pos, 2i+1)]`.
   - Construire explicitement `R_k = [[cos(k*w_i), sin(k*w_i)], [-sin(k*w_i), cos(k*w_i)]]` avec `w_i = 10000^(-2i/d)` (verifier le signe par le test !)
   - Verifier numeriquement pour `k ∈ {1, 5, 17}` et toutes les positions valides : erreur max < 1e-8
   - Expliquer en commentaire pourquoi cette propriete permet en theorie d'apprendre des relations de position RELATIVES avec un encodage ABSOLU

3. **Produit scalaire et distance** : calculer la matrice `PE @ PE.T` et verifier que `PE(p) . PE(q)` ne depend (quasi) que de `|p - q|` : pour plusieurs paires a meme ecart, l'ecart-type des produits scalaires doit etre < 1% de leur moyenne. Tracer (ASCII ou matplotlib) le produit scalaire en fonction de l'offset.

4. **ALiBi** : implementer les biais `bias[h, i, j] = -slope_h * (i - j)` pour `j <= i`, avec les pentes geometriques `slope_h = 2^(-8h/H)` pour `h = 1..H` (H=8 tetes).
   - Verifier les pentes : pour H=8, slopes = 1/2^1 ... 1/2^8
   - Ajouter ces biais a des scores d'attention UNIFORMES (Q@K^T = 0) et montrer que les poids d'attention decroissent avec la distance : pour la derniere position (T=32), `weights[t_dernier, j]` doit etre strictement decroissant quand la distance `i-j` augmente
   - Comparer tete 1 (pente forte → tres locale) et tete 8 (pente faible → quasi uniforme) : l'entropie de la distribution d'attention de la tete 8 doit etre superieure a celle de la tete 1

### Criteres de reussite

- [ ] La propriete de rotation `PE(pos+k) = R_k @ PE(pos)` est verifiee a 1e-8 pour les 3 valeurs de k
- [ ] La quasi-dependance de `PE(p).PE(q)` a `|p-q|` seul est quantifiee (std < 1% de la moyenne par offset)
- [ ] Les pentes ALiBi sont exactes et les poids decroissent strictement avec la distance
- [ ] Le contraste local/global entre tetes ALiBi est demontre par les entropies
- [ ] Les liens conceptuels (PE sinusoidal → RoPE ; ALiBi → extrapolation de longueur) sont expliques en commentaires

# Exercices Medium — Jour 18 : Long context (Flash Attention, RoPE scaling)

---

## Exercice 4 : Tiled (Flash) attention from scratch

### Objectif

Implementer une attention complete en tiles avec online softmax (l'algorithme de Flash Attention v1, simule en numpy) et prouver numeriquement qu'elle produit **exactement** la meme sortie que l'attention naive — c'est le claim porteur du papier de Tri Dao.

### Consigne

L'attention naive materialise la matrice `S = Q @ K^T` de taille `N x N`. Flash Attention ne la materialise jamais : elle traite `Q` par blocs (boucle externe), iterte `K, V` par blocs (boucle interne), et maintient un softmax en ligne (running max + running sum + accumulateur de sortie rescale).

1. **naive_attention(Q, K, V)** : reference. `S = Q @ K^T / sqrt(d)`, `P = softmax(S)`, `O = P @ V`. Retourner `O`.

2. **tiled_attention(Q, K, V, block_size)** : sans materialiser `S` :
   - Pour chaque bloc de `Q` (`Qi`), initialiser un accumulateur `Oi = 0`, un running max `mi = -inf`, un running denom `li = 0`.
   - Pour chaque bloc de `K, V` (`Kj, Vj`) : calculer la tuile `Sij = Qi @ Kj^T / sqrt(d)` (seule chose en "SRAM").
   - Mettre a jour l'online softmax : `mi_new = max(mi, max(Sij))`, `alpha = exp(mi - mi_new)`, `Pij = exp(Sij - mi_new)`, puis `li = alpha*li + sum(Pij)` et `Oi = alpha*Oi + Pij @ Vj`.
   - A la fin du bloc de `Q` : normaliser `Oi /= li`.
   - (Reutilise la logique exacte de `tiled_attention` dans `02-code/18-long-context-attention-scaling.py`.)

3. **Verification** : sur `N = 256`, `d = 32`, generer `Q, K, V` aleatoires. Asserter `max|O_naive - O_tiled| < 1e-4` (en pratique ~1e-6).

4. **Independance au block_size** : verifier que le resultat est identique pour `block_size` dans `{32, 64, 128}`. Le tiling ne doit JAMAIS changer le resultat, seulement le pic memoire.

5. **Pic memoire** : afficher le pic vanilla (`N*N*2`) vs tiled (`block_size^2*2`) pour montrer le gain.

### Criteres de reussite

- [ ] `naive_attention` et `tiled_attention` implementees en numpy pur
- [ ] L'online softmax rescale correctement l'accumulateur ET le denominateur a chaque bloc
- [ ] `assert max|O_naive - O_tiled| < 1e-4` PASSE
- [ ] Le resultat est invariant au `block_size` (3 valeurs testees)
- [ ] Le pic memoire tiled est affiche et bien inferieur au vanilla
- [ ] Code commente avec le POURQUOI du rescale (max delta entre blocs)

---

## Exercice 5 : Sliding window attention + receptive field

### Objectif

Construire un masque d'attention causale a fenetre glissante (style Mistral 7B), verifier que l'attention au-dela de la fenetre est **exactement** nulle, et calculer le receptive field cumule a travers plusieurs couches.

### Consigne

En sliding window, le token `i` n'attend que sur `[max(0, i-W+1), i]` (causal + fenetre `W`). Memoire et compute deviennent `O(N*W)` au lieu de `O(N^2)`, et le KV cache est borne.

1. **make_sliding_mask(N, W)** : construire la matrice masque `N x N` (1 = visible, 0 = masque), telle que la ligne `i` a des 1 sur `[max(0, i-W+1), i]` seulement.

2. **attention_with_mask(Q, K, V, mask)** : attention naive avec masque additif (`S = where(mask>0, S, -1e9)` puis softmax). Retourner la matrice de probas `P` et la sortie.

3. **Comparaison full vs sliding** : avec `N = 64`, `W = 8`, comparer le nombre moyen de tokens attendus par ligne en full causal vs sliding.

4. **Attention hors-fenetre = 0** : au dernier token `i = N-1`, sommer l'attention vers les tokens situes a plus de `W` positions en arriere (colonnes `0 .. N-W-1`). En full causal c'est non nul ; en sliding ca doit etre **exactement 0** (asserter `< 1e-7`).

5. **Receptive field** : apres `L` couches empilees de sliding window `W`, le receptive field theorique est `L * W`. Pour Mistral 7B (`L = 32`, `W = 4096`), calculer le receptive field (~131K) et expliquer pourquoi il reste **theorique** (dilution a chaque hop).

### Criteres de reussite

- [ ] Le masque sliding est correct (causal + borne a `W`, verifie sur quelques lignes)
- [ ] Full causal : ~`(N+1)/2` tokens/ligne en moyenne ; sliding : ≤ `W` tokens/ligne
- [ ] `assert` que l'attention hors-fenetre du dernier token est nulle (< 1e-7) en sliding
- [ ] Receptive field `L*W = 32*4096 = 131072` tokens calcule
- [ ] Comprehension : receptive field cumule != attention exacte long-range (info diluee)

---

## Exercice 6 : RoPE scaling — NTK-aware et YaRN par bande

### Objectif

Implementer NTK-aware (changement de base) et YaRN (scaling par bande : hautes freqs intactes, basses freqs PI-like) et comparer numeriquement la preservation des hautes vs basses frequences face a PI.

### Consigne

Reutiliser `rope_frequencies`, `rope_pi_frequencies`, `rope_ntk_frequencies`, `rope_yarn_frequencies` du `02-code/18`. Parametres : `d = 64`, `base = 10000`, `L_train = 4096`, `L_target = 32768`, `scale = 8`.

1. **NTK-aware** : au lieu de comprimer les positions, on **augmente la base** :
   ```
   new_base = base * scale^(d / (d - 2))
   f_ntk = rope_frequencies(d, new_base)
   ```
   Implementer et afficher.

2. **YaRN par bande** : interpolation differenciee selon la longueur d'onde `lambda_i = 2*pi / theta_i` :
   - haute freq (`lambda << L_train`) -> garder `theta_i` original ;
   - basse freq (`lambda >> L_train`) -> appliquer PI (`theta_i / scale`) ;
   - bande de transition -> rampe lineaire entre les deux.
   Utiliser la rampe `ramp = clip((ratio - alpha)/(beta - alpha), 0, 1)` avec `ratio = L_train / lambda`, et `f_yarn = (1-ramp)*f_pi + ramp*f_orig`.

3. **Tableau comparatif** : pour quelques indices de paires repartis sur le spectre (`0, 4, 8, 16, 24, 30`), afficher `f_orig`, `f_pi`, `f_ntk`, `f_yarn`.

4. **Ratios de preservation** :
   - Haute freq (paire `k=0`) : ratio `f_method / f_orig`. PI doit valoir `1/scale = 0.125` (ecrase), NTK et YaRN doivent etre proches de `1.0` (preserve).
   - Basse freq (derniere paire) : ratio `f_method / f_orig`. PI vaut `1/scale`, YaRN doit etre PI-like (proche de `1/scale`).

5. **Conclusion** : expliquer pourquoi YaRN (hautes freqs intactes + basses comprimees) donne la meilleure qualite long-context (standard 2024-2026).

### Criteres de reussite

- [ ] NTK : `new_base = base * scale^(d/(d-2))` correct (~86 800 pour d=64, scale=8)
- [ ] YaRN : rampe par bande correcte, hautes freqs intactes, basses freqs PI-like
- [ ] Tableau `f_orig / f_pi / f_ntk / f_yarn` affiche sur plusieurs paires
- [ ] Preservation haute freq : `PI ≈ 0.125` (ecrase), `NTK ≈ YaRN ≈ 1.0` (preserve)
- [ ] Preservation basse freq : YaRN ≈ PI ≈ `1/scale` (long-range comprime)
- [ ] Code commente avec le POURQUOI du scaling par bande

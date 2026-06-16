# Exercices Hard — Jour 17 : State Space Models

---

## Exercice 7 : Parallel scan (Blelloch) — derouler la recurrence en O(log N) depth

### Objectif

Implementer le parallel scan associatif (cur du selective scan de Mamba) et verifier qu'il produit exactement le meme resultat que la recurrence sequentielle, mais avec une profondeur O(log N) au lieu de O(N).

### Consigne

1. **Rappel** : la recurrence d'un SSM `h_t = a_t * h_{t-1} + b_t` (forme scalaire par canal, `a_t = A_bar_t`, `b_t = B_bar_t x_t`) est un **scan associatif**. L'operateur de composition de deux segments `(a1, b1)` et `(a2, b2)` est :
   ```
   (a1, b1) o (a2, b2) = (a1 * a2,  a2 * b1 + b2)
   ```
   Verifier a la main (sur 2-3 steps) que composer ces tuples reproduit bien la recurrence.

2. **Scan sequentiel (reference)** : implementer `sequential_scan(a, b)` qui applique la recurrence step par step et renvoie tous les `h_t`. C'est la verite terrain.

3. **Parallel scan (Blelloch / Hillis-Steele)** : implementer `parallel_scan(a, b)` qui calcule les memes `h_t` via la composition associative par doublement de pas (`d = 1, 2, 4, ...`). A chaque etage, on combine `element[i]` avec `element[i - d]` via l'operateur ci-dessus. O(N log N) work, O(log N) depth.

4. **Verifier l'equivalence** numerique : `max|h_seq - h_par|` doit etre ~1e-12 sur une sequence aleatoire (N=64, a_t dans (0,1) pour la stabilite).

5. **Selectivite** : rendre `a_t` et `b_t` **input-dependent** (Mamba) — `a_t = exp(-softplus(w_a . x_t))` (dans (0,1)), `b_t = (w_b . x_t) * x_t`. Re-verifier que le parallel scan marche toujours (le scan est agnostique a la provenance des coeffs).

6. **Analyse** :
   - Pourquoi l'associativite de l'operateur est-elle la condition pour paralleliser ? (sans elle, pas de regroupement par arbre)
   - Pourquoi Mamba ne peut PAS utiliser la FFT (comme S4) alors qu'il peut utiliser le parallel scan ? (relier au cours : kernel non-stationnaire car a_t depend de x_t)
   - Relier au truc hardware de Tri Dao (scan en SRAM, recompute) : qu'est-ce que le parallel scan resout (depth) vs qu'est-ce que le kernel SRAM resout (bande passante) ?

### Criteres de reussite

- [ ] L'operateur de composition associatif est implemente et son associativite verifiee
- [ ] `sequential_scan` et `parallel_scan` produisent le meme resultat (diff ~1e-12)
- [ ] Le parallel scan fonctionne en coeffs fixes ET input-dependent (selectif)
- [ ] L'analyse explique : associativite -> parallelisme ; pourquoi pas de FFT en selectif ; depth vs bandwidth
- [ ] Code numpy, seed, commente WHY

---

## Exercice 8 : Discretisation ZOH et stabilite — du continu au discret

### Objectif

Implementer la discretisation Zero-Order Hold (`A_bar = exp(Delta * A)`, `B_bar = (A_bar - I) A^{-1} B`) qui transforme l'EDO continue en recurrence discrete, et explorer le role du pas `Delta` (la cle de la selectivite de Mamba).

### Consigne

1. **Systeme continu** : `h'(t) = A h(t) + B x(t)`, `y(t) = C h(t)` avec `A` diagonale stable (valeurs propres negatives, ex `diag(-0.1, -0.5, -1.0, -2.0)`), `B`, `C` aleatoires.

2. **Discretisation ZOH** : implementer `discretize(A, B, Delta)` qui renvoie
   - `A_bar = exp(Delta * A)` (exponentielle de matrice ; pour A diagonale c'est `exp(Delta * diag)`) ;
   - `B_bar = (A_bar - I) @ inv(A) @ B` (forme exacte du cours).
   Gerer le cas A diagonale efficacement.

3. **Verifier** : pour un input `x` donne, comparer la sortie de la recurrence discrete `h_t = A_bar h_{t-1} + B_bar x_t` avec une integration numerique fine (Euler a tres petit pas) de l'EDO continue. Elles doivent concorder.

4. **Effet de Delta** : faire varier `Delta in {0.01, 0.1, 1.0, 5.0}`. Observer :
   - petit Delta -> `A_bar` proche de I -> l'etat retient longtemps (memoire longue, peu de mise a jour) ;
   - grand Delta -> `A_bar` proche de 0 -> l'etat oublie vite (memoire courte, forte absorption de l'input).
   Tabuler le "temps de demi-vie" de la memoire en fonction de Delta.

5. **Lien Mamba** : dans Mamba, `Delta_t = softplus(Linear(x_t))` est **input-dependent**. Implementer un `Delta_t` qui depend de l'input (ex grand quand le token est "important", petit sinon) et montrer sur une mini-sequence que le modele "ouvre" (grand Delta) ou "ferme" (petit Delta) sa memoire selon le token. C'est par Delta que la selectivite agit (note du cours).

6. **Analyse** :
   - Pourquoi exige-t-on la stabilite (`Re(eigenvalues(A)) < 0` en continu, `|eig(A_bar)| < 1` en discret) ?
   - Pourquoi Mamba garde `A` fixe mais rend `Delta` selectif plutot que de rendre `A` directement selectif ? (cout, stabilite, HiPPO)

### Criteres de reussite

- [ ] `discretize` calcule `A_bar = exp(Delta A)` et `B_bar = (A_bar - I) A^{-1} B` correctement
- [ ] La recurrence discrete concorde avec l'integration fine de l'EDO continue
- [ ] Le balayage de Delta montre le tradeoff memoire longue (petit Delta) vs absorption (grand Delta)
- [ ] Un Delta_t input-dependent demontre l'ouverture/fermeture selective de la memoire
- [ ] L'analyse couvre stabilite + pourquoi A fixe / Delta selectif
- [ ] Code numpy, seed, commente WHY

---

## Exercice 9 : Capacite memoire de l'etat — quand le compresseur lossy sature

### Objectif

Quantifier rigoureusement la limite fondamentale des SSM (idee fausse n°4) : l'etat de taille fixe `D` est un compresseur lossy, et au-dela d'un certain nombre d'informations a retenir, le recall s'effondre — la ou un transformer (KV cache) tient.

### Consigne

1. **Tache de capacite** : memoriser `M` paires cle->valeur (vecteurs aleatoires) presentees sequentiellement, puis les recuperer par requete sur la cle. C'est une tache d'**associative memory** pure.

2. **Memoire SSM (etat fixe)** : modeliser un SSM lineaire comme une **memoire associative lineaire** : `H = sum_m v_m k_m^T` (outer products accumules dans une matrice d'etat `D x D`), recall par `v_hat = H k_query`. C'est exactement ce que fait un SSM lineaire / attention lineaire (lien Mamba-2 / SSD du cours). Implementer ecriture + lecture.

3. **Mesurer la degradation** : pour un etat de taille `D` fixe, faire croitre `M` (nombre de paires). Mesurer l'erreur de recall (cosine similarity entre `v_hat` et le vrai `v`) en fonction de M. Identifier le **point de saturation** (ou l'erreur decolle) en fonction de D.

4. **Comparer au "transformer" (KV cache)** : un transformer stocke litteralement chaque paire (`M` slots), donc recall ~parfait jusqu'a M arbitraire (au cout d'une memoire O(M)). Montrer la divergence : SSM (memoire O(D) fixe) sature, transformer (memoire O(M)) tient.

5. **Loi de capacite** : verifier empiriquement que la capacite utile du SSM scale ~lineairement avec `D` (ex : un etat `D` peut stocker ~`c*D` paires avant saturation, pour une constante c liee a la dimension des cles/orthogonalite). Tracer M_saturation vs D.

6. **Analyse** :
   - Relier au benchmark MQAR du cours (Mamba pur 30-50% vs transformer 95%+).
   - Pourquoi est-ce une limite **fondamentale** (theorie de l'information : compresser M*d floats dans D*D < M*d est lossy) et pas un defaut d'implementation ?
   - Pourquoi les hybrides (Jamba) sont la reponse pragmatique ?

### Criteres de reussite

- [ ] La memoire associative lineaire (outer products + recall) est implementee
- [ ] L'erreur de recall en fonction de M est mesuree pour un D fixe ; point de saturation identifie
- [ ] La comparaison SSM (O(D) sature) vs transformer (O(M) tient) est demontree
- [ ] La capacite utile ~ lineaire en D est verifiee empiriquement (M_saturation vs D)
- [ ] L'analyse relie a MQAR, argumente la limite informationnelle, justifie les hybrides
- [ ] Code numpy, seed, commente WHY

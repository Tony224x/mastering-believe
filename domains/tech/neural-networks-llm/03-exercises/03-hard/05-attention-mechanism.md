# Exercices Hard — Jour 5 : Attention Mechanism

---

## Exercice 7 : Backward de l'attention (softmax Jacobian) + gradient check

### Objectif

Deriver et implementer le backward complet du scaled dot-product attention, en passant par le point delicat : la Jacobienne du softmax. Prouver la correction par gradient check.

### Consigne

Le forward est `O = softmax(Q @ K^T / sqrt(d_k)) @ V`. On veut `dQ, dK, dV` etant donne `dO`.

1. **Jacobienne du softmax** : pour `p = softmax(s)`, montrer que `ds = J^T @ dp` ou la Jacobienne vaut `J_ij = p_i (delta_ij - p_j)`. En pratique, pour une ligne :
   ```
   ds = p * (dp - sum(dp * p))
   ```
   Implementer `softmax_backward(dp, p)` avec cette formule vectorisee.

2. **Backward complet** de l'attention :
   - `O = A @ V` (avec `A` = poids d'attention) → `dV = A^T @ dO`, `dA = dO @ V^T`
   - `A = softmax(S)` ligne par ligne → `dS = softmax_backward(dA, A)` (par ligne)
   - `S = (Q @ K^T) / sqrt(d_k)` → `dQ = (dS @ K) / sqrt(d_k)`, `dK = (dS^T @ Q) / sqrt(d_k)`

3. **Gradient check** : avec une loss `0.5 * ||O||^2` (donc `dO = O`), compare `dQ, dK, dV` analytiques aux gradients numeriques (eps = 1e-5, erreur relative < 1e-5).

4. **Avec masque causal** : refaire le gradient check en presence d'un masque causal. Verifier que le gradient des positions masquees ne "fuit" pas (les positions a `-inf` ont un poids softmax de 0, donc gradient 0).

5. Question : pourquoi le gradient `dQ` depend-il de TOUTES les keys (et pas seulement de celle qui a recu le plus de poids) ? (Indice : la Jacobienne du softmax est dense.)

### Criteres de reussite

- [ ] `softmax_backward` implemente correctement `p * (dp - sum(dp * p))`
- [ ] `dV, dA, dS, dQ, dK` sont derives et implementes correctement
- [ ] Le gradient check passe pour `dQ, dK, dV` (erreur < 1e-5)
- [ ] Le gradient check passe AUSSI avec masque causal, sans fuite vers les positions masquees
- [ ] La reponse explique que la Jacobienne du softmax couple toutes les sorties → `dQ` melange l'info de toutes les keys

---

## Exercice 8 : FlashAttention-style — attention online (streaming softmax)

### Objectif

Implementer le coeur algorithmique de FlashAttention : un softmax "online" qui calcule l'attention en blocs sans jamais materialiser la matrice `L x L` complete. Comprendre le gain memoire O(L) au lieu de O(L^2).

### Consigne

L'idee cle de FlashAttention : on parcourt les keys/values par blocs et on maintient un accumulateur de l'output avec une renormalisation incrementale du softmax (trick du "running max" + "running sum").

1. **Softmax online sur un vecteur** : implementer une fonction qui calcule `softmax(s) @ V` en parcourant `s` et `V` par blocs, en maintenant :
   - `m` = max courant (running max)
   - `l` = somme courante des exp (renormalisee a chaque nouveau max)
   - `acc` = accumulateur de l'output renormalise

   La regle de fusion quand un nouveau bloc arrive avec son propre max `m_block` :
   ```
   m_new = max(m, m_block)
   l = l * exp(m - m_new) + l_block * exp(m_block - m_new)
   acc = acc * exp(m - m_new) + acc_block * exp(m_block - m_new)
   ```
   (puis `output = acc / l` a la fin).

2. **Verifier l'exactitude** : compare l'output de l'attention online avec l'attention "naive" (softmax complet materialise). L'erreur doit etre < 1e-10. C'est le point crucial : FlashAttention est EXACT, pas une approximation.

3. **Attention complete par blocs** : etendre a `Q (n_q, d), K (n_k, d), V (n_k, d)`. Pour chaque bloc de queries, parcourir les blocs de keys/values en streaming et accumuler.

4. **Cout memoire** : mesurer (ou raisonner) la memoire pic. L'attention naive stocke `(n_q, n_k)`. L'attention online ne stocke qu'un bloc `(block_q, block_k)` + les accumulateurs `(n_q, d)`. Montrer que pour `n_q = n_k = L` et bloc fixe, la memoire est O(L) au lieu de O(L^2).

5. Question : pourquoi le "running max" est-il indispensable pour la stabilite numerique ? Que se passerait-il si on accumulait directement `sum(exp(s))` sans soustraire le max ?

### Criteres de reussite

- [ ] Le softmax online sur un vecteur donne le meme resultat que le softmax direct (< 1e-10)
- [ ] La regle de fusion (rescaling par `exp(m - m_new)`) est correcte
- [ ] L'attention online par blocs est EXACTE vs l'attention naive (< 1e-10)
- [ ] L'analyse memoire montre O(L) (accumulateurs) vs O(L^2) (matrice complete)
- [ ] La reponse : sans running max, `exp(s)` overflow pour de grands scores ; le max soustrait borne les exposants a <= 0

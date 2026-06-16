# Exercices Medium — Jour 5 : Attention Mechanism

---

## Exercice 4 : Self-attention layer complete en NumPy (Q/K/V appris)

### Objectif

Implementer une couche de self-attention complete (projections Q/K/V apprises + scaled dot-product + masque optionnel) et verifier ses proprietes numeriques.

### Consigne

En te basant sur `02-code/05-attention-mechanism.py` :

1. Implementer `scaled_dot_product_attention(Q, K, V, mask=None)` qui renvoie l'output ET les poids d'attention. Le masque utilise la convention du cours : `1 = bloque`, `0 = autorise` (on met `-inf` avant softmax).

2. Implementer une classe `SelfAttention` avec 3 matrices apprises `W_Q, W_K, W_V` (shape `d_model x d_k`). `forward(X, mask=None)` projette `X (seq, d_model)` puis applique l'attention.

3. **Verifier 3 proprietes** :
   - **Lignes stochastiques** : chaque ligne des poids d'attention somme a 1 (a 1e-10 pres).
   - **Effet du masque causal** : avec un masque triangulaire, le token en position `i` ne met aucun poids sur les positions `> i` (verifier que la partie triangulaire superieure stricte est nulle).
   - **Invariance par permutation (sans masque ni PE)** : permuter les tokens d'entree permute les outputs de la meme facon. C'est LA raison pour laquelle on a besoin de positional encoding.

4. Question : pourquoi divise-t-on par `sqrt(d_k)` et pas par `d_k` ou par rien ? Demontre-le en calculant la variance de `Q_i . K_j` quand les composantes sont iid de variance 1.

### Criteres de reussite

- [ ] `scaled_dot_product_attention` est correct (scores, scale, masque, softmax, weighted sum)
- [ ] Les poids d'attention somment a 1 par ligne
- [ ] Le masque causal annule strictement la partie triangulaire superieure des poids
- [ ] L'invariance par permutation est demontree numeriquement (permuter X permute l'output)
- [ ] La justification du `sqrt(d_k)` est correcte : `Var(Q.K) = d_k` (somme de d_k produits de variance 1), donc diviser par `sqrt(d_k)` ramene l'ecart-type a ~1

---

## Exercice 5 : Multi-head attention en NumPy + cout memoire

### Objectif

Implementer la multi-head attention (MHA) complete en NumPy, prouver l'equivalence "1 grosse projection + reshape" vs "N petites tetes", et analyser le cout.

### Consigne

1. Implementer `MultiHeadAttention(d_model, n_heads)` :
   - Projections `W_Q, W_K, W_V` de shape `(d_model, d_model)` et projection de sortie `W_O (d_model, d_model)`.
   - Reshape `(seq, d_model) -> (n_heads, seq, d_head)` avec `d_head = d_model / n_heads`.
   - Attention par tete, concat, puis projection `W_O`.

2. **Verifier les shapes** : pour `d_model=64, n_heads=8, seq=10`, tracer la shape apres chaque etape (projection, reshape, attention par tete, concat, output).

3. **Equivalence** : montrer numeriquement que faire la projection en une fois `X @ W_Q` puis reshaper en tetes donne EXACTEMENT le meme resultat que projeter chaque tete separement avec les sous-blocs colonnes de `W_Q`. (erreur < 1e-10)

4. **Cout memoire** : pour `seq_len = L`, la matrice d'attention a une taille `n_heads * L^2`. Calculer la memoire (en MB, float32) pour `L ∈ {512, 2048, 8192}` avec `n_heads=12`. Verifier la croissance quadratique en `L`.

5. Question : la MHA a-t-elle plus de parametres que la single-head attention avec le meme `d_model` ? Pourquoi multi-head est-il prefere malgre un cout parametrique identique ?

### Criteres de reussite

- [ ] La MHA est correcte (reshape en tetes, attention par tete, concat, W_O)
- [ ] Les shapes sont tracees correctement, `d_head = 8`
- [ ] L'equivalence "grosse projection + reshape" vs "tetes separees" est verifiee (< 1e-10)
- [ ] La memoire de la matrice d'attention croit en O(L^2) (verifie pour les 3 valeurs)
- [ ] La reponse : meme nombre de params (les projections sont `d_model x d_model` dans les 2 cas) ; multi-head permet a chaque tete de se specialiser dans un sous-espace different

---

## Exercice 6 : Cross-attention et le role de Q vs K/V

### Objectif

Comprendre la difference self-attention vs cross-attention et le role asymetrique de la query par rapport aux key/value.

### Consigne

En cross-attention, la query vient d'une source (ex : decoder) et les key/value d'une autre (ex : encoder).

1. Implementer `cross_attention(X_query, X_context, W_Q, W_K, W_V)` :
   - `Q = X_query @ W_Q` (depuis la source query)
   - `K = X_context @ W_K`, `V = X_context @ W_V` (depuis le contexte)
   - Shapes : `X_query (n_q, d_model)`, `X_context (n_c, d_model)`, output `(n_q, d_v)`.

2. **Verifier les shapes asymetriques** : avec `n_q=3` (3 tokens de query) et `n_c=5` (5 tokens de contexte), la matrice d'attention est `(3, 5)` et l'output est `(3, d_v)`. La longueur de sortie suit la query, pas le contexte.

3. **Cas degenere** : montrer que la self-attention est un cas particulier de cross-attention ou `X_query == X_context`.

4. **Experience** : construis un mini "retrieval" : 1 query qui doit "recuperer" la value du contexte dont la key lui ressemble le plus. Construis `Q`, `K`, `V` tels que la query soit quasi alignee avec une seule key, et verifie que l'output est dominé par la value correspondante.

5. Question : dans un Transformer encoder-decoder (traduction), a quelle etape la cross-attention intervient-elle et que permet-elle au decoder de faire ?

### Criteres de reussite

- [ ] `cross_attention` est correct (Q de la source, K/V du contexte)
- [ ] La matrice d'attention a la shape `(n_q, n_c)` et l'output `(n_q, d_v)`
- [ ] La self-attention est bien retrouvee comme cas `X_query == X_context`
- [ ] Le mini-retrieval montre l'output dominé par la value de la key la plus proche
- [ ] La reponse : la cross-attention permet au decoder, a chaque pas de generation, d'aller "regarder" les representations de la phrase source (alignement source-cible)

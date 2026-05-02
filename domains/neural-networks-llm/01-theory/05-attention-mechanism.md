# Jour 5 — Attention Mechanism : l'innovation qui change tout

> **Temps estime** : 6h | **Prerequis** : Jour 4 (RNN, LSTM), Jour 3 (Embeddings)

---

## 1. Le probleme que l'attention resout

### Le goulot d'etranglement du seq2seq encoder-decoder

En 2014, l'etat de l'art en traduction automatique etait le modele **encoder-decoder** (Sutskever et al., Cho et al.) :

```
Phrase source  ────[ LSTM encoder ]────→ [ vecteur c ]────[ LSTM decoder ]────→ traduction
"le chat mange"                           (taille fixe)                          "the cat eats"
```

**L'idee** : l'encoder lit la phrase source mot par mot et compresse TOUT dans un seul vecteur `c` de taille fixe (typiquement 512 ou 1024 dims). Le decoder part de ce vecteur et genere la traduction mot par mot.

**Le probleme** : compresser une phrase de 50 mots dans un vecteur de 512 dims, c'est comme resumer un livre en un tweet. L'information est perdue.

### Symptomes observes en 2014

1. **Performance s'effondre pour les longues phrases** : pour des phrases de 60+ mots, le BLEU score chute de facon dramatique.
2. **Le decodeur oublie le debut** : dans "The cat that I adopted last year from a shelter in Paris is black", le mot "cat" doit voyager a travers 15 pas avant d'influencer "is black".
3. **Impossible de faire mieux en scalant** : ajouter plus de couches LSTM n'aide pas, car le bottleneck reste le vecteur `c`.

### L'insight : chaque mot genere doit pouvoir "regarder" toute la phrase source

Au lieu de compresser toute la source en un seul vecteur, on garde **tous les etats caches** de l'encoder : `h_1, h_2, ..., h_T`. Quand le decoder genere le mot de sortie `y_t`, il decide DYNAMIQUEMENT lesquels de ces etats regarder.

```
Source : "le chat mange le poisson"
           ↓     ↓     ↓     ↓     ↓
           h_1   h_2   h_3   h_4   h_5

Pour generer "eats" (mot 3 de la cible), l'attention regarde surtout h_3 ("mange")
Pour generer "cat"  (mot 2 de la cible), l'attention regarde surtout h_2 ("chat")
```

C'est l'idee de **l'attention** (Bahdanau et al., 2014). Et c'est la base de tout ce qui viendra ensuite.

---

## 2. Attention : l'intuition

### Metaphore : recherche dans un dictionnaire

Tu veux trouver la traduction d'un mot. Tu as un dictionnaire avec des cles (mots francais) et des valeurs (traductions anglaises). Tu cherches avec une requete (le mot que tu veux traduire).

```
Query (Q)    : "chat"
Keys (K)     : ["chat", "chien", "maison", "voiture", ...]
Values (V)   : ["cat",  "dog",   "house",  "car",     ...]

Similarity(Q, K) → [0.95, 0.40, 0.10, 0.05, ...]  (softmax)
                    ↓
Output : weighted sum of V
       = 0.95*"cat" + 0.40*"dog" + 0.10*"house" + ...
       ≈ "cat"
```

L'attention, c'est exactement ca. Sauf que :

- Les cles et valeurs sont des **vecteurs** (pas des strings)
- La similarite est un **produit scalaire** (pas une comparaison exacte)
- Le resultat est une **moyenne ponderee continue** (pas une selection discrete)

### Terminologie Q, K, V — pourquoi 3 objets differents ?

- **Query (Q)** : "qu'est-ce que je cherche ?" — represente le token courant qui veut recevoir de l'info
- **Key (K)** : "qu'est-ce que j'offre comme index ?" — represente chaque token source, servant a determiner la pertinence
- **Value (V)** : "si tu me choisis, voici ce que je donne" — l'information reelle a transmettre

**Pourquoi K et V sont distincts ?** Parce qu'on peut etre pertinent comme index (matching) mais transmettre quelque chose de different (content). Exemple : dans un moteur de recherche, le titre sert a matcher (K), mais c'est l'article complet qu'on retourne (V).

---

## 3. Scaled Dot-Product Attention : la formule

### La formule de base (version matricielle)

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Shapes :
  Q : (n_q, d_k)   — n_q queries, each of dim d_k
  K : (n_k, d_k)   — n_k keys, each of dim d_k
  V : (n_k, d_v)   — n_k values, each of dim d_v

  Q @ K^T          : (n_q, n_k)   — similarity scores
  softmax(...)     : (n_q, n_k)   — attention weights (rows sum to 1)
  ... @ V          : (n_q, d_v)   — output for each query
```

### Decomposition etape par etape

**Etape 1 : scores de similarite**

```
scores = Q @ K^T           # (n_q, n_k)
```

Chaque cellule `scores[i][j] = Q_i · K_j` est le produit scalaire entre la query `i` et la key `j`. Plus c'est grand, plus la key `j` est pertinente pour la query `i`.

**Etape 2 : scaling par sqrt(d_k)**

```
scores_scaled = scores / sqrt(d_k)
```

**Pourquoi diviser par sqrt(d_k) ?** C'est subtil mais crucial.

Si Q et K ont des composantes aleatoires de moyenne 0 et variance 1, alors leur produit scalaire `Q · K` a une variance de `d_k`. Autrement dit, l'ecart-type est `sqrt(d_k)`.

```
d_k = 64  → std ≈ 8
d_k = 512 → std ≈ 22
```

Sans scaling, les scores sont tres grands en valeur absolue. Le softmax devient alors quasi-one-hot (tout concentre sur un seul token), ce qui tue le gradient — le softmax sature et sa derivee est ~0 pour les autres positions.

**Le scaling par `sqrt(d_k)`** ramene l'ecart-type des scores a 1, ce qui maintient le softmax dans une zone "douce" ou le gradient circule bien.

**Etape 3 : softmax sur les keys**

```
weights = softmax(scores_scaled, axis=-1)   # (n_q, n_k), rows sum to 1
```

Pour chaque query, on convertit les scores en une distribution de probabilite sur les keys. `weights[i][j]` = combien la query `i` fait attention a la key `j`.

**Etape 4 : moyenne ponderee des values**

```
output = weights @ V        # (n_q, d_v)
```

Pour chaque query, on calcule la somme ponderee des values. C'est le resultat de l'attention : une combinaison continue des values selon les poids.

### Visualisation du mecanisme

```
Phrase source : "le chat mange le poisson"
                  V_1   V_2   V_3   V_4   V_5

Query = "cat" (token cible qu'on genere)

  Similarity      :  0.05  0.92  0.01  0.01  0.01
  (softmaxed)
                     ↓     ↓     ↓     ↓     ↓
                  0.05*V_1 + 0.92*V_2 + 0.01*V_3 + 0.01*V_4 + 0.01*V_5
                  ≈ V_2  (le vecteur de "chat")

→ Le modele recupere l'info "chat" pour generer "cat"
```

---

## 4. Self-Attention : l'attention de la sequence sur elle-meme

### La difference avec le seq2seq

Dans le seq2seq d'origine (Bahdanau), les queries viennent du decoder et les K/V viennent de l'encoder. C'est l'**attention croisee** (cross-attention).

Dans le Transformer, il y a aussi (et surtout) la **self-attention** : Q, K, V viennent TOUS de la meme sequence. Chaque token regarde tous les autres tokens de la meme sequence.

```
Phrase : "le chat noir dort"
           x_1  x_2  x_3  x_4

Pour chaque token, on calcule Q, K, V :
  Q_i = W_Q @ x_i   (ce que je cherche)
  K_i = W_K @ x_i   (mon index)
  V_i = W_V @ x_i   (mon contenu)

Puis : Attention(Q, K, V) → nouveau vecteur pour chaque position

Pour le token "noir" :
  - Q_3 mesure la similarite avec K_1, K_2, K_3, K_4
  - Les poids resultants disent "noir" regarde quoi
  - Probablement : forte attention sur "chat" (car "noir" est un adjectif de "chat")
```

### L'effet spectaculaire

Apres une couche de self-attention, chaque token a un representation qui **integre le contexte de toute la phrase**. Le mot "banque" dans "banque centrale" aura une representation differente de "banque du fleuve" car les poids d'attention vers les autres mots sont differents.

C'est exactement ce que ELMo faisait avec des BiLSTM (jour 3), mais en parallele et en une seule etape.

### Les 3 matrices apprises : W_Q, W_K, W_V

```
W_Q : (d_model, d_k)
W_K : (d_model, d_k)
W_V : (d_model, d_v)
```

Ces matrices sont **apprises** par backprop. Ce sont elles qui determinent "comment chaque token formule sa query, sa key et sa value".

**Pourquoi 3 matrices differentes et pas juste une ?** Parce que le role de "ce que tu cherches" (Q) est different de "comment tu es indexe" (K) qui est different de "ce que tu transmets" (V). Avoir 3 projections permet au modele de specialiser chaque role.

---

## 5. Masking : attention causale et padding

### Masking causal (pour la generation)

Dans un Transformer **decoder** (GPT, LLaMA), on genere le texte token par token. Quand on calcule l'attention pour le token `i`, on ne doit PAS voir les tokens futurs (`i+1`, `i+2`, ...) — sinon le modele "triche" en regardant la reponse.

```
Phrase : "le chat mange la souris"
           0    1    2    3   4

Attention autorisee pour chaque position :

  pos 0 ("le")      : peut voir [0]
  pos 1 ("chat")    : peut voir [0, 1]
  pos 2 ("mange")   : peut voir [0, 1, 2]
  pos 3 ("la")      : peut voir [0, 1, 2, 3]
  pos 4 ("souris")  : peut voir [0, 1, 2, 3, 4]
```

On l'implemente avec une **matrice de mask triangulaire inferieure** :

```
mask = 1 -∞  -∞  -∞  -∞
       1  1  -∞  -∞  -∞
       1  1   1  -∞  -∞
       1  1   1   1  -∞
       1  1   1   1   1
```

On ajoute ce mask aux scores AVANT le softmax. `-inf + score = -inf`, et `softmax([-inf, ...]) = 0`. Les positions futures recoivent 0% d'attention.

### Masking de padding

Quand on bat des phrases de longueurs differentes, on pad avec des tokens speciaux. On ne veut pas que l'attention aille vers ces tokens pad. Meme technique : on met `-inf` aux positions pad.

```
Phrase 1 : "le chat mange"               (3 tokens)
Phrase 2 : "le chien"                    (2 tokens, 1 padding)

Batch tokens : [["le", "chat", "mange"], ["le", "chien", "<pad>"]]

Mask phrase 2 : [1, 1, 0]   (0 = ignorer)
```

---

## 6. Multi-Head Attention : l'idee qui multiplie la capacite

### Le probleme avec une seule tete d'attention

Une seule passe d'attention force le modele a compresser toute la relation entre tokens dans un seul jeu de poids. Mais il y a plusieurs "types" de relations qu'on veut capturer :

- Relations syntaxiques (sujet-verbe)
- Relations semantiques (synonymes)
- Relations de coreference (pronom → antecedent)
- Relations positionnelles (mot adjacent)

**L'idee du multi-head** : au lieu d'avoir UNE attention avec des vecteurs de dim `d_model`, on a `h` attentions **paralleles** avec des vecteurs de dim `d_model / h`. Chaque "tete" peut se specialiser dans un type de relation.

### La formule

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) @ W_O

head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

Avec :
  d_model = 512 (total)
  h = 8 (nombre de tetes)
  d_k = d_v = d_model / h = 64
```

### Decomposition etape par etape

**Etape 1 : projections dans les sous-espaces**

```
Q : (seq, d_model=512)
W_Q : (d_model, d_model) → on peut le voir comme h matrices (d_model, d_k=64) empilees

Q_projected = Q @ W_Q = (seq, d_model=512)
Reshape → (seq, h=8, d_k=64)
Transpose → (h=8, seq, d_k=64)  (facilite le calcul parallele)
```

Meme chose pour K et V.

**Etape 2 : attention dans chaque sous-espace**

```
Pour chaque tete i :
  head_i = Attention(Q_i, K_i, V_i)   # (seq, d_v=64)
```

Les 8 tetes calculent en parallele (c'est ca la force du Transformer : tout est parallelise).

**Etape 3 : concatenation**

```
concat = [head_1, head_2, ..., head_8]  # (seq, 8*64) = (seq, 512)
```

On remet les tetes bout a bout pour reformer un vecteur de taille `d_model`.

**Etape 4 : projection finale**

```
output = concat @ W_O    # (seq, d_model)
W_O : (d_model, d_model)
```

Cette derniere projection mixe les informations des differentes tetes.

### Visualisation

```
     Input (seq, d_model=512)
          │
          ├─ W_Q, reshape → (h=8, seq, 64)
          ├─ W_K, reshape → (h=8, seq, 64)
          └─ W_V, reshape → (h=8, seq, 64)
                 │
                 ▼
    ┌─────────┬─────────┬─────────┬─────────┐
    │ head_1  │ head_2  │ head_3  │   ...   │
    │ Attention(Q_i, K_i, V_i) each         │
    └─────────┴─────────┴─────────┴─────────┘
                 │
                 ▼
    Concat → (seq, 8*64=512)
                 │
                 ▼
              W_O
                 │
                 ▼
     Output (seq, d_model=512)
```

### Interpretabilite : ce que les tetes apprennent

Des analyses (Clark et al., 2019 sur BERT) montrent que les tetes se specialisent spontanement :

- Tete 1 : attention a la position precedente (comme un n-gram)
- Tete 2 : attention aux mots identiques ailleurs dans la phrase
- Tete 3 : resolution de coreference
- Tete 4 : syntaxe (sujet-verbe)
- ...

Chaque tete devient un "detecteur" d'un type de relation. Avec 8-16 tetes et 12-96 couches, un Transformer moderne capture une structure linguistique tres riche.

---

## 7. Attention vs RNN : la revolution en un tableau

|                                    | RNN / LSTM                         | Attention / Transformer                  |
| ---------------------------------- | ---------------------------------- | ---------------------------------------- |
| Parallelisation                    | Sequentiel (h_t depend de h_{t-1}) | Tout en parallele                        |
| Longueur max effective             | ~100-500 tokens                    | 10 000+ tokens                           |
| Gradient entre positions distantes | Passe a travers T etapes           | Passe direct (1 etape)                   |
| Complexite par couche              | O(T * d^2)                         | O(T^2 * d)                               |
| Utilisation GPU                    | ~1% (rempli-attente)               | ~95% (matrices denses)                   |
| Interpretabilite                   | Faible                             | Bonne (matrice d'attention visualisable) |

**La seule faiblesse** : `O(T^2)` en memoire. Pour T = 10 000 tokens, la matrice d'attention fait 100M d'entrees. C'est ce que les techniques recentes (Flash Attention, Linear Attention, Sliding Window) cherchent a ameliorer.

---

## 8. Flash Cards — Active Recall

### Q1 : Quel probleme fondamental l'attention resout-elle par rapport au seq2seq encoder-decoder classique ?

<details>
<summary>Reponse</summary>

Le seq2seq classique compresse toute la phrase source dans un **vecteur de taille fixe** `c` (typiquement 512 dims). C'est un goulot d'etranglement : l'information des 60e mots est diluee ou perdue.

L'attention **garde tous les etats caches** de l'encoder et laisse le decoder choisir dynamiquement lesquels regarder a chaque pas de generation. Le decoder peut "retourner en arriere" vers n'importe quelle position source.

Consequence : les longues phrases sont traitees correctement, et le gradient voyage directement entre n'importe quel token source et n'importe quel token cible (au lieu de traverser T etapes LSTM).

</details>

### Q2 : Ecris la formule du scaled dot-product attention. Pourquoi diviser par sqrt(d_k) ?

<details>
<summary>Reponse</summary>

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Pourquoi /sqrt(d_k) ?** Si Q et K ont des composantes de moyenne 0 et variance 1, alors `Q · K` a une variance de `d_k`, donc un ecart-type de `sqrt(d_k)`.

Sans scaling, pour `d_k = 512`, les scores auraient un ecart-type de ~22. Le softmax de tels scores serait quasi-one-hot (toute la masse sur un seul token), ce qui :

1. Tue le gradient (softmax sature)
2. Empeche le modele d'apprendre des patterns graduels

En divisant par `sqrt(d_k)`, on ramene l'ecart-type a ~1, ce qui garde le softmax dans sa zone "sensible".

</details>

### Q3 : Qu'est-ce que la self-attention ? Quelle est la difference avec l'attention croisee (cross-attention) ?

<details>
<summary>Reponse</summary>

**Self-attention** : Q, K, V viennent tous de la **meme sequence**. Chaque token regarde tous les autres tokens de la meme sequence. C'est l'operation centrale dans un Transformer encoder ou decoder.

**Cross-attention** : Q vient d'une sequence, K et V viennent d'une **autre sequence**. Exemple typique : dans un Transformer encoder-decoder pour la traduction, le decoder fait de la cross-attention ou Q vient du decoder (ce qu'on genere) et K/V viennent de l'encoder (la source).

Dans GPT (decoder-only), il n'y a que de la self-attention. Dans BERT (encoder-only), il n'y a que de la self-attention. Dans T5 / l'original Transformer (encoder-decoder), les deux existent.

</details>

### Q4 : A quoi sert le masking causal ? Comment est-il implemente ?

<details>
<summary>Reponse</summary>

Le **masking causal** empeche un token a la position `i` de regarder les tokens futurs (`i+1`, `i+2`, ...) pendant l'entrainement. C'est indispensable pour la generation autoregressive (GPT, LLaMA) : si le modele voyait la reponse, il tricherait.

**Implementation** : on construit une matrice triangulaire `M` de forme `(T, T)` avec :

- `M[i][j] = 0` si `j <= i` (autorise)
- `M[i][j] = -inf` si `j > i` (interdit)

On l'ajoute aux scores AVANT le softmax :

```
scores = Q @ K^T / sqrt(d_k) + M
weights = softmax(scores)
```

`softmax(-inf) = 0`, donc les positions futures recoivent exactement 0% d'attention. Le gradient y est aussi nul.

</details>

### Q5 : Pourquoi le multi-head attention est-il meilleur qu'une seule tete avec d_model dimensions ?

<details>
<summary>Reponse</summary>

Une seule tete force le modele a capturer TOUS les types de relations (syntaxiques, semantiques, coreference, positionnelles, ...) dans un seul jeu de poids W_Q, W_K, W_V.

**Multi-head** : on a `h` attentions paralleles, chacune dans un sous-espace de dimension `d_model / h`. Chaque tete peut se specialiser dans un type de relation.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
```

Des analyses post-hoc (BERT, GPT) montrent que les tetes se specialisent spontanement : une tete fait de la coreference, une autre regarde le token precedent, une autre capture la syntaxe sujet-verbe, etc.

Cout computationnel : **identique** a une seule tete avec d_model dims, car les calculs se font dans des sous-espaces plus petits qui s'additionnent exactement.

</details>

---

## 9. Key Takeaways

1. **L'attention est un lookup differentiable** : queries, keys, values. Les scores sont des produits scalaires, softmaxed, puis utilises pour ponderer les values.
2. **La formule standard** : `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V`. Le `/sqrt(d_k)` empeche le softmax de saturer.
3. **Self-attention** : Q, K, V viennent de la meme sequence. Chaque token integre le contexte de toute la sequence en une etape. C'est ce qui remplace l'etat cache du RNN.
4. **Masking causal** pour la generation autoregressive : matrice triangulaire `-inf` qui empeche de voir le futur.
5. **Multi-head attention** : plusieurs attentions paralleles dans des sous-espaces, permettant aux tetes de se specialiser. Cout computationnel identique a une seule tete de meme taille totale.
6. **Complexite O(T^2)** en memoire. C'est le point faible actuel, que Flash Attention et consorts cherchent a ameliorer.

---

**Prochain jour** : J6 — le bloc Transformer complet (attention + FFN + LayerNorm + residuals + positional encoding).


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS25 V2 — Karpathy (Introduction to Transformers)** — la conference de reference pour l'intuition self-attention.
- **CMU 11-711 (Neubig, Fa24) — Lec. 4 (Attention & Transformers)** — derivation pas-a-pas Q/K/V et multi-head.
- **CMU 11-711 (Welleck, Sp25) — Lec. 5 (Attention & Transformers)** — version 2025, focus sur l'usage moderne.
- **Stanford CS231N — Lec. 8 (Attention & Transformers)** — angle vision + masking causal.

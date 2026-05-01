# Exercices Faciles — Jour 5 : Attention Mechanism

---

## Exercice 1 : Calculer une tete d'attention a la main

### Objectif

Faire tourner le mecanisme d'attention avec de vraies valeurs numeriques, sans framework.

### Consigne

Soit 2 tokens d'entree, chacun projete en Q, K, V de dimension 2 :

```
Q = [[1, 0],
     [0, 1]]        # 2 queries

K = [[1, 0],
     [0, 1]]        # 2 keys

V = [[10, 0],
     [0, 10]]       # 2 values
```

1. Calculer `Q @ K^T` (matrice 2x2).

2. Diviser par `sqrt(d_k)` ou `d_k = 2`. Que vaut `sqrt(2)` ? Ecrire les scores scales.

3. Appliquer softmax sur chaque ligne. Pour rappel :
   ```
   softmax([a, b]) = [exp(a), exp(b)] / (exp(a) + exp(b))
   ```
   Pour `exp(1/sqrt(2))` et `exp(0)`, donner les valeurs numeriques arrondies a 4 decimales.

4. Calculer `output = weights @ V` (matrice 2x2 finale).

5. Interpreter : la query 0 "regarde" surtout laquelle des values ? Pourquoi ?

6. **Bonus** : si on retire le `/ sqrt(d_k)`, que deviennent les poids d'attention ? Sont-ils plus "durs" ou plus "doux" ?

### Criteres de reussite

- [ ] `Q @ K^T = [[1, 0], [0, 1]]` (identite)
- [ ] Scores scales = `[[1/sqrt(2), 0], [0, 1/sqrt(2)]] ≈ [[0.707, 0], [0, 0.707]]`
- [ ] Les poids softmax sont corrects : pour chaque ligne, la majorite de la masse sur la diagonale
- [ ] L'output penche clairement vers la value correspondante (ex: query 0 → mostly V_0)
- [ ] L'explication : plus les scores sont grands, plus le softmax est concentre (pique)

---

## Exercice 2 : Calculer QKV a partir d'un input et de matrices de projection

### Objectif

Comprendre comment les vecteurs Q, K, V sont derives des embeddings d'entree via des matrices apprenables.

### Consigne

Soit 3 tokens d'entree, chacun en dimension 4 :

```
X = [[1, 0, 1, 0],    # token "le"
     [0, 1, 0, 1],    # token "chat"
     [1, 1, 0, 0]]    # token "dort"
```

Les matrices de projection (dim_k = dim_v = 2, d_model = 4) :

```
W_Q = [[1, 0],
       [0, 1],
       [0, 0],
       [0, 0]]

W_K = [[0, 0],
       [0, 0],
       [1, 0],
       [0, 1]]

W_V = [[1, 0],
       [1, 0],
       [0, 1],
       [0, 1]]
```

1. Calculer `Q = X @ W_Q` (matrice 3x2).

2. Calculer `K = X @ W_K` (matrice 3x2).

3. Calculer `V = X @ W_V` (matrice 3x2).

4. Calculer les scores `Q @ K^T / sqrt(2)`.

5. Observer : pour la query du token "le" (ligne 0), quelle key a le score le plus eleve ? Est-ce attendu vu la structure de W_Q et W_K ?

6. Question conceptuelle : pourquoi utiliser 3 matrices distinctes W_Q, W_K, W_V au lieu d'une seule W qui donnerait les 3 en meme temps ?

### Criteres de reussite

- [ ] Les 3 matrices Q, K, V sont calculees correctement
  - `Q = [[1,0], [0,1], [1,1]]`
  - `K = [[1,0], [0,1], [0,0]]`
  - `V = [[1,0], [1,0], [2,0]]`
- [ ] La matrice des scores `Q @ K^T` est calculee
- [ ] L'observation sur les scores est correcte
- [ ] L'explication des 3 matrices : chaque role (interroger, etre interroge, fournir) est distinct et specialise. Avoir 3 projections permet au modele d'apprendre des comportements differents pour chaque role.

---

## Exercice 3 : Positional encoding sinusoidal a la main

### Objectif

Calculer les valeurs du positional encoding du Transformer pour comprendre leur structure.

### Consigne

La formule du positional encoding sinusoidal (Vaswani et al., 2017) :

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Avec :
- `pos` : position du token dans la sequence (0, 1, 2, ...)
- `i`   : index de dimension (0, 1, 2, ..., d_model/2 - 1)
- `d_model` : dimension du modele

Soit `d_model = 4`. On veut calculer PE pour les positions 0, 1, 2, 3.

1. Pour chaque dimension (0, 1, 2, 3), donner :
   - Si c'est pair ou impair
   - L'index i associe (i = dim // 2)
   - Le denominateur `10000^(2i/d_model)` numeriquement
   - La fonction utilisee (sin ou cos)

2. Calculer la matrice PE de shape (4, 4) : 4 positions, 4 dimensions.

3. Observer :
   - PE(pos=0) = [?, ?, ?, ?] (que vaut sin(0) et cos(0) ?)
   - Les dimensions basses (0, 1) varient-elles rapidement ou lentement avec pos ?
   - Les dimensions hautes (2, 3) varient-elles rapidement ou lentement avec pos ?

4. Question conceptuelle : pourquoi utiliser des sinus/cosinus a frequences multiples plutot qu'un simple numero de position (0, 1, 2, ...) ?

5. Question bonus : une propriete magique des sinus est que `PE(pos+k)` peut etre exprime comme une combinaison lineaire de `PE(pos)` pour n'importe quel offset `k`. Pourquoi c'est utile ? (Indice : pensez a la capacite du modele a generaliser a des positions non vues pendant l'entrainement.)

### Criteres de reussite

- [ ] La matrice PE (4, 4) est calculee avec les bonnes fonctions et denominateurs
- [ ] PE(0) = [0, 1, 0, 1] (sin(0)=0, cos(0)=1)
- [ ] Les dimensions basses varient RAPIDEMENT (haute frequence), les hautes varient LENTEMENT (basse frequence)
- [ ] L'explication : encoder la position comme un scalaire (0, 1, 2...) pose 2 problemes : (1) les valeurs explosent pour les longues sequences, (2) il n'y a pas de structure geometrique exploitable. Les sinusoides encodent la position dans une "horloge" multi-echelles ou chaque dimension tourne a une vitesse differente.
- [ ] Bonus : la propriete de linearite permet au modele d'apprendre l'operation "decaler de k positions" comme une transformation lineaire, ce qui facilite la generalisation aux longueurs non vues.

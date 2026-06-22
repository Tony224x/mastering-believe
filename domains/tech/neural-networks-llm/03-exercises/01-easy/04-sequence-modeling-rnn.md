# Exercices Faciles — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)

---

## Exercice 1 : Forward pass d'un RNN a la main

### Objectif

Comprendre concretement comment un RNN maintient et met a jour son etat cache a travers le temps.

### Consigne

Soit un RNN avec :
- Input dim = 2
- Hidden dim = 2
- `W_xh = [[0.5, -0.3], [0.1, 0.4]]`
- `W_hh = [[0.2, 0.1], [-0.1, 0.3]]`
- `b_h = [0.0, 0.0]`
- Etat initial `h_0 = [0.0, 0.0]`

Et la sequence d'entree (3 pas de temps) :
```
x_1 = [1.0, 0.0]
x_2 = [0.0, 1.0]
x_3 = [1.0, 1.0]
```

1. Calculer `h_1` en detaillant les etapes :
   - Calculer `W_xh @ x_1`
   - Calculer `W_hh @ h_0`
   - Somme + tanh

2. Calculer `h_2` en utilisant `h_1` comme etat precedent.

3. Calculer `h_3` en utilisant `h_2`.

4. Question : si `x_1 = [0, 0]`, `x_2 = [0, 0]`, `x_3 = [0, 0]` (toutes entrees a zero), que devient `h_3` ? Pourquoi ?

5. Observer : pour t=3, l'etat cache `h_3` contient-il encore de l'information sur `x_1` ? Sous quelle forme ?

### Criteres de reussite

- [ ] Les 3 etats caches sont calcules numeriquement (precision 3 decimales)
- [ ] Le calcul intermediaire `W_xh @ x_t + W_hh @ h_{t-1}` est explicite a chaque pas
- [ ] L'explication sur la propagation de l'information est juste : `h_t` contient une compression non-lineaire de tous les `x_1..x_t`
- [ ] Pour la question 4, on obtient `h_3 = [0, 0]` (tanh(0) = 0, et toutes les entrees sont nulles)

---

## Exercice 2 : Vanishing gradient — calcul d'un produit de Jacobiens

### Objectif

Voir concretement pourquoi le gradient disparait (ou explose) dans un RNN.

### Consigne

Le gradient `dL/dh_0` est (en gros) le produit :
```
dL/dh_0 = dL/dh_T * prod_{t=1..T} dh_t/dh_{t-1}

avec dh_t/dh_{t-1} = diag(1 - h_t^2) @ W_hh
```

Pour simplifier, suppose que `diag(1 - h_t^2) ≈ I` (l'activation est dans sa zone lineaire). Le produit devient :
```
dL/dh_0 ≈ dL/dh_T * W_hh^T
```

1. Soit `W_hh = [[0.5, 0], [0, 0.5]]` (matrice diagonale avec spectre 0.5).
   - Pour T = 5, 10, 20, 50, calculer `||W_hh^T||` (norme de Frobenius).
   - Que se passe-t-il ? Comment appelle-t-on ce phenomene ?

2. Soit `W_hh = [[1.5, 0], [0, 1.5]]` (spectre 1.5).
   - Meme calcul pour T = 5, 10, 20, 50.
   - Que se passe-t-il maintenant ?

3. Soit `W_hh = [[1.0, 0], [0, 1.0]]` (spectre 1.0 = l'identite).
   - Meme calcul pour les memes T.
   - Pourquoi est-ce le cas "ideal" ? Est-ce realiste en pratique ?

4. Question conceptuelle : en termes de valeurs propres `lambda` de `W_hh`, donne le lien entre `lambda` et le comportement du gradient. Complete :
   - `lambda < 1` → ...
   - `lambda = 1` → ...
   - `lambda > 1` → ...

5. Cite 2 techniques qui mitigent ce probleme dans un RNN vanilla (sans changer l'architecture).

### Criteres de reussite

- [ ] Les 4 normes sont calculees pour `lambda = 0.5` (0.03125, ~10^-3, ~10^-6, ~10^-15)
- [ ] Les 4 normes sont calculees pour `lambda = 1.5` (7.6, 57.7, ~3300, ~6*10^8)
- [ ] Les 4 normes pour `lambda = 1` restent constantes (~1)
- [ ] Les noms sont corrects : vanishing, exploding, stable
- [ ] Gradient clipping et initialisation orthogonale sont cites

---

## Exercice 3 : LSTM gates — lequel fait quoi ?

### Objectif

Identifier le role de chaque gate du LSTM et comprendre l'equation du cell state.

### Consigne

Les 4 gates d'un LSTM sont :
```
f_t = sigmoid(W_f @ [h_{t-1}, x_t])
i_t = sigmoid(W_i @ [h_{t-1}, x_t])
o_t = sigmoid(W_o @ [h_{t-1}, x_t])
c~_t = tanh(W_c @ [h_{t-1}, x_t])
```

Et les mises a jour :
```
c_t = f_t * c_{t-1} + i_t * c~_t
h_t = o_t * tanh(c_t)
```

1. Pour chaque gate (`f`, `i`, `o`, `c~`), repondre :
   - Activation utilisee (sigmoid ou tanh) ?
   - Domaine de sortie ([0,1] ou [-1,1]) ?
   - Role fonctionnel ?

2. Scenario : a un pas de temps donne, on observe :
   ```
   f_t = [1.0, 1.0, 1.0]
   i_t = [0.0, 0.0, 0.0]
   ```
   Que vaut `c_t` en fonction de `c_{t-1}` ? Qu'est-ce que ce comportement represente ? (Indice : pensez a la metaphore du carnet de notes).

3. Scenario inverse :
   ```
   f_t = [0.0, 0.0, 0.0]
   i_t = [1.0, 1.0, 1.0]
   ```
   Que vaut `c_t` ? Qu'est-ce que ce comportement represente ?

4. Scenario mixte :
   ```
   f_t = [1.0, 0.0, 1.0]
   i_t = [0.0, 1.0, 0.0]
   c_{t-1} = [2.0, 3.0, 4.0]
   c~_t   = [5.0, 6.0, 7.0]
   ```
   Calculer `c_t` composante par composante. Interpreter : chaque dimension fait quoi ?

5. Pourquoi l'equation `c_t = f_t * c_{t-1} + i_t * c~_t` permet au gradient de NE PAS vanish (contrairement au RNN vanilla) ? Explique en une phrase.

### Criteres de reussite

- [ ] Tableau correct : f, i, o en sigmoid (domaine [0,1]) ; c~ en tanh (domaine [-1,1])
- [ ] Scenario 1 : `c_t = c_{t-1}` (on garde la memoire intacte, "carnet fige")
- [ ] Scenario 2 : `c_t = c~_t` (on remplace completement, "page blanche")
- [ ] Scenario 3 : `c_t = [2.0, 6.0, 4.0]` (chaque dim applique son regle : dim 1 et 3 gardent, dim 2 remplace)
- [ ] L'explication du no-vanishing mentionne : quand `f_t ≈ 1`, `c_{t-1}` passe directement, pas de multiplication par `W_hh` (le gradient passe a travers l'addition)

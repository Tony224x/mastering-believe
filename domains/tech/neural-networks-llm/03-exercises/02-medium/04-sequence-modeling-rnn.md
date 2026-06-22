# Exercices Medium — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)

---

## Exercice 4 : RNN cell complete (forward + backward) avec gradient check

### Objectif

Implementer une cellule RNN vanilla complete (forward ET backward) en NumPy et prouver la correction du backward par differences finies.

### Consigne

En te basant sur les equations du cours :
```
h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
y_t = W_hy @ h_t + b_y
```

1. Implementer `rnn_cell_forward(x_t, h_prev, params)` qui renvoie `h_t`, `y_t` et un cache.

2. Implementer `rnn_cell_backward(dh_t, dy_t, cache, params)` qui renvoie les gradients :
   - `dW_xh`, `dW_hh`, `dW_hy`, `db_h`, `db_y`
   - `dh_prev` (le gradient qui remonte vers le pas de temps precedent)
   - Detailler la chain rule : derivee de `tanh` = `1 - h_t^2`, `dW_xh = outer(dz_h, x_t)`, etc.

3. **Gradient check** : pour une cellule isolee (un seul pas de temps), avec une loss MSE arbitraire sur `y_t`, compare chaque gradient analytique au gradient numerique :
   ```
   grad_num = (loss(param + eps) - loss(param - eps)) / (2 * eps)   avec eps = 1e-5
   ```
   L'erreur relative doit etre < 1e-5 pour TOUS les parametres.

4. Question : pourquoi `dh_prev` est le terme qui cause vanishing/exploding quand on l'applique sur T pas de temps ? (Indice : il multiplie `W_hh^T` a chaque pas.)

### Criteres de reussite

- [ ] `rnn_cell_forward` renvoie les bonnes shapes et un cache reutilisable
- [ ] Les 5 gradients de parametres + `dh_prev` sont calcules avec la chain rule explicitee
- [ ] Le gradient check passe pour tous les parametres (erreur relative < 1e-5)
- [ ] La derivee de tanh est bien `1 - h_t^2` (et non `1 - tanh(z)^2` recalcule a partir de z)
- [ ] L'explication relie `dh_prev = W_hh^T @ dz_h` au phenomene vanishing/exploding

---

## Exercice 5 : BPTT complet sur une sequence + monitoring du gradient

### Objectif

Implementer la Backpropagation Through Time (BPTT) sur une sequence complete et observer empiriquement comment la norme du gradient evolue selon la longueur de sequence.

### Consigne

1. Implementer une boucle d'entrainement char-level sur un mini-corpus repetitif (ex : `"abcabcabc..."` ou `"le chat dort. "*10`) :
   - Forward : derouler le RNN sur `seq_len` pas, accumuler la cross-entropy
   - Backward (BPTT) : remonter le temps en accumulant les gradients (les memes parametres sont utilises a chaque pas → on **somme** les contributions)
   - Gradient clipping par norme globale (clip a 5.0)
   - Mise a jour SGD

2. Entrainer sur `seq_len ∈ {5, 15, 30}` et comparer :
   - La loss finale
   - La norme moyenne du gradient (avant clipping) sur les 50 dernieres iterations

3. **Monitoring vanishing** : pour une sequence de longueur T, decompose la norme du gradient `||dL/dh_0||` en fonction du pas de temps. Trace (texte ou matplotlib) la norme `||dL/dh_t||` pour t = T, T-1, ..., 0. Observe-t-elle une decroissance exponentielle vers t=0 ?

4. Question : pourquoi accumule-t-on (somme) les gradients de `W_hh` sur tous les pas plutot que de les moyenner ou de prendre le dernier ?

### Criteres de reussite

- [ ] La BPTT accumule correctement les gradients de chaque parametre partage sur tous les pas
- [ ] `dh_next` (gradient venant du futur) est combine avec `dy` (gradient du output local) a chaque pas
- [ ] Le gradient clipping est applique sur la norme globale de tous les gradients
- [ ] L'observation montre que `||dL/dh_t||` decroit en remontant vers t=0 (vanishing) pour un RNN vanilla
- [ ] L'explication de l'accumulation est correcte : un parametre partage recoit un gradient = somme des derivees partielles de chaque usage (regle de la somme sur les chemins)

---

## Exercice 6 : LSTM cell from scratch (forward)

### Objectif

Implementer le forward pass complet d'une cellule LSTM et verifier numeriquement le comportement des gates.

### Consigne

Equations du LSTM (cours) :
```
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)    # forget gate
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)    # input gate
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    # output gate
c~_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)      # candidate
c_t = f_t * c_{t-1} + i_t * c~_t             # new cell state
h_t = o_t * tanh(c_t)                        # new hidden state
```

1. Implementer `lstm_cell_forward(x_t, h_prev, c_prev, params)`. Utilise la concatenation `[h_{t-1}, x_t]` et une seule matrice par gate.

2. Derouler la cellule sur une sequence de 4 pas et afficher `c_t` et `h_t` a chaque pas.

3. **Verification des gates** : force manuellement les biais pour creer 2 regimes et verifie le comportement :
   - **Memoire figee** : initialise `b_f` tres grand (forget ≈ 1) et `b_i` tres negatif (input ≈ 0). Verifie que `c_t ≈ c_{t-1}` (la memoire est conservee).
   - **Reset complet** : initialise `b_f` tres negatif (forget ≈ 0). Verifie que `c_t ≈ i_t * c~_t` (l'ancien etat est oublie).

4. **Comparaison RNN vs LSTM sur le gradient** : avec `f_t ≈ 1`, montre numeriquement que `dc_t/dc_{t-1} = f_t ≈ 1` (le gradient du cell state passe quasi intact, contrairement au RNN ou `dh_t/dh_{t-1}` implique une multiplication par `W_hh`).

### Criteres de reussite

- [ ] Le forward LSTM est correct (4 gates + cell state + hidden state)
- [ ] La concatenation `[h_{t-1}, x_t]` et les shapes des matrices sont coherentes
- [ ] Le regime "memoire figee" donne `c_t ≈ c_{t-1}` numeriquement
- [ ] Le regime "reset complet" donne `c_t ≈ i_t * c~_t`
- [ ] La demonstration `dc_t/dc_{t-1} = f_t` explique pourquoi le LSTM ne vanish pas (chemin additif/gate au lieu de matmul repetee)

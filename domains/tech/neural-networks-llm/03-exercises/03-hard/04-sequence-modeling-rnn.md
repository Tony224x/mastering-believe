# Exercices Hard — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)

---

## Exercice 7 : LSTM complet (forward + BPTT) avec gradient check

### Objectif

Implementer le backward pass complet d'une cellule LSTM (le plus dur des exercices de la journee) et prouver sa correction par gradient check sur une sequence multi-pas.

### Consigne

Tu as deja le forward LSTM (Medium ex. 6). Ici on derive et on implemente le backward.

1. **Backward d'une cellule** : etant donne `dh_t` (du futur + du output) et `dc_t` (du futur), calculer :
   - `do_t = dh_t * tanh(c_t)` puis backprop a travers la sigmoid de `o`
   - `dc_t += dh_t * o_t * (1 - tanh(c_t)^2)` (le gradient sur `c_t` recoit aussi la contribution de `h_t`)
   - `df_t = dc_t * c_{t-1}`, `di_t = dc_t * c~_t`, `dc~_t = dc_t * i_t`
   - Backprop a travers les sigmoids (`f, i, o`) et le tanh (`c~`)
   - Gradients des matrices de poids et des biais
   - `dh_prev` et `dc_prev` (les deux gradients qui remontent au pas precedent)

2. **BPTT sur une sequence** : derouler le forward sur T pas (en cachant chaque etape), puis remonter le temps en accumulant les gradients des parametres et en propageant `dh_next` ET `dc_next`.

3. **Gradient check rigoureux** : sur une sequence de T=4 pas avec une loss = somme des MSE sur tous les `h_t`, compare chaque gradient analytique (matrices + biais) au gradient numerique (eps = 1e-5). Erreur relative < 1e-4 pour tous.

4. **Experience vanishing** : entraine un LSTM et un RNN vanilla sur une tache "copy" : reproduire le premier caractere apres un delai de T pas (`X..........X` → cible = premier char). Compare l'accuracy pour T ∈ {5, 20, 50}. Le LSTM doit tenir bien plus longtemps.

### Criteres de reussite

- [ ] Le backward LSTM est correct : `dc_t` recoit DEUX contributions (du futur `dc_next` et via `h_t = o_t * tanh(c_t)`)
- [ ] Les 8 matrices/biais (4 gates) ont leurs gradients calcules avec la chain rule complete
- [ ] `dh_prev` et `dc_prev` sont tous deux propages au pas precedent
- [ ] Le gradient check passe pour TOUS les parametres sur une sequence multi-pas (erreur < 1e-4)
- [ ] L'experience copy montre que le LSTM conserve l'accuracy a T=50 la ou le RNN vanilla s'effondre

---

## Exercice 8 : GRU from scratch + comparaison rigoureuse LSTM/GRU/RNN

### Objectif

Implementer le GRU complet (forward + backward), le comparer au LSTM et au RNN sur une vraie tache, et analyser le tradeoff capacite/parametres.

### Consigne

Equations GRU (cours) :
```
z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)        # update gate
r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)        # reset gate
h~_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)    # candidate (note le r_t * h_{t-1})
h_t = (1 - z_t) * h_{t-1} + z_t * h~_t           # final hidden
```

1. **Forward GRU** : implementer `gru_cell_forward`. Attention au terme `r_t * h_{t-1}` DANS le calcul du candidat (piege classique).

2. **Backward GRU** : deriver et implementer le backward. Le point delicat : `h_{t-1}` apparait a TROIS endroits (dans `z_t`, `r_t`, et via `r_t * h_{t-1}` dans le candidat, plus le terme `(1 - z_t) * h_{t-1}`). Tous ces chemins contribuent a `dh_prev`. Valide par gradient check (erreur < 1e-4).

3. **Comptage de parametres** : pour `hidden_dim = H` et `input_dim = D`, donner le nombre exact de parametres de chaque cellule (RNN, LSTM, GRU). Verifier que GRU ≈ 75% du LSTM.
   - RNN : `H*(D+H) + H` (biais)
   - LSTM : `4 * (H*(D+H) + H)`
   - GRU : `3 * (H*(D+H) + H)`

4. **Benchmark** : entrainer RNN, LSTM, GRU avec le MEME budget de parametres (ajuster `hidden_dim` pour egaliser) sur une tache de prediction de sequence (char-level next-token). Comparer : loss finale, vitesse de convergence, temps par iteration. Produire un tableau et conclure.

### Criteres de reussite

- [ ] Le forward GRU est correct, avec `r_t * h_{t-1}` bien place dans le candidat
- [ ] Le backward GRU gere les trois (voire quatre) chemins de `h_{t-1}` vers `h_t`
- [ ] Le gradient check passe pour tous les parametres (erreur < 1e-4)
- [ ] Le comptage de parametres est exact et la ratio GRU/LSTM ≈ 0.75 est verifiee
- [ ] Le benchmark a budget de parametres egal est equitable (meme nombre de params, meme seed, meme corpus) et l'analyse est juste (differences marginales en pratique, GRU plus rapide par iteration)

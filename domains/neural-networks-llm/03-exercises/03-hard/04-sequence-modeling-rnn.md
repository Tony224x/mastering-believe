# Exercices Hard — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)

---

## Exercice 7 : LSTM from scratch + mesure du gradient longue distance

### Objectif

Implementer un LSTM complet (cellule + deroulement sur T timesteps) et **prouver numeriquement** qu'il propage les gradients sur de longues distances la ou le RNN vanilla les perd.

### Consigne

1. Implementer `lstm_forward(X, h0, c0, params)` qui deroule la cellule LSTM sur une sequence `X` de shape `(T, input_dim)` et retourne tous les `h_t` et `c_t`
   - Equations : `f, i, o = sigmoid(W. @ [h_prev; x] + b.)`, `g = tanh(Wg @ [h_prev; x] + bg)`, `c = f * c_prev + i * g`, `h = o * tanh(c)`
   - **Initialiser le bias du forget gate a +1.0** (astuce standard : au depart, le LSTM retient tout)

2. Implementer le RNN vanilla equivalent `rnn_forward` (meme input_dim, meme hidden_dim)

3. Mesurer la sensibilite de la loss au PREMIER input :
   - Loss : `L = sum(h_T ** 2)` (depend uniquement du dernier hidden state)
   - Calculer `dL/dx_0` par **differences finies** (perturber chaque composante de `x_0` de ±1e-5, refaire le forward complet) — pas besoin de backward analytique ici
   - Norme du gradient `||dL/dx_0||` pour `T ∈ {5, 20, 40}`

4. Comparer RNN vs LSTM dans un tableau : `T | ||dL/dx_0|| RNN | ||dL/dx_0|| LSTM | ratio LSTM/RNN`

5. Expliquer en commentaire POURQUOI le LSTM gagne : le chemin `c_t = f * c_{t-1} + ...` est additif (le gradient traverse via `f` sans multiplication par une matrice ni saturation tanh repetee)

Dimensions : `input_dim=4`, `hidden_dim=8`, seed fixe, poids `randn * 0.5` pour forcer le vanishing du RNN.

### Criteres de reussite

- [ ] Le LSTM est implemente from scratch (4 gates, pas de framework) et les shapes sont correctes
- [ ] Le forget bias est initialise a +1.0 et son role est explique en commentaire
- [ ] Le gradient par differences finies est calcule pour les 3 longueurs T pour les DEUX modeles
- [ ] A T=40, `||dL/dx_0||` du RNN est < 1e-6 ET le ratio LSTM/RNN est > 1e3 (le LSTM conserve un signal exploitable)
- [ ] Le tableau comparatif est affiche et l'explication du chemin additif de la cell state est presente
- [ ] Tout tourne en < 30 s (les differences finies sur x_0 ne demandent que 2 * input_dim forwards par config)

---

## Exercice 8 : BPTT complet from scratch + exploding gradients & clipping

### Objectif

Implementer la Backpropagation Through Time complete d'un RNN vanilla, la valider par gradient check, puis provoquer et corriger une explosion de gradient avec le clipping.

### Consigne

1. Implementer `rnn_bptt(X, y, h0, params)` pour un RNN vanilla avec loss MSE sur le dernier output :
   - Forward : stocker tous les `h_t` et `z_t` (pre-activations)
   - Backward : derouler la chaine a l'envers, accumuler `dW_xh`, `dW_hh`, `db_h` **a chaque timestep** (les poids sont partages dans le temps !)
   - Ne pas oublier le terme recurrent : `dL/dh_{t-1} += (dL/dz_t) @ W_hh.T`

2. **Gradient check** : comparer chaque element de `dW_xh`, `dW_hh`, `db_h` aux differences finies centrees (eps=1e-5) sur une sequence T=6. Erreur relative < 1e-5 partout.

3. **Provoquer l'explosion** : re-initialiser `W_hh` avec un rayon spectral ~1.5 (`W_hh = 1.5 * W / max(|eigvals(W)|)`) et utiliser des inputs TRES petits (`X * 0.01`) pour rester dans le regime quasi-lineaire de tanh. Mesurer `||dW_hh||` pour `T ∈ {3, 6, 9, 12}` et montrer la croissance exponentielle (~1.5^T). **Observation a documenter** : si on laisse les inputs grands ou T tres long, la saturation de tanh (derivee → 0) finit par PLAFONNER l'explosion — mesurer aussi T=40 pour le constater. C'est pour ca que l'explosion en pratique arrive par "spikes" quand les etats traversent la zone lineaire.

4. **Clipping par norme globale** : implementer
   ```python
   def clip_gradients(grads, max_norm):
       # global norm over ALL gradients, rescale if above threshold
   ```
   et verifier : si la norme globale depasse `max_norm`, apres clipping elle vaut exactement `max_norm` ; sinon les gradients sont inchanges.

5. Mini demonstration d'entrainement : sur une tache jouet (predire `sum(X)` a partir de la sequence), entrainer 200 steps avec et sans clipping (lr=0.01, W_hh explosif au depart). Sans clipping la loss doit diverger (NaN ou > 1e3) ; avec clipping elle doit decroitre.

### Criteres de reussite

- [ ] Le BPTT accumule bien les gradients sur tous les timesteps (poids partages)
- [ ] Gradient check : erreur relative max < 1e-5 sur les 3 gradients, tous elements testes
- [ ] La croissance de `||dW_hh||` est mesuree en regime quasi-lineaire : `||dW_hh||(T=12) > 20 * ||dW_hh||(T=3)` ; le plafonnement par saturation tanh a T=40 est constate et explique
- [ ] `clip_gradients` preserve la DIRECTION du gradient (rescale global, pas de clip element par element) — verifie par un test de colinearite
- [ ] L'entrainement sans clipping diverge, avec clipping converge (loss finale < loss initiale / 10)
- [ ] Execution totale < 30 s

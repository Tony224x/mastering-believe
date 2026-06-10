# Exercices Medium — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)

---

## Exercice 4 : RNN forward batche — predire puis verifier les shapes

### Objectif

Implementer le forward pass d'un RNN vanilla sur un batch complet de sequences et maitriser les shapes a chaque etape (la source n°1 de bugs en sequence modeling).

### Consigne

1. **Avant de coder**, predire sur papier les shapes de chaque tenseur pour :
   - `batch_size=3`, `T=5` (timesteps), `input_dim=4`, `hidden_dim=6`, `output_dim=2`
   - A predire : `X`, `h_t`, `W_xh`, `W_hh`, `W_hy`, `H` (tous les hidden states), `Y` (tous les outputs)

2. Implementer :

```python
def rnn_forward(X, h0, W_xh, W_hh, W_hy, b_h, b_y):
    """X: (batch, T, input_dim), h0: (batch, hidden_dim).
    Returns H: (batch, T, hidden_dim), Y: (batch, T, output_dim)."""
```

   avec la recurrence `h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)` et `y_t = h_t @ W_hy + b_y`.

3. Verifier l'implementation batchee contre une boucle de reference qui traite **un sample a la fois, un timestep a la fois** (version "naive" sans batch). Les deux doivent donner exactement les memes valeurs.

4. Tester avec un `h0` non nul et verifier que `H[:, 0]` depend bien de `h0` (changer `h0` doit changer `H[:, 0]`).

### Criteres de reussite

- [ ] Les 7 shapes predites sur papier sont correctes (verifiees par `assert tensor.shape == ...`)
- [ ] `H.shape == (3, 5, 6)` et `Y.shape == (3, 5, 2)`
- [ ] Difference max entre version batchee et version naive < 1e-12
- [ ] Le test de dependance a `h0` passe (les hidden states changent quand h0 change)
- [ ] Aucune boucle sur la dimension batch dans `rnn_forward` (seule la boucle sur T est permise)

---

## Exercice 5 : Gradient check d'une cellule RNN

### Objectif

Deriver et implementer le backward pass d'UNE cellule RNN (un seul timestep), puis le valider par differences finies — la technique de verification universelle.

### Consigne

Pour une cellule `h = tanh(x @ W_xh + h_prev @ W_hh + b_h)` et une loss scalaire `L = 0.5 * sum(h**2)` :

1. Deriver a la main les 5 gradients : `dL/dW_xh`, `dL/dW_hh`, `dL/db_h`, `dL/dx`, `dL/dh_prev`
   - Rappel : `d(tanh(z))/dz = 1 - tanh(z)^2`, et `dL/dh = h` pour cette loss
2. Implementer `rnn_cell_backward(x, h_prev, W_xh, W_hh, b_h)` qui retourne les 5 gradients analytiques
3. Implementer un gradient check par differences finies centrees :
   `grad_num = (L(theta + eps) - L(theta - eps)) / (2 * eps)` avec `eps = 1e-6`
4. Verifier les 5 gradients element par element avec l'erreur relative
   `|analytique - numerique| / (|analytique| + |numerique| + 1e-8)`

Dimensions de test : `input_dim=3`, `hidden_dim=4`, valeurs initialisees avec `np.random.randn * 0.5`, seed fixe.

### Criteres de reussite

- [ ] Les 5 gradients analytiques sont implementes (pas seulement W_xh et W_hh)
- [ ] L'erreur relative max sur TOUS les elements de TOUS les gradients est < 1e-6
- [ ] Le gradient check teste chaque element individuellement (pas juste la norme globale)
- [ ] La derivation papier est retranscrite en commentaires (chaque ligne du backward citee)
- [ ] Le code distingue clairement `dL/dh` (entrant) et la propagation vers `dL/dh_prev` (sortant)

---

## Exercice 6 : Debugger une cellule LSTM cassee

### Objectif

Detecter et corriger des bugs classiques dans une implementation LSTM — exactement le genre d'erreurs silencieuses qui "entrainent quand meme" mais n'apprennent rien a long terme.

### Consigne

Le code suivant contient **3 bugs**. Trouver chacun, expliquer pourquoi c'est faux (quel comportement ca casse), et corriger :

```python
def lstm_cell_buggy(x, h_prev, c_prev, params):
    Wf, Wi, Wg, Wo, bf, bi, bg, bo = params
    z = np.concatenate([h_prev, x])

    f = np.tanh(Wf @ z + bf)            # BUG ?
    i = sigmoid(Wi @ z + bi)
    g = np.tanh(Wg @ z + bg)
    o = sigmoid(Wo @ z + bo)

    c = f * g                            # BUG ?
    h = o * c                            # BUG ?
    return h, c
```

1. Identifier les 3 bugs et ecrire pour chacun : symptome attendu pendant l'entrainement
2. Ecrire `lstm_cell_fixed` avec les equations correctes :
   `f, i, o = sigmoid(...)`, `g = tanh(...)`, `c = f * c_prev + i * g`, `h = o * tanh(c)`
3. Verifier sur des valeurs imposees (seed 42, `input_dim=3`, `hidden_dim=4`) que :
   - les 3 gates `f, i, o` sont dans `[0, 1]`
   - avec `f = 1` force et `i = 0` force, `c == c_prev` (la memoire est preservee parfaitement)
   - avec `f = 0` force et `i = 1` force, `c == g` (la memoire est ecrasee par le candidat)
4. Comparer `lstm_cell_buggy` et `lstm_cell_fixed` sur le meme input et mesurer l'ecart

### Criteres de reussite

- [ ] Les 3 bugs sont identifies : forget gate en tanh (peut etre negative → c oscille), c sans `c_prev` (aucune memoire ne passe → equivalent a un reseau sans recurrence de cellule), h sans `tanh(c)` (h non borne → activations qui explosent)
- [ ] La version corrigee respecte les equations LSTM standard
- [ ] Les tests de gates forcees passent exactement (`c == c_prev` a 1e-12 pres quand f=1, i=0)
- [ ] Chaque bug est commente avec son symptome d'entrainement
- [ ] Les gates de la version corrigee sont verifiees dans [0, 1]

# Exercices Hard — Jour 2 : Reseaux denses (MLP)

---

## Exercice 7 : Implementer un mini-framework de deep learning

### Objectif

Creer un micro-framework modulaire de deep learning (style PyTorch simplifie) avec des couches empilables, perte composable, et optimizers interchangeables.

### Consigne

Implementer les classes suivantes en NumPy pur :

**1. Couches (Layers) — chaque couche a un `forward()` et un `backward()` :**

```python
class Linear:
    """Z = X @ W + b. Stocke X pour le backward."""
    def forward(self, X): ...
    def backward(self, grad_output): ...   # retourne grad_input

class ReLULayer:
    """A = max(0, Z). Stocke Z pour le backward."""
    def forward(self, Z): ...
    def backward(self, grad_output): ...

class SigmoidLayer:
    def forward(self, Z): ...
    def backward(self, grad_output): ...

class SoftmaxLayer:
    def forward(self, Z): ...
    # Note: backward combine avec la loss (pas standalone)

class DropoutLayer:
    def __init__(self, rate=0.5): ...
    def forward(self, A, training=True): ...
    def backward(self, grad_output): ...

class BatchNormLayer:
    """Normalize activations to mean=0, var=1, then scale/shift.
    BN(x) = γ * (x - μ) / √(σ² + ε) + β
    γ and β are learnable. μ and σ are batch statistics (training) or
    running averages (inference)."""
    def forward(self, Z, training=True): ...
    def backward(self, grad_output): ...
```

**2. Modele sequentiel :**

```python
class Sequential:
    """Empile des couches et propage forward/backward automatiquement."""
    def __init__(self, layers): ...
    def forward(self, X, training=True): ...
    def backward(self, grad_loss): ...
    def parameters(self): ...   # retourne tous les (W, dW) pour l'optimizer
```

**3. Losses :**
```python
class MSELoss:
    def forward(self, y_pred, y_true): ...   # retourne loss scalaire
    def backward(self): ...                   # retourne gradient

class BCEWithLogitsLoss:
    """Combine sigmoid + BCE en une seule classe (numeriquement stable)."""
    def forward(self, logits, y_true): ...
    def backward(self): ...

class CrossEntropyLoss:
    """Combine softmax + CCE en une seule classe."""
    def forward(self, logits, y_true_onehot): ...
    def backward(self): ...
```

**4. Tester le framework :**

```python
model = Sequential([
    Linear(2, 64),
    BatchNormLayer(64),
    ReLULayer(),
    DropoutLayer(0.3),
    Linear(64, 32),
    ReLULayer(),
    Linear(32, 1),
])
criterion = BCEWithLogitsLoss()
optimizer = AdamOptimizer(model.parameters(), lr=0.001)

# Training loop
for epoch in range(500):
    logits = model.forward(X_train, training=True)
    loss = criterion.forward(logits, y_train)
    grad = criterion.backward()
    model.backward(grad)
    optimizer.step()
```

Tester sur 3 datasets : spirales, moons, cercles concentriques. Accuracy > 90% sur les 3.

**5. Gradient check** : verifier que les gradients de CHAQUE couche (Linear, BatchNorm) sont corrects en les comparant aux differences finies.

### Criteres de reussite

- [ ] Architecture modulaire : chaque couche est independante et testable
- [ ] Le forward pass chaine les couches automatiquement
- [ ] Le backward pass propage les gradients en sens inverse automatiquement
- [ ] BatchNorm est implemente correctement (comportement different train/inference)
- [ ] Dropout est implemente avec inverted dropout
- [ ] Les 3 losses fonctionnent et sont numeriquement stables
- [ ] Gradient check passe pour toutes les couches (erreur < 1e-5)
- [ ] Les 3 datasets sont resolus avec accuracy > 90%
- [ ] Le code est modulaire et reutilisable (on peut ajouter une nouvelle couche facilement)

---

## Exercice 8 : Learning rate finder et optimizer landscape

### Objectif

Implementer le LR Range Test (Smith 2017) et visualiser le paysage d'optimisation pour comprendre les differences entre optimizers.

### Consigne

**Partie A — LR Range Test :**

1. Implementer un "LR finder" qui augmente progressivement le learning rate pendant l'entrainement :
   ```
   lr_min = 1e-7, lr_max = 10
   Pour chaque batch :
     lr_current = lr_min * (lr_max / lr_min) ^ (batch / n_batches)
     Entrainer avec lr_current
     Enregistrer (lr_current, loss)
   ```

2. Tracer log(lr) vs loss. Le LR optimal est juste avant que la loss ne commence a augmenter (typiquement la zone avec la pente la plus negative).

3. Tester sur un MLP [2, 64, 32, 1] avec le dataset spirales. Comparer le LR trouve avec les heuristiques classiques (0.001, 0.01).

**Partie B — Visualisation du paysage d'optimisation :**

4. Pour un petit MLP [2, 4, 1] avec seulement ~17 parametres, visualiser la trajectoire des 4 optimizers :
   - Projeter les 17 parametres en 2D avec PCA (sur la trajectoire des poids au cours de l'entrainement)
   - Tracer les trajectoires des 4 optimizers sur le meme plan 2D
   - Superposer les contours de loss (interpoles)

5. Observer et expliquer :
   - Pourquoi SGD fait des zigzags ?
   - Pourquoi Momentum "depasse" parfois le minimum ?
   - Pourquoi Adam converge en ligne plus directe ?
   - Les 4 optimizers convergent-ils au meme point final ?

**Partie C — Cosine Annealing :**

6. Implementer un cosine annealing LR schedule :
   ```
   lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
   ```

7. Comparer Adam fixe vs Adam + cosine annealing sur 2000 epochs. Le schedule aide-t-il ?

### Criteres de reussite

- [ ] Le LR finder est implemente correctement et produit une courbe interpretable
- [ ] Le LR optimal identifie par le finder est coherent avec les resultats experimentaux
- [ ] La projection PCA des trajectoires est correcte et visualisable
- [ ] Les trajectoires des 4 optimizers montrent des comportements distincts
- [ ] Les observations sont expliquees avec les formules des optimizers
- [ ] Le cosine annealing est implemente correctement
- [ ] L'analyse finale est argumentee avec des resultats empiriques

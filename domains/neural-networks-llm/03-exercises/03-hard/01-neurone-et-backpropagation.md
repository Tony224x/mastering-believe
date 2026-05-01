# Exercices Hard — Jour 1 : Le neurone & backpropagation

---

## Exercice 7 : Reseau multi-couche generalise from scratch

### Objectif

Implementer un reseau de neurones a N couches (profondeur arbitraire) avec backpropagation generalisee, sans aucune librairie de deep learning.

### Consigne

Creer une classe `DeepNetwork` qui accepte une architecture arbitraire :

```python
# Exemple : 2 inputs, 2 couches cachees de 8 et 4 neurones, 1 output
net = DeepNetwork(layer_sizes=[2, 8, 4, 1], activations=['relu', 'relu', 'sigmoid'])
```

La classe doit implementer :

1. **Initialisation** : Xavier init pour sigmoid/tanh, He init pour ReLU
   - Xavier : `W * sqrt(1/n_in)`
   - He : `W * sqrt(2/n_in)` (ReLU a besoin de variance plus grande car il tue la moitie des neurones)

2. **Forward pass** generique pour N couches :
   ```
   Pour chaque couche l = 1, ..., L :
       z_l = a_{l-1} @ W_l + b_l
       a_l = activation_l(z_l)
   ```

3. **Backward pass** generique pour N couches :
   ```
   Pour l = L, L-1, ..., 1 (en remontant) :
       δ_l = ∂L/∂z_l
       ∂L/∂W_l = a_{l-1}^T @ δ_l
       ∂L/∂b_l = sum(δ_l)
       propager δ vers la couche precedente
   ```

4. **Support des 3 activations** : sigmoid, tanh, ReLU (avec leurs derivees)

5. **Support de 2 losses** : MSE et Binary Cross-Entropy

6. **Tester sur 3 problemes** :
   - XOR (2→4→1) — verifie que ca marche sur le cas simple
   - Cercles concentriques (`sklearn.datasets.make_circles`) — necessite profondeur
   - Spirale (`sklearn.datasets.make_moons`) — frontiere de decision complexe

7. **Verification** : pour chaque poids, comparer le gradient analytique (backprop) avec le gradient numerique (differences finies). L'erreur relative doit etre < 1e-5.

```python
def gradient_check(net, X, y, epsilon=1e-7):
    """Compare analytical gradients with numerical gradients."""
    for l in range(len(net.weights)):
        for i in range(net.weights[l].shape[0]):
            for j in range(net.weights[l].shape[1]):
                # Save original weight
                original = net.weights[l][i, j]

                # Compute f(w + epsilon)
                net.weights[l][i, j] = original + epsilon
                net.forward(X)
                loss_plus = net.compute_loss(y)

                # Compute f(w - epsilon)
                net.weights[l][i, j] = original - epsilon
                net.forward(X)
                loss_minus = net.compute_loss(y)

                # Numerical gradient
                numerical = (loss_plus - loss_minus) / (2 * epsilon)

                # Restore original weight
                net.weights[l][i, j] = original

                # Compare with analytical gradient
                analytical = net.grad_W[l][i, j]
                error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-8)
                if error > 1e-5:
                    print(f"MISMATCH layer {l}, weight [{i},{j}]: "
                          f"analytical={analytical:.8f}, numerical={numerical:.8f}, error={error:.8f}")
```

### Criteres de reussite

- [ ] La classe gere une architecture arbitraire (testee avec au moins 3 architectures differentes)
- [ ] L'initialisation est correcte (Xavier pour sigmoid/tanh, He pour ReLU)
- [ ] Le forward pass est generique (boucle sur les couches, pas de code en dur)
- [ ] Le backward pass est generique et correct (gradient check passe pour TOUTES les couches)
- [ ] Les 3 problemes sont resolus avec accuracy > 95%
- [ ] Le gradient check confirme l'exactitude des gradients analytiques (erreur < 1e-5)
- [ ] Le code est structure et commente

---

## Exercice 8 : Visualiser la surface de loss et la trajectoire du gradient

### Objectif

Comprendre geometriquement ce que fait le gradient descent en visualisant la surface de loss et la trajectoire des poids.

### Consigne

Pour un neurone UNIQUE avec 2 poids (w1, w2) et bias fixe a 0, sur un probleme de classification simple (2 points) :

```python
# 2 points seulement pour pouvoir visualiser la loss surface en 3D
X = np.array([[1.0, 0.0], [0.0, 1.0]])
y = np.array([[1.0], [0.0]])
# Neurone : a = sigmoid(w1*x1 + w2*x2)
# Loss : MSE
```

1. **Calculer la loss pour une grille de (w1, w2)** : de -5 a 5 avec un pas de 0.1 → matrice 100x100 de valeurs de loss

2. **Tracer la surface de loss en 3D** (matplotlib `plot_surface`) :
   - Axes x et y : w1 et w2
   - Axe z : loss
   - Observer : ou est le minimum ? La surface est-elle convexe ?

3. **Tracer les contours en 2D** (`contour` ou `contourf`) et superposer la trajectoire du gradient descent :
   - Initialiser w1=4, w2=-4 (loin du minimum)
   - Entrainer pendant 100 epochs avec lr=0.5
   - A chaque epoch, enregistrer (w1, w2)
   - Tracer la trajectoire comme une serie de fleches sur les contours

4. **Comparer 3 trajectoires** avec 3 learning rates differents (0.1, 1.0, 5.0) sur le meme plot de contours

5. **Question** : la surface de loss d'un reseau avec couche cachee est-elle convexe ? Pourquoi est-ce un probleme en pratique ? (repondre dans un commentaire du code)

### Criteres de reussite

- [ ] La surface de loss 3D est correcte et visualisable
- [ ] Les contours 2D sont corrects
- [ ] La trajectoire du gradient descent est tracee et montre la convergence vers le minimum
- [ ] Les 3 LR differents montrent des comportements distincts (lent, optimal, oscillation)
- [ ] La reponse a la question sur la convexite est correcte : non convexe → minima locaux, saddle points, le GD peut converger vers un mauvais minimum selon l'initialisation
- [ ] Le code est propre et les visualisations sont lisibles (labels, titres, legendes)

# Exercices Medium — Jour 1 : Le neurone & backpropagation

---

## Exercice 4 : Backpropagation a la main (reseau 2-1)

### Objectif

Calculer TOUS les gradients d'un mini-reseau a la main, sans code, en utilisant la chain rule.

### Consigne

Reseau : 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid), loss MSE.

Parametres :
```
x1 = 1.0, x2 = 0.0
w1 = 0.5 (x1→h1), w2 = -0.3 (x2→h1), b_h1 = 0.0
w3 = 0.1 (x1→h2), w4 = 0.7 (x2→h2), b_h2 = 0.0
w5 = 0.4 (h1→o),  w6 = -0.6 (h2→o),  b_o = 0.1
y = 1.0
```

**A calculer sur papier (ou dans un fichier texte) :**

1. Forward pass complet : z_h1, a_h1, z_h2, a_h2, z_o, a_o
2. Loss MSE
3. Backprop — tous les gradients :
   - ∂L/∂w5, ∂L/∂w6, ∂L/∂b_o
   - ∂L/∂w1, ∂L/∂w2, ∂L/∂b_h1
   - ∂L/∂w3, ∂L/∂w4, ∂L/∂b_h2
4. Mise a jour des poids avec lr = 0.1
5. Verifier en code : reimplementer le forward + backward et comparer les valeurs calculees a la main avec les valeurs du code

### Criteres de reussite

- [ ] Forward pass correct (6 valeurs intermediaires)
- [ ] Les 9 gradients sont calcules correctement avec la chain rule explicitee a chaque etape
- [ ] Les poids mis a jour sont corrects
- [ ] Le code de verification confirme les calculs manuels (erreur < 1e-4)
- [ ] Aucune etape sautee — chaque multiplication de la chain rule est visible

---

## Exercice 5 : Impact du learning rate — etude empirique

### Objectif

Observer experimentalement l'effet du learning rate et trouver le LR optimal pour un probleme donne.

### Consigne

En utilisant le reseau 2 couches du cours (2→4→1, sigmoid, MSE) sur le probleme XOR :

1. Entrainer le reseau avec 10 learning rates differents : [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
2. Pour chaque LR, entrainer pendant 5000 epochs et enregistrer la loss a chaque epoch
3. Produire un tableau recapitulatif : LR | Loss finale | Nombre d'epochs pour atteindre loss < 0.01 (ou "jamais") | Convergence (oui/non/diverge)
4. Tracer les courbes de loss (matplotlib ou ASCII)
5. Determiner le LR "optimal" — celui qui converge le plus vite tout en restant stable
6. **Bonus** : implementer un LR scheduler simple qui divise le LR par 2 tous les 1000 epochs. Comparer avec un LR fixe.

### Criteres de reussite

- [ ] Les 10 LR sont testes avec le meme seed (comparaison equitable)
- [ ] Le tableau recapitulatif est correct et lisible
- [ ] Le LR optimal identifie est justifie (pas juste "le plus rapide" mais aussi "stable")
- [ ] L'explication distingue 3 regimes : trop petit (lent), optimal (rapide+stable), trop grand (oscillations/divergence)
- [ ] Le code est propre et commente

---

## Exercice 6 : Mini-batch from scratch

### Objectif

Implementer les 3 variantes de gradient descent (batch, SGD, mini-batch) et comparer leur comportement.

### Consigne

1. Generer un dataset de classification binaire de 200 points (2 classes en spirale ou 2 gaussiennes) :

```python
# Suggestion : 2 gaussiennes
np.random.seed(42)
n = 100
X_class0 = np.random.randn(n, 2) * 0.5 + np.array([1, 1])
X_class1 = np.random.randn(n, 2) * 0.5 + np.array([-1, -1])
X = np.vstack([X_class0, X_class1])
y = np.array([0]*n + [1]*n).reshape(-1, 1)
```

2. Implementer une classe `NeuralNetwork` avec 3 methodes d'entrainement :
   - `train_batch(X, y, lr, epochs)` — gradient sur tout le dataset
   - `train_sgd(X, y, lr, epochs)` — gradient sur 1 echantillon aleatoire
   - `train_minibatch(X, y, lr, epochs, batch_size)` — gradient sur un mini-batch

3. Pour chaque variante, enregistrer :
   - La courbe de loss
   - Le temps d'execution (avec `time.time()`)
   - L'accuracy finale

4. Comparer dans un tableau et expliquer les differences observees

### Criteres de reussite

- [ ] Les 3 variantes sont implementees correctement (pas de copie de la solution du cours — reimplementer)
- [ ] Le shuffling des donnees est fait a chaque epoch pour mini-batch et SGD
- [ ] Les courbes de loss montrent le comportement attendu : batch (lisse), SGD (bruite), mini-batch (intermediaire)
- [ ] L'accuracy finale est similaire pour les 3 (elles convergent au meme endroit, juste differemment)
- [ ] L'analyse explique pourquoi mini-batch est le standard en pratique (compromis vitesse/stabilite + parallelisme GPU)

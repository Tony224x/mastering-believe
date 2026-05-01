# Exercices Medium — Jour 2 : Reseaux denses (MLP)

---

## Exercice 4 : MLP multi-classes from scratch — classification de chiffres

### Objectif

Implementer un MLP complet avec softmax + categorical cross-entropy pour un probleme a plus de 2 classes.

### Consigne

1. Generer un dataset synthetique de 3 classes (3 clusters gaussiens en 2D) :

```python
np.random.seed(42)
n = 100  # points per class
X0 = np.random.randn(n, 2) * 0.4 + np.array([0, 2])
X1 = np.random.randn(n, 2) * 0.4 + np.array([-1.5, -1])
X2 = np.random.randn(n, 2) * 0.4 + np.array([1.5, -1])
X = np.vstack([X0, X1, X2])
y = np.array([0]*n + [1]*n + [2]*n)
# Convertir en one-hot
y_onehot = np.zeros((3*n, 3))
y_onehot[np.arange(3*n), y] = 1
```

2. Implementer un MLP [2, 32, 16, 3] avec :
   - ReLU pour les couches cachees
   - Softmax pour la couche de sortie (implementer avec le log-sum-exp trick)
   - Categorical cross-entropy comme loss
   - Adam optimizer

3. Entrainer pendant 500 epochs, afficher loss et accuracy tous les 50 epochs

4. Afficher la matrice de confusion finale (3x3) :
```
             Predicted 0  Predicted 1  Predicted 2
Actual 0         98           2            0
Actual 1          1          96            3
Actual 2          0           1           99
```

5. **Verification numerique** : pour 5 poids aleatoires, comparer le gradient analytique (backprop) avec le gradient numerique (differences finies). L'erreur relative doit etre < 1e-5.

### Criteres de reussite

- [ ] Softmax implementee correctement avec le log-sum-exp trick
- [ ] CCE loss calculee correctement
- [ ] Backprop fonctionne pour la combinaison softmax+CCE (gradient = p - y_onehot)
- [ ] Accuracy finale > 95% sur l'ensemble d'entrainement
- [ ] Matrice de confusion correcte et lisible
- [ ] Gradient check passe pour les 5 poids testes (erreur < 1e-5)

---

## Exercice 5 : Comparaison systematique des 4 optimizers

### Objectif

Etudier empiriquement les forces et faiblesses de SGD, Momentum, RMSProp et Adam.

### Consigne

1. Generer le dataset "moons" (deux croissants entrelaces) :

```python
def make_moons(n=300, noise=0.15, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, np.pi, n)
    x1 = np.c_[np.cos(t), np.sin(t)]                    # upper moon
    x2 = np.c_[np.cos(t) + 0.5, -np.sin(t) + 0.5]      # lower moon (shifted)
    X = np.vstack([x1, x2]) + np.random.randn(2*n, 2) * noise
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    perm = np.random.permutation(2*n)
    return X[perm], y[perm]
```

2. Pour chaque optimizer (SGD lr=0.1, Momentum lr=0.05 β=0.9, RMSProp lr=0.005, Adam lr=0.005), entrainer un MLP [2, 32, 16, 1] pendant 1000 epochs avec le meme seed

3. Mesurer et comparer :
   - Courbe de loss (train)
   - Nombre d'epochs pour atteindre loss < 0.2
   - Accuracy finale sur un set de validation (30% des donnees)
   - Temps d'execution

4. Faire une deuxieme experience : ajouter du bruit aux donnees (noise=0.4) et re-comparer. Quel optimizer est le plus robuste au bruit ?

5. Produire un tableau recapitulatif et une analyse argumentee de quand utiliser chaque optimizer

### Criteres de reussite

- [ ] Les 4 optimizers sont implementes from scratch (pas de librairie)
- [ ] Meme seed et meme architecture pour une comparaison equitable
- [ ] Les courbes de loss montrent les differences attendues (SGD lent, Adam rapide)
- [ ] L'experience avec bruit montre la robustesse relative des optimizers
- [ ] L'analyse est argumentee avec les resultats empiriques, pas juste de la theorie
- [ ] Le tableau recapitulatif est complet (speed, accuracy, robustesse)

---

## Exercice 6 : Etude empirique de l'overfitting et des regularizations

### Objectif

Observer l'overfitting se produire, le diagnostiquer avec les courbes train/val, et le resoudre avec differentes techniques.

### Consigne

1. Generer un petit dataset de classification (100 points, 2 classes, 2D) et split 60/40 train/val

2. Creer un reseau deliberement trop gros : [2, 256, 128, 64, 1]

3. **Experiment A** — Baseline (pas de regularization) :
   - Entrainer 2000 epochs avec Adam
   - Tracer train loss ET val loss sur le meme graphe
   - Identifier visuellement le point d'overfitting

4. **Experiment B** — L2 regularization :
   - Tester λ = [0, 0.0001, 0.001, 0.01, 0.1]
   - Pour chaque λ, entrainer et noter la best val accuracy
   - Tracer un graphe λ vs best_val_accuracy pour trouver le λ optimal

5. **Experiment C** — Dropout :
   - Tester dropout rate = [0, 0.1, 0.3, 0.5, 0.7]
   - Meme procedure que pour L2
   - Trouver le taux optimal

6. **Experiment D** — Early stopping :
   - Implementer un early stopping avec patience = 50 :
     ```
     Si val_loss n'a pas ameliore depuis 50 epochs → arreter
     Restaurer les poids du meilleur epoch
     ```
   - Comparer avec l'entrainement complet

7. **Experiment E** — Combinaison optimale : L2 optimal + dropout optimal + early stopping

### Criteres de reussite

- [ ] L'overfitting est clairement demontre dans l'experience A (train/val gap croissant)
- [ ] Les courbes L2 et dropout montrent l'effet de chaque hyperparametre
- [ ] Early stopping est implemente correctement (sauvegarde des meilleurs poids)
- [ ] La combinaison E donne les meilleurs resultats de generalisation
- [ ] L'analyse explique POURQUOI chaque technique fonctionne (pas juste "ca marche mieux")
- [ ] Les resultats sont presentes dans des tableaux clairs

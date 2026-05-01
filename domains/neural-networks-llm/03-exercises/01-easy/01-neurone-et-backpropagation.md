# Exercices Faciles — Jour 1 : Le neurone & backpropagation

---

## Exercice 1 : Forward pass a la main

### Objectif

Savoir calculer la sortie d'un neurone unique pas a pas, sans code.

### Consigne

Un neurone a les parametres suivants :

- Inputs : x1 = 1.0, x2 = -0.5, x3 = 0.3
- Poids : w1 = 0.2, w2 = 0.8, w3 = -0.5
- Biais : b = 0.15
- Activation : sigmoid

**Calculer a la main :**

1. La somme ponderee z
2. L'activation a = sigmoid(z)
3. La loss MSE si la valeur cible est y = 0.0

Montrer chaque etape de calcul avec les valeurs intermediaires.

### Criteres de reussite

- [ ] z calcule correctement (verifier : z = 0.2*1.0 + 0.8*(-0.5) + (-0.5)*0.3 + 0.15)
- [ ] sigmoid appliquee correctement avec le bon z
- [ ] MSE calculee comme (a - y)^2
- [ ] Chaque etape intermediaire est explicite (pas de saut)

---

## Exercice 2 : Implementer les 3 fonctions d'activation

### Objectif

Coder sigmoid, tanh et ReLU from scratch, verifier leurs proprietes.

### Consigne

Ecrire un fichier Python qui :

1. Implemente `sigmoid(z)`, `tanh(z)`, `relu(z)` en NumPy (PAS avec des fonctions built-in de scipy ou autre)
2. Implemente leurs derivees respectives : `sigmoid_deriv(z)`, `tanh_deriv(z)`, `relu_deriv(z)`
3. Pour z = [-3, -1, 0, 1, 3], afficher un tableau avec les valeurs et derivees des 3 fonctions
4. Verifier numeriquement que la derivee est correcte en utilisant la methode des differences finies :

```python
# Verification numerique de la derivee
epsilon = 1e-7
numerical_deriv = (f(z + epsilon) - f(z - epsilon)) / (2 * epsilon)
# Doit etre tres proche de la derivee analytique
```

### Criteres de reussite

- [ ] Les 3 fonctions d'activation sont implementees correctement
- [ ] Les 3 derivees analytiques sont implementees correctement
- [ ] La verification numerique montre une erreur < 1e-5 pour chaque point
- [ ] Le tableau de sortie est clair et lisible

---

## Exercice 3 : Calcul de loss — MSE vs Cross-Entropy

### Objectif

Comprendre quand utiliser MSE vs Cross-Entropy et voir la difference sur des exemples concrets.

### Consigne

Ecrire un script Python qui :

1. Definit 5 paires (prediction, cible) :

   - (0.9, 1.0) — prediction correcte, confiance haute
   - (0.5, 1.0) — prediction incertaine
   - (0.1, 1.0) — prediction tres fausse
   - (0.9, 0.0) — prediction tres fausse (inversee)
   - (0.01, 0.0) — prediction correcte, confiance haute
2. Pour chaque paire, calcule :

   - MSE loss : (pred - target)^2
   - Binary Cross-Entropy loss : -[y*log(pred) + (1-y)*log(1-pred)]
3. Affiche un tableau comparatif
4. Calcule le gradient ∂L/∂pred pour MSE et BCE dans chaque cas
5. Reponds a la question : dans quel cas BCE penalise plus fortement qu'MSE ? Pourquoi est-ce desirable pour la classification ?

### Criteres de reussite

- [ ] Les deux losses sont calculees correctement pour les 5 cas
- [ ] Les gradients sont calcules correctement
- [ ] Le tableau est clair
- [ ] La reponse a la question montre la comprehension : BCE penalise fortement les predictions tres confiantes et fausses (pred=0.9, target=0.0 → loss tres haute), et le gradient ne sature pas grace a l'annulation avec sigmoid

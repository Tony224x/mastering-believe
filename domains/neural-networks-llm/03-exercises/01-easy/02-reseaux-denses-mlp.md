# Exercices Faciles — Jour 2 : Reseaux denses (MLP)

---

## Exercice 1 : Forward pass matriciel a la main

### Objectif

Savoir calculer un forward pass complet en notation matricielle, avec dimensions annotees a chaque etape.

### Consigne

Reseau MLP : [3, 2, 1] (3 inputs, 2 hidden neurons ReLU, 1 output sigmoid).

Parametres :
```
X = [[1.0, 0.5, -0.3]]        (1, 3) — un seul exemple

W1 = [[ 0.2,  0.5],           (3, 2)
      [-0.1,  0.3],
      [ 0.4, -0.2]]

b1 = [0.1, -0.1]              (1, 2)

W2 = [[ 0.6],                 (2, 1)
      [-0.4]]

b2 = [0.05]                   (1, 1)
```

**Calculer a la main :**
1. Z1 = X @ W1 + b1 — ecrire le produit matriciel element par element, annoter les dimensions
2. A1 = ReLU(Z1) — appliquer ReLU element par element
3. Z2 = A1 @ W2 + b2 — annoter les dimensions
4. A2 = sigmoid(Z2) — prediction finale
5. Loss BCE si y = 1.0

### Criteres de reussite

- [ ] Chaque produit matriciel est ecrit avec les dimensions explicites : (1,3) @ (3,2) = (1,2)
- [ ] ReLU correctement applique (les valeurs negatives deviennent 0)
- [ ] Sigmoid calculee correctement sur Z2
- [ ] BCE calculee comme -[y*log(p) + (1-y)*log(1-p)]
- [ ] Aucune etape sautee — toutes les valeurs intermediaires sont visibles

---

## Exercice 2 : Comparer les 3 loss functions

### Objectif

Comprendre le comportement de MSE, BCE et CCE sur des exemples concrets et savoir laquelle choisir.

### Consigne

1. Pour la classification binaire, calculer MSE et BCE pour ces 6 predictions (cible y=1) :
   ```
   p = [0.99, 0.9, 0.7, 0.5, 0.1, 0.01]
   ```

2. Pour chaque loss, calculer aussi le gradient ∂L/∂p

3. Tracer (ou calculer) la courbe loss vs prediction pour les deux losses. Observer :
   - Quand p → 0 (prediction completement fausse), BCE → +∞ mais MSE → 1 seulement
   - Quand p → 1 (prediction correcte), les deux → 0

4. Pour la classification multi-classes (K=3), calculer CCE pour :
   ```
   Classe vraie : 0 (one-hot : [1, 0, 0])
   Logits cas A : [3.0, 1.0, 0.5]   (confiant et juste)
   Logits cas B : [0.5, 3.0, 1.0]   (confiant et faux)
   Logits cas C : [1.0, 1.0, 1.0]   (incertain)
   ```
   Pour chaque cas : calculer softmax, puis CCE, puis le gradient ∂L/∂z

5. Repondre : pourquoi BCE est meilleure que MSE pour la classification ?

### Criteres de reussite

- [ ] Les 6 valeurs MSE et BCE sont correctes
- [ ] Les gradients sont corrects
- [ ] Les 3 cas multi-classes sont calcules correctement (softmax puis CCE)
- [ ] Le gradient CCE = softmax - onehot est verifie sur les 3 cas
- [ ] La reponse explique : BCE a un gradient proportionnel a l'erreur (pas de saturation), MSE + sigmoid a un gradient qui sature quand la prediction est tres fausse

---

## Exercice 3 : Implementer Adam pas a pas

### Objectif

Comprendre chaque composante d'Adam en l'implementant pour un seul parametre.

### Consigne

Simuler 10 steps d'Adam sur un SEUL poids w, avec des gradients fournis :

```python
w = 5.0                    # poids initial
gradients = [2.0, 1.8, 2.1, -0.5, -0.3, 1.5, 1.0, 0.8, 0.5, 0.2]
lr = 0.1
beta1, beta2 = 0.9, 0.999
eps = 1e-8
```

Pour chaque step t = 1, ..., 10 :
1. Calculer m_t = β1 * m_{t-1} + (1-β1) * g_t
2. Calculer v_t = β2 * v_{t-1} + (1-β2) * g_t^2
3. Calculer m_hat = m_t / (1 - β1^t) (bias correction)
4. Calculer v_hat = v_t / (1 - β2^t)
5. Calculer Δw = lr * m_hat / (√v_hat + ε)
6. Mettre a jour w = w - Δw

Afficher un tableau avec toutes les valeurs intermediaires a chaque step.

**Bonus** : comparer avec SGD simple (w -= lr * g) sur les memes gradients. Observer la difference de trajectoire.

### Criteres de reussite

- [ ] Les 10 steps sont calcules correctement avec toutes les valeurs intermediaires
- [ ] m (1er moment) montre l'effet de "direction" — lissage des gradients
- [ ] v (2eme moment) montre l'effet d'"echelle" — normalisation adaptative
- [ ] La bias correction est significative aux premieres steps (m_hat >> m)
- [ ] Le bonus montre que Adam converge plus regulierement que SGD brut

# Jour 2 — Reseaux denses (MLP) : Forward pass, Loss functions, Optimizers, Regularization

> **Temps estime** : 5h | **Prerequis** : Jour 1 (neurone unique, backprop, gradient descent basique)

---

## 1. Du neurone unique au reseau multi-couches

### Pourquoi empiler des couches ?

Un neurone unique = une frontiere de decision lineaire. Il peut separer des classes avec une droite (ou un hyperplan), point.

Deux classes en spirale ? Un XOR ? Un cercle a l'interieur d'un autre ? Impossible avec une droite.

En empilant des couches, chaque couche **transforme l'espace des donnees** :
- La 1ere couche cachee cree des features non-lineaires a partir des inputs
- La 2eme couche cachee combine ces features en patterns plus abstraits
- La couche de sortie prend la decision finale dans cet espace transforme

```
Espace original (non separable)     →     Espace transforme (separable)

  o x o x                                    o o o o
  x o x o          hidden layers              ------
  o x o x         ===========>               x x x x
  x o x o                                    x x x x
```

### Universal Approximation Theorem — l'intuition

**Enonce simplifie** : un reseau avec UNE SEULE couche cachee et suffisamment de neurones peut approximer n'importe quelle fonction continue sur un domaine borne, avec une precision arbitraire.

**Intuition** : chaque neurone ReLU decoupe l'espace avec un "coude". Avec assez de coudes, on peut approximer n'importe quelle courbe — comme une approximation par morceaux lineaires.

```
Fonction cible :       Approximation MLP (6 neurones ReLU) :

    /\                     /\
   /  \                   / \
  /    \                 /   \
_/      \_              /     \_
            \_               \__
```

**Attention** : le theoreme dit que ca EXISTE, pas que le gradient descent va le TROUVER. En pratique, on utilise plusieurs couches (deep) plutot qu'une seule couche tres large, parce que :
1. Les couches profondes apprennent des representations hierarchiques (features simples → complexes)
2. Beaucoup moins de parametres necessaires pour la meme expressivite
3. Meilleure generalisation empirique

---

## 2. Architecture MLP — notation matricielle

### Structure

```
Input Layer       Hidden Layer 1      Hidden Layer 2      Output Layer
(n_0 neurones)    (n_1 neurones)      (n_2 neurones)      (n_3 neurones)

  x_1 ─────┐
            ├──→  h1_1 ─────┐
  x_2 ─────┤     h1_2       ├──→  h2_1 ─────┐
            ├──→  h1_3       │     h2_2       ├──→  y_1
  x_3 ─────┤     h1_4 ─────┘     h2_3 ─────┘     y_2
            ├──→
  x_4 ─────┘
```

### Notation matricielle — dimensions

Pour un reseau avec L couches, les parametres de la couche l sont :

```
W_l : matrice de poids   — dimensions (n_{l-1}, n_l)
b_l : vecteur de biais   — dimensions (1, n_l)
```

**Exemple concret** : reseau [4, 8, 6, 2]
- Couche 1 : W1 de taille (4, 8), b1 de taille (1, 8) → 4*8 + 8 = 40 parametres
- Couche 2 : W2 de taille (8, 6), b2 de taille (1, 6) → 8*6 + 6 = 54 parametres
- Couche 3 : W3 de taille (6, 2), b3 de taille (1, 2) → 6*2 + 2 = 14 parametres
- **Total** : 108 parametres

### Pourquoi ces dimensions ?

```
X      @    W1     +    b1     =    Z1
(m, 4) @ (4, 8)   + (1, 8)    = (m, 8)
  ↑         ↑          ↑           ↑
batch    n_in→n_out  broadcast   batch de
size                  sur m       pre-activations
```

La dimension "interieure" du produit matriciel (4) doit correspondre : `n_colonnes(X) = n_lignes(W1) = n_in`. Le resultat a `n_colonnes(W1) = n_out` colonnes.

---

## 3. Forward pass en notation matricielle

### Calcul complet avec dimensions annotees

Pour un batch de m exemples sur un reseau [4, 8, 6, 2] :

```
Couche 1 (input → hidden 1) :
  Z1 = X @ W1 + b1           (m,4) @ (4,8) + (1,8) = (m,8)
  A1 = relu(Z1)              (m,8) — element-wise

Couche 2 (hidden 1 → hidden 2) :
  Z2 = A1 @ W2 + b2          (m,8) @ (8,6) + (1,6) = (m,6)
  A2 = relu(Z2)              (m,6)

Couche 3 (hidden 2 → output) :
  Z3 = A2 @ W3 + b3          (m,6) @ (6,2) + (1,2) = (m,2)
  A3 = softmax(Z3)           (m,2) — probabilites de sortie
```

### Exemple numerique complet

Reseau [2, 3, 1], batch de 2 exemples :

```
X = [[0.5, 0.8],     (2, 2) — 2 exemples, 2 features
     [0.1, 0.9]]

W1 = [[ 0.4, -0.2,  0.3],   (2, 3)
      [-0.1,  0.5,  0.2]]

b1 = [0.1, -0.1, 0.0]       (1, 3)

W2 = [[ 0.6],                (3, 1)
      [-0.3],
      [ 0.4]]

b2 = [0.1]                   (1, 1)
```

**Etape 1 — Couche cachee :**
```
Z1 = X @ W1 + b1

X @ W1 = [[0.5*0.4 + 0.8*(-0.1),  0.5*(-0.2) + 0.8*0.5,  0.5*0.3 + 0.8*0.2],
           [0.1*0.4 + 0.9*(-0.1),  0.1*(-0.2) + 0.9*0.5,  0.1*0.3 + 0.9*0.2]]

       = [[0.20 - 0.08,  -0.10 + 0.40,  0.15 + 0.16],
          [0.04 - 0.09,  -0.02 + 0.45,  0.03 + 0.18]]

       = [[0.12,  0.30,  0.31],
          [-0.05, 0.43,  0.21]]

Z1 = [[0.12 + 0.1,  0.30 - 0.1,  0.31 + 0.0],
      [-0.05 + 0.1, 0.43 - 0.1,  0.21 + 0.0]]

   = [[0.22,  0.20,  0.31],
      [0.05,  0.33,  0.21]]

A1 = relu(Z1) = [[0.22, 0.20, 0.31],    (tout > 0, rien ne change)
                  [0.05, 0.33, 0.21]]
```

**Etape 2 — Couche de sortie :**
```
Z2 = A1 @ W2 + b2

A1 @ W2 = [[0.22*0.6 + 0.20*(-0.3) + 0.31*0.4],
            [0.05*0.6 + 0.33*(-0.3) + 0.21*0.4]]

        = [[0.132 - 0.060 + 0.124],
           [0.030 - 0.099 + 0.084]]

        = [[0.196],
           [0.015]]

Z2 = [[0.196 + 0.1],
      [0.015 + 0.1]]

   = [[0.296],
      [0.115]]

A2 = sigmoid(Z2) = [[sigmoid(0.296)],   = [[0.5735],
                     [sigmoid(0.115)]]      [0.5287]]
```

Le batch entier est traite en UNE operation matricielle par couche. C'est pour ca que les GPUs sont si efficaces : chaque couche = un gros produit matriciel.

---

## 4. Loss functions en profondeur

### 4.1 MSE — Mean Squared Error

```
L = (1/m) * Σ (y_pred - y_true)^2
```

**Gradient** :
```
∂L/∂y_pred = (2/m) * (y_pred - y_true)
```

**Quand l'utiliser** : regression (predire une valeur continue — prix, temperature, age).

**Exemple** :
```
Predictions :  [2.5, 3.8, 1.2]
Cibles :       [3.0, 4.0, 1.0]
Erreurs :      [-0.5, -0.2, 0.2]
Carres :       [0.25, 0.04, 0.04]
MSE :          (0.25 + 0.04 + 0.04) / 3 = 0.11
```

**Probleme pour la classification** : combiner MSE avec sigmoid cause un gradient qui sature. Quand sigmoid(z) → 0 ou 1, la derivee de sigmoid → 0, ce qui ralentit l'apprentissage exactement quand la prediction est tres fausse.

### 4.2 Binary Cross-Entropy (BCE) — derivation intuitive

**D'ou vient la formule ?** Du maximum likelihood.

On modelise le probleme comme un tirage de Bernoulli. Pour une prediction p et un label y :

```
P(y=1 | x) = p        — le modele predit la probabilite d'etre dans la classe 1
P(y=0 | x) = 1 - p    — la probabilite complementaire

On peut ecrire les deux cas en une formule :
P(y | x) = p^y * (1-p)^(1-y)
```

La **vraisemblance** (likelihood) pour m exemples :
```
L = Π p_i^{y_i} * (1-p_i)^{1-y_i}
```

Le **log-vraisemblance** (plus stable numeriquement, transforme le produit en somme) :
```
log L = Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

On veut **maximiser** la vraisemblance = **minimiser** la negative log-vraisemblance :
```
BCE = -(1/m) * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

**Gradient** :
```
∂BCE/∂p = -(y/p) + (1-y)/(1-p)
```

**Et combine avec sigmoid (z → p = sigmoid(z)), ca simplifie magiquement** :
```
∂BCE/∂z = p - y = sigmoid(z) - y
```

Le gradient est proportionnel a l'erreur. Pas de saturation.

**Exemple** :
```
Prediction : p = 0.2, Cible : y = 1
BCE = -[1*log(0.2) + 0*log(0.8)]
    = -log(0.2)
    = -(-1.6094)
    = 1.6094        ← penalite forte (confiant et faux)

Prediction : p = 0.9, Cible : y = 1
BCE = -log(0.9)
    = 0.1054        ← penalite faible (confiant et juste)
```

### 4.3 Categorical Cross-Entropy + Softmax

Pour les problemes multi-classes (K classes), on utilise softmax + categorical cross-entropy.

**Softmax** — transforme un vecteur de scores en probabilites :

```
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}
```

**Exemple** avec 3 classes :
```
z = [2.0, 1.0, 0.5]

e^z = [e^2.0, e^1.0, e^0.5] = [7.389, 2.718, 1.649]
somme = 7.389 + 2.718 + 1.649 = 11.756

softmax(z) = [7.389/11.756, 2.718/11.756, 1.649/11.756]
           = [0.6285, 0.2312, 0.1403]

Somme = 1.0 ✓   Toutes positives ✓
```

**Categorical Cross-Entropy** :
```
L = -Σ_k y_k * log(p_k)
```

Ou y est un vecteur one-hot (ex: classe 0 → [1, 0, 0]).

```
y = [1, 0, 0],  p = [0.6285, 0.2312, 0.1403]

L = -[1*log(0.6285) + 0*log(0.2312) + 0*log(0.1403)]
  = -log(0.6285)
  = 0.4643
```

Seule la probabilite de la vraie classe compte. Si le modele donne 0.99 a la bonne classe → loss ≈ 0.01. S'il donne 0.01 → loss ≈ 4.6. Penalite exponentielle.

### Le probleme de stabilite numerique — log-sum-exp trick

**Le probleme** : si z_i est grand (ex: 1000), e^{1000} = overflow (inf). Si z_i est tres negatif, e^{-1000} = underflow (0).

**La solution** : soustraire le max avant de calculer l'exponentielle.

```
softmax(z_i) = e^{z_i} / Σ_j e^{z_j}

Soit M = max(z). On soustrait M de tous les z :

softmax(z_i) = e^{z_i - M} / Σ_j e^{z_j - M}
```

Mathematiquement equivalent (les e^M s'annulent numerateur/denominateur), mais numeriquement stable car le plus grand exposant est maintenant 0.

```
z = [1000, 999, 998]

Sans le trick :
  e^1000 = inf    ← OVERFLOW

Avec le trick (M = 1000) :
  z - M = [0, -1, -2]
  e^0 = 1.000,  e^{-1} = 0.368,  e^{-2} = 0.135
  somme = 1.503
  softmax = [0.665, 0.245, 0.090]   ← stable ✓
```

**Le gradient combine softmax + categorical cross-entropy** simplifie aussi :
```
∂L/∂z_i = p_i - y_i    (meme forme elegante que sigmoid + BCE !)
```

### Recapitulatif des losses

| Probleme | Loss | Activation sortie | Gradient ∂L/∂z |
|---|---|---|---|
| Regression | MSE | Lineaire (identite) | (2/m)(y_pred - y_true) |
| Classification binaire | BCE | Sigmoid | p - y |
| Classification multi-classes | Categorical CE | Softmax | p - y (vecteur) |

---

## 5. Optimizers — l'evolution du gradient descent

### 5.1 SGD — le basique

```
w = w - lr * ∂L/∂w
```

C'est le gradient descent classique. Simple, mais avec des problemes :

**Probleme 1 — Oscillations** : si la surface de loss est une "vallee allongee" (un gradient fort dans une direction, faible dans l'autre), SGD oscille perpendiculairement a la direction optimale.

```
Surface de loss (vue de dessus — contours elliptiques) :

   _______________
  /               \
 /  ╱╲╱╲╱╲╱╲╱╲    \     ← SGD oscille dans la direction "raide"
/   →→→→→→→→→→→    /     ← mais avance lentement dans la bonne direction
 \               /
  \_____________/
```

**Probleme 2 — Saddle points** : en haute dimension, il y a beaucoup plus de saddle points (gradient = 0 mais pas un minimum) que de vrais minima. SGD reste bloque.

**Probleme 3 — Learning rate unique** : tous les parametres utilisent le meme LR, mais certains features sont rares (gradients rares mais informatifs) et d'autres sont frequents.

### 5.2 Momentum — la balle qui roule

**Analogie** : imaginez une balle qui roule sur la surface de loss. Elle accumule de la vitesse dans les directions coherentes et amortit les oscillations.

**Formule** :
```
v_t = β * v_{t-1} + ∂L/∂w          (accumuler la vitesse)
w = w - lr * v_t                      (mettre a jour avec la vitesse)
```

- **β** (momentum coefficient) : typiquement 0.9
- **v** (velocity) : la "memoire" des gradients passes

**Pourquoi ca aide** :
- Dans les directions coherentes : les gradients s'additionnent → acceleration
- Dans les directions oscillantes : les gradients s'annulent → amortissement

**Exemple numerique** :
```
β = 0.9, lr = 0.01

Epoch 1 : gradient = 2.0
  v_1 = 0.9 * 0 + 2.0 = 2.0
  w -= 0.01 * 2.0 = -0.02

Epoch 2 : gradient = 1.8
  v_2 = 0.9 * 2.0 + 1.8 = 3.6     ← accelere !
  w -= 0.01 * 3.6 = -0.036

Epoch 3 : gradient = -0.5 (direction change)
  v_3 = 0.9 * 3.6 + (-0.5) = 2.74  ← freine mais continue
  w -= 0.01 * 2.74 = -0.0274
```

### 5.3 RMSProp — adaptive learning rate par parametre

**Idee** : les parametres avec des gradients frequemment grands doivent avoir un LR plus petit, et inversement.

**Formule** :
```
s_t = β * s_{t-1} + (1 - β) * (∂L/∂w)^2    (moyenne mobile du carre des gradients)
w = w - lr * ∂L/∂w / (√s_t + ε)              (normaliser par la "taille typique" du gradient)
```

- **s** : estimation de la variance du gradient (moyenne mobile exponentielle)
- **ε** : petit nombre pour eviter la division par zero (typiquement 1e-8)
- **β** : typiquement 0.999

**Pourquoi ca aide** :
- Parametres avec gradients grands → s grand → division par un grand nombre → petit pas
- Parametres avec gradients petits → s petit → division par un petit nombre → grand pas
- Chaque parametre a effectivement son propre learning rate adaptatif

### 5.4 Adam — le best-of

Adam combine les idees de Momentum (1er moment — moyenne des gradients) et RMSProp (2eme moment — variance des gradients).

**Formule** :
```
m_t = β1 * m_{t-1} + (1 - β1) * g_t          (1er moment : moyenne des gradients)
v_t = β2 * v_{t-1} + (1 - β2) * g_t^2        (2eme moment : variance des gradients)

# Bias correction (crucial au debut quand m et v sont proches de 0)
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)

w = w - lr * m_hat / (√v_hat + ε)
```

Hyperparametres par defaut (quasi universels) :
- **β1** = 0.9
- **β2** = 0.999
- **ε** = 1e-8
- **lr** = 0.001

**Pourquoi la bias correction ?**

Au debut de l'entrainement, m et v sont initialises a 0. Comme on fait une moyenne mobile exponentielle, les premieres estimations sont biaisees vers 0.

```
Sans correction (t=1, β1=0.9) :
  m_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1   ← 10x trop petit !

Avec correction :
  m_hat = 0.1 * g_1 / (1 - 0.9^1) = 0.1 * g_1 / 0.1 = g_1   ← correct !
```

La correction disparait apres quelques dizaines de steps (β1^t → 0).

### Tableau comparatif

| Optimizer | Vitesse | Hyperparametres a tuner | Meilleur pour | Faiblesses |
|---|---|---|---|---|
| SGD | Lent | lr (critique) | Convexe, grande generalisation | Oscillations, saddle points |
| SGD+Momentum | Bon | lr + β | La plupart des problemes classiques | Peut depasser le minimum |
| RMSProp | Bon | lr + β | RNN, non-stationnaire | Pas de momentum |
| **Adam** | **Rapide** | **lr (le reste = defauts)** | **Defaut pour tout** | Peut mal generaliser vs SGD tune |

> **Regle pratique** : commencer par Adam (lr=0.001). Si le modele overfitte ou generalise mal, essayer SGD+Momentum avec LR schedule. Pour les LLMs et les tres gros modeles, AdamW (Adam avec weight decay decouple) est le standard.

---

## 6. Regularization — combattre l'overfitting

### 6.1 Le probleme : overfitting

L'overfitting, c'est quand le modele **memorise** les donnees d'entrainement au lieu d'apprendre les patterns generaux. Il performe bien sur le train set mais mal sur de nouvelles donnees.

**Diagnostic** : la courbe classique en U.

```
   Loss
    │
    │ ╲           ╱         ← validation loss (remonte = overfitting)
    │  ╲         ╱
    │   ╲       ╱
    │    ╲     ╱
    │     ╲___╱              ← sweet spot (early stopping)
    │
    │  ╲
    │   ╲
    │    ╲
    │     ╲__________        ← training loss (descend toujours)
    │
    └────────────────── Epochs
```

**Quand ca arrive** :
- Trop de parametres par rapport au nombre d'exemples
- Entrainement trop long
- Modele trop expressif pour un probleme simple

### 6.2 L1 vs L2 Regularization

L'idee : ajouter une penalite sur les poids a la loss, pour empecher les poids de devenir trop grands.

**L2 Regularization (Ridge / Weight Decay)**
```
L_total = L_original + λ * Σ w_i^2
```

- Le gradient additionnel : `∂/∂w = 2λ * w` → pousse les poids vers 0 proportionnellement a leur taille
- Les gros poids sont penalises beaucoup plus (quadratique)
- Resultat : poids **petits mais non-nuls** — le reseau utilise toutes ses connexions a faible intensite

**L1 Regularization (Lasso)**
```
L_total = L_original + λ * Σ |w_i|
```

- Le gradient additionnel : `∂/∂w = λ * sign(w)` → pousse vers 0 avec une force constante
- Meme penalite quel que soit la taille du poids
- Resultat : poids **exactement zero** pour les connexions inutiles → **sparsity**

**Comparaison visuelle de l'effet sur un poids** :

```
L2 (push proportionnel)              L1 (push constant)

  Force                               Force
   │                                   │
   │         ╱                         │    ___________
   │       ╱                           │   |
   │     ╱                             │   |
   │   ╱                               │   |
   │ ╱                                 │   |
───┼──────── w                     ────┼───┼────────── w
   │╲                                  │   |
   │  ╲                                │   |
   │    ╲                              │   |___________
   │                                   │
```

**Exemple numerique** (λ = 0.01) :
```
Poids initial : w = 5.0

L2 : penalite = 0.01 * 5.0^2 = 0.25  | gradient = 2 * 0.01 * 5.0 = 0.10
L1 : penalite = 0.01 * |5.0| = 0.05  | gradient = 0.01 * sign(5.0) = 0.01

Poids initial : w = 0.01

L2 : penalite = 0.01 * 0.01^2 = 0.000001  | gradient = 2 * 0.01 * 0.01 = 0.0002
L1 : penalite = 0.01 * |0.01| = 0.0001    | gradient = 0.01 * sign(0.01) = 0.01
                                                        ← meme force ! pousse a zero
```

> **Regle** : L2 (weight decay) est le standard. L1 si on veut de la sparsity (selection de features). En pratique, pour les reseaux de neurones, on utilise presque toujours L2.

### 6.3 Dropout — ensemble de sous-reseaux

**Principe** : a chaque batch pendant l'entrainement, chaque neurone a une probabilite p d'etre **desactive** (mis a zero).

```
Sans dropout :                     Avec dropout (p=0.5) :

  x1 ──→ h1 ──→ h3 ──→ y          x1 ──→ h1 ──→ [0] ──→ y
         ╲╱     ╲╱                         ╲╱
         ╱╲     ╱╲                         ╱╲
  x2 ──→ h2 ──→ h4 ──→ y          x2 ──→ [0] ──→ h4 ──→ y

  Tous les neurones actifs          h2 et h3 desactives aleatoirement
```

**Intuition** : c'est comme entrainer un **ensemble** de sous-reseaux differents, puis moyenner leurs predictions. Chaque sous-reseau doit apprendre a fonctionner sans les autres neurones → pas de co-dependance → meilleure generalisation.

**Implementation** :
```
Pendant le training :
  mask = random(shape) > p    (masque binaire aleatoire)
  A = A * mask                (desactiver les neurones)
  A = A / (1 - p)             (inverted dropout: scale up pour compenser)

Pendant l'inference :
  Rien a faire ! (grace a l'inverted dropout, les magnitudes sont deja correctes)
```

**Pourquoi diviser par (1-p) ?** Si on desactive 50% des neurones, la somme des activations est 2x plus petite. En divisant par 0.5, on restaure la magnitude attendue. Ainsi, le meme modele marche pour l'inference sans modification.

**Taux typiques** :
- Couches cachees : p = 0.5 (50% de dropout) — valeur standard de Hinton
- Couches proches de l'input : p = 0.2 (moins de dropout — les features brutes sont precieuses)
- Couche de sortie : jamais de dropout

### 6.4 Autres techniques (preview)

**Early stopping** : arreter l'entrainement quand la validation loss remonte (patience = nombre d'epochs a attendre avant de decider). Techniquement, c'est une forme de regularization car ca limite la capacite effective du modele.

**Batch Normalization** : normaliser les activations de chaque couche a moyenne 0 et variance 1, puis re-scaler avec des parametres apprenables. Stabilise l'entrainement, permet des LR plus grands, et a un leger effet regularisant (le bruit du mini-batch). On le verra en detail au Jour 3.

---

## 7. Flash Cards — Active Recall

### Q1 : Quelle est la dimension de la matrice de poids W pour une couche avec 64 entrees et 32 sorties ?

<details>
<summary>Reponse</summary>

```
W : (64, 32)
b : (1, 32)
```

Convention : W a `n_in` lignes et `n_out` colonnes. Le produit `X @ W` avec X de shape `(batch, 64)` donne `(batch, 32)`.

Nombre de parametres : 64 * 32 + 32 = 2080.

</details>

### Q2 : Pourquoi softmax et categorical cross-entropy vont ensemble ? Que donne leur gradient combine ?

<details>
<summary>Reponse</summary>

Softmax transforme les scores bruts (logits) en probabilites (somme = 1, toutes positives). Categorical cross-entropy penalise le modele en fonction de la probabilite assignee a la bonne classe.

Combines, leur gradient simplifie a :
```
∂L/∂z_i = p_i - y_i
```

Meme forme elegante que sigmoid + BCE. Le gradient est directement proportionnel a l'erreur, sans saturation.

</details>

### Q3 : Expliquer Adam en une phrase : quel est le role de m (1er moment) et v (2eme moment) ?

<details>
<summary>Reponse</summary>

**m** (1er moment) = moyenne mobile des gradients = **direction** du mouvement (comme Momentum — accelere dans les directions coherentes).

**v** (2eme moment) = moyenne mobile des gradients au carre = **taille typique** du gradient pour ce parametre (comme RMSProp — normalise pour donner un LR adaptatif par parametre).

Adam = Momentum (m) + RMSProp (v) + bias correction.

</details>

### Q4 : Quelle est la difference entre L1 et L2 regularization ? Quand utiliser chacune ?

<details>
<summary>Reponse</summary>

- **L2** (weight decay) : penalite `λ * Σ w^2`. Pousse les poids vers des petites valeurs proportionnellement a leur taille. Resultat : poids petits mais non-nuls. **Standard pour les reseaux de neurones.**

- **L1** (lasso) : penalite `λ * Σ |w|`. Pousse les poids vers zero avec une force constante. Resultat : beaucoup de poids exactement a zero (**sparsity**). Utile pour la **selection de features**.

En pratique pour le deep learning : L2 (souvent sous forme de weight decay dans AdamW).

</details>

### Q5 : Comment fonctionne dropout ? Pourquoi divise-t-on par (1-p) pendant l'entrainement ?

<details>
<summary>Reponse</summary>

Pendant l'entrainement, chaque neurone est desactive avec probabilite p (mis a zero aleatoirement). Cela force le reseau a ne pas dependre d'un seul neurone et ameliore la generalisation.

On divise par (1-p) = **inverted dropout** : si on desactive 50% des neurones, la somme des activations est 2x plus petite. En divisant par 0.5, on restaure la magnitude attendue. Ainsi, pendant l'inference, on n'a rien a modifier — le modele fonctionne tel quel.

</details>

---

## Pour aller plus loin

Lectures couvrant MLP, regularisation et optimisation (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS230 — Lec. 3 (Full Cycle of a DL project)** — Adam, batch norm, dropout en contexte projet.
- **Stanford CS231N — Lec. 3 (Regularization & Optimization)** — couverture exhaustive optims modernes.
- **MIT 6.S191 — Lecture "Introduction to Deep Learning"** (édition récente) — vue compacte regularisation.


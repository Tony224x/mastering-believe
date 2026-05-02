# J2 — Transformations 3D : SE(3), rotations, twists

> Manipuler des poses rigides en 3D (rotation + translation), composer des transformations le long d'une chaîne cinematique, comprendre les twists et pourquoi la formulation Product of Exponentials (PoE) supplante Denavit-Hartenberg.
> Sources : `[Lynch & Park, 2017, ch. 3]`, `[Khatib CS223A, L2-3]`.

---

## 1. Exemple concret : composer 2 transformations sur un bras 2-DOF planaire

Imagine un bras planaire avec deux liens de longueur `L1 = 1` et `L2 = 1`, ancré à l'origine du repère monde `{w}`. Le premier joint tourne d'un angle `q1`, le second de `q2`.

- Repère `{1}` = base du lien 1 (fixe sur `{w}` au joint 1)
- Repère `{2}` = base du lien 2 (fixé au bout du lien 1)
- Repère `{e}` = end-effector (fixé au bout du lien 2)

La pose du end-effector dans le monde est la composition :

```
T_we = T_w1 · T_12 · T_2e
```

Chaque `T_ij` est une **matrice homogène 4x4** qui empile une rotation et une translation. Pour `q1 = q2 = 0` et avec les axes locaux usuels, l'end-effector est à `(L1+L2, 0, 0) = (2, 0, 0)` dans le monde — c'est exactement ce que la composition `T_w1 · T_12 · T_2e` redonne. C'est la mécanique de base de la cinematique directe (J3) : on multiplie les poses le long de la chaîne.

> **Take-away** : composer des transformations rigides = multiplier des matrices 4x4. SE(3) est le groupe (au sens algébrique) qui formalise ces matrices et leur composition.

---

## 2. Rotations : SO(3)

Une rotation 3D est une matrice `R ∈ R^{3x3}` qui satisfait :

- `R^T R = I` (orthogonalité)
- `det(R) = +1` (préserve l'orientation, exclut les réflexions)

L'ensemble de ces matrices forme le groupe spécial orthogonal `SO(3)`.

### 2.1 Trois représentations canoniques

| Représentation | Paramètres | Avantages | Inconvénients |
|---|---|---|---|
| **Matrice** `R` | 9 (avec 6 contraintes) | composition = produit, action sur vecteur = produit | redondant, peut dériver hors SO(3) |
| **Axe-angle** `(ω̂, θ)` | 3 (`ω̂ θ` ∈ R^3, norme = θ) | minimal, intuitif, sans singularité | composition non-triviale |
| **Quaternion** `q = (w, x, y, z)` | 4 (norme 1) | composition O(1), pas de gimbal lock, slerp propre | ambiguïté `q ≡ -q` |

> Les angles d'Euler (roll/pitch/yaw) ne figurent pas dans ce tableau parce qu'ils souffrent du **gimbal lock** (perte d'un degré de liberté à certaines configurations). On ne les utilise que pour l'affichage humain, jamais pour les calculs internes `[Lynch & Park, 2017, §3.2.3]`.

### 2.2 Formule de Rodrigues : axe-angle → matrice

Pour un axe unitaire `ω̂ ∈ R^3` et un angle `θ`, la matrice de rotation est :

```
R = I + sin(θ) · [ω̂] + (1 - cos(θ)) · [ω̂]^2
```

où `[ω̂]` est la matrice antisymetrique 3x3 (skew-symmetric matrix) :

```
        [  0   -ω_z   ω_y ]
[ω̂] =  [ ω_z    0   -ω_x ]
        [-ω_y   ω_x    0  ]
```

`[ω̂]` encode le produit vectoriel : `[ω̂] v = ω̂ × v`. Cette matrice est l'**algèbre de Lie** `so(3)` ; l'exponentielle matricielle envoie `so(3)` sur `SO(3)`. Rodrigues est exactement la forme close de `exp([ω̂] θ)` `[Lynch & Park, 2017, §3.2.3]`.

### 2.3 Quaternions : composition rapide et stable

Un quaternion unitaire `q = (w, x, y, z)` avec `w^2 + x^2 + y^2 + z^2 = 1` représente la rotation d'angle `θ` autour de `ω̂` :

```
q = (cos(θ/2),  ω̂_x sin(θ/2),  ω̂_y sin(θ/2),  ω̂_z sin(θ/2))
```

La composition de deux rotations s'écrit `q_total = q_1 ⊗ q_2` (produit de Hamilton), 16 multiplications/8 additions — moins coûteux qu'un produit matriciel 27/18 et numériquement plus stable. C'est pour ça que les contrôleurs et les estimateurs d'état (EKF, UKF) parlent quaternions en interne.

> **Mnémo** : R coûte 9 floats à stocker, quaternion 4. Pour des milliers de transformations (un trajectoire IK, un nuage de points), la différence se voit.

---

## 3. SE(3) : rotations + translations

Une **pose rigide** est un couple `(R, p)` avec `R ∈ SO(3)` et `p ∈ R^3`. L'ensemble des poses forme le groupe `SE(3)` (Special Euclidean group).

### 3.1 Matrice homogène 4x4

On encode la pose dans une matrice augmentée :

```
       [ R   p ]
T  =   [ 0   1 ]    ∈ R^{4x4}
```

L'intérêt : la composition de deux poses `(R_1, p_1)` puis `(R_2, p_2)` se fait en multipliant simplement `T_1 · T_2`. Sans cette astuce, il faudrait gérer rotation + translation séparément à chaque étape.

Pour transformer un point `x ∈ R^3` du repère `{B}` vers `{A}`, on l'augmente en `[x; 1]` puis on multiplie :

```
[x_A]     [R_AB  p_AB ] [x_B]
[ 1 ]  =  [ 0     1   ] [ 1 ]
```

Concrètement : `x_A = R_AB · x_B + p_AB`. La forme homogène cache cette mécanique pour permettre la composition directe.

### 3.2 Inverse d'une transformation

L'inverse de `T = (R, p)` n'est PAS `(R^{-1}, -p)`. C'est :

```
            [ R^T   -R^T p ]
T^{-1}  =   [  0       1   ]
```

Démonstration courte : si `x_A = R x_B + p`, alors `x_B = R^T (x_A - p) = R^T x_A - R^T p`. La rotation s'inverse par transposition (puisque `R^T R = I`), mais la translation doit être réexprimee dans le nouveau repère, d'où le `-R^T p`. **Erreur classique** : oublier ce détail produit des bugs subtils dans les IK numériques.

### 3.3 Composition le long d'une chaîne

Pour une chaîne cinematique avec n joints, la pose du end-effector dans le repère monde est :

```
T_we(q) = T_01(q_1) · T_12(q_2) · ... · T_{n-1,n}(q_n)
```

C'est exactement la cinematique directe `[Lynch & Park, 2017, §4.1]`. La beauté de la formulation matrice homogène : c'est un simple produit matriciel, et numpy le fait nativement avec `@`.

---

## 4. Twists et screws : la representation duale

### 4.1 Idée

Une transformation rigide peut être vue comme une rotation autour d'un **axe screw** (axe dans l'espace, pas forcément passant par l'origine) + une translation le long de cet axe. Cette représentation s'appelle un **twist** ou **screw motion** (théorème de Chasles).

### 4.2 Espace tangent à SE(3) : se(3)

Tout comme `so(3)` est l'espace tangent à `SO(3)` (vecteurs angulaires `ω`), `se(3)` est l'espace tangent à `SE(3)`. Un élément de `se(3)`, appelé **twist spatial**, est un vecteur 6D :

```
V = (ω, v) ∈ R^6
```

où `ω ∈ R^3` est la vitesse angulaire et `v ∈ R^3` la vitesse linéaire (les deux exprimées dans le repère spatial). Sa forme matricielle 4x4 est :

```
         [ [ω]   v ]
[V]  =   [  0    0 ]
```

L'exponentielle matricielle envoie `se(3)` sur `SE(3)` :

```
T(t) = exp([V] t)
```

C'est l'équivalent direct de Rodrigues, mais en dimension 6. La formule fermée pour `exp([V])` est connue (Lynch & Park, eq. 3.88) — c'est ce qui rend la PoE praticable.

### 4.3 Screw axis = axe + pitch

Un twist normalisé `S = V / ‖ω‖` (si `ω ≠ 0`) s'interprète comme un **screw axis** : un axe spatial `ω̂` autour duquel on tourne, plus un déplacement linéaire le long de `ω̂` proportionnel à l'angle (le ratio = **pitch h**). Si `ω = 0`, c'est une translation pure.

> **Take-away** : un twist code à la fois rotation et translation dans un seul vecteur 6D, et son exponentielle est une transformation SE(3). C'est l'objet central de la cinematique moderne.

---

## 5. Pourquoi PoE > DH

### 5.1 Denavit-Hartenberg : le standard historique

DH paramétrise chaque joint par 4 nombres (`a`, `α`, `d`, `θ`) suivant des conventions strictes sur le placement des repères. Inventé en 1955, c'est ce qu'on apprend dans la plupart des manuels `[Lynch & Park, 2017, §A.2]`.

### 5.2 Product of Exponentials (PoE) : la formulation moderne

PoE écrit la cinematique directe d'un manipulateur n-DOF comme :

```
T(q) = exp([S_1] q_1) · exp([S_2] q_2) · ... · exp([S_n] q_n) · M
```

où :
- `S_i ∈ R^6` est le screw axis du joint i (exprimé dans le repère espace **fixe**, ou dans le repère du end-effector pour la formulation body)
- `M ∈ SE(3)` est la pose du end-effector quand tous les `q_i = 0` (configuration "home")

### 5.3 Avantages PoE

1. **Pas de placement contraint des repères** : DH impose une convention où l'axe `x_i` doit être perpendiculaire à `z_{i-1}`, ce qui force un placement précis des repères. PoE laisse choisir librement les repères tant qu'on connaît `M` et les axes screw `S_i`.
2. **Une seule équation** : la même formule marche pour les joints rotoides et prismatiques (un screw rotoide a `‖ω‖ = 1`, un screw prismatique a `ω = 0` et `‖v‖ = 1`).
3. **Conceptuellement direct** : un screw axis est physique (l'axe autour duquel le joint tourne), pas un artefact de paramétrisation.
4. **Calcul du Jacobien quasi-immédiat** : les colonnes du Jacobien spatial sont directement les screws transformés (J4).

> **Mnémo** : DH = "tu paramètres ton robot en torturant les repères". PoE = "tu lis les axes physiques sur le CAD, tu les exprimes dans un repère monde, t'as fini".

---

## 6. Take-aways

| Concept | À retenir |
|---|---|
| `SO(3)` | rotations 3D, `R^T R = I`, `det R = +1` |
| `SE(3)` | poses rigides = (R, p), encodées en matrice homogène 4x4 |
| Inverse SE(3) | `T^{-1} = [R^T, -R^T p; 0, 1]`, pas `[R^T, -p]` |
| Composition | produit matriciel `T_1 @ T_2`, ordre = du repère le plus à gauche |
| Quaternion | 4 floats, composition rapide, pas de gimbal lock |
| Twist `V = (ω, v)` | élément de `se(3)`, 6D, exponentié donne SE(3) |
| Screw axis | twist normalisé : axe + pitch |
| PoE vs DH | PoE = product of `exp([S_i] q_i)` × M, plus propre, repères libres |

---

## 7. Flash-cards (spaced repetition)

**Q1** — Comment vérifier numériquement qu'une matrice 3x3 est dans SO(3) ?
**R1** — `np.allclose(R.T @ R, np.eye(3))` ET `np.isclose(np.linalg.det(R), 1.0)`. Les deux conditions sont nécessaires (sans le det, on inclut les réflexions).

**Q2** — Quelle est l'inverse de la matrice homogène `T = [R, p; 0, 1]` ?
**R2** — `T^{-1} = [R^T, -R^T p; 0, 1]`. La rotation s'inverse par transposition ; la translation doit être ré-exprimee dans le nouveau repère.

**Q3** — Pourquoi préférer un quaternion à une matrice de rotation pour stocker une orientation ?
**R3** — 4 floats vs 9, composition O(1) plus stable, pas de gimbal lock, `slerp` propre pour interpoler. La matrice reste utile pour appliquer la rotation à un vecteur ou pour la lecture humaine.

**Q4** — Qu'est-ce qu'un twist `V ∈ R^6` et comment passe-t-on de `V` à `T ∈ SE(3)` ?
**R4** — `V = (ω, v)` empile vitesse angulaire et vitesse linéaire (6D). On forme la matrice 4x4 `[V] = [[ω], v; 0, 0]` puis on prend l'exponentielle matricielle : `T = exp([V])`. C'est la forme cinematique de Rodrigues, dimension 6.

**Q5** — Énonce la formule PoE pour la cinematique directe d'un manipulateur n-DOF (formulation espace).
**R5** — `T(q) = exp([S_1] q_1) · ... · exp([S_n] q_n) · M`, où `S_i` est le screw axis du joint i exprimé dans le repère monde (configuration home), et `M` est la pose du end-effector quand tous les `q_i = 0`. Voir `[Lynch & Park, 2017, §4.1.2]`.

---

## Sources

- `[Lynch & Park, 2017]` — *Modern Robotics*, ch. 3 (Rigid-Body Motions). http://hades.mech.northwestern.edu/index.php/Modern_Robotics
- `[Khatib CS223A, L2-3]` — Stanford CS223A Introduction to Robotics, lectures 2-3 (rotations, transformations homogènes). https://www.youtube.com/playlist?list=PL65CC0384A1798ADF

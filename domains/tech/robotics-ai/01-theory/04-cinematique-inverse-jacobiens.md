# J4 — Cinématique inverse + Jacobiens

> **Acquis visé** : résoudre IK numériquement pour un bras (jusqu'à Franka 7-DOF), calculer un Jacobien, gérer une singularité.
> **Prérequis** : J2 (SE(3), twists), J3 (FK via PoE ou DH).
> **Sources** : `[Lynch & Park, 2017, ch. 5-6]`, `CS223A L6-8 (Khatib)`.

---

## 0. Le probleme concret

Tu as un bras planaire 2-DOF. Les deux longueurs valent `L1 = L2 = 1`. Tu veux que l'effecteur atteigne le point cible `(x, y) = (1.2, 0.6)`. **Quels angles `q1, q2` faut-il commander ?**

C'est ca, la cinematique inverse (IK) : etant donnee une pose desiree de l'end-effector, trouver la configuration articulaire qui la realise.

C'est l'inverse exact de J3 (FK : `q -> T`). Sauf que l'inverse est **mille fois plus dur** :

| Probleme | FK (J3) | IK (J4) |
|---|---|---|
| Existence | toujours | pas garantie (cible hors workspace) |
| Unicite | toujours unique | souvent plusieurs solutions (coude haut/bas) |
| Linearite | non lineaire mais explicite | systeme non lineaire a resoudre |
| Forme close | oui (multiplication de matrices) | parfois (analytique) sinon iteratif |

> **Key takeaway** : FK est une **fonction** ; IK est un **probleme inverse**. Pas symetrique.

---

## 1. IK analytique sur le bras 2-DOF planaire

Posons les notations. Les angles sont `q1` (epaule) et `q2` (coude), mesures par rapport a la base et au segment precedent.

La FK donne :

```
x = L1 cos(q1) + L2 cos(q1 + q2)
y = L1 sin(q1) + L2 sin(q1 + q2)
```

On veut inverser ce systeme. Deux equations, deux inconnues, mais non lineaires (sinus, cosinus, somme d'angles).

### 1.1. Trouver `q2` par la loi des cosinus

L'astuce : calculer la distance carree de la base a la cible.

```
x^2 + y^2 = L1^2 + L2^2 + 2 L1 L2 cos(q2)
```

Donc :

```
cos(q2) = (x^2 + y^2 - L1^2 - L2^2) / (2 L1 L2)
```

D'ou :

```
q2 = +/- acos( (x^2 + y^2 - L1^2 - L2^2) / (2 L1 L2) )
```

**Deux solutions** : `+acos` (coude vers le bas) et `-acos` (coude vers le haut). C'est la fameuse **multiplicite** de l'IK.

> Si `|cos(q2)| > 1`, le point est hors workspace. Pas de solution.

### 1.2. Trouver `q1` par atan2

Une fois `q2` fixe, on resout :

```
q1 = atan2(y, x) - atan2(L2 sin(q2), L1 + L2 cos(q2))
```

L'astuce d'atan2 (et non atan) : il gere les 4 quadrants sans ambiguite. **Toujours utiliser atan2 en robotique.**

### 1.3. Pourquoi ca marche ici, et pas (ou rarement) en 6-DOF

Le 2-DOF se ramene a une geometrie de triangle. Pour un bras industriel 6-DOF avec **wrist spherique** (les 3 derniers axes concourants — UR5, Franka modulo une legere offset, KUKA), il existe une solution analytique (decoupage position/orientation). Pour la plupart des bras avec geometrie generale, **pas de forme close** : il faut iterer.

**Exercice mental** : compte les solutions analytiques d'un 6-DOF avec wrist spherique. Reponse : jusqu'a 8 (combinatoire shoulder/elbow/wrist).

---

## 2. Le Jacobien — pourquoi tout passe par lui

Pour iterer, on a besoin de la derivee de la FK. Cette derivee, c'est le **Jacobien**.

### 2.1. Definition

Soit `f : R^n -> R^m` la fonction FK qui mappe `q` (config articulaire, dim n) vers `x` (pose end-effector, dim m). Le Jacobien est :

```
J(q) = df/dq   [matrice m x n]
```

Pour le 2-DOF planaire (`m = 2` : x, y) :

```
J = [ -L1 s1 - L2 s12   -L2 s12 ]
    [  L1 c1 + L2 c12    L2 c12 ]
```

avec `s1 = sin(q1)`, `c1 = cos(q1)`, `s12 = sin(q1+q2)`, `c12 = cos(q1+q2)`.

### 2.2. Vitesse end-effector vs vitesse articulaire

Le Jacobien lie les vitesses :

```
x_dot = J(q) q_dot
```

C'est **lineaire** dans `q_dot`. Concretement : si tu commandes des vitesses articulaires `q_dot`, tu obtiens une vitesse `x_dot` de l'end-effector. Si tu veux une vitesse `x_dot` desiree, tu inverses :

```
q_dot = J^{-1}(q) x_dot      (si J est carre et inversible)
q_dot = J^+(q)   x_dot       (general, pseudo-inverse)
```

`J^+` est la pseudo-inverse de Moore-Penrose. C'est elle qui sert quand `n != m` (redondance).

### 2.3. Forces statiques — le Jacobien transpose

Principe de la dualite cinematique/statique : si l'end-effector applique une force/couple `F` sur l'environnement, les couples articulaires necessaires sont :

```
tau = J^T(q) F
```

Cette equation est **gratuite** : meme Jacobien, juste transpose. Tres utile pour le controle de force et les implementations d'**impedance**.

> **Key takeaway** : le Jacobien fait trois jobs en meme temps :
> 1. derivee de la FK (pour iterer en IK),
> 2. mapping vitesse articulaire <-> vitesse cartesienne,
> 3. mapping force cartesienne <-> couple articulaire (par sa transposee).

### 2.4. Jacobien geometrique vs analytique

Deux variantes existent ; c'est une source classique de bugs.

- **Jacobien geometrique** `J_g` : exprime le twist spatial `V = (omega, v)` en 6D (3 rotation + 3 translation). C'est le plus naturel quand on travaille en SE(3) : `V = J_g(q) q_dot`.
- **Jacobien analytique** `J_a` : derivee de la pose parametree (par ex. (x,y,z) + angles d'Euler). Sa partie rotation depend du choix de parametrisation.

**Pour la position seule**, les deux coincident (J_pos). Pour l'orientation, **prefere le geometrique** : pas de singularite de parametrisation. Lynch & Park ch. 5 utilise quasi exclusivement le Jacobien geometrique.

---

## 3. IK numerique : Newton-Raphson

Quand pas de forme close, on itere. Le principe est generique : Newton-Raphson sur `f(q) - x_target = 0`.

### 3.1. L'iteration

Soit `e(q) = x_target - f(q)` le residu (erreur en pose). On linearise `f` autour de `q_k` :

```
f(q_k + dq) ~ f(q_k) + J(q_k) dq
```

On veut `f(q_k + dq) = x_target`, donc `J(q_k) dq = e(q_k)`. Soit :

```
dq = J^{-1}(q_k) e(q_k)         (si J inversible)
q_{k+1} = q_k + dq
```

Critere d'arret : `||e|| < tol` (typiquement 1e-4 m pour la position).

**Convergence** : quadratique pres de la solution. Mais si on demarre loin, ca peut diverger ou osciller. D'ou l'importance d'une bonne initialisation (en pratique : config courante du robot).

### 3.2. Cas redondant : pseudo-inverse

Pour Franka 7-DOF, la pose est 6D et les articulations sont 7D : `J` est `6 x 7`. Pas inversible. On utilise la pseudo-inverse :

```
dq = J^+ e
```

Parmi les solutions `q_dot` qui realisent le mouvement souhaite, `J^+` choisit celle de **norme minimale**. Geometriquement : projection orthogonale.

Le **noyau** de J (`null(J)`, dim 1 pour Franka) represente les mouvements articulaires qui ne bougent pas l'end-effector — c'est la **redondance**, exploitable pour eviter des obstacles ou des limites.

---

## 4. Singularites — ou Newton-Raphson casse

Une **singularite** est une configuration ou `J(q)` perd du rang. Pour un 6-DOF, c'est `det(J) = 0`. Concretement : il y a des directions cartesiennes que le robot ne peut pas atteindre instantanement, peu importe les vitesses articulaires.

### 4.1. Trois familles classiques (cf. Khatib L7)

1. **Singularite de bras tendu** (workspace boundary) : bras totalement deplie, perte d'un DOF radial.
2. **Singularite d'epaule** : poignet aligne avec l'axe vertical de l'epaule, perte du DOF azimutal.
3. **Singularite de poignet** : axes 4 et 6 alignes (configuration de cardan), un DOF de rotation s'effondre.

### 4.2. Symptome dans Newton-Raphson

Pres d'une singularite, `J` est mal conditionnee. La pseudo-inverse `J^+` explose : pour une petite erreur cartesienne, on demande des vitesses articulaires gigantesques. Le robot tremble, depasse, casse.

### 4.3. Le fix : Damped Least Squares (DLS)

L'astuce de Wampler/Nakamura : remplacer la pseudo-inverse par une version **regularisee**.

```
dq = J^T (J J^T + lambda^2 I)^{-1} e
```

Avec `lambda` un petit parametre d'amortissement (typ. 0.01 a 0.1). Au lieu de minimiser `||e||^2` strict, on minimise :

```
||e - J dq||^2 + lambda^2 ||dq||^2
```

C'est le **classique trade-off** : on accepte une legere imprecision de tracking pres de la singularite, en echange de la stabilite.

**Reglage adaptatif** : on peut faire varier `lambda` en fonction du conditionnement de `J` (faible loin de la singularite, eleve pres).

> **Key takeaway** : DLS = Newton-Raphson + Tikhonov. C'est le **standard de facto** pour l'IK numerique en robotique (utilise par MuJoCo, ROS MoveIt, Drake, et tous les bras industriels).

---

## 5. Recap algo : IK numerique avec DLS

```
Entrees : x_target (pose desiree), q_init, lambda, tol, max_iter
Sortie  : q tel que f(q) ~ x_target

q = q_init
pour k = 1..max_iter :
    e = x_target - f(q)
    si ||e|| < tol : retourner q (succes)
    J = jacobien(q)
    dq = J^T (J J^T + lambda^2 I)^{-1} e   # DLS
    q = q + dq
    q = clip(q, q_min, q_max)               # respecter limites
retourner q (echec : pas converge)
```

Trois **astuces praticiennes** :

1. **Initialiser proche** : utiliser la config courante du robot, ou la solution de l'iteration precedente (controle en boucle).
2. **Step-size cap** : limiter `||dq||` a une valeur max (ex. 0.5 rad) pour eviter les sauts.
3. **Multiple restarts** : si echec, redemarrer depuis une config aleatoire dans les limites.

---

## 6. Quand utiliser quoi ?

| Cas | Methode |
|---|---|
| Bras 2-DOF / 3-DOF planaire | analytique (rapide, exact, multiplicite explicite) |
| 6-DOF avec wrist spherique (UR5, KUKA) | analytique (8 solutions) si dispo, sinon numerique |
| Franka 7-DOF, geometries exotiques, contraintes (limites, obstacles) | **numerique avec DLS** |
| Controle temps reel a 1 kHz | DLS warm-started + step-size cap |
| Optimisation avec contraintes (limites, evitement) | scipy.optimize.minimize avec QP/SLSQP |

---

## 7. Connexion vers J5-J6

- J5 (dynamique) reutilise `J^T` pour calculer les couples articulaires necessaires a une force d'effecteur (computed torque, impedance).
- J6 (controle classique) introduit l'**operational space control** de Khatib, qui est essentiellement du controle direct dans l'espace cartesien via Jacobien et son inverse dynamique.
- L'IK n'est jamais une "fin en soi" en robotique moderne — c'est une **brique** dans des pipelines de planning (J8) et de controle.

---

## 8. Flash-cards (spaced repetition)

1. **Q :** Pourquoi `atan2(y, x)` et pas `atan(y/x)` ?
   **R :** `atan2` distingue les 4 quadrants (gere le signe de x et y separement) et evite la division par zero quand x=0. `atan` est ambigu pi/-pi.

2. **Q :** Pourquoi le bras 2-DOF planaire a-t-il (en general) 2 solutions IK ?
   **R :** `q2 = +/- acos(...)` : configuration coude-haut et coude-bas. Toutes deux atteignent la meme position d'end-effector.

3. **Q :** Que represente le noyau de `J(q)` pour un robot redondant ?
   **R :** L'ensemble des vitesses articulaires `q_dot` qui ne bougent pas l'end-effector. C'est la dimension de **redondance** (n - m) exploitee pour des objectifs secondaires (eviter obstacles, limites).

4. **Q :** Que se passe-t-il si on applique pseudo-inverse pure pres d'une singularite ?
   **R :** `J J^T` devient mal conditionnee, sa pseudo-inverse explose, on demande des vitesses articulaires enormes. Le robot devient instable. Solution : DLS (regularisation Tikhonov avec `lambda^2 I`).

5. **Q :** Quelle est la formule "magique" qui lie couples articulaires et force cartesienne en statique ?
   **R :** `tau = J^T(q) F`. Le Jacobien transpose mappe les efforts cartesiens (force/couple sur l'effecteur) vers les couples articulaires necessaires a les equilibrer.

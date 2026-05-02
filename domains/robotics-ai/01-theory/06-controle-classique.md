# J6 — Controle classique : PID, computed torque, LQR

> Objectif du jour : stabiliser un pendule inverse avec LQR, comprendre pourquoi un PID brut oscille, et savoir tracker une trajectoire articulaire avec computed torque.

---

## 0. L'exemple qui declenche tout : pendule inverse linearise

Considere un pendule inverse monte sur un chariot. L'etat est `x = [theta, theta_dot]^T` (angle par rapport a la verticale, vitesse angulaire). L'entree est un couple `u` applique a l'articulation. Linearise autour de l'equilibre instable `theta = 0`, la dynamique devient :

```
x_dot = A x + B u
A = [[0,        1],
     [g/l,      0]]
B = [[0],
     [1/(m*l^2)]]
```

avec `g = 9.81`, `m = 1.0 kg`, `l = 1.0 m`. La valeur propre de `A` est `±sqrt(g/l) ≈ ±3.13` : un mode strictement instable. Sans controle, theta diverge exponentiellement.

**Ce qu'on fait dans le code du jour** :

1. On essaie un PID brut sur `theta` (consigne `theta = 0`). Resultat : oscillations entretenues, le PID lutte contre l'inertie sans anticiper la dynamique.
2. On calcule le gain LQR `K` via `scipy.linalg.solve_continuous_are` avec `Q = diag(10, 1)` et `R = [[1]]`. La loi de commande devient `u = -K x`.
3. On simule. Le LQR stabilise en ~1.5 s sans overshoot, le PID oscille a ±0.4 rad indefiniment.

C'est la difference entre "regulateur reactif" (PID) et "regulateur optimal model-based" (LQR). Le LQR exploite la connaissance complete de `(A, B)` ; le PID n'a que l'erreur scalaire.

Maintenant qu'on sait OU on va, on peut deplier les 4 grandes familles de controle articulaire.

---

## 1. PID articulaire : la baseline industrielle

### 1.1 Forme canonique

Pour un degre de liberte avec consigne `q_d` :

```
e(t) = q_d - q(t)
u(t) = Kp e(t) + Ki integrale(e) dt + Kd de/dt
```

- **Kp** (proportionnel) : raideur virtuelle, repond a l'erreur courante.
- **Ki** (integral) : annule l'erreur statique (frottement sec, gravite mal compensee).
- **Kd** (derivee) : amortit, anticipe.

### 1.2 Trois pieges classiques

**a) Anti-windup**. Si l'actuateur sature (couple max atteint), `Ki * integrale(e)` continue a gonfler. Quand l'erreur change de signe, le terme integral met du temps a redescendre — le systeme overshoot massivement. Solutions :
- Clamping : freezer l'integrale quand `|u| > u_max`.
- Back-calculation : `dI/dt = e - Kt * (u_sat - u_unsat)`.

**b) Derivee du measurement, pas de l'erreur**. Si on derive `e = q_d - q` et que `q_d` saute (consigne en marche), le terme `Kd * de/dt` produit un pic enorme ("derivative kick"). Solution : deriver uniquement `q` (mesure), pas la consigne :

```
u = Kp * (q_d - q) + Ki * integrale(q_d - q) - Kd * dq/dt
```

**c) Bruit sur la derivee**. Numeriquement, `(q[t] - q[t-1])/dt` amplifie le bruit capteur. En pratique on filtre passe-bas la derivee (filtre du premier ordre, tau ≈ 5-10 ms).

### 1.3 Reglage : Ziegler-Nichols et au-dela

Methode pragmatique : monter `Kp` jusqu'a oscillation entretenue (`Kp_u`, periode `T_u`), puis :
- PI : `Kp = 0.45 Kp_u`, `Ki = 1.2 Kp_u / T_u`
- PID : `Kp = 0.6 Kp_u`, `Ki = 2 Kp_u / T_u`, `Kd = Kp_u T_u / 8`

Mais : sur un robot multi-articule avec couplages dynamiques, le PID independant par axe **ignore les forces de Coriolis et la gravite variable**. C'est pour ca qu'on passe au computed torque.

> **Cle** : le PID est un controleur **lineaire local** sans modele. Il marche partout, mal partout. Pour un robot non-lineaire, on veut un modele dans la boucle.

---

## 2. Computed torque control (CTC)

### 2.1 L'idee : feedforward dynamique + feedback

Rappel J5 : la dynamique d'un manipulateur s'ecrit

```
M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau
```

Si on connait `M, C, g` (via URDF/MuJoCo), on peut **inverser la dynamique**. Avec une trajectoire de reference `q_d(t), q_dot_d(t), q_ddot_d(t)`, on choisit :

```
tau = M(q) [q_ddot_d + Kp e + Kd e_dot] + C(q, q_dot) q_dot + g(q)
```

avec `e = q_d - q`. En remplacant dans la dynamique :

```
M(q) q_ddot + C q_dot + g = M (q_ddot_d + Kp e + Kd e_dot) + C q_dot + g
=> q_ddot = q_ddot_d + Kp e + Kd e_dot
=> e_ddot + Kd e_dot + Kp e = 0
```

Magie : on a transforme un systeme non-lineaire couple en `n` doubles integrateurs decouples avec dynamique d'erreur lineaire choisie. On regle `Kp, Kd` pour amortissement critique : `Kd = 2 sqrt(Kp)`.

### 2.2 Limites

- Sensibilite aux erreurs de modele. Si la masse reelle differe de 20 %, le decouplage est imparfait, l'erreur ne tend plus vers 0.
- Couts CPU : recalculer `M(q), C(q, q_dot), g(q)` a chaque pas. Sur MuJoCo c'est gratuit (`mj_rne` calcule les forces inertielles).
- En presence de contacts/frottements non-modelises, le CTC pur diverge. On ajoute souvent un terme adaptatif (Slotine-Li adaptive control [Siciliano et al., 2009, ch. 8]).

### 2.3 Operational space control (Khatib)

Variation : au lieu de tracker `q_d(t)` dans l'espace articulaire, on tracke `x_d(t)` dans l'espace cartesien (position de l'end-effector). La loi de Khatib (1987) est :

```
F = Lambda(q) x_ddot_d + mu(q, q_dot) + p(q)
tau = J^T F
```

ou `Lambda = (J M^-1 J^T)^-1` est la **masse operationnelle**. C'est le standard pour la manipulation impedance-based (ex : DLR, Franka). Pour le detail : [Siciliano et al., 2009, ch. 8.6].

> **Cle** : computed torque = "lineariser le robot par feedback" via le modele dynamique. Sans modele, on retombe au PID.

---

## 3. LQR — Linear Quadratic Regulator

### 3.1 Probleme et formulation

Soit un systeme lineaire continu `x_dot = A x + B u`. On cherche `u(t)` qui minimise

```
J = integrale_0^infty (x^T Q x + u^T R u) dt
```

- `Q` (n x n, semi-definie positive) : penalise les ecarts d'etat.
- `R` (m x m, definie positive) : penalise l'effort de commande.

Le LQR est l'**unique** controleur optimal pour ce critere quadratique sur une dynamique lineaire [Tedrake, ch. 7].

### 3.2 Solution : equation de Riccati

La solution prend la forme `u = -K x` ou `K = R^-1 B^T P` et `P` resout l'**Algebraic Riccati Equation (ARE)** :

```
A^T P + P A - P B R^-1 B^T P + Q = 0
```

Numeriquement : `scipy.linalg.solve_continuous_are(A, B, Q, R)` retourne `P`. Pour le cas discret, c'est `solve_discrete_are`.

### 3.3 Finite-horizon vs infinite-horizon

- **Infinite-horizon** : `J = integrale_0^infty (...)`. `K` est constant. ARE algebrique. Stable si `(A, B)` controllable et `(A, sqrt(Q))` observable.
- **Finite-horizon** : `J = integrale_0^T (...) + x(T)^T Q_f x(T)`. `K(t)` depend du temps via une **Riccati differentielle** retrograde. C'est ce qu'utilise iLQR (J8).

Pour stabiliser un equilibre, on prend toujours infinite-horizon. Pour suivre une trajectoire de reference avec horizon fini, finite-horizon.

### 3.4 Choix de Q et R

Heuristique de Bryson : normaliser par les valeurs max acceptables.

```
Q_ii = 1 / (x_i_max)^2
R_jj = 1 / (u_j_max)^2
```

Plus `Q/R` est grand, plus le controleur est agressif (etat reagit vite, effort eleve). Equilibre selon les saturations actuateurs.

### 3.5 LQR sur systeme non-lineaire : linearisation autour d'equilibre

Pour le pendule inverse, la dynamique reelle est non-lineaire :

```
theta_ddot = (g/l) sin(theta) + u / (m l^2)
```

On linearise autour de `theta = 0` (equilibre instable). Jacobiens evalues en 0 donnent `(A, B)` ci-dessus. Le LQR calcule sur cette linearisation **stabilise localement** le systeme non-lineaire — region d'attraction limitee, mais suffisante pour un balancier deja proche de la verticale.

Au-dela : on peut chainer un swing-up energetique (J12) qui amene le pendule pres de la verticale, puis switcher au LQR. Ou utiliser iLQR qui resout la trajectoire optimale non-lineaire complete [Tedrake, ch. 8].

> **Cle** : LQR est le **PID optimal** pour systemes lineaires. Pas d'overshoot, gains globalement coherents, garantie de stabilite. Mais necessite un modele lineaire fiable.

---

## 4. Tableau de decision : quel controleur quand ?

| Situation | Choix |
|-----------|-------|
| Position simple, modele inconnu | PID |
| Tracking trajectoire, modele connu | Computed torque |
| Manipulation force/position espace cartesien | Operational space (Khatib) |
| Stabiliser equilibre instable, lineaire | LQR infinite-horizon |
| Swing-up + stabilisation | iLQR ou trajopt + LQR final |
| Contraintes dures (couples, obstacles) | MPC (J12) |

---

## 5. Connexion vers la suite

- **J7-J8** : on quitte le controle pour aller en perception/planning.
- **J12 (model-based RL)** : on revient au controle avec MPC + iLQR, qui generalisent LQR aux contraintes et non-linearites.
- **VLA modernes (J19+)** : les action heads de pi0/OpenVLA fournissent souvent des **trajectoires** que des controleurs classiques (PID, impedance) executent en bas-niveau a 100+ Hz. Le bas-niveau du jour est toujours present dans les VLA frontier 2025-2026.

---

## 6. Flash cards (revision)

**Q1.** Pourquoi un PID brut oscille-t-il sur le pendule inverse ?
**R1.** Le PID est lineaire local sans modele. Il ignore l'instabilite intrinseque (mode propre `+sqrt(g/l)`) et reagit seulement a l'erreur passee. Sans terme anticipatif, il poursuit une cible mouvante.

**Q2.** Donne la loi de computed torque pour tracker `q_d(t)`.
**R2.** `tau = M(q) [q_ddot_d + Kp(q_d - q) + Kd(q_dot_d - q_dot)] + C(q, q_dot) q_dot + g(q)`. Resultat : dynamique d'erreur `e_ddot + Kd e_dot + Kp e = 0`.

**Q3.** Que resout `scipy.linalg.solve_continuous_are(A, B, Q, R)` ?
**R3.** L'equation algebrique de Riccati `A^T P + P A - P B R^-1 B^T P + Q = 0`. Le gain LQR est ensuite `K = R^-1 B^T P`.

**Q4.** Quelle est la difference entre LQR finite-horizon et infinite-horizon ?
**R4.** Finite-horizon : `K(t)` depend du temps, calcule via Riccati differentielle retrograde sur `[0, T]`. Infinite-horizon : `K` constant, ARE algebrique, valable pour stabiliser un equilibre indefiniment.

**Q5.** Pourquoi deriver la mesure et pas l'erreur dans un PID ?
**R5.** Eviter le "derivative kick" : un saut de consigne `q_d` produit une derivee infinie de `e = q_d - q`. Deriver `q` seul lisse la commande aux changements de consigne.

---

## Sources

- [Siciliano et al., 2009, ch. 8] — Robotics: Modelling, Planning and Control. Computed torque, impedance, operational space, anti-windup.
- [Tedrake, ch. 7] — Underactuated Robotics. LQR, Riccati, controlabilite, region d'attraction.
- [Tedrake, ch. 8] — Underactuated Robotics. Lyapunov, region of attraction du LQR sur systeme non-lineaire.

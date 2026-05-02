# Exercice J8 — Hard : RRT* + planning dans le C-space d'un bras 2-DOF

## Objectif

Deux choses serieuses dans cet exercice :

1. **Implementer RRT*** : choix optimal du parent + rewiring des voisins.
2. **Faire du planning dans un C-space non-trivial** : un bras 2-DOF planar
   avec des obstacles dans le **workspace**, dont la projection en C-space
   `(theta_1, theta_2)` est tordue et non-convexe.

C'est la combinaison qui demontre la puissance de sampling-based : on planifie
dans un C-space `(theta_1, theta_2) in [0, 2pi]^2` sans jamais calculer
explicitement `C_obs`.

## Consigne

### Partie A — RRT*

A partir de ton RRT (cours ou exo medium), ajoute :

1. **Choix du parent intelligent** : quand un nouveau noeud `q_new` est
   produit par `steer`, considere **tous les voisins** dans une boule de rayon
   `r` autour de `q_new`. Choisis comme parent celui qui minimise
   `cost(q_start -> q_neigh) + dist(q_neigh, q_new)` parmi ceux pour qui le
   segment est libre.
2. **Rewiring** : pour chacun des memes voisins, verifie si passer par
   `q_new` ameliore son cout courant. Si oui, recable son parent vers
   `q_new`. Propage : pas besoin de recalculer en cascade pour cet exercice
   (mais documente pourquoi on devrait en theorie).
3. **Rayon dynamique** : `r = min(eps * 5, gamma * (log(n) / n)^(1/2))`
   pour un C-space 2D, avec `gamma` constante reglable (commence a 1.5).

### Partie B — Bras 2-DOF planar

Un bras planar a deux liens `L_1 = L_2 = 1.0`, base a l'origine.
Pour `q = (theta_1, theta_2)` :

- Joint 1 (epaule) : `(0, 0)`
- Joint 2 (coude)  : `(L_1 cos(theta_1), L_1 sin(theta_1))`
- End-effector     : `joint_2 + L_2 (cos(theta_1 + theta_2), sin(theta_1 + theta_2))`

Le **workspace** contient deux disques obstacles :
- `(x = 1.2, y = 0.8, r = 0.3)`
- `(x = -0.5, y = 1.5, r = 0.4)`

Ecris :

1. `forward_kinematics(q) -> (joint_2, end_effector)` — utile pour collision et viz.
2. `is_collision_free(q)` qui teste que **les deux liens** (segments
   `(0, 0) -> joint_2` et `joint_2 -> end_effector`) ne touchent **aucun**
   disque. (Distance point-segment a un cercle, formule classique.)
3. Plan avec RRT* dans `[0, 2pi]^2` (attention : le wrapping angulaire peut
   etre ignore pour cet exercice si tu choisis bien `q_start` et `q_goal`).
4. Configurations imposees :
   - `q_start = (0.1, 0.1)` (bras presque tendu vers x positif)
   - `q_goal  = (2.5, -1.0)` (bras pointant ailleurs, evite les obstacles)
5. Visualisations exigees :
   - L'arbre RRT* dans le C-space `(theta_1, theta_2)`.
   - Une animation (ou 5-10 snapshots equireparties) du bras dans le
     workspace en suivant le chemin solution, avec les disques obstacles.

### Partie C — Comparaison RRT vs RRT*

- Sur 10 seeds, mesure le **cout** du chemin (somme des longueurs d'aretes
  dans le C-space) pour RRT et RRT*.
- Trace l'evolution du cout en fonction de `n` (nombre de samples) pour les
  deux.
- Verifie empiriquement que RRT* converge vers un cout plus faible quand
  `n` augmente (RRT, lui, garde le cout du premier chemin trouve).

## Criteres de reussite

- RRT* trouve un chemin **plus court** (cout C-space) que RRT classique sur
  au moins 8 des 10 seeds, pour `n = 2000` samples.
- Le bras suit visuellement le chemin sans traverser les disques obstacles
  (dans le workspace).
- Tu peux articuler en une phrase pourquoi RRT* a un cout
  asymptotiquement optimal et RRT non.

## Indices

- Distance point-segment : projeter le centre du disque sur le segment, prendre
  la distance min entre cette projection (clampee sur le segment) et le
  centre. Collision si distance < rayon.
- Pour les voisins dans la boule : `np.linalg.norm(self.nodes - q_new, axis=1) < r`.
- Le cout d'un noeud peut etre cache dans un tableau `cost[i]` mis a jour
  quand on rewire.
- Si le rewiring devient instable (oscillations), reduit `gamma` ou la taille
  du voisinage.

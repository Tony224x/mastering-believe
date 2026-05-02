# J8 — Motion planning : RRT, PRM, trajectory optimization

> Comment un robot trouve-t-il un chemin sans collision dans un monde encombre ?

---

## 0. Exemple concret : un robot 2D dans un labyrinthe

Imagine un robot ponctuel dans une piece de 10x10 metres. Quelques caisses
rectangulaires sont posees au sol. Tu lui donnes une position de depart
`q_start = (1, 1)` et une cible `q_goal = (9, 9)`. Question : comment trouve-t-il
une trajectoire continue qui evite les caisses ?

Approches naives :

1. **Ligne droite** : `q_start -> q_goal`. Marche si rien sur le chemin, casse
   des qu'une caisse coupe le segment.
2. **Discretiser le sol en grille + A*** : fonctionne en 2D mais explose en haute
   dimension (un bras robot 7-DOF aurait `100^7 = 10^14` cellules).
3. **Sampling-based** : tirer des points au hasard et construire un graphe des
   chemins valides. **Echelle bien jusqu'a >10 dimensions.** C'est la cle.

C'est la classe d'algorithmes RRT / PRM, inventes par LaValle et Kavraki dans les
annees 1990, qui restent l'etat de l'art en motion planning classique
[LaValle, 2006, ch. 5].

---

## 1. Configuration space (C-space) : le concept central

### 1.1 Definition

Le **configuration space** `C` est l'ensemble de toutes les configurations
possibles d'un robot. Chaque point de `C` est une configuration unique `q`.

| Robot | Dimension de `C` | Exemple de `q` |
|-------|------------------|----------------|
| Robot ponctuel 2D | 2 | `(x, y)` |
| Robot mobile non-holonome | 3 | `(x, y, theta)` |
| Bras 2-DOF planar | 2 | `(theta_1, theta_2)` |
| Bras Franka Panda | 7 | `(q_1, ..., q_7)` |
| Humanoide complet | 30+ | toutes les articulations |

### 1.2 Obstacles dans C-space

Un obstacle physique dans le **monde de travail** (workspace `W`) devient une
**region interdite** `C_obs` dans le C-space.

- `C_free = C \ C_obs` : ensemble des configurations sans collision.
- Le motion planning consiste a trouver un **chemin continu** dans `C_free`
  reliant `q_start` a `q_goal`.

**Subtilite** : pour un robot ponctuel 2D, `C_obs` a la meme forme que les
obstacles dans `W`. Pour un bras articule, `C_obs` est une forme tordue dans un
espace de haute dimension qu'on ne peut pas calculer explicitement. **D'ou
l'idee : ne jamais construire `C_obs` ; juste tester `q in C_free`** via une
fonction `is_collision_free(q)`.

```
+-----------------------+         +-----------------------+
|         q_goal        |         |  C_obs (zone hachee)  |
|           *           |         |     XXXXX             |
|       /---/ \-\       |         |     XXXXX             |
|  obs1 \   \  \ \      |         |          C_free       |
|       \---\ obs2      |         |   q_start *--->* q_goal|
|  q_start *            |         |                       |
+-----------------------+         +-----------------------+
       Workspace                      Configuration Space
```

### 1.3 Why sampling-based

Au lieu de construire `C_obs` explicitement, on **echantillonne** des `q` dans
`C` et on garde ceux qui sont libres. Avantage : la verification de collision
est generique (boites englobantes, capsules, FCL/Bullet). On peut faire du
planning dans 50 dimensions.

---

## 2. RRT (Rapidly-exploring Random Tree)

### 2.1 Idee centrale

Construire un **arbre** enracine en `q_start` qui s'etend rapidement vers les
zones inexplorees de `C_free`. Invente par LaValle (1998) precisement pour les
hautes dimensions [LaValle, 2006, ch. 5.5].

### 2.2 Algorithme

```
T = arbre avec un seul noeud q_start
pour i = 1 ... N :
    q_rand <- sample_uniform(C)            # parfois biaise vers q_goal (5-10%)
    q_near <- noeud de T le plus proche de q_rand
    q_new  <- steer(q_near, q_rand, eps)   # avance d'un pas eps depuis q_near
    si is_collision_free(q_near, q_new) :
        ajouter q_new a T avec parent q_near
        si distance(q_new, q_goal) < tol :
            retourner chemin de q_start a q_new
retourner echec
```

### 2.3 Pourquoi ca marche

**Propriete Voronoi-bias** : les noeuds avec une grande cellule de Voronoi
(donc proches des frontieres inexplorees) sont selectionnes plus souvent. Cela
pousse l'arbre vers l'exterieur, d'ou le "rapidly-exploring".

### 2.4 Garanties et limites

| Propriete | Statut |
|-----------|--------|
| Probabilistically complete | Oui : si une solution existe, P(la trouver) -> 1 quand N -> infini |
| Asymptotiquement optimal | **Non** : RRT converge vers une solution sous-optimale |
| Lisse | Non : trajectoire en zigzag, post-processing requis |

---

## 3. RRT* : la version optimale

Karaman & Frazzoli (2011). **Meme cout asymptotique que RRT, mais converge vers
le chemin optimal** quand N -> infini.

Trois changements vs RRT :

1. **Choix du parent intelligent** : au lieu de prendre `q_near` comme parent
   de `q_new`, on cherche dans une boule de rayon `r` autour de `q_new` le
   noeud `q_parent` qui minimise `cost(q_start -> q_parent) + dist(q_parent, q_new)`.
2. **Rewiring** : on verifie si certains voisins de `q_new` peuvent ameliorer
   leur cout en passant par `q_new` comme parent. Si oui, on recable.
3. **Rayon de connexion** `r ~ (log(n)/n)^(1/d)` : decroit avec `n` pour rester
   en `O(n log n)` total.

```
   RRT classique :          RRT* apres rewiring :
                                                
   q_start                  q_start
     |                        |
     a---b                    a
     |   |                    |\
     c   d---q_new            c d
                              |/
                              q_new (recable via le meilleur cout)
```

---

## 4. RRT-Connect : bidirectional speed-up

Kuffner & LaValle (2000). Idee simple : **deux arbres** simultanes, l'un
enracine en `q_start`, l'autre en `q_goal`. A chaque iteration, l'un grandit
vers un sample, l'autre tente de **rejoindre** la nouvelle feuille.

- 5x a 50x plus rapide que RRT en pratique sur des problemes type bras 6-DOF.
- **Pas asymptotiquement optimal** (sauf variante BiRRT*).
- Standard de fait dans MoveIt / OMPL pour les bras industriels.

---

## 5. PRM (Probabilistic Roadmap)

Kavraki, Svestka, Latombe, Overmars (1996). Approche **multi-query** : on
construit une fois un graphe (roadmap) couvrant `C_free`, puis on l'utilise
pour repondre a plusieurs requetes de planification.

### 5.1 Phase de construction (offline)

```
V = ensemble vide ; E = ensemble vide
tant que |V| < N :
    q <- sample_uniform(C)
    si is_collision_free(q) :
        V <- V U {q}
        pour chaque q' dans V parmi les k plus proches voisins de q :
            si is_collision_free(q, q') :
                E <- E U {(q, q')}
```

### 5.2 Phase de requete (online)

Etant donne `(q_start, q_goal)` :
1. Connecter chacun a la roadmap (k plus proches voisins).
2. **Dijkstra ou A*** sur le graphe.

### 5.3 RRT vs PRM : quand quoi ?

| | RRT / RRT-Connect | PRM |
|---|---|---|
| Type de requete | Single-query | Multi-query |
| Environnement | Statique ou dynamique | **Statique** (sinon il faut reconstruire) |
| Cout offline | Aucun | Eleve (construction roadmap) |
| Cout online | Modere | **Tres faible** (Dijkstra) |
| Use case typique | Bras qui change de tache souvent | Robot mobile dans un entrepot fixe |

---

## 6. Trajectory optimization : raffiner la trajectoire

RRT/PRM produisent un chemin **geometrique** (sequence de configurations sans
collision). Mais en robotique on veut souvent une **trajectoire dynamique** :
une fonction `q(t)` qui respecte aussi `q_dot_max`, `tau_max`, et minimise un
cout (energie, jerk, temps).

[Tedrake, 2024, ch. 10] propose deux familles principales.

### 6.1 Direct shooting

On parametrise le **controle** `u(t)` (en general piece-wise constant sur N pas)
et on integre la dynamique `x_{k+1} = f(x_k, u_k)`. Le probleme devient :

```
min_{u_0, ..., u_{N-1}}  sum_k  L(x_k, u_k)
sous  x_0 = x_init
      x_{k+1} = f(x_k, u_k)
      g(x_k, u_k) <= 0     # contraintes (limites joints, obstacles)
```

- **Avantage** : peu de variables (juste les `u_k`).
- **Inconvenient** : non-lineaire, ill-conditione pour horizons longs (l'erreur
  s'integre).

### 6.2 Direct collocation (DIRCOL)

On parametrise **a la fois** `x(t)` et `u(t)` aux noeuds de collocation.
La dynamique devient une **contrainte d'egalite** entre noeuds successifs
(typiquement integration de Hermite-Simpson). On obtient un grand probleme
sparse passe a un solveur NLP (IPOPT, SNOPT).

- **Avantage** : robuste sur horizons longs, contraintes d'etat directes.
- **Inconvenient** : beaucoup de variables, code plus complexe.
- **Standard** pour atterrissage SpaceX, optimisation marche humanoide,
  acrobaties de drones.

### 6.3 iLQR / DDP

Variante de Newton sur le shooting, exploite la structure temporelle pour
factoriser efficacement. Tres rapide, utilise dans MPC en temps reel.

---

## 7. Combinaison planning + control

En pratique :

1. **Planner** (RRT*, PRM) genere un chemin **geometrique** sans collision.
2. **Trajectory smoother** (shortcutting + spline cubic) lisse le chemin et
   produit une trajectoire de reference `q_ref(t)`.
3. **MPC ou computed torque** (cf. J6) suit `q_ref(t)` malgre les
   perturbations.

```
[RRT*]  --chemin geometrique-->  [smoothing]  --q_ref(t)-->  [MPC]  --tau-->  [robot]
                                                                        ^
                                                                        |
                                                                  feedback q,q_dot
```

Cette **separation** est la doctrine classique. Les approches end-to-end
modernes (Diffusion Policy, VLA — voir J16, J19+) court-circuitent ce pipeline
en apprenant directement la trajectoire depuis les pixels.

---

## 8. Take-aways

- **C-space** abstrait n'importe quel robot en un point dans `R^d`. La
  collision check `is_collision_free(q)` suffit, on ne calcule jamais
  explicitement `C_obs`.
- **RRT** : single-query, probabilistically complete, sous-optimal mais rapide.
- **RRT*** : asymptotiquement optimal grace au rewiring, cout `O(n log n)`.
- **RRT-Connect** : bidirectionnel, 5-50x plus rapide en pratique.
- **PRM** : multi-query, brille en environnement statique.
- **Trajectory optimization** raffine un chemin geometrique en trajectoire
  dynamique (DIRCOL est le standard).
- **Pipeline classique** : `RRT* -> smoothing -> MPC` ; pipeline moderne :
  `Diffusion Policy / VLA -> action`.

---

## 9. Flash-cards (active recall)

> Cache la reponse, donne ta version, puis verifie.

**Q1.** Quelle est la complexite typique de `RRT*` apres `n` iterations ?
> `O(n log n)` total grace au rayon de rewiring `r ~ (log n / n)^(1/d)`.

**Q2.** Pourquoi sampling-based bat A* en haute dimension ?
> A* discretise le C-space (cout exponentiel en `d`). Sampling-based ne
> discretise rien, on echantillonne juste — cout linaire en nombre de samples.

**Q3.** Qu'est-ce que `C_obs` et pourquoi on ne le calcule jamais ?
> Region interdite du C-space. Pour un bras 7-DOF c'est une variete de
> dimension 7 implicite, pas de forme analytique. On teste juste `q` un par un.

**Q4.** Direct shooting vs direct collocation : la difference cle ?
> Shooting parametrise `u(t)` seul, integre la dynamique. Collocation
> parametrise `x(t)` ET `u(t)` et impose la dynamique comme contrainte
> d'egalite — plus robuste sur horizons longs.

**Q5.** Pourquoi RRT-Connect est-il typiquement 10x plus rapide que RRT ?
> Deux arbres se rencontrent au milieu. Probabilite de rencontre ~ produit
> des aires explorees, qui croit beaucoup plus vite qu'un seul arbre vers
> `q_goal`.

---

## Sources

- [LaValle, 2006, ch. 5] — *Planning Algorithms*, Cambridge UP. http://lavalle.pl/planning/
- [Tedrake, 2024, ch. 10] — *Underactuated Robotics*, MIT 6.832 (trajectory optimization). https://underactuated.csail.mit.edu/

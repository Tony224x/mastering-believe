# Analyse — A* sur grille ponderee (FleetSim)

Discussion des choix de conception et de la complexite, en complement de
`astar.py` (corrige commente ligne a ligne) et des mesures de `benchmarks/bench_large.py`.

## 1. Complexite

On note `V` le nombre de cases franchissables et `E` le nombre d'aretes
(`E ~ 4V` en 4-connexe, `~ 8V` en 8-connexe).

| Operation | Cout |
|-----------|------|
| Pop du tas (best `f`) | O(log V) |
| Relachement d'une arete (push) | O(log V) |
| Total temps | **O(E log V)** = O(V log V) sur une grille |
| Total espace | **O(V)** (`g_score`, `came_from`, `closed`, le tas) |

A* a la **meme borne pire-cas** que Dijkstra : `O(E log V)`. L'heuristique ne
change pas la borne asymptotique, elle change la **constante** — le nombre de
noeuds reellement etendus avant d'atteindre le goal.

### Pourquoi `closed` sans re-expansion est correct

L'heuristique Manhattan (4-conn) / Chebyshev (8-conn) est **admissible**
(jamais surestimee) ET **consistante** (`h(a) <= cout(a,b) + h(b)`). Avec une
heuristique consistante, le premier `pop` d'un noeud le sort deja avec son cout
optimal : on peut le fermer definitivement sans jamais le re-ouvrir. Si
l'heuristique etait seulement admissible (pas consistante), il faudrait
autoriser la re-ouverture d'un noeud ferme.

### Determinisme

Le tas pousse `(f, counter, node)`. Sans le `counter`, `heapq` comparerait les
`Point` a `f` egal, et l'ordre dependrait de la sequence d'insertion — donc du
parcours. Le `counter` monotone impose un ordre total reproductible : meme
entree => meme chemin, octet pour octet (verifie par `test_determinism`).

## 2. Lecture des benchmarks — l'honnetete sur le "speedup"

Sur la grille a couts varies de `bench_large.py` (500x500), A* est **~1.8-2x**
plus rapide que le Dijkstra naif, pour un **cout optimal identique** (les deux
sont exacts). Ce n'est PAS un facteur 5-10x, et c'est attendu :

- **Grille a couts varies** : l'heuristique reste informative, A* etend nettement
  moins de noeuds. Gain reel mais modere.
- **Grille uniforme ouverte** (tout a cout 1, pas d'obstacle) : Manhattan est
  *exactement tendue*. Tous les noeuds sur l'enveloppe optimale partagent le
  meme `f`, donc A* etend quand meme un **losange complet** de cases — speedup
  proche de **1x**. C'est le cas defavorable classique d'A*.
- **Grille avec murs / couloirs** : c'est la que l'heuristique guide le plus,
  on voit les plus gros gains (la frontiere de Dijkstra explose dans toutes les
  directions, A* fonce vers le goal).

> **Lecon** : la metrique honnete est le **nombre de noeuds etendus**, pas
> seulement le wall-clock. `bench_large.py` affiche les expansions de Dijkstra
> pour rendre ce point tangible. En entretien, dire "A* est plus rapide" sans
> nuancer le type de terrain est une erreur courante.

## 3. Pourquoi pas... ?

- **Dijkstra bidirectionnel** : utile quand on n'a *pas* de bonne heuristique.
  Ici Manhattan est gratuite et excellente — A* unidirectionnel suffit et evite
  la machinerie de rencontre des deux frontieres (critere d'arret subtil, source
  de bugs).
- **Jump Point Search (JPS)** : 10x+ en pratique, mais **seulement sur grille a
  cout uniforme**. Nos allees ont des couts varies (1/2/5/10), ce qui casse
  l'hypothese de symetrie des chemins exploitee par JPS. Hors scope v1.
- **A* hierarchique (HPA*)** : indispensable au-dela de ~1000x1000 avec
  re-planification frequente (precalcul de portails entre regions), mais lourd a
  implementer et a maintenir. Reserve a une v2 si le profil le justifie.

## 4. Limites connues de cette implementation

- Pas de partage de calculs : 10 AGV vers le meme dock = 10 recherches. Un
  unique Dijkstra inverse depuis le goal donnerait les 10 chemins (cf "Pour
  aller plus loin" du README).
- Couts statiques : pas de couche d'evitement dynamique (zone humaine variable
  selon les shifts).
- `g_score`/`came_from` en `dict` : choix memoire correct pour des grilles
  creuses ou de tres grande taille, legerement plus lent qu'un tableau dense sur
  petites grilles (overhead de hachage).

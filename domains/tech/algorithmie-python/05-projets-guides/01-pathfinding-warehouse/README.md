# Projet 01 — Pathfinding entrepot A*

## Contexte metier

Dans FleetSim, un AGV qui recoit l'ordre "deplacer cette palette en zone B-12" ne peut pas filer a vol d'oiseau : il traverse des allees aux couts de mouvement varies (allee large, allee etroite, zone humaine, zone humide, rack en travaux) et doit eviter les zones occupees par d'autres flottes ou par des operateurs. Le moteur doit recalculer des **milliers de trajets par tick** quand plusieurs flottes manoeuvrent en parallele.

La v1 du moteur utilisait un Dijkstra basique, trop lent au-dela de 500 robots. Ta mission : reecrire en A* avec une heuristique bien choisie et prouver le gain.

## Objectif technique

Implementer un pathfinder `astar(zone, start, goal) -> list[tuple[int, int]]` qui :
- Prend une grille 2D ou chaque case a un **cout de mouvement** (1 = allee principale, 2 = allee secondaire, 5 = zone densement peuplee, 10 = zone humaine prioritaire, `inf` = rack/mur infranchissable)
- Retourne la liste des cases du chemin optimal (du start au goal inclus), ou `None` si pas de chemin
- Utilise une heuristique admissible (Manhattan ou Chebyshev selon 4 ou 8 directions)

## Consigne

```python
GridCell = int | float  # cout, ou math.inf si infranchissable
Grid = list[list[GridCell]]
Point = tuple[int, int]  # (row, col)

def astar(zone: Grid, start: Point, goal: Point, allow_diagonal: bool = False) -> list[Point] | None:
    ...
```

Contraintes :
- Tie-breaker deterministe (deux cases a meme f-score doivent etre traitees dans un ordre reproductible)
- Pas de recursion (grilles jusqu'a 2000x2000)
- Valide ton implementation avec la suite de tests fournie (`tests/test_astar.py`, stdlib only)

## Etapes guidees

1. **Modele** — represente l'etat ouvert avec un `heapq`. Le tuple pousse est `(f, counter, node)` : le `counter` assure le tie-breaker deterministe.
2. **Voisinage** — factorise `neighbors(node, allow_diagonal)` pour ne pas dupliquer la logique. Pense aux bornes de la grille.
3. **Heuristique** — Manhattan pour 4-connexe, Chebyshev pour 8-connexe. Prouve qu'elle est admissible (jamais surestime).
4. **Reconstruction** — garde un `came_from: dict[Point, Point]` et remonte du goal vers le start, puis reverse.
5. **Cout d'arete** — le cout pour passer de `a` a `b` est `zone[b]` (cout de la case d'arrivee). Pour 8-connexe, multiplie par `sqrt(2)` en diagonale.
6. **Early exit** — des que `goal` sort du heap, reconstruis. Ne pas attendre que le heap soit vide.

## Criteres de reussite

- Tous les tests de `tests/test_astar.py` passent (`python tests/test_astar.py`)
- Sur `benchmarks/bench_large.py` (grille 500x500 a couts varies), A* trouve **le meme cout optimal** que Dijkstra naif tout en **etendant moins de noeuds** et en etant plus rapide (compter ~2x sur ce terrain ; le gain depend fortement du type de grille — voir `solution/analyse.md`)
- Les chemins sont **reproductibles** entre deux executions (pas de randomness)
- Gere une grille sans solution en retournant `None` (pas d'exception)

> **Piege a comprendre** : sur une grille *uniforme* parfaitement ouverte, l'heuristique Manhattan est exactement tendue et A* etend quand meme un losange complet — speedup proche de 1x. Le gros gain d'A* apparait sur les grilles a murs/couloirs ou a couts varies. La metrique honnete est le **nombre de noeuds etendus**, pas seulement le wall-clock.

## Solution

Voir `solution/astar.py` pour la correction commentee ligne par ligne, et `solution/analyse.md` pour la discussion des choix (complexite, determinisme, lecture honnete du benchmark, pourquoi pas Dijkstra bidirectionnel / JPS / HPA*).

## Pour aller plus loin

- **Jump Point Search (JPS)** sur grille uniforme — 10x plus rapide en pratique
- **Hierarchical pathfinding (HPA*)** — precalcul de regions, indispensable au-dela de 1000x1000
- **Cost layers dynamiques** — ajouter une couche "evitement zone humaine" qui evolue dans le temps (poste de pause, shift change)
- **Partage de calculs** — quand 10 AGV vont au meme dock, faire un seul pathfinding inverse depuis le goal

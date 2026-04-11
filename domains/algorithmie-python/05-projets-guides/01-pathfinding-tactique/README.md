# Projet 01 — Pathfinding tactique A*

## Contexte metier

Dans SWORD, un peloton d'infanterie qui recoit l'ordre "se deplacer au point X" ne peut pas filer a vol d'oiseau : il traverse du terrain varie (route, champ, foret, cours d'eau, zone urbaine) avec des couts de progression tres differents, et doit eviter les zones couvertes par un ennemi connu. Le moteur doit recalculer des **milliers de trajets par tick** quand plusieurs brigades manoeuvrent en parallele.

La v1 du moteur utilisait un Dijkstra basique, trop lent au-dela de 500 unites. Ta mission : reecrire en A* avec une heuristique bien choisie et prouver le gain.

## Objectif technique

Implementer un pathfinder `astar(grid, start, goal) -> list[tuple[int, int]]` qui :
- Prend une grille 2D ou chaque case a un **cout de mouvement** (1 = route, 2 = plaine, 5 = foret, 10 = zone urbaine, `inf` = eau/infranchissable)
- Retourne la liste des cases du chemin optimal (du start au goal inclus), ou `None` si pas de chemin
- Utilise une heuristique admissible (Manhattan ou Chebyshev selon 4 ou 8 directions)

## Consigne

```python
GridCell = int | float  # cout, ou math.inf si infranchissable
Grid = list[list[GridCell]]
Point = tuple[int, int]  # (row, col)

def astar(grid: Grid, start: Point, goal: Point, allow_diagonal: bool = False) -> list[Point] | None:
    ...
```

Contraintes :
- Tie-breaker deterministe (deux cases a meme f-score doivent etre traitees dans un ordre reproductible)
- Pas de recursion (grilles jusqu'a 2000x2000)
- Test sur `tests/` fourni

## Etapes guidees

1. **Modele** — represente l'etat ouvert avec un `heapq`. Le tuple pousse est `(f, counter, node)` : le `counter` assure le tie-breaker deterministe.
2. **Voisinage** — factorise `neighbors(node, allow_diagonal)` pour ne pas dupliquer la logique. Pense aux bornes de la grille.
3. **Heuristique** — Manhattan pour 4-connexe, Chebyshev pour 8-connexe. Prouve qu'elle est admissible (jamais surestime).
4. **Reconstruction** — garde un `came_from: dict[Point, Point]` et remonte du goal vers le start, puis reverse.
5. **Cout d'arete** — le cout pour passer de `a` a `b` est `grid[b]` (cout de la case d'arrivee). Pour 8-connexe, multiplie par `sqrt(2)` en diagonale.
6. **Early exit** — des que `goal` sort du heap, reconstruis. Ne pas attendre que le heap soit vide.

## Criteres de reussite

- Tous les tests de `tests/test_astar.py` passent
- Sur `benchmarks/bench_large.py`, A* est au moins **5x plus rapide** que Dijkstra naif sur une grille 500x500
- Les chemins sont **reproductibles** entre deux executions (pas de randomness)
- Gere une grille sans solution en retournant `None` (pas d'exception)

## Solution

Voir `solution/astar.py` pour la correction commentee ligne par ligne, et `solution/analyse.md` pour la discussion des choix (pourquoi pas Dijkstra bidirectionnel, pourquoi pas JPS, etc.).

## Pour aller plus loin

- **Jump Point Search (JPS)** sur grille uniforme — 10x plus rapide en pratique
- **Hierarchical pathfinding (HPA*)** — precalcul de regions, indispensable au-dela de 1000x1000
- **Cost layers dynamiques** — ajouter une couche "peur du feu ennemi" qui evolue dans le temps
- **Partage de calculs** — quand 10 unites vont au meme endroit, faire un seul pathfinding inverse depuis le goal

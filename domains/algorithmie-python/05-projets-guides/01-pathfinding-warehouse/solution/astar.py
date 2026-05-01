"""
Correction commentee — A* tactique pour FleetSim.

Objectif : pathfinding deterministe sur grille ponderee avec couts terrain.
Cle de lecture : chaque commentaire explique le POURQUOI (choix de design),
pas le QUOI (lisible dans le code).
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable

Point = tuple[int, int]
Grid = list[list[float]]


# 4-connexe d'abord : deterministe, pas de probleme d'obstacle "corner cutting".
# En 8-connexe on ajoute les diagonales. L'ordre est fixe (pas de set) pour
# garantir le determinisme quand deux voisins ont meme f-score.
_MOVES_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MOVES_8 = _MOVES_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _heuristic(a: Point, b: Point, allow_diagonal: bool) -> float:
    """Manhattan pour 4-connexe, Chebyshev pour 8-connexe.

    Les deux sont admissibles (ne surestiment jamais le cout reel) si le cout
    minimal d'un pas vaut 1. Chebyshev est plus tight en 8-connexe car elle
    prend en compte les diagonales. Octile serait encore meilleure (ajoute un
    terme sqrt(2)-1 pour les diagonales), laissee en extension.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if allow_diagonal:
        return max(dr, dc)
    return dr + dc


def _in_bounds(zone: Grid, p: Point) -> bool:
    r, c = p
    return 0 <= r < len(zone) and 0 <= c < len(zone[0])


def _neighbors(zone: Grid, node: Point, allow_diagonal: bool) -> Iterable[tuple[Point, float]]:
    """Yield (voisin, cout_arete). Cout = cout de la case d'arrivee (convention).

    On multiplie par sqrt(2) en diagonale : un pas diagonal couvre plus de
    distance mais les deux cases adjacentes doivent etre franchissables
    (evite le "diagonal corner cutting" a travers un mur).
    """
    moves = _MOVES_8 if allow_diagonal else _MOVES_4
    for dr, dc in moves:
        np = (node[0] + dr, node[1] + dc)
        if not _in_bounds(zone, np):
            continue
        cost = zone[np[0]][np[1]]
        if cost == math.inf:
            continue
        if dr != 0 and dc != 0:
            # Anti-corner-cutting : les deux cases orthogonales doivent etre libres.
            if zone[node[0] + dr][node[1]] == math.inf:
                continue
            if zone[node[0]][node[1] + dc] == math.inf:
                continue
            yield np, cost * math.sqrt(2)
        else:
            yield np, cost


def astar(zone: Grid, start: Point, goal: Point, allow_diagonal: bool = False) -> list[Point] | None:
    """A* sur grille ponderee. Retourne la liste des cases (start -> goal)
    ou None si pas de chemin.

    Pourquoi A* et pas Dijkstra : l'heuristique guide l'exploration vers le
    goal et reduit drastiquement le nombre de noeuds developpes. Sur une
    grille 500x500 ouverte, Dijkstra developpe ~O(N*M), A* developpe ~O(N)
    ou N est la longueur du chemin.

    Pourquoi un counter dans le tuple heap : heapq compare les tuples
    lexicographiquement. Si deux noeuds ont meme f-score, Python comparerait
    les Points, et l'ordre dependrait de l'insertion -> non deterministe
    quand on change les voisins. Le counter garantit un ordre strict.
    """
    if not _in_bounds(zone, start) or not _in_bounds(zone, goal):
        return None
    if zone[start[0]][start[1]] == math.inf or zone[goal[0]][goal[1]] == math.inf:
        return None

    # g_score[node] = cout reel depuis start. On n'initialise pas toute la
    # grille (gaspillage memoire sur 2000x2000), on utilise un dict sparse.
    g_score: dict[Point, float] = {start: 0.0}
    came_from: dict[Point, Point] = {}

    counter = 0  # tie-breaker monotone, garantit le determinisme
    open_heap: list[tuple[float, int, Point]] = [(_heuristic(start, goal, allow_diagonal), counter, start)]

    # closed_set : noeuds dont on a fixe le cout optimal. On ne les redeveloppe
    # pas. Valide parce que l'heuristique est admissible ET consistante
    # (ce qui est le cas pour Manhattan/Chebyshev sur grille).
    closed: set[Point] = set()

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruction : remonter came_from depuis goal, puis reverse.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed:
            # On peut avoir plusieurs entrees pour un meme noeud si on a
            # trouve un meilleur chemin apres la premiere insertion. On skip
            # les obsoletes (variante "lazy deletion" du heap).
            continue
        closed.add(current)

        for neighbor, edge_cost in _neighbors(zone, current, allow_diagonal):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_neighbor = tentative_g + _heuristic(neighbor, goal, allow_diagonal)
                counter += 1
                heapq.heappush(open_heap, (f_neighbor, counter, neighbor))

    return None  # open vide, goal non atteint


if __name__ == "__main__":
    # Demo : petite carte avec foret (cout 5), route (cout 1), obstacle (inf).
    INF = math.inf
    demo: Grid = [
        [1, 1, 1, 5, 5, 5],
        [5, 5, 1, 5, INF, 5],
        [5, 5, 1, 1, 1, 5],
        [5, INF, INF, INF, 1, 5],
        [5, 5, 5, 5, 1, 1],
    ]
    path = astar(demo, (0, 0), (4, 5))
    print("Chemin:", path)
    # Le chemin optimal passe par la route (cout 1) pour eviter la foret.

"""
Commented solution — tactical A* for FleetSim.

Goal: deterministic pathfinding on a weighted grid with terrain costs.
Reading key: every comment explains the WHY (design choice),
not the WHAT (readable in the code).
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable

Point = tuple[int, int]
Grid = list[list[float]]


# 4-connected first: deterministic, no "corner cutting" obstacle issue.
# 8-connected adds the diagonals. The order is fixed (no set) to
# guarantee determinism when two neighbors share the same f-score.
_MOVES_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MOVES_8 = _MOVES_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _heuristic(a: Point, b: Point, allow_diagonal: bool) -> float:
    """Manhattan for 4-connected, Chebyshev for 8-connected.

    Both are admissible (never overestimate the true cost) as long as the
    minimal step cost is 1. Chebyshev is tighter in 8-connected mode because
    it accounts for diagonals. Octile would be even better (adds a
    sqrt(2)-1 term for diagonals), left as an extension.
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
    """Yield (neighbor, edge_cost). Cost = cost of the destination cell (convention).

    We multiply by sqrt(2) on diagonals: a diagonal step covers more
    distance, and both adjacent cells must be traversable
    (prevents "diagonal corner cutting" through a wall).
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
            # Anti-corner-cutting: both orthogonal cells must be free.
            if zone[node[0] + dr][node[1]] == math.inf:
                continue
            if zone[node[0]][node[1] + dc] == math.inf:
                continue
            yield np, cost * math.sqrt(2)
        else:
            yield np, cost


def astar(zone: Grid, start: Point, goal: Point, allow_diagonal: bool = False) -> list[Point] | None:
    """A* on a weighted grid. Returns the list of cells (start -> goal)
    or None if there is no path.

    Why A* and not Dijkstra: the heuristic steers the exploration toward the
    goal and drastically reduces the number of expanded nodes. On an open
    500x500 grid, Dijkstra expands ~O(N*M), A* expands ~O(N)
    where N is the path length.

    Why a counter inside the heap tuple: heapq compares tuples
    lexicographically. If two nodes have the same f-score, Python would
    compare the Points, and the order would depend on insertion -> non
    deterministic when the neighbors change. The counter guarantees a strict order.
    """
    if not _in_bounds(zone, start) or not _in_bounds(zone, goal):
        return None
    if zone[start[0]][start[1]] == math.inf or zone[goal[0]][goal[1]] == math.inf:
        return None

    # g_score[node] = actual cost from start. We do not initialize the whole
    # grid (memory waste on 2000x2000), we use a sparse dict instead.
    g_score: dict[Point, float] = {start: 0.0}
    came_from: dict[Point, Point] = {}

    counter = 0  # monotonic tie-breaker, guarantees determinism
    open_heap: list[tuple[float, int, Point]] = [(_heuristic(start, goal, allow_diagonal), counter, start)]

    # closed_set: nodes whose optimal cost is settled. We never re-expand
    # them. Valid because the heuristic is admissible AND consistent
    # (which is the case for Manhattan/Chebyshev on a grid).
    closed: set[Point] = set()

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruction: walk came_from back from goal, then reverse.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed:
            # The same node can have several heap entries if a better path
            # was found after the first insertion. We skip the stale ones
            # ("lazy deletion" variant of the heap).
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

    return None  # open set empty, goal not reached


if __name__ == "__main__":
    # Demo: small map with forest (cost 5), road (cost 1), obstacle (inf).
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
    # The optimal path follows the road (cost 1) to avoid the forest.

"""
Benchmark: A* vs. naive Dijkstra on a large open grid.

Pure stdlib (`time`, `random`). Run with:
    python benchmarks/bench_large.py            # default 500x500
    python benchmarks/bench_large.py 300        # custom side length

What it shows: on an open grid the goal-directed heuristic lets A* expand far
fewer nodes than Dijkstra (which fans out uniformly). We report wall-clock time,
the optimal cost found by each (must be EQUAL — same convention, both optimal),
and the speedup ratio.

Reading key: this is a demonstration harness, not a micro-benchmark. Both
solvers share the same edge-cost convention as solution/astar.py so the costs
are directly comparable.
"""
from __future__ import annotations

import heapq
import math
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "solution"))

from astar import astar  # noqa: E402

Point = tuple[int, int]
Grid = list[list[float]]
INF = math.inf
_MOVES_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def dijkstra(zone: Grid, start: Point, goal: Point) -> tuple[list[Point] | None, float, int]:
    """Naive uniform-cost search (no heuristic). Returns (path, cost, expansions)."""
    rows, cols = len(zone), len(zone[0])
    if zone[start[0]][start[1]] == INF or zone[goal[0]][goal[1]] == INF:
        return None, INF, 0
    dist = {start: 0.0}
    came_from: dict[Point, Point] = {}
    counter = 0
    expansions = 0
    pq: list[tuple[float, int, Point]] = [(0.0, 0, start)]
    closed: set[Point] = set()
    while pq:
        d, _, node = heapq.heappop(pq)
        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path, d, expansions
        if node in closed:
            continue
        closed.add(node)
        expansions += 1
        r, c = node
        for dr, dc in _MOVES_4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = zone[nr][nc]
            if cell == INF:
                continue
            nd = d + cell
            if nd < dist.get((nr, nc), INF):
                dist[(nr, nc)] = nd
                came_from[(nr, nc)] = node
                counter += 1
                heapq.heappush(pq, (nd, counter, (nr, nc)))
    return None, INF, expansions


def build_grid(n: int, seed: int = 7) -> Grid:
    """Open grid with a few cost bands and sparse obstacles, corner-to-corner solvable."""
    rng = random.Random(seed)
    grid: Grid = [[rng.choice([1, 1, 1, 1, 2, 5]) for _ in range(n)] for _ in range(n)]
    # Sprinkle obstacles but keep the two diagonals clear so a path exists.
    for _ in range(n * n // 12):
        r, c = rng.randrange(n), rng.randrange(n)
        if r == c or r + c == n - 1:
            continue
        grid[r][c] = INF
    grid[0][0] = grid[n - 1][n - 1] = 1
    return grid


def _path_cost(zone: Grid, path: list[Point]) -> float:
    return sum(zone[r][c] for (r, c) in path[1:]) if path else INF


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Building {n}x{n} grid...")
    grid = build_grid(n)
    start, goal = (0, 0), (n - 1, n - 1)

    t0 = time.perf_counter()
    a_path = astar(grid, start, goal)
    t_astar = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_path, _, d_exp = dijkstra(grid, start, goal)
    t_dij = time.perf_counter() - t0

    a_cost = _path_cost(grid, a_path) if a_path else None
    d_cost = _path_cost(grid, d_path) if d_path else None

    print(f"A*       : {t_astar:7.3f}s   cost={a_cost}")
    print(f"Dijkstra : {t_dij:7.3f}s   cost={d_cost}   expansions={d_exp}")
    if a_cost is not None and d_cost is not None:
        same = math.isclose(a_cost, d_cost)
        print(f"Same optimal cost: {same}")
        if t_astar > 0:
            print(f"Speedup (Dijkstra / A*): {t_dij / t_astar:.1f}x")
        print(
            "Note: the speedup depends heavily on the terrain. On grids with "
            "varied costs the heuristic stays informative; on a perfectly "
            "uniform open grid the Manhattan heuristic is exactly tight and A* "
            "still expands a full diamond -> little to no speedup. The honest "
            "metric is *nodes expanded*, not just wall-clock time."
        )
        if not same:
            print("WARNING: costs differ — one solver is wrong.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

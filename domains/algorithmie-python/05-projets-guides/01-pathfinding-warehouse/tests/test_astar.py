"""
Minimal test suite for the A* pathfinder.

Pure stdlib (no pytest): run with `python tests/test_astar.py`.
Exit code 0 = all green, 1 = at least one failure.

Reading key: each test states the INVARIANT it protects, not just the I/O.
The reference oracle is a brute Dijkstra (Section "oracle") — slow but obviously
correct — so we check A* against ground truth, not against itself.
"""
from __future__ import annotations

import heapq
import math
import os
import sys

# Allow `python tests/test_astar.py` from any cwd: add solution/ to path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "solution"))

from astar import astar  # noqa: E402

Point = tuple[int, int]
Grid = list[list[float]]
INF = math.inf


# ---------------------------------------------------------------------------
# Oracle: a deliberately simple Dijkstra. Same edge-cost convention as astar
# (cost = destination cell, sqrt(2) on diagonals). We only return the optimal
# COST here; A* returns a path, so we sum the path's edge costs and compare.
# ---------------------------------------------------------------------------
def _dijkstra_cost(zone: Grid, start: Point, goal: Point, allow_diagonal: bool = False) -> float | None:
    rows, cols = len(zone), len(zone[0])
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if allow_diagonal:
        moves = moves + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    if zone[start[0]][start[1]] == INF or zone[goal[0]][goal[1]] == INF:
        return None
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if node == goal:
            return d
        if d > dist.get(node, INF):
            continue
        r, c = node
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = zone[nr][nc]
            if cell == INF:
                continue
            if dr != 0 and dc != 0:
                if zone[r + dr][c] == INF or zone[r][c + dc] == INF:
                    continue
                step = cell * math.sqrt(2)
            else:
                step = cell
            nd = d + step
            if nd < dist.get((nr, nc), INF):
                dist[(nr, nc)] = nd
                heapq.heappush(pq, (nd, (nr, nc)))
    return None


def _path_cost(zone: Grid, path: list[Point], allow_diagonal: bool) -> float:
    total = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        cell = zone[r2][c2]
        if r1 != r2 and c1 != c2:
            total += cell * math.sqrt(2)
        else:
            total += cell
    return total


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_trivial():
    """Start == goal returns a single-cell path."""
    grid = [[1, 1], [1, 1]]
    p = astar(grid, (0, 0), (0, 0))
    check("trivial: start == goal", p == [(0, 0)], repr(p))


def test_straight_line():
    """Open uniform grid: optimal cost == Manhattan distance of unit cells."""
    grid = [[1] * 5 for _ in range(5)]
    p = astar(grid, (0, 0), (4, 4))
    cost = _path_cost(grid, p, allow_diagonal=False)
    check("straight line: path is connected", _is_connected(p, False))
    check("straight line: optimal cost", math.isclose(cost, _dijkstra_cost(grid, (0, 0), (4, 4))))


def test_weighted_detour():
    """A* must route around an expensive band, not bulldoze through it."""
    grid = [
        [1, 1, 1, 1, 1],
        [9, 9, 9, 9, 1],
        [1, 1, 1, 1, 1],
        [1, 9, 9, 9, 9],
        [1, 1, 1, 1, 1],
    ]
    p = astar(grid, (0, 0), (4, 4))
    cost = _path_cost(grid, p, allow_diagonal=False)
    check("weighted detour: optimal cost matches Dijkstra",
          math.isclose(cost, _dijkstra_cost(grid, (0, 0), (4, 4))), f"got {cost}")


def test_no_path():
    """A wall splitting the grid -> None, not an exception."""
    grid = [
        [1, 1, INF, 1, 1],
        [1, 1, INF, 1, 1],
        [1, 1, INF, 1, 1],
    ]
    check("no path: returns None", astar(grid, (0, 0), (0, 4)) is None)


def test_blocked_endpoints():
    """Start or goal on an obstacle -> None."""
    grid = [[INF, 1], [1, 1]]
    check("blocked start: None", astar(grid, (0, 0), (1, 1)) is None)
    check("blocked goal: None", astar(grid, (1, 1), (0, 0)) is None)


def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    check("out of bounds: None", astar(grid, (0, 0), (5, 5)) is None)


def test_determinism():
    """Same inputs, two runs -> byte-identical path (no randomness)."""
    grid = [
        [1, 2, 1, 1, 1],
        [1, 2, 5, 2, 1],
        [1, 1, 1, 2, 1],
        [5, 5, 1, 1, 1],
        [1, 1, 1, 5, 1],
    ]
    a = astar(grid, (0, 0), (4, 4))
    b = astar(grid, (0, 0), (4, 4))
    check("determinism: identical across runs", a == b)


def test_diagonal_optimality():
    """8-connected: A* cost must match the 8-connected Dijkstra oracle."""
    grid = [[1] * 6 for _ in range(6)]
    p = astar(grid, (0, 0), (5, 5), allow_diagonal=True)
    cost = _path_cost(grid, p, allow_diagonal=True)
    check("diagonal: connected (8-conn)", _is_connected(p, True))
    check("diagonal: optimal cost",
          math.isclose(cost, _dijkstra_cost(grid, (0, 0), (5, 5), allow_diagonal=True)), f"got {cost}")


def test_optimality_random():
    """Fuzz: many random grids, A* cost must equal the Dijkstra oracle."""
    import random
    rng = random.Random(42)  # fixed seed -> reproducible failure if any
    bad = 0
    for _ in range(200):
        n = rng.randint(3, 8)
        grid = [[rng.choice([1, 1, 1, 2, 5, INF]) for _ in range(n)] for _ in range(n)]
        start, goal = (0, 0), (n - 1, n - 1)
        grid[0][0] = 1
        grid[n - 1][n - 1] = 1
        p = astar(grid, start, goal)
        oracle = _dijkstra_cost(grid, start, goal)
        if p is None:
            if oracle is not None:
                bad += 1
        else:
            if oracle is None or not math.isclose(_path_cost(grid, p, False), oracle):
                bad += 1
    check("fuzz: A* matches Dijkstra on 200 random grids", bad == 0, f"{bad} mismatches")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_connected(path: list[Point], allow_diagonal: bool) -> bool:
    if not path:
        return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = abs(r1 - r2), abs(c1 - c2)
        if allow_diagonal:
            if max(dr, dc) != 1:
                return False
        else:
            if dr + dc != 1:
                return False
    return True


def main() -> int:
    tests = [
        test_trivial, test_straight_line, test_weighted_detour, test_no_path,
        test_blocked_endpoints, test_out_of_bounds, test_determinism,
        test_diagonal_optimality, test_optimality_random,
    ]
    print("Running A* tests...")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

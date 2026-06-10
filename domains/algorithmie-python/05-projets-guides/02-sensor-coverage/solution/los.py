"""
Symmetric line-of-sight via Bresenham — solution.

The main trap: standard Bresenham is not symmetric on half-cells.
We canonicalize by always tracing from the lexicographically smaller Point.
"""
from __future__ import annotations

Point = tuple[int, int]
Grid = list[list[bool]]  # True = obstacle


def line_of_sight(zone: Grid, a: Point, b: Point) -> bool:
    """True if the LOS from a to b is clear. a and b excluded from the test."""
    # Canonicalization: guarantees the symmetry LOS(a,b) == LOS(b,a).
    # Without it, the line traced from (0,0) to (4,3) can differ from the one
    # traced from (4,3) to (0,0) when the slope falls between two octants.
    if (a[0], a[1]) > (b[0], b[1]):
        a, b = b, a

    r0, c0 = a
    r1, c1 = b

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    # Exclude the starting point: we step once before checking.
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        # Obstacle test, except on the arrival point
        if (r, c) != (r1, c1) and zone[r][c]:
            return False

    return True


def bench() -> None:
    """Benchmark: 40,000 calls on a 200x200 grid."""
    import random
    import time

    random.seed(42)  # determinism of the bench itself
    SIZE = 200
    zone: Grid = [[random.random() < 0.15 for _ in range(SIZE)] for _ in range(SIZE)]

    pairs = [
        ((random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)),
         (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)))
        for _ in range(40_000)
    ]

    start = time.perf_counter()
    visible = sum(1 for a, b in pairs if line_of_sight(zone, a, b))
    elapsed = time.perf_counter() - start

    print(f"40 000 LOS en {elapsed*1000:.1f} ms ({visible} paires visibles)")
    # Target: < 50 ms on a modern machine.


if __name__ == "__main__":
    # Sanity test: clear line vs blocked line
    g: Grid = [[False] * 5 for _ in range(5)]
    assert line_of_sight(g, (0, 0), (4, 4))
    g[2][2] = True
    assert not line_of_sight(g, (0, 0), (4, 4))
    # Symmetry
    assert line_of_sight(g, (0, 0), (4, 4)) == line_of_sight(g, (4, 4), (0, 0))
    print("Tests OK")
    bench()

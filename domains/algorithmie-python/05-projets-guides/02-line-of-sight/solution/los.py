"""
Line-of-sight symetrique par Bresenham — correction.

Le piege principal : Bresenham standard n'est pas symetrique sur les demi-cases.
On canonicalise en tracant toujours depuis le Point lexicographiquement inferieur.
"""
from __future__ import annotations

Point = tuple[int, int]
Grid = list[list[bool]]  # True = obstacle


def line_of_sight(grid: Grid, a: Point, b: Point) -> bool:
    """True si la LOS de a vers b est libre. a et b exclus du test."""
    # Canonicalisation : garantit la symetrie LOS(a,b) == LOS(b,a).
    # Sans ca, la ligne tracee de (0,0) vers (4,3) peut differer de celle
    # tracee de (4,3) vers (0,0) quand la pente tombe entre deux octants.
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
    # On exclut le point de depart : on fait un "step" avant de checker.
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        # Test d'obstacle sauf sur le point d'arrivee
        if (r, c) != (r1, c1) and grid[r][c]:
            return False

    return True


def bench() -> None:
    """Benchmark : 40 000 appels sur grille 200x200."""
    import random
    import time

    random.seed(42)  # determinisme du bench lui-meme
    SIZE = 200
    grid: Grid = [[random.random() < 0.15 for _ in range(SIZE)] for _ in range(SIZE)]

    pairs = [
        ((random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)),
         (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)))
        for _ in range(40_000)
    ]

    start = time.perf_counter()
    visible = sum(1 for a, b in pairs if line_of_sight(grid, a, b))
    elapsed = time.perf_counter() - start

    print(f"40 000 LOS en {elapsed*1000:.1f} ms ({visible} paires visibles)")
    # Objectif : < 50 ms sur une machine moderne.


if __name__ == "__main__":
    # Test sanity : ligne libre vs bloquee
    g: Grid = [[False] * 5 for _ in range(5)]
    assert line_of_sight(g, (0, 0), (4, 4))
    g[2][2] = True
    assert not line_of_sight(g, (0, 0), (4, 4))
    # Symetrie
    assert line_of_sight(g, (0, 0), (4, 4)) == line_of_sight(g, (4, 4), (0, 0))
    print("Tests OK")
    bench()

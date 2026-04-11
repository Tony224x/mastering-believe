# Projet 02 — Line-of-sight sur grille (Bresenham)

## Contexte metier

Chaque tick de SWORD, le moteur doit calculer "qui voit qui" pour determiner les detections, les tirs possibles et la fog of war. Pour N unites amies et M unites ennemies, c'est N*M tests de visibilite. Sur un exercice de brigade (200 unites par camp), ca fait **40 000 tests par tick** a 20 Hz = **800 000 appels/seconde** dans le pire cas.

La primitive fondamentale : etant donne un tireur en `(r1, c1)`, une cible en `(r2, c2)`, et une grille d'elevations / obstacles, est-ce que la ligne entre les deux est bloquee ?

## Objectif technique

Implementer `line_of_sight(grid, a, b) -> bool` qui :
- Prend une grille 2D de booleens (`True` = obstacle, `False` = passable) ou d'elevations
- Retourne `True` si la ligne de `a` a `b` n'est bloquee par aucune cellule, `False` sinon
- Utilise l'algorithme de **Bresenham** pour tracer la ligne en O(max(dr, dc)) sans flottants
- Sort immediatement au premier obstacle rencontre (early exit)

## Consigne

```python
Point = tuple[int, int]
Grid = list[list[bool]]  # True = bloque

def line_of_sight(grid: Grid, a: Point, b: Point) -> bool:
    """True si la LOS de a vers b est libre (a et b exclus du test)."""
```

Version elevations (extension) :
```python
ElevGrid = list[list[float]]

def line_of_sight_3d(grid: ElevGrid, a: Point, b: Point, a_height: float, b_height: float) -> bool:
    """Vrai si aucune cellule intermediaire n'a une elevation > a la ligne interpolee."""
```

## Etapes guidees

1. **Bresenham 2D** — implementer l'algorithme "octant zero" d'abord, puis generaliser via les signes et le swap dr/dc quand `abs(dc) < abs(dr)`.
2. **Ne pas allouer de liste** — iterer avec un generator ou directement dans la boucle. Allouer une liste pour chaque appel casse le cache et multiplie le GC.
3. **Exclure a et b** — le tireur et la cible ne bloquent pas leur propre vue.
4. **Early exit** — des qu'on touche un obstacle, `return False`.
5. **Benchmark** — mesurer avec `timeit` le cout d'un appel seul, puis le cout pour N*M = 40 000 appels.
6. **Extension elevation** — interpoler lineairement l'altitude du tireur a la cible, comparer a l'elevation de chaque cellule traversee.

## Criteres de reussite

- Tests unitaires passent (ligne horizontale, verticale, diagonale parfaite, avec/sans obstacle)
- 40 000 appels en **moins de 100 ms** sur une grille 200x200 en pur Python (benchmark fourni, mesure reference : ~51 ms)
- Pas d'allocation de liste intermediaire (verifie avec `tracemalloc`)
- Determinisme : LOS de `a -> b` == LOS de `b -> a` (la ligne doit etre symetrique)

**Note realisme** : en pur Python, la solution reference tourne autour de 50 ms sur une machine moderne. Le critere de 100 ms laisse de la marge pour une machine plus lente ou une implementation un peu moins optimisee. Pour descendre significativement sous les 20 ms, il faut **vectoriser avec numpy** (traiter les paires en batch) ou descendre en **C extension / Cython**. Voir "Pour aller plus loin".

## Piege classique

Bresenham standard **n'est pas symetrique** : `line(a, b)` peut passer par des cellules differentes de `line(b, a)` quand la pente est pile entre deux octants. Pour SWORD c'est inacceptable (si A voit B alors B voit A). Deux solutions :
- Canonicaliser : toujours tracer du point avec `(r, c)` lexicographiquement plus petit vers l'autre
- Utiliser une variante symetrique (ex: algo de Xiaolin Wu sans anti-aliasing)

## Solution

Voir `solution/los.py` pour la correction avec la variante symetrique et le benchmark.

## Pour aller plus loin

- **Pre-computed visibility** — pour des postes d'observation fixes, precalculer une carte de visibilite et faire O(1) au runtime
- **Vectorisation numpy** — batch 1000 LOS en un seul appel numpy (gain 10-100x si on traite des salves de tir)
- **Shadow casting** — algorithme de roguelike pour calculer d'un coup tout le champ de vision d'une unite (utile pour fog of war)

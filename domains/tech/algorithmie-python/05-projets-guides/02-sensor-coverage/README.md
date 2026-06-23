# Projet 02 — Couverture sensorielle sur grille (Bresenham)

## Contexte metier

Chaque tick de FleetSim, le moteur doit calculer "quel capteur voit quoi" pour determiner les detections, les pickups possibles, et la **couverture** entre deux points de l'entrepot. Pour N capteurs (cameras, lidars, drones) et M cibles potentielles (colis, AGV, operateurs), c'est N*M tests de visibilite. Sur un shift complet (200 capteurs / camera + 200 cibles), ca fait **40 000 tests par tick** a 20 Hz = **800 000 appels/seconde** dans le pire cas.

La primitive fondamentale : etant donne un capteur en `(r1, c1)`, une cible en `(r2, c2)`, et une grille d'obstacles (racks, murs, machines), est-ce que la ligne directe entre les deux est libre ?

## Objectif technique

Implementer `coverage(zone, a, b) -> bool` qui :
- Prend une grille 2D de booleens (`True` = obstacle, `False` = passable) ou de hauteurs de rack
- Retourne `True` si la ligne de `a` a `b` n'est bloquee par aucune cellule, `False` sinon
- Utilise l'algorithme de **Bresenham** pour tracer la ligne en O(max(dr, dc)) sans flottants
- Sort immediatement au premier obstacle rencontre (early exit)

## Consigne

```python
Point = tuple[int, int]
Grid = list[list[bool]]  # True = bloque

def coverage(zone: Grid, a: Point, b: Point) -> bool:
    """True si la couverture de a vers b est libre (a et b exclus du test)."""
```

Version 3D (extension) :
```python
HeightGrid = list[list[float]]

def coverage_3d(zone: HeightGrid, a: Point, b: Point, a_height: float, b_height: float) -> bool:
    """Vrai si aucune cellule intermediaire n'a une hauteur > a la ligne interpolee."""
```

## Etapes guidees

1. **Bresenham 2D** — implementer l'algorithme "octant zero" d'abord, puis generaliser via les signes et le swap dr/dc quand `abs(dc) < abs(dr)`.
2. **Ne pas allouer de liste** — iterer avec un generator ou directement dans la boucle. Allouer une liste pour chaque appel casse le cache et multiplie le GC.
3. **Exclure a et b** — le capteur et la cible ne bloquent pas leur propre vue.
4. **Early exit** — des qu'on touche un obstacle, `return False`.
5. **Benchmark** — mesurer avec `timeit` le cout d'un appel seul, puis le cout pour N*M = 40 000 appels.
6. **Extension hauteur** — interpoler lineairement la hauteur du capteur a la cible, comparer a la hauteur de rack de chaque cellule traversee.

## Criteres de reussite

- Tests unitaires passent (ligne horizontale, verticale, diagonale parfaite, avec/sans obstacle)
- 40 000 appels en **moins de 100 ms** sur une grille 200x200 en pur Python (benchmark fourni, mesure reference : ~51 ms)
- Pas d'allocation de liste intermediaire (verifie avec `tracemalloc`)
- Determinisme : couverture de `a -> b` == couverture de `b -> a` (la ligne doit etre symetrique)

**Note realisme** : en pur Python, la solution reference tourne autour de 50 ms sur une machine moderne. Le critere de 100 ms laisse de la marge pour une machine plus lente ou une implementation un peu moins optimisee. Pour descendre significativement sous les 20 ms, il faut **vectoriser avec numpy** (traiter les paires en batch) ou descendre en **C extension / Cython**. Voir "Pour aller plus loin".

## Piege classique

Bresenham standard **n'est pas symetrique** : `line(a, b)` peut passer par des cellules differentes de `line(b, a)` quand la pente est pile entre deux octants. Pour FleetSim c'est inacceptable (si capteur A voit B alors B doit etre vu de A). Deux solutions :
- Canonicaliser : toujours tracer du point avec `(r, c)` lexicographiquement plus petit vers l'autre
- Utiliser une variante symetrique (ex: algo de Xiaolin Wu sans anti-aliasing)

## Solution

Voir `solution/los.py` pour la correction avec la variante symetrique et le benchmark.

## Pour aller plus loin

- **Pre-computed visibility** — pour des cameras fixes, precalculer une carte de couverture et faire O(1) au runtime
- **Vectorisation numpy** — batch 1000 tests de couverture en un seul appel numpy (gain 10-100x si on traite des salves de detections)
- **Shadow casting** — algorithme de roguelike pour calculer d'un coup tout le champ de vision d'un capteur (utile pour planifier le placement de cameras)

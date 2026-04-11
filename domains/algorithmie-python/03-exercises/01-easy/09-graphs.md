# Exercices Easy — Graphs

---

## Exercice 1 : DFS sur grille — Number of Islands

### Objectif

Maitriser le pattern "grille 2D = graphe" en utilisant DFS pour "couler" (flood-fill) les cellules connectees.

### Consigne

Etant donne une grille 2D de `"1"` (terre) et `"0"` (eau), compte le nombre d'iles. Une ile est un groupe de `"1"` connectes horizontalement ou verticalement (pas en diagonale).

```python
def num_islands(grid: list[list[str]]) -> int:
    """
    Count the number of islands in the grid.
    """
    pass
```

### Tests

```python
grid1 = [
    ["1", "1", "1", "1", "0"],
    ["1", "1", "0", "1", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "0", "0", "0"],
]
assert num_islands([row[:] for row in grid1]) == 1

grid2 = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
]
assert num_islands([row[:] for row in grid2]) == 3

assert num_islands([]) == 0
assert num_islands([["0"]]) == 0
assert num_islands([["1"]]) == 1
```

### Criteres de reussite

- [ ] DFS ou BFS a partir de chaque cellule `"1"` non visitee
- [ ] Marque les cellules visitees (soit dans un set, soit en mutant la grille)
- [ ] Complexite O(R * C) temps et espace
- [ ] Tous les tests passent

---

## Exercice 2 : BFS — Shortest Path in Binary Matrix

### Objectif

Utiliser BFS pour trouver le plus court chemin dans une grille avec voisinage 8-directions.

### Consigne

Dans une grille `n x n` de 0 et 1, trouve la longueur du plus court chemin du coin haut-gauche `(0, 0)` au coin bas-droit `(n-1, n-1)`. Tu peux te deplacer dans les 8 directions (y compris diagonales), mais uniquement sur des cellules de valeur 0. Retourne `-1` si aucun chemin.

```python
def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    """
    Return the length of the shortest clear path, or -1 if none.
    Path length = number of cells visited (start and end included).
    """
    pass
```

### Tests

```python
assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1
assert shortest_path_binary_matrix([[0]]) == 1
assert shortest_path_binary_matrix([[1]]) == -1
```

### Criteres de reussite

- [ ] BFS avec deque (pas DFS)
- [ ] Gere le cas ou `(0, 0)` ou `(n-1, n-1)` est bloque par un 1
- [ ] 8 directions (delta_r, delta_c) incluant les diagonales
- [ ] Complexite O(n^2) temps et espace

---

## Exercice 3 : Topological Sort — Course Schedule

### Objectif

Appliquer le topological sort (Kahn ou DFS) pour detecter si un ensemble de cours avec prerequisites peut etre complete.

### Consigne

Il y a `num_courses` cours numerotes de 0 a `num_courses - 1`. Chaque prerequisite est un tuple `[a, b]` signifiant "pour prendre le cours a, il faut d'abord prendre b". Retourne `True` si on peut terminer tous les cours, `False` sinon.

```python
def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """
    Return True if all courses can be completed (no cyclic dependency).
    """
    pass
```

### Tests

```python
assert can_finish(2, [[1, 0]]) == True
assert can_finish(2, [[1, 0], [0, 1]]) == False   # Cycle
assert can_finish(4, [[1, 0], [2, 1], [3, 2]]) == True
assert can_finish(3, [[0, 1], [1, 2], [2, 0]]) == False  # Cycle
assert can_finish(1, []) == True
assert can_finish(5, []) == True
```

### Criteres de reussite

- [ ] Construit un graphe (dict ou list of list)
- [ ] Utilise Kahn (in-degree + queue) OU DFS avec 3 couleurs
- [ ] Detecte correctement les cycles
- [ ] Complexite O(V + E) temps et espace

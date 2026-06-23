# Exercices Hard — Graphs

> Dijkstra utilise `heapq`. Union-Find s'implemente avec path compression + union by rank. Les grilles 2D ponderees sont des graphes.

---

## Exercice 7 : Dijkstra — Network Delay Time

### Objectif

Appliquer Dijkstra a un graphe oriente pondere : trouver le temps pour qu'un signal atteigne TOUS les noeuds depuis une source. C'est le "shortest path to all nodes" classique, ou la reponse est le **max** des plus courtes distances.

### Consigne

Un reseau de `n` noeuds (numerotes `1..n`). `times[i] = [u, v, w]` est une arete orientee de `u` vers `v` avec un delai `w`. Un signal part du noeud `k`. Retourne le temps minimal pour que **tous** les noeuds recoivent le signal, ou `-1` si certains sont inatteignables.

```python
def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """
    Return the time for all n nodes to receive a signal from node k, or -1.
    """
    pass
```

**Indice** : Dijkstra depuis `k`. La reponse est `max(dist[node] for node in 1..n)` si tous sont atteints, sinon `-1` (un `inf` subsiste). Utilise un min-heap `(distance, node)` et skip les entrees obsoletes (`d > dist[u]`).

### Tests

```python
assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
assert network_delay_time([[1, 2, 1]], 2, 1) == 1
assert network_delay_time([[1, 2, 1]], 2, 2) == -1     # Node 1 unreachable from 2
assert network_delay_time([[1, 2, 1], [2, 3, 2], [1, 3, 4]], 3, 1) == 3   # 1->2->3 = 3 < 4
assert network_delay_time([[1, 2, 1], [2, 1, 3]], 2, 1) == 1
```

### Criteres de reussite

- [ ] Dijkstra avec min-heap `(dist, node)` et relaxation
- [ ] Skip des entrees obsoletes du heap (`d > dist[u]: continue`)
- [ ] La reponse est le max des distances ; `-1` si un noeud reste a l'infini
- [ ] Gere un noeud inatteignable, un graphe a deux noeuds
- [ ] Complexite O((V + E) log V) temps, O(V + E) espace

---

## Exercice 8 : Union-Find — Number of Connected Components

### Objectif

Implementer Union-Find from scratch (path compression + union by rank) et l'appliquer au comptage de composantes connexes. Union-Find bat BFS/DFS des qu'on a beaucoup de queries de connectivite.

### Consigne

Etant donne `n` noeuds (`0..n-1`) et une liste d'aretes non orientees `edges`, retourne le **nombre de composantes connexes** du graphe.

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    """
    Return the number of connected components using Union-Find.
    """
    pass
```

**Indice** : commence avec `n` composantes. Chaque `union(a, b)` qui fusionne deux composantes distinctes decremente le compteur de 1. Une union de noeuds deja connectes ne change rien (ne pas decrementer). Implemente `find` avec path compression et `union` by rank.

### Tests

```python
assert count_components(5, [[0, 1], [1, 2], [3, 4]]) == 2
assert count_components(5, [[0, 1], [1, 2], [2, 3], [3, 4]]) == 1
assert count_components(4, []) == 4                    # No edges = n components
assert count_components(1, []) == 1
assert count_components(3, [[0, 1], [1, 0], [0, 1]]) == 2   # Duplicate edges
assert count_components(6, [[0, 1], [2, 3], [4, 5]]) == 3
```

### Criteres de reussite

- [ ] `find` avec path compression
- [ ] `union` by rank (attacher le plus petit arbre au plus grand)
- [ ] Le compteur ne decremente que sur une fusion EFFECTIVE
- [ ] Gere les aretes dupliquees (pas de double comptage), zero arete, un noeud
- [ ] Complexite O(n + E * alpha(n)) ~ O(n + E) temps

---

## Exercice 9 : BFS multi-source 2D — Pacific Atlantic Water Flow

### Objectif

Probleme hard de grille ou la cle est d'**inverser le sens du flux** : au lieu de chercher d'ou l'eau peut atteindre les oceans, on part des oceans et on remonte. Deux parcours multi-source, puis intersection.

### Consigne

Une grille `heights` (R x C) ou `heights[r][c]` est l'altitude d'une cellule. L'ocean Pacifique borde les cotes **haut et gauche**, l'Atlantique borde les cotes **bas et droite**. L'eau s'ecoule d'une cellule vers une voisine (4-directions) d'altitude **inferieure ou egale**. Retourne la liste des coordonnees `[r, c]` depuis lesquelles l'eau peut atteindre **les deux** oceans.

```python
def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    Return all cells from which water can flow to BOTH oceans.
    Order of the returned cells does not matter.
    """
    pass
```

**Approche attendue** :
1. Inverse le probleme : depuis chaque ocean, fais un DFS/BFS qui "remonte" vers les cellules d'altitude **superieure ou egale** (flux inverse).
2. `pacific` = ensemble des cellules atteignables depuis les bords haut/gauche ; `atlantic` depuis bas/droite.
3. La reponse est l'**intersection** des deux ensembles.

### Tests

```python
def to_set(cells):
    return {tuple(c) for c in cells}

heights = [
    [1, 2, 2, 3, 5],
    [3, 2, 3, 4, 4],
    [2, 4, 5, 3, 1],
    [6, 7, 1, 4, 5],
    [5, 1, 1, 2, 4],
]
expected = {(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)}
assert to_set(pacific_atlantic(heights)) == expected

assert to_set(pacific_atlantic([[1]])) == {(0, 0)}    # Single cell touches both
assert to_set(pacific_atlantic([[2, 1], [1, 2]])) == {(0, 0), (0, 1), (1, 0), (1, 1)}
```

### Criteres de reussite

- [ ] Inversion du flux : parcours depuis les oceans vers l'amont (altitude >=)
- [ ] Deux ensembles (pacific, atlantic) construits par BFS/DFS multi-source
- [ ] Resultat = intersection des deux ensembles
- [ ] Gere une grille 1x1 et une grille 2x2
- [ ] Complexite O(R * C) temps et espace

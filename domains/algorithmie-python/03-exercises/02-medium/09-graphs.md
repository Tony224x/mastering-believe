# Exercices Medium — Graphs

---

## Exercice 4 : DFS + Hash Map — Clone Graph

### Objectif

Copier un graphe avec cycles sans boucler a l'infini : le pattern "map original → copie" qui sert aussi pour copy-list-with-random-pointer. Cible la confusion classique entre "visite" et "deja copie".

### Consigne

Etant donne un noeud d'un **graphe connexe non oriente** ou chaque noeud a une valeur unique et une liste `neighbors`, retourne une **copie profonde** du graphe (aucun noeud partage avec l'original).

```python
class Node:
    def __init__(self, val: int = 0, neighbors: list | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Node | None) -> Node | None:
    """
    Return a deep copy of the connected undirected graph.
    """
    pass
```

**Indice** : un dict `original → clone` joue les deux roles : marqueur de visite ET annuaire des copies. Cree le clone AVANT de recurser sur les voisins, sinon les cycles bouclent a l'infini.

### Tests

```python
def build_graph(adjacency: list[list[int]]) -> Node | None:
    """adjacency[i] = neighbors (1-indexed) of node i+1."""
    if not adjacency:
        return None
    nodes = [Node(i + 1) for i in range(len(adjacency))]
    for i, neighbors in enumerate(adjacency):
        nodes[i].neighbors = [nodes[j - 1] for j in neighbors]
    return nodes[0]

def check_clone(original: Node | None, clone: Node | None) -> bool:
    """Same structure/values, but ZERO shared node objects."""
    seen = {}
    def dfs(o, c):
        if o is None or c is None:
            return o is None and c is None
        if o is c:
            return False                    # Shared object: not a deep copy
        if o in seen:
            return seen[o] is c
        if o.val != c.val or len(o.neighbors) != len(c.neighbors):
            return False
        seen[o] = c
        return all(dfs(on, cn) for on, cn in zip(o.neighbors, c.neighbors))
    return dfs(original, clone)

# Square with both diagonals of adjacency: 1-2, 1-4, 2-3, 3-4
g = build_graph([[2, 4], [1, 3], [2, 4], [1, 3]])
assert check_clone(g, clone_graph(g))

assert clone_graph(None) is None

single = build_graph([[]])                  # One node, no neighbor
assert check_clone(single, clone_graph(single))

two = build_graph([[2], [1]])               # Two nodes, mutual edge (cycle of 2)
assert check_clone(two, clone_graph(two))
```

### Criteres de reussite

- [ ] Un seul dict `original → clone` sert de visited ET d'annuaire
- [ ] Le clone est enregistre dans le dict AVANT la recursion sur les voisins (sinon recursion infinie sur les cycles)
- [ ] Aucun objet partage entre original et copie (`check_clone` le verifie)
- [ ] O(V + E) temps, O(V) espace
- [ ] Tous les tests passent

---

## Exercice 5 : Union-Find — Number of Connected Components

### Objectif

Implementer Union-Find from scratch avec ses deux optimisations (path compression + union by rank), et savoir quand le preferer au DFS : edges en vrac, requetes incrementales.

### Consigne

Etant donne `n` noeuds (numerotes de 0 a n-1) et une liste d'aretes non orientees `edges`, retourne le **nombre de composantes connexes**.

**Contrainte** : utiliser Union-Find (pas DFS/BFS). Implemente `find` avec path compression et `union` by rank (ou by size). Compte les composantes en decrementant un compteur a chaque union **effective**.

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    """
    Return the number of connected components using Union-Find.
    """
    pass
```

### Tests

```python
assert count_components(5, [[0, 1], [1, 2], [3, 4]]) == 2
assert count_components(5, [[0, 1], [1, 2], [2, 3], [3, 4]]) == 1
assert count_components(4, []) == 4                      # No edges: n components
assert count_components(1, []) == 1
assert count_components(5, [[0, 1], [0, 2], [1, 2], [3, 4]]) == 2   # Redundant edge
assert count_components(3, [[0, 1], [0, 1]]) == 2        # Duplicate edge
```

### Criteres de reussite

- [ ] `find` applique la path compression (rattachement direct a la racine)
- [ ] `union` by rank/size — l'arbre reste plat
- [ ] Le compteur n'est decremente QUE si les deux racines etaient differentes (aretes redondantes ignorees)
- [ ] Complexite quasi O(E * α(n)) ≈ O(E) amortie — tu sais citer α (inverse d'Ackermann)
- [ ] Tous les tests passent

---

## Exercice 6 : Dijkstra — Network Delay Time

### Objectif

Implementer Dijkstra avec `heapq` et le pattern "lazy deletion" (on pousse des doublons, on ignore les entrees perimees au pop). C'est l'implementation Python standard en entretien.

### Consigne

On te donne `times`, une liste d'aretes orientees ponderees `(u, v, w)` : un signal envoye de `u` atteint `v` en `w` unites de temps. Les noeuds sont numerotes de 1 a `n`.

Un signal part du noeud `k`. Retourne le **temps minimum pour que TOUS les noeuds recoivent le signal**, ou `-1` si certains noeuds sont inatteignables.

```python
def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """
    Return the time for the signal from node k to reach all n nodes,
    or -1 if some node is unreachable. Dijkstra with a min-heap.
    """
    pass
```

**Pieges** :
- Skip au pop : si le noeud a deja une distance finalisee, ignorer l'entree (lazy deletion).
- La reponse est le **max** des distances finalisees (le dernier noeud atteint), pas leur somme.

### Tests

```python
assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
assert network_delay_time([[1, 2, 1]], 2, 1) == 1
assert network_delay_time([[1, 2, 1]], 2, 2) == -1       # Node 1 unreachable from 2
assert network_delay_time([], 1, 1) == 0                  # Single node, nothing to do
assert network_delay_time([[1, 2, 1], [2, 3, 2], [1, 3, 4]], 3, 1) == 3   # Path 1->2->3 beats direct edge
assert network_delay_time([[1, 2, 1], [2, 1, 3]], 2, 2) == 3
```

### Criteres de reussite

- [ ] Min-heap de tuples `(distance, node)` — la distance d'abord pour l'ordre du heap
- [ ] Lazy deletion : entree ignoree au pop si le noeud est deja finalise
- [ ] Chaque noeud est finalise UNE fois ; les voisins sont relaxes a ce moment-la
- [ ] O((V + E) log V) temps, O(V + E) espace
- [ ] Le resultat est le max des distances, -1 si un noeud manque
- [ ] Tous les tests passent

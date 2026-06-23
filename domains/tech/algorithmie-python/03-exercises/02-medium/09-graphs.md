# Exercices Medium — Graphs

> Les graphes sont representes en **adjacency list** (`dict` ou `list[list[int]]`). Les grilles 2D sont des graphes deguises : chaque cellule a 4 voisins.

---

## Exercice 4 : BFS sur grille — Rotting Oranges

### Objectif

Reconnaitre un probleme de grille comme un BFS **multi-source** : on lance la diffusion depuis TOUTES les sources en meme temps et on compte le nombre de "vagues" (= temps). Pattern omnipresent (fire spread, infection, flood fill chronometre).

### Consigne

Dans une grille `grid`, chaque cellule vaut `0` (vide), `1` (orange fraiche) ou `2` (orange pourrie). Chaque minute, toute orange fraiche **adjacente (4-directions)** a une orange pourrie devient pourrie. Retourne le nombre minimal de minutes pour que toutes les oranges fraiches pourrissent. Si c'est impossible (une orange fraiche reste isolee), retourne `-1`.

```python
def oranges_rotting(grid: list[list[int]]) -> int:
    """
    Return minutes until no fresh orange remains, or -1 if impossible.
    Multi-source BFS from all initially rotten oranges.
    """
    pass
```

**Indice** : initialise la queue avec **toutes** les oranges pourries au depart, et compte les fraiches restantes. A chaque "vague" de BFS (un `level_size`), pourris les voisins frais et incremente le temps. A la fin, s'il reste des fraiches, retourne `-1`.

### Tests

```python
assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1   # Bottom-left isolated
assert oranges_rotting([[0, 2]]) == 0                              # No fresh orange
assert oranges_rotting([[0]]) == 0
assert oranges_rotting([[1]]) == -1                                # Fresh, no source
assert oranges_rotting([[2, 2], [1, 1]]) == 1
```

### Criteres de reussite

- [ ] BFS multi-source : toutes les sources dans la queue au depart
- [ ] Le temps = nombre de vagues BFS (traitement par `level_size`)
- [ ] Compte les fraiches restantes pour detecter l'impossibilite (`-1`)
- [ ] Gere "aucune fraiche" → 0, et "fraiche isolee" → -1
- [ ] Complexite O(R * C) temps et espace

---

## Exercice 5 : Topological Sort — Course Schedule II

### Objectif

Produire un ORDRE topologique valide (pas juste detecter la faisabilite). C'est la version "II" du Course Schedule : Kahn's algorithm (BFS + in-degree) renvoie l'ordre, et un ordre vide signale un cycle.

### Consigne

Tu dois suivre `num_courses` cours (numerotes `0..num_courses-1`). `prerequisites[i] = [a, b]` signifie qu'il faut avoir fait `b` avant `a`. Retourne **un** ordre valide pour suivre tous les cours. S'il est impossible (cycle), retourne `[]`.

```python
def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Return a valid course ordering, or [] if impossible (cycle).
    """
    pass
```

**Indice** : construis le graphe `b -> a` et calcule l'in-degree de chaque cours. Kahn : enfile les cours d'in-degree 0, depile-les dans `order`, decremente l'in-degree des voisins, enfile ceux qui tombent a 0. Si `len(order) != num_courses`, il y a un cycle → `[]`.

### Tests

```python
def is_valid_order(order, num_courses, prerequisites):
    if len(order) != num_courses or len(set(order)) != num_courses:
        return False
    pos = {c: i for i, c in enumerate(order)}
    return all(pos[b] < pos[a] for a, b in prerequisites)   # b before a

assert find_order(2, [[1, 0]]) == [0, 1]
assert is_valid_order(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]), 4,
                      [[1, 0], [2, 0], [3, 1], [3, 2]])
assert find_order(1, []) == [0]
assert find_order(2, [[0, 1], [1, 0]]) == []          # Cycle
assert find_order(3, [[1, 0], [2, 1]]) == [0, 1, 2]
```

### Criteres de reussite

- [ ] Construit le graphe et l'in-degree correctement (sens `b -> a`)
- [ ] Kahn's algorithm : queue des in-degree 0, decrement, enfilement
- [ ] Detecte le cycle via `len(order) != num_courses` → `[]`
- [ ] Gere zero prerequis, un seul cours, cycle
- [ ] Complexite O(V + E) temps et espace

---

## Exercice 6 : Clone Graph — Copie profonde avec hash map

### Objectif

Cloner un graphe connexe (avec cycles) en O(V + E). Le piege : les cycles. La cle est une `dict {original: copie}` qui sert a la fois de "visited" et de table de correspondance pour relier les copies entre elles.

### Consigne

Un noeud de graphe non oriente est defini par `Node(val, neighbors)`. Etant donne une reference vers un noeud du graphe, retourne une **copie profonde** (deep copy) du graphe entier : memes valeurs, meme topologie, mais des objets distincts.

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: "Node | None") -> "Node | None":
    """
    Return a deep copy of the connected undirected graph.
    """
    pass
```

**Indice** : DFS ou BFS avec une `dict {original: clone}`. Quand tu visites un noeud, cree sa copie et enregistre-la AVANT de recurser sur ses voisins (sinon, un cycle te fait boucler a l'infini). Pour chaque voisin, clone-le (ou recupere son clone deja cree) et ajoute-le aux `neighbors` de la copie.

### Tests

```python
def build_graph(adj):
    """adj: 1-indexed adjacency list. Returns node 1 (or None if empty)."""
    if not adj:
        return None
    nodes = {i + 1: Node(i + 1) for i in range(len(adj))}
    for i, neighbors in enumerate(adj):
        nodes[i + 1].neighbors = [nodes[n] for n in neighbors]
    return nodes[1]

def signature(node):
    """BFS signature: {val: sorted neighbor vals}. Identical for isomorphic graphs."""
    if not node:
        return {}
    from collections import deque
    seen, q, sig = {node}, deque([node]), {}
    while q:
        cur = q.popleft()
        sig[cur.val] = sorted(n.val for n in cur.neighbors)
        for nb in cur.neighbors:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return sig

def check(adj):
    original = build_graph(adj)
    cloned = clone_graph(original)
    # Same shape...
    assert signature(original) == signature(cloned)
    # ...but distinct objects
    if original:
        assert cloned is not original

check([[2, 4], [1, 3], [2, 4], [1, 3]])   # 4-cycle
check([[2], [1]])                          # 2 nodes
check([[]])                                # single isolated node
assert clone_graph(None) is None           # empty graph
```

### Criteres de reussite

- [ ] Utilise une `dict {original: clone}` comme visited + correspondance
- [ ] Enregistre le clone AVANT de recurser (gere les cycles sans boucle infinie)
- [ ] Les copies sont des objets distincts (`cloned is not original`)
- [ ] Gere le graphe vide (None) et le noeud isole
- [ ] Complexite O(V + E) temps et espace

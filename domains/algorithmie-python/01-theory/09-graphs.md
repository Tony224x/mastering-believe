# Jour 9 — Graphs : DFS, BFS, Topological Sort, Dijkstra, Union-Find

> **Temps estime** : 60-75 min de lecture active | **Objectif** : maitriser les representations de graphes et les 6 algorithmes incontournables en entretien

---

## 1. Pourquoi les graphes sont l'etape "senior"

Un graphe generalise un arbre : **un arbre est un graphe connexe sans cycle**. En entretien, les graphes apparaissent des qu'on parle de :

- Reseaux sociaux, routes, dependances, workflows
- Grilles 2D (chaque cellule = noeud, ses 4 voisins = aretes)
- Course prerequisites, build systems (topological sort)
- Plus courts chemins (Dijkstra, BFS sans poids)
- Connectivite (union-find, number of islands)

**Le piege** : beaucoup de problemes de grilles (island count, word search, shortest path in a maze) sont en fait des problemes de graphes deguises. Savoir reconnaitre ca = gagner 10 min en entretien.

---

## 2. Representations

### Adjacency List (la plus utile en entretien)

```python
# graph[u] = liste des voisins de u
graph = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"],
}
# Ou avec des entiers :
graph = [[1, 2], [0, 3], [0, 3], [1, 2]]   # graph[i] = voisins de i

# Poids : (voisin, poids)
weighted = {
    "A": [("B", 2), ("C", 5)],
    "B": [("A", 2), ("D", 1)],
}
```

**Avantages** : O(V + E) espace, iteration efficace sur les voisins
**Inconvenients** : tester si une arete (u, v) existe prend O(deg(u))

### Adjacency Matrix

```python
# matrix[i][j] = 1 si arete, 0 sinon (ou poids pour graphe pondere)
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]
```

**Avantages** : O(1) pour tester une arete
**Inconvenients** : O(V^2) espace — genant pour graphes sparse

### Edge List

```python
edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
```

Compact mais il faut le convertir en adjacency list pour la plupart des algos.

### Convention : directed vs undirected

```python
# Undirected : ajouter les deux sens
def add_edge_undirected(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)

# Directed : un seul sens
def add_edge_directed(graph, u, v):
    graph[u].append(v)
```

---

## 3. Pattern 1 — DFS sur graphe

La grosse difference avec les arbres : **il peut y avoir des cycles**. Il faut un `visited` set pour eviter les boucles infinies.

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)                      # Visit
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
# Time: O(V + E), Space: O(V)
```

### Version iterative

```python
def dfs_iter(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return visited
```

### Application — Number of Islands (LeetCode 200)

```python
def num_islands(zone):
    if not zone:
        return 0
    rows, cols = len(zone), len(zone[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or zone[r][c] != "1":
            return
        zone[r][c] = "0"               # Mark visited by overwriting
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if zone[r][c] == "1":
                count += 1
                dfs(r, c)              # Sink the entire island
    return count
# Time: O(R * C), Space: O(R * C) worst case recursion
```

> **Astuce** : modifier la grille en place (`zone[r][c] = "0"`) evite un set visited supplementaire. A l'entretien, verifie avant si le candidat a le droit de muter l'input.

---

## 4. Pattern 2 — BFS sur graphe

Meme logique que pour les arbres, avec un visited set. BFS donne le **plus court chemin en nombre d'aretes** dans un graphe **non pondere**.

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)   # IMPORTANT: marquer a l'ajout, pas au pop
                queue.append(neighbor)
```

> **Piege** : marquer `visited` au moment du **pop** au lieu de l'ajout provoque des doublons dans la queue → BFS devient O(V^2).

### Shortest Path (unweighted)

```python
def shortest_path(graph, start, end):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1
```

---

## 5. Pattern 3 — Cycle Detection

### Undirected graph (DFS + parent tracking)

```python
def has_cycle_undirected(graph):
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:   # Back edge to non-parent = cycle
                return True
        return False
    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    return False
```

### Directed graph (DFS + 3 colors)

```python
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY             # Currently in the recursion stack
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True            # Back edge to node in progress = cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK            # Fully processed
        return False

    for node in graph:
        if color[node] == WHITE and dfs(node):
            return True
    return False
```

> **Cle** : GRAY = "dans la stack de recursion". Une arete vers un noeud GRAY = back edge = cycle.

---

## 6. Pattern 4 — Topological Sort

Ordonner les noeuds d'un DAG (Directed Acyclic Graph) tel que pour toute arete `u → v`, `u` apparaisse avant `v`. Usages : course schedule, build dependencies, task scheduling.

### Kahn's Algorithm (BFS avec in-degree)

```python
from collections import deque

def topo_sort_kahn(graph):
    # 1. Calculer l'in-degree de chaque noeud
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    # 2. Commencer avec les noeuds d'in-degree 0
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Si tous les noeuds n'ont pas ete visites → cycle detecte
    return result if len(result) == len(graph) else []
```

### DFS version (post-order)

```python
def topo_sort_dfs(graph):
    visited = set()
    result = []
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)            # Post-order
    for node in graph:
        if node not in visited:
            dfs(node)
    return result[::-1]                # Reverse: post-order gives reverse topo
```

### Application — Course Schedule

```python
def can_finish(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    return len(topo_sort_kahn(graph)) == num_courses
```

---

## 7. Pattern 5 — Dijkstra (shortest path, weighted, positive)

Plus court chemin d'un noeud source vers tous les autres dans un graphe pondere **avec poids positifs**.

```python
import heapq

def dijkstra(graph, start):
    # graph: {u: [(v, weight), ...]}
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]                # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                    # Entree obsolete, on skip
        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist
# Time: O((V + E) log V), Space: O(V)
```

> **Important** : Dijkstra ne fonctionne pas avec des poids **negatifs**. Pour ca, utiliser Bellman-Ford (O(V*E)).

---

## 8. Pattern 6 — Union-Find (Disjoint Set Union)

Structure pour gerer des ensembles disjoints avec 2 operations :
- `find(x)` : retourne le representant de l'ensemble contenant x
- `union(x, y)` : fusionne les ensembles de x et y

Avec path compression + union by rank : **O(alpha(n)) ~ O(1)** amorti.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))   # Chaque element est son propre parent
        self.rank = [0] * n            # Approximation de la hauteur

    def find(self, x):
        # Path compression: chaque noeud pointe directement vers la racine
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False               # Deja dans le meme ensemble
        # Union by rank: attacher le plus petit au plus grand
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Applications classiques

- **Number of Connected Components** : `n - nombre d'unions reussies`
- **Redundant Connection** : trouver l'arete qui cree un cycle dans un arbre
- **Accounts Merge** : grouper les comptes partageant un email

---

## 9. Decision Tree — Quel algo ?

```
Type de probleme ?
|
├── Parcourir / visiter tous les noeuds atteignables ?
│   ├── Ordre sans importance → DFS (plus simple a coder)
│   └── Par distance (BFS) ou plus court chemin non pondere → BFS
|
├── Plus court chemin ?
│   ├── Non pondere → BFS
│   ├── Poids positifs → Dijkstra
│   └── Poids negatifs autorises → Bellman-Ford
|
├── Detecter un cycle ?
│   ├── Undirected → DFS + parent tracking
│   └── Directed → DFS 3 couleurs, ou topo sort echoue
|
├── Ordonner des taches avec dependances ?
│   └── Topological sort (Kahn ou DFS post-order)
|
├── Connectivite / composantes / union de groupes ?
│   └── Union-Find (plus rapide que BFS pour des queries repetees)
|
└── Grille 2D avec voisins ?
    └── C'est un graphe — DFS ou BFS avec (row, col) comme noeud
```

**Signaux dans l'enonce** :

| Signal | Algo |
|--------|------|
| "shortest path" non pondere | BFS |
| "shortest path" pondere | Dijkstra |
| "islands", "regions", "connected cells" | DFS/BFS sur grille |
| "course schedule", "build order", "prerequisites" | Topological sort |
| "cycle", "can reach", "valid ordering" | DFS + cycle detection |
| "friends of friends", "connected groups" | Union-Find |
| "number of components" | Union-Find ou DFS |

---

## 10. Complexites

| Algo | Temps | Espace |
|------|-------|--------|
| DFS / BFS | O(V + E) | O(V) |
| Topological Sort (Kahn/DFS) | O(V + E) | O(V) |
| Dijkstra avec heap | O((V + E) log V) | O(V) |
| Bellman-Ford | O(V * E) | O(V) |
| Union-Find (find/union) | O(alpha(n)) ~ O(1) amorti | O(n) |
| Floyd-Warshall (toutes paires) | O(V^3) | O(V^2) |

---

## 11. Pieges courants

**Piege 1 — Oublier visited dans DFS**
Boucle infinie garantie sur un graphe avec cycles.

**Piege 2 — Marquer visited au pop dans BFS**
Cause des doublons dans la queue. TOUJOURS marquer a l'ajout.

**Piege 3 — Dijkstra sur poids negatifs**
Dijkstra suppose que, une fois qu'un noeud est extrait du heap avec une distance d, cette distance est definitive. Avec des poids negatifs, cette hypothese est fausse. Utiliser Bellman-Ford.

**Piege 4 — Confondre DAG et arbre**
Un arbre a exactement V-1 aretes. Un DAG peut en avoir plus. La topological sort fonctionne sur un DAG, pas sur un graphe avec cycles.

**Piege 5 — Grille non rectangulaire**
Certains problemes ont des grilles en escalier (jagged). Toujours verifier les bornes avec `r < len(zone)` ET `c < len(zone[r])`.

---

## 12. Flash Cards — Revision espacee

**Q1** : Pourquoi doit-on marquer un noeud comme visited au moment de l'ajouter a la queue BFS, et non au moment de le depop ?
> **R1** : Si on marque au pop, plusieurs ancetres peuvent pousser le meme noeud dans la queue avant qu'on ne l'extraie, generant des doublons et passant la complexite en O(V*E) ou pire. Marquer a l'ajout garantit qu'un noeud entre dans la queue une seule fois.

**Q2** : Quand utiliser Dijkstra plutot que BFS pour un plus court chemin ?
> **R2** : Des que le graphe est **pondere**. BFS suppose toutes les aretes de poids 1. Si les poids sont positifs et varies, utiliser Dijkstra. Si des poids sont negatifs, Bellman-Ford.

**Q3** : Comment fonctionne le 3-couleurs pour detecter un cycle dans un graphe dirige ?
> **R3** : WHITE = non visite, GRAY = dans la stack de recursion en cours, BLACK = completement traite. Si on rencontre une arete vers un noeud GRAY, c'est une back edge → cycle. Les aretes vers BLACK sont ok (elles traversent un sous-arbre deja fini).

**Q4** : Pourquoi Kahn's algorithm detecte-t-il un cycle automatiquement ?
> **R4** : Kahn n'enfile que les noeuds d'in-degree 0. Dans un graphe avec cycle, les noeuds du cycle ne voient jamais leur in-degree tomber a 0 (chacun depend d'un autre du cycle). Donc si `len(result) != len(graph)`, il y a un cycle.

**Q5** : Quelle est la complexite amortie de find/union avec path compression et union by rank ?
> **R5** : O(alpha(n)) ou alpha est la fonction inverse d'Ackermann. C'est < 5 pour tous les n realistes, donc on dit "O(1) amorti" en entretien.

---

## Resume — Key Takeaways

1. **Adjacency list** est la representation par defaut pour les entretiens
2. **DFS et BFS** sont les deux briques de base — tout le reste en derive
3. **Visited set** obligatoire pour eviter les boucles infinies
4. **Topological sort** : Kahn (BFS + in-degree) OU DFS post-order reverse
5. **Dijkstra** : heap + relaxation, UNIQUEMENT poids positifs
6. **Union-Find** : path compression + union by rank = O(1) amorti
7. **Les grilles 2D sont des graphes** : chaque cellule a 4 voisins, applique DFS/BFS

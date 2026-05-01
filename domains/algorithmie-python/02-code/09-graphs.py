"""
Day 9 — Graphs: DFS, BFS, Topological Sort, Dijkstra, Union-Find
Run: python domains/algorithmie-python/02-code/09-graphs.py
"""

import heapq
from collections import deque, defaultdict


# =============================================================================
# SAMPLE GRAPHS USED ACROSS DEMOS
# =============================================================================

def sample_undirected():
    """
        A --- B
        |     |
        C --- D --- E
    """
    return {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C", "E"],
        "E": ["D"],
    }


def sample_directed_dag():
    """
    A -> B -> D
     \        ^
      \-> C --/
    """
    return {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }


def sample_weighted():
    """
        (A)--2--(B)
         |       |
         5       1
         |       |
        (C)--3--(D)
    """
    return {
        "A": [("B", 2), ("C", 5)],
        "B": [("A", 2), ("D", 1)],
        "C": [("A", 5), ("D", 3)],
        "D": [("B", 1), ("C", 3)],
    }


# =============================================================================
# SECTION 1: DFS
# =============================================================================

def dfs_recursive(graph, start, visited=None, order=None):
    """
    Standard recursive DFS.
    WHY the visited set: graphs can have cycles, unlike trees. Without
    visited, dfs would loop forever on 'A -> B -> A -> B -> ...'.
    """
    if visited is None:
        visited = set()
        order = []
    visited.add(start)
    order.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, order)
    return order


def dfs_iterative(graph, start):
    """
    Iterative DFS with an explicit stack.
    Useful when recursion depth would exceed Python's default limit.
    """
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return order


# =============================================================================
# SECTION 2: BFS
# =============================================================================

def bfs(graph, start):
    """
    Standard BFS traversal.
    CRITICAL: we mark neighbors as visited WHEN WE ENQUEUE THEM, not when
    we pop them. Marking at pop causes duplicates in the queue.
    """
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)   # <-- mark here, not at pop
                queue.append(neighbor)
    return order


def shortest_path_unweighted(graph, start, end):
    """
    BFS gives shortest path in terms of NUMBER OF EDGES when the graph
    is unweighted. We store (node, distance) tuples in the queue.
    """
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


# =============================================================================
# SECTION 3: NUMBER OF ISLANDS (DFS on a zone)
# =============================================================================

def num_islands(zone):
    """
    Classic zone-as-graph problem.
    We iterate every cell; when we find a '1', we DFS to sink the entire
    island (turning its cells to '0') and increment the count by 1.

    Time : O(R * C)
    Space: O(R * C) worst case recursion stack (spiral island)
    """
    if not zone or not zone[0]:
        return 0
    rows, cols = len(zone), len(zone[0])
    count = 0

    def dfs(r, c):
        # Bounds + water check all in one
        if r < 0 or r >= rows or c < 0 or c >= cols or zone[r][c] != "1":
            return
        zone[r][c] = "0"                # Sink the cell so we don't revisit
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if zone[r][c] == "1":
                count += 1
                dfs(r, c)
    return count


# =============================================================================
# SECTION 4: CYCLE DETECTION
# =============================================================================

def has_cycle_undirected(graph):
    """
    DFS with parent tracking.
    In an undirected graph, the edge back to your immediate parent is
    expected (it's the same edge viewed from the other side). Any OTHER
    visited neighbor means we've closed a cycle.
    """
    visited = set()

    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True             # Back edge to non-parent = cycle
        return False

    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True
    return False


def has_cycle_directed(graph):
    """
    DFS 3-colors.
    WHITE = unseen, GRAY = currently on the recursion stack, BLACK = done.
    Any edge into a GRAY node means we've found a back edge = cycle.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE and dfs(node):
            return True
    return False


# =============================================================================
# SECTION 5: TOPOLOGICAL SORT
# =============================================================================

def topo_sort_kahn(graph):
    """
    Kahn's algorithm: BFS-style topological sort using in-degrees.
    Steps:
      1. Compute in-degree of every node.
      2. Start with all nodes of in-degree 0.
      3. Pop a node, append to result, decrement neighbors' in-degrees.
      4. If a neighbor's in-degree hits 0, enqueue it.
    If the final result misses nodes, the graph has a cycle.
    """
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    queue = deque([n for n in graph if in_degree[n] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't include every node, there must be a cycle
    return result if len(result) == len(graph) else []


def topo_sort_dfs(graph):
    """
    DFS post-order version.
    We add a node to the result AFTER visiting all its descendants.
    Reversing the post-order yields a valid topological ordering.
    """
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)              # Post-order

    for node in graph:
        if node not in visited:
            dfs(node)
    return order[::-1]


# =============================================================================
# SECTION 6: DIJKSTRA
# =============================================================================

def dijkstra(graph, start):
    """
    Single-source shortest path on a weighted graph (positive weights only).
    Invariant: when we pop (d, u) from the heap, d is the final shortest
    distance to u if d == dist[u]. Stale entries (d > dist[u]) are skipped.

    Time : O((V + E) log V)
    Space: O(V)
    """
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]                 # min-heap on distance

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                    # Stale entry, a shorter one was processed
        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist


# =============================================================================
# SECTION 7: UNION-FIND
# =============================================================================

class UnionFind:
    """
    Disjoint Set Union with path compression + union by rank.
    Amortized O(alpha(n)) per op — effectively O(1).
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n             # Number of distinct components

    def find(self, x):
        # Path compression: point x directly to the root while recursing up
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False                # Already in the same set
        # Union by rank: hang the shorter tree under the taller one
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


def count_components(n, edges):
    """
    Count connected components using Union-Find.
    """
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.components


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    g = sample_undirected()

    print("DFS recursive from A :", dfs_recursive(g, "A"))
    print("DFS iterative from A :", dfs_iterative(g, "A"))
    print("BFS from A           :", bfs(g, "A"))
    print("Shortest path A->E   :", shortest_path_unweighted(g, "A", "E"))

    print("\nNum islands:")
    zone = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    print(" ", num_islands(zone))       # Expected: 3

    print("\nUndirected cycle? :", has_cycle_undirected(g))
    dag = sample_directed_dag()
    print("Directed cycle (DAG)? :", has_cycle_directed(dag))
    cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
    print("Directed cycle (cyclic)? :", has_cycle_directed(cyclic))

    print("\nTopo sort (Kahn):", topo_sort_kahn(dag))
    print("Topo sort (DFS) :", topo_sort_dfs(dag))

    print("\nDijkstra from A:", dijkstra(sample_weighted(), "A"))

    print("\nUnion-Find demo:")
    uf = UnionFind(6)
    for u, v in [(0, 1), (1, 2), (3, 4)]:
        uf.union(u, v)
    print("  components:", uf.components)     # 3 -> {0,1,2}, {3,4}, {5}
    print("  0 ~ 2?    :", uf.connected(0, 2))
    print("  0 ~ 3?    :", uf.connected(0, 3))

"""
Solutions — Day 9: Graphs (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/09-graphs-hard.py

Each solution is numbered to match the exercise file (03-hard/09-graphs.md).
All solutions are verified with assertions at the end.
"""

import heapq
from collections import defaultdict, deque


# =============================================================================
# EXERCISE 7 (Hard): Network Delay Time — Dijkstra
# =============================================================================

def network_delay_time(times, n, k):
    """
    Dijkstra from node k; answer is the max shortest-distance over all nodes,
    or -1 if some node stays unreachable (infinite).

    Time: O((V + E) log V), Space: O(V + E)
    """
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {node: float('inf') for node in range(1, n + 1)}
    dist[k] = 0
    heap = [(0, k)]                     # (distance, node)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                   # Stale entry, skip
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    longest = max(dist.values())
    return longest if longest < float('inf') else -1


def test_exercise_7():
    print("\nExercise 7: Network Delay Time")

    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1
    assert network_delay_time([[1, 2, 1], [2, 3, 2], [1, 3, 4]], 3, 1) == 3
    assert network_delay_time([[1, 2, 1], [2, 1, 3]], 2, 1) == 1

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): Number of Connected Components — Union-Find
# =============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n                 # Number of disjoint components

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False               # Already connected: no merge
        if self.rank[px] < self.rank[py]:
            px, py = py, px            # Attach smaller tree under larger
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1                # One fewer component
        return True


def count_components(n, edges):
    """
    Start with n components; each effective union reduces the count by 1.

    Time: O(n + E * alpha(n)), Space: O(n)
    """
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    return uf.count


def test_exercise_8():
    print("\nExercise 8: Number of Connected Components")

    assert count_components(5, [[0, 1], [1, 2], [3, 4]]) == 2
    assert count_components(5, [[0, 1], [1, 2], [2, 3], [3, 4]]) == 1
    assert count_components(4, []) == 4
    assert count_components(1, []) == 1
    assert count_components(3, [[0, 1], [1, 0], [0, 1]]) == 2
    assert count_components(6, [[0, 1], [2, 3], [4, 5]]) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Pacific Atlantic Water Flow — Reversed multi-source BFS
# =============================================================================

def pacific_atlantic(heights):
    """
    Reverse the flow: from each ocean's border cells, BFS uphill (>= height).
    The answer is the intersection of cells reachable from BOTH oceans.

    Time: O(R * C), Space: O(R * C)
    """
    if not heights or not heights[0]:
        return []
    rows, cols = len(heights), len(heights[0])

    def bfs(starts):
        reachable = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                        (nr, nc) not in reachable and
                        heights[nr][nc] >= heights[r][c]):   # Uphill (reverse flow)
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
        return reachable

    pacific_starts = [(0, c) for c in range(cols)] + [(r, 0) for r in range(rows)]
    atlantic_starts = ([(rows - 1, c) for c in range(cols)] +
                       [(r, cols - 1) for r in range(rows)])

    pacific = bfs(pacific_starts)
    atlantic = bfs(atlantic_starts)
    return [[r, c] for (r, c) in pacific & atlantic]


def test_exercise_9():
    print("\nExercise 9: Pacific Atlantic Water Flow")

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

    assert to_set(pacific_atlantic([[1]])) == {(0, 0)}
    assert to_set(pacific_atlantic([[2, 1], [1, 2]])) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 9: Graphs (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

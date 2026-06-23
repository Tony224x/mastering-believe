"""
Solutions — Day 9: Graphs (MEDIUM)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/09-graphs-medium.py

Each solution is numbered to match the exercise file (02-medium/09-graphs.md).
All solutions are verified with assertions at the end.
"""

from collections import deque


# =============================================================================
# EXERCISE 4 (Medium): Rotting Oranges — Multi-source BFS
# =============================================================================

def oranges_rotting(grid):
    """
    Multi-source BFS starting from every rotten orange at once.
    Each BFS wave = one minute. Track remaining fresh oranges to detect -1.

    Time: O(R * C), Space: O(R * C)
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))   # All initial sources
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0                       # Nothing to rot

    minutes = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue and fresh > 0:
        minutes += 1
        for _ in range(len(queue)):    # Process one wave
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2   # Rot it now (mark visited)
                    fresh -= 1
                    queue.append((nr, nc))
    return minutes if fresh == 0 else -1


def test_exercise_4():
    print("\nExercise 4: Rotting Oranges")

    assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert oranges_rotting([[0, 2]]) == 0
    assert oranges_rotting([[0]]) == 0
    assert oranges_rotting([[1]]) == -1
    assert oranges_rotting([[2, 2], [1, 1]]) == 1

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Course Schedule II — Topological sort (Kahn)
# =============================================================================

def find_order(num_courses, prerequisites):
    """
    Kahn's algorithm: enqueue in-degree-0 courses, pop into the order, decrement
    neighbors. If the order misses some course, a cycle exists -> [].

    Time: O(V + E), Space: O(V + E)
    """
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses
    for course, prereq in prerequisites:
        graph[prereq].append(course)   # prereq -> course (b before a)
        in_degree[course] += 1

    queue = deque(c for c in range(num_courses) if in_degree[c] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return order if len(order) == num_courses else []


def test_exercise_5():
    print("\nExercise 5: Course Schedule II")

    def is_valid_order(order, num_courses, prerequisites):
        if len(order) != num_courses or len(set(order)) != num_courses:
            return False
        pos = {c: i for i, c in enumerate(order)}
        return all(pos[b] < pos[a] for a, b in prerequisites)

    assert find_order(2, [[1, 0]]) == [0, 1]
    assert is_valid_order(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]), 4,
                          [[1, 0], [2, 0], [3, 1], [3, 2]])
    assert find_order(1, []) == [0]
    assert find_order(2, [[0, 1], [1, 0]]) == []
    assert find_order(3, [[1, 0], [2, 1]]) == [0, 1, 2]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Clone Graph — Deep copy with hash map
# =============================================================================

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    """
    DFS with a {original: clone} map serving as both visited set and lookup.
    Register the clone BEFORE recursing so cycles don't loop forever.

    Time: O(V + E), Space: O(V)
    """
    if not node:
        return None
    clones = {}

    def dfs(orig):
        if orig in clones:
            return clones[orig]
        copy = Node(orig.val)
        clones[orig] = copy            # Register BEFORE recursing (cycle safety)
        for nb in orig.neighbors:
            copy.neighbors.append(dfs(nb))
        return copy

    return dfs(node)


def test_exercise_6():
    print("\nExercise 6: Clone Graph")

    def build_graph(adj):
        if not adj:
            return None
        nodes = {i + 1: Node(i + 1) for i in range(len(adj))}
        for i, neighbors in enumerate(adj):
            nodes[i + 1].neighbors = [nodes[n] for n in neighbors]
        return nodes[1]

    def signature(node):
        if not node:
            return {}
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
        assert signature(original) == signature(cloned)
        if original:
            assert cloned is not original
            # Verify deep copy: no shared node objects between graphs
            orig_nodes, clone_nodes = set(), set()
            for n, store in ((original, orig_nodes), (cloned, clone_nodes)):
                seen, q = {id(n)}, deque([n])
                store.add(id(n))
                while q:
                    cur = q.popleft()
                    for nb in cur.neighbors:
                        if id(nb) not in seen:
                            seen.add(id(nb))
                            store.add(id(nb))
                            q.append(nb)
            assert orig_nodes.isdisjoint(clone_nodes)

    check([[2, 4], [1, 3], [2, 4], [1, 3]])
    check([[2], [1]])
    check([[]])
    assert clone_graph(None) is None

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 9: Graphs (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

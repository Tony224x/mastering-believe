"""
Solutions — Day 9 Graphs (easy, medium and hard exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/09-graphs.py
"""

import heapq
from collections import defaultdict, deque


# =============================================================================
# Exercise 1: Number of Islands
# =============================================================================

def num_islands(zone):
    """
    DFS flood-fill.
    For each unvisited '1' cell, increment the counter and sink the entire
    island by DFS (mark every connected '1' as '0').

    Time : O(R * C) — each cell visited at most once
    Space: O(R * C) recursion worst case
    """
    if not zone or not zone[0]:
        return 0
    rows, cols = len(zone), len(zone[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or zone[r][c] != "1":
            return
        zone[r][c] = "0"                # Mark visited by mutation
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
# Exercise 2: Shortest Path in Binary Matrix
# =============================================================================

def shortest_path_binary_matrix(zone):
    """
    BFS from (0,0) with 8-directional moves.
    We return the path length (number of cells). BFS guarantees that the
    first time we reach the target, it's via the shortest path.

    Time : O(n^2)
    Space: O(n^2)
    """
    n = len(zone)
    if n == 0 or zone[0][0] != 0 or zone[n - 1][n - 1] != 0:
        return -1                        # Blocked start or end

    # 8 directions: N, S, E, W, NE, NW, SE, SW
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    visited = {(0, 0)}
    queue = deque([(0, 0, 1)])           # (row, col, path_length)

    while queue:
        r, c, length = queue.popleft()
        if r == n - 1 and c == n - 1:
            return length
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and
                    zone[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))    # Mark at enqueue time
                queue.append((nr, nc, length + 1))

    return -1


# =============================================================================
# Exercise 3: Course Schedule
# =============================================================================

def can_finish(num_courses, prerequisites):
    """
    Topological sort via Kahn's algorithm.
    Build the graph: prereq -> course. If we can't process every course,
    there's a cycle (courses in the cycle never reach in-degree 0).

    Time : O(V + E)
    Space: O(V + E)
    """
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with all courses that have no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    processed = 0

    while queue:
        node = queue.popleft()
        processed += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return processed == num_courses


# Alternative DFS version with 3 colors
def can_finish_dfs(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * num_courses

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return False             # Back edge = cycle
            if color[neighbor] == WHITE and not dfs(neighbor):
                return False
        color[node] = BLACK
        return True

    for i in range(num_courses):
        if color[i] == WHITE and not dfs(i):
            return False
    return True



# =============================================================================
# Exercise 4 (Medium): Clone Graph
# =============================================================================

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    """
    DFS with a single dict playing two roles.

    THE DICT original -> clone IS:
    - the visited marker (a node in the dict has been reached), AND
    - the directory used to wire neighbor pointers to existing clones.

    ORDER MATTERS:
    - Register the clone in the dict BEFORE recursing on neighbors.
      Otherwise a cycle (A - B - A) recurses forever: cloning A clones B,
      which tries to clone A again, which clones B again...

    Time: O(V + E), Space: O(V)
    """
    if node is None:
        return None

    clones = {}                         # original node -> its clone

    def dfs(original):
        if original in clones:
            return clones[original]     # Already cloned: reuse, do not recurse

        copy = Node(original.val)
        clones[original] = copy         # Register BEFORE visiting neighbors
        copy.neighbors = [dfs(n) for n in original.neighbors]
        return copy

    return dfs(node)


# =============================================================================
# Exercise 5 (Medium): Number of Connected Components (Union-Find)
# =============================================================================

def count_components(n, edges):
    """
    Union-Find with both standard optimizations.

    PATH COMPRESSION (in find):
    - While walking up to the root, reattach each visited node directly
      to the root: future finds become near-O(1).

    UNION BY RANK (in union):
    - Always hang the shallower tree under the deeper one, so tree height
      stays O(log n) even before compression kicks in.

    Together: amortized O(alpha(n)) per operation, where alpha is the
    inverse Ackermann function (<= 4 for any realistic n).

    COMPONENT COUNTING:
    - Start at n; decrement ONLY when a union actually merges two
      different roots (redundant/duplicate edges change nothing).
    """
    parent = list(range(n))
    rank = [0] * n
    components = n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:        # Path compression pass
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        nonlocal components
        ra, rb = find(a), find(b)
        if ra == rb:
            return                      # Already connected: no merge
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra             # ra is the deeper (or equal) tree
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        components -= 1                 # One real merge = one fewer component

    for a, b in edges:
        union(a, b)

    return components


# =============================================================================
# Exercise 6 (Medium): Network Delay Time (Dijkstra)
# =============================================================================

def network_delay_time(times, n, k):
    """
    Dijkstra with heapq and lazy deletion.

    LAZY DELETION:
    - heapq cannot decrease-key, so we push duplicate entries when a
      shorter path to a node is found. At pop time, an entry whose node
      is already finalized is simply skipped.

    WHY THE ANSWER IS A MAX:
    - Each node receives the signal at its own shortest distance; ALL
      nodes have it once the farthest one does -> max of distances.

    Time: O((V + E) log V), Space: O(V + E)
    """
    graph = defaultdict(list)           # u -> [(v, w), ...]
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {}                           # node -> finalized shortest distance
    heap = [(0, k)]                     # (distance, node) — distance first!

    while heap:
        d, node = heapq.heappop(heap)
        if node in dist:
            continue                    # Stale entry (lazy deletion): skip
        dist[node] = d                  # First pop = shortest distance (weights >= 0)
        for neighbor, w in graph[node]:
            if neighbor not in dist:
                heapq.heappush(heap, (d + w, neighbor))

    if len(dist) < n:
        return -1                       # Some node never reached
    return max(dist.values())


# =============================================================================
# Exercise 7 (Hard): Word Ladder (BFS on an implicit graph)
# =============================================================================

def ladder_length(begin_word, end_word, word_list):
    """
    BFS over wildcard buckets instead of explicit edges.

    WHY BUCKETS:
    - Comparing all word pairs costs O(N^2 * L). Instead, each word of
      length L belongs to L wildcard patterns ("hot" -> "*ot", "h*t",
      "ho*"). Two words are neighbors iff they share a pattern.
      Building + consuming buckets is O(N * L^2) (L patterns of length L
      per word).

    WHY BFS GIVES THE SHORTEST SEQUENCE:
    - All edges have weight 1; BFS explores by increasing distance, so
      the first time end_word is dequeued, its level is minimal.

    BUCKET EMPTYING:
    - After expanding a bucket once, clear it. Every later word reaching
      the same bucket would only rediscover visited words.
    """
    words = set(word_list)
    if end_word not in words:
        return 0                        # Target unreachable by definition

    L = len(begin_word)
    buckets = defaultdict(list)         # pattern -> words matching it
    for word in words | {begin_word}:
        for i in range(L):
            buckets[word[:i] + "*" + word[i + 1:]].append(word)

    visited = {begin_word}
    queue = deque([(begin_word, 1)])    # (word, sequence length so far)

    while queue:
        word, length = queue.popleft()
        if word == end_word:
            return length
        for i in range(L):
            pattern = word[:i] + "*" + word[i + 1:]
            for neighbor in buckets[pattern]:
                if neighbor not in visited:
                    visited.add(neighbor)       # Mark at enqueue
                    queue.append((neighbor, length + 1))
            buckets[pattern] = []       # Bucket consumed: never re-scan it

    return 0


# =============================================================================
# Exercise 8 (Hard): Alien Dictionary (topological sort)
# =============================================================================

def alien_order(words):
    """
    Build a constraint graph from adjacent word pairs, then Kahn's algorithm.

    GRAPH CONSTRUCTION:
    - For each ADJACENT pair, only the FIRST differing character carries
      information: c1 -> c2 (c1 comes before c2). Later characters are
      unconstrained by this pair.
    - Use sets for edges: duplicate edges would inflate in-degrees and
      break Kahn's termination check.

    TWO DISTINCT INVALID CASES:
    1. Inverted prefix (["abc", "ab"]): no differing char but the first
       word is longer — impossible in any lexicographic order. This is
       NOT a cycle; it must be caught during construction.
    2. Contradictory constraints: Kahn finishes with unprocessed letters
       (a cycle) -> "".

    Time: O(C) where C = total length of all words.
    """
    # Every letter that appears must exist in the graph, even with no edges
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in graph}

    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:         # Avoid duplicate edges
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break                           # Only the first difference counts
        else:
            # No differing char: w2 must not be a strict prefix... w1 must
            # not be LONGER than w2 (["abc", "ab"] is invalid)
            if len(w1) > len(w2):
                return ""

    # Kahn's algorithm: repeatedly consume letters with in-degree 0
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for nxt in graph[c]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # Letters left unconsumed => cycle => no valid order
    return "".join(order) if len(order) == len(graph) else ""


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
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
    print("Exercise 1 (num_islands): OK")

    # -- Exercise 2 --
    assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
    assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
    assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1
    assert shortest_path_binary_matrix([[0]]) == 1
    assert shortest_path_binary_matrix([[1]]) == -1
    print("Exercise 2 (shortest_path_binary_matrix): OK")

    # -- Exercise 3 --
    assert can_finish(2, [[1, 0]]) == True
    assert can_finish(2, [[1, 0], [0, 1]]) == False
    assert can_finish(4, [[1, 0], [2, 1], [3, 2]]) == True
    assert can_finish(3, [[0, 1], [1, 2], [2, 0]]) == False
    assert can_finish(1, []) == True
    assert can_finish(5, []) == True
    # DFS variant matches
    assert can_finish_dfs(3, [[0, 1], [1, 2], [2, 0]]) == False
    assert can_finish_dfs(4, [[1, 0], [2, 1], [3, 2]]) == True
    print("Exercise 3 (can_finish): OK")

    # -- Exercise 4 --
    def build_graph(adjacency):
        if not adjacency:
            return None
        nodes = [Node(i + 1) for i in range(len(adjacency))]
        for i, neighbors in enumerate(adjacency):
            nodes[i].neighbors = [nodes[j - 1] for j in neighbors]
        return nodes[0]

    def check_clone(original, clone):
        seen = {}

        def dfs(o, c):
            if o is None or c is None:
                return o is None and c is None
            if o is c:
                return False            # Shared object: not a deep copy
            if o in seen:
                return seen[o] is c
            if o.val != c.val or len(o.neighbors) != len(c.neighbors):
                return False
            seen[o] = c
            return all(dfs(on, cn) for on, cn in zip(o.neighbors, c.neighbors))

        return dfs(original, clone)

    g = build_graph([[2, 4], [1, 3], [2, 4], [1, 3]])
    assert check_clone(g, clone_graph(g))
    assert clone_graph(None) is None
    single = build_graph([[]])
    assert check_clone(single, clone_graph(single))
    two = build_graph([[2], [1]])
    assert check_clone(two, clone_graph(two))
    print("Exercise 4 (clone_graph): OK")

    # -- Exercise 5 --
    assert count_components(5, [[0, 1], [1, 2], [3, 4]]) == 2
    assert count_components(5, [[0, 1], [1, 2], [2, 3], [3, 4]]) == 1
    assert count_components(4, []) == 4
    assert count_components(1, []) == 1
    assert count_components(5, [[0, 1], [0, 2], [1, 2], [3, 4]]) == 2
    assert count_components(3, [[0, 1], [0, 1]]) == 2
    print("Exercise 5 (count_components): OK")

    # -- Exercise 6 --
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1
    assert network_delay_time([], 1, 1) == 0
    assert network_delay_time([[1, 2, 1], [2, 3, 2], [1, 3, 4]], 3, 1) == 3
    assert network_delay_time([[1, 2, 1], [2, 1, 3]], 2, 2) == 3
    print("Exercise 6 (network_delay_time): OK")

    # -- Exercise 7 --
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
    assert ladder_length("a", "c", ["a", "b", "c"]) == 2
    assert ladder_length("hot", "dog", ["hot", "dog"]) == 0
    assert ladder_length("hit", "hit", ["hit"]) == 1
    assert ladder_length("ab", "cd", ["ad", "cd"]) == 3
    print("Exercise 7 (ladder_length): OK")

    # -- Exercise 8 --
    assert alien_order(["wrt", "wrf", "er", "ett", "rftt"]) == "wertf"
    assert alien_order(["z", "x"]) == "zx"
    assert alien_order(["z", "x", "z"]) == ""               # Cycle
    assert alien_order(["abc", "ab"]) == ""                  # Inverted prefix
    assert alien_order(["ab", "abc"]) != ""
    result = alien_order(["zyx"])
    assert sorted(result) == ["x", "y", "z"]
    result = alien_order(["ac", "ab", "zc", "zb"])
    assert set(result) == {"a", "b", "c", "z"}
    assert result.index("c") < result.index("b")
    print("Exercise 8 (alien_order): OK")

    print("\nAll Day 9 solutions pass!")

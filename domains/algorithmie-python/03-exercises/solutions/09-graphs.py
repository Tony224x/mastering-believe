"""
Solutions — Day 9 Graphs (easy exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/09-graphs.py
"""

from collections import deque


# =============================================================================
# Exercise 1: Number of Islands
# =============================================================================

def num_islands(grid):
    """
    DFS flood-fill.
    For each unvisited '1' cell, increment the counter and sink the entire
    island by DFS (mark every connected '1' as '0').

    Time : O(R * C) — each cell visited at most once
    Space: O(R * C) recursion worst case
    """
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"                # Mark visited by mutation
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count


# =============================================================================
# Exercise 2: Shortest Path in Binary Matrix
# =============================================================================

def shortest_path_binary_matrix(grid):
    """
    BFS from (0,0) with 8-directional moves.
    We return the path length (number of cells). BFS guarantees that the
    first time we reach the target, it's via the shortest path.

    Time : O(n^2)
    Space: O(n^2)
    """
    n = len(grid)
    if n == 0 or grid[0][0] != 0 or grid[n - 1][n - 1] != 0:
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
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
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

    print("\nAll Day 9 solutions pass!")

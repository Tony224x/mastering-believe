"""
Solutions — Day 4: Stacks & Queues
Run: python domains/algorithmie-python/03-exercises/solutions/04-stacks-queues.py

Each solution is numbered to match the exercise file.
All solutions are verified with assertions at the end.
"""

from collections import deque


# =============================================================================
# EXERCISE 1 (Easy): Valid Parentheses
# =============================================================================

def is_valid(s: str) -> bool:
    """
    Stack-based matching.

    APPROACH:
    - Push every opener onto the stack.
    - When we see a closer, the top of the stack MUST be the matching opener.
      If not (or stack is empty), the string is invalid.
    - At the end, the stack must be empty (no unmatched openers).

    WHY A STACK:
    - The rule "innermost pair closes first" is exactly LIFO.

    Time: O(n) — single pass
    Space: O(n) — worst case all openers like "((((("
    """
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            # Empty stack OR mismatch -> invalid
            if not stack or stack.pop() != pairs[c]:
                return False
        # Any other character is ignored here (not expected per the problem)
    # Any leftover opener means unmatched
    return not stack


def test_exercise_1():
    print("\nExercise 1: Valid Parentheses")

    assert is_valid("()") == True
    assert is_valid("()[]{}") == True
    assert is_valid("(]") == False
    assert is_valid("([)]") == False
    assert is_valid("{[]}") == True
    assert is_valid("") == True
    assert is_valid("(((") == False
    assert is_valid(")(") == False
    assert is_valid("((()))") == True

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 2 (Easy): Number of Islands (BFS)
# =============================================================================

def num_islands(grid):
    """
    Multi-source BFS over a grid.

    APPROACH:
    - Scan every cell. When we find an unvisited '1', that's a new island.
    - Run BFS from that cell to mark every connected '1' as visited.
    - Increment the island counter once per BFS launch.

    WHY A QUEUE:
    - We need to explore every cell of one island before moving on.
    - BFS with a queue ensures we flood-fill breadth-first. A stack (DFS) also
      works here since we only care about connectivity, not shortest path.

    VISITED MARKING:
    - We mutate the grid to '0' when enqueuing to avoid a second visit.
    - Marking at enqueue (not dequeue) is critical: otherwise the same cell
      could be added to the queue multiple times, blowing up the cost.

    Time: O(rows * cols) — every cell enqueued at most once
    Space: O(rows * cols) — worst case the queue holds one full row/column
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "1":
                continue

            # Found a new island — BFS from here
            islands += 1
            queue = deque([(r, c)])
            grid[r][c] = "0"                # Mark at enqueue

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and grid[nr][nc] == "1"):
                        grid[nr][nc] = "0"   # Mark BEFORE enqueue
                        queue.append((nr, nc))

    return islands


def test_exercise_2():
    print("\nExercise 2: Number of Islands")

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

    assert num_islands([["0"]]) == 0
    assert num_islands([["1"]]) == 1
    assert num_islands([]) == 0

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 3 (Easy): Next Greater Element I
# =============================================================================

def next_greater_element(nums1, nums2):
    """
    Monotonic decreasing stack on nums2, then lookup for nums1.

    APPROACH:
    - Walk nums2 left to right with a DECREASING stack of values.
    - When the current value is greater than the stack top, the stack top's
      "next greater" is the current value. Pop and record in a dict.
    - After the loop, anything still in the stack has no next greater -> -1.
    - For each element of nums1, look up the dict in O(1).

    WHY STORE VALUES (not indices):
    - nums2 has distinct values per the problem, and we only need the VALUE
      of the next greater element, not its position. Storing values avoids a
      second indirection.

    Time: O(n + m) where n = len(nums2), m = len(nums1)
    Space: O(n)
    """
    next_greater = {}       # value -> its next greater in nums2
    stack = []              # Decreasing stack of VALUES from nums2

    for v in nums2:
        # Every stack entry smaller than v has its next greater resolved
        while stack and stack[-1] < v:
            next_greater[stack.pop()] = v
        stack.append(v)

    # Anything left has no greater element to the right
    for v in stack:
        next_greater[v] = -1

    # Second pass: lookup for nums1
    return [next_greater[v] for v in nums1]


def test_exercise_3():
    print("\nExercise 3: Next Greater Element I")

    assert next_greater_element([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
    assert next_greater_element([2, 4], [1, 2, 3, 4]) == [3, -1]
    assert next_greater_element([1], [1]) == [-1]
    assert next_greater_element([1, 3, 5, 2, 4], [6, 5, 4, 3, 2, 1, 7]) == [7, 7, 7, 7, 7]
    assert next_greater_element([5], [4, 5, 6]) == [6]

    print("  PASS — all test cases")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    print("\nAll Day 4 exercise solutions passed.")

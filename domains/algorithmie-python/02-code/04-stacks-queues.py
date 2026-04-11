"""
Day 4 — Stacks & Queues: LIFO, FIFO, Monotonic Stack & BFS Foundations
Run: python domains/algorithmie-python/02-code/04-stacks-queues.py

Each section shows a canonical example of one pattern. The goal: feel WHY
each data structure is the right tool for the problem. Comments explain the
invariants, not the syntax.
"""

import time
from collections import deque


def timed(func):
    """Decorator that prints execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


# =============================================================================
# SECTION 1: STACK — Valid Parentheses (matching pairs)
# =============================================================================

def is_valid_parentheses(s: str) -> bool:
    """
    LIFO matching: the most recently opened bracket must be the first closed.

    INTUITION:
    - Every closing bracket must match the LAST unmatched opener.
    - A stack naturally models "last unmatched" thanks to LIFO.
    - Empty stack at the end = every opener was matched.

    Time: O(n), Space: O(n) in the worst case (all openers)
    """
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)                       # Opener goes on the stack
        elif c in ")]}":
            # Two failure modes: empty stack, or mismatch with top
            if not stack or stack.pop() != pairs[c]:
                return False
    # Any remaining opener means unmatched
    return not stack


# =============================================================================
# SECTION 2: MONOTONIC STACK — Daily Temperatures (next greater)
# =============================================================================

def daily_temperatures_brute(temps):
    """
    Brute force: for each day, scan forward for a warmer day.

    Time: O(n^2) — quadratic when temperatures are sorted decreasing
    Space: O(n)
    """
    n = len(temps)
    result = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if temps[j] > temps[i]:
                result[i] = j - i
                break
    return result


def daily_temperatures(temps):
    """
    Monotonic DECREASING stack of indices.

    INVARIANT:
    - The stack always holds INDICES whose temperatures form a decreasing
      sequence from bottom to top. This is exactly the set of days still
      "waiting" for a warmer day.
    - When we encounter a warmer day, we pop every waiting day that it resolves.

    WHY O(n) DESPITE THE NESTED WHILE:
    - Each index is pushed at most once and popped at most once.
    - Total push + pop operations across the entire run is bounded by 2n.

    Time: O(n), Space: O(n)
    """
    n = len(temps)
    result = [0] * n
    stack = []      # Stack of indices, NOT values

    for i, t in enumerate(temps):
        # Resolve every waiting day whose temperature is strictly lower
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            result[j] = i - j                 # Distance in days
        stack.append(i)

    # Any index still in the stack has no warmer day — result[j] stays 0
    return result


# =============================================================================
# SECTION 3: MONOTONIC STACK — Largest Rectangle in Histogram
# =============================================================================

def largest_rectangle(heights):
    """
    Monotonic INCREASING stack to compute, for each bar, the largest rectangle
    where that bar is the SMALLEST height.

    KEY INSIGHT:
    - For each bar h[i], we need the nearest smaller bar on the LEFT and RIGHT.
    - The monotonic stack gives both in amortized O(1).
    - Sentinel 0 at the end flushes the stack so we process every bar.

    Time: O(n), Space: O(n)
    """
    stack = []                 # Indices with strictly increasing heights
    best = 0
    # Append a sentinel 0 to force all remaining bars to be popped at the end
    for i, h in enumerate(heights + [0]):
        # While current bar breaks the increasing invariant
        while stack and heights[stack[-1]] > h:
            top_idx = stack.pop()
            top_h = heights[top_idx]
            # Left boundary: previous bar in stack (or -1 if none)
            left = stack[-1] if stack else -1
            # Width spans from left+1 to i-1 inclusive
            width = i - left - 1
            best = max(best, top_h * width)
        stack.append(i)
    return best


# =============================================================================
# SECTION 4: STACK — Evaluate Reverse Polish Notation
# =============================================================================

def eval_rpn(tokens):
    """
    Classic stack-based expression evaluator.

    INTUITION:
    - RPN = operands first, operator last.
    - Push operands. When you see an operator, pop the last TWO operands,
      apply, and push the result.
    - Order matters: second pop is the LEFT operand (it was pushed first).

    TRAP:
    - "/" in LeetCode's RPN problem truncates toward zero, not floor.
      Use int(a / b) and NOT a // b.

    Time: O(n), Space: O(n)
    """
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),     # Truncate toward zero
    }

    for tok in tokens:
        if tok in ops:
            b = stack.pop()                # Right operand (pushed LAST)
            a = stack.pop()                # Left operand  (pushed FIRST)
            stack.append(ops[tok](a, b))
        else:
            stack.append(int(tok))

    return stack[0]


# =============================================================================
# SECTION 5: QUEUE — BFS shortest path on a grid
# =============================================================================

def bfs_shortest_path(grid, start, target):
    """
    FIFO BFS on a 4-connected grid.

    WHY QUEUE AND NOT STACK:
    - BFS processes nodes in order of distance from the source.
    - FIFO guarantees we always dequeue the closest unexplored node first.
    - Using a stack would give DFS, which does NOT find shortest paths.

    VISITED MARKING:
    - We mark a cell as visited the moment we enqueue it, not when we dequeue.
    - Otherwise the same cell may be enqueued multiple times -> wrong complexity.

    Time: O(rows * cols), Space: O(rows * cols)
    """
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1 or grid[target[0]][target[1]] == 1:
        return -1
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (r, c), dist = queue.popleft()    # O(1) with deque, never list.pop(0)
        if (r, c) == target:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))      # Mark at enqueue time
                queue.append(((nr, nc), dist + 1))
    return -1


# =============================================================================
# SECTION 6: QUEUE WITH TWO STACKS
# =============================================================================

class MyQueue:
    """
    FIFO queue implemented with two LIFO stacks.

    AMORTIZED O(1):
    - push = append to in_stack (pure O(1))
    - pop  = if out_stack is empty, flush in_stack into out_stack (reversing order),
             then pop out_stack.
    - Each element is moved from in to out AT MOST ONCE.
    - Total cost over n operations is O(n), so O(1) amortized per op.
    """

    def __init__(self):
        self.in_stack = []        # Where pushes land
        self.out_stack = []       # Where pops come from

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self._ensure_out()
        return self.out_stack.pop()

    def peek(self):
        self._ensure_out()
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack

    def _ensure_out(self):
        """If the output stack is empty, transfer everything — reversing order."""
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())


# =============================================================================
# DEMOS
# =============================================================================

def demo_parentheses():
    print("\n" + "=" * 70)
    print("STACK #1: Valid Parentheses")
    print("=" * 70)
    tests = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True),
        ("(((", False),
    ]
    for s, expected in tests:
        got = is_valid_parentheses(s)
        assert got == expected, f"{s!r}: expected {expected}, got {got}"
    print("  All test cases passed")


def demo_daily_temperatures():
    print("\n" + "=" * 70)
    print("MONOTONIC STACK: Daily Temperatures (next warmer day)")
    print("=" * 70)
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    expected = [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures(temps) == expected
    assert daily_temperatures_brute(temps) == expected
    print(f"  temps    = {temps}")
    print(f"  expected = {expected}")

    # Performance comparison on a worst-case input
    print("\n  Performance comparison on 20_000 decreasing temps (brute = O(n^2)):")
    worst = list(range(20_000, 0, -1))
    timed(daily_temperatures_brute)(worst)
    timed(daily_temperatures)(worst)


def demo_largest_rectangle():
    print("\n" + "=" * 70)
    print("MONOTONIC STACK: Largest Rectangle in Histogram")
    print("=" * 70)
    tests = [
        ([2, 1, 5, 6, 2, 3], 10),
        ([2, 4], 4),
        ([1], 1),
        ([4, 2, 0, 3, 2, 5], 6),
    ]
    for heights, expected in tests:
        got = largest_rectangle(heights)
        assert got == expected, f"{heights}: expected {expected}, got {got}"
    print("  All test cases passed")


def demo_rpn():
    print("\n" + "=" * 70)
    print("STACK: Evaluate Reverse Polish Notation")
    print("=" * 70)
    tests = [
        (["2", "1", "+", "3", "*"], 9),                   # (2+1)*3
        (["4", "13", "5", "/", "+"], 6),                  # 4 + (13/5) = 4+2 = 6
        (["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"], 22),
    ]
    for tokens, expected in tests:
        got = eval_rpn(tokens)
        assert got == expected, f"{tokens}: expected {expected}, got {got}"
    print("  All test cases passed")


def demo_bfs_grid():
    print("\n" + "=" * 70)
    print("QUEUE: BFS shortest path on a grid (0 = free, 1 = wall)")
    print("=" * 70)
    grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ]
    assert bfs_shortest_path(grid, (0, 0), (3, 3)) == 6
    assert bfs_shortest_path([[0, 1], [1, 0]], (0, 0), (1, 1)) == -1
    assert bfs_shortest_path([[0]], (0, 0), (0, 0)) == 0
    print("  All test cases passed")


def demo_my_queue():
    print("\n" + "=" * 70)
    print("CLASSIC: Queue with two Stacks")
    print("=" * 70)
    q = MyQueue()
    q.push(1)
    q.push(2)
    q.push(3)
    assert q.peek() == 1
    assert q.pop() == 1
    assert q.pop() == 2
    q.push(4)
    assert q.pop() == 3
    assert q.pop() == 4
    assert q.empty()
    print("  All operations passed")


if __name__ == "__main__":
    demo_parentheses()
    demo_daily_temperatures()
    demo_largest_rectangle()
    demo_rpn()
    demo_bfs_grid()
    demo_my_queue()
    print("\nAll Day 4 demos passed.")

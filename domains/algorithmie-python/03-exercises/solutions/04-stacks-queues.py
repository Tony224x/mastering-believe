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

def num_islands(zone):
    """
    Multi-source BFS over a zone.

    APPROACH:
    - Scan every cell. When we find an unvisited '1', that's a new island.
    - Run BFS from that cell to mark every connected '1' as visited.
    - Increment the island counter once per BFS launch.

    WHY A QUEUE:
    - We need to explore every cell of one island before moving on.
    - BFS with a queue ensures we flood-fill breadth-first. A stack (DFS) also
      works here since we only care about connectivity, not shortest path.

    VISITED MARKING:
    - We mutate the zone to '0' when enqueuing to avoid a second visit.
    - Marking at enqueue (not dequeue) is critical: otherwise the same cell
      could be added to the queue multiple times, blowing up the cost.

    Time: O(rows * cols) — every cell enqueued at most once
    Space: O(rows * cols) — worst case the queue holds one full row/column
    """
    if not zone or not zone[0]:
        return 0

    rows, cols = len(zone), len(zone[0])
    islands = 0

    for r in range(rows):
        for c in range(cols):
            if zone[r][c] != "1":
                continue

            # Found a new island — BFS from here
            islands += 1
            queue = deque([(r, c)])
            zone[r][c] = "0"                # Mark at enqueue

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and zone[nr][nc] == "1"):
                        zone[nr][nc] = "0"   # Mark BEFORE enqueue
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
# EXERCISE 4 (Medium): Daily Temperatures
# =============================================================================

def daily_temperatures(temperatures):
    """
    Monotonic decreasing stack of INDICES.

    APPROACH:
    - Walk left to right keeping a stack of indices whose answer is unknown.
    - The stack is decreasing in temperature: when a warmer day arrives,
      every colder index on the stack just found its answer.

    WHY INDICES (not values):
    - The answer is a DISTANCE (i - popped_index), so we need positions.
      Next Greater Element (easy ex. 3) only needed values.

    Time: O(n) — each index pushed once, popped at most once
    Space: O(n) — worst case strictly decreasing temperatures
    """
    answer = [0] * len(temperatures)    # Default 0 = "no warmer day"
    stack = []                          # Indices with unresolved answers

    for i, temp in enumerate(temperatures):
        # Strict comparison: an EQUAL temperature is not warmer
        while stack and temperatures[stack[-1]] < temp:
            j = stack.pop()
            answer[j] = i - j           # Distance, not the temperature itself
        stack.append(i)

    # Indices left on the stack keep their default 0
    return answer


def test_exercise_4():
    print("\nExercise 4: Daily Temperatures")

    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert daily_temperatures([30, 60, 90]) == [1, 1, 0]
    assert daily_temperatures([90, 60, 30]) == [0, 0, 0]
    assert daily_temperatures([70]) == [0]
    assert daily_temperatures([70, 70, 70]) == [0, 0, 0]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Evaluate Reverse Polish Notation
# =============================================================================

def eval_rpn(tokens):
    """
    Stack evaluation of a postfix expression.

    APPROACH:
    - Numbers are pushed. An operator pops TWO operands and pushes the result.
    - Operand order matters: the FIRST pop is the RIGHT operand
      (e.g. ["3", "4", "-"] means 3 - 4, so right=4 is popped first).

    THE DIVISION TRAP:
    - Python's // floors toward -infinity: -7 // 2 == -4.
    - The problem requires truncation toward zero: int(-7 / 2) == -3.

    Time: O(n) — one pass over tokens
    Space: O(n) — stack of operands
    """
    stack = []
    operators = {"+", "-", "*", "/"}

    for token in tokens:
        if token in operators:
            right = stack.pop()         # First pop = RIGHT operand
            left = stack.pop()
            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            else:
                # int() truncates toward zero, unlike // which floors
                stack.append(int(left / right))
        else:
            # int() handles negative numbers like "-11" — they are NOT
            # operators because they have more than one character
            stack.append(int(token))

    return stack[0]                     # Valid expression leaves exactly one value


def test_exercise_5():
    print("\nExercise 5: Evaluate Reverse Polish Notation")

    assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
    assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
    assert eval_rpn(["42"]) == 42
    assert eval_rpn(["-7", "2", "/"]) == -3     # Truncation toward zero
    assert eval_rpn(["7", "-2", "/"]) == -3
    assert eval_rpn(["3", "4", "-"]) == -1      # Operand order

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Implement Queue using Two Stacks
# =============================================================================

class MyQueue:
    """
    FIFO queue built from two LIFO stacks.

    INVARIANT:
    - inbox receives every push.
    - outbox serves pops/peeks; its top is always the queue front.
    - We only refill outbox (by draining inbox) when outbox is EMPTY —
      transferring while outbox still has elements would break FIFO order.

    AMORTIZED ANALYSIS:
    - A single pop can cost O(n) (the transfer), but each element is moved
      from inbox to outbox AT MOST ONCE in its lifetime.
    - Total cost of n pushes + n pops = O(n) → O(1) amortized per operation.
    """

    def __init__(self):
        self.inbox = []     # Receives pushes (newest on top)
        self.outbox = []    # Serves pops (oldest on top after transfer)

    def push(self, x):
        """O(1) — just stack the new element."""
        self.inbox.append(x)

    def _transfer_if_needed(self):
        # Only when outbox is empty: reversing inbox puts the oldest on top
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())

    def pop(self):
        """O(1) amortized — transfer happens at most once per element."""
        self._transfer_if_needed()
        return self.outbox.pop()

    def peek(self):
        """O(1) amortized."""
        self._transfer_if_needed()
        return self.outbox[-1]

    def empty(self):
        """O(1) — queue is empty iff both stacks are empty."""
        return not self.inbox and not self.outbox


def test_exercise_6():
    print("\nExercise 6: Queue using Two Stacks")

    q = MyQueue()
    q.push(1)
    q.push(2)
    assert q.peek() == 1
    assert q.pop() == 1
    assert q.empty() == False
    assert q.pop() == 2
    assert q.empty() == True

    # Interleaved push/pop — detects incorrect transfers
    q2 = MyQueue()
    q2.push(1)
    q2.push(2)
    assert q2.pop() == 1
    q2.push(3)
    assert q2.pop() == 2    # Must come from outbox, NOT the fresh push
    assert q2.pop() == 3
    assert q2.empty() == True

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 7 (Hard): Largest Rectangle in Histogram
# =============================================================================

def largest_rectangle_brute(heights):
    """
    O(n^2) oracle — for each bar, expand left and right while bars are >= it.
    Kept only to cross-check the stack version on random inputs.
    """
    best = 0
    n = len(heights)
    for i in range(n):
        left = i
        while left > 0 and heights[left - 1] >= heights[i]:
            left -= 1
        right = i
        while right < n - 1 and heights[right + 1] >= heights[i]:
            right += 1
        best = max(best, heights[i] * (right - left + 1))
    return best


def largest_rectangle_area(heights):
    """
    Monotonic INCREASING stack of indices — O(n).

    KEY IDEA:
    - For each bar, the largest rectangle using its FULL height extends from
      the first shorter bar on its left to the first shorter bar on its right.
    - Keep an increasing stack. When a shorter bar arrives, every popped bar
      has just discovered its RIGHT boundary (current index i); its LEFT
      boundary is the new stack top (first shorter bar to the left).

    WIDTH TRAP:
    - If the stack becomes empty after the pop, the popped bar was the
      smallest so far → its rectangle spans from index 0, so width = i
      (NOT i - stack[-1] - 1, which would crash).

    SENTINEL:
    - Appending a virtual 0-height bar at the end forces every remaining
      index to be popped and measured — no special post-loop drain needed.

    Time: O(n) — each index pushed once, popped once
    Space: O(n)
    """
    best = 0
    stack = []                              # Indices with increasing heights

    # The sentinel height 0 flushes the whole stack at the end
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            # Left boundary: new stack top; empty stack → spans from index 0
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)

    return best


def test_exercise_7():
    print("\nExercise 7: Largest Rectangle in Histogram")

    assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
    assert largest_rectangle_area([2, 4]) == 4
    assert largest_rectangle_area([1]) == 1
    assert largest_rectangle_area([]) == 0
    assert largest_rectangle_area([0, 0, 0]) == 0
    assert largest_rectangle_area([5, 5, 5, 5]) == 20
    assert largest_rectangle_area([1, 2, 3, 4, 5]) == 9
    assert largest_rectangle_area([5, 4, 3, 2, 1]) == 9
    assert largest_rectangle_area([2, 1, 2]) == 3

    # Cross-check against the brute force oracle
    import random
    for _ in range(200):
        arr = [random.randint(0, 20) for _ in range(random.randint(0, 30))]
        assert largest_rectangle_area(arr) == largest_rectangle_brute(arr), arr

    # Benchmark: stack version should scale ~linearly, brute force ~quadratically
    import time
    print("  Benchmark (random heights):")
    print(f"    {'n':>6} | {'stack O(n)':>12} | {'brute O(n^2)':>13}")
    for n in [1000, 2000, 4000, 8000]:
        arr = [random.randint(0, 1000) for _ in range(n)]
        start = time.perf_counter()
        r1 = largest_rectangle_area(arr)
        t_stack = time.perf_counter() - start
        start = time.perf_counter()
        r2 = largest_rectangle_brute(arr)
        t_brute = time.perf_counter() - start
        assert r1 == r2
        print(f"    {n:>6,} | {t_stack:>11.5f}s | {t_brute:>12.5f}s")

    print("  PASS — all test cases + oracle + benchmark")


# =============================================================================
# EXERCISE 8 (Hard): Rotting Oranges (multi-source BFS)
# =============================================================================

def oranges_rotting(grid):
    """
    Multi-source BFS, counting levels (= minutes).

    APPROACH:
    - Enqueue ALL initially rotten oranges BEFORE starting the BFS, and count
      the fresh ones. All sources spread simultaneously, which is exactly
      what a single shared queue models.
    - Process the queue level by level: one level = one minute.
    - Decrement the fresh counter on each contamination; if any fresh orange
      remains at the end, it is unreachable → -1.

    WHY NOT re-scan the grid each minute:
    - Re-scanning is O(R*C) per minute → O((R*C)^2) worst case.
      The BFS touches each cell once → O(R*C) total.

    MINUTE COUNTING TRAP:
    - The initial wave (minute 0) must not count: only increment when a level
      actually contaminates at least one orange.

    Time: O(rows * cols), Space: O(rows * cols)
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Multi-source init: every rotten orange starts in the queue
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0                        # Nothing to rot — 0 minutes, even if no source

    minutes = 0
    while queue and fresh:
        # Freeze the level size: everything currently queued rots neighbors
        # during the SAME minute
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2    # Mark at enqueue — no double visit
                    fresh -= 1
                    queue.append((nr, nc))
        minutes += 1

    return minutes if fresh == 0 else -1


def test_exercise_8():
    print("\nExercise 8: Rotting Oranges")

    assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert oranges_rotting([[0, 2]]) == 0
    assert oranges_rotting([[0]]) == 0
    assert oranges_rotting([[1]]) == -1
    assert oranges_rotting([[2]]) == 0
    assert oranges_rotting([[2, 2], [1, 1], [0, 0], [2, 0]]) == 1

    print("  PASS — all test cases")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    test_exercise_6()
    test_exercise_7()
    test_exercise_8()
    print("\nAll Day 4 exercise solutions passed.")

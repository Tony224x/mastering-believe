"""
Solutions — Day 12 Backtracking & Recursion (easy, medium and hard exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/12-backtracking-recursion.py
"""


# =============================================================================
# Exercise 1: Subsets
# =============================================================================

def subsets(nums):
    """
    Backtracking with start index.
    KEY INSIGHT: every intermediate path is itself a valid subset, so we
    snapshot path[:] at every recursive call (not just at a leaf).

    Total subsets: 2^n. Each one is a path from root to some node in the
    decision tree where at each level we choose "include nums[i] or not".

    Time : O(2^n * n) — 2^n subsets, each copy is O(n)
    Space: O(n) for recursion + O(2^n * n) for output
    """
    result = []

    def backtrack(start, path):
        result.append(path[:])         # Snapshot
        for i in range(start, len(nums)):
            path.append(nums[i])       # Choose
            backtrack(i + 1, path)     # Explore (i+1 to avoid re-picking)
            path.pop()                 # Unchoose

    backtrack(0, [])
    return result


# =============================================================================
# Exercise 2: Permutations
# =============================================================================

def permute(nums):
    """
    Backtracking with a used[] array.
    At each level, we try every unused element. When len(path) == len(nums),
    we have a complete permutation.

    Time : O(n! * n)
    Space: O(n)
    """
    result = []
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


# =============================================================================
# Exercise 3: Generate Parentheses
# =============================================================================

def generate_parentheses(n):
    """
    Backtracking with two counters.
    Two invariants that guarantee validity:
      - open_count < n   -> we can still open more
      - close_count < open_count -> we haven't closed more than we opened

    Because we only take these two actions when legal, every leaf reached
    is automatically valid — no post-validation needed.

    Time : O(4^n / sqrt(n)) — Catalan number growth
    Space: O(n) recursion + output
    """
    result = []

    def backtrack(path, open_count, close_count):
        if len(path) == 2 * n:
            result.append("".join(path))
            return
        if open_count < n:
            path.append("(")
            backtrack(path, open_count + 1, close_count)
            path.pop()
        if close_count < open_count:
            path.append(")")
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result



# =============================================================================
# Exercise 4 (Medium): Combination Sum
# =============================================================================

def combination_sum(candidates, target):
    """
    Backtracking with a start index AND element reuse.

    THE i vs i+1 NUANCE:
    - Recursing with `i` allows picking candidates[i] again (unlimited
      reuse) but never anything BEFORE i. Combinations are thus built in
      non-decreasing index order — a canonical form that kills duplicates
      like [2,3] vs [3,2] without any dedup set.
    - Recursing with i+1 would forbid reuse (that's Combination Sum II).

    PRUNING:
    - Sorting + break as soon as a candidate exceeds the remaining target:
      all later candidates are even bigger.

    Time: exponential in the worst case (bounded by target/min(candidate)
    depth), Space: O(target/min) recursion depth.
    """
    candidates = sorted(candidates)     # Enables the early break
    result = []
    path = []

    def backtrack(start, remaining):
        if remaining == 0:
            result.append(path[:])      # COPY — path keeps mutating after this
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                   # Sorted: everything after is too big
            path.append(candidates[i])              # Choose
            backtrack(i, remaining - candidates[i])  # `i`, not i+1: reuse OK
            path.pop()                               # Un-choose

    backtrack(0, target)
    return result


# =============================================================================
# Exercise 5 (Medium): Permutations II (duplicates, pruned)
# =============================================================================

def permute_unique(nums):
    """
    Permutations with duplicate values, deduplicated DURING recursion.

    THE SKIP RULE (after sorting):
        if nums[i] == nums[i-1] and not used[i-1]: continue

    WHY IT WORKS:
    - Among identical copies, the rule says: copy i may only be placed if
      copy i-1 is already in the current path. So identical copies always
      appear in index order — ONE canonical interleaving per multiset
      arrangement, instead of k! identical permutations.
    - Filtering at the end with a set still EXPLORES all n! branches;
      pruning cuts those branches before they grow.

    Time: O(#unique permutations * n), Space: O(n)
    """
    nums = sorted(nums)                 # Precondition for the skip rule
    result = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            # Skip a duplicate unless its left twin is already placed
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return result


# =============================================================================
# Exercise 6 (Medium): Word Search (grid backtracking)
# =============================================================================

def exist(board, word):
    """
    DFS from every starting cell, marking and RESTORING cells.

    MARK + RESTORE:
    - board[r][c] = "#" before recursing prevents reusing the cell within
      the current path. Restoring the letter afterwards is essential:
      other paths (and other starting cells) must still see it.
      Forgetting the restore is the classic bug — the test that re-checks
      board integrity catches it.

    WHY O(R * C * 3^L):
    - Each of the R*C starting cells explores at most 3 directions per
      step after the first (never going back where it came from), for L
      steps.
    """
    if not word:
        return True
    if not board or not board[0]:
        return False

    rows, cols = len(board), len(board[0])

    def dfs(r, c, k):
        # k = index of the character we are trying to match at (r, c)
        if board[r][c] != word[k]:
            return False
        if k == len(word) - 1:
            return True

        board[r][c] = "#"               # Mark: used by the current path
        found = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dfs(nr, nc, k + 1):
                found = True
                break
        board[r][c] = word[k]           # RESTORE — other paths need this cell
        return found

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))


# =============================================================================
# Exercise 7 (Hard): N-Queens
# =============================================================================

def solve_n_queens(n):
    """
    Row-by-row backtracking with O(1) conflict tests.

    THE THREE SETS:
    - cols: columns already occupied.
    - diag1 (r - c): constant along "\\" diagonals.
    - diag2 (r + c): constant along "/" diagonals.
    Placing a queen is legal iff none of the three keys is taken — three
    O(1) set lookups instead of scanning the board (O(n)).

    ONE QUEEN PER ROW BY CONSTRUCTION:
    - The recursion places exactly one queen per row, so row conflicts
      can simply never happen — no need to track them.

    Time: O(n!) with heavy pruning in practice, Space: O(n)
    """
    result = []
    queens = []                         # queens[r] = column of the queen on row r
    cols, diag1, diag2 = set(), set(), set()

    def backtrack(r):
        if r == n:
            # Build the string board only when a full solution exists
            result.append(["." * c + "Q" + "." * (n - c - 1) for c in queens])
            return
        for c in range(n):
            if c in cols or (r - c) in diag1 or (r + c) in diag2:
                continue                # O(1) conflict test
            queens.append(c)
            cols.add(c); diag1.add(r - c); diag2.add(r + c)
            backtrack(r + 1)
            queens.pop()
            cols.discard(c); diag1.discard(r - c); diag2.discard(r + c)

    backtrack(0)
    return result


# =============================================================================
# Exercise 8 (Hard): Sudoku Solver
# =============================================================================

def solve_sudoku(board):
    """
    Backtracking over precomputed constraint sets + MRV cell ordering.

    PRECOMPUTED SETS (the required optimization):
    - rows[r], cols[c], boxes[(r//3)*3 + c//3] hold the digits already
      used. "Can digit d go at (r, c)?" is three O(1) lookups — a naive
      is_valid() that re-scans row/column/box costs O(27) per attempt and
      multiplies the whole search time.

    MRV — MINIMUM REMAINING VALUES (the bonus, included because it makes
    hard puzzles tractable):
    - Instead of filling cells in reading order, always branch on the
      empty cell with the FEWEST legal candidates. Cells with one
      candidate are forced moves; failing early near the root prunes
      gigantic subtrees. This is what keeps Inkala's grid well under a
      second in pure Python.
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empties = []

    # Single initial pass building the 27 constraint sets
    for r in range(9):
        for c in range(9):
            d = board[r][c]
            if d == ".":
                empties.append((r, c))
            else:
                rows[r].add(d)
                cols[c].add(d)
                boxes[(r // 3) * 3 + c // 3].add(d)

    def candidates(r, c):
        b = (r // 3) * 3 + c // 3
        return [d for d in "123456789"
                if d not in rows[r] and d not in cols[c] and d not in boxes[b]]

    def backtrack(remaining):
        if not remaining:
            return True                 # Every cell filled

        # MRV: branch on the most constrained cell
        idx = min(range(len(remaining)),
                  key=lambda i: len(candidates(*remaining[i])))
        r, c = remaining[idx]
        rest = remaining[:idx] + remaining[idx + 1:]
        b = (r // 3) * 3 + c // 3

        for d in candidates(r, c):
            board[r][c] = d
            rows[r].add(d); cols[c].add(d); boxes[b].add(d)
            if backtrack(rest):
                return True
            board[r][c] = "."           # Undo symmetrically
            rows[r].discard(d); cols[c].discard(d); boxes[b].discard(d)

        return False                    # Dead end: trigger backtracking above

    backtrack(empties)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
    def sorted_subsets(result):
        return sorted([sorted(s) for s in result])

    assert sorted_subsets(subsets([])) == [[]]
    assert sorted_subsets(subsets([0])) == [[], [0]]
    assert sorted_subsets(subsets([1, 2, 3])) == [
        [], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]
    ]
    assert len(subsets([1, 2, 3, 4])) == 16
    assert len(subsets([1, 2, 3, 4, 5])) == 32
    print("Exercise 1 (subsets): OK")

    # -- Exercise 2 --
    def sorted_perms(result):
        return sorted([tuple(p) for p in result])

    assert sorted_perms(permute([1])) == [(1,)]
    assert sorted_perms(permute([1, 2])) == [(1, 2), (2, 1)]
    assert sorted_perms(permute([1, 2, 3])) == [
        (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)
    ]
    assert len(permute([1, 2, 3, 4])) == 24
    assert len(permute([1, 2, 3, 4, 5])) == 120
    print("Exercise 2 (permute): OK")

    # -- Exercise 3 --
    assert sorted(generate_parentheses(1)) == ["()"]
    assert sorted(generate_parentheses(2)) == ["(())", "()()"]
    assert sorted(generate_parentheses(3)) == [
        "((()))", "(()())", "(())()", "()(())", "()()()"
    ]
    assert len(generate_parentheses(4)) == 14
    assert len(generate_parentheses(5)) == 42
    print("Exercise 3 (generate_parentheses): OK")

    # -- Exercise 4 --
    def normalize_combos(result):
        return sorted([sorted(c) for c in result])

    assert normalize_combos(combination_sum([2, 3, 6, 7], 7)) == [[2, 2, 3], [7]]
    assert normalize_combos(combination_sum([2, 3, 5], 8)) == [
        [2, 2, 2, 2], [2, 3, 3], [3, 5]
    ]
    assert combination_sum([2], 1) == []
    assert normalize_combos(combination_sum([1], 2)) == [[1, 1]]
    assert combination_sum([3, 5], 2) == []
    assert normalize_combos(combination_sum([7], 7)) == [[7]]
    print("Exercise 4 (combination_sum): OK")

    # -- Exercise 5 --
    def normalize_perms(result):
        return sorted([tuple(p) for p in result])

    assert normalize_perms(permute_unique([1, 1, 2])) == [(1, 1, 2), (1, 2, 1), (2, 1, 1)]
    assert normalize_perms(permute_unique([1, 2, 3])) == [
        (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)
    ]
    assert normalize_perms(permute_unique([1, 1])) == [(1, 1)]
    assert normalize_perms(permute_unique([1])) == [(1,)]
    assert len(permute_unique([1, 1, 2, 2])) == 6
    assert len(permute_unique([2, 2, 2])) == 1
    print("Exercise 5 (permute_unique): OK")

    # -- Exercise 6 --
    board = [
        ["A", "B", "C", "E"],
        ["S", "F", "C", "S"],
        ["A", "D", "E", "E"],
    ]
    assert exist([row[:] for row in board], "ABCCED") == True
    assert exist([row[:] for row in board], "SEE") == True
    assert exist([row[:] for row in board], "ABCB") == False
    assert exist([row[:] for row in board], "") == True
    assert exist([["A"]], "A") == True
    assert exist([["A"]], "AA") == False
    assert exist([["A", "A"]], "AAA") == False
    b = [row[:] for row in board]
    exist(b, "ABCCED")
    assert b == board                   # Restoration check
    print("Exercise 6 (exist): OK")

    # -- Exercise 7 --
    result = solve_n_queens(4)
    assert len(result) == 2
    expected = {
        (".Q..", "...Q", "Q...", "..Q."),
        ("..Q.", "Q...", "...Q", ".Q.."),
    }
    assert {tuple(sol) for sol in result} == expected
    assert len(solve_n_queens(1)) == 1
    assert solve_n_queens(2) == []
    assert solve_n_queens(3) == []
    assert len(solve_n_queens(6)) == 4
    assert len(solve_n_queens(8)) == 92
    for sol in solve_n_queens(6):
        queens = [(r, row.index("Q")) for r, row in enumerate(sol)]
        assert len({c for _, c in queens}) == 6
        assert len({r - c for r, c in queens}) == 6
        assert len({r + c for r, c in queens}) == 6
    print("Exercise 7 (solve_n_queens): OK")

    # -- Exercise 8 --
    def is_solved(bd):
        full = set("123456789")
        for i in range(9):
            if {bd[i][j] for j in range(9)} != full:
                return False
            if {bd[j][i] for j in range(9)} != full:
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box = {bd[br + dr][bc + dc] for dr in range(3) for dc in range(3)}
                if box != full:
                    return False
        return True

    puzzle = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    solve_sudoku(puzzle)
    assert is_solved(puzzle)
    assert puzzle[0] == ["5", "3", "4", "6", "7", "8", "9", "1", "2"]

    import time
    hard = [
        ["8", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", "3", "6", ".", ".", ".", ".", "."],
        [".", "7", ".", ".", "9", ".", "2", ".", "."],
        [".", "5", ".", ".", ".", "7", ".", ".", "."],
        [".", ".", ".", ".", "4", "5", "7", ".", "."],
        [".", ".", ".", "1", ".", ".", ".", "3", "."],
        [".", ".", "1", ".", ".", ".", ".", "6", "8"],
        [".", ".", "8", "5", ".", ".", ".", "1", "."],
        [".", "9", ".", ".", ".", ".", "4", ".", "."],
    ]
    start = time.perf_counter()
    solve_sudoku(hard)
    elapsed = time.perf_counter() - start
    assert is_solved(hard)
    assert elapsed < 5, f"Sudoku too slow: {elapsed:.2f}s"
    print(f"Exercise 8 (solve_sudoku): OK (hard grid in {elapsed:.3f}s)")

    print("\nAll Day 12 solutions pass!")

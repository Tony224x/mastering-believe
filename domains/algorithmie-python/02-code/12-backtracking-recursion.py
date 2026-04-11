"""
Day 12 — Backtracking & Recursion: the universal choose/explore/unchoose template
Run: python domains/algorithmie-python/02-code/12-backtracking-recursion.py
"""


# =============================================================================
# SECTION 1: SUBSETS
# =============================================================================

def subsets(nums):
    """
    Generate all 2^n subsets of nums.
    Each partial path IS a subset — we append at every recursive call,
    not only when we hit a "base case".

    `start` ensures we never revisit an earlier index, which avoids
    producing [1,2] and [2,1] (both represent the same subset).
    """
    result = []

    def backtrack(start, path):
        result.append(path[:])        # Snapshot the current subset
        for i in range(start, len(nums)):
            path.append(nums[i])      # Choose
            backtrack(i + 1, path)    # Explore with narrowed range
            path.pop()                # Unchoose

    backtrack(0, [])
    return result


# =============================================================================
# SECTION 2: PERMUTATIONS
# =============================================================================

def permutations(nums):
    """
    Generate all n! permutations of nums.
    We track used[] to avoid reusing an element within one permutation.
    A permutation is complete when len(path) == len(nums).
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


def permutations_unique(nums):
    """
    Permutations with duplicates -- skip symmetric branches.
    TRICK: sort first, then skip nums[i] if the previous equal element
    has NOT been used yet (which would mean we're about to generate the
    same permutation as if we'd taken the previous one).
    """
    result = []
    nums.sort()
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


# =============================================================================
# SECTION 3: COMBINATIONS
# =============================================================================

def combinations(n, k):
    """
    All combinations of k numbers chosen from [1..n].
    PRUNING: if we still need (k - len(path)) more elements, we must
    start at most at n - (k - len(path)) + 1 -- otherwise we can't
    possibly finish.
    """
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        need = k - len(path)
        for i in range(start, n - need + 2):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result


def combination_sum(candidates, target):
    """
    Find all unique combinations (unlimited reuse) summing to target.
    We sort to enable pruning (`break` as soon as candidate > remaining).
    """
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                  # All further candidates too big
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # `i` = reuse allowed
            path.pop()

    backtrack(0, [], target)
    return result


# =============================================================================
# SECTION 4: N-QUEENS
# =============================================================================

def solve_n_queens(n):
    """
    Place n queens so none attack each other.
    TWO KEYS:
      1. We place one queen per row (no two queens in the same row ever)
      2. For attack checks, we use 3 sets:
           - cols: columns already occupied
           - diag1: diagonals where row - col is constant
           - diag2: anti-diagonals where row + col is constant
    Each check/update is O(1).
    """
    result = []
    cols = set()
    diag1 = set()
    diag2 = set()
    board = [["."] * n for _ in range(n)]

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = "Q"
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            board[row][col] = "."
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result


# =============================================================================
# SECTION 5: WORD SEARCH
# =============================================================================

def word_search(board, word):
    """
    DFS with in-place cell marking.
    When we visit a cell we overwrite it with '#' so we can't reuse it
    in the same path. We restore the original character on the way back
    (classic backtracking undo).

    Time : O(R * C * 4^L), L = len(word)
    Space: O(L) recursion
    """
    if not board or not board[0]:
        return False
    rows, cols = len(board), len(board[0])

    def dfs(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
            return False
        tmp = board[r][c]
        board[r][c] = "#"               # Mark as visited
        found = (dfs(r + 1, c, idx + 1) or
                 dfs(r - 1, c, idx + 1) or
                 dfs(r, c + 1, idx + 1) or
                 dfs(r, c - 1, idx + 1))
        board[r][c] = tmp               # Restore
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False


# =============================================================================
# SECTION 6: GENERATE PARENTHESES
# =============================================================================

def generate_parentheses(n):
    """
    All valid combinations of n pairs of parentheses.
    INVARIANT: at every point, we can only add '(' if open < n, and we
    can only add ')' if close < open (otherwise we'd close unopened pairs).
    These two rules alone guarantee a valid result — no separate validity
    check needed.
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
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("Subsets [1,2,3]:")
    print(" ", subsets([1, 2, 3]))

    print("\nPermutations [1,2,3]:")
    print(" ", permutations([1, 2, 3]))

    print("\nUnique permutations [1,1,2]:")
    print(" ", permutations_unique([1, 1, 2]))

    print("\nCombinations C(4, 2):")
    print(" ", combinations(4, 2))

    print("\nCombination sum [2,3,6,7] target=7:")
    print(" ", combination_sum([2, 3, 6, 7], 7))

    print("\nN-Queens n=4:")
    for sol in solve_n_queens(4):
        for row in sol:
            print(" ", row)
        print()

    print("Word search 'ABCCED':")
    board = [
        ["A", "B", "C", "E"],
        ["S", "F", "C", "S"],
        ["A", "D", "E", "E"],
    ]
    print(" ", word_search([row[:] for row in board], "ABCCED"))  # True
    print(" ", word_search([row[:] for row in board], "ABCB"))    # False

    print("\nGenerate parentheses n=3:")
    print(" ", generate_parentheses(3))

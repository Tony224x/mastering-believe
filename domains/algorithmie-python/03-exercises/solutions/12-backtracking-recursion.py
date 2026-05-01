"""
Solutions — Day 12 Backtracking & Recursion (easy exercises).
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

    print("\nAll Day 12 solutions pass!")

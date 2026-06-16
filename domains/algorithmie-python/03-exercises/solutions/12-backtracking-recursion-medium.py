"""
Solutions — Day 12: Backtracking & Recursion (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/12-backtracking-recursion-medium.py

Each solution is numbered to match the exercise file (02-medium/12-backtracking-recursion.md).
All solutions are verified with assertions at the end.
"""


# =============================================================================
# EXERCISE 4 (Medium): Combination Sum — reuse allowed
# =============================================================================

def combination_sum(candidates, target):
    """
    Backtrack passing `i` (not i+1) so the same candidate can be reused.
    Sort + break prune branches once a candidate exceeds the remainder.

    Time: O(n^(target/min)) worst case, Space: O(target/min) recursion
    """
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])     # Copy the completed combination
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                  # Sorted: everything after is bigger too
            path.append(candidates[i])             # Choose
            backtrack(i, path, remaining - candidates[i])   # i: allow reuse
            path.pop()                             # Unchoose

    backtrack(0, [], target)
    return result


def test_exercise_4():
    print("\nExercise 4: Combination Sum")

    def sort_combos(combos):
        return sorted(sorted(c) for c in combos)

    assert sort_combos(combination_sum([2, 3, 6, 7], 7)) == sort_combos([[2, 2, 3], [7]])
    assert sort_combos(combination_sum([2, 3, 5], 8)) == sort_combos(
        [[2, 2, 2, 2], [2, 3, 3], [3, 5]])
    assert combination_sum([2], 1) == []
    assert sort_combos(combination_sum([1], 2)) == [[1, 1]]
    assert combination_sum([3, 5], 1) == []
    assert sort_combos(combination_sum([2, 4], 6)) == sort_combos([[2, 2, 2], [2, 4]])

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Subsets II — powerset with duplicates
# =============================================================================

def subsets_with_dup(nums):
    """
    Sort, then skip duplicates at the SAME tree level
    (i > start and nums[i] == nums[i-1]) to avoid duplicate subsets.
    Each partial state is a valid subset.

    Time: O(2^n * n), Space: O(n) recursion
    """
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path[:])         # Every partial state is a subset
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue               # Skip duplicate at this level
            path.append(nums[i])       # Choose
            backtrack(i + 1, path)     # i+1: no reuse
            path.pop()                 # Unchoose

    backtrack(0, [])
    return result


def test_exercise_5():
    print("\nExercise 5: Subsets II")

    def sort_subsets(subs):
        return sorted(sorted(s) for s in subs)

    assert sort_subsets(subsets_with_dup([1, 2, 2])) == sort_subsets(
        [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]])
    assert sort_subsets(subsets_with_dup([0])) == [[], [0]]
    assert sort_subsets(subsets_with_dup([])) == [[]]
    assert len(subsets_with_dup([1, 1, 1])) == 4
    assert sort_subsets(subsets_with_dup([4, 4, 4, 1, 4])) == sort_subsets(
        [[], [1], [1, 4], [1, 4, 4], [1, 4, 4, 4], [1, 4, 4, 4, 4],
         [4], [4, 4], [4, 4, 4], [4, 4, 4, 4]])

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Generate Parentheses — constrained generation
# =============================================================================

def generate_parentheses(n):
    """
    Two constraints generate only valid sequences:
    - add '(' while open_count < n
    - add ')' while close_count < open_count

    Time: O(4^n / sqrt(n)) — Catalan number of results
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
        if close_count < open_count:   # Never close more than opened
            path.append(")")
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result


def test_exercise_6():
    print("\nExercise 6: Generate Parentheses")

    assert sorted(generate_parentheses(1)) == ["()"]
    assert sorted(generate_parentheses(2)) == ["(())", "()()"]
    assert sorted(generate_parentheses(3)) == sorted(
        ["((()))", "(()())", "(())()", "()(())", "()()()"])
    assert generate_parentheses(0) == [""]
    assert len(generate_parentheses(4)) == 14

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 12: Backtracking & Recursion (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

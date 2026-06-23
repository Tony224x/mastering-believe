"""
Solutions — Day 12: Backtracking & Recursion (HARD)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/12-backtracking-recursion-hard.py

Each solution is numbered to match the exercise file (03-hard/12-backtracking-recursion.md).
All solutions are verified with assertions at the end.
"""

import copy


# =============================================================================
# EXERCISE 7 (Hard): Word Search — backtracking on a grid
# =============================================================================

def exist(board, word):
    """
    DFS + undo: mark the current cell '#', explore 4 neighbours, restore it.
    The board is fully restored to its initial state after the call.

    Time: O(R * C * 4^L), Space: O(L) recursion (L = len(word))
    """
    if not word:
        return True
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
                board[r][c] != word[idx]):
            return False
        tmp = board[r][c]
        board[r][c] = "#"              # Choose (block reuse in this path)
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))
        board[r][c] = tmp             # Unchoose (restore)
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False


def test_exercise_7():
    print("\nExercise 7: Word Search")

    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    snapshot = copy.deepcopy(board)
    assert exist(board, "ABCCED") is True
    assert exist(board, "SEE") is True
    assert exist(board, "ABCB") is False
    assert board == snapshot           # Board restored after each call

    assert exist([["A"]], "A") is True
    assert exist([["A"]], "B") is False
    assert exist([["A", "B"], ["C", "D"]], "ACDB") is True
    assert exist([["A", "A"]], "AAA") is False

    print("  PASS — all test cases (incl. board restoration)")


# =============================================================================
# EXERCISE 8 (Hard): N-Queens — constraint placement, O(1) attack check
# =============================================================================

def total_n_queens(n):
    """
    One queen per row. A column is safe if col not in cols, (row-col) not in
    diag1, (row+col) not in diag2 — all O(1) set lookups.

    Time: O(n!) with pruning, Space: O(n)
    """
    cols = set()
    diag1 = set()                      # row - col (the "\" diagonals)
    diag2 = set()                      # row + col (the "/" diagonals)
    count = 0

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col); diag1.add(row - col); diag2.add(row + col)   # Choose
            backtrack(row + 1)                                          # Explore
            cols.discard(col); diag1.discard(row - col); diag2.discard(row + col)  # Unchoose

    backtrack(0)
    return count


def test_exercise_8():
    print("\nExercise 8: N-Queens (count)")

    assert total_n_queens(1) == 1
    assert total_n_queens(2) == 0
    assert total_n_queens(3) == 0
    assert total_n_queens(4) == 2
    assert total_n_queens(5) == 10
    assert total_n_queens(6) == 4
    assert total_n_queens(8) == 92

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Palindrome Partitioning
# =============================================================================

def partition(s):
    """
    Backtrack by prefix: for each end, if s[start:end] is a palindrome, take it
    and recurse from end. Record when start reaches the end of the string.

    Time: O(n * 2^n), Space: O(n) recursion
    """
    result = []
    n = len(s)

    def is_palindrome(lo, hi):         # s[lo:hi] inclusive-exclusive check
        i, j = lo, hi - 1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    def backtrack(start, path):
        if start == n:
            result.append(path[:])     # Complete partition
            return
        for end in range(start + 1, n + 1):
            if is_palindrome(start, end):
                path.append(s[start:end])     # Choose palindromic prefix
                backtrack(end, path)          # Explore the rest
                path.pop()                    # Unchoose

    backtrack(0, [])
    return result


def test_exercise_9():
    print("\nExercise 9: Palindrome Partitioning")

    def sort_parts(parts):
        return sorted(parts)

    assert sort_parts(partition("aab")) == sort_parts([["a", "a", "b"], ["aa", "b"]])
    assert sort_parts(partition("a")) == [["a"]]
    assert sort_parts(partition("")) == [[]]
    assert sort_parts(partition("aba")) == sort_parts([["a", "b", "a"], ["aba"]])
    assert sort_parts(partition("abc")) == sort_parts([["a", "b", "c"]])
    assert sort_parts(partition("aaa")) == sort_parts(
        [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]])

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 12: Backtracking & Recursion (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

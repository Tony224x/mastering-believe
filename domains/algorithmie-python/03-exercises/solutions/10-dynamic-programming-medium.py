"""
Solutions — Day 10: Dynamic Programming (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/10-dynamic-programming-medium.py

Each solution is numbered to match the exercise file (02-medium/10-dynamic-programming.md).
All solutions are verified with assertions at the end.
"""


# =============================================================================
# EXERCISE 4 (Medium): House Robber II — circular linear DP
# =============================================================================

def rob_circular(nums: list[int]) -> int:
    """
    Houses in a CIRCLE: house 0 and house n-1 are adjacent.

    KEY REDUCTION:
    - In a circle you can never rob BOTH the first and the last house.
    - So the optimum is the best of two LINEAR House Robber runs:
        * houses [0 .. n-2]  (allow first, forbid last)
        * houses [1 .. n-1]  (forbid first, allow last)
    - The single-house case has no neighbor, handle it separately.

    Time: O(n), Space: O(1)
    """
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    def rob_linear(houses: list[int]) -> int:
        # Classic House Robber: prev2 = best up to i-2, prev1 = best up to i-1
        prev2, prev1 = 0, 0
        for gold in houses:
            prev2, prev1 = prev1, max(prev1, prev2 + gold)
        return prev1

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))


def test_exercise_4():
    print("\nExercise 4: House Robber II (circular)")

    assert rob_circular([2, 3, 2]) == 3
    assert rob_circular([1, 2, 3, 1]) == 4
    assert rob_circular([0]) == 0
    assert rob_circular([5]) == 5
    assert rob_circular([1, 2]) == 2
    assert rob_circular([2, 7, 9, 3, 1]) == 11
    assert rob_circular([1, 2, 3]) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Longest Common Subsequence — 2D DP (rolling rows)
# =============================================================================

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    dp[i][j] = LCS length of text1[:i] and text2[:j].

    RECURRENCE:
    - text1[i-1] == text2[j-1] -> dp[i][j] = dp[i-1][j-1] + 1   (diagonal + 1)
    - otherwise               -> dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    SPACE OPTIMIZATION:
    - dp[i][j] only needs the previous row and the current row, so we keep
      two 1D arrays and index on the SHORTER string so width = min(m, n).

    Time: O(m * n), Space: O(min(m, n))
    """
    # Make `b` the shorter string so the rolling width is min(m, n)
    a, b = (text1, text2) if len(text1) >= len(text2) else (text2, text1)
    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, len(a) + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev          # roll the rows without reallocating

    return prev[n]


def _lcs_grid(text1: str, text2: str) -> int:
    """Reference full-grid version used as a cross-check oracle."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def test_exercise_5():
    print("\nExercise 5: Longest Common Subsequence")

    cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
        ("", "abc", 0),
        ("abc", "", 0),
        ("bl", "yby", 1),
        ("ezupkr", "ubmrapg", 2),
        ("oxcpqrsvwf", "shmtulqrypy", 2),
    ]
    for t1, t2, expected in cases:
        got = longest_common_subsequence(t1, t2)
        assert got == expected, f"{t1!r},{t2!r}: got {got}, want {expected}"
        # Cross-check rolling version against the full-grid oracle
        assert got == _lcs_grid(t1, t2)

    print("  PASS — all test cases (incl. full-grid cross-check)")


# =============================================================================
# EXERCISE 6 (Medium): Decode Ways — count DP with conditional transitions
# =============================================================================

def num_decodings(s: str) -> int:
    """
    Count decodings of a digit string with mapping A=1..Z=26.

    dp[i] = number of ways to decode s[:i].
    Base: dp[0] = 1 (empty string decodes one way).

    TRANSITIONS (reading s[:i]):
    - One char  s[i-1]:    valid iff s[i-1] != '0'      -> dp[i] += dp[i-1]
    - Two chars s[i-2:i]:  valid iff "10" <= s[i-2:i] <= "26" -> dp[i] += dp[i-2]

    The two-digit check also rejects "00", "30".."99" automatically and the
    one-digit check rejects standalone '0' (so "0", "06", "100" -> 0).

    Time: O(n), Space: O(1) (two rolling variables)
    """
    if not s:
        return 0

    prev2 = 1                            # dp[0]
    prev1 = 1 if s[0] != '0' else 0      # dp[1]

    for i in range(2, len(s) + 1):
        current = 0
        if s[i - 1] != '0':             # single-digit decode
            current += prev1
        two = int(s[i - 2:i])           # two-digit decode (10..26)
        if 10 <= two <= 26:
            current += prev2
        prev2, prev1 = prev1, current

    return prev1


def test_exercise_6():
    print("\nExercise 6: Decode Ways")

    assert num_decodings("12") == 2
    assert num_decodings("226") == 3
    assert num_decodings("0") == 0
    assert num_decodings("06") == 0
    assert num_decodings("10") == 1
    assert num_decodings("100") == 0
    assert num_decodings("1") == 1
    assert num_decodings("27") == 1
    assert num_decodings("2101") == 1
    assert num_decodings("11106") == 2

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 10: Dynamic Programming (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

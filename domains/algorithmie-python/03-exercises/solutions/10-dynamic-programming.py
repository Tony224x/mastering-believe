"""
Solutions — Day 10 Dynamic Programming (easy exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/10-dynamic-programming.py
"""


# =============================================================================
# Exercise 1: Climbing Stairs (O(1) space)
# =============================================================================

def climb_stairs(n):
    """
    Linear DP with O(1) space.
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    Base: dp[0] = 1 (one way: do nothing), dp[1] = 1

    We only need the last two values, so two rolling variables suffice.

    Time : O(n)
    Space: O(1)
    """
    if n < 2:
        return 1
    prev2, prev1 = 1, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev1 + prev2
    return prev1


# Memoized variant (top-down) — same complexity, different style
def climb_stairs_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n < 2:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    return memo[n]


# =============================================================================
# Exercise 2: Coin Change (minimum coins)
# =============================================================================

def coin_change(coins, amount):
    """
    Bottom-up tabulation.
    dp[a] = min coins to make amount `a`; initialized to +inf.
    dp[0] = 0 (zero coins to make zero).

    For each target a from 1 to amount:
        For each coin c, if c <= a:
            dp[a] = min(dp[a], dp[a - c] + 1)

    Time : O(amount * len(coins))
    Space: O(amount)
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


# =============================================================================
# Exercise 3: Unique Paths
# =============================================================================

def unique_paths(m, n):
    """
    2D tabulation.
    dp[i][j] = number of unique paths from (0, 0) to (i, j).
    Base case: the entire first row and first column are 1 (only one way
    to reach any cell on the top edge or left edge — moving straight).
    Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1].

    Time : O(m * n)
    Space: O(m * n)
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]


def unique_paths_rolling(m, n):
    """
    Space-optimized version using a single row of size n.
    row[j] starts as the previous row's value; after update, row[j-1] is
    the current row's left neighbor. So row[j] += row[j-1] is exactly
    dp[i][j] = dp[i-1][j] + dp[i][j-1].

    Time : O(m * n)
    Space: O(n)
    """
    row = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            row[j] += row[j - 1]
    return row[n - 1]


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
    assert climb_stairs(1) == 1
    assert climb_stairs(2) == 2
    assert climb_stairs(3) == 3
    assert climb_stairs(4) == 5
    assert climb_stairs(5) == 8
    assert climb_stairs(10) == 89
    assert climb_stairs(45) == 1836311903
    assert climb_stairs_memo(10) == 89
    print("Exercise 1 (climb_stairs): OK")

    # -- Exercise 2 --
    assert coin_change([1, 2, 5], 11) == 3
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0
    assert coin_change([1, 2, 5], 0) == 0
    assert coin_change([1], 2) == 2
    assert coin_change([2, 5, 10, 1], 27) == 4
    assert coin_change([186, 419, 83, 408], 6249) == 20
    print("Exercise 2 (coin_change): OK")

    # -- Exercise 3 --
    assert unique_paths(3, 7) == 28
    assert unique_paths(3, 2) == 3
    assert unique_paths(1, 1) == 1
    assert unique_paths(1, 10) == 1
    assert unique_paths(10, 1) == 1
    assert unique_paths(7, 3) == 28
    assert unique_paths(10, 10) == 48620
    assert unique_paths_rolling(3, 7) == 28
    assert unique_paths_rolling(10, 10) == 48620
    print("Exercise 3 (unique_paths): OK")

    print("\nAll Day 10 solutions pass!")

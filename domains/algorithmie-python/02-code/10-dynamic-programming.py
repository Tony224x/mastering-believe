"""
Day 10 — Dynamic Programming: memoization, tabulation, classic patterns
Run: python domains/algorithmie-python/02-code/10-dynamic-programming.py
"""

from functools import lru_cache


# =============================================================================
# SECTION 1: FIBONACCI — the gateway drug to DP
# =============================================================================

def fib_naive(n):
    """
    Exponential blow-up: O(2^n).
    Never use this for n > 30.
    """
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n, memo=None):
    """
    Top-down memoization: cache results in a dict.
    Each fib(i) computed once -> O(n) time, O(n) space.
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n < 2:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def fib_tab(n):
    """
    Bottom-up tabulation: iterative fill.
    """
    if n < 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


def fib_opt(n):
    """
    Bottom-up with O(1) space — we only need the last 2 values.
    """
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


# =============================================================================
# SECTION 2: CLIMBING STAIRS + HOUSE ROBBER (linear DP)
# =============================================================================

def climb_stairs(n):
    """
    Same recurrence as Fibonacci (dp[i] = dp[i-1] + dp[i-2]).
    Reason: to reach step i, you come from step i-1 (1 step) or i-2 (2 steps).
    """
    if n < 2:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def rob(nums):
    """
    House Robber: cannot rob two adjacent houses.
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    Translation: at house i, either skip it (dp[i-1]) or take it (dp[i-2] + nums[i]).
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2 = nums[0]                    # dp[i-2]
    prev1 = max(nums[0], nums[1])      # dp[i-1]
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1


# =============================================================================
# SECTION 3: KNAPSACK 0/1
# =============================================================================

def knapsack_01(weights, values, W):
    """
    Classic 2D DP.
    dp[i][w] = max value using the first i items with capacity w.
    Recurrence:
        - If weights[i-1] > w: can't take it, dp[i][w] = dp[i-1][w]
        - Else: dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(
                    dp[i - 1][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1],
                )
    return dp[n][W]


def knapsack_01_1d(weights, values, W):
    """
    Space-optimized 1D version.
    CRITICAL: we iterate w from right to left so that dp[w - weight] still
    refers to the PREVIOUS item's row. Going left-to-right would allow the
    same item to be taken multiple times (turning it into unbounded knapsack).
    """
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]


# =============================================================================
# SECTION 4: COIN CHANGE
# =============================================================================

def coin_change_min(coins, amount):
    """
    Minimum coins to make `amount`. Unlimited coins available.
    dp[a] = min coins to make a
    Recurrence: dp[a] = min(dp[a - c] + 1 for c in coins if c <= a)
    """
    INF = float('inf')
    dp = [INF] * (amount + 1)
    dp[0] = 0                          # 0 coins to make amount 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


def coin_change_combinations(amount, coins):
    """
    Number of COMBINATIONS (order doesn't matter) to reach `amount`.
    KEY: outer loop is coins, inner loop is amount.
    This ordering counts each coin only AFTER earlier coins so permutations
    aren't double-counted.
    """
    dp = [0] * (amount + 1)
    dp[0] = 1                          # One way to make 0: empty set
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]


# =============================================================================
# SECTION 5: LONGEST COMMON SUBSEQUENCE
# =============================================================================

def lcs(s1, s2):
    """
    dp[i][j] = LCS length of s1[:i] and s2[:j].
    Recurrence:
        - If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
        - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# =============================================================================
# SECTION 6: LONGEST INCREASING SUBSEQUENCE
# =============================================================================

def lis_dp(nums):
    """
    O(n^2) DP.
    dp[i] = length of LIS ending at index i (inclusive).
    """
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lis_patience(nums):
    """
    O(n log n) via patience sorting.
    tails[i] = the smallest possible tail of an increasing subsequence of length i+1.
    For each num, binary-search where it fits; if it extends, append; else replace.
    The length of tails is the LIS length.
    """
    import bisect
    tails = []
    for num in nums:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    return len(tails)


# =============================================================================
# SECTION 7: UNIQUE PATHS (grid DP)
# =============================================================================

def unique_paths(m, n):
    """
    Robot in an m x n grid moving only right or down.
    Number of distinct paths from (0,0) to (m-1, n-1).

    dp[i][j] = number of paths to cell (i, j)
    Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]
    """
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]


def unique_paths_rolling(m, n):
    """
    Space-optimized: we only need the previous row.
    """
    row = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            row[j] += row[j - 1]       # row[j-1] is "left" (already updated this row)
                                        # row[j] (before +=) is "above" from previous row
    return row[n - 1]


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("Fibonacci:")
    for fn in (fib_memo, fib_tab, fib_opt):
        print(f"  {fn.__name__}(20) = {fn(20)}")

    print("\nClimbing stairs(5):", climb_stairs(5))
    print("House robber([2,7,9,3,1]):", rob([2, 7, 9, 3, 1]))

    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    print(f"\nKnapsack 0/1 (W=7): 2D={knapsack_01(weights, values, 7)} "
          f"1D={knapsack_01_1d(weights, values, 7)}")

    print("\nCoin change min for 11 with [1,2,5]:",
          coin_change_min([1, 2, 5], 11))
    print("Coin change combinations for 5 with [1,2,5]:",
          coin_change_combinations(5, [1, 2, 5]))

    print("\nLCS('abcde', 'ace'):", lcs("abcde", "ace"))

    print("\nLIS([10,9,2,5,3,7,101,18]):")
    print("  DP       :", lis_dp([10, 9, 2, 5, 3, 7, 101, 18]))
    print("  Patience :", lis_patience([10, 9, 2, 5, 3, 7, 101, 18]))

    print("\nUnique paths 3x7:", unique_paths(3, 7))
    print("Unique paths 3x7 (rolling):", unique_paths_rolling(3, 7))

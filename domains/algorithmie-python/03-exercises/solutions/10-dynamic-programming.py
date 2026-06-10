"""
Solutions — Day 10 Dynamic Programming (easy, medium and hard exercises).
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
# Exercise 4 (Medium): House Robber II (circular street)
# =============================================================================

def rob_line(nums):
    """
    Linear House Robber in O(1) space.
    rob_prev = best including consideration up to i-1
    rob_prev2 = best up to i-2
    At each house: take it (rob_prev2 + num) or skip it (rob_prev).
    """
    rob_prev2, rob_prev = 0, 0
    for num in nums:
        rob_prev2, rob_prev = rob_prev, max(rob_prev, rob_prev2 + num)
    return rob_prev


def rob_circular(nums):
    """
    Reduce the circular problem to TWO linear ones.

    KEY ARGUMENT:
    - House 0 and house n-1 are adjacent on the circle, so at most one of
      them is robbed. Every optimal solution therefore lives entirely in
      nums[:-1] (house n-1 excluded) OR in nums[1:] (house 0 excluded).
      Take the max of both linear solutions.

    THE len == 1 TRAP:
    - Both slices are empty for a single house, returning 0 instead of
      nums[0]. Handle it explicitly.

    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    return max(rob_line(nums[:-1]), rob_line(nums[1:]))


# =============================================================================
# Exercise 5 (Medium): Longest Increasing Subsequence (O(n^2) and O(n log n))
# =============================================================================

def lis_dp(nums):
    """
    Classic O(n^2) DP.
    dp[i] = length of the best increasing subsequence ENDING at i.
    Transition: extend any j < i with nums[j] < nums[i] (strict).
    """
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:           # Strict: duplicates can't chain
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lis_fast(nums):
    """
    Patience sorting — O(n log n).

    INVARIANT of tails:
    - tails[k] = the SMALLEST possible tail value of an increasing
      subsequence of length k+1 seen so far. tails is always sorted.
    - tails is NOT an actual subsequence; only its LENGTH is the answer.

    UPDATE RULE:
    - For each num, find the leftmost slot >= num (bisect_left) and
      replace it; if num is bigger than every tail, append (the LIS just
      got longer).

    WHY bisect_left (not bisect_right):
    - Strictly increasing: an equal value must REPLACE the existing tail,
      not extend after it ([7,7,7] must stay at length 1).
    """
    import bisect

    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)               # Extends the longest subsequence
        else:
            tails[pos] = num                # Better (smaller) tail for length pos+1
    return len(tails)


# =============================================================================
# Exercise 6 (Medium): Coin Change II — combinations vs permutations
# =============================================================================

def change(amount, coins):
    """
    Count COMBINATIONS: outer loop over coins.

    WHY THE LOOP ORDER MATTERS:
    - With coins outside, dp[a] only accumulates ways that use coins in a
      CANONICAL order (all 1s decided before any 2 is considered). Each
      multiset of coins is counted exactly once: 1+2 and 2+1 are the same
      path "use one 1 (during coin=1 pass), then one 2 (during coin=2)".
    - With amount outside (see change_permutations), every coin can be
      appended at every step, so each ORDERING is counted separately.

    Base case: dp[0] = 1 — the empty combination makes amount 0.

    Time: O(amount * len(coins)), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:                      # Coins outside → combinations
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]


def change_permutations(amount, coins):
    """
    Counterpart with amount outside → counts ORDERED sequences.
    For amount=3, coins=[1,2]: 1+1+1, 1+2, 2+1 → 3 (vs 2 combinations).
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    for a in range(1, amount + 1):          # Amount outside → permutations
        for coin in coins:
            if coin <= a:
                dp[a] += dp[a - coin]
    return dp[amount]


# =============================================================================
# Exercise 7 (Hard): Edit Distance (2D + space-optimized 1D)
# =============================================================================

def min_distance(word1, word2):
    """
    Levenshtein distance, full 2D table.

    dp[i][j] = min operations to turn word1[:i] into word2[:j].

    BASE CASES (often botched):
    - dp[i][0] = i — delete all i characters to reach the empty string.
    - dp[0][j] = j — insert all j characters starting from empty.

    TRANSITIONS (when characters differ, each option costs 1):
    - dp[i-1][j-1] + 1 → REPLACE word1[i-1] by word2[j-1]
    - dp[i-1][j]   + 1 → DELETE word1[i-1]
    - dp[i][j-1]   + 1 → INSERT word2[j-1]

    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]     # Free: characters match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],           # replace
                    dp[i - 1][j],               # delete
                    dp[i][j - 1],               # insert
                )
    return dp[m][n]


def min_distance_1d(word1, word2):
    """
    Same recurrence with ONE row — O(min(m, n)) space.

    THE DIAGONAL TRICK:
    - row[j] currently holds dp[i-1][j]; writing the new value destroys
      dp[i-1][j] which is the DIAGONAL for column j+1. Save it in a
      variable before overwriting.
    """
    # Iterate over the shorter word as columns → O(min(m, n)) memory
    if len(word2) > len(word1):
        word1, word2 = word2, word1

    m, n = len(word1), len(word2)
    row = list(range(n + 1))                # dp[0][j] = j

    for i in range(1, m + 1):
        diagonal = row[0]                   # dp[i-1][0]
        row[0] = i                          # dp[i][0]
        for j in range(1, n + 1):
            saved = row[j]                  # dp[i-1][j], next diagonal
            if word1[i - 1] == word2[j - 1]:
                row[j] = diagonal
            else:
                row[j] = 1 + min(diagonal, row[j], row[j - 1])
            diagonal = saved
    return row[n]


# =============================================================================
# Exercise 8 (Hard): 0/1 Knapsack — value + reconstruction + 1D
# =============================================================================

def knapsack_2d(weights, values, capacity):
    """
    Full table + backtracking to recover the chosen items.

    dp[i][w] = best value using the first i items with capacity w.

    RECONSTRUCTION:
    - Walk back from dp[n][capacity]. If dp[i][w] != dp[i-1][w], skipping
      item i-1 cannot explain the value → item i-1 was taken: record it
      and subtract its weight. Otherwise move up one row.

    Time: O(n * W), Space: O(n * W) (needed to reconstruct)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i, v_i = weights[i - 1], values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                 # Skip item i-1
            if w_i <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - w_i] + v_i)  # Take it

    # Backtrack to find WHICH items were taken
    chosen = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:                # Value changed → item taken
            chosen.append(i - 1)
            w -= weights[i - 1]
    chosen.reverse()
    return dp[n][capacity], chosen


def knapsack_1d(weights, values, capacity):
    """
    1D optimization — O(W) space.

    WHY THE CAPACITY LOOP MUST BE DESCENDING:
    - dp[w - weight] must still hold the PREVIOUS item-row value.
      Ascending order would read a cell already updated with the CURRENT
      item, allowing it to be taken twice (that's the UNBOUNDED knapsack).
      Counterexample: weights=[3], values=[5], capacity=9 → ascending
      gives 15 (item used 3 times), descending correctly gives 5.
    """
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for w in range(capacity, weight - 1, -1):   # DESCENDING — 0/1 guarantee
            dp[w] = max(dp[w], dp[w - weight] + value)
    return dp[capacity]


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

    # -- Exercise 4 --
    assert rob_circular([2, 3, 2]) == 3
    assert rob_circular([1, 2, 3, 1]) == 4
    assert rob_circular([1, 2, 3]) == 3
    assert rob_circular([5]) == 5
    assert rob_circular([]) == 0
    assert rob_circular([1, 2]) == 2
    assert rob_circular([200, 3, 140, 20, 10]) == 340
    print("Exercise 4 (rob_circular): OK")

    # -- Exercise 5 --
    for lis in (lis_dp, lis_fast):
        assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
        assert lis([0, 1, 0, 3, 2, 3]) == 4
        assert lis([7, 7, 7, 7]) == 1
        assert lis([]) == 0
        assert lis([5]) == 1
        assert lis([1, 2, 3, 4]) == 4
        assert lis([4, 3, 2, 1]) == 1
    import random
    for _ in range(100):
        arr = [random.randint(0, 20) for _ in range(random.randint(0, 30))]
        assert lis_dp(arr) == lis_fast(arr), arr
    print("Exercise 5 (lis_dp / lis_fast): OK")

    # -- Exercise 6 --
    assert change(5, [1, 2, 5]) == 4
    assert change(3, [2]) == 0
    assert change(0, [7]) == 1
    assert change(0, []) == 1
    assert change(10, [10]) == 1
    assert change(3, [1, 2]) == 2
    assert change_permutations(3, [1, 2]) == 3   # Order matters → more ways
    assert change(500, [3, 5, 7, 8, 9, 10, 11]) == 35502874
    print("Exercise 6 (change): OK")

    # -- Exercise 7 --
    for f in (min_distance, min_distance_1d):
        assert f("horse", "ros") == 3
        assert f("intention", "execution") == 5
        assert f("", "") == 0
        assert f("", "abc") == 3
        assert f("abc", "") == 3
        assert f("abc", "abc") == 0
        assert f("a", "b") == 1
        assert f("ab", "ba") == 2
    for _ in range(100):
        w1 = "".join(random.choices("ab", k=random.randint(0, 10)))
        w2 = "".join(random.choices("ab", k=random.randint(0, 10)))
        assert min_distance(w1, w2) == min_distance_1d(w1, w2), (w1, w2)
    print("Exercise 7 (min_distance): OK")

    # -- Exercise 8 --
    best, chosen = knapsack_2d([1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert best == 9
    assert sorted(chosen) == [1, 2]
    best, chosen = knapsack_2d([2, 2, 2], [3, 3, 3], 4)
    assert best == 6 and len(chosen) == 2
    assert knapsack_2d([], [], 10) == (0, [])
    assert knapsack_2d([5], [10], 4) == (0, [])
    assert knapsack_1d([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    assert knapsack_1d([2, 2, 2], [3, 3, 3], 4) == 6
    assert knapsack_1d([3], [5], 9) == 5     # 0/1: item NOT reused
    for _ in range(50):
        n = random.randint(0, 10)
        ws = [random.randint(1, 10) for _ in range(n)]
        vs = [random.randint(1, 20) for _ in range(n)]
        cap = random.randint(0, 30)
        best2d, chosen = knapsack_2d(ws, vs, cap)
        assert best2d == knapsack_1d(ws, vs, cap)
        assert sum(ws[i] for i in chosen) <= cap
        assert sum(vs[i] for i in chosen) == best2d
    print("Exercise 8 (knapsack): OK")

    print("\nAll Day 10 solutions pass!")

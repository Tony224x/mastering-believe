"""
Solutions — Day 11: DP Avance & Greedy (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/11-dp-avance-greedy-hard.py

Each solution is numbered to match the exercise file (03-hard/11-dp-avance-greedy.md).
All solutions are verified with assertions at the end.
"""

from collections import Counter


# =============================================================================
# EXERCISE 7 (Hard): Stock IV (at most k transactions) — state machine DP
# =============================================================================

def max_profit_k(k: int, prices: list[int]) -> int:
    """
    AT MOST k transactions.

    SHORTCUT: if k >= len(prices)//2, k is effectively unlimited -> greedy:
    sum every positive consecutive rise (LeetCode 122).

    DP otherwise:
    - buy[t]  = best profit after the t-th BUY (we hold a share)
    - sell[t] = best profit after the t-th SELL (we hold nothing)
    For each price:
        buy[t]  = max(buy[t],  sell[t-1] - price)
        sell[t] = max(sell[t], buy[t]   + price)
    sell[0] is conceptually 0 (no transaction done yet).

    Time: O(n * k) (or O(n) in the unlimited branch), Space: O(k)
    """
    n = len(prices)
    if n == 0 or k == 0:
        return 0

    # Unlimited-transactions shortcut
    if k >= n // 2:
        return sum(max(0, prices[i + 1] - prices[i]) for i in range(n - 1))

    buy = [float('-inf')] * (k + 1)        # buy[0] unused (no 0-th buy)
    sell = [0] * (k + 1)                    # sell[0] = 0 baseline

    for price in prices:
        for t in range(1, k + 1):
            buy[t] = max(buy[t], sell[t - 1] - price)
            sell[t] = max(sell[t], buy[t] + price)

    return sell[k]


def _max_profit_k_brute(k: int, prices: list[int]) -> int:
    """Memoized brute force oracle: state = (index, transactions_left, holding)."""
    from functools import lru_cache
    n = len(prices)

    @lru_cache(maxsize=None)
    def dp(i: int, left: int, holding: bool) -> int:
        if i == n or left == 0:
            return 0
        best = dp(i + 1, left, holding)            # do nothing today
        if holding:
            best = max(best, prices[i] + dp(i + 1, left - 1, False))  # sell
        else:
            best = max(best, -prices[i] + dp(i + 1, left, True))      # buy
        return best

    return dp(0, k, False)


def test_exercise_7():
    print("\nExercise 7: Stock IV (k transactions)")

    cases = [
        (2, [2, 4, 1], 2),
        (2, [3, 2, 6, 5, 0, 3], 7),
        (0, [1, 3, 5], 0),
        (2, [], 0),
        (1, [1, 2, 3, 4, 5], 4),
        (100, [1, 2, 3, 4, 5], 4),
        (2, [1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 13),
        (3, [5, 4, 3, 2, 1], 0),
    ]
    for k, prices, expected in cases:
        got = max_profit_k(k, prices)
        assert got == expected, f"k={k}, {prices}: got {got}, want {expected}"
        # Cross-check against the memoized brute force on small inputs
        if len(prices) <= 12:
            assert got == _max_profit_k_brute(k, prices)

    print("  PASS — all test cases (incl. brute-force cross-check)")


# =============================================================================
# EXERCISE 8 (Hard): Burst Balloons — interval DP on the LAST balloon
# =============================================================================

def max_coins(nums: list[int]) -> int:
    """
    Add virtual 1s at both ends: arr = [1] + nums + [1].
    dp[i][j] = max coins from bursting all balloons strictly between i and j.

    Think of the LAST balloon m burst in the open interval (i, j): at that
    moment its neighbors are exactly i and j (everything else is gone), so it
    yields arr[i]*arr[m]*arr[j], and the two sides are already solved:
        dp[i][j] = max over i<m<j of dp[i][m] + arr[i]*arr[m]*arr[j] + dp[m][j]

    Fill by increasing interval length so sub-intervals are ready.

    Time: O(n^3), Space: O(n^2)
    """
    if not nums:
        return 0

    arr = [1] + nums + [1]
    size = len(arr)
    dp = [[0] * size for _ in range(size)]

    # length = distance between the exclusive bounds i and j
    for length in range(2, size):
        for i in range(size - length):
            j = i + length
            best = 0
            for m in range(i + 1, j):
                coins = dp[i][m] + arr[i] * arr[m] * arr[j] + dp[m][j]
                if coins > best:
                    best = coins
            dp[i][j] = best

    return dp[0][size - 1]


def test_exercise_8():
    print("\nExercise 8: Burst Balloons")

    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10
    assert max_coins([]) == 0
    assert max_coins([5]) == 5
    assert max_coins([7, 9, 8, 0, 7, 1, 3, 5, 5, 2, 3]) == 1654
    assert max_coins([1]) == 1
    assert max_coins([9, 76, 64, 21]) == 116718

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Task Scheduler — greedy closed-form
# =============================================================================

def least_interval(tasks: list[str], n: int) -> int:
    """
    GREEDY closed form. Let f_max be the highest task frequency and n_max the
    number of tasks sharing that frequency. Arrange (f_max - 1) frames of size
    (n + 1), each led by the most frequent task, then append the n_max tasks
    that hit the final slot:
        frame = (f_max - 1) * (n + 1) + n_max

    If there are many distinct tasks, no idle is needed and the answer is just
    len(tasks). Hence:
        answer = max(frame, len(tasks))

    Time: O(T) (T = number of tasks), Space: O(1) (at most 26 letters)
    """
    if not tasks:
        return 0

    counts = Counter(tasks)
    f_max = max(counts.values())
    n_max = sum(1 for c in counts.values() if c == f_max)

    frame = (f_max - 1) * (n + 1) + n_max
    return max(frame, len(tasks))


# Oracle de simulation retire : il etait bugge (sur-comptait l'idle de fin de frame).
# `least_interval` ci-dessus (formule (f_max-1)*(n+1)+n_max, bornee par len(tasks))
# est l'implementation de reference correcte.


def test_exercise_9():
    print("\nExercise 9: Task Scheduler")

    assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
    assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
    assert least_interval(["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"], 4) == 10
    assert least_interval(["A"], 2) == 1
    assert least_interval(
        ["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2) == 16
    assert least_interval(["A", "A", "A", "B", "B", "B"], 3) == 10
    # A and B both appear twice; with cooldown n=2 the optimal schedule is
    # "A B _ A B" (same task needs n+1=3 spacing), so the answer is 5, not 4.
    assert least_interval(["A", "B", "A", "B"], 2) == 5

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 11: DP Avance & Greedy (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

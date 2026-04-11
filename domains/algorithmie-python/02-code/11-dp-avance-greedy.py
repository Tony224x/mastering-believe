"""
Day 11 — Advanced DP & Greedy: state machines, interval DP, classic greedy
Run: python domains/algorithmie-python/02-code/11-dp-avance-greedy.py
"""


# =============================================================================
# SECTION 1: STATE MACHINE DP — stock problems
# =============================================================================

def max_profit_unlimited(prices):
    """
    Best Time to Buy and Sell Stock II: unlimited transactions.
    State machine with two states:
      - cash: we currently hold no stock
      - hold: we currently hold one stock
    Transitions each day:
      cash_new = max(cash, hold + price)    # do nothing, or sell
      hold_new = max(hold, cash - price)    # do nothing, or buy
    Answer: cash at the end (you can't end holding a share).
    """
    cash, hold = 0, float('-inf')
    for p in prices:
        cash, hold = max(cash, hold + p), max(hold, cash - p)
    return cash


def max_profit_cooldown(prices):
    """
    Buy and Sell with 1-day cooldown after selling.
    Three states: hold (holding), cash (just sold, in cooldown), rest (free).
    """
    if not prices:
        return 0
    hold, cash, rest = -prices[0], 0, 0
    for p in prices[1:]:
        prev_cash = cash
        cash = hold + p               # sell -> move to cash, triggers cooldown
        hold = max(hold, rest - p)    # stay holding, or buy from rest
        rest = max(rest, prev_cash)   # stay resting, or cooldown ended from cash
    return max(cash, rest)            # not holding at the end is always better


# =============================================================================
# SECTION 2: INTERVAL DP — longest palindromic substring
# =============================================================================

def longest_palindromic_substring(s):
    """
    Interval DP filled by increasing length.
    dp[i][j] = True if s[i..j] is a palindrome.
    Base: single chars are palindromes; same-char pairs are palindromes.
    Recurrence: dp[i][j] = (s[i] == s[j]) AND dp[i+1][j-1]

    CRITICAL: we fill by length so dp[i+1][j-1] is always computed before dp[i][j].

    Time : O(n^2)
    Space: O(n^2)
    """
    n = len(s)
    if n == 0:
        return ""
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    # Length 1 palindromes
    for i in range(n):
        dp[i][i] = True
    # Length 2 and above
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_len:
                        start, max_len = i, length
    return s[start:start + max_len]


def min_palindrome_cuts(s):
    """
    Minimum cuts to partition s into palindromic substrings.
    Step 1: precompute is_palin[i][j] with interval DP.
    Step 2: dp[i] = min cuts for s[:i+1].
      If s[:i+1] is already a palindrome, dp[i] = 0.
      Else, dp[i] = min(dp[j-1] + 1) for j where s[j..i] is a palindrome.
    """
    n = len(s)
    if n == 0:
        return 0
    is_palin = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palin[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or is_palin[i + 1][j - 1]):
                is_palin[i][j] = True

    dp = [0] * n
    for i in range(n):
        if is_palin[0][i]:
            dp[i] = 0
            continue
        dp[i] = i                      # Worst case: cut before every char
        for j in range(1, i + 1):
            if is_palin[j][i]:
                dp[i] = min(dp[i], dp[j - 1] + 1)
    return dp[n - 1]


# =============================================================================
# SECTION 3: PARTITION DP — word break
# =============================================================================

def word_break(s, word_dict):
    """
    dp[i] = True if s[:i] can be segmented into dictionary words.
    dp[0] = True (empty string).
    dp[i] = True if exists j < i with dp[j] AND s[j:i] in dict.
    """
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]


# =============================================================================
# SECTION 4: GREEDY — interval scheduling
# =============================================================================

def max_non_overlapping(intervals):
    """
    Maximum number of non-overlapping intervals.
    GREEDY: sort by END time, then pick each interval whose start is
    >= the end of the last picked.

    WHY sort by end: the interval that ends earliest leaves the most room
    for subsequent intervals. Exchange argument: if any other choice were
    optimal, we could replace its first pick with the earliest-ending
    interval without hurting the count.

    Time : O(n log n) — sorting dominates
    Space: O(1)
    """
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 1
    last_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= last_end:
            count += 1
            last_end = end
    return count


def erase_overlap_intervals(intervals):
    """
    Minimum number of intervals to remove to make the rest non-overlapping.
    = n - max_non_overlapping.
    """
    return len(intervals) - max_non_overlapping(intervals)


# =============================================================================
# SECTION 5: GREEDY — jump game
# =============================================================================

def can_jump(nums):
    """
    True if we can reach the last index.
    Greedy: track the farthest reachable index as we scan left to right.
    If at any point i > max_reach, we're stuck.
    """
    max_reach = 0
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    return True


def jump(nums):
    """
    Minimum number of jumps to reach the last index.
    Greedy BFS: maintain a 'frontier' [current_end, farthest]. When i
    reaches current_end, we commit to a jump and extend the frontier.
    """
    if len(nums) <= 1:
        return 0
    jumps, current_end, farthest = 0, 0, 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:           # End of current reach, must jump
            jumps += 1
            current_end = farthest
    return jumps


# =============================================================================
# SECTION 6: GREEDY — gas station
# =============================================================================

def can_complete_circuit(gas, cost):
    """
    Return the starting gas station index where you can complete the circuit,
    or -1 if impossible.

    Greedy O(n):
      - total_tank tracks the global balance; if < 0, impossible.
      - curr_tank tracks the balance from the current candidate start.
      - When curr_tank goes negative at i, RESET start to i+1 and curr_tank to 0.

    PROOF (exchange argument): if starting at s fails at index i (curr_tank < 0),
    then no index j in [s, i] can be a valid starting point, because starting
    from j means less accumulated gas than starting from s (since s -> j was
    all non-negative). So we can safely skip to i+1.
    """
    total_tank, curr_tank, start = 0, 0, 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        curr_tank += diff
        if curr_tank < 0:
            start = i + 1
            curr_tank = 0
    return start if total_tank >= 0 else -1


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("Stock unlimited [7,1,5,3,6,4]:",
          max_profit_unlimited([7, 1, 5, 3, 6, 4]))        # 7
    print("Stock cooldown [1,2,3,0,2]:",
          max_profit_cooldown([1, 2, 3, 0, 2]))            # 3

    print("\nLongest palindromic substring in 'babad':",
          longest_palindromic_substring("babad"))
    print("Min palindrome cuts 'aab':", min_palindrome_cuts("aab"))  # 1

    print("\nWord break 'leetcode' ['leet','code']:",
          word_break("leetcode", ["leet", "code"]))        # True

    print("\nMax non-overlapping intervals:")
    print(" ", max_non_overlapping([[1, 3], [2, 4], [3, 5]]))  # 2
    print("Erase overlap intervals:")
    print(" ", erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]))  # 1

    print("\nCan jump [2,3,1,1,4]:", can_jump([2, 3, 1, 1, 4]))  # True
    print("Can jump [3,2,1,0,4]:", can_jump([3, 2, 1, 0, 4]))    # False
    print("Min jumps [2,3,1,1,4]:", jump([2, 3, 1, 1, 4]))       # 2

    print("\nGas station:")
    print(" ", can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))  # 3
    print(" ", can_complete_circuit([2, 3, 4], [3, 4, 3]))              # -1

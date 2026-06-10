"""
Solutions — Day 11 DP Avance & Greedy (easy, medium and hard exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/11-dp-avance-greedy.py
"""


# =============================================================================
# Exercise 1: Max Profit (unlimited transactions)
# =============================================================================

def max_profit(prices):
    """
    State machine with two states.
    cash = max profit today if we DO NOT hold a share
    hold = max profit today if we DO    hold a share

    Transitions:
      new_cash = max(cash, hold + p)   # stay in cash OR sell today
      new_hold = max(hold, cash - p)   # stay in hold OR buy today

    Initial: cash=0, hold=-inf (we can't start holding out of nothing).
    Answer: cash at the end (no point in holding past the last day).

    Time : O(n)
    Space: O(1)
    """
    cash, hold = 0, float('-inf')
    for p in prices:
        cash, hold = max(cash, hold + p), max(hold, cash - p)
    return cash


# Equivalent formulation: sum up every positive daily delta.
# WHY this works: any multi-day up-trend can be decomposed into consecutive
# daily deltas, and we can buy/sell on every up-day without restriction.
def max_profit_delta(prices):
    return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices)))


# =============================================================================
# Exercise 2: Erase Overlap Intervals
# =============================================================================

def erase_overlap_intervals(intervals):
    """
    Greedy: max non-overlapping = n - min_to_remove.
    Sort by END ascending. Walk through: keep intervals whose start is
    >= the last kept end.

    WHY sort by end: earliest-ending interval frees the schedule the most.
    Any solution that doesn't pick the earliest-ending interval can be
    rewritten (exchange argument) to use it without losing count.

    Time : O(n log n)
    Space: O(1) (ignoring sort's allocation)
    """
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    kept = 1
    last_end = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= last_end:          # No overlap (touching endpoints OK)
            kept += 1
            last_end = end
    return len(intervals) - kept


# =============================================================================
# Exercise 3: Can Jump
# =============================================================================

def can_jump(nums):
    """
    Greedy one-pass.
    max_reach = the farthest index we know we can reach so far.
    At each step i, if i > max_reach we're stuck (no prior step could
    get us here). Otherwise extend max_reach with i + nums[i].

    Early exit: if max_reach >= len(nums) - 1 we can stop.

    Time : O(n)
    Space: O(1)
    """
    max_reach = 0
    last = len(nums) - 1
    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
        if max_reach >= last:
            return True
    return True



# =============================================================================
# Exercise 4 (Medium): Best Time to Buy and Sell Stock with Cooldown
# =============================================================================

def max_profit_cooldown(prices):
    """
    Three-state machine, O(1) space.

    STATES (best cash achievable while being in that state today):
    - hold: currently holding one share
    - sold: sold TODAY (tomorrow is forced cooldown)
    - rest: holding nothing, free to buy

    TRANSITIONS (the cooldown lives in the rest <- sold edge):
    - hold = max(hold, rest - price)   # buy only allowed from rest
    - sold = hold + price              # sell what we hold
    - rest = max(rest, sold)           # yesterday's sale becomes "rested" today

    INITIALIZATION TRAP:
    - hold starts at -prices[0] (buying the first day), NOT 0; sold/rest
      start at 0/-inf equivalents. Using 0 for hold would mean getting a
      share for free.

    ANSWER: max(sold, rest) — ending while still holding a share is
    never optimal (the buy price was paid for nothing).

    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0

    hold = -prices[0]                   # Bought on day 0
    sold = float("-inf")                # Can't have sold yet
    rest = 0                            # Doing nothing

    for price in prices[1:]:
        prev_hold, prev_sold, prev_rest = hold, sold, rest
        hold = max(prev_hold, prev_rest - price)   # Buy only after resting
        sold = prev_hold + price                    # Sell today
        rest = max(prev_rest, prev_sold)            # Cooldown happens here

    return max(sold, rest)


# =============================================================================
# Exercise 5 (Medium): Word Break
# =============================================================================

def word_break(s, word_dict):
    """
    Partition DP: dp[i] = "s[:i] is segmentable".

    WHY DP (not greedy): taking the longest matching word first fails on
    "cars" with ["car", "ca", "rs"] — "car" leaves "s" unmatched while
    "ca" + "rs" works. DP explores all split points.

    PERF DETAILS:
    - set(word_dict): membership in O(1) instead of O(#words).
    - The inner loop only looks back max_len characters: a window longer
      than the longest dictionary word can never match.

    Time: O(n * max_len) with O(1) average substring hashing cost per
    lookup (substring construction is O(max_len)), Space: O(n)
    """
    words = set(word_dict)
    if not words:
        return s == ""
    max_len = max(len(w) for w in words)

    dp = [False] * (len(s) + 1)
    dp[0] = True                        # Empty prefix is trivially segmentable

    for i in range(1, len(s) + 1):
        # Only the last max_len characters can form a dictionary word
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break                   # One valid split is enough
    return dp[len(s)]


# =============================================================================
# Exercise 6 (Medium): Jump Game II (minimum jumps)
# =============================================================================

def jump(nums):
    """
    Greedy by windows — a disguised BFS over indices.

    BFS ANALOGY:
    - Level 0 = {index 0}. Level d = all indices first reachable with d
      jumps. current_end is the right edge of the current level; farthest
      is the right edge of the NEXT level, grown as we scan.
    - When i reaches current_end, the current level is exhausted: we are
      forced to take one more jump → jumps += 1, window slides.

    OFF-BY-ONE TRAP:
    - Iterate up to len(nums) - 2 included: landing exactly on the last
      index must NOT trigger an extra jump.

    Time: O(n), Space: O(1)
    """
    jumps = 0
    current_end = 0                     # Edge of the window reachable with `jumps`
    farthest = 0                        # Best reach seen inside the window

    for i in range(len(nums) - 1):      # Last index excluded — no ghost jump
        farthest = max(farthest, i + nums[i])
        if i == current_end:            # Window exhausted: must jump
            jumps += 1
            current_end = farthest

    return jumps


# =============================================================================
# Exercise 7 (Hard): Palindrome Partitioning II (min cuts)
# =============================================================================

def min_cut(s):
    """
    Two cooperating DP tables — O(n^2) total.

    TABLE 1 — is_palin[i][j]: s[i..j] is a palindrome.
    - Filled by increasing LENGTH so that is_palin[i+1][j-1] (a shorter
      span) is always ready. No slicing: each cell costs O(1), table O(n^2).
      The naive "s[i:j+1] == reversed" check inside the cut loop would
      cost O(n) per test → O(n^3) overall.

    TABLE 2 — cuts[i]: min cuts for the prefix s[:i+1].
    - If the whole prefix is a palindrome: 0 cuts (short-circuit).
    - Else: min over j of cuts[j] + 1 where s[j+1..i] is a palindrome
      (checked via table 1 in O(1)).

    Time: O(n^2), Space: O(n^2)
    """
    n = len(s)
    if n <= 1:
        return 0

    # Table 1: palindromes by increasing length
    is_palin = [[False] * n for _ in range(n)]
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length <= 2 or is_palin[i + 1][j - 1]):
                is_palin[i][j] = True

    # Table 2: minimum cuts per prefix
    cuts = [0] * n
    for i in range(n):
        if is_palin[0][i]:
            cuts[i] = 0                 # Whole prefix is a palindrome
            continue
        cuts[i] = min(cuts[j] + 1 for j in range(i) if is_palin[j + 1][i])

    return cuts[n - 1]


# =============================================================================
# Exercise 8 (Hard): Best Time to Buy and Sell Stock IV (k transactions)
# =============================================================================

def max_profit_k(k, prices):
    """
    Generalized state machine: 2k states.

    THE MANDATORY SHORTCUT:
    - A transaction needs >= 2 days (buy one day, sell another), so at
      most n // 2 transactions fit. If k >= n // 2 the cap is irrelevant:
      fall back to the unlimited-transactions greedy (sum of positive
      deltas) in O(n). Without this, k = 10^6 would make the DP loop
      O(n * k) explode for nothing.

    STATES (per transaction index t in 1..k):
    - hold[t]: best cash while holding a share during transaction t
    - cash[t]: best cash after CLOSING t transactions
    Transitions per price:
    - hold[t] = max(hold[t], cash[t-1] - price)   # open transaction t
    - cash[t] = max(cash[t], hold[t] + price)     # close transaction t

    SPECIAL CASES embedded: k=1 is Stock I, k=inf (shortcut) is Stock II.

    Time: O(n * min(k, n)), Space: O(k)
    """
    n = len(prices)
    if n < 2 or k == 0:
        return 0

    if k >= n // 2:
        # Unlimited transactions: harvest every positive daily delta
        return sum(max(prices[i + 1] - prices[i], 0) for i in range(n - 1))

    hold = [float("-inf")] * (k + 1)    # hold[0] unused (no transaction open)
    cash = [0] * (k + 1)                # cash[0] = 0: zero transactions closed

    for price in prices:
        for t in range(1, k + 1):
            hold[t] = max(hold[t], cash[t - 1] - price)
            cash[t] = max(cash[t], hold[t] + price)

    return cash[k]


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
    assert max_profit([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit([1, 2, 3, 4, 5]) == 4
    assert max_profit([7, 6, 4, 3, 1]) == 0
    assert max_profit([]) == 0
    assert max_profit([5]) == 0
    assert max_profit([2, 4, 1, 7]) == 8
    assert max_profit_delta([7, 1, 5, 3, 6, 4]) == 7
    assert max_profit_delta([1, 2, 3, 4, 5]) == 4
    print("Exercise 1 (max_profit): OK")

    # -- Exercise 2 --
    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
    assert erase_overlap_intervals([]) == 0
    assert erase_overlap_intervals([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]) == 2
    print("Exercise 2 (erase_overlap_intervals): OK")

    # -- Exercise 3 --
    assert can_jump([2, 3, 1, 1, 4]) == True
    assert can_jump([3, 2, 1, 0, 4]) == False
    assert can_jump([0]) == True
    assert can_jump([1]) == True
    assert can_jump([0, 1]) == False
    assert can_jump([1, 0, 1, 0]) == False
    assert can_jump([2, 0, 0]) == True
    print("Exercise 3 (can_jump): OK")

    # -- Exercise 4 --
    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
    assert max_profit_cooldown([1]) == 0
    assert max_profit_cooldown([]) == 0
    assert max_profit_cooldown([1, 2]) == 1
    assert max_profit_cooldown([2, 1]) == 0
    assert max_profit_cooldown([1, 2, 4]) == 3
    assert max_profit_cooldown([2, 1, 4]) == 3
    assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6
    print("Exercise 4 (max_profit_cooldown): OK")

    # -- Exercise 5 --
    assert word_break("leetcode", ["leet", "code"]) == True
    assert word_break("applepenapple", ["apple", "pen"]) == True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) == False
    assert word_break("", ["a"]) == True
    assert word_break("a", []) == False
    assert word_break("aaaaaaa", ["aaaa", "aaa"]) == True
    assert word_break("aaaaaaab", ["aaaa", "aaa"]) == False
    assert word_break("cars", ["car", "ca", "rs"]) == True
    print("Exercise 5 (word_break): OK")

    # -- Exercise 6 --
    assert jump([2, 3, 1, 1, 4]) == 2
    assert jump([2, 3, 0, 1, 4]) == 2
    assert jump([0]) == 0
    assert jump([1, 2]) == 1
    assert jump([1, 1, 1, 1]) == 3
    assert jump([5, 1, 1, 1, 1]) == 1
    assert jump([1, 2, 1, 1, 1]) == 3
    assert jump([4, 1, 1, 3, 1, 1, 1]) == 2
    print("Exercise 6 (jump): OK")

    # -- Exercise 7 --
    assert min_cut("aab") == 1
    assert min_cut("a") == 0
    assert min_cut("ab") == 1
    assert min_cut("aaaa") == 0
    assert min_cut("abba") == 0
    assert min_cut("abcba") == 0
    assert min_cut("abcde") == 4
    assert min_cut("aabaa") == 0
    assert min_cut("aabba") == 1
    assert min_cut("cabababcbc") == 3
    # O(n^2) sanity check: n = 1200 must answer almost instantly
    import time
    start = time.perf_counter()
    assert min_cut("ab" * 600) == 1     # "a" | "bab...b" — one giant palindrome
    elapsed = time.perf_counter() - start
    assert elapsed < 5, f"min_cut too slow: {elapsed:.2f}s"
    print(f"Exercise 7 (min_cut): OK (n=1200 in {elapsed:.3f}s)")

    # -- Exercise 8 --
    assert max_profit_k(2, [2, 4, 1]) == 2
    assert max_profit_k(2, [3, 2, 6, 5, 0, 3]) == 7
    assert max_profit_k(1, [3, 2, 6, 5, 0, 3]) == 4
    assert max_profit_k(0, [1, 5]) == 0
    assert max_profit_k(2, []) == 0
    assert max_profit_k(2, [5, 4, 3, 2, 1]) == 0
    assert max_profit_k(100, [1, 2, 3, 4, 5]) == 4
    assert max_profit_k(1000000, list(range(1000)) * 2) == 1998   # Shortcut path
    assert max_profit_k(3, [1, 5, 2, 8, 3, 10]) == 17
    print("Exercise 8 (max_profit_k): OK")

    print("\nAll Day 11 solutions pass!")

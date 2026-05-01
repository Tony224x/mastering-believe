"""
Solutions — Day 11 DP Avance & Greedy (easy exercises).
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

    print("\nAll Day 11 solutions pass!")

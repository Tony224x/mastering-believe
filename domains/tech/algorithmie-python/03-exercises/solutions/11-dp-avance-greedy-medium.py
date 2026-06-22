"""
Solutions — Day 11: DP Avance & Greedy (MEDIUM)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/11-dp-avance-greedy-medium.py

Each solution is numbered to match the exercise file (02-medium/11-dp-avance-greedy.md).
All solutions are verified with assertions at the end.
"""


# =============================================================================
# EXERCISE 4 (Medium): Stock with Cooldown — 3-state machine DP
# =============================================================================

def max_profit_cooldown(prices: list[int]) -> int:
    """
    Three states tracked per day:
    - hold: max profit while currently HOLDING a share
    - sold: max profit on a day we JUST SOLD (so tomorrow is cooldown)
    - rest: max profit while holding nothing and NOT in cooldown

    TRANSITIONS (using yesterday's values):
    - hold = max(hold, rest - price)   # keep holding, or buy today (only from rest)
    - sold = hold + price              # sell today
    - rest = max(rest, sold)           # keep resting, or come out of cooldown

    The answer is the best "not holding" state: max(sold, rest).

    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0

    hold = float('-inf')               # cannot hold before buying anything
    sold = 0
    rest = 0

    for p in prices:
        prev_sold = sold
        sold = hold + p
        hold = max(hold, rest - p)
        rest = max(rest, prev_sold)

    return max(sold, rest)


def test_exercise_4():
    print("\nExercise 4: Stock with Cooldown")

    assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
    assert max_profit_cooldown([1]) == 0
    assert max_profit_cooldown([]) == 0
    assert max_profit_cooldown([1, 2, 4]) == 3
    assert max_profit_cooldown([5, 4, 3, 2, 1]) == 0
    assert max_profit_cooldown([2, 1, 4]) == 3
    assert max_profit_cooldown([6, 1, 3, 2, 4, 7]) == 6

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Non-overlapping Intervals — greedy by earliest end
# =============================================================================

def erase_overlap_intervals(intervals: list[list[int]]) -> int:
    """
    GREEDY: sort by END ascending, always keep the interval that finishes
    earliest because it leaves the most room for the rest (exchange argument).

    Walk through, tracking the last accepted end. If the next interval starts
    BEFORE that end, it overlaps -> remove it (count++). Otherwise accept it
    and advance the end.

    Touching endpoints ([1,2] and [2,3]) do NOT overlap, hence `start < end`.

    Time: O(n log n), Space: O(1)
    """
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[1])     # earliest finishing first
    removed = 0
    last_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start < last_end:               # strict: touching is allowed
            removed += 1                   # overlap -> drop this one
        else:
            last_end = end                 # accept, move the boundary
    return removed


def test_exercise_5():
    print("\nExercise 5: Non-overlapping Intervals")

    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
    assert erase_overlap_intervals([]) == 0
    assert erase_overlap_intervals([[1, 100], [11, 22], [1, 11], [2, 12]]) == 2
    assert erase_overlap_intervals([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]) == 2
    assert erase_overlap_intervals([[1, 2]]) == 0

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Gas Station — one-pass greedy
# =============================================================================

def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    TWO INVARIANTS:
    1. If sum(gas) < sum(cost), the loop is impossible -> -1.
    2. Otherwise a unique start exists. Scanning once, if the running tank
       drops below 0 at index i, no station in [start..i] can be the start
       (each gives less accumulated gas than `start`), so jump to i+1.

    EXCHANGE ARGUMENT: starting at any j in (start, i] accumulates strictly
    less than starting at `start`, which was already non-negative up to j-1,
    so j fails no later than i. Safe to skip to i+1.

    Time: O(n), Space: O(1)
    """
    if sum(gas) < sum(cost):
        return -1

    start = 0
    tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1                  # candidates [old_start..i] all fail
            tank = 0
    return start


def test_exercise_6():
    print("\nExercise 6: Gas Station")

    assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1
    assert can_complete_circuit([5], [4]) == 0
    assert can_complete_circuit([2], [2]) == 0
    assert can_complete_circuit([3, 1, 1], [1, 2, 2]) == 0
    assert can_complete_circuit([1, 1, 1], [2, 1, 1]) == -1
    assert can_complete_circuit([4, 5, 2, 6, 5, 3], [3, 2, 7, 3, 2, 9]) == -1

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 11: DP Avance & Greedy (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

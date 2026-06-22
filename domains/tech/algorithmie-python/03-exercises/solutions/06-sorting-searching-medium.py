"""
Solutions — Day 6: Sorting & Searching (MEDIUM)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/06-sorting-searching-medium.py

Each solution is numbered to match the exercise file (02-medium/06-sorting-searching.md).
All solutions are verified with assertions at the end.
"""

import math
from functools import cmp_to_key


# =============================================================================
# EXERCISE 4 (Medium): Find First and Last Position — lower/upper bound
# =============================================================================

def search_range(nums: list[int], target: int) -> list[int]:
    """
    Two binary searches: lower_bound (first index >= target) and upper_bound
    (first index > target). The occurrence range is [low, high-1].

    Time: O(log n), Space: O(1)
    """
    def lower_bound(t):
        lo, hi = 0, len(nums)          # hi = len, half-open interval
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    left = lower_bound(target)
    # target absent if index out of range or value mismatch
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    right = lower_bound(target + 1) - 1   # upper_bound(target) - 1
    return [left, right]


def test_exercise_4():
    print("\nExercise 4: Find First and Last Position")

    assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert search_range([], 0) == [-1, -1]
    assert search_range([1], 1) == [0, 0]
    assert search_range([2, 2, 2, 2], 2) == [0, 3]
    assert search_range([1, 2, 3], 3) == [2, 2]
    assert search_range([1, 2, 3], 1) == [0, 0]
    assert search_range([1, 4], 2) == [-1, -1]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Koko Eating Bananas — Binary search on answer
# =============================================================================

def min_eating_speed(piles: list[int], h: int) -> int:
    """
    Binary search the minimum eating speed k in [1, max(piles)].

    FEASIBILITY:
    - For speed k, hours needed = sum(ceil(pile / k)) — monotonically DECREASING
      in k. We want the smallest k whose hours_needed <= h.

    Time: O(n log(max(piles))), Space: O(1)
    """
    def hours_needed(k):
        return sum(math.ceil(pile / k) for pile in piles)

    lo, hi = 1, max(piles)             # Search space of possible speeds
    while lo < hi:
        mid = (lo + hi) // 2
        if hours_needed(mid) <= h:
            hi = mid                   # mid works; try smaller
        else:
            lo = mid + 1               # mid too slow; need faster
    return lo


def test_exercise_5():
    print("\nExercise 5: Koko Eating Bananas")

    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
    assert min_eating_speed([1], 1) == 1
    assert min_eating_speed([1000000000], 2) == 500000000
    assert min_eating_speed([3, 6, 7, 11], 4) == 11
    assert min_eating_speed([312884470], 312884469) == 2

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Largest Number — cmp_to_key custom comparator
# =============================================================================

def largest_number(nums: list[int]) -> str:
    """
    Sort string forms so that 'a' precedes 'b' iff a+b > b+a (string compare).

    EDGE CASE:
    - If the largest arrangement starts with '0', the whole list is zeros, so
      the answer is "0" (not "000").

    Time: O(n log n * k), Space: O(n) where k = max number length.
    """
    def compare(a, b):
        if a + b > b + a:
            return -1                  # a should come first
        if a + b < b + a:
            return 1
        return 0

    strs = [str(n) for n in nums]
    strs.sort(key=cmp_to_key(compare))
    result = ''.join(strs)
    return "0" if result[0] == "0" else result   # Collapse all-zeros case


def test_exercise_6():
    print("\nExercise 6: Largest Number")

    assert largest_number([10, 2]) == "210"
    assert largest_number([3, 30, 34, 5, 9]) == "9534330"
    assert largest_number([1]) == "1"
    assert largest_number([10]) == "10"
    assert largest_number([0, 0]) == "0"
    assert largest_number([0]) == "0"
    assert largest_number([34, 3, 32]) == "34332"
    assert largest_number([121, 12]) == "12121"

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 6: Sorting & Searching (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

"""
Solutions — Day 7: Sprint Complementaire (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/07-sprint-exercices-medium.py

New problems complementing the easy sprint P1-P10 (no duplication).
Matches the exercise file (02-medium/07-sprint-exercices.md).
All solutions are verified with assertions at the end.
"""

from collections import Counter


# =============================================================================
# EXERCISE 4 (Medium): Permutation in String — Fixed-size sliding window
# =============================================================================

def check_inclusion(s1: str, s2: str) -> bool:
    """
    Slide a fixed window of size len(s1) over s2, comparing frequency counters.

    O(1) UPDATE PER SLIDE:
    - Add the entering char, remove the leaving char; compare counters.
    - Counter equality ignores zero-count keys, so window == need works cleanly.

    Time: O(len(s2)), Space: O(1) (bounded alphabet)
    """
    if len(s1) > len(s2):
        return False

    need = Counter(s1)
    window = Counter(s2[:len(s1)])
    if window == need:
        return True

    for i in range(len(s1), len(s2)):
        window[s2[i]] += 1                       # New char enters the window
        left = s2[i - len(s1)]
        window[left] -= 1                        # Old char leaves the window
        if window[left] == 0:
            del window[left]                     # Keep counter clean for ==
        if window == need:
            return True

    return False


def test_exercise_4():
    print("\nExercise 4: Permutation in String")

    assert check_inclusion("ab", "eidbaooo") == True
    assert check_inclusion("ab", "eidboaoo") == False
    assert check_inclusion("a", "a") == True
    assert check_inclusion("abc", "ab") == False
    assert check_inclusion("adc", "dcda") == True
    assert check_inclusion("hello", "ooolleoooleh") == False
    assert check_inclusion("ab", "ab") == True

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): 3Sum Closest — Sort + two pointers
# =============================================================================

def three_sum_closest(nums: list[int], target: int) -> int:
    """
    Sort, then two-pointer scan per fixed index, tracking the closest sum.

    Time: O(n^2), Space: O(1) excluding the sort.
    """
    nums.sort()
    n = len(nums)
    best = nums[0] + nums[1] + nums[2]           # Any valid initial triple

    for i in range(n - 2):
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if abs(total - target) < abs(best - target):
                best = total
            if total == target:
                return total                      # Exact match: cannot do better
            elif total < target:
                lo += 1
            else:
                hi -= 1

    return best


def test_exercise_5():
    print("\nExercise 5: 3Sum Closest")

    assert three_sum_closest([-1, 2, 1, -4], 1) == 2
    assert three_sum_closest([0, 0, 0], 1) == 0
    assert three_sum_closest([1, 1, 1, 0], -100) == 2
    assert three_sum_closest([1, 2, 3], 6) == 6
    assert three_sum_closest([-3, -2, -5, 3, -4], -1) == -2
    assert three_sum_closest([0, 1, 2], 3) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Product of Array Except Self — Prefix/suffix products
# =============================================================================

def product_except_self(nums: list[int]) -> list[int]:
    """
    Two passes: prefix products into answer, then multiply by running suffix.

    NO DIVISION:
    - Handles zeros naturally (division would fail on a zero element).

    Time: O(n), Space: O(1) auxiliary (the output does not count).
    """
    n = len(nums)
    answer = [1] * n

    prefix = 1
    for i in range(n):
        answer[i] = prefix                        # Product of everything left of i
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix                       # Multiply by product right of i
        suffix *= nums[i]

    return answer


def test_exercise_6():
    print("\nExercise 6: Product of Array Except Self")

    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    assert product_except_self([2, 3]) == [3, 2]
    assert product_except_self([0, 0]) == [0, 0]
    assert product_except_self([5]) == [1]
    assert product_except_self([1, 0]) == [0, 1]

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 7: Sprint Complementaire (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

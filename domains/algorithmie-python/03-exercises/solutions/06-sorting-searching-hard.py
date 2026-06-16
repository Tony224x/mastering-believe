"""
Solutions — Day 6: Sorting & Searching (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/06-sorting-searching-hard.py

Each solution is numbered to match the exercise file (03-hard/06-sorting-searching.md).
All solutions are verified with assertions at the end.
"""


# =============================================================================
# EXERCISE 7 (Hard): Median of Two Sorted Arrays — Binary search on partition
# =============================================================================

def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Binary search the partition on the SMALLER array in O(log(min(m, n))).

    IDEA:
    - Put 'i' elements of nums1 and 'j = half - i' of nums2 on the left side.
    - Valid partition: max(left) <= min(right) on both sides, i.e.
      left1 <= right2 AND left2 <= right1 (using +/- inf at the borders).
    - Then the median comes from max(left1, left2) and min(right1, right2).

    Time: O(log(min(m, n))), Space: O(1)
    """
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1    # Ensure nums1 is the shorter array

    m, n = len(nums1), len(nums2)
    total = m + n
    half = total // 2

    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2             # Elements taken from nums1 on the left
        j = half - i                   # Elements taken from nums2 on the left

        left1 = nums1[i - 1] if i > 0 else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j - 1] if j > 0 else float('-inf')
        right2 = nums2[j] if j < n else float('inf')

        if left1 <= right2 and left2 <= right1:
            if total % 2:              # Odd total: median is the first right element
                return float(min(right1, right2))
            return (max(left1, left2) + min(right1, right2)) / 2
        elif left1 > right2:
            hi = i - 1                 # Took too many from nums1
        else:
            lo = i + 1                 # Took too few from nums1

    raise ValueError("Inputs are not sorted arrays")   # Unreachable on valid input


def test_exercise_7():
    print("\nExercise 7: Median of Two Sorted Arrays")

    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
    assert find_median_sorted_arrays([], [1]) == 1.0
    assert find_median_sorted_arrays([2], []) == 2.0
    assert find_median_sorted_arrays([1, 3], [2, 7]) == 2.5
    assert find_median_sorted_arrays([1, 2, 3, 4, 5], [6, 7, 8]) == 4.5
    assert find_median_sorted_arrays([1, 1, 1], [1, 1, 1]) == 1.0
    assert find_median_sorted_arrays([0, 0], [0, 0]) == 0.0
    assert find_median_sorted_arrays([1, 3, 5, 7], [2, 4, 6]) == 4.0

    # Brute-force cross-check
    import statistics
    samples = [
        ([1, 5, 9], [2, 3, 4, 10]),
        ([10, 20, 30], [1, 2, 3, 40, 50]),
        ([-5, -3, 0], [-4, 1, 2]),
    ]
    for a, b in samples:
        expected = statistics.median(sorted(a + b))
        assert find_median_sorted_arrays(a, b) == expected

    print("  PASS — all test cases (incl. brute-force cross-check)")


# =============================================================================
# EXERCISE 8 (Hard): Count of Smaller Numbers After Self — Merge sort
# =============================================================================

def count_smaller(nums: list[int]) -> list[int]:
    """
    Instrumented merge sort over INDICES, counting cross-pair inversions.

    During a merge, when an element from the RIGHT half is placed before some
    elements still pending in the LEFT half, that right element is smaller than
    each of those left elements: bump their counts. We track this with a running
    count of right-half elements already merged.

    Time: O(n log n), Space: O(n)
    """
    n = len(nums)
    counts = [0] * n
    indices = list(range(n))           # Sort indices, not values

    def merge_sort(idx):
        if len(idx) <= 1:
            return idx
        mid = len(idx) // 2
        left = merge_sort(idx[:mid])
        right = merge_sort(idx[mid:])

        merged = []
        i = j = 0
        right_smaller = 0              # How many right-half values already merged
        while i < len(left) and j < len(right):
            # Strictly smaller: right value must be < left value to count
            if nums[right[j]] < nums[left[i]]:
                right_smaller += 1
                merged.append(right[j])
                j += 1
            else:
                counts[left[i]] += right_smaller
                merged.append(left[i])
                i += 1
        # Remaining left elements: all pending right values were smaller
        while i < len(left):
            counts[left[i]] += right_smaller
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
        return merged

    merge_sort(indices)
    return counts


def test_exercise_8():
    print("\nExercise 8: Count of Smaller Numbers After Self")

    assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_smaller([-1]) == [0]
    assert count_smaller([-1, -1]) == [0, 0]
    assert count_smaller([]) == []
    assert count_smaller([2, 0, 1]) == [2, 0, 0]
    assert count_smaller([1, 2, 3, 4]) == [0, 0, 0, 0]
    assert count_smaller([4, 3, 2, 1]) == [3, 2, 1, 0]
    assert count_smaller([5, 5, 5]) == [0, 0, 0]

    # Brute-force cross-check
    def brute(nums):
        return [sum(1 for j in range(i + 1, len(nums)) if nums[j] < nums[i])
                for i in range(len(nums))]

    import random
    rng = random.Random(42)
    for _ in range(50):
        arr = [rng.randint(-10, 10) for _ in range(rng.randint(0, 12))]
        assert count_smaller(arr) == brute(arr)

    print("  PASS — all test cases (incl. randomized brute-force cross-check)")


# =============================================================================
# EXERCISE 9 (Hard): Split Array Largest Sum — Binary search on answer + greedy
# =============================================================================

def split_array(nums: list[int], k: int) -> int:
    """
    Binary search the answer in [max(nums), sum(nums)].

    FEASIBILITY (greedy):
    - For a cap, count how many groups are needed so no group sum exceeds cap.
    - Fewer-or-equal-to-k groups means cap is achievable.
    - groups_needed is monotonically NON-INCREASING in cap, so we binary search
      the smallest feasible cap.

    Time: O(n log(sum(nums))), Space: O(1)
    """
    def groups_needed(cap):
        groups = 1
        current = 0
        for x in nums:
            if current + x > cap:
                groups += 1            # Start a new group
                current = x
            else:
                current += x
        return groups

    lo, hi = max(nums), sum(nums)      # Search space of possible largest sums
    while lo < hi:
        mid = (lo + hi) // 2
        if groups_needed(mid) <= k:
            hi = mid                   # cap works; try smaller
        else:
            lo = mid + 1               # too tight; allow bigger cap
    return lo


def test_exercise_9():
    print("\nExercise 9: Split Array Largest Sum")

    assert split_array([7, 2, 5, 10, 8], 2) == 18
    assert split_array([1, 2, 3, 4, 5], 2) == 9
    assert split_array([1, 4, 4], 3) == 4
    assert split_array([1], 1) == 1
    assert split_array([1, 2, 3, 4, 5], 1) == 15
    assert split_array([1, 2, 3, 4, 5], 5) == 5
    assert split_array([2, 3, 1, 1, 1, 1, 1], 5) == 3

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 6: Sorting & Searching (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

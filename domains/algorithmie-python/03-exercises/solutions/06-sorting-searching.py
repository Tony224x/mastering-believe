"""
Solutions — Day 6: Sorting & Searching
Run: python domains/algorithmie-python/03-exercises/solutions/06-sorting-searching.py

Each solution is numbered to match the exercise file.
All solutions are verified with assertions at the end.
"""

import heapq
import random
from collections import Counter


# =============================================================================
# EXERCISE 1 (Easy): Binary Search
# =============================================================================

def binary_search(nums, target):
    """
    Classic iterative binary search with inclusive bounds [lo, hi].

    LOOP INVARIANT:
    - If target exists in nums, its index is in the range [lo, hi].
    - When lo > hi, the range is empty and target is absent.

    WHY `lo <= hi` and not `lo < hi`:
    - With inclusive bounds, when lo == hi, there's still one unchecked element.
    - The last iteration sets either lo = mid + 1 or hi = mid - 1, so we exit
      cleanly with lo > hi.

    Time: O(log n) — the range halves each iteration
    Space: O(1)
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2           # Python has big-int, so overflow is not a concern
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1               # Target is strictly to the right of mid
        else:
            hi = mid - 1               # Target is strictly to the left of mid
    return -1


def test_exercise_1():
    print("\nExercise 1: Binary Search")

    assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
    assert binary_search([5], 5) == 0
    assert binary_search([5], -5) == -1
    assert binary_search([], 0) == -1
    assert binary_search([1, 2, 3, 4, 5], 1) == 0
    assert binary_search([1, 2, 3, 4, 5], 5) == 4

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 2 (Easy): Sort Characters by Frequency
# =============================================================================

def frequency_sort(s):
    """
    Count frequencies, then sort characters by frequency descending.

    APPROACH:
    - Counter gives us frequencies in O(n).
    - sorted with key=lambda uses Timsort, O(k log k) on the unique characters.
    - Reconstruct the output by multiplying each character by its frequency.

    Time: O(n + k log k) where n = len(s) and k = number of unique characters
          Since k <= n, this is O(n log n) worst case.
    Space: O(n) for the output
    """
    freq = Counter(s)
    # Sort unique characters by count descending. Stability of Timsort means
    # ties keep their original Counter insertion order, which is fine here.
    ordered = sorted(freq.items(), key=lambda kv: -kv[1])
    return ''.join(char * count for char, count in ordered)


def test_exercise_2():
    print("\nExercise 2: Sort Characters by Frequency")

    def check(s, valid_outputs):
        result = frequency_sort(s)
        assert len(result) == len(s), f"Length mismatch for {s!r}"
        assert result in valid_outputs, f"Got {result!r}, expected one of {valid_outputs}"

    check("tree", {"eert", "eetr"})
    check("cccaaa", {"cccaaa", "aaaccc"})
    check("Aabb", {"bbAa", "bbaA"})
    check("", {""})
    check("a", {"a"})

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 3 (Easy): Kth Largest Element
# =============================================================================

def kth_largest(nums, k):
    """
    Min-heap of size k approach.

    INTUITION:
    - Maintain a min-heap that always contains the k LARGEST elements seen so far.
    - The root of the heap is the smallest of those k -> that's the k-th largest.
    - When a new element is larger than the root, replace the root.

    WHY THIS BEATS sorted(nums)[-k]:
    - Time: O(n log k) vs O(n log n). When k << n this is a big win.
    - Space: O(k) instead of O(n).

    Time: O(n log k), Space: O(k)
    """
    # Initialize the heap with the first k elements
    heap = nums[:k]
    heapq.heapify(heap)                   # O(k)

    # Process remaining elements
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)  # Pop smallest AND push num in one op

    return heap[0]                         # Root is the smallest of the k largest


def kth_largest_quickselect(nums, k):
    """
    Alternative solution using randomized quickselect. Shown for comparison.

    Time: O(n) average, O(n^2) worst case (extremely rare with random pivot)
    Space: O(1) iterative
    """
    # k-th LARGEST in 1-indexed = (n - k)-th SMALLEST in 0-indexed
    target_idx = len(nums) - k
    arr = nums[:]                          # Local copy — don't mutate caller's list

    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        pivot_idx = _partition(arr, lo, hi)
        if pivot_idx == target_idx:
            return arr[pivot_idx]
        elif pivot_idx < target_idx:
            lo = pivot_idx + 1
        else:
            hi = pivot_idx - 1
    raise IndexError("k out of bounds")


def _partition(arr, lo, hi):
    """Lomuto partition with a random pivot."""
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
    pivot = arr[hi]
    i = lo
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[hi] = arr[hi], arr[i]
    return i


def test_exercise_3():
    print("\nExercise 3: Kth Largest Element")

    for impl in (kth_largest, kth_largest_quickselect):
        assert impl([3, 2, 1, 5, 6, 4], 2) == 5
        assert impl([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
        assert impl([1], 1) == 1
        assert impl([1, 2], 1) == 2
        assert impl([1, 2], 2) == 1
        assert impl([7, 7, 7, 7], 2) == 7

    print("  PASS — all test cases (heap and quickselect)")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    print("\nAll Day 6 exercise solutions passed.")

"""
Day 6 — Sorting & Searching: Binary Search, Quickselect & Custom Comparators
Run: python domains/algorithmie-python/02-code/06-sorting-searching.py

Each section highlights ONE pattern. Comments explain invariants and WHY the
boundaries are what they are (off-by-one is where most bugs live).
"""

import random
import time
import heapq
from functools import cmp_to_key
from bisect import bisect_left, bisect_right


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


# =============================================================================
# SECTION 1: BUILT-IN SORT WITH CUSTOM KEYS
# =============================================================================

def sort_by_length_then_alpha(words):
    """
    Sort words primarily by length (ascending), then alphabetically.

    TECHNIQUE:
    - Wrap multiple criteria in a tuple. Python compares tuples lexicographically,
      so the first differing element decides the order.
    - This is always O(n log n) — the tuple construction is O(1) per element.
    """
    return sorted(words, key=lambda s: (len(s), s))


def sort_students_by_grade_desc_then_name_asc(students):
    """
    students = list of (name, grade). Sort by grade DESCENDING, then name ASCENDING.

    TRICK: negate the numeric key to invert its order while keeping string keys
    in natural order. Works for ints/floats but NOT for strings (they can't be negated).
    """
    return sorted(students, key=lambda s: (-s[1], s[0]))


def largest_number(nums):
    """
    Concatenate integers to form the largest possible number. Classic cmp_to_key.

    INSIGHT:
    - For two numbers a and b, a should come before b iff a+b > b+a (as strings).
    - This ordering is a total order (transitive) but NOT expressible as a key.

    Example: [3, 30, 34, 5, 9] -> "9534330"
    """
    def cmp(a, b):
        if a + b > b + a:
            return -1            # a comes first
        elif a + b < b + a:
            return 1
        return 0

    str_nums = [str(n) for n in nums]
    str_nums.sort(key=cmp_to_key(cmp))
    result = ''.join(str_nums)
    # Edge case: leading zeros -> "000..." should become "0"
    return '0' if result[0] == '0' else result


# =============================================================================
# SECTION 2: BINARY SEARCH — 4 VARIANTS
# =============================================================================

def binary_search_exact(arr, target):
    """
    Exact match. Return index or -1.

    TEMPLATE: lo <= hi, hi = len - 1, mid +/- 1 on both sides.
    This is the ONLY variant where lo and hi always shrink.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2        # No overflow in Python
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1             # Target is strictly right of mid
        else:
            hi = mid - 1             # Target is strictly left of mid
    return -1


def lower_bound(arr, target):
    """
    First index i such that arr[i] >= target. Returns len(arr) if none.

    TEMPLATE: lo < hi, hi = len, hi = mid (NOT mid - 1).
    The idea: hi is ALWAYS a valid upper-bound candidate, so we never exclude it.
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1             # mid is too small, target must be to the right
        else:
            hi = mid                 # mid might be the answer, search [lo, mid)
    return lo


def upper_bound(arr, target):
    """
    First index i such that arr[i] > target. Returns len(arr) if none.
    Symmetric with lower_bound but with <= instead of <.
    """
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def count_occurrences(arr, target):
    """Count how many times target appears in a sorted array. O(log n)."""
    return upper_bound(arr, target) - lower_bound(arr, target)


def search_rotated(nums, target):
    """
    Search in a rotated sorted array with distinct values. Returns index or -1.

    KEY INSIGHT:
    - At each step, at least one half (left-of-mid or right-of-mid) is fully sorted.
    - We check which half is sorted, then check if target lies INSIDE that sorted half.

    Time: O(log n), Space: O(1)
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:
            # Left half [lo..mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1         # target is in the sorted left half
            else:
                lo = mid + 1         # target is in the right half (possibly still rotated)
        else:
            # Right half [mid..hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1         # target is in the sorted right half
            else:
                hi = mid - 1         # target is in the left half

    return -1


# =============================================================================
# SECTION 3: QUICKSELECT
# =============================================================================

def quickselect(arr, k):
    """
    Return the k-th smallest element (0-indexed) of arr.

    APPROACH:
    - Pick a random pivot, partition, then recurse into ONE side only.
    - Expected O(n) because the partition reduces the search space by a
      constant factor on average.
    - Randomized pivot makes the worst case O(n^2) essentially impossible
      on adversarial inputs.

    NOTE: mutates arr in place. Pass a copy if you need the original preserved.
    """
    arr = arr[:]                    # Local copy so callers are not surprised
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        pivot_idx = _partition(arr, lo, hi)
        if pivot_idx == k:
            return arr[pivot_idx]
        elif pivot_idx < k:
            lo = pivot_idx + 1       # k-th is to the right
        else:
            hi = pivot_idx - 1       # k-th is to the left
    raise IndexError("k out of bounds")


def _partition(arr, lo, hi):
    """
    Lomuto partition with a randomized pivot. Returns the final index of the pivot.

    INVARIANT AFTER PARTITION:
    - arr[lo..i-1] < pivot
    - arr[i..hi-1] >= pivot
    - arr[i] is the pivot in its final position
    """
    # Randomize to avoid O(n^2) on sorted/adversarial input
    pivot_idx = random.randint(lo, hi)
    arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

    pivot = arr[hi]
    i = lo                           # Next slot for elements < pivot
    for j in range(lo, hi):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[hi] = arr[hi], arr[i]    # Place pivot in its final slot
    return i


def kth_largest_with_heap(nums, k):
    """
    Alternative: k-th largest using a min-heap of size k.

    Time: O(n log k), Space: O(k)
    Better than quickselect when k is small (k << n).
    """
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]


# =============================================================================
# SECTION 4: BINARY SEARCH ON A 2D MATRIX
# =============================================================================

def search_matrix_sorted(matrix, target):
    """
    Matrix is fully sorted row-wise AND end-of-row < start-of-next-row.
    Treat as a 1D array of length m*n using div/mod.

    Time: O(log(m*n)), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target:
            return True
        elif val < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


def search_matrix_staircase(matrix, target):
    """
    Matrix has rows sorted AND columns sorted, independently.
    Start at top-right: left decreases, down increases. Each step eliminates
    a full row or column.

    Time: O(m + n), Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    r, c = 0, len(matrix[0]) - 1
    while r < len(matrix) and c >= 0:
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] > target:
            c -= 1                   # All cells below are strictly greater
        else:
            r += 1                   # All cells to the left are strictly smaller
    return False


# =============================================================================
# DEMOS
# =============================================================================

def demo_custom_sort():
    print("\n" + "=" * 70)
    print("CUSTOM SORT: key= and cmp_to_key")
    print("=" * 70)
    words = ["banana", "apple", "fig", "kiwi", "cherry"]
    assert sort_by_length_then_alpha(words) == ["fig", "kiwi", "apple", "banana", "cherry"]

    students = [("Alice", 85), ("Bob", 92), ("Charlie", 85), ("Dave", 92)]
    assert sort_students_by_grade_desc_then_name_asc(students) == [
        ("Bob", 92), ("Dave", 92), ("Alice", 85), ("Charlie", 85)
    ]

    assert largest_number([3, 30, 34, 5, 9]) == "9534330"
    assert largest_number([10, 2]) == "210"
    assert largest_number([0, 0]) == "0"
    print("  All custom sort tests passed")


def demo_binary_search():
    print("\n" + "=" * 70)
    print("BINARY SEARCH variants")
    print("=" * 70)
    arr = [1, 2, 4, 4, 4, 7, 10, 15]
    assert binary_search_exact(arr, 4) in (2, 3, 4)
    assert binary_search_exact(arr, 6) == -1
    assert lower_bound(arr, 4) == 2
    assert upper_bound(arr, 4) == 5
    assert count_occurrences(arr, 4) == 3

    # Rotated
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated([1], 0) == -1
    assert search_rotated([1], 1) == 0

    # Sanity check against bisect
    assert bisect_left(arr, 4) == lower_bound(arr, 4)
    assert bisect_right(arr, 4) == upper_bound(arr, 4)

    print("  All binary search tests passed")


def demo_quickselect():
    print("\n" + "=" * 70)
    print("QUICKSELECT vs HEAP")
    print("=" * 70)
    nums = [3, 2, 1, 5, 6, 4]
    # 2nd largest = 5 -> k-th smallest (0-indexed) = len - 2 = 4
    assert quickselect(nums, len(nums) - 2) == 5
    assert kth_largest_with_heap(nums, 2) == 5

    # Performance on 100k elements
    big = [random.randint(0, 10**6) for _ in range(100_000)]
    k = 50_000                                           # median
    t0 = time.perf_counter()
    a = quickselect(big, k)
    t1 = time.perf_counter()
    # Compare to true value via sorted (baseline)
    b = sorted(big)[k]
    t2 = time.perf_counter()
    assert a == b
    print(f"  quickselect: {t1 - t0:.4f}s  |  sorted baseline: {t2 - t1:.4f}s")


def demo_matrix_search():
    print("\n" + "=" * 70)
    print("MATRIX SEARCH: sorted (LC 74) and staircase (LC 240)")
    print("=" * 70)
    m = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 60],
    ]
    assert search_matrix_sorted(m, 16) is True
    assert search_matrix_sorted(m, 15) is False

    m2 = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    assert search_matrix_staircase(m2, 5) is True
    assert search_matrix_staircase(m2, 20) is False
    print("  All matrix search tests passed")


if __name__ == "__main__":
    demo_custom_sort()
    demo_binary_search()
    demo_quickselect()
    demo_matrix_search()
    print("\nAll Day 6 demos passed.")

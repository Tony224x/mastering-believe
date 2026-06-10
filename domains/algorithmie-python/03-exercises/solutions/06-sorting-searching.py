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

# =============================================================================
# EXERCISE 4 (Medium): Find First and Last Position (lower/upper bound)
# =============================================================================

def search_range(nums, target):
    """
    Two binary searches via a single lower_bound helper.

    KEY IDEA:
    - lower_bound(x) = first index i with nums[i] >= x.
    - first occurrence of target  = lower_bound(target)
    - last occurrence of target   = lower_bound(target + 1) - 1
      (the element just BEFORE where target+1 would be inserted)

    WHY NOT "find one index then expand linearly":
    - On [5, 5, 5, ..., 5] the expansion is O(n), defeating the binary search.

    Time: O(log n), Space: O(1)
    """
    def lower_bound(x):
        # Invariant: answer is in [lo, hi]; hi = len(nums) means "not found"
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] >= x:
                hi = mid            # mid could be the answer — keep it
            else:
                lo = mid + 1        # everything <= mid is too small
        return lo

    first = lower_bound(target)
    if first == len(nums) or nums[first] != target:
        return [-1, -1]
    return [first, lower_bound(target + 1) - 1]


def test_exercise_4():
    print("\nExercise 4: Find First and Last Position")

    assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert search_range([], 0) == [-1, -1]
    assert search_range([1], 1) == [0, 0]
    assert search_range([2, 2], 2) == [0, 1]
    assert search_range([5, 5, 5, 5, 5], 5) == [0, 4]
    assert search_range([1, 2, 3], 0) == [-1, -1]
    assert search_range([1, 2, 3], 4) == [-1, -1]

    # Oracle: bisect implements the exact same bounds
    import bisect
    for _ in range(200):
        nums = sorted(random.choices(range(20), k=random.randint(0, 30)))
        target = random.randint(0, 20)
        lo = bisect.bisect_left(nums, target)
        hi = bisect.bisect_right(nums, target) - 1
        expected = [lo, hi] if lo <= hi else [-1, -1]
        assert search_range(nums, target) == expected

    print("  PASS — all test cases + bisect oracle")


# =============================================================================
# EXERCISE 5 (Medium): Largest Number (cmp_to_key)
# =============================================================================

def largest_number(nums):
    """
    Sort with a CONCATENATION comparator, not a simple key.

    WHY key=str IS WRONG:
    - Lexicographic order puts "30" before "3"? Actually "3" > "30"
      lexicographically, but "34" > "3" too — yet 3 must come BEFORE 34
      ("334" < "343"). No per-element key captures this: the right order
      depends on the PAIR being compared.

    THE COMPARATOR:
    - a before b  iff  str(a) + str(b) > str(b) + str(a).
    - This relation is transitive (provable), so it defines a valid order.

    Time: O(n log n * k) where k = average digit count
    """
    from functools import cmp_to_key

    def compare(a, b):
        # Negative return = a first. We want the bigger concatenation first.
        if a + b > b + a:
            return -1
        if a + b < b + a:
            return 1
        return 0

    digits = sorted(map(str, nums), key=cmp_to_key(compare))
    result = "".join(digits)
    # All-zero input would produce "000..." — normalize to "0"
    return result.lstrip("0") or "0"


def test_exercise_5():
    print("\nExercise 5: Largest Number")

    assert largest_number([10, 2]) == "210"
    assert largest_number([3, 30, 34, 5, 9]) == "9534330"
    assert largest_number([0, 0]) == "0"
    assert largest_number([0]) == "0"
    assert largest_number([1]) == "1"
    assert largest_number([432, 43243]) == "43243432"
    assert largest_number([121, 12]) == "12121"
    assert largest_number([8308, 830]) == "8308830"

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Find Peak Element
# =============================================================================

def find_peak_element(nums):
    """
    Binary search WITHOUT a sorted array.

    THE INVARIANT (what makes binary search legal here):
    - If nums[mid] < nums[mid+1], the slope goes UP at mid. Following an
      ascending slope must eventually hit a peak (values can't ascend
      forever — the virtual -inf boundary stops them). So a peak exists
      strictly right of mid → lo = mid + 1.
    - Otherwise nums[mid] > nums[mid+1] (no equals allowed): mid itself
      may be the peak of a descending slope → hi = mid (keep mid).

    BOUNDS SAFETY:
    - Loop condition lo < hi guarantees mid + 1 <= hi <= len-1: no
      out-of-range access.

    Time: O(log n), Space: O(1)
    """
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1            # Ascending slope: peak is to the right
        else:
            hi = mid                # Descending: mid could be the peak
    return lo


def test_exercise_6():
    print("\nExercise 6: Find Peak Element")

    assert find_peak_element([1, 2, 3, 1]) == 2
    assert find_peak_element([1, 2, 1, 3, 5, 6, 4]) in (1, 5)
    assert find_peak_element([1]) == 0
    assert find_peak_element([1, 2]) == 1
    assert find_peak_element([2, 1]) == 0
    assert find_peak_element([1, 2, 3, 4, 5]) == 4
    assert find_peak_element([5, 4, 3, 2, 1]) == 0

    # Property check: the returned index must really be a peak
    for _ in range(100):
        n = random.randint(1, 50)
        arr = random.sample(range(1000), n)     # Distinct values
        i = find_peak_element(arr)
        left = arr[i - 1] if i > 0 else float("-inf")
        right = arr[i + 1] if i < n - 1 else float("-inf")
        assert left < arr[i] > right, (arr, i)

    print("  PASS — all test cases + property check")


# =============================================================================
# EXERCISE 7 (Hard): Median of Two Sorted Arrays
# =============================================================================

def find_median_sorted_arrays(nums1, nums2):
    """
    Binary search on the PARTITION of the smaller array.

    KEY IDEA:
    - We look for a cut: i elements of nums1 and j elements of nums2 on the
      left side, with i + j = (m + n + 1) // 2 (left side holds the extra
      element when the total is odd).
    - The cut is valid when every left element <= every right element,
      which (arrays being sorted) reduces to TWO comparisons:
        nums1[i-1] <= nums2[j]  and  nums2[j-1] <= nums1[i]
    - If nums1[i-1] > nums2[j], i is too big → search left. Otherwise too
      small → search right. Classic binary search on i.

    EDGE HANDLING:
    - i == 0 or i == m means one side of the cut is empty: substitute
      -inf/+inf so the comparisons hold trivially.

    Time: O(log(min(m, n))), Space: O(1)
    """
    # Always binary search on the smaller array → log(min(m, n))
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2             # Size of the left side
    lo, hi = 0, m
    INF = float("inf")

    while lo <= hi:
        i = (lo + hi) // 2              # Elements of nums1 on the left
        j = half - i                    # Elements of nums2 on the left

        left1 = nums1[i - 1] if i > 0 else -INF
        right1 = nums1[i] if i < m else INF
        left2 = nums2[j - 1] if j > 0 else -INF
        right2 = nums2[j] if j < n else INF

        if left1 <= right2 and left2 <= right1:
            # Valid partition found
            if (m + n) % 2 == 1:
                return float(max(left1, left2))
            return (max(left1, left2) + min(right1, right2)) / 2.0
        elif left1 > right2:
            hi = i - 1                  # Too many nums1 elements on the left
        else:
            lo = i + 1                  # Not enough

    raise ValueError("Inputs must be sorted arrays")    # Unreachable on valid input


def test_exercise_7():
    print("\nExercise 7: Median of Two Sorted Arrays")

    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
    assert find_median_sorted_arrays([], [1]) == 1.0
    assert find_median_sorted_arrays([2], []) == 2.0
    assert find_median_sorted_arrays([1, 1], [1, 1]) == 1.0
    assert find_median_sorted_arrays([1, 2, 3, 4, 5], [6, 7, 8, 9]) == 5.0
    assert find_median_sorted_arrays([6, 7, 8, 9], [1, 2, 3, 4, 5]) == 5.0
    assert find_median_sorted_arrays([1, 5], [2, 3, 4]) == 3.0

    # Oracle: statistics.median on the merged arrays
    import statistics
    for _ in range(200):
        a = sorted(random.choices(range(50), k=random.randint(0, 20)))
        b = sorted(random.choices(range(50), k=random.randint(0, 20)))
        if not a and not b:
            continue
        assert abs(find_median_sorted_arrays(a, b) - statistics.median(a + b)) < 1e-9

    print("  PASS — all test cases + statistics.median oracle")


# =============================================================================
# EXERCISE 8 (Hard): Count Inversions (augmented merge sort)
# =============================================================================

def count_inversions_brute(arr):
    """O(n^2) oracle — check every pair (i < j, arr[i] > arr[j])."""
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count


def count_inversions(arr):
    """
    Augmented merge sort — O(n log n).

    THE ONE EXTRA LINE:
    - During the merge, when we take right[j] while left[i:] is not empty,
      right[j] is smaller than EVERY remaining left element. Each of those
      (len(left) - i) elements appears before right[j] in the original
      order → that many inversions, counted in O(1).

    DUPLICATES:
    - Equal values are NOT inversions, so ties must take from the LEFT
      first (left[i] <= right[j] → take left, count nothing).

    Time: O(n log n), Space: O(n) for merge buffers (input untouched).
    """
    def sort_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, c_left = sort_count(a[:mid])
        right, c_right = sort_count(a[mid:])

        merged = []
        count = c_left + c_right
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:     # <= : equal pairs are not inversions
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                count += len(left) - i  # All remaining left elements invert with right[j]
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, count

    return sort_count(list(arr))[1]     # Copy: the caller's array stays intact


def test_exercise_8():
    print("\nExercise 8: Count Inversions")

    assert count_inversions([1, 2, 3, 4]) == 0
    assert count_inversions([4, 3, 2, 1]) == 6
    assert count_inversions([2, 4, 1, 3, 5]) == 3
    assert count_inversions([]) == 0
    assert count_inversions([1]) == 0
    assert count_inversions([3, 3, 3]) == 0
    assert count_inversions([2, 1, 2, 1]) == 3

    # Input must not be modified
    original = [4, 3, 2, 1]
    count_inversions(original)
    assert original == [4, 3, 2, 1]

    # Oracle on random inputs
    for _ in range(100):
        arr = [random.randint(0, 30) for _ in range(random.randint(0, 40))]
        assert count_inversions(arr) == count_inversions_brute(arr), arr

    # Benchmark on REVERSED arrays (worst case: n(n-1)/2 inversions)
    import time
    print("  Benchmark (reversed array — worst case):")
    print(f"    {'n':>6} | {'merge O(n log n)':>17} | {'brute O(n^2)':>13}")
    for n in [2000, 4000, 8000]:
        arr = list(range(n, 0, -1))
        start = time.perf_counter()
        r1 = count_inversions(arr)
        t_merge = time.perf_counter() - start
        start = time.perf_counter()
        r2 = count_inversions_brute(arr)
        t_brute = time.perf_counter() - start
        assert r1 == r2 == n * (n - 1) // 2
        print(f"    {n:>6,} | {t_merge:>16.5f}s | {t_brute:>12.5f}s")

    print("  PASS — all test cases + oracle + benchmark")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    test_exercise_6()
    test_exercise_7()
    test_exercise_8()
    print("\nAll Day 6 exercise solutions passed.")

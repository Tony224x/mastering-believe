"""
Day 2 — Arrays & Strings: Two Pointers, Sliding Window, Prefix Sum
Run: python domains/algorithmie-python/02-code/02-arrays-strings.py

Each section shows the BRUTE FORCE first, then the OPTIMIZED version,
so you can FEEL the improvement and understand WHY the pattern matters.
"""

import time
import random
from collections import Counter


# =============================================================================
# HELPER: Timing decorator
# =============================================================================

def timed(func):
    """Decorator that prints execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


# =============================================================================
# SECTION 1: TWO POINTERS
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 1: Two Sum II — Input Array Is Sorted (LeetCode 167)
# Given a SORTED array and a target, find two numbers that sum to target.
# Return their 1-indexed positions.
# ---------------------------------------------------------------------------

@timed
def two_sum_brute(numbers, target):
    """
    Brute force: try every pair.
    Time: O(n^2) — nested loops
    Space: O(1)
    """
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if numbers[i] + numbers[j] == target:
                return [i + 1, j + 1]  # 1-indexed
    return []


@timed
def two_sum_sorted(numbers, target):
    """
    Two pointers (convergence): exploit sorted order.

    INTUITION:
    - left points to smallest, right points to largest
    - If sum < target: we need a bigger sum → move left right (bigger number)
    - If sum > target: we need a smaller sum → move right left (smaller number)
    - If sum == target: found it!

    WHY this works: at each step, we eliminate an entire row or column
    of the n×n pair matrix. We never need to revisit.

    Time: O(n) — each pointer moves at most n times
    Space: O(1) — just two variables
    """
    left, right = 0, len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            return [left + 1, right + 1]  # 1-indexed
        elif current_sum < target:
            left += 1      # Sum too small → increase left value
        else:
            right -= 1     # Sum too big → decrease right value

    return []


def demo_two_sum():
    print("\n" + "=" * 70)
    print("TWO POINTERS #1: Two Sum (sorted array)")
    print("=" * 70)

    # Correctness
    nums = [2, 7, 11, 15]
    target = 9
    assert two_sum_brute(nums, target) == [1, 2]
    assert two_sum_sorted(nums, target) == [1, 2]

    # Performance: brute force vs two pointers on large input
    print("\n  Performance comparison:")
    n = 10_000
    nums_large = list(range(1, n + 1))  # Sorted: [1, 2, 3, ..., n]
    target_large = nums_large[-1] + nums_large[-2]  # Worst case: last two elements

    two_sum_brute(nums_large, target_large)
    two_sum_sorted(nums_large, target_large)


# ---------------------------------------------------------------------------
# Problem 2: Container With Most Water (LeetCode 11)
# Given heights, find two lines that form a container holding most water.
# ---------------------------------------------------------------------------

@timed
def max_area_brute(height):
    """
    Brute force: try every pair of lines.
    Time: O(n^2)
    Space: O(1)
    """
    max_water = 0
    n = len(height)
    for i in range(n):
        for j in range(i + 1, n):
            # Water = width × min(left_height, right_height)
            water = (j - i) * min(height[i], height[j])
            max_water = max(max_water, water)
    return max_water


@timed
def max_area(height):
    """
    Two pointers (convergence): start from widest container.

    INTUITION:
    - Start with the widest container (left=0, right=n-1)
    - Width can only DECREASE as we move inward
    - To potentially increase area, we must find a TALLER bar
    - Always move the SHORTER bar inward:
      * Moving the taller bar can't help (min height stays the same or decreases)
      * Moving the shorter bar might find something taller

    PROOF of correctness (greedy argument):
    - When height[left] < height[right], ANY container using left with a
      position < right would have: smaller width AND height ≤ height[left]
      → strictly worse. So we can safely skip all those.

    Time: O(n) — one pass
    Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        # Move the shorter bar — it's the bottleneck
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water


def demo_container():
    print("\n" + "=" * 70)
    print("TWO POINTERS #2: Container With Most Water")
    print("=" * 70)

    # Correctness
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    assert max_area_brute(height) == 49   # Between index 1 (h=8) and 8 (h=7)
    assert max_area(height) == 49

    # Performance
    print("\n  Performance comparison:")
    n = 5_000
    height_large = [random.randint(1, 10000) for _ in range(n)]
    r1 = max_area_brute(height_large)
    r2 = max_area(height_large)
    assert r1 == r2, f"Results differ: {r1} vs {r2}"


# ---------------------------------------------------------------------------
# Problem 3: Trapping Rain Water (LeetCode 42) — THE classic hard
# Given elevation map, compute how much rain water can be trapped.
# ---------------------------------------------------------------------------

@timed
def trap_brute(height):
    """
    Brute force: for each bar, find max height to its left and right.
    Water at position i = min(max_left, max_right) - height[i]

    Time: O(n^2) — for each element, scan left and right
    Space: O(1)
    """
    n = len(height)
    total = 0
    for i in range(n):
        # Find tallest bar to the left of i (including i)
        max_left = max(height[:i + 1])     # O(i) scan
        # Find tallest bar to the right of i (including i)
        max_right = max(height[i:])        # O(n-i) scan
        # Water at i = min of two maxes minus current height
        total += min(max_left, max_right) - height[i]
    return total


@timed
def trap_prefix(height):
    """
    Prefix/suffix max approach: precompute left_max and right_max arrays.

    Time: O(n) — three passes
    Space: O(n) — two extra arrays
    """
    n = len(height)
    if n == 0:
        return 0

    # left_max[i] = max height from 0 to i
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # right_max[i] = max height from i to n-1
    right_max = [0] * n
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # Water at each position
    total = 0
    for i in range(n):
        total += min(left_max[i], right_max[i]) - height[i]
    return total


@timed
def trap_two_pointers(height):
    """
    Two pointers: O(n) time, O(1) space — the optimal solution.

    INTUITION:
    - We don't need to know the EXACT max on both sides
    - We just need to know which side is the bottleneck
    - If left_max < right_max: water at left is determined by left_max
      (right side is guaranteed to be at least right_max)
    - Process the side with the smaller max

    Time: O(n) — single pass
    Space: O(1) — just four variables
    """
    n = len(height)
    if n == 0:
        return 0

    left, right = 0, n - 1
    left_max, right_max = height[left], height[right]
    total = 0

    while left < right:
        if left_max < right_max:
            # Left side is the bottleneck — process left
            left += 1
            left_max = max(left_max, height[left])
            total += left_max - height[left]  # Water = max_level - current_height
        else:
            # Right side is the bottleneck — process right
            right -= 1
            right_max = max(right_max, height[right])
            total += right_max - height[right]

    return total


def demo_trapping_rain():
    print("\n" + "=" * 70)
    print("TWO POINTERS #3: Trapping Rain Water (Hard)")
    print("=" * 70)

    # Correctness
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    expected = 6
    assert trap_brute(height) == expected
    assert trap_prefix(height) == expected
    assert trap_two_pointers(height) == expected
    print("  All three approaches give correct result: 6")

    # Edge cases
    assert trap_two_pointers([]) == 0          # Empty
    assert trap_two_pointers([3]) == 0         # Single bar
    assert trap_two_pointers([3, 2]) == 0      # Two bars — no trap
    assert trap_two_pointers([4, 2, 3]) == 1   # Simple valley
    print("  Edge cases passed")

    # Performance
    print("\n  Performance comparison:")
    n = 5_000
    h = [random.randint(0, 100) for _ in range(n)]
    r1 = trap_brute(h)
    r2 = trap_prefix(h)
    r3 = trap_two_pointers(h)
    assert r1 == r2 == r3, "Results differ!"


# =============================================================================
# SECTION 2: SLIDING WINDOW
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 4: Maximum Average Subarray I (LeetCode 643) — Fixed window
# Find contiguous subarray of length k with maximum average.
# ---------------------------------------------------------------------------

@timed
def max_avg_brute(nums, k):
    """
    Brute force: compute sum of every window of size k.
    Time: O(n*k) — sum() is O(k) per window
    Space: O(1)
    """
    max_sum = float('-inf')
    for i in range(len(nums) - k + 1):
        window_sum = sum(nums[i:i + k])  # O(k) recalculation each time
        max_sum = max(max_sum, window_sum)
    return max_sum / k


@timed
def max_avg(nums, k):
    """
    Sliding window (fixed size): maintain a running sum.

    INTUITION:
    - First window: sum(nums[0..k-1])
    - Slide right by 1: ADD nums[i], REMOVE nums[i-k]
    - Net change per slide: O(1) instead of O(k)

    Time: O(n) — each element is added and removed exactly once
    Space: O(1)
    """
    # Initialize first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide: add new element, remove old element
    for i in range(k, len(nums)):
        window_sum += nums[i]        # New element enters window from right
        window_sum -= nums[i - k]    # Old element leaves window from left
        max_sum = max(max_sum, window_sum)

    return max_sum / k


def demo_max_avg():
    print("\n" + "=" * 70)
    print("SLIDING WINDOW #1: Maximum Average Subarray (fixed window)")
    print("=" * 70)

    # Correctness
    nums = [1, 12, -5, -6, 50, 3]
    k = 4
    expected = 12.75  # [12, -5, -6, 50] → sum=51, avg=12.75
    assert abs(max_avg_brute(nums, k) - expected) < 1e-9
    assert abs(max_avg(nums, k) - expected) < 1e-9

    # Performance
    print("\n  Performance comparison:")
    n = 100_000
    k = 1000
    nums_large = [random.randint(-100, 100) for _ in range(n)]
    r1 = max_avg_brute(nums_large, k)
    r2 = max_avg(nums_large, k)
    assert abs(r1 - r2) < 1e-9


# ---------------------------------------------------------------------------
# Problem 5: Longest Substring Without Repeating Characters (LeetCode 3)
# Variable window — find longest substring with all unique characters.
# ---------------------------------------------------------------------------

@timed
def length_of_longest_brute(s):
    """
    Brute force: check every substring for uniqueness.
    Time: O(n^3) — O(n^2) substrings, O(n) to check each
    Space: O(n) — for the set check
    """
    best = 0
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            # Check if s[i..j] has all unique characters
            substring = s[i:j + 1]          # O(j-i+1) to create
            if len(set(substring)) == len(substring):  # O(j-i+1) to check
                best = max(best, j - i + 1)
            else:
                break  # If duplicate found, extending won't help
    return best


@timed
def length_of_longest(s):
    """
    Sliding window (variable) with hash map.

    INTUITION:
    - Maintain a window [left, right] with all unique characters
    - Expand right: add char to window
    - If duplicate found: jump left past the previous occurrence
    - Track last seen index of each char to enable the jump

    WHY char_index[s[right]] >= left matters:
    - A char might exist in our map from a PREVIOUS window position
    - We only care about duplicates WITHIN the current window [left..right]
    - If the previous occurrence is before left, it's not in our window → ignore

    Time: O(n) — right moves n times, left moves at most n times total
    Space: O(min(n, 26)) — at most 26 entries for lowercase English letters
    """
    char_index = {}   # char → last seen index
    left = 0
    best = 0

    for right in range(len(s)):
        char = s[right]

        # If this char is already in the current window, shrink past it
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1  # Jump left past the duplicate

        # Update the last seen position of this char
        char_index[char] = right

        # Current window [left..right] has all unique chars
        best = max(best, right - left + 1)

    return best


def demo_longest_substring():
    print("\n" + "=" * 70)
    print("SLIDING WINDOW #2: Longest Substring Without Repeating Characters")
    print("=" * 70)

    # Correctness
    test_cases = [
        ("abcabcbb", 3),   # "abc"
        ("bbbbb", 1),      # "b"
        ("pwwkew", 3),     # "wke"
        ("", 0),           # Empty string
        ("abcdefg", 7),    # Entire string — no repeats
        (" ", 1),          # Single space
    ]
    for s, expected in test_cases:
        assert length_of_longest(s) == expected, f"Failed for '{s}'"
    print("  All test cases passed")

    # Performance
    print("\n  Performance comparison:")
    # Generate a string with characters from 'a' to 'z', repeated randomly
    n = 5_000
    s_large = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
    r1 = length_of_longest_brute(s_large)
    r2 = length_of_longest(s_large)
    assert r1 == r2, f"Results differ: {r1} vs {r2}"


# ---------------------------------------------------------------------------
# Problem 6: Minimum Window Substring (LeetCode 76) — THE classic hard
# Find smallest substring of s containing all characters of t.
# ---------------------------------------------------------------------------

@timed
def min_window_brute(s, t):
    """
    Brute force: check every substring, verify it contains all chars of t.
    Time: O(n^2 * m) where n = len(s), m = len(t)
    Space: O(m) for the Counter
    """
    if not s or not t:
        return ""

    need = Counter(t)
    best = ""

    for i in range(len(s)):
        for j in range(i + len(t), len(s) + 1):  # Window must be at least len(t)
            window = Counter(s[i:j])
            # Check if window contains all chars in need with sufficient counts
            if all(window.get(c, 0) >= need[c] for c in need):
                if not best or (j - i) < len(best):
                    best = s[i:j]
                break  # Found smallest starting at i, no need to extend further
    return best


@timed
def min_window(s, t):
    """
    Sliding window (variable) with two counters.

    INTUITION:
    - Maintain counts of chars we NEED (from t) and chars we HAVE (in window)
    - Expand right to satisfy requirements
    - Once satisfied, shrink left to minimize window size
    - Track the smallest valid window found

    The key trick: instead of comparing entire counters each time (O(m)),
    we track 'have' = number of unique chars that have met their required count.
    When have == need_count, the window is valid.

    Time: O(n + m) — each pointer moves at most n times, t is scanned once
    Space: O(m + alphabet) — two counters
    """
    if not s or not t:
        return ""

    need = Counter(t)       # What we need: {char: required_count}
    need_count = len(need)  # How many unique chars must be satisfied

    window = {}             # What we have in current window
    have = 0                # How many unique chars meet their requirement
    left = 0
    best = (float('inf'), 0, 0)  # (length, left, right)

    for right in range(len(s)):
        # EXPAND: add s[right] to window
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # Check if adding this char just satisfied a requirement
        if char in need and window[char] == need[char]:
            have += 1  # One more unique char is now fully satisfied

        # SHRINK: while all requirements are met, try to minimize
        while have == need_count:
            # Update best if current window is smaller
            window_len = right - left + 1
            if window_len < best[0]:
                best = (window_len, left, right)

            # Remove s[left] from window and move left forward
            left_char = s[left]
            window[left_char] -= 1
            # Did removing this char break a requirement?
            if left_char in need and window[left_char] < need[left_char]:
                have -= 1  # We just lost a required character
            left += 1

    length, lo, hi = best
    return s[lo:hi + 1] if length != float('inf') else ""


def demo_min_window():
    print("\n" + "=" * 70)
    print("SLIDING WINDOW #3: Minimum Window Substring (Hard)")
    print("=" * 70)

    # Correctness
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""  # Can't form "aa" from single "a"
    assert min_window("", "abc") == ""   # Empty s
    assert min_window("abc", "") == ""   # Empty t
    print("  All test cases passed")

    # Performance
    print("\n  Performance comparison:")
    n = 2_000
    # Generate s with random uppercase letters, t with a few chars to find
    s_large = "".join(random.choice("ABCDEFGHIJ") for _ in range(n))
    t_large = "ABCDE"
    r1 = min_window_brute(s_large, t_large)
    r2 = min_window(s_large, t_large)
    assert r1 == r2, f"Results differ: '{r1}' vs '{r2}'"


# =============================================================================
# SECTION 3: PREFIX SUM
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 7: Range Sum Query (LeetCode 303)
# Given an array, answer multiple range sum queries efficiently.
# ---------------------------------------------------------------------------

class RangeSumBrute:
    """
    Brute force: recalculate sum for each query.
    Build: O(1)
    Query: O(n) per query
    """
    def __init__(self, nums):
        self.nums = nums

    def sum_range(self, left, right):
        return sum(self.nums[left:right + 1])  # O(right - left + 1)


class RangeSumPrefix:
    """
    Prefix sum: precompute cumulative sums.

    prefix[0] = 0  (empty prefix — crucial for edge case i=0)
    prefix[i] = nums[0] + nums[1] + ... + nums[i-1]

    sum(nums[left..right]) = prefix[right+1] - prefix[left]

    Build: O(n)
    Query: O(1) — just one subtraction!
    Space: O(n)
    """
    def __init__(self, nums):
        n = len(nums)
        self.prefix = [0] * (n + 1)      # n+1 elements, prefix[0] = 0
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + nums[i]

    def sum_range(self, left, right):
        # Sum of nums[left..right] = prefix[right+1] - prefix[left]
        return self.prefix[right + 1] - self.prefix[left]


def demo_range_sum():
    print("\n" + "=" * 70)
    print("PREFIX SUM #1: Range Sum Query")
    print("=" * 70)

    # Correctness
    nums = [-2, 0, 3, -5, 2, -1]
    brute = RangeSumBrute(nums)
    prefix = RangeSumPrefix(nums)

    queries = [(0, 2), (2, 5), (0, 5)]
    expected = [1, -1, -3]

    for (l, r), exp in zip(queries, expected):
        assert brute.sum_range(l, r) == exp
        assert prefix.sum_range(l, r) == exp
    print("  Correctness: all queries match")

    # Performance: 10,000 queries on a large array
    print("\n  Performance: 10,000 queries on array of size 50,000")
    n = 50_000
    nums_large = [random.randint(-100, 100) for _ in range(n)]
    queries = [(random.randint(0, n // 2), random.randint(n // 2, n - 1)) for _ in range(10_000)]

    # Brute force
    rs_brute = RangeSumBrute(nums_large)
    start = time.perf_counter()
    results_brute = [rs_brute.sum_range(l, r) for l, r in queries]
    brute_time = time.perf_counter() - start
    print(f"  Brute force (10k queries): {brute_time:.4f}s")

    # Prefix sum
    start = time.perf_counter()
    rs_prefix = RangeSumPrefix(nums_large)
    build_time = time.perf_counter() - start
    start = time.perf_counter()
    results_prefix = [rs_prefix.sum_range(l, r) for l, r in queries]
    query_time = time.perf_counter() - start
    print(f"  Prefix sum (build): {build_time:.6f}s")
    print(f"  Prefix sum (10k queries): {query_time:.6f}s")
    print(f"  Total prefix: {build_time + query_time:.6f}s")
    print(f"  Speedup: {brute_time / max(build_time + query_time, 1e-9):.0f}x")

    assert results_brute == results_prefix, "Results differ!"


# ---------------------------------------------------------------------------
# Problem 8: Subarray Sum Equals K (LeetCode 560)
# Count the number of contiguous subarrays that sum to k.
# NOTE: array can contain NEGATIVE numbers (sliding window won't work!)
# ---------------------------------------------------------------------------

@timed
def subarray_sum_brute(nums, k):
    """
    Brute force: try every subarray, compute its sum.
    Time: O(n^2)
    Space: O(1)
    """
    count = 0
    n = len(nums)
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]     # Running sum avoids O(n^3)
            if current_sum == k:
                count += 1
    return count


@timed
def subarray_sum(nums, k):
    """
    Prefix sum + hash map: the canonical pattern.

    INTUITION:
    - Let prefix[j] = sum of nums[0..j-1]
    - If prefix[j] - prefix[i] = k, then subarray nums[i..j-1] sums to k
    - Rearranging: prefix[i] = prefix[j] - k
    - For each j, count how many earlier prefix sums equal (prefix[j] - k)
    - A hash map stores the frequency of each prefix sum seen so far

    WHY seen = {0: 1} ?
    - An "empty prefix" of sum 0 exists before index 0
    - This handles the case where a subarray starting at index 0 sums to k
    - Without it, we'd miss subarray nums[0..j] when prefix[j+1] = k

    WHY sliding window doesn't work here:
    - Sliding window assumes expanding window increases the sum (or shrinking decreases it)
    - With NEGATIVE numbers, expanding can decrease the sum → invalid assumption
    - Prefix sum + hashmap works regardless of negative numbers

    Time: O(n) — single pass, O(1) per hash map operation
    Space: O(n) — hash map stores at most n prefix sums
    """
    count = 0
    prefix = 0            # Running prefix sum (no need for an array!)
    seen = {0: 1}         # {prefix_sum: count of occurrences}

    for num in nums:
        prefix += num     # Update running prefix sum

        # How many earlier prefix sums equal (prefix - k)?
        # Each one represents a subarray ending here that sums to k
        complement = prefix - k
        if complement in seen:
            count += seen[complement]

        # Record this prefix sum
        seen[prefix] = seen.get(prefix, 0) + 1

    return count


def demo_subarray_sum():
    print("\n" + "=" * 70)
    print("PREFIX SUM #2: Subarray Sum Equals K")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([1, 1, 1], 2, 2),           # [1,1] starting at 0 and 1
        ([1, 2, 3], 3, 2),           # [1,2] and [3]
        ([1, -1, 0], 0, 3),          # [1,-1], [-1,0], [1,-1,0]
        ([0, 0, 0], 0, 6),           # Every subarray: [0],[0],[0],[0,0],[0,0],[0,0,0]
        ([3, 4, 7, 2, -3, 1, 4, 2], 7, 4),
    ]
    for nums, k, expected in test_cases:
        brute_result = subarray_sum_brute(nums, k)
        optimized_result = subarray_sum(nums, k)
        assert brute_result == expected, f"Brute failed for {nums}, k={k}: got {brute_result}"
        assert optimized_result == expected, f"Optimized failed for {nums}, k={k}: got {optimized_result}"
    print("  All test cases passed")

    # Performance
    print("\n  Performance comparison:")
    n = 10_000
    nums_large = [random.randint(-50, 50) for _ in range(n)]
    k = 42
    r1 = subarray_sum_brute(nums_large, k)
    r2 = subarray_sum(nums_large, k)
    assert r1 == r2, f"Results differ: {r1} vs {r2}"


# ---------------------------------------------------------------------------
# Problem 9: Product of Array Except Self (LeetCode 238)
# For each element, compute the product of all other elements.
# Constraint: no division allowed!
# ---------------------------------------------------------------------------

@timed
def product_except_self_brute(nums):
    """
    Brute force: for each index, multiply everything else.
    Time: O(n^2) — n elements, each needs n-1 multiplications
    Space: O(n) for the result (O(1) extra)
    """
    n = len(nums)
    result = [1] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                result[i] *= nums[j]
    return result


@timed
def product_except_self(nums):
    """
    Prefix product + suffix product: two passes.

    INTUITION:
    - result[i] should be (product of all left of i) × (product of all right of i)
    - Pass 1 (left → right): build left prefix products in result
      result[i] = nums[0] * nums[1] * ... * nums[i-1]
    - Pass 2 (right → left): multiply by right suffix products
      result[i] *= nums[i+1] * nums[i+2] * ... * nums[n-1]

    WHY no division:
    - Division fails when there are zeros in the array (division by zero)
    - This two-pass approach handles zeros naturally
    - It's also what interviewers expect — they want to see the prefix/suffix trick

    Time: O(n) — two passes
    Space: O(1) extra — we use the result array for prefix, then multiply suffix in-place
    """
    n = len(nums)
    result = [1] * n

    # Pass 1: left-to-right — result[i] = product of everything to the LEFT
    left_product = 1
    for i in range(n):
        result[i] = left_product          # Product of nums[0..i-1]
        left_product *= nums[i]           # Include nums[i] for next iteration

    # Pass 2: right-to-left — multiply result[i] by product of everything to the RIGHT
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product        # Multiply by product of nums[i+1..n-1]
        right_product *= nums[i]          # Include nums[i] for next iteration

    return result


def demo_product_except_self():
    print("\n" + "=" * 70)
    print("PREFIX SUM #3: Product of Array Except Self")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([1, 2, 3, 4], [24, 12, 8, 6]),
        ([-1, 1, 0, -3, 3], [0, 0, 9, 0, 0]),
        ([2, 3], [3, 2]),
        ([0, 0], [0, 0]),
    ]
    for nums, expected in test_cases:
        assert product_except_self(nums) == expected, f"Failed for {nums}"
    print("  All test cases passed")

    # Performance
    print("\n  Performance comparison:")
    n = 3_000
    nums_large = [random.randint(1, 10) for _ in range(n)]
    r1 = product_except_self_brute(nums_large)
    r2 = product_except_self(nums_large)
    assert r1 == r2, "Results differ!"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   Day 2 -- Arrays & Strings: Two Pointers, Sliding Window, Prefix Sum")
    print("=" * 70)

    # Two Pointers
    demo_two_sum()
    demo_container()
    demo_trapping_rain()

    # Sliding Window
    demo_max_avg()
    demo_longest_substring()
    demo_min_window()

    # Prefix Sum
    demo_range_sum()
    demo_subarray_sum()
    demo_product_except_self()

    print("\n" + "=" * 70)
    print("DONE — Key observations:")
    print("  1. Two Pointers on sorted array: O(n^2) -> O(n) for pair problems")
    print("  2. Trapping Rain Water: brute O(n^2) -> prefix O(n)+O(n) -> two pointers O(n)+O(1)")
    print("  3. Fixed sliding window: O(n*k) -> O(n) by maintaining running sum")
    print("  4. Variable sliding window: O(n^3) -> O(n) with hashmap to track state")
    print("  5. Prefix sum: O(n) per query -> O(1) per query after O(n) build")
    print("  6. Subarray sum = K with negatives: prefix sum + hashmap, NOT sliding window")
    print("=" * 70)

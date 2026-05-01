"""
Solutions — Day 2: Arrays & Strings (Two Pointers, Sliding Window, Prefix Sum)
Run: python domains/algorithmie-python/03-exercises/solutions/02-arrays-strings.py

Each solution is numbered to match the exercise file.
All solutions are verified with assertions at the end.
"""

import time
from collections import deque, Counter


# =============================================================================
# EXERCISE 1 (Easy): Remove Duplicates from Sorted Array — Two Pointers
# =============================================================================

def remove_duplicates(nums: list[int]) -> int:
    """
    Two pointers (same direction):
    - slow = position to write the next unique element
    - fast = position scanning ahead for new values

    Since the array is SORTED, duplicates are adjacent.
    We only advance slow when we find a value different from nums[slow-1].

    Time: O(n) — single pass
    Space: O(1) — in-place, no extra array
    """
    if not nums:
        return 0

    slow = 1  # First element is always unique, start writing from index 1

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow - 1]:  # Found a new unique value
            nums[slow] = nums[fast]        # Place it at the slow pointer
            slow += 1                      # Advance write position

    return slow


def test_exercise_1():
    print("\nExercise 1: Remove Duplicates from Sorted Array")

    nums = [1, 1, 2]
    k = remove_duplicates(nums)
    assert k == 2 and nums[:k] == [1, 2]

    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k = remove_duplicates(nums)
    assert k == 5 and nums[:k] == [0, 1, 2, 3, 4]

    nums = [1]
    k = remove_duplicates(nums)
    assert k == 1 and nums[:k] == [1]

    nums = []
    k = remove_duplicates(nums)
    assert k == 0

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 2 (Easy): Maximum Sum of Subarray of Size K — Sliding Window
# =============================================================================

def max_subarray_sum(nums: list[int], k: int) -> int:
    """
    Sliding window (fixed size k):
    1. Compute sum of first window [0..k-1]
    2. Slide: add nums[i], remove nums[i-k]
    3. Track the maximum sum seen

    Time: O(n) — each element added once, removed once
    Space: O(1)
    """
    # Initialize first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(nums)):
        window_sum += nums[i]        # New element enters on the right
        window_sum -= nums[i - k]    # Old element leaves on the left
        max_sum = max(max_sum, window_sum)

    return max_sum


def test_exercise_2():
    print("\nExercise 2: Maximum Sum of Subarray of Size K")

    assert max_subarray_sum([2, 1, 5, 1, 3, 2], 3) == 9
    assert max_subarray_sum([2, 3, 4, 1, 5], 2) == 7
    assert max_subarray_sum([1, -1, 5, -2, 3], 3) == 6
    assert max_subarray_sum([10], 1) == 10
    assert max_subarray_sum([-1, -2, -3, -4], 2) == -3
    assert max_subarray_sum([1, 2, 3, 4, 5], 5) == 15

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 3 (Easy): Running Sum & Range Sum Query — Prefix Sum
# =============================================================================

def running_sum(nums: list[int]) -> list[int]:
    """
    Compute cumulative sum in-place style.

    result[i] = nums[0] + nums[1] + ... + nums[i]

    Time: O(n)
    Space: O(n) for the result (O(1) extra if modifying in-place)
    """
    result = []
    cumsum = 0
    for num in nums:
        cumsum += num
        result.append(cumsum)
    return result


def range_sum_query(nums: list[int], queries: list[tuple[int, int]]) -> list[int]:
    """
    Build prefix sum array once, then answer each query in O(1).

    prefix[0] = 0  (crucial: the empty prefix sum)
    prefix[i+1] = prefix[i] + nums[i]
    sum(nums[left..right]) = prefix[right+1] - prefix[left]

    Build: O(n)
    Each query: O(1)
    Total for Q queries: O(n + Q)
    """
    n = len(nums)
    # Build prefix sum with extra element at index 0
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    # Answer queries
    results = []
    for left, right in queries:
        results.append(prefix[right + 1] - prefix[left])  # O(1) per query

    return results


def test_exercise_3():
    print("\nExercise 3: Running Sum & Range Sum Query")

    # Part A
    assert running_sum([1, 2, 3, 4]) == [1, 3, 6, 10]
    assert running_sum([1, 1, 1, 1, 1]) == [1, 2, 3, 4, 5]
    assert running_sum([3, 1, 2, 10, 1]) == [3, 4, 6, 16, 17]
    assert running_sum([5]) == [5]

    # Part B
    nums = [1, 2, 3, 4, 5]
    assert range_sum_query(nums, [(0, 2)]) == [6]
    assert range_sum_query(nums, [(1, 3)]) == [9]
    assert range_sum_query(nums, [(0, 4)]) == [15]
    assert range_sum_query(nums, [(2, 2)]) == [3]
    assert range_sum_query(nums, [(0, 0), (4, 4), (1, 3)]) == [1, 5, 9]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 4 (Medium): 3Sum — Two Pointers
# =============================================================================

def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Sort + Two Pointers.

    STRATEGY:
    1. Sort the array — O(n log n)
    2. For each element nums[i] (the "anchor"):
       - Use two pointers on the rest to find pairs summing to -nums[i]
       - Skip duplicates for both the anchor and the pointers

    DEDUPLICATION:
    - Anchor: if nums[i] == nums[i-1], skip (same anchor = same results)
    - Left pointer: after finding a triplet, skip duplicates of nums[left]
    - Right pointer: after finding a triplet, skip duplicates of nums[right]

    Time: O(n^2) — O(n log n) sort + O(n) per anchor × O(n) anchors
    Space: O(1) extra (ignoring sort internal space and output)
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):  # Need at least 3 elements
        # Skip duplicate anchors
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Early termination: if smallest possible sum > 0, no more triplets
        if nums[i] > 0:
            break

        target = -nums[i]  # We need left + right = target
        left, right = i + 1, n - 1

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates on both sides
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < target:
                left += 1      # Need bigger sum
            else:
                right -= 1    # Need smaller sum

    return result


def test_exercise_4():
    print("\nExercise 4: 3Sum")

    result = three_sum([-1, 0, 1, 2, -1, -4])
    assert sorted([sorted(t) for t in result]) == sorted([[-1, -1, 2], [-1, 0, 1]])

    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]

    result = three_sum([-2, 0, 1, 1, 2])
    assert sorted([sorted(t) for t in result]) == sorted([[-2, 0, 2], [-2, 1, 1]])

    assert three_sum([]) == []
    assert three_sum([1]) == []

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Longest Repeating Character Replacement — Sliding Window
# =============================================================================

def character_replacement(s: str, k: int) -> int:
    """
    Sliding window (variable) with frequency counter.

    INTUITION:
    - For a window of size (right - left + 1):
      * max_freq = count of the most frequent char in window
      * replacements_needed = window_size - max_freq
      * If replacements_needed <= k, the window is valid
    - Expand right to grow the window
    - If invalid, shrink left

    OPTIMIZATION TRICK:
    - We don't need to DECREMENT max_freq when shrinking
    - max_freq only matters when it INCREASES (to find a bigger window)
    - A stale (too-high) max_freq just means we won't shrink unnecessarily
    - The answer only improves when max_freq increases, so this is safe

    Time: O(n) — each pointer moves at most n times
    Space: O(26) = O(1) — frequency counter for uppercase letters
    """
    count = {}       # char → frequency in current window
    max_freq = 0     # Max frequency of any single char in current window
    left = 0
    best = 0

    for right in range(len(s)):
        char = s[right]
        count[char] = count.get(char, 0) + 1
        max_freq = max(max_freq, count[char])  # Update if this char is now most frequent

        # Window size = right - left + 1
        # Replacements needed = window_size - max_freq
        window_size = right - left + 1
        if window_size - max_freq > k:
            # Too many replacements needed → shrink from left
            count[s[left]] -= 1
            left += 1
            # Note: we do NOT update max_freq here (optimization trick above)

        best = max(best, right - left + 1)

    return best


def test_exercise_5():
    print("\nExercise 5: Longest Repeating Character Replacement")

    assert character_replacement("ABAB", 2) == 4
    assert character_replacement("AABABBA", 1) == 4
    assert character_replacement("AAAA", 0) == 4
    assert character_replacement("ABCD", 3) == 4
    assert character_replacement("A", 0) == 1
    assert character_replacement("ABAB", 0) == 1
    assert character_replacement("AAAB", 0) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Contiguous Array (Equal 0s and 1s) — Prefix Sum
# =============================================================================

def find_max_length(nums: list[int]) -> int:
    """
    Prefix sum + hashmap after transforming 0 → -1.

    TRANSFORMATION:
    - Replace every 0 with -1
    - Now: subarray with equal 0s and 1s ↔ subarray with sum = 0
    - subarray sum = 0 ↔ prefix[i] == prefix[j] for some i < j

    ALGORITHM:
    - Maintain running prefix sum
    - Store the FIRST index where each prefix sum value appears
    - If we see the same prefix sum again, the subarray between those
      two indices has sum 0 → equal 0s and 1s

    WHY store the FIRST index (not the last)?
    - We want the LONGEST subarray, so we want the earliest occurrence
      of each prefix sum to maximize (current_index - first_index)

    WHY seen = {0: -1}?
    - A prefix sum of 0 at index j means nums[0..j] sums to 0
    - We treat it as if we saw prefix sum 0 at index -1 (before the array)
    - Length = j - (-1) = j + 1 (the entire subarray from start)

    Time: O(n) — single pass
    Space: O(n) — hash map stores at most n+1 distinct prefix sums
    """
    if not nums:
        return 0

    prefix = 0
    seen = {0: -1}  # prefix_sum → first index where this sum appeared
    best = 0

    for i, num in enumerate(nums):
        # Transform: 0 → -1, 1 stays 1
        prefix += 1 if num == 1 else -1

        if prefix in seen:
            # Subarray from seen[prefix]+1 to i has sum 0 → equal 0s and 1s
            length = i - seen[prefix]
            best = max(best, length)
        else:
            # First time seeing this prefix sum — record the index
            seen[prefix] = i

    return best


def test_exercise_6():
    print("\nExercise 6: Contiguous Array (Equal 0s and 1s)")

    assert find_max_length([0, 1]) == 2
    assert find_max_length([0, 1, 0]) == 2
    assert find_max_length([0, 0, 1, 0, 0, 0, 1, 1]) == 6
    assert find_max_length([0, 0, 0, 1, 1, 1]) == 6
    assert find_max_length([1, 1, 1, 1]) == 0
    assert find_max_length([0]) == 0
    assert find_max_length([]) == 0
    assert find_max_length([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]) == 12

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 7 (Hard): Sliding Window Maximum — Monotonic Deque
# =============================================================================

def max_sliding_window_brute(nums: list[int], k: int) -> list[int]:
    """
    Brute force: compute max() for each window.
    Time: O(n * k) — max() is O(k) per window
    Space: O(1) extra
    """
    if not nums:
        return []
    result = []
    for i in range(len(nums) - k + 1):
        result.append(max(nums[i:i + k]))
    return result


def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    Monotonic decreasing deque of INDICES.

    INVARIANT:
    - Deque contains indices in DECREASING order of their values
    - Front of deque = index of maximum in current window
    - Any index in deque whose value is less than nums[i] is useless
      (it can never be the max while nums[i] is in the window)

    OPERATIONS at each step i:
    1. EXPIRE: remove indices from front that are outside the window (< i - k + 1)
    2. CLEAN: remove indices from back whose values ≤ nums[i]
       (they'll never be the max — nums[i] is bigger AND stays longer)
    3. ADD: append index i to back
    4. RECORD: if window is full (i >= k-1), front of deque is the max

    WHY indices (not values)?
    - We need to know WHEN to expire an element (is it still in the window?)
    - Values alone don't tell us position

    Time: O(n) — each index is pushed and popped at most once from the deque
    Space: O(k) — deque holds at most k indices
    """
    if not nums:
        return []

    dq = deque()  # Stores INDICES, not values
    result = []

    for i in range(len(nums)):
        # 1. EXPIRE: remove indices outside the current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 2. CLEAN: remove indices whose values are <= current
        #    They can never be the max — nums[i] is bigger AND newer
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        # 3. ADD: push current index
        dq.append(i)

        # 4. RECORD: once we have a full window, record the max
        if i >= k - 1:
            result.append(nums[dq[0]])  # Front of deque = max of window

    return result


def test_exercise_7():
    print("\nExercise 7: Sliding Window Maximum")

    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([1, -1], 1) == [1, -1]
    assert max_sliding_window([9, 11], 2) == [11]
    assert max_sliding_window([4, 3, 2, 1], 3) == [4, 3]
    assert max_sliding_window([1, 2, 3, 4], 3) == [3, 4]
    assert max_sliding_window([5, 5, 5, 5], 2) == [5, 5, 5]
    assert max_sliding_window([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11], 3) == [10, 9, 8, 7, 6, 5, 4, 3, 11]

    # Verify against brute force on random input
    import random
    nums = [random.randint(-100, 100) for _ in range(1000)]
    k = 50
    assert max_sliding_window(nums, k) == max_sliding_window_brute(nums, k), "Mismatch with brute force!"

    print("  PASS — all test cases (including random brute-force verification)")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 100_000
    k = 1_000
    nums_large = [random.randint(-1000, 1000) for _ in range(n)]

    start = time.perf_counter()
    r1 = max_sliding_window_brute(nums_large, k)
    brute_time = time.perf_counter() - start
    print(f"  Brute force O(n*k): {brute_time:.4f}s")

    start = time.perf_counter()
    r2 = max_sliding_window(nums_large, k)
    deque_time = time.perf_counter() - start
    print(f"  Monotonic deque O(n): {deque_time:.6f}s")
    print(f"  Speedup: {brute_time / max(deque_time, 1e-9):.0f}x")

    assert r1 == r2, "Results differ!"


# =============================================================================
# EXERCISE 8 (Hard): Trapping Rain Water with Follow-ups
# =============================================================================

# --- Part A: Three approaches ---

def trap_brute(height: list[int]) -> int:
    """
    Brute force: for each bar, find max left and max right.
    Water at position i = min(max_left, max_right) - height[i]

    Time: O(n^2) — scanning left/right for each position
    Space: O(1)
    """
    n = len(height)
    if n < 3:
        return 0

    total = 0
    for i in range(n):
        max_left = max(height[:i + 1])    # O(i)
        max_right = max(height[i:])       # O(n-i)
        total += min(max_left, max_right) - height[i]
    return total


def trap_prefix(height: list[int]) -> int:
    """
    Prefix/suffix max arrays: precompute the max from each direction.

    left_max[i] = max(height[0], ..., height[i])
    right_max[i] = max(height[i], ..., height[n-1])

    Water at i = min(left_max[i], right_max[i]) - height[i]

    Time: O(n) — three passes
    Space: O(n) — two extra arrays
    """
    n = len(height)
    if n < 3:
        return 0

    # Build left_max: max height seen from left up to each index
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # Build right_max: max height seen from right up to each index
    right_max = [0] * n
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # Compute water at each position
    total = 0
    for i in range(n):
        total += min(left_max[i], right_max[i]) - height[i]
    return total


def trap_two_pointers(height: list[int]) -> int:
    """
    Two pointers: O(n) time, O(1) space — the optimal solution.

    INVARIANT:
    - left_max tracks the max height seen from the left up to 'left'
    - right_max tracks the max height seen from the right up to 'right'
    - Process the side with the SMALLER max:
      * If left_max < right_max: water at 'left' is bounded by left_max
        (right side is guaranteed to have something >= right_max >= left_max)
      * If left_max >= right_max: water at 'right' is bounded by right_max

    WHY we can trust the opposite side:
    - If left_max < right_max, we know there exists a bar on the right
      that is at least right_max tall. So the bottleneck is definitely left_max.
    - We don't need to know the exact max on the other side — just that it's
      at least as big as our current max.

    Time: O(n)
    Space: O(1)
    """
    n = len(height)
    if n < 3:
        return 0

    left, right = 0, n - 1
    left_max, right_max = height[left], height[right]
    total = 0

    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            total += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            total += right_max - height[right]

    return total


# --- Part B: Detailed water per position ---

def trap_detailed(height: list[int]) -> tuple[int, list[int]]:
    """
    Return total water AND water at each position.
    Uses the prefix/suffix approach to get per-position detail.

    Time: O(n)
    Space: O(n)
    """
    n = len(height)
    if n < 3:
        return 0, [0] * n

    # Build left_max and right_max
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    right_max = [0] * n
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # Water at each position
    water = [0] * n
    total = 0
    for i in range(n):
        water[i] = min(left_max[i], right_max[i]) - height[i]
        total += water[i]

    return total, water


# --- Part C: Count separate pools ---

def count_pools(height: list[int]) -> int:
    """
    Count contiguous groups of positions with trapped water > 0.

    Strategy:
    1. Compute water at each position (using trap_detailed)
    2. Count transitions from 0 to positive → each is a new pool

    Time: O(n)
    Space: O(n) for the water array
    """
    _, water = trap_detailed(height)

    pools = 0
    in_pool = False

    for w in water:
        if w > 0 and not in_pool:
            pools += 1      # Start of a new pool
            in_pool = True
        elif w == 0:
            in_pool = False  # End of current pool (if any)

    return pools


def test_exercise_8():
    print("\nExercise 8: Trapping Rain Water with Follow-ups")

    # Part A
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    assert trap_brute(height) == 6
    assert trap_prefix(height) == 6
    assert trap_two_pointers(height) == 6

    assert trap_two_pointers([4, 2, 0, 3, 2, 5]) == 9
    assert trap_two_pointers([]) == 0
    assert trap_two_pointers([1]) == 0
    assert trap_two_pointers([1, 2]) == 0
    assert trap_two_pointers([2, 0, 2]) == 2
    assert trap_two_pointers([3, 0, 0, 0, 3]) == 9
    print("  Part A — PASS")

    # Part B
    total, water = trap_detailed([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
    assert total == 6
    assert water == [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]
    print("  Part B — PASS")

    # Part C
    # water = [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]
    # pools: [2] → pool 1, [4,5,6] → pool 2, [9] → pool 3
    assert count_pools([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 3
    assert count_pools([2, 0, 2]) == 1
    assert count_pools([3, 0, 0, 0, 3]) == 1
    assert count_pools([1, 2, 3]) == 0
    assert count_pools([3, 1, 2, 1, 3]) == 1
    print("  Part C — PASS")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 2: Arrays & Strings")
    print("=" * 70)

    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    test_exercise_6()
    test_exercise_7()
    test_exercise_8()

    print("\n" + "=" * 70)
    print("ALL SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

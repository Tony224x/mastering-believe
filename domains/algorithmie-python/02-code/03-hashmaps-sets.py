"""
Day 3 — Hash Maps & Sets: Frequency Counting, Grouping, Two-Sum Patterns
Run: python domains/algorithmie-python/02-code/03-hashmaps-sets.py

Each section shows the BRUTE FORCE first, then the OPTIMIZED version with hash maps,
so you can FEEL the improvement and understand WHY hash maps are the #1 interview tool.
"""

import time
import random
from collections import Counter, defaultdict


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
# SECTION 1: FREQUENCY COUNTING
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 1: Valid Anagram (LeetCode 242)
# Given two strings, determine if t is an anagram of s.
# ---------------------------------------------------------------------------

@timed
def is_anagram_brute(s: str, t: str) -> bool:
    """
    Brute force: sort both strings and compare.
    Time: O(n log n) — sorting dominates
    Space: O(n) — sorted copies
    """
    return sorted(s) == sorted(t)


@timed
def is_anagram(s: str, t: str) -> bool:
    """
    Frequency counting with Counter.

    INTUITION:
    - Two strings are anagrams if and only if they have the same
      character frequencies.
    - Counter builds a frequency dict in one pass.

    Time: O(n) — two passes to build counters, O(1) comparison (bounded by alphabet)
    Space: O(1) — at most 26 entries for lowercase English letters
    """
    if len(s) != len(t):
        return False                   # Quick exit: different lengths can't be anagrams
    return Counter(s) == Counter(t)    # Counter equality checks all frequencies


@timed
def is_anagram_manual(s: str, t: str) -> bool:
    """
    Manual frequency counting without imports — useful when interviewer
    wants to see you build it from scratch.

    TECHNIQUE: increment for s, decrement for t. If any count goes negative, fail.

    Time: O(n)
    Space: O(1) — bounded by alphabet size
    """
    if len(s) != len(t):
        return False

    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1    # .get avoids KeyError, returns 0 if missing
    for c in t:
        freq[c] = freq.get(c, 0) - 1    # Decrement for each char in t
        if freq[c] < 0:                  # More of this char in t than in s
            return False
    return True


def demo_anagram():
    print("\n" + "=" * 70)
    print("FREQUENCY COUNTING #1: Valid Anagram")
    print("=" * 70)

    # Correctness
    test_cases = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("", "", True),
        ("a", "a", True),
        ("ab", "a", False),     # Different lengths
    ]
    for s, t, expected in test_cases:
        assert is_anagram(s, t) == expected, f"Failed for ({s}, {t})"
        assert is_anagram_manual(s, t) == expected, f"Manual failed for ({s}, {t})"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 100_000
    chars = "abcdefghijklmnopqrstuvwxyz"
    s_large = "".join(random.choice(chars) for _ in range(n))
    # Create an anagram by shuffling
    t_large = "".join(random.sample(s_large, len(s_large)))

    is_anagram_brute(s_large, t_large)
    is_anagram(s_large, t_large)
    is_anagram_manual(s_large, t_large)


# ---------------------------------------------------------------------------
# Problem 2: Top K Frequent Elements (LeetCode 347)
# Given an integer array and an integer k, return the k most frequent elements.
# ---------------------------------------------------------------------------

@timed
def top_k_frequent_brute(nums: list[int], k: int) -> list[int]:
    """
    Brute force: count frequencies, sort by frequency, take top k.
    Time: O(n log n) — sorting all unique elements
    Space: O(n)
    """
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1
    # Sort by frequency (descending), take first k
    sorted_by_freq = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    return sorted_by_freq[:k]


@timed
def top_k_frequent_counter(nums: list[int], k: int) -> list[int]:
    """
    Counter.most_common — Pythonic and clean.

    Under the hood: most_common uses heapq.nlargest which is O(n + k log n).
    For small k, this is effectively O(n).

    Time: O(n + k log n)
    Space: O(n)
    """
    return [x for x, _ in Counter(nums).most_common(k)]


@timed
def top_k_frequent_bucket(nums: list[int], k: int) -> list[int]:
    """
    Bucket sort approach — TRUE O(n).

    INTUITION:
    - Max possible frequency = len(nums)
    - Create buckets[i] = list of elements with frequency i
    - Walk buckets from highest to lowest, collecting k elements

    WHY this is O(n):
    - Building frequency map: O(n)
    - Filling buckets: O(n) (each element placed once)
    - Collecting top k: O(n) worst case (but usually much less)

    Time: O(n)
    Space: O(n) — buckets array
    """
    freq = Counter(nums)

    # Buckets: index = frequency, value = list of elements with that frequency
    # Max frequency possible = len(nums) (all same element)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    # Collect from highest frequency down
    result = []
    for i in range(len(buckets) - 1, 0, -1):  # Start from highest frequency
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result


def demo_top_k():
    print("\n" + "=" * 70)
    print("FREQUENCY COUNTING #2: Top K Frequent Elements")
    print("=" * 70)

    # Correctness
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    # Top 2: [1, 2] (order doesn't matter between same-frequency elements)
    result_counter = set(top_k_frequent_counter(nums, k))
    result_bucket = set(top_k_frequent_bucket(nums, k))
    assert result_counter == {1, 2}
    assert result_bucket == {1, 2}

    # Single element
    assert top_k_frequent_counter([1], 1) == [1]
    assert top_k_frequent_bucket([1], 1) == [1]
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 200_000
    nums_large = [random.randint(1, 1000) for _ in range(n)]
    k = 10

    top_k_frequent_brute(nums_large, k)
    top_k_frequent_counter(nums_large, k)
    top_k_frequent_bucket(nums_large, k)


# ---------------------------------------------------------------------------
# Problem 3: First Unique Character in a String (LeetCode 387)
# Return the index of the first non-repeating character.
# ---------------------------------------------------------------------------

@timed
def first_unique_char_brute(s: str) -> int:
    """
    Brute force: for each character, scan entire string for duplicates.
    Time: O(n^2) — nested scan
    Space: O(1)
    """
    for i in range(len(s)):
        is_unique = True
        for j in range(len(s)):
            if i != j and s[i] == s[j]:
                is_unique = False
                break
        if is_unique:
            return i
    return -1


@timed
def first_unique_char(s: str) -> int:
    """
    Two-pass frequency counting.

    Pass 1: build frequency map — O(n)
    Pass 2: find first char with frequency 1 — O(n)

    IMPORTANT: we iterate over the STRING in pass 2 (not the dict)
    because we need the first index by POSITION, not by insertion order.

    Time: O(n)
    Space: O(1) — at most 26 chars
    """
    freq = Counter(s)                  # Pass 1: count all characters
    for i, c in enumerate(s):          # Pass 2: find first with count == 1
        if freq[c] == 1:
            return i
    return -1


def demo_first_unique():
    print("\n" + "=" * 70)
    print("FREQUENCY COUNTING #3: First Unique Character")
    print("=" * 70)

    # Correctness
    test_cases = [
        ("leetcode", 0),        # 'l' is first unique
        ("loveleetcode", 2),    # 'v' is first unique
        ("aabb", -1),           # No unique character
        ("a", 0),               # Single char
        ("aadadaad", -1),       # All repeated
    ]
    for s, expected in test_cases:
        assert first_unique_char(s) == expected, f"Failed for '{s}'"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 10_000
    # Create string where only the last char is unique
    s_large = "".join(random.choice("abcdefghij") for _ in range(n - 1)) + "z"

    first_unique_char_brute(s_large)
    first_unique_char(s_large)


# =============================================================================
# SECTION 2: GROUPING / BUCKETING
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 4: Group Anagrams (LeetCode 49)
# Group strings that are anagrams of each other.
# ---------------------------------------------------------------------------

@timed
def group_anagrams_brute(strs: list[str]) -> list[list[str]]:
    """
    Brute force: compare every pair of strings for anagram status.
    Time: O(n^2 * k) — n strings, comparing each pair, each comparison O(k)
    Space: O(n * k)
    """
    used = [False] * len(strs)
    groups = []

    for i in range(len(strs)):
        if used[i]:
            continue
        group = [strs[i]]
        used[i] = True
        for j in range(i + 1, len(strs)):
            if not used[j] and sorted(strs[i]) == sorted(strs[j]):
                group.append(strs[j])
                used[j] = True
        groups.append(group)

    return groups


@timed
def group_anagrams_sort(strs: list[str]) -> list[list[str]]:
    """
    Grouping with sorted string as key.

    KEY INSIGHT: all anagrams produce the same sorted string.
    "eat" → "aet", "tea" → "aet", "ate" → "aet"

    We use a defaultdict(list) to group all strings sharing the same sorted form.

    Time: O(n * k log k) — sorting each string of length k
    Space: O(n * k) — storing all strings in groups
    """
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))      # Sorted chars as tuple (hashable)
        groups[key].append(s)
    return list(groups.values())


@timed
def group_anagrams_count(strs: list[str]) -> list[list[str]]:
    """
    Optimal: character COUNT as key (avoids sorting).

    Instead of sorting each string O(k log k), we build a frequency
    tuple in O(k). This is faster when strings are long.

    Time: O(n * k) — counting chars is O(k), no sorting
    Space: O(n * k)
    """
    groups = defaultdict(list)
    for s in strs:
        # Build frequency array for 26 lowercase letters
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        key = tuple(count)           # Tuple is hashable, list is not
        groups[key].append(s)
    return list(groups.values())


def demo_group_anagrams():
    print("\n" + "=" * 70)
    print("GROUPING #1: Group Anagrams")
    print("=" * 70)

    # Correctness
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result_sort = group_anagrams_sort(strs)
    result_count = group_anagrams_count(strs)

    # Convert to sets of frozensets for order-independent comparison
    def normalize(groups):
        return set(frozenset(g) for g in groups)

    expected = normalize([["eat", "tea", "ate"], ["tan", "nat"], ["bat"]])
    assert normalize(result_sort) == expected
    assert normalize(result_count) == expected

    # Edge cases
    assert group_anagrams_sort([""]) == [[""]]
    assert group_anagrams_sort(["a"]) == [["a"]]
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 5_000
    k = 20   # Average string length
    chars = "abcdefghijklmnopqrstuvwxyz"
    strs_large = ["".join(random.choice(chars) for _ in range(random.randint(1, k)))
                   for _ in range(n)]

    group_anagrams_brute(strs_large)
    group_anagrams_sort(strs_large)
    group_anagrams_count(strs_large)


# ---------------------------------------------------------------------------
# Problem 5: Group by Digit Sum
# Group numbers that share the same digit sum.
# ---------------------------------------------------------------------------

@timed
def group_by_digit_sum_brute(nums: list[int]) -> dict[int, list[int]]:
    """
    Brute force: for each number, find its group by scanning all previous.
    Time: O(n^2) in worst case (comparing digit sums)
    Space: O(n)
    """
    groups = []
    sums = []

    for num in nums:
        dsum = sum(int(d) for d in str(abs(num)))
        found = False
        for i, s in enumerate(sums):
            if s == dsum:
                groups[i].append(num)
                found = True
                break
        if not found:
            sums.append(dsum)
            groups.append([num])

    return {sums[i]: groups[i] for i in range(len(sums))}


@timed
def group_by_digit_sum(nums: list[int]) -> dict[int, list[int]]:
    """
    defaultdict(list) — O(n) grouping.

    PATTERN: compute a "key" for each element, append to that key's group.
    This is the canonical grouping pattern.

    Time: O(n * d) where d = number of digits per number
    Space: O(n)
    """
    groups = defaultdict(list)
    for num in nums:
        digit_sum = sum(int(d) for d in str(abs(num)))
        groups[digit_sum].append(num)
    return dict(groups)


def demo_group_digit_sum():
    print("\n" + "=" * 70)
    print("GROUPING #2: Group by Digit Sum")
    print("=" * 70)

    # Correctness
    nums = [18, 36, 27, 45, 99, 123, 15]
    result = group_by_digit_sum(nums)
    # 18→9, 36→9, 27→9, 45→9, 99→18, 123→6, 15→6
    assert set(result[9]) == {18, 36, 27, 45}
    assert set(result[18]) == {99}
    assert set(result[6]) == {123, 15}

    # Edge case: single element
    result = group_by_digit_sum([0])
    assert result == {0: [0]}
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 10_000
    nums_large = [random.randint(0, 99999) for _ in range(n)]
    group_by_digit_sum_brute(nums_large)
    group_by_digit_sum(nums_large)


# =============================================================================
# SECTION 3: TWO-SUM FAMILY (Complement Lookup)
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 6: Two Sum (LeetCode 1) — THE most famous interview problem
# Given an array and a target, return indices of two numbers that sum to target.
# ---------------------------------------------------------------------------

@timed
def two_sum_brute(nums: list[int], target: int) -> list[int]:
    """
    Brute force: try every pair.
    Time: O(n^2) — nested loops
    Space: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


@timed
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Hash map as complement lookup table.

    INTUITION:
    - For each number nums[i], we need to find target - nums[i] somewhere else
    - Instead of scanning the entire array (O(n)), look it up in a dict (O(1))
    - Store each number as we go: seen[value] = index

    CRITICAL: store AFTER checking the complement.
    - If we store first, we might use the same element twice
    - Example: nums=[3,3], target=6
      * i=0: complement=3, not in seen → store seen[3]=0
      * i=1: complement=3, found in seen → return [0, 1] ✓
    - If we stored first: i=0: store seen[3]=0, complement=3, found! → [0, 0] ✗

    Time: O(n) — single pass, O(1) per lookup
    Space: O(n) — hash map stores at most n elements
    """
    seen = {}     # value → index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i                    # Store AFTER checking (avoid self-match)

    return []


def demo_two_sum():
    print("\n" + "=" * 70)
    print("TWO-SUM FAMILY #1: Two Sum (the classic)")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),       # Duplicate values
    ]
    for nums, target, expected in test_cases:
        assert two_sum(nums, target) == expected, f"Failed for {nums}, target={target}"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 10_000
    nums_large = list(range(n))
    target_large = nums_large[-1] + nums_large[-2]  # Worst case: last two elements

    two_sum_brute(nums_large, target_large)
    two_sum(nums_large, target_large)


# ---------------------------------------------------------------------------
# Problem 7: 4Sum II (LeetCode 454)
# Given four arrays, count tuples (i,j,k,l) where A[i]+B[j]+C[k]+D[l]=0.
# ---------------------------------------------------------------------------

@timed
def four_sum_count_brute(nums1, nums2, nums3, nums4):
    """
    Brute force: try every combination.
    Time: O(n^4) — four nested loops
    Space: O(1)
    """
    count = 0
    for a in nums1:
        for b in nums2:
            for c in nums3:
                for d in nums4:
                    if a + b + c + d == 0:
                        count += 1
    return count


@timed
def four_sum_count(nums1, nums2, nums3, nums4):
    """
    Split into two groups + hash map lookup.

    INTUITION:
    - A + B + C + D = 0  ⟺  A + B = -(C + D)
    - Precompute ALL sums of A+B, store their frequencies
    - For each C+D sum, look up how many A+B sums are the complement

    This is a generalization of Two Sum to four arrays:
    - Two Sum: for each element, find its complement → O(n)
    - 4Sum II: for each pair (A,B), find complement pair (C,D) → O(n^2)

    Time: O(n^2) — two passes of O(n^2) each
    Space: O(n^2) — storing all A+B sums
    """
    # Phase 1: compute all A+B sums and their frequencies
    ab_sums = Counter()               # sum → count of pairs producing this sum
    for a in nums1:
        for b in nums2:
            ab_sums[a + b] += 1

    # Phase 2: for each C+D, find complement in ab_sums
    count = 0
    for c in nums3:
        for d in nums4:
            complement = -(c + d)
            count += ab_sums[complement]  # Counter returns 0 if key missing

    return count


def demo_four_sum():
    print("\n" + "=" * 70)
    print("TWO-SUM FAMILY #2: 4Sum II")
    print("=" * 70)

    # Correctness
    nums1 = [1, 2]
    nums2 = [-2, -1]
    nums3 = [-1, 2]
    nums4 = [0, 2]
    # Tuples: (0,0,0,1), (1,1,0,0) → count = 2
    assert four_sum_count(nums1, nums2, nums3, nums4) == 2

    # All zeros
    assert four_sum_count([0], [0], [0], [0]) == 1

    # No solution
    assert four_sum_count([1], [1], [1], [1]) == 0
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 50  # n=50 → brute force tries 50^4 = 6.25M combinations
    nums = [[random.randint(-100, 100) for _ in range(n)] for _ in range(4)]

    r1 = four_sum_count_brute(*nums)
    r2 = four_sum_count(*nums)
    assert r1 == r2, f"Results differ: {r1} vs {r2}"


# ---------------------------------------------------------------------------
# Problem 8: Count Pairs with Difference K (variant of LeetCode 532)
# Count pairs (i,j) with i < j such that |nums[i] - nums[j]| == k.
# ---------------------------------------------------------------------------

@timed
def count_pairs_diff_brute(nums: list[int], k: int) -> int:
    """
    Brute force: check every pair.
    Time: O(n^2)
    Space: O(1)
    """
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(nums[i] - nums[j]) == k:
                count += 1
    return count


@timed
def count_pairs_diff(nums: list[int], k: int) -> int:
    """
    Frequency counting + complement lookup.

    INTUITION:
    - |a - b| == k  means  a - b == k  OR  b - a == k
    - Equivalently: for each number a, check if (a + k) or (a - k) exists
    - Use a Counter to know how many of each value exist

    EDGE CASE k == 0:
    - We need pairs of identical values: C(count, 2) = count*(count-1)/2

    Time: O(n)
    Space: O(n)
    """
    freq = Counter(nums)
    count = 0

    if k < 0:
        k = abs(k)   # Absolute difference, so k must be >= 0

    for num in freq:
        if k > 0:
            # Check if num + k exists (avoids double-counting)
            if num + k in freq:
                count += freq[num] * freq[num + k]
        else:
            # k == 0: pairs within the same value → C(n, 2)
            count += freq[num] * (freq[num] - 1) // 2

    return count


def demo_count_pairs_diff():
    print("\n" + "=" * 70)
    print("TWO-SUM FAMILY #3: Count Pairs with Difference K")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([1, 5, 3, 4, 2], 2, 3),       # (1,3), (3,5), (2,4)
        ([1, 2, 3, 4, 5], 1, 4),       # (1,2), (2,3), (3,4), (4,5)
        ([3, 1, 4, 1, 5], 0, 1),       # (1,1) — one pair of duplicates
        ([1, 1, 1, 1], 0, 6),          # C(4,2) = 6 pairs
        ([1, 2, 3], 5, 0),             # No pair with diff 5
    ]
    for nums, k, expected in test_cases:
        brute = count_pairs_diff_brute(nums, k)
        optimized = count_pairs_diff(nums, k)
        assert brute == expected, f"Brute failed for {nums}, k={k}: got {brute}"
        assert optimized == expected, f"Optimized failed for {nums}, k={k}: got {optimized}"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 10_000
    nums_large = [random.randint(1, 1000) for _ in range(n)]
    k = 5

    r1 = count_pairs_diff_brute(nums_large, k)
    r2 = count_pairs_diff(nums_large, k)
    assert r1 == r2, f"Results differ: {r1} vs {r2}"


# =============================================================================
# SECTION 4: SEEN SET / VISITED
# =============================================================================

# ---------------------------------------------------------------------------
# Problem 9: Contains Duplicate II (LeetCode 219)
# Check if there exist two indices i, j with nums[i]==nums[j] and |i-j|<=k.
# ---------------------------------------------------------------------------

@timed
def contains_nearby_duplicate_brute(nums: list[int], k: int) -> bool:
    """
    Brute force: check every pair within distance k.
    Time: O(n * k) — for each element, check up to k neighbors
    Space: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, min(i + k + 1, n)):
            if nums[i] == nums[j]:
                return True
    return False


@timed
def contains_nearby_duplicate(nums: list[int], k: int) -> bool:
    """
    Sliding window as a SET.

    INTUITION:
    - Maintain a set of the last k elements
    - For each new element: check if it's in the set (O(1))
    - If yes → duplicate within k distance
    - If no → add it, and evict the oldest if set size > k

    This is the Seen Set pattern combined with a sliding window.
    The set acts as a fixed-size window of the last k values.

    Time: O(n) — each element: one O(1) lookup + one O(1) add + possibly one O(1) remove
    Space: O(k) — set holds at most k elements
    """
    window = set()

    for i, num in enumerate(nums):
        if num in window:
            return True             # Duplicate found within window of size k
        window.add(num)
        if len(window) > k:
            window.remove(nums[i - k])  # Remove element that just left the window
    return False


def demo_contains_nearby():
    print("\n" + "=" * 70)
    print("SEEN SET #1: Contains Duplicate Within K")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([1, 2, 3, 1], 3, True),           # nums[0]==nums[3], |0-3|=3 <= 3
        ([1, 0, 1, 1], 1, True),           # nums[2]==nums[3], |2-3|=1 <= 1
        ([1, 2, 3, 1, 2, 3], 2, False),    # Closest duplicate is at distance 3
        ([1, 2, 1], 0, False),              # k=0, no same-index allowed
        ([1], 1, False),                     # Single element
        ([], 0, False),                      # Empty
    ]
    for nums, k, expected in test_cases:
        assert contains_nearby_duplicate(nums, k) == expected, f"Failed for {nums}, k={k}"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 50_000
    k = 100
    nums_large = list(range(n))  # No duplicates → worst case (full scan)

    contains_nearby_duplicate_brute(nums_large, k)
    contains_nearby_duplicate(nums_large, k)


# ---------------------------------------------------------------------------
# Problem 10: Longest Consecutive Sequence (LeetCode 128)
# Find the length of the longest consecutive elements sequence. Must be O(n).
# ---------------------------------------------------------------------------

@timed
def longest_consecutive_brute(nums: list[int]) -> int:
    """
    Brute force: sort then scan for consecutive runs.
    Time: O(n log n) — sorting
    Space: O(n) — sorted copy (or O(1) if sorting in-place)
    """
    if not nums:
        return 0

    nums_sorted = sorted(set(nums))     # Remove duplicates + sort
    best = 1
    current = 1

    for i in range(1, len(nums_sorted)):
        if nums_sorted[i] == nums_sorted[i - 1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


@timed
def longest_consecutive(nums: list[int]) -> int:
    """
    Set-based O(n) solution.

    INTUITION:
    - Put all numbers in a set for O(1) lookup
    - For each number, check if it's the START of a sequence
      (i.e., num-1 is NOT in the set)
    - If it's the start, count how far the sequence goes (num+1, num+2, ...)

    WHY this is O(n) despite nested loop:
    - The "if num - 1 not in num_set" check ensures we only start
      counting from the BEGINNING of each sequence
    - Each number is visited AT MOST TWICE:
      * Once in the outer for loop
      * Once inside a while loop (as part of exactly ONE sequence)
    - Total work = O(n) + O(n) = O(n)

    Time: O(n)
    Space: O(n) — the set
    """
    if not nums:
        return 0

    num_set = set(nums)     # O(n) build, O(1) lookups
    best = 0

    for num in num_set:
        # KEY: only start counting from the BEGINNING of a sequence
        if num - 1 not in num_set:
            # This is the start of a new sequence
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            best = max(best, length)

    return best


def demo_longest_consecutive():
    print("\n" + "=" * 70)
    print("SEEN SET #2: Longest Consecutive Sequence")
    print("=" * 70)

    # Correctness
    test_cases = [
        ([100, 4, 200, 1, 3, 2], 4),         # [1,2,3,4]
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9), # [0,1,2,3,4,5,6,7,8]
        ([1, 2, 0, 1], 3),                    # [0,1,2] — duplicates don't count
        ([], 0),
        ([1], 1),
        ([1, 3, 5, 7], 1),                    # No consecutive, max sequence = 1
    ]
    for nums, expected in test_cases:
        brute = longest_consecutive_brute(nums)
        optimized = longest_consecutive(nums)
        assert brute == expected, f"Brute failed for {nums}: got {brute}"
        assert optimized == expected, f"Optimized failed for {nums}: got {optimized}"
    print("  All test cases passed")

    # Performance comparison
    print("\n  Performance comparison:")
    n = 100_000
    nums_large = random.sample(range(n * 10), n)  # Sparse numbers

    longest_consecutive_brute(nums_large)
    longest_consecutive(nums_large)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   Day 3 -- Hash Maps & Sets: Frequency Counting, Grouping, Two-Sum")
    print("=" * 70)

    # Frequency Counting
    demo_anagram()
    demo_top_k()
    demo_first_unique()

    # Grouping
    demo_group_anagrams()
    demo_group_digit_sum()

    # Two-Sum Family
    demo_two_sum()
    demo_four_sum()
    demo_count_pairs_diff()

    # Seen Set / Visited
    demo_contains_nearby()
    demo_longest_consecutive()

    print("\n" + "=" * 70)
    print("DONE — Key observations:")
    print("  1. Frequency Counting: Counter/dict turns O(n^2) comparisons into O(n)")
    print("  2. Grouping: defaultdict(list) with a computed key = canonical pattern")
    print("  3. Two Sum pattern: dict as lookup table eliminates inner loop")
    print("  4. 4Sum II: split 4 arrays into 2 groups -> O(n^4) to O(n^2)")
    print("  5. Seen Set: sliding window + set for duplicate detection within K")
    print("  6. Longest Consecutive: 'start of sequence' check makes nested loop O(n)")
    print("=" * 70)

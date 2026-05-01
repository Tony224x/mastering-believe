"""
Day 7 — Sprint Solutions: 10 timed problems covering J1-J6 patterns
Run: python domains/algorithmie-python/02-code/07-sprint-exercices.py

Each problem:
- Comment block with target time (15 min) and pattern hint
- Final solution with complexity
- Tests via asserts

Use this file to COMPARE your sprint answers after you attempted the problems
on your own. Do NOT read this file before your sprint attempt.
"""

import time
from collections import Counter, defaultdict, deque
from typing import List, Optional


def timed(label):
    """Context manager that prints the elapsed time of a block."""
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"  {label}: {elapsed:.6f}s")
    return _Timer()


# =============================================================================
# P1 — TWO SUM (Easy) — target 5 min
# Pattern: J3 hash map complement lookup
# =============================================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Single pass with a dict storing value -> index.
    For each num, check if target - num was seen. Store AFTER the check.

    Time: O(n), Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# =============================================================================
# P2 — VALID ANAGRAM (Easy) — target 3 min
# Pattern: J3 frequency counting
# =============================================================================

def is_anagram(s: str, t: str) -> bool:
    """
    Two strings are anagrams iff they have the same character frequencies.

    Time: O(n), Space: O(1) — bounded by alphabet size
    """
    if len(s) != len(t):
        return False                 # Quick exit: different lengths can't match
    return Counter(s) == Counter(t)


# =============================================================================
# P3 — VALID PARENTHESES (Easy) — target 5 min
# Pattern: J4 stack matching
# =============================================================================

def is_valid_parens(s: str) -> bool:
    """
    LIFO matching: the most recently opened bracket must close first.

    Time: O(n), Space: O(n)
    """
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack                 # Leftover openers = unmatched


# =============================================================================
# P4 — BEST TIME TO BUY AND SELL STOCK (Easy) — target 5 min
# Pattern: J1 single pass with running minimum
# =============================================================================

def max_profit(prices: List[int]) -> int:
    """
    Track the minimum price seen so far. For each day, compute the profit
    as price - min_so_far and keep the maximum.

    Time: O(n), Space: O(1)
    """
    if not prices:
        return 0
    min_price = prices[0]
    best = 0
    for price in prices[1:]:
        best = max(best, price - min_price)      # Try selling today
        min_price = min(min_price, price)        # Consider buying today for future
    return best


# =============================================================================
# P5 — REVERSE LINKED LIST (Easy) — target 5 min
# Pattern: J5 three-pointer reversal
# =============================================================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative reversal. Save next BEFORE rewriting curr.next.

    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head
    while curr:
        nxt = curr.next               # Save before clobbering
        curr.next = prev
        prev = curr
        curr = nxt
    return prev                       # New head = old tail


# =============================================================================
# P6 — LONGEST SUBSTRING WITHOUT REPEATING CHARS (Medium) — target 10 min
# Pattern: J2 sliding window + hash set
# =============================================================================

def length_of_longest_substring(s: str) -> int:
    """
    Sliding window [left, right] that always contains distinct characters.
    When we see a duplicate, shrink from the left until it's gone.

    Time: O(n) — each character is added and removed at most once
    Space: O(min(n, alphabet))
    """
    seen = set()
    left = 0
    best = 0
    for right, c in enumerate(s):
        while c in seen:
            seen.remove(s[left])
            left += 1
        seen.add(c)
        best = max(best, right - left + 1)
    return best


# =============================================================================
# P7 — GROUP ANAGRAMS (Medium) — target 10 min
# Pattern: J3 grouping with defaultdict
# =============================================================================

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Use the sorted string as a grouping key. All anagrams share the same sorted form.

    Time: O(n * k log k) where k = max string length
    Space: O(n * k)
    """
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))        # Tuple is hashable; "ate" -> ('a','e','t')
        groups[key].append(s)
    return list(groups.values())


# =============================================================================
# P8 — TOP K FREQUENT ELEMENTS (Medium) — target 10 min
# Pattern: J3 + J6, Counter + bucket sort
# =============================================================================

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Bucket sort: index each bucket by frequency. O(n) total.

    Time: O(n), Space: O(n)
    """
    freq = Counter(nums)
    # Buckets: index i holds numbers appearing exactly i times. max freq <= n.
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    # Walk buckets from highest frequency down
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result


# =============================================================================
# P9 — SEARCH IN ROTATED SORTED ARRAY (Medium) — target 12 min
# Pattern: J6 modified binary search
# =============================================================================

def search_rotated(nums: List[int], target: int) -> int:
    """
    At each iteration, ONE half is guaranteed sorted. Check which, then decide.

    Time: O(log n), Space: O(1)
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:
            # Left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


# =============================================================================
# P10 — NUMBER OF ISLANDS (Medium) — target 12 min
# Pattern: J4 BFS on zone
# =============================================================================

def num_islands(zone: List[List[str]]) -> int:
    """
    Scan every cell. When we find an unvisited land cell, BFS to flood-fill
    the whole island and increment the counter.

    Time: O(rows * cols), Space: O(rows * cols)
    """
    if not zone or not zone[0]:
        return 0
    rows, cols = len(zone), len(zone[0])
    islands = 0

    for r in range(rows):
        for c in range(cols):
            if zone[r][c] != "1":
                continue
            islands += 1
            # BFS flood fill
            queue = deque([(r, c)])
            zone[r][c] = "0"          # Mark at enqueue
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and zone[nr][nc] == "1"):
                        zone[nr][nc] = "0"
                        queue.append((nr, nc))
    return islands


# =============================================================================
# TEST SUITE — run each problem and assert correctness
# =============================================================================

def run_all_sprint():
    print("\n" + "=" * 70)
    print("SPRINT SOLUTIONS — running all 10 problems")
    print("=" * 70)

    # P1
    with timed("P1 two_sum"):
        assert two_sum([2, 7, 11, 15], 9) == [0, 1]
        assert two_sum([3, 2, 4], 6) == [1, 2]
        assert two_sum([3, 3], 6) == [0, 1]

    # P2
    with timed("P2 is_anagram"):
        assert is_anagram("anagram", "nagaram") is True
        assert is_anagram("rat", "car") is False
        assert is_anagram("", "") is True

    # P3
    with timed("P3 is_valid_parens"):
        assert is_valid_parens("()[]{}") is True
        assert is_valid_parens("([)]") is False
        assert is_valid_parens("") is True

    # P4
    with timed("P4 max_profit"):
        assert max_profit([7, 1, 5, 3, 6, 4]) == 5
        assert max_profit([7, 6, 4, 3, 1]) == 0
        assert max_profit([]) == 0

    # P5
    def build(vals):
        dummy = ListNode(0)
        t = dummy
        for v in vals:
            t.next = ListNode(v)
            t = t.next
        return dummy.next

    def to_list(h):
        out = []
        while h:
            out.append(h.val)
            h = h.next
        return out

    with timed("P5 reverse_list"):
        assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
        assert to_list(reverse_list(build([1]))) == [1]
        assert reverse_list(None) is None

    # P6
    with timed("P6 length_of_longest_substring"):
        assert length_of_longest_substring("abcabcbb") == 3
        assert length_of_longest_substring("bbbbb") == 1
        assert length_of_longest_substring("pwwkew") == 3
        assert length_of_longest_substring("") == 0

    # P7
    with timed("P7 group_anagrams"):
        result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
        normalized = sorted([sorted(g) for g in result])
        expected = sorted([sorted(g) for g in [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]])
        assert normalized == expected

    # P8
    with timed("P8 top_k_frequent"):
        assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
        assert top_k_frequent([1], 1) == [1]

    # P9
    with timed("P9 search_rotated"):
        assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
        assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
        assert search_rotated([1], 0) == -1

    # P10
    with timed("P10 num_islands"):
        zone = [
            ["1", "1", "0", "0", "0"],
            ["1", "1", "0", "0", "0"],
            ["0", "0", "1", "0", "0"],
            ["0", "0", "0", "1", "1"],
        ]
        assert num_islands([row[:] for row in zone]) == 3
        assert num_islands([["0"]]) == 0
        assert num_islands([["1"]]) == 1

    print("\nAll 10 sprint problems passed.")


if __name__ == "__main__":
    run_all_sprint()

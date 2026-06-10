"""
Solutions — Day 7 Sprint (10 problems)
Run: python domains/algorithmie-python/03-exercises/solutions/07-sprint-exercices.py

Each solution includes:
- Pattern used (from J1-J6)
- Time/space complexity
- Key insight explained in comments

Use this file ONLY after attempting each problem on your own for <= 15 min.
"""

from collections import Counter, defaultdict, deque
from typing import List, Optional


# =============================================================================
# P1 — TWO SUM
# Pattern: J3 hash map complement lookup
# =============================================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Classic complement lookup.

    INSIGHT:
    - For each num, we want target - num in the array.
    - A dict mapping value -> index gives O(1) lookup.
    - Store AFTER checking to avoid self-matching (e.g. [3,3], target=6).

    Time: O(n) single pass
    Space: O(n) worst case for the dict
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i                # Only store after checking
    return []


# =============================================================================
# P2 — VALID ANAGRAM
# Pattern: J3 frequency counting
# =============================================================================

def is_anagram(s: str, t: str) -> bool:
    """
    Two strings are anagrams iff they have identical character frequencies.

    Time: O(n)
    Space: O(1) — bounded by alphabet size
    """
    if len(s) != len(t):
        return False                 # Can't be anagrams if lengths differ
    return Counter(s) == Counter(t)  # Counter equality checks all frequencies


# =============================================================================
# P3 — VALID PARENTHESES
# Pattern: J4 stack matching
# =============================================================================

def is_valid_parens(s: str) -> bool:
    """
    LIFO matching: each closer must pop the most recent opener.

    INVARIANT:
    - At any point, the stack holds the openers that haven't been matched yet.
    - An empty stack at the end means every opener was matched.

    Time: O(n)
    Space: O(n) for the stack
    """
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            # Two failure modes: empty stack, or mismatched pair
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


# =============================================================================
# P4 — BEST TIME TO BUY AND SELL STOCK
# Pattern: J1 single pass with running minimum
# =============================================================================

def max_profit(prices: List[int]) -> int:
    """
    Track the minimum price seen so far and the best profit we could make
    by selling at each subsequent day.

    INSIGHT:
    - For each day i, the best profit if we sell today is price[i] - min(prices[0..i-1]).
    - We maintain min_so_far in one pass, updating it AFTER computing profit.

    Time: O(n)
    Space: O(1)
    """
    if not prices:
        return 0
    min_price = prices[0]
    best = 0
    for price in prices[1:]:
        best = max(best, price - min_price)   # Sell today hypothesis
        min_price = min(min_price, price)     # Update for future days
    return best


# =============================================================================
# P5 — REVERSE LINKED LIST
# Pattern: J5 three-pointer reversal
# =============================================================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Canonical iterative reversal.

    STEPS (in this exact order each iteration):
    1. Save curr.next into nxt (before we clobber it)
    2. Rewire curr.next to prev (flip the arrow backwards)
    3. Advance prev to curr (prev is now the new head of the reversed prefix)
    4. Advance curr to nxt

    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head
    while curr:
        nxt = curr.next              # Save next before clobbering
        curr.next = prev             # Flip the pointer
        prev = curr                  # Advance prev into reversed prefix
        curr = nxt                   # Advance curr into remaining tail
    return prev                      # New head = old tail


# =============================================================================
# P6 — LONGEST SUBSTRING WITHOUT REPEATING CHARS
# Pattern: J2 sliding window + hash set
# =============================================================================

def length_of_longest_substring(s: str) -> int:
    """
    Sliding window [left, right] with a set of chars in the current window.

    INVARIANT:
    - The window s[left..right] always contains distinct characters.
    - When we see a duplicate, we shrink from the left until the duplicate is gone.

    AMORTIZED O(n):
    - Each character is added to the set once and removed at most once.
    - Total work across the run is bounded by 2n.

    Time: O(n)
    Space: O(min(n, alphabet size))
    """
    seen = set()
    left = 0
    best = 0
    for right, c in enumerate(s):
        while c in seen:             # Shrink until duplicate is gone
            seen.remove(s[left])
            left += 1
        seen.add(c)
        best = max(best, right - left + 1)
    return best


# =============================================================================
# P7 — GROUP ANAGRAMS
# Pattern: J3 grouping with defaultdict
# =============================================================================

def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    Use the SORTED characters as a grouping key.

    INSIGHT:
    - All anagrams produce the same sorted string: "eat" -> "aet", "tea" -> "aet".
    - A defaultdict(list) collects strings that share the same key.

    Time: O(n * k log k) where k = max string length
    Space: O(n * k) for the grouped output
    """
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))        # Tuple for hashability
        groups[key].append(s)
    return list(groups.values())


# =============================================================================
# P8 — TOP K FREQUENT ELEMENTS
# Pattern: J3 + J6, Counter + bucket sort (O(n) optimal)
# =============================================================================

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Bucket sort by frequency.

    INSIGHT:
    - Max frequency is bounded by len(nums). We create n+1 buckets indexed by
      frequency. Each bucket holds the numbers that appear exactly that many times.
    - Walking buckets from highest to lowest gives top-k in O(n).

    WHY BETTER THAN Counter.most_common(k):
    - most_common uses a heap: O(n + k log n). Bucket sort is O(n).
    - For k ~ n, the difference is negligible. For k small, most_common is fine too.

    Time: O(n)
    Space: O(n)
    """
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    result = []
    for i in range(len(buckets) - 1, 0, -1):   # Walk from highest freq down
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result


# =============================================================================
# P9 — SEARCH IN ROTATED SORTED ARRAY
# Pattern: J6 modified binary search
# =============================================================================

def search_rotated(nums: List[int], target: int) -> int:
    """
    At each step, at least one half of [lo..mid] or [mid..hi] is sorted.
    Identify which, then check if target is in that sorted half.

    INSIGHT:
    - A rotation breaks sorted order at exactly ONE point.
    - If nums[lo] <= nums[mid], the left half is sorted.
    - Otherwise the right half is sorted.

    Time: O(log n)
    Space: O(1)
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid

        if nums[lo] <= nums[mid]:
            # Left half [lo..mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1         # target lies in the sorted left half
            else:
                lo = mid + 1         # target is somewhere in the right half
        else:
            # Right half [mid..hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1         # target lies in the sorted right half
            else:
                hi = mid - 1         # target is in the left half
    return -1


# =============================================================================
# P10 — NUMBER OF ISLANDS
# Pattern: J4 BFS flood fill
# =============================================================================

def num_islands(zone: List[List[str]]) -> int:
    """
    Scan every cell. When we find an unvisited '1', launch a BFS flood-fill
    that marks every connected '1' and increment the counter.

    KEY POINTS:
    - Use deque (NEVER list.pop(0)) for O(1) dequeue.
    - Mark cells as visited at ENQUEUE time (mutate '1' -> '0').
    - 4-connected only: up, down, left, right.

    Time: O(rows * cols) — every cell is touched at most twice
    Space: O(rows * cols) for the queue worst case
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
            # Flood-fill with BFS
            queue = deque([(r, c)])
            zone[r][c] = "0"          # Mark before processing (enqueue-time)
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
# EXERCISE 4 (Medium sprint) — PRODUCT OF ARRAY EXCEPT SELF
# Pattern: J1/J2 prefix-suffix accumulation, no division
# =============================================================================

def product_except_self(nums: List[int]) -> List[int]:
    """
    Two sweeps: prefix products then suffix products.

    WHY NO DIVISION:
    - total_product / nums[i] crashes on zeros (and the problem forbids it).

    KEY IDEA:
    - answer[i] = (product of everything LEFT of i) * (product RIGHT of i).
    - Sweep 1 fills answer with left products.
    - Sweep 2 walks right-to-left, multiplying by a running suffix product
      held in a single variable → O(1) extra space.

    Time: O(n), Space: O(1) auxiliary (output array excluded)
    """
    n = len(nums)
    answer = [1] * n

    prefix = 1
    for i in range(n):
        answer[i] = prefix          # Product of nums[0..i-1]
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix         # Multiply by product of nums[i+1..]
        suffix *= nums[i]

    return answer


# =============================================================================
# EXERCISE 5 (Medium sprint) — LONGEST CONSECUTIVE SEQUENCE
# Pattern: J3 set membership + smart start detection
# =============================================================================

def longest_consecutive(nums: List[int]) -> int:
    """
    Set + only start counting from sequence STARTS.

    WHY NOT SORT:
    - Sorting gives O(n log n); the interviewer asks for O(n).

    THE O(n) ARGUMENT:
    - The inner while looks like a nested loop, but it only runs from
      numbers whose predecessor is absent (sequence starts). Each number
      is therefore visited at most twice (once by the outer loop, once
      inside the walk of its own sequence) → O(n) total.

    Time: O(n), Space: O(n) for the set
    """
    seen = set(nums)                # Also deduplicates
    best = 0

    for num in seen:
        if num - 1 in seen:
            continue                # Not a sequence start — skip (the key trick)
        length = 1
        while num + length in seen:
            length += 1
        best = max(best, length)

    return best


# =============================================================================
# EXERCISE 6 (Medium sprint) — MERGE INTERVALS
# Pattern: J6 sort by start + linear sweep
# =============================================================================

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Sort by start, then sweep and extend.

    WHY SORT FIRST:
    - After sorting, an interval can only overlap with the LAST merged one;
      without sorting, overlaps can appear anywhere and the sweep is wrong.

    THE CONTAINMENT TRAP:
    - [1, 4] then [2, 3]: extend with max(end, current_end), never just
      assign — otherwise the merged interval would SHRINK to [1, 3].

    Time: O(n log n) for the sort, Space: O(n) for the result
    """
    if not intervals:
        return []

    intervals = sorted(intervals)   # Sorts by start (then end) lexicographically
    merged = [intervals[0][:]]      # Copy: don't mutate the caller's data

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # <= : touching bounds DO merge
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


# =============================================================================
# EXERCISE 7 (Hard sprint) — FIRST MISSING POSITIVE
# Pattern: J1 index-as-hash (cyclic sort), O(n) time + O(1) space
# =============================================================================

def first_missing_positive(nums: List[int]) -> int:
    """
    Cyclic sort: the array is its own hash map.

    SUBOPTIMAL ANSWERS TO ANNOUNCE FIRST:
    - set of values: O(n) time but O(n) space — violates the constraint.
    - sort then scan: O(n log n) — violates the time constraint.

    KEY OBSERVATION:
    - With n elements, the answer is in [1, n+1]. Values <= 0 or > n are
      irrelevant noise. So value v in [1, n] belongs at index v - 1.

    THE DUPLICATE TRAP:
    - Swap only while nums[i] != nums[nums[i] - 1]. Checking
      nums[i] != i + 1 alone loops forever on [1, 1] (swapping equal
      values back and forth).

    Time: O(n) — each swap places one value at its final index forever,
          so there are at most n swaps in total.
    Space: O(1) — in-place mutation only.
    """
    n = len(nums)

    for i in range(n):
        # Keep swapping until the value here is out of range or already home
        while 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
            target = nums[i] - 1
            nums[i], nums[target] = nums[target], nums[i]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1            # First slot whose owner is missing

    return n + 1                    # All of [1, n] present


# =============================================================================
# EXERCISE 8 (Hard sprint) — REVERSE NODES IN K-GROUP
# Pattern: J5 reversal by blocks + dummy head
# =============================================================================

def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Reverse the list k nodes at a time; leave a short final group as is.

    STRUCTURE:
    - group_prev points to the node BEFORE the current group (starts at
      the dummy, so the first group needs no special case).
    - Step 1: probe k nodes ahead. Fewer than k left → done.
    - Step 2: standard reversal of exactly k nodes, with prev initialized
      to group_next so the reversed block is ALREADY connected to the rest.
    - Step 3: reconnect group_prev to the new block head; the old block
      head (now the tail) becomes the next group_prev.

    Time: O(n) — each node is probed once and reversed once
    Space: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy

    while True:
        # Step 1: find the k-th node of the current group
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next   # Fewer than k nodes left: keep as is

        group_next = kth.next       # First node AFTER the group

        # Step 2: reverse the k nodes; prev starts at group_next so the
        # block's future tail already points to the rest of the list
        prev, curr = group_next, group_prev.next
        while curr is not group_next:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        # Step 3: reconnect. The old group head is now the group tail.
        old_head = group_prev.next
        group_prev.next = kth       # kth is the new head of the block
        group_prev = old_head


# =============================================================================
# TESTS
# =============================================================================

def test_all():
    print("\nRunning all 10 sprint solutions...")

    # P1
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]

    # P2
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False
    assert is_anagram("", "") is True

    # P3
    assert is_valid_parens("()[]{}") is True
    assert is_valid_parens("([)]") is False
    assert is_valid_parens("") is True
    assert is_valid_parens("((") is False

    # P4
    assert max_profit([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit([7, 6, 4, 3, 1]) == 0
    assert max_profit([]) == 0
    assert max_profit([1]) == 0

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

    assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
    assert to_list(reverse_list(build([1]))) == [1]
    assert reverse_list(None) is None

    # P6
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3
    assert length_of_longest_substring("") == 0

    # P7
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    normalized = sorted([sorted(g) for g in result])
    expected = sorted([sorted(g) for g in [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]])
    assert normalized == expected

    # P8
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent([1], 1) == [1]

    # P9
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated([1], 0) == -1
    assert search_rotated([1], 1) == 0

    # P10
    grid1 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert num_islands([row[:] for row in grid1]) == 3
    assert num_islands([["0"]]) == 0
    assert num_islands([["1"]]) == 1

    print("All 10 sprint solutions PASS")


def test_medium_hard():
    print("\nRunning medium/hard sprint solutions (exercises 4-8)...")

    # Exercise 4 — Product of Array Except Self
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    assert product_except_self([2, 3]) == [3, 2]
    assert product_except_self([0, 0]) == [0, 0]
    assert product_except_self([5, 0]) == [0, 5]
    assert product_except_self([1, 1, 1, 1]) == [1, 1, 1, 1]
    print("Exercise 4 (product_except_self): OK")

    # Exercise 5 — Longest Consecutive Sequence
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longest_consecutive([]) == 0
    assert longest_consecutive([1]) == 1
    assert longest_consecutive([1, 1, 1]) == 1
    assert longest_consecutive([5, 3, 1]) == 1
    assert longest_consecutive([-2, -1, 0, 1]) == 4
    print("Exercise 5 (longest_consecutive): OK")

    # Exercise 6 — Merge Intervals
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 4]]) == [[1, 4]]
    assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
    assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
    assert merge_intervals([[2, 2], [2, 2], [2, 2]]) == [[2, 2]]
    print("Exercise 6 (merge_intervals): OK")

    # Exercise 7 — First Missing Positive
    assert first_missing_positive([1, 2, 0]) == 3
    assert first_missing_positive([3, 4, -1, 1]) == 2
    assert first_missing_positive([7, 8, 9, 11, 12]) == 1
    assert first_missing_positive([]) == 1
    assert first_missing_positive([1]) == 2
    assert first_missing_positive([2]) == 1
    assert first_missing_positive([1, 1]) == 2
    assert first_missing_positive([2, 2, 2, 2]) == 1
    assert first_missing_positive([1, 2, 3, 4, 5]) == 6
    assert first_missing_positive([-1, -2, -3]) == 1
    print("Exercise 7 (first_missing_positive): OK")

    # Exercise 8 — Reverse Nodes in k-Group
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

    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 1)) == [1, 2, 3, 4, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 5)) == [5, 4, 3, 2, 1]
    assert to_list(reverse_k_group(build([1, 2]), 3)) == [1, 2]
    assert to_list(reverse_k_group(build([]), 2)) == []
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6]), 2)) == [2, 1, 4, 3, 6, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6, 7]), 3)) == [3, 2, 1, 6, 5, 4, 7]
    print("Exercise 8 (reverse_k_group): OK")

    print("All medium/hard sprint solutions PASS")


if __name__ == "__main__":
    test_all()
    test_medium_hard()

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

def num_islands(grid: List[List[str]]) -> int:
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
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    islands = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "1":
                continue
            islands += 1
            # Flood-fill with BFS
            queue = deque([(r, c)])
            grid[r][c] = "0"          # Mark before processing (enqueue-time)
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and grid[nr][nc] == "1"):
                        grid[nr][nc] = "0"
                        queue.append((nr, nc))
    return islands


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


if __name__ == "__main__":
    test_all()

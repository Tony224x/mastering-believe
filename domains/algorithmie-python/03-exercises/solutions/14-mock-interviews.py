"""
Solutions — Day 14 Mock Interviews (fresh problems, easy to hard).
Run: python domains/algorithmie-python/03-exercises/solutions/14-mock-interviews.py

Each solution is narrated like a presentation: the "why" matters more than
the "what". If you're reading these, you've already tried the problems —
compare your reasoning to the one below.
"""

from collections import defaultdict


# =============================================================================
# Mock 1: Move Zeroes
# =============================================================================

def move_zeroes(nums):
    """
    Two-pointer in-place solution.

    Idea: maintain a "write" pointer that tracks where the next non-zero
    should go. Walk through the array; whenever we see a non-zero, write
    it at `write` and advance `write`. After the first pass, all non-zeros
    are at positions 0..write-1 in their original order. Fill the rest
    with zeros.

    Why not swap? Swap also works, but write-then-fill is clearer and has
    the same complexity.

    Time : O(n)
    Space: O(1)
    """
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write] = nums[read]
            write += 1
    # Fill the remainder with zeros
    for i in range(write, len(nums)):
        nums[i] = 0


# Alternative: single-pass swap version
def move_zeroes_swap(nums):
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1


# =============================================================================
# Mock 2: Valid Palindrome
# =============================================================================

def is_palindrome(s):
    """
    Two-pointer approach.
    Move left and right toward the center, skipping non-alphanumeric
    characters on both sides. Compare lowercased characters at each step.

    WHY not clean-then-reverse? Because cleaning and reversing allocates
    O(n) extra memory. Two pointers work in O(1) extra space.

    Time : O(n)
    Space: O(1)
    """
    left, right = 0, len(s) - 1
    while left < right:
        # Advance left past any non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        # Compare case-insensitively
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


# =============================================================================
# Mock 3: Group Anagrams
# =============================================================================

def group_anagrams(strs):
    """
    Grouping with a hash map keyed by a canonical anagram signature.

    Two key choices:
      - tuple(sorted(s))  -> O(k log k) per string
      - tuple of 26 counts -> O(k) per string, better for long strings

    We pick the count-based key for optimal asymptotic complexity.

    Time : O(n * k), n = number of strings, k = max length
    Space: O(n * k) for the grouping map
    """
    groups = defaultdict(list)
    for s in strs:
        # Build a 26-length frequency tuple as the signature
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord("a")] += 1
        groups[tuple(counts)].append(s)
    return list(groups.values())


# Alternative keyed by sorted tuple (shorter but O(k log k))
def group_anagrams_sorted(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())


# =============================================================================
# TESTS
# =============================================================================


# =============================================================================
# Mock 4: Spiral Matrix
# =============================================================================

def spiral_order(matrix):
    """
    Four shrinking boundaries — pure simulation, zero clever algorithm.

    THE NARRATION THAT MATTERS IN A MOCK:
    - State the invariant up front: top/bottom/left/right delimit the
      not-yet-visited ring; each side pass consumes one boundary line.

    THE TWO GUARDS:
    - After the right and down passes, the matrix may be exhausted in one
      dimension. Without `if top <= bottom` before the leftward pass, a
      single-row matrix is read twice (right then left). Same for the
      upward pass on single-column matrices.

    Time: O(m * n) — each cell visited once. Space: O(1) auxiliary.
    """
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):            # → along the top row
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):            # ↓ along the right column
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:                           # Guard: row still exists
            for c in range(right, left - 1, -1):    # ← along the bottom row
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:                           # Guard: column still exists
            for r in range(bottom, top - 1, -1):    # ↑ along the left column
                result.append(matrix[r][left])
            left += 1

    return result


# =============================================================================
# Mock 5: Insert Interval
# =============================================================================

def insert_interval(intervals, new_interval):
    """
    Three-phase single pass — exploits the sorted/disjoint precondition.

    WHY NOT append + sort + merge:
    - O(n log n) and throws away the precondition. The interviewer gave
      you "sorted and disjoint" for a reason: use it for O(n).

    PHASES:
    1. Intervals ending strictly before new_interval starts: copy as is.
    2. Overlapping intervals (start <= new_end AND end >= new_start):
       absorb them into new_interval by widening its bounds.
    3. Everything else starts after new_interval ends: copy as is.

    Touching bounds ([1,5] + [5,7]) count as overlapping → merge.

    Time: O(n), Space: O(n) for the result.
    """
    result = []
    new_start, new_end = new_interval
    i, n = 0, len(intervals)

    # Phase 1: strictly before (end < new_start)
    while i < n and intervals[i][1] < new_start:
        result.append(intervals[i])
        i += 1

    # Phase 2: overlap — widen new_interval instead of appending
    while i < n and intervals[i][0] <= new_end:     # <= : touching merges
        new_start = min(new_start, intervals[i][0])
        new_end = max(new_end, intervals[i][1])
        i += 1
    result.append([new_start, new_end])

    # Phase 3: strictly after
    result.extend(intervals[i:])
    return result


# =============================================================================
# Mock 6: Insert Delete GetRandom O(1)
# =============================================================================

class RandomizedSet:
    """
    list + dict combo — each structure covers the other's weakness.

    WHY TWO STRUCTURES:
    - A set alone: random.choice needs indexing, sets aren't indexable
      (list(set) is O(n) per call).
    - A list alone: remove-by-value is O(n) (search + shift).
    - dict gives O(1) lookup of an element's position; the list gives
      O(1) uniform random indexing.

    THE SWAP-WITH-LAST TRICK (remove):
    - Removing from the middle of a list is O(n) because of shifting.
      Instead, overwrite the removed slot with the LAST element, update
      that element's index in the dict, then pop the tail — O(1).
    - Edge case: removing the last element swaps it with itself, which
      is harmless (the dict update writes then deletes the same key —
      hence delete AFTER the swap bookkeeping).
    """

    def __init__(self):
        self.values = []                # Indexable storage for get_random
        self.index_of = {}              # value -> its index in self.values

    def insert(self, val):
        """O(1): append + record index."""
        if val in self.index_of:
            return False
        self.index_of[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val):
        """O(1): swap with last, then pop the tail."""
        if val not in self.index_of:
            return False
        idx = self.index_of[val]
        last = self.values[-1]
        # Move the last element into the vacated slot (THE classic bug:
        # forgetting to update the dict for the moved element)
        self.values[idx] = last
        self.index_of[last] = idx
        self.values.pop()
        del self.index_of[val]          # AFTER bookkeeping: handles val == last
        return True

    def get_random(self):
        """O(1): uniform thanks to the dense list."""
        import random
        return random.choice(self.values)


# =============================================================================
# Mock 7: Longest Valid Parentheses
# =============================================================================

def longest_valid_parentheses(s):
    """
    Stack of INDICES with a sentinel base — O(n) single pass.

    CHANGE OF REPRESENTATION (vs day 4's Valid Parentheses):
    - We don't need validity (a boolean); we need LENGTHS. Lengths are
      distances between indices, so the stack must hold indices.

    THE BASE INVARIANT:
    - stack[0] is always the index of the last character that BREAKS
      validity (initially the virtual index -1). Above it: indices of
      unmatched '('.
    - On ')': pop. If the stack empties, this ')' is unmatched → it
      becomes the new base. Otherwise the valid window ends at i and
      starts right after stack[-1]: length = i - stack[-1]. This
      naturally CHAINS adjacent valid blocks ("()()" → 4), which naive
      pair-counting approaches get wrong.

    DP ALTERNATIVE (sketch): dp[i] = longest valid substring ENDING at i;
    when s[i] == ')', look at the char before the dp[i-1] block — if it
    is '(', dp[i] = dp[i-1] + 2 + dp[i - dp[i-1] - 2].

    Time: O(n), Space: O(n)
    """
    best = 0
    stack = [-1]                        # Sentinel base: index before the string

    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)         # Unmatched ')': new base
            else:
                best = max(best, i - stack[-1])

    return best


# =============================================================================
# Mock 8: Basic Calculator
# =============================================================================

def calculate(s):
    """
    Single pass with (result, sign) saved on a stack at each '('.

    STATE:
    - num: multi-digit number being read (num = num * 10 + digit).
    - sign: +1/-1 applied to the NEXT operand.
    - result: accumulated sum of the CURRENT parenthesis level.

    PARENTHESES = CALL STACK:
    - '(' suspends the current computation exactly like a function call:
      push (result, sign), reset both, start the inner level from scratch.
    - ')' returns: the inner result is treated as a single operand of the
      outer level → result = saved_result + saved_sign * inner_result.

    UNARY MINUS FOR FREE:
    - "-(3+4)": when '-' is read, num is 0 (nothing flushed), result
      stays 0 and sign becomes -1 — no special case needed.

    FLUSH POINTS (the classic bugs):
    - A number is folded into result when hitting an operator, a ')',
      or the END of the string (the final flush is easy to forget).

    Time: O(n), Space: O(depth of parentheses)
    """
    result = 0
    sign = 1
    num = 0
    stack = []                          # Saved (result, sign) per open paren

    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)    # Multi-digit accumulation
        elif ch in "+-":
            result += sign * num        # Flush the pending number
            num = 0
            sign = 1 if ch == "+" else -1
        elif ch == "(":
            stack.append((result, sign))    # Suspend the outer level
            result, sign = 0, 1
        elif ch == ")":
            result += sign * num        # Flush inside the parens
            num = 0
            saved_result, saved_sign = stack.pop()
            result = saved_result + saved_sign * result
        # Spaces: ignored

    return result + sign * num          # Final flush — easy to forget


if __name__ == "__main__":
    # -- Mock 1 --
    def check_move(nums, expected):
        move_zeroes(nums)
        assert nums == expected, f"Got {nums}, expected {expected}"

    check_move([0, 1, 0, 3, 12], [1, 3, 12, 0, 0])
    check_move([0], [0])
    check_move([1, 2, 3], [1, 2, 3])
    check_move([0, 0, 0], [0, 0, 0])
    check_move([4, 2, 4, 0, 0, 3, 0, 5, 1, 0], [4, 2, 4, 3, 5, 1, 0, 0, 0, 0])
    check_move([], [])
    # Verify swap variant matches
    nums2 = [0, 1, 0, 3, 12]
    move_zeroes_swap(nums2)
    assert nums2 == [1, 3, 12, 0, 0]
    print("Mock 1 (move_zeroes): OK")

    # -- Mock 2 --
    assert is_palindrome("A man, a plan, a canal: Panama") == True
    assert is_palindrome("race a car") == False
    assert is_palindrome(" ") == True
    assert is_palindrome("") == True
    assert is_palindrome("a.") == True
    assert is_palindrome("0P") == False
    assert is_palindrome("ab_a") == True
    print("Mock 2 (is_palindrome): OK")

    # -- Mock 3 --
    def sort_groups(result):
        return sorted([sorted(group) for group in result])

    assert sort_groups(group_anagrams(
        ["eat", "tea", "tan", "ate", "nat", "bat"])) == [
        ["ate", "eat", "tea"], ["bat"], ["nat", "tan"]
    ]
    assert sort_groups(group_anagrams([""])) == [[""]]
    assert sort_groups(group_anagrams(["a"])) == [["a"]]
    assert sort_groups(group_anagrams([])) == []
    assert sort_groups(group_anagrams(["abc", "bca", "cab", "xyz"])) == [
        ["abc", "bca", "cab"], ["xyz"]
    ]
    # Verify sorted-key variant matches
    assert sort_groups(group_anagrams_sorted(
        ["eat", "tea", "tan", "ate", "nat", "bat"])) == [
        ["ate", "eat", "tea"], ["bat"], ["nat", "tan"]
    ]
    print("Mock 3 (group_anagrams): OK")

    # -- Mock 4 --
    assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [
        1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7
    ]
    assert spiral_order([[1]]) == [1]
    assert spiral_order([[1, 2, 3]]) == [1, 2, 3]
    assert spiral_order([[1], [2], [3]]) == [1, 2, 3]
    assert spiral_order([]) == []
    assert spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]
    assert spiral_order([[1, 2], [4, 5], [7, 8]]) == [1, 2, 5, 8, 7, 4]
    print("Mock 4 (spiral_order): OK")

    # -- Mock 5 --
    assert insert_interval([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [
        [1, 2], [3, 10], [12, 16]
    ]
    assert insert_interval([], [5, 7]) == [[5, 7]]
    assert insert_interval([[1, 5]], [2, 3]) == [[1, 5]]
    assert insert_interval([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]
    assert insert_interval([[6, 8]], [1, 5]) == [[1, 5], [6, 8]]
    assert insert_interval([[1, 5]], [5, 7]) == [[1, 7]]
    assert insert_interval([[3, 4]], [1, 2]) == [[1, 2], [3, 4]]
    print("Mock 5 (insert_interval): OK")

    # -- Mock 6 --
    rs = RandomizedSet()
    assert rs.insert(1) == True
    assert rs.insert(1) == False
    assert rs.remove(2) == False
    assert rs.insert(2) == True
    assert rs.get_random() in (1, 2)
    assert rs.remove(1) == True
    assert rs.get_random() == 2
    assert rs.remove(2) == True
    assert rs.insert(2) == True

    rs2 = RandomizedSet()
    rs2.insert(10); rs2.insert(20); rs2.insert(30)
    assert rs2.remove(30) == True       # Remove the LAST element (self-swap)
    assert rs2.remove(10) == True
    assert rs2.get_random() == 20

    rs3 = RandomizedSet()
    for v in range(3):
        rs3.insert(v)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(3000):
        counts[rs3.get_random()] += 1
    assert all(c > 700 for c in counts.values()), counts
    print("Mock 6 (RandomizedSet): OK")

    # -- Mock 7 --
    assert longest_valid_parentheses("(()") == 2
    assert longest_valid_parentheses(")()())") == 4
    assert longest_valid_parentheses("") == 0
    assert longest_valid_parentheses("(") == 0
    assert longest_valid_parentheses(")") == 0
    assert longest_valid_parentheses("()(()") == 2
    assert longest_valid_parentheses("()(())") == 6
    assert longest_valid_parentheses("(()())") == 6
    assert longest_valid_parentheses("())((())") == 4
    assert longest_valid_parentheses("()()") == 4
    print("Mock 7 (longest_valid_parentheses): OK")

    # -- Mock 8 --
    assert calculate("1 + 1") == 2
    assert calculate(" 2-1 + 2 ") == 3
    assert calculate("(1+(4+5+2)-3)+(6+8)") == 23
    assert calculate("123") == 123
    assert calculate("0") == 0
    assert calculate("-(3 + 4)") == -7
    assert calculate("(-2)") == -2
    assert calculate("2-(5-6)") == 3
    assert calculate("- (3 + (4 + 5))") == -12
    assert calculate("1-(-7)") == 8
    assert calculate("10 - (2 + 3) + (4 - 1)") == 8
    print("Mock 8 (calculate): OK")

    print("\nAll Day 14 mock solutions pass!")

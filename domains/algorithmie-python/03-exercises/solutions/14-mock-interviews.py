"""
Solutions — Day 14 Mock Interviews (fresh problems).
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

    print("\nAll Day 14 mock solutions pass!")

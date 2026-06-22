"""
Solutions — Day 13: Bit Manipulation, Heaps & Tries (MEDIUM)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/13-bit-heap-trie-medium.py

Each solution is numbered to match the exercise file (02-medium/13-bit-heap-trie.md).
All solutions are verified with assertions at the end.
"""

import heapq
from collections import Counter


# =============================================================================
# EXERCISE 4 (Medium): Top K Frequent Elements — Counter + size-k heap
# =============================================================================

def top_k_frequent(nums, k):
    """
    Count frequencies, then keep a size-k min-heap of (freq, value).
    The heap discards the least frequent whenever it overflows.

    Time: O(n log k), Space: O(n)
    """
    freq = Counter(nums)
    heap = []                          # Min-heap of (frequency, value)
    for value, count in freq.items():
        heapq.heappush(heap, (count, value))
        if len(heap) > k:
            heapq.heappop(heap)        # Drop the least frequent so far
    return [value for _, value in heap]


def test_exercise_4():
    print("\nExercise 4: Top K Frequent Elements")

    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent([1], 1) == [1]
    assert sorted(top_k_frequent([4, 4, 4, 5, 5, 6], 2)) == [4, 5]
    assert sorted(top_k_frequent([1, 2, 3, 4], 4)) == [1, 2, 3, 4]
    assert top_k_frequent([7, 7, 7], 1) == [7]
    assert sorted(top_k_frequent([-1, -1, -2, -2, -2, 3], 2)) == [-2, -1]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Implement Trie (Prefix Tree)
# =============================================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    """
    Standard prefix tree. insert/search/starts_with all run in O(L).
    is_end distinguishes a complete word from a mere prefix.
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node


def test_exercise_5():
    print("\nExercise 5: Implement Trie")

    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.starts_with("app") is True
    trie.insert("app")
    assert trie.search("app") is True

    t2 = Trie()
    assert t2.search("anything") is False
    assert t2.starts_with("") is True
    t2.insert("")
    assert t2.search("") is True

    t3 = Trie()
    for w in ["a", "ab", "abc"]:
        t3.insert(w)
    assert all(t3.search(w) for w in ["a", "ab", "abc"])
    assert t3.starts_with("ab") is True
    assert t3.search("abcd") is False

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Counting Bits — DP over bits
# =============================================================================

def count_bits(n):
    """
    dp[i] = dp[i >> 1] + (i & 1): i>>1 drops the last bit, (i&1) restores it.

    Time: O(n), Space: O(n)
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


def test_exercise_6():
    print("\nExercise 6: Counting Bits")

    assert count_bits(0) == [0]
    assert count_bits(2) == [0, 1, 1]
    assert count_bits(5) == [0, 1, 1, 2, 1, 2]
    for n in range(0, 50):
        assert count_bits(n) == [bin(i).count("1") for i in range(n + 1)]

    print("  PASS — all test cases (incl. naive cross-check)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 13: Bit Manipulation, Heaps & Tries (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

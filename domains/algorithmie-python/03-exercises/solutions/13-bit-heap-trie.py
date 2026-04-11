"""
Solutions — Day 13 Bit, Heap, Trie (easy exercises).
Run: python domains/algorithmie-python/03-exercises/solutions/13-bit-heap-trie.py
"""

import heapq


# =============================================================================
# Exercise 1: Single Number (XOR)
# =============================================================================

def single_number(nums):
    """
    XOR every element together.
    Because XOR is commutative, associative, and x ^ x = 0, every pair
    cancels out and the lone element is the final result.
    0 is the identity for XOR, so starting from 0 is safe.

    Time : O(n)
    Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result


# =============================================================================
# Exercise 2: Kth Largest Element (min-heap of size k)
# =============================================================================

def find_kth_largest(nums, k):
    """
    Min-heap of size k.
    After processing all numbers, the heap holds exactly the k largest.
    The smallest of those k (heap[0]) is the kth largest overall.

    Time : O(n log k)
    Space: O(k)
    """
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)  # pop + push in one op
    return heap[0]


# Alternative using heapq.nlargest (same Big-O)
def find_kth_largest_nlargest(nums, k):
    return heapq.nlargest(k, nums)[-1]


# =============================================================================
# Exercise 3: Trie
# =============================================================================

class TrieNode:
    """
    Node in the prefix tree.
    - children: dict mapping char -> TrieNode
    - is_end: True iff a complete word ends at this node
    """
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    """
    Prefix tree. All operations run in O(L) where L = len(query).
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Walk/create the path for each character, mark the last as end."""
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        """A word is present iff we can walk its path AND the last node is end."""
        if word == "":
            return self.root.is_end
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        """Any prefix walkable through the trie is a valid starts_with."""
        if prefix == "":
            return True                # The empty prefix matches anything
        return self._walk(prefix) is not None

    def _walk(self, s):
        """Descend the trie following `s`; return the last node or None."""
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    # -- Exercise 1 --
    assert single_number([2, 2, 1]) == 1
    assert single_number([4, 1, 2, 1, 2]) == 4
    assert single_number([1]) == 1
    assert single_number([-1, -1, -2]) == -2
    assert single_number([0, 1, 0]) == 1
    assert single_number([7, 7, 3, 5, 3]) == 5
    print("Exercise 1 (single_number): OK")

    # -- Exercise 2 --
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert find_kth_largest([1], 1) == 1
    assert find_kth_largest([1, 2], 1) == 2
    assert find_kth_largest([1, 2], 2) == 1
    assert find_kth_largest([7, 6, 5, 4, 3, 2, 1], 3) == 5
    # Double check with nlargest variant
    assert find_kth_largest_nlargest([3, 2, 1, 5, 6, 4], 2) == 5
    print("Exercise 2 (find_kth_largest): OK")

    # -- Exercise 3 --
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") == True
    assert trie.search("app") == False
    assert trie.starts_with("app") == True
    trie.insert("app")
    assert trie.search("app") == True

    trie2 = Trie()
    trie2.insert("hello")
    trie2.insert("help")
    trie2.insert("helicopter")
    assert trie2.starts_with("hel") == True
    assert trie2.starts_with("helix") == False
    assert trie2.search("hell") == False
    assert trie2.search("help") == True

    trie3 = Trie()
    assert trie3.search("") == False
    assert trie3.starts_with("") == True
    print("Exercise 3 (Trie): OK")

    print("\nAll Day 13 solutions pass!")

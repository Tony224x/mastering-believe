"""
Day 13 — Bit Manipulation, Heaps & Tries
Run: python domains/algorithmie-python/02-code/13-bit-heap-trie.py
"""

import heapq


# =============================================================================
# SECTION 1: BIT MANIPULATION
# =============================================================================

def single_number(nums):
    """
    Find the one element that appears exactly once when all others
    appear exactly twice.

    KEY: XOR is commutative, associative, and x ^ x = 0.
    So xoring everything cancels all pairs and leaves the unique value.

    Time : O(n)
    Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result


def count_bits(n):
    """
    Return a list where result[i] = number of 1-bits in i, for i in [0, n].

    DP trick: dp[i] = dp[i >> 1] + (i & 1)
    Explanation: shifting i right by 1 removes the last bit, so dp[i >> 1]
    counts all bits except the last one; (i & 1) adds back the last bit.

    Time : O(n)
    Space: O(n)
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


def count_bits_kernighan(n):
    """
    Alternative with Brian Kernighan's trick: dp[i] = dp[i & (i-1)] + 1.
    i & (i-1) clears the LOWEST set bit of i, so dp[i & (i-1)] counts one
    fewer bit. Adding 1 gives the total.
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i & (i - 1)] + 1
    return dp


def is_power_of_two(n):
    """
    A power of 2 has exactly one bit set. Clearing it with n & (n-1)
    should give 0. Negative numbers and zero are NOT powers of 2.
    """
    return n > 0 and (n & (n - 1)) == 0


def hamming_weight(n):
    """Count set bits of n using Brian Kernighan's trick."""
    count = 0
    while n:
        n &= n - 1                     # Drop the lowest set bit
        count += 1
    return count


# =============================================================================
# SECTION 2: HEAPS — top K, kth largest
# =============================================================================

def k_largest(nums, k):
    """
    Return the k largest elements (in any order).
    Strategy: keep a min-heap of size k; any time we see a number larger
    than the smallest of the heap, replace it. After the scan, the heap
    holds exactly the k largest.

    Time : O(n log k)
    Space: O(k)
    """
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)   # pop + push in one go, more efficient
    return heap


def kth_largest_element(nums, k):
    """
    Return THE kth largest element (a single value).
    """
    return k_largest(nums, k)[0]       # Smallest in the size-k min-heap


def merge_k_sorted(lists):
    """
    Merge k sorted lists into a single sorted list.

    KEY: push tuples (value, list_index, element_index) so ties break by
    index instead of comparing list objects (which would raise TypeError).

    Time : O(N log k), where N = total number of elements
    Space: O(k) heap + O(N) output
    """
    heap = []
    # Seed the heap with the first element of each non-empty list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        next_elem_idx = elem_idx + 1
        if next_elem_idx < len(lists[list_idx]):
            next_val = lists[list_idx][next_elem_idx]
            heapq.heappush(heap, (next_val, list_idx, next_elem_idx))

    return result


def max_heap_demo(nums):
    """
    Python only offers a min-heap; to simulate a max-heap, negate values.
    Remember to negate again on pop to get the real value.
    """
    max_heap = []
    for x in nums:
        heapq.heappush(max_heap, -x)
    # Pop all elements in descending order
    result = []
    while max_heap:
        result.append(-heapq.heappop(max_heap))
    return result


# =============================================================================
# SECTION 3: TRIE
# =============================================================================

class TrieNode:
    """A single node in the trie: map of char -> child + end-of-word flag."""
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    """
    Prefix tree supporting insert, search, and startsWith in O(L)
    where L = length of the query.
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True             # Mark the last char as end-of-word

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        """Descend the trie following s, return the last node or None."""
        node = self.root
        for c in s:
            if c not in node.children:
                return None
            node = node.children[c]
        return node


# =============================================================================
# SECTION 4: WORD SEARCH II (trie + DFS on a zone)
# =============================================================================

def find_words(board, words):
    """
    Find every word from `words` that can be formed by a path in the zone
    (up/down/left/right, no cell reuse within a word).

    KEY: build a trie of all words, then run DFS from each cell, descending
    the trie character by character. If the current prefix isn't in the
    trie, we prune immediately — way faster than searching each word
    independently when the word list is long.
    """
    # Build trie, stash the full word at terminal nodes
    root = TrieNode()
    for word in words:
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True
        node.word = word               # Attribute added dynamically

    if not board or not board[0]:
        return []
    rows, cols = len(board), len(board[0])
    found = []

    def dfs(r, c, node):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return
        char = board[r][c]
        if char == "#" or char not in node.children:
            return
        next_node = node.children[char]
        if next_node.is_end:
            found.append(next_node.word)
            next_node.is_end = False   # Avoid duplicates in the output
        board[r][c] = "#"              # Mark visited
        dfs(r + 1, c, next_node)
        dfs(r - 1, c, next_node)
        dfs(r, c + 1, next_node)
        dfs(r, c - 1, next_node)
        board[r][c] = char             # Restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return found


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    print("Single number [4,1,2,1,2]:", single_number([4, 1, 2, 1, 2]))  # 4
    print("Count bits up to 5:", count_bits(5))                          # [0,1,1,2,1,2]
    print("Count bits Kernighan up to 5:", count_bits_kernighan(5))
    print("Is power of two 16?", is_power_of_two(16))
    print("Is power of two 18?", is_power_of_two(18))
    print("Hamming weight of 11:", hamming_weight(11))                   # 3 (1011)

    print("\nK largest (k=3) of [3,2,1,5,6,4]:",
          sorted(k_largest([3, 2, 1, 5, 6, 4], 3), reverse=True))
    print("Kth (k=2) largest [3,2,1,5,6,4]:",
          kth_largest_element([3, 2, 1, 5, 6, 4], 2))                    # 5

    print("\nMerge k sorted [[1,4,5],[1,3,4],[2,6]]:",
          merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]]))

    print("\nMax-heap demo:", max_heap_demo([3, 1, 4, 1, 5, 9, 2, 6]))

    print("\nTrie demo:")
    trie = Trie()
    for w in ["apple", "app", "apricot", "banana"]:
        trie.insert(w)
    print("  search 'apple' :", trie.search("apple"))    # True
    print("  search 'app'   :", trie.search("app"))      # True
    print("  search 'ap'    :", trie.search("ap"))       # False
    print("  starts 'apri'  :", trie.starts_with("apri"))# True
    print("  starts 'ban'   :", trie.starts_with("ban")) # True
    print("  starts 'cat'   :", trie.starts_with("cat")) # False

    print("\nWord search II:")
    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    words = ["oath", "pea", "eat", "rain"]
    print(" ", sorted(find_words([row[:] for row in board], words)))

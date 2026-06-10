"""
Solutions — Day 13 Bit, Heap, Trie (easy, medium and hard exercises).
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
# Exercise 4 (Medium): Single Number II (bit counting mod 3)
# =============================================================================

def single_number_ii_oracle(nums):
    """O(n) space oracle with a Counter — used only to cross-check."""
    from collections import Counter
    for value, count in Counter(nums).items():
        if count == 1:
            return value


def single_number_ii(nums):
    """
    Count each bit position modulo 3 — O(n) time, O(1) space.

    WHY XOR IS NOT ENOUGH:
    - XOR cancels PAIRS. With triples, each bit of the duplicated numbers
      contributes 3 (= 1 mod 2) — XOR would mix them into the answer.
    - Counting per bit: duplicated numbers contribute 0 mod 3, so
      (bit count) % 3 isolates the single number's bit.

    THE PYTHON NEGATIVE TRAP:
    - Python ints have arbitrary precision: -5 has infinitely many sign
      bits, so "iterate 32 bits" needs an explicit two's-complement view.
      Mask each number to its bit (num >> i) & 1 works for negatives too
      in Python BUT the reconstruction needs the 32-bit convention: if
      bit 31 of the result is set, the value is negative → subtract 2^32.
    """
    result = 0
    for i in range(32):
        # Sum of bit i across all numbers (two's complement view)
        count = sum((num >> i) & 1 for num in nums)
        if count % 3:
            result |= 1 << i

    # Bit 31 set means the 32-bit value is negative
    if result >= 1 << 31:
        result -= 1 << 32
    return result


# =============================================================================
# Exercise 5 (Medium): K Closest Points to Origin (bounded max-heap)
# =============================================================================

def k_closest(points, k):
    """
    Max-heap of size k holding the k closest points seen so far.

    THE THREE APPROACHES (and why this one):
    - sort all: O(n log n) — wasteful, we only need k of them.
    - heapify all + k pops: O(n + k log n) — O(n) memory.
    - BOUNDED heap of size k: O(n log k) time, O(k) memory — best when
      k << n (and what the interviewer wants to hear).

    MAX-HEAP VIA NEGATION:
    - heapq is a min-heap; pushing -dist puts the FARTHEST of the kept
      points on top, ready to be evicted by anything closer.

    NO sqrt:
    - sqrt is monotonic, so comparing squared distances gives the same
      order without float cost/precision issues.
    """
    import heapq

    heap = []                           # (-squared_distance, x, y), size <= k
    for x, y in points:
        d = x * x + y * y
        if len(heap) < k:
            heapq.heappush(heap, (-d, x, y))
        elif -heap[0][0] > d:           # Farthest kept point is farther than this one
            heapq.heappushpop(heap, (-d, x, y))

    return [[x, y] for _, x, y in heap]


# =============================================================================
# Exercise 6 (Medium): Counting Bits (O(n) DP)
# =============================================================================

def count_bits(n):
    """
    DP over bits: ans[i] = ans[i >> 1] + (i & 1).

    WHY IT WORKS:
    - i >> 1 is i without its last bit (already computed: i >> 1 < i),
      and (i & 1) adds that last bit back. O(1) per value.

    Time: O(n) total, Space: O(n) for the output itself.
    """
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans


def count_bits_kernighan(n):
    """
    Alternative recurrence: ans[i] = ans[i & (i - 1)] + 1.

    WHY i & (i - 1) CLEARS THE LOWEST SET BIT:
    - Subtracting 1 flips the lowest set bit to 0 and all bits below it
      to 1 (borrow propagation). ANDing keeps everything above unchanged
      and zeroes that whole low region — exactly "drop one 1-bit".
    """
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i & (i - 1)] + 1
    return ans


# =============================================================================
# Exercise 7 (Hard): Word Search II (trie + DFS)
# =============================================================================

def find_words(board, words):
    """
    One trie-guided DFS instead of one grid search per word.

    WHY A TRIE:
    - Searching W words independently costs O(W * R*C * 4^L). Words
      sharing a prefix share the SAME grid exploration in the trie walk:
      the DFS advances board and trie together and dies as soon as the
      current path is no word's prefix.

    DEDUP BY CONSTRUCTION:
    - The word is stored on its terminal node. On collection we set
      node.word = None: a second path to the same word finds nothing.

    DEAD LEAF PRUNING (bonus):
    - After exploring a child with no remaining children and no word,
      delete it from its parent: future DFS branches skip it entirely.
    """
    # Build the trie: plain dicts + a "word" slot on terminal nodes
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = word                # Terminal marker holds the word itself

    if not board or not board[0]:
        return []

    rows, cols = len(board), len(board[0])
    found = []

    def dfs(r, c, node):
        ch = board[r][c]
        child = node.get(ch)
        if child is None:
            return                      # No word continues with this letter

        word = child.get("$")
        if word is not None:
            found.append(word)
            child["$"] = None           # Collected: never report it again

        board[r][c] = "#"               # Mark for the current path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, child)
        board[r][c] = ch                # Restore

        # Dead leaf pruning: child exhausted → unlink it
        if not child or all(k == "$" and v is None for k, v in child.items()):
            node.pop(ch, None)

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, trie)

    return found


# =============================================================================
# Exercise 8 (Hard): Maximum XOR of Two Numbers (binary trie)
# =============================================================================

BITS = 32                               # Fixed depth: prefixes never mix


def find_maximum_xor_brute(nums):
    """O(n^2) oracle — try every pair (i <= j)."""
    best = 0
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            best = max(best, nums[i] ^ nums[j])
    return best


def find_maximum_xor(nums):
    """
    Binary trie + greedy descent — O(n * 32).

    TRIE SHAPE:
    - Each number is a fixed-length path of 32 bits, most significant
      bit first. Fixed length matters: without it, 3 (11) would collide
      with 7 (111) prefixes and the greedy walk would be wrong.

    GREEDY ARGUMENT:
    - Walking from the most significant bit, taking the OPPOSITE branch
      sets a 1 at that XOR position. A 1 at bit i outweighs ALL lower
      bits combined (2^i > 2^i - 1), so the greedy choice is always safe.

    ORDERING:
    - Insert each number BEFORE querying it: the first query then runs
      against the number itself (XOR = 0), which handles single-element
      arrays without special cases.
    """
    root = {}
    best = 0

    for num in nums:
        # Insert num (32-bit path, MSB first)
        node = root
        for i in range(BITS - 1, -1, -1):
            bit = (num >> i) & 1
            node = node.setdefault(bit, {})

        # Query: prefer the opposite bit at every level
        node = root
        xor_value = 0
        for i in range(BITS - 1, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit
            if opposite in node:
                xor_value |= 1 << i     # This level contributes a 1
                node = node[opposite]
            else:
                node = node[bit]        # Forced: same bit, contributes 0
        best = max(best, xor_value)

    return best


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

    # -- Exercise 4 --
    import random
    assert single_number_ii([2, 2, 3, 2]) == 3
    assert single_number_ii([0, 1, 0, 1, 0, 1, 99]) == 99
    assert single_number_ii([7]) == 7
    assert single_number_ii([-2, -2, 1, -2]) == 1
    assert single_number_ii([-4, -4, -4, -5]) == -5
    assert single_number_ii([1, 1, 1, 0]) == 0
    for _ in range(50):
        triples = random.sample(range(-50, 50), 6)
        single = random.choice([x for x in range(-50, 50) if x not in triples])
        arr = triples * 3 + [single]
        random.shuffle(arr)
        assert single_number_ii(arr) == single == single_number_ii_oracle(arr)
    print("Exercise 4 (single_number_ii): OK")

    # -- Exercise 5 --
    def normalize_pts(pts):
        return sorted(map(tuple, pts))

    assert normalize_pts(k_closest([[1, 3], [-2, 2]], 1)) == [(-2, 2)]
    assert normalize_pts(k_closest([[3, 3], [5, -1], [-2, 4]], 2)) == [(-2, 4), (3, 3)]
    assert normalize_pts(k_closest([[0, 1], [1, 0]], 2)) == [(0, 1), (1, 0)]
    assert normalize_pts(k_closest([[5, 5]], 1)) == [(5, 5)]
    assert normalize_pts(k_closest([[1, 1], [2, 2], [3, 3]], 3)) == [(1, 1), (2, 2), (3, 3)]
    assert normalize_pts(k_closest([[0, 0], [10, 10], [1, 1]], 2)) == [(0, 0), (1, 1)]
    print("Exercise 5 (k_closest): OK")

    # -- Exercise 6 --
    assert count_bits(2) == [0, 1, 1]
    assert count_bits(5) == [0, 1, 1, 2, 1, 2]
    assert count_bits(0) == [0]
    assert count_bits(1) == [0, 1]
    assert count_bits(16)[16] == 1
    assert count_bits(15)[15] == 4
    assert count_bits(255)[255] == 8
    assert count_bits(1000) == [bin(i).count("1") for i in range(1001)]
    assert count_bits_kernighan(1000) == count_bits(1000)
    print("Exercise 6 (count_bits): OK")

    # -- Exercise 7 --
    ws_board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    assert sorted(find_words([row[:] for row in ws_board],
                             ["oath", "pea", "eat", "rain"])) == ["eat", "oath"]
    assert find_words([["a", "b"], ["c", "d"]], ["abcb"]) == []
    assert sorted(find_words([["a", "a"]], ["a", "aa", "aaa"])) == ["a", "aa"]
    assert find_words([["a"]], []) == []
    assert sorted(find_words([["a", "b", "c"]], ["ab", "abc"])) == ["ab", "abc"]
    assert find_words([["a", "a"], ["a", "a"]], ["aa"]) == ["aa"]
    print("Exercise 7 (find_words): OK")

    # -- Exercise 8 --
    for f in (find_maximum_xor_brute, find_maximum_xor):
        assert f([3, 10, 5, 25, 2, 8]) == 28
        assert f([0]) == 0
        assert f([2, 4]) == 6
        assert f([8, 10, 2]) == 10
        assert f([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]) == 127
        assert f([0, 0, 0]) == 0
        assert f([1]) == 0
    for _ in range(50):
        arr = [random.randint(0, 1 << 20) for _ in range(random.randint(1, 60))]
        assert find_maximum_xor(arr) == find_maximum_xor_brute(arr), arr

    # Benchmark: trie linear vs brute quadratic
    import time
    print("  Benchmark (random 31-bit values):")
    print(f"    {'n':>6} | {'trie O(n*32)':>13} | {'brute O(n^2)':>13}")
    for n in [2000, 4000, 8000]:
        arr = [random.randint(0, (1 << 31) - 1) for _ in range(n)]
        start = time.perf_counter()
        r1 = find_maximum_xor(arr)
        t_trie = time.perf_counter() - start
        start = time.perf_counter()
        r2 = find_maximum_xor_brute(arr)
        t_brute = time.perf_counter() - start
        assert r1 == r2
        print(f"    {n:>6,} | {t_trie:>12.4f}s | {t_brute:>12.4f}s")
    print("Exercise 8 (find_maximum_xor): OK")

    print("\nAll Day 13 solutions pass!")

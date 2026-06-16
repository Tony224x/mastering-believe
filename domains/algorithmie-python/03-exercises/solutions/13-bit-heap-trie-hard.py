"""
Solutions — Day 13: Bit Manipulation, Heaps & Tries (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/13-bit-heap-trie-hard.py

Each solution is numbered to match the exercise file (03-hard/13-bit-heap-trie.md).
All solutions are verified with assertions at the end.
"""

import heapq


# =============================================================================
# EXERCISE 7 (Hard): Find Median from Data Stream — two heaps
# =============================================================================

class MedianFinder:
    """
    `low` is a max-heap (stored negated) for the lower half; `high` is a min-heap
    for the upper half. Invariants: every low <= every high, and
    len(low) - len(high) in {0, 1}.

    add_num: O(log n), find_median: O(1)
    """
    def __init__(self):
        self.low = []                  # Max-heap via negation
        self.high = []                 # Min-heap

    def add_num(self, num):
        heapq.heappush(self.low, -num)
        # Move the largest of low into high to keep ordering
        heapq.heappush(self.high, -heapq.heappop(self.low))
        # Re-balance so low has equal or one more element than high
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def find_median(self):
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2


def test_exercise_7():
    print("\nExercise 7: Find Median from Data Stream")

    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0

    mf2 = MedianFinder()
    mf2.add_num(5)
    assert mf2.find_median() == 5.0

    import statistics
    import random
    mf3 = MedianFinder()
    rng = random.Random(7)
    seen = []
    for _ in range(200):
        x = rng.randint(-50, 50)
        mf3.add_num(x)
        seen.append(x)
        assert mf3.find_median() == statistics.median(seen)

    print("  PASS — all test cases (incl. statistics.median cross-check)")


# =============================================================================
# EXERCISE 8 (Hard): Word Search II — trie + DFS
# =============================================================================

class _Node:
    __slots__ = ("children", "word")
    def __init__(self):
        self.children = {}
        self.word = None               # Stores the full word at terminal nodes


def find_words(board, words):
    """
    Build a trie of all words, then run one DFS per cell that descends the trie,
    pruning as soon as the current prefix is absent.

    Time: O(R * C * 4^L), independent of the number of words.
    """
    root = _Node()
    for word in words:
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = _Node()
            node = node.children[c]
        node.word = word

    if not board or not board[0]:
        return []
    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return
        nxt = node.children[char]
        if nxt.word is not None:
            result.append(nxt.word)
            nxt.word = None            # De-duplicate
        board[r][c] = "#"              # Mark visited
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, nxt)
        board[r][c] = char             # Restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return result


def test_exercise_8():
    print("\nExercise 8: Word Search II")

    board = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    assert sorted(find_words(board, ["oath", "pea", "eat", "rain"])) == ["eat", "oath"]
    assert find_words([["a"]], ["b"]) == []
    assert sorted(find_words([["a", "b"]], ["a", "b", "ab", "ba"])) == ["a", "ab", "b", "ba"]
    assert find_words([["a", "a"]], ["aaa"]) == []
    assert sorted(find_words([["a"]], ["a"])) == ["a"]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Add and Search Word — trie + wildcard DFS
# =============================================================================

class WordDictionary:
    """
    Standard trie insert. search handles '.' by trying ALL children at that node
    via DFS. Success requires both length match and is_end at the final node.

    search: O(L) without wildcards, O(26^d) worst case with d dots.
    """
    def __init__(self):
        self.root = {}                 # Nested dicts; "$" key marks word end

    def add_word(self, word):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node["$"] = True               # End-of-word marker

    def search(self, word):
        def dfs(node, idx):
            if idx == len(word):
                return "$" in node
            c = word[idx]
            if c == ".":
                for key, child in node.items():
                    if key != "$" and dfs(child, idx + 1):
                        return True
                return False
            if c not in node:
                return False
            return dfs(node[c], idx + 1)

        return dfs(self.root, 0)


def test_exercise_9():
    print("\nExercise 9: Add and Search Word")

    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")
    assert wd.search("pad") is False
    assert wd.search("bad") is True
    assert wd.search(".ad") is True
    assert wd.search("b..") is True
    assert wd.search("...") is True
    assert wd.search("....") is False
    assert wd.search("..") is False

    wd2 = WordDictionary()
    assert wd2.search("a") is False
    assert wd2.search(".") is False

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 13: Bit Manipulation, Heaps & Tries (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

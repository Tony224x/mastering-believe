"""
Solutions — Day 14 Mock Interviews (HARD).
Run: python domains/algorithmie-python/03-exercises/solutions/14-mock-interviews-hard.py

Each solution maps to the exercise file (03-hard/14-mock-interviews.md):
  Exercise 7 -> Merge Intervals
  Exercise 8 -> LRU Cache (design)
  Exercise 9 -> Trapping Rain Water

These are narrated like a debrief: the "why" (design choices, complexity) matters
more than the "what". All solutions are stdlib-only and verified with assertions.
"""

from collections import OrderedDict


# =============================================================================
# EXERCISE 7 (Hard): Merge Intervals — sort then sweep
# =============================================================================

def merge_intervals(intervals):
    """
    Sorting is the unlock: once intervals are ordered by start, any overlap can
    only happen with the LAST interval we kept. So a single sweep suffices.

    For each interval, either it overlaps the last result (start <= last_end ->
    extend last_end with max, because an interval may be fully contained), or it
    is disjoint (append a fresh copy).

    Why `max` on the end? Case [[1,4],[2,3]]: the second is inside the first, so
    we must keep end=4, not overwrite with 3.

    Endpoint touch ([1,4] & [4,5]) counts as overlap -> use `<=`.

    Time : O(n log n)  (the sort dominates)
    Space: O(n)        (the output list)
    """
    if not intervals:
        return []
    # Sort by start; copy each pair so we never mutate the caller's lists.
    ordered = sorted(([s, e] for s, e in intervals), key=lambda x: x[0])
    result = [ordered[0]]
    for start, end in ordered[1:]:
        last = result[-1]
        if start <= last[1]:           # Overlap (incl. endpoint touch)
            last[1] = max(last[1], end)
        else:
            result.append([start, end])
    return result


def test_exercise_7():
    print("\nExercise 7: Merge Intervals")

    def norm(out):
        return sorted([list(x) for x in out])

    assert norm(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])) == [[1, 6], [8, 10], [15, 18]]
    assert norm(merge_intervals([[1, 4], [4, 5]])) == [[1, 5]]
    assert norm(merge_intervals([[1, 4], [2, 3]])) == [[1, 4]]
    assert norm(merge_intervals([[1, 4], [0, 4]])) == [[0, 4]]
    assert merge_intervals([]) == []
    assert merge_intervals([[5, 7]]) == [[5, 7]]
    assert norm(merge_intervals([[1, 4], [0, 0]])) == [[0, 0], [1, 4]]

    # Caller's input must not be mutated.
    src = [[2, 3], [1, 5]]
    _ = merge_intervals(src)
    assert src == [[2, 3], [1, 5]]

    print("  PASS — all test cases (incl. no input mutation)")


# =============================================================================
# EXERCISE 8 (Hard): LRU Cache — hashmap + doubly linked list, O(1)
# =============================================================================

class _DNode:
    """Doubly linked list node. Sentinels avoid None checks at the ends."""
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    """
    Two structures composed to make both ops O(1):
      - dict `key -> node`           : O(1) lookup
      - doubly linked list           : O(1) move-to-front and tail eviction

    Layout: head <-> (most recent) ... (least recent) <-> tail.
    Sentinel head/tail mean every real node always has both neighbors, so the
    splice helpers never branch on None.

    get : O(1)   put : O(1)   space: O(capacity)
    """

    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}                      # key -> _DNode
        self.head = _DNode()               # Sentinel: most-recent side
        self.tail = _DNode()               # Sentinel: least-recent side
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        # Unlink a node from wherever it sits.
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        # Insert right after head -> becomes most recently used.
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)                 # Touch => refresh recency
        self._add_front(node)
        return node.val

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.val = value               # Update value...
            self._remove(node)             # ...and refresh recency
            self._add_front(node)
            return
        if len(self.map) >= self.cap:
            # Evict the true LRU = the node just before tail.
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        node = _DNode(key, value)
        self.map[key] = node
        self._add_front(node)


# An idiomatic alternative the interviewer accepts: OrderedDict + move_to_end.
class LRUCacheOrdered:
    def __init__(self, capacity):
        self.cap = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)         # Mark most recent
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.cap:
            self.data.popitem(last=False)  # Pop least recent (front)


def test_exercise_8():
    print("\nExercise 8: LRU Cache")

    for Impl in (LRUCache, LRUCacheOrdered):
        c = Impl(2)
        c.put(1, 1)
        c.put(2, 2)
        assert c.get(1) == 1
        c.put(3, 3)                        # Evicts 2 (LRU)
        assert c.get(2) == -1
        c.put(4, 4)                        # Evicts 1
        assert c.get(1) == -1
        assert c.get(3) == 3
        assert c.get(4) == 4

        c2 = Impl(2)
        c2.put(2, 1)
        c2.put(2, 2)                       # Update, no eviction
        assert c2.get(2) == 2
        c2.put(1, 1)
        c2.put(4, 1)                       # Evicts 2 (LRU)
        assert c2.get(2) == -1

        c3 = Impl(1)                       # Capacity 1 edge
        c3.put(1, 10)
        c3.put(2, 20)                      # Evicts 1
        assert c3.get(1) == -1
        assert c3.get(2) == 20

    print("  PASS — all test cases (both DLL and OrderedDict impls)")


# =============================================================================
# EXERCISE 9 (Hard): Trapping Rain Water — two pointers, O(1) space
# =============================================================================

def trap(height):
    """
    Water above index i = min(max_left, max_right) - height[i] (when positive).

    Naive: recompute both maxes per index -> O(n^2).
    Better: precompute prefix/suffix maxes -> O(n) time but O(n) space.
    Best (this): two pointers in O(1) space.

    Key insight for two pointers: move the side with the SMALLER running max.
    That side is the limiting wall, so the water there is fully determined by
    its own running max -- we can bank it immediately without knowing the exact
    other-side max (we only need to know the other side is >= this one).

    Time : O(n)   Space: O(1)
    """
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    total = 0
    while left < right:
        if left_max <= right_max:
            # Left wall is the limiter: water here depends only on left_max.
            left += 1
            left_max = max(left_max, height[left])
            total += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            total += right_max - height[right]
    return total


# Reference O(n)-space version, used as an independent cross-check in tests.
def trap_prefix_suffix(height):
    n = len(height)
    if n == 0:
        return 0
    pre = [0] * n
    suf = [0] * n
    pre[0] = height[0]
    for i in range(1, n):
        pre[i] = max(pre[i - 1], height[i])
    suf[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        suf[i] = max(suf[i + 1], height[i])
    return sum(min(pre[i], suf[i]) - height[i] for i in range(n))


def test_exercise_9():
    print("\nExercise 9: Trapping Rain Water")

    cases = [
        [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
        [4, 2, 0, 3, 2, 5],
        [],
        [1],
        [2, 2],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [3, 0, 3],
        [0, 0, 0],
    ]
    expected = [6, 9, 0, 0, 0, 0, 0, 3, 0]
    for h, exp in zip(cases, expected):
        assert trap(h) == exp, f"{h} -> {trap(h)} != {exp}"
        # Cross-check the O(1)-space answer against the O(n)-space reference.
        assert trap(h) == trap_prefix_suffix(h)

    print("  PASS — all test cases (two-pointer == prefix/suffix reference)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 14: Mock Interviews (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

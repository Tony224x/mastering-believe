"""
Solutions — Day 5: Linked Lists (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/05-linked-lists-hard.py

Each solution is numbered to match the exercise file (03-hard/05-linked-lists.md).
All solutions are verified with assertions at the end.
"""

import heapq
from typing import Optional


# =============================================================================
# DATA STRUCTURE + HELPERS
# =============================================================================

class ListNode:
    """Singly linked list node."""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def build(values) -> Optional[ListNode]:
    dummy = ListNode(0)
    tail = dummy
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> list:
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


# =============================================================================
# EXERCISE 7 (Hard): Reverse Nodes in k-Group
# =============================================================================

def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Reverse the list in groups of k; leave a trailing group of < k untouched.

    APPROACH (iterative, O(1) space):
    - group_prev points to the node before the current group.
    - First check there are k nodes ahead (kth). If not, stop.
    - Reverse exactly k nodes, then re-stitch:
        group_prev -> (reversed group) -> node after the group.

    Time: O(n), Space: O(1)
    """
    if k <= 1 or head is None:
        return head

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        # Find the k-th node from group_prev; bail if fewer than k remain
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next

        group_next = kth.next          # First node of the NEXT group
        # Reverse the group [group_prev.next .. kth]
        prev, curr = group_next, group_prev.next
        while curr is not group_next:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        # group_prev.next was the group's head, now it's the group's tail
        tmp = group_prev.next
        group_prev.next = kth          # kth is the new head of this group
        group_prev = tmp               # Old head is the tail -> next group's prev


def test_exercise_7():
    print("\nExercise 7: Reverse Nodes in k-Group")

    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 1)) == [1, 2, 3, 4, 5]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 5)) == [5, 4, 3, 2, 1]
    assert to_list(reverse_k_group(build([1, 2, 3, 4]), 2)) == [2, 1, 4, 3]
    assert to_list(reverse_k_group(build([1]), 2)) == [1]
    assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6, 7]), 3)) == [3, 2, 1, 6, 5, 4, 7]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): Merge k Sorted Lists — Min-heap
# =============================================================================

def merge_k_lists(lists) -> Optional[ListNode]:
    """
    Merge k sorted lists using a min-heap of (value, tiebreaker, node).

    WHY THE TIEBREAKER:
    - heapq compares tuples element by element. If two values are equal it would
      try to compare ListNode objects, which raises TypeError. A unique counter
      breaks ties before reaching the node.

    Time: O(N log k) where N = total nodes, k = number of lists.
    Space: O(k) for the heap (reuses existing nodes for the output).
    """
    heap = []
    counter = 0                    # Unique, monotonically increasing tiebreaker

    # Seed the heap with the head of each non-empty list
    for node in lists:
        if node:
            heapq.heappush(heap, (node.val, counter, node))
            counter += 1

    dummy = ListNode(0)
    tail = dummy

    while heap:
        _, _, node = heapq.heappop(heap)
        tail.next = node           # Reuse the existing node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1

    tail.next = None               # Terminate cleanly
    return dummy.next


def test_exercise_8():
    print("\nExercise 8: Merge k Sorted Lists")

    r = merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])
    assert to_list(r) == [1, 1, 2, 3, 4, 4, 5, 6]

    assert to_list(merge_k_lists([])) == []
    assert to_list(merge_k_lists([build([])])) == []
    assert to_list(merge_k_lists([build([]), build([1]), build([])])) == [1]
    assert to_list(merge_k_lists([build([1, 2, 3])])) == [1, 2, 3]
    assert to_list(merge_k_lists([build([5]), build([4]), build([3]), build([2]), build([1])])) == [1, 2, 3, 4, 5]

    lists = [build([i, i + 10, i + 20]) for i in range(10)]
    assert to_list(merge_k_lists(lists)) == sorted(range(30))

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Copy List with Random Pointer — Interleaving (O(1) aux)
# =============================================================================

class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def copy_random_list(head: Optional[Node]) -> Optional[Node]:
    """
    Deep copy with the O(1)-auxiliary interleaving technique.

    THREE PASSES:
    1. Insert a clone right after each original node: A -> A' -> B -> B' -> ...
    2. Wire clones' random: clone.random = original.random.next (the original's
       random's clone sits right after it).
    3. Detach the two interleaved lists, restoring the original's next pointers.

    Time: O(n), Space: O(1) auxiliary (ignoring the output itself).
    """
    if head is None:
        return None

    # Pass 1: interleave clones
    curr = head
    while curr:
        clone = Node(curr.val, curr.next, None)
        curr.next = clone
        curr = clone.next

    # Pass 2: set random pointers on the clones
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # Pass 3: unweave the original and the copy
    curr = head
    copy_head = head.next
    while curr:
        clone = curr.next
        curr.next = clone.next                  # Restore original
        clone.next = clone.next.next if clone.next else None
        curr = curr.next

    return copy_head


# --- Test helpers (mirrors of the exercise file) ---

def build_with_random(arr):
    nodes = [Node(v) for v, _ in arr]
    for i, (_, r) in enumerate(arr):
        nodes[i].next = nodes[i + 1] if i + 1 < len(nodes) else None
        nodes[i].random = nodes[r] if r is not None else None
    return nodes[0] if nodes else None


def serialize(head):
    idx, n = {}, head
    i = 0
    while n:
        idx[n] = i
        n = n.next
        i += 1
    out, n = [], head
    while n:
        out.append((n.val, idx[n.random] if n.random else None))
        n = n.next
    return out


def assert_deep_copy(original, copy):
    assert serialize(original) == serialize(copy)
    orig_ids, n = set(), original
    while n:
        orig_ids.add(id(n))
        n = n.next
    n = copy
    while n:
        assert id(n) not in orig_ids          # Every node is brand new
        n = n.next


def test_exercise_9():
    print("\nExercise 9: Copy List with Random Pointer")

    cases = [
        [(7, None), (13, 0), (11, 4), (10, 2), (1, 0)],
        [(1, 1), (2, 1)],
        [(3, None), (3, 0), (3, None)],
        [],
        [(1, None)],
        [(1, 0)],                                # Self-reference random
    ]
    for case in cases:
        src = build_with_random(case)
        cpy = copy_random_list(src)
        assert_deep_copy(src, cpy)

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 5: Linked Lists (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

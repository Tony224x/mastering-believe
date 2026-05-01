"""
Day 5 — Linked Lists: Fast/Slow Pointers, Reversal & Merge
Run: python domains/algorithmie-python/02-code/05-linked-lists.py

Focus: manipulation of pointers with CORRECT invariants. Every non-obvious
line has a comment explaining WHY the assignment is safe and what invariant
it maintains.
"""

from typing import Optional


# =============================================================================
# DATA STRUCTURE
# =============================================================================

class ListNode:
    """Singly linked list node."""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

    # Pretty printing for demos
    def __repr__(self):
        vals = []
        curr = self
        # Guard against cycles to avoid infinite repr
        for _ in range(50):
            if curr is None:
                break
            vals.append(str(curr.val))
            curr = curr.next
        return " -> ".join(vals) + (" -> None" if curr is None else " -> ...")


def build_list(values) -> Optional[ListNode]:
    """Build a linked list from an iterable of values. Returns the head."""
    dummy = ListNode(0)
    tail = dummy
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]):
    """Convert a linked list to a Python list. Caps at 1000 nodes for safety."""
    out = []
    curr = head
    for _ in range(1000):
        if curr is None:
            break
        out.append(curr.val)
        curr = curr.next
    return out


# =============================================================================
# SECTION 1: FAST & SLOW POINTERS
# =============================================================================

def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Return the middle node. For even-length lists, return the second middle.

    INVARIANT:
    - slow advances by 1 per step, fast by 2.
    - When fast reaches the end (or one step before), slow is at the middle.

    Time: O(n), Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Floyd's cycle detection.

    INTUITION:
    - If there's a cycle, fast will lap slow inside it. Each iteration the gap
      between them (measured along the cycle) decreases by 1.
    - If there's no cycle, fast reaches None and we exit.

    Time: O(n), Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:             # Same node -> cycle confirmed
            return True
    return False


def detect_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Return the node where the cycle begins, or None if no cycle.

    TWO-PHASE FLOYD:
    - Phase 1: detect the meeting point inside the cycle.
    - Phase 2: restart one pointer from head at speed 1; when they meet again,
      that's the cycle entry.

    Time: O(n), Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None                  # No cycle

    # Phase 2
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next
    return slow


# =============================================================================
# SECTION 2: REVERSAL
# =============================================================================

def reverse_iterative(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative reversal — the canonical three-pointer dance.

    INVARIANT AT LOOP START:
    - prev points to the already-reversed prefix's head
    - curr points to the next node to process
    - Everything from curr onward is still in ORIGINAL order

    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head
    while curr:
        nxt = curr.next          # 1. Save next BEFORE we clobber the link
        curr.next = prev         # 2. Flip the arrow backwards
        prev = curr              # 3. Advance prev into the reversed prefix
        curr = nxt               # 4. Advance curr into the remaining tail
    return prev                  # New head = old tail (the last non-None prev)


def reverse_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive reversal.

    WHY THE head.next = None LINE:
    - After head.next.next = head, the former second node now points back to head.
    - If we don't reset head.next to None, head still points to the second node,
      creating a 2-node cycle. Always break the forward link.

    Time: O(n), Space: O(n) due to recursion stack
    """
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head            # The second node now points back to head
    head.next = None                 # Break the forward link to avoid a cycle
    return new_head


# =============================================================================
# SECTION 3: MERGE TWO SORTED LISTS
# =============================================================================

def merge_two_sorted(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Merge two sorted lists into one sorted list — reusing existing nodes.

    WHY A DUMMY HEAD:
    - We need a "previous tail" at the very first iteration. The dummy gives us one
      for free, eliminating the edge case "what if result is empty?"

    Time: O(n + m), Space: O(1)
    """
    dummy = ListNode(0)
    tail = dummy

    while l1 and l2:
        # Attach the smaller node, advance that list
        if l1.val <= l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next             # The new tail is the node we just attached

    # At most one of l1, l2 is non-None — append the remainder in O(1)
    tail.next = l1 or l2

    return dummy.next


# =============================================================================
# SECTION 4: TWO POINTERS WITH GAP
# =============================================================================

def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Remove the nth node from the end using a gap of n+1 between two pointers.

    WHY GAP = n+1 (not n):
    - We want `slow` to land ONE NODE BEFORE the target so we can delete
      `slow.next`. Starting both at `dummy` and advancing fast by n+1, when
      fast reaches None, slow is at position (length - n), which is exactly
      one before the target.

    Time: O(n), Space: O(1)
    """
    dummy = ListNode(0, head)
    fast = slow = dummy

    # Open the gap
    for _ in range(n + 1):
        fast = fast.next              # Caller guarantees n <= length, so this is safe

    # Walk both pointers together until fast reaches None
    while fast:
        fast = fast.next
        slow = slow.next

    # slow is just before the node to remove
    slow.next = slow.next.next        # Skip over the target node

    return dummy.next


def get_intersection(headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the intersection node of two linked lists in O(1) extra space.

    TRICK:
    - Each pointer walks A then B (and the other walks B then A). Total distance
      covered by each is len(A) + len(B), so they are in lockstep modulo the
      intersection. They meet at the intersection (or both at None).

    Time: O(n + m), Space: O(1)
    """
    if not headA or not headB:
        return None
    a, b = headA, headB
    while a is not b:
        # When a pointer reaches the end, it jumps to the OTHER list's head
        a = a.next if a else headB
        b = b.next if b else headA
    return a                          # Either the intersection node or None


# =============================================================================
# DEMOS
# =============================================================================

def demo_fast_slow():
    print("\n" + "=" * 70)
    print("FAST & SLOW: middle node and cycle detection")
    print("=" * 70)

    head = build_list([1, 2, 3, 4, 5])
    assert find_middle(head).val == 3

    head = build_list([1, 2, 3, 4])
    assert find_middle(head).val == 3           # Second middle for even length

    # Cycle detection
    h = build_list([1, 2, 3, 4, 5])
    assert has_cycle(h) is False
    # Build a cycle: 5 -> 3
    last = h
    while last.next:
        last = last.next
    last.next = h.next.next                     # Tail points to node '3'
    assert has_cycle(h) is True
    start = detect_cycle_start(h)
    assert start.val == 3
    # Break the cycle for safety
    last.next = None
    print("  All fast/slow tests passed")


def demo_reversal():
    print("\n" + "=" * 70)
    print("REVERSAL: iterative and recursive")
    print("=" * 70)

    head = build_list([1, 2, 3, 4, 5])
    reversed_head = reverse_iterative(head)
    assert to_list(reversed_head) == [5, 4, 3, 2, 1]

    head = build_list([1, 2, 3])
    reversed_head = reverse_recursive(head)
    assert to_list(reversed_head) == [3, 2, 1]

    # Edge cases
    assert reverse_iterative(None) is None
    single = ListNode(42)
    assert reverse_iterative(single).val == 42
    print("  All reversal tests passed")


def demo_merge():
    print("\n" + "=" * 70)
    print("MERGE two sorted lists")
    print("=" * 70)

    l1 = build_list([1, 2, 4])
    l2 = build_list([1, 3, 4])
    merged = merge_two_sorted(l1, l2)
    assert to_list(merged) == [1, 1, 2, 3, 4, 4]

    # Edge cases: one list empty
    assert to_list(merge_two_sorted(None, build_list([1, 2]))) == [1, 2]
    assert to_list(merge_two_sorted(build_list([1, 2]), None)) == [1, 2]
    assert merge_two_sorted(None, None) is None
    print("  All merge tests passed")


def demo_gap_pointers():
    print("\n" + "=" * 70)
    print("TWO POINTERS WITH GAP: remove nth from end, intersection")
    print("=" * 70)

    head = build_list([1, 2, 3, 4, 5])
    out = remove_nth_from_end(head, 2)           # Remove node with value 4
    assert to_list(out) == [1, 2, 3, 5]

    # Remove head
    head = build_list([1, 2])
    out = remove_nth_from_end(head, 2)
    assert to_list(out) == [2]

    # Intersection demo: two lists that share a tail
    # A: 4 -> 1 -\
    #             8 -> 4 -> 5
    # B: 5 -> 6 -> 1 -/
    shared = build_list([8, 4, 5])
    a = ListNode(4, ListNode(1, shared))
    b = ListNode(5, ListNode(6, ListNode(1, shared)))
    inter = get_intersection(a, b)
    assert inter is shared
    assert inter.val == 8

    # No intersection
    c = build_list([1, 2, 3])
    d = build_list([4, 5, 6])
    assert get_intersection(c, d) is None
    print("  All gap-pointer tests passed")


if __name__ == "__main__":
    demo_fast_slow()
    demo_reversal()
    demo_merge()
    demo_gap_pointers()
    print("\nAll Day 5 demos passed.")

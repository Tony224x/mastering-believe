"""
Solutions — Day 5: Linked Lists (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/05-linked-lists-medium.py

Each solution is numbered to match the exercise file (02-medium/05-linked-lists.md).
All solutions are verified with assertions at the end.
"""

from typing import Optional


# =============================================================================
# DATA STRUCTURE + HELPERS (shared by all exercises)
# =============================================================================

class ListNode:
    """Singly linked list node."""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def build(values) -> Optional[ListNode]:
    """Build a linked list from an iterable. Return the head."""
    dummy = ListNode(0)
    tail = dummy
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> list:
    """Convert a linked list to a Python list."""
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


# =============================================================================
# EXERCISE 4 (Medium): Reverse Linked List II — Head-insertion trick
# =============================================================================

def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """
    Reverse the sublist [left, right] (1-indexed) in one pass, O(1) space.

    APPROACH (head-insertion):
    - dummy -> head lets us treat left = 1 uniformly.
    - Walk 'prev' to the node just BEFORE position left.
    - 'curr' is the first node to reverse. Repeatedly take curr.next ('nxt')
      and splice it right after 'prev'. After (right - left) moves, the segment
      is reversed.

    Time: O(n), Space: O(1)
    """
    if head is None or left == right:
        return head

    dummy = ListNode(0, head)
    prev = dummy
    for _ in range(left - 1):          # Stop just before position 'left'
        prev = prev.next

    curr = prev.next                   # First node of the segment (stays last)
    for _ in range(right - left):
        nxt = curr.next                # Node to move to the front of the segment
        curr.next = nxt.next           # Detach nxt
        nxt.next = prev.next           # nxt jumps to the front
        prev.next = nxt                # Re-link the front to prev

    return dummy.next


def test_exercise_4():
    print("\nExercise 4: Reverse Linked List II")

    assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
    assert to_list(reverse_between(build([5]), 1, 1)) == [5]
    assert to_list(reverse_between(build([1, 2]), 1, 2)) == [2, 1]
    assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 1, 5)) == [5, 4, 3, 2, 1]
    assert to_list(reverse_between(build([1, 2, 3, 4, 5]), 3, 3)) == [1, 2, 3, 4, 5]
    assert to_list(reverse_between(build([3, 5]), 1, 1)) == [3, 5]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Add Two Numbers — Carry propagation
# =============================================================================

def add_two_numbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Digit-by-digit addition with carry. Digits are stored least-significant first.

    INVARIANT:
    - At each step, total = l1.val + l2.val + carry.
    - New digit = total % 10, new carry = total // 10.
    - Loop while either list has nodes OR carry is nonzero (final carry creates
      a new leading node, e.g. 99 + 1 -> 100).

    Time: O(max(n, m)), Space: O(max(n, m)) for the result.
    """
    dummy = ListNode(0)
    tail = dummy
    carry = 0

    while l1 or l2 or carry:
        a = l1.val if l1 else 0
        b = l2.val if l2 else 0
        total = a + b + carry
        carry, digit = divmod(total, 10)
        tail.next = ListNode(digit)
        tail = tail.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next


def test_exercise_5():
    print("\nExercise 5: Add Two Numbers")

    assert to_list(add_two_numbers(build([2, 4, 3]), build([5, 6, 4]))) == [7, 0, 8]
    assert to_list(add_two_numbers(build([0]), build([0]))) == [0]
    assert to_list(add_two_numbers(build([9, 9, 9, 9, 9, 9, 9]), build([9, 9, 9, 9]))) == [8, 9, 9, 9, 0, 0, 0, 1]
    assert to_list(add_two_numbers(build([5]), build([5]))) == [0, 1]
    assert to_list(add_two_numbers(build([1, 8]), build([0]))) == [1, 8]
    assert to_list(add_two_numbers(build([9, 9]), build([1]))) == [0, 0, 1]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Odd Even Linked List — Two interleaved sublists
# =============================================================================

def odd_even_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Split nodes by position parity, then concatenate odd-positions + even-positions.

    APPROACH:
    - 'odd' walks the 1st, 3rd, 5th ... nodes; 'even' walks the 2nd, 4th ...
    - Keep 'even_head' to re-attach the even sublist after the odd one.
    - Advance by jumps of two, weaving the two sublists in place.

    Time: O(n), Space: O(1) — no node allocation.
    """
    if head is None or head.next is None:
        return head

    odd = head
    even = head.next
    even_head = even               # Remember start of the even sublist

    while even and even.next:
        odd.next = even.next       # Link odd to the next odd node
        odd = odd.next
        even.next = odd.next       # Link even to the next even node
        even = even.next

    odd.next = even_head           # Attach even sublist after the odd tail
    return head


def test_exercise_6():
    print("\nExercise 6: Odd Even Linked List")

    assert to_list(odd_even_list(build([1, 2, 3, 4, 5]))) == [1, 3, 5, 2, 4]
    assert to_list(odd_even_list(build([2, 1, 3, 5, 6, 4, 7]))) == [2, 3, 6, 7, 1, 5, 4]
    assert to_list(odd_even_list(build([1]))) == [1]
    assert to_list(odd_even_list(build([1, 2]))) == [1, 2]
    assert to_list(odd_even_list(build([]))) == []
    assert to_list(odd_even_list(build([1, 2, 3, 4]))) == [1, 3, 2, 4]

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 5: Linked Lists (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

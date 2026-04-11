"""
Solutions — Day 5: Linked Lists
Run: python domains/algorithmie-python/03-exercises/solutions/05-linked-lists.py

Each solution is numbered to match the exercise file.
All solutions are verified with assertions at the end.
"""

from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def build(values) -> Optional[ListNode]:
    """Build a linked list from a Python list."""
    dummy = ListNode(0)
    t = dummy
    for v in values:
        t.next = ListNode(v)
        t = t.next
    return dummy.next


def to_list(head: Optional[ListNode]):
    """Convert a linked list to a Python list."""
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


# =============================================================================
# EXERCISE 1 (Easy): Reverse Linked List
# =============================================================================

def reverse_list(head):
    """
    Iterative reversal using three pointers.

    INVARIANT AT LOOP ENTRY:
    - prev points to the head of the already-reversed prefix
    - curr points to the next node to flip
    - Everything reachable from curr is still in ORIGINAL order

    STEPS (in this exact order):
    1. nxt = curr.next   — save the rest BEFORE we clobber curr.next
    2. curr.next = prev  — flip the arrow backwards
    3. prev = curr       — advance prev (it's now the new head of the reversed prefix)
    4. curr = nxt        — advance curr into the un-reversed tail

    Time: O(n), Space: O(1)
    """
    prev = None
    curr = head
    while curr:
        nxt = curr.next         # Must happen first, or we lose the rest
        curr.next = prev
        prev = curr
        curr = nxt
    # When curr is None, prev is the last node we touched — the new head
    return prev


def test_exercise_1():
    print("\nExercise 1: Reverse Linked List")

    assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
    assert to_list(reverse_list(build([1, 2]))) == [2, 1]
    assert to_list(reverse_list(build([1]))) == [1]
    assert reverse_list(None) is None

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 2 (Easy): Middle of the Linked List
# =============================================================================

def middle_node(head):
    """
    Fast & Slow pointers — the canonical one-pass midpoint algorithm.

    INTUITION:
    - slow moves 1 step per iteration, fast moves 2.
    - When fast reaches the end, slow has moved exactly half as far,
      so it's at the middle.
    - For even length, this implementation returns the SECOND middle:
      that matches LeetCode 876 spec.

    CONDITION `while fast and fast.next`:
    - fast.next.next would crash if fast.next is None.
    - Checking both ensures we can always advance 2 steps.

    Time: O(n), Space: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def test_exercise_2():
    print("\nExercise 2: Middle of the Linked List")

    assert middle_node(build([1, 2, 3, 4, 5])).val == 3
    assert middle_node(build([1, 2, 3, 4, 5, 6])).val == 4    # Second middle
    assert middle_node(build([1])).val == 1
    assert middle_node(build([1, 2])).val == 2

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 3 (Easy): Merge Two Sorted Lists
# =============================================================================

def merge_two_lists(list1, list2):
    """
    Merge using a dummy head + tail pointer.

    WHY DUMMY HEAD:
    - On the first iteration, there's no "previous node" to attach to.
    - A sentinel gives us that "previous node" for free, removing a special case.

    REUSE OF NODES:
    - We never create new ListNode instances. We only rewire existing `.next`
      pointers. This gives O(1) extra space.

    TAIL OF ONE LIST REMAINING:
    - At most one of list1/list2 is non-None after the loop.
    - `tail.next = list1 or list2` attaches the rest in O(1):
      Python's `or` returns the first truthy operand, or the second if both falsy.

    Time: O(n + m), Space: O(1)
    """
    dummy = ListNode(0)
    tail = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next            # tail is always the last attached node

    # Attach the remaining non-empty list (or None if both are empty)
    tail.next = list1 or list2

    return dummy.next               # Skip the sentinel to return the real head


def test_exercise_3():
    print("\nExercise 3: Merge Two Sorted Lists")

    assert to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
    assert to_list(merge_two_lists(build([]), build([]))) == []
    assert to_list(merge_two_lists(build([]), build([0]))) == [0]
    assert to_list(merge_two_lists(build([1, 2, 3]), build([]))) == [1, 2, 3]
    assert to_list(merge_two_lists(build([5]), build([1, 2, 3]))) == [1, 2, 3, 5]

    print("  PASS — all test cases")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    print("\nAll Day 5 exercise solutions passed.")

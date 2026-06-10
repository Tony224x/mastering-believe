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

# =============================================================================
# EXERCISE 4 (Medium): Linked List Cycle II (cycle start)
# =============================================================================

def detect_cycle_oracle(head):
    """
    O(n) space oracle: remember every visited node in a set.
    The FIRST node seen twice is the cycle start.
    Kept to cross-check Floyd's O(1)-space version.
    """
    seen = set()
    node = head
    while node:
        if node in seen:
            return node
        seen.add(node)
        node = node.next
    return None


def detect_cycle(head):
    """
    Floyd's algorithm, both phases. O(n) time, O(1) space.

    PHASE 1 — detect: fast moves 2 steps, slow moves 1. If they meet,
    there is a cycle.

    PHASE 2 — locate the start: reset one pointer to head, advance both
    ONE step at a time; they meet exactly at the cycle start.

    WHY PHASE 2 WORKS:
    - Let a = distance head -> cycle start, b = distance cycle start ->
      meeting point, L = cycle length.
    - At the meeting point: slow walked a + b, fast walked 2(a + b).
      fast's extra distance (a + b) is a whole number of laps: a + b = kL.
    - So a = kL - b, i.e. walking `a` steps from the meeting point lands
      on the cycle start (b + a = kL ≡ 0 mod L). Hence the two pointers
      (one from head, one from the meeting point) collide at the start.
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:                # Identity check, not value equality
            # Phase 2: same speed from head and from the meeting point
            probe = head
            while probe is not slow:
                probe = probe.next
                slow = slow.next
            return probe
    return None                          # fast hit the end: no cycle


def test_exercise_4():
    print("\nExercise 4: Linked List Cycle II")

    # 3 -> 2 -> 0 -> -4 -> (back to 2)
    n1, n2, n3, n4 = ListNode(3), ListNode(2), ListNode(0), ListNode(-4)
    n1.next, n2.next, n3.next, n4.next = n2, n3, n4, n2
    assert detect_cycle(n1) is n2
    assert detect_cycle_oracle(n1) is n2

    a, b = ListNode(1), ListNode(2)
    a.next, b.next = b, b
    assert detect_cycle(a) is b

    assert detect_cycle(build([1, 2, 3])) is None
    assert detect_cycle(None) is None
    assert detect_cycle(ListNode(1)) is None

    c = ListNode(1)
    c.next = c
    assert detect_cycle(c) is c

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Remove Nth Node From End
# =============================================================================

def remove_nth_from_end(head, n):
    """
    One pass with two pointers separated by a fixed gap.

    APPROACH:
    - From a dummy head, advance `lead` by n+1 steps.
    - Then move `lead` and `lag` together until `lead` is None.
    - The gap guarantees `lag` stops on the PREDECESSOR of the target,
      so deletion is just lag.next = lag.next.next.

    WHY THE DUMMY HEAD:
    - When the head itself must be removed (n == length), the predecessor
      is the dummy — no special case needed.

    Time: O(n) single pass, Space: O(1)
    """
    dummy = ListNode(0, head)
    lead = lag = dummy

    # Gap of n+1 so that lag ends on the node BEFORE the one to delete
    for _ in range(n + 1):
        lead = lead.next

    while lead:
        lead = lead.next
        lag = lag.next

    lag.next = lag.next.next            # Unlink the target node
    return dummy.next


def test_exercise_5():
    print("\nExercise 5: Remove Nth Node From End")

    assert to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert to_list(remove_nth_from_end(build([1]), 1)) == []
    assert to_list(remove_nth_from_end(build([1, 2]), 2)) == [2]    # Head removal
    assert to_list(remove_nth_from_end(build([1, 2]), 1)) == [1]
    assert to_list(remove_nth_from_end(build([1, 2, 3]), 3)) == [2, 3]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Reorder List
# =============================================================================

def reorder_list(head):
    """
    Three patterns chained: find middle + reverse second half + interleave.

    STEP 1 — middle (fast/slow): with fast starting at head, slow lands on
    the middle for odd lengths and on the FIRST of the two middles for even
    lengths, so the first half is never shorter than the second.

    STEP 2 — cut and reverse: terminate the first half with None (critical:
    forgetting this creates a cycle), then reverse the second half.

    STEP 3 — interleave: alternately link one node from each half. The
    first half is >= the second, so it always ends the merge.

    Time: O(n), Space: O(1) — pointers only, no auxiliary container.
    """
    if not head or not head.next:
        return

    # Step 1: slow ends on the middle (first middle for even length)
    slow, fast = head, head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: cut after slow, reverse the second half
    second = slow.next
    slow.next = None                    # Terminate first half — avoids a cycle
    prev = None
    while second:
        nxt = second.next
        second.next = prev
        prev = second
        second = nxt
    second = prev                       # Head of the reversed second half

    # Step 3: interleave first and second halves
    first = head
    while second:
        f_next, s_next = first.next, second.next
        first.next = second
        second.next = f_next
        first, second = f_next, s_next


def test_exercise_6():
    print("\nExercise 6: Reorder List")

    lst = build([1, 2, 3, 4])
    reorder_list(lst)
    assert to_list(lst) == [1, 4, 2, 3]

    lst = build([1, 2, 3, 4, 5])
    reorder_list(lst)
    assert to_list(lst) == [1, 5, 2, 4, 3]

    lst = build([1])
    reorder_list(lst)
    assert to_list(lst) == [1]

    lst = build([1, 2])
    reorder_list(lst)
    assert to_list(lst) == [1, 2]

    lst = build([1, 2, 3])
    reorder_list(lst)
    assert to_list(lst) == [1, 3, 2]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 7 (Hard): Merge K Sorted Lists
# =============================================================================

def merge_k_lists(lists):
    """
    Min-heap of at most k nodes.

    APPROACH:
    - Push the head of every non-empty list into a min-heap.
    - Pop the global minimum, append it to the result, push its successor.
    - Every node enters and leaves the heap exactly once: O(N log k).

    TIE-BREAKER TRAP:
    - heapq compares tuple elements left to right. With equal values it
      would try to compare ListNode objects → TypeError. A monotonically
      increasing counter as second element breaks ties safely.

    WHY NOT merge one-by-one (O(N*k)):
    - Merging list i into the accumulated result re-walks all previously
      merged nodes; the first list's nodes are traversed ~k times.

    Time: O(N log k), Space: O(k) for the heap.
    """
    import heapq

    heap = []
    counter = 0                         # Tie-breaker for equal values
    for node in lists:
        if node:                        # Skip empty (None) lists
            heap.append((node.val, counter, node))
            counter += 1
    heapq.heapify(heap)

    dummy = ListNode(0)
    tail = dummy
    while heap:
        _, _, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1

    tail.next = None                    # Detach any leftover chain
    return dummy.next


def test_exercise_7():
    print("\nExercise 7: Merge K Sorted Lists")

    result = merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])
    assert to_list(result) == [1, 1, 2, 3, 4, 4, 5, 6]

    assert merge_k_lists([]) is None
    assert merge_k_lists([None]) is None
    assert to_list(merge_k_lists([None, build([1]), None])) == [1]
    assert to_list(merge_k_lists([build([5]), build([1]), build([3])])) == [1, 3, 5]

    result = merge_k_lists([build([2, 2]), build([2, 2]), build([2])])
    assert to_list(result) == [2, 2, 2, 2, 2]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): Sort List (merge sort on pointers)
# =============================================================================

def sort_list(head):
    """
    Top-down merge sort on a linked list — O(n log n) time, O(log n) stack.

    WHY MERGE SORT (not quicksort):
    - Merging two sorted linked lists is O(1) space (relink pointers),
      while quicksort's partition degrades to O(n^2) on sorted input
      and random pivot access is O(n) on a list.

    SPLIT TRAP:
    - fast starts at head.next (not head) so that slow stops BEFORE the
      middle. With fast = head, a 2-node list splits into (2, 0) and the
      recursion never terminates.
    """
    if not head or not head.next:
        return head                     # 0 or 1 node: already sorted

    # Split: slow stops before the middle thanks to fast = head.next
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None                    # Cut the list in two

    left = sort_list(head)
    right = sort_list(mid)

    # Merge two sorted lists (same pattern as easy exercise 3)
    dummy = ListNode(0)
    tail = dummy
    while left and right:
        if left.val <= right.val:
            tail.next, left = left, left.next
        else:
            tail.next, right = right, right.next
        tail = tail.next
    tail.next = left if left else right
    return dummy.next


def test_exercise_8():
    print("\nExercise 8: Sort List")

    assert to_list(sort_list(build([4, 2, 1, 3]))) == [1, 2, 3, 4]
    assert to_list(sort_list(build([-1, 5, 3, 4, 0]))) == [-1, 0, 3, 4, 5]
    assert to_list(sort_list(build([]))) == []
    assert to_list(sort_list(build([1]))) == [1]
    assert to_list(sort_list(build([2, 1]))) == [1, 2]
    assert to_list(sort_list(build([1, 1, 1]))) == [1, 1, 1]
    assert to_list(sort_list(build([5, 4, 3, 2, 1]))) == [1, 2, 3, 4, 5]

    # Random cross-check against sorted() as oracle
    import random
    for _ in range(20):
        values = [random.randint(-100, 100) for _ in range(random.randint(0, 50))]
        assert to_list(sort_list(build(values))) == sorted(values)

    # Benchmark: time should roughly double (x~2.2) when n doubles — n log n
    import time
    print("  Benchmark (random values):")
    prev = None
    for n in [1000, 2000, 4000, 8000]:
        values = [random.randint(0, 1_000_000) for _ in range(n)]
        lst = build(values)
        start = time.perf_counter()
        result = sort_list(lst)
        elapsed = time.perf_counter() - start
        assert to_list(result) == sorted(values)
        ratio = f" (x{elapsed / prev:.1f})" if prev else ""
        print(f"    n={n:>5,}: {elapsed:.5f}s{ratio}")
        prev = elapsed

    print("  PASS — all test cases + oracle + benchmark")


# =============================================================================
# RUN ALL
# =============================================================================

if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    test_exercise_6()
    test_exercise_7()
    test_exercise_8()
    print("\nAll Day 5 exercise solutions passed.")

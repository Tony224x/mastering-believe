"""
Solutions — Day 4: Stacks & Queues (HARD)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/04-stacks-queues-hard.py

Each solution is numbered to match the exercise file (03-hard/04-stacks-queues.md).
All solutions are verified with assertions at the end.
"""

from collections import deque


# =============================================================================
# EXERCISE 7 (Hard): Sliding Window Maximum — Monotonic Deque
# =============================================================================

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    Monotonic DECREASING deque of indices.

    INVARIANT:
    - dq holds indices whose values are decreasing from front to back.
    - dq[0] is always the index of the current window's maximum.

    EACH STEP:
    1. Pop from the BACK every index whose value <= nums[i]: those can never be
       the max again (nums[i] is newer and >=).
    2. Append i.
    3. Pop from the FRONT the index that has slid out of the window (<= i - k).
    4. Once the first window is complete (i >= k-1), record nums[dq[0]].

    WHY O(n):
    - Each index is appended once and removed once -> 2n deque operations total.

    Time: O(n), Space: O(k)
    """
    if not nums or k <= 0:
        return []

    dq = deque()                   # Indices, values decreasing front -> back
    result = []

    for i, val in enumerate(nums):
        # 1. Drop dominated indices from the back
        while dq and nums[dq[-1]] <= val:
            dq.pop()
        dq.append(i)

        # 3. Drop the front if it left the window
        if dq[0] <= i - k:
            dq.popleft()

        # 4. Window of size k is complete starting at i = k - 1
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


def test_exercise_7():
    print("\nExercise 7: Sliding Window Maximum")

    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([9, 8, 7, 6], 2) == [9, 8, 7]
    assert max_sliding_window([1, 2, 3, 4], 2) == [2, 3, 4]
    assert max_sliding_window([4, 4, 4, 4], 2) == [4, 4, 4]
    assert max_sliding_window([1, 3, 1, 2, 0, 5], 3) == [3, 3, 2, 5]
    assert max_sliding_window([7, 2, 4], 2) == [7, 4]
    assert max_sliding_window([1, -1], 1) == [1, -1]

    # Cross-check against a brute-force oracle on random-ish inputs
    def brute(nums, k):
        return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]

    samples = [
        ([5, 3, 8, 1, 9, 2, 7], 1),
        ([5, 3, 8, 1, 9, 2, 7], 3),
        ([5, 3, 8, 1, 9, 2, 7], 7),
        ([-7, -8, -1, -3], 2),
    ]
    for nums, k in samples:
        assert max_sliding_window(nums, k) == brute(nums, k)

    print("  PASS — all test cases (incl. brute-force cross-check)")


# =============================================================================
# EXERCISE 8 (Hard): Basic Calculator (+ - parentheses) — Stack of context
# =============================================================================

def calculate(s: str) -> int:
    """
    Evaluate +, -, ( ) on non-negative integers with a single pass and a stack.

    STATE CARRIED:
    - result: running total of the current parenthesis level
    - number: the integer currently being parsed
    - sign:   +1 or -1, the sign that applies to the NEXT number/parenthesis

    ON '(':  push (result, sign), then reset result=0, sign=+1 to start fresh.
    ON ')':  finalize the current number, then pop (prev_result, prev_sign):
             result = prev_result + prev_sign * result.

    Time: O(n), Space: O(n) (stack depth = parenthesis nesting).
    """
    result = 0
    number = 0
    sign = 1
    stack = []

    for c in s:
        if c.isdigit():
            number = number * 10 + int(c)      # Multi-digit numbers
        elif c == '+':
            result += sign * number
            number = 0
            sign = 1
        elif c == '-':
            result += sign * number
            number = 0
            sign = -1
        elif c == '(':
            # Save the context, then start a brand-new sub-expression
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif c == ')':
            result += sign * number            # Finalize last number in the block
            number = 0
            prev_sign = stack.pop()
            prev_result = stack.pop()
            result = prev_result + prev_sign * result
        # spaces are ignored

    return result + sign * number              # Add the trailing number, if any


def test_exercise_8():
    print("\nExercise 8: Basic Calculator")

    assert calculate("1 + 1") == 2
    assert calculate(" 2-1 + 2 ") == 3
    assert calculate("(1+(4+5+2)-3)+(6+8)") == 23
    assert calculate("- (3 + (4 + 5))") == -12
    assert calculate("2147483647") == 2147483647
    assert calculate("1-(     -2)") == 3
    assert calculate("(1)") == 1
    assert calculate("10 - (2 + 3) - 1") == 4
    assert calculate("0") == 0

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Trapping Rain Water — Monotonic Stack
# =============================================================================

def trap(height: list[int]) -> int:
    """
    Monotonic DECREASING stack of indices, accumulating water layer by layer.

    IDEA:
    - When height[i] rises above the top of the stack, we found a basin.
    - Pop 'bottom'. The new top is 'left' (left wall), and i is 'right' (right wall).
    - Bounded water height = min(height[left], height[i]) - height[bottom].
    - Width between walls (exclusive) = i - left - 1.
    - Add bounded_height * width.

    Time: O(n), Space: O(n).
    """
    stack = []                     # Indices, heights decreasing bottom -> top
    water = 0

    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break              # No left wall -> water spills out
            left = stack[-1]
            bounded = min(height[left], h) - height[bottom]
            width = i - left - 1
            water += bounded * width
        stack.append(i)

    return water


def trap_two_pointers(height: list[int]) -> int:
    """
    Alternative O(1)-space two-pointer solution, used here as a cross-check oracle.

    Time: O(n), Space: O(1).
    """
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max <= right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    return water


def test_exercise_9():
    print("\nExercise 9: Trapping Rain Water")

    cases = [
        ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        ([4, 2, 0, 3, 2, 5], 9),
        ([], 0),
        ([1, 2, 3], 0),
        ([3, 2, 1], 0),
        ([5], 0),
        ([2, 0, 2], 2),
        ([5, 4, 1, 2], 1),
    ]
    for arr, expected in cases:
        assert trap(arr) == expected, f"{arr}: got {trap(arr)}, want {expected}"
        # Cross-check the monotonic-stack answer against the two-pointer oracle
        assert trap(arr) == trap_two_pointers(arr)

    print("  PASS — all test cases (incl. two-pointer cross-check)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 4: Stacks & Queues (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

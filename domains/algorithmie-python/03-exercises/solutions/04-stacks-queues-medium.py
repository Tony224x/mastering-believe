"""
Solutions — Day 4: Stacks & Queues (MEDIUM)
Run: python domains/algorithmie-python/03-exercises/solutions/04-stacks-queues-medium.py

Each solution is numbered to match the exercise file (02-medium/04-stacks-queues.md).
All solutions are verified with assertions at the end.
"""


# =============================================================================
# EXERCISE 4 (Medium): Daily Temperatures — Monotonic Stack (indices)
# =============================================================================

def daily_temperatures(temperatures: list[int]) -> list[int]:
    """
    Monotonic DECREASING stack of indices.

    INVARIANT:
    - The stack holds indices whose temperatures are decreasing from bottom to
      top. These are the days still "waiting" for a warmer day.
    - When today is strictly warmer than the temperature at the top, we resolve
      that day: its answer is the distance (i - j).

    WHY INDICES, NOT VALUES:
    - We need the DISTANCE between days, so we must remember positions.

    WHY O(n) DESPITE THE NESTED WHILE:
    - Each index is pushed once and popped at most once -> 2n operations total.

    Time: O(n), Space: O(n)
    """
    n = len(temperatures)
    answer = [0] * n               # Default 0 = no warmer day in the future
    stack = []                     # Indices, decreasing temps from bottom to top

    for i, t in enumerate(temperatures):
        # Resolve every waiting day strictly colder than today
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j      # Days waited
        stack.append(i)

    return answer


def test_exercise_4():
    print("\nExercise 4: Daily Temperatures")

    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert daily_temperatures([60, 50, 40, 30]) == [0, 0, 0, 0]
    assert daily_temperatures([30]) == [0]
    assert daily_temperatures([30, 30, 30]) == [0, 0, 0]
    assert daily_temperatures([89, 62, 70, 58, 47, 47, 46, 76, 100, 70]) == [8, 1, 5, 4, 3, 2, 1, 1, 0, 0]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Decode String — Stack of context
# =============================================================================

def decode_string(s: str) -> str:
    """
    Stack of (previous_string, repeat_count) to handle nesting.

    APPROACH:
    - Walk char by char.
    - Digit: build the current multiplier (handle multi-digit numbers).
    - '[' : push (current_string, current_number), then reset both.
    - ']' : pop (prev_string, k); current_string = prev_string + k * current_string.
    - letter: append to current_string.

    Time: O(N) where N = length of the DECODED output (each char emitted once).
    Space: O(N) for the stack + output buffers.
    """
    num_stack = []                 # Repeat counts to apply on closing ']'
    str_stack = []                 # Strings accumulated before each '['
    current = []                   # Current segment as a list of chars (fast join)
    num = 0                        # Current multiplier being parsed

    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)        # Support multi-digit k, e.g. "10[a]"
        elif c == '[':
            num_stack.append(num)
            str_stack.append(current)
            current = []                   # Start a fresh segment for the block
            num = 0
        elif c == ']':
            k = num_stack.pop()
            prev = str_stack.pop()
            prev.extend(current * k)       # prev + (current repeated k times)
            current = prev
        else:
            current.append(c)              # Plain letter

    return ''.join(current)


def test_exercise_5():
    print("\nExercise 5: Decode String")

    assert decode_string("3[a]2[bc]") == "aaabcbc"
    assert decode_string("3[a2[c]]") == "accaccacc"
    assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"
    assert decode_string("abc") == "abc"
    assert decode_string("10[a]") == "aaaaaaaaaa"
    assert decode_string("2[3[a]b]") == "aaabaaab"
    assert decode_string("") == ""

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Min Stack — Auxiliary minimum stack
# =============================================================================

class MinStack:
    """
    Stack with O(1) get_min by tracking the running minimum at each level.

    KEY IDEA:
    - A second stack 'mins' stores, at each level, the minimum of the main stack
      up to and including that element.
    - push(val): also push min(val, current_min). For an empty stack, push val.
    - pop(): pop BOTH stacks so 'mins' stays aligned.
    - get_min(): just read the top of 'mins' — O(1).

    DUPLICATE HANDLING:
    - Because we push the running min unconditionally, duplicates of the minimum
      are tracked level by level, so popping one duplicate keeps the min correct.

    Time: O(1) for all operations. Space: O(n).
    """

    def __init__(self):
        self.stack = []
        self.mins = []             # mins[i] = min of stack[0..i]

    def push(self, val: int) -> None:
        self.stack.append(val)
        # Running minimum: either val or the previous min, whichever is smaller
        current_min = val if not self.mins else min(val, self.mins[-1])
        self.mins.append(current_min)

    def pop(self) -> None:
        self.stack.pop()
        self.mins.pop()            # Keep the two stacks in lockstep

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.mins[-1]


def test_exercise_6():
    print("\nExercise 6: Min Stack")

    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    assert ms.get_min() == -3
    ms.pop()
    assert ms.top() == 0
    assert ms.get_min() == -2

    ms = MinStack()
    ms.push(5)
    assert ms.get_min() == 5
    assert ms.top() == 5
    ms.push(5)
    assert ms.get_min() == 5
    ms.pop()
    assert ms.get_min() == 5

    ms = MinStack()
    ms.push(2)
    ms.push(0)
    ms.push(3)
    ms.push(0)
    assert ms.get_min() == 0
    ms.pop()
    assert ms.get_min() == 0
    ms.pop()
    assert ms.get_min() == 0
    ms.pop()
    assert ms.get_min() == 2

    print("  PASS — all test cases")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 4: Stacks & Queues (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

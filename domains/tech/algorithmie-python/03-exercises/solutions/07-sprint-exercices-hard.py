"""
Solutions — Day 7: Sprint Complementaire (HARD)
Run: python domains/tech/algorithmie-python/03-exercises/solutions/07-sprint-exercices-hard.py

New problems complementing the easy sprint P1-P10 (no duplication).
Matches the exercise file (03-hard/07-sprint-exercices.md).
All solutions are verified with assertions at the end.
"""

import heapq


# =============================================================================
# EXERCISE 7 (Hard): Evaluate Expression with precedence — Stack
# =============================================================================

def evaluate(s: str) -> int:
    """
    Evaluate + - * / with precedence using a single stack (no parentheses).

    KEY IDEA:
    - Keep the LAST operator seen. When the next number is finalized:
        '+' -> push +num ; '-' -> push -num
        '*' -> pop, push popped * num ; '/' -> pop, push trunc(popped / num)
    - High-precedence ops act immediately on the stack top; low-precedence ops
      just push, deferring summation. The final answer is sum(stack).

    Time: O(n), Space: O(n)
    """
    stack = []
    num = 0
    op = '+'                                       # Implicit leading '+'

    for i, c in enumerate(s):
        if c.isdigit():
            num = num * 10 + int(c)
        if (not c.isdigit() and c != ' ') or i == len(s) - 1:
            # Flush at any operator or at the very end of the string
            if op == '+':
                stack.append(num)
            elif op == '-':
                stack.append(-num)
            elif op == '*':
                stack.append(stack.pop() * num)
            elif op == '/':
                stack.append(int(stack.pop() / num))   # Truncate toward zero
            op = c
            num = 0

    return sum(stack)


def test_exercise_7():
    print("\nExercise 7: Evaluate Expression with precedence")

    assert evaluate("3+2*2") == 7
    assert evaluate(" 3/2 ") == 1
    assert evaluate(" 3+5 / 2 ") == 5
    assert evaluate("2*3+4") == 10
    assert evaluate("14-3/2") == 13
    assert evaluate("1") == 1
    assert evaluate("2*2*2*2") == 16
    assert evaluate("100/3/3") == 11
    assert evaluate("0") == 0

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): Find Median from Data Stream — Two heaps
# =============================================================================

class MedianFinder:
    """
    Two balanced heaps:
    - low:  max-heap (store negatives) holding the smaller half
    - high: min-heap holding the larger half

    INVARIANT: len(low) == len(high) OR len(low) == len(high) + 1.

    add_num: O(log n) ; find_median: O(1).
    """

    def __init__(self):
        self.low = []                              # Max-heap via negation
        self.high = []                             # Min-heap

    def add_num(self, num: int) -> None:
        # Always push to low first, then move the max of low into high
        heapq.heappush(self.low, -num)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        # Rebalance so low holds the extra element when sizes differ
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def find_median(self) -> float:
        if len(self.low) > len(self.high):
            return float(-self.low[0])             # Odd count: top of low
        return (-self.low[0] + self.high[0]) / 2   # Even count: average of tops


def test_exercise_8():
    print("\nExercise 8: Find Median from Data Stream")

    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0

    mf = MedianFinder()
    mf.add_num(5)
    assert mf.find_median() == 5.0

    mf = MedianFinder()
    for x in [6, 10, 2, 6, 5, 0]:
        mf.add_num(x)
    assert mf.find_median() == 5.5

    mf = MedianFinder()
    for x in [-1, -2, -3]:
        mf.add_num(x)
    assert mf.find_median() == -2.0

    # Cross-check against statistics.median on a random stream
    import random
    import statistics
    rng = random.Random(7)
    mf = MedianFinder()
    seen = []
    for _ in range(200):
        x = rng.randint(-50, 50)
        mf.add_num(x)
        seen.append(x)
        assert mf.find_median() == statistics.median(seen)

    print("  PASS — all test cases (incl. randomized cross-check)")


# =============================================================================
# EXERCISE 9 (Hard): Longest Increasing Subsequence — Patience + binary search
# =============================================================================

def length_of_lis(nums: list[int]) -> int:
    """
    Patience sorting: tails[i] = smallest possible tail of an increasing
    subsequence of length i+1. For each num, bisect_left its position:
    - position == len(tails): extend (new longest)
    - else: replace tails[pos] (greedy: keep tails as small as possible)

    Strictly increasing -> bisect_left (a duplicate replaces, never extends).

    Time: O(n log n), Space: O(n)
    """
    from bisect import bisect_left

    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)                      # Extend the LIS
        else:
            tails[pos] = num                       # Tighten an existing length
    return len(tails)


def test_exercise_9():
    print("\nExercise 9: Longest Increasing Subsequence")

    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
    assert length_of_lis([7, 7, 7, 7]) == 1
    assert length_of_lis([1]) == 1
    assert length_of_lis([]) == 0
    assert length_of_lis([4, 10, 4, 3, 8, 9]) == 3
    assert length_of_lis([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6

    # Cross-check against an O(n^2) DP oracle
    def lis_dp(nums):
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    import random
    rng = random.Random(11)
    for _ in range(50):
        arr = [rng.randint(0, 20) for _ in range(rng.randint(0, 15))]
        assert length_of_lis(arr) == lis_dp(arr)

    print("  PASS — all test cases (incl. O(n^2) DP cross-check)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 7: Sprint Complementaire (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

"""
Solutions — Day 14 Mock Interviews (MEDIUM).
Run: python domains/algorithmie-python/03-exercises/solutions/14-mock-interviews-medium.py

Each solution maps to the exercise file (02-medium/14-mock-interviews.md):
  Exercise 4 -> Group Anagrams           (hashing / signature grouping)
  Exercise 5 -> Product of Array Except Self (prefix/suffix, O(1) extra)
  Exercise 6 -> Coin Change              (bottom-up DP)

These are mock-interview debriefs: the "why" (the pivot away from the naive
solution, the chosen complexity) matters as much as the code. All solutions are
stdlib-only and verified with assertions, including the cases the exercise's
auto-eval grid singles out (empty input, zeros, impossible amount...).
"""

from collections import defaultdict


# =============================================================================
# EXERCISE 4 (Medium): Group Anagrams — signature as a dict key
# =============================================================================

def group_anagrams(strs):
    """
    Two words are anagrams iff they share the same multiset of letters. So we
    map each word to a canonical SIGNATURE and bucket by it.

    Naive (verbalised in the interview): compare every pair by sorting both ->
    O(n^2 * k log k). We pivot to a single pass keyed by signature.

    Signature choice: a 26-count tuple. That is O(k) per word (k = word length),
    beating the "sorted string" signature which is O(k log k). Both are valid;
    the count tuple is the optimal one to mention.

    Time : O(n * k)   (one pass, O(k) signature per word)
    Space: O(n * k)   (the buckets hold every input string)
    """
    buckets = defaultdict(list)
    for word in strs:
        counts = [0] * 26                  # a..z frequency vector
        for ch in word:
            counts[ord(ch) - ord("a")] += 1
        buckets[tuple(counts)].append(word)  # tuple is hashable -> dict key
    return list(buckets.values())


def test_exercise_4():
    print("\nExercise 4: Group Anagrams")

    def normalize(groups):
        # Order of groups and within a group is irrelevant -> canonicalise.
        return sorted(sorted(g) for g in groups)

    assert normalize(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])) == \
        normalize([["bat"], ["nat", "tan"], ["ate", "eat", "tea"]])
    assert group_anagrams([""]) == [[""]]          # single empty string
    assert group_anagrams(["a"]) == [["a"]]        # single word
    assert normalize(group_anagrams([])) == []     # empty input
    assert normalize(group_anagrams(["abc", "bca", "cab", "xyz"])) == \
        normalize([["abc", "bca", "cab"], ["xyz"]])
    # Duplicates land in the same bucket (kept, not deduped).
    assert normalize(group_anagrams(["ab", "ba", "ab"])) == \
        normalize([["ab", "ba", "ab"]])

    print("  PASS — all test cases (incl. [], [\"\"], single word, duplicates)")


# =============================================================================
# EXERCISE 5 (Medium): Product of Array Except Self — prefix/suffix, no division
# =============================================================================

def product_except_self(nums):
    """
    The trap is division: answer[i] = total_product / nums[i] is O(n) but BREAKS
    on zeros (division by zero, and a single zero already makes the formula
    ambiguous for the other indices). So division is off the table.

    Pivot: answer[i] = (product of everything LEFT of i) * (product of everything
    RIGHT of i). Two passes:
      1) left-to-right, accumulating the prefix product INTO the output.
      2) right-to-left, multiplying each slot by the running suffix product.

    The output array doubles as the accumulator, so no extra array is needed.

    Time : O(n)
    Space: O(1) extra (the output is not counted as extra space)
    """
    n = len(nums)
    answer = [1] * n

    prefix = 1
    for i in range(n):
        answer[i] = prefix                 # product of all elements before i
        prefix *= nums[i]

    suffix = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix                # fold in product of all after i
        suffix *= nums[i]

    return answer


def test_exercise_5():
    print("\nExercise 5: Product of Array Except Self")

    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    assert product_except_self([0, 1, 2]) == [2, 0, 0]      # one zero
    assert product_except_self([0, 0, 3]) == [0, 0, 0]      # two zeros
    assert product_except_self([2, 3]) == [3, 2]
    assert product_except_self([5]) == [1]                  # single element

    # The output is the only allocation (O(1) extra confirmed by construction);
    # the input is never mutated.
    src = [1, 2, 3, 4]
    _ = product_except_self(src)
    assert src == [1, 2, 3, 4]

    print("  PASS — all test cases (incl. 1 zero, 2 zeros, single element)")


# =============================================================================
# EXERCISE 6 (Medium): Coin Change — bottom-up DP
# =============================================================================

def coin_change(coins, amount):
    """
    Recognise the DP: the min coins for `amount` is 1 + the best of (min coins
    for amount - c) over every coin c. Optimal substructure + overlapping
    subproblems -> bottom-up table.

    Naive brute force (verbalised): branch on every coin recursively ->
    exponential. DP collapses the shared subproblems.

    dp[x] = fewest coins to make x.
      dp[0] = 0                      (zero amount needs zero coins)
      dp[x] = 1 + min(dp[x - c] for c in coins if c <= x)
    Unreachable amounts stay at +inf; we report -1 if the target is unreachable.

    Time : O(amount * len(coins))
    Space: O(amount)
    """
    INF = float("inf")
    dp = [0] + [INF] * amount              # dp[0]=0, rest = unreachable
    for x in range(1, amount + 1):
        for c in coins:
            if c <= x and dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
    return dp[amount] if dp[amount] != INF else -1


def test_exercise_6():
    print("\nExercise 6: Coin Change")

    assert coin_change([1, 2, 5], 11) == 3        # 5 + 5 + 1
    assert coin_change([2], 3) == -1              # odd target, even coin
    assert coin_change([1], 0) == 0              # zero amount -> zero coins
    assert coin_change([1], 2) == 2
    assert coin_change([2, 5, 10, 1], 27) == 4   # 10 + 10 + 5 + 2
    assert coin_change([186, 419, 83, 408], 6249) == 20
    assert coin_change([5], 3) == -1

    print("  PASS — all test cases (incl. amount=0 and impossible -> -1)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 14: Mock Interviews (MEDIUM)")
    print("=" * 70)

    test_exercise_4()
    test_exercise_5()
    test_exercise_6()

    print("\n" + "=" * 70)
    print("ALL MEDIUM SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

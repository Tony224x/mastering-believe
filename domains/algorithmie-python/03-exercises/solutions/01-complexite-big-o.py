"""
Solutions — Day 1: Complexite & Big-O
Run: python domains/algorithmie-python/03-exercises/solutions/01-complexite-big-o.py

Each solution is numbered to match the exercise file.
"""

import time
import heapq
from collections import defaultdict, Counter


# =============================================================================
# EXERCISE 1: Identifier la complexite
# =============================================================================

def exercise_1_answers():
    """
    Fonction A: sum of array
      Time: O(n)   — single loop over n elements
      Space: O(1)  — only one variable (total)

    Fonction B: has duplicate (brute force)
      Time: O(n^2)  — nested loops, both iterate over arr
      Space: O(1)   — no extra data structure

    Fonction C: double until >= n
      Time: O(log n) — i doubles each iteration, reaches n in log2(n) steps
      Space: O(1)    — single variable

    Fonction D: sorted(set(arr))
      Time: O(n log n) — set() is O(n), sorted() is O(n log n), total = O(n log n)
      Space: O(n)      — set creates a copy of up to n elements

    Fonction E: string concatenation in loop (THE TRAP)
      Time: O(n^2) — each += copies the growing string (length 1, 2, 3, ... n)
                      total copies = 1+2+3+...+n = n(n+1)/2 = O(n^2)
      Space: O(n)  — the result string grows to length n
    """
    print("Exercise 1: See docstring above for answers")
    print("  Key trap: Fonction E looks O(n) but is O(n^2) due to string immutability")


# =============================================================================
# EXERCISE 2: Classer par complexite
# =============================================================================

def exercise_2_answers():
    """
    Correct order (smallest to largest as n → ∞):

    1. O(1)          — constant
    2. O(log n)      — logarithmic
    3. O(sqrt(n))    — sub-linear (between log n and n)
    4. O(n)          — linear
    5. O(100n)       — linear (same class as O(n), constant factor ignored)
    6. O(n log n)    — linearithmic
    7. O(n^2)        — quadratic
    8. O(n^3)        — cubic
    9. O(2^n)        — exponential
    10. O(n!)        — factorial

    Note: O(n) and O(100n) are the SAME Big-O class. In practice 100n is slower,
    but Big-O ignores constant factors.

    Ratios at n = 1000:
      log(1000) ≈ 10     vs  sqrt(1000) ≈ 31.6    → ~3x
      sqrt(1000) ≈ 31.6  vs  1000                  → ~31x
      1000               vs  1000*10 = 10,000       → 10x
      10,000             vs  1,000,000               → 100x
      1,000,000          vs  1,000,000,000           → 1000x
    """
    import math
    n = 1000
    values = {
        "1":        1,
        "log n":    math.log2(n),
        "sqrt(n)":  math.sqrt(n),
        "n":        n,
        "100n":     100 * n,
        "n log n":  n * math.log2(n),
        "n^2":      n ** 2,
        "n^3":      n ** 3,
        "2^n":      2 ** n,  # Astronomically large
    }

    print("Exercise 2: Values at n=1000")
    for name, val in values.items():
        if val < 1e15:
            print(f"  {name:>10} = {val:,.0f}")
        else:
            print(f"  {name:>10} = {val:.2e}")


# =============================================================================
# EXERCISE 3: Optimise le lookup
# =============================================================================

def common_elements_brute(list1, list2):
    """O(n*m) — linear scan of list2 for each element in list1."""
    result = []
    for x in list1:
        if x in list2:          # O(m) per lookup
            result.append(x)
    return result


def common_elements_optimized(list1, list2):
    """
    O(n + m) — convert the smaller list to a set for O(1) lookups.

    Steps:
    1. Build a set from list2: O(m) time, O(m) space
    2. Iterate list1, check membership in set: O(n) * O(1) = O(n)
    3. Use a 'seen' set to avoid duplicates in result: O(1) per check
    Total: O(n + m) time, O(min(n, m)) space for the set
    """
    set2 = set(list2)           # O(m) to build — converts O(m) lookups to O(1) each
    seen = set()                # Track what we've already added to avoid duplicates
    result = []
    for x in list1:             # O(n) iterations
        if x in set2 and x not in seen:  # Both O(1) lookups
            result.append(x)
            seen.add(x)
    return result


def exercise_3_demo():
    print("\nExercise 3: Common elements optimization")
    list1 = list(range(10_000))
    list2 = list(range(5_000, 15_000))

    start = time.perf_counter()
    r1 = common_elements_brute(list1, list2)
    brute_time = time.perf_counter() - start

    start = time.perf_counter()
    r2 = common_elements_optimized(list1, list2)
    opt_time = time.perf_counter() - start

    assert sorted(r1) == sorted(r2), "Results should match"
    print(f"  Brute force: {brute_time:.4f}s")
    print(f"  Optimized:   {opt_time:.6f}s")
    print(f"  Speedup:     {brute_time / max(opt_time, 1e-9):.0f}x")


# =============================================================================
# EXERCISE 4: Analyse de complexite recursive
# =============================================================================

def exercise_4_answers():
    """
    Fonction F: func_f(n) = 2 * func_f(n-1)
      Time: O(2^n)
        - Each call spawns 2 sub-calls, depth = n
        - Tree has 2^0 + 2^1 + ... + 2^n = 2^(n+1) - 1 nodes
      Space: O(n)
        - Max depth of call stack = n

    Fonction G: func_g(n) = func_g(n-1) + func_g(n-2)  [Fibonacci pattern]
      Time: O(2^n) — technically O(phi^n) where phi ≈ 1.618
        - Same class as F (exponential), but with a smaller base
        - F's tree is a PERFECT binary tree (every node has 2 children)
        - G's tree is UNBALANCED (right subtree is smaller by 1 level)
        - Both are O(2^n) in Big-O, but F is ~2x faster in practice
      Space: O(n) — call stack depth = n

    Fonction H: divide-and-conquer on array (merge sort pattern)
      Time: O(n log n)
        - Splits array in half each time → log n levels
        - At each level, total work = O(n) (merge step touches all elements)
        - Total: O(n) per level * O(log n) levels = O(n log n)
      Space: O(n) for merge buffer + O(log n) for call stack = O(n)

    Call tree for F with n=4:
                    f(4)
                /         \\
            f(3)           f(3)
           /    \\         /    \\
        f(2)   f(2)    f(2)   f(2)
        / \\   / \\    / \\   / \\
      f(1) f(1) ...  ...  ...  ...
      = 2^4 - 1 = 15 calls
    """
    print("\nExercise 4: See docstring for full analysis")
    print("  F: O(2^n) time, O(n) space — perfect binary tree of calls")
    print("  G: O(2^n) time (O(phi^n) precise), O(n) space — Fibonacci tree")
    print("  H: O(n log n) time, O(n) space — merge sort pattern")


# =============================================================================
# EXERCISE 5: Profiler et optimiser
# =============================================================================

def find_duplicates_brute(arr):
    """O(n^2) — compare every pair."""
    duplicates = []
    for i in range(len(arr)):
        if arr[i] in duplicates:        # Skip if already found
            continue
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
                break                   # Found one duplicate, move on
    return duplicates


def find_duplicates_optimized(arr):
    """O(n) — count occurrences with a dict, filter those > 1."""
    counts = Counter(arr)               # O(n) — single pass to count
    return [x for x, c in counts.items() if c > 1]  # O(n) — single pass to filter


def exercise_5_demo():
    print("\nExercise 5: Profiling duplicates detection")
    print(f"  {'n':>8} | {'Brute O(n^2)':>14} | {'Dict O(n)':>14} | {'Ratio':>8}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")

    import random
    for n in [1_000, 5_000, 10_000, 20_000]:
        # Create array with ~10% duplicates
        arr = list(range(n)) + random.sample(range(n), n // 10)
        random.shuffle(arr)

        start = time.perf_counter()
        r1 = find_duplicates_brute(arr)
        brute_time = time.perf_counter() - start

        start = time.perf_counter()
        r2 = find_duplicates_optimized(arr)
        opt_time = time.perf_counter() - start

        assert set(r1) == set(r2), f"Results differ at n={n}"
        print(f"  {n:>8,} | {brute_time:>13.4f}s | {opt_time:>13.6f}s | {brute_time/max(opt_time,1e-9):>7.0f}x")

    print("\n  Observation: when n doubles, brute force time ~4x (confirms O(n^2))")
    print("  Trade-off: O(n) extra space (Counter dict) → O(n^2) → O(n) time")


# =============================================================================
# EXERCISE 6: Piege des operations cachees
# =============================================================================

def process_logs_broken(logs: list) -> dict:
    """Original broken version — O(n^2) due to hidden costs."""
    result = {}
    all_users = []                          # BUG: list instead of set

    for log in logs:
        parts = log.split(":")
        user, action = parts[0], parts[1]

        if user not in all_users:           # O(k) where k = unique users — scans entire list
            all_users.append(user)

        if user not in result:
            result[user] = ""
        result[user] = result[user] + action + ","  # O(len) — copies growing string each time

    final = {}
    for user in all_users:
        actions = result[user].split(",")
        actions = [a for a in actions if a]
        if len(actions) > 1:
            final[user] = actions

    return final


def process_logs_fixed(logs: list) -> dict:
    """
    Fixed version — O(n) time, O(n) space.

    Fixes:
    1. Use defaultdict(list) instead of string concatenation
       → Appending to a list is O(1) amortized (vs O(k) string copy)
    2. Use a set for tracking unique users (not needed here since dict handles it)
       → O(1) membership test (vs O(k) list scan)
    """
    actions_by_user = defaultdict(list)     # O(1) access + O(1) append

    for log in logs:                        # O(n) iterations
        user, action = log.split(":", 1)    # split with maxsplit=1 for safety
        actions_by_user[user].append(action)  # O(1) amortized

    # Filter users with > 1 action — O(n) total across all users
    return {
        user: actions
        for user, actions in actions_by_user.items()
        if len(actions) > 1                 # O(1) — len() is constant time
    }


def exercise_6_demo():
    print("\nExercise 6: Hidden operations trap")
    import random

    n = 20_000
    users = [f"user{i}" for i in range(100)]  # 100 unique users
    actions = ["login", "click", "scroll", "logout", "purchase"]
    logs = [f"{random.choice(users)}:{random.choice(actions)}" for _ in range(n)]

    start = time.perf_counter()
    r1 = process_logs_broken(logs)
    broken_time = time.perf_counter() - start

    start = time.perf_counter()
    r2 = process_logs_fixed(logs)
    fixed_time = time.perf_counter() - start

    # Verify same results
    assert set(r1.keys()) == set(r2.keys()), "Different users in results"
    for user in r1:
        assert sorted(r1[user]) == sorted(r2[user]), f"Different actions for {user}"

    print(f"  Broken version ({n:,} logs): {broken_time:.4f}s")
    print(f"  Fixed version  ({n:,} logs): {fixed_time:.6f}s")
    print(f"  Speedup: {broken_time / max(fixed_time, 1e-9):.0f}x")
    print("\n  Problems found:")
    print("    1. `user not in all_users` — O(k) scan on list, should use set or dict")
    print("    2. `result[user] + action + ','` — O(k) string copy, should use list.append")


# =============================================================================
# EXERCISE 7: Master Theorem
# =============================================================================

def exercise_7_answers():
    """
    Master Theorem: T(n) = aT(n/b) + O(n^d)
    Compare log_b(a) vs d:
      - If log_b(a) > d  → O(n^(log_b(a)))     [recursion dominates]
      - If log_b(a) == d → O(n^d * log n)       [balanced]
      - If log_b(a) < d  → O(n^d)               [work-per-level dominates]

    1. T(n) = 2T(n/2) + O(n)
       a=2, b=2, d=1 → log_2(2) = 1 == d → O(n log n)  ✓ (merge sort)

    2. T(n) = 2T(n/2) + O(1)
       a=2, b=2, d=0 → log_2(2) = 1 > 0 → O(n^1) = O(n)  ✓ (tree traversal)

    3. T(n) = T(n/2) + O(n)
       a=1, b=2, d=1 → log_2(1) = 0 < 1 → O(n^1) = O(n)  ✓ (work dominates)
       Intuition: n + n/2 + n/4 + ... = 2n = O(n)

    4. T(n) = 3T(n/3) + O(n)
       a=3, b=3, d=1 → log_3(3) = 1 == d → O(n log n)  (same as merge sort!)

    5. T(n) = 2T(n/2) + O(n^2)
       a=2, b=2, d=2 → log_2(2) = 1 < 2 → O(n^2)  ✓ (merge step too expensive)
    """
    import math

    print("\nExercise 7: Master Theorem results")
    cases = [
        (2, 2, 1, "O(n log n)", "merge sort"),
        (2, 2, 0, "O(n)",       "tree traversal"),
        (1, 2, 1, "O(n)",       "work dominates"),
        (3, 3, 1, "O(n log n)", "3-way merge sort"),
        (2, 2, 2, "O(n^2)",     "expensive merge"),
    ]

    for i, (a, b, d, expected, desc) in enumerate(cases, 1):
        log_ba = math.log(a) / math.log(b) if a > 0 else 0
        if log_ba > d:
            result = f"O(n^{log_ba:.1f})"
            case_type = "recursion dominates"
        elif abs(log_ba - d) < 0.01:
            result = f"O(n^{d} log n)" if d > 0 else "O(log n)"
            case_type = "balanced"
        else:
            result = f"O(n^{d})"
            case_type = "work dominates"
        print(f"  {i}. a={a}, b={b}, d={d} → log_b(a)={log_ba:.1f}, "
              f"case: {case_type} → {expected} ({desc})")

    # Empirical verification for case 1: merge sort pattern
    print("\n  Empirical verification — case 1 (merge sort pattern):")

    def merge_sort_count(arr):
        """Returns (sorted_arr, operation_count)."""
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, ops_l = merge_sort_count(arr[:mid])
        right, ops_r = merge_sort_count(arr[mid:])
        # Merge step: O(n) work
        merged = []
        i = j = 0
        ops = ops_l + ops_r
        while i < len(left) and j < len(right):
            ops += 1
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        ops += len(left) - i + len(right) - j
        return merged, ops

    import random
    for n in [1000, 2000, 4000, 8000]:
        arr = list(range(n))
        random.shuffle(arr)
        _, ops = merge_sort_count(arr)
        ratio = ops / (n * math.log2(n))  # Should be roughly constant if O(n log n)
        print(f"    n={n:>5,}: ops={ops:>10,}, ops/(n*log2(n))={ratio:.2f}")


# =============================================================================
# EXERCISE 8: MedianFinder with two heaps
# =============================================================================

class MedianFinder:
    """
    Find median from a data stream using two heaps.

    Invariants:
    1. max_heap stores the SMALLER half (negated for Python's min-heap)
    2. min_heap stores the LARGER half
    3. len(max_heap) == len(min_heap) OR len(max_heap) == len(min_heap) + 1
    4. max of smaller half <= min of larger half

    Why two heaps?
    - Naive (sort each time): add_num = O(n log n) — too slow
    - Sorted list + bisect.insort: add_num = O(n) — insertion shifts elements
    - Two heaps: add_num = O(log n), find_median = O(1) ✓
    """

    def __init__(self):
        self.max_heap = []  # Stores smaller half (NEGATED values for max-heap behavior)
        self.min_heap = []  # Stores larger half (normal min-heap)

    def add_num(self, num: int) -> None:
        """Add a number to the data structure. O(log n)."""
        # Step 1: Add to max_heap (smaller half) by default
        heapq.heappush(self.max_heap, -num)  # Negate for max-heap behavior

        # Step 2: Ensure max of smaller half <= min of larger half
        # If the newly added element is larger than the smallest in min_heap,
        # move it over
        if self.min_heap and -self.max_heap[0] > self.min_heap[0]:
            val = -heapq.heappop(self.max_heap)   # Pop max from smaller half
            heapq.heappush(self.min_heap, val)      # Push to larger half

        # Step 3: Rebalance sizes — max_heap can have at most 1 more than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        elif len(self.min_heap) > len(self.max_heap):
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)

    def find_median(self) -> float:
        """Return the median of all elements added so far. O(1)."""
        if not self.max_heap:
            raise ValueError("No numbers added yet")

        if len(self.max_heap) > len(self.min_heap):
            # Odd count: median is the top of max_heap
            return float(-self.max_heap[0])
        else:
            # Even count: median is average of both tops
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0


def exercise_8_demo():
    print("\nExercise 8: MedianFinder with two heaps")

    # --- Correctness test ---
    print("\n  Correctness test:")
    mf = MedianFinder()
    test_values = [5, 2, 8, 1, 9, 3, 7, 4, 6, 10]
    all_added = []

    for val in test_values:
        mf.add_num(val)
        all_added.append(val)
        expected = sorted(all_added)
        n = len(expected)
        if n % 2 == 1:
            expected_median = float(expected[n // 2])
        else:
            expected_median = (expected[n // 2 - 1] + expected[n // 2]) / 2.0

        actual_median = mf.find_median()
        status = "OK" if abs(actual_median - expected_median) < 1e-9 else "FAIL"
        print(f"    Added {val:>2}, stream={all_added}, "
              f"median={actual_median:.1f} (expected {expected_median:.1f}) [{status}]")

    # --- Performance test ---
    print("\n  Performance test (add_num should be O(log n)):")
    for n in [10_000, 50_000, 100_000, 200_000]:
        mf = MedianFinder()
        import random
        nums = [random.randint(0, 1_000_000) for _ in range(n)]

        start = time.perf_counter()
        for num in nums:
            mf.add_num(num)
        add_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(1000):
            mf.find_median()
        median_time = time.perf_counter() - start

        print(f"    n={n:>7,}: add_num total={add_time:.4f}s "
              f"({add_time/n*1e6:.1f}us/op), "
              f"find_median 1000x={median_time:.6f}s")

    # --- Edge cases ---
    print("\n  Edge cases:")
    mf = MedianFinder()

    # Empty stream
    try:
        mf.find_median()
        print("    Empty stream: FAIL (should raise)")
    except ValueError as e:
        print(f"    Empty stream: OK (raised ValueError: {e})")

    # Single element
    mf.add_num(42)
    assert mf.find_median() == 42.0
    print("    Single element: OK (median=42.0)")

    # Duplicates
    mf2 = MedianFinder()
    for x in [5, 5, 5, 5, 5]:
        mf2.add_num(x)
    assert mf2.find_median() == 5.0
    print("    All duplicates [5,5,5,5,5]: OK (median=5.0)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 1: Complexite & Big-O")
    print("=" * 70)

    exercise_1_answers()
    exercise_2_answers()
    exercise_3_demo()
    exercise_4_answers()
    exercise_5_demo()
    exercise_6_demo()
    exercise_7_answers()
    exercise_8_demo()

    print("\n" + "=" * 70)
    print("ALL SOLUTIONS COMPLETE")
    print("=" * 70)

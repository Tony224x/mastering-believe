"""
Day 1 — Complexity & Big-O: Runnable Examples
Run: python domains/algorithmie-python/02-code/01-complexite-big-o.py

Each section demonstrates a complexity class with a real function,
then times it so you can FEEL the difference.
"""

import time
import random
from collections import deque


# =============================================================================
# HELPER: Timing decorator to measure execution time
# =============================================================================

def timed(func):
    """Decorator that prints execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()          # perf_counter = highest resolution timer
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  {func.__name__}: {elapsed:.6f}s")
        return result
    return wrapper


# =============================================================================
# SECTION 1: Each complexity class — real functions
# =============================================================================

# --- O(1) — Constant time ---------------------------------------------------

@timed
def o1_dict_lookup(data: dict, key: str):
    """Dict lookup is O(1) average — hash table under the hood."""
    return data.get(key)  # Single hash computation + bucket access


# --- O(log n) — Logarithmic time --------------------------------------------

@timed
def o_logn_binary_search(arr: list, target: int) -> int:
    """Classic binary search on a sorted array. O(log n)."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2         # Split search space in half each time
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1              # Discard left half
        else:
            hi = mid - 1              # Discard right half
    return -1


# --- O(n) — Linear time -----------------------------------------------------

@timed
def o_n_find_max(arr: list) -> int:
    """Single pass through the array. O(n)."""
    max_val = arr[0]
    for x in arr:                     # Touch each element exactly once
        if x > max_val:
            max_val = x
    return max_val


# --- O(n log n) — Linearithmic time -----------------------------------------

@timed
def o_nlogn_sort(arr: list) -> list:
    """Python's Timsort: O(n log n) worst case. Returns sorted copy."""
    return sorted(arr)                # sorted() creates a new list


# --- O(n^2) — Quadratic time ------------------------------------------------

@timed
def o_n2_bubble_sort(arr: list) -> list:
    """Bubble sort: O(n^2). Classic example of nested loops."""
    arr = arr.copy()                  # Don't mutate the original
    n = len(arr)
    for i in range(n):                # n iterations
        for j in range(0, n - i - 1): # n-1, n-2, ... iterations
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap
    return arr


# --- O(2^n) — Exponential time ----------------------------------------------

@timed
def o_2n_fibonacci(n: int) -> int:
    """Naive recursive Fibonacci. O(2^n) — each call spawns 2 sub-calls."""
    if n <= 1:
        return n
    return o_2n_fibonacci.__wrapped__(n - 1) + o_2n_fibonacci.__wrapped__(n - 2)

# Store the raw function for recursive calls (bypass the timing decorator)
o_2n_fibonacci.__wrapped__ = lambda n: n if n <= 1 else o_2n_fibonacci.__wrapped__(n - 1) + o_2n_fibonacci.__wrapped__(n - 2)


# --- O(n!) — Factorial time --------------------------------------------------

@timed
def o_nfact_permutations(arr: list) -> list:
    """Generate all permutations. O(n!) results, each of length n."""
    if len(arr) <= 1:
        return [arr[:]]              # Base case: single element
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]   # Remove element i
        for p in o_nfact_permutations.__wrapped__(rest):
            result.append([arr[i]] + p)
    return result

o_nfact_permutations.__wrapped__ = lambda arr: [arr[:]] if len(arr) <= 1 else [
    [arr[i]] + p
    for i in range(len(arr))
    for p in o_nfact_permutations.__wrapped__(arr[:i] + arr[i+1:])
]


# =============================================================================
# SECTION 2: Timing comparisons — FEEL the difference
# =============================================================================

def demo_timing_comparisons():
    """Run all complexity classes and compare their wall-clock times."""
    print("=" * 70)
    print("TIMING COMPARISONS — same data, different algorithms")
    print("=" * 70)

    # --- O(1) vs O(n) vs O(n^2) on increasing sizes -------------------------
    sizes = [1_000, 10_000, 50_000]

    for n in sizes:
        print(f"\n--- n = {n:,} ---")
        data = list(range(n))
        random.shuffle(data)
        data_dict = {i: i for i in range(n)}

        o1_dict_lookup(data_dict, n // 2)     # O(1)
        o_logn_binary_search(sorted(data), n // 2)  # O(log n) — needs sorted input
        o_n_find_max(data)                     # O(n)
        o_nlogn_sort(data)                     # O(n log n)

        if n <= 10_000:                        # O(n^2) is too slow for n > 10k
            o_n2_bubble_sort(data)

    # --- Exponential: watch it explode ---------------------------------------
    print(f"\n--- Exponential: Fibonacci ---")
    for k in [20, 25, 30, 35]:
        print(f"  fib({k}):", end="")
        start = time.perf_counter()
        result = o_2n_fibonacci.__wrapped__(k)
        elapsed = time.perf_counter() - start
        print(f" = {result}, took {elapsed:.4f}s")

    # --- Factorial: permutations ---------------------------------------------
    print(f"\n--- Factorial: Permutations ---")
    for k in [6, 8, 10]:
        start = time.perf_counter()
        result = o_nfact_permutations.__wrapped__(list(range(k)))
        elapsed = time.perf_counter() - start
        print(f"  perms({k}): {len(result):,} permutations, took {elapsed:.4f}s")


# =============================================================================
# SECTION 3: Python-specific gotchas
# =============================================================================

def demo_list_vs_set_lookup():
    """
    Piege #1: `in` on list is O(n), on set is O(1).
    This is THE most common Python complexity trap in interviews.
    """
    print("\n" + "=" * 70)
    print("GOTCHA #1: list lookup vs set lookup")
    print("=" * 70)

    n = 100_000
    data = list(range(n))
    data_set = set(data)                  # O(n) to build, but O(1) per lookup after
    targets = random.sample(range(n), 1000)  # 1000 random lookups

    # List lookup: O(n) per check * 1000 checks = O(1000 * n) = ~10^8
    start = time.perf_counter()
    count = sum(1 for t in targets if t in data)      # `in` scans the list linearly
    list_time = time.perf_counter() - start
    print(f"  List lookup (1000 searches in {n:,} elements): {list_time:.4f}s")

    # Set lookup: O(1) per check * 1000 checks = O(1000)
    start = time.perf_counter()
    count = sum(1 for t in targets if t in data_set)   # `in` hashes and checks bucket
    set_time = time.perf_counter() - start
    print(f"  Set lookup  (1000 searches in {n:,} elements): {set_time:.4f}s")
    print(f"  Speedup: {list_time / max(set_time, 1e-9):.0f}x faster with set")


def demo_string_concatenation():
    """
    Piege #2: String concatenation in a loop is O(n^2).
    Strings are immutable — each += creates a brand new string object.
    """
    print("\n" + "=" * 70)
    print("GOTCHA #2: string concatenation += vs ''.join()")
    print("=" * 70)

    n = 50_000
    strings = ["x" for _ in range(n)]     # n small strings to concatenate

    # BAD: += in loop — O(n^2) because each step copies the growing string
    start = time.perf_counter()
    result = ""
    for s in strings:
        result += s                        # New string object every iteration!
    concat_time = time.perf_counter() - start
    print(f"  += loop ({n:,} strings):   {concat_time:.6f}s")

    # GOOD: join — O(n) total, single allocation
    start = time.perf_counter()
    result = "".join(strings)              # Calculates total size first, one malloc
    join_time = time.perf_counter() - start
    print(f"  ''.join() ({n:,} strings): {join_time:.6f}s")
    print(f"  Speedup: {concat_time / max(join_time, 1e-9):.0f}x faster with join")


def demo_insert_front():
    """
    Piege #3: list.insert(0, x) is O(n) — shifts all elements.
    Use collections.deque for O(1) appendleft.
    """
    print("\n" + "=" * 70)
    print("GOTCHA #3: list.insert(0, x) vs deque.appendleft()")
    print("=" * 70)

    n = 50_000

    # BAD: list.insert(0, x) — O(n) per insert, O(n^2) total
    start = time.perf_counter()
    arr = []
    for i in range(n):
        arr.insert(0, i)                   # Shifts ALL existing elements right
    list_time = time.perf_counter() - start
    print(f"  list.insert(0, x) ({n:,} ops): {list_time:.4f}s")

    # GOOD: deque.appendleft — O(1) per insert, O(n) total
    start = time.perf_counter()
    dq = deque()
    for i in range(n):
        dq.appendleft(i)                   # Doubly-linked list, no shifting
    deque_time = time.perf_counter() - start
    print(f"  deque.appendleft ({n:,} ops):  {deque_time:.4f}s")
    print(f"  Speedup: {list_time / max(deque_time, 1e-9):.0f}x faster with deque")


# =============================================================================
# SECTION 4: Step-by-step complexity analysis function
# =============================================================================

def analyze_complexity_example():
    """
    Walk through analyzing a real function's complexity, step by step.
    This teaches the PROCESS, not just the answer.
    """
    print("\n" + "=" * 70)
    print("STEP-BY-STEP ANALYSIS: Two Sum problem")
    print("=" * 70)

    # --- Version 1: Brute force ---
    def two_sum_brute(nums, target):
        """Find two indices whose values sum to target."""
        for i in range(len(nums)):             # Step 1: outer loop = n iterations
            for j in range(i + 1, len(nums)):  # Step 2: inner loop = n-1, n-2, ... iterations
                if nums[i] + nums[j] == target:  # Step 3: O(1) comparison
                    return [i, j]
        return []

    # Analysis:
    # - Outer loop: n iterations
    # - Inner loop: average n/2 iterations per outer iteration
    # - Total: n * n/2 = n^2/2
    # - Big-O: O(n^2)
    # - Space: O(1) — no extra data structures
    print("\n  Version 1 — Brute force:")
    print("    Outer loop: n iterations")
    print("    Inner loop: ~n/2 iterations per outer step")
    print("    Total: n * n/2 → O(n^2)")
    print("    Space: O(1)")

    # --- Version 2: Hash map optimization ---
    def two_sum_hash(nums, target):
        """Find two indices whose values sum to target — optimized."""
        seen = {}                              # Step 1: empty dict, O(1) space initially
        for i, num in enumerate(nums):         # Step 2: single loop = n iterations
            complement = target - num          # Step 3: O(1) arithmetic
            if complement in seen:             # Step 4: O(1) dict lookup (not list!)
                return [seen[complement], i]
            seen[num] = i                      # Step 5: O(1) dict insert
        return []
    # grows to at most n entries

    # Analysis:
    # - Single loop: n iterations
    # - Each iteration: O(1) dict lookup + O(1) insert
    # - Total: n * O(1) = O(n)
    # - Space: O(n) — dict stores up to n entries
    print("\n  Version 2 — Hash map:")
    print("    Single loop: n iterations")
    print("    Each step: O(1) dict lookup + O(1) insert")
    print("    Total: n * O(1) → O(n)")
    print("    Space: O(n) — dict stores up to n entries")

    # --- Timing comparison ---
    print("\n  Timing comparison:")
    n = 10_000
    nums = list(range(n))
    target = n - 2 + n - 1  # Worst case: answer is the last two elements

    start = time.perf_counter()
    two_sum_brute(nums, target)
    brute_time = time.perf_counter() - start
    print(f"    Brute force (n={n:,}): {brute_time:.4f}s")

    start = time.perf_counter()
    two_sum_hash(nums, target)
    hash_time = time.perf_counter() - start
    print(f"    Hash map    (n={n:,}): {hash_time:.6f}s")
    print(f"    Speedup: {brute_time / max(hash_time, 1e-9):.0f}x")

    # --- KEY INSIGHT for interviews ---
    print("\n  KEY INSIGHT:")
    print("    Trade-off: O(n) extra space → O(n^2) → O(n) time")
    print("    This space-for-time trade is THE fundamental optimization pattern.")


# =============================================================================
# SECTION 5: Amortized complexity — list.append() deep dive
# =============================================================================

def demo_amortized_append():
    """
    Show that list.append() is O(1) amortized by measuring individual appends.
    Most are instant; occasional spikes = reallocation.
    """
    print("\n" + "=" * 70)
    print("AMORTIZED COMPLEXITY: list.append() timing")
    print("=" * 70)

    import sys

    n = 100_000
    arr = []
    spikes = []  # Track expensive appends (reallocation events)
    prev_size = sys.getsizeof(arr)

    for i in range(n):
        arr.append(i)
        new_size = sys.getsizeof(arr)
        if new_size != prev_size:
            # Reallocation happened! Memory jumped.
            spikes.append((i, prev_size, new_size))
            prev_size = new_size

    print(f"  {n:,} appends performed")
    print(f"  Reallocations detected: {len(spikes)}")
    print(f"  Last 5 reallocations (index, old_bytes, new_bytes):")
    for idx, old, new in spikes[-5:]:
        growth = new / old
        print(f"    append #{idx:,}: {old:,} → {new:,} bytes (growth factor: {growth:.2f}x)")
    print(f"\n  Conclusion: {len(spikes)} reallocations over {n:,} appends")
    print(f"  = {len(spikes)/n*100:.2f}% of appends trigger a reallocation")
    print(f"  → The amortized cost per append is O(1)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   Day 1 -- Complexity & Big-O: Interactive Examples")
    print("=" * 70)

    demo_timing_comparisons()
    demo_list_vs_set_lookup()
    demo_string_concatenation()
    demo_insert_front()
    analyze_complexity_example()
    demo_amortized_append()

    print("\n" + "=" * 70)
    print("DONE — Review the output above. Notice how:")
    print("  1. O(n^2) becomes visibly slow around n=10,000")
    print("  2. O(2^n) explodes between n=30 and n=35")
    print("  3. Set lookups are 100x+ faster than list lookups")
    print("  4. ''.join() crushes += concatenation")
    print("  5. deque.appendleft() crushes list.insert(0, x)")
    print("=" * 70)

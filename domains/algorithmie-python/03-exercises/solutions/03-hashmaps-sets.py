"""
Solutions — Day 3: Hash Maps & Sets (Frequency Counting, Grouping, Two-Sum Patterns)
Run: python domains/algorithmie-python/03-exercises/solutions/03-hashmaps-sets.py

Each solution is numbered to match the exercise file.
All solutions are verified with assertions at the end.
"""

from collections import Counter, defaultdict, OrderedDict


# =============================================================================
# EXERCISE 1 (Easy): Ransom Note — Frequency Counting
# =============================================================================

def can_construct(ransom_note: str, magazine: str) -> bool:
    """
    Frequency counting: check if magazine has enough of each character.

    APPROACH:
    - Count character frequencies in magazine
    - For each char in ransom_note, decrement the count
    - If any count goes below 0, magazine doesn't have enough → False

    Alternative (even simpler): Counter subtraction
    - Counter(ransom_note) - Counter(magazine) should be empty

    Time: O(n + m) where n = len(ransom_note), m = len(magazine)
    Space: O(1) — bounded by alphabet size (26 lowercase letters)
    """
    # Build frequency map from magazine (the "supply")
    available = Counter(magazine)

    # Check if each char in ransom_note can be "consumed"
    for c in ransom_note:
        if available[c] <= 0:    # Counter returns 0 for missing keys (no KeyError)
            return False
        available[c] -= 1

    return True

    # ONE-LINER alternative:
    # return not (Counter(ransom_note) - Counter(magazine))
    # Counter subtraction removes keys with count <= 0
    # If result is empty (falsy), ransom_note is constructible


def test_exercise_1():
    print("\nExercise 1: Ransom Note")

    assert can_construct("a", "b") == False
    assert can_construct("aa", "ab") == False
    assert can_construct("aa", "aab") == True
    assert can_construct("", "anything") == True
    assert can_construct("abc", "abc") == True
    assert can_construct("abc", "cba") == True
    assert can_construct("aab", "baa") == True
    assert can_construct("fihjjjjei", "hjibagacbhadfaefdjaeaebgi") == False

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 2 (Easy): Contains Duplicate — Seen Set
# =============================================================================

def contains_duplicate(nums: list[int]) -> bool:
    """
    Seen set: add elements one by one, check if already seen.

    SHORT-CIRCUIT: return True immediately on first duplicate found.
    This avoids scanning the rest of the array unnecessarily.

    Alternative one-liner: return len(nums) != len(set(nums))
    But this ALWAYS processes the entire array (no short-circuit).

    Time: O(n) — worst case visits all elements
    Space: O(n) — set stores at most n elements
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True     # Duplicate found — stop immediately
        seen.add(num)
    return False


def test_exercise_2():
    print("\nExercise 2: Contains Duplicate")

    assert contains_duplicate([1, 2, 3, 1]) == True
    assert contains_duplicate([1, 2, 3, 4]) == False
    assert contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) == True
    assert contains_duplicate([]) == False
    assert contains_duplicate([1]) == False
    assert contains_duplicate([1, 1]) == True

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 3 (Easy): Two Sum — Complement Lookup
# =============================================================================

def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Hash map as complement lookup table.

    For each element nums[i]:
    1. Compute complement = target - nums[i]
    2. Check if complement was seen before (O(1) lookup)
    3. If yes → return the two indices
    4. If no → store nums[i] with its index for future lookups

    CRITICAL: store AFTER checking.
    - Prevents self-match (using the same element twice)
    - Example: nums=[5], target=10 → complement=5, but it's the same element
    - By checking before storing, we ensure complement comes from a DIFFERENT index

    Time: O(n) — single pass
    Space: O(n) — hash map stores at most n entries
    """
    seen = {}     # value → index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]    # Found pair: [earlier_index, current_index]

        seen[num] = i    # Store current element for future lookups

    return []    # No solution found (shouldn't happen per problem constraints)


def test_exercise_3():
    print("\nExercise 3: Two Sum")

    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    assert two_sum([1, 5, 8, 3], 4) == [0, 3]
    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 4 (Medium): Group Anagrams — Grouping
# =============================================================================

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    Grouping with character frequency tuple as key.

    Two approaches for the key:
    A) tuple(sorted(s)) — O(k log k) per string, simpler code
    B) Frequency tuple — O(k) per string, faster for long strings

    We implement approach B (optimal).

    KEY INSIGHT: all anagrams produce the same frequency signature.
    "eat" → (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)
    "tea" → same tuple
    "ate" → same tuple

    Time: O(n * k) where n = number of strings, k = max string length
    Space: O(n * k) — storing all strings in groups
    """
    groups = defaultdict(list)

    for s in strs:
        # Build frequency array for 26 lowercase letters
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1

        # Use tuple as dict key (lists are NOT hashable)
        key = tuple(count)
        groups[key].append(s)

    return list(groups.values())


def test_exercise_4():
    print("\nExercise 4: Group Anagrams")

    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    normalized = sorted([sorted(g) for g in result])
    assert normalized == sorted([["ate", "eat", "tea"], ["nat", "tan"], ["bat"]])

    result = group_anagrams([""])
    assert result == [[""]]

    result = group_anagrams(["a"])
    assert result == [["a"]]

    result = group_anagrams(["abc", "bca", "cab"])
    assert len(result) == 1
    assert sorted(result[0]) == ["abc", "bca", "cab"]

    result = group_anagrams(["abc", "def", "ghi"])
    assert len(result) == 3

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 5 (Medium): Intersection of Two Arrays II — Frequency + Set
# =============================================================================

def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Frequency counting on the smaller array, then iterate the larger.

    APPROACH:
    1. Build Counter on the SMALLER array (saves memory)
    2. Iterate through the LARGER array
    3. For each element: if count > 0 in Counter, include it and decrement

    WHY Counter on smaller array:
    - Follow-up: if nums1 is much smaller, we use O(min(n,m)) space
    - If nums2 is on disk, we only need nums1 in memory

    Time: O(n + m)
    Space: O(min(n, m)) — Counter on the smaller array
    """
    # Ensure nums1 is the smaller array (for space optimization)
    if len(nums1) > len(nums2):
        return intersect(nums2, nums1)

    freq = Counter(nums1)         # Counter on smaller array
    result = []

    for num in nums2:
        if freq[num] > 0:         # This element exists in nums1 (with remaining count)
            result.append(num)
            freq[num] -= 1        # "Consume" one occurrence

    return result


def test_exercise_5():
    print("\nExercise 5: Intersection of Two Arrays II")

    assert sorted(intersect([1, 2, 2, 1], [2, 2])) == [2, 2]
    assert sorted(intersect([4, 9, 5], [9, 4, 9, 8, 4])) == [4, 9]
    assert intersect([1, 2, 3], [4, 5, 6]) == []
    assert sorted(intersect([1, 1, 1], [1, 1])) == [1, 1]
    assert intersect([], [1, 2, 3]) == []
    assert intersect([1, 2, 3], []) == []
    assert sorted(intersect([3, 1, 2], [1, 1])) == [1]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 6 (Medium): Subarray Sum Equals K — Prefix Sum + Hash Map
# =============================================================================

def subarray_sum(nums: list[int], k: int) -> int:
    """
    Prefix sum + hash map for O(n) counting of subarrays with sum = k.

    CORE IDEA:
    - Let prefix[j] = sum of nums[0..j-1]
    - If prefix[j] - prefix[i] = k, then subarray nums[i..j-1] sums to k
    - Rearranging: prefix[i] = prefix[j] - k
    - For each j, count how many earlier prefix sums equal (prefix[j] - k)
    - A hash map stores: {prefix_sum_value: count_of_times_seen}

    WHY seen = {0: 1}?
    - There is one "empty prefix" with sum 0 (before the array starts)
    - This handles subarrays starting from index 0:
      if prefix[j] = k, then prefix[j] - k = 0, and seen[0] = 1 → count += 1

    WHY sliding window fails:
    - Sliding window assumes monotonic behavior (expand increases sum)
    - With NEGATIVE numbers, expanding can decrease the sum
    - prefix sum + hashmap works regardless of sign

    Time: O(n) — single pass, O(1) per hash operation
    Space: O(n) — hash map stores at most n distinct prefix sums
    """
    count = 0
    prefix = 0          # Running prefix sum (no need for an array)
    seen = {0: 1}       # {prefix_sum: number_of_times_seen}

    for num in nums:
        prefix += num   # Update running prefix sum

        # How many earlier prefix sums equal prefix - k?
        complement = prefix - k
        if complement in seen:
            count += seen[complement]

        # Record current prefix sum
        seen[prefix] = seen.get(prefix, 0) + 1

    return count


def test_exercise_6():
    print("\nExercise 6: Subarray Sum Equals K")

    assert subarray_sum([1, 1, 1], 2) == 2
    assert subarray_sum([1, 2, 3], 3) == 2
    assert subarray_sum([1, -1, 0], 0) == 3
    assert subarray_sum([0, 0, 0], 0) == 6
    assert subarray_sum([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
    assert subarray_sum([-1, -1, 1], 0) == 1
    assert subarray_sum([1], 0) == 0
    assert subarray_sum([1], 1) == 1

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 7 (Hard): Minimum Window Substring — Sliding Window + Hash Maps
# =============================================================================

def min_window(s: str, t: str) -> str:
    """
    Sliding window with two hash maps (need + window) and a 'have' counter.

    ALGORITHM:
    1. Build 'need' Counter from t — what we must have
    2. Expand right pointer, updating 'window' Counter
    3. When a char in window reaches its needed count → increment 'have'
    4. While have == len(need) (all requirements met):
       - Update best if current window is smaller
       - Shrink from left: decrement window counter
       - If a char drops below needed count → decrement 'have'
    5. Return the smallest valid window found

    KEY OPTIMIZATION:
    - Instead of comparing the entire window Counter to need Counter at every step
      (which would be O(|t|)), we maintain 'have' as a running count of satisfied
      unique characters. We only update 'have' when a threshold is crossed.
    - This makes the comparison O(1) instead of O(|t|).

    Time: O(n + m) — each pointer moves at most n times, t scanned once
    Space: O(m + |alphabet|) — two counters
    """
    if not s or not t:
        return ""

    need = Counter(t)            # Required character counts
    need_count = len(need)       # Number of unique chars to satisfy

    window = {}                  # Current window character counts
    have = 0                     # Number of unique chars with sufficient count
    left = 0
    best = (float('inf'), 0, 0)  # (length, left_index, right_index)

    for right in range(len(s)):
        # EXPAND: add s[right] to window
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # Did adding this char satisfy a requirement?
        if char in need and window[char] == need[char]:
            have += 1            # This unique char is now fully satisfied

        # SHRINK: while all requirements met, try to minimize window
        while have == need_count:
            # Update best if current window is smaller
            window_len = right - left + 1
            if window_len < best[0]:
                best = (window_len, left, right)

            # Remove s[left] from window
            left_char = s[left]
            window[left_char] -= 1

            # Did removing this char break a requirement?
            if left_char in need and window[left_char] < need[left_char]:
                have -= 1        # Lost a required character

            left += 1           # Shrink window

    length, lo, hi = best
    return s[lo:hi + 1] if length != float('inf') else ""


def test_exercise_7():
    print("\nExercise 7: Minimum Window Substring")

    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""
    assert min_window("", "abc") == ""
    assert min_window("abc", "") == ""
    assert min_window("aa", "aa") == "aa"
    assert min_window("bba", "ab") == "ba"
    assert min_window("aaflslflsldkalskaaa", "aaa") == "aaa"
    assert min_window("abcdef", "z") == ""
    assert min_window("aaaa", "a") == "a"
    assert min_window("aaaa", "aa") == "aa"

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 8 (Hard): LRU Cache — OrderedDict (Hash Map + Doubly Linked List)
# =============================================================================

# --- Solution A: Using OrderedDict (Pythonic, recommended in Python interviews) ---

class LRUCache:
    """
    LRU Cache using OrderedDict.

    OrderedDict maintains insertion order AND supports:
    - move_to_end(key): O(1) — mark as most recently used
    - popitem(last=False): O(1) — remove least recently used (front)

    Under the hood, OrderedDict is a regular dict + a doubly-linked list.
    This gives us O(1) for get, put, and eviction.

    INVARIANT:
    - Front of OrderedDict = least recently used (LRU)
    - Back of OrderedDict = most recently used (MRU)
    - get() → move_to_end (mark as MRU)
    - put() → if new key, add to end (MRU); if existing, update + move_to_end
    - eviction → popitem(last=False) removes the front (LRU)

    Time: O(1) for get and put
    Space: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()   # key → value, ordered by recency

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Mark as most recently used by moving to end
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing: change value + move to end (most recent)
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Insert new entry
            self.cache[key] = value
            # Evict LRU if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove FIRST item (LRU)


# --- Solution B: Manual implementation with dict + doubly-linked list ---
# (Bonus: this is what interviewers expect if they say "don't use OrderedDict")

class _Node:
    """Doubly-linked list node."""
    __slots__ = ('key', 'value', 'prev', 'next')

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCacheManual:
    """
    LRU Cache with dict + doubly-linked list.

    STRUCTURE:
    - dict: key → Node (for O(1) lookup)
    - Doubly-linked list: ordered by recency
      * head.next = LRU (least recently used)
      * tail.prev = MRU (most recently used)
      * head and tail are dummy sentinel nodes

    OPERATIONS:
    - get(key): find node in dict, move to end (before tail), return value
    - put(key, value): if exists, update + move; if new, add before tail, evict if needed
    - _remove(node): unlink a node from the list
    - _add_to_end(node): add a node just before tail (MRU position)

    Time: O(1) for all operations
    Space: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}           # key → Node

        # Sentinel nodes (avoid null checks)
        self.head = _Node()       # Dummy head → LRU side
        self.tail = _Node()       # Dummy tail → MRU side
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: _Node):
        """Remove node from the doubly-linked list. O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_end(self, node: _Node):
        """Add node just before tail (MRU position). O(1)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # Move to end (mark as most recently used)
        self._remove(node)
        self._add_to_end(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_end(node)
        else:
            # Create new node
            node = _Node(key, value)
            self.cache[key] = node
            self._add_to_end(node)
            # Evict LRU if over capacity
            if len(self.cache) > self.capacity:
                lru = self.head.next   # The node right after head is LRU
                self._remove(lru)
                del self.cache[lru.key]


def test_exercise_8():
    print("\nExercise 8: LRU Cache")

    # Test both implementations
    for CacheClass, name in [(LRUCache, "OrderedDict"), (LRUCacheManual, "Manual DLL")]:
        # Basic test from LeetCode 146
        cache = CacheClass(2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1

        cache.put(3, 3)                 # Evicts key 2
        assert cache.get(2) == -1

        cache.put(4, 4)                 # Evicts key 1
        assert cache.get(1) == -1
        assert cache.get(3) == 3
        assert cache.get(4) == 4

        # Capacity 1
        cache = CacheClass(1)
        cache.put(1, 1)
        cache.put(2, 2)                 # Evicts key 1
        assert cache.get(1) == -1
        assert cache.get(2) == 2

        # Update existing key
        cache = CacheClass(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)                # Update key 1
        assert cache.get(1) == 10
        cache.put(3, 3)                 # Evicts key 2 (not key 1, which was just updated)
        assert cache.get(2) == -1
        assert cache.get(1) == 10

        # Large sequence
        cache = CacheClass(3)
        for i in range(10):
            cache.put(i, i * 10)
        assert cache.get(6) == -1
        assert cache.get(7) == 70
        assert cache.get(8) == 80
        assert cache.get(9) == 90

        print(f"  {name} — PASS")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 3: Hash Maps & Sets")
    print("=" * 70)

    test_exercise_1()
    test_exercise_2()
    test_exercise_3()
    test_exercise_4()
    test_exercise_5()
    test_exercise_6()
    test_exercise_7()
    test_exercise_8()

    print("\n" + "=" * 70)
    print("ALL SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

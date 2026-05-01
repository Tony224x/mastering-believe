"""
Day 14 — Mock Interviews: 3 full walkthroughs (easy, medium, hard)
Run: python domains/algorithmie-python/02-code/14-mock-interviews.py

Each solution is narrated like a real interview: clarification, approach,
complexity, and verification. Read the comments as if you were presenting
your reasoning out loud to an interviewer.
"""


# =============================================================================
# MOCK 1 (EASY) — Valid Parentheses
# =============================================================================
#
# PROBLEM:
#   Given a string containing only '()[]{}', determine if it's valid:
#   every opening bracket has a matching closing bracket in the correct order.
#
# CLARIFICATION QUESTIONS YOU SHOULD ASK:
#   - Is empty string valid? (yes, conventionally)
#   - Only these 6 characters? (yes)
#   - Any size limit? (typically <= 10^4)
#
# INTUITION:
#   "()[]{}" is valid, but "([)]" is NOT — even though the counts match.
#   We need ORDER, which means we need a LIFO structure: a stack.
#
# APPROACH:
#   Push opening brackets onto a stack. On a closing bracket, the top of
#   the stack must be the corresponding opener. If not, or if the stack
#   is empty, the string is invalid. At the end, the stack must be empty
#   (every opener has been matched).
#
# COMPLEXITY:
#   Time : O(n) — one pass over the string
#   Space: O(n) — the stack can hold up to n/2 characters

def is_valid_parentheses(s):
    # Mapping closing -> opening. Using a dict makes the check O(1).
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in pairs:
            # Closing bracket: must match the top of the stack
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            # Opening bracket: push it
            stack.append(c)
    # All openers matched iff stack is empty
    return not stack


# =============================================================================
# MOCK 2 (MEDIUM) — Longest Substring Without Repeating Characters
# =============================================================================
#
# PROBLEM:
#   Given a string s, find the length of the longest substring without
#   repeating characters.
#
# CLARIFICATION:
#   - Empty string -> 0
#   - Uppercase vs lowercase count as different? Yes.
#   - ASCII only? Typically yes, but algorithm works with any alphabet.
#
# INTUITION:
#   A "substring" is contiguous. We can expand a window to the right as
#   long as no duplicate appears. When a duplicate shows up, we shrink
#   the window from the left.
#
# APPROACH (sliding window with a dict):
#   last[c] = the most recent index of character c in the window.
#   When we see c at index i:
#     - If c is in last and last[c] >= left, we've found a duplicate,
#       so we move left to last[c] + 1.
#     - Update last[c] = i.
#     - best = max(best, i - left + 1).
#
# COMPLEXITY:
#   Time : O(n) — each character visited at most twice
#   Space: O(min(n, alphabet))

def length_of_longest_substring(s):
    last = {}                          # char -> last seen index
    left = 0
    best = 0
    for right, c in enumerate(s):
        if c in last and last[c] >= left:
            # Move left past the previous occurrence of c
            left = last[c] + 1
        last[c] = right
        best = max(best, right - left + 1)
    return best


# =============================================================================
# MOCK 3 (HARD) — Merge Intervals
# =============================================================================
#
# PROBLEM:
#   Given a list of intervals [start, end], merge overlapping intervals
#   and return the non-overlapping result.
#
# CLARIFICATION:
#   - Does [1,4] and [4,5] count as overlapping? Yes (touching endpoints
#     merge into [1,5] — this is the conventional interpretation).
#   - Input might be empty or unsorted.
#
# INTUITION:
#   After sorting by start, any overlap must be between consecutive
#   intervals (proof: if interval i overlaps j > i+1, then i also
#   overlaps i+1 because starts are sorted). So one linear scan after
#   sorting is enough.
#
# APPROACH:
#   1. Sort intervals by start ascending.
#   2. Walk through. For each interval, if it overlaps the LAST one in
#      the result (current.start <= last.end), extend last.end to
#      max(last.end, current.end). Otherwise, append it.
#
# COMPLEXITY:
#   Time : O(n log n) — sorting dominates
#   Space: O(n) — the output list

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]         # Copy so we don't mutate input
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            # Overlap — extend the last interval's end
            last[1] = max(last[1], end)
        else:
            # No overlap — append a fresh one
            merged.append([start, end])
    return merged


# =============================================================================
# MAIN DEMO — run tests for all three mocks
# =============================================================================

if __name__ == "__main__":
    # Mock 1
    print("Mock 1 — Valid Parentheses:")
    for s, expected in [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True),
        ("(", False),
    ]:
        got = is_valid_parentheses(s)
        mark = "OK" if got == expected else "FAIL"
        print(f"  {mark} is_valid_parentheses({s!r}) = {got} (expected {expected})")

    # Mock 2
    print("\nMock 2 — Longest Substring Without Repeating:")
    for s, expected in [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        (" ", 1),
        ("dvdf", 3),
        ("abba", 2),
    ]:
        got = length_of_longest_substring(s)
        mark = "OK" if got == expected else "FAIL"
        print(f"  {mark} length_of_longest_substring({s!r}) = {got} (expected {expected})")

    # Mock 3
    print("\nMock 3 — Merge Intervals:")
    cases = [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
        ([[1, 4], [0, 4]], [[0, 4]]),
        ([[1, 4], [2, 3]], [[1, 4]]),
        ([], []),
        ([[1, 5]], [[1, 5]]),
    ]
    for intervals, expected in cases:
        got = merge_intervals([lst[:] for lst in intervals])
        mark = "OK" if got == expected else "FAIL"
        print(f"  {mark} merge_intervals({intervals}) = {got} (expected {expected})")

"""
Solutions — Day 10: Dynamic Programming (HARD)
Run: python domains/algorithmie-python/03-exercises/solutions/10-dynamic-programming-hard.py

Each solution is numbered to match the exercise file (03-hard/10-dynamic-programming.md).
All solutions are verified with assertions at the end.
"""

from functools import lru_cache


# =============================================================================
# EXERCISE 7 (Hard): Edit Distance (Levenshtein) — 2D DP (rolling rows)
# =============================================================================

def min_distance(word1: str, word2: str) -> int:
    """
    dp[i][j] = min ops to turn word1[:i] into word2[:j].

    BASE:
    - dp[i][0] = i  (delete i chars)
    - dp[0][j] = j  (insert j chars)

    TRANSITIONS:
    - word1[i-1] == word2[j-1] -> dp[i][j] = dp[i-1][j-1]      (no op)
    - else dp[i][j] = 1 + min(dp[i-1][j],    # delete from word1
                              dp[i][j-1],    # insert into word1
                              dp[i-1][j-1])  # replace

    SPACE: keep only previous + current row -> O(min(m, n)) by indexing the
    inner loop over the SHORTER string.

    Time: O(m * n), Space: O(min(m, n))
    """
    a, b = (word1, word2) if len(word1) >= len(word2) else (word2, word1)
    n = len(b)

    prev = list(range(n + 1))            # dp[0][j] = j
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * n             # dp[i][0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[n]


def _edit_distance_grid(word1: str, word2: str) -> int:
    """Reference full-grid version used as a cross-check oracle."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def test_exercise_7():
    print("\nExercise 7: Edit Distance (Levenshtein)")

    cases = [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("abc", "abc", 0),
        ("a", "b", 1),
        ("sunday", "saturday", 3),
        ("kitten", "sitting", 3),
    ]
    for w1, w2, expected in cases:
        got = min_distance(w1, w2)
        assert got == expected, f"{w1!r},{w2!r}: got {got}, want {expected}"
        assert got == _edit_distance_grid(w1, w2)   # cross-check

    print("  PASS — all test cases (incl. full-grid cross-check)")


# =============================================================================
# EXERCISE 8 (Hard): Word Break II — reconstruction with memoized DFS by index
# =============================================================================

def word_break_all(s: str, word_dict: list[str]) -> list[str]:
    """
    Return ALL sentences. Memoize by START INDEX so each suffix is solved once.

    dfs(start) = list of all sentences formable from s[start:].
    For each end where s[start:end] is a word:
        for tail in dfs(end):
            sentence = word + (" " + tail if tail else "")

    Base: dfs(len(s)) = [""]  (empty suffix -> one empty sentence).

    WHY MEMOIZE BY INDEX:
    - The same suffix s[start:] can be reached through many prefixes; caching
      dfs(start) avoids recomputing it, taming the exponential blow-up.

    Time: O(n^2 + n * #words * output) ; Space: O(n + output)
    """
    words = set(word_dict)
    n = len(s)
    memo: dict[int, list[str]] = {}

    def dfs(start: int) -> list[str]:
        if start == n:
            return [""]                  # one way to "finish": empty tail
        if start in memo:
            return memo[start]

        sentences = []
        for end in range(start + 1, n + 1):
            prefix = s[start:end]
            if prefix in words:
                for tail in dfs(end):
                    sentences.append(prefix if not tail else prefix + " " + tail)
        memo[start] = sentences
        return sentences

    return dfs(0)


def test_exercise_8():
    print("\nExercise 8: Word Break II")

    res = word_break_all("catsanddog", ["cat", "cats", "and", "sand", "dog"])
    assert sorted(res) == sorted(["cats and dog", "cat sand dog"])

    res = word_break_all("pineapplepenapple",
                         ["apple", "pen", "applepen", "pine", "pineapple"])
    assert sorted(res) == sorted([
        "pine apple pen apple",
        "pineapple pen apple",
        "pine applepen apple",
    ])

    assert word_break_all("catsandog", ["cats", "dog", "sand", "and", "cat"]) == []
    assert word_break_all("", ["a"]) == [""]
    assert word_break_all("a", ["a"]) == ["a"]

    print("  PASS — all test cases")


# =============================================================================
# EXERCISE 9 (Hard): Regular Expression Matching ('.' and '*') — 2D DP
# =============================================================================

def is_match(s: str, p: str) -> bool:
    """
    dp[i][j] = does s[:i] match p[:j]?

    BASE:
    - dp[0][0] = True
    - dp[0][j] : a pattern like "a*b*c*" can match "" -> if p[j-1]=='*',
      dp[0][j] = dp[0][j-2].

    TRANSITIONS:
    - If p[j-1] in (s[i-1], '.')  -> dp[i][j] = dp[i-1][j-1]   (consume one char)
    - If p[j-1] == '*':
        * zero occurrence of p[j-2]: dp[i][j] = dp[i][j-2]
        * OR one more occurrence if p[j-2] matches s[i-1]: dp[i-1][j]

    Time: O(m * n), Space: O(m * n). No use of the `re` module.
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Patterns that can match the empty string: x*, x*y*, ...
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pc = p[j - 1]
            if pc == '*':
                # Zero occurrence of the char before '*'
                dp[i][j] = dp[i][j - 2]
                # One more occurrence if the preceding pattern matches s[i-1]
                prev = p[j - 2]
                if prev == '.' or prev == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif pc == '.' or pc == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            # else stays False

    return dp[m][n]


def _is_match_brute(s: str, p: str) -> bool:
    """Recursive brute-force oracle (no `re`) for cross-checking on small inputs."""
    @lru_cache(maxsize=None)
    def rec(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)
        first = i < len(s) and (p[j] == s[i] or p[j] == '.')
        if j + 1 < len(p) and p[j + 1] == '*':
            return rec(i, j + 2) or (first and rec(i + 1, j))
        return first and rec(i + 1, j + 1)
    return rec(0, 0)


def test_exercise_9():
    print("\nExercise 9: Regular Expression Matching")

    cases = [
        ("aa", "a", False),
        ("aa", "a*", True),
        ("ab", ".*", True),
        ("aab", "c*a*b", True),
        ("misissipi", "mis*is*p*.", False),
        ("", "", True),
        ("", "a*", True),
        ("", ".*", True),
        ("abc", "", False),
        ("aaa", "a*a", True),
        ("aaa", "ab*a*c*a", True),
        ("a", ".*..a*", False),
    ]
    for s, p, expected in cases:
        got = is_match(s, p)
        assert got == expected, f"is_match({s!r},{p!r}): got {got}, want {expected}"
        assert got == _is_match_brute(s, p)   # cross-check vs recursive oracle

    print("  PASS — all test cases (incl. recursive cross-check)")


# =============================================================================
# MAIN — Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SOLUTIONS — Day 10: Dynamic Programming (HARD)")
    print("=" * 70)

    test_exercise_7()
    test_exercise_8()
    test_exercise_9()

    print("\n" + "=" * 70)
    print("ALL HARD SOLUTIONS COMPLETE — All assertions passed")
    print("=" * 70)

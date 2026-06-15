"""
Day 17 -- Solutions to the easy exercises for verifiers & self-improvement.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/17-verifiers-self-improvement.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared helpers from the day-17 code module
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

day17 = import_module("17-verifiers-self-improvement")
_evaluate_expr = day17._evaluate_expr
_extract_answer = day17._extract_answer
best_of_n = day17.best_of_n
weighted_majority = day17.weighted_majority
OutcomeVerifier = day17.OutcomeVerifier
Lesson = day17.Lesson

random.seed(0)

# ===========================================================================
# SOLUTION 1 -- Parity Verifier (ORM) and Parity Process Verifier (PRM)
# ===========================================================================

print("\n" + "=" * 60)
print("  SOLUTION 1 — ParityVerifier & ParityProcessVerifier")
print("=" * 60)


class ParityVerifier:
    """
    ORM: scores a final answer string based on whether the last numeric
    value is even (1.0), odd (0.0), or unparseable (0.5).
    """

    def score(self, answer: str) -> float:
        tokens = answer.replace("=", " ").split()
        for tok in reversed(tokens):
            val = _evaluate_expr(tok)
            if val is not None:
                return 1.0 if int(val) % 2 == 0 else 0.0
        return 0.5  # cannot parse — ambiguous


class ParityProcessVerifier:
    """
    PRM: scores a single reasoning step based on whether an intermediate
    result is even (1.0), odd (0.4), or absent (0.2).
    """

    def score_step(self, step: str) -> float:
        # Scan tokens right-to-left so we pick the *result* of an expression
        # (e.g. in "3 + 5 = 8" we want 8, not 3).
        tokens = step.replace("=", " ").replace("->", " ").split()
        for tok in reversed(tokens):
            val = _evaluate_expr(tok)
            if val is not None:
                return 1.0 if int(val) % 2 == 0 else 0.4
        return 0.2  # no numeric result found in this step


parity_verifier = ParityVerifier()
parity_prm = ParityProcessVerifier()

test_expressions = [
    "step1: 3 + 5 = 8",   # even intermediate
    "step1: 3 + 4 = 7",   # odd intermediate
    "answer: 12",          # even final
    "answer: 7",           # odd final
    "answer: ??",          # ambiguous
]

print(f"\n  {'Expression':<35} {'ORM score':>10} {'PRM score':>10}")
print(f"  {'-' * 55}")
for expr in test_expressions:
    orm_s = parity_verifier.score(expr)
    prm_s = parity_prm.score_step(expr)
    print(f"  {expr!r:<35} {orm_s:>10.2f} {prm_s:>10.2f}")

# Assertions
assert parity_verifier.score("answer: 12") == 1.0, "12 should score 1.0"
assert parity_verifier.score("answer: 7") == 0.0, "7 should score 0.0"
assert parity_verifier.score("answer: ??") == 0.5, "ambiguous should score 0.5"
assert parity_prm.score_step("step1: 3 + 5 = 8") == 1.0, "8 is even -> 1.0"
assert parity_prm.score_step("step1: 3 + 4 = 7") == 0.4, "7 is odd -> 0.4"
print("\n  [OK] All parity verifier assertions passed.")


# ===========================================================================
# SOLUTION 2 -- Compare best-of-N vs weighted majority (noisy setting)
# ===========================================================================

print("\n" + "=" * 60)
print("  SOLUTION 2 — Best-of-N vs Weighted Majority (noisy)")
print("=" * 60)


def noisy_generator(target: float, extra_context: str = "") -> str:
    """
    Mock generator with three outcome modes:
      50%  -> correct answer
      30%  -> close but wrong (target + 1..5)
      20%  -> completely random (1..100)
    """
    p = random.random()
    if p < 0.50:
        return f"answer: {int(target)}"
    elif p < 0.80:
        return f"answer: {int(target) + random.randint(1, 5)}"
    else:
        return f"answer: {random.randint(1, 100)}"


class NoisyOutcomeVerifier:
    """
    ORM with Gaussian noise on the score.  Simulates an imperfect verifier.
    """

    def __init__(self, target: float, noise: float = 0.15) -> None:
        self.target = target
        self.noise = noise
        self._base = OutcomeVerifier(target)

    def score(self, answer: str) -> float:
        base_score = self._base.score(answer)
        noisy = base_score + random.gauss(0, self.noise)
        return max(0.0, min(1.0, noisy))


TARGET = 24.0
REPEATS = 20
N = 10

bon_scores: list[float] = []
wm_scores: list[float] = []

for _ in range(REPEATS):
    nv = NoisyOutcomeVerifier(TARGET, noise=0.15)
    _, s_bon = best_of_n(noisy_generator, nv, TARGET, n=N)
    bon_scores.append(s_bon)
    _, s_wm = weighted_majority(noisy_generator, nv, TARGET, n=N)
    wm_scores.append(s_wm)

mean_bon = sum(bon_scores) / REPEATS
std_bon = (sum((s - mean_bon) ** 2 for s in bon_scores) / REPEATS) ** 0.5
mean_wm = sum(wm_scores) / REPEATS
std_wm = (sum((s - mean_wm) ** 2 for s in wm_scores) / REPEATS) ** 0.5

print(f"\n  Target = {TARGET:.0f}, n = {N}, repeats = {REPEATS}")
print(f"\n  {'Method':<25} {'Mean score':>12} {'Std dev':>10}")
print(f"  {'-' * 47}")
print(f"  {'Best-of-N':<25} {mean_bon:>12.4f} {std_bon:>10.4f}")
print(f"  {'Weighted Majority':<25} {mean_wm:>12.4f} {std_wm:>10.4f}")
print(
    "\n  Interpretation : weighted majority tends to have a lower std dev "
    "because it aggregates evidence from all candidates rather than relying "
    "on a single argmax, making it more robust to score noise."
)

assert REPEATS == 20, "Experiment must run exactly 20 times"
assert N == 10, "n must be 10"
print("\n  [OK] Comparison experiment completed.")


# ===========================================================================
# SOLUTION 3 -- Lessons store with expiration (TTL + confidence filter)
# ===========================================================================

print("\n" + "=" * 60)
print("  SOLUTION 3 — Lessons store with TTL expiration")
print("=" * 60)


@dataclass
class LessonWithTTL:
    context: str
    observation: str
    score_before: float
    score_after: float
    timestamp: str
    occurrences: int
    confidence: float
    last_seen: str       # ISO datetime of last occurrence
    ttl_days: int = 7   # lesson expires after this many days of inactivity


def filter_lessons(
    lessons: list[LessonWithTTL],
    now: str,
    min_confidence: float = 0.4,
) -> list[LessonWithTTL]:
    """
    Return lessons that are:
      - Still within their TTL window (last_seen + ttl_days >= now)
      - Above the minimum confidence threshold
    Sorted by confidence descending.
    """
    now_dt = datetime.fromisoformat(now)
    kept: list[LessonWithTTL] = []

    for lesson in lessons:
        last_seen_dt = datetime.fromisoformat(lesson.last_seen)
        age_days = (now_dt - last_seen_dt).days
        expired = age_days > lesson.ttl_days
        low_conf = lesson.confidence < min_confidence

        if expired and low_conf:
            print(f"  [EXCLUDED] {lesson.context!r:35s} — expired ({age_days}d) AND low confidence ({lesson.confidence:.2f})")
        elif expired:
            print(f"  [EXCLUDED] {lesson.context!r:35s} — expired ({age_days}d, ttl={lesson.ttl_days}d)")
        elif low_conf:
            print(f"  [EXCLUDED] {lesson.context!r:35s} — low confidence ({lesson.confidence:.2f} < {min_confidence})")
        else:
            kept.append(lesson)

    return sorted(kept, key=lambda l: l.confidence, reverse=True)


now_str = "2026-06-15T12:00:00"
now_dt = datetime.fromisoformat(now_str)

test_lessons = [
    LessonWithTTL(  # fresh + confident -> kept
        context="case_A_fresh_confident",
        observation="always decompose before substituting",
        score_before=0.3, score_after=0.9,
        timestamp=now_str, occurrences=3, confidence=0.8,
        last_seen=(now_dt - timedelta(days=2)).isoformat(), ttl_days=7,
    ),
    LessonWithTTL(  # fresh + confident (higher) -> kept
        context="case_B_fresh_confident_2",
        observation="check boundary cases first",
        score_before=0.4, score_after=0.95,
        timestamp=now_str, occurrences=5, confidence=0.92,
        last_seen=(now_dt - timedelta(days=1)).isoformat(), ttl_days=7,
    ),
    LessonWithTTL(  # old (8 days > 7 TTL) but confident -> expired
        context="case_C_old_confident",
        observation="normalize before applying quadratic formula",
        score_before=0.5, score_after=0.85,
        timestamp=now_str, occurrences=2, confidence=0.75,
        last_seen=(now_dt - timedelta(days=8)).isoformat(), ttl_days=7,
    ),
    LessonWithTTL(  # fresh but low confidence -> filtered
        context="case_D_fresh_low_conf",
        observation="try random restarts",
        score_before=0.2, score_after=0.35,
        timestamp=now_str, occurrences=1, confidence=0.2,
        last_seen=(now_dt - timedelta(days=1)).isoformat(), ttl_days=7,
    ),
    LessonWithTTL(  # old AND low confidence -> both reasons
        context="case_E_old_low_conf",
        observation="avoid negative intermediates",
        score_before=0.1, score_after=0.3,
        timestamp=now_str, occurrences=1, confidence=0.15,
        last_seen=(now_dt - timedelta(days=10)).isoformat(), ttl_days=7,
    ),
]

print(f"\n  Lessons before filtering : {len(test_lessons)}")
print()
surviving = filter_lessons(test_lessons, now_str, min_confidence=0.4)
print(f"\n  Lessons after  filtering : {len(surviving)}")
print(f"\n  Survivors (sorted by confidence desc):")
for lesson in surviving:
    print(f"    - {lesson.context!r:35s} conf={lesson.confidence:.2f}")

# Assertions
assert len(surviving) == 2, f"Expected 2 survivors, got {len(surviving)}"
assert surviving[0].confidence >= surviving[1].confidence, "Should be sorted desc by confidence"
assert all(l.context in ("case_A_fresh_confident", "case_B_fresh_confident_2") for l in surviving)
print("\n  [OK] Lesson filtering: exactly 2 survivors, sorted correctly.")

print("\nAll Day-17 solutions completed successfully.\n")

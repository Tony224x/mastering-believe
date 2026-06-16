"""
Day 17 -- Verifiers, Process Reward Models & Self-Improvement Loops.

Demonstrates:
  1. OutcomeVerifier      -- scores a final answer (ORM style)
  2. ProcessRewardModel   -- scores each reasoning step independently
  3. best_of_n            -- generate N candidates, pick the highest-scoring one
  4. weighted_majority    -- vote across candidates weighted by verifier score
  5. beam_search          -- PRM-guided beam search over reasoning steps
  6. SelfRefine           -- generator / critic / refiner loop (intra-run)
  7. SelfImprovingAgent   -- loads lessons from a JSON store at startup,
                             injects them into generation, and writes new
                             lessons at the end — improvement is measurable
                             run-over-run without touching model weights.

The "LLM" is entirely mocked with deterministic heuristics so the file
runs with stdlib only and zero API keys.

Dependencies: stdlib only (json, pathlib, dataclasses, random, math, hashlib)
Run:
    python domains/agentic-ai/02-code/17-verifiers-self-improvement.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ===========================================================================
# 1. PROBLEM DOMAIN — evaluate arithmetic expressions
# ===========================================================================
# The toy task: given a target integer T, generate a multi-step reasoning
# chain that produces T using only +, -, *, with numbers 1-9.
# Example target=24 -> steps: ["2 * 12", "12 = 3 * 4", "3 + 1 = 4", "answer: 24"]
#
# A real verifier checks each step; here we mock it with heuristics.


def _evaluate_expr(expr: str) -> float | None:
    """Safely evaluate a simple arithmetic expression string. Returns None on failure."""
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


# ===========================================================================
# 2. OUTCOME VERIFIER (ORM)
# ===========================================================================

class OutcomeVerifier:
    """
    Scores a final answer string against a target value.

    Score ∈ [0, 1].  1.0 means the answer matches exactly; 0.0 means wrong.
    Partial credit for being numerically close (simulates a soft ORM).
    """

    def __init__(self, target: float) -> None:
        self.target = target

    def score(self, answer: str) -> float:
        """Parse the last number in *answer* and compare to target."""
        # Extract last numeric token from the answer string
        tokens = answer.replace("=", " ").split()
        for tok in reversed(tokens):
            val = _evaluate_expr(tok)
            if val is not None:
                if math.isclose(val, self.target, rel_tol=1e-6):
                    return 1.0
                # Soft partial credit: 1 / (1 + |delta|)
                return 1.0 / (1.0 + abs(val - self.target))
        return 0.0


# ===========================================================================
# 3. PROCESS REWARD MODEL (PRM)
# ===========================================================================

class ProcessRewardModel:
    """
    Assigns a score ∈ [0, 1] to each reasoning *step* in isolation.

    In production this would be a fine-tuned classifier.  Here we use:
      - +0.3 base if the step contains an arithmetic expression that evaluates
      - +0.4 bonus if the intermediate result is a "useful" integer (divides target)
      - -0.2 penalty if the expression evaluates to 0 or negative (dead end)
      - random noise ~ N(0, 0.05) to simulate model uncertainty
    """

    def __init__(self, target: float, noise_std: float = 0.05) -> None:
        self.target = target
        self.noise_std = noise_std

    def score_step(self, step: str) -> float:
        # Find any sub-expression like "X op Y"
        tokens = step.replace("=", " ").replace("->", " ").split()
        score = 0.0
        for tok in tokens:
            val = _evaluate_expr(tok)
            if val is not None and val != 0.0:
                score = 0.3  # step has a valid numeric value
                if self.target != 0 and self.target % val == 0:
                    score += 0.4  # intermediate result divides the target
                if val <= 0:
                    score -= 0.2  # negative or zero is usually a dead end
                break

        # Add Gaussian noise (clamped to [0, 1])
        noise = random.gauss(0, self.noise_std)
        return max(0.0, min(1.0, score + noise))

    def score_trajectory(self, steps: list[str]) -> float:
        """Aggregate step scores with a min-product approach."""
        if not steps:
            return 0.0
        step_scores = [self.score_step(s) for s in steps]
        # Use min to surface any single bad step (conservative)
        return min(step_scores)


# ===========================================================================
# 4. MOCK GENERATOR
# ===========================================================================

# A mock "LLM" generator that builds arithmetic chains toward a target.
# Quality improves when extra_context (lessons) is provided.


def _mock_generator(target: float, extra_context: str = "") -> str:
    """
    Produce a reasoning chain attempting to reach *target*.
    extra_context simulates lessons injected into the prompt.

    Without lessons: random walks, ~40% chance of correct answer.
    With lessons: applies simple heuristics, ~70% chance.
    """
    use_lessons = "always check" in extra_context.lower() or len(extra_context) > 20

    # Simple strategy: pick two factors if target is composite
    steps: list[str] = []
    found = False

    if use_lessons:
        # Lessons teach us to try factor decomposition first
        for a in range(2, 10):
            if target % a == 0:
                b = int(target // a)
                steps.append(f"step1: decompose {int(target)} = {a} * {b}")
                steps.append(f"step2: verify {a} * {b} = {a * b}")
                steps.append(f"answer: {int(a * b)}")
                found = True
                break
        if not found:
            # Fallback: add/subtract
            a = random.randint(1, int(target) - 1)
            b = int(target) - a
            steps.append(f"step1: split {int(target)} = {a} + {b}")
            steps.append(f"step2: verify {a} + {b} = {a + b}")
            steps.append(f"answer: {int(a + b)}")
            found = True
    else:
        # Without lessons: naive random arithmetic (often wrong)
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        result = random.choice([a + b, a * b, a - b])
        steps.append(f"step1: try {a} op {b} = {result}")
        steps.append(f"step2: adjust if needed -> {result}")
        steps.append(f"answer: {result}")

    return "\n".join(steps)


# ===========================================================================
# 5. BEST-OF-N
# ===========================================================================

def best_of_n(
    generator: Callable[[float, str], str],
    verifier: OutcomeVerifier,
    target: float,
    n: int = 6,
    extra_context: str = "",
) -> tuple[str, float]:
    """
    Generate *n* independent candidates and return the one with the highest
    outcome verifier score.

    Returns (best_candidate, best_score).
    """
    candidates = [generator(target, extra_context) for _ in range(n)]
    scores = [verifier.score(c) for c in candidates]
    best_idx = scores.index(max(scores))
    return candidates[best_idx], scores[best_idx]


# ===========================================================================
# 6. WEIGHTED MAJORITY VOTING
# ===========================================================================

def _extract_answer(candidate: str) -> str:
    """Pull the 'answer: X' line from a candidate string."""
    for line in candidate.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return candidate.strip().splitlines()[-1]


def weighted_majority(
    generator: Callable[[float, str], str],
    verifier: OutcomeVerifier,
    target: float,
    n: int = 8,
    extra_context: str = "",
) -> tuple[str, float]:
    """
    Generate *n* candidates.  Tally verifier scores per unique answer.
    The answer with the highest total score wins.

    Returns (winning_answer, total_score_of_winning_group).
    """
    candidates = [generator(target, extra_context) for _ in range(n)]
    scores = [verifier.score(c) for c in candidates]

    tallies: dict[str, float] = defaultdict(float)
    answer_map: dict[str, str] = {}  # normalized answer -> first full candidate
    for cand, score in zip(candidates, scores):
        ans = _extract_answer(cand)
        tallies[ans] += score
        answer_map.setdefault(ans, cand)

    best_ans = max(tallies, key=lambda k: tallies[k])
    return answer_map[best_ans], tallies[best_ans]


# ===========================================================================
# 7. BEAM SEARCH GUIDED BY PRM
# ===========================================================================

@dataclass
class BeamNode:
    steps: list[str]
    prm_score: float  # cumulative PRM score so far


def beam_search(
    target: float,
    prm: ProcessRewardModel,
    outcome_verifier: OutcomeVerifier,
    beam_width: int = 3,
    max_depth: int = 3,
) -> tuple[list[str], float]:
    """
    Expand reasoning steps using PRM scores to prune the beam.

    At each depth level:
      - Each live node spawns *beam_width* child steps.
      - We keep only the top *beam_width* nodes by PRM score.
    Finally, the node whose full trajectory has the best *outcome* score wins.

    Returns (best_steps, outcome_score).
    """
    # Seed beam with single-step candidates
    beam: list[BeamNode] = []
    for a in range(2, 2 + beam_width + 1):
        if target % a == 0:
            step = f"step1: {int(target)} / {a} = {int(target // a)}"
        else:
            step = f"step1: {int(target)} - {a} = {int(target - a)}"
        node = BeamNode(steps=[step], prm_score=prm.score_step(step))
        beam.append(node)

    beam.sort(key=lambda n: n.prm_score, reverse=True)
    beam = beam[:beam_width]

    for depth in range(1, max_depth):
        next_beam: list[BeamNode] = []
        for node in beam:
            # Generate a few expansions from the current state
            for j in range(beam_width):
                # Mock: reduce remaining value by a random divisor
                remaining_val = target / max(1, depth * (j + 1))
                new_step = f"step{depth + 1}: reduce to {remaining_val:.1f}"
                new_steps = node.steps + [new_step]
                prm_s = prm.score_step(new_step)
                cumulative = min(node.prm_score, prm_s)
                next_beam.append(BeamNode(steps=new_steps, prm_score=cumulative))

        next_beam.sort(key=lambda n: n.prm_score, reverse=True)
        beam = next_beam[:beam_width]

    # Add a final answer step to each beam node and pick by outcome score
    best_steps: list[str] = []
    best_outcome: float = -1.0
    for node in beam:
        # Mock: last PRM-scored value as the final answer
        last_num = node.steps[-1].split()[-1]
        final_step = f"answer: {last_num}"
        full_steps = node.steps + [final_step]
        outcome = outcome_verifier.score(final_step)
        if outcome > best_outcome:
            best_outcome = outcome
            best_steps = full_steps

    return best_steps, best_outcome


# ===========================================================================
# 8. SELF-REFINE (intra-run loop)
# ===========================================================================

@dataclass
class SelfRefineResult:
    iterations: int
    final_output: str
    final_score: float
    history: list[tuple[str, str, float]]  # (output, critique, score)


def self_refine(
    initial_output: str,
    target: float,
    verifier: OutcomeVerifier,
    max_iterations: int = 3,
) -> SelfRefineResult:
    """
    Simulate the Self-Refine loop:
      Generator -> Critic -> Refiner -> repeat.

    The critic identifies the deviation from target.
    The refiner nudges the answer toward target.
    """

    def critic(output: str, target: float) -> str:
        """Mock critic: point out the numerical error."""
        ans_str = _extract_answer(output)
        val = _evaluate_expr(ans_str)
        if val is None:
            return "critique: cannot parse the answer — re-derive from scratch."
        delta = target - val
        if abs(delta) < 1e-6:
            return "critique: answer is correct — no changes needed."
        return (
            f"critique: answer {val:.1f} is off by {delta:+.1f}. "
            f"Adjust the final step to reach {target}."
        )

    def refiner(output: str, critique: str, target: float) -> str:
        """Mock refiner: apply the critique to correct the answer."""
        if "no changes" in critique:
            return output
        # Extract delta from critique and patch the answer line
        parts = critique.split("off by")
        if len(parts) == 2:
            delta_str = parts[1].split(".")[0].strip().replace("+", "")
            try:
                delta = float(delta_str)
                ans_str = _extract_answer(output)
                old_val = _evaluate_expr(ans_str) or 0.0
                new_val = old_val + delta
                lines = output.splitlines()
                lines[-1] = f"answer: {new_val:.1f}"
                return "\n".join(lines)
            except ValueError:
                pass
        return output  # fallback: no change

    history: list[tuple[str, str, float]] = []
    current = initial_output

    for i in range(max_iterations):
        score = verifier.score(current)
        crit = critic(current, target)
        history.append((current, crit, score))
        if "no changes" in crit:
            break
        current = refiner(current, crit, target)

    final_score = verifier.score(current)
    return SelfRefineResult(
        iterations=len(history),
        final_output=current,
        final_score=final_score,
        history=history,
    )


# ===========================================================================
# 9. SELF-IMPROVING AGENT WITH PERSISTENT LESSONS STORE
# ===========================================================================

LESSONS_FILE = Path(__file__).parent / "17-lessons-store.json"


@dataclass
class Lesson:
    context: str
    observation: str
    score_before: float
    score_after: float
    timestamp: str
    occurrences: int = 1
    confidence: float = 0.5


def _load_lessons(filepath: Path) -> list[Lesson]:
    """Load lessons from a JSON file.  Returns empty list if missing."""
    if not filepath.exists():
        return []
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Lesson(**item) for item in data.get("lessons", [])]


def _save_lessons(lessons: list[Lesson], filepath: Path) -> None:
    """Persist lessons to a JSON file (append-friendly, creates if missing)."""
    payload = {"lessons": [lesson.__dict__ for lesson in lessons]}
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _lessons_to_context(lessons: list[Lesson], min_confidence: float = 0.4) -> str:
    """Format high-confidence lessons as a context string for the generator."""
    relevant = [l for l in lessons if l.confidence >= min_confidence]
    if not relevant:
        return ""
    lines = ["## Lecons des runs precedents (a prioriser) :"]
    for i, lesson in enumerate(relevant[-5:], 1):  # last 5 lessons max
        lines.append(
            f"{i}. [{lesson.context}] {lesson.observation} "
            f"(confiance={lesson.confidence:.2f})"
        )
    return "\n".join(lines)


def _extract_lesson(
    target: float,
    score_before: float,
    score_after: float,
) -> Lesson | None:
    """
    Derive a lesson from run metrics.
    Only creates a lesson if there was a meaningful improvement.
    """
    if score_after - score_before < 0.1:
        return None  # no lesson worth recording

    observation = (
        f"always check factor decomposition first when target={int(target)} "
        f"(score improved from {score_before:.2f} to {score_after:.2f})"
    )
    return Lesson(
        context=f"arithmetic_target_{int(target)}",
        observation=observation,
        score_before=score_before,
        score_after=score_after,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        occurrences=1,
        confidence=0.5 + 0.1 * min(5, int(score_after * 10 - 5)),
    )


class SelfImprovingAgent:
    """
    An agent that learns across runs by persisting lessons to a JSON store.

    Each run:
      1. Loads lessons from the store.
      2. Injects lessons as extra context into the generator.
      3. Solves *n_problems* using best-of-N.
      4. Derives new lessons from the improvement observed.
      5. Saves updated lessons back to the store.

    Run this script multiple times to see the score improve progressively.
    """

    def __init__(
        self,
        lessons_file: Path = LESSONS_FILE,
        n: int = 6,
    ) -> None:
        self.lessons_file = lessons_file
        self.n = n

    def run(self, targets: list[float]) -> dict:
        """Execute one run and return a results summary."""
        lessons = _load_lessons(self.lessons_file)
        context = _lessons_to_context(lessons)

        scores_without_lessons: list[float] = []
        scores_with_lessons: list[float] = []

        for target in targets:
            verifier = OutcomeVerifier(target)

            # Score WITHOUT lessons (baseline)
            _, s_without = best_of_n(
                _mock_generator, verifier, target, n=self.n, extra_context=""
            )
            scores_without_lessons.append(s_without)

            # Score WITH lessons injected
            _, s_with = best_of_n(
                _mock_generator, verifier, target, n=self.n, extra_context=context
            )
            scores_with_lessons.append(s_with)

            # Extract and merge lessons
            lesson = _extract_lesson(target, s_without, s_with)
            if lesson:
                # Check if a similar lesson already exists (update occurrences)
                existing = next(
                    (l for l in lessons if l.context == lesson.context), None
                )
                if existing:
                    existing.occurrences += 1
                    existing.confidence = min(
                        0.99, existing.confidence + 0.05
                    )
                else:
                    lessons.append(lesson)

        _save_lessons(lessons, self.lessons_file)

        mean_without = sum(scores_without_lessons) / len(scores_without_lessons)
        mean_with = sum(scores_with_lessons) / len(scores_with_lessons)
        delta = mean_with - mean_without

        return {
            "n_lessons_loaded": len(_load_lessons(self.lessons_file)),
            "mean_score_without_lessons": round(mean_without, 4),
            "mean_score_with_lessons": round(mean_with, 4),
            "improvement_delta": round(delta, 4),
            "targets": targets,
            "scores_without": [round(s, 3) for s in scores_without_lessons],
            "scores_with": [round(s, 3) for s in scores_with_lessons],
        }


# ===========================================================================
# DEMO
# ===========================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    TARGET = 24.0
    verifier = OutcomeVerifier(TARGET)
    prm = ProcessRewardModel(TARGET)

    # --- 1. Outcome Verifier -------------------------------------------------
    _banner("1. OutcomeVerifier — score final answers")
    for answer in ["answer: 24", "answer: 20", "answer: 6 * 4"]:
        score = verifier.score(answer)
        print(f"  {answer!r:30s} -> score={score:.3f}")

    # --- 2. ProcessRewardModel -----------------------------------------------
    _banner("2. ProcessRewardModel — score individual steps")
    steps = [
        "step1: decompose 24 = 6 * 4",
        "step2: verify 6 * 4 = 24",
        "step3: done, 24 / 0 is undefined",  # bad step
    ]
    for step in steps:
        score = prm.score_step(step)
        print(f"  {step!r:45s} -> score={score:.3f}")

    # --- 3. Best-of-N --------------------------------------------------------
    _banner("3. Best-of-N (n=6, no lessons)")
    best, best_score = best_of_n(_mock_generator, verifier, TARGET, n=6)
    print(f"  Best candidate (score={best_score:.3f}):\n")
    for line in best.splitlines():
        print(f"    {line}")

    # --- 4. Weighted Majority Voting -----------------------------------------
    _banner("4. Weighted Majority Voting (n=8)")
    winner, total_score = weighted_majority(_mock_generator, verifier, TARGET, n=8)
    print(f"  Winning answer: {_extract_answer(winner)!r}  (total weighted score={total_score:.3f})")

    # --- 5. Beam Search guided by PRM ----------------------------------------
    _banner("5. Beam Search guided by PRM (beam_width=3, max_depth=3)")
    beam_steps, beam_score = beam_search(TARGET, prm, verifier, beam_width=3, max_depth=3)
    print(f"  Outcome score: {beam_score:.3f}")
    for step in beam_steps:
        print(f"    {step}")

    # --- 6. Self-Refine ------------------------------------------------------
    _banner("6. Self-Refine (intra-run, max 3 iterations)")
    initial = _mock_generator(TARGET, extra_context="")
    result = self_refine(initial, TARGET, verifier, max_iterations=3)
    print(f"  Completed in {result.iterations} iteration(s)")
    print(f"  Final score: {result.final_score:.3f}")
    print(f"  Final output:\n")
    for line in result.final_output.splitlines():
        print(f"    {line}")

    # --- 7. Self-Improving Agent ---------------------------------------------
    _banner("7. SelfImprovingAgent — persists lessons between runs")

    # Clean up lessons store before fresh demo so results are deterministic
    if LESSONS_FILE.exists():
        LESSONS_FILE.unlink()

    agent = SelfImprovingAgent(lessons_file=LESSONS_FILE, n=6)
    targets = [12.0, 24.0, 36.0, 48.0]

    print("\n  === Run 1 (no prior lessons) ===")
    r1 = agent.run(targets)
    print(f"  Lessons en store (apres run) : {r1['n_lessons_loaded']}")
    print(f"  Mean WITHOUT    : {r1['mean_score_without_lessons']:.4f}")
    print(f"  Mean WITH       : {r1['mean_score_with_lessons']:.4f}")
    print(f"  Delta           : {r1['improvement_delta']:+.4f}")

    print("\n  === Run 2 (lessons from Run 1 loaded) ===")
    r2 = agent.run(targets)
    print(f"  Lessons en store (apres run) : {r2['n_lessons_loaded']}")
    print(f"  Mean WITHOUT    : {r2['mean_score_without_lessons']:.4f}")
    print(f"  Mean WITH       : {r2['mean_score_with_lessons']:.4f}")
    print(f"  Delta           : {r2['improvement_delta']:+.4f}")

    print("\n  === Run 3 (lessons from Run 1+2 loaded) ===")
    r3 = agent.run(targets)
    print(f"  Lessons en store (apres run) : {r3['n_lessons_loaded']}")
    print(f"  Mean WITHOUT    : {r3['mean_score_without_lessons']:.4f}")
    print(f"  Mean WITH       : {r3['mean_score_with_lessons']:.4f}")
    print(f"  Delta           : {r3['improvement_delta']:+.4f}")

    # Verify improvement trend
    deltas = [r1["improvement_delta"], r2["improvement_delta"], r3["improvement_delta"]]
    print(f"\n  Improvement deltas over 3 runs: {[f'{d:+.4f}' for d in deltas]}")
    assert any(d > 0 for d in deltas), "Expected at least one positive delta across runs"
    print("\n  [OK] Self-improving agent: positive delta confirmed.")

    # Clean up lessons file after demo
    if LESSONS_FILE.exists():
        LESSONS_FILE.unlink()

    print("\nAll demos completed successfully.\n")

"""
Solutions -- Day 17 (MEDIUM): Verifiers & self-improvement

Contains solutions for:
  - Medium Ex 1: Best-of-N driven by a RUBRIC verifier (weighted multi-criteria
                 scoring); proves the selected candidate maximizes the aggregate.
  - Medium Ex 2: Self-refine loop with a MONOTONE (non-regressing) guarantee --
                 generator/critic/refiner, best-so-far guard against regressions.
  - Medium Ex 3: Verifier ENSEMBLE (format checker + content rubric + in-memory
                 unit-test runner) with a blocking-failure rule.

Self-contained: deterministic stubs only, runs OFFLINE with stdlib, zero API
keys. The in-memory unit-test runner uses exec() in an isolated namespace.

Run:  python 03-exercises/solutions/17-verifiers-self-improvement-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Shared tiny helpers (re-embedded so the file is fully self-contained)
# ---------------------------------------------------------------------------

def _safe_eval(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression; None on any failure."""
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def _extract_answer(candidate: str) -> str:
    """Pull the 'answer: X' line; fallback to the last non-empty line."""
    for line in candidate.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    lines = [ln for ln in candidate.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _answer_value(candidate: str) -> float | None:
    """Numeric value of the final answer, if parseable."""
    return _safe_eval(_extract_answer(candidate))


# ==========================================================================
# MEDIUM EXERCISE 1 -- Best-of-N driven by a rubric verifier
# ==========================================================================

@dataclass
class RubricCriterion:
    """One weighted dimension of quality. check() returns a score in [0, 1]."""
    name: str
    weight: float
    check: Callable[[str], float]


class RubricVerifier:
    """
    Weighted multi-criteria verifier. Unlike a bare outcome verifier (which only
    reads the final value), this scores PROCESS dimensions too, so two equally
    correct candidates can still be ranked.
    """

    def __init__(self, target: float) -> None:
        self.target = target
        self.criteria: list[RubricCriterion] = [
            RubricCriterion("final_correct", 0.5, self._c_final_correct),
            RubricCriterion("has_verification", 0.3, self._c_has_verification),
            RubricCriterion("no_dead_step", 0.2, self._c_no_dead_step),
        ]

    # -- criteria (each returns [0, 1]) -------------------------------------

    def _c_final_correct(self, cand: str) -> float:
        val = _answer_value(cand)
        if val is None:
            return 0.0
        return 1.0 if math.isclose(val, self.target, rel_tol=1e-9) else 0.0

    def _c_has_verification(self, cand: str) -> float:
        # Reward explicit verification steps ("verify", "check", "=").
        steps = [ln for ln in cand.splitlines() if "verify" in ln.lower() or "check" in ln.lower()]
        return 1.0 if steps else 0.0

    def _c_no_dead_step(self, cand: str) -> float:
        # Penalize dead ends: division by zero, or a step evaluating to <= 0.
        for line in cand.splitlines():
            for tok in line.replace("=", " ").split():
                if "/0" in tok.replace(" ", ""):
                    return 0.0
                val = _safe_eval(tok)
                if val is not None and val < 0:
                    return 0.0
        return 1.0

    # -- public API ---------------------------------------------------------

    def breakdown(self, candidate: str) -> dict:
        return {c.name: c.check(candidate) for c in self.criteria}

    def score(self, candidate: str) -> float:
        total_w = sum(c.weight for c in self.criteria)
        s = sum(c.weight * c.check(candidate) for c in self.criteria)
        return s / total_w if total_w else 0.0


def best_of_n_rubric(
    generator: Callable[[float, int], str],
    verifier: RubricVerifier,
    target: float,
    n: int,
) -> tuple[str, float, list[float]]:
    """Generate n candidates, score each, return (best, best_score, all_scores)."""
    candidates = [generator(target, i) for i in range(n)]
    scores = [verifier.score(c) for c in candidates]
    best_idx = max(range(n), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx], scores


def _rubric_generator(target: float, seed: int) -> str:
    """
    Deterministic candidate factory keyed by seed, producing a SPREAD of quality:
      seed 0 -> wrong final answer
      seed 1 -> correct, but NO verification step (process-poor)
      seed 2 -> correct AND verifies its steps (process-rich, should win)
      seed 3 -> correct but has a dead step (negative intermediate)
      others -> correct, no verification
    """
    t = int(target)
    if seed % 5 == 0:
        return f"step1: guess {t - 3}\nanswer: {t - 3}"           # wrong
    if seed % 5 == 1:
        return f"step1: decompose {t} = 2 * {t // 2}\nanswer: {t}"  # correct, no verify
    if seed % 5 == 2:
        return (
            f"step1: decompose {t} = 2 * {t // 2}\n"
            f"step2: verify 2 * {t // 2} = {t}\n"
            f"check: {t} == {t}\n"
            f"answer: {t}"
        )                                                          # correct + verify (best)
    if seed % 5 == 3:
        return f"step1: {t} - {t + 5} = -5\nstep2: correct sign\nanswer: {t}"  # dead step
    return f"step1: add up to {t}\nanswer: {t}"                    # correct, no verify


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Best-of-N driven by a rubric verifier")
    print("=" * 70)

    target = 24.0
    verifier = RubricVerifier(target)
    best, best_score, all_scores = best_of_n_rubric(_rubric_generator, verifier, target, n=6)

    print(f"  all scores: {[round(s, 3) for s in all_scores]}")
    print(f"  best score: {round(best_score, 3)}")
    print(f"  best breakdown: { {k: round(v,2) for k,v in verifier.breakdown(best).items()} }")
    print("  best candidate:")
    for line in best.splitlines():
        print(f"    {line}")

    # Best-of-N really returns the argmax.
    assert math.isclose(best_score, max(all_scores)), (best_score, all_scores)
    assert all(s <= best_score + 1e-12 for s in all_scores), "a candidate beat the selected one"

    # Two CORRECT candidates separated by process criteria: verify-rich beats verify-poor.
    correct_no_verify = _rubric_generator(target, 1)   # correct, no verification
    correct_verify = _rubric_generator(target, 2)      # correct, verifies
    s_poor = verifier.score(correct_no_verify)
    s_rich = verifier.score(correct_verify)
    assert verifier.breakdown(correct_no_verify)["final_correct"] == 1.0
    assert verifier.breakdown(correct_verify)["final_correct"] == 1.0
    assert s_rich > s_poor, (s_rich, s_poor)  # both correct, rubric breaks the tie

    print(f"  tie-break: correct+verify={s_rich:.3f} > correct-only={s_poor:.3f}")
    print("[Verification] PASS -- rubric argmax + process tie-break confirmed")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Self-refine with monotone (non-regressing) guarantee
# ==========================================================================

def _quality_scorer(output: str, target: float) -> float:
    """
    Quality = correctness (0.7) + presence of a verification step (0.3).
    Combines outcome AND process so refinement has room to climb.
    """
    val = _answer_value(output)
    correct = 0.7 if (val is not None and math.isclose(val, target, rel_tol=1e-9)) else 0.0
    verified = 0.3 if any("verify" in ln.lower() for ln in output.splitlines()) else 0.0
    return correct + verified


def _critic(output: str, target: float) -> str:
    """Identify the single most severe remaining defect."""
    val = _answer_value(output)
    if val is None or not math.isclose(val, target, rel_tol=1e-9):
        return "wrong_result"
    if not any("verify" in ln.lower() for ln in output.splitlines()):
        return "missing_verification"
    return "ok"


def _refiner(output: str, critique: str, target: float) -> str:
    """Apply ONE targeted fix dictated by the critique."""
    t = int(target)
    if critique == "wrong_result":
        # Re-derive correctly.
        return f"step1: decompose {t} = 2 * {t // 2}\nanswer: {t}"
    if critique == "missing_verification":
        lines = output.splitlines()
        # Insert a verification step before the answer line.
        ins = f"verify: 2 * {t // 2} = {t}"
        out = [ln for ln in lines if not ln.lower().startswith("answer:")]
        out.append(ins)
        out += [ln for ln in lines if ln.lower().startswith("answer:")]
        return "\n".join(out)
    return output  # nothing to do


def self_refine_monotone(
    initial: str,
    target: float,
    scorer: Callable[[str, float], float],
    refiner: Callable[[str, str, float], str],
    max_iters: int = 5,
    threshold: float = 1.0,
) -> dict:
    """
    Generator/critic/refiner loop with an anti-regression guard: a refined
    candidate is ACCEPTED only if it does not lower the best-so-far score.
    Stops on threshold reached OR budget exhausted.
    """
    best_output = initial
    best_score = scorer(initial, target)
    history: list[tuple[int, str, float]] = [(0, initial, best_score)]
    best_curve: list[float] = [best_score]

    for it in range(1, max_iters + 1):
        if best_score >= threshold:
            break
        critique = _critic(best_output, target)
        if critique == "ok":
            break
        candidate = refiner(best_output, critique, target)
        cand_score = scorer(candidate, target)
        history.append((it, candidate, cand_score))
        # Anti-regression: only accept if it does not lower the best score.
        if cand_score >= best_score:
            best_output, best_score = candidate, cand_score
        best_curve.append(best_score)

    return {
        "final_output": best_output,
        "final_score": best_score,
        "history": history,
        "best_curve": best_curve,
        "iterations": len(history) - 1,
        "reached_threshold": best_score >= threshold,
    }


def _harmful_refiner(output: str, critique: str, target: float) -> str:
    """A deliberately BAD refiner: it corrupts the answer (regression test)."""
    return "step1: oops\nanswer: -999"


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Self-refine with monotone (non-regressing) guarantee")
    print("=" * 70)

    target = 24.0
    # Favourable case: starts wrong, climbs to the threshold.
    initial = "step1: wild guess\nanswer: 17"
    res = self_refine_monotone(initial, target, _quality_scorer, _refiner,
                               max_iters=5, threshold=1.0)
    print(f"  best-score curve: {[round(s, 2) for s in res['best_curve']]}")
    print(f"  final score: {res['final_score']:.2f}  reached_threshold={res['reached_threshold']}")
    print("  final output:")
    for line in res["final_output"].splitlines():
        print(f"    {line}")

    # Monotone: the retained best-score sequence never decreases.
    curve = res["best_curve"]
    assert all(curve[i] <= curve[i + 1] + 1e-12 for i in range(len(curve) - 1)), curve
    # Final >= initial.
    assert res["final_score"] >= curve[0]
    # Threshold reached within budget on the favourable case.
    assert res["reached_threshold"], res["final_score"]

    # Hard case: a refiner that can't make progress still stops cleanly at budget.
    stuck = self_refine_monotone("answer: nope", target, _quality_scorer,
                                 lambda o, c, t: "answer: still-bad",
                                 max_iters=3, threshold=1.0)
    assert not stuck["reached_threshold"]
    assert stuck["iterations"] <= 3
    print(f"  stuck case: stopped at iter={stuck['iterations']} score={stuck['final_score']:.2f}")

    # Anti-regression guard: a harmful refiner must NOT degrade a good start.
    good_start = "step1: decompose 24 = 2 * 12\nverify: 2 * 12 = 24\nanswer: 24"  # score 1.0
    guarded = self_refine_monotone(good_start, target, _quality_scorer,
                                   _harmful_refiner, max_iters=3, threshold=2.0)
    assert guarded["final_score"] >= _quality_scorer(good_start, target), guarded["final_score"]
    print(f"  guard: harmful refiner could not lower score "
          f"({guarded['final_score']:.2f} kept)")
    print("[Verification] PASS -- monotone improvement, clean budget stop, guard holds")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Verifier ensemble (format + rubric + unit tests)
# ==========================================================================

# Each verifier: candidate (code string) -> (score in [0,1], is_blocking_failure)
Verifier = Callable[[str], "tuple[float, bool]"]


def format_checker(candidate: str) -> tuple[float, bool]:
    """Structural check: has a 'def f(' and a 'return'. Never blocking."""
    has_def = bool(re.search(r"def\s+\w+\s*\(", candidate))
    has_return = "return" in candidate
    score = 0.5 * has_def + 0.5 * has_return
    return score, False


def rubric_checker(candidate: str) -> tuple[float, bool]:
    """Content rubric: docstring + an edge-case guard ('if'). Never blocking."""
    has_doc = '"""' in candidate or "'''" in candidate
    has_guard = bool(re.search(r"\bif\b", candidate))
    score = 0.5 * has_doc + 0.5 * has_guard
    return score, False


def make_unittest_runner(cases: list[tuple[tuple, object]], func_name: str = "f") -> Verifier:
    """
    Build a verifier that exec()s the candidate code in an ISOLATED namespace,
    runs `cases` against the defined function, and returns
    (passed_ratio, is_blocking_failure). Any compile/run exception => blocking.
    """

    def runner(candidate: str) -> tuple[float, bool]:
        ns: dict = {}
        try:
            exec(candidate, {"__builtins__": __builtins__}, ns)  # noqa: S102
            func = ns.get(func_name)
            if not callable(func):
                return 0.0, True  # no function defined -> blocking
        except Exception:
            return 0.0, True      # compile/exec error -> blocking

        passed = 0
        crashed = False
        for args, expected in cases:
            try:
                if func(*args) == expected:
                    passed += 1
            except Exception:
                crashed = True    # a case raised -> blocking failure, caught here
        ratio = passed / len(cases) if cases else 1.0
        blocking = (passed < len(cases)) or crashed
        return ratio, blocking

    return runner


def ensemble_verdict(
    candidate: str,
    verifiers: list[tuple[str, Verifier, float]],
    threshold: float = 0.7,
) -> dict:
    """
    Aggregate heterogeneous verifiers. Accepted iff the weighted mean clears the
    threshold AND no blocking verifier failed.
    verifiers: list of (name, verifier, weight).
    """
    details: dict[str, dict] = {}
    total_w = sum(w for _, _, w in verifiers)
    agg = 0.0
    any_block = False
    for name, vf, w in verifiers:
        score, blocking = vf(candidate)
        details[name] = {"score": round(score, 3), "blocking": blocking}
        agg += w * score
        any_block = any_block or blocking
    aggregate = agg / total_w if total_w else 0.0
    accepted = (aggregate >= threshold) and (not any_block)
    return {"aggregate": round(aggregate, 3), "accepted": accepted,
            "blocked": any_block, "details": details}


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Verifier ensemble (format + rubric + unit tests)")
    print("=" * 70)

    runner = make_unittest_runner(
        [(([0],), 0), (([1, -2, 3],), 4), (([],), 0), (([5],), 5)],
        func_name="f",
    )
    verifiers = [
        ("format", format_checker, 0.2),
        ("rubric", rubric_checker, 0.2),
        ("unittests", runner, 0.6),
    ]

    correct = (
        'def f(xs):\n'
        '    """Sum of positive numbers; empty list -> 0."""\n'
        '    if not xs:\n'
        '        return 0\n'
        '    return sum(x for x in xs if x > 0)\n'
    )
    wrong = (
        'def f(xs):\n'
        '    """Looks fine but forgets the > 0 filter."""\n'
        '    if not xs:\n'
        '        return 0\n'
        '    return sum(xs)\n'   # fails on [1, -2, 3] -> 2, expected 4
    )
    crashes = (
        'def f(xs):\n'
        '    """Crashes on empty list (index error)."""\n'
        '    return xs[0] + sum(x for x in xs[1:] if x > 0)\n'
    )

    v_ok = ensemble_verdict(correct, verifiers)
    v_wrong = ensemble_verdict(wrong, verifiers)
    v_crash = ensemble_verdict(crashes, verifiers)

    for label, v in [("correct", v_ok), ("wrong", v_wrong), ("crashes", v_crash)]:
        print(f"  {label:8} -> accepted={v['accepted']} aggregate={v['aggregate']} "
              f"blocked={v['blocked']}")
        print(f"           details={v['details']}")

    # Correct code is accepted.
    assert v_ok["accepted"], v_ok
    # Well-formatted but wrong: blocking failure overrides a high format/rubric score.
    assert not v_wrong["accepted"], v_wrong
    assert v_wrong["blocked"], v_wrong
    assert v_wrong["details"]["format"]["score"] == 1.0  # format is perfect, yet rejected
    # Crashing code is rejected cleanly (ensemble itself does not crash).
    assert not v_crash["accepted"], v_crash
    assert v_crash["details"]["unittests"]["blocking"], v_crash

    print("[Verification] PASS -- ensemble blocks wrong/crashing code despite good format")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 17 MEDIUM Solutions -- Verifiers & self-improvement")
    print("  (stdlib only -- runs fully offline, no API key)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

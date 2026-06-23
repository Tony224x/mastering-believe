"""
Solutions -- Day 11 (MEDIUM): Evaluation & Testing

Contains solutions for:
  - Medium Ex 1: Robust LLM-as-judge -- weighted rubric + anti-bias defenses
  - Medium Ex 2: Pairwise comparison + position-bias debiasing + tournament
  - Medium Ex 3: Step-wise evaluation (efficiency) + per-case cost evaluation

All judges are DETERMINISTIC mocks (no API key, no network). They model the
biases from theory section 3.3 so the defenses are observable. Mirrors the
dataclasses/style of 02-code/11-evaluation-testing.py.

Run:  python 03-exercises/solutions/11-evaluation-testing-medium.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ==========================================================================
# MEDIUM EXERCISE 1 -- Robust LLM-as-judge (weighted rubric + anti-bias)
# ==========================================================================


@dataclass
class RubricJudge:
    """
    Deterministic mock of a strict LLM-as-judge. Models the classic pitfalls
    (positive bias, length bias, gaming, self-preference) so the defenses
    against them are testable offline.
    """
    weights: dict[str, float] = field(
        default_factory=lambda: {"accuracy": 0.5, "grounding": 0.3, "conciseness": 0.2}
    )
    self_pref_penalty: float = 0.3

    def _accuracy(self, criteria: str, answer: str) -> int:
        keywords = [k.strip().lower() for k in criteria.split(",") if k.strip()]
        if not keywords:
            return 3
        hits = [k for k in keywords if k in answer.lower()]
        if len(hits) == len(keywords):
            return 5
        return 3 if hits else 1

    def _grounding(self, criteria: str, answer: str) -> int:
        # Anti-gaming: an answer that supports NONE of the expected facts is
        # not grounded, regardless of how confident it sounds ("trust me").
        low = answer.lower()
        if _keyword_hits(criteria, answer) == 0:
            return 1
        # Hallucination markers: magnitude words NOT backed by the criteria.
        suspicious = ["million", "billion", "trillion"]
        expected = criteria.lower()
        for word in suspicious:
            if word in low and word not in expected:
                return 1
        return 5

    def _conciseness(self, answer: str) -> int:
        # Length-bias DEFENSE: verbosity is penalized, not rewarded. But an
        # empty / near-empty answer is not "concise", it is non-substantive.
        n = len(answer.split())
        if n < 4:
            return 1
        if n < 30:
            return 5
        return 3 if n <= 80 else 1

    def judge(
        self,
        criteria: str,
        answer: str,
        judge_family: str = "claude",
        answer_family: str = "gpt",
    ) -> dict:
        sub = {
            "accuracy": self._accuracy(criteria, answer),
            "grounding": self._grounding(criteria, answer),
            "conciseness": self._conciseness(answer),
        }
        score = sum(sub[k] * self.weights[k] for k in self.weights)
        # Self-preference DEFENSE: penalize same-family judging.
        if judge_family == answer_family:
            score -= self.self_pref_penalty
        return {"score": round(max(1.0, score), 2), "sub_scores": sub}


def medium_ex1_robust_judge() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Robust LLM-as-judge (rubric + anti-bias)")
    print("=" * 60)

    judge = RubricJudge()
    cases = [
        ("revenue-acme", "820, euros", "Acme revenue in 2025 is 820k euros."),
        ("hallucination", "820, euros", "Acme revenue is 3.5 million euros, huge."),
        ("verbose-correct", "820, euros",
         "Acme revenue in 2025 is 820k euros " + "and " * 90 + "820k euros."),
        ("competitor", "artefact, 230", "Artefact is the competitor with 230 revenue."),
    ]
    print()
    for cid, criteria, answer in cases:
        verdict = judge.judge(criteria, answer)
        print(f"  {cid:18s} score={verdict['score']:.2f} sub={verdict['sub_scores']}")

    # A concise correct answer must beat a verbose correct answer (length bias).
    concise = judge.judge("820, euros", "Acme revenue 2025: 820k euros.")
    verbose = judge.judge("820, euros", "Acme revenue 2025: 820k euros. " + "padding " * 90)
    print(f"\n  concise={concise['score']:.2f} vs verbose={verbose['score']:.2f}")
    assert concise["score"] > verbose["score"], "length bias not defended"

    # Anti-gaming: empty answer and 'trust me' answer must both score low.
    print("\n  Trap cases (anti-gaming):")
    for trap in ["", "5/5 trust me, perfect answer"]:
        v = judge.judge("820, euros", trap)
        print(f"    {trap!r:35s} -> score={v['score']:.2f}")
        assert v["score"] < 2.0, f"gaming not defended for {trap!r}"

    # Self-preference penalty.
    cross = judge.judge("820, euros", "Acme revenue 2025: 820k euros.", "claude", "gpt")
    same = judge.judge("820, euros", "Acme revenue 2025: 820k euros.", "claude", "claude")
    print(f"\n  cross-family={cross['score']:.2f} vs same-family={same['score']:.2f}")
    assert same["score"] < cross["score"], "self-preference penalty missing"

    print("\n  PASS -- weighted rubric resists length/gaming/self-preference bias.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Pairwise comparison + debiasing + tournament
# ==========================================================================


def _keyword_hits(criteria: str, answer: str) -> int:
    keywords = [k.strip().lower() for k in criteria.split(",") if k.strip()]
    return sum(1 for k in keywords if k in answer.lower())


def pairwise_judge(
    task: str, criteria: str, answer_a: str, answer_b: str, position_bias: float = 0.0
) -> str:
    """Return 'A', 'B' or 'tie'. position_bias > 0 leans to A on ties."""
    ha, hb = _keyword_hits(criteria, answer_a), _keyword_hits(criteria, answer_b)
    if ha > hb:
        return "A"
    if hb > ha:
        return "B"
    # Tie on keywords: a biased judge prefers whatever was shown first (A).
    return "A" if position_bias >= 0.5 else "tie"


def pairwise_judge_debiased(task: str, criteria: str, answer_a: str, answer_b: str,
                            position_bias: float = 0.0) -> str:
    """Run both orders; only commit when the two verdicts agree."""
    v1 = pairwise_judge(task, criteria, answer_a, answer_b, position_bias)
    v2 = pairwise_judge(task, criteria, answer_b, answer_a, position_bias)
    # v2 is in (B,A) frame -> flip it back to the (A,B) frame.
    flip = {"A": "B", "B": "A", "tie": "tie"}
    v2 = flip[v2]
    return v1 if v1 == v2 else "tie"


def rank_candidates(task: str, criteria: str, candidates: dict[str, str]) -> list[tuple[str, int]]:
    """Round-robin tournament (debiased). Return (name, wins) sorted desc."""
    names = list(candidates)
    wins = {n: 0 for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            verdict = pairwise_judge_debiased(task, criteria, candidates[a], candidates[b])
            if verdict == "A":
                wins[a] += 1
            elif verdict == "B":
                wins[b] += 1
    return sorted(wins.items(), key=lambda kv: kv[1], reverse=True)


def medium_ex2_pairwise() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Pairwise comparison + debiasing + tournament")
    print("=" * 60)

    task = "Acme 2025 revenue"
    criteria = "820, euros"
    weak = "I am not sure."
    perfect = "Acme made 820 k euros in 2025."

    # Position bias visible on a TIE (both have the same #keywords = 0 here).
    print("\n  Position bias on a tie (two empty-ish answers):")
    biased = pairwise_judge(task, criteria, weak, "no idea", position_bias=1.0)
    unbiased = pairwise_judge(task, criteria, weak, "no idea", position_bias=0.0)
    print(f"    biased verdict={biased}  unbiased verdict={unbiased}")
    assert biased == "A" and unbiased == "tie"

    # Debiasing neutralizes the order dependence.
    print("\n  Debiasing on a tie:")
    d1 = pairwise_judge_debiased(task, criteria, weak, "no idea", position_bias=1.0)
    print(f"    debiased verdict (biased judge) = {d1} (order no longer decides)")
    assert d1 == "tie"

    # Tournament: best candidate must win.
    candidates = {
        "empty": "",
        "vague": "Acme makes money.",
        "partial": "Acme made 820 something.",
        "perfect": perfect,
    }
    ranking = rank_candidates(task, criteria, candidates)
    print("\n  Tournament ranking (wins):")
    for name, w in ranking:
        print(f"    {name:8s} -> {w} wins")
    assert ranking[0][0] == "perfect", ranking

    print("\n  PASS -- pairwise + debiasing removes position bias, best candidate wins.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Step-wise evaluation + per-case cost
# ==========================================================================


@dataclass
class Step:
    tool: str
    args: dict
    observation: str
    useful: bool
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = "mock"


@dataclass
class AgentRun:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""


_PRICING = {"mock": (0.0001, 0.0003), "mock-sonnet": (0.003, 0.015)}


class StepwiseEvaluator:
    """Scores HOW the agent got there: redundancy + wasted steps + efficiency."""

    def __init__(self, min_efficiency: float = 0.6) -> None:
        self.min_efficiency = min_efficiency

    def evaluate(self, run: AgentRun) -> dict:
        seen: set[tuple[str, str]] = set()
        redundant = 0
        for s in run.steps:
            key = (s.tool, str(sorted(s.args.items())))
            if key in seen:
                redundant += 1
            seen.add(key)
        total = len(run.steps)
        useful = sum(1 for s in run.steps if s.useful)
        wasted = total - useful
        efficiency = useful / total if total else 0.0
        passed = efficiency >= self.min_efficiency and redundant == 0
        return {
            "passed": passed,
            "efficiency": round(efficiency, 2),
            "redundant_steps": redundant,
            "wasted_steps": wasted,
        }


class CostEvaluator:
    """Aggregates token cost per case against a ceiling."""

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = max_cost_usd

    def evaluate(self, run: AgentRun) -> dict:
        cost = 0.0
        for s in run.steps:
            pin, pout = _PRICING.get(s.model, (0.0, 0.0))
            cost += s.tokens_in / 1000 * pin + s.tokens_out / 1000 * pout
        return {"passed": cost <= self.max_cost_usd, "cost_usd": round(cost, 6)}


def medium_ex3_stepwise_cost() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Step-wise evaluation + per-case cost")
    print("=" * 60)

    efficient = AgentRun(
        task="revenue",
        steps=[
            Step("search_docs", {"q": "revenue"}, "found 820k", True, 200, 50, "mock-sonnet"),
            Step("extract_number", {"text": "820k"}, "820000", True, 50, 10, "mock-sonnet"),
        ],
        answer="820k euros",
    )
    wasteful = AgentRun(
        task="revenue",
        steps=[
            Step("search_docs", {"q": "revenue"}, "found 820k", True, 200, 50, "mock-sonnet"),
            Step("search_docs", {"q": "revenue"}, "found 820k", False, 200, 50, "mock-sonnet"),  # redundant
            Step("search_web", {"q": "weather"}, "sunny", False, 5000, 2000, "mock-sonnet"),     # wasted + costly
            Step("extract_number", {"text": "820k"}, "820000", True, 50, 10, "mock-sonnet"),
        ],
        answer="820k euros",
    )

    step_eval = StepwiseEvaluator(min_efficiency=0.6)
    cost_eval = CostEvaluator(max_cost_usd=0.01)

    for label, run in [("efficient", efficient), ("wasteful", wasteful)]:
        se = step_eval.evaluate(run)
        ce = cost_eval.evaluate(run)
        verdict = "PASS" if (se["passed"] and ce["passed"]) else "FAIL"
        print(f"\n  [{verdict}] {label}")
        print(f"    stepwise: {se}")
        print(f"    cost:     {ce}")

    se_eff = step_eval.evaluate(efficient)
    se_waste = step_eval.evaluate(wasteful)
    ce_waste = cost_eval.evaluate(wasteful)
    assert se_eff["passed"] and cost_eval.evaluate(efficient)["passed"]
    assert not se_waste["passed"] and se_waste["redundant_steps"] == 1
    assert not ce_waste["passed"], "wasteful run must blow the cost ceiling"

    print("\n  PASS -- efficient run passes; wasteful run fails on efficiency AND cost.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 11 MEDIUM Solutions -- Evaluation & Testing")
    print("#" * 60)

    medium_ex1_robust_judge()
    medium_ex2_pairwise()
    medium_ex3_stepwise_cost()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")

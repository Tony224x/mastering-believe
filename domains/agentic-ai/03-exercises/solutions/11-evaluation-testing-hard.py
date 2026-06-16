"""
Solutions -- Day 11 (HARD): Evaluation & Testing

Contains solutions for:
  - Hard Ex 1: CIEvalGate -- stratified pass rates, flakiness detection,
               regression budget policy (the gate you wire into CI)
  - Hard Ex 2: RagEvaluator -- faithfulness, context precision/recall,
               answer relevance (RAGAS-style) fully offline & deterministic

No API key, no network. The "non-determinism" used to exercise flakiness is a
controlled, seeded toggle so the test itself stays reproducible. Style mirrors
02-code/11-evaluation-testing.py.

Run:  python 03-exercises/solutions/11-evaluation-testing-hard.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# ==========================================================================
# HARD EXERCISE 1 -- CIEvalGate
# ==========================================================================


@dataclass
class TestCase:
    id: str
    task: str
    expected_keywords: str
    tags: list[str] = field(default_factory=list)

    @property
    def difficulty(self) -> str:
        for d in ("easy", "medium", "hard"):
            if d in self.tags:
                return d
        return "medium"


def _keyword_hits(criteria: str, answer: str) -> int:
    kws = [k.strip().lower() for k in criteria.split(",") if k.strip()]
    return sum(1 for k in kws if k in answer.lower())


@dataclass
class ScriptedAgent:
    """
    Mock agent. `answers[id]` maps a case to its answer. A case listed in
    `flaky` flips its answer on alternate runs (controlled, seeded), which the
    gate must surface even though the case may "pass on average".
    """
    answers: dict[str, str]
    flaky: set[str] = field(default_factory=set)
    _run_counter: dict[str, int] = field(default_factory=dict)

    def run(self, case: TestCase) -> str:
        n = self._run_counter.get(case.id, 0)
        self._run_counter[case.id] = n + 1
        base = self.answers.get(case.id, "I do not know.")
        if case.id in self.flaky and n % 2 == 1:
            return "I do not know."     # deterministic flip
        return base


class CIEvalGate:
    """Stratified, flakiness-aware, regression-budgeted CI gate."""

    STRATA_THRESHOLDS = {"easy": 1.0, "medium": 0.8, "hard": 0.5}

    def __init__(self, cases: list[TestCase], pass_threshold_keywords: int = 1,
                 n_runs: int = 5) -> None:
        self.cases = cases
        self.pass_threshold = pass_threshold_keywords
        self.n_runs = n_runs

    def _case_passes(self, case: TestCase, answer: str) -> bool:
        return _keyword_hits(case.expected_keywords, answer) >= self.pass_threshold

    def run(self, agent: ScriptedAgent) -> dict:
        """Run each case n_runs times. Verdict = majority; flag flakiness."""
        verdicts: dict[str, str] = {}
        flaky_cases: list[str] = []
        per_run: dict[str, list[bool]] = {}
        for case in self.cases:
            outcomes = [self._case_passes(case, agent.run(case)) for _ in range(self.n_runs)]
            per_run[case.id] = outcomes
            if len(set(outcomes)) > 1:        # verdict varies across runs
                flaky_cases.append(case.id)
            # Majority verdict.
            verdicts[case.id] = "PASS" if sum(outcomes) > self.n_runs / 2 else "FAIL"
        return {"verdicts": verdicts, "flaky": flaky_cases, "per_run": per_run}

    def per_tag_rates(self, verdicts: dict[str, str]) -> dict[str, float]:
        by_strata: dict[str, list[bool]] = {}
        for case in self.cases:
            by_strata.setdefault(case.difficulty, []).append(verdicts[case.id] == "PASS")
        return {k: round(sum(v) / len(v), 2) for k, v in by_strata.items()}

    def gate(self, baseline: dict[str, str], current_run: dict) -> dict:
        current = current_run["verdicts"]
        rates = self.per_tag_rates(current)
        blocking: list[str] = []

        # 1. Strata thresholds.
        for strata, rate in rates.items():
            threshold = self.STRATA_THRESHOLDS.get(strata, 0.0)
            if rate < threshold:
                blocking.append(f"strata '{strata}' at {rate:.0%} < required {threshold:.0%}")

        # 2. Regression budget.
        diff_by_strata = {c.id: c.difficulty for c in self.cases}
        regressions = [i for i in baseline if baseline[i] == "PASS" and current.get(i) == "FAIL"]
        fixes = [i for i in baseline if baseline[i] == "FAIL" and current.get(i) == "PASS"]
        easy_regressions = [i for i in regressions if diff_by_strata.get(i) == "easy"]
        other_regressions = [i for i in regressions if diff_by_strata.get(i) != "easy"]

        if easy_regressions:
            blocking.append(f"easy regressions (zero tolerance): {easy_regressions}")
        if len(other_regressions) > 1 or (other_regressions and len(fixes) < 2):
            blocking.append(
                f"regression budget blown: {len(other_regressions)} medium/hard "
                f"regression(s) vs {len(fixes)} fix(es)"
            )

        # 3. Flakiness is reported (warning, also blocks here for strictness).
        if current_run["flaky"]:
            blocking.append(f"flaky cases must be fixed: {current_run['flaky']}")

        return {
            "passed": not blocking,
            "per_tag_rates": rates,
            "flaky_cases": current_run["flaky"],
            "regressions": regressions,
            "fixes": fixes,
            "blocking_reasons": blocking,
        }


def _build_cases() -> list[TestCase]:
    # 5 medium cases so a single medium regression keeps the strata at 80%
    # (lets us showcase the regression-budget policy, not just the threshold).
    return [
        TestCase("e1", "revenue", "820", ["easy", "rag"]),
        TestCase("e2", "acme-immo", "real estate", ["easy", "docs"]),
        TestCase("m1", "competitor", "artefact", ["medium", "rag"]),
        TestCase("m2", "customers", "insurance", ["medium", "docs"]),
        TestCase("m3", "stack", "langgraph", ["medium", "tech"]),
        TestCase("m4", "hiring", "engineer", ["medium", "docs"]),
        TestCase("m5", "costs", "api", ["medium", "finance"]),
        TestCase("h1", "forecast", "scenario", ["hard", "reasoning"]),
    ]


def hard_ex1_ci_gate() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: CIEvalGate -- stratified + flakiness + regression budget")
    print("=" * 60)

    cases = _build_cases()
    gate = CIEvalGate(cases, n_runs=5)

    # Baseline: everything passes except h1 (hard, allowed to be 50%+).
    baseline_answers = {
        "e1": "820k euros", "e2": "real estate platform",
        "m1": "Artefact competitor", "m2": "two insurance companies",
        "m3": "uses langgraph for orchestration", "m4": "hired a data engineer",
        "m5": "main cost is api spend",
        "h1": "I do not know.",   # baseline FAIL on hard
    }
    baseline_agent = ScriptedAgent(answers=baseline_answers)
    baseline_run = gate.run(baseline_agent)
    baseline = baseline_run["verdicts"]
    print(f"\n  Baseline verdicts: {baseline}")
    print(f"  Baseline per-tag:  {gate.per_tag_rates(baseline)}")

    # Scenario A: candidate regresses an EASY case -> must BLOCK.
    print("\n  --- Scenario A: regress an easy case ---")
    cand_a = ScriptedAgent(answers={**baseline_answers, "e1": "no idea"})
    res_a = gate.gate(baseline, gate.run(cand_a))
    for r in res_a["blocking_reasons"]:
        print(f"    BLOCK: {r}")
    assert not res_a["passed"]
    assert any("easy regressions" in r for r in res_a["blocking_reasons"])

    # Scenario B: regress 1 medium but FIX 2 (h1 + a previously-failing) -> PASS.
    print("\n  --- Scenario B: 1 medium regression compensated by 2 fixes ---")
    # Make baseline have 2 failing cases to fix.
    base2_answers = {**baseline_answers, "m2": "I do not know."}  # m2 also fails in baseline
    base2 = gate.run(ScriptedAgent(answers=base2_answers))["verdicts"]
    cand_b = ScriptedAgent(answers={
        **baseline_answers,
        "m1": "no idea",                  # regress 1 medium
        "m2": "two insurance companies",  # fix
        "h1": "scenario analysis done",   # fix
    })
    res_b = gate.gate(base2, gate.run(cand_b))
    print(f"    regressions={res_b['regressions']} fixes={res_b['fixes']}")
    print(f"    per-tag={res_b['per_tag_rates']} passed={res_b['passed']}")
    print(f"    blocking={res_b['blocking_reasons']}")
    assert res_b["passed"], res_b["blocking_reasons"]

    # Scenario C: a flaky case is always reported.
    print("\n  --- Scenario C: flaky case ---")
    cand_c = ScriptedAgent(answers=baseline_answers, flaky={"m1"})
    run_c = gate.run(cand_c)
    res_c = gate.gate(baseline, run_c)
    print(f"    flaky_cases={res_c['flaky_cases']}")
    assert "m1" in res_c["flaky_cases"]
    assert any("flaky" in r for r in res_c["blocking_reasons"])

    print("\n  PASS -- strata thresholds, flakiness, regression budget all enforced.\n")


# ==========================================================================
# HARD EXERCISE 2 -- RagEvaluator (RAGAS-style, offline)
# ==========================================================================


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class RagCase:
    question: str
    answer: str
    retrieved_contexts: list[str]
    ground_truth_contexts: list[str]


class RagEvaluator:
    """Deterministic RAG metrics. No external LLM; 'support' = token overlap."""

    def __init__(self, support_threshold: float = 0.25,
                 weights: dict[str, float] | None = None) -> None:
        self.support_threshold = support_threshold
        self.weights = weights or {
            "faithfulness": 0.4, "context_precision": 0.2,
            "context_recall": 0.2, "answer_relevance": 0.2,
        }

    @staticmethod
    def extract_claims(answer: str) -> list[str]:
        return [c.strip() for c in answer.split(".") if c.strip()]

    def _supported(self, claim: str, contexts: list[str]) -> bool:
        ct = _tokens(claim)
        if not ct:
            return True
        for ctx in contexts:
            overlap = len(ct & _tokens(ctx)) / len(ct)
            if overlap >= self.support_threshold:
                return True
        return False

    def faithfulness(self, case: RagCase) -> float:
        claims = self.extract_claims(case.answer)
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if self._supported(c, case.retrieved_contexts))
        return round(supported / len(claims), 2)

    @staticmethod
    def _ctx_match(a: str, b: str) -> bool:
        # Two chunks are "the same fact" if they share enough tokens.
        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= 0.4

    def context_precision(self, case: RagCase) -> float:
        if not case.retrieved_contexts:
            return 0.0
        relevant = sum(
            1 for r in case.retrieved_contexts
            if any(self._ctx_match(r, g) for g in case.ground_truth_contexts)
        )
        return round(relevant / len(case.retrieved_contexts), 2)

    def context_recall(self, case: RagCase) -> float:
        if not case.ground_truth_contexts:
            return 1.0
        found = sum(
            1 for g in case.ground_truth_contexts
            if any(self._ctx_match(g, r) for r in case.retrieved_contexts)
        )
        return round(found / len(case.ground_truth_contexts), 2)

    def answer_relevance(self, case: RagCase) -> float:
        q, a = _tokens(case.question), _tokens(case.answer)
        # ignore generic stopwords-ish short tokens to avoid trivial overlap
        q = {t for t in q if len(t) > 3}
        if not q:
            return 1.0
        return round(len(q & a) / len(q), 2)

    def evaluate(self, case: RagCase) -> dict:
        metrics = {
            "faithfulness": self.faithfulness(case),
            "context_precision": self.context_precision(case),
            "context_recall": self.context_recall(case),
            "answer_relevance": self.answer_relevance(case),
        }
        composite = round(sum(metrics[k] * self.weights[k] for k in self.weights), 3)
        return {**metrics, "composite": composite, "passed": composite >= 0.7}


def hard_ex2_rag_eval() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: RagEvaluator -- faithfulness + precision/recall + relevance")
    print("=" * 60)

    ev = RagEvaluator()

    gt = [
        "Acme reported revenue of 820000 euros in 2025 up from 180000 in 2024",
        "Consulting accounts for 72 percent of revenue SaaS for 28 percent",
    ]

    ideal = RagCase(
        question="What was Acme revenue in 2025 and the split?",
        answer="Acme revenue in 2025 was 820000 euros. Consulting was 72 percent of revenue.",
        retrieved_contexts=gt,
        ground_truth_contexts=gt,
    )
    hallucinated = RagCase(
        question="What was Acme revenue in 2025?",
        answer="Acme revenue in 2025 was 820000 euros. Acme also acquired a competitor for 5 billion dollars.",
        retrieved_contexts=[gt[0]],
        ground_truth_contexts=[gt[0]],
    )
    bad_retriever = RagCase(
        question="What was Acme revenue in 2025 and the split?",
        answer="Acme revenue in 2025 was 820000 euros.",
        retrieved_contexts=[gt[0]],            # missing the split chunk
        ground_truth_contexts=gt,
    )

    print()
    for label, case in [("ideal", ideal), ("hallucinated", hallucinated), ("bad_retriever", bad_retriever)]:
        r = ev.evaluate(case)
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"  [{verdict}] {label:14s} faith={r['faithfulness']} "
              f"prec={r['context_precision']} recall={r['context_recall']} "
              f"relev={r['answer_relevance']} composite={r['composite']}")

    r_ideal = ev.evaluate(ideal)
    r_hall = ev.evaluate(hallucinated)
    r_bad = ev.evaluate(bad_retriever)

    assert r_ideal["faithfulness"] == 1.0 and r_ideal["context_recall"] == 1.0
    assert r_ideal["passed"]
    assert r_hall["faithfulness"] < 1.0, "hallucinated claim must drop faithfulness"
    assert r_bad["context_recall"] < 1.0, "missing chunk must drop recall"

    print("\n  PASS -- hallucination caught by faithfulness, retriever gap by recall.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 11 HARD Solutions -- Evaluation & Testing")
    print("#" * 60)

    hard_ex1_ci_gate()
    hard_ex2_rag_eval()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

"""
Day 14 -- CAPSTONE solutions.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/14-capstone.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day14 = import_module("14-capstone")
MockLLM = day14.MockLLM
TinyTfIdfRetriever = day14.TinyTfIdfRetriever
InputGuardrail = day14.InputGuardrail
OutputGuardrail = day14.OutputGuardrail
HITLGate = day14.HITLGate
CANARY_TOKEN = day14.CANARY_TOKEN
CORPUS = day14.CORPUS
KaliraResearcher = day14.KaliraResearcher
KaliraState = day14.KaliraState
BudgetTracker = day14.BudgetTracker
BudgetExceeded = day14.BudgetExceeded
Flag = day14.Flag
SupervisorAgent = day14.SupervisorAgent
ResearcherAgent = day14.ResearcherAgent
AnalyzerAgent = day14.AnalyzerAgent
WriterAgent = day14.WriterAgent
EvalCase = day14.EvalCase
build_eval_cases = day14.build_eval_cases
run_eval = day14.run_eval
traced = day14.traced
Tracer = day14.Tracer


# ===========================================================================
# SOLUTION 1 -- CriticAgent
# ===========================================================================

class CriticMockLLM(MockLLM):
    def critique(self, draft: str) -> tuple[dict, dict]:
        self.call_count += 1
        issues: list[str] = []
        if len(draft) < 200:
            issues.append("too short")
        if "##" not in draft:
            issues.append("missing sections")
        if not any(ch.isdigit() for ch in draft):
            issues.append("missing numbers")
        revised = draft + f"\n\n[Critic revision: {len(issues)} issues addressed]"
        return (
            {"issues": issues, "revised": revised},
            self._charge_meta(60, 40),
        )


@dataclass
class CriticAgent:
    llm: CriticMockLLM

    @traced("critic.run")
    def run(self, state: KaliraState) -> KaliraState:
        report, meta = self.llm.critique(state.draft_report)
        state.budget.charge(meta["model"], meta["tokens_in"], meta["tokens_out"])
        # We stash the critique on an ad-hoc attribute for demo purposes
        state.draft_report = report["revised"]
        setattr(state, "critic_report", report)
        return state


class KaliraResearcherWithCritic(KaliraResearcher):
    """Subclass that inserts a CriticAgent between Writer and Guardrail."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Upgrade llm to critic-aware
        if not isinstance(self.llm, CriticMockLLM):
            raise ValueError("llm must be a CriticMockLLM for this subclass")
        self.critic = CriticAgent(llm=self.llm)

    def run(self, user_id: str, query: str) -> KaliraState:
        trace_id = day14._GLOBAL_TRACER.start_trace()
        state = KaliraState(
            user_id=user_id,
            query=query,
            trace_id=trace_id,
            budget=BudgetTracker(max_cost_usd=self.max_budget_usd),
        )
        ok, flags = self.input_guard.scan(query)
        state.flags.extend(flags)
        if not ok:
            state.verdict = "blocked_input"
            return state
        try:
            state = self.supervisor.plan(state)
            state = self.researcher.run(state)
            state = self.analyzer.run(state)
            state = self.writer.run(state)
            state = self.critic.run(state)
        except BudgetExceeded as exc:
            state.verdict = "budget_exceeded"
            state.flags.append(Flag("budget", str(exc), "block"))
            return state
        out_ok, out_flags = self.output_guard.scan(state.draft_report)
        state.flags.extend(out_flags)
        if not out_ok:
            state.verdict = "blocked_output"
            return state
        if not self.hitl.approve(state.draft_report):
            state.verdict = "rejected_by_human"
            return state
        state.final_report = state.draft_report
        state.verdict = "ok"
        return state


def solution_1() -> None:
    print("\n=== Solution 1: CriticAgent ===")
    llm = CriticMockLLM()
    day14._GLOBAL_TRACER = Tracer()
    researcher = KaliraResearcherWithCritic(
        llm=llm,
        retriever=TinyTfIdfRetriever(CORPUS),
        input_guard=InputGuardrail(),
        output_guard=OutputGuardrail(canary=CANARY_TOKEN),
        hitl=HITLGate(approver=lambda content: True),
        max_budget_usd=0.50,
    )
    state = researcher.run("anthony", "What is the revenue of Kalira in 2025?")
    print(f"  verdict={state.verdict}")
    assert "[Critic revision" in state.final_report
    critic_span_names = [s.name for s in day14._GLOBAL_TRACER.spans if s.name == "critic.run"]
    print(f"  critic spans: {len(critic_span_names)}")
    assert critic_span_names, "critic span should be recorded"


# ===========================================================================
# SOLUTION 2 -- Swarm-based researcher
# ===========================================================================

@dataclass
class SwarmKaliraResearcher:
    """Alternative to KaliraResearcher with explicit handoffs, no supervisor."""

    llm: MockLLM
    retriever: TinyTfIdfRetriever
    input_guard: InputGuardrail
    output_guard: OutputGuardrail
    hitl: HITLGate
    max_budget_usd: float = 0.50

    def __post_init__(self) -> None:
        self.researcher = ResearcherAgent(llm=self.llm, retriever=self.retriever)
        self.analyzer = AnalyzerAgent(llm=self.llm)
        self.writer = WriterAgent(llm=self.llm)

    def _handoff(self, current: str) -> str | None:
        chain = {"researcher": "analyzer", "analyzer": "writer", "writer": None}
        return chain.get(current)

    def _run_agent(self, name: str, state: KaliraState) -> KaliraState:
        if name == "researcher":
            return self.researcher.run(state)
        if name == "analyzer":
            return self.analyzer.run(state)
        if name == "writer":
            return self.writer.run(state)
        raise ValueError(name)

    def run(self, user_id: str, query: str) -> KaliraState:
        trace_id = day14._GLOBAL_TRACER.start_trace()
        state = KaliraState(
            user_id=user_id,
            query=query,
            trace_id=trace_id,
            budget=BudgetTracker(max_cost_usd=self.max_budget_usd),
        )
        ok, flags = self.input_guard.scan(query)
        state.flags.extend(flags)
        if not ok:
            state.verdict = "blocked_input"
            return state
        try:
            current: str | None = "researcher"
            while current is not None:
                state = self._run_agent(current, state)
                current = self._handoff(current)
        except BudgetExceeded as exc:
            state.verdict = "budget_exceeded"
            state.flags.append(Flag("budget", str(exc), "block"))
            return state
        out_ok, out_flags = self.output_guard.scan(state.draft_report)
        state.flags.extend(out_flags)
        if not out_ok:
            state.verdict = "blocked_output"
            return state
        if not self.hitl.approve(state.draft_report):
            state.verdict = "rejected_by_human"
            return state
        state.final_report = state.draft_report
        state.verdict = "ok"
        return state


def solution_2() -> None:
    print("\n=== Solution 2: Swarm researcher ===")
    day14._GLOBAL_TRACER = Tracer()
    llm = MockLLM()
    researcher = SwarmKaliraResearcher(
        llm=llm,
        retriever=TinyTfIdfRetriever(CORPUS),
        input_guard=InputGuardrail(),
        output_guard=OutputGuardrail(canary=CANARY_TOKEN),
        hitl=HITLGate(approver=lambda content: True),
        max_budget_usd=0.50,
    )
    state = researcher.run("anthony", "What is the revenue of Kalira in 2025?")
    print(f"  verdict={state.verdict}")
    # Verify no supervisor spans
    sup_spans = [s for s in day14._GLOBAL_TRACER.spans if s.name == "supervisor.plan"]
    print(f"  supervisor spans: {len(sup_spans)} (should be 0)")
    assert len(sup_spans) == 0
    assert state.verdict == "ok"


# ===========================================================================
# SOLUTION 3 -- out-of-domain eval case
# ===========================================================================

STOPWORDS = {
    "what", "is", "the", "of", "a", "an", "and", "or", "in", "on", "at",
    "to", "for", "by", "with", "who", "where", "when", "how", "which",
    "are", "was", "were", "be", "been", "do", "does", "did", "has", "have",
    "this", "that", "these", "those", "its",
}


def _content_tokens(text: str) -> set[str]:
    """Tokens minus stopwords and short tokens -- captures the real topic words."""
    import re as _re
    return {
        t for t in _re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    }


class OODMockLLM(MockLLM):
    """MockLLM that gracefully degrades when findings do not cover the query topic."""

    def write(self, query, findings, analysis):  # type: ignore[override]
        self.call_count += 1
        query_topic = _content_tokens(query)
        if not findings or not query_topic:
            weak = True
        else:
            # Check whether any topic word from the query appears in the findings
            finding_text = " ".join(f["excerpt"].lower() for f in findings)
            weak = not any(t in finding_text for t in query_topic)
        if weak:
            report = (
                "I do not have enough information in the internal corpus to "
                f"answer: {query}. No relevant documents were found."
            )
            return report, self._charge_meta(20, 30)
        return super().write(query, findings, analysis)


def build_extended_cases() -> list[EvalCase]:
    cases = build_eval_cases()
    cases.append(
        EvalCase(
            id="out-of-domain",
            query="What is the capital of Mars?",
            expected_keywords="",
            expected_verdict="ok",
        )
    )
    return cases


def extended_run_eval(researcher: KaliraResearcher) -> list[dict]:
    results: list[dict] = []
    for case in build_extended_cases():
        state = researcher.run("eval-user", case.query)
        verdict_ok = state.verdict == case.expected_verdict
        low = state.final_report.lower() if state.final_report else ""
        keyword_hits = all(k.strip().lower() in low for k in case.expected_keywords.split(",") if k.strip())
        graceful = "do not have" in low or "not found" in low
        if case.expected_keywords:
            passed = verdict_ok and keyword_hits
        else:
            # Out-of-domain case: either the agent was blocked or it gracefully degraded
            passed = verdict_ok and (graceful or state.verdict != "ok")
        results.append({
            "id": case.id,
            "verdict": state.verdict,
            "passed": passed,
            "graceful": graceful,
        })
    return results


def solution_3() -> None:
    print("\n=== Solution 3: out-of-domain case ===")
    day14._GLOBAL_TRACER = Tracer()
    llm = OODMockLLM()
    researcher = KaliraResearcher(
        llm=llm,
        retriever=TinyTfIdfRetriever(CORPUS),
        input_guard=InputGuardrail(),
        output_guard=OutputGuardrail(canary=CANARY_TOKEN),
        hitl=HITLGate(approver=lambda content: True),
        max_budget_usd=0.50,
    )
    results = extended_run_eval(researcher)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        tag = " out_of_domain_handled" if r["graceful"] else ""
        print(f"  [{status}] {r['id']:18} verdict={r['verdict']:14}{tag}")
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    print(f"  pass rate: {pass_rate:.0%}")
    assert pass_rate == 1.0, "expected 100 percent pass rate"


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

"""
Solutions -- Day 14 (MEDIUM): Capstone extensions (AcmeResearcher)

Contains solutions for:
  - Medium Ex 1: Semantic findings cache + cross-request reuse
  - Medium Ex 2: Complexity-adaptive budget + early-exit
  - Medium Ex 3: Verifiable citations + uncited-claim detection

These EXTEND the capstone. To run fully offline & standalone, this file embeds
a faithful mini-AcmeResearcher (corpus + TF-IDF retriever + MockLLM + budget +
agents), mirroring 02-code/14-capstone.py without importing it.

Run:  python 03-exercises/solutions/14-capstone-medium.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ==========================================================================
# EMBEDDED MINI-CAPSTONE (offline stand-in for 02-code/14-capstone.py)
# ==========================================================================

CORPUS: list[tuple[str, str, str]] = [
    ("k1", "company", "Acme is an AI consulting and SaaS studio founded in 2024 by Alex."),
    ("k2", "product", "acme-immo is Acme's flagship SaaS for real estate price transparency, launched in Conakry."),
    ("k4", "finance", "Acme reported revenue of 820000 euros in 2025, up from 180000 in 2024."),
    ("k5", "market", "The main competitor is Artefact, which reported 230 million euros in revenue."),
    ("k9", "customer", "Main customers in France: two mid-size insurance companies and a law firm network."),
    ("k10", "finance", "Acme's main cost line in 2025 was LLM API spend at 90000 euros."),
]

_STOP = {"the", "of", "in", "is", "a", "and", "what", "who", "are", "to", "for", "was"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _significant(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOP and len(t) > 2}


class TinyTfIdfRetriever:
    def __init__(self, docs: list[tuple[str, str, str]]) -> None:
        self.docs = docs
        self._tf = [Counter(_tokenize(t)) for _, _, t in docs]
        self._df: Counter = Counter()
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        return math.log((1 + self._n) / (1 + self._df.get(term, 0))) + 1.0

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, str, str, float]]:
        q = _tokenize(query)
        scored = []
        for i, (doc_id, source, text) in enumerate(self.docs):
            s = sum(self._tf[i].get(t, 0) * self._idf(t) for t in q)
            if s > 0:
                scored.append((doc_id, source, text, s))
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]


MODEL_PRICING = {"mock-sonnet": (0.003, 0.015)}


def compute_cost(model: str, tin: int, tout: int) -> float:
    pin, pout = MODEL_PRICING.get(model, (0.0, 0.0))
    return tin / 1000 * pin + tout / 1000 * pout


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetTracker:
    max_cost_usd: float
    current_cost_usd: float = 0.0

    def charge(self, tin: int, tout: int, model: str = "mock-sonnet") -> float:
        cost = compute_cost(model, tin, tout)
        self.current_cost_usd += cost
        if self.current_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"budget {self.max_cost_usd}$ exceeded at {self.current_cost_usd:.4f}$")
        return cost


class MockLLM:
    def research(self, query, chunks):
        findings = [{"doc_id": d, "source": s, "excerpt": t[:120], "relevance": round(sc, 2)}
                    for d, s, t, sc in chunks]
        return findings, (40, 60)

    def analyze(self, findings):
        return {"insights": [f"Insight {f['doc_id']}" for f in findings[:3]],
                "conclusion": "Acme shows momentum."}, (80, 40)

    def write(self, query, findings, analysis):
        ctx = "\n".join(f"- {f['doc_id']}: {f['excerpt'][:60]}" for f in findings[:3])
        report = f"# {query}\n{ctx}\n## Conclusion\n{analysis['conclusion']}"
        return report, (120, 200)


# ==========================================================================
# MEDIUM EXERCISE 1 -- Semantic findings cache
# ==========================================================================


class FindingsCache:
    """Reuse findings across semantically-close queries (token overlap mock)."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self._store: list[tuple[set[str], str, list[dict]]] = []

    def lookup(self, query: str) -> list[dict] | None:
        q = _significant(query)
        if not q:
            return None
        for toks, _orig, findings in self._store:
            overlap = len(q & toks) / len(q | toks) if (q | toks) else 0.0
            if overlap >= self.threshold:
                return findings
        return None

    def store(self, query: str, findings: list[dict]) -> None:
        self._store.append((_significant(query), query, findings))


@dataclass
class CachedResearcher:
    llm: MockLLM
    retriever: TinyTfIdfRetriever
    cache: FindingsCache

    def run(self, query: str, budget: BudgetTracker) -> dict:
        state: dict[str, Any] = {"query": query}
        cached = self.cache.lookup(query)
        if cached is not None:
            state["cache_status"] = "hit"
            state["findings"] = cached
        else:
            state["cache_status"] = "miss"
            chunks = self.retriever.search(query, top_k=3)
            findings, (tin, tout) = self.llm.research(query, chunks)
            budget.charge(tin, tout)            # researcher cost only on miss
            state["findings"] = findings
            self.cache.store(query, findings)
        # analyze + write run either way (cheap relative to research here)
        analysis, (tin, tout) = self.llm.analyze(state["findings"])
        budget.charge(tin, tout)
        report, (tin, tout) = self.llm.write(query, state["findings"], analysis)
        budget.charge(tin, tout)
        state["cost"] = round(budget.current_cost_usd, 5)
        state["report"] = report
        return state


def medium_ex1_findings_cache() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Semantic findings cache")
    print("=" * 60)

    retr = TinyTfIdfRetriever(CORPUS)
    cache = FindingsCache(threshold=0.5)
    researcher = CachedResearcher(MockLLM(), retr, cache)

    queries = [
        "What is the revenue of Acme in 2025?",
        "Acme 2025 revenue?",                 # semantically close -> hit
        "Who are Acme customers?",            # different -> miss
    ]
    costs = []
    print()
    for q in queries:
        b = BudgetTracker(max_cost_usd=1.0)
        st = researcher.run(q, b)
        costs.append(st["cost"])
        print(f"  {q:38s} -> cache={st['cache_status']:5s} cost=${st['cost']}")

    # 2nd query is a hit -> cheaper than the 1st (researcher skipped).
    assert costs[1] < costs[0], (costs[0], costs[1])

    # Cross-check: total cost WITH cache (close 2nd query) < WITHOUT cache.
    no_cache_total = 0.0
    for q in [queries[0], queries[1]]:
        r = CachedResearcher(MockLLM(), retr, FindingsCache(threshold=0.5))  # fresh per query
        no_cache_total += r.run(q, BudgetTracker(1.0))["cost"]
    cache_shared = FindingsCache(threshold=0.5)
    r_shared = CachedResearcher(MockLLM(), retr, cache_shared)
    with_cache_total = sum(
        r_shared.run(q, BudgetTracker(1.0))["cost"] for q in [queries[0], queries[1]]
    )
    print(f"\n  total cost (no cache)={no_cache_total:.5f} vs (shared cache)={with_cache_total:.5f}")
    assert with_cache_total < no_cache_total

    print("\n  PASS -- close query hits cache, researcher skipped, cost reduced.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Complexity-adaptive budget + early-exit
# ==========================================================================


def complexity_estimator(query: str) -> str:
    low = query.lower()
    signals = low.count(" and ") + query.count(",")
    deep_words = any(w in low for w in ("compare", "analyze", "forecast", "trade-off"))
    if deep_words or signals >= 3 or len(query) > 120:
        return "deep"
    if signals >= 1 or len(query) > 60:
        return "composed"
    return "simple"


COMPLEXITY_BUDGET = {"simple": 0.10, "composed": 0.30, "deep": 0.60}


@dataclass
class AdaptiveResearcher:
    llm: MockLLM
    retriever: TinyTfIdfRetriever

    def run(self, query: str) -> dict:
        complexity = complexity_estimator(query)
        allocated = COMPLEXITY_BUDGET[complexity]
        budget = BudgetTracker(max_cost_usd=allocated)
        state: dict[str, Any] = {"query": query, "complexity": complexity,
                                 "allocated_budget": allocated, "early_exit": False,
                                 "verdict": "ok"}
        try:
            chunks = self.retriever.search(query, top_k=3)
            findings, (tin, tout) = self.llm.research(query, chunks)
            budget.charge(tin, tout)
            analysis, (tin, tout) = self.llm.analyze(findings)
            budget.charge(tin, tout)

            # Early-exit: if top finding is highly relevant, skip any re-search.
            confidence = findings[0]["relevance"] if findings else 0.0
            if complexity == "simple" and confidence >= 1.0:
                state["early_exit"] = True
            else:
                # A 'deep' query would do an extra retrieval pass here.
                if complexity == "deep":
                    extra, (tin, tout) = self.llm.research(query + " details", chunks)
                    budget.charge(tin, tout)

            report, (tin, tout) = self.llm.write(query, findings, analysis)
            budget.charge(tin, tout)
            state["report"] = report
        except BudgetExceeded as exc:
            state["verdict"] = "budget_exceeded"
            state["error"] = str(exc)
        state["cost"] = round(budget.current_cost_usd, 5)
        return state


def medium_ex2_adaptive_budget() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Complexity-adaptive budget + early-exit")
    print("=" * 60)

    researcher = AdaptiveResearcher(MockLLM(), TinyTfIdfRetriever(CORPUS))
    queries = [
        "Acme revenue?",                                              # simple
        "What is acme-immo and where was it launched?",              # composed
        "Compare Acme consulting and SaaS revenue and analyze trade-offs",  # deep
    ]
    print()
    seen_complexities = set()
    for q in queries:
        st = researcher.run(q)
        seen_complexities.add(st["complexity"])
        print(f"  [{st['complexity']:8s}] budget=${st['allocated_budget']} "
              f"cost=${st['cost']} early_exit={st['early_exit']} verdict={st['verdict']}")
        assert st["verdict"] != "budget_exceeded", f"{q} blew its budget"
        assert st["cost"] <= st["allocated_budget"] + 1e-9

    assert len(seen_complexities) == 3, "all three complexity classes must appear"
    simple_state = researcher.run("Acme revenue?")
    assert simple_state["early_exit"] is True, "simple confident query should early-exit"

    print("\n  PASS -- budget scales with complexity, simple query early-exits, no overrun.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Verifiable citations
# ==========================================================================


class CitingLLM(MockLLM):
    """Writer variant that attaches [doc_id] citations to claims."""

    def write_cited(self, query, findings, inject_phantom=False, drop_citation=False):
        ids = [f["doc_id"] for f in findings]
        lines = ["# Report"]
        # One cited claim per finding.
        for f in findings:
            lines.append(f"Acme fact from {f['source']} is documented [{f['doc_id']}].")
        # Add a numeric claim, optionally uncited / phantom.
        if drop_citation:
            lines.append("Acme made 820000 euros in 2025.")           # NO citation
        elif inject_phantom:
            lines.append("Acme made 820000 euros in 2025 [k99].")     # phantom id
        else:
            cited = ids[0] if ids else "k4"
            lines.append(f"Acme made 820000 euros in 2025 [{cited}].")
        lines.append("## Conclusion: Acme shows momentum.")
        return "\n".join(lines)


class CitationEvaluator:
    """Verifies every claim is grounded by a doc_id that exists in findings."""

    CIT = re.compile(r"\[([a-z0-9]+)\]")
    FACT = re.compile(r"\d|[A-Z][a-z]+")    # has a number or a proper-noun-ish token

    def evaluate(self, report: str, findings: list[dict]) -> dict:
        valid_ids = {f["doc_id"] for f in findings}
        cited_claims = uncited_claims = 0
        phantom: list[str] = []
        for line in report.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cites = self.CIT.findall(line)
            is_factual = bool(self.FACT.search(self.CIT.sub("", line)))
            if cites:
                cited_claims += 1
                phantom.extend(c for c in cites if c not in valid_ids)
            elif is_factual:
                uncited_claims += 1
        total = cited_claims + uncited_claims
        grounding_ratio = round(cited_claims / total, 2) if total else 1.0
        passed = grounding_ratio >= 0.8 and not phantom
        return {"cited_claims": cited_claims, "uncited_claims": uncited_claims,
                "phantom_citations": phantom, "grounding_ratio": grounding_ratio,
                "passed": passed}


def medium_ex3_citations() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Verifiable citations + uncited-claim detection")
    print("=" * 60)

    retr = TinyTfIdfRetriever(CORPUS)
    chunks = retr.search("Acme revenue 2025", top_k=3)
    findings = [{"doc_id": d, "source": s, "excerpt": t[:80]} for d, s, t, _ in chunks]
    llm = CitingLLM()
    ev = CitationEvaluator()

    print()
    good = ev.evaluate(llm.write_cited("revenue", findings), findings)
    print(f"  well-sourced  : {good}")
    assert good["passed"] and not good["phantom_citations"]

    phantom = ev.evaluate(llm.write_cited("revenue", findings, inject_phantom=True), findings)
    print(f"  phantom cite  : {phantom}")
    assert phantom["phantom_citations"] == ["k99"] and not phantom["passed"]

    uncited = ev.evaluate(llm.write_cited("revenue", findings, drop_citation=True), findings)
    print(f"  uncited claim : {uncited}")
    assert uncited["uncited_claims"] >= 1 and not uncited["passed"]

    print("\n  PASS -- phantom citations and uncited factual claims both caught.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 14 MEDIUM Solutions -- Capstone extensions")
    print("#" * 60)

    medium_ex1_findings_cache()
    medium_ex2_adaptive_budget()
    medium_ex3_citations()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")

"""
Day 8 -- Solutions to the easy and medium exercises for Agentic RAG.

Easy:   solution_1 (3-level grader), solution_2 (LLM budget), solution_3 (multi-hop)
Medium: solution_m1 (grading-driven reformulation loop),
        solution_m2 (hybrid retrieval + Reciprocal Rank Fusion),
        solution_m3 (cited synthesis + faithfulness check)

Run the whole file to execute every solution with its test scenario.

    python domains/agentic-ai/03-exercises/solutions/08-rag-agentique.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the shared code importable (tiny retriever + MockLLM from day 8 code)
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
from importlib import import_module

day8 = import_module("08-rag-agentique")
TinyTfIdfRetriever = day8.TinyTfIdfRetriever
MockLLM = day8.MockLLM
AgenticRAG = day8.AgenticRAG
AgenticRAGConfig = day8.AgenticRAGConfig
SubQuestionTrace = day8.SubQuestionTrace
CORPUS = list(day8.CORPUS)
_tokenize = day8._tokenize


# ===========================================================================
# SOLUTION 1 -- 3-level retrieval grader
# ===========================================================================

class ThreeLevelMockLLM(MockLLM):
    """MockLLM overriding grade to return HIGHLY / PARTIALLY / IRRELEVANT."""

    def _grade(self, query: str, chunk: str) -> str:  # type: ignore[override]
        q_terms = {t for t in _tokenize(query) if len(t) > 3}
        c_terms = set(_tokenize(chunk))
        overlap = len(q_terms & c_terms)
        query_wants_number = any(
            w in query.lower() for w in ["revenue", "ca", "how much", "price"]
        )
        chunk_has_number = bool(re.search(r"\d", chunk))
        if query_wants_number and not chunk_has_number:
            return "IRRELEVANT"
        if overlap >= 4 and (not query_wants_number or chunk_has_number):
            return "HIGHLY_RELEVANT"
        if overlap >= 2:
            return "PARTIALLY_RELEVANT"
        return "IRRELEVANT"


@dataclass
class ThreeLevelTrace(SubQuestionTrace):
    highly_relevant_count: int = 0
    partially_relevant_count: int = 0


class ThreeLevelAgenticRAG(AgenticRAG):
    """Prioritize HIGHLY_RELEVANT chunks, top up with PARTIALLY_RELEVANT if sparse."""

    def _answer_sub_question(self, sub_question: str, trace: SubQuestionTrace) -> list[str]:  # type: ignore[override]
        sources = self.llm("route", {"query": sub_question})["sources"]
        trace.sources_used = sources
        self._log(f"  [sub] {sub_question}  sources={sources}")

        current = sub_question
        highly: list[str] = []
        partially: list[str] = []
        for attempt in range(self.config.max_retries + 1):
            trace.attempts = attempt + 1
            hits = self.retriever.search(
                current, top_k=self.config.top_k, allowed_sources=sources
            )
            for doc_id, source, text, _score in hits:
                verdict = self.llm("grade", {"query": sub_question, "chunk": text})["verdict"]
                label = f"{doc_id} ({source})"
                if verdict == "HIGHLY_RELEVANT":
                    highly.append(text)
                    trace.kept_chunks.append(label)
                elif verdict == "PARTIALLY_RELEVANT":
                    partially.append(text)
                    trace.kept_chunks.append(label)
                else:
                    trace.dropped_chunks.append(label)

            if highly:
                break  # Good enough, stop retrying
            if attempt < self.config.max_retries:
                current = self.llm("reformulate", {"query": current})["query"]

        # Expose counts on the trace if it is a ThreeLevelTrace
        if isinstance(trace, ThreeLevelTrace):
            trace.highly_relevant_count = len(highly)
            trace.partially_relevant_count = len(partially)

        # Keep at most 2 chunks: prefer HIGHLY, top up with PARTIALLY
        final = highly[:2]
        if len(final) < 2:
            final.extend(partially[: 2 - len(final)])
        return final


def solution_1() -> None:
    print("\n=== Solution 1: 3-level grader ===")
    retriever = TinyTfIdfRetriever(CORPUS)
    llm = ThreeLevelMockLLM()
    rag = ThreeLevelAgenticRAG(retriever, llm, AgenticRAGConfig(verbose=False))

    # Monkey-patch trace factory to use the extended trace dataclass
    original_answer = rag.answer

    def answer_with_three_level_traces(q: str) -> str:
        rag.traces = []
        sub_questions = llm("decompose", {"query": q})["sub_questions"]
        all_context: list[str] = []
        for sq in sub_questions:
            trace = ThreeLevelTrace(sub_question=sq)
            rag.traces.append(trace)
            kept = rag._answer_sub_question(sq, trace)
            all_context.extend(kept)
            trace.answered = bool(kept)
        return llm("synthesize", {"query": q, "context": all_context})["answer"]

    rag.answer = answer_with_three_level_traces  # type: ignore

    q = "What is the revenue of Acme in 2025 compared to its main competitor?"
    print(rag.answer(q))
    for t in rag.traces:
        assert isinstance(t, ThreeLevelTrace)
        print(
            f"  {t.sub_question}  "
            f"highly={t.highly_relevant_count}  "
            f"partially={t.partially_relevant_count}"
        )


# ===========================================================================
# SOLUTION 2 -- LLM call budget
# ===========================================================================

class BudgetedAgenticRAG(AgenticRAG):
    """AgenticRAG variant with a hard ceiling on LLM calls."""

    def __init__(
        self,
        retriever: TinyTfIdfRetriever,
        llm: MockLLM,
        config: AgenticRAGConfig | None = None,
        llm_budget: int = 20,
    ) -> None:
        super().__init__(retriever, llm, config)
        self.llm_budget = llm_budget
        self._llm_calls = 0
        self._warned_80 = False

    def _call_llm(self, task: str, payload: dict) -> dict:
        self._llm_calls += 1
        # Warning at 80 percent
        if not self._warned_80 and self._llm_calls >= int(0.8 * self.llm_budget):
            self._warned_80 = True
            print(
                f"[AgenticRAG] warning: {self._llm_calls}/{self.llm_budget} LLM calls used"
            )
        if self._llm_calls > self.llm_budget:
            raise RuntimeError("LLM budget exceeded")
        return self.llm(task, payload)

    # Override everywhere the base class calls self.llm(...) so our helper is used.
    def answer(self, query: str) -> str:
        self._llm_calls = 0
        self._warned_80 = False
        self.traces = []
        sub_questions = self._call_llm("decompose", {"query": query})["sub_questions"]
        sub_questions = sub_questions[: self.config.max_sub_questions]
        all_context: list[str] = []
        for sq in sub_questions:
            trace = SubQuestionTrace(sub_question=sq)
            self.traces.append(trace)
            kept = self._answer_sub_question(sq, trace)
            all_context.extend(kept)
            trace.answered = bool(kept)
        final = self._call_llm("synthesize", {"query": query, "context": all_context})["answer"]
        return final

    def _answer_sub_question(self, sub_question: str, trace: SubQuestionTrace) -> list[str]:  # type: ignore[override]
        sources = self._call_llm("route", {"query": sub_question})["sources"]
        trace.sources_used = sources
        current = sub_question
        for attempt in range(self.config.max_retries + 1):
            trace.attempts = attempt + 1
            hits = self.retriever.search(
                current, top_k=self.config.top_k, allowed_sources=sources
            )
            kept: list[str] = []
            for doc_id, source, text, _score in hits:
                verdict = self._call_llm("grade", {"query": sub_question, "chunk": text})["verdict"]
                if verdict == "RELEVANT":
                    kept.append(text)
                    trace.kept_chunks.append(f"{doc_id} ({source})")
                else:
                    trace.dropped_chunks.append(f"{doc_id} ({source})")
            if kept:
                return kept
            if attempt < self.config.max_retries:
                current = self._call_llm("reformulate", {"query": current})["query"]
        return []

    def explain(self) -> str:
        base = super().explain()
        return base + f"\n  total LLM calls: {self._llm_calls}/{self.llm_budget}"


def solution_2() -> None:
    print("\n=== Solution 2: LLM call budget ===")
    retriever = TinyTfIdfRetriever(CORPUS)
    llm = MockLLM()

    # Tight budget -- should raise
    rag_tight = BudgetedAgenticRAG(
        retriever, llm, AgenticRAGConfig(verbose=False), llm_budget=5
    )
    try:
        rag_tight.answer(
            "What is the revenue of Acme in 2025 compared to its main competitor "
            "and what is the webhook API status?"
        )
    except RuntimeError as exc:
        print(f"  tight budget: correctly raised -> {exc}")

    # Normal budget -- should succeed
    rag_normal = BudgetedAgenticRAG(
        retriever, llm, AgenticRAGConfig(verbose=False), llm_budget=40
    )
    rag_normal.answer("What is acme-immo?")
    print("  normal budget: pipeline finished")
    print("  " + rag_normal.explain().replace("\n", "\n  "))


# ===========================================================================
# SOLUTION 3 -- Multi-hop reasoning
# ===========================================================================

EXTRA_DOCS = [
    ("d10", "blog",
     "Artefact was founded by Vincent Luciani and Philippe Rolet in 2015."),
    ("d11", "blog",
     "Vincent Luciani lives in Paris and speaks at AI conferences in Europe."),
]


class MultiHopMockLLM(MockLLM):
    """Extend the MockLLM with an extract_entity task."""

    def __call__(self, task: str, payload: dict) -> dict:  # type: ignore[override]
        if task == "extract_entity":
            return {"entity": self._extract_entity(payload["chunks"])}
        return super().__call__(task, payload)

    def _extract_entity(self, chunks: list[str]) -> str | None:
        """First capitalized token of length >= 2 that is not at sentence start only."""
        for chunk in chunks:
            for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", chunk):
                if tok.lower() not in {"the", "a", "an", "in", "on", "of", "and"}:
                    return tok
        return None


def multi_hop_answer(
    query: str,
    retriever: TinyTfIdfRetriever,
    llm: MultiHopMockLLM,
    max_hops: int = 3,
) -> list[dict]:
    """
    Run a very naive multi-hop loop.
    Returns a list of hop records for inspection.
    """
    hops: list[dict] = []
    current_query = query
    for hop in range(max_hops):
        hits = retriever.search(current_query, top_k=3)
        chunks = [text for _, _, text, _ in hits]
        entity = llm("extract_entity", {"chunks": chunks})["entity"]
        record = {
            "hop": hop + 1,
            "query": current_query,
            "chunks": chunks,
            "entity": entity,
        }
        hops.append(record)
        if not entity:
            break
        # Build the next sub-question from the entity
        if hop == 0:
            current_query = f"who founded {entity}"
        elif hop == 1:
            current_query = f"where does {entity} live"
        else:
            break
    return hops


def solution_3() -> None:
    print("\n=== Solution 3: multi-hop ===")
    corpus = CORPUS + EXTRA_DOCS
    retriever = TinyTfIdfRetriever(corpus)
    llm = MultiHopMockLLM()

    query = "main French competitor of Acme"
    hops = multi_hop_answer(query, retriever, llm, max_hops=3)
    for h in hops:
        print(f"  hop {h['hop']}: {h['query']}  -> entity={h['entity']}")
        for c in h["chunks"]:
            print(f"     chunk: {c[:80]}...")


# ===========================================================================
# MEDIUM SOLUTION 1 -- Grading-driven reformulation loop (expand + HyDE)
# ===========================================================================

STOP = {"what", "is", "the", "of", "in", "a", "an", "and", "to", "for", "acme",
        "which", "who", "how", "does", "with"}


def _content_words(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in STOP and len(t) > 1}


def grade_chunk(query_used: str, chunk: str) -> str:
    """Local grading: >= 2 content-word overlap -> RELEVANT."""
    overlap = _content_words(query_used) & _content_words(chunk)
    return "RELEVANT" if len(overlap) >= 2 else "IRRELEVANT"


def reformulate_expand(query: str) -> str:
    """Strategy 1: append synonyms / neighbour terms."""
    expansions = {
        "ca": "chiffre d'affaires revenue annual sales",
        "staff": "team hires hiring engineer employees",
    }
    extra = " ".join(v for k, v in expansions.items() if k in _tokenize(query))
    return f"{query} {extra}".strip() if extra else f"{query} details"


def reformulate_hyde(query: str) -> str:
    """Strategy 2 (HyDE): write a plausible hypothetical ANSWER, search with it."""
    q = query.lower()
    if "rival" in q or "competitor" in q:
        return ("The main French competitor of Acme in AI consulting is a firm "
                "such as Artefact, with hundreds of millions of euros of revenue.")
    if "ca" in _tokenize(query) or "revenue" in q:
        return "Acme reported revenue of several hundred thousand euros in 2025."
    return f"A plausible answer to '{query}' would mention Acme products."


def retrieve_with_retry(query: str, retriever, max_attempts: int = 3) -> dict:
    strategies = [("raw", lambda q: q),
                  ("expand", reformulate_expand),
                  ("hyde", reformulate_hyde)]
    attempts: list[dict] = []

    for strategy_name, transform in strategies[:max_attempts]:
        current = transform(query)
        hits = retriever.search(current, top_k=3)
        relevant = [(doc_id, text) for doc_id, _src, text, _s in hits
                    if grade_chunk(current, text) == "RELEVANT"]
        attempts.append({"strategy": strategy_name, "query": current,
                         "chunks": len(hits), "relevant": len(relevant)})
        if relevant:
            return {"chunks": relevant, "attempts": attempts,
                    "strategy_used": strategy_name}

    return {"chunks": [], "attempts": attempts, "strategy_used": "none"}


def solution_m1() -> None:
    print("\n=== Medium 1: reformulation loop ===")
    retriever = TinyTfIdfRetriever(CORPUS)

    cases = [
        ("What is the rate limit of the Acme API?", "raw"),
        ("What is the CA of Acme in 2025?", "expand"),       # vocab mismatch
        ("Which company is the biggest rival of Acme?", "hyde"),
        ("What is the weather in Tokyo today?", "none"),     # hopeless
    ]
    for query, expected_strategy in cases:
        result = retrieve_with_retry(query, retriever)
        print(f"\n  Q: {query}")
        for a in result["attempts"]:
            print(f"    [{a['strategy']:6}] relevant={a['relevant']}/{a['chunks']}"
                  f"  query={a['query'][:70]}")
        print(f"    -> strategy_used={result['strategy_used']}")
        assert result["strategy_used"] == expected_strategy, (
            f"{query}: expected {expected_strategy}, got {result['strategy_used']}"
        )
        if result["strategy_used"] == "none":
            answer = "Insufficient context: I cannot answer this question."
            print(f"    honest answer: {answer}")
            assert not any(c.isdigit() for c in answer)
    print("\n  [Verification] PASS -- raw / expand / hyde / none all as expected")


# ===========================================================================
# MEDIUM SOLUTION 2 -- Hybrid retrieval + Reciprocal Rank Fusion
# ===========================================================================

HYBRID_CORPUS: list[tuple[str, str, str]] = [
    ("c1", "blog", "Acme revenue reached 820000 euros in 2025, a strong increase."),
    ("c2", "docs_api", "The Acme API rate limit is 60 requests per minute; "
                       "exceeding it returns HTTP 429."),
    ("c3", "docs_produit", "Acme launched acme-immo, a real estate price "
                           "transparency platform, first in Conakry."),
    ("c4", "blog", "Artefact is the main French competitor of Acme in AI consulting."),
    # Spam-like doc designed to fool naive keyword counting:
    ("c5", "blog", "Acme Acme Acme error error error errors guide: handling "
                   "common errors and limits in agent systems."),
]


class KeywordRetriever:
    """Naive exact-match retriever: occurrence count + x2 bonus for exact numbers."""

    def __init__(self, docs: list[tuple[str, str, str]]) -> None:
        self.docs = docs

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        q_numbers = {t for t in q_tokens if t.isdigit()}
        scored = []
        for doc_id, _src, text in self.docs:
            d_tokens = _tokenize(text)
            score = float(sum(d_tokens.count(t) for t in q_tokens))
            score += sum(2.0 for n in q_numbers if n in d_tokens)  # exact-number bonus
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over rankings of 1 / (k + rank), rank starts at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def hybrid_search(query: str, tfidf, keyword, top_k: int = 3) -> dict:
    tfidf_rank = [doc_id for doc_id, _s, _t, _sc in
                  [(d, s, t, sc) for d, s, t, sc in tfidf.search(query, top_k=5)]]
    kw_rank = [doc_id for doc_id, _ in keyword.search(query, top_k=5)]
    fused = rrf_fuse([tfidf_rank, kw_rank])[:top_k]
    return {"tfidf": tfidf_rank, "keyword": kw_rank, "fused": [d for d, _ in fused],
            "fused_scores": fused}


def solution_m2() -> None:
    print("\n=== Medium 2: hybrid retrieval + RRF ===")

    # Unit-check RRF on a hand-computable case
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    expected_a = 1 / 61 + 1 / 62
    assert abs(dict(fused)["a"] - expected_a) < 1e-12
    assert abs(dict(fused)["a"] - dict(fused)["b"]) < 1e-12  # symmetric tie
    print("  rrf_fuse unit check OK (1/61 + 1/62)")

    tfidf = TinyTfIdfRetriever(HYBRID_CORPUS)
    keyword = KeywordRetriever(HYBRID_CORPUS)

    cases = [
        # (label, query, expected best doc)
        ("semantic", "How did Acme errors handling guide describe the rate limit?", "c2"),
        ("exact",    "Why does the API return error 429?", "c2"),
        ("exact-number", "Which product reached 820000 euros?", "c1"),
    ]
    print(f"\n  {'case':14} | {'tfidf top1':10} | {'keyword top1':12} | hybrid top1")
    failures_single = 0
    for label, query, expected in cases:
        out = hybrid_search(query, tfidf, keyword)
        t1 = out["tfidf"][0] if out["tfidf"] else "-"
        k1 = out["keyword"][0] if out["keyword"] else "-"
        h1 = out["fused"][0]
        print(f"  {label:14} | {t1:10} | {k1:12} | {h1}")
        assert h1 == expected, f"{label}: hybrid top1={h1}, expected {expected}"
        failures_single += int(t1 != expected) + int(k1 != expected)

    assert failures_single >= 1, "expected at least one single-retriever failure"
    print(f"\n  [Verification] PASS -- hybrid top-1 correct on all cases, "
          f"{failures_single} single-retriever failure(s)")


# ===========================================================================
# MEDIUM SOLUTION 3 -- Cited synthesis + faithfulness check
# ===========================================================================

CHUNKS_BY_ID = {
    "doc_finance_01": "Acme generated EUR 50M of revenue in 2023, a record year.",
    "doc_hr_02": "By December the Acme team grew to 85 people across 3 offices.",
}


def check_faithfulness(answer: str, chunks_by_id: dict[str, str]) -> dict:
    """Verify every sentence is cited and every number is backed by its source."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    violations: list[dict] = []
    ok_count = 0

    for sentence in sentences:
        cites = re.findall(r"\[([^\]]+)\]", sentence)
        if not cites:
            violations.append({"type": "uncited", "sentence": sentence})
            continue
        bad = False
        for cite in cites:
            if cite not in chunks_by_id:
                violations.append({"type": "bad_citation", "citation": cite,
                                   "sentence": sentence})
                bad = True
        if bad:
            continue
        # Numbers in the sentence body (citations stripped) must appear in sources
        body = re.sub(r"\[[^\]]+\]", "", sentence)
        cited_text = " ".join(chunks_by_id[c] for c in cites)
        for number in re.findall(r"\d+(?:\.\d+)?", body):
            if number not in cited_text:
                violations.append({"type": "unsupported_number", "number": number,
                                   "sentence": sentence})
                bad = True
        if not bad:
            ok_count += 1

    coverage = ok_count / len(sentences) if sentences else 0.0
    return {"faithful": not violations, "violations": violations,
            "coverage": round(coverage, 2)}


def mock_synthesize_cited(retry_with_violations: list | None = None) -> str:
    """Mock 'synthesize_cited' task: a faulty draft, then a corrected one on retry."""
    if retry_with_violations:
        return ("Acme generated EUR 50M in 2023 [doc_finance_01]. "
                "The team grew to 85 people [doc_hr_02].")
    return ("Acme generated EUR 75M in 2023 [doc_finance_01]. "          # wrong number
            "The team grew to 85 people [doc_hr_02]. "
            "Acme plans to triple revenue next year.")                   # uncited


def solution_m3() -> None:
    print("\n=== Medium 3: cited synthesis + faithfulness ===")

    # Case 1: fully correct answer
    good = ("Acme generated EUR 50M in 2023 [doc_finance_01]. "
            "The team grew to 85 people [doc_hr_02].")
    r1 = check_faithfulness(good, CHUNKS_BY_ID)
    print(f"  correct answer:   faithful={r1['faithful']} coverage={r1['coverage']}")
    assert r1["faithful"] and r1["coverage"] == 1.0

    # Case 2: hallucinated number
    hallu = "Acme generated EUR 75M in 2023 [doc_finance_01]."
    r2 = check_faithfulness(hallu, CHUNKS_BY_ID)
    kinds2 = {v["type"] for v in r2["violations"]}
    print(f"  hallucinated num: faithful={r2['faithful']} violations={kinds2}")
    assert "unsupported_number" in kinds2

    # Case 3: missing citation + bad citation
    uncited = ("Acme is doing very well. "
               "The team grew to 85 people [doc_unknown_99].")
    r3 = check_faithfulness(uncited, CHUNKS_BY_ID)
    kinds3 = {v["type"] for v in r3["violations"]}
    print(f"  uncited/bad cite: faithful={r3['faithful']} violations={kinds3}")
    assert {"uncited", "bad_citation"} <= kinds3

    # Pipeline integration: faulty draft -> verify -> 1 retry -> deliver
    draft = mock_synthesize_cited()
    verdict = check_faithfulness(draft, CHUNKS_BY_ID)
    print(f"\n  pipeline draft 1: faithful={verdict['faithful']} "
          f"({len(verdict['violations'])} violations)")
    if not verdict["faithful"]:
        draft = mock_synthesize_cited(retry_with_violations=verdict["violations"])
        verdict = check_faithfulness(draft, CHUNKS_BY_ID)
        print(f"  pipeline draft 2: faithful={verdict['faithful']} "
              f"coverage={verdict['coverage']}")
    assert verdict["faithful"] and verdict["coverage"] == 1.0
    print("\n  [Verification] PASS -- detection + retry produce a faithful answer")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

    solution_m1()
    solution_m2()
    solution_m3()

    # Hard exercises are substantial projects -- key hints:
    #
    # Hard Ex 1 (Corrective RAG):
    #   - Verdict from the best chunk confidence: CORRECT / AMBIGUOUS / INCORRECT
    #   - Knowledge strips: split chunks into sentences, re-grade each strip
    #   - Mock web search only fires on AMBIGUOUS / INCORRECT
    #   - Every claim carries [corpus:doc] or [web:result] attribution
    #
    # Hard Ex 2 (multi-index router RAG):
    #   - One TinyTfIdfRetriever per corpus (products / finance / hr)
    #   - route_index by keyword rules; self_eval lists missing aspects
    #   - Central call counter shared by every llm(...) call enforces the budget

"""
Day 8 -- Solutions to the easy exercises for Agentic RAG.

Each exercise is a self-contained function named solution_1 / solution_2 / solution_3.
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


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

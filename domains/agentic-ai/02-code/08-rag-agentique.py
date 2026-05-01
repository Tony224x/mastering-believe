"""
Day 8 -- Agentic RAG: a tiny but complete agentic RAG pipeline from scratch.

Demonstrates:
  1. TinyTfIdfRetriever      -- naive TF-IDF retriever over an in-memory corpus
  2. MockLLM                 -- deterministic LLM stub that decomposes, grades,
                                reformulates and synthesizes (no API key needed)
  3. QueryDecomposer         -- split a complex query into sub-questions
  4. Router                  -- pick the right knowledge source per sub-question
  5. RetrievalGrader         -- LLM judges whether each chunk is relevant
  6. AdaptiveRetriever       -- retry with a reformulated query if nothing useful
  7. AgenticRAG              -- the full pipeline: decompose -> route -> retrieve
                                -> grade -> retry -> synthesize
  8. A demo comparing a vanilla RAG and the agentic RAG on the same query

Dependencies: stdlib only. Optional: langchain / anthropic / openai if installed,
but the MockLLM fallback guarantees the demo runs offline.

Run:
    python domains/agentic-ai/02-code/08-rag-agentique.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Optional LLM bindings -- fall back to MockLLM if not available
# ---------------------------------------------------------------------------

HAS_ANTHROPIC = False
HAS_OPENAI = False
try:
    import anthropic  # noqa: F401
    HAS_ANTHROPIC = True
except ImportError:
    pass
try:
    import openai  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1. TINY CORPUS -- pretend this is your knowledge base
# ---------------------------------------------------------------------------

# Each doc is (id, source, text). Source is used by the router.
# Kept tiny on purpose so the demo output stays readable.
CORPUS: list[tuple[str, str, str]] = [
    ("d1", "docs_produit",
     "Acme is an AI consulting and SaaS studio founded in 2024 by Alex. "
     "It targets B2B niches in France and West Africa."),
    ("d2", "docs_produit",
     "Acme's flagship product for West Africa is acme-immo, a real estate "
     "price transparency platform launched first in Conakry, Guinea."),
    ("d3", "blog",
     "In 2025 the French AI consulting market grew 34 percent year over year, "
     "driven by demand for retrieval augmented generation and agentic systems."),
    ("d4", "blog",
     "Acme reported revenue of 820000 euros in 2025, up from 180000 euros in 2024. "
     "Most growth came from consulting mandates."),
    ("d5", "jira",
     "Ticket KAL-142: decision to drop the chatbot feature because retention "
     "after 7 days was below 5 percent. Replaced by an email summary flow."),
    ("d6", "jira",
     "Ticket KAL-188: hired a junior data engineer in March 2026 to maintain "
     "the loyalty platform that serves sports federations in production."),
    ("d7", "docs_api",
     "The Acme API exposes a POST /webhook endpoint that accepts JSON events. "
     "The signature must be validated with the header X-Acme-Signature."),
    ("d8", "docs_api",
     "Rate limit on the Acme API is 60 requests per minute per API key. "
     "Exceeding the limit returns HTTP 429."),
    ("d9", "blog",
     "Acme's main French competitor in AI consulting is Artefact, which reported "
     "230 million euros of revenue in 2025."),
]


# ---------------------------------------------------------------------------
# 2. NAIVE TF-IDF RETRIEVER
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Keeps things simple."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TinyTfIdfRetriever:
    """
    A tiny in-memory TF-IDF retriever.

    Not meant for real use -- this is purely pedagogical so you can see
    every step without a dependency on a real vector store.
    """

    def __init__(self, docs: list[tuple[str, str, str]]) -> None:
        self.docs = docs
        # Precompute term frequencies per doc
        self._tf: list[Counter] = [Counter(_tokenize(text)) for _, _, text in docs]
        # Document frequency: how many docs contain each term
        self._df: Counter = Counter()
        for tf in self._tf:
            for term in tf.keys():
                self._df[term] += 1
        self._n_docs = len(docs)

    def _idf(self, term: str) -> float:
        """Classic IDF with smoothing so unknown terms stay finite."""
        df = self._df.get(term, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Sum of tf * idf over query terms."""
        tf = self._tf[doc_idx]
        return sum(tf.get(t, 0) * self._idf(t) for t in query_tokens)

    def search(
        self,
        query: str,
        top_k: int = 3,
        allowed_sources: list[str] | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """
        Return up to top_k docs as (doc_id, source, text, score).

        If allowed_sources is provided, restrict the search to those sources
        -- this is the metadata filtering path used by the router.
        """
        q = _tokenize(query)
        scored: list[tuple[str, str, str, float]] = []
        for i, (doc_id, source, text) in enumerate(self.docs):
            if allowed_sources and source not in allowed_sources:
                continue
            score = self._score(q, i)
            if score > 0:
                scored.append((doc_id, source, text, score))
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 3. MOCK LLM -- deterministic stub that behaves like a real LLM would
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic LLM stub with enough intelligence to make the demo work.

    Supports four "skills", selected by the task string:
      - decompose: split a query into sub-questions
      - route:     pick knowledge sources for a sub-question
      - grade:     judge chunk relevance (RELEVANT / IRRELEVANT)
      - reformulate: reformulate a query that returned nothing useful
      - synthesize: write a final answer from the collected chunks

    In a real system each of these would be a prompted LLM call.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, task: str, payload: dict) -> dict:
        self.call_count += 1
        if task == "decompose":
            return {"sub_questions": self._decompose(payload["query"])}
        if task == "route":
            return {"sources": self._route(payload["query"])}
        if task == "grade":
            return {"verdict": self._grade(payload["query"], payload["chunk"])}
        if task == "reformulate":
            return {"query": self._reformulate(payload["query"])}
        if task == "synthesize":
            return {"answer": self._synthesize(payload["query"], payload["context"])}
        raise ValueError(f"Unknown MockLLM task: {task}")

    # --- individual skills ------------------------------------------------

    def _decompose(self, query: str) -> list[str]:
        """Very naive decomposition: split on 'and', 'compared to', 'et', etc."""
        q = query.strip().rstrip("?")
        pieces: list[str]
        if " compared to " in q or " vs " in q or " et " in q:
            parts = re.split(r" compared to | vs | et ", q)
            pieces = [p.strip() + "?" for p in parts if p.strip()]
        elif " and " in q and len(q.split()) > 8:
            parts = q.split(" and ")
            pieces = [p.strip() + "?" for p in parts if p.strip()]
        else:
            # Simple question -- do not decompose
            pieces = [q + "?"]
        return pieces

    def _route(self, query: str) -> list[str]:
        """Keyword based routing. A real LLM would do this with a prompt."""
        q = query.lower()
        sources: list[str] = []
        if any(w in q for w in ["api", "endpoint", "webhook", "rate limit", "post "]):
            sources.append("docs_api")
        if any(w in q for w in ["ticket", "decision", "hire", "drop", "kal-"]):
            sources.append("jira")
        if any(w in q for w in ["revenue", "ca", "market", "competitor", "growth"]):
            sources.append("blog")
        if any(w in q for w in ["product", "acme", "immo", "launched", "founded"]):
            sources.append("docs_produit")
        return sources or ["docs_produit", "blog", "docs_api", "jira"]

    def _grade(self, query: str, chunk: str) -> str:
        """Grade a chunk as RELEVANT if it shares enough content words with the query."""
        q_terms = {t for t in _tokenize(query) if len(t) > 3}
        c_terms = set(_tokenize(chunk))
        overlap = len(q_terms & c_terms)
        # Keyword-specific boost: numeric questions need numeric chunks
        query_has_number_word = any(w in query.lower() for w in ["revenue", "ca", "how much", "price"])
        chunk_has_number = bool(re.search(r"\d", chunk))
        if query_has_number_word and not chunk_has_number:
            return "IRRELEVANT"
        return "RELEVANT" if overlap >= 2 else "IRRELEVANT"

    def _reformulate(self, query: str) -> str:
        """Add a few synonyms / French to English translation to widen the search."""
        q = query.lower()
        replacements = {
            "ca": "revenue",
            "chiffre d affaires": "revenue",
            "concurrent": "competitor",
            "entreprise": "company",
        }
        for k, v in replacements.items():
            if k in q:
                q = q.replace(k, v)
        # Drop stop words so the query becomes denser
        stop = {"what", "is", "the", "of", "in", "a", "an", "for", "to", "quelle", "est", "le", "la"}
        tokens = [t for t in q.split() if t not in stop]
        return " ".join(tokens)

    def _synthesize(self, query: str, context: list[str]) -> str:
        """Return a short answer that cites the chunks used."""
        if not context:
            return f"I could not find information to answer: {query}"
        bullets = "\n".join(f"- {c}" for c in context)
        return (
            f"Answer based on {len(context)} retrieved chunk(s):\n{bullets}\n"
            f"(query: {query})"
        )


# ---------------------------------------------------------------------------
# 4. VANILLA RAG -- for comparison
# ---------------------------------------------------------------------------

def vanilla_rag(query: str, retriever: TinyTfIdfRetriever, llm: MockLLM) -> str:
    """Single retrieve + single generate. No decomposition, no grading."""
    hits = retriever.search(query, top_k=3)
    context = [text for _, _, text, _ in hits]
    return llm("synthesize", {"query": query, "context": context})["answer"]


# ---------------------------------------------------------------------------
# 5. AGENTIC RAG -- the full pipeline
# ---------------------------------------------------------------------------

@dataclass
class AgenticRAGConfig:
    """All the knobs of the agentic pipeline in one place."""
    top_k: int = 3
    max_retries: int = 2   # how many reformulations per sub-question
    max_sub_questions: int = 5
    verbose: bool = True


@dataclass
class SubQuestionTrace:
    """Observability record for each sub-question processed."""
    sub_question: str
    sources_used: list[str] = field(default_factory=list)
    attempts: int = 0
    kept_chunks: list[str] = field(default_factory=list)
    dropped_chunks: list[str] = field(default_factory=list)
    answered: bool = False


class AgenticRAG:
    """
    Full agentic RAG: decompose -> route -> retrieve -> grade -> retry -> synthesize.
    """

    def __init__(
        self,
        retriever: TinyTfIdfRetriever,
        llm: MockLLM,
        config: AgenticRAGConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or AgenticRAGConfig()
        self.traces: list[SubQuestionTrace] = []

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def answer(self, query: str) -> str:
        """Main entry point. Returns a synthesized final answer."""
        self.traces = []
        self._log(f"\n[AgenticRAG] Query: {query}")

        # 1. Decompose
        sub_questions = self.llm("decompose", {"query": query})["sub_questions"]
        sub_questions = sub_questions[: self.config.max_sub_questions]
        self._log(f"[AgenticRAG] Sub-questions: {sub_questions}")

        # 2. Process each sub-question independently
        all_context: list[str] = []
        for sq in sub_questions:
            trace = SubQuestionTrace(sub_question=sq)
            self.traces.append(trace)
            kept = self._answer_sub_question(sq, trace)
            all_context.extend(kept)
            trace.answered = bool(kept)

        # 3. Synthesize final answer from all kept chunks
        final = self.llm("synthesize", {"query": query, "context": all_context})["answer"]
        return final

    def _answer_sub_question(
        self, sub_question: str, trace: SubQuestionTrace
    ) -> list[str]:
        """
        For one sub-question, retrieve + grade + optionally reformulate and retry.
        Returns the list of kept (relevant) chunk texts.
        """
        # Route: which sources should we hit?
        sources = self.llm("route", {"query": sub_question})["sources"]
        trace.sources_used = sources
        self._log(f"  [sub] {sub_question}  sources={sources}")

        current_query = sub_question
        for attempt in range(self.config.max_retries + 1):
            trace.attempts = attempt + 1
            hits = self.retriever.search(
                current_query, top_k=self.config.top_k, allowed_sources=sources
            )
            self._log(f"    attempt {attempt + 1}: {len(hits)} hits")

            kept: list[str] = []
            for doc_id, source, text, score in hits:
                verdict = self.llm("grade", {"query": sub_question, "chunk": text})["verdict"]
                if verdict == "RELEVANT":
                    kept.append(text)
                    trace.kept_chunks.append(f"{doc_id} ({source})")
                    self._log(f"      KEEP  {doc_id} ({source}) score={score:.2f}")
                else:
                    trace.dropped_chunks.append(f"{doc_id} ({source})")
                    self._log(f"      DROP  {doc_id} ({source}) score={score:.2f}")

            if kept:
                return kept

            # Nothing kept -- try to reformulate and retry
            if attempt < self.config.max_retries:
                current_query = self.llm("reformulate", {"query": current_query})["query"]
                self._log(f"    reformulated -> {current_query}")

        self._log("    [sub] gave up, no relevant chunks found")
        return []

    def explain(self) -> str:
        """Human readable summary of the traces from the last answer call."""
        lines = ["Agentic RAG trace:"]
        for t in self.traces:
            status = "OK" if t.answered else "NOT FOUND"
            lines.append(
                f"  [{status}] {t.sub_question}  "
                f"sources={t.sources_used} attempts={t.attempts} "
                f"kept={len(t.kept_chunks)} dropped={len(t.dropped_chunks)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

def demo() -> None:
    """Compare vanilla RAG and agentic RAG on the same composed query."""
    retriever = TinyTfIdfRetriever(CORPUS)
    llm = MockLLM()

    print("=" * 70)
    print("Backends available: "
          f"anthropic={HAS_ANTHROPIC} openai={HAS_OPENAI} -- using MockLLM")
    print("=" * 70)

    complex_query = (
        "What is the revenue of Acme in 2025 compared to its main competitor "
        "and what is the status of the webhook API?"
    )

    print("\n--- VANILLA RAG ---")
    llm.call_count = 0
    answer_v = vanilla_rag(complex_query, retriever, llm)
    print(answer_v)
    print(f"LLM calls: {llm.call_count}")

    print("\n--- AGENTIC RAG ---")
    llm.call_count = 0
    rag = AgenticRAG(retriever, llm, AgenticRAGConfig(verbose=True))
    answer_a = rag.answer(complex_query)
    print("\n" + answer_a)
    print(f"\nLLM calls: {llm.call_count}")
    print("\n" + rag.explain())

    print("\n--- AGENTIC RAG on a simple query (should NOT over-decompose) ---")
    llm.call_count = 0
    simple = "What is acme-immo?"
    print(rag.answer(simple))
    print(f"LLM calls: {llm.call_count}")


if __name__ == "__main__":
    demo()

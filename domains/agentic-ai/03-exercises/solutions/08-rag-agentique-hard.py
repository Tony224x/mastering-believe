"""
Solutions -- Day 8 (HARD): RAG agentique

Contains solutions for:
  - Hard Ex 1: Full self-corrective agentic RAG loop
               retrieve -> grade -> (reformulate&retry | answer)
               -> groundedness check -> (re-retrieve | finish), with a
               max-iteration guard and proof that a wrong query self-corrects.
  - Hard Ex 2: Multi-hop RAG -- hop-1 result feeds hop-2 sub-question,
               proves both hops fired and the answer combines both, with a
               guard when hop-1 extracts no entity.

Self-contained: embeds a tiny deterministic mock-embedding retriever
(hashlib sha256 -> vector -> cosine, numpy-optional) + a deterministic MockLLM
+ a small mock corpus, so the file RUNS OFFLINE with zero dependencies.

Run:  python 03-exercises/solutions/08-rag-agentique-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# numpy is optional: cosine only needs dot/norm. Try numpy, fall back to a tiny
# pure-Python shim so the file RUNS OFFLINE with zero dependencies.
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy
    _HAS_NUMPY = False

    class _NumpyShim:
        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        class linalg:
            @staticmethod
            def norm(v):
                return sum(x * x for x in v) ** 0.5

    np = _NumpyShim()  # type: ignore


# ==========================================================================
# SHARED -- deterministic mock-embedding retriever + corpus + MockLLM
# ==========================================================================

_EMB_DIM = 32


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) > 3}


def mock_embed(text: str) -> list[float]:
    """Deterministic, offline 'embedding': hash tokens into buckets."""
    vec = [0.0] * _EMB_DIM
    for tok in _tokenize(text):
        h = hashlib.sha256(tok.encode()).digest()
        vec[h[0] % _EMB_DIM] += 1.0 + (h[1] % 5) / 10.0
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class Doc:
    doc_id: str
    source: str
    text: str


# Corpus designed so:
#  - a French query about revenue/competitor FAILS first, self-corrects (Ex1)
#  - a 2-hop chain works: product -> builder company -> founder + location (Ex2)
CORPUS: list[Doc] = [
    Doc("d1", "blog",
        "Acme reported revenue of 820000 euros in 2025, well above 2024."),
    Doc("d2", "blog",
        "Acme's main French competitor Artefact reported 230 million euros of "
        "revenue in 2025."),
    Doc("d3", "docs_produit",
        "The logistics product acme-shield was built by the company Globex in "
        "2025."),
    Doc("d4", "blog",
        "Globex was founded by Dana Reyes, and the founder is based in Lyon."),
    Doc("d5", "docs_api",
        "The Acme API exposes a POST webhook endpoint validated by a signature."),
    Doc("d6", "jira",
        "Ticket KAL-142: decision to drop the chatbot feature for low retention."),
]


class MockLLM:
    """
    Deterministic LLM stub with the skills needed by the hard exercises.

    Skills:
      - reformulate:    widen a failing query (FR->EN, drop stop words)
      - synthesize:     short answer quoting the kept chunks
      - extract_entity: first 'meaningful' proper noun from chunks (multi-hop)
      - plan_multihop:  produce a 2-step plan with a placeholder in hop 2
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, task: str, payload: dict) -> dict:
        self.call_count += 1
        if task == "reformulate":
            return {"query": self._reformulate(payload["query"])}
        if task == "synthesize":
            return {"answer": self._synthesize(payload["query"], payload["chunks"])}
        if task == "extract_entity":
            return {"entity": self._extract_entity(payload["chunks"],
                                                   payload.get("avoid", set()))}
        if task == "plan_multihop":
            return {"plan": self._plan_multihop(payload["query"])}
        raise ValueError(f"Unknown MockLLM task: {task}")

    def _reformulate(self, query: str) -> str:
        q = query.lower()
        replacements = {
            "chiffre d affaires": "revenue", "chiffre affaires": "revenue",
            "chiffre": "revenue", "affaires": "revenue",
            "concurrent": "competitor", "entreprise": "company",
            "fonde": "founded", "fondateur": "founder", "base": "based",
        }
        for fr, en in replacements.items():
            q = q.replace(fr, en)
        stop = {"quel", "quelle", "est", "le", "la", "les", "de", "du", "des",
                "what", "is", "the", "of", "in", "a", "an", "for", "to", "and"}
        return " ".join(t for t in _tokenize(q) if t not in stop)

    def _synthesize(self, query: str, chunks: list[str]) -> str:
        if not chunks:
            return f"I could not find information to answer: {query}"
        return " ".join(chunks)

    def _extract_entity(self, chunks: list[str], avoid: set[str]) -> str | None:
        """First capitalized proper-noun token not in a small stoplist/avoid set."""
        skip = {"The", "A", "An", "In", "On", "Of", "And", "Acme"} | set(avoid)
        for chunk in chunks:
            for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", chunk):
                if tok not in skip:
                    return tok
        return None

    def _plan_multihop(self, query: str) -> list[str]:
        """
        Naive planner: returns a 2-step plan whose 2nd step references the 1st
        result via a {hop1} placeholder. A real LLM would emit this from the
        question structure.
        """
        return ["who built the product mentioned",
                "who founded {hop1} and where is the founder based"]


class MockEmbeddingRetriever:
    """Tiny dense retriever over the mock embeddings (cosine similarity)."""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._emb = {d.doc_id: mock_embed(d.text) for d in docs}

    def search(self, query: str, top_k: int = 3) -> list[tuple[Doc, float]]:
        q = mock_embed(query)
        scored = [(d, cosine(q, self._emb[d.doc_id])) for d in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def grade_retrieval(query: str, chunks: list[str]) -> str:
    """Binary triage used by the self-corrective loop: CORRECT / INCORRECT."""
    q_terms = _content_words(query)
    return "CORRECT" if any(len(q_terms & _content_words(c)) >= 2
                            for c in chunks) else "INCORRECT"


# ==========================================================================
# HARD EXERCISE 1 -- Full self-corrective agentic RAG loop
# ==========================================================================

@dataclass
class CorrectiveState:
    query: str
    current_query: str
    iterations: int = 0
    kept_chunks: list[str] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""
    trace: list[str] = field(default_factory=list)


class SelfCorrectiveRAG:
    """
    Explicit state machine:
      retrieve -> grade -> (reformulate -> retrieve | answer)
               -> groundedness -> (reformulate -> retrieve | finish)
    Bounded by max_iterations; always terminates gracefully.
    """

    def __init__(self, retriever: MockEmbeddingRetriever, llm: MockLLM,
                 top_k: int = 3, max_iterations: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_iterations = max_iterations

    def run(self, query: str) -> CorrectiveState:
        st = CorrectiveState(query=query, current_query=query)
        while st.iterations < self.max_iterations:
            st.iterations += 1

            # --- retrieve ---
            st.trace.append("retrieve")
            hits = self.retriever.search(st.current_query, top_k=self.top_k)
            chunks = [d.text for d, _ in hits]

            # --- grade ---
            verdict = grade_retrieval(st.current_query, chunks)
            st.trace.append(f"grade:{verdict}")
            if verdict == "INCORRECT":
                st.trace.append("reformulate")
                st.current_query = self.llm(
                    "reformulate", {"query": st.current_query})["query"]
                continue

            # keep chunks that actually overlap the current query
            q_terms = _content_words(st.current_query)
            st.kept_chunks = [c for c in chunks
                              if len(q_terms & _content_words(c)) >= 2]

            # --- answer ---
            st.trace.append("answer")
            st.answer = self.llm(
                "synthesize", {"query": query, "chunks": st.kept_chunks})["answer"]

            # --- groundedness ---
            if self._is_grounded(st.answer, st.kept_chunks):
                st.trace.append("ground:ok")
                st.stop_reason = "grounded_answer"
                return st
            # not grounded -> reformulate and re-retrieve
            st.trace.append("ground:fail")
            st.trace.append("reformulate")
            st.current_query = self.llm(
                "reformulate", {"query": st.current_query})["query"]

        # iteration guard hit
        st.stop_reason = "not_found" if not st.kept_chunks else "max_iterations"
        return st

    @staticmethod
    def _is_grounded(answer: str, kept: list[str]) -> bool:
        if not kept:
            return False
        a_terms = _content_words(answer)
        return any(len(a_terms & _content_words(c)) >= 2 for c in kept)


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Full self-corrective agentic RAG loop")
    print("=" * 70)
    retriever = MockEmbeddingRetriever(CORPUS)
    llm = MockLLM()
    rag = SelfCorrectiveRAG(retriever, llm, max_iterations=5)

    # --- self-correction: French query fails, then corrects -----------------
    st = rag.run("chiffre affaires concurrent acme")
    print(f"  trace: {st.trace}")
    print(f"  stop_reason={st.stop_reason} iterations={st.iterations}")
    print(f"  answer (truncated): {st.answer[:70]}...")
    # an INCORRECT must appear strictly before a CORRECT (proof of correction)
    idx_bad = next(i for i, s in enumerate(st.trace) if s == "grade:INCORRECT")
    idx_good = next(i for i, s in enumerate(st.trace) if s == "grade:CORRECT")
    assert idx_bad < idx_good, st.trace
    assert "reformulate" in st.trace
    assert st.stop_reason == "grounded_answer", st.stop_reason
    # grounded against the kept chunk content (competitor revenue fact)
    assert any("competitor" in c.lower() for c in st.kept_chunks), st.kept_chunks
    assert st.iterations <= 5

    # --- impossible query: info absent -> graceful stop, no infinite loop ---
    st2 = rag.run("quelle est la couleur preferee du dragon de komodo violet")
    print(f"\n  impossible trace: {st2.trace}")
    print(f"  stop_reason={st2.stop_reason} iterations={st2.iterations}")
    assert st2.stop_reason in {"not_found", "max_iterations"}, st2.stop_reason
    assert st2.iterations == 5, st2.iterations  # hit the guard, did not crash
    print("[Verification] PASS -- self-correction proven + guard prevents loop")


# ==========================================================================
# HARD EXERCISE 2 -- Multi-hop RAG with chained retrieves
# ==========================================================================

@dataclass
class HopRecord:
    query: str
    kept_chunks: list[str]
    entity: str | None


class MultiHopRAG:
    """Two-hop RAG: hop-1 entity is substituted into the hop-2 sub-question."""

    def __init__(self, retriever: MockEmbeddingRetriever, llm: MockLLM,
                 top_k: int = 3) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.hops: list[HopRecord] = []
        self.final_answer: str = ""

    def run(self, question: str) -> dict:
        self.hops = []
        plan = self.llm("plan_multihop", {"query": question})["plan"]

        # --- HOP 1 ---
        hop1_query = f"{plan[0]} : {question}"
        hits1 = self.retriever.search(hop1_query, top_k=self.top_k)
        kept1 = [d.text for d, _ in hits1]
        entity = self.llm("extract_entity", {"chunks": kept1})["entity"]
        self.hops.append(HopRecord(hop1_query, kept1, entity))

        # Guard: hop-1 produced no entity -> cannot chain
        if not entity:
            return {"status": "hop1_failed", "hops": self.hops, "final_answer": ""}

        # --- HOP 2 (references hop-1 result via the {hop1} placeholder) ---
        hop2_query = plan[1].replace("{hop1}", entity)
        hits2 = self.retriever.search(hop2_query, top_k=self.top_k)
        # keep chunks that overlap the (substituted) hop-2 query
        q2_terms = _content_words(hop2_query)
        kept2 = [d.text for d, _ in hits2
                 if len(q2_terms & _content_words(d.text)) >= 2]
        entity2 = self.llm("extract_entity",
                           {"chunks": kept2, "avoid": {entity}})["entity"]
        self.hops.append(HopRecord(hop2_query, kept2, entity2))

        # final answer combines a hop-1 fact and a hop-2 fact
        self.final_answer = (
            f"The product was built by {entity}. " + (" ".join(kept2))
        )
        return {"status": "ok", "hops": self.hops,
                "final_answer": self.final_answer, "entity1": entity}


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- Multi-hop RAG (chained retrieves)")
    print("=" * 70)
    retriever = MockEmbeddingRetriever(CORPUS)
    llm = MockLLM()
    rag = MultiHopRAG(retriever, llm)

    question = ("Who founded the company that built acme-shield, and where is "
                "that founder based?")
    res = rag.run(question)
    for i, h in enumerate(res["hops"], 1):
        print(f"  hop {i}: query={h.query[:50]!r} entity={h.entity}")
        for c in h.kept_chunks:
            print(f"     chunk: {c[:60]}...")
    print(f"  final: {res['final_answer'][:90]}...")

    # exactly two hops executed
    assert res["status"] == "ok"
    assert len(res["hops"]) == 2, res["hops"]
    # hop-1 extracted the builder company
    entity1 = res["entity1"]
    assert entity1 == "Globex", entity1
    # CHAINING proof: the hop-1 entity appears inside the hop-2 query
    assert entity1 in res["hops"][1].query, res["hops"][1].query
    # final answer COMBINES a hop-1 fact (Globex) and a hop-2 fact (Lyon/Dana)
    assert "Globex" in res["final_answer"]
    assert ("Lyon" in res["final_answer"] or "Dana" in res["final_answer"]), \
        res["final_answer"]

    # --- guard: hop-1 yields no entity -> hop1_failed, no crash --------------
    empty = MockEmbeddingRetriever([
        Doc("x1", "blog", "nothing relevant here just filler words around"),
    ])
    rag2 = MultiHopRAG(empty, MockLLM())
    res2 = rag2.run(question)
    print(f"\n  guard case status: {res2['status']} (hops={len(res2['hops'])})")
    assert res2["status"] == "hop1_failed", res2["status"]
    assert len(res2["hops"]) == 1  # hop-2 never fired
    print("[Verification] PASS -- 2 hops chained, answer combines both, guard works")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 8 HARD Solutions -- RAG agentique"
          f"  (numpy={'yes' if _HAS_NUMPY else 'shim'})")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

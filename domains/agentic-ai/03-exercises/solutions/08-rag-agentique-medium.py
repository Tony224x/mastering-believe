"""
Solutions -- Day 8 (MEDIUM): RAG agentique

Contains solutions for:
  - Medium Ex 1: Corrective RAG (CRAG) -- triage CORRECT/AMBIGUOUS/INCORRECT
                 with reformulate + re-retrieve only when INCORRECT
  - Medium Ex 2: Two-stage reranking (cosine similarity, then deterministic
                 keyword-overlap + recency rerank) -- order changes
  - Medium Ex 3: Citation-grounded answer with a checker that flags unknown
                 docs and unsupported claims

Self-contained: embeds a tiny deterministic mock-embedding retriever
(hashlib sha256 -> vector -> cosine, numpy-optional) + a deterministic MockLLM
+ a small mock corpus, so the file RUNS OFFLINE with zero dependencies.

Run:  python 03-exercises/solutions/08-rag-agentique-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# numpy is optional: we only need dot/norm for cosine, doable in pure Python.
# Try numpy first, fall back to a tiny shim so the file RUNS OFFLINE with zero
# dependencies (same pattern as solutions/03-memory-state.py).
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
    """Lowercase, strip punctuation, split. Mirrors the day-8 code tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(text: str) -> set[str]:
    """Content words = tokens longer than 3 chars (drops most stop words)."""
    return {t for t in _tokenize(text) if len(t) > 3}


def mock_embed(text: str) -> list[float]:
    """
    Deterministic 'embedding': hash each token to a bucket and accumulate.
    This is NOT a real embedding, but it is stable, offline and gives a usable
    cosine signal (texts sharing tokens get closer vectors).
    """
    vec = [0.0] * _EMB_DIM
    for tok in _tokenize(text):
        # sha256 -> stable integer -> bucket index + a small stable weight
        h = hashlib.sha256(tok.encode()).digest()
        idx = h[0] % _EMB_DIM
        weight = 1.0 + (h[1] % 5) / 10.0  # 1.0 .. 1.4, stable per token
        vec[idx] += weight
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
    year: int = 2025


# A tiny corpus. English content on purpose so a French query can FAIL first
# (Ex 1) and self-correct after FR->EN reformulation.
CORPUS: list[Doc] = [
    Doc("d1", "blog",
        "Acme reported revenue of 820000 euros in 2025, up from 180000 in 2024.",
        year=2025),
    Doc("d2", "blog",
        "Acme's main French competitor in AI consulting is Artefact, which "
        "reported 230 million euros of revenue in 2025.", year=2025),
    Doc("d3", "docs_produit",
        "Acme is an AI consulting and SaaS studio founded in 2024, targeting "
        "B2B niches in France and West Africa.", year=2024),
    Doc("d4", "docs_api",
        "The Acme API exposes a POST webhook endpoint that accepts JSON events "
        "validated with a signature header.", year=2023),
    Doc("d5", "jira",
        "Ticket KAL-142: decision to drop the chatbot feature because day-7 "
        "retention was below 5 percent.", year=2022),
    Doc("d6", "blog",
        "In 2026 the French AI consulting market kept growing, driven by demand "
        "for retrieval augmented generation and agentic systems.", year=2026),
]


class MockLLM:
    """
    Deterministic LLM stub with the skills needed by the medium exercises.

    Skills (selected by `task`):
      - reformulate: widen a failing query (FR->EN synonyms, drop stop words)
      - synthesize:  short answer that quotes the kept chunks
      - cite:        produce one cited sentence per kept chunk
    """

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, task: str, payload: dict) -> dict:
        self.call_count += 1
        if task == "reformulate":
            return {"query": self._reformulate(payload["query"])}
        if task == "synthesize":
            return {"answer": self._synthesize(payload["query"], payload["chunks"])}
        if task == "cite":
            return {"claims": self._cite(payload["retrieved"])}
        raise ValueError(f"Unknown MockLLM task: {task}")

    def _reformulate(self, query: str) -> str:
        q = query.lower()
        replacements = {
            "chiffre d affaires": "revenue",
            "chiffre affaires": "revenue",
            "chiffre": "revenue",
            "affaires": "revenue",
            "concurrent": "competitor",
            "entreprise": "company",
            "fonde": "founded",
            "marche": "market",
        }
        for fr, en in replacements.items():
            q = q.replace(fr, en)
        stop = {"quel", "quelle", "est", "le", "la", "les", "de", "du", "des",
                "what", "is", "the", "of", "in", "a", "an", "for", "to"}
        tokens = [t for t in _tokenize(q) if t not in stop]
        return " ".join(tokens)

    def _synthesize(self, query: str, chunks: list[str]) -> str:
        if not chunks:
            return f"I could not find information to answer: {query}"
        return " ".join(chunks)

    def _cite(self, retrieved: list[tuple[str, str]]) -> list[dict]:
        """retrieved = [(doc_id, text), ...] -> one cited sentence per doc."""
        claims = []
        for doc_id, text in retrieved:
            # The "sentence" reuses the chunk text so it is grounded by design.
            claims.append({"sentence": text, "cites": [doc_id]})
        return claims


class MockEmbeddingRetriever:
    """Tiny dense retriever over the mock embeddings (cosine similarity)."""

    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self._emb = {d.doc_id: mock_embed(d.text) for d in docs}

    def search(self, query: str, top_k: int = 5,
               allowed_sources: list[str] | None = None) -> list[tuple[Doc, float]]:
        q = mock_embed(query)
        scored: list[tuple[Doc, float]] = []
        for d in self.docs:
            if allowed_sources and d.source not in allowed_sources:
                continue
            scored.append((d, cosine(q, self._emb[d.doc_id])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ==========================================================================
# MEDIUM EXERCISE 1 -- Corrective RAG (CRAG)
# ==========================================================================

def grade_retrieval(query: str, chunks: list[str]) -> str:
    """Triage the WHOLE retrieval: CORRECT / AMBIGUOUS / INCORRECT."""
    q_terms = _content_words(query)
    query_wants_number = any(w in query.lower()
                             for w in ["revenue", "ca", "how much", "price", "much"])
    best_overlap = 0
    has_number_match = False
    for c in chunks:
        overlap = len(q_terms & _content_words(c))
        best_overlap = max(best_overlap, overlap)
        if overlap >= 2 and re.search(r"\d", c):
            has_number_match = True
    if best_overlap < 2:
        return "INCORRECT"
    if best_overlap >= 3 and (not query_wants_number or has_number_match):
        return "CORRECT"
    return "AMBIGUOUS"


def corrective_rag(query: str, retriever: MockEmbeddingRetriever, llm: MockLLM,
                   top_k: int = 4, max_corrections: int = 2) -> dict:
    """CRAG loop: grade the retrieval, only re-retrieve when INCORRECT."""
    current = query
    verdict_history: list[str] = []
    corrections = 0
    kept: list[str] = []
    confidence = "high"

    for _ in range(max_corrections + 1):
        hits = retriever.search(current, top_k=top_k)
        chunks = [d.text for d, _ in hits]
        # Grade against the CURRENT (possibly reformulated) query: that is the
        # query that actually drove this retrieval. Grading against the original
        # French terms would stay INCORRECT forever even on good chunks.
        verdict = grade_retrieval(current, chunks)
        verdict_history.append(verdict)

        if verdict == "CORRECT":
            # keep chunks that actually overlap the current query
            q_terms = _content_words(current)
            kept = [c for c in chunks if len(q_terms & _content_words(c)) >= 2]
            confidence = "high"
            break
        if verdict == "AMBIGUOUS":
            q_terms = _content_words(current)
            kept = [c for c in chunks if len(q_terms & _content_words(c)) >= 2]
            confidence = "low"
            break
        # INCORRECT -> reformulate and re-retrieve (bounded)
        if corrections < max_corrections:
            current = llm("reformulate", {"query": current})["query"]
            corrections += 1
            continue
        confidence = "none"
        break

    answer = llm("synthesize", {"query": query, "chunks": kept})["answer"]
    return {
        "answer": answer,
        "verdict_history": verdict_history,
        "corrections": corrections,
        "confidence": confidence,
        "kept": kept,
    }


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Corrective RAG (triage + bounded re-retrieve)")
    print("=" * 70)
    retriever = MockEmbeddingRetriever(CORPUS)
    llm = MockLLM()

    # --- All three verdicts exist -------------------------------------------
    assert grade_retrieval("revenue Acme 2025",
                           ["Acme reported revenue of 820000 euros in 2025."]) == "CORRECT"
    assert grade_retrieval("revenue Acme 2025",
                           ["Acme is a studio founded in 2024."]) == "INCORRECT"
    assert grade_retrieval("acme founded studio",
                           ["Acme founded studio targeting nothing useful"]) in {
        "CORRECT", "AMBIGUOUS"}

    # --- Query CORRECT on the first try -> zero corrections -----------------
    res_ok = corrective_rag("Acme revenue 2025 competitor", retriever, llm)
    print(f"  direct query: history={res_ok['verdict_history']} "
          f"corrections={res_ok['corrections']}")
    assert res_ok["verdict_history"][0] == "CORRECT"
    assert res_ok["corrections"] == 0

    # --- French query FAILS first, then self-corrects after reformulation ---
    fr_query = "chiffre affaires concurrent acme"  # corpus is in English
    res_fix = corrective_rag(fr_query, retriever, llm, max_corrections=2)
    print(f"  french query: history={res_fix['verdict_history']} "
          f"corrections={res_fix['corrections']}")
    print(f"    -> answer (truncated): {res_fix['answer'][:70]}...")
    assert res_fix["verdict_history"][0] == "INCORRECT"
    assert res_fix["verdict_history"][-1] == "CORRECT"
    assert res_fix["corrections"] >= 1
    # grounded: the kept chunks must include the competitor revenue fact
    assert any("competitor" in c.lower() or "artefact" in c.lower()
               for c in res_fix["kept"]), res_fix["kept"]

    # --- AMBIGUOUS case -> confidence low -----------------------------------
    # craft a chunk with overlap exactly 2 (market, consulting) and a non-number
    # query, so it is neither INCORRECT (>=2) nor CORRECT (needs >=3).
    amb = grade_retrieval("market growth consulting",
                          ["the french consulting market kept growing strongly"])
    print(f"  ambiguous grade demo: {amb}")
    assert amb == "AMBIGUOUS"
    print(f"  bounded: corrections never exceed max ({res_fix['corrections']} <= 2)")
    assert res_fix["corrections"] <= 2
    print("[Verification] PASS -- CRAG triage + bounded self-correction")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Two-stage reranking
# ==========================================================================

def rerank(query: str, candidates: list[tuple[Doc, float]],
           alpha: float = 0.5) -> list[tuple[Doc, float]]:
    """
    Stage 2: rerank candidates by alpha*keyword_overlap + (1-alpha)*recency,
    both normalized over the candidate set. Returns (doc, rerank_score) sorted.
    """
    q_terms = _content_words(query)
    overlaps = [len(q_terms & _content_words(d.text)) for d, _ in candidates]
    years = [d.year for d, _ in candidates]
    max_ov = max(overlaps) or 1
    min_y, max_y = min(years), max(years)
    span = (max_y - min_y) or 1

    scored: list[tuple[Doc, float]] = []
    for (d, _sim), ov in zip(candidates, overlaps):
        kw = ov / max_ov
        recency = (d.year - min_y) / span
        score = alpha * kw + (1 - alpha) * recency
        scored.append((d, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Two-stage reranking (similarity then keyword+recency)")
    print("=" * 70)
    retriever = MockEmbeddingRetriever(CORPUS)

    # Query mixing a recent-market topic and AI consulting keywords.
    query = "french ai consulting market revenue"

    # Stage 1: candidate set ordered by cosine similarity.
    candidates = retriever.search(query, top_k=5)
    sim_order = [d.doc_id for d, _ in candidates]
    print(f"  stage-1 (similarity) order: {sim_order}")

    # Stage 2: rerank by keyword overlap + recency.
    reranked = rerank(query, candidates, alpha=0.5)
    rr_order = [d.doc_id for d, _ in reranked]
    print(f"  stage-2 (rerank)     order: {rr_order}")

    # The order MUST change between the two stages.
    assert sim_order != rr_order, (sim_order, rr_order)

    # Top-1 after rerank has the maximal rerank_score.
    top_score = reranked[0][1]
    assert all(top_score >= s for _, s in reranked)

    # alpha sweep: keyword-pure vs recency-pure must give a different top.
    top_kw = rerank(query, candidates, alpha=1.0)[0][0].doc_id   # keywords only
    top_rec = rerank(query, candidates, alpha=0.0)[0][0].doc_id  # recency only
    print(f"  alpha=1.0 (keywords) top={top_kw} | alpha=0.0 (recency) top={top_rec}")
    assert top_kw != top_rec, "both signals must actually matter"
    # recency-only top must be the most recent candidate
    most_recent = max(candidates, key=lambda c: c[0].year)[0].doc_id
    assert top_rec == most_recent, (top_rec, most_recent)
    print("[Verification] PASS -- rerank changes order, both signals matter")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Citation-grounded answer + checker
# ==========================================================================

def answer_with_citations(query: str, retrieved: list[Doc],
                          llm: MockLLM) -> list[dict]:
    """One cited sentence per retrieved doc: {'sentence', 'cites':[doc_id]}."""
    pairs = [(d.doc_id, d.text) for d in retrieved]
    return llm("cite", {"retrieved": pairs})["claims"]


def check_groundedness(claims: list[dict], retrieved: list[Doc]) -> dict:
    """
    Verify each claim: (1) every cited doc_id exists in retrieved,
    (2) the sentence shares >= 2 content words with the cited doc text.
    """
    by_id = {d.doc_id: d for d in retrieved}
    violations: list[dict] = []
    for claim in claims:
        sentence = claim["sentence"]
        s_terms = _content_words(sentence)
        for doc_id in claim["cites"]:
            doc = by_id.get(doc_id)
            if doc is None:
                violations.append({"sentence": sentence, "doc_id": doc_id,
                                   "reason": "unknown_doc"})
                continue
            if len(s_terms & _content_words(doc.text)) < 2:
                violations.append({"sentence": sentence, "doc_id": doc_id,
                                   "reason": "unsupported"})
    return {"grounded": not violations, "violations": violations}


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Citation-grounded answer + checker")
    print("=" * 70)
    retriever = MockEmbeddingRetriever(CORPUS)
    llm = MockLLM()

    query = "Acme revenue competitor"
    retrieved = [d for d, _ in retriever.search(query, top_k=3)]

    # --- Nominal: every sentence cites an existing, supporting doc -----------
    claims = answer_with_citations(query, retrieved, llm)
    report = check_groundedness(claims, retrieved)
    print(f"  nominal: {len(claims)} claims, grounded={report['grounded']}")
    assert report["grounded"] is True
    assert report["violations"] == []

    # --- Adversarial: inject one unknown-doc citation + one unsupported ------
    adversarial = list(claims) + [
        {"sentence": "Acme was acquired by Globex in 2027.", "cites": ["d99"]},
        {"sentence": "Bananas are a great source of potassium.",
         "cites": [retrieved[0].doc_id]},
    ]
    report2 = check_groundedness(adversarial, retrieved)
    reasons = sorted(v["reason"] for v in report2["violations"])
    print(f"  adversarial violations: {[ (v['reason'], v['doc_id']) for v in report2['violations'] ]}")
    assert report2["grounded"] is False
    assert reasons == ["unknown_doc", "unsupported"], reasons
    # the unknown-doc violation must point at d99, the unsupported at a real doc
    unknown = next(v for v in report2["violations"] if v["reason"] == "unknown_doc")
    unsupported = next(v for v in report2["violations"] if v["reason"] == "unsupported")
    assert unknown["doc_id"] == "d99"
    assert unsupported["doc_id"] == retrieved[0].doc_id
    print("[Verification] PASS -- detects unknown_doc + unsupported, nominal is grounded")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 8 MEDIUM Solutions -- RAG agentique"
          f"  (numpy={'yes' if _HAS_NUMPY else 'shim'})")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

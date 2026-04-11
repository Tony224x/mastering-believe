"""
Jour 10 -- RAG end-to-end sans dependances externes.

Usage:
    python 10-rag-architecture.py

Pipeline mini mais complet :
  1. Corpus de documents en dur
  2. Chunker (recursive)
  3. Embedding naif via TF-IDF (aucun modele requis)
  4. Index dense (cosine) + index sparse (BM25 simplifie)
  5. Hybrid search via Reciprocal Rank Fusion
  6. Reranker simule (scoring base sur term overlap pondere)
  7. Generation d'une reponse formatee avec citations
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Corpus
# =============================================================================


CORPUS = {
    "doc1": (
        "RAG Architecture overview.\n\n"
        "Retrieval-Augmented Generation combines a retriever with a generator. "
        "The retriever fetches relevant documents from a knowledge base, and the "
        "generator produces an answer conditioned on those documents. RAG is the "
        "dominant pattern for LLM applications that need factual grounding."
    ),
    "doc2": (
        "Vector databases for embeddings.\n\n"
        "Popular vector databases include Pinecone, Qdrant, Weaviate, and pgvector. "
        "They store high-dimensional vectors and support approximate nearest "
        "neighbor search using HNSW or IVF indexes. Choose pgvector for small "
        "prototypes and Qdrant or Pinecone for production workloads."
    ),
    "doc3": (
        "Chunking strategies.\n\n"
        "Fixed-size chunking splits text into equal token windows. Recursive "
        "chunking uses a hierarchy of separators like paragraph, sentence, word. "
        "Semantic chunking cuts where the meaning shifts between adjacent sentences. "
        "Typical chunk size is 300 to 1000 tokens with 10 percent overlap."
    ),
    "doc4": (
        "Hybrid search.\n\n"
        "Hybrid search combines dense vector retrieval with sparse BM25 keyword "
        "retrieval and fuses the two ranked lists. The most common fusion "
        "algorithm is Reciprocal Rank Fusion, which scores each document by the "
        "sum of 1 over rank plus a constant. Hybrid improves recall by 10 to 20 "
        "percent over pure dense retrieval."
    ),
    "doc5": (
        "Reranking with cross-encoders.\n\n"
        "A cross-encoder jointly encodes the query and each candidate document, "
        "producing a precise relevance score. Rerankers such as Cohere Rerank or "
        "BGE reranker are applied to the top 20 or top 50 retrieved candidates to "
        "boost the best answers to the top. Reranking typically adds 15 to 30 "
        "percent MRR on top of retrieval."
    ),
    "doc6": (
        "Evaluating RAG systems.\n\n"
        "Retrieval quality is measured using recall at k and mean reciprocal rank. "
        "Generation quality is measured using faithfulness, context precision, and "
        "answer relevance. Tools like Ragas and TruLens automate these metrics. "
        "You must build a gold set of question and expected answer pairs."
    ),
}


# =============================================================================
# SECTION 2 : Tokenizer et chunker recursifs
# =============================================================================


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "as", "by", "from", "it", "its", "into", "over", "under",
}


def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanum + drop stopwords."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def recursive_chunk(text: str, target_size: int = 120, overlap: int = 20) -> list[str]:
    """Recursive-ish chunking : split on paragraphs, then on sentences.

    target_size and overlap are in characters for simplicity.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for p in parts:
        if len(p) <= target_size:
            chunks.append(p)
            continue
        # fallback : split on sentences
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= target_size:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    chunks.append(buf)
                # overlap: keep the last few chars of previous chunk
                carry = buf[-overlap:] if overlap and buf else ""
                buf = (carry + " " + s).strip()
        if buf:
            chunks.append(buf)
    return chunks


# =============================================================================
# SECTION 3 : Dense retrieval via TF-IDF + cosine (no model required)
# =============================================================================


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    tfidf: dict[str, float] = field(default_factory=dict)


class TFIDFIndex:
    """Tiny TF-IDF to play the role of an embedder.

    In real life you'd call an embedding model, but for a standalone demo
    TF-IDF gives us dense-ish vectors where cosine similarity is meaningful.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.df: Counter = Counter()  # document frequency
        self.n_docs: int = 0

    def add(self, chunk: Chunk) -> None:
        chunk.tokens = tokenize(chunk.text)
        self.chunks.append(chunk)
        for t in set(chunk.tokens):
            self.df[t] += 1
        self.n_docs += 1

    def build(self) -> None:
        """Compute TF-IDF vectors for all chunks once DF is final."""
        for c in self.chunks:
            tf = Counter(c.tokens)
            vec = {}
            for term, freq in tf.items():
                idf = math.log((self.n_docs + 1) / (self.df[term] + 1)) + 1
                vec[term] = (freq / max(len(c.tokens), 1)) * idf
            c.tfidf = vec

    def _cosine(self, q_vec: dict[str, float], c_vec: dict[str, float]) -> float:
        dot = sum(q_vec[t] * c_vec.get(t, 0.0) for t in q_vec)
        nq = math.sqrt(sum(v * v for v in q_vec.values()))
        nc = math.sqrt(sum(v * v for v in c_vec.values()))
        return dot / (nq * nc + 1e-9)

    def vectorize_query(self, query: str) -> dict[str, float]:
        q_tokens = tokenize(query)
        tf = Counter(q_tokens)
        vec = {}
        for term, freq in tf.items():
            idf = math.log((self.n_docs + 1) / (self.df.get(term, 0) + 1)) + 1
            vec[term] = (freq / max(len(q_tokens), 1)) * idf
        return vec

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.vectorize_query(query)
        scored = [(c, self._cosine(q, c.tfidf)) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# =============================================================================
# SECTION 4 : Sparse retrieval (BM25 simplifie)
# =============================================================================


class BM25Index:
    """Simplified BM25 for keyword matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def add(self, chunk: Chunk) -> None:
        if not chunk.tokens:
            chunk.tokens = tokenize(chunk.text)
        self.chunks.append(chunk)
        for t in set(chunk.tokens):
            self.df[t] += 1

    def build(self) -> None:
        if self.chunks:
            self.avgdl = sum(len(c.tokens) for c in self.chunks) / len(self.chunks)

    def _score(self, query_tokens: list[str], chunk: Chunk) -> float:
        n = len(self.chunks)
        score = 0.0
        tf = Counter(chunk.tokens)
        dl = len(chunk.tokens)
        for t in query_tokens:
            if t not in tf:
                continue
            idf = math.log((n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5) + 1)
            num = tf[t] * (self.k1 + 1)
            den = tf[t] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += idf * num / den
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q_tokens = tokenize(query)
        scored = [(c, self._score(q_tokens, c)) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# =============================================================================
# SECTION 5 : Hybrid via Reciprocal Rank Fusion
# =============================================================================


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """RRF: each chunk gets sum(1/(rank+k)) across lists, then sort."""
    score: dict[str, float] = defaultdict(float)
    by_id: dict[str, Chunk] = {}
    for rank, (c, _) in enumerate(dense_results):
        score[c.chunk_id] += 1 / (rank + 1 + k)
        by_id[c.chunk_id] = c
    for rank, (c, _) in enumerate(sparse_results):
        score[c.chunk_id] += 1 / (rank + 1 + k)
        by_id[c.chunk_id] = c
    fused = [(by_id[cid], s) for cid, s in score.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# =============================================================================
# SECTION 6 : Reranker simule
# =============================================================================


def simulated_cross_encoder(query: str, chunk: Chunk) -> float:
    """Simulated cross-encoder: weighted term overlap + phrase bonus.

    In production this is a transformer. Here we reward exact query tokens
    and give a small bonus if the query tokens appear close together.
    """
    q_tokens = set(tokenize(query))
    c_tokens = set(chunk.tokens)
    overlap = len(q_tokens & c_tokens)
    uniq = len(q_tokens) or 1
    base = overlap / uniq
    # small co-occurrence bonus : reward chunks where query tokens cluster
    text = chunk.text.lower()
    co_bonus = 0.0
    q_list = list(q_tokens)
    for i, a in enumerate(q_list):
        for b in q_list[i + 1 :]:
            if a in text and b in text:
                co_bonus += 0.05
    return base + min(co_bonus, 0.2)


def rerank(query: str, candidates: list[tuple[Chunk, float]], top_k: int = 3) -> list[Chunk]:
    scored = [(c, simulated_cross_encoder(query, c)) for c, _ in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


# =============================================================================
# SECTION 7 : Answer formatting with inline citations
# =============================================================================


def format_answer(query: str, top_chunks: list[Chunk]) -> str:
    """Compose a simulated 'LLM' answer that cites the top chunks.

    No real LLM : we build the response from the chunks directly. The point
    is to show the citation pattern (inline [N] references).
    """
    lines = [f"Question: {query}", "", "Answer (grounded, with citations):"]
    for i, c in enumerate(top_chunks, start=1):
        snippet = c.text.replace("\n", " ")[:200]
        lines.append(f"  - {snippet} [{i}]")
    lines.append("")
    lines.append("Sources:")
    for i, c in enumerate(top_chunks, start=1):
        lines.append(f"  [{i}] {c.doc_id}#{c.chunk_id}")
    return "\n".join(lines)


# =============================================================================
# SECTION 8 : Pipeline demo
# =============================================================================


def build_indexes() -> tuple[TFIDFIndex, BM25Index]:
    dense = TFIDFIndex()
    sparse = BM25Index()
    for doc_id, text in CORPUS.items():
        pieces = recursive_chunk(text, target_size=160, overlap=20)
        for i, chunk_text in enumerate(pieces):
            cid = f"{doc_id}::{i}"
            c = Chunk(chunk_id=cid, doc_id=doc_id, text=chunk_text)
            dense.add(c)
    dense.build()
    # reuse the same chunks (same tokens already computed) for sparse
    for c in dense.chunks:
        sparse.add(c)
    sparse.build()
    return dense, sparse


def rag_query(query: str, dense: TFIDFIndex, sparse: BM25Index) -> str:
    dense_res = dense.search(query, k=10)
    sparse_res = sparse.search(query, k=10)
    fused = reciprocal_rank_fusion(dense_res, sparse_res)
    top = rerank(query, fused, top_k=3)
    return format_answer(query, top)


def demo() -> None:
    print(SEPARATOR)
    print("MINI RAG -- indexation + hybrid + reranker + citations")
    print(SEPARATOR)
    dense, sparse = build_indexes()
    print(f"\nIndexed {len(dense.chunks)} chunks across {len(CORPUS)} documents.")

    queries = [
        "How does hybrid search combine dense and sparse retrieval?",
        "What chunk size should I use for technical docs?",
        "How do I evaluate a RAG pipeline?",
    ]
    for q in queries:
        print("\n" + SEPARATOR)
        print(rag_query(q, dense, sparse))


if __name__ == "__main__":
    demo()

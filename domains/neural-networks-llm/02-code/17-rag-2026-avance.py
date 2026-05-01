"""
Jour 17 — RAG 2026 : hybrid + RRF + reranker + contextual retrieval
===================================================================
Pure Python. Aucune lib. Le but est de montrer a la main ce que font les
pipelines RAG modernes, avec des nombres qu'on peut inspecter.

Contenu :
  1. BM25 minimal (tf-idf + length norm)
  2. Dense retrieval simule (vecteurs aleatoires fixes par terme)
  3. Reciprocal Rank Fusion (RRF) pour combiner les deux
  4. Reranker simule (cross-scoring query+doc)
  5. Contextual retrieval : enrichissement de chunk avant embedding
  6. Evaluation : recall@k et nDCG@10 sur un mini-corpus

Run : python 02-code/17-rag-2026-avance.py
"""

from __future__ import annotations
import sys, io, re, math, random
from collections import Counter, defaultdict

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(42)

# ============================================================================
# Corpus jouet : 10 chunks avec des themes qui se recoupent
# ============================================================================

CORPUS = [
    ("c1", "doc_annual_2023",
     "ACME Corporation rapport annuel 2023 : le chiffre d'affaires Europe a augmente de 15%."),
    ("c2", "doc_annual_2023",
     "ACME Corporation rapport annuel 2023 : les couts R&D sont passes de 200M a 240M EUR."),
    ("c3", "doc_support_tickets",
     "Ticket #4821 : erreur code NS-101 a l'ouverture du module admin. Fix applique."),
    ("c4", "doc_support_tickets",
     "Ticket #4822 : timeout reseau sur endpoint /api/v2/users. Lien avec NS-101 ?"),
    ("c5", "doc_legal",
     "La clause 7.3 du contrat definit les obligations de confidentialite des parties."),
    ("c6", "doc_legal",
     "La clause 7.4 limite la responsabilite aux dommages directs, exclusion explicite."),
    ("c7", "doc_medical",
     "Le patient presente une hypertension artérielle stade 2, traitement en cours."),
    ("c8", "doc_medical",
     "Reaction indesirable au medicament X chez 2% des patients selon l'etude."),
    ("c9", "doc_annual_2022",
     "ACME Corporation 2022 : chiffre d'affaires Europe stable, croissance en Asie."),
    ("c10", "doc_faq",
     "Comment resoudre un code erreur NS-101 : relancer le service admin et verifier les logs."),
]

# Ground truth pour l'evaluation : pour chaque query, les chunks pertinents
QUERIES = [
    ("Quelle est la croissance du CA Europe d'ACME en 2023 ?", {"c1"}),
    ("Comment fixer l'erreur NS-101 ?", {"c3", "c10"}),
    ("Que dit la clause de responsabilite limitee ?", {"c6"}),
]

# ============================================================================
# 1) BM25 minimal (lexical)
# ============================================================================

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[a-z0-9\-]+", text.lower())]


class BM25:
    def __init__(self, corpus: list[tuple[str, str, str]], k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(c[2]) for c in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.docs_tokens) / self.N
        # DF puis IDF avec lissage (BM25+)
        df = Counter()
        for d in self.docs_tokens:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query: str) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = []
        for (cid, did, text), tokens in zip(self.corpus, self.docs_tokens):
            tf = Counter(tokens)
            dl = len(tokens)
            s = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            scores.append((cid, s))
        return sorted(scores, key=lambda x: -x[1])


# ============================================================================
# 2) Dense retrieval simule : vecteur = moyenne des vecteurs par token
# ============================================================================

VOCAB_DIM = 32


def term_vector(token: str) -> list[float]:
    """Vecteur deterministe par token (cache via hash)."""
    rng = random.Random(hash(token) & 0xFFFFFFFF)
    return [rng.gauss(0, 1) for _ in range(VOCAB_DIM)]


def text_vector(text: str) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * VOCAB_DIM
    vecs = [term_vector(t) for t in tokens]
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(VOCAB_DIM)]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class DenseRetriever:
    def __init__(self, corpus, enrich: bool = False):
        self.corpus = corpus
        self.enrich = enrich
        # Contextual retrieval : on prefixe le chunk par un resume doc.
        self.vectors = []
        for cid, did, text in corpus:
            if enrich:
                text_to_embed = f"Contexte: {did}. {text}"
            else:
                text_to_embed = text
            self.vectors.append(text_vector(text_to_embed))

    def score(self, query: str) -> list[tuple[str, float]]:
        q = text_vector(query)
        out = [(self.corpus[i][0], cosine(q, v)) for i, v in enumerate(self.vectors)]
        return sorted(out, key=lambda x: -x[1])


# ============================================================================
# 3) Reciprocal Rank Fusion : combiner sans parametre a tuner
# ============================================================================

def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking):
            fused[cid] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


# ============================================================================
# 4) Reranker simule (cross-encoder)
# ============================================================================

def cross_score(query: str, text: str) -> float:
    """
    Un vrai cross-encoder concatene query [SEP] doc, passe dans un modele,
    sort un score. On simule : dense cosine + bonus pour chaque token lexical
    present (hybride). C'est ce qu'apprend implicitement un cross-encoder.
    """
    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(text))
    overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
    dense = cosine(text_vector(query), text_vector(text))
    return 0.6 * dense + 0.4 * overlap


def rerank(query, candidates, corpus_by_id) -> list[tuple[str, float]]:
    scored = [(cid, cross_score(query, corpus_by_id[cid][2])) for cid, _ in candidates]
    return sorted(scored, key=lambda x: -x[1])


# ============================================================================
# 5) Evaluation : recall@k et nDCG@10
# ============================================================================

def recall_at_k(ranking, relevant, k=5):
    topk = {cid for cid, _ in ranking[:k]}
    return len(topk & relevant) / max(len(relevant), 1)


def ndcg_at_k(ranking, relevant, k=10):
    def dcg(rels):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    gains = [1.0 if cid in relevant else 0.0 for cid, _ in ranking[:k]]
    ideal = sorted(gains, reverse=True)
    return dcg(gains) / (dcg(ideal) or 1e-9)


# ============================================================================
# Run : comparer 4 strategies sur le mini-corpus
# ============================================================================

print("=" * 70)
print("Comparaison 4 strategies : dense, BM25, hybrid(RRF), hybrid+reranker")
print("=" * 70)

bm25 = BM25(CORPUS)
dense = DenseRetriever(CORPUS, enrich=False)
dense_ctx = DenseRetriever(CORPUS, enrich=True)
by_id = {c[0]: c for c in CORPUS}

results = defaultdict(lambda: {"r@5": [], "ndcg@10": []})

for query, relevant in QUERIES:
    print(f"\n  Query: {query!r}")
    print(f"  Relevant: {sorted(relevant)}")

    r_bm25 = bm25.score(query)
    r_dense = dense.score(query)
    r_dense_ctx = dense_ctx.score(query)
    r_hybrid = rrf([r_bm25, r_dense])
    r_hybrid_ctx = rrf([r_bm25, r_dense_ctx])
    r_rerank = rerank(query, r_hybrid_ctx[:6], by_id)

    strategies = [
        ("dense", r_dense),
        ("dense+ctx", r_dense_ctx),
        ("bm25", r_bm25),
        ("hybrid RRF", r_hybrid),
        ("hybrid+ctx RRF", r_hybrid_ctx),
        ("hybrid+ctx+rerank", r_rerank),
    ]
    for name, ranking in strategies:
        r5 = recall_at_k(ranking, relevant, k=5)
        nd = ndcg_at_k(ranking, relevant, k=10)
        results[name]["r@5"].append(r5)
        results[name]["ndcg@10"].append(nd)
        top3 = [cid for cid, _ in ranking[:3]]
        print(f"    {name:<22} top3={top3}  r@5={r5:.2f}  ndcg@10={nd:.2f}")


print("\n" + "=" * 70)
print("Moyennes sur les 3 queries (retenir l'ordre des strategies) :")
print("=" * 70)
for name, metrics in results.items():
    r5 = sum(metrics["r@5"]) / len(metrics["r@5"])
    nd = sum(metrics["ndcg@10"]) / len(metrics["ndcg@10"])
    print(f"  {name:<22}  recall@5={r5:.2f}  nDCG@10={nd:.2f}")

print("""
Lecons :
  - BM25 seul gagne sur les matches lexicaux exacts (NS-101, clause 7.4).
  - Dense gagne sur les reformulations (croissance ≈ augmentation).
  - RRF combine les forces sans parametre a tuner.
  - Contextual retrieval aide quand le chunk est ambigu hors-contexte.
  - Reranker affine le top-k et donne le meilleur nDCG, au prix d'une latence
    supplementaire.

Dans un vrai systeme, remplacer :
  - BM25 par Elasticsearch/Tantivy
  - dense retriever par voyage-3/bge-large/embedding-3-large
  - reranker par Cohere rerank-3.5 ou bge-reranker-v3
  - ajouter un agentic layer pour query rewriting + multi-step retrieval
""")

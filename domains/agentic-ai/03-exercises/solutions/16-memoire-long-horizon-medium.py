"""
Solutions -- Day 16 (MEDIUM): Long-horizon memory

Contains solutions for:
  - Medium Ex 1: SalienceStore -- relevance scoring (recency + importance +
                 similarity) with top-K retrieval ranking. Proves the most
                 relevant memories surface and a recent-but-off-topic memory
                 does NOT pollute the top-2.
  - Medium Ex 2: Consolidation pass -- dedup of near-duplicate memories +
                 decay/forgetting of stale entries (importance shield). Proves
                 the store shrinks while important memories survive.
  - Medium Ex 3: Reflection -- derive higher-level semantic facts from raw
                 episodic events with traceable source_ids.

Self-contained: embeds deterministic fake embeddings + memory entries, so the
file RUNS OFFLINE with zero dependencies (no torch, no langgraph, no API key).

Run:  python 03-exercises/solutions/16-memoire-long-horizon-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ==========================================================================
# SHARED OFFLINE PRIMITIVES (mirror 02-code/16-memoire-long-horizon.py)
# ==========================================================================

def _tokenize(text: str) -> list[str]:
    """Lowercase, keep alphabetic tokens only."""
    return re.findall(r"[a-z]+", text.lower())


def fake_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words pseudo-embedding, L2-normalized."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        # Stable per-token bucket (hash() is salted, so use a fixed scheme).
        bucket = (sum(ord(c) for c in tok) * 131) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1] for non-negative bag-of-words vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


_AUTO_ID = [0]


def _next_id() -> str:
    _AUTO_ID[0] += 1
    return f"ep{_AUTO_ID[0]:03d}"


@dataclass
class MemoryEntry:
    """A single long-term memory unit with salience and timestamps."""
    content: str
    importance: float = 0.5
    created_at: float = 0.0
    last_accessed: float = 0.0
    tags: list[str] = field(default_factory=list)
    merged_count: int = 0           # how many near-duplicates folded into this one
    id: str = field(default_factory=_next_id)
    embedding: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = fake_embed(self.content)


HOUR = 3600.0
DAY = 24 * HOUR


def recency(entry: MemoryEntry, t_now: float, lambda_: float = 0.02) -> float:
    """Exponential decay on elapsed hours -> score in (0, 1]."""
    hours = max(0.0, (t_now - entry.created_at) / HOUR)
    return math.exp(-lambda_ * hours)


# ==========================================================================
# MEDIUM EXERCISE 1 -- SalienceStore: scoring + retrieval ranking
# ==========================================================================

class SalienceStore:
    """
    Memory store ranking entries by:
        score = w_rec*recency + w_imp*importance + w_sim*similarity
    Similarity is cosine in [0, 1]; recency is exponential decay.
    """

    def __init__(self, weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
                 lambda_: float = 0.02) -> None:
        self.w_rec, self.w_imp, self.w_sim = weights
        self.lambda_ = lambda_
        self._entries: list[MemoryEntry] = []

    def add(self, content: str, importance: float = 0.5,
            created_at: float = 0.0, tags: list[str] | None = None) -> MemoryEntry:
        e = MemoryEntry(content=content, importance=importance,
                        created_at=created_at, last_accessed=created_at,
                        tags=tags or [])
        self._entries.append(e)
        return e

    def _score(self, e: MemoryEntry, q_emb: list[float], t_now: float) -> float:
        rec = recency(e, t_now, self.lambda_)
        sim = cosine_similarity(e.embedding, q_emb)  # already in [0, 1]
        return self.w_rec * rec + self.w_imp * e.importance + self.w_sim * sim

    def retrieve(self, query: str, k: int, t_now: float) -> list[tuple[float, MemoryEntry]]:
        q_emb = fake_embed(query)
        scored = [(self._score(e, q_emb, t_now), e) for e in self._entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        # Touch the retrieved entries (recall reinforces).
        for _, e in scored[:k]:
            e.last_accessed = t_now
        return scored[:k]


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- SalienceStore: scoring + retrieval ranking")
    print("=" * 70)

    now = 100 * DAY                      # arbitrary "now"
    store = SalienceStore(weights=(0.25, 0.3, 0.45), lambda_=0.02)

    # 2 highly relevant memories (strong lexical overlap with the query).
    relevant_a = store.add("user preference export format CSV report",
                           importance=0.6, created_at=now - 2 * DAY)
    relevant_b = store.add("the user wants the export format to be CSV not PDF",
                           importance=0.55, created_at=now - 1 * DAY)
    # 1 old-but-very-important salient memory (low lexical overlap).
    old_important = store.add("contact email of the account owner changed",
                              importance=0.95, created_at=now - 30 * DAY)
    # 1 very recent but totally off-topic, low importance (the trap).
    recent_noise = store.add("weather api timeout retry scheduled tonight",
                             importance=0.2, created_at=now - 0.1 * HOUR)
    # 2 mid noise.
    store.add("agent restarted after a deployment last week",
              importance=0.4, created_at=now - 5 * DAY)
    store.add("invoice pipeline ran nightly without issues",
              importance=0.4, created_at=now - 3 * DAY)

    query = "user preference export format CSV"
    top = store.retrieve(query, k=4, t_now=now)
    q_emb = fake_embed(query)

    print(f"  Query: '{query}'")
    print(f"  {'rk':>2} {'score':>7} {'rec':>6} {'imp':>5} {'sim':>6}  content")
    for rank, (score, e) in enumerate(top, 1):
        rec = recency(e, now, store.lambda_)
        sim = cosine_similarity(e.embedding, q_emb)
        print(f"  {rank:>2} {score:7.3f} {rec:6.3f} {e.importance:5.2f} "
              f"{sim:6.3f}  {e.content[:48]}")

    top_entries = [e for _, e in top]
    top2_ids = {e.id for e in top_entries[:2]}

    # Top-1 must be a genuinely relevant memory, not the recent off-topic one.
    assert top_entries[0].id in (relevant_a.id, relevant_b.id), \
        f"top-1 should be a relevant memory, got {top_entries[0].content!r}"
    # The recent off-topic noise must NOT crack the top-2.
    assert recent_noise.id not in top2_ids, \
        "recent-but-off-topic memory polluted the top-2"
    # The old-but-important memory survives in the top-K thanks to salience.
    assert old_important.id in {e.id for e in top_entries}, \
        "old-but-important memory should surface via salience"

    print("[Verification] PASS -- relevant memories surface, salience preserved, "
          "recent noise filtered")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Consolidation: dedup near-duplicates + decay stale
# ==========================================================================

def dedup(entries: list[MemoryEntry], sim_threshold: float = 0.6) -> list[MemoryEntry]:
    """
    Greedy clustering by cosine similarity. Each cluster keeps ONE representative:
    the highest-importance entry (ties broken by recency / created_at).
    The representative inherits the cluster's max importance and records how many
    near-duplicates were folded in via `merged_count`.
    """
    reps: list[MemoryEntry] = []
    for e in entries:
        placed = False
        for rep in reps:
            if cosine_similarity(e.embedding, rep.embedding) >= sim_threshold:
                # Fold e into rep's cluster.
                rep.merged_count += 1
                rep.importance = max(rep.importance, e.importance)
                # Prefer the more important / more recent content as the anchor.
                if (e.importance > rep.importance) or (
                        e.importance == rep.importance and e.created_at > rep.created_at):
                    rep.content = e.content
                    rep.embedding = e.embedding
                placed = True
                break
        if not placed:
            reps.append(e)
    return reps


def forget(entries: list[MemoryEntry], t_now: float,
           recency_threshold: float = 0.05, lambda_: float = 0.02) -> list[MemoryEntry]:
    """
    Drop stale entries. effective_score = max(recency, importance*0.5); the
    importance term shields critical memories from decay-based forgetting.
    """
    kept: list[MemoryEntry] = []
    for e in entries:
        rec = recency(e, t_now, lambda_)
        effective = max(rec, e.importance * 0.5)
        if effective >= recency_threshold:
            kept.append(e)
    return kept


def consolidate(entries: list[MemoryEntry], t_now: float,
                sim_threshold: float = 0.6,
                recency_threshold: float = 0.05,
                lambda_: float = 0.02) -> list[MemoryEntry]:
    """dedup then forget."""
    deduped = dedup(entries, sim_threshold)
    return forget(deduped, t_now, recency_threshold, lambda_)


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Consolidation: dedup near-duplicates + decay stale")
    print("=" * 70)

    now = 100 * DAY
    entries: list[MemoryEntry] = []

    # 3 near-duplicates of the same fact (high mutual similarity).
    entries.append(MemoryEntry("user prefers CSV exports", importance=0.6, created_at=now - 1 * DAY))
    entries.append(MemoryEntry("the user prefers CSV exports", importance=0.5, created_at=now - 2 * DAY))
    entries.append(MemoryEntry("user prefers CSV exports format", importance=0.7, created_at=now - 0.5 * DAY))

    # 2 stale + low importance (very old, will be forgotten). Distinct wording so
    # dedup leaves them separate; importance*0.5 stays under the forget threshold.
    entries.append(MemoryEntry("scratch heartbeat ping alpha trace", importance=0.08,
                               created_at=now - 400 * DAY))
    entries.append(MemoryEntry("verbose diagnostic dump beta channel", importance=0.09,
                               created_at=now - 380 * DAY))

    # 1 stale BUT critical (very old, must survive via importance shield).
    critical = MemoryEntry("master account recovery key rotated",
                           importance=0.9, created_at=now - 500 * DAY)
    entries.append(critical)

    # 4 normal varied entries (recent enough, distinct content).
    entries.append(MemoryEntry("invoice batch processed monday morning", importance=0.4,
                               created_at=now - 3 * DAY))
    entries.append(MemoryEntry("dashboard latency improved after caching", importance=0.45,
                               created_at=now - 4 * DAY))
    entries.append(MemoryEntry("new teammate onboarded to the project", importance=0.5,
                               created_at=now - 6 * DAY))
    entries.append(MemoryEntry("weekly report sent to stakeholders", importance=0.4,
                               created_at=now - 2 * DAY))

    initial_count = len(entries)
    print(f"  initial entries: {initial_count}")

    # Step 1: dedup.
    deduped = dedup(entries, sim_threshold=0.6)
    csv_reps = [e for e in deduped if "csv" in e.content.lower()]
    print(f"  after dedup: {len(deduped)} entries "
          f"(CSV cluster -> {len(csv_reps)} representative)")

    # The 3 CSV near-duplicates collapse to exactly 1 representative.
    assert len(csv_reps) == 1, f"CSV near-duplicates should collapse to 1, got {len(csv_reps)}"
    assert csv_reps[0].merged_count == 2, \
        f"representative should record 2 folded duplicates, got {csv_reps[0].merged_count}"
    # The representative inherits the max importance of the cluster (0.7).
    assert abs(csv_reps[0].importance - 0.7) < 1e-9, csv_reps[0].importance
    assert len(deduped) < initial_count, "dedup must reduce the count"

    # Step 2: forget on the deduped set.
    final = forget(deduped, t_now=now, recency_threshold=0.05, lambda_=0.02)
    contents = {e.content for e in final}
    print(f"  after forget: {len(final)} entries")

    # Stale low-importance entries are gone.
    assert "scratch heartbeat ping alpha trace" not in contents, "stale entry should be forgotten"
    assert "verbose diagnostic dump beta channel" not in contents, "stale entry should be forgotten"
    # Critical-but-stale entry survives via the importance shield.
    assert critical.content in contents, "critical stale entry must survive (importance shield)"
    # Whole consolidation strictly shrinks the store.
    assert len(final) < initial_count, \
        f"final {len(final)} should be < initial {initial_count}"

    print(f"  net: {initial_count} -> {len(final)} entries")
    print("[Verification] PASS -- dedup folded duplicates, stale forgotten, "
          "critical shielded")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Reflection: derive traceable semantic facts
# ==========================================================================

@dataclass
class SemanticFact:
    """A higher-level fact distilled from episodes, with full traceability."""
    content: str
    importance: float
    source_ids: list[str]
    confidence: float


_STOP = {"the", "a", "an", "is", "was", "are", "to", "in", "on", "and", "or",
         "for", "of", "user", "with", "at", "it", "this", "that", "asked",
         "said", "after"}


def _salient_token(content: str) -> str | None:
    # Keep short but meaningful acronyms like "csv"; drop 1-2 char noise.
    toks = [t for t in _tokenize(content) if t not in _STOP and len(t) >= 3]
    if not toks:
        return None
    return Counter(toks).most_common(1)[0][0]


def reflect(episodes: list[MemoryEntry], min_support: int = 2) -> list[SemanticFact]:
    """
    Group episodes by their salient keyword; for each group with enough support,
    distill a SemanticFact that references the source episode ids.
    """
    groups: dict[str, list[MemoryEntry]] = {}
    for ep in episodes:
        key = _salient_token(ep.content)
        if key is None:
            continue
        groups.setdefault(key, []).append(ep)

    facts: list[SemanticFact] = []
    for keyword, group in sorted(groups.items()):
        if len(group) < min_support:
            continue
        source_ids = [e.id for e in group]
        avg_imp = sum(e.importance for e in group) / len(group)
        facts.append(SemanticFact(
            content=f"Across {len(group)} episodes, '{keyword}' is a recurring theme",
            importance=min(1.0, avg_imp + 0.1),
            source_ids=source_ids,
            confidence=min(0.95, 0.5 + 0.1 * len(group)),
        ))
    return facts


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Reflection: derive traceable semantic facts")
    print("=" * 70)

    # 8+ episodes; CSV (3), payment (2), 3 unique-theme noise episodes.
    # The theme word repeats within each episode so it is unambiguously the
    # salient token (most frequent), mirroring how reflection clusters a stream.
    stream = [
        MemoryEntry("CSV requested: user wants the CSV report", importance=0.7, created_at=1 * DAY),
        MemoryEntry("CSV export generated, CSV emailed to user", importance=0.6, created_at=2 * DAY),
        MemoryEntry("user thanked us for the CSV, CSV format kept", importance=0.5, created_at=3 * DAY),
        MemoryEntry("payment gateway payment returned an error", importance=0.8, created_at=4 * DAY),
        MemoryEntry("payment retry: payment eventually succeeded", importance=0.6, created_at=5 * DAY),
        MemoryEntry("dashboard widget colors adjusted", importance=0.3, created_at=6 * DAY),
        MemoryEntry("teammate joined the standup", importance=0.3, created_at=7 * DAY),
        MemoryEntry("backup snapshot completed overnight", importance=0.4, created_at=8 * DAY),
    ]
    existing_ids = {e.id for e in stream}

    facts = reflect(stream, min_support=2)
    print(f"  episodes: {len(stream)} | derived facts: {len(facts)}")
    for f in facts:
        print(f"    conf={f.confidence:.2f} imp={f.importance:.2f} "
              f"sources={f.source_ids} | {f.content}")

    by_kw = {}
    for f in facts:
        # The salient keyword appears in the fact content.
        for kw in ("csv", "payment"):
            if kw in f.content.lower():
                by_kw[kw] = f

    # CSV fact: >= 3 source ids, all real.
    assert "csv" in by_kw, "expected a CSV fact"
    csv_fact = by_kw["csv"]
    assert len(csv_fact.source_ids) >= 3, csv_fact.source_ids
    assert all(sid in existing_ids for sid in csv_fact.source_ids)

    # Payment fact: exactly 2 sources.
    assert "payment" in by_kw, "expected a payment fact"
    pay_fact = by_kw["payment"]
    assert len(pay_fact.source_ids) == 2, pay_fact.source_ids

    # Unique-theme episodes produce NO fact (only CSV + payment qualify).
    assert len(facts) == 2, f"only 2 themes have >= min_support, got {len(facts)} facts"

    # Confidence grows with support: CSV (3) > payment (2).
    assert csv_fact.confidence > pay_fact.confidence

    # Full traceability: every source id of every fact really exists.
    for f in facts:
        assert f.source_ids, "a derived fact must reference at least one source"
        assert all(sid in existing_ids for sid in f.source_ids), \
            "all source_ids must point to real episodes"

    print("[Verification] PASS -- facts derived, traceable sources, confidence "
          "scales with support")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 16 MEDIUM Solutions -- Long-horizon memory")
    print("  (offline, deterministic, stdlib only -- no API key)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

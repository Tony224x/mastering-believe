"""
Solutions -- Day 16 (HARD): Long-horizon memory

Contains solutions for:
  - Hard Ex 1: LongHorizonMemory -- full pipeline (episodic write ->
               consolidation/forgetting -> semantic reflection -> retrieval)
               run over a multi-session simulation. Proves recall improves
               after reflection AND the store stays bounded.
  - Hard Ex 2: BoundedMemory -- eviction policy under a hard capacity budget
               combining recency (LRU), frequency (LFU) and salience. Proves
               important memories are never evicted while capacity is respected.

Self-contained, fully offline. Embeds deterministic fake embeddings + memory
entries -- no torch, no langgraph, no API key.

Run:  python 03-exercises/solutions/16-memoire-long-horizon-hard.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# ==========================================================================
# SHARED OFFLINE PRIMITIVES
# ==========================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def fake_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words pseudo-embedding, L2-normalized."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        bucket = (sum(ord(c) for c in tok) * 131) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


HOUR = 3600.0
DAY = 24 * HOUR

_AUTO_ID = [0]


def _next_id(prefix: str = "ep") -> str:
    _AUTO_ID[0] += 1
    return f"{prefix}{_AUTO_ID[0]:03d}"


@dataclass
class MemoryEntry:
    content: str
    importance: float = 0.5
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    kind: str = "episodic"          # "episodic" | "semantic"
    source_ids: list[str] = field(default_factory=list)
    merged_count: int = 0
    id: str = field(default_factory=lambda: _next_id("ep"))
    embedding: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.embedding = fake_embed(self.content)
        if self.last_access == 0.0:
            self.last_access = self.created_at


def recency(t_ref: float, t_now: float, lambda_: float = 0.02) -> float:
    """Exponential decay on elapsed hours -> (0, 1]."""
    hours = max(0.0, (t_now - t_ref) / HOUR)
    return math.exp(-lambda_ * hours)


# ==========================================================================
# HARD EXERCISE 1 -- LongHorizonMemory: full pipeline over multi-session sim
# ==========================================================================

_STOP = {"the", "a", "an", "is", "was", "are", "to", "in", "on", "and", "or",
         "for", "of", "user", "with", "at", "it", "this", "that", "asked",
         "said", "again", "after", "still"}


def _salient_token(content: str) -> str | None:
    # Keep short but meaningful acronyms like "csv"; drop 1-2 char noise.
    toks = [t for t in _tokenize(content) if t not in _STOP and len(t) >= 3]
    if not toks:
        return None
    return Counter(toks).most_common(1)[0][0]


class LongHorizonMemory:
    """
    Orchestrates the three memory stages of module 16:
        observe (episodic write)
          -> consolidate (dedup + forget + reflection -> semantic)
          -> recall (retrieval over {episodes} U {semantic facts})
    """

    def __init__(self, sim_threshold: float = 0.65,
                 recency_threshold: float = 0.05, lambda_: float = 0.02,
                 protect_importance: float = 0.85) -> None:
        self.sim_threshold = sim_threshold
        self.recency_threshold = recency_threshold
        self.lambda_ = lambda_
        self.protect_importance = protect_importance
        self.episodic: list[MemoryEntry] = []
        self.semantic: list[MemoryEntry] = []
        self.total_raw_written = 0          # count of every raw episode ever written

    # ---- write -------------------------------------------------------
    def observe(self, content: str, importance: float, t: float) -> MemoryEntry:
        e = MemoryEntry(content=content, importance=importance, created_at=t,
                        last_access=t, kind="episodic")
        self.episodic.append(e)
        self.total_raw_written += 1
        return e

    # ---- consolidation: dedup + forget + reflection ------------------
    def _dedup(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        reps: list[MemoryEntry] = []
        for e in entries:
            placed = False
            for rep in reps:
                if cosine_similarity(e.embedding, rep.embedding) >= self.sim_threshold:
                    rep.merged_count += 1
                    if e.importance > rep.importance:
                        rep.importance = e.importance
                        rep.content = e.content
                        rep.embedding = e.embedding
                    placed = True
                    break
            if not placed:
                reps.append(e)
        return reps

    def _forget(self, entries: list[MemoryEntry], t_now: float) -> list[MemoryEntry]:
        kept = []
        for e in entries:
            rec = recency(e.created_at, t_now, self.lambda_)
            effective = max(rec, e.importance * 0.5)  # importance shield
            if effective >= self.recency_threshold:
                kept.append(e)
        return kept

    def _reflect(self, episodes: list[MemoryEntry], t_now: float,
                 min_support: int = 2) -> list[MemoryEntry]:
        groups: dict[str, list[MemoryEntry]] = {}
        for ep in episodes:
            key = _salient_token(ep.content)
            if key is None:
                continue
            groups.setdefault(key, []).append(ep)

        new_facts: list[MemoryEntry] = []
        existing = {f.id: f for f in self.semantic}
        for keyword, group in sorted(groups.items()):
            if len(group) < min_support:
                continue
            fact_id = f"sem_{keyword}"
            if fact_id in existing:
                # Theme already distilled AND still live in the episodic store:
                # refresh the fact so its recency reflects the latest evidence
                # (a recurring/relevant fact should not "age out" while the
                # underlying episodes are still present). This keeps the recall
                # signal honest -- a still-true semantic fact stays retrievable.
                fact = existing[fact_id]
                fact.created_at = t_now
                fact.last_access = t_now
                known = set(fact.source_ids)
                fact.source_ids.extend(e.id for e in group if e.id not in known)
                continue
            avg_imp = sum(e.importance for e in group) / len(group)
            fact = MemoryEntry(
                content=f"semantic: '{keyword}' is a recurring theme over {len(group)} episodes",
                importance=min(1.0, avg_imp + 0.1),
                created_at=t_now,
                last_access=t_now,
                kind="semantic",
                source_ids=[e.id for e in group],
            )
            fact.id = fact_id
            new_facts.append(fact)
        self.semantic.extend(new_facts)
        return new_facts

    def consolidate(self, t_now: float, min_support: int = 2) -> list[MemoryEntry]:
        self.episodic = self._dedup(self.episodic)
        self.episodic = self._forget(self.episodic, t_now)
        return self._reflect(self.episodic, t_now, min_support)

    # ---- recall ------------------------------------------------------
    def recall(self, query: str, k: int, t_now: float,
               weights: tuple[float, float, float] = (0.25, 0.3, 0.45)) -> list[MemoryEntry]:
        w_rec, w_imp, w_sim = weights
        q_emb = fake_embed(query)
        pool = self.episodic + self.semantic
        scored = []
        for e in pool:
            rec = recency(e.created_at, t_now, self.lambda_)
            sim = cosine_similarity(e.embedding, q_emb)
            score = w_rec * rec + w_imp * e.importance + w_sim * sim
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def total_units(self) -> int:
        return len(self.episodic) + len(self.semantic)


def hard_ex1_pipeline() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: LongHorizonMemory pipeline (multi-session)")
    print("=" * 60)

    # sim_threshold tuned so genuine near-duplicates fold, but the distinctly
    # worded theme episodes (low mutual cosine) are NOT folded -- reflection
    # still sees the whole cluster.
    mem = LongHorizonMemory(sim_threshold=0.75, recency_threshold=0.05,
                            lambda_=0.02, protect_importance=0.85)

    # ----- Session 1 (J0) -----
    # The "csv" theme recurs across ALL sessions. Each csv episode repeats "csv"
    # so it is the unambiguous salient token, but the rest of the wording differs
    # so episodes are not near-duplicates of each other (they survive dedup and
    # are all visible to reflection).
    s1 = 0 * DAY
    mem.observe("csv csv requested by the user for the quarterly report", 0.6, s1)
    mem.observe("payment gateway returned a transient error once", 0.5, s1 + 1 * HOUR)
    mem.observe("ephemeral debug trace token alpha printed locally", 0.04, s1 + 2 * HOUR)
    critical = mem.observe("master recovery secret rotated by admin", 0.95, s1 + 3 * HOUR)

    query = "csv export preference"

    # Recall BEFORE any consolidation: only raw episodes exist.
    before = mem.recall(query, k=3, t_now=s1 + 4 * HOUR)
    before_has_semantic = any(e.kind == "semantic" for e in before)
    print(f"\n  [J0] recall BEFORE consolidation -> top kinds: "
          f"{[e.kind for e in before]}")
    assert not before_has_semantic, "no semantic fact should exist before consolidation"

    # ----- Session 2 (J3) -----
    s2 = 3 * DAY
    mem.observe("csv csv delivered, generated and emailed successfully", 0.5, s2)
    # A genuine near-duplicate pair (for the dedup / bounded-store demonstration).
    mem.observe("nightly backup snapshot completed without errors", 0.4, s2 + 1 * HOUR)
    mem.observe("nightly backup snapshot completed with no errors", 0.4, s2 + 90 * 60)
    mem.observe("ephemeral debug trace token beta printed locally", 0.04, s2 + 2 * HOUR)
    mem.consolidate(t_now=s2 + 3 * HOUR, min_support=2)

    # ----- Session 3 (J10) -- the theme recurs one last time (freshest) --------
    s3 = 10 * DAY
    mem.observe("csv csv confirmed as the wanted persistent export format", 0.6, s3 + 1 * HOUR)
    final_facts = mem.consolidate(t_now=s3 + 3 * HOUR, min_support=2)

    print(f"  raw episodes ever written: {mem.total_raw_written}")
    print(f"  after sim: {len(mem.episodic)} episodes + {len(mem.semantic)} "
          f"semantic = {mem.total_units()} units")
    print(f"  semantic facts: {[f.id for f in mem.semantic]}")

    # Recall AFTER consolidation: a consolidated semantic fact must surface.
    after = mem.recall(query, k=5, t_now=s3 + 4 * HOUR)
    after_kinds = [e.kind for e in after]
    print(f"  [J10] recall AFTER consolidation -> top kinds: {after_kinds}")
    has_semantic_csv = any(e.kind == "semantic" and "csv" in e.content.lower()
                           for e in after)

    # (a) Recall improved: a decontextualized semantic CSV fact now appears,
    #     a signal a single raw episode could not provide.
    assert has_semantic_csv, "a consolidated CSV semantic fact should surface after reflection"

    # (b) Store bounded: total units < raw episodes ever written.
    assert mem.total_units() < mem.total_raw_written, \
        f"store not bounded: {mem.total_units()} units vs {mem.total_raw_written} raw"

    # (c) Traceability: at least one derived fact has valid source ids.
    all_ep_ids = {e.id for e in mem.episodic} | {  # ids may have been forgotten;
        i for f in mem.semantic for i in f.source_ids}
    written_ids_seen = any(f.source_ids for f in mem.semantic)
    assert written_ids_seen, "at least one derived fact must reference sources"
    # Every source id must have been a real episode id (ep-prefixed, generated).
    for f in mem.semantic:
        for sid in f.source_ids:
            assert sid.startswith("ep"), f"bad source id {sid}"

    # (d) Importance shield: the critical secret survives despite ageing 10 days.
    assert any(e.id == critical.id for e in mem.episodic), \
        "critical episode must survive forgetting (importance shield)"

    print("\n  PASS -- recall improved, store bounded, sources traceable, "
          "critical shielded.\n")


# ==========================================================================
# HARD EXERCISE 2 -- BoundedMemory: hybrid LRU/LFU/salience eviction
# ==========================================================================

class BoundedMemory:
    """
    Capacity-bounded store with a hybrid eviction policy:
        evict_score = w_rec*recency + w_freq*norm_freq + w_sal*importance
    Lowest evict_score is evicted first. Entries whose importance >=
    protect_threshold are immune (protected set) and never chosen as victims.
    If the store overflows but ALL entries are protected, raise MemoryError
    rather than dropping a critical memory.
    """

    def __init__(self, capacity: int, protect_threshold: float = 0.8,
                 weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
                 lambda_: float = 0.02) -> None:
        self.capacity = capacity
        self.protect_threshold = protect_threshold
        self.w_rec, self.w_freq, self.w_sal = weights
        self.lambda_ = lambda_
        self._entries: dict[str, MemoryEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def add(self, key: str, content: str, importance: float, t_now: float) -> MemoryEntry:
        e = MemoryEntry(content=content, importance=importance,
                        created_at=t_now, last_access=t_now)
        e.id = key
        self._entries[key] = e
        # Enforce the hard capacity budget.
        while len(self._entries) > self.capacity:
            self._evict(t_now)
        assert len(self._entries) <= self.capacity  # invariant
        return e

    def get(self, key: str, t_now: float) -> MemoryEntry | None:
        e = self._entries.get(key)
        if e is not None:
            e.last_access = t_now          # LRU signal
            e.access_count += 1            # LFU signal
        return e

    def _evict(self, t_now: float) -> str:
        max_access = max((e.access_count for e in self._entries.values()), default=0)
        candidates = [e for e in self._entries.values()
                      if e.importance < self.protect_threshold]
        if not candidates:
            # Every entry is protected and we are over capacity -> cannot proceed.
            raise MemoryError(
                "capacity exceeded but all entries are protected (importance shield)")

        def evict_score(e: MemoryEntry) -> float:
            rec = recency(e.last_access, t_now, self.lambda_)         # LRU
            norm_freq = e.access_count / (max_access + 1)             # LFU
            return self.w_rec * rec + self.w_freq * norm_freq + self.w_sal * e.importance

        victim = min(candidates, key=evict_score)
        del self._entries[victim.id]
        return victim.id


def hard_ex2_eviction() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: BoundedMemory -- hybrid LRU/LFU/salience eviction")
    print("=" * 60)

    cap = 5
    store = BoundedMemory(capacity=cap, protect_threshold=0.8,
                          weights=(0.4, 0.3, 0.3), lambda_=0.02)
    t = 0.0

    # Critical entry inserted very early, never touched again.
    store.add("crit", "master recovery key (critical)", importance=0.95, t_now=t)
    t += 1 * DAY

    # 4 normal entries.
    store.add("n1", "routine note one", importance=0.4, t_now=t); t += 1 * HOUR
    store.add("n2", "routine note two", importance=0.4, t_now=t); t += 1 * HOUR
    fav = "fav"
    store.add(fav, "frequently used note", importance=0.4, t_now=t); t += 1 * HOUR
    store.add("n3", "routine note three", importance=0.4, t_now=t); t += 1 * HOUR
    assert len(store) <= cap

    # Hammer the favourite so its frequency keeps it alive.
    for _ in range(8):
        store.get(fav, t_now=t)
        t += 0.5 * HOUR

    # An equal-importance peer that is NEVER accessed (the LFU loser).
    store.add("lonely", "never accessed peer", importance=0.4, t_now=t); t += 1 * HOUR
    assert len(store) <= cap, "capacity invariant after lonely insert"

    # Spam insertions to force several evictions over time.
    print(f"\n  capacity = {cap}; spamming inserts to force evictions...")
    for i in range(6):
        store.add(f"spam{i}", f"ephemeral spam entry {i}", importance=0.35, t_now=t)
        # Capacity invariant must hold after EVERY insertion.
        assert len(store) <= cap, f"capacity overflow after spam{i}: {len(store)}"
        t += 2 * HOUR

    keys = set(store._entries.keys())
    print(f"  surviving keys: {sorted(keys)} (size={len(store)})")

    # (1) Capacity respected.
    assert len(store) <= cap

    # (2) Critical entry always present despite age + zero access (protected).
    assert "crit" in store, "critical protected entry was evicted!"

    # (3) Frequently accessed favourite survives vs the never-accessed peer.
    assert fav in store, "frequently accessed entry should survive (LFU)"
    # The lonely equal-importance peer, never accessed, should be gone by now.
    assert "lonely" not in store, "never-accessed peer should have been evicted"

    # (4) An old never-accessed banal entry (n1) was evicted.
    assert "n1" not in store, "old never-accessed banal entry should be evicted"

    # (5) All-protected overflow raises MemoryError instead of dropping a critical.
    print("\n  all-protected overflow test:")
    pstore = BoundedMemory(capacity=2, protect_threshold=0.8)
    pstore.add("p1", "protected one", importance=0.9, t_now=0.0)
    pstore.add("p2", "protected two", importance=0.9, t_now=1.0)
    try:
        pstore.add("p3", "protected three", importance=0.9, t_now=2.0)
        raise AssertionError("expected MemoryError when all entries are protected")
    except MemoryError as e:
        print(f"    raised MemoryError: {e}")
        assert "protected" in str(e)

    print("\n  PASS -- capacity bounded, critical never evicted, LFU favours "
          "frequent, all-protected guarded.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 16 HARD Solutions -- Long-horizon memory")
    print("#" * 60)

    hard_ex1_pipeline()
    hard_ex2_eviction()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

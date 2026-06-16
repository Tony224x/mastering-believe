"""
Day 16 -- Long-horizon memory: episodic, semantic, procedural & MemGPT-style paging.

Demonstrates:
  1. EpisodicMemory  -- timestamped event log with decay
  2. SemanticMemory  -- fact store with confidence & source tracking
  3. ProceduralMemory -- skill/strategy store with success rate
  4. Fake embeddings  -- deterministic bag-of-words cosine (no API key needed)
  5. RelevanceScorer  -- recency + importance + similarity (Generative Agents formula)
  6. HierarchicalMemory -- MemGPT-style main context (limited) + archival (unlimited)
                           with page_in / page_out and consolidate()

Dependencies: stdlib only (no API key, no external packages)
Run:
    python domains/agentic-ai/02-code/16-memoire-long-horizon.py
"""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

# ===========================================================================
# FAKE EMBEDDINGS — deterministic, no API key
# ===========================================================================
# We represent text as a normalized bag-of-words vector over a small fixed
# vocabulary derived from the words themselves.  This gives us a real cosine
# similarity that reflects lexical overlap — good enough for demo retrieval.

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    import re
    return re.findall(r"[a-z]+", text.lower())


def fake_embed(text: str, dim: int = 64) -> list[float]:
    """
    Produce a deterministic pseudo-embedding by hashing each token to a
    dimension bucket and accumulating counts, then L2-normalizing.
    Not semantically rich, but cosine similarity is still meaningful for
    texts sharing words.
    """
    tokens = _tokenize(text)
    vec = [0.0] * dim
    for token in tokens:
        # Map each character's ordinal to a bucket in [0, dim)
        bucket = hash(token) % dim
        vec[bucket] += 1.0
    # L2-normalize so cosine similarity works correctly
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity between two L2-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    # Vectors are already normalized, but guard against floating-point drift
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


# ===========================================================================
# 1. BASE MEMORY ENTRY
# ===========================================================================

MemoryType = Literal["episodic", "semantic", "procedural"]


@dataclass
class MemoryEntry:
    """
    A single unit of long-term memory.

    Fields
    ------
    content     : the text stored
    memory_type : episodic | semantic | procedural
    importance  : 0.0-1.0 score assigned at insertion (or updated by reflection)
    created_at  : UNIX timestamp of creation
    last_accessed : UNIX timestamp of last retrieval (used for recency boosting)
    tags        : optional labels for filtering
    embedding   : fake deterministic embedding of content
    id          : unique identifier
    """
    content: str
    memory_type: MemoryType
    importance: float = 0.5          # 0.0 (trivial) to 1.0 (critical)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(init=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        self.embedding = fake_embed(self.content)


# ===========================================================================
# 2. RELEVANCE SCORER — recency + importance + similarity
# ===========================================================================

class RelevanceScorer:
    """
    Implements the three-component scoring from Generative Agents (Park et al., 2023):

        relevance = w_rec * recency + w_imp * importance + w_sim * similarity

    Parameters
    ----------
    decay_lambda : controls how fast recency fades (per hour)
                   0.01 -> slow decay (~50% after 70h)
                   0.1  -> fast decay (~50% after 7h)
    weights      : (w_rec, w_imp, w_sim) must sum to 1.0
    """

    def __init__(
        self,
        decay_lambda: float = 0.02,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
    ) -> None:
        self.decay_lambda = decay_lambda
        self.w_rec, self.w_imp, self.w_sim = weights

    def recency(self, entry: MemoryEntry, t_now: float) -> float:
        """Exponential decay: score = exp(-lambda * hours_elapsed)."""
        hours = (t_now - entry.created_at) / 3600.0
        return math.exp(-self.decay_lambda * hours)

    def score(
        self,
        entry: MemoryEntry,
        query_embedding: list[float],
        t_now: float | None = None,
    ) -> float:
        """Compute the composite relevance score for retrieval."""
        t_now = t_now or time.time()
        rec = self.recency(entry, t_now)
        imp = entry.importance
        sim = cosine_similarity(entry.embedding, query_embedding)
        return self.w_rec * rec + self.w_imp * imp + self.w_sim * sim

    def top_k(
        self,
        entries: list[MemoryEntry],
        query: str,
        k: int = 5,
        t_now: float | None = None,
    ) -> list[tuple[float, MemoryEntry]]:
        """Return top-k entries sorted by relevance descending."""
        t_now = t_now or time.time()
        query_emb = fake_embed(query)
        scored = [(self.score(e, query_emb, t_now), e) for e in entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# ===========================================================================
# 3. EPISODIC MEMORY — the "what happened" store
# ===========================================================================

class EpisodicMemory:
    """
    Append-only log of timestamped events (conversations, tool calls, outcomes).
    Supports decay-based garbage collection.
    """

    def __init__(self, decay_threshold: float = 0.05) -> None:
        self._entries: list[MemoryEntry] = []
        # Entries whose effective score falls below this are eligible for GC
        self.decay_threshold = decay_threshold
        self._scorer = RelevanceScorer()

    def add(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        created_at: float | None = None,
    ) -> MemoryEntry:
        """Record a new episodic event."""
        entry = MemoryEntry(
            content=content,
            memory_type="episodic",
            importance=importance,
            tags=tags or [],
        )
        if created_at is not None:
            entry.created_at = created_at
            entry.last_accessed = created_at
        self._entries.append(entry)
        return entry

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Return top-k most relevant episodes."""
        t_now = time.time()
        results = self._scorer.top_k(self._entries, query, k, t_now)
        for _, entry in results:
            entry.last_accessed = t_now
        return [e for _, e in results]

    def gc(self, t_now: float | None = None) -> int:
        """
        Garbage-collect stale low-importance episodes.
        Returns the number of entries removed.
        """
        t_now = t_now or time.time()
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            # Importance shields important events from deletion
            if e.importance >= 0.7 or self._scorer.recency(e, t_now) >= self.decay_threshold
        ]
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[MemoryEntry]:
        return list(self._entries)


# ===========================================================================
# 4. SEMANTIC MEMORY — the "what we know" store
# ===========================================================================

class SemanticMemory:
    """
    Stores generalized facts derived from episodes (via consolidation) or
    entered directly.  Facts can be updated in place.
    """

    def __init__(self) -> None:
        self._facts: dict[str, MemoryEntry] = {}  # key -> MemoryEntry
        self._scorer = RelevanceScorer(decay_lambda=0.005)  # facts decay slowly

    def add(
        self,
        key: str,
        content: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Upsert a semantic fact.  If the key exists, update confidence."""
        entry = MemoryEntry(
            content=content,
            memory_type="semantic",
            importance=importance,
            tags=tags or [],
        )
        self._facts[key] = entry
        return entry

    def update(self, key: str, new_content: str, new_importance: float | None = None) -> bool:
        """Update an existing fact.  Returns False if the key does not exist."""
        if key not in self._facts:
            return False
        entry = self._facts[key]
        entry.content = new_content
        entry.embedding = fake_embed(new_content)
        if new_importance is not None:
            entry.importance = new_importance
        return True

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """Return top-k most relevant facts."""
        return [e for _, e in self._scorer.top_k(list(self._facts.values()), query, k)]

    def get(self, key: str) -> MemoryEntry | None:
        return self._facts.get(key)

    def __len__(self) -> int:
        return len(self._facts)

    def all(self) -> list[MemoryEntry]:
        return list(self._facts.values())


# ===========================================================================
# 5. PROCEDURAL MEMORY — the "how to do it" store
# ===========================================================================

@dataclass
class Skill:
    """
    An executable strategy: a named sequence of steps with a tracked success rate.
    """
    name: str
    description: str
    steps: list[str]
    success_rate: float = 1.0
    usage_count: int = 0
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(init=False)

    def __post_init__(self) -> None:
        # Embed concatenation of name + description for retrieval
        self.embedding = fake_embed(self.name + " " + self.description)

    def record_outcome(self, success: bool) -> None:
        """Update exponential moving average of success rate."""
        alpha = 0.2  # learning rate
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.usage_count += 1


class ProceduralMemory:
    """Registry of named skills (strategies / action sequences)."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def retrieve(self, query: str, k: int = 3) -> list[Skill]:
        """Return up to k most relevant skills by cosine similarity."""
        q_emb = fake_embed(query)
        scored = [
            (cosine_similarity(s.embedding, q_emb), s)
            for s in self._skills.values()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def record_outcome(self, skill_name: str, success: bool) -> None:
        if skill_name in self._skills:
            self._skills[skill_name].record_outcome(success)

    def __len__(self) -> int:
        return len(self._skills)


# ===========================================================================
# 6. HIERARCHICAL MEMORY — MemGPT-style main context + archival
# ===========================================================================

MAX_MAIN_CONTEXT = 5   # simulate a small "context window" (RAM limit)


class HierarchicalMemory:
    """
    MemGPT-inspired two-tier memory system:

    * main_context : active entries visible to the LLM (limited to MAX_MAIN_CONTEXT)
    * archival     : unlimited persistent store (the "disk")

    page_in(query)  -> load most relevant archival entries into main context
    page_out(entry) -> evict entry from main context to archival
    consolidate()   -> run mock consolidation: group episodes into semantic facts
    """

    def __init__(self) -> None:
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self._scorer = RelevanceScorer()
        # main_context holds references to MemoryEntry objects currently "in RAM"
        self._main_context: list[MemoryEntry] = []
        # archival holds everything else (episodic + semantic together for simplicity)
        self._archival: list[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def observe(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """
        Record an episodic observation.
        If main context is full, automatically page_out the least relevant entry.
        """
        entry = self.episodic.add(content, importance, tags)
        self._archival.append(entry)

        # Add to main context if space is available
        if len(self._main_context) < MAX_MAIN_CONTEXT:
            self._main_context.append(entry)
        else:
            # Page out the least relevant entry (by importance * recency)
            t_now = time.time()
            scored = [
                (self._scorer.recency(e, t_now) * e.importance, e)
                for e in self._main_context
            ]
            scored.sort(key=lambda x: x[0])
            evicted = scored[0][1]
            self._main_context.remove(evicted)
            self._main_context.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Paging
    # ------------------------------------------------------------------

    def page_in(self, query: str, k: int = 3) -> list[MemoryEntry]:
        """
        Search archival storage and load the top-k most relevant entries
        into main context, evicting the least relevant if needed.
        """
        top = self._scorer.top_k(self._archival, query, k)
        loaded: list[MemoryEntry] = []
        for _, entry in top:
            if entry not in self._main_context:
                # Make room if context is full
                if len(self._main_context) >= MAX_MAIN_CONTEXT:
                    self._page_out_weakest(query)
                self._main_context.append(entry)
            loaded.append(entry)
        return loaded

    def _page_out_weakest(self, query: str) -> MemoryEntry | None:
        """Evict the least relevant entry from main context."""
        if not self._main_context:
            return None
        q_emb = fake_embed(query)
        t_now = time.time()
        scored = [(self._scorer.score(e, q_emb, t_now), e) for e in self._main_context]
        scored.sort(key=lambda x: x[0])
        weakest = scored[0][1]
        self._main_context.remove(weakest)
        return weakest

    # ------------------------------------------------------------------
    # Consolidation (mock reflection: episodic -> semantic)
    # ------------------------------------------------------------------

    def consolidate(self, min_episodes: int = 2) -> list[MemoryEntry]:
        """
        Mock consolidation: find groups of episodes sharing a common word
        and merge them into a semantic fact.

        In production this would be an LLM prompt:
          "What general facts can you extract from these episodes?"

        Returns the newly created semantic entries.
        """
        episodes = self.episodic.all()
        if len(episodes) < min_episodes:
            return []

        # Group episodes by their most frequent non-trivial token
        STOP = {"the", "a", "an", "is", "was", "are", "to", "in", "on", "and", "or"}
        groups: dict[str, list[MemoryEntry]] = {}
        for ep in episodes:
            tokens = [t for t in _tokenize(ep.content) if t not in STOP and len(t) > 3]
            if not tokens:
                continue
            top_token = Counter(tokens).most_common(1)[0][0]
            groups.setdefault(top_token, []).append(ep)

        new_facts: list[MemoryEntry] = []
        for keyword, group in groups.items():
            if len(group) < min_episodes:
                continue
            # Build a synthetic fact from the group
            sources = ", ".join(f"ep:{e.id}" for e in group)
            avg_importance = sum(e.importance for e in group) / len(group)
            fact_content = (
                f"[Consolidated from {len(group)} episodes about '{keyword}'] "
                + group[-1].content  # use the most recent episode as the core content
            )
            key = f"consolidated_{keyword}"
            fact = self.semantic.add(
                key=key,
                content=fact_content,
                importance=min(avg_importance + 0.1, 1.0),  # slight boost for consolidation
                tags=["consolidated", keyword],
            )
            new_facts.append(fact)

        return new_facts

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve from ALL memory types: episodic + semantic + procedural embeddings.
        Page in top results to main context automatically.
        """
        all_entries = self._archival + self.semantic.all()
        top = self._scorer.top_k(all_entries, query, k)
        # Page in results
        loaded = self.page_in(query, k=min(k, MAX_MAIN_CONTEXT))
        return [e for _, e in top]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "main_context_entries": len(self._main_context),
            "archival_entries": len(self._archival),
            "semantic_facts": len(self.semantic),
            "skills": len(self.procedural),
        }

    def show_main_context(self) -> None:
        print(f"  Main context ({len(self._main_context)}/{MAX_MAIN_CONTEXT} slots):")
        for e in self._main_context:
            print(f"    [{e.memory_type:10}] imp={e.importance:.2f} | {e.content[:60]}")


# ===========================================================================
# 7. DEMO
# ===========================================================================

def _hr(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_episodic_memory() -> None:
    _hr("1. EPISODIC MEMORY — timestamped events with decay")
    mem = EpisodicMemory()

    # Simulate events spread over time
    now = time.time()
    one_hour = 3600

    mem.add("User asked for a CSV report on Q3 sales", importance=0.8,
            created_at=now - 48 * one_hour)  # 2 days ago
    mem.add("Tool fetch_weather timed out", importance=0.4,
            created_at=now - 24 * one_hour)  # 1 day ago
    mem.add("User said 'perfect, exactly what I needed'", importance=0.6,
            created_at=now - 2 * one_hour)   # 2 hours ago
    mem.add("User asked for a PDF report on Q4 sales", importance=0.8,
            created_at=now - 1 * one_hour)   # 1 hour ago
    mem.add("Payment service returned 503 error", importance=0.9,
            created_at=now - 10 * 60)        # 10 min ago

    print(f"  Total episodes stored: {len(mem)}")
    print("  Retrieving top-3 for query: 'sales report CSV'")
    results = mem.retrieve("sales report CSV", k=3)
    for i, e in enumerate(results, 1):
        print(f"    {i}. [{e.id}] {e.content[:60]}")

    # Garbage collect: stale low-importance entries (simulate 7 days passing)
    removed = mem.gc(t_now=now + 7 * 24 * one_hour)
    print(f"  After GC (simulating 7 days later): {removed} entry/ies removed, {len(mem)} remain")


def demo_semantic_memory() -> None:
    _hr("2. SEMANTIC MEMORY — consolidated facts")
    mem = SemanticMemory()

    mem.add("user_pref_format", "User prefers CSV exports over PDF", importance=0.9)
    mem.add("user_pref_brevity", "User prefers concise bullet-point answers", importance=0.85)
    mem.add("payment_service", "Payment service is unreliable on Fridays", importance=0.8)
    mem.add("product_compat", "Product X requires Python 3.10 or newer", importance=0.75)

    print(f"  Total facts: {len(mem)}")
    print("  Query: 'what format does the user prefer?'")
    results = mem.retrieve("what format does the user prefer?", k=2)
    for e in results:
        print(f"    -> {e.content}")

    # Update a fact (new information overrides the old one)
    updated = mem.update("user_pref_format", "User now prefers JSON exports", new_importance=0.95)
    print(f"  Updated 'user_pref_format': {updated}")
    print(f"  New value: {mem.get('user_pref_format').content}")


def demo_procedural_memory() -> None:
    _hr("3. PROCEDURAL MEMORY — skills and strategies")
    mem = ProceduralMemory()

    mem.add(Skill(
        name="generate_csv_report",
        description="Generate a sales report as CSV and email it to the user",
        steps=["query SQL table", "format with tabulate", "export CSV", "send email"],
        tags=["report", "csv"],
    ))
    mem.add(Skill(
        name="debug_import_error",
        description="Troubleshoot Python import errors step by step",
        steps=["check venv", "run pip list", "check PYTHONPATH", "reinstall package"],
        tags=["debug", "python"],
    ))
    mem.add(Skill(
        name="handle_impatient_user",
        description="When user is in a hurry: answer first, explain later",
        steps=["give direct answer", "add brief justification", "offer follow-up"],
        tags=["communication"],
    ))

    print(f"  Total skills: {len(mem)}")
    print("  Query: 'how to export data as CSV?'")
    results = mem.retrieve("how to export data as CSV?", k=2)
    for s in results:
        print(f"    Skill: {s.name} (success_rate={s.success_rate:.2f})")
        for step in s.steps:
            print(f"      - {step}")

    # Record outcomes to update success rate
    mem.record_outcome("generate_csv_report", success=True)
    mem.record_outcome("generate_csv_report", success=False)
    after = mem._skills["generate_csv_report"].success_rate
    print(f"  generate_csv_report success_rate after 1 success + 1 failure: {after:.3f}")


def demo_relevance_scorer() -> None:
    _hr("4. RELEVANCE SCORING — recency + importance + similarity")
    scorer = RelevanceScorer(decay_lambda=0.05, weights=(0.3, 0.3, 0.4))
    now = time.time()

    entries = [
        MemoryEntry("User wants CSV format for all reports",
                    "episodic", importance=0.9, created_at=now - 0),       # just now
        MemoryEntry("Tool weather_api timed out",
                    "episodic", importance=0.3, created_at=now - 3600),    # 1h ago
        MemoryEntry("User asked about CSV export options",
                    "episodic", importance=0.7, created_at=now - 86400),   # 1d ago
        MemoryEntry("Payment service down on Fridays",
                    "semantic", importance=0.8, created_at=now - 86400 * 7), # 7d ago
    ]

    query = "export format CSV report"
    q_emb = fake_embed(query)
    print(f"  Query: '{query}'")
    print(f"  {'Content':50s} {'rec':>5} {'imp':>5} {'sim':>5} {'total':>6}")
    print(f"  {'-' * 75}")
    for e in entries:
        rec = scorer.recency(e, now)
        imp = e.importance
        sim = cosine_similarity(e.embedding, q_emb)
        total = scorer.score(e, q_emb, now)
        print(f"  {e.content[:50]:50s} {rec:5.3f} {imp:5.2f} {sim:5.3f} {total:6.3f}")


def demo_hierarchical_memory() -> None:
    _hr("5. HIERARCHICAL MEMORY — MemGPT-style paging + consolidation")
    hmem = HierarchicalMemory()

    # Seed some procedural skills
    hmem.procedural.add(Skill(
        name="csv_report",
        description="generate and send CSV report",
        steps=["sql query", "format csv", "email user"],
    ))

    print("  Observing 7 episodes (context limit = 5)...")
    hmem.observe("Alice requested Q1 sales report CSV", importance=0.8, tags=["alice", "report"])
    hmem.observe("Generated Q1 CSV report successfully", importance=0.6, tags=["report"])
    hmem.observe("Alice said 'great, thank you!'", importance=0.5, tags=["alice"])
    hmem.observe("Bob requested Q2 sales report CSV", importance=0.8, tags=["bob", "report"])
    hmem.observe("Generated Q2 CSV report successfully", importance=0.6, tags=["report"])
    hmem.observe("Payment service returned 503", importance=0.9, tags=["payment", "error"])
    hmem.observe("Bob asked for PDF report, PDF not supported", importance=0.7, tags=["bob"])

    print("\n  Status after 7 observations:")
    print(f"    {hmem.status()}")
    hmem.show_main_context()

    # Paging in: load archival entries relevant to "report CSV"
    print("\n  Paging in: query = 'report CSV'")
    paged = hmem.page_in("report CSV", k=3)
    for e in paged:
        print(f"    paged in [{e.id}]: {e.content[:60]}")
    hmem.show_main_context()

    # Consolidation: extract semantic facts from episodes
    print("\n  Running consolidate()...")
    new_facts = hmem.consolidate(min_episodes=2)
    print(f"  {len(new_facts)} new semantic fact(s) created:")
    for f in new_facts:
        print(f"    [{f.id}] {f.content[:80]}")

    # Final retrieval spanning all memory types
    print("\n  Final retrieve: query = 'sales report user preference'")
    results = hmem.retrieve("sales report user preference", k=4)
    for i, e in enumerate(results, 1):
        print(f"    {i}. [{e.memory_type:10}] imp={e.importance:.2f} | {e.content[:60]}")

    print(f"\n  Final status: {hmem.status()}")


if __name__ == "__main__":
    demo_episodic_memory()
    demo_semantic_memory()
    demo_procedural_memory()
    demo_relevance_scorer()
    demo_hierarchical_memory()
    print("\n[Day 16 demo complete]\n")

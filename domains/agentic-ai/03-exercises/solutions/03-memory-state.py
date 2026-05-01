"""
Solutions -- Day 3: Memory & State

Contains solutions for:
  - Easy Ex 1:  Token-aware sliding window with stats
  - Easy Ex 2:  Working memory with type validation
  - Easy Ex 3:  Checkpoint serialisation round-trip
  - Medium Ex 1: Adaptive hybrid memory with importance markers
  - Medium Ex 2: Vector memory with recency weighting and TTL
  - Medium Ex 3: Immutable state + reducers + time-travel

Run:  python 03-exercises/solutions/03-memory-state.py
Each solution is self-contained.
"""

import json
import hashlib
import time
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ==========================================================================
# SHARED UTILS
# ==========================================================================

def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def estimate_message_tokens(msg: dict) -> int:
    return estimate_tokens(msg.get("role", "")) + estimate_tokens(msg.get("content", "")) + 4


def mock_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic mock embedding via SHA-256 hash."""
    h = hashlib.sha256(text.encode()).digest()
    while len(h) < dim:
        h += hashlib.sha256(h).digest()
    raw = np.array([(b / 127.5) - 1.0 for b in h[:dim]], dtype=np.float64)
    norm = np.linalg.norm(raw)
    return raw / norm if norm > 0 else raw


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


# ==========================================================================
# EASY EXERCISE 1 — Token-aware sliding window with stats
# ==========================================================================

class SlidingWindowMemoryWithStats:
    """Sliding window memory that tracks detailed statistics."""

    def __init__(self, max_messages: int = 8, max_tokens: int = 300):
        self._messages: list[dict] = []
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._total_seen = 0       # Total messages ever added
        self._dropped_count = 0    # Messages dropped by trimming

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        self._total_seen += 1
        self._trim()

    def _trim(self) -> None:
        # Trim by count
        while len(self._messages) > self._max_messages:
            self._messages.pop(0)
            self._dropped_count += 1
        # Trim by token budget
        while self._get_total_tokens() > self._max_tokens and len(self._messages) > 1:
            self._messages.pop(0)
            self._dropped_count += 1

    def _get_total_tokens(self) -> int:
        return sum(estimate_message_tokens(m) for m in self._messages)

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def get_stats(self) -> dict:
        """Return 5 stats about the current state of memory."""
        token_count = self._get_total_tokens()
        window_size = len(self._messages)
        return {
            "total_messages_seen": self._total_seen,
            "current_window_size": window_size,
            "current_token_count": token_count,
            "dropped_messages": self._dropped_count,
            "avg_tokens_per_message": round(token_count / window_size, 1) if window_size > 0 else 0,
        }


def demo_easy_1():
    """Token-aware sliding window with stats."""
    print("\n" + "=" * 60)
    print("  EASY 1: Sliding Window with Stats")
    print("=" * 60)

    mem = SlidingWindowMemoryWithStats(max_messages=8, max_tokens=300)

    # 15 messages of varying sizes
    messages = [
        ("user", "Hi"),
        ("assistant", "Hello! How can I help?"),
        ("user", "I need a laptop"),
        ("assistant", "Sure! What's your budget?"),
        ("user", "500 EUR maximum, this is very important, please remember this constraint throughout our conversation"),
        ("assistant", "Got it, 500 EUR max. Let me search for options that fit your budget."),
        ("user", "Ok"),
        ("assistant", "I found 3 laptops: ASUS VivoBook (449 EUR), Lenovo IdeaPad (399 EUR), Acer Aspire (479 EUR). All under your 500 EUR budget."),
        ("user", "Tell me about the ASUS"),
        ("assistant", "The ASUS VivoBook features a Ryzen 5 processor, 8GB RAM, 256GB SSD. Good for daily tasks and light development work. Weighs 1.7kg."),
        ("user", "And the Lenovo?"),
        ("assistant", "The Lenovo IdeaPad has an Intel i5, 8GB RAM, 512GB SSD. Better storage but slightly heavier at 1.9kg. The screen is average quality."),
        ("user", "Which one do you recommend?"),
        ("assistant", "I recommend the ASUS VivoBook for the better screen and lighter weight. The Lenovo is a good alternative if you need more storage."),
        ("user", "Thanks! What was my budget again?"),
    ]

    for i, (role, content) in enumerate(messages):
        mem.add_message(role, content)
        stats = mem.get_stats()
        print(f"  Msg {i+1:2d} | Window: {stats['current_window_size']} msgs, "
              f"~{stats['current_token_count']:3d} tok, "
              f"dropped: {stats['dropped_messages']}, "
              f"avg: {stats['avg_tokens_per_message']:.1f} tok/msg")

    print(f"\n  Final window contents:")
    for msg in mem.get_messages():
        print(f"    [{msg['role']}] {msg['content'][:70]}{'...' if len(msg['content']) > 70 else ''}")

    stats = mem.get_stats()
    print(f"\n  Final stats: {json.dumps(stats, indent=2)}")
    print(f"  NOTE: The budget message from msg 5 may have been dropped!")


# ==========================================================================
# EASY EXERCISE 2 — Working memory with type validation
# ==========================================================================

class TypedWorkingMemory:
    """Working memory with runtime type checking."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._types: dict[str, type] = {}  # Expected type per key

    def set(self, key: str, value: Any, source: str = "agent") -> None:
        """Untyped set — for backwards compatibility."""
        self._store[key] = value

    def set_typed(self, key: str, value: Any, expected_type: type, source: str = "agent") -> None:
        """
        Set a value with type checking.
        Raises TypeError if value doesn't match expected_type.
        """
        if not isinstance(value, expected_type):
            raise TypeError(
                f"WorkingMemory type error: key '{key}' expects {expected_type.__name__}, "
                f"got {type(value).__name__} (value: {repr(value)[:50]})"
            )
        self._store[key] = value
        self._types[key] = expected_type

    def get(self, key: str, default: Any = None) -> Any:
        """Untyped get — for backwards compatibility."""
        return self._store.get(key, default)

    def get_typed(self, key: str, expected_type: type) -> Any:
        """
        Get a value with type checking at read time.
        Raises TypeError if stored value doesn't match expected_type.
        Raises KeyError if key not found.
        """
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in working memory")

        value = self._store[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"WorkingMemory type error: key '{key}' contains {type(value).__name__}, "
                f"but was read as {expected_type.__name__} (value: {repr(value)[:50]})"
            )
        return value

    def __repr__(self):
        return f"TypedWorkingMemory({list(self._store.keys())})"


def demo_easy_2():
    """Working memory with type validation."""
    print("\n" + "=" * 60)
    print("  EASY 2: Typed Working Memory")
    print("=" * 60)

    wm = TypedWorkingMemory()

    # Valid cases
    print("\n  --- Valid cases ---")
    test_cases_valid = [
        ("budget", 500, int),
        ("currency", "EUR", str),
        ("in_stock", True, bool),
        ("candidates", ["ASUS", "Lenovo"], list),
        ("scores", {"ASUS": 4.1, "Lenovo": 3.9}, dict),
        ("avg_price", 424.0, float),
    ]
    for key, value, typ in test_cases_valid:
        wm.set_typed(key, value, typ)
        print(f"  set_typed('{key}', {repr(value)}, {typ.__name__}) -> OK")

    # Invalid cases — set_typed
    print("\n  --- Invalid set_typed cases ---")
    test_cases_invalid_set = [
        ("budget", "five hundred", int),
        ("currency", 42, str),
        ("in_stock", "yes", bool),
    ]
    for key, value, typ in test_cases_invalid_set:
        try:
            wm.set_typed(key, value, typ)
            print(f"  set_typed('{key}', {repr(value)}, {typ.__name__}) -> UNEXPECTED OK")
        except TypeError as e:
            print(f"  set_typed('{key}', {repr(value)}, {typ.__name__}) -> TypeError: {e}")

    # Invalid cases — get_typed
    print("\n  --- Invalid get_typed cases ---")
    test_cases_invalid_get = [
        ("budget", str),      # Stored as int, read as str
        ("currency", int),    # Stored as str, read as int
        ("in_stock", list),   # Stored as bool, read as list
    ]
    for key, typ in test_cases_invalid_get:
        try:
            val = wm.get_typed(key, typ)
            print(f"  get_typed('{key}', {typ.__name__}) -> UNEXPECTED: {val}")
        except TypeError as e:
            print(f"  get_typed('{key}', {typ.__name__}) -> TypeError: {e}")

    # Untyped set still works
    print("\n  --- Untyped set (backwards compat) ---")
    wm.set("anything", {"mixed": [1, "two", 3.0]})
    print(f"  set('anything', ...) -> OK, value: {wm.get('anything')}")


# ==========================================================================
# EASY EXERCISE 3 — Checkpoint serialisation round-trip
# ==========================================================================

@dataclass
class SimpleAgentState:
    """Typed state for checkpoint exercise."""
    messages: list[dict] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    task: str = ""
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "working_memory": self.working_memory,
            "iteration": self.iteration,
            "task": self.task,
            "tools_used": self.tools_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimpleAgentState":
        return cls(
            messages=data.get("messages", []),
            working_memory=data.get("working_memory", {}),
            iteration=data.get("iteration", 0),
            task=data.get("task", ""),
            tools_used=data.get("tools_used", []),
        )


def save_checkpoint(state: SimpleAgentState, filepath: str) -> None:
    """Serialize state to JSON file."""
    checkpoint = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "state": state.to_dict(),
    }
    Path(filepath).write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))


def load_checkpoint(filepath: str) -> SimpleAgentState:
    """
    Deserialize state from JSON file.
    Raises ValueError on corrupted files, FileNotFoundError if missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    raw = path.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted checkpoint (invalid JSON): {e}")

    if "state" not in data:
        raise ValueError("Corrupted checkpoint: missing 'state' key")

    return SimpleAgentState.from_dict(data["state"])


def demo_easy_3():
    """Checkpoint serialisation round-trip."""
    print("\n" + "=" * 60)
    print("  EASY 3: Checkpoint Serialisation")
    print("=" * 60)

    # Create a realistic state
    original = SimpleAgentState(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Find me a laptop under 500 EUR."},
            {"role": "assistant", "content": "Let me search for laptops."},
            {"role": "tool", "content": '{"results": [{"name": "ASUS", "price": 449}]}'},
            {"role": "assistant", "content": "I found the ASUS VivoBook at 449 EUR."},
        ],
        working_memory={
            "budget_max": 500,
            "currency": "EUR",
            "top_pick": "ASUS VivoBook",
        },
        iteration=5,
        task="laptop_recommendation",
        tools_used=["search_products", "calculate", "get_reviews"],
    )

    # Save
    tmp = tempfile.mktemp(suffix=".json")
    save_checkpoint(original, tmp)
    print(f"\n  Saved checkpoint to: {Path(tmp).name}")

    # Load
    loaded = load_checkpoint(tmp)
    print(f"  Loaded checkpoint successfully")

    # Compare ALL fields
    print(f"\n  Verification:")
    checks = [
        ("messages", original.messages == loaded.messages),
        ("working_memory", original.working_memory == loaded.working_memory),
        ("iteration", original.iteration == loaded.iteration),
        ("task", original.task == loaded.task),
        ("tools_used", original.tools_used == loaded.tools_used),
    ]
    all_ok = True
    for field_name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"    {field_name}: {status}")
        if not ok:
            all_ok = False
    assert all_ok, "Round-trip failed — data loss detected!"
    print(f"  All fields match — round-trip successful!")

    # Test corrupted file
    print(f"\n  --- Corrupted file test ---")
    corrupt_path = tempfile.mktemp(suffix=".json")
    Path(corrupt_path).write_text("{ this is not valid json !!!")
    try:
        load_checkpoint(corrupt_path)
        print(f"  UNEXPECTED: no error on corrupted file")
    except ValueError as e:
        print(f"  Caught ValueError: {e}")

    # Test missing file
    try:
        load_checkpoint("/nonexistent/path/checkpoint.json")
        print(f"  UNEXPECTED: no error on missing file")
    except FileNotFoundError as e:
        print(f"  Caught FileNotFoundError: {e}")

    # Cleanup
    Path(tmp).unlink(missing_ok=True)
    Path(corrupt_path).unlink(missing_ok=True)


# ==========================================================================
# MEDIUM EXERCISE 1 — Adaptive hybrid memory with importance markers
# ==========================================================================

IMPORTANCE_MARKERS = {"important", "budget", "deadline", "remember", "n'oublie pas", "critical", "must", "constraint"}


def mock_summarize_for_hybrid(messages: list[dict]) -> tuple[str, str]:
    """
    Mock dual summarizer — returns (executive_summary, detailed_summary).
    In production, two LLM calls with different prompts.
    """
    facts = []
    for msg in messages:
        c = msg.get("content", "").lower()
        if any(kw in c for kw in IMPORTANCE_MARKERS):
            facts.append(msg["content"][:100])

    executive = f"Key facts: {'; '.join(facts[:3])}" if facts else "No critical facts."
    detailed = f"Conversation covered {len(messages)} exchanges. " + " | ".join(facts[:6]) if facts else f"General conversation ({len(messages)} msgs)."
    return executive[:200], detailed[:500]


class AdaptiveHybridMemory:
    """
    Hybrid memory that adapts summarization frequency and protects important messages.

    Features:
    - Adaptive threshold based on conversation pace
    - Important messages are protected from summarization
    - Dual-level summaries (executive + detailed)
    """

    def __init__(self, recent_window: int = 8, base_threshold: int = 20, max_tokens: int = 2000):
        self._messages: list[dict] = []
        self._important_messages: list[dict] = []  # Protected from summarization
        self._executive_summary: str = ""
        self._detailed_summary: str = ""
        self._recent_window = recent_window
        self._base_threshold = base_threshold
        self._max_tokens = max_tokens
        self._msgs_since_last_summary = 0
        self._summary_count = 0
        self._current_threshold = base_threshold

    def _is_important(self, content: str) -> bool:
        """Check if a message contains importance markers."""
        lower = content.lower()
        return any(marker in lower for marker in IMPORTANCE_MARKERS)

    def add_message(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self._messages.append(msg)
        self._msgs_since_last_summary += 1

        # Protect important messages
        if self._is_important(content):
            self._important_messages.append(msg)

        # Check if we need to summarize
        if len(self._messages) > self._current_threshold:
            self._condense()

    def _condense(self) -> None:
        """Summarize, adapting threshold based on conversation pace."""
        # Adaptive threshold: fast conversations get more frequent summaries
        if self._msgs_since_last_summary < 5:
            self._current_threshold = max(10, self._base_threshold - 5)  # Lower threshold
        elif self._msgs_since_last_summary > 20:
            self._current_threshold = self._base_threshold + 10  # Higher threshold

        # Split messages: keep recent window + important
        cutoff = len(self._messages) - self._recent_window
        if cutoff <= 0:
            return

        old = self._messages[:cutoff]
        self._messages = self._messages[cutoff:]

        # Generate dual summaries
        self._executive_summary, self._detailed_summary = mock_summarize_for_hybrid(old)

        self._msgs_since_last_summary = 0
        self._summary_count += 1

    def get_messages(self) -> list[dict]:
        """Return: executive summary + important messages + recent buffer."""
        result = []

        if self._executive_summary:
            result.append({"role": "system", "content": f"EXECUTIVE SUMMARY: {self._executive_summary}"})
            result.append({"role": "system", "content": f"DETAILED CONTEXT: {self._detailed_summary}"})

        # Add protected important messages (deduplicated with recent)
        recent_contents = {m["content"] for m in self._messages}
        for imp in self._important_messages:
            if imp["content"] not in recent_contents:
                result.append({"role": "system", "content": f"[IMPORTANT] {imp['content']}"})

        result.extend(self._messages[-self._recent_window:])
        return result

    def get_token_count(self) -> int:
        return sum(estimate_message_tokens(m) for m in self.get_messages())


def demo_medium_1():
    """Adaptive hybrid memory with importance markers."""
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Adaptive Hybrid Memory")
    print("=" * 60)

    mem = AdaptiveHybridMemory(recent_window=6, base_threshold=12, max_tokens=2000)

    # 30 messages — message 5 has an important constraint
    messages = [
        ("user", "Hi, I'm looking for a laptop."),
        ("assistant", "I'd be happy to help!"),
        ("user", "I'm a developer."),
        ("assistant", "Great, what kind of development?"),
        ("user", "IMPORTANT: My budget is exactly 500 EUR, do NOT go over this."),  # ← CRITICAL
        ("assistant", "Understood, 500 EUR max."),
        ("user", "I prefer light laptops."),
        ("assistant", "Noted, lightweight preference."),
        ("user", "Ok"),
        ("assistant", "Let me search for options."),
        ("user", "Sounds good"),
        ("assistant", "I found 3 options under 500 EUR."),
        ("user", "Tell me about the first one."),
        ("assistant", "The ASUS VivoBook: 449 EUR, 1.7kg, Ryzen 5."),
        ("user", "And the second?"),
        ("assistant", "The Lenovo IdeaPad: 399 EUR, 1.9kg, Intel i5."),
        ("user", "Interesting"),
        ("assistant", "Shall I compare them?"),
        ("user", "Yes please"),
        ("assistant", "ASUS has better screen, Lenovo has more storage."),
        ("user", "Hmm"),
        ("assistant", "Both are good choices for development."),
        ("user", "Any reviews?"),
        ("assistant", "ASUS: 4.1/5, Lenovo: 3.9/5."),
        ("user", "Ok"),
        ("assistant", "The ASUS seems like the better choice overall."),
        ("user", "Makes sense"),
        ("assistant", "Want me to check warranty?"),
        ("user", "Remind me — what was my budget constraint again?"),  # ← tests memory
        ("assistant", "Checking..."),
    ]

    for i, (role, content) in enumerate(messages):
        mem.add_message(role, content)

    print(f"\n  After 30 messages:")
    print(f"  Token count: ~{mem.get_token_count()}")
    print(f"  Summaries generated: {mem._summary_count}")

    print(f"\n  Messages sent to LLM:")
    for msg in mem.get_messages():
        c = msg['content']
        tag = ""
        if "[IMPORTANT]" in c:
            tag = " <<< PROTECTED"
        if "EXECUTIVE" in c or "DETAILED" in c:
            tag = " <<< SUMMARY"
        print(f"    [{msg['role']}] {c[:80]}{'...' if len(c) > 80 else ''}{tag}")

    # Verify the budget constraint is still accessible
    all_content = " ".join(m["content"] for m in mem.get_messages())
    budget_preserved = "500" in all_content
    print(f"\n  Budget constraint (500 EUR) preserved: {'YES' if budget_preserved else 'NO — BUG!'}")


# ==========================================================================
# MEDIUM EXERCISE 2 — Vector memory with recency weighting and TTL
# ==========================================================================

@dataclass
class TimedMemoryEntry:
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0           # Unix timestamp
    ttl: float | None = None          # Seconds until expiry (None = never)
    importance: float = 0.5           # 0.0 to 1.0
    access_count: int = 0             # Times returned in search results
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:12]
        if self.created_at == 0.0:
            self.created_at = time.time()

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class VectorMemoryWithRecency:
    """
    Vector memory with recency weighting, TTL, and importance scoring.

    Score = similarity * w_sim + recency * w_rec + importance * w_imp
    """

    def __init__(
        self,
        embed_fn=mock_embed,
        dim: int = 64,
        w_similarity: float = 0.5,
        w_recency: float = 0.3,
        w_importance: float = 0.2,
        recency_half_life: float = 3600.0,  # Seconds — score halves every hour
    ):
        self._entries: list[TimedMemoryEntry] = []
        self._embed_fn = embed_fn
        self._dim = dim
        self._w_sim = w_similarity
        self._w_rec = w_recency
        self._w_imp = w_importance
        self._half_life = recency_half_life
        self._seen_ids: set[str] = set()

    def _recency_score(self, created_at: float) -> float:
        """Exponential decay: 1.0 at t=0, 0.5 at t=half_life."""
        age = time.time() - created_at
        return 2.0 ** (-age / self._half_life) if self._half_life > 0 else 1.0

    def store(self, text: str, metadata: dict | None = None, ttl: float | None = None,
              created_at: float | None = None) -> str:
        entry = TimedMemoryEntry(
            text=text,
            embedding=self._embed_fn(text, self._dim),
            metadata=metadata or {},
            ttl=ttl,
        )
        if created_at is not None:
            entry.created_at = created_at
        if entry.id in self._seen_ids:
            return entry.id
        self._entries.append(entry)
        self._seen_ids.add(entry.id)
        return entry.id

    def search(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict]:
        query_emb = self._embed_fn(query, self._dim)
        results = []

        for entry in self._entries:
            # Skip expired entries
            if entry.is_expired():
                continue
            # Metadata filter
            if metadata_filter:
                if not all(entry.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            sim = cosine_similarity(query_emb, entry.embedding)
            rec = self._recency_score(entry.created_at)
            imp = entry.importance

            # Composite score
            score = sim * self._w_sim + rec * self._w_rec + imp * self._w_imp

            results.append({
                "text": entry.text,
                "score": round(score, 4),
                "similarity": round(sim, 4),
                "recency": round(rec, 4),
                "importance": round(imp, 4),
                "metadata": entry.metadata,
                "id": entry.id,
            })

            # Boost importance when accessed
            entry.access_count += 1
            entry.importance = min(1.0, entry.importance + 0.1)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        before = len(self._entries)
        expired_ids = [e.id for e in self._entries if e.is_expired()]
        self._entries = [e for e in self._entries if not e.is_expired()]
        for eid in expired_ids:
            self._seen_ids.discard(eid)
        return before - len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)


def demo_medium_2():
    """Vector memory with recency weighting and TTL."""
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Vector Memory with Recency & TTL")
    print("=" * 60)

    vmem = VectorMemoryWithRecency(
        dim=64,
        w_similarity=0.5,
        w_recency=0.3,
        w_importance=0.2,
        recency_half_life=60.0,  # 1 minute half-life for demo
    )

    now = time.time()

    # Store entries at different times
    entries = [
        ("Budget is 500 EUR maximum.", {"type": "constraint"}, None, now),          # Recent, no TTL
        ("User prefers ASUS brand.", {"type": "preference"}, None, now - 30),       # 30s ago
        ("ThinkPad X1 was too expensive.", {"type": "history"}, None, now - 120),   # 2min ago (old)
        ("API key: sk-test123", {"type": "system"}, 2.0, now - 5),                 # TTL=2s, 5s ago -> EXPIRED
        ("User is a developer.", {"type": "profile"}, None, now - 10),             # 10s ago
        ("Temporary: checking price...", {"type": "temp"}, 1.0, now - 3),          # TTL=1s -> EXPIRED
    ]

    for text, meta, ttl, ts in entries:
        vmem.store(text, metadata=meta, ttl=ttl, created_at=ts)

    print(f"\n  Stored {vmem.size} entries (2 with expired TTL)")

    # Search — should NOT return expired entries
    print(f"\n  Search: 'laptop budget'")
    results = vmem.search("laptop budget", top_k=5)
    for r in results:
        print(f"    Score {r['score']:.3f} (sim={r['similarity']:.3f}, rec={r['recency']:.3f}, imp={r['importance']:.3f}): {r['text'][:50]}")

    # Verify expired entries are excluded
    result_texts = [r["text"] for r in results]
    assert "API key: sk-test123" not in result_texts, "Expired entry should be excluded!"
    assert "Temporary: checking price..." not in result_texts, "Expired entry should be excluded!"
    print(f"\n  Expired entries correctly excluded from results!")

    # Cleanup expired
    removed = vmem.cleanup_expired()
    print(f"  cleanup_expired() removed {removed} entries. Size now: {vmem.size}")

    # Importance boost — search again, same entry should have higher importance
    print(f"\n  Search again to see importance boost:")
    results2 = vmem.search("laptop budget", top_k=3)
    for r in results2:
        print(f"    imp={r['importance']:.2f}: {r['text'][:50]}")


# ==========================================================================
# MEDIUM EXERCISE 3 — Immutable state + reducers + time-travel
# ==========================================================================

@dataclass(frozen=True)
class ImmutableAgentState:
    """Frozen dataclass — truly immutable. Any 'mutation' creates a new instance."""
    messages: tuple = ()
    working_memory: tuple = ()  # tuple of (key, value) pairs since frozen
    iteration: int = 0
    total_tokens: int = 0
    status: str = "running"  # running | paused | done | error
    error_message: str = ""


def reduce(state: ImmutableAgentState, action: dict) -> ImmutableAgentState:
    """
    Pure reducer: takes (state, action), returns NEW state. No side effects.

    This is the core of the functional state management pattern.
    Every state transition is explicit, traceable, and reversible.
    """
    match action["type"]:
        case "ADD_MESSAGE":
            msg = (action["role"], action["content"])
            return replace(state, messages=state.messages + (msg,))

        case "SET_MEMORY":
            # Add/update a key-value pair in working memory
            key, value = action["key"], action["value"]
            # Remove existing key if present, then add new
            existing = tuple((k, v) for k, v in state.working_memory if k != key)
            return replace(state, working_memory=existing + ((key, value),))

        case "INCREMENT":
            tokens = action.get("tokens", 0)
            return replace(
                state,
                iteration=state.iteration + 1,
                total_tokens=state.total_tokens + tokens,
            )

        case "FINISH":
            return replace(state, status="done")

        case "ERROR":
            return replace(
                state,
                status="error",
                error_message=action.get("message", "Unknown error"),
            )

        case "PAUSE":
            return replace(state, status="paused")

        case _:
            return state  # Unknown action — state unchanged


class StateHistory:
    """
    Records every (state, action) pair for time-travel debugging.

    Features:
    - get_state_at(step): return state at any step
    - replay_from(step): return all states from step onwards
    - diff(a, b): show what changed between two steps
    - find_action(predicate): find first action matching a condition
    """

    def __init__(self):
        self._history: list[tuple[ImmutableAgentState, dict]] = []

    def push(self, state: ImmutableAgentState, action: dict) -> None:
        self._history.append((state, action))

    def get_state_at(self, step: int) -> ImmutableAgentState | None:
        if 0 <= step < len(self._history):
            return self._history[step][0]
        return None

    def get_action_at(self, step: int) -> dict | None:
        if 0 <= step < len(self._history):
            return self._history[step][1]
        return None

    def replay_from(self, step: int) -> list[tuple[ImmutableAgentState, dict]]:
        return self._history[step:]

    def diff(self, step_a: int, step_b: int) -> dict:
        """Show what changed between two steps."""
        state_a = self.get_state_at(step_a)
        state_b = self.get_state_at(step_b)
        if state_a is None or state_b is None:
            return {"error": f"Invalid steps: {step_a}, {step_b}"}

        changes = {}
        for field_name in ["messages", "working_memory", "iteration", "total_tokens", "status", "error_message"]:
            val_a = getattr(state_a, field_name)
            val_b = getattr(state_b, field_name)
            if val_a != val_b:
                changes[field_name] = {"from": val_a, "to": val_b}
        return changes

    def find_action(self, predicate: Callable[[dict], bool]) -> tuple[int, dict] | None:
        """Find first action matching predicate. Returns (step, action) or None."""
        for i, (_, action) in enumerate(self._history):
            if predicate(action):
                return (i, action)
        return None

    @property
    def length(self) -> int:
        return len(self._history)


def demo_medium_3():
    """Immutable state + reducers + time-travel."""
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Immutable State + Reducers + Time-Travel")
    print("=" * 60)

    history = StateHistory()
    state = ImmutableAgentState()

    # 8 actions simulating an agent run
    actions = [
        {"type": "ADD_MESSAGE", "role": "user", "content": "Find me a laptop under 500 EUR"},
        {"type": "SET_MEMORY", "key": "budget", "value": 500},
        {"type": "INCREMENT", "tokens": 150},
        {"type": "ADD_MESSAGE", "role": "assistant", "content": "Searching for laptops..."},
        {"type": "SET_MEMORY", "key": "results_count", "value": 3},
        {"type": "INCREMENT", "tokens": 200},
        {"type": "ERROR", "message": "Tool search_products timed out after 30s"},
        {"type": "ADD_MESSAGE", "role": "assistant", "content": "Sorry, search failed. Retrying..."},
    ]

    print(f"\n  Executing {len(actions)} actions:")
    for i, action in enumerate(actions):
        history.push(state, action)
        new_state = reduce(state, action)
        print(f"    Step {i}: {action['type']:<15} -> iter={new_state.iteration}, "
              f"msgs={len(new_state.messages)}, status={new_state.status}")
        state = new_state

    # Time-travel: show state at step 3
    print(f"\n  --- Time-travel to step 3 ---")
    s3 = history.get_state_at(3)
    if s3:
        print(f"    Iteration: {s3.iteration}")
        print(f"    Messages: {len(s3.messages)}")
        print(f"    Working memory: {dict(s3.working_memory)}")
        print(f"    Status: {s3.status}")

    # Diff between step 2 and step 5
    print(f"\n  --- Diff: step 2 vs step 5 ---")
    changes = history.diff(2, 5)
    for field_name, change in changes.items():
        from_val = repr(change['from'])[:60]
        to_val = repr(change['to'])[:60]
        print(f"    {field_name}: {from_val} -> {to_val}")

    # Find the error action
    print(f"\n  --- Find error action ---")
    result = history.find_action(lambda a: a["type"] == "ERROR")
    if result:
        step, action = result
        print(f"    Found at step {step}: {action}")

    # Branching: load state at step 4, apply different actions
    print(f"\n  --- Branching from step 4 ---")
    branch_state = history.get_state_at(4)
    if branch_state:
        # Branch A: original (error happened)
        branch_a = reduce(branch_state, actions[5])  # INCREMENT
        branch_a = reduce(branch_a, actions[6])       # ERROR

        # Branch B: alternative (success instead of error)
        branch_b = reduce(branch_state, {"type": "INCREMENT", "tokens": 180})
        branch_b = reduce(branch_b, {"type": "SET_MEMORY", "key": "top_pick", "value": "ASUS VivoBook"})
        branch_b = reduce(branch_b, {"type": "FINISH"})

        print(f"    Branch A (original): status={branch_a.status}, error='{branch_a.error_message}'")
        print(f"    Branch B (alt path): status={branch_b.status}, memory={dict(branch_b.working_memory)}")
        print(f"    Same starting point, different outcomes — that's time-travel!")


# ==========================================================================
# MAIN — Run all solutions
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Solutions — Day 3: Memory & State")
    print("=" * 60)

    # Easy exercises
    demo_easy_1()
    demo_easy_2()
    demo_easy_3()

    # Medium exercises
    demo_medium_1()
    demo_medium_2()
    demo_medium_3()

    print("\n" + "=" * 60)
    print("  All solutions complete!")
    print("  Hard exercises are architecture-heavy — see the exercise")
    print("  descriptions for implementation guidance.")
    print("=" * 60)

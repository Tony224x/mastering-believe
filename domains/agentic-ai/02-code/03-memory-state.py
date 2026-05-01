"""
Day 3 -- Memory & State: Complete Memory System from Scratch

Demonstrates:
  1. ConversationBufferMemory   — keep all messages (simplest)
  2. ConversationSummaryMemory  — summarize old messages (mock LLM summarizer)
  3. SlidingWindowMemory        — keep last N messages with token counting
  4. HybridMemory               — summary for old + buffer for recent
  5. WorkingMemory              — scratchpad with key-value store
  6. VectorMemory               — numpy embeddings + cosine similarity search
  7. Checkpointing              — save/restore agent state to/from JSON
  8. Demo agent that uses ALL memory types for a multi-step research task

Dependencies: stdlib + numpy only. No frameworks.

Two modes:
  - SIMULATED mode: Works without any API key (default)
  - LIVE mode: Uses a real OpenAI-compatible API (set OPENAI_API_KEY env var)

Run:
    python 02-code/03-memory-state.py
    OPENAI_API_KEY=sk-... python 02-code/03-memory-state.py
"""

import json
import os
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# UTILS — Token counting, message formatting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Rough token count estimation.
    Rule of thumb: ~4 characters per token for English.
    In production, use tiktoken for exact counts.
    """
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single message (role + content)."""
    role_tokens = estimate_tokens(message.get("role", ""))
    content_tokens = estimate_tokens(message.get("content", ""))
    return role_tokens + content_tokens + 4  # +4 for message framing overhead


def format_messages_for_display(messages: list[dict], max_per_msg: int = 100) -> str:
    """Pretty-print messages for debugging. Truncate long content."""
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if len(content) > max_per_msg:
            content = content[:max_per_msg] + "..."
        lines.append(f"  [{role}] {content}")
    return "\n".join(lines)


# ============================================================================
# 1. CONVERSATION BUFFER MEMORY — Keep everything
# ============================================================================

class ConversationBufferMemory:
    """
    Simplest memory: store every message, no trimming.

    Good for: short conversations (< 20 messages), demos, debugging.
    Bad for: long conversations — tokens grow unbounded, cost explodes.
    """

    def __init__(self):
        self._messages: list[dict] = []  # The full message history

    def add_message(self, role: str, content: str) -> None:
        """Append a new message to history."""
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        """Return ALL messages — no filtering, no trimming."""
        return list(self._messages)  # Return a copy to prevent external mutation

    def get_token_count(self) -> int:
        """Total tokens across all stored messages."""
        return sum(estimate_message_tokens(m) for m in self._messages)

    def clear(self) -> None:
        """Wipe all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"BufferMemory({len(self)} messages, ~{self.get_token_count()} tokens)"


# ============================================================================
# 2. CONVERSATION SUMMARY MEMORY — Summarize old messages
# ============================================================================

def mock_llm_summarize(messages: list[dict]) -> str:
    """
    Mock LLM summarizer — in production, call a real LLM to summarize.

    This mock extracts key patterns from messages to create a realistic summary.
    It looks for common info types: names, preferences, facts, tool results.
    """
    # Extract key information from messages heuristically
    facts = []
    for msg in messages:
        content = msg.get("content", "")
        # Look for preference patterns like "budget is X" or "I prefer Y"
        if any(kw in content.lower() for kw in ["budget", "prefer", "want", "need", "looking for"]):
            # Keep the whole message as it likely contains a preference
            facts.append(content[:150])
        # Look for tool results (they tend to contain structured data)
        elif msg.get("role") == "tool" or "result:" in content.lower():
            facts.append(f"Tool result: {content[:100]}")
        # Look for decisions or conclusions
        elif any(kw in content.lower() for kw in ["decided", "conclusion", "found", "confirmed"]):
            facts.append(content[:150])

    if not facts:
        # Fallback: just note how many messages were summarized
        return f"[Summary of {len(messages)} messages — no key facts extracted]"

    summary = "Previous conversation summary:\n" + "\n".join(f"- {f}" for f in facts[:10])
    return summary


class ConversationSummaryMemory:
    """
    Summarizes old messages to save tokens.

    Strategy: when buffer exceeds max_messages, summarize the oldest half
    and keep the summary + recent messages.

    Good for: long conversations where old details don't matter.
    Trade-off: loses fine-grained details from old messages.
    """

    def __init__(self, max_messages: int = 20, summarizer=mock_llm_summarize):
        self._messages: list[dict] = []
        self._summary: str = ""          # Running summary of old messages
        self._max_messages: int = max_messages
        self._summarizer = summarizer     # Pluggable — swap with real LLM call
        self._summaries_count: int = 0    # How many times we've summarized

    def add_message(self, role: str, content: str) -> None:
        """Add message, trigger summarization if buffer is full."""
        self._messages.append({"role": role, "content": content})

        # Summarize when we exceed the limit
        if len(self._messages) > self._max_messages:
            self._condense()

    def _condense(self) -> None:
        """Summarize the oldest half of messages."""
        midpoint = len(self._messages) // 2
        old_messages = self._messages[:midpoint]
        recent_messages = self._messages[midpoint:]

        # Summarize old messages (including any previous summary as context)
        if self._summary:
            # Prepend existing summary as context for the new summarization
            old_messages = [{"role": "system", "content": self._summary}] + old_messages

        self._summary = self._summarizer(old_messages)
        self._messages = recent_messages
        self._summaries_count += 1

    def get_messages(self) -> list[dict]:
        """Return summary (if exists) + recent messages."""
        result = []
        if self._summary:
            # Inject summary as a system-level context message
            result.append({"role": "system", "content": self._summary})
        result.extend(self._messages)
        return result

    def get_token_count(self) -> int:
        """Token count of summary + recent messages."""
        total = estimate_tokens(self._summary) if self._summary else 0
        total += sum(estimate_message_tokens(m) for m in self._messages)
        return total

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""
        self._summaries_count = 0

    def __repr__(self) -> str:
        return (
            f"SummaryMemory({len(self._messages)} recent msgs, "
            f"~{self.get_token_count()} tokens, "
            f"{self._summaries_count} summarizations done)"
        )


# ============================================================================
# 3. SLIDING WINDOW MEMORY — Keep last N messages with token budget
# ============================================================================

class SlidingWindowMemory:
    """
    Keep the most recent messages that fit within a token budget.

    Two limits (whichever is hit first):
      - max_messages: hard limit on number of messages
      - max_tokens: soft limit on total token count

    Good for: predictable cost, when old context doesn't matter.
    Bad for: tasks where early context is crucial (user gave a spec at message 1).
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        self._messages: list[dict] = []
        self._max_messages: int = max_messages
        self._max_tokens: int = max_tokens
        self._dropped_count: int = 0  # Track how many messages were dropped

    def add_message(self, role: str, content: str) -> None:
        """Add message and trim to stay within limits."""
        self._messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        """Remove oldest messages until within both limits."""
        # Trim by message count first (fast)
        while len(self._messages) > self._max_messages:
            self._messages.pop(0)
            self._dropped_count += 1

        # Then trim by token count (more precise)
        while self._get_total_tokens() > self._max_tokens and len(self._messages) > 1:
            self._messages.pop(0)
            self._dropped_count += 1

    def _get_total_tokens(self) -> int:
        return sum(estimate_message_tokens(m) for m in self._messages)

    def get_messages(self) -> list[dict]:
        """Return the current window of messages."""
        return list(self._messages)

    def get_token_count(self) -> int:
        return self._get_total_tokens()

    @property
    def dropped_count(self) -> int:
        """How many messages have been dropped since creation."""
        return self._dropped_count

    def clear(self) -> None:
        self._messages.clear()
        self._dropped_count = 0

    def __repr__(self) -> str:
        return (
            f"SlidingWindowMemory({len(self._messages)} msgs, "
            f"~{self.get_token_count()} tokens, "
            f"{self._dropped_count} dropped)"
        )


# ============================================================================
# 4. HYBRID MEMORY — Summary of old + buffer of recent (production pattern)
# ============================================================================

class HybridMemory:
    """
    Production-grade pattern: summarize old messages, keep recent ones verbatim.

    How it works:
      1. Messages accumulate in the buffer
      2. When buffer exceeds threshold, oldest messages are summarized
      3. Summary is prepended to context; recent messages are kept verbatim
      4. The LLM sees: [summary of past] + [last N messages in full detail]

    This is the strategy used by most production agents (including LangGraph
    memory implementations).
    """

    def __init__(
        self,
        recent_window: int = 10,      # Keep last N messages verbatim
        summary_threshold: int = 20,   # Trigger summary when total > this
        max_tokens: int = 4000,        # Max total tokens (summary + recent)
        summarizer=mock_llm_summarize,
    ):
        self._all_messages: list[dict] = []  # Full history (for summarization)
        self._summary: str = ""               # Running summary
        self._recent_window: int = recent_window
        self._summary_threshold: int = summary_threshold
        self._max_tokens: int = max_tokens
        self._summarizer = summarizer

    def add_message(self, role: str, content: str) -> None:
        """Add message and re-balance if needed."""
        self._all_messages.append({"role": role, "content": content})

        # Trigger summarization when we accumulate too many messages
        if len(self._all_messages) > self._summary_threshold:
            self._rebalance()

    def _rebalance(self) -> None:
        """Summarize old messages, keep recent window."""
        # Split: everything except the recent window goes to summary
        cutoff = len(self._all_messages) - self._recent_window
        if cutoff <= 0:
            return  # Not enough messages to summarize

        old_messages = self._all_messages[:cutoff]

        # Include existing summary for continuity
        if self._summary:
            context = [{"role": "system", "content": self._summary}] + old_messages
        else:
            context = old_messages

        # Generate new summary
        self._summary = self._summarizer(context)

        # Keep only the recent window in the buffer
        self._all_messages = self._all_messages[cutoff:]

    def get_messages(self) -> list[dict]:
        """Return [summary] + recent messages, respecting token budget."""
        result = []

        # Add summary if it exists
        if self._summary:
            result.append({"role": "system", "content": self._summary})

        # Add recent messages, trimmed to token budget
        recent = self._all_messages[-self._recent_window:]
        summary_tokens = estimate_tokens(self._summary) if self._summary else 0
        remaining_budget = self._max_tokens - summary_tokens

        # Add recent messages from newest to oldest until budget runs out
        trimmed_recent = []
        used_tokens = 0
        for msg in reversed(recent):
            msg_tokens = estimate_message_tokens(msg)
            if used_tokens + msg_tokens > remaining_budget:
                break
            trimmed_recent.append(msg)
            used_tokens += msg_tokens

        result.extend(reversed(trimmed_recent))  # Restore chronological order
        return result

    def get_token_count(self) -> int:
        messages = self.get_messages()
        return sum(estimate_message_tokens(m) for m in messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def total_messages_seen(self) -> int:
        """Total messages ever added (including summarized ones)."""
        return len(self._all_messages) + (1 if self._summary else 0)

    def clear(self) -> None:
        self._all_messages.clear()
        self._summary = ""

    def __repr__(self) -> str:
        return (
            f"HybridMemory("
            f"summary={'yes' if self._summary else 'no'}, "
            f"{len(self._all_messages)} recent, "
            f"~{self.get_token_count()} tokens)"
        )


# ============================================================================
# 5. WORKING MEMORY — Scratchpad with typed key-value store
# ============================================================================

class WorkingMemory:
    """
    Structured scratchpad for intermediate results during a task.

    Unlike conversation memory (which stores messages), working memory stores
    structured data: variables, hypotheses, partial results, task state.

    Think of it as the agent's "notepad" — organized, compact, queryable.

    In production, this is what separates a good agent from a great one.
    An agent that maintains a clean scratchpad is 10x more effective.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}        # Main key-value store
        self._history: list[dict] = []           # Audit trail of all writes
        self._metadata: dict[str, dict] = {}     # Metadata per key (type, timestamp, source)

    def set(self, key: str, value: Any, source: str = "agent") -> None:
        """
        Store a value with metadata.

        Args:
            key: Name of the variable (e.g., "budget_max", "findings")
            value: Any Python value
            source: Who set this value ("agent", "user", "tool_result")
        """
        self._store[key] = value
        self._metadata[key] = {
            "type": type(value).__name__,
            "source": source,
            "updated_at": datetime.now().isoformat(),
        }
        # Audit trail — useful for debugging "who changed what when"
        self._history.append({
            "action": "set",
            "key": key,
            "value": repr(value)[:200],  # Truncate for large values
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key. Returns default if not found."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in working memory."""
        return key in self._store

    def delete(self, key: str) -> None:
        """Remove a key from working memory."""
        if key in self._store:
            self._history.append({
                "action": "delete",
                "key": key,
                "timestamp": datetime.now().isoformat(),
            })
            del self._store[key]
            del self._metadata[key]

    def append_to(self, key: str, value: Any) -> None:
        """Append to a list-type value. Creates the list if key doesn't exist."""
        if key not in self._store:
            self._store[key] = []
        if not isinstance(self._store[key], list):
            raise TypeError(f"Cannot append to non-list key '{key}' (type: {type(self._store[key]).__name__})")
        self._store[key].append(value)
        self._metadata[key] = {
            "type": "list",
            "source": "agent",
            "updated_at": datetime.now().isoformat(),
        }

    def get_snapshot(self) -> dict:
        """Return a copy of the entire working memory for checkpointing."""
        return {
            "store": dict(self._store),  # Shallow copy
            "metadata": dict(self._metadata),
        }

    def to_context_string(self) -> str:
        """
        Format working memory as a string to inject into the LLM context.
        This is how the agent "reads" its scratchpad.
        """
        if not self._store:
            return "[Working memory is empty]"

        lines = ["Current working memory:"]
        for key, value in self._store.items():
            meta = self._metadata.get(key, {})
            source = meta.get("source", "?")
            # Format value compactly
            val_str = repr(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            lines.append(f"  - {key} ({source}): {val_str}")
        return "\n".join(lines)

    @property
    def history(self) -> list[dict]:
        """Full audit trail of writes and deletes."""
        return list(self._history)

    def clear(self) -> None:
        self._store.clear()
        self._metadata.clear()
        self._history.clear()

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._store)} keys: {list(self._store.keys())})"


# ============================================================================
# 6. VECTOR MEMORY — Numpy embeddings + cosine similarity
# ============================================================================

def mock_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Mock embedding function — produces deterministic pseudo-embeddings.

    Uses SHA-256 hash of the text to generate a reproducible vector.
    Texts with similar words will NOT have similar embeddings (it's a hash,
    not a real encoder). For real similarity, use OpenAI/Voyage embeddings.

    But it's enough to demonstrate the vector memory architecture.
    """
    # Hash the text to get deterministic bytes
    hash_bytes = hashlib.sha256(text.encode()).digest()
    # Extend hash bytes if needed to fill the dimension
    while len(hash_bytes) < dim:
        hash_bytes += hashlib.sha256(hash_bytes).digest()
    # Convert each byte to a float in [-1, 1] range (avoids float32 overflow)
    raw = np.array([(b / 127.5) - 1.0 for b in hash_bytes[:dim]], dtype=np.float64)
    # Normalize to unit vector (required for cosine similarity to be meaningful)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    return raw


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns: 1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.

    This is THE distance metric for embedding search. Euclidean distance
    also works but cosine is more robust to magnitude differences.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass
class MemoryEntry:
    """A single entry in vector memory — text + embedding + metadata."""
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = ""

    def __post_init__(self):
        if not self.id:
            # Auto-generate a unique ID from content hash
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:12]


class VectorMemory:
    """
    Long-term memory backed by vector embeddings + cosine similarity search.

    In production, this would use Chroma, Pinecone, Qdrant, or pgvector.
    Here we implement it from scratch with numpy to show what's under the hood.

    Features:
      - Store text with embeddings and metadata
      - Similarity search (top-k nearest neighbors)
      - Metadata filtering (filter before similarity search)
      - Deduplication (don't store the same text twice)
    """

    def __init__(self, embed_fn=mock_embed, dim: int = 64):
        self._entries: list[MemoryEntry] = []
        self._embed_fn = embed_fn
        self._dim = dim
        self._seen_ids: set[str] = set()  # For deduplication

    def store(self, text: str, metadata: dict | None = None) -> str:
        """
        Store a text in vector memory.

        Returns: the entry ID (for retrieval or deletion).
        Deduplicates: same text won't be stored twice.
        """
        entry = MemoryEntry(
            text=text,
            embedding=self._embed_fn(text, self._dim),
            metadata=metadata or {},
        )

        # Deduplication check
        if entry.id in self._seen_ids:
            return entry.id  # Already stored, skip

        self._entries.append(entry)
        self._seen_ids.add(entry.id)
        return entry.id

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search: find the most similar stored memories.

        Args:
            query: text to search for
            top_k: number of results to return
            min_score: minimum cosine similarity threshold
            metadata_filter: dict of {key: value} that entries must match

        Returns: list of {"text", "score", "metadata", "id"} sorted by score desc
        """
        if not self._entries:
            return []

        query_embedding = self._embed_fn(query, self._dim)

        # Step 1: Apply metadata filter (if any) — reduces search space
        candidates = self._entries
        if metadata_filter:
            candidates = [
                e for e in candidates
                if all(e.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        # Step 2: Compute cosine similarity for each candidate
        scored = []
        for entry in candidates:
            score = cosine_similarity(query_embedding, entry.embedding)
            if score >= min_score:
                scored.append({
                    "text": entry.text,
                    "score": round(score, 4),
                    "metadata": entry.metadata,
                    "id": entry.id,
                })

        # Step 3: Sort by score (descending) and return top-k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                self._entries.pop(i)
                self._seen_ids.discard(entry_id)
                return True
        return False

    def list_all(self) -> list[dict]:
        """List all entries (without embeddings — too large to display)."""
        return [
            {"id": e.id, "text": e.text[:100], "metadata": e.metadata}
            for e in self._entries
        ]

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._seen_ids.clear()

    def __repr__(self) -> str:
        return f"VectorMemory({self.size} entries, dim={self._dim})"


# ============================================================================
# 7. CHECKPOINTING — Save/restore agent state to/from JSON
# ============================================================================

@dataclass
class AgentState:
    """
    Complete agent state — everything needed to resume execution.

    This is the "photo" of the agent at a point in time.
    Using a typed dataclass prevents the typo/bad-type bugs of plain dicts.
    """
    messages: list[dict] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    total_tokens_used: int = 0
    task: str = ""
    model: str = "simulated"
    done: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON export."""
        return {
            "messages": self.messages,
            "working_memory": self.working_memory,
            "iteration": self.iteration,
            "total_tokens_used": self.total_tokens_used,
            "task": self.task,
            "model": self.model,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """Deserialize from a plain dict."""
        return cls(
            messages=data.get("messages", []),
            working_memory=data.get("working_memory", {}),
            iteration=data.get("iteration", 0),
            total_tokens_used=data.get("total_tokens_used", 0),
            task=data.get("task", ""),
            model=data.get("model", "simulated"),
            done=data.get("done", False),
        )


class CheckpointManager:
    """
    Save and restore agent state to JSON files.

    Features:
      - Save checkpoint at any point during execution
      - List all checkpoints for a task
      - Load a specific checkpoint by step number
      - Time-travel: load any previous checkpoint and resume from there
      - Cleanup: delete old checkpoints

    File format: {checkpoint_dir}/{task_id}/step_{N}.json
    """

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState, task_id: str = "default") -> str:
        """
        Save a checkpoint. Returns the file path.

        The checkpoint includes the state + metadata (timestamp, version).
        """
        task_dir = self._dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "step": state.iteration,
            "state": state.to_dict(),
        }

        filepath = task_dir / f"step_{state.iteration:04d}.json"
        filepath.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))
        return str(filepath)

    def load(self, task_id: str, step: int) -> AgentState | None:
        """
        Load a checkpoint by task ID and step number.
        Returns None if checkpoint doesn't exist.
        """
        filepath = self._dir / task_id / f"step_{step:04d}.json"
        if not filepath.exists():
            return None

        data = json.loads(filepath.read_text())
        return AgentState.from_dict(data["state"])

    def list_checkpoints(self, task_id: str) -> list[dict]:
        """List all checkpoints for a task, sorted by step."""
        task_dir = self._dir / task_id
        if not task_dir.exists():
            return []

        checkpoints = []
        for f in sorted(task_dir.glob("step_*.json")):
            data = json.loads(f.read_text())
            checkpoints.append({
                "step": data["step"],
                "timestamp": data["timestamp"],
                "file": str(f),
                "tokens": data["state"].get("total_tokens_used", 0),
            })
        return checkpoints

    def load_latest(self, task_id: str) -> AgentState | None:
        """Load the most recent checkpoint for a task."""
        checkpoints = self.list_checkpoints(task_id)
        if not checkpoints:
            return None
        latest_step = checkpoints[-1]["step"]
        return self.load(task_id, latest_step)

    def cleanup(self, task_id: str) -> int:
        """Delete all checkpoints for a task. Returns number deleted."""
        task_dir = self._dir / task_id
        if not task_dir.exists():
            return 0
        count = 0
        for f in task_dir.glob("step_*.json"):
            f.unlink()
            count += 1
        # Remove directory if empty
        try:
            task_dir.rmdir()
        except OSError:
            pass  # Directory not empty — other files present
        return count


# ============================================================================
# 8. DEMO AGENT — Uses ALL memory types for a multi-step research task
# ============================================================================

# --- Simulated tool functions ---

def tool_search_products(query: str) -> str:
    """Mock product search."""
    products = {
        "laptops": [
            {"name": "ThinkPad X1", "price": 1299, "category": "laptop", "rating": 4.5},
            {"name": "MacBook Air M4", "price": 1199, "category": "laptop", "rating": 4.7},
            {"name": "ASUS VivoBook", "price": 449, "category": "laptop", "rating": 4.1},
            {"name": "Lenovo IdeaPad", "price": 399, "category": "laptop", "rating": 3.9},
            {"name": "HP Pavilion", "price": 549, "category": "laptop", "rating": 4.2},
        ],
        "monitors": [
            {"name": "Dell U2723QE", "price": 619, "category": "monitor", "rating": 4.6},
            {"name": "LG 27UK850", "price": 449, "category": "monitor", "rating": 4.3},
        ],
    }
    # Simple keyword matching
    for key, items in products.items():
        if key in query.lower():
            return json.dumps(items, indent=2)
    return json.dumps(products["laptops"], indent=2)  # Default to laptops


def tool_calculate(expression: str) -> str:
    """Safe math calculator."""
    import re as _re
    if not _re.match(r'^[\d\s\+\-\*\/\.\(\)\%]+$', expression):
        return f"Error: unsafe expression '{expression}'"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def tool_get_reviews(product_name: str) -> str:
    """Mock review fetcher."""
    reviews = {
        "ThinkPad X1": "Excellent build quality, great keyboard. Pricey but worth it.",
        "MacBook Air M4": "Best battery life in class. M4 chip is blazing fast.",
        "ASUS VivoBook": "Great value for the price. Decent performance for daily tasks.",
        "Lenovo IdeaPad": "Budget-friendly. Screen could be better.",
        "HP Pavilion": "Good all-rounder. Solid mid-range option.",
    }
    return reviews.get(product_name, f"No reviews found for '{product_name}'.")


# --- Simulated LLM responses for the demo ---

DEMO_RESPONSES = [
    # Step 0: Agent analyzes the task and sets up working memory
    {
        "thought": "User wants laptop recommendations under budget. Let me first check working memory for user preferences, then search for products.",
        "action": "search_products",
        "action_input": {"query": "laptops"},
    },
    # Step 1: Agent stores findings and filters by budget
    {
        "thought": "Found 5 laptops. The budget from working memory is 500. Let me filter and calculate which ones fit.",
        "action": "calculate",
        "action_input": {"expression": "449 + 399 + 549"},
        "memory_update": {"filtered_products": ["ASUS VivoBook (449)", "Lenovo IdeaPad (399)"]},
    },
    # Step 2: Agent gets reviews for top pick
    {
        "thought": "2 laptops under 500: ASUS VivoBook (449, 4.1 stars) and Lenovo IdeaPad (399, 3.9 stars). Let me get reviews for the ASUS since it has better ratings.",
        "action": "get_reviews",
        "action_input": {"product_name": "ASUS VivoBook"},
        "memory_update": {"top_pick": "ASUS VivoBook", "reason": "Best rated under budget"},
    },
    # Step 3: Agent stores finding in long-term memory and finishes
    {
        "thought": "The ASUS VivoBook is the best option: 449 EUR, 4.1 stars, good value. Let me store this recommendation and finish.",
        "action": "finish",
        "action_input": "Based on your budget of 500 EUR, I recommend the ASUS VivoBook (449 EUR, 4.1/5 stars). It offers great value for daily tasks. The Lenovo IdeaPad (399 EUR) is a cheaper alternative but has a lower rating. The ASUS has good reviews: 'Great value for the price.'",
        "long_term_store": "User prefers laptops under 500 EUR. Recommended ASUS VivoBook (449 EUR, 4.1 stars).",
    },
]

_demo_step = 0


def demo_simulated_llm(messages: list[dict]) -> dict:
    """
    Simulated LLM that returns pre-scripted actions.
    In a real agent, this would call the LLM API with the full message context.
    """
    global _demo_step
    if _demo_step >= len(DEMO_RESPONSES):
        return {"thought": "Done.", "action": "finish", "action_input": "Task complete."}
    response = DEMO_RESPONSES[_demo_step]
    _demo_step += 1
    return response


def run_demo_agent():
    """
    Full demo: agent uses all memory types to answer
    "Recommend a laptop under my budget."

    Shows:
      - Working memory: stores budget, task state, intermediate results
      - Hybrid conversation memory: manages message history efficiently
      - Vector memory: stores and retrieves long-term knowledge
      - Checkpointing: saves state at each step for time-travel debugging
    """
    global _demo_step
    _demo_step = 0  # Reset simulation

    print("\n" + "=" * 70)
    print("  DEMO: Multi-Memory Agent — Laptop Recommendation")
    print("  Using: HybridMemory + WorkingMemory + VectorMemory + Checkpointing")
    print("=" * 70)

    # --- Initialize all memory systems ---
    conversation = HybridMemory(recent_window=6, summary_threshold=15, max_tokens=3000)
    scratchpad = WorkingMemory()
    long_term = VectorMemory(dim=64)
    checkpointer = CheckpointManager(checkpoint_dir=".demo_checkpoints")
    task_id = "laptop_reco"

    # Clean up previous checkpoints
    checkpointer.cleanup(task_id)

    # --- Pre-load some long-term memories (simulating past sessions) ---
    long_term.store(
        "User profile: AI Engineer, uses Python daily, needs portable laptop.",
        metadata={"type": "user_profile", "source": "onboarding"},
    )
    long_term.store(
        "Previous recommendation: suggested a ThinkPad X1 for coding (Jan 2026).",
        metadata={"type": "recommendation", "source": "session_2026_01"},
    )
    long_term.store(
        "User feedback: ThinkPad was too expensive. Prefers budget options.",
        metadata={"type": "user_feedback", "source": "session_2026_02"},
    )

    print(f"\n  Long-term memory pre-loaded: {long_term.size} entries")

    # --- Set up initial working memory ---
    scratchpad.set("budget_max", 500, source="user")
    scratchpad.set("currency", "EUR", source="user")
    scratchpad.set("task", "Find laptop under budget", source="agent")
    scratchpad.set("remaining_steps", ["search", "filter", "review", "recommend"], source="agent")

    print(f"  Working memory initialized: {scratchpad}")

    # --- Retrieve relevant long-term memories ---
    print("\n  --- Retrieving relevant long-term memories ---")
    relevant = long_term.search("laptop budget recommendation", top_k=3)
    for mem in relevant:
        print(f"    Score {mem['score']:.3f}: {mem['text'][:80]}...")

    # Add retrieved memories to conversation context
    if relevant:
        context_str = "Relevant memories from past sessions:\n"
        context_str += "\n".join(f"- {m['text']}" for m in relevant)
        conversation.add_message("system", context_str)

    # --- Add user message ---
    user_msg = "Recommend me a laptop. My budget is 500 EUR maximum."
    conversation.add_message("user", user_msg)
    print(f"\n  User: {user_msg}")

    # --- Tool dispatch ---
    tools = {
        "search_products": lambda p: tool_search_products(p.get("query", "")),
        "calculate": lambda p: tool_calculate(p.get("expression", "")),
        "get_reviews": lambda p: tool_get_reviews(p.get("product_name", "")),
    }

    # --- Agent loop ---
    state = AgentState(task="laptop_recommendation", model="simulated")

    for i in range(10):  # Max iterations safety
        # Get LLM decision (simulated)
        context_messages = conversation.get_messages()
        # Add working memory to context
        context_messages.append({"role": "system", "content": scratchpad.to_context_string()})

        decision = demo_simulated_llm(context_messages)

        # Update state
        state.iteration = i
        state.messages = conversation.get_messages()
        state.working_memory = scratchpad.get_snapshot()["store"]
        state.total_tokens_used += sum(estimate_message_tokens(m) for m in context_messages)

        # Checkpoint BEFORE executing the action
        cp_path = checkpointer.save(state, task_id)

        # Display step
        print(f"\n  --- Step {i} ---")
        print(f"  Thought: {decision['thought']}")
        print(f"  Action:  {decision['action']}")

        # Update working memory if the step specifies it
        if "memory_update" in decision:
            for key, val in decision["memory_update"].items():
                scratchpad.set(key, val, source="agent")
                print(f"  [Scratchpad] Set '{key}' = {repr(val)[:80]}")

        # Check for finish
        if decision["action"] == "finish":
            answer = decision["action_input"]
            conversation.add_message("assistant", answer)

            # Store in long-term memory if specified
            if "long_term_store" in decision:
                doc = decision["long_term_store"]
                long_term.store(doc, metadata={"type": "recommendation", "source": "session_current"})
                print(f"  [Long-term] Stored: {doc[:80]}...")

            state.done = True
            state.iteration = i + 1
            checkpointer.save(state, task_id)

            print(f"\n  {'=' * 60}")
            print(f"  FINAL ANSWER: {answer}")
            print(f"  {'=' * 60}")
            break

        # Execute tool
        action_name = decision["action"]
        action_input = decision.get("action_input", {})
        if isinstance(action_input, str):
            action_input = {"query": action_input}

        if action_name in tools:
            result = tools[action_name](action_input)
            print(f"  Tool result: {result[:120]}{'...' if len(result) > 120 else ''}")
        else:
            result = f"Error: unknown tool '{action_name}'"
            print(f"  {result}")

        # Add to conversation memory
        conversation.add_message("assistant", f"Thought: {decision['thought']}\nAction: {action_name}")
        conversation.add_message("tool", f"Result: {result}")

    # --- Post-run: show memory state ---
    print("\n" + "=" * 70)
    print("  POST-RUN MEMORY STATE")
    print("=" * 70)

    print(f"\n  Conversation memory: {conversation}")
    print(f"  Working memory: {scratchpad}")
    print(f"  Long-term memory: {long_term}")
    print(f"  Total tokens used: ~{state.total_tokens_used}")

    print(f"\n  Working memory contents:")
    print(f"  {scratchpad.to_context_string()}")

    # --- Demonstrate checkpointing: list and time-travel ---
    print(f"\n  --- Checkpoints ---")
    checkpoints = checkpointer.list_checkpoints(task_id)
    for cp in checkpoints:
        print(f"    Step {cp['step']}: {cp['timestamp']} (~{cp['tokens']} tokens)")

    # Time-travel demo: load step 1
    if len(checkpoints) >= 2:
        old_state = checkpointer.load(task_id, step=1)
        if old_state:
            print(f"\n  --- Time-travel to Step 1 ---")
            print(f"  Task at step 1: {old_state.task}")
            print(f"  Working memory at step 1: {list(old_state.working_memory.keys())}")
            print(f"  Messages at step 1: {len(old_state.messages)}")
            print(f"  (Could resume execution from this point with different decisions)")

    # Cleanup demo checkpoints
    deleted = checkpointer.cleanup(task_id)
    print(f"\n  Cleaned up {deleted} checkpoint files.")


# ============================================================================
# INDIVIDUAL DEMOS — Run each memory type standalone
# ============================================================================

def demo_buffer_memory():
    """Show how buffer memory works and when it breaks."""
    print("\n" + "-" * 50)
    print("  Demo 1: ConversationBufferMemory")
    print("-" * 50)

    mem = ConversationBufferMemory()

    # Simulate a short conversation
    exchanges = [
        ("user", "Hi! I'm looking for a laptop under 500 EUR."),
        ("assistant", "I'd be happy to help you find a laptop! Let me search for options."),
        ("assistant", "I found 3 laptops under 500 EUR:\n1. ASUS VivoBook - 449 EUR\n2. Lenovo IdeaPad - 399 EUR\n3. Acer Aspire - 479 EUR"),
        ("user", "What about the ASUS? Is it good for programming?"),
        ("assistant", "The ASUS VivoBook has a Ryzen 5 processor, 8GB RAM, 256GB SSD. Decent for light programming."),
    ]

    for role, content in exchanges:
        mem.add_message(role, content)
        print(f"  Added [{role}]: {content[:60]}... | Total: ~{mem.get_token_count()} tokens")

    print(f"\n  Final state: {mem}")
    print(f"  All {len(mem)} messages are kept — no information loss")
    print(f"  Problem: with 100 messages, this would be ~25,000+ tokens per LLM call")


def demo_sliding_window():
    """Show how sliding window drops old messages."""
    print("\n" + "-" * 50)
    print("  Demo 2: SlidingWindowMemory (window=5, max_tokens=500)")
    print("-" * 50)

    mem = SlidingWindowMemory(max_messages=5, max_tokens=500)

    messages = [
        ("user", "My budget is 500 EUR. Remember this!"),       # msg 1 — critical info
        ("assistant", "Got it, budget is 500 EUR."),              # msg 2
        ("user", "Show me laptops."),                             # msg 3
        ("assistant", "Here are 5 laptops: ..."),                 # msg 4
        ("user", "Tell me about the ASUS."),                      # msg 5
        ("assistant", "The ASUS VivoBook costs 449 EUR..."),      # msg 6
        ("user", "And the Lenovo?"),                              # msg 7
        ("assistant", "The Lenovo IdeaPad costs 399 EUR..."),     # msg 8
        ("user", "What was my budget again?"),                    # msg 9 — oops!
    ]

    for i, (role, content) in enumerate(messages):
        mem.add_message(role, content)
        in_window = len(mem.get_messages())
        print(f"  Msg {i+1} added [{role}]: {content[:50]:<50} | Window: {in_window} msgs, dropped: {mem.dropped_count}")

    print(f"\n  Final state: {mem}")
    print(f"  Messages in window:")
    for msg in mem.get_messages():
        print(f"    [{msg['role']}] {msg['content'][:70]}")
    print(f"\n  PROBLEM: 'My budget is 500 EUR' was dropped! The agent forgot the budget.")


def demo_hybrid_memory():
    """Show hybrid memory in action."""
    print("\n" + "-" * 50)
    print("  Demo 3: HybridMemory (summary + recent buffer)")
    print("-" * 50)

    mem = HybridMemory(recent_window=4, summary_threshold=8, max_tokens=2000)

    # Simulate a longer conversation
    messages = [
        ("user", "My name is Alex, I'm an AI engineer."),
        ("assistant", "Nice to meet you, Alex!"),
        ("user", "My budget for a laptop is 500 EUR maximum."),
        ("assistant", "Noted, budget is 500 EUR."),
        ("user", "I need it for Python development and Claude Code."),
        ("assistant", "For dev work, you'll want good RAM and SSD."),
        ("user", "I prefer ASUS or Lenovo brands."),
        ("assistant", "Let me search for ASUS and Lenovo laptops under 500 EUR."),
        ("user", "Also, I travel a lot. Weight matters."),
        ("assistant", "I'll prioritize lightweight models too."),
        ("user", "What do you recommend?"),
    ]

    for role, content in messages:
        mem.add_message(role, content)

    print(f"  State: {mem}")
    print(f"\n  Messages sent to LLM:")
    for msg in mem.get_messages():
        content = msg['content']
        if len(content) > 100:
            content = content[:100] + "..."
        print(f"    [{msg['role']}] {content}")

    print(f"\n  Summary preserves key facts (budget, preferences)")
    print(f"  Recent messages give full context for current question")
    if mem.summary:
        print(f"\n  Generated summary:\n    {mem.summary[:200]}")


def demo_vector_memory():
    """Show vector memory storage and retrieval."""
    print("\n" + "-" * 50)
    print("  Demo 4: VectorMemory (embedding search)")
    print("-" * 50)

    vmem = VectorMemory(dim=64)

    # Store some memories
    docs = [
        ("User prefers ASUS brand laptops.", {"type": "preference"}),
        ("Budget constraint: maximum 500 EUR.", {"type": "constraint"}),
        ("Previously recommended ThinkPad X1 (too expensive).", {"type": "history"}),
        ("User is an AI engineer, needs good CPU and RAM.", {"type": "profile"}),
        ("User travels frequently, prefers lightweight devices.", {"type": "preference"}),
        ("API key for product search is stored in env vars.", {"type": "system"}),
    ]

    for text, meta in docs:
        entry_id = vmem.store(text, metadata=meta)
        print(f"  Stored: [{entry_id}] {text[:60]}")

    print(f"\n  Total entries: {vmem.size}")

    # Search without filter
    print(f"\n  Search: 'laptop recommendation for developer'")
    results = vmem.search("laptop recommendation for developer", top_k=3)
    for r in results:
        print(f"    Score {r['score']:.3f}: {r['text'][:60]}")

    # Search with metadata filter
    print(f"\n  Search with filter type='preference':")
    results = vmem.search("laptop", top_k=3, metadata_filter={"type": "preference"})
    for r in results:
        print(f"    Score {r['score']:.3f}: {r['text'][:60]}")

    # Deduplication demo
    print(f"\n  Deduplication test:")
    vmem.store("Budget constraint: maximum 500 EUR.", {"type": "constraint"})
    print(f"  Stored same text again -> size still {vmem.size} (deduplicated)")


def demo_working_memory():
    """Show working memory as a structured scratchpad."""
    print("\n" + "-" * 50)
    print("  Demo 5: WorkingMemory (scratchpad)")
    print("-" * 50)

    wm = WorkingMemory()

    # Agent builds up working memory during task
    wm.set("task", "Find laptop under budget", source="agent")
    wm.set("budget_max", 500, source="user")
    wm.set("currency", "EUR", source="user")

    print("  After initial setup:")
    print(f"  {wm.to_context_string()}")

    # Agent finds products and stores partial results
    wm.append_to("candidates", {"name": "ASUS VivoBook", "price": 449, "rating": 4.1})
    wm.append_to("candidates", {"name": "Lenovo IdeaPad", "price": 399, "rating": 3.9})
    wm.set("top_pick", "ASUS VivoBook", source="agent")
    wm.set("confidence", 0.85, source="agent")

    print("\n  After research:")
    print(f"  {wm.to_context_string()}")

    # Check audit trail
    print(f"\n  Audit trail ({len(wm.history)} operations):")
    for entry in wm.history:
        print(f"    {entry['action']}: {entry['key']}" + (f" = {entry.get('value', '?')[:50]}" if 'value' in entry else ""))

    # Snapshot for checkpointing
    snapshot = wm.get_snapshot()
    print(f"\n  Snapshot keys: {list(snapshot['store'].keys())}")


def demo_checkpointing():
    """Show checkpoint save, list, load, time-travel."""
    print("\n" + "-" * 50)
    print("  Demo 6: Checkpointing (save/load/time-travel)")
    print("-" * 50)

    mgr = CheckpointManager(checkpoint_dir=".demo_checkpoints_unit")
    task_id = "demo_task"
    mgr.cleanup(task_id)  # Start fresh

    # Simulate 5 steps of agent execution
    state = AgentState(task="analyze sales data", model="simulated")

    for step in range(5):
        state.iteration = step
        state.total_tokens_used += 500  # Simulated token usage
        state.messages.append({"role": "assistant", "content": f"Step {step} action..."})
        state.working_memory[f"step_{step}_result"] = f"Finding {step}"

        path = mgr.save(state, task_id)
        print(f"  Saved checkpoint: step {step} -> {Path(path).name}")

    # List all checkpoints
    print(f"\n  All checkpoints:")
    for cp in mgr.list_checkpoints(task_id):
        print(f"    Step {cp['step']}: {cp['timestamp'][:19]}, ~{cp['tokens']} tokens")

    # Time-travel: load step 2
    print(f"\n  Time-travel to step 2:")
    old_state = mgr.load(task_id, step=2)
    if old_state:
        print(f"    Iteration: {old_state.iteration}")
        print(f"    Messages: {len(old_state.messages)}")
        print(f"    Working memory keys: {list(old_state.working_memory.keys())}")
        print(f"    Tokens used at step 2: {old_state.total_tokens_used}")
        print(f"    -> Could resume from here with different decisions!")

    # Load latest
    latest = mgr.load_latest(task_id)
    if latest:
        print(f"\n  Latest checkpoint: step {latest.iteration}, {len(latest.messages)} messages")

    # Cleanup
    deleted = mgr.cleanup(task_id)
    print(f"\n  Cleaned up {deleted} checkpoint files.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Day 3 — Memory & State: Complete Memory System from Scratch")
    print("  Dependencies: stdlib + numpy | No frameworks needed")
    print("=" * 70)

    # Run individual demos first (educational — one concept at a time)
    demo_buffer_memory()
    demo_sliding_window()
    demo_hybrid_memory()
    demo_vector_memory()
    demo_working_memory()
    demo_checkpointing()

    # Run the full agent demo (all memory types working together)
    run_demo_agent()

    print("\n" + "=" * 70)
    print("  All demos complete! Key takeaways:")
    print("  - BufferMemory: simple but doesn't scale")
    print("  - SlidingWindow: predictable cost, loses old context")
    print("  - HybridMemory: best compromise for production")
    print("  - VectorMemory: long-term recall via similarity search")
    print("  - WorkingMemory: structured scratchpad for intermediate state")
    print("  - Checkpointing: save/restore for resilience & debugging")
    print("=" * 70)

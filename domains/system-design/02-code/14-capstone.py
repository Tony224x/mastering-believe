"""
Jour 14 -- Capstone : calculators d'estimation et structures de donnees
references pour 2 designs complets.

Usage:
    python 14-capstone.py

Contenu :
  1. Capacity planning helpers (users, QPS, storage, bandwidth, cost)
  2. Reference data structures pour Dropbox-like (chunk metadata store)
  3. Reference data structures pour LLM Support Assistant (conversation memory)
  4. Demo : on lance une estimation Dropbox + une estimation LLM support
     puis on manipule les structures de donnees pour montrer comment on
     "repondrait a la question" en live.
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from collections import defaultdict

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Capacity planning calculators
# =============================================================================


def human(n: float) -> str:
    """Turn a big number into K / M / G / T units."""
    for u, thr in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= thr:
            return f"{n/thr:.2f}{u}"
    return f"{n:.2f}"


def bytes_human(n: float) -> str:
    for u, thr in (("PB", 1024 ** 5), ("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if abs(n) >= thr:
            return f"{n/thr:.2f} {u}"
    return f"{n:.0f} B"


@dataclass
class CapacityEstimate:
    label: str
    daily_users: int
    ops_per_user_per_day: float
    bytes_per_op: int
    peak_factor: float = 3.0

    @property
    def ops_per_day(self) -> float:
        return self.daily_users * self.ops_per_user_per_day

    @property
    def avg_qps(self) -> float:
        return self.ops_per_day / 86_400

    @property
    def peak_qps(self) -> float:
        return self.avg_qps * self.peak_factor

    @property
    def bytes_per_day(self) -> float:
        return self.ops_per_day * self.bytes_per_op

    @property
    def bytes_per_year(self) -> float:
        return self.bytes_per_day * 365

    @property
    def bandwidth_bps(self) -> float:
        # bytes per sec -> bits per sec
        return self.avg_qps * self.bytes_per_op * 8

    def report(self) -> str:
        lines = [
            f"=== {self.label} ===",
            f"  daily users       : {human(self.daily_users)}",
            f"  ops / user / day  : {self.ops_per_user_per_day}",
            f"  ops / day         : {human(self.ops_per_day)}",
            f"  avg QPS           : {self.avg_qps:.1f} ({human(self.avg_qps)} ops/s)",
            f"  peak QPS          : {self.peak_qps:.1f}",
            f"  bytes / day       : {bytes_human(self.bytes_per_day)}",
            f"  bytes / year      : {bytes_human(self.bytes_per_year)}",
            f"  bandwidth (avg)   : {human(self.bandwidth_bps / 1e9)} Gbps",
        ]
        return "\n".join(lines)


def llm_cost_estimate(
    daily_convs: int,
    turns_per_conv: float,
    tokens_in: int,
    tokens_out: int,
    price_in_per_million: float,
    price_out_per_million: float,
) -> dict:
    """Estimate $/day and $/month for an LLM product."""
    calls = daily_convs * turns_per_conv
    in_tok = calls * tokens_in
    out_tok = calls * tokens_out
    cost_day = in_tok * price_in_per_million / 1e6 + out_tok * price_out_per_million / 1e6
    return {
        "calls_per_day": calls,
        "in_tokens_per_day": in_tok,
        "out_tokens_per_day": out_tok,
        "cost_per_day_usd": cost_day,
        "cost_per_month_usd": cost_day * 30,
        "cost_per_conversation_usd": cost_day / max(daily_convs, 1),
    }


# =============================================================================
# SECTION 2 : Dropbox reference data structures
# =============================================================================


@dataclass
class FileBlock:
    """A 4 MB chunk of a file, uniquely identified by content hash (SHA-256).

    Deduplication: if two files share a block (same hash), the block is
    stored only once in the block store, and ref_count is incremented.
    """

    hash: str                  # content hash
    size: int                  # bytes
    storage_location: str      # e.g. s3://bucket/key
    ref_count: int = 1


@dataclass
class FileVersion:
    version_id: str
    file_id: str
    size: int
    created_at: str
    block_hashes: list[str] = field(default_factory=list)  # ordered


@dataclass
class DropboxMetadataStore:
    """In-memory simulation of the metadata store.

    Real implementation: PostgreSQL (partitioned by user_id) + Redis cache
    on hot folders + S3-compatible block store.
    """

    blocks: dict[str, FileBlock] = field(default_factory=dict)
    files: dict[str, dict] = field(default_factory=dict)
    versions: dict[str, FileVersion] = field(default_factory=dict)

    def put_block(self, hash_: str, size: int, location: str) -> bool:
        """Return True if the block was newly created, False if it was a dedup hit."""
        if hash_ in self.blocks:
            self.blocks[hash_].ref_count += 1
            return False
        self.blocks[hash_] = FileBlock(hash=hash_, size=size, storage_location=location)
        return True

    def create_version(self, file_id: str, version_id: str, block_hashes: list[str]) -> FileVersion:
        size = sum(self.blocks[h].size for h in block_hashes if h in self.blocks)
        v = FileVersion(
            version_id=version_id,
            file_id=file_id,
            size=size,
            created_at="2026-04-11T10:00:00Z",
            block_hashes=block_hashes,
        )
        self.versions[version_id] = v
        return v

    def storage_saved_from_dedup(self) -> int:
        """Bytes saved thanks to dedup : sum((ref_count - 1) * block.size)."""
        saved = 0
        for b in self.blocks.values():
            saved += (b.ref_count - 1) * b.size
        return saved


# =============================================================================
# SECTION 3 : LLM Support Assistant reference data structures
# =============================================================================


@dataclass
class Message:
    role: str   # user | assistant | tool | system
    content: str
    ts: float = 0.0
    tokens: int = 0


@dataclass
class ConversationMemory:
    """Short-term memory for a single user conversation.

    Strategy :
      - keep the last N messages verbatim
      - when exceeded, summarize the oldest ones into a compact note
    """

    user_id: str
    session_id: str
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    max_messages: int = 10

    def add(self, role: str, content: str, tokens: int = 0) -> None:
        self.messages.append(Message(role=role, content=content, tokens=tokens))
        if len(self.messages) > self.max_messages:
            self._summarize_oldest()

    def _summarize_oldest(self) -> None:
        # Real system: call a small LLM. Here we just concat and truncate.
        old = self.messages[: -self.max_messages]
        keep = self.messages[-self.max_messages :]
        new_part = " | ".join(f"{m.role}:{m.content[:40]}" for m in old)
        self.summary = (self.summary + " || " + new_part)[:800]
        self.messages = keep

    def build_prompt_context(self) -> list[dict]:
        ctx: list[dict] = []
        if self.summary:
            ctx.append({"role": "system", "content": f"Conversation summary: {self.summary}"})
        for m in self.messages:
            ctx.append({"role": m.role, "content": m.content})
        return ctx


@dataclass
class LongTermUserMemory:
    """Persistent store of user preferences and facts.

    Real system : key-value store + optional vector store for fuzzy recall.
    """

    user_id: str
    facts: dict[str, Any] = field(default_factory=dict)
    episodic: list[dict] = field(default_factory=list)

    def remember_fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def remember_event(self, event: dict) -> None:
        self.episodic.append(event)


# =============================================================================
# SECTION 4 : Demos
# =============================================================================


def demo_capacity() -> None:
    print(SEPARATOR)
    print("CAPACITY ESTIMATES")
    print(SEPARATOR)

    # Dropbox
    dbx_upload = CapacityEstimate(
        label="Dropbox -- uploads",
        daily_users=50_000_000,
        ops_per_user_per_day=2,
        bytes_per_op=2 * 1024 * 1024,  # 2 MB avg
    )
    dbx_download = CapacityEstimate(
        label="Dropbox -- downloads",
        daily_users=50_000_000,
        ops_per_user_per_day=10,
        bytes_per_op=2 * 1024 * 1024,
    )
    print(dbx_upload.report())
    print()
    print(dbx_download.report())

    # LLM support
    print("\n" + SEPARATOR)
    print("LLM Support Assistant -- cost estimate")
    est = llm_cost_estimate(
        daily_convs=500_000,
        turns_per_conv=5,
        tokens_in=3000,
        tokens_out=500,
        price_in_per_million=0.15,   # gpt-5.4-mini-ish
        price_out_per_million=0.60,
    )
    for k, v in est.items():
        if isinstance(v, float):
            # show more precision for per-conversation cost
            fmt = f"{v:,.5f}" if "conversation" in k else f"{v:,.2f}"
            print(f"  {k:<30} {fmt}")
        else:
            print(f"  {k:<30} {v:,}")


def demo_dropbox_dedup() -> None:
    print("\n" + SEPARATOR)
    print("DROPBOX METADATA + DEDUP DEMO")
    print(SEPARATOR)
    store = DropboxMetadataStore()
    # Two users upload the same cat.jpg
    block_a = "hash_A"; block_b = "hash_B"; block_c = "hash_C"
    store.put_block(block_a, 4 * 1024 * 1024, "s3://bucket/A")
    store.put_block(block_b, 4 * 1024 * 1024, "s3://bucket/B")
    store.put_block(block_c, 1 * 1024 * 1024, "s3://bucket/C")
    # User 1: cat.jpg = A + B + C
    store.create_version(file_id="file1", version_id="v1", block_hashes=[block_a, block_b, block_c])
    # User 2: same cat.jpg, same blocks
    store.put_block(block_a, 4 * 1024 * 1024, "s3://bucket/A")
    store.put_block(block_b, 4 * 1024 * 1024, "s3://bucket/B")
    store.put_block(block_c, 1 * 1024 * 1024, "s3://bucket/C")
    store.create_version(file_id="file2", version_id="v2", block_hashes=[block_a, block_b, block_c])
    # User 3: cat_edited.jpg shares A and B, new block D
    block_d = "hash_D"
    store.put_block(block_a, 4 * 1024 * 1024, "s3://bucket/A")
    store.put_block(block_b, 4 * 1024 * 1024, "s3://bucket/B")
    store.put_block(block_d, 2 * 1024 * 1024, "s3://bucket/D")
    store.create_version(file_id="file3", version_id="v3", block_hashes=[block_a, block_b, block_d])

    print(f"  Blocks stored  : {len(store.blocks)}")
    print(f"  Versions       : {len(store.versions)}")
    for h, b in store.blocks.items():
        print(f"    {h} size={bytes_human(b.size)} ref_count={b.ref_count}")
    print(f"  Storage saved  : {bytes_human(store.storage_saved_from_dedup())}")


def demo_llm_memory() -> None:
    print("\n" + SEPARATOR)
    print("LLM CONVERSATION MEMORY DEMO")
    print(SEPARATOR)
    mem = ConversationMemory(user_id="u42", session_id="s1", max_messages=6)
    mem.add("user", "Hi, my order 123 is late.", tokens=10)
    mem.add("assistant", "Let me check order 123.", tokens=8)
    mem.add("tool", "order_123 -> shipped 2 days ago", tokens=10)
    mem.add("assistant", "It shipped 2 days ago, should arrive tomorrow.", tokens=12)
    mem.add("user", "Can I have a refund if it doesn't arrive?", tokens=13)
    mem.add("assistant", "If no delivery in 5 days I can offer a refund.", tokens=15)
    mem.add("user", "ok thanks. By the way my address changed.", tokens=12)
    mem.add("assistant", "Please share the new address.", tokens=8)
    mem.add("user", "New address: 10 rue de Paris", tokens=10)
    mem.add("assistant", "Noted, I updated your account.", tokens=9)
    mem.add("user", "Also can you cancel my subscription?", tokens=11)  # triggers summarize

    ctx = mem.build_prompt_context()
    print(f"  messages kept: {len(mem.messages)}")
    print(f"  summary: {mem.summary[:200]}...")
    print(f"  prompt ctx size: {len(ctx)}")
    for c in ctx[:3]:
        print(f"    - {c['role']}: {c['content'][:60]}")


def demo() -> None:
    demo_capacity()
    demo_dropbox_dedup()
    demo_llm_memory()
    print("\n" + SEPARATOR)
    print("Cheat sheet : every system design interview -> estimate first, structures second.")


if __name__ == "__main__":
    demo()

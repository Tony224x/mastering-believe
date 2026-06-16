"""
Solutions -- Day 3 (HARD): Memory & State

Contains solutions for:
  - Hard Ex 1: MemGPT-lite -- an agent that manages its own memory through
               meta-tools, under a strict 2000-token context window
               (inspired by MemGPT, Packer et al., 2023)
  - Hard Ex 2: Distributed multi-agent memory -- a SharedMemoryBus with
               namespace isolation, locks, event log and broadcast

LLM decisions are mocked (scripted, deterministic) -- "mode simule" as the
exercise statement allows -- but every memory component is REAL: the vector
store really embeds and searches, the context window really counts tokens
and overflows, the bus really locks and logs.

Run:  python 03-exercises/solutions/03-memory-state-hard.py
Each solution is self-contained.
"""

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# numpy is optional: vector math (embed/search) needs only array/dot/norm, done
# in pure Python below. Try numpy first, fall back to a tiny shim so the file
# RUNS OFFLINE with zero dependencies (see 02-code/03-memory-state.py).
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numpy
    _HAS_NUMPY = False

    class _NumpyShim:
        ndarray = list
        float64 = float

        @staticmethod
        def array(seq, dtype=float):
            return [dtype(x) for x in seq]

        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        class linalg:
            @staticmethod
            def norm(v):
                return sum(x * x for x in v) ** 0.5

    np = _NumpyShim()  # type: ignore

# ==========================================================================
# SHARED UTILS (same conventions as 03-memory-state.py)
# ==========================================================================

def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def mock_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic mock embedding via SHA-256 hash."""
    h = hashlib.sha256(text.encode()).digest()
    while len(h) < dim:
        h += hashlib.sha256(h).digest()
    raw = np.array([(b / 127.5) - 1.0 for b in h[:dim]], dtype=np.float64)
    norm = np.linalg.norm(raw)
    if norm <= 0:
        return raw
    # Scalar division on numpy arrays; elementwise on the pure-Python fallback.
    return raw / norm if _HAS_NUMPY else [x / norm for x in raw]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


# ==========================================================================
# HARD EXERCISE 1 -- MemGPT-lite: autonomous memory management
# ==========================================================================
#
# The core MemGPT idea: the context window is a SCARCE resource (here: hard
# 2000-token cap), so the agent gets memory operations as TOOLS and decides
# itself when to save, search, summarize or forget. The decisions below are
# scripted (mock LLM), but the memory machinery is fully functional.

# --- MEMGPT START ----------------------------------------------------------

class ContextOverflowError(RuntimeError):
    """Raised when adding a message would exceed the hard token cap."""


class VectorStore:
    """Minimal long-term memory: embed on store, cosine search on query."""

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}  # id -> {text, category, embedding}

    def store(self, text: str, category: str) -> str:
        mem_id = hashlib.md5(text.encode()).hexdigest()[:10]
        self._entries[mem_id] = {"text": text, "category": category,
                                 "embedding": mock_embed(text)}
        return mem_id

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = mock_embed(query)
        scored = [{"id": i, "text": e["text"], "category": e["category"],
                   "score": round(cosine_similarity(q, e["embedding"]), 4)}
                  for i, e in self._entries.items()]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def forget(self, mem_id: str) -> bool:
        return self._entries.pop(mem_id, None) is not None

    @property
    def size(self) -> int:
        return len(self._entries)


class Scratchpad:
    """Working memory: small key-value state the agent keeps in context."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def update(self, key: str, value: str) -> None:
        self._data[key] = value

    def render(self) -> str:
        return json.dumps(self._data) if self._data else "(empty)"


class ContextWindow:
    """
    Hard-capped context. add() ROLLS BACK and raises on overflow, so the
    cap can never be silently violated -- the agent MUST manage its memory.
    """

    def __init__(self, max_tokens: int = 2000,
                 system_prompt: str = "You are a product analyst with memory meta-tools."):
        self.max_tokens = max_tokens
        self._system = system_prompt
        self._summary = ""           # Compressed history lives here
        self._messages: list[dict] = []

    def token_count(self) -> int:
        text = self._system + self._summary + "".join(
            m["role"] + m["content"] for m in self._messages)
        return estimate_tokens(text)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if self.token_count() > self.max_tokens:
            self._messages.pop()  # Rollback -- the cap is inviolable
            raise ContextOverflowError(
                f"Adding {estimate_tokens(content)} tokens would exceed the "
                f"{self.max_tokens}-token cap (now at {self.token_count()})")

    def compress(self, summarizer: Callable[[str, list[dict]], str],
                 keep_last: int = 2) -> str:
        """Replace old messages with a summary, keep the recent tail."""
        old, tail = self._messages[:-keep_last], self._messages[-keep_last:]
        self._summary = summarizer(self._summary, old)
        self._messages = tail
        return self._summary


PRODUCT_NAMES = ["Falcon Air 13", "Nova Edge 15", "Orbit Book 14",
                 "Pulse Go 15", "Zephyr Slim 14"]


def mock_summarizer(previous_summary: str, old_messages: list[dict]) -> str:
    """
    Mock of the summarization LLM call: keeps ONLY the product names seen
    so far. The details are safe because the agent saved them to long-term
    memory BEFORE summarizing -- that is the whole MemGPT contract.
    """
    text = previous_summary + " ".join(m["content"] for m in old_messages)
    seen = [n for n in PRODUCT_NAMES if n in text]
    return (f"[Summary] Products analyzed so far: {', '.join(seen) or 'none'}. "
            f"Detailed facts are stored in long-term memory.")


class MemoryAgent:
    """Agent wiring: 6 memory meta-tools + 2 classic tools, all callable."""

    def __init__(self, context: ContextWindow):
        self.context = context
        self.long_term = VectorStore()
        self.scratchpad = Scratchpad()
        self.stats = {"save": 0, "search_lt": 0, "scratch": 0,
                      "summarize": 0, "forget": 0}

    # --- 6 memory meta-tools (the agent calls these like normal tools) ----

    def save_to_long_term(self, text: str, category: str) -> str:
        self.stats["save"] += 1
        return f"saved id={self.long_term.store(text, category)}"

    def search_long_term(self, query: str, top_k: int = 3) -> str:
        self.stats["search_lt"] += 1
        hits = self.long_term.search(query, top_k)
        return json.dumps([{"id": h["id"], "text": h["text"]} for h in hits])

    def update_scratchpad(self, key: str, value: str) -> str:
        self.stats["scratch"] += 1
        self.scratchpad.update(key, value)
        return f"scratchpad[{key}] = {value}"

    def read_scratchpad(self) -> str:
        return self.scratchpad.render()

    def summarize_context(self) -> str:
        self.stats["summarize"] += 1
        before = self.context.token_count()
        summary = self.context.compress(mock_summarizer)
        return f"context compressed {before} -> {self.context.token_count()} tokens. {summary}"

    def forget(self, memory_id: str) -> str:
        self.stats["forget"] += 1
        ok = self.long_term.forget(memory_id)
        return f"forgot {memory_id}" if ok else f"no memory {memory_id}"

    # --- 2 classic tools ----------------------------------------------------

    def search(self, query: str) -> str:
        return mock_product_search(query)

    def calculate(self, expression: str) -> str:
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            raise ValueError(f"Unsafe expression: {expression!r}")
        return str(eval(expression))

    def call(self, tool: str, params: dict) -> str:
        return getattr(self, tool)(**params)

# --- MEMGPT END ------------------------------------------------------------


# Long mock search results: each product sheet is deliberately verbose
# (~450 tokens) so that 5 of them CANNOT fit in a 2000-token context --
# the agent is forced to externalize facts and summarize.
_FILLER = "synthetic benchmark row with thermals and battery data; " * 32

PRODUCT_SHEETS = {
    "Falcon Air 13": "Falcon Air 13: 549 EUR, 14h battery, 1.1 kg, 8GB RAM, "
                     "great portability but weak GPU. " + _FILLER,
    "Nova Edge 15": "Nova Edge 15: 599 EUR, Ryzen 5 7530U, 16GB RAM, 512GB SSD, "
                    "best raw specs for developers. " + _FILLER,
    "Orbit Book 14": "Orbit Book 14: 579 EUR, 8GB soldered RAM, dim 250-nit "
                     "screen, mediocre value. " + _FILLER,
    "Pulse Go 15": "Pulse Go 15: 449 EUR, Celeron N4500, 4GB RAM, too weak "
                   "for development work. " + _FILLER,
    "Zephyr Slim 14": "Zephyr Slim 14: 619 EUR, nice chassis but EXCEEDS the "
                      "600 EUR budget. " + _FILLER,
}


def mock_product_search(query: str) -> str:
    for name, sheet in PRODUCT_SHEETS.items():
        if name.lower() in query.lower():
            return sheet
    return "Catalog: " + ", ".join(PRODUCT_NAMES)


# Scripted agent decisions -- 19 steps, far beyond what fits in 2000 tokens.
# "$temp_id" is resolved at runtime (the id is only known after saving).
MEMGPT_SCRIPT: list[dict] = [
    {"thought": "Pin the task constraints in working memory first.",
     "tool": "update_scratchpad", "params": {"key": "task", "value": "recommend dev laptop <= 600 EUR"}},
    {"thought": "List the candidate products.",
     "tool": "search", "params": {"query": "laptops under 650 EUR"}},
    {"thought": "Track progress in the scratchpad.",
     "tool": "update_scratchpad", "params": {"key": "pending", "value": "5 products to analyze"}},
    {"thought": "Analyze product 1.",
     "tool": "search", "params": {"query": "Falcon Air 13 review"}},
    {"thought": "The sheet is huge -- keep only the distilled fact, long-term.",
     "tool": "save_to_long_term",
     "params": {"text": "Falcon Air 13: 549 EUR, 14h battery, 1.1kg, 8GB RAM - portable but weak GPU",
                "category": "product_fact"}},
    {"thought": "Analyze product 2.",
     "tool": "search", "params": {"query": "Nova Edge 15 review"}},
    {"thought": "Distill and store product 2.",
     "tool": "save_to_long_term",
     "params": {"text": "Nova Edge 15: 599 EUR, Ryzen 5, 16GB RAM, 512GB SSD - best specs for a developer",
                "category": "product_fact"}},
    {"thought": "Temporary note while waiting for a price confirmation.",
     "tool": "save_to_long_term", "capture_id": "temp_id",
     "params": {"text": "TEMP: double-check Nova Edge 15 price before final answer",
                "category": "temp"}},
    {"thought": "Context is filling up -- the facts are safe in long-term, compress.",
     "tool": "summarize_context", "params": {}},
    {"thought": "Analyze product 3.",
     "tool": "search", "params": {"query": "Orbit Book 14 review"}},
    {"thought": "Distill and store product 3.",
     "tool": "save_to_long_term",
     "params": {"text": "Orbit Book 14: 579 EUR, 8GB soldered RAM, dim screen - mediocre value",
                "category": "product_fact"}},
    {"thought": "Analyze product 4.",
     "tool": "search", "params": {"query": "Pulse Go 15 review"}},
    {"thought": "Distill and store product 4.",
     "tool": "save_to_long_term",
     "params": {"text": "Pulse Go 15: 449 EUR, Celeron, 4GB RAM - too weak for development",
                "category": "product_fact"}},
    {"thought": "Near the cap again -- compress before product 5.",
     "tool": "summarize_context", "params": {}},
    {"thought": "Analyze product 5.",
     "tool": "search", "params": {"query": "Zephyr Slim 14 review"}},
    {"thought": "Distill and store product 5.",
     "tool": "save_to_long_term",
     "params": {"text": "Zephyr Slim 14: 619 EUR - over the 600 EUR budget, excluded",
                "category": "product_fact"}},
    {"thought": "Price confirmed via the sheet; the temp note is obsolete.",
     "tool": "forget", "params": {"memory_id": "$temp_id"}},
    {"thought": "Recall every stored fact to build the comparison.",
     "tool": "search_long_term",
     "params": {"query": "laptop EUR RAM developer budget", "top_k": 5}},
    {"thought": "How much headroom does the winner leave?",
     "tool": "calculate", "params": {"expression": "600 - 599"}},
]


def hard_ex1_memgpt_lite():
    """
    Solution: MemGPT-lite. The context never exceeds 2000 tokens even though
    the task ingests far more -- because the agent saves, summarizes, forgets.
    """
    print("\n" + "=" * 60)
    print("  HARD 1: MemGPT-lite -- Autonomous Memory Management")
    print("=" * 60)

    ctx = ContextWindow(max_tokens=2000)
    agent = MemoryAgent(ctx)
    captured: dict[str, str] = {}      # Runtime ids (e.g. the temp memory id)
    max_tokens_seen, total_ingested = 0, 0
    retrieved_facts = ""

    print(f"\n  Task: analyze 5 products, recommend the best for a developer "
          f"(budget 600 EUR)\n  Hard cap: {ctx.max_tokens} tokens\n")

    for i, step in enumerate(MEMGPT_SCRIPT, start=1):
        params = {k: captured[v[1:]] if isinstance(v, str) and v.startswith("$") else v
                  for k, v in step["params"].items()}
        observation = agent.call(step["tool"], params)
        if step.get("capture_id"):
            captured[step["capture_id"]] = observation.split("id=")[1]
        if step["tool"] == "search_long_term":
            retrieved_facts = observation

        # The agent's context receives the thought + observation, like a
        # real message history would. add() raises if the cap is broken.
        ctx.add("assistant", step["thought"])
        ctx.add("tool", observation)

        tokens = ctx.token_count()
        max_tokens_seen = max(max_tokens_seen, tokens)
        total_ingested += estimate_tokens(step["thought"] + observation)
        assert tokens <= 2000, f"Context cap violated at step {i}: {tokens}"
        marker = " <<< COMPRESS" if step["tool"] == "summarize_context" else ""
        print(f"  Step {i:2d} | {step['tool']:18s} | context: {tokens:4d}/2000 tok{marker}")

    # Final answer: mock generation grounded in the RETRIEVED facts only --
    # proof that information from all steps survived via long-term memory.
    facts = [h["text"] for h in json.loads(retrieved_facts)]
    final_answer = ("Recommendation: Nova Edge 15 at 599 EUR (1 EUR under budget) -- "
                    "best specs for a developer (16GB RAM, 512GB SSD). "
                    "Full comparison: " + " | ".join(facts))
    print(f"\n  Final answer:\n    {final_answer[:200]}...")

    # --- Verify the success criteria --------------------------------------
    assert max_tokens_seen <= 2000, "Cap exceeded at some step"
    assert total_ingested > 2000, "Task must be impossible without memory mgmt"
    for name in PRODUCT_NAMES:
        assert name in final_answer, f"Final answer is missing {name}"
    assert agent.stats["save"] == 6 and agent.stats["search_lt"] == 1
    assert agent.stats["summarize"] == 2 and agent.stats["forget"] == 1
    assert agent.stats["scratch"] >= 2
    assert agent.long_term.size == 5, "Temp note must be forgotten (5 facts left)"

    # Code-size criterion: the architecture (between sentinels) is < 400 lines
    source = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(source) if "MEMGPT START" in l)
    end = next(i for i, l in enumerate(source) if "MEMGPT END" in l)
    arch_lines = end - start - 1
    print(f"\n  Stats: {agent.stats}")
    print(f"  Peak context: {max_tokens_seen}/2000 tok | total ingested: "
          f"~{total_ingested} tok | architecture: {arch_lines} lines (< 400)")
    assert arch_lines < 400

    print("\n  PASS -- 19 steps, cap never broken, answer uses all 5 products.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Distributed multi-agent shared memory
# ==========================================================================
#
# Three specialists share ONE long-term memory bus. Each agent: reads
# everything, writes only to its own namespace (enforced), keeps a private
# working memory. Important writes are broadcast to the other agents.

@dataclass
class MemoryRecord:
    id: str
    namespace: str
    author: str
    text: str
    category: str
    timestamp: float
    embedding: np.ndarray = field(repr=False, default=None)


class SharedMemoryBus:
    """Shared long-term memory with locks, namespaces, event log, broadcast."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._lock = threading.Lock()          # Serializes concurrent writes
        self._event_log: list[dict] = []
        self._agents: set[str] = set()
        self._notifications: dict[str, list[str]] = {}

    def register_agent(self, name: str) -> None:
        self._agents.add(name)
        self._notifications.setdefault(name, [])

    def _log(self, agent: str, op: str, detail: str) -> None:
        self._event_log.append({"ts": round(time.time(), 4), "agent": agent,
                                "op": op, "detail": detail})

    def write(self, author: str, namespace: str, text: str,
              category: str = "fact", broadcast: bool = False) -> str:
        # Namespace isolation: an agent may ONLY write under its own name.
        if namespace != author:
            self._log(author, "WRITE_DENIED", f"tried namespace '{namespace}'")
            raise PermissionError(
                f"{author} cannot write to namespace '{namespace}' "
                f"(agents only write to their own namespace)")
        with self._lock:  # Even if rounds are sequential, the code must guard
            rec = MemoryRecord(
                id=f"{namespace}-{len(self._records):03d}", namespace=namespace,
                author=author, text=text, category=category,
                timestamp=time.time(), embedding=mock_embed(text))
            self._records.append(rec)
            self._log(author, "WRITE", f"[{rec.id}] {text[:50]}")
            if broadcast:
                # Push a notification into every OTHER agent's inbox; they
                # receive it at the start of their next round.
                for other in self._agents - {author}:
                    self._notifications[other].append(f"from {author}: {text}")
                self._log(author, "BROADCAST", f"[{rec.id}] to {len(self._agents) - 1} agents")
        return rec.id

    def read(self, reader: str, namespace: str | None = None) -> list[MemoryRecord]:
        """Reads are open: every agent can read every namespace."""
        self._log(reader, "READ", f"namespace={namespace or 'ALL'}")
        return [r for r in self._records if namespace is None or r.namespace == namespace]

    def search(self, reader: str, query: str, top_k: int = 3) -> list[MemoryRecord]:
        self._log(reader, "SEARCH", query[:50])
        q = mock_embed(query)
        ranked = sorted(self._records,
                        key=lambda r: cosine_similarity(q, r.embedding), reverse=True)
        return ranked[:top_k]

    def drain_notifications(self, agent: str) -> list[str]:
        notes, self._notifications[agent] = self._notifications[agent], []
        return notes

    @property
    def event_log(self) -> list[dict]:
        return list(self._event_log)


class BaseAgent:
    """Common shape: private working memory + bus access. Extending the
    system = subclass this and append to the schedule (no edits elsewhere)."""

    def __init__(self, name: str, bus: SharedMemoryBus):
        self.name = name
        self.bus = bus
        self.working_memory: dict[str, Any] = {}  # PRIVATE -- never shared
        bus.register_agent(name)

    def begin_round(self, round_no: int) -> list[str]:
        notes = self.bus.drain_notifications(self.name)
        self.working_memory[f"round{round_no}_notifications"] = notes
        return notes

    def run_round(self, round_no: int, task: str) -> str:
        raise NotImplementedError


class ResearchAgent(BaseAgent):
    """Finds facts (mock search) and stores them, broadcasting key ones."""

    FACTS_R1 = [
        "Fly.io: usage-based pricing, ~5 USD/mo entry, machines API, global edge regions",
        "Render: fixed-tier pricing from 7 USD/mo, zero-config deploys, good Postgres story",
        "Vercel: free hobby tier, 20 USD/seat pro, frontend-first, serverless functions only",
    ]
    FACTS_R4 = [
        "Fly.io: no managed Postgres anymore (unmanaged only) - ops burden for a small team",
        "Render: autoscaling limited on starter tiers; Vercel: backend needs serverless rewrite",
    ]

    def run_round(self, round_no: int, task: str) -> str:
        self.begin_round(round_no)
        facts = self.FACTS_R1 if round_no == 1 else self.FACTS_R4
        for fact in facts:
            self.bus.write(self.name, self.name, fact, "fact", broadcast=True)
        self.working_memory["facts_stored"] = self.working_memory.get("facts_stored", 0) + len(facts)
        return f"stored {len(facts)} facts"


class AnalysisAgent(BaseAgent):
    """Reads research facts, produces conclusions, broadcasts them."""

    def run_round(self, round_no: int, task: str) -> str:
        notes = self.begin_round(round_no)
        facts = self.bus.read(self.name, namespace="ResearchAgent")
        # The conclusions are hardcoded (mock LLM) but provably grounded:
        # they cite how many facts/notifications they were derived from.
        if round_no == 2:
            conclusions = [
                f"Pricing: Render most predictable (fixed tiers); derived from {len(facts)} facts",
                f"DX: Vercel best for frontend, Fly.io best for global low-latency backends",
            ]
        else:
            conclusions = [
                f"Revised: Render wins for a B2B SaaS backend (managed Postgres, predictable cost), "
                f"based on {len(facts)} facts incl. round-4 findings",
            ]
        for c in conclusions:
            self.bus.write(self.name, self.name, c, "conclusion", broadcast=True)
        self.working_memory["facts_seen"] = len(facts)
        return f"saw {len(notes)} notifications, wrote {len(conclusions)} conclusions"


class ReportAgent(BaseAgent):
    """Reads conclusions and writes the (final) report to its namespace."""

    def run_round(self, round_no: int, task: str) -> str:
        self.begin_round(round_no)
        conclusions = self.bus.read(self.name, namespace="AnalysisAgent")
        facts = self.bus.read(self.name, namespace="ResearchAgent")
        label = "FINAL report" if round_no >= 6 else "draft report"
        report = (f"{label} -- {task}. Recommendation: Render. "
                  f"Grounded in {len(facts)} research facts and {len(conclusions)} "
                  f"analysis conclusions. Platforms compared: Fly.io, Render, Vercel. "
                  f"Key conclusion: {conclusions[-1].text if conclusions else 'n/a'}")
        self.bus.write(self.name, self.name, report, "report")
        self.working_memory["last_report"] = report
        return label


class FactCheckAgent(BaseAgent):
    """4th agent added WITHOUT modifying any existing class -- extensibility."""

    def run_round(self, round_no: int, task: str) -> str:
        everything = self.bus.read(self.name)  # Can read all namespaces
        verdict = f"Fact-check: {len(everything)} records reviewed, no contradiction found"
        self.bus.write(self.name, self.name, verdict, "audit")
        return verdict


def hard_ex2_shared_memory_bus():
    """
    Solution: 3 specialized agents collaborating through a SharedMemoryBus
    over 6 sequential rounds, then a 4th agent added without code changes.
    """
    print("\n" + "=" * 60)
    print("  HARD 2: Distributed Multi-Agent Shared Memory")
    print("=" * 60)

    task = "Comparative analysis of Fly.io vs Render vs Vercel for a B2B SaaS startup"
    bus = SharedMemoryBus()
    research = ResearchAgent("ResearchAgent", bus)
    analysis = AnalysisAgent("AnalysisAgent", bus)
    report = ReportAgent("ReportAgent", bus)
    schedule = [research, analysis, report, research, analysis, report]

    print(f"\n  Task: {task}\n")
    for round_no, agent in enumerate(schedule, start=1):
        outcome = agent.run_round(round_no, task)
        print(f"  Round {round_no} | {agent.name:14s} | {outcome}")

    # --- Criterion: namespace isolation (write denied across namespaces) ---
    print("\n  --- Namespace isolation test ---")
    try:
        bus.write("ResearchAgent", "AnalysisAgent", "sneaky overwrite attempt")
        raise AssertionError("Cross-namespace write should have been denied")
    except PermissionError as e:
        print(f"  Denied as expected: {e}")

    # --- Criterion: broadcast (analysis saw research facts next round) -----
    notes_r2 = analysis.working_memory["round2_notifications"]
    assert any("Fly.io" in n and "ResearchAgent" in n for n in notes_r2), notes_r2
    print(f"\n  Broadcast check: AnalysisAgent round 2 received "
          f"{len(notes_r2)} notifications, e.g.:\n    {notes_r2[0][:90]}")

    # --- Criterion: locks under real concurrency ---------------------------
    print("\n  --- Concurrency test: 5 threads writing simultaneously ---")
    bus2 = SharedMemoryBus()
    for i in range(5):
        bus2.register_agent(f"worker{i}")
    threads = [threading.Thread(
        target=lambda n=f"worker{i}": [bus2.write(n, n, f"{n} entry {j}") for j in range(20)])
        for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [r.id for r in bus2.read("auditor")]
    assert len(ids) == 100 and len(set(ids)) == 100, "Lost or duplicated writes!"
    print(f"  100 concurrent writes -> {len(set(ids))} unique records (no loss).")

    # --- Criterion: event log traces who did what --------------------------
    authors = {e["agent"] for e in bus.event_log}
    assert {"ResearchAgent", "AnalysisAgent", "ReportAgent"} <= authors
    print(f"\n  --- Event log ({len(bus.event_log)} events) -- first 6 ---")
    for e in bus.event_log[:6]:
        print(f"    {e['agent']:14s} {e['op']:10s} {e['detail'][:55]}")

    # --- Criterion: final report aggregates all 3 agents -------------------
    final = report.working_memory["last_report"]
    print(f"\n  Final report:\n    {final[:220]}...")
    assert "FINAL report" in final
    for kw in ("Fly.io", "Render", "Vercel"):       # From ResearchAgent
        assert kw in final
    assert "research facts" in final and "conclusions" in final  # Data lineage
    assert "Recommendation" in final                 # ReportAgent's own output

    # --- Criterion: private working memories stay private ------------------
    assert "facts_stored" in research.working_memory
    assert "facts_stored" not in analysis.working_memory
    assert research.working_memory is not analysis.working_memory

    # --- Criterion: extensible -- 4th agent, zero changes to existing code -
    print("\n  --- Extensibility: adding FactCheckAgent (round 7) ---")
    checker = FactCheckAgent("FactCheckAgent", bus)
    print(f"  Round 7 | FactCheckAgent | {checker.run_round(7, task)}")
    assert any(r.namespace == "FactCheckAgent" for r in bus.read("auditor"))

    print("\n  PASS -- isolation, broadcast, locks, event log, 3-agent report, "
          "4th agent added cleanly.\n")


# ==========================================================================
# MAIN -- Run both hard solutions
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 3 HARD Solutions -- Memory & State")
    print("#" * 60)

    hard_ex1_memgpt_lite()
    hard_ex2_shared_memory_bus()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

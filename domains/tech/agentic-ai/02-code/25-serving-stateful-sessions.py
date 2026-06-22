"""
Day 25 -- Serving stateful agents & session management at scale.

Demonstrates:
  1. Checkpointer interface (abstract base) with two concrete implementations:
       - MemoryCheckpointer  : dict-based, in-process, zero persistence
       - SQLiteCheckpointer  : sqlite3 stdlib, real file or :memory:, durable
  2. SessionManager         : thread-scoped state per thread_id
  3. Horizontal scaling sim : two stateless "workers" sharing a SQLiteCheckpointer,
                              one worker can resume a session started by the other
  4. OnlineDriftMonitor     : sliding-window success-rate tracker with drift alert

Dependencies: stdlib only (sqlite3, threading, uuid, collections, dataclasses).

Run:
    python domains/tech/agentic-ai/02-code/25-serving-stateful-sessions.py
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# 1. CHECKPOINTER INTERFACE
# ===========================================================================

@dataclass
class Checkpoint:
    """
    Snapshot of the state of a single thread at one point in time.

    Attributes
    ----------
    thread_id   : unique identifier for the conversation / session
    step        : monotonically increasing counter (each turn = +1)
    state       : arbitrary JSON-serialisable dict (messages, variables, …)
    created_at  : unix timestamp of this snapshot
    """
    thread_id: str
    step: int
    state: dict[str, Any]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "step": self.step,
            "state": self.state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            thread_id=d["thread_id"],
            step=d["step"],
            state=d["state"],
            created_at=d["created_at"],
        )


class BaseCheckpointer(ABC):
    """
    Abstract checkpointer: load the latest state for a thread, save a new one.

    In LangGraph this is `BaseCheckpointSaver`; we reproduce the same shape
    with stdlib only for pedagogical purposes.
    """

    @abstractmethod
    def load(self, thread_id: str) -> Checkpoint | None:
        """Return the most-recent checkpoint for *thread_id*, or None."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Persist *checkpoint* (upsert by thread_id + step)."""

    @abstractmethod
    def list_threads(self) -> list[str]:
        """Return all known thread_ids."""


# ===========================================================================
# 2. IN-MEMORY CHECKPOINTER  (dev / tests only)
# ===========================================================================

class MemoryCheckpointer(BaseCheckpointer):
    """
    Stores checkpoints in a plain Python dict.

    Thread-safe via a reentrant lock so it can be shared across threads in tests.
    Downside: all data is lost when the process exits — not suitable for production.
    """

    def __init__(self) -> None:
        # key = thread_id, value = list[Checkpoint] sorted by step
        self._store: dict[str, list[Checkpoint]] = {}
        self._lock = threading.RLock()

    def load(self, thread_id: str) -> Checkpoint | None:
        with self._lock:
            history = self._store.get(thread_id, [])
            return history[-1] if history else None

    def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._store.setdefault(checkpoint.thread_id, []).append(checkpoint)

    def list_threads(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


# ===========================================================================
# 3. SQLITE CHECKPOINTER  (single-node production / edge)
# ===========================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id   TEXT    NOT NULL,
    step        INTEGER NOT NULL,
    state_json  TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (thread_id, step)
);
"""


class SQLiteCheckpointer(BaseCheckpointer):
    """
    Durable checkpointer backed by sqlite3 (stdlib — no extra dependencies).

    Use conn_string=":memory:" for a transient in-process DB (useful in tests);
    use a real path like "./sessions.db" for actual durability.

    Thread-safe: each thread opens its own connection (check_same_thread=False
    is the simplest approach for a single file with WAL mode enabled).
    """

    def __init__(self, conn_string: str = ":memory:") -> None:
        self._conn_string = conn_string
        self._local = threading.local()   # one connection per thread
        # Initialise the schema on the main thread
        self._get_conn()

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or lazily create) this thread's SQLite connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                self._conn_string,
                check_same_thread=False,  # we handle synchronisation ourselves
            )
            conn.execute("PRAGMA journal_mode=WAL;")  # concurrent readers + 1 writer
            conn.execute(_SCHEMA)
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def load(self, thread_id: str) -> Checkpoint | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT step, state_json, created_at FROM checkpoints "
            "WHERE thread_id = ? ORDER BY step DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        step, state_json, created_at = row
        return Checkpoint(
            thread_id=thread_id,
            step=step,
            state=json.loads(state_json),
            created_at=created_at,
        )

    def save(self, checkpoint: Checkpoint) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, step, state_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                checkpoint.thread_id,
                checkpoint.step,
                json.dumps(checkpoint.state),
                checkpoint.created_at,
            ),
        )
        conn.commit()

    def list_threads(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
        return [r[0] for r in rows]


# ===========================================================================
# 4. SESSION MANAGER  (thread-scoped state orchestration)
# ===========================================================================

class SessionManager:
    """
    High-level API that wraps a checkpointer.

    Each `thread_id` represents one user conversation.
    The SessionManager loads / saves checkpoints transparently and exposes
    a simple `process_turn` method that simulates an agent turn.
    """

    def __init__(self, checkpointer: BaseCheckpointer) -> None:
        self._cp = checkpointer

    def get_or_create(self, thread_id: str) -> Checkpoint:
        """Load the current state, or create a fresh empty checkpoint."""
        existing = self._cp.load(thread_id)
        if existing is not None:
            return existing
        return Checkpoint(
            thread_id=thread_id,
            step=0,
            state={"messages": [], "metadata": {}},
        )

    def process_turn(
        self,
        thread_id: str,
        user_message: str,
        worker_id: str = "worker-?",
    ) -> str:
        """
        Simulate one agent turn:
          1. Load the current checkpoint
          2. Append the user message to state["messages"]
          3. Generate a mock assistant reply
          4. Save a new checkpoint (step + 1)
          5. Return the assistant reply

        The `worker_id` is logged to show WHICH worker handled this turn —
        proving that any worker can resume any session.
        """
        ckpt = self.get_or_create(thread_id)

        # Append user message
        ckpt.state["messages"].append({"role": "user", "content": user_message})

        # Simulate assistant response (mock LLM — no API key needed)
        history_len = len(ckpt.state["messages"])
        reply = (
            f"[{worker_id}] Turn #{ckpt.step + 1}: "
            f"I see {history_len} message(s) so far. "
            f"You said: '{user_message}'"
        )
        ckpt.state["messages"].append({"role": "assistant", "content": reply})

        # Persist new checkpoint
        new_ckpt = Checkpoint(
            thread_id=thread_id,
            step=ckpt.step + 1,
            state=ckpt.state,
        )
        self._cp.save(new_ckpt)

        return reply


# ===========================================================================
# 5. HORIZONTAL SCALING SIMULATION
# ===========================================================================

def simulate_horizontal_scaling(checkpointer: BaseCheckpointer) -> None:
    """
    Spin up two simulated stateless workers that share the same checkpointer.

    Worker A starts session A and session B.
    Worker B then CONTINUES session A and session B.
    This proves that state is not tied to a specific worker process.
    """
    print("\n" + "=" * 60)
    print("HORIZONTAL SCALING SIMULATION")
    print("Two stateless workers sharing one SQLiteCheckpointer")
    print("=" * 60)

    session_a = f"user-alice-{uuid.uuid4().hex[:6]}"
    session_b = f"user-bob-{uuid.uuid4().hex[:6]}"

    mgr_a = SessionManager(checkpointer)   # Worker A's view
    mgr_b = SessionManager(checkpointer)   # Worker B's view (same store)

    # --- Worker A handles two sessions ---
    print("\n[Worker-A] Starting session Alice")
    reply = mgr_a.process_turn(session_a, "Hello, I need help with invoices.", "Worker-A")
    print(f"  -> {reply}")

    print("[Worker-A] Starting session Bob")
    reply = mgr_a.process_turn(session_b, "Hi, can you check my fleet status?", "Worker-A")
    print(f"  -> {reply}")

    print("[Worker-A] Second turn for Alice")
    reply = mgr_a.process_turn(session_a, "Specifically invoice #1042.", "Worker-A")
    print(f"  -> {reply}")

    # --- Worker B picks up BOTH sessions (stateless handoff) ---
    print("\n[Worker-B] Resuming Alice's session (started by Worker-A)")
    reply = mgr_b.process_turn(session_a, "What was that invoice number again?", "Worker-B")
    print(f"  -> {reply}")

    print("[Worker-B] Resuming Bob's session (started by Worker-A)")
    reply = mgr_b.process_turn(session_b, "Any anomalies detected?", "Worker-B")
    print(f"  -> {reply}")

    # Verify: Alice's session now has 3 turns (2 by A + 1 by B)
    final_ckpt = checkpointer.load(session_a)
    assert final_ckpt is not None
    assert final_ckpt.step == 3, f"Expected step 3, got {final_ckpt.step}"
    print(f"\nVerification OK: Alice's session is at step {final_ckpt.step} "
          f"({len(final_ckpt.state['messages'])} messages total)")
    print("Any worker can resume any session — stateless handoff confirmed.")


# ===========================================================================
# 6. CONCURRENT WORKERS  (threading demo)
# ===========================================================================

def simulate_concurrent_workers(checkpointer: BaseCheckpointer) -> None:
    """
    Two OS threads (simulating real concurrent workers) process turns on
    DIFFERENT sessions simultaneously.

    If they processed the SAME session concurrently we would need a distributed
    lock (advisory lock in Postgres, SETNX in Redis). Here we show the safe
    pattern: each session is processed by one worker at a time.
    """
    print("\n" + "=" * 60)
    print("CONCURRENT WORKERS (threading)")
    print("=" * 60)

    results: list[str] = []
    lock = threading.Lock()

    def worker(worker_id: str, thread_id: str, messages: list[str]) -> None:
        mgr = SessionManager(checkpointer)
        for msg in messages:
            reply = mgr.process_turn(thread_id, msg, worker_id)
            with lock:
                results.append(reply)
                print(f"  {reply}")

    session_x = f"session-x-{uuid.uuid4().hex[:6]}"
    session_y = f"session-y-{uuid.uuid4().hex[:6]}"

    t1 = threading.Thread(
        target=worker,
        args=("Worker-1", session_x, ["What is LangGraph?", "How does checkpointing work?"]),
    )
    t2 = threading.Thread(
        target=worker,
        args=("Worker-2", session_y, ["Explain Redis TTL.", "What is WAL mode in SQLite?"]),
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"\n{len(results)} turns completed across 2 concurrent workers.")


# ===========================================================================
# 7. ONLINE DRIFT MONITOR
# ===========================================================================

class OnlineDriftMonitor:
    """
    Sliding-window success-rate tracker for online eval.

    Records the outcome (success=True/False) of each agent session.
    Reports the success rate over the last `window_size` sessions.
    Fires an alert if the rate drops below `alert_threshold`.

    In production this would be backed by a time-series DB (Prometheus, InfluxDB).
    Here we use a deque for simplicity.

    Corresponds to the "fenetre glissante" pattern described in the theory (J25 § 5.2).
    """

    def __init__(
        self,
        window_size: int = 10,
        alert_threshold: float = 0.70,
        baseline: float | None = None,
    ) -> None:
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        # baseline comes from offline eval (J11); used to compute drift delta
        self.baseline = baseline
        self._window: deque[bool] = deque(maxlen=window_size)
        self._total = 0
        self._successes = 0

    def record(self, success: bool) -> None:
        """Log the outcome of one session."""
        self._window.append(success)
        self._total += 1
        if success:
            self._successes += 1

    @property
    def current_rate(self) -> float:
        """Success rate over the current window."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def overall_rate(self) -> float:
        """Overall success rate since start."""
        if self._total == 0:
            return 0.0
        return self._successes / self._total

    def check_drift(self) -> dict[str, Any]:
        """
        Compare the current window rate against the alert threshold.
        If a baseline is set, also compute the delta vs offline-eval baseline.
        """
        rate = self.current_rate
        alert = rate < self.alert_threshold
        result: dict[str, Any] = {
            "window_size": len(self._window),
            "current_rate": round(rate, 3),
            "alert_threshold": self.alert_threshold,
            "alert": alert,
        }
        if self.baseline is not None:
            delta = rate - self.baseline
            result["baseline"] = self.baseline
            result["delta_vs_baseline"] = round(delta, 3)
            result["drift_detected"] = delta < -0.10  # >10 pp drop signals drift
        return result


def simulate_online_eval() -> None:
    """
    Simulate a production traffic stream:
      - First 15 sessions: healthy (high success rate)
      - Next 10 sessions:  degraded (low success rate) → drift alert
    """
    print("\n" + "=" * 60)
    print("ONLINE EVAL & DRIFT DETECTION")
    print("=" * 60)

    # Baseline from offline eval (J11): 87% success
    monitor = OnlineDriftMonitor(window_size=10, alert_threshold=0.70, baseline=0.87)

    import random
    rng = random.Random(42)  # fixed seed for reproducibility

    # Phase 1: healthy traffic
    print("\nPhase 1 — healthy traffic (success rate ~85%)")
    for i in range(15):
        outcome = rng.random() < 0.85
        monitor.record(outcome)
        if (i + 1) % 5 == 0:
            status = monitor.check_drift()
            print(f"  After {i+1} sessions: rate={status['current_rate']:.0%}  "
                  f"alert={status['alert']}  "
                  f"drift={status.get('drift_detected', 'N/A')}")

    # Phase 2: degraded traffic (simulates LLM model update / distribution shift)
    print("\nPhase 2 — degraded traffic (success rate ~40%) — DRIFT expected")
    for i in range(10):
        outcome = rng.random() < 0.40
        monitor.record(outcome)
        if (i + 1) % 5 == 0:
            status = monitor.check_drift()
            flag = "  <-- DRIFT ALERT" if status.get("drift_detected") else ""
            print(f"  After {15 + i+1} sessions: rate={status['current_rate']:.0%}  "
                  f"alert={status['alert']}  "
                  f"delta={status.get('delta_vs_baseline', 'N/A')}{flag}")

    final = monitor.check_drift()
    print(f"\nFinal summary: overall_rate={monitor.overall_rate:.0%}  "
          f"window_rate={final['current_rate']:.0%}  "
          f"drift={final.get('drift_detected', 'N/A')}")
    assert final.get("drift_detected") is True, "Expected drift to be detected"
    print("Drift detection assertion passed.")


# ===========================================================================
# 8. BACKEND COMPARISON DEMO
# ===========================================================================

def compare_backends() -> None:
    """
    Run the same sequence of turns against both backend types and show
    that the sessions survive a backend 'restart' (new instance, same DB).
    """
    print("\n" + "=" * 60)
    print("BACKEND COMPARISON: MemoryCheckpointer vs SQLiteCheckpointer")
    print("=" * 60)

    thread_id = "demo-thread-compare"

    # --- MemoryCheckpointer ---
    mem_cp = MemoryCheckpointer()
    mgr = SessionManager(mem_cp)
    mgr.process_turn(thread_id, "First message", "mem-worker")
    mgr.process_turn(thread_id, "Second message", "mem-worker")
    ckpt = mem_cp.load(thread_id)
    print(f"\n[MemoryCheckpointer] step={ckpt.step}, "
          f"messages={len(ckpt.state['messages'])}")

    # Simulate restart: create a NEW MemoryCheckpointer instance
    mem_cp2 = MemoryCheckpointer()
    ckpt_after = mem_cp2.load(thread_id)
    print(f"[MemoryCheckpointer] After 'restart': checkpoint={ckpt_after}  "
          f"(None = data lost)")
    assert ckpt_after is None, "MemoryCheckpointer should lose state on restart"

    # --- SQLiteCheckpointer with :memory: (durable within the same connection) ---
    # We use the SAME instance to show persistence across SessionManager instances
    db_cp = SQLiteCheckpointer(":memory:")
    mgr1 = SessionManager(db_cp)
    mgr1.process_turn(thread_id, "First message", "sql-worker-1")
    mgr1.process_turn(thread_id, "Second message", "sql-worker-1")

    # A second SessionManager pointing to the SAME checkpointer object
    mgr2 = SessionManager(db_cp)
    reply = mgr2.process_turn(thread_id, "Third message from new manager", "sql-worker-2")
    ckpt = db_cp.load(thread_id)
    print(f"\n[SQLiteCheckpointer] step={ckpt.step}, "
          f"messages={len(ckpt.state['messages'])}")
    print(f"[SQLiteCheckpointer] Resumed by different manager: '{reply}'")
    assert ckpt.step == 3, f"Expected step 3, got {ckpt.step}"
    print("[SQLiteCheckpointer] Persistence across managers: OK")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("Day 25 — Serving Stateful Agents & Sessions at Scale")
    print("stdlib only — no LangGraph, no external services required\n")

    # 1. Compare the two backends
    compare_backends()

    # 2. Simulate horizontal scaling with a shared SQLiteCheckpointer
    shared_db = SQLiteCheckpointer(":memory:")
    simulate_horizontal_scaling(shared_db)

    # 3. Concurrent workers on different sessions
    simulate_concurrent_workers(shared_db)

    # 4. Online eval & drift detection
    simulate_online_eval()

    print("\n" + "=" * 60)
    print("All demos completed successfully. Exit code 0.")
    print("=" * 60)

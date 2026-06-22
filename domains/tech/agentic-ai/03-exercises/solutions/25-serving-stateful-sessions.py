"""
Day 25 -- Solutions to the easy exercises for serving stateful agents & sessions.

Run the whole file to execute every solution:
    python domains/tech/agentic-ai/03-exercises/solutions/25-serving-stateful-sessions.py
"""

from __future__ import annotations

import sys
import time
import uuid
from collections import deque
from importlib import import_module
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import from the day-25 code module (same pattern as 13-securite-robustesse.py)
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

day25 = import_module("25-serving-stateful-sessions")
BaseCheckpointer = day25.BaseCheckpointer
Checkpoint = day25.Checkpoint
SQLiteCheckpointer = day25.SQLiteCheckpointer
SessionManager = day25.SessionManager
OnlineDriftMonitor = day25.OnlineDriftMonitor


# ===========================================================================
# SOLUTION 1 -- TTL & session expiration
# ===========================================================================

class TTLSQLiteCheckpointer(SQLiteCheckpointer):
    """
    SQLiteCheckpointer extended with an automatic TTL mechanism.

    Sessions whose last checkpoint is older than `ttl_seconds` are considered
    expired: `load` returns None for them, and `purge_expired` deletes them.
    """

    def __init__(self, conn_string: str = ":memory:", ttl_seconds: float = 3600.0) -> None:
        super().__init__(conn_string)
        self.ttl_seconds = ttl_seconds

    def _is_expired(self, checkpoint: Checkpoint) -> bool:
        """True if the checkpoint was created more than ttl_seconds ago."""
        return (time.time() - checkpoint.created_at) > self.ttl_seconds

    def load(self, thread_id: str) -> Checkpoint | None:
        """Return the latest checkpoint, or None if it is expired."""
        ckpt = super().load(thread_id)
        if ckpt is None:
            return None
        if self._is_expired(ckpt):
            return None   # treat as non-existent — purge_expired will clean up
        return ckpt

    def purge_expired(self) -> int:
        """
        Delete all threads whose most-recent checkpoint is older than ttl_seconds.
        Returns the number of threads (not rows) deleted.
        """
        cutoff = time.time() - self.ttl_seconds
        conn = self._get_conn()

        # Find thread_ids where the LATEST checkpoint is expired
        expired_threads = conn.execute(
            """
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            HAVING MAX(created_at) < ?
            """,
            (cutoff,),
        ).fetchall()

        if not expired_threads:
            return 0

        ids = [row[0] for row in expired_threads]
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})", ids
        )
        conn.commit()
        return len(ids)


def solution_1() -> None:
    print("\n=== Solution 1: TTL & expiration ===")
    cp = TTLSQLiteCheckpointer(":memory:", ttl_seconds=60.0)

    # Helper: save a checkpoint with a custom created_at (bypasses TTL on write)
    def _force_save(thread_id: str, age_seconds: float) -> None:
        ckpt = Checkpoint(
            thread_id=thread_id,
            step=1,
            state={"messages": []},
            created_at=time.time() - age_seconds,
        )
        # Call parent save to skip the TTL check on load
        super(TTLSQLiteCheckpointer, cp).save(ckpt)

    # Create 3 sessions with different ages
    _force_save("session-fresh",       age_seconds=5.0)    # 5 s old  -> alive
    _force_save("session-old",         age_seconds=120.0)  # 120 s old -> expired
    _force_save("session-borderline",  age_seconds=59.0)   # 59 s old  -> alive

    # Before purge: load each
    print(f"  Before purge:")
    print(f"    session-fresh       : {cp.load('session-fresh')}")
    print(f"    session-old         : {cp.load('session-old')} (None = expired via load)")
    print(f"    session-borderline  : {cp.load('session-borderline')}")

    # Purge expired sessions from the DB
    deleted = cp.purge_expired()
    print(f"\n  purge_expired() deleted {deleted} thread(s)")

    # After purge: verify
    assert cp.load("session-old") is None,         "session-old should be gone"
    assert cp.load("session-fresh") is not None,   "session-fresh should survive"
    assert cp.load("session-borderline") is not None, "session-borderline should survive"

    remaining = cp.list_threads()
    print(f"  Remaining threads: {remaining}")
    assert "session-old" not in remaining
    assert "session-fresh" in remaining
    assert "session-borderline" in remaining
    print("  All assertions passed.")


# ===========================================================================
# SOLUTION 2 -- Multi-user isolation
# ===========================================================================

class SecureSessionManager:
    """
    SessionManager wrapper that enforces owner-based access control.

    Each session has an `owner_id` stored in state.
    Any `process_turn` call with a mismatched `user_id` raises PermissionError.
    """

    def __init__(self, checkpointer: BaseCheckpointer) -> None:
        self._mgr = SessionManager(checkpointer)
        self._cp = checkpointer

    def create_session(self, user_id: str) -> str:
        """
        Create a new session owned by *user_id*.
        Returns the opaque thread_id (UUID v4).
        """
        thread_id = str(uuid.uuid4())
        initial = Checkpoint(
            thread_id=thread_id,
            step=0,
            state={"messages": [], "owner_id": user_id},
        )
        self._cp.save(initial)
        return thread_id

    def process_turn(self, thread_id: str, user_id: str, message: str) -> str:
        """
        Process one turn, but only if *user_id* owns this thread.
        Raises PermissionError otherwise.
        """
        ckpt = self._cp.load(thread_id)
        if ckpt is None:
            raise KeyError(f"thread not found: {thread_id}")
        owner = ckpt.state.get("owner_id")
        if owner != user_id:
            raise PermissionError(
                f"access denied: thread owned by another user"
            )
        return self._mgr.process_turn(thread_id, message, f"secure-worker/{user_id}")


def solution_2() -> None:
    print("\n=== Solution 2: multi-user isolation ===")
    cp = SQLiteCheckpointer(":memory:")
    mgr = SecureSessionManager(cp)

    # Alice creates her session
    alice_thread = mgr.create_session("alice")
    print(f"  Alice's thread_id : {alice_thread} (opaque UUID)")

    # Bob tries to intrude
    try:
        mgr.process_turn(alice_thread, "bob", "Steal Alice's data!")
        print("  ERROR: should have raised PermissionError")
    except PermissionError as e:
        print(f"  Bob's attempt blocked: {e}")
        assert "access denied" in str(e)

    # Alice can still use her session
    reply = mgr.process_turn(alice_thread, "alice", "My first real message.")
    print(f"  Alice's turn processed: {reply[:60]}...")

    # Bob creates his own session (should work fine)
    bob_thread = mgr.create_session("bob")
    reply_bob = mgr.process_turn(bob_thread, "bob", "My own session, no problem.")
    print(f"  Bob's own session: {reply_bob[:60]}...")

    assert alice_thread != bob_thread
    print("  All isolation assertions passed.")


# ===========================================================================
# SOLUTION 3 -- Advanced drift monitor with per-type breakdown
# ===========================================================================

class AdvancedDriftMonitor(OnlineDriftMonitor):
    """
    Extends OnlineDriftMonitor to track success rates per query type.

    Each `record` call takes an optional `query_type` parameter.
    The sliding window also applies per type (via separate deques).
    """

    def __init__(
        self,
        window_size: int = 10,
        alert_threshold: float = 0.70,
        baseline: float | None = None,
    ) -> None:
        super().__init__(window_size, alert_threshold, baseline)
        # type -> deque[bool] of outcomes within the window
        self._type_windows: dict[str, deque[bool]] = {}

    def record(self, success: bool, query_type: str = "default") -> None:  # type: ignore[override]
        """Record a session outcome with its query type."""
        super().record(success)
        if query_type not in self._type_windows:
            self._type_windows[query_type] = deque(maxlen=self.window_size)
        self._type_windows[query_type].append(success)

    def success_rate_by_type(self) -> dict[str, float]:
        """Compute success rate per type over the current window."""
        rates: dict[str, float] = {}
        for qtype, window in self._type_windows.items():
            if window:
                rates[qtype] = round(sum(window) / len(window), 3)
            else:
                rates[qtype] = 0.0
        return rates

    def detect_type_drift(
        self,
        baseline_by_type: dict[str, float],
        threshold: float = 0.15,
    ) -> list[str]:
        """
        Return query types whose current rate is more than `threshold` below
        their baseline (e.g., threshold=0.15 means a 15 pp drop triggers drift).
        """
        current = self.success_rate_by_type()
        drifted: list[str] = []
        for qtype, base_rate in baseline_by_type.items():
            current_rate = current.get(qtype, 0.0)
            if (base_rate - current_rate) > threshold:
                drifted.append(qtype)
        return drifted


def solution_3() -> None:
    print("\n=== Solution 3: per-type drift detection ===")
    import random
    rng = random.Random(99)

    baseline_by_type = {"faq": 0.95, "support": 0.80, "complex": 0.70}

    monitor = AdvancedDriftMonitor(window_size=20, alert_threshold=0.70)

    # Phase 1: healthy traffic
    print("\n  Phase 1 — healthy traffic")
    for _ in range(30):
        for qtype, rate in baseline_by_type.items():
            monitor.record(rng.random() < rate, query_type=qtype)

    rates_healthy = monitor.success_rate_by_type()
    print(f"  Rates by type (healthy): {rates_healthy}")
    drifted = monitor.detect_type_drift(baseline_by_type)
    print(f"  Drifted types (healthy): {drifted}")
    assert drifted == [], f"No drift expected in healthy phase, got {drifted}"

    # Phase 2: complex degrades to 30%
    print("\n  Phase 2 — 'complex' degrades to 30%")
    degraded_rates = {"faq": 0.95, "support": 0.80, "complex": 0.30}
    for _ in range(30):
        for qtype, rate in degraded_rates.items():
            monitor.record(rng.random() < rate, query_type=qtype)

    rates_degraded = monitor.success_rate_by_type()
    print(f"  Rates by type (degraded): {rates_degraded}")
    drifted = monitor.detect_type_drift(baseline_by_type)
    print(f"  Drifted types (degraded): {drifted}  <-- should contain 'complex'")

    assert "complex" in drifted, f"'complex' should be drifted, got {drifted}"
    assert "faq" not in drifted,     f"'faq' should NOT be drifted"
    assert "support" not in drifted, f"'support' should NOT be drifted"
    print("  All per-type drift assertions passed.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("Day 25 — Solutions: Serving Stateful Agents & Sessions")

    solution_1()
    solution_2()
    solution_3()

    print("\nAll solutions completed. Exit code 0.")

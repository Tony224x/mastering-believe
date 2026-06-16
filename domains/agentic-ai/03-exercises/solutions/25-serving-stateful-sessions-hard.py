"""
Solutions -- Day 25 (HARD): Serving stateful agents & session management

Contains solutions for:
  - Hard Ex 1: Fault-tolerant stateful serving cluster. A load balancer with
               session affinity routes turns to stateless workers backed by a
               SHARED checkpoint store. Killing the owning node triggers failover
               + session migration + restore -- proven with zero lost turns.
  - Hard Ex 2: Concurrency-safe session manager. Optimistic locking
               (compare-and-swap on a version) detects a write conflict between
               two interleaved requests to the same thread; the loser retries and
               wins, no lost update. Idempotency keys make retries safe.

Self-contained & offline. No network, no Redis/Postgres, no real threads:
concurrency is replayed via an explicit, deterministic op sequence so the
write conflict is reproducible every run. Checkpoint/BaseCheckpointer are a
minimal embed of 02-code/25-serving-stateful-sessions.py (not imported).

Run:  python 03-exercises/solutions/25-serving-stateful-sessions-hard.py
"""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ==========================================================================
# Shared minimal embed of Checkpoint / store (from 02-code, trimmed)
# ==========================================================================

@dataclass
class Checkpoint:
    """Snapshot of one thread's state (step = monotonically increasing turn id)."""
    thread_id: str
    step: int
    state: dict[str, Any]


class SharedCheckpointStore:
    """
    Durable shared store: the single source of truth.

    It outlives any worker object -- killing a worker loses NO checkpoint,
    which is exactly why stateless workers + external store beat sticky sessions.
    """

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    def load(self, thread_id: str) -> Checkpoint | None:
        ckpt = self._store.get(thread_id)
        # Defensive deep copy: a worker mutating its view never corrupts the store.
        return None if ckpt is None else copy.deepcopy(ckpt)

    def save(self, checkpoint: Checkpoint) -> None:
        self._store[checkpoint.thread_id] = copy.deepcopy(checkpoint)


# ==========================================================================
# HARD EXERCISE 1 -- Fault-tolerant stateful serving (failover + migration)
# ==========================================================================

class NoLiveWorker(RuntimeError):
    """Raised when no worker is alive to serve a request."""


class Worker:
    """Stateless worker: load -> compute -> save. Holds no per-session memory."""

    def __init__(self, worker_id: str, store: SharedCheckpointStore) -> None:
        self.worker_id = worker_id
        self.store = store
        self.alive = True

    def handle(self, thread_id: str, message: str) -> str:
        ckpt = self.store.load(thread_id) or Checkpoint(thread_id, 0, {"messages": []})
        msgs = ckpt.state.get("messages", [])
        msgs.append({"role": "user", "content": message})
        reply = f"[{self.worker_id}] turn#{ckpt.step + 1}: ack '{message}'"
        msgs.append({"role": "assistant", "content": reply, "by": self.worker_id})
        self.store.save(Checkpoint(thread_id, ckpt.step + 1, {"messages": msgs}))
        return reply


class Router:
    """
    Session-affinity router over the set of LIVE workers.

    A thread_id maps deterministically (stable hash) to one of the live workers.
    When its current owner dies, the next live worker becomes the new owner --
    that is the failover. Affinity holds as long as the owner stays alive.
    """

    def __init__(self, workers: list[Worker]) -> None:
        self.workers = workers

    def _live(self) -> list[Worker]:
        return [w for w in self.workers if w.alive]

    def owner(self, thread_id: str) -> Worker:
        live = self._live()
        if not live:
            raise NoLiveWorker("no live worker to serve the request")
        # Stable hash modulo the SORTED live set -> deterministic, reproducible.
        live_sorted = sorted(live, key=lambda w: w.worker_id)
        h = int(hashlib.md5(thread_id.encode()).hexdigest(), 16)
        return live_sorted[h % len(live_sorted)]


class Cluster:
    """Glue: router + stateless workers + shared store, with migration logging."""

    def __init__(self, n_workers: int) -> None:
        self.store = SharedCheckpointStore()
        self.workers = [Worker(f"w{i}", self.store) for i in range(n_workers)]
        self.router = Router(self.workers)
        self._last_owner: dict[str, str] = {}                  # thread_id -> worker_id
        self.migrations: list[tuple[str, str, str]] = []        # (tid, from, to)

    def dispatch(self, thread_id: str, message: str) -> dict[str, Any]:
        owner = self.router.owner(thread_id)
        prev = self._last_owner.get(thread_id)
        migrated = prev is not None and prev != owner.worker_id
        if migrated:
            self.migrations.append((thread_id, prev, owner.worker_id))
        self._last_owner[thread_id] = owner.worker_id
        # The (possibly new) owner restores state from the shared store inside handle().
        reply = owner.handle(thread_id, message)
        return {"worker": owner.worker_id, "reply": reply, "migrated": migrated}

    def kill(self, worker_id: str) -> None:
        for w in self.workers:
            if w.worker_id == worker_id:
                w.alive = False


def hard_ex1_fault_tolerant_serving() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: Fault-tolerant stateful serving (failover + migration)")
    print("=" * 60)

    cluster = Cluster(n_workers=3)
    tid = "session-S"

    # 3 turns -> all land on the same owner (affinity).
    owners = []
    for msg in ("t1", "t2", "t3"):
        r = cluster.dispatch(tid, msg)
        owners.append(r["worker"])
        assert r["migrated"] is False
    assert len(set(owners)) == 1, "affinity: all early turns must hit the same owner"
    original_owner = owners[0]
    assert cluster.store.load(tid).step == 3
    print(f"\n  3 turns on owner '{original_owner}' (affinity) -> step={cluster.store.load(tid).step}")

    # Kill the owner. Next turns must fail over to another LIVE worker.
    cluster.kill(original_owner)
    print(f"  killed owner '{original_owner}' -> failover expected")
    failover_workers = []
    for msg in ("t4", "t5"):
        r = cluster.dispatch(tid, msg)
        failover_workers.append(r["worker"])
    assert all(w != original_owner for w in failover_workers), "must not route to dead owner"
    assert len(set(failover_workers)) == 1, "failover target should be stable too"
    assert any(m[0] == tid for m in cluster.migrations), "a migration must be logged for S"

    # Continuity: 5 turns total, none lost, none duplicated, ordered.
    final = cluster.store.load(tid)
    assert final.step == 5, f"expected step 5 (3+2, no lost turn), got {final.step}"
    user_msgs = [m["content"] for m in final.state["messages"] if m["role"] == "user"]
    assert user_msgs == ["t1", "t2", "t3", "t4", "t5"], f"history order wrong: {user_msgs}"
    last_by = final.state["messages"][-1]["by"]
    assert last_by != original_owner and last_by == failover_workers[-1]
    print(f"  after failover -> step={final.step}, history={user_msgs}")
    print(f"  migrations logged: {cluster.migrations}")

    # Edge: kill workers until ONE survivor; session converges onto it.
    live_ids = sorted(w.worker_id for w in cluster.workers if w.alive)
    survivor = live_ids[0]
    for wid in live_ids[1:]:
        cluster.kill(wid)
    r = cluster.dispatch(tid, "t6")
    assert r["worker"] == survivor, "session must converge to the sole survivor"
    assert cluster.store.load(tid).step == 6
    print(f"  killed all but '{survivor}' -> session served by survivor, step=6")

    # Edge: kill the last worker -> NoLiveWorker, store still intact.
    cluster.kill(survivor)
    raised = False
    try:
        cluster.dispatch(tid, "t7")
    except NoLiveWorker as e:
        raised = True
        print(f"  killed last worker -> {type(e).__name__}: {e}")
    assert raised, "dispatching with no live worker must raise NoLiveWorker"
    assert cluster.store.load(tid).step == 6, "store survives worker death (no lost state)"

    print("\n  PASS -- failover migrated the session with zero lost turns; store durable.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Concurrent session manager (optimistic locking + idempotency)
# ==========================================================================

class VersionedStore:
    """
    Per-thread (version, state) with optimistic concurrency control.

    compare_and_swap only applies if the caller's expected_version still matches
    the current version. A stale writer (someone else committed first) is
    rejected -> the classic optimistic-locking conflict.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[int, dict[str, Any]]] = {}

    def read(self, thread_id: str) -> tuple[int, dict[str, Any]]:
        version, state = self._data.get(thread_id, (0, {"messages": [], "applied_keys": []}))
        # Return a defensive deep copy so callers can't mutate the store directly.
        return version, copy.deepcopy(state)

    def compare_and_swap(self, thread_id: str, expected_version: int,
                         new_state: dict[str, Any]) -> bool:
        current_version = self._data.get(thread_id, (0, None))[0]
        if expected_version != current_version:
            return False  # someone committed in between -> conflict
        self._data[thread_id] = (current_version + 1, copy.deepcopy(new_state))
        return True

    def version(self, thread_id: str) -> int:
        return self._data.get(thread_id, (0, None))[0]


class ConcurrentSessionManager:
    """
    Applies a turn under optimistic locking with bounded retries + idempotency.

    - Idempotency: a turn whose idempotency_key is already recorded is a no-op
      ('duplicate') -- safe to retry a request whose response was lost.
    - Conflict handling: on a failed CAS, re-read the latest version and retry,
      so a concurrent writer never causes a lost update.
    """

    def __init__(self, store: VersionedStore) -> None:
        self.store = store

    def apply_turn(self, thread_id: str, user_message: str, idempotency_key: str,
                   max_retries: int = 3) -> dict[str, Any]:
        for attempt in range(max_retries + 1):
            version, state = self.store.read(thread_id)

            # Idempotency: already applied -> do not apply again.
            if idempotency_key in state.get("applied_keys", []):
                return {"status": "duplicate", "version": version, "retries": attempt}

            new_state = copy.deepcopy(state)
            new_state["messages"].append({"role": "user", "content": user_message,
                                          "key": idempotency_key})
            new_state["applied_keys"].append(idempotency_key)

            if self.store.compare_and_swap(thread_id, version, new_state):
                return {"status": "applied", "retries": attempt,
                        "version": self.store.version(thread_id)}
            # else: conflict -> loop, re-read the newest version, retry.

        return {"status": "conflict_exhausted", "retries": max_retries}


def hard_ex2_concurrent_session_manager() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: Concurrent session manager (optimistic locking + idempotency)")
    print("=" * 60)

    store = VersionedStore()
    mgr = ConcurrentSessionManager(store)
    tid = "thread-X"

    # --- Explicit interleave of two requests on the SAME thread ---
    # We replay the race deterministically (no real threads): both read v0, R1
    # commits first, R2's CAS(v0) fails, R2 retries against v1 and commits.

    # R1 reads v0=0.
    v_r1, s_r1 = store.read(tid)
    assert v_r1 == 0
    # R2 reads v0=0 too (concurrent snapshot).
    v_r2, s_r2 = store.read(tid)
    assert v_r2 == 0

    # R1 commits its turn against v0 -> succeeds (version -> 1).
    s_r1["messages"].append({"role": "user", "content": "R1-msg", "key": "k-R1"})
    s_r1["applied_keys"].append("k-R1")
    assert store.compare_and_swap(tid, v_r1, s_r1) is True
    print(f"\n  R1 CAS(v0)         -> success, version now {store.version(tid)}")

    # R2 tries to commit against the now-stale v0 -> conflict (CAS returns False).
    s_r2["messages"].append({"role": "user", "content": "R2-msg", "key": "k-R2"})
    s_r2["applied_keys"].append("k-R2")
    assert store.compare_and_swap(tid, v_r2, s_r2) is False
    print(f"  R2 CAS(v0)         -> CONFLICT detected (version moved to {store.version(tid)})")

    # R2 retries via the manager: re-reads v1, CAS(v1) succeeds (version -> 2).
    r2_final = mgr.apply_turn(tid, "R2-msg-retry", idempotency_key="k-R2b")
    print(f"  R2 retry           -> {r2_final}")
    assert r2_final["status"] == "applied"
    assert r2_final["retries"] == 0, "via the manager the re-read sees v1 and commits at once"
    assert store.version(tid) == 2

    # No lost update: both R1's and R2's messages are present.
    _, final_state = store.read(tid)
    contents = [m["content"] for m in final_state["messages"]]
    assert "R1-msg" in contents and "R2-msg-retry" in contents, f"lost update! {contents}"
    print(f"  final messages     -> {contents} (no lost update)")

    # --- Drive the FULL race through the manager to assert retry counters ---
    store2 = VersionedStore()
    mgr2 = ConcurrentSessionManager(store2)
    tid2 = "thread-Y"

    # R1 reads v0, holds it. R2 will go first via the manager (commits v0->v1).
    v0, snap_r1 = store2.read(tid2)
    res_r2 = mgr2.apply_turn(tid2, "Y-R2", idempotency_key="y-R2")  # version 0 -> 1
    assert res_r2 == {"status": "applied", "retries": 0, "version": 1}

    # Now R1 attempts its CAS against the stale v0 -> conflict, then retries.
    snap_r1["messages"].append({"role": "user", "content": "Y-R1", "key": "y-R1"})
    snap_r1["applied_keys"].append("y-R1")
    assert store2.compare_and_swap(tid2, v0, snap_r1) is False  # stale -> conflict
    res_r1 = mgr2.apply_turn(tid2, "Y-R1", idempotency_key="y-R1")  # re-read v1, commit -> v2
    print(f"\n  forced race: R2={res_r2['retries']} retry, "
          f"R1 recovers via manager -> {res_r1}")
    assert res_r1["status"] == "applied" and store2.version(tid2) == 2

    # --- Idempotency: replay the SAME key -> duplicate, version unchanged ---
    before = store.version(tid)
    dup = mgr.apply_turn(tid, "R1-msg-again", idempotency_key="k-R1")  # k-R1 already applied
    print(f"\n  replay key 'k-R1'  -> {dup}")
    assert dup["status"] == "duplicate"
    assert store.version(tid) == before, "duplicate must NOT advance the version"
    _, after_state = store.read(tid)
    k_r1_count = sum(1 for m in after_state["messages"] if m.get("key") == "k-R1")
    assert k_r1_count == 1, f"idempotent replay must not duplicate the message (got {k_r1_count})"

    # --- Defensive copy: mutating a read copy must not corrupt the store ---
    _, leaked = store.read(tid)
    leaked["messages"].append({"role": "user", "content": "INJECTED"})
    _, fresh = store.read(tid)
    assert all(m.get("content") != "INJECTED" for m in fresh["messages"]), "read must be a copy"
    print(f"  defensive copy     -> store unaffected by mutating a read snapshot")

    print("\n  PASS -- conflict detected, retried without lost update; retries idempotent.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 25 HARD Solutions -- Serving Stateful Agents & Sessions")
    print("#" * 60)

    hard_ex1_fault_tolerant_serving()
    hard_ex2_concurrent_session_manager()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

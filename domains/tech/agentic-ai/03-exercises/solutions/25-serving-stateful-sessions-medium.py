"""
Solutions -- Day 25 (MEDIUM): Serving stateful agents & session management

Contains solutions for:
  - Medium Ex 1: LRUTTLSessionStore -- bounded session store with LRU eviction
                 + TTL expiry, deterministic via an injectable clock (no sleep).
  - Medium Ex 2: ConsistentHashRouter -- session affinity via a consistent-hash
                 ring; stable routing + minimal remapping when a node is added.
  - Medium Ex 3: ResilientWorker -- checkpoint/restore so a session resumes on a
                 different worker after a simulated crash, with no lost turns.

Self-contained & offline. No network, no Redis/Postgres, no API key.
The Checkpoint / BaseCheckpointer classes are a minimal embed of
02-code/25-serving-stateful-sessions.py (not imported). Deterministic by design.

Run:  python 03-exercises/solutions/25-serving-stateful-sessions-medium.py
"""

from __future__ import annotations

import bisect
import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


# ==========================================================================
# Shared minimal embed of Checkpoint / BaseCheckpointer (from 02-code, trimmed)
# ==========================================================================

@dataclass
class Checkpoint:
    """Snapshot of one thread's state at a point in time (step = turn counter)."""
    thread_id: str
    step: int
    state: dict[str, Any]
    created_at: float = 0.0


class BaseCheckpointer(ABC):
    @abstractmethod
    def load(self, thread_id: str) -> Checkpoint | None: ...

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None: ...


class InMemoryCheckpointer(BaseCheckpointer):
    """Durable-within-process dict store: survives the death of a 'worker' object."""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}

    def load(self, thread_id: str) -> Checkpoint | None:
        return self._store.get(thread_id)

    def save(self, checkpoint: Checkpoint) -> None:
        self._store[checkpoint.thread_id] = checkpoint


# ==========================================================================
# MEDIUM EXERCISE 1 -- LRU + TTL session store with eviction
# ==========================================================================

class LRUTTLSessionStore:
    """
    Bounded session cache: LRU eviction (by capacity) + TTL expiry (by age).

    Models a Redis L1 cache (theory 3.4 / 5.2). The clock is injectable so the
    test is fully deterministic -- we never call time.sleep.
    """

    def __init__(self, capacity: int, ttl: float, clock: Callable[[], float] | None = None) -> None:
        assert capacity >= 1
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock or __import__("time").time
        # OrderedDict preserves access order: front = LRU, back = most-recent.
        self._data: "OrderedDict[str, tuple[float, Checkpoint]]" = OrderedDict()

    def _expired(self, stamp: float) -> bool:
        return (self.clock() - stamp) >= self.ttl

    def save(self, checkpoint: Checkpoint) -> list[str]:
        tid = checkpoint.thread_id
        # Upsert and mark as most-recently-used (move to the back).
        if tid in self._data:
            self._data.move_to_end(tid)
        self._data[tid] = (self.clock(), checkpoint)
        self._data.move_to_end(tid)

        evicted: list[str] = []
        # Evict from the FRONT (least-recently-used) until within capacity.
        while len(self._data) > self.capacity:
            victim, _ = self._data.popitem(last=False)
            evicted.append(victim)
        return evicted

    def load(self, thread_id: str) -> Checkpoint | None:
        entry = self._data.get(thread_id)
        if entry is None:
            return None
        stamp, ckpt = entry
        if self._expired(stamp):
            # Lazy expiry: drop the stale entry on read.
            del self._data[thread_id]
            return None
        # A successful read refreshes LRU order (protects from eviction).
        self._data.move_to_end(thread_id)
        return ckpt

    def purge_expired(self) -> list[str]:
        expired = [tid for tid, (stamp, _) in self._data.items() if self._expired(stamp)]
        for tid in expired:
            del self._data[tid]
        return expired

    def keys_in_lru_order(self) -> list[str]:
        return list(self._data.keys())


class _FakeClock:
    """Deterministic clock: advance() instead of time.sleep()."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _ckpt(tid: str, step: int = 1) -> Checkpoint:
    return Checkpoint(thread_id=tid, step=step, state={"messages": [], "metadata": {}})


def medium_ex1_lru_ttl_store() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: LRU + TTL session store with eviction")
    print("=" * 60)

    clock = _FakeClock(1000.0)
    store = LRUTTLSessionStore(capacity=3, ttl=60.0, clock=clock)

    # Fill the cache: A, B, C (A is now LRU, C is most-recent).
    assert store.save(_ckpt("A")) == []
    assert store.save(_ckpt("B")) == []
    assert store.save(_ckpt("C")) == []
    print(f"\n  after save A,B,C   -> LRU order: {store.keys_in_lru_order()}")

    # Touch A: it becomes most-recently-used, so B is now the true LRU.
    assert store.load("A") is not None
    print(f"  after load(A)      -> LRU order: {store.keys_in_lru_order()}")

    # Insert D -> capacity exceeded -> evict the LRU, which is B (NOT A).
    evicted = store.save(_ckpt("D"))
    print(f"  after save D       -> evicted: {evicted}  LRU order: {store.keys_in_lru_order()}")
    assert evicted == ["B"], f"expected B evicted (true LRU), got {evicted}"
    assert store.load("A") is not None, "A was protected by the recent load"
    assert store.load("B") is None, "B should have been evicted"

    # --- TTL expiry (deterministic via fake clock) ---
    # Current entries: A, C, D (all stamped near t=1000). Advance past the TTL.
    clock.advance(61.0)
    assert store.load("A") is None, "A must be expired by TTL"
    print(f"\n  advanced clock +61s -> load(A) expired = {store.load('A')}")

    # Re-add a fresh entry, advance only a little, and purge: only stale ones go.
    store.save(_ckpt("E"))            # E stamped at t=1061
    clock.advance(30.0)              # t=1091 -> E age 30 < 60 (alive), C/D age 91 (expired)
    purged = sorted(store.purge_expired())
    print(f"  purge_expired       -> {purged}  remaining: {store.keys_in_lru_order()}")
    assert "E" not in purged, "E is still within TTL"
    assert set(purged) == {"C", "D"}, f"expected C,D purged, got {purged}"
    assert store.keys_in_lru_order() == ["E"]

    print("\n  PASS -- LRU order correct (B evicted before A), TTL expiry deterministic.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Consistent-hash router with session affinity
# ==========================================================================

class ConsistentHashRouter:
    """
    Consistent-hash ring for session affinity (theory 4.1).

    Each physical node owns `vnodes` virtual points on the ring. A thread_id is
    hashed and routed to the first vnode clockwise. Adding a node only remaps the
    keys that fall into the new node's arcs -- far fewer than a naive hash % N.
    Uses hashlib (stable across runs), NOT the builtin hash (salted per process).
    """

    def __init__(self, nodes: list[str], vnodes: int = 100) -> None:
        self.vnodes = vnodes
        self._ring: dict[int, str] = {}      # position -> node
        self._sorted: list[int] = []          # sorted positions for bisect
        for n in nodes:
            self._add_to_ring(n)

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _add_to_ring(self, node: str) -> None:
        for i in range(self.vnodes):
            pos = self._hash(f"{node}#{i}")
            self._ring[pos] = node
        self._sorted = sorted(self._ring)

    def add_node(self, node: str) -> None:
        self._add_to_ring(node)

    def remove_node(self, node: str) -> None:
        self._ring = {p: n for p, n in self._ring.items() if n != node}
        self._sorted = sorted(self._ring)

    def route(self, thread_id: str) -> str:
        if not self._sorted:
            raise RuntimeError("empty ring")
        h = self._hash(thread_id)
        idx = bisect.bisect_right(self._sorted, h)
        if idx == len(self._sorted):
            idx = 0  # wrap-around to the first vnode
        return self._ring[self._sorted[idx]]


def medium_ex2_consistent_hash() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Consistent-hash router with session affinity")
    print("=" * 60)

    router = ConsistentHashRouter(["node-A", "node-B", "node-C"], vnodes=150)
    ids = [f"thread-{i}" for i in range(1000)]

    # Initial assignment.
    assign = {tid: router.route(tid) for tid in ids}
    assert all(v in {"node-A", "node-B", "node-C"} for v in assign.values())

    # Affinity: routing the same ids again is perfectly stable (deterministic).
    assert all(router.route(tid) == assign[tid] for tid in ids), "routing must be stable"
    print(f"\n  affinity check     -> 1000/1000 ids map to the same node on re-route")

    # Add a 4th node; measure how many keys move.
    router.add_node("node-D")
    assign2 = {tid: router.route(tid) for tid in ids}
    moved = sum(1 for tid in ids if assign[tid] != assign2[tid])
    frac = moved / len(ids)
    print(f"  add node-D         -> consistent-hash remapped {moved}/1000 = {frac:.1%}")

    # Naive hash % N baseline: from 3 to 4 buckets remaps the vast majority.
    def naive(tid: str, n: int) -> int:
        return int(hashlib.md5(tid.encode()).hexdigest(), 16) % n
    naive_moved = sum(1 for tid in ids if naive(tid, 3) != naive(tid, 4))
    print(f"  naive hash % N     -> remapped {naive_moved}/1000 = {naive_moved/len(ids):.1%}")

    assert frac < 0.5, f"consistent hash should remap < 50%, got {frac:.1%}"
    assert moved < naive_moved, "consistent hash must remap strictly fewer keys than hash % N"

    # Keys that did NOT move keep their original owner (preserved affinity).
    for tid in ids:
        if assign[tid] != "node-D" and assign2[tid] != "node-D":
            assert assign[tid] == assign2[tid]

    # remove_node reroutes only the removed node's sessions to live nodes.
    router.remove_node("node-D")
    assign3 = {tid: router.route(tid) for tid in ids}
    assert all(v != "node-D" for v in assign3.values())
    # Sessions that were on A/B/C before adding D return to their original owner.
    for tid in ids:
        if assign[tid] != "node-D":
            assert assign3[tid] == assign[tid]
    print(f"  remove node-D      -> all sessions back on live nodes, affinity restored")

    print("\n  PASS -- stable affinity; add_node remaps minimally (< naive hash % N).\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Checkpoint / restore: resume after worker crash
# ==========================================================================

class WorkerCrash(RuntimeError):
    """Simulated worker crash raised mid-turn, BEFORE the checkpoint is saved."""


class ResilientWorker:
    """
    Stateless worker over a shared checkpointer.

    Atomicity rule: we only call store.save() AFTER the full turn is computed.
    A crash before save leaves NO partial state -- the step counter never moves
    for a turn that didn't finish, so another worker resumes cleanly.
    """

    def __init__(self, worker_id: str, store: BaseCheckpointer) -> None:
        self.worker_id = worker_id
        self.store = store

    def _get_or_create(self, thread_id: str) -> Checkpoint:
        existing = self.store.load(thread_id)
        if existing is not None:
            return existing
        return Checkpoint(thread_id=thread_id, step=0, state={"messages": []})

    def process_turn(self, thread_id: str, user_message: str,
                     crash_after_compute: bool = False) -> str:
        ckpt = self._get_or_create(thread_id)
        # Work on a fresh state copy so a crash cannot mutate the stored object.
        messages = list(ckpt.state.get("messages", []))
        messages.append({"role": "user", "content": user_message})
        reply = f"[{self.worker_id}] turn#{ckpt.step + 1}: ack '{user_message}'"
        messages.append({"role": "assistant", "content": reply, "by": self.worker_id})

        if crash_after_compute:
            # Crash BEFORE persisting -> nothing is saved, step stays put.
            raise WorkerCrash(f"{self.worker_id} died before saving turn")

        self.store.save(Checkpoint(
            thread_id=thread_id,
            step=ckpt.step + 1,
            state={"messages": messages},
        ))
        return reply


def medium_ex3_checkpoint_restore() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Checkpoint / restore -- resume after worker crash")
    print("=" * 60)

    store = InMemoryCheckpointer()      # shared, durable across worker objects
    tid = "thread-42"

    w1 = ResilientWorker("W1", store)
    w1.process_turn(tid, "hello")
    w1.process_turn(tid, "how are you?")
    assert store.load(tid).step == 2
    print(f"\n  W1 processed 2 turns -> step = {store.load(tid).step}")

    # 3rd turn crashes before save: state must be untouched.
    crashed = False
    try:
        w1.process_turn(tid, "this one crashes", crash_after_compute=True)
    except WorkerCrash as e:
        crashed = True
        print(f"  W1 crashed mid-turn  -> {e}")
    assert crashed, "WorkerCrash should have been raised"
    assert store.load(tid).step == 2, "crashed turn must NOT advance the step (no ghost turn)"
    assert len(store.load(tid).state["messages"]) == 4, "no partial messages persisted"

    # A different worker (same store) resumes from the last durable checkpoint.
    w2 = ResilientWorker("W2", store)
    reply = w2.process_turn(tid, "this one crashes")   # retry the lost turn
    print(f"  W2 resumed session   -> '{reply}'")

    final = store.load(tid)
    assert final.step == 3, f"expected step 3 after resume, got {final.step}"
    msgs = final.state["messages"]
    assert len(msgs) == 6, f"expected 6 messages (3 turns), got {len(msgs)}"
    # No duplicate of the crashed turn: only ONE 'this one crashes' user message.
    crash_count = sum(1 for m in msgs if m.get("content") == "this one crashes" and m["role"] == "user")
    assert crash_count == 1, f"crashed turn must not be duplicated (got {crash_count})"
    # The last assistant turn was handled by W2 (proof of stateless handoff).
    assert msgs[-1]["by"] == "W2", "last turn must be served by W2"
    print(f"  final state          -> step={final.step}, msgs={len(msgs)}, last_by={msgs[-1]['by']}")

    print("\n  PASS -- crash lost no turn; W2 resumed cleanly with no duplication.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 25 MEDIUM Solutions -- Serving Stateful Agents & Sessions")
    print("#" * 60)

    medium_ex1_lru_ttl_store()
    medium_ex2_consistent_hash()
    medium_ex3_checkpoint_restore()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")

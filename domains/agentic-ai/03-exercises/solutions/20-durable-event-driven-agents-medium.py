"""
Solutions -- Day 20 (MEDIUM): Durable & event-driven agents

Contains solutions for:
  - Medium Ex 1: Event-sourced state rebuilt by replay, with snapshot + tail
                 replay proven equal to full replay (event sourcing core)
  - Medium Ex 2: Crash-and-resume harness with a checkpoint store -- committed
                 steps are memoized, not re-executed (exactly-once side effects)
  - Medium Ex 3: Idempotency layer over an at-least-once channel -- duplicate
                 deliveries are deduped by event_id (each effect applied once)

Self-contained: pure-stdlib, deterministic, runs OFFLINE with zero
dependencies (no langgraph, no API key, no network, no external server).

Run:  python 03-exercises/solutions/20-durable-event-driven-agents-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import copy
from typing import Any, Callable


# ==========================================================================
# MEDIUM EXERCISE 1 -- Event-sourced state rebuilt by replay (+ snapshot)
# ==========================================================================

# Event types (cours section 2.1).
ITEM_ADDED = "ITEM_ADDED"
ITEM_REMOVED = "ITEM_REMOVED"
CART_CLEARED = "CART_CLEARED"


def empty_state() -> dict:
    """The genesis state -- every replay starts from here."""
    return {"items": {}, "version": 0}


def apply(state: dict, event: dict) -> dict:
    """
    Pure reducer: state + event -> new state. NEVER mutates `state` in place.

    This is the only place the cart state changes; the outside world only
    appends events, never touches the state directly (event sourcing).
    """
    new = {"items": dict(state["items"]), "version": state["version"] + 1}
    etype, payload = event["type"], event.get("payload", {})

    if etype == ITEM_ADDED:
        sku, qty = payload["sku"], payload["qty"]
        new["items"][sku] = new["items"].get(sku, 0) + qty
    elif etype == ITEM_REMOVED:
        sku, qty = payload["sku"], payload["qty"]
        remaining = new["items"].get(sku, 0) - qty
        if remaining > 0:
            new["items"][sku] = remaining
        else:
            new["items"].pop(sku, None)  # qty hits 0 -> sku disappears
    elif etype == CART_CLEARED:
        new["items"] = {}
    else:
        raise ValueError(f"Unknown event type: {etype}")
    return new


def replay(events: list[dict]) -> dict:
    """Reconstruct state from scratch by folding every event in order."""
    state = empty_state()
    for ev in events:
        state = apply(state, ev)
    return state


def replay_from(snapshot: dict, tail_events: list[dict]) -> dict:
    """Resume from a snapshot's state and apply only the tail of the log."""
    state = copy.deepcopy(snapshot["state"])
    for ev in tail_events:
        state = apply(state, ev)
    return state


def _ev(etype: str, **payload) -> dict:
    return {"type": etype, "payload": payload}


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Event-sourced state rebuilt by replay (+ snapshot)")
    print("=" * 70)

    events = [
        _ev(ITEM_ADDED, sku="A", qty=3),
        _ev(ITEM_ADDED, sku="B", qty=2),
        _ev(ITEM_REMOVED, sku="A", qty=1),   # A: 3 -> 2
        _ev(ITEM_ADDED, sku="C", qty=5),
        _ev(ITEM_REMOVED, sku="B", qty=2),   # B: 2 -> 0 -> disappears
        _ev(ITEM_ADDED, sku="A", qty=4),     # A: 2 -> 6
    ]

    full = replay(events)
    print(f"  full replay state: {full}")

    # version == number of events applied.
    assert full["version"] == len(events), full["version"]
    # B dropped to 0 and disappeared; A and C remain.
    assert "B" not in full["items"], full["items"]
    assert full["items"] == {"A": 6, "C": 5}, full["items"]

    # Snapshot at the middle, then tail-replay the rest.
    k = 3
    snapshot = {"state": replay(events[:k]), "at": k}
    tail = replay_from(snapshot, events[k:])
    print(f"  snapshot@{k} state: {snapshot['state']}")
    print(f"  snapshot + tail:    {tail}")

    # snapshot + tail must equal a full replay (the durable-restart invariant).
    assert tail == full, (tail, full)

    # Determinism: replaying the same log twice yields identical state.
    assert replay(events) == replay(events)

    # apply() did not mutate the caller's state (purity check).
    base = empty_state()
    _ = apply(base, _ev(ITEM_ADDED, sku="X", qty=1))
    assert base == {"items": {}, "version": 0}, "apply() must not mutate input"

    print("[Verification] PASS -- replay deterministic, snapshot+tail == full replay")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Crash-and-resume harness with checkpointing
# ==========================================================================

# Side-effect counters: prove each step runs for real exactly once.
SIDE_EFFECTS: dict[str, int] = {}


def _do_step(name: str) -> Callable[[], str]:
    """Build a step function that records its real execution and is deterministic."""
    def fn() -> str:
        SIDE_EFFECTS[name] = SIDE_EFFECTS.get(name, 0) + 1
        return f"result_of_{name}"
    return fn


class CheckpointStore:
    """
    In-memory durable checkpoint. In production this would be a DB row / KV
    entry flushed to disk; here a dict is enough to model commit + restart.
    """

    def __init__(self) -> None:
        self.done: dict[str, str] = {}   # step name -> committed result
        self.next_index: int = 0         # index of the next step to run

    def is_done(self, name: str) -> bool:
        return name in self.done

    def commit(self, index: int, name: str, result: str) -> None:
        # Persist the result, THEN advance the cursor (commit point).
        self.done[name] = result
        self.next_index = index + 1


def run_workflow(steps: list[tuple[str, Callable[[], str]]],
                 store: CheckpointStore,
                 crash_after: str | None = None) -> dict[str, str]:
    """
    Run a linear workflow with checkpoint-based resume.

    - If a step's result is already committed, memoize it (no re-execution).
    - Otherwise run it, commit immediately, then move on.
    - `crash_after` raises AFTER the named step has been committed (crash
      happens BETWEEN steps, never mid-commit).
    """
    results: dict[str, str] = dict(store.done)  # carry over committed results
    for index, (name, fn) in enumerate(steps):
        if store.is_done(name):
            # Memoized from the checkpoint -- do NOT re-execute the side effect.
            results[name] = store.done[name]
            continue
        result = fn()
        store.commit(index, name, result)        # durable commit
        results[name] = result
        if crash_after == name:
            raise RuntimeError(f"CRASH after {name}")
    return results


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Crash-and-resume harness (exactly-once side effects)")
    print("=" * 70)

    def build_steps() -> list[tuple[str, Callable[[], str]]]:
        return [(n, _do_step(n)) for n in ("step_a", "step_b", "step_c", "step_d")]

    # --- Reference run on a fresh store (no crash) to know the target output ---
    SIDE_EFFECTS.clear()
    ref_store = CheckpointStore()
    reference = run_workflow(build_steps(), ref_store)
    print(f"  reference (no crash): {reference}")

    # --- Crash run, then resume on the SAME store ---
    SIDE_EFFECTS.clear()
    store = CheckpointStore()

    # Run 1: crashes after step_b.
    try:
        run_workflow(build_steps(), store, crash_after="step_b")
        raise AssertionError("expected the simulated crash")
    except RuntimeError as e:
        print(f"  run 1 crashed: {e} | committed so far: {list(store.done)}")
    assert SIDE_EFFECTS == {"step_a": 1, "step_b": 1}, SIDE_EFFECTS

    # Run 2: resume on the same store, no crash -> should finish.
    final = run_workflow(build_steps(), store)
    print(f"  run 2 resumed -> final: {final}")
    print(f"  side-effect counts: {SIDE_EFFECTS}")

    # Exactly-once: every side effect ran exactly once across crash + resume.
    assert all(c == 1 for c in SIDE_EFFECTS.values()), SIDE_EFFECTS
    assert set(SIDE_EFFECTS) == {"step_a", "step_b", "step_c", "step_d"}
    # Run 2 re-executed ONLY the remaining steps (a/b stayed at 1).
    assert SIDE_EFFECTS["step_a"] == 1 and SIDE_EFFECTS["step_b"] == 1
    # Resumed result identical to the reference run.
    assert final == reference, (final, reference)

    print("[Verification] PASS -- committed steps memoized, exactly-once on resume")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Idempotency layer over an at-least-once channel
# ==========================================================================

class IdempotentConsumer:
    """
    Dedupe deliveries by event_id (the idempotency key, cours section 2.3).

    An at-least-once channel may deliver the same event several times; this
    consumer guarantees the business handler applies each event_id once,
    giving the consumer an exactly-once *view*.
    """

    def __init__(self, handler: Callable[[dict], None]) -> None:
        self.handler = handler
        self.processed_ids: set[str] = set()  # the idempotency store
        self.handler_calls = 0

    def handle(self, event: dict) -> dict:
        eid = event["event_id"]
        if eid in self.processed_ids:
            return {"status": "duplicate", "event_id": eid}
        # First time we see this id: apply the side effect, then record it.
        self.handler(event)
        self.processed_ids.add(eid)
        self.handler_calls += 1
        return {"status": "applied", "event_id": eid}


def deliver(consumer: IdempotentConsumer, events: list[dict]) -> list[dict]:
    """Simulate an at-least-once channel: just pass each delivery through."""
    return [consumer.handle(ev) for ev in events]


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Idempotency layer (at-least-once -> exactly-once view)")
    print("=" * 70)

    # Business state: a running balance credited by each distinct event.
    balance = {"value": 0}

    def credit(event: dict) -> None:
        balance["value"] += event["payload"]["amount"]

    consumer = IdempotentConsumer(credit)

    # 3 distinct events, each worth +10/+20/+30. The at-least-once channel
    # redelivers some ids and interleaves the order.
    e1 = {"event_id": "id1", "type": "credit", "payload": {"amount": 10}}
    e2 = {"event_id": "id2", "type": "credit", "payload": {"amount": 20}}
    e3 = {"event_id": "id3", "type": "credit", "payload": {"amount": 30}}

    # Interleaved deliveries with duplicates of id1 (x2 extra) and id2 (x1 extra).
    deliveries = [e1, e2, e1, e3, e2, e1]
    statuses = deliver(consumer, deliveries)
    outcomes = [s["status"] for s in statuses]
    print(f"  deliveries: {[e['event_id'] for e in deliveries]}")
    print(f"  outcomes:   {outcomes}")
    print(f"  balance:    {balance['value']} | handler calls: {consumer.handler_calls}")

    # Final state == each distinct id applied once: 10 + 20 + 30 = 60.
    assert balance["value"] == 60, balance["value"]
    # Handler called once per DISTINCT id, not per delivery.
    assert consumer.handler_calls == 3, consumer.handler_calls
    # First sight of each id -> applied; redeliveries -> duplicate.
    assert outcomes == ["applied", "applied", "duplicate",
                        "applied", "duplicate", "duplicate"], outcomes

    # Robustness: a different interleaving yields the same final state.
    balance2 = {"value": 0}
    consumer2 = IdempotentConsumer(lambda ev: balance2.__setitem__(
        "value", balance2["value"] + ev["payload"]["amount"]))
    deliver(consumer2, [e1, e2, e3])         # clean once-each delivery
    assert balance2["value"] == balance["value"], (balance2, balance)
    assert consumer2.handler_calls == 3

    print("[Verification] PASS -- duplicates deduped, each effect applied once")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 20 MEDIUM Solutions -- Durable & event-driven agents")
    print("  (stdlib only -- runs offline, no API key, no network)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

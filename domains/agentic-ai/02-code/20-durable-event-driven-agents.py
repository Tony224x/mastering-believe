"""
Day 20 -- Durable & event-driven agents: survive crashes, replay history, HITL advanced.

Demonstrates (stdlib only, no API key, no external server):
  1. DurableEventLog     -- append-only JSON-lines event log on disk
  2. DurableEngine       -- mini workflow engine: run activities, journal results,
                            replay from log to skip already-done activities
  3. Crash simulation    -- workflow crashes mid-way, engine restarts and resumes
                            exactly where it left off (already-done activities NOT
                            re-executed -- proved by counters)
  4. EventBus            -- in-process pub/sub: publish/subscribe/unsubscribe
  5. HITL interrupt/resume -- workflow pauses, external caller edits state,
                              workflow resumes with updated state

Run:
    python domains/agentic-ai/02-code/20-durable-event-driven-agents.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ===========================================================================
# PART 1 -- DURABLE EVENT LOG
# ===========================================================================

class DurableEventLog:
    """
    Append-only event journal stored as JSON lines.

    Each line is a JSON object:
        {"seq": int, "event": str, "payload": dict, "prev_hash": str}

    The `prev_hash` field chains entries: sha256 of the raw previous line.
    This makes tampering detectable (same principle as J13 audit log, but
    here used to guarantee replay fidelity, not just tamper-evidence).

    Key guarantees:
        - Append-only: lines are never deleted or overwritten.
        - Deterministic replay: reading lines in order reconstructs state.
    """

    GENESIS_HASH = "0" * 64  # hash for the first entry (no predecessor)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Open in append+read mode; create if absent.
        self._fp = self.path.open("a+", encoding="utf-8")
        self._seq = self._count_existing_lines()
        self._last_raw: str = self._last_line_raw()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def append(self, event: str, payload: dict) -> dict:
        """Write a new event to the log. Returns the record written."""
        prev_hash = (
            hashlib.sha256(self._last_raw.encode()).hexdigest()
            if self._last_raw
            else self.GENESIS_HASH
        )
        record = {
            "seq": self._seq,
            "event": event,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        raw = json.dumps(record, sort_keys=True)
        self._fp.write(raw + "\n")
        self._fp.flush()
        # Update internal state.
        self._seq += 1
        self._last_raw = raw
        return record

    def read_all(self) -> list[dict]:
        """Return every record in the log, in order."""
        self._fp.seek(0)
        records = []
        for line in self._fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def close(self) -> None:
        self._fp.close()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _count_existing_lines(self) -> int:
        self._fp.seek(0)
        return sum(1 for ln in self._fp if ln.strip())

    def _last_line_raw(self) -> str:
        self._fp.seek(0)
        last = ""
        for ln in self._fp:
            if ln.strip():
                last = ln.strip()
        return last


# ===========================================================================
# PART 2 -- DURABLE WORKFLOW ENGINE
# ===========================================================================

class ActivityAlreadyDoneError(Exception):
    """Raised internally to signal that an activity result is in the log."""


class DurableEngine:
    """
    Mini Temporal-like durable execution engine.

    Concept:
        - A *workflow* is a Python callable that calls `engine.run_activity(...)`.
        - An *activity* is a Python callable that does real I/O (mocked here).
        - The engine journals ACTIVITY_STARTED / ACTIVITY_COMPLETED events.
        - On replay (engine.replay()), completed activities are NOT re-executed:
          the cached result is returned directly from the log.

    This simulates what Temporal does server-side, in a single process using
    a JSON-lines file as the event store.
    """

    WORKFLOW_STARTED = "WORKFLOW_STARTED"
    WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
    ACTIVITY_STARTED = "ACTIVITY_STARTED"
    ACTIVITY_COMPLETED = "ACTIVITY_COMPLETED"
    HITL_INTERRUPTED = "HITL_INTERRUPTED"
    HITL_RESUMED = "HITL_RESUMED"

    def __init__(self, log: DurableEventLog) -> None:
        self.log = log
        # activity_name -> result cached from log (populated during replay).
        self._cached_results: dict[str, Any] = {}
        # Human edits injected before resume.
        self._state_edits: dict[str, Any] = {}
        # Shared mutable state passed through the workflow.
        self.state: dict[str, Any] = {}
        # Execution counters -- only incremented on real (non-replayed) runs.
        self.real_executions: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Replay: load completed activities from the log
    # -----------------------------------------------------------------------

    def replay(self) -> None:
        """
        Reread the event log and cache completed activity results.

        After replay(), run_activity() will skip activities whose
        ACTIVITY_COMPLETED event already exists in the log.
        """
        for record in self.log.read_all():
            if record["event"] == self.ACTIVITY_COMPLETED:
                name = record["payload"]["name"]
                self._cached_results[name] = record["payload"]["result"]
            elif record["event"] == self.HITL_RESUMED:
                # Restore any state edits that were applied at resume time.
                self._state_edits.update(record["payload"].get("edits", {}))

    # -----------------------------------------------------------------------
    # Core: run an activity (or return cached result on replay)
    # -----------------------------------------------------------------------

    def run_activity(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute an activity or return its cached result if already done.

        Workflow code calls this instead of calling `fn` directly.
        This is the boundary between deterministic workflow and non-deterministic I/O.
        """
        if name in self._cached_results:
            # Already done in a previous run -- return cached result WITHOUT calling fn.
            print(f"  [REPLAY] Activity '{name}' skipped -- result from log: {self._cached_results[name]!r}")
            return self._cached_results[name]

        # First real execution of this activity.
        self.log.append(self.ACTIVITY_STARTED, {"name": name, "args": list(args)})
        # Track how many times each activity runs for real (should be 1).
        self.real_executions[name] = self.real_executions.get(name, 0) + 1

        result = fn(*args, **kwargs)

        self.log.append(self.ACTIVITY_COMPLETED, {"name": name, "result": result})
        # Cache it so sub-calls within the same run are also idempotent.
        self._cached_results[name] = result
        print(f"  [EXEC]   Activity '{name}' ran for real -> {result!r}")
        return result

    # -----------------------------------------------------------------------
    # HITL: interrupt and resume
    # -----------------------------------------------------------------------

    def interrupt(self, question: str, context: dict) -> dict:
        """
        Pause the workflow. Store the interrupt in the log.
        The caller (or an external system) must call `resume()` to continue.
        Returns the edits applied during the pause (empty on first interrupt).
        """
        # Idempotent on replay: when the workflow re-runs from the top after a
        # resume, this interrupt is already in the log -- don't append it again
        # (re-appending would place HITL_INTERRUPTED after its own HITL_RESUMED
        # and break the "read the log in order to reconstruct state" invariant).
        already = any(
            r["event"] == self.HITL_INTERRUPTED and r["payload"].get("question") == question
            for r in self.log.read_all()
        )
        if not already:
            self.log.append(self.HITL_INTERRUPTED, {"question": question, "context": context})
        print(f"\n  [HITL] INTERRUPTED -- waiting for human decision")
        print(f"         Question : {question}")
        print(f"         Context  : {context}")
        # Return state edits that were queued before this resume (may be empty).
        return dict(self._state_edits)

    def resume(self, edits: dict | None = None) -> None:
        """
        Resume the workflow after an interrupt.
        Optionally pass state edits (the human's corrections).
        """
        edits = edits or {}
        self._state_edits.update(edits)
        self.log.append(self.HITL_RESUMED, {"edits": edits})
        print(f"  [HITL] RESUMED with edits: {edits}")

    # -----------------------------------------------------------------------
    # Workflow lifecycle
    # -----------------------------------------------------------------------

    def start_workflow(self, workflow_id: str) -> None:
        """Log that a new workflow execution has started (idempotent on replay)."""
        # On resume the workflow re-runs from the top; only one WORKFLOW_STARTED
        # should ever exist per workflow_id, so don't re-append on replay.
        already = any(
            r["event"] == self.WORKFLOW_STARTED and r["payload"].get("workflow_id") == workflow_id
            for r in self.log.read_all()
        )
        if not already:
            self.log.append(self.WORKFLOW_STARTED, {"workflow_id": workflow_id})

    def complete_workflow(self, result: Any) -> None:
        """Log successful completion."""
        self.log.append(self.WORKFLOW_COMPLETED, {"result": result})


# ===========================================================================
# PART 3 -- SAMPLE ACTIVITIES (the non-deterministic I/O layer)
# ===========================================================================

# These counters let us verify that activities are NOT re-executed on replay.
ACTIVITY_CALL_COUNTS: dict[str, int] = {}


def _track(name: str) -> None:
    ACTIVITY_CALL_COUNTS[name] = ACTIVITY_CALL_COUNTS.get(name, 0) + 1


def activity_summarize_order(order_id: str) -> str:
    """Mock LLM call: summarize an order. Non-deterministic in production."""
    _track("summarize_order")
    # In real code: return llm.invoke(f"Summarize order {order_id}")
    return f"Order {order_id}: 3 items, total $149.90, expedited shipping"


def activity_charge_payment(order_id: str) -> str:
    """Mock payment API call. Must be idempotent (idempotency key = order_id)."""
    _track("charge_payment")
    # In real code: stripe.charge(amount=..., idempotency_key=order_id)
    return f"payment_ok:{order_id}:txn_{order_id[:4]}"


def activity_send_email(order_id: str, summary: str) -> str:
    """Mock email send. Idempotent via idempotency key."""
    _track("send_email")
    # In real code: sendgrid.send(to=..., body=summary, idempotency_key=...)
    return f"email_sent:{order_id}"


def activity_update_inventory(order_id: str) -> str:
    """Mock inventory update. Idempotent via upsert."""
    _track("update_inventory")
    return f"inventory_updated:{order_id}"


# ===========================================================================
# PART 4 -- THE WORKFLOW (deterministic orchestrator)
# ===========================================================================

def order_workflow(engine: DurableEngine, order_id: str, simulate_crash_after: str | None = None) -> str:
    """
    A durable order-processing workflow.

    Rules for workflow code:
        - No direct I/O (no HTTP, no file writes, no random).
        - All I/O goes through engine.run_activity().
        - Idempotent: can be replayed from the beginning.

    `simulate_crash_after` triggers an exception after the named activity,
    simulating a process crash.
    """
    engine.start_workflow(order_id)

    # Step 1: Summarize order via LLM.
    summary = engine.run_activity("summarize_order", activity_summarize_order, order_id)

    if simulate_crash_after == "summarize_order":
        raise RuntimeError("SIMULATED CRASH after summarize_order")

    # Step 2: Charge payment.
    payment_ref = engine.run_activity("charge_payment", activity_charge_payment, order_id)

    if simulate_crash_after == "charge_payment":
        raise RuntimeError("SIMULATED CRASH after charge_payment")

    # Step 3: HITL -- human reviews before email is sent.
    edits = engine.interrupt(
        question="Approve email to customer?",
        context={"summary": summary, "payment": payment_ref},
    )
    # Apply any state corrections the human made.
    if "summary" in edits:
        summary = edits["summary"]
        print(f"  [HITL] Human corrected summary -> {summary!r}")

    # Step 4: Send email.
    email_ref = engine.run_activity("send_email", activity_send_email, order_id, summary)

    if simulate_crash_after == "send_email":
        raise RuntimeError("SIMULATED CRASH after send_email")

    # Step 5: Update inventory.
    inv_ref = engine.run_activity("update_inventory", activity_update_inventory, order_id)

    result = {
        "order_id": order_id,
        "summary": summary,
        "payment": payment_ref,
        "email": email_ref,
        "inventory": inv_ref,
    }
    engine.complete_workflow(result)
    return str(result)


# ===========================================================================
# PART 5 -- EVENT BUS (pub/sub)
# ===========================================================================

class EventBus:
    """
    Simple in-process pub/sub bus.

    Agents subscribe to topic strings. When an event is published on a topic,
    all subscribed handlers are called in registration order.

    In production this would be backed by Kafka, RabbitMQ, or AWS SQS.
    The interface is the same; only the transport changes.
    """

    def __init__(self) -> None:
        # topic -> list of (handler_id, callable)
        self._subscribers: dict[str, list[tuple[str, Callable]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> str:
        """Register a handler for a topic. Returns a subscription ID."""
        sub_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._subscribers.setdefault(topic, []).append((sub_id, handler))
        return sub_id

    def unsubscribe(self, topic: str, sub_id: str) -> None:
        """Remove a handler by subscription ID."""
        with self._lock:
            if topic in self._subscribers:
                self._subscribers[topic] = [
                    (sid, h) for sid, h in self._subscribers[topic] if sid != sub_id
                ]

    def publish(self, topic: str, event: dict) -> int:
        """Publish an event. Returns number of handlers called."""
        event_with_meta = {**event, "_topic": topic, "_id": str(uuid.uuid4())[:8]}
        handlers = []
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for _, handler in handlers:
            handler(event_with_meta)
        return len(handlers)


# ===========================================================================
# PART 6 -- DEMO
# ===========================================================================

def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def demo_crash_and_resume() -> None:
    """
    Demonstrates the core durable execution feature:
        1. Run the workflow until it crashes after 'charge_payment'.
        2. Restart the engine on the same log file.
        3. Replay: engine sees summarize_order + charge_payment already done.
        4. Continue from send_email (HITL first, then email, then inventory).
        5. Prove that summarize_order and charge_payment were NOT re-executed.
    """
    separator("DEMO 1 -- Durable execution: crash & resume")

    order_id = "ORD-42"
    log_path = Path(tempfile.mkdtemp()) / "workflow.jsonl"
    print(f"\nEvent log: {log_path}")

    # ---- RUN 1: crash after charge_payment ----
    print("\n--- Run 1: normal start, CRASH after charge_payment ---")
    ACTIVITY_CALL_COUNTS.clear()

    engine1 = DurableEngine(DurableEventLog(log_path))
    try:
        order_workflow(engine1, order_id, simulate_crash_after="charge_payment")
    except RuntimeError as exc:
        print(f"\n  *** PROCESS CRASH: {exc} ***")

    engine1.log.close()

    counts_after_run1 = dict(ACTIVITY_CALL_COUNTS)
    print(f"\n  Activity real executions after Run 1: {counts_after_run1}")
    # Expected: summarize_order=1, charge_payment=1, send_email=0, update_inventory=0

    # ---- RUN 2: restart, replay, continue ----
    print("\n--- Run 2: restart, replay log, resume from send_email ---")

    engine2 = DurableEngine(DurableEventLog(log_path))
    engine2.replay()  # Load cached results from the log.

    print(f"\n  Cached activity results loaded from log: {list(engine2._cached_results.keys())}")

    # Simulate the human responding to the HITL interrupt, with a state edit.
    engine2.resume(edits={"summary": "CORRECTED BY HUMAN: Order ORD-42 with priority flag"})

    # Re-run the full workflow -- activities already in the log will be skipped.
    result = order_workflow(engine2, order_id, simulate_crash_after=None)
    engine2.log.close()

    counts_after_run2 = dict(ACTIVITY_CALL_COUNTS)
    print(f"\n  Activity real executions after Run 2: {counts_after_run2}")
    # Expected: summarize_order=1 (NOT 2), charge_payment=1 (NOT 2)

    print(f"\n  Final workflow result: {result}")

    # Verification.
    assert counts_after_run2.get("summarize_order") == 1, "summarize_order must run exactly once!"
    assert counts_after_run2.get("charge_payment") == 1, "charge_payment must run exactly once!"
    assert counts_after_run2.get("send_email") == 1, "send_email must run in run 2!"
    assert counts_after_run2.get("update_inventory") == 1, "update_inventory must run in run 2!"
    print("\n  [OK] All activities ran exactly once despite crash and replay.")

    # Show the full event log.
    print("\n  Event log contents:")
    log3 = DurableEventLog(log_path)
    for rec in log3.read_all():
        print(f"    seq={rec['seq']:02d} | {rec['event']:25s} | {json.dumps(rec['payload'])[:60]}")
    log3.close()


def demo_event_bus() -> None:
    """
    Demonstrates pub/sub event-driven agent communication.
        - Two 'agents' subscribe to different topics.
        - Events are published; each agent reacts only to its topic.
        - Shows decouplage: the publisher does not know who is subscribed.
    """
    separator("DEMO 2 -- Event-driven pub/sub")

    bus = EventBus()

    # Agent A: reacts to new orders.
    received_by_a: list[dict] = []

    def agent_order_processor(event: dict) -> None:
        received_by_a.append(event)
        print(f"  [AgentA] Processing order event: order_id={event.get('order_id')}")

    # Agent B: reacts to payment alerts.
    received_by_b: list[dict] = []

    def agent_payment_monitor(event: dict) -> None:
        received_by_b.append(event)
        print(f"  [AgentB] Payment alert: amount={event.get('amount')}, status={event.get('status')}")

    # Agent C: subscribes to ALL events via a wildcard-like catch-all topic.
    received_by_c: list[dict] = []

    def agent_audit_logger(event: dict) -> None:
        received_by_c.append(event)
        print(f"  [AgentC] Audit log: topic={event.get('_topic')}, id={event.get('_id')}")

    sub_a = bus.subscribe("order.created", agent_order_processor)
    sub_b = bus.subscribe("payment.alert", agent_payment_monitor)
    bus.subscribe("order.created", agent_audit_logger)
    bus.subscribe("payment.alert", agent_audit_logger)

    print("\n  Publishing events...")
    print()

    n1 = bus.publish("order.created", {"order_id": "ORD-99", "amount": 250.0})
    n2 = bus.publish("payment.alert", {"order_id": "ORD-99", "amount": 250.0, "status": "declined"})
    n3 = bus.publish("order.created", {"order_id": "ORD-100", "amount": 45.0})

    print(f"\n  Handlers called: order.created x2 -> {n1},{n3} | payment.alert -> {n2}")
    print(f"  AgentA received {len(received_by_a)} events, AgentB {len(received_by_b)}, AgentC {len(received_by_c)}")

    # Unsubscribe AgentA, publish again -- AgentA no longer notified.
    bus.unsubscribe("order.created", sub_a)
    print("\n  [Unsubscribed AgentA] Publishing another order.created...")
    n4 = bus.publish("order.created", {"order_id": "ORD-101", "amount": 10.0})
    print(f"  Handlers called: {n4} (only AgentC now)")

    assert len(received_by_a) == 2, "AgentA should have received 2 events (before unsubscribe)"
    assert len(received_by_b) == 1
    # AgentC is still subscribed to both topics and receives all 4 publishes:
    # order.created x3 (ORD-99, ORD-100, ORD-101) + payment.alert x1 = 4 total
    assert len(received_by_c) == 4
    print("\n  [OK] Pub/sub routing correct.")


def demo_hitl_edit_state() -> None:
    """
    Demonstrates interrupt / edit-state / resume pattern:
        - Workflow runs, hits an interrupt point.
        - 'Human' inspects context and edits the state.
        - Workflow resumes with the corrected state.
    """
    separator("DEMO 3 -- HITL: interrupt / edit-state / resume")

    log_path = Path(tempfile.mkdtemp()) / "hitl_workflow.jsonl"
    ACTIVITY_CALL_COUNTS.clear()

    engine = DurableEngine(DurableEventLog(log_path))

    print("\n  Step 1: Workflow runs up to the HITL interrupt...")
    engine.start_workflow("HITL-DEMO")
    summary = engine.run_activity("summarize_order", activity_summarize_order, "ORD-HITL")
    payment = engine.run_activity("charge_payment", activity_charge_payment, "ORD-HITL")

    # Workflow pauses here, waiting for human.
    edits = engine.interrupt(
        question="Please review the summary before sending the email.",
        context={"summary": summary, "payment": payment},
    )

    # At this point in a real system: the interrupt is persisted in the log.
    # The process could crash and restart. The human is notified asynchronously.
    # Here we simulate the human responding synchronously.

    print("\n  Step 2: Human reviews and edits the state...")
    corrected_summary = "HUMAN EDIT: Order ORD-HITL confirmed, VIP customer, expedite"
    engine.resume(edits={"summary": corrected_summary})

    # Apply human edits to local state (engine.interrupt returns edits after resume).
    # In a real system the engine would re-call interrupt() which returns the edits.
    # Here we model it simply: after resume(), re-call interrupt to get the edits.
    # We already have them from the first interrupt() call above; refresh:
    edits = engine._state_edits  # edits logged at resume time

    if "summary" in edits:
        summary = edits["summary"]

    print(f"\n  Step 3: Workflow resumes with corrected summary: {summary!r}")
    email = engine.run_activity("send_email", activity_send_email, "ORD-HITL", summary)
    inv = engine.run_activity("update_inventory", activity_update_inventory, "ORD-HITL")
    engine.complete_workflow({"email": email, "inventory": inv})
    engine.log.close()

    assert "HUMAN EDIT" in summary, "Summary should contain the human's correction"
    print("\n  [OK] HITL edit-state-and-resume worked correctly.")

    print("\n  Event log:")
    log2 = DurableEventLog(log_path)
    for rec in log2.read_all():
        print(f"    seq={rec['seq']:02d} | {rec['event']:25s} | {json.dumps(rec['payload'])[:70]}")
    log2.close()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Day 20 -- Durable & event-driven agents demo")
    print("Stdlib only, no API key, no external server.\n")

    demo_crash_and_resume()
    demo_event_bus()
    demo_hitl_edit_state()

    separator("ALL DEMOS PASSED")
    print("\nKey takeaways demonstrated:")
    print("  1. Activities run ONCE despite workflow being replayed after a crash.")
    print("  2. Event log is append-only; replay reconstructs state without re-executing.")
    print("  3. Pub/sub decouples publishers from subscribers; easy to add/remove agents.")
    print("  4. HITL interrupt/resume: human edits state mid-workflow, engine continues.")

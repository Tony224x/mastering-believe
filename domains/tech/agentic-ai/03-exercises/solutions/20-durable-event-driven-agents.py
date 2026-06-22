"""
Day 20 -- Solutions to the easy exercises for durable & event-driven agents.

Imports the Day 20 code module to reuse DurableEventLog, DurableEngine, EventBus.
Run the whole file to execute all three solutions.

    python domains/tech/agentic-ai/03-exercises/solutions/20-durable-event-driven-agents.py
"""

from __future__ import annotations

import fnmatch
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Bootstrap: add 02-code to sys.path so we can import the day module.
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# importlib handles the hyphen in the module filename.
import importlib

day20 = importlib.import_module("20-durable-event-driven-agents")

DurableEventLog = day20.DurableEventLog
DurableEngine = day20.DurableEngine
EventBus = day20.EventBus


# ===========================================================================
# SOLUTION 1 -- ACTIVITY_FAILED event + replay re-execution
# ===========================================================================

class DurableEngineWithFailures(DurableEngine):
    """
    Extends DurableEngine to handle activity failures gracefully.

    New behaviour:
        - If an activity raises, log ACTIVITY_FAILED and re-raise.
        - On replay, activities listed in _failed_activities are RE-EXECUTED
          (failures are treated as transient; only successes are cached).
    """

    ACTIVITY_FAILED = "ACTIVITY_FAILED"

    def __init__(self, log: DurableEventLog) -> None:
        super().__init__(log)
        # name -> error message for activities that failed in a previous run
        self._failed_activities: dict[str, str] = {}

    # Override replay to also load failures.
    def replay(self) -> None:
        for record in self.log.read_all():
            if record["event"] == self.ACTIVITY_COMPLETED:
                name = record["payload"]["name"]
                self._cached_results[name] = record["payload"]["result"]
            elif record["event"] == self.ACTIVITY_FAILED:
                name = record["payload"]["name"]
                # Remove from cache in case a previous run had started it.
                self._cached_results.pop(name, None)
                self._failed_activities[name] = record["payload"]["error"]
            elif record["event"] == self.HITL_RESUMED:
                self._state_edits.update(record["payload"].get("edits", {}))

    # Override run_activity to catch exceptions and journal ACTIVITY_FAILED.
    def run_activity(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if name in self._cached_results:
            print(f"  [REPLAY-OK]   Activity '{name}' skipped -- result from log")
            return self._cached_results[name]

        if name in self._failed_activities:
            print(f"  [REPLAY-FAIL] Activity '{name}' previously failed ({self._failed_activities[name]}), re-executing...")
            # Remove from failed set so we try again.
            del self._failed_activities[name]

        self.log.append(self.ACTIVITY_STARTED, {"name": name, "args": list(args)})

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            # Journal the failure, then re-raise so the workflow can decide.
            self.log.append(
                self.ACTIVITY_FAILED,
                {"name": name, "error": str(exc), "error_type": type(exc).__name__},
            )
            print(f"  [FAILED]  Activity '{name}' raised {type(exc).__name__}: {exc}")
            raise

        self.log.append(self.ACTIVITY_COMPLETED, {"name": name, "result": result})
        self._cached_results[name] = result
        print(f"  [EXEC]    Activity '{name}' completed -> {result!r}")
        return result


def solution1() -> None:
    """Exercise 1: ACTIVITY_FAILED event + replay re-execution of transient failures."""
    print("\n" + "=" * 60)
    print("  SOLUTION 1 -- ACTIVITY_FAILED + replay re-execution")
    print("=" * 60)

    log_path = Path(tempfile.mkdtemp()) / "sol1.jsonl"
    attempt_count = [0]  # mutable counter shared across calls

    def activity_flaky(attempt_count: list) -> str:
        """Fails on first call, succeeds on second."""
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            raise ValueError("Transient error on first attempt")
        return f"flaky_result_attempt_{attempt_count[0]}"

    # --- Run 1: activity_flaky fails ---
    print("\n--- Run 1: activity fails ---")
    engine1 = DurableEngineWithFailures(DurableEventLog(log_path))
    engine1.start_workflow("sol1-wf")
    try:
        engine1.run_activity("flaky_step", activity_flaky, attempt_count)
    except ValueError:
        print("  Workflow caught the failure; stopping run 1.")
    engine1.log.close()

    # --- Run 2: reload, replay, retry ---
    print("\n--- Run 2: reload, replay, re-execute flaky_step ---")
    engine2 = DurableEngineWithFailures(DurableEventLog(log_path))
    engine2.replay()
    result = engine2.run_activity("flaky_step", activity_flaky, attempt_count)
    engine2.log.close()

    # Verify log contents.
    log_check = DurableEventLog(log_path)
    events = [r["event"] for r in log_check.read_all()]
    log_check.close()

    print(f"\n  Log events: {events}")
    assert "ACTIVITY_FAILED" in events, "ACTIVITY_FAILED must appear in log"
    assert events.count("ACTIVITY_COMPLETED") == 1, "Exactly one ACTIVITY_COMPLETED"
    # ACTIVITY_FAILED must come before ACTIVITY_COMPLETED for the same step.
    assert events.index("ACTIVITY_FAILED") < events.index("ACTIVITY_COMPLETED")
    assert result.startswith("flaky_result"), f"Unexpected result: {result}"
    print("\n  [OK] Solution 1 passed.")


# ===========================================================================
# SOLUTION 2 -- EventBus with wildcard topic matching
# ===========================================================================

class WildcardEventBus(EventBus):
    """
    EventBus extended with AMQP-style wildcard topic matching:
        *  -- exactly one segment
        #  -- zero or more segments
    Segments are separated by '.'.
    """

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        """
        Match a topic against a pattern with * and # wildcards.

        Algorithm: recursive descent on segments.
            pattern segment '*' -> matches exactly one topic segment
            pattern segment '#' -> matches zero or more topic segments (greedy)
        """
        p_parts = pattern.split(".")
        t_parts = topic.split(".")

        def _match(pi: int, ti: int) -> bool:
            # Both exhausted: success.
            if pi == len(p_parts) and ti == len(t_parts):
                return True
            # Pattern exhausted but topic has more segments: failure.
            if pi == len(p_parts):
                return False
            seg = p_parts[pi]
            if seg == "#":
                # '#' matches zero or more segments: try all possibilities.
                for skip in range(len(t_parts) - ti + 1):
                    if _match(pi + 1, ti + skip):
                        return True
                return False
            elif seg == "*":
                # '*' matches exactly one segment.
                if ti >= len(t_parts):
                    return False
                return _match(pi + 1, ti + 1)
            else:
                # Literal segment: must match exactly.
                if ti >= len(t_parts) or seg != t_parts[ti]:
                    return False
                return _match(pi + 1, ti + 1)

        return _match(0, 0)

    def publish(self, topic: str, event: dict) -> int:
        """Publish event to all handlers whose pattern matches the topic."""
        import uuid as _uuid
        event_with_meta = {**event, "_topic": topic, "_id": str(_uuid.uuid4())[:8]}
        count = 0
        with self._lock:
            all_subs = list(self._subscribers.items())
        for pattern, handlers in all_subs:
            if self._matches(pattern, topic):
                for _, handler in handlers:
                    handler(event_with_meta)
                    count += 1
        return count


def solution2() -> None:
    """Exercise 2: EventBus with * and # wildcard topic matching."""
    print("\n" + "=" * 60)
    print("  SOLUTION 2 -- EventBus with wildcard topics")
    print("=" * 60)

    # Unit-test the matching logic first.
    M = WildcardEventBus._matches
    assert M("order.*", "order.created") is True,  "order.* should match order.created"
    assert M("order.*", "order.item.added") is False, "order.* should NOT match order.item.added"
    assert M("order.#", "order.item.added") is True,  "order.# should match order.item.added"
    assert M("order.#", "order") is True,             "order.# should match 'order' (0 extra segments)"
    assert M("#", "anything.goes.here") is True,      "# should match anything"
    assert M("payment.*", "order.created") is False,  "payment.* should NOT match order.created"
    print("  [OK] _matches() unit tests passed.")

    bus = WildcardEventBus()

    received: dict[str, list] = {"A": [], "B": [], "C": [], "all": []}

    bus.subscribe("order.*", lambda e: received["A"].append(e["_topic"]))
    bus.subscribe("order.#", lambda e: received["B"].append(e["_topic"]))
    bus.subscribe("payment.*", lambda e: received["C"].append(e["_topic"]))
    bus.subscribe("#", lambda e: received["all"].append(e["_topic"]))

    print("\n  Publishing 4 events...")
    bus.publish("order.created",   {"data": 1})
    bus.publish("order.item.added", {"data": 2})
    bus.publish("payment.failed",  {"data": 3})
    bus.publish("system.health",   {"data": 4})

    print(f"  handler_A  (order.*)    received: {received['A']}")
    print(f"  handler_B  (order.#)    received: {received['B']}")
    print(f"  handler_C  (payment.*)  received: {received['C']}")
    print(f"  handler_all (#)         received: {received['all']}")

    assert received["A"] == ["order.created"], f"A: {received['A']}"
    assert set(received["B"]) == {"order.created", "order.item.added"}, f"B: {received['B']}"
    assert received["C"] == ["payment.failed"], f"C: {received['C']}"
    assert set(received["all"]) == {"order.created", "order.item.added", "payment.failed", "system.health"}, f"all: {received['all']}"
    print("\n  [OK] Solution 2 passed.")


# ===========================================================================
# SOLUTION 3 -- Saga pattern with compensations
# ===========================================================================

@dataclass
class SagaStep:
    """A step in a Saga: an action and its compensation (rollback)."""
    name: str
    activity: Callable[..., Any]
    compensation: Callable[..., Any]


class SagaEngine:
    """
    Runs a list of SagaSteps using a DurableEngine.

    On failure at step N:
        - Executes compensations for steps N-1, N-2, ..., 0 (reverse order).
        - Journals SAGA_COMPENSATING and SAGA_FAILED events.
    """

    SAGA_COMPENSATING = "SAGA_COMPENSATING"
    SAGA_FAILED = "SAGA_FAILED"
    SAGA_COMPLETED = "SAGA_COMPLETED"

    def __init__(self, engine: DurableEngine, steps: list[SagaStep]) -> None:
        self.engine = engine
        self.steps = steps

    def run(self, args: dict) -> dict:
        completed: list[SagaStep] = []
        failed_at: str | None = None

        for step in self.steps:
            try:
                result = self.engine.run_activity(step.name, step.activity, args)
                args = {**args, step.name + "_result": result}
                completed.append(step)
            except Exception as exc:
                failed_at = step.name
                print(f"  [SAGA] Step '{step.name}' failed: {exc}")
                break

        if failed_at is None:
            self.engine.log.append(self.SAGA_COMPLETED, {"steps": [s.name for s in self.steps]})
            return {"status": "completed"}

        # Compensate in reverse order.
        print(f"  [SAGA] Compensating {len(completed)} completed steps (reverse order)...")
        self.engine.log.append(
            self.SAGA_COMPENSATING,
            {"failed_at": failed_at, "compensating": [s.name for s in reversed(completed)]},
        )
        for step in reversed(completed):
            try:
                self.engine.run_activity(
                    step.name + "_compensation",
                    step.compensation,
                    args,
                )
                print(f"  [SAGA] Compensated '{step.name}'")
            except Exception as comp_exc:
                print(f"  [SAGA] WARNING: compensation for '{step.name}' failed: {comp_exc}")

        self.engine.log.append(self.SAGA_FAILED, {"failed_at": failed_at})
        return {"status": "compensated", "failed_at": failed_at}


def solution3() -> None:
    """Exercise 3: Saga pattern with compensations on partial failure."""
    print("\n" + "=" * 60)
    print("  SOLUTION 3 -- Saga pattern with compensations")
    print("=" * 60)

    log_path = Path(tempfile.mkdtemp()) / "sol3.jsonl"

    # Track which activities actually ran.
    ran: list[str] = []

    def reserve_stock(args: dict) -> str:
        ran.append("reserve_stock")
        return "stock_reserved"

    def release_stock(args: dict) -> str:
        ran.append("release_stock")
        return "stock_released"

    def charge_payment(args: dict) -> str:
        ran.append("charge_payment")
        return "payment_charged"

    def refund_payment(args: dict) -> str:
        ran.append("refund_payment")
        return "payment_refunded"

    def ship_order(args: dict) -> str:
        ran.append("ship_order")
        raise RuntimeError("Shipping service unavailable")  # intentional failure

    def cancel_shipment(args: dict) -> str:
        ran.append("cancel_shipment")
        return "shipment_cancelled"

    steps = [
        SagaStep("reserve_stock", reserve_stock, release_stock),
        SagaStep("charge_payment", charge_payment, refund_payment),
        SagaStep("ship_order", ship_order, cancel_shipment),
    ]

    engine = DurableEngine(DurableEventLog(log_path))
    saga = SagaEngine(engine, steps)
    result = saga.run({"order_id": "ORD-SAGA"})
    engine.log.close()

    print(f"\n  Saga result: {result}")
    print(f"  Activities that ran (in order): {ran}")

    # Verify result.
    assert result["status"] == "compensated"
    assert result["failed_at"] == "ship_order"

    # Verify compensation order: refund_payment must come before release_stock.
    assert "refund_payment" in ran
    assert "release_stock" in ran
    assert ran.index("refund_payment") < ran.index("release_stock"), (
        "refund_payment should compensate before release_stock (reverse order)"
    )
    # ship_order ran (and failed); cancel_shipment should NOT have run (ship didn't complete).
    assert "cancel_shipment" not in ran, "cancel_shipment should not run if ship_order failed"

    # Verify log events.
    log_check = DurableEventLog(log_path)
    events = [r["event"] for r in log_check.read_all()]
    log_check.close()
    print(f"\n  Log events: {events}")
    assert "SAGA_COMPENSATING" in events
    assert "SAGA_FAILED" in events
    assert events.index("SAGA_COMPENSATING") < events.index("SAGA_FAILED")

    print("\n  [OK] Solution 3 passed.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Day 20 -- Solutions runner")
    print("=" * 60)

    solution1()
    solution2()
    solution3()

    print("\n" + "=" * 60)
    print("  ALL SOLUTIONS PASSED")
    print("=" * 60)

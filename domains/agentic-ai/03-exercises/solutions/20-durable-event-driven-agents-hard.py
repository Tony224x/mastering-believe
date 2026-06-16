"""
Solutions -- Day 20 (HARD): Durable & event-driven agents

Contains solutions for:
  - Hard Ex 1: WorkflowEngine with step journaling + deterministic replay.
               Completed steps are memoized; side_effect() replays captured
               non-deterministic values. Proves exactly-once side effects and
               identical replay values across a simulated mid-workflow crash.
  - Hard Ex 2: DurableSagaOrchestrator -- journaled actions/compensations,
               reverse-order compensation, crash-during-compensation resume
               (compensations idempotent), and dead-letter for non-compensable
               failures.

stdlib only, fully offline, deterministic. No langgraph, no API key, no
network. The append-only journal is modeled as an in-memory list of records
(same semantics as the JSON-lines log in 02-code/20-...py).

Run:  python 03-exercises/solutions/20-durable-event-driven-agents-hard.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ==========================================================================
# Shared: an append-only journal (in-memory, same semantics as JSON-lines)
# ==========================================================================

class Journal:
    """Append-only event log. Never deleted or overwritten -> deterministic replay."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def append(self, kind: str, **fields: Any) -> dict:
        rec = {"seq": len(self._records), "kind": kind, **fields}
        self._records.append(rec)
        return rec

    def read_all(self) -> list[dict]:
        return list(self._records)

    def kinds(self) -> list[str]:
        return [r["kind"] for r in self._records]


# ==========================================================================
# HARD EXERCISE 1 -- WorkflowEngine: step journaling + deterministic replay
# ==========================================================================

# Real-execution counters: prove memoization (no re-execution on replay).
EXEC: dict[str, int] = {}
SIDE_EFFECT_PRODUCED: dict[str, int] = {}


class WorkflowEngine:
    """
    Mini Temporal-like engine over a shared append-only journal.

    Determinism boundary:
        - engine.step(name, fn, ...)   -> journaled STEP_COMPLETED, memoized on replay
        - engine.side_effect(key, prod)-> journaled SIDE_EFFECT, REPLAYED verbatim
    The workflow itself must be pure: all non-determinism flows through these.
    """

    STEP_COMPLETED = "STEP_COMPLETED"
    SIDE_EFFECT = "SIDE_EFFECT"
    WORKFLOW_DONE = "WORKFLOW_DONE"

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        # Load already-journaled results so the next run skips them (replay).
        self._step_results: dict[str, Any] = {}
        self._side_effects: dict[str, Any] = {}
        for r in journal.read_all():
            if r["kind"] == self.STEP_COMPLETED:
                self._step_results[r["key"]] = r["value"]
            elif r["kind"] == self.SIDE_EFFECT:
                self._side_effects[r["key"]] = r["value"]

    def step(self, name: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Execute a step once; on replay return the journaled result, no re-run."""
        if name in self._step_results:
            return self._step_results[name]  # memoized -- fn NOT called
        result = fn(*args)
        self.journal.append(self.STEP_COMPLETED, key=name, value=result)
        self._step_results[name] = result
        return result

    def side_effect(self, key: str, producer: Callable[[], Any]) -> Any:
        """Capture a non-deterministic value once; replay it verbatim afterwards."""
        if key in self._side_effects:
            return self._side_effects[key]  # replayed -- producer NOT called
        value = producer()
        self.journal.append(self.SIDE_EFFECT, key=key, value=value)
        self._side_effects[key] = value
        return value

    def done(self, result: Any) -> None:
        # Idempotent: only one WORKFLOW_DONE per workflow.
        if self.WORKFLOW_DONE not in self.journal.kinds():
            self.journal.append(self.WORKFLOW_DONE, value=result)


def booking_workflow(engine: WorkflowEngine, crash_after: str | None = None) -> dict:
    """
    Deterministic booking workflow. Non-determinism (id, timestamp) is captured
    via engine.side_effect so it replays identically. Side effects via step().
    """
    # Non-deterministic values, captured once, replayed verbatim later.
    booking_id = engine.side_effect(
        "booking_id", lambda: _produce("booking_id", "BK-" + str(7 * 13 + 4)))
    created_at = engine.side_effect(
        "created_at", lambda: _produce("created_at", 1_700_000_000))

    def reserve() -> str:
        EXEC["reserve"] = EXEC.get("reserve", 0) + 1
        return f"reserved:{booking_id}"

    def pay() -> str:
        EXEC["pay"] = EXEC.get("pay", 0) + 1
        return f"paid:{booking_id}:at{created_at}"

    def confirm() -> str:
        EXEC["confirm"] = EXEC.get("confirm", 0) + 1
        return f"confirmed:{booking_id}"

    def notify() -> str:
        EXEC["notify"] = EXEC.get("notify", 0) + 1
        return f"notified:{booking_id}"

    r1 = engine.step("reserve", reserve)
    r2 = engine.step("pay", pay)
    if crash_after == "pay":
        raise RuntimeError("CRASH after pay")
    r3 = engine.step("confirm", confirm)
    r4 = engine.step("notify", notify)

    result = {"booking_id": booking_id, "created_at": created_at,
              "reserve": r1, "pay": r2, "confirm": r3, "notify": r4}
    engine.done(result)
    return result


def _produce(key: str, value: Any) -> Any:
    """Marks that a non-deterministic producer actually ran (should be once)."""
    SIDE_EFFECT_PRODUCED[key] = SIDE_EFFECT_PRODUCED.get(key, 0) + 1
    return value


def hard_ex1_durable_workflow() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Durable WorkflowEngine: journaling + deterministic replay")
    print("=" * 70)

    # --- Reference run (no crash) on a fresh journal: the target state. ---
    EXEC.clear(); SIDE_EFFECT_PRODUCED.clear()
    ref = booking_workflow(WorkflowEngine(Journal()))
    print(f"  reference (no crash): id={ref['booking_id']} at={ref['created_at']}")

    # --- Crash run + resume on the SAME journal. ---
    EXEC.clear(); SIDE_EFFECT_PRODUCED.clear()
    journal = Journal()

    # Run 1: crashes after 'pay'.
    try:
        booking_workflow(WorkflowEngine(journal), crash_after="pay")
        raise AssertionError("expected the simulated crash")
    except RuntimeError as e:
        print(f"  run 1 crashed: {e} | journal kinds: {journal.kinds()}")
    assert EXEC == {"reserve": 1, "pay": 1}, EXEC
    run1_id = next(r["value"] for r in journal.read_all()
                   if r.get("key") == "booking_id")

    # Run 2: new engine on the SAME journal -> replays then finishes.
    final = booking_workflow(WorkflowEngine(journal))
    print(f"  run 2 resumed -> {final}")
    print(f"  exec counts: {EXEC} | producers called: {SIDE_EFFECT_PRODUCED}")

    # Exactly-once: each side-effecting step ran once across crash + resume.
    assert EXEC == {"reserve": 1, "pay": 1, "confirm": 1, "notify": 1}, EXEC
    # Deterministic replay: id/timestamp identical between run 1 and run 2...
    assert final["booking_id"] == run1_id, (final["booking_id"], run1_id)
    assert final["booking_id"] == ref["booking_id"]
    assert final["created_at"] == ref["created_at"]
    # ...and the non-deterministic producers were each called only ONCE.
    assert SIDE_EFFECT_PRODUCED == {"booking_id": 1, "created_at": 1}, SIDE_EFFECT_PRODUCED
    # WORKFLOW_DONE appears exactly once.
    assert journal.kinds().count("WORKFLOW_DONE") == 1, journal.kinds()
    # Crash+resume final state == reference run state.
    assert final == ref, (final, ref)

    print("[Verification] PASS -- exactly-once steps, replayed values identical")


# ==========================================================================
# HARD EXERCISE 2 -- DurableSagaOrchestrator (compensation + replay + DLQ)
# ==========================================================================

@dataclass
class SagaStep:
    name: str
    action: Callable[[dict], None]
    compensation: Callable[[dict], None]
    compensable: bool = True


class DurableSagaOrchestrator:
    """
    Saga over an append-only journal.

    - Runs steps in order; on the first failing action, compensates the
      already-committed steps in REVERSE order.
    - Each successful compensation journals COMPENSATED {name}, so a restart
      after a crash mid-compensation replays ONLY the remaining ones.
    - A non-compensable failure (compensable=False or a compensation that
      raises) goes to the dead_letter_queue, journaled as DEAD_LETTER, WITHOUT
      blocking the other compensations.
    """

    STEP_DONE = "STEP_DONE"
    STEP_FAILED = "STEP_FAILED"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    SAGA_ROLLED_BACK = "SAGA_ROLLED_BACK"
    DEAD_LETTER = "DEAD_LETTER"

    def __init__(self, journal: Journal, steps: list[SagaStep], world: dict) -> None:
        self.journal = journal
        self.steps = {s.name: s for s in steps}
        self.order = [s.name for s in steps]
        self.world = world
        self.dead_letter_queue: list[dict] = []

    def _already_compensated(self) -> set[str]:
        return {r["name"] for r in self.journal.read_all()
                if r["kind"] == self.COMPENSATED}

    def _committed_steps(self) -> list[str]:
        """Steps whose action succeeded (in journal order)."""
        return [r["name"] for r in self.journal.read_all()
                if r["kind"] == self.STEP_DONE]

    def run_forward(self) -> str | None:
        """Run actions in order; return the name of the failing step or None."""
        for name in self.order:
            step = self.steps[name]
            try:
                step.action(self.world)
                self.journal.append(self.STEP_DONE, name=name)
            except Exception as exc:  # noqa: BLE001 -- saga must catch & compensate
                self.journal.append(self.STEP_FAILED, name=name, error=str(exc))
                return name
        return None

    def compensate(self, failed_at: str | None = None) -> None:
        """
        Compensate committed steps in reverse order, skipping any already
        COMPENSATED in the journal (idempotent across a crash/restart).
        """
        committed = self._committed_steps()
        to_do = list(reversed(committed))
        already = self._already_compensated()
        if failed_at is not None:
            self.journal.append(self.COMPENSATING, failed_at=failed_at,
                                plan=[n for n in to_do if n not in already])

        for name in to_do:
            if name in already:
                continue  # crash-resume: don't replay a done compensation
            step = self.steps[name]
            if not step.compensable:
                self._dead_letter(name, "step is non-compensable")
                continue
            try:
                step.compensation(self.world)
                self.journal.append(self.COMPENSATED, name=name)
            except Exception as exc:  # compensation itself failed -> dead-letter
                self._dead_letter(name, f"compensation raised: {exc}")
                # Do NOT abort: keep compensating the rest of the steps.

        self.journal.append(self.SAGA_ROLLED_BACK)

    def _dead_letter(self, name: str, error: str) -> None:
        item = {"step": name, "error": error}
        self.dead_letter_queue.append(item)
        self.journal.append(self.DEAD_LETTER, step=name, error=error)


def hard_ex2_durable_saga() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- DurableSagaOrchestrator: compensation + replay + dead-letter")
    print("=" * 70)

    comp_counts: dict[str, int] = {}

    def make_world() -> dict:
        return {"stock_reserved": False, "payment_charged": False, "shipped": False}

    def reserve(w): w["stock_reserved"] = True
    def release(w):
        comp_counts["release_stock"] = comp_counts.get("release_stock", 0) + 1
        w["stock_reserved"] = False
    def charge(w): w["payment_charged"] = True
    def refund(w):
        comp_counts["refund_payment"] = comp_counts.get("refund_payment", 0) + 1
        w["payment_charged"] = False
    def ship_fail(w): raise RuntimeError("shipping service down")
    def cancel_ship(w): w["shipped"] = False

    # ---- Case A: nominal failure -> full reverse-order rollback ----
    print("\n  Case A: ship_order fails -> compensate refund then release")
    comp_counts.clear()
    world = make_world()
    initial = dict(world)
    j = Journal()
    steps = [
        SagaStep("reserve_stock", reserve, release),
        SagaStep("charge_payment", charge, refund),
        SagaStep("ship_order", ship_fail, cancel_ship),
    ]
    orch = DurableSagaOrchestrator(j, steps, world)
    failed = orch.run_forward()
    assert failed == "ship_order", failed
    orch.compensate(failed_at=failed)

    print(f"    world after rollback: {world}")
    print(f"    compensation order in journal: "
          f"{[r['name'] for r in j.read_all() if r['kind']=='COMPENSATED']}")
    # World restored to its initial state (full rollback).
    assert world == initial, (world, initial)
    # Reverse order: refund_payment compensated before release_stock.
    comp_seq = [r["name"] for r in j.read_all() if r["kind"] == "COMPENSATED"]
    assert comp_seq == ["charge_payment", "reserve_stock"], comp_seq
    assert comp_counts == {"refund_payment": 1, "release_stock": 1}, comp_counts

    # ---- Case B: crash DURING compensation -> resume skips the done one ----
    print("\n  Case B: crash after first compensation -> resume only the rest")
    comp_counts.clear()
    world = make_world()
    j2 = Journal()
    # Pre-seed journal as if forward ran (reserve, charge committed) and ship failed.
    j2.append("STEP_DONE", name="reserve_stock")
    j2.append("STEP_DONE", name="charge_payment")
    j2.append("STEP_FAILED", name="ship_order", error="shipping service down")
    # Reflect that committed actions already mutated the world.
    world["stock_reserved"] = True
    world["payment_charged"] = True
    # First compensation (refund) ran and was journaled, THEN the process crashed.
    refund(world)                                  # comp_counts.refund_payment = 1
    j2.append("COMPENSATED", name="charge_payment")
    print(f"    pre-crash: refund done, world={world}, "
          f"counts={comp_counts}")

    # Restart: new orchestrator reads the journal and resumes compensation.
    orch2 = DurableSagaOrchestrator(j2, steps, world)
    orch2.compensate(failed_at="ship_order")
    print(f"    world after resume: {world} | counts={comp_counts}")

    # refund_payment NOT replayed (already COMPENSATED); only release ran now.
    assert comp_counts == {"refund_payment": 1, "release_stock": 1}, comp_counts
    assert world == make_world(), world  # back to initial
    # Each compensation journaled exactly once (no duplicate COMPENSATED).
    done_names = [r["name"] for r in j2.read_all() if r["kind"] == "COMPENSATED"]
    assert sorted(done_names) == ["charge_payment", "reserve_stock"], done_names

    # ---- Case C: non-compensable failure -> dead-letter, others proceed ----
    print("\n  Case C: a compensation is non-compensable -> dead-letter queue")
    world = make_world()
    j3 = Journal()
    steps_dlq = [
        SagaStep("reserve_stock", reserve, release),
        # This step's compensation cannot run -> must go to the DLQ.
        SagaStep("charge_payment", charge, refund, compensable=False),
        SagaStep("ship_order", ship_fail, cancel_ship),
    ]
    orch3 = DurableSagaOrchestrator(j3, steps_dlq, world)
    failed3 = orch3.run_forward()
    orch3.compensate(failed_at=failed3)

    print(f"    dead_letter_queue: {orch3.dead_letter_queue}")
    print(f"    world after: {world}")
    # The non-compensable step is dead-lettered and journaled.
    assert len(orch3.dead_letter_queue) == 1, orch3.dead_letter_queue
    assert orch3.dead_letter_queue[0]["step"] == "charge_payment"
    assert "DEAD_LETTER" in j3.kinds()
    # The OTHER compensation (release_stock) still ran despite the DLQ item.
    assert world["stock_reserved"] is False, world
    # payment_charged stays True (its compensation never ran -> dead-lettered).
    assert world["payment_charged"] is True, world

    print("[Verification] PASS -- reverse rollback, crash-resume idempotent, DLQ")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  Day 20 HARD Solutions -- Durable & event-driven agents")
    print("  (stdlib only -- runs offline, no API key, no network)")
    print("#" * 70)

    hard_ex1_durable_workflow()
    hard_ex2_durable_saga()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70 + "\n")

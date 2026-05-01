"""
Day 12 -- Solutions to the easy exercises for production & observability.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/12-production-observabilite.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day12 = import_module("12-production-observabilite")
Tracer = day12.Tracer
Span = day12.Span
set_global_tracer = day12.set_global_tracer
traced = day12.traced
compute_cost = day12.compute_cost
BudgetExceeded = day12.BudgetExceeded
TransientError = day12.TransientError


# ===========================================================================
# SOLUTION 1 -- Trace summary report
# ===========================================================================

def trace_report(tracer: Tracer, trace_id: str) -> dict:
    spans = [s for s in tracer.spans if s.trace_id == trace_id]
    if not spans:
        return {"trace_id": trace_id, "total_spans": 0}
    slowest = max(spans, key=lambda s: s.duration_ms)
    most_expensive = max(spans, key=lambda s: s.cost_usd)
    errors = [s.name for s in spans if s.error]
    return {
        "trace_id": trace_id,
        "total_spans": len(spans),
        "total_duration_ms": sum(s.duration_ms for s in spans),
        "total_cost_usd": round(sum(s.cost_usd for s in spans), 6),
        "total_tokens_in": sum(s.tokens_in for s in spans),
        "total_tokens_out": sum(s.tokens_out for s in spans),
        "errors": errors,
        "slowest_span": {"name": slowest.name, "duration_ms": slowest.duration_ms},
        "most_expensive_span": {
            "name": most_expensive.name,
            "cost_usd": round(most_expensive.cost_usd, 6),
        },
    }


def solution_1() -> None:
    print("\n=== Solution 1: trace report ===")
    tracer = Tracer()
    set_global_tracer(tracer)

    @traced("fast_step")
    def fast_step(x: int) -> tuple[int, dict]:
        return x * 2, {"tokens_in": 5, "tokens_out": 2, "cost_usd": 0.0001}

    @traced("slow_step")
    def slow_step(x: int) -> tuple[int, dict]:
        time.sleep(0.02)
        return x + 100, {"tokens_in": 50, "tokens_out": 20, "cost_usd": 0.001}

    @traced("failing_step")
    def failing_step() -> int:
        raise RuntimeError("boom")

    for i, trace_name in enumerate(["t1", "t2", "t3"], start=1):
        trace_id = tracer.start_trace(f"report-test-{i}")
        fast_step(i)
        slow_step(i)
        if i == 3:
            try:
                failing_step()
            except RuntimeError:
                pass
        report = trace_report(tracer, trace_id)
        print(f"  {trace_name}: {report}")
        assert report["total_spans"] >= 2
        if i == 3:
            assert "failing_step" in report["errors"]


# ===========================================================================
# SOLUTION 2 -- Per-user daily budget
# ===========================================================================

class DailyBudgetExceeded(Exception):
    def __init__(self, user_id: str, current_cost: float) -> None:
        super().__init__(f"budget exceeded for {user_id}: {current_cost:.6f}$")
        self.user_id = user_id
        self.current_cost = current_cost


@dataclass
class DailyUserBudget:
    max_cost_per_user_per_day: float
    ledger: dict[str, dict] = field(default_factory=dict)

    def _today(self) -> str:
        return date.today().isoformat()

    def charge(
        self, user_id: str, model: str, tokens_in: int, tokens_out: int
    ) -> float:
        cost = compute_cost(model, tokens_in, tokens_out)
        today = self._today()
        entry = self.ledger.get(user_id)
        if entry is None or entry["date"] != today:
            entry = {"date": today, "cost": 0.0}
            self.ledger[user_id] = entry
        if entry["cost"] + cost > self.max_cost_per_user_per_day:
            raise DailyBudgetExceeded(user_id, entry["cost"] + cost)
        entry["cost"] += cost
        return cost

    def get_remaining(self, user_id: str) -> float:
        entry = self.ledger.get(user_id)
        if entry is None or entry["date"] != self._today():
            return self.max_cost_per_user_per_day
        return self.max_cost_per_user_per_day - entry["cost"]


def solution_2() -> None:
    print("\n=== Solution 2: per-user daily budget ===")
    budget = DailyUserBudget(max_cost_per_user_per_day=0.01)

    # user_A stays under budget
    for _ in range(5):
        budget.charge("user_A", "gpt-5.4-mini", 100, 50)
    print(f"  user_A remaining: {budget.get_remaining('user_A'):.6f}$")
    assert budget.get_remaining("user_A") > 0

    # user_B breaches
    try:
        for _ in range(100):
            budget.charge("user_B", "claude-opus-4-6", 1000, 1000)
    except DailyBudgetExceeded as exc:
        print(f"  user_B breach caught: {exc}")

    # user_C simulate yesterday then today
    budget.charge("user_C", "gpt-5.4-mini", 100, 50)
    # Manually rewind the date so we can prove the reset works
    budget.ledger["user_C"]["date"] = (date.today() - timedelta(days=1)).isoformat()
    # New charge should now consider the budget reset
    budget.charge("user_C", "gpt-5.4-mini", 100, 50)
    entry = budget.ledger["user_C"]
    print(f"  user_C entry after reset: {entry}")
    assert entry["date"] == date.today().isoformat()


# ===========================================================================
# SOLUTION 3 -- Circuit breaker
# ===========================================================================

class CircuitOpen(Exception):
    pass


class CircuitBreaker:
    """Three states: CLOSED (ok), OPEN (reject all), HALF_OPEN (probe)."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 3, reset_seconds: float = 2.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at: float | None = None

    def _check_half_open(self) -> None:
        if self.state == self.OPEN and self.opened_at is not None:
            if time.time() - self.opened_at >= self.reset_seconds:
                self.state = self.HALF_OPEN

    def call(self, fn: Callable[[], Any]) -> Any:
        self._check_half_open()
        if self.state == self.OPEN:
            raise CircuitOpen("breaker is open, rejecting call")
        try:
            result = fn()
        except Exception:
            self.failures += 1
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                self.opened_at = time.time()
                raise
            if self.failures >= self.failure_threshold:
                self.state = self.OPEN
                self.opened_at = time.time()
            raise
        # Success
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
        self.failures = 0
        return result


def make_flaky_service(fail_n_times: int) -> Callable[[], str]:
    state = {"remaining": fail_n_times}

    def service() -> str:
        if state["remaining"] > 0:
            state["remaining"] -= 1
            raise TransientError("simulated failure")
        return "ok"

    return service


def solution_3() -> None:
    print("\n=== Solution 3: circuit breaker ===")
    breaker = CircuitBreaker(failure_threshold=3, reset_seconds=0.5)
    # Service: fails 5 times then works
    service = make_flaky_service(fail_n_times=5)

    attempts: list[tuple[int, str]] = []
    for i in range(6):
        try:
            out = breaker.call(service)
            attempts.append((i, f"ok:{out}"))
        except TransientError:
            attempts.append((i, "transient"))
        except CircuitOpen:
            attempts.append((i, "rejected"))
    print("  first pass attempts:", attempts, "state=", breaker.state)
    assert breaker.state == "OPEN"

    # Wait for reset, then retry
    time.sleep(0.6)
    # At this point the service still fails -- half_open probe should fail -> OPEN again
    try:
        breaker.call(service)
    except TransientError:
        print("  probe failed, state=", breaker.state)
    assert breaker.state == "OPEN"

    # Wait again and force service to heal
    time.sleep(0.6)
    # Monkey-heal the service
    service_healed = lambda: "healed"
    out = breaker.call(service_healed)
    print("  healed call:", out, "state=", breaker.state)
    assert breaker.state == "CLOSED"


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

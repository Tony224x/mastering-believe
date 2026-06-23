"""
Solutions -- Day 12 (HARD): Production & Observability

Contains solutions for:
  - Hard Ex 1: ObservabilityPipeline -- SLOs, error budget, z-score anomaly
               detection, deduplicating AlertManager
  - Hard Ex 2: AdmissionController -- concurrency limiter + adaptive load
               shedding with priority classes and hysteresis

stdlib only, fully offline and deterministic. Concurrency / time are modeled
logically (single-thread) so the test is reproducible.

Run:  python 03-exercises/solutions/12-production-observabilite-hard.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ==========================================================================
# HARD EXERCISE 1 -- ObservabilityPipeline
# ==========================================================================


@dataclass
class Trace:
    trace_id: str
    latency_ms: float
    cost_usd: float
    ok: bool


@dataclass
class SLO:
    availability: float = 0.99
    latency_p95_ms: float = 1500.0
    cost_per_request_usd: float = 0.02


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    rank = 0.95 * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo), 2)


def zscore_anomalies(series: list[float], window: int = 10, threshold: float = 3.0) -> list[int]:
    """Return indices flagged as anomalies via rolling mean/std z-score."""
    anomalies: list[int] = []
    for i in range(len(series)):
        if i < window:                      # not enough history yet
            continue
        ref = series[i - window:i]
        mean = sum(ref) / len(ref)
        var = sum((x - mean) ** 2 for x in ref) / len(ref)
        std = var ** 0.5
        if std == 0:
            continue
        z = (series[i] - mean) / std
        if abs(z) > threshold:
            anomalies.append(i)
    return anomalies


class AlertManager:
    """Fires alerts but deduplicates per type within a cooldown window."""

    def __init__(self, cooldown: int = 5) -> None:
        self.cooldown = cooldown
        self._last_fired: dict[str, int] = {}
        self.fired: list[dict] = []

    def maybe_fire(self, tick: int, alert_type: str, detail: str) -> bool:
        last = self._last_fired.get(alert_type)
        if last is not None and tick - last < self.cooldown:
            return False                    # suppressed (dedup)
        self._last_fired[alert_type] = tick
        self.fired.append({"tick": tick, "type": alert_type, "detail": detail})
        return True


class ObservabilityPipeline:
    def __init__(self, slo: SLO | None = None) -> None:
        self.slo = slo or SLO()
        self.alerts = AlertManager(cooldown=5)

    def evaluate_slos(self, traces: list[Trace]) -> dict:
        total = len(traces)
        failed = sum(1 for t in traces if not t.ok)
        availability = (total - failed) / total if total else 1.0
        p95 = _p95([t.latency_ms for t in traces])
        avg_cost = sum(t.cost_usd for t in traces) / total if total else 0.0
        return {
            "availability": {"value": round(availability, 4),
                             "ok": availability >= self.slo.availability},
            "latency_p95": {"value": p95, "ok": p95 <= self.slo.latency_p95_ms},
            "cost_per_request": {"value": round(avg_cost, 5),
                                 "ok": avg_cost <= self.slo.cost_per_request_usd},
        }

    def error_budget(self, traces: list[Trace]) -> dict:
        total = len(traces)
        failed = sum(1 for t in traces if not t.ok)
        budget = 1.0 - self.slo.availability          # e.g. 1%
        consumed = failed / total if total else 0.0
        remaining = budget - consumed
        return {
            "budget": round(budget, 4),
            "consumed": round(consumed, 4),
            "remaining": round(remaining, 4),
            "exhausted": remaining <= 0,
            "recommendation": "FREEZE features" if remaining <= 0 else "ship freely",
        }

    def run(self, traces: list[Trace]) -> dict:
        slo_status = self.evaluate_slos(traces)
        budget = self.error_budget(traces)
        anomalies = zscore_anomalies([t.latency_ms for t in traces])

        # Alerting (with dedup) -- walk traces in order as a time series.
        for i, t in enumerate(traces):
            if i in anomalies:
                self.alerts.maybe_fire(i, "latency_anomaly", f"trace {t.trace_id} z>3")
        if budget["exhausted"]:
            self.alerts.maybe_fire(len(traces), "error_budget", "availability budget exhausted")
        for name, status in slo_status.items():
            if not status["ok"]:
                self.alerts.maybe_fire(len(traces), f"slo_{name}", f"{name}={status['value']}")

        return {
            "slo_status": slo_status,
            "error_budget": budget,
            "anomalies": anomalies,
            "alerts_fired": self.alerts.fired,
        }


def hard_ex1_observability() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: ObservabilityPipeline -- SLO + error budget + anomaly")
    print("=" * 60)

    traces: list[Trace] = []
    # 40 healthy traces.
    for i in range(40):
        traces.append(Trace(f"t{i}", latency_ms=100.0 + (i % 5) * 10, cost_usd=0.005, ok=True))
    # Inject a latency spike (anomaly) at index 25.
    traces[25] = Trace("t25", latency_ms=3000.0, cost_usd=0.005, ok=True)
    # Inject a burst of failures to blow the error budget (>1%).
    for i in range(36, 40):
        traces[i] = Trace(f"t{i}", latency_ms=120.0, cost_usd=0.005, ok=False)

    pipeline = ObservabilityPipeline()
    report = pipeline.run(traces)

    print("\n  SLO status:")
    for name, st in report["slo_status"].items():
        print(f"    {name:18s} value={st['value']} ok={st['ok']}")
    print(f"\n  Error budget: {report['error_budget']}")
    print(f"  Anomalies at indices: {report['anomalies']}")
    print(f"\n  Alerts fired ({len(report['alerts_fired'])}):")
    for a in report["alerts_fired"]:
        print(f"    [{a['type']}] {a['detail']}")

    assert 25 in report["anomalies"], "latency spike must be detected"
    assert report["error_budget"]["exhausted"], "4/40 failures > 1% budget"
    assert "FREEZE" in report["error_budget"]["recommendation"]
    # Dedup: only one latency_anomaly alert type fired.
    anomaly_alerts = [a for a in report["alerts_fired"] if a["type"] == "latency_anomaly"]
    assert len(anomaly_alerts) == 1, "anomaly alerts must be deduplicated"
    assert not report["slo_status"]["availability"]["ok"]

    print("\n  PASS -- SLOs, error budget, anomaly detection, dedup alerting.\n")


# ==========================================================================
# HARD EXERCISE 2 -- AdmissionController (concurrency + load shedding)
# ==========================================================================

PRIORITY_RANK = {"critical": 0, "high": 1, "low": 2}


@dataclass
class AdmissionController:
    max_in_flight: int = 5
    degraded_threshold_ms: float = 1000.0
    # hysteresis: enter shed above HIGH, leave only below LOW (avoid flapping)
    enter_shed_mult: float = 1.0           # >= threshold -> degraded
    leave_shed_mult: float = 0.8           # < 0.8*threshold -> recover
    in_flight: int = 0
    degraded: bool = False
    metrics: dict = field(default_factory=lambda: {
        "admitted": {"critical": 0, "high": 0, "low": 0},
        "queued": {"critical": 0, "high": 0, "low": 0},
        "shed": {"critical": 0, "high": 0, "low": 0},
    })

    def _update_mode(self, observed_latency_ms: float) -> None:
        if not self.degraded and observed_latency_ms >= self.degraded_threshold_ms * self.enter_shed_mult:
            self.degraded = True
        elif self.degraded and observed_latency_ms < self.degraded_threshold_ms * self.leave_shed_mult:
            self.degraded = False

    def _shed_fraction(self, observed_latency_ms: float) -> float:
        """0.0 at threshold, ramps to 1.0 at 2x threshold."""
        if not self.degraded:
            return 0.0
        over = observed_latency_ms / self.degraded_threshold_ms
        return max(0.0, min(1.0, over - 1.0))   # 1.0x->0%, 1.5x->50%, 2.0x->100%

    def admit(self, seq: int, priority: str, observed_latency_ms: float) -> str:
        """Return 'admitted', 'queued' or 'shed'. seq drives deterministic shedding."""
        self._update_mode(observed_latency_ms)

        # Adaptive load shedding -- low priority first, never critical.
        if priority == "low":
            frac = self._shed_fraction(observed_latency_ms)
            # Deterministic: shed the first `frac` fraction of a 10-slot cycle.
            if frac > 0 and (seq % 10) < round(frac * 10):
                self.metrics["shed"]["low"] += 1
                return "shed"

        # Concurrency limiter.
        if self.in_flight < self.max_in_flight:
            self.in_flight += 1
            self.metrics["admitted"][priority] += 1
            return "admitted"

        # Over capacity: critical/high queue, low is shed.
        if priority in ("critical", "high"):
            self.metrics["queued"][priority] += 1
            return "queued"
        self.metrics["shed"]["low"] += 1
        return "shed"

    def complete(self) -> None:
        if self.in_flight > 0:
            self.in_flight -= 1


def hard_ex2_admission_control() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: AdmissionController -- concurrency + adaptive shedding")
    print("=" * 60)

    ctrl = AdmissionController(max_in_flight=5, degraded_threshold_ms=1000.0)

    # Build a deterministic mixed workload: 30 critical, 30 high, 40 low.
    workload: list[str] = (["critical"] * 30 + ["high"] * 30 + ["low"] * 40)
    # Interleave so priorities are mixed through time.
    workload.sort(key=lambda p: (hash(p) % 7))   # stable-ish shuffle, deterministic

    # Latency ramps: healthy -> degraded (1800ms peak) -> recovery.
    def latency_at(i: int) -> float:
        if 30 <= i < 70:
            return 1800.0          # degraded window (1.8x threshold -> 80% shed of low)
        return 200.0               # healthy

    saw_degraded = saw_recovered = False
    for i, prio in enumerate(workload):
        verdict = ctrl.admit(i, prio, latency_at(i))
        if ctrl.degraded:
            saw_degraded = True
        elif saw_degraded:
            saw_recovered = True
        # Free a slot quickly to keep the limiter cycling.
        if verdict == "admitted":
            ctrl.complete()

    print(f"\n  metrics admitted: {ctrl.metrics['admitted']}")
    print(f"  metrics queued:   {ctrl.metrics['queued']}")
    print(f"  metrics shed:     {ctrl.metrics['shed']}")
    print(f"  final degraded mode: {ctrl.degraded}")

    # Critical is NEVER shed.
    assert ctrl.metrics["shed"]["critical"] == 0, "critical must never be shed"
    # Low traffic IS shed during the degraded window.
    assert ctrl.metrics["shed"]["low"] > 0, "low must be shed under degradation"
    # Mode transitions happened with hysteresis and recovered at the end.
    assert saw_degraded and saw_recovered, "must enter degraded then recover"
    assert ctrl.degraded is False, "must recover to normal after latency drops"

    print("\n  PASS -- critical protected, low shed under load, hysteresis recovery.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 12 HARD Solutions -- Production & Observability")
    print("#" * 60)

    hard_ex1_observability()
    hard_ex2_admission_control()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

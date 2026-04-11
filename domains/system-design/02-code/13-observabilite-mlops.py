"""
Jour 13 -- Mini observability lib + PSI drift detector.

Usage:
    python 13-observabilite-mlops.py

Ce module contient :
  1. Une mini lib de tracing LLM : spans, latence, tokens, cost, attributes
  2. Un exporter qui affiche un arbre d'execution
  3. Un detecteur de drift via PSI (Population Stability Index)
  4. Une demo qui instrumente un faux agent et fait drifter une feature

Le but est de montrer les structures de donnees derriere Langfuse / LangSmith
et de rendre tangible la formule PSI.
"""

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Span + Trace data model
# =============================================================================


@dataclass
class Span:
    """One unit of work inside a trace.

    In OpenTelemetry terms : an operation bound in time, with attributes
    and optional parent. For LLM calls, attributes include model name,
    tokens, cost, etc.
    """

    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    start_ts: float
    end_ts: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"  # ok | error

    def finish(self, status: str = "ok") -> None:
        self.end_ts = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_ts is None:
            return 0.0
        return (self.end_ts - self.start_ts) * 1000


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)


# =============================================================================
# SECTION 2 : Tracer (in-memory)
# =============================================================================


class Tracer:
    """Thread-unsafe tracer for illustration purposes.

    In production you would send spans to Langfuse / LangSmith / OTLP.
    """

    def __init__(self) -> None:
        self.traces: dict[str, Trace] = {}
        self._current_stack: list[Span] = []

    def start_trace(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        trace_id = uuid.uuid4().hex[:12]
        self.traces[trace_id] = Trace(trace_id=trace_id, user_id=user_id, session_id=session_id)
        return trace_id

    def start_span(self, trace_id: str, name: str, **attrs: Any) -> Span:
        parent = self._current_stack[-1].span_id if self._current_stack else None
        span = Span(
            span_id=uuid.uuid4().hex[:8],
            trace_id=trace_id,
            parent_id=parent,
            name=name,
            start_ts=time.time(),
            attributes=dict(attrs),
        )
        self.traces[trace_id].spans.append(span)
        self._current_stack.append(span)
        return span

    def end_span(self, span: Span, status: str = "ok", **extra_attrs: Any) -> None:
        span.attributes.update(extra_attrs)
        span.finish(status=status)
        if self._current_stack and self._current_stack[-1].span_id == span.span_id:
            self._current_stack.pop()

    def add_event(self, span: Span, event: str, **data: Any) -> None:
        span.events.append({"t": time.time(), "event": event, **data})


# =============================================================================
# SECTION 3 : Render a trace as a tree (like Langfuse UI)
# =============================================================================


def render_trace(trace: Trace) -> str:
    """Pretty print a trace as an indented tree with timings and cost."""
    by_parent: dict[Optional[str], list[Span]] = {}
    for s in trace.spans:
        by_parent.setdefault(s.parent_id, []).append(s)

    lines = [
        f"Trace {trace.trace_id}  user={trace.user_id}  session={trace.session_id}  tags={trace.tags}"
    ]

    total_cost = sum(s.attributes.get("cost_usd", 0.0) for s in trace.spans)
    total_tokens_in = sum(s.attributes.get("tokens_in", 0) for s in trace.spans)
    total_tokens_out = sum(s.attributes.get("tokens_out", 0) for s in trace.spans)
    total_duration = 0.0
    for s in trace.spans:
        if s.parent_id is None:
            total_duration = max(total_duration, s.duration_ms)
    lines.append(
        f"  total: {total_duration:.0f}ms tokens_in={total_tokens_in} "
        f"tokens_out={total_tokens_out} cost=${total_cost:.5f}"
    )

    def walk(parent: Optional[str], depth: int) -> None:
        for s in by_parent.get(parent, []):
            cost = s.attributes.get("cost_usd", 0.0)
            tokens_in = s.attributes.get("tokens_in", 0)
            tokens_out = s.attributes.get("tokens_out", 0)
            model = s.attributes.get("model", "")
            prefix = "  " + "|  " * depth + "+- "
            extra = []
            if model:
                extra.append(f"model={model}")
            if tokens_in or tokens_out:
                extra.append(f"tok={tokens_in}/{tokens_out}")
            if cost:
                extra.append(f"${cost:.5f}")
            extra_str = " ".join(extra)
            lines.append(f"{prefix}{s.name}  [{s.duration_ms:.0f}ms] {extra_str}")
            walk(s.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)


# =============================================================================
# SECTION 4 : Fake agent instrumented with tracer
# =============================================================================


def run_fake_agent(tracer: Tracer, user_id: str, question: str) -> Trace:
    """Simulate an agent that does : retrieve -> 1 LLM call -> rerank -> 1 LLM call.

    Each step is a span with fake timings and costs.
    """
    trace_id = tracer.start_trace(user_id=user_id, session_id="sess-" + user_id)
    trace = tracer.traces[trace_id]
    trace.tags = ["demo", "agent"]

    root = tracer.start_span(trace_id, "agent.run", user_question=question[:40])
    time.sleep(0)

    retr = tracer.start_span(trace_id, "retrieval", k=10)
    time.sleep(0.001)
    tracer.end_span(retr, num_hits=5)

    llm1 = tracer.start_span(
        trace_id,
        "llm.call",
        model="mini",
        tokens_in=500,
        tokens_out=120,
        cost_usd=500 * 0.00015 / 1000 + 120 * 0.0006 / 1000,
    )
    time.sleep(0.001)
    tracer.end_span(llm1)

    rer = tracer.start_span(trace_id, "reranker", top=10)
    time.sleep(0.001)
    tracer.end_span(rer)

    llm2 = tracer.start_span(
        trace_id,
        "llm.call",
        model="std",
        tokens_in=1200,
        tokens_out=350,
        cost_usd=1200 * 0.0025 / 1000 + 350 * 0.01 / 1000,
    )
    time.sleep(0.001)
    tracer.end_span(llm2)

    tracer.end_span(root, status="ok")
    return trace


# =============================================================================
# SECTION 5 : PSI drift detector
# =============================================================================


def _bin_edges(baseline: list[float], n_bins: int) -> list[float]:
    """Return n_bins+1 edges based on the baseline distribution quantiles."""
    if not baseline:
        return []
    sorted_bl = sorted(baseline)
    edges = []
    for i in range(n_bins + 1):
        idx = int(i * (len(sorted_bl) - 1) / n_bins)
        edges.append(sorted_bl[idx])
    # Deduplicate while preserving order
    dedup: list[float] = []
    for e in edges:
        if not dedup or e != dedup[-1]:
            dedup.append(e)
    return dedup


def _bucketize(values: list[float], edges: list[float]) -> list[int]:
    """Bucket counts : how many values fall in each [edges[i], edges[i+1]) bin."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        placed = False
        for i in range(len(edges) - 1):
            if (v >= edges[i]) and (v <= edges[i + 1] if i == len(edges) - 2 else v < edges[i + 1]):
                counts[i] += 1
                placed = True
                break
        if not placed:
            # out of range : put in closest bin
            if v < edges[0]:
                counts[0] += 1
            else:
                counts[-1] += 1
    return counts


def population_stability_index(
    baseline: list[float], current: list[float], n_bins: int = 10
) -> float:
    """Compute the PSI between two distributions of a numeric feature.

    PSI = sum( (p_base - p_curr) * ln(p_base / p_curr) ) over all bins.
    Rule of thumb :
        < 0.1  : no drift
        < 0.25 : moderate drift
        > 0.25 : significant drift
    """
    if not baseline or not current:
        return 0.0
    edges = _bin_edges(baseline, n_bins)
    base_counts = _bucketize(baseline, edges)
    curr_counts = _bucketize(current, edges)
    n_base = sum(base_counts) or 1
    n_curr = sum(curr_counts) or 1
    psi = 0.0
    for b, c in zip(base_counts, curr_counts):
        p_b = max(b / n_base, 1e-6)  # avoid log(0)
        p_c = max(c / n_curr, 1e-6)
        psi += (p_b - p_c) * math.log(p_b / p_c)
    return psi


def drift_verdict(psi: float) -> str:
    if psi < 0.1:
        return "no drift"
    if psi < 0.25:
        return "moderate drift (watch)"
    return "significant drift (act)"


# =============================================================================
# SECTION 6 : Demo
# =============================================================================


def demo_tracing() -> None:
    print(SEPARATOR)
    print("TRACING DEMO")
    print(SEPARATOR)
    tracer = Tracer()
    for i, q in enumerate(["price of our SaaS?", "how do I reset my password?"]):
        trace = run_fake_agent(tracer, user_id=f"user_{i}", question=q)
        print("\n" + render_trace(trace))


def demo_drift() -> None:
    print("\n" + SEPARATOR)
    print("DRIFT DETECTION DEMO (PSI)")
    print(SEPARATOR)
    random.seed(42)

    baseline = [random.gauss(0.0, 1.0) for _ in range(2000)]

    scenarios = {
        "identical (same gen)": [random.gauss(0.0, 1.0) for _ in range(2000)],
        "shifted mean (+0.5)":  [random.gauss(0.5, 1.0) for _ in range(2000)],
        "shifted mean (+1.5)":  [random.gauss(1.5, 1.0) for _ in range(2000)],
        "wider variance (x2)":  [random.gauss(0.0, 2.0) for _ in range(2000)],
        "totally different":    [random.gauss(5.0, 2.0) for _ in range(2000)],
    }
    print(f"\n{'scenario':<25} {'PSI':>8}  verdict")
    print("-" * 60)
    for label, current in scenarios.items():
        psi = population_stability_index(baseline, current, n_bins=10)
        print(f"{label:<25} {psi:>8.3f}  {drift_verdict(psi)}")


def demo() -> None:
    demo_tracing()
    demo_drift()
    print("\nTake-aways :")
    print("  - Traces make agent behavior debuggable (cost, latency, tool use).")
    print("  - PSI is cheap, stable, and should run daily on every production feature.")
    print("  - Thresholds 0.1 / 0.25 are standard but YMMV per feature.")


if __name__ == "__main__":
    demo()

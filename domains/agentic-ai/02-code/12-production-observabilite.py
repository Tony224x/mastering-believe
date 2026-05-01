"""
Day 12 -- Production & Observability: tracing, cost tracking, fallbacks.

Demonstrates:
  1. @traced decorator -- captures function name, input, output, duration,
     tokens, and writes one JSONL span per call
  2. Tracer          -- in-memory + file-backed span store with session support
  3. BudgetTracker   -- cost and call accounting with hard enforcement
  4. FallbackChain   -- primary LLM -> secondary LLM -> template answer
  5. retry_with_backoff -- transient error recovery with exponential backoff
  6. Demo            -- instrumented mini agent that uses all of the above

Dependencies: stdlib only. Optional: langfuse, openai, anthropic. Everything
falls back to MockLLMs so the demo runs offline.

Run:
    python domains/agentic-ai/02-code/12-production-observabilite.py
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Optional bindings
# ---------------------------------------------------------------------------

HAS_LANGFUSE = False
try:
    import langfuse  # noqa: F401
    HAS_LANGFUSE = True
except ImportError:
    pass


# ===========================================================================
# 1. SPAN AND TRACER
# ===========================================================================

@dataclass
class Span:
    """One recorded operation. The span is what you will see in Langfuse / OTel."""
    span_id: str
    trace_id: str
    name: str
    input: Any
    output: Any
    duration_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    parent_span_id: str | None = None
    timestamp: float = field(default_factory=time.time)


class Tracer:
    """
    Minimalist tracer that stores spans in memory and writes them to JSONL.

    In production you would swap this for Langfuse / LangSmith / OTel. The API
    surface is intentionally similar so you can port code over with a handful
    of changes.
    """

    def __init__(self, jsonl_path: str | None = None) -> None:
        self.jsonl_path = jsonl_path
        self.spans: list[Span] = []
        self._current_trace_id: str | None = None
        self._span_stack: list[str] = []

    def start_trace(self, trace_id: str | None = None) -> str:
        """Open a new trace (one user request = one trace)."""
        self._current_trace_id = trace_id or f"trace-{uuid.uuid4().hex[:8]}"
        self._span_stack = []
        return self._current_trace_id

    def record(self, span: Span) -> None:
        """Store a span and append to the JSONL file if configured."""
        self.spans.append(span)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(span), default=str) + "\n")

    def get_current_trace_id(self) -> str:
        return self._current_trace_id or "untraced"

    def push_parent(self, span_id: str) -> None:
        self._span_stack.append(span_id)

    def pop_parent(self) -> None:
        if self._span_stack:
            self._span_stack.pop()

    def current_parent(self) -> str | None:
        return self._span_stack[-1] if self._span_stack else None


# The tracing decorator is module-level so tests can swap the tracer.
_GLOBAL_TRACER = Tracer()


def set_global_tracer(tracer: Tracer) -> None:
    global _GLOBAL_TRACER
    _GLOBAL_TRACER = tracer


def traced(name: str | None = None) -> Callable[..., Any]:
    """
    Decorator that records one span per call.

    Usage:
        @traced("plan_step")
        def plan(x): return f"plan({x})"

    The decorated function can return either a plain value or a tuple
    `(value, {"tokens_in": N, "tokens_out": M, "cost_usd": C})` to record
    cost metadata on the span.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            span_id = f"span-{uuid.uuid4().hex[:6]}"
            trace_id = _GLOBAL_TRACER.get_current_trace_id()
            parent = _GLOBAL_TRACER.current_parent()
            start = time.perf_counter()
            _GLOBAL_TRACER.push_parent(span_id)
            error: str | None = None
            output: Any = None
            meta: dict = {}
            try:
                result = fn(*args, **kwargs)
                # If the function returned (value, meta_dict), capture meta for
                # the span but still pass the full tuple through to the caller.
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    output, meta = result
                else:
                    output = result
                return result
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                raise
            finally:
                _GLOBAL_TRACER.pop_parent()
                duration_ms = int((time.perf_counter() - start) * 1000)
                span = Span(
                    span_id=span_id,
                    trace_id=trace_id,
                    name=span_name,
                    input={"args": args, "kwargs": kwargs},
                    output=output,
                    duration_ms=duration_ms,
                    tokens_in=meta.get("tokens_in", 0),
                    tokens_out=meta.get("tokens_out", 0),
                    cost_usd=meta.get("cost_usd", 0.0),
                    error=error,
                    parent_span_id=parent,
                )
                _GLOBAL_TRACER.record(span)

        return wrapper

    return decorator


# ===========================================================================
# 2. BUDGET TRACKER
# ===========================================================================

class BudgetExceeded(Exception):
    """Raised when the cost ceiling is breached."""


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # price per 1K tokens (input, output) as of 2026 (example values)
    "claude-opus-4-6": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "gpt-5.4": (0.005, 0.015),
    "gpt-5.4-mini": (0.0005, 0.0015),
    "mock": (0.0001, 0.0003),
}


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for one LLM call given the token counts."""
    price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return tokens_in / 1000 * price_in + tokens_out / 1000 * price_out


@dataclass
class BudgetTracker:
    """Hard budget enforced in USD. Also keeps a break-down per model."""
    max_cost_usd: float
    current_cost_usd: float = 0.0
    calls: int = 0
    per_model: dict[str, float] = field(default_factory=dict)

    def charge(self, model: str, tokens_in: int, tokens_out: int) -> float:
        cost = compute_cost(model, tokens_in, tokens_out)
        self.current_cost_usd += cost
        self.calls += 1
        self.per_model[model] = self.per_model.get(model, 0.0) + cost
        if self.current_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"Budget {self.max_cost_usd:.4f}$ exceeded "
                f"(current: {self.current_cost_usd:.4f}$)"
            )
        return cost


# ===========================================================================
# 3. RETRY WITH EXPONENTIAL BACKOFF
# ===========================================================================

class TransientError(Exception):
    """Error we should retry on (rate limits, timeouts, 5xx)."""


def retry_with_backoff(
    fn: Callable[[], Any],
    max_attempts: int = 3,
    base_delay: float = 0.2,
    jitter: float = 0.1,
) -> Any:
    """
    Retry a callable on TransientError with exponential backoff + jitter.

    Non-transient errors bubble up immediately. In tests we keep delays small.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
            time.sleep(min(delay, 1.0))
    assert last_exc is not None
    raise last_exc


# ===========================================================================
# 4. FALLBACK CHAIN
# ===========================================================================

class FallbackChain:
    """
    Primary -> secondary -> static answer. Also integrated with the tracer
    so that each attempt becomes a child span.
    """

    def __init__(
        self,
        primary: Callable[[str], tuple[str, dict]],
        secondary: Callable[[str], tuple[str, dict]],
        static_fallback: str = "Sorry, the service is temporarily unavailable.",
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.static_fallback = static_fallback

    @traced("fallback_chain")
    def __call__(self, prompt: str) -> tuple[str, dict]:
        try:
            return retry_with_backoff(lambda: self.primary(prompt), max_attempts=2)
        except Exception:
            pass
        try:
            return retry_with_backoff(lambda: self.secondary(prompt), max_attempts=2)
        except Exception:
            pass
        return self.static_fallback, {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}


# ===========================================================================
# 5. MOCK LLMS -- one reliable, one flaky
# ===========================================================================

def mock_primary_llm(prompt: str) -> tuple[str, dict]:
    """Reliable mock LLM -- a deterministic transformation."""
    out = f"[primary] processed: {prompt[:40]}..."
    meta = {"tokens_in": len(prompt) // 4, "tokens_out": 20, "cost_usd": 0.0}
    meta["cost_usd"] = compute_cost("claude-opus-4-6", meta["tokens_in"], meta["tokens_out"])
    return out, meta


def make_flaky_llm(fail_n_times: int) -> Callable[[str], tuple[str, dict]]:
    """Build a mock LLM that raises TransientError the first N calls."""
    state = {"remaining": fail_n_times}

    def flaky(prompt: str) -> tuple[str, dict]:
        if state["remaining"] > 0:
            state["remaining"] -= 1
            raise TransientError("simulated 503")
        out = f"[flaky-ok] {prompt[:30]}..."
        meta = {"tokens_in": 20, "tokens_out": 15, "cost_usd": 0.0}
        meta["cost_usd"] = compute_cost("gpt-5.4-mini", 20, 15)
        return out, meta

    return flaky


def mock_secondary_llm(prompt: str) -> tuple[str, dict]:
    out = f"[secondary] cheaper answer for: {prompt[:30]}..."
    meta = {"tokens_in": 15, "tokens_out": 12, "cost_usd": 0.0}
    meta["cost_usd"] = compute_cost("gpt-5.4-mini", 15, 12)
    return out, meta


# ===========================================================================
# 6. INSTRUMENTED MINI AGENT
# ===========================================================================

class ProdAgent:
    """
    Toy agent wrapping: tracing + budget + retry + fallback.

    Each user query runs inside a single trace. Each internal step is a span.
    If the budget explodes mid-run, the agent aborts gracefully.
    """

    def __init__(
        self,
        llm_chain: FallbackChain,
        budget: BudgetTracker,
    ) -> None:
        self.llm_chain = llm_chain
        self.budget = budget

    @traced("plan_step")
    def plan(self, query: str) -> tuple[list[str], dict]:
        plan = [f"analyze({query})", f"synthesize({query})"]
        return plan, {"tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0}

    @traced("llm_step")
    def llm_step(self, prompt: str, model: str = "claude-opus-4-6") -> tuple[str, dict]:
        out, meta = self.llm_chain(prompt)
        # Attribute cost to the budget
        self.budget.charge(model, meta["tokens_in"], meta["tokens_out"])
        return out, meta

    def run(self, query: str) -> dict:
        trace_id = _GLOBAL_TRACER.start_trace()
        try:
            plan = self.plan(query)
            outputs = []
            for step in plan:
                out, _ = self.llm_step(step)
                outputs.append(out)
            return {
                "trace_id": trace_id,
                "answer": " | ".join(outputs),
                "cost": self.budget.current_cost_usd,
                "status": "ok",
            }
        except BudgetExceeded as exc:
            return {
                "trace_id": trace_id,
                "answer": None,
                "cost": self.budget.current_cost_usd,
                "status": "budget_exceeded",
                "error": str(exc),
            }


# ===========================================================================
# 7. DEMO
# ===========================================================================

def demo() -> None:
    print("=" * 70)
    print(f"Backends available: langfuse={HAS_LANGFUSE} -- using stdlib tracer")
    print("=" * 70)

    # Per-demo tracer writing to a jsonl file so you can inspect offline
    out_path = Path(__file__).parent / "_traces.jsonl"
    if out_path.exists():
        out_path.unlink()
    tracer = Tracer(jsonl_path=str(out_path))
    set_global_tracer(tracer)

    print("\n--- 1. normal run ---")
    chain = FallbackChain(primary=mock_primary_llm, secondary=mock_secondary_llm)
    budget = BudgetTracker(max_cost_usd=1.0)
    agent = ProdAgent(llm_chain=chain, budget=budget)
    result = agent.run("summarize recent Acme revenue growth")
    print(json.dumps(result, indent=2))

    print("\n--- 2. primary LLM flaky -> retry / fallback kicks in ---")
    flaky = make_flaky_llm(fail_n_times=1)
    chain_flaky = FallbackChain(primary=flaky, secondary=mock_secondary_llm)
    budget2 = BudgetTracker(max_cost_usd=1.0)
    agent2 = ProdAgent(llm_chain=chain_flaky, budget=budget2)
    result2 = agent2.run("analyze Q1 metrics")
    print(json.dumps(result2, indent=2))

    print("\n--- 3. budget exceeded during run ---")
    tight_budget = BudgetTracker(max_cost_usd=0.0001)
    agent3 = ProdAgent(
        llm_chain=FallbackChain(primary=mock_primary_llm, secondary=mock_secondary_llm),
        budget=tight_budget,
    )
    result3 = agent3.run("this run should get cut short")
    print(json.dumps(result3, indent=2))

    print("\n--- 4. trace summary ---")
    print(f"Total spans recorded: {len(tracer.spans)}")
    for s in tracer.spans[-6:]:
        print(
            f"  {s.trace_id[-8:]} {s.name:18} "
            f"duration={s.duration_ms:4}ms "
            f"cost=${s.cost_usd:.6f} "
            f"error={s.error}"
        )
    print(f"\nTraces written to {out_path}")


if __name__ == "__main__":
    demo()

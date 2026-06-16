"""
Solutions -- Day 18 (MEDIUM): Orchestration compared & failure modes

Contains solutions for:
  - Medium Ex 1: Benchmark harness comparing 3 topologies (single-agent /
                 sequential pipeline / parallel fan-out+fan-in) on the SAME
                 task set, measuring tokens, steps and perceived latency, and
                 asserting the expected trade-offs (single cheaper than pipeline,
                 fan-out lower latency than pipeline).
  - Medium Ex 2: Fault-injection framework -- injects a dropped-handoff payload
                 OR a malformed agent output into an A->B->C pipeline and proves
                 a schema validator flags it at the right stage (vs a naive run
                 that lets the fault propagate silently).
  - Medium Ex 3: Cascading-failure simulator across an agent chain (error
                 amplified hop by hop) + a circuit-breaker that contains it.

Self-contained: embeds a deterministic mock LLM (call + token counter) so the
file RUNS OFFLINE with zero dependencies (no langgraph, no API key). langgraph
is referenced behind a try/except only to mirror the course code.

Run:  python 03-exercises/solutions/18-orchestration-comparee-failure-modes-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Optional binding -- the mock LLM guarantees offline execution either way.
HAS_LANGGRAPH = False
try:  # pragma: no cover - environment dependent
    import langgraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    pass


# ==========================================================================
# MEDIUM EXERCISE 1 -- Benchmark harness: 3 topologies on the same task set
# ==========================================================================

@dataclass
class InstrumentedLLM:
    """
    Deterministic mock LLM with a per-call cost ledger.
    Each call records (topology, agent, tokens_in, tokens_out, latency).
    `latency` is a SIMULATED fixed cost per call -- no time.sleep, so the
    benchmark stays deterministic and offline.
    """
    LATENCY_PER_CALL: float = 1.0
    log: list[dict] = field(default_factory=list)

    def call(self, topology: str, agent: str, prompt: str, out_tokens: int = 12) -> str:
        tokens_in = len(prompt.split())
        self.log.append({
            "topology": topology, "agent": agent,
            "tokens_in": tokens_in, "tokens_out": out_tokens,
            "latency": self.LATENCY_PER_CALL,
        })
        return f"[{agent}] handled: {prompt[:40]}"

    def reset(self) -> None:
        self.log.clear()

    def totals(self, topology: str) -> dict:
        rows = [e for e in self.log if e["topology"] == topology]
        return {
            "llm_calls": len(rows),
            "tokens_total": sum(e["tokens_in"] + e["tokens_out"] for e in rows),
        }


def single_agent(llm: InstrumentedLLM, subtasks: list[str]) -> dict:
    """One LLM call handles ALL subtasks in a single shared context."""
    prompt = "Solve all of: " + " ; ".join(subtasks)
    out = llm.call("single", "solo_agent", prompt)
    # Perceived latency = a single call.
    return {"steps": 1, "perceived_latency": llm.LATENCY_PER_CALL,
            "success": True, "output": out}


def pipeline(llm: InstrumentedLLM, subtasks: list[str]) -> dict:
    """One agent per subtask, SEQUENTIAL, each re-injects the accumulated context."""
    context = ""
    steps = 0
    latency = 0.0
    for i, sub in enumerate(subtasks):
        # Context grows with every hop -> token cost accumulates (section 4.3).
        prompt = f"{context} | now do: {sub}".strip(" |")
        out = llm.call("pipeline", f"agent_{i}", prompt)
        context += " " + out  # accumulate
        steps += 1
        latency += llm.LATENCY_PER_CALL  # sequential -> latencies add up
    return {"steps": steps, "perceived_latency": latency,
            "success": True, "output": context.strip()}


def parallel_fanout(llm: InstrumentedLLM, subtasks: list[str]) -> dict:
    """One agent per INDEPENDENT subtask (no accumulation) + a fan-in synthesizer."""
    partials = []
    branch_latencies = []
    for i, sub in enumerate(subtasks):
        out = llm.call("parallel", f"branch_{i}", f"do: {sub}")  # independent ctx
        partials.append(out)
        branch_latencies.append(llm.LATENCY_PER_CALL)
    # Fan-in: one synthesis call.
    synth = llm.call("parallel", "fan_in", "synthesize: " + " ; ".join(partials))
    # Branches run concurrently -> latency = slowest branch + the fan-in step.
    perceived = (max(branch_latencies) if branch_latencies else 0.0) + llm.LATENCY_PER_CALL
    return {"steps": len(subtasks) + 1, "perceived_latency": perceived,
            "success": True, "output": synth}


def benchmark(subtasks: list[str]) -> dict:
    """Run the SAME subtask set through all 3 topologies and collect metrics."""
    report: dict = {}
    for name, fn in (("single", single_agent),
                     ("pipeline", pipeline),
                     ("parallel", parallel_fanout)):
        llm = InstrumentedLLM()  # fresh ledger per topology
        res = fn(llm, subtasks)
        totals = llm.totals(name)
        report[name] = {
            "llm_calls": totals["llm_calls"],
            "tokens_total": totals["tokens_total"],
            "steps": res["steps"],
            "perceived_latency": res["perceived_latency"],
            "success": res["success"],
        }
    return report


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Benchmark harness: 3 topologies on the same task set")
    print("=" * 70)

    subtasks = [
        "analyse document A",
        "analyse document B",
        "analyse document C",
        "analyse document D",
    ]
    report = benchmark(subtasks)

    print(f"  {'topology':10} | {'calls':>5} | {'tokens':>6} | {'steps':>5} | {'latency':>7}")
    for name, m in report.items():
        print(f"  {name:10} | {m['llm_calls']:>5} | {m['tokens_total']:>6} | "
              f"{m['steps']:>5} | {m['perceived_latency']:>7.1f}")

    # Trade-off 1: single agent is cheaper in tokens than the sequential pipeline
    # (no context accumulation across hops).
    assert report["single"]["tokens_total"] < report["pipeline"]["tokens_total"], report
    # Trade-off 2: on independent subtasks, parallel fan-out has lower perceived
    # latency than the sequential pipeline.
    assert report["parallel"]["perceived_latency"] < report["pipeline"]["perceived_latency"], report
    # The benchmark measures trade-offs, not failures: all topologies succeed.
    assert all(m["success"] for m in report.values())
    # Sanity: single is exactly one call; parallel is n branches + 1 fan-in.
    assert report["single"]["llm_calls"] == 1
    assert report["parallel"]["llm_calls"] == len(subtasks) + 1
    print("[Verification] PASS -- single cheaper than pipeline, "
          "parallel lower latency than pipeline")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Fault-injection framework + schema validator
# ==========================================================================

class HandoffValidationError(Exception):
    """Raised by the validator when a handoff/output breaks the expected schema."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"[{stage}] {detail}")
        self.stage = stage
        self.detail = detail


# Expected OUTPUT schema per stage: key -> expected python type.
STAGE_SCHEMA: dict[str, dict[str, type]] = {
    "researcher": {"sources": list},
    "writer": {"draft": str},
    "reviewer": {"verdict": str},
}

# Required INPUT keys per stage: upstream contributions the stage needs to read.
# This is what makes a dropped handoff detectable at the boundary.
STAGE_INPUT_REQS: dict[str, dict[str, type]] = {
    "researcher": {},                 # entry stage: only the task
    "writer": {"sources": list},      # needs research output
    "reviewer": {"draft": str},       # needs the writer's draft
}


def validate_input(stage: str, payload: dict) -> None:
    """Validate the payload ENTERING a stage: required upstream keys present + typed."""
    for key, typ in STAGE_INPUT_REQS[stage].items():
        if key not in payload or payload[key] is None:
            raise HandoffValidationError(stage, f"missing upstream key '{key}' (dropped handoff?)")
        if not isinstance(payload[key], typ):
            raise HandoffValidationError(
                stage, f"upstream key '{key}' has type {type(payload[key]).__name__}, expected {typ.__name__}")


def validate_output(stage: str, payload: dict) -> None:
    """Validate the payload PRODUCED by a stage: required output keys present + typed."""
    for key, typ in STAGE_SCHEMA[stage].items():
        if key not in payload or payload[key] is None:
            raise HandoffValidationError(stage, f"missing output key '{key}'")
        if not isinstance(payload[key], typ):
            raise HandoffValidationError(
                stage, f"output key '{key}' has type {type(payload[key]).__name__}, expected {typ.__name__}")


# --- The 3 well-behaved agents (agent(payload) -> payload) ----------------

def researcher(payload: dict) -> dict:
    return {"sources": ["src1", "src2", "src3"], **payload}


def writer(payload: dict) -> dict:
    n = len(payload.get("sources", []))
    return {"draft": f"Draft synthesizing {n} sources.", **payload}


def reviewer(payload: dict) -> dict:
    return {"verdict": f"approved: {payload.get('draft', '')[:20]}", **payload}


PIPELINE = [("researcher", researcher), ("writer", writer), ("reviewer", reviewer)]


@dataclass
class FaultInjector:
    """
    Parametrable fault injector. Activates ONE failure mode:
      - dropped_handoff: drops a key from the payload BEFORE the next stage
      - malformed_output: corrupts the type of an agent's output key
    """
    mode: Optional[str] = None
    target_stage: Optional[str] = None  # stage whose OUTPUT is tampered with
    drop_key: Optional[str] = None

    def after_agent(self, stage: str, payload: dict) -> dict:
        """Tamper with an agent's output (malformed_output)."""
        if self.mode == "malformed_output" and stage == self.target_stage:
            corrupted = dict(payload)
            # Wrong type for the stage's required key (e.g. draft = None / list).
            req_key = next(iter(STAGE_SCHEMA[stage]))
            corrupted[req_key] = None  # invalid: validator expects a non-None typed value
            return corrupted
        return payload

    def on_handoff(self, stage: str, payload: dict) -> dict:
        """Tamper with the payload in transit (dropped_handoff)."""
        if self.mode == "dropped_handoff" and stage == self.target_stage:
            dropped = dict(payload)
            dropped.pop(self.drop_key, None)
            return dropped
        return payload


def run_pipeline(agents, injector: Optional[FaultInjector] = None,
                 validate: bool = True) -> dict:
    """
    Execute A->B->C. If validate=True, check the schema at each boundary:
      - INPUT validation (catches a dropped handoff: required upstream key gone)
      - OUTPUT validation (catches a malformed output: wrong type / missing key)
    Returns {"ok": True, "result": ...} or {"ok": False, "error": stage, ...}.
    """
    payload: dict = {"task": "report on failure modes"}
    for stage, fn in agents:
        # Validate what THIS stage receives (a dropped handoff surfaces here).
        if validate:
            try:
                validate_input(stage, payload)
            except HandoffValidationError as e:
                return {"ok": False, "error": e.stage, "detail": e.detail}
        out = fn(payload)
        if injector:
            out = injector.after_agent(stage, out)  # corrupt this agent's output
        # Validate what THIS stage produced (a malformed output surfaces here).
        if validate:
            try:
                validate_output(stage, out)
            except HandoffValidationError as e:
                return {"ok": False, "error": e.stage, "detail": e.detail}
        # Handoff to the next stage (injector may drop a key in transit).
        payload = injector.on_handoff(stage, out) if injector else out
    return {"ok": True, "result": payload}


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Fault-injection framework + schema validator")
    print("=" * 70)

    # Baseline: no fault -> pipeline completes.
    ok = run_pipeline(PIPELINE)
    print(f"  no fault        -> ok={ok['ok']}")
    assert ok["ok"] is True and "verdict" in ok["result"]

    # Fault A: drop the 'draft' key at the writer->reviewer handoff.
    inj_drop = FaultInjector(mode="dropped_handoff", target_stage="writer", drop_key="draft")
    res_drop = run_pipeline(PIPELINE, injector=inj_drop)
    print(f"  dropped_handoff -> ok={res_drop['ok']} error={res_drop.get('error')} "
          f"({res_drop.get('detail')})")
    # The 'draft' key is dropped after writer -> the reviewer's INPUT validation
    # detects the missing upstream key at the reviewer boundary (info loss surfaced
    # exactly there, not further down).
    assert res_drop["ok"] is False
    assert res_drop["error"] == "reviewer", res_drop
    assert "dropped handoff" in res_drop["detail"]

    # Fault B: writer emits a malformed output (draft set to None / wrong type).
    inj_bad = FaultInjector(mode="malformed_output", target_stage="writer")
    res_bad = run_pipeline(PIPELINE, injector=inj_bad)
    print(f"  malformed_output-> ok={res_bad['ok']} error={res_bad.get('error')} "
          f"({res_bad.get('detail')})")
    assert res_bad["ok"] is False
    assert res_bad["error"] == "writer", res_bad

    # Contrast: NAIVE run (no validator) lets the dropped-handoff fault propagate
    # silently -- the reviewer receives an empty/corrupt draft and nobody flags it.
    naive = run_pipeline(PIPELINE, injector=inj_drop, validate=False)
    print(f"  naive (no valid)-> ok={naive['ok']} (fault propagated silently)")
    assert naive["ok"] is True, "naive run hides the fault -- no detection"
    # Evidence the fault really got through: the reviewer's verdict was built from
    # an EMPTY draft because the key was dropped before it ran.
    assert "approved: " in naive["result"]["verdict"]
    print("[Verification] PASS -- faults flagged at the right stage; "
          "naive run propagates silently")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Cascading-failure simulator + circuit-breaker
# ==========================================================================

@dataclass
class ChainResult:
    final_confidence: float
    validated: bool          # did the last agent present it as "valid"?
    degraded: bool           # did a breaker mark the run as degraded?
    cut_stage: Optional[int] # stage index where the breaker tripped (None if not)
    breaker_state: str       # "closed" | "open"


@dataclass
class CircuitBreaker:
    """Trips OPEN as soon as confidence drops below min_confidence."""
    min_confidence: float = 0.6
    state: str = "closed"
    cut_stage: Optional[int] = None

    def inspect(self, stage: int, confidence: float) -> bool:
        """Return True if the chain should be cut (circuit opened)."""
        if confidence < self.min_confidence:
            self.state = "open"
            self.cut_stage = stage
            return True
        return False


def run_chain(seed_confidence: float, n_agents: int = 4,
              breaker: Optional[CircuitBreaker] = None) -> ChainResult:
    """
    A chain of n agents. Each agent degrades confidence when its input looks
    suspect (confidence * factor). With no breaker the chain runs to the end and
    the LAST agent presents the output as "validated" regardless (section 4.4 --
    cross-agent legitimation). With a breaker it cuts as soon as it drops below
    the threshold.
    """
    confidence = seed_confidence
    for stage in range(n_agents):
        # An already-suspect input gets amplified downward (error propagation).
        factor = 0.7 if confidence < 0.8 else 1.0
        confidence = round(confidence * factor, 4)
        if breaker and breaker.inspect(stage, confidence):
            # Circuit opened: stop and DO NOT pretend it is validated.
            return ChainResult(final_confidence=confidence, validated=False,
                               degraded=True, cut_stage=stage,
                               breaker_state=breaker.state)
    # Reached the end. The final agent "validates" no matter what -> the trap.
    return ChainResult(final_confidence=confidence, validated=True,
                       degraded=False, cut_stage=None,
                       breaker_state=breaker.state if breaker else "closed")


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Cascading-failure simulator + circuit-breaker")
    print("=" * 70)

    THRESHOLD = 0.6
    SEED = 0.4  # injected error at the source (low-confidence input)

    # --- Naive run: no breaker -> cascade to the end, falsely "validated" ---
    naive = run_chain(SEED, n_agents=4, breaker=None)
    print(f"  naive   -> final_conf={naive.final_confidence} "
          f"validated={naive.validated} degraded={naive.degraded}")
    assert naive.final_confidence < THRESHOLD, naive
    assert naive.validated is True, "naive chain legitimizes a corrupt output"
    assert naive.degraded is False

    # --- Protected run: breaker cuts early, marks degraded, no fake validation ---
    breaker = CircuitBreaker(min_confidence=THRESHOLD)
    protected = run_chain(SEED, n_agents=4, breaker=breaker)
    print(f"  protected-> cut_stage={protected.cut_stage} "
          f"state={protected.breaker_state} validated={protected.validated} "
          f"degraded={protected.degraded}")
    assert protected.breaker_state == "open"
    assert protected.cut_stage is not None and protected.cut_stage < 4, protected
    assert protected.degraded is True
    assert protected.validated is False, "breaker must not present output as valid"

    # The protected run cuts strictly BEFORE the naive run reaches the end.
    print("[Verification] PASS -- naive legitimizes corruption; breaker contains "
          "the cascade")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 18 MEDIUM Solutions -- Orchestration compared & failure modes")
    print(f"  (langgraph available: {HAS_LANGGRAPH} -- running offline either way)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

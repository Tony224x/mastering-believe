"""
Solutions -- Day 18 (HARD): Orchestration compared & failure modes

Contains solutions for:
  - Hard Ex 1: FailureClassifier -- given multi-agent execution traces, detect
               and categorize >=4 distinct failure modes (info_loss_handoff,
               role_violation, infinite_loop, conflicting_writes, + bonus
               cascade_unvalidated). Each is proven with a forged trace; a clean
               trace yields zero labels (no false positive).
  - Hard Ex 2: RobustOrchestrator -- compares/runs a multi-agent topology under
               surveillance and applies graded recovery (retry -> reroute ->
               fallback single-agent) when a failure mode is detected, proving
               end-to-end that the recovered run succeeds where the naive run
               fails.

stdlib only, fully offline, deterministic. No network, no API key.

Run:  python 03-exercises/solutions/18-orchestration-comparee-failure-modes-hard.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional


# ==========================================================================
# HARD EXERCISE 1 -- FailureClassifier from execution traces
# ==========================================================================

@dataclass
class Event:
    """One step of a multi-agent execution trace."""
    step: int
    agent: str
    kind: str           # 'call' | 'handoff' | 'write' | 'output'
    data: dict = field(default_factory=dict)


# Role -> the state keys / tools each role is ALLOWED to touch.
ROLE_PERMS: dict[str, set[str]] = {
    "researcher": {"sources", "research"},
    "writer": {"draft"},
    "reviewer": {"verdict", "review"},
    "tooluser": {"search", "summarize"},
}


class FailureClassifier:
    """
    Inspects a trace and returns the set of failure-mode labels present.
    Detects at least 4 distinct multi-agent failure modes (course section 4 +
    MAST taxonomy).
    """

    LOOP_THRESHOLD = 3          # repeats without state progress -> livelock
    LOW_CONFIDENCE = 0.5        # output relayed below this w/o validation = cascade

    def classify(self, trace: list[Event]) -> list[str]:
        labels: set[str] = set()
        self._detect_info_loss(trace, labels)
        self._detect_role_violation(trace, labels)
        self._detect_infinite_loop(trace, labels)
        self._detect_conflicting_writes(trace, labels)
        self._detect_cascade_unvalidated(trace, labels)
        return sorted(labels)

    # -- 1. info loss across a handoff -------------------------------------
    def _detect_info_loss(self, trace, labels) -> None:
        # Track the context keys "in flight"; a handoff that drops a key loses info.
        carried: set[str] = set()
        for ev in trace:
            if ev.kind == "write":
                carried |= set(ev.data.get("keys", []))
            elif ev.kind == "handoff":
                forwarded = set(ev.data.get("payload_keys", []))
                # A key that existed but is not forwarded is lost information.
                if carried and not carried.issubset(forwarded):
                    labels.add("info_loss_handoff")
                carried = forwarded  # downstream only sees what was forwarded

    # -- 2. role violation -------------------------------------------------
    def _detect_role_violation(self, trace, labels) -> None:
        for ev in trace:
            allowed = ROLE_PERMS.get(ev.agent, set())
            if ev.kind == "write":
                for key in ev.data.get("keys", []):
                    if key not in allowed:
                        labels.add("role_violation")
            elif ev.kind == "call":
                tool = ev.data.get("tool")
                if tool is not None and tool not in allowed:
                    labels.add("role_violation")

    # -- 3. infinite loop / livelock --------------------------------------
    def _detect_infinite_loop(self, trace, labels) -> None:
        # Count consecutive (agent, kind) repeats that write NO new state key.
        run_len = 0
        prev_sig: Optional[tuple] = None
        for ev in trace:
            sig = (ev.agent, ev.kind)
            wrote_new = ev.kind == "write" and bool(ev.data.get("keys"))
            if sig == prev_sig and not wrote_new:
                run_len += 1
            else:
                run_len = 1
                prev_sig = sig
            if run_len >= self.LOOP_THRESHOLD:
                labels.add("infinite_loop")

    # -- 4. conflicting writes --------------------------------------------
    def _detect_conflicting_writes(self, trace, labels) -> None:
        last_value: dict[str, object] = {}
        for ev in trace:
            if ev.kind != "write":
                continue
            for key, val in ev.data.get("values", {}).items():
                if key in last_value and last_value[key] != val:
                    labels.add("conflicting_writes")
                last_value[key] = val

    # -- 5. (bonus) cascade relayed without validation --------------------
    def _detect_cascade_unvalidated(self, trace, labels) -> None:
        validated_since_low = True
        for ev in trace:
            if ev.kind in ("call", "handoff") and ev.data.get("confidence") is not None:
                if ev.data["confidence"] < self.LOW_CONFIDENCE:
                    validated_since_low = False
            if ev.kind == "write" and ev.data.get("validation"):
                validated_since_low = True
            if ev.kind == "output" and not validated_since_low:
                labels.add("cascade_unvalidated")


def hard_ex1_failure_classifier() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: FailureClassifier from execution traces")
    print("=" * 60)

    clf = FailureClassifier()

    # Clean trace: a well-formed research -> write -> review -> output.
    clean = [
        Event(0, "researcher", "write", {"keys": ["sources"], "values": {"sources": ["a"]}}),
        Event(1, "researcher", "handoff", {"payload_keys": ["sources"]}),
        Event(2, "writer", "write", {"keys": ["draft"], "values": {"draft": "d"}}),
        Event(3, "writer", "handoff", {"payload_keys": ["sources", "draft"]}),
        Event(4, "reviewer", "write", {"keys": ["verdict"], "values": {"verdict": "ok"},
                                       "validation": True}),
        Event(5, "reviewer", "output", {"confidence": 0.9}),
    ]
    print(f"  clean trace            -> {clf.classify(clean)}")
    assert clf.classify(clean) == [], "clean trace must yield no labels"

    # 1. info loss: writer carries {sources, draft} but forwards only {draft}.
    info_loss = [
        Event(0, "researcher", "write", {"keys": ["sources"]}),
        Event(1, "researcher", "handoff", {"payload_keys": ["sources"]}),
        Event(2, "writer", "write", {"keys": ["draft"]}),
        Event(3, "writer", "handoff", {"payload_keys": ["draft"]}),  # 'sources' lost
    ]
    out = clf.classify(info_loss)
    print(f"  info_loss_handoff      -> {out}")
    assert "info_loss_handoff" in out

    # 2. role violation: the reviewer writes the 'draft' key (writer's territory).
    role_viol = [
        Event(0, "reviewer", "write", {"keys": ["draft"]}),
    ]
    out = clf.classify(role_viol)
    print(f"  role_violation         -> {out}")
    assert "role_violation" in out
    # also: an agent calling a tool outside its role.
    role_viol_tool = [Event(0, "writer", "call", {"tool": "delete_db"})]
    assert "role_violation" in clf.classify(role_viol_tool)

    # 3. infinite loop: same agent 'calls' repeatedly without writing new state.
    loop = [
        Event(0, "agent_A", "call", {}),
        Event(1, "agent_A", "call", {}),
        Event(2, "agent_A", "call", {}),
        Event(3, "agent_A", "call", {}),
    ]
    out = clf.classify(loop)
    print(f"  infinite_loop          -> {out}")
    assert "infinite_loop" in out

    # 4. conflicting writes: two agents write different values to the same key.
    conflict = [
        Event(0, "researcher", "write", {"keys": ["sources"], "values": {"sources": ["a"]}}),
        Event(1, "writer", "write", {"keys": ["draft"], "values": {"sources": ["b"]}}),  # clash
    ]
    out = clf.classify(conflict)
    print(f"  conflicting_writes     -> {out}")
    assert "conflicting_writes" in out

    # 5. bonus cascade: low-confidence relayed to a final output, never validated.
    cascade = [
        Event(0, "researcher", "handoff", {"payload_keys": ["sources"], "confidence": 0.3}),
        Event(1, "writer", "output", {}),  # no validation event between
    ]
    out = clf.classify(cascade)
    print(f"  cascade_unvalidated    -> {out}")
    assert "cascade_unvalidated" in out

    # Coverage: at least 4 distinct failure modes are detectable.
    all_modes = set()
    for tr in (info_loss, role_viol, loop, conflict, cascade):
        all_modes |= set(clf.classify(tr))
    assert len({"info_loss_handoff", "role_violation", "infinite_loop",
                "conflicting_writes"} & all_modes) >= 4
    print("\n  PASS -- >=4 failure modes classified, clean trace stays clean.\n")


# ==========================================================================
# HARD EXERCISE 2 -- RobustOrchestrator (retry -> reroute -> fallback)
# ==========================================================================

class StepFailure(Exception):
    """A sub-step produced an invalid/malformed result."""


@dataclass
class ScriptedAgent:
    """
    Deterministic faillible agent. `fail_first` calls fail (transient), the rest
    succeed. If `always_fail`, every call fails (permanent -> needs reroute).
    """
    name: str
    fail_first: int = 0
    always_fail: bool = False
    _calls: int = 0

    def __call__(self, subtask: str) -> dict:
        self._calls += 1
        if self.always_fail or self._calls <= self.fail_first:
            raise StepFailure(f"{self.name} failed on call #{self._calls}")
        return {"by": self.name, "result": f"done: {subtask}"}


def is_valid(out: object) -> bool:
    """Minimal schema + failure-mode check: non-empty dict with a 'result' str."""
    return isinstance(out, dict) and isinstance(out.get("result"), str) and bool(out["result"])


@dataclass
class RobustOrchestrator:
    """
    Runs sub-tasks through primary agents under surveillance; on a detected
    failure it escalates: retry -> reroute (alternate agent) -> fallback to a
    single generalist agent (circuit-breaker, "one good agent > N agents").
    """
    primaries: dict[str, ScriptedAgent]        # subtask -> primary agent
    alternates: dict[str, ScriptedAgent]       # subtask -> reroute agent
    generalist: ScriptedAgent                  # final fallback (single-agent)
    max_retries: int = 2
    step_budget: int = 50

    def _run_step(self, subtask: str, report: dict) -> Optional[dict]:
        """Try primary with retries, then the alternate. Returns out or None."""
        primary = self.primaries[subtask]
        # retry loop on the primary (handles transient failures)
        for attempt in range(self.max_retries + 1):
            report["steps"] += 1
            self._check_budget(report)
            try:
                out = primary(subtask)
                if is_valid(out):
                    if attempt > 0:
                        report["strategy_used"].append("retry")
                        report["retries"] += attempt
                    return out
                raise StepFailure("invalid output")
            except StepFailure:
                continue  # retry
        # primary exhausted retries -> reroute to the alternate agent
        report["strategy_used"].append("retry")  # retries were spent and failed
        report["retries"] += self.max_retries
        alt = self.alternates.get(subtask)
        if alt is not None:
            report["steps"] += 1
            self._check_budget(report)
            try:
                out = alt(subtask)
                if is_valid(out):
                    report["strategy_used"].append("reroute")
                    report["rerouted"] += 1
                    return out
            except StepFailure:
                pass
        return None  # both primary and alternate failed -> caller falls back

    def _check_budget(self, report: dict) -> None:
        if report["steps"] > self.step_budget:
            raise RuntimeError("step budget exceeded -- aborting to avoid infinite loop")

    def run(self, subtasks: list[str]) -> dict:
        report = {"success": False, "strategy_used": [], "retries": 0,
                  "rerouted": 0, "fell_back": False, "steps": 0, "outputs": []}
        for sub in subtasks:
            out = self._run_step(sub, report)
            if out is None:
                # Multi-agent path failed for this subtask -> fall back whole task
                # to a single generalist agent (circuit-breaker).
                report["fell_back"] = True
                report["strategy_used"].append("fallback")
                gen_out = []
                for s in subtasks:
                    report["steps"] += 1
                    self._check_budget(report)
                    try:
                        gen_out.append(self.generalist(s))
                    except StepFailure:
                        # Even the fallback failed -> graceful failure, no loop.
                        gen_out.append({"by": "generalist", "result": ""})
                report["outputs"] = gen_out
                report["success"] = all(is_valid(o) for o in gen_out)
                return report
            report["outputs"].append(out)
        report["success"] = all(is_valid(o) for o in report["outputs"])
        return report


def naive_run(primaries: dict[str, ScriptedAgent], subtasks: list[str]) -> dict:
    """No recovery: one shot per primary, any failure fails the whole run."""
    outputs = []
    for sub in subtasks:
        try:
            out = primaries[sub](sub)
        except StepFailure:
            return {"success": False, "outputs": outputs}
        if not is_valid(out):
            return {"success": False, "outputs": outputs}
        outputs.append(out)
    return {"success": True, "outputs": outputs}


def hard_ex2_robust_orchestrator() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: RobustOrchestrator -- retry -> reroute -> fallback")
    print("=" * 60)

    subtasks = ["plan", "code", "review"]

    # --- Scenario A: transient + reroutable failures, multi-agent recovers ---
    # 'code' primary fails twice then succeeds (transient -> retry fixes it).
    # 'review' primary always fails (permanent -> reroute to alternate).
    def make_primaries():
        return {
            "plan": ScriptedAgent("planner"),
            "code": ScriptedAgent("coder", fail_first=2),
            "review": ScriptedAgent("reviewer", always_fail=True),
        }
    alternates = {
        "review": ScriptedAgent("reviewer_alt"),  # works
    }

    primaries = make_primaries()
    naive = naive_run(primaries, subtasks)
    print(f"  naive run        -> success={naive['success']}")
    assert naive["success"] is False, "naive run must fail on the scripted failures"

    primaries = make_primaries()  # fresh state (call counters reset)
    orch = RobustOrchestrator(primaries=primaries, alternates=alternates,
                              generalist=ScriptedAgent("generalist"))
    res = orch.run(subtasks)
    print(f"  robust run       -> success={res['success']} "
          f"strategies={res['strategy_used']} retries={res['retries']} "
          f"rerouted={res['rerouted']} fell_back={res['fell_back']}")
    assert res["success"] is True, "robust orchestrator must succeed on same scenario"
    assert "retry" in res["strategy_used"]      # transient 'code' fixed by retry
    assert "reroute" in res["strategy_used"]    # permanent 'review' fixed by reroute
    assert res["fell_back"] is False            # multi-agent path recovered

    # --- Scenario B: everything multi-agent breaks -> single-agent fallback ---
    broken_primaries = {
        "plan": ScriptedAgent("planner", always_fail=True),
        "code": ScriptedAgent("coder", always_fail=True),
        "review": ScriptedAgent("reviewer", always_fail=True),
    }
    broken_alts = {
        "plan": ScriptedAgent("planner_alt", always_fail=True),
    }
    orch2 = RobustOrchestrator(primaries=broken_primaries, alternates=broken_alts,
                               generalist=ScriptedAgent("generalist"))  # generalist works
    res2 = orch2.run(subtasks)
    print(f"  all-broken run   -> success={res2['success']} "
          f"fell_back={res2['fell_back']} strategies={res2['strategy_used']}")
    assert res2["fell_back"] is True, "must fall back to single-agent when multi-agent fails"
    assert res2["success"] is True, "single-agent fallback must save the run"
    assert "fallback" in res2["strategy_used"]

    # --- Bounded execution: even an all-broken stack (incl. generalist) must
    #     terminate gracefully (success=False), never loop forever. ---
    orch3 = RobustOrchestrator(primaries=broken_primaries, alternates={},
                               generalist=ScriptedAgent("gen_broken", always_fail=True),
                               step_budget=50)
    res3 = orch3.run(subtasks)
    print(f"  all-fail run     -> success={res3['success']} fell_back={res3['fell_back']} "
          f"steps={res3['steps']} (bounded, no infinite loop)")
    assert res3["success"] is False, "everything broken -> graceful failure"
    assert res3["fell_back"] is True
    assert res3["steps"] <= orch3.step_budget, "execution must stay within the budget"

    print("\n  PASS -- naive fails, robust recovers; fallback single-agent saves "
          "the all-broken run.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 18 HARD Solutions -- Orchestration compared & failure modes")
    print("#" * 60)

    hard_ex1_failure_classifier()
    hard_ex2_robust_orchestrator()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

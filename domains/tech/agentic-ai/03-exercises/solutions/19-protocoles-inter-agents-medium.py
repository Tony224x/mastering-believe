"""
Solutions -- Day 19 (MEDIUM): Inter-agent protocols (A2A)

Contains solutions for:
  - Medium Ex 1: Typed message Envelope + capability-based MessageRouter that
                 validates, authorizes (ACL) and routes -- rejecting malformed
                 and unauthorized messages cleanly.
  - Medium Ex 2: Request/response task delegation with correlation IDs, a
                 status lifecycle (submitted -> working -> completed/failed)
                 and DETERMINISTIC timeout handling (injected logical clock).
  - Medium Ex 3: Versioned handshake + capability negotiation between a client
                 and an agent (highest common version, capability intersection).

Self-contained: pure stdlib, no import of 02-code, no threads, no time.sleep.
Runs OFFLINE with zero dependencies (no langgraph, no API key).

Run:  python 03-exercises/solutions/19-protocoles-inter-agents-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ==========================================================================
# MEDIUM EXERCISE 1 -- Typed envelope + capability router with ACL
# ==========================================================================

SUPPORTED_VERSIONS = {"1.0", "1.1"}


@dataclass
class Envelope:
    """A typed inter-agent message envelope (A2A Message + routing header)."""
    msg_id: str
    sender: str
    capability: str            # the capability the sender wants invoked
    payload: dict
    protocol_version: str = "1.0"


def validate_envelope(env: Envelope) -> tuple[bool, str]:
    """Reject malformed envelopes; return (ok, reason)."""
    for field_name in ("msg_id", "sender", "capability"):
        value = getattr(env, field_name, None)
        if not value:  # missing or empty string
            return False, f"missing_field:{field_name}"
    if not isinstance(env.payload, dict):
        return False, "payload_not_dict"
    if env.protocol_version not in SUPPORTED_VERSIONS:
        return False, f"unsupported_version:{env.protocol_version}"
    return True, "ok"


class CapabilityAgent:
    """A receiver agent declaring one or more capabilities."""

    def __init__(self, name: str, capabilities: set[str]) -> None:
        self.name = name
        self.capabilities = capabilities

    def handle(self, env: Envelope) -> dict:
        # Deterministic mock work keyed by capability.
        if env.capability == "optimize_routes":
            stops = env.payload.get("stops", [])
            return {"optimized_order": sorted(stops), "distance_km": len(stops) * 12.4}
        if env.capability == "get_weather":
            city = env.payload.get("city", "Unknown")
            return {"city": city, "temperature_c": 18.5, "condition": "Partly cloudy"}
        return {"echo": env.payload}


class MessageRouter:
    """
    Validates, authorizes and routes envelopes to the agent that declares
    the requested capability. Rejection is a first-class outcome.
    """

    def __init__(self, acl: dict[str, set[str]]) -> None:
        # capability -> set of senders authorized to invoke it
        self._acl = acl
        # capability -> agent that declares it (first wins)
        self._by_capability: dict[str, CapabilityAgent] = {}
        self._agents: list[CapabilityAgent] = []

    def register(self, agent: CapabilityAgent) -> None:
        self._agents.append(agent)
        for cap in agent.capabilities:
            self._by_capability.setdefault(cap, agent)

    def route(self, env: Envelope) -> dict:
        # 1. Validation
        ok, reason = validate_envelope(env)
        if not ok:
            return {"status": "rejected", "reason": f"invalid:{reason}"}

        # 2. Authorization (ACL on the requested capability)
        allowed_senders = self._acl.get(env.capability, set())
        if env.sender not in allowed_senders:
            return {"status": "rejected", "reason": "unauthorized"}

        # 3. Find a competent receiver
        agent = self._by_capability.get(env.capability)
        if agent is None:
            return {"status": "rejected", "reason": "no_route"}

        # 4. Dispatch
        result = agent.handle(env)
        return {"status": "routed", "agent": agent.name, "result": result}


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Typed envelope + capability router with ACL")
    print("=" * 70)

    router = MessageRouter(acl={
        "optimize_routes": {"orchestrator", "dispatcher"},
        "get_weather": {"orchestrator"},
    })
    router.register(CapabilityAgent("RouteAgent", {"optimize_routes"}))
    router.register(CapabilityAgent("WeatherAgent", {"get_weather"}))

    # Valid + authorized -> routed to the right agent
    good = Envelope("m1", "orchestrator", "optimize_routes",
                    {"stops": ["Lyon", "Paris", "Nice"]}, "1.1")
    r_good = router.route(good)
    print(f"  valid/authorized   -> {r_good['status']} via {r_good.get('agent')}")
    assert r_good["status"] == "routed"
    assert r_good["agent"] == "RouteAgent"
    assert r_good["result"]["optimized_order"] == ["Lyon", "Nice", "Paris"]

    # Malformed: empty capability
    bad_field = Envelope("m2", "orchestrator", "", {"stops": []}, "1.0")
    r_bf = router.route(bad_field)
    print(f"  empty capability   -> {r_bf['status']} ({r_bf['reason']})")
    assert r_bf["status"] == "rejected" and r_bf["reason"] == "invalid:missing_field:capability"

    # Malformed: payload not a dict
    bad_payload = Envelope("m3", "orchestrator", "optimize_routes", ["not", "a", "dict"], "1.0")  # type: ignore[arg-type]
    r_bp = router.route(bad_payload)
    print(f"  payload not dict   -> {r_bp['status']} ({r_bp['reason']})")
    assert r_bp["status"] == "rejected" and r_bp["reason"] == "invalid:payload_not_dict"

    # Malformed: unsupported version
    bad_ver = Envelope("m4", "orchestrator", "optimize_routes", {"stops": []}, "9.9")
    r_bv = router.route(bad_ver)
    print(f"  bad version        -> {r_bv['status']} ({r_bv['reason']})")
    assert r_bv["status"] == "rejected" and r_bv["reason"] == "invalid:unsupported_version:9.9"

    # Unauthorized sender (weather restricted to 'orchestrator')
    unauth = Envelope("m5", "dispatcher", "get_weather", {"city": "Lyon"}, "1.0")
    r_ua = router.route(unauth)
    print(f"  unauthorized       -> {r_ua['status']} ({r_ua['reason']})")
    assert r_ua["status"] == "rejected" and r_ua["reason"] == "unauthorized"

    # No route: capability nobody declares (but ACL-allowed to isolate the case)
    router._acl["translate"] = {"orchestrator"}
    no_route = Envelope("m6", "orchestrator", "translate", {"text": "hi"}, "1.0")
    r_nr = router.route(no_route)
    print(f"  no competent agent -> {r_nr['status']} ({r_nr['reason']})")
    assert r_nr["status"] == "rejected" and r_nr["reason"] == "no_route"

    print("[Verification] PASS -- valid routed, malformed/unauthorized/no_route rejected")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Task delegation: correlation IDs, lifecycle, timeout
# ==========================================================================

class Status(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


_TERMINAL = {Status.COMPLETED, Status.FAILED, Status.TIMED_OUT}


@dataclass
class TaskRecord:
    corr_id: str
    status: Status
    submitted_at: float
    deadline: float
    work_fn: Callable[[], dict]
    result: dict | None = None
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "corr_id": self.corr_id,
            "status": self.status.value,
            "result": self.result,
            "history": list(self.history),
        }


class DelegationBroker:
    """
    Server side of a request/response delegation contract.

    Drives tasks one step at a time so the lifecycle is observable and the
    timeout is deterministic (no threads, logical clock injected via `now`).
    """

    def __init__(self, now: Callable[[], float]) -> None:
        self._now = now
        self._tasks: dict[str, TaskRecord] = {}
        self._ids = ("corr-%04d" % n for n in itertools.count(1))

    def submit(self, work_fn: Callable[[], dict], deadline_s: float) -> str:
        corr_id = next(self._ids)
        t = self._now()
        rec = TaskRecord(
            corr_id=corr_id,
            status=Status.SUBMITTED,
            submitted_at=t,
            deadline=t + deadline_s,
            work_fn=work_fn,
        )
        rec.history.append(Status.SUBMITTED.value)
        self._tasks[corr_id] = rec
        return corr_id

    def _transition(self, rec: TaskRecord, status: Status) -> None:
        rec.status = status
        rec.history.append(status.value)

    def step(self, corr_id: str) -> None:
        rec = self._tasks[corr_id]
        if rec.status in _TERMINAL:
            return

        # Deadline check happens BEFORE any transition / work execution.
        if self._now() > rec.deadline:
            self._transition(rec, Status.TIMED_OUT)
            return

        if rec.status is Status.SUBMITTED:
            self._transition(rec, Status.WORKING)
            return

        if rec.status is Status.WORKING:
            try:
                rec.result = rec.work_fn()
                self._transition(rec, Status.COMPLETED)
            except Exception as exc:  # noqa: BLE001
                rec.result = {"error": str(exc)}
                self._transition(rec, Status.FAILED)

    def get(self, corr_id: str) -> dict:
        return self._tasks[corr_id].to_dict()


def delegate_and_poll(broker: DelegationBroker, work_fn: Callable[[], dict],
                      deadline_s: float, max_steps: int = 10) -> dict:
    """Client side: submit, then poll step/get until a terminal state."""
    corr_id = broker.submit(work_fn, deadline_s)
    for _ in range(max_steps):
        snap = broker.get(corr_id)
        if snap["status"] in {s.value for s in _TERMINAL}:
            return snap
        broker.step(corr_id)
    return broker.get(corr_id)


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Task delegation: correlation IDs, lifecycle, timeout")
    print("=" * 70)

    # A mutable logical clock we can advance by hand (deterministic timeout).
    clock = {"t": 100.0}
    now = lambda: clock["t"]  # noqa: E731

    # Effect counter to prove a timed-out task NEVER runs its work_fn.
    effects = {"count": 0}

    def fast_work() -> dict:
        effects["count"] += 1
        return {"answer": 42}

    def boom() -> dict:
        effects["count"] += 1
        raise RuntimeError("worker exploded")

    # --- 1) Happy path: submitted -> working -> completed
    broker = DelegationBroker(now=now)
    final = delegate_and_poll(broker, fast_work, deadline_s=50)
    print(f"  happy path   -> {final['status']} history={final['history']}")
    assert final["corr_id"].startswith("corr-")
    assert final["status"] == "completed"
    assert final["history"] == ["submitted", "working", "completed"]
    assert final["result"] == {"answer": 42}

    # --- 2) Failure path: work_fn raises -> failed (not completed)
    clock["t"] = 100.0
    broker2 = DelegationBroker(now=now)
    failed = delegate_and_poll(broker2, boom, deadline_s=50)
    print(f"  failure path -> {failed['status']} history={failed['history']}")
    assert failed["status"] == "failed"
    assert failed["history"] == ["submitted", "working", "failed"]
    assert "exploded" in failed["result"]["error"]

    # --- 3) Timeout path: advance the clock past the deadline before stepping
    clock["t"] = 100.0
    broker3 = DelegationBroker(now=now)
    effects_before = effects["count"]
    corr = broker3.submit(fast_work, deadline_s=10)  # deadline = 110.0
    clock["t"] = 200.0                                # jump well past deadline
    # Poll: first step should detect the timeout immediately.
    snap = broker3.get(corr)
    steps = 0
    while snap["status"] not in {s.value for s in _TERMINAL} and steps < 10:
        broker3.step(corr)
        snap = broker3.get(corr)
        steps += 1
    print(f"  timeout path -> {snap['status']} history={snap['history']}")
    assert snap["status"] == "timed_out"
    # work_fn must NOT have run for the timed-out task
    assert effects["count"] == effects_before, "timed-out task must not execute work_fn"

    print(f"  total work_fn invocations: {effects['count']} (happy=1 + failure=1, timeout=0)")
    assert effects["count"] == 2
    print("[Verification] PASS -- lifecycle traced, failure + deterministic timeout")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Versioned handshake + capability negotiation
# ==========================================================================

def version_key(v: str) -> tuple[int, ...]:
    """'1.10' -> (1, 10) so that '1.10' > '1.9' (no naive lexicographic compare)."""
    return tuple(int(p) for p in v.split("."))


def negotiate(client_profile: dict, server_profile: dict) -> dict:
    """Agree on the highest common version and the capability intersection."""
    common_versions = set(client_profile["versions"]) & set(server_profile["versions"])
    if not common_versions:
        return {"ok": False, "reason": "version_mismatch"}
    chosen = max(common_versions, key=version_key)

    common_caps = set(client_profile["capabilities"]) & set(server_profile["capabilities"])
    if not common_caps:
        return {"ok": False, "reason": "no_common_capability"}

    return {"ok": True, "version": chosen, "capabilities": sorted(common_caps)}


def dispatch(session: dict, capability: str) -> str:
    """After a successful handshake, only negotiated capabilities are callable."""
    if not session.get("ok"):
        raise ValueError("no active session")
    if capability not in set(session["capabilities"]):
        raise ValueError(f"capability not negotiated: {capability}")
    return f"dispatched {capability} over protocol v{session['version']}"


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Versioned handshake + capability negotiation")
    print("=" * 70)

    client = {"versions": ["1.0", "1.1"],
              "capabilities": {"optimize_routes", "estimate_eta", "translate"}}
    server = {"versions": ["1.1", "1.0"],
              "capabilities": {"optimize_routes", "estimate_eta", "get_weather"}}

    session = negotiate(client, server)
    print(f"  compatible   -> ok={session['ok']} v={session['version']} "
          f"caps={session['capabilities']}")
    assert session["ok"] is True
    assert session["version"] == "1.1"  # highest common, not first found
    assert session["capabilities"] == ["estimate_eta", "optimize_routes"]

    # Negotiated capabilities are dispatchable; others are not.
    msg = dispatch(session, "optimize_routes")
    print(f"  dispatch ok  -> {msg}")
    try:
        dispatch(session, "translate")  # client has it, server doesn't -> not negotiated
        raise AssertionError("translate should not be dispatchable")
    except ValueError as e:
        print(f"  dispatch off -> rejected: {e}")
        assert "not negotiated" in str(e)

    # Version mismatch (disjoint version sets)
    old_server = {"versions": ["2.0", "2.1"], "capabilities": {"optimize_routes"}}
    vm = negotiate(client, old_server)
    print(f"  version gap  -> ok={vm['ok']} reason={vm['reason']}")
    assert vm == {"ok": False, "reason": "version_mismatch"}

    # No common capability (versions match, capabilities disjoint)
    odd_server = {"versions": ["1.0"], "capabilities": {"render_3d", "send_email"}}
    nc = negotiate(client, odd_server)
    print(f"  no caps      -> ok={nc['ok']} reason={nc['reason']}")
    assert nc == {"ok": False, "reason": "no_common_capability"}

    # version_key ordering: 1.10 must outrank 1.9
    assert version_key("1.10") > version_key("1.9")
    both = {"versions": ["1.9", "1.10"], "capabilities": {"optimize_routes"}}
    chosen = negotiate(both, both)
    print(f"  semver pick  -> v={chosen['version']} (1.10 > 1.9)")
    assert chosen["version"] == "1.10"

    print("[Verification] PASS -- highest common version, intersection, scoped dispatch")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 19 MEDIUM Solutions -- Inter-agent protocols (A2A)")
    print("  (pure stdlib, offline, deterministic -- no threads, no API key)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

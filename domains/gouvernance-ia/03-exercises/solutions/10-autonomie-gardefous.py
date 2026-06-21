"""
Solutions — Day 10 "Autonomy, guardrails & operations" (easy / medium / hard).

One file, three sections separated by markers, plus a smoke test in __main__.

# requires: stdlib only
Run:
    python 10-autonomie-gardefous.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# === EASY ===================================================================
# Calibrate autonomy level, a tiny action-rail guardrail, and a fail-safe
# kill-switch. WHY: autonomy is a per-action dial; the kill-switch defaults to
# "stop" when in doubt (unknown agent / non-active state).


def autonomy_level(impact: int, irreversible: bool) -> str:
    """Map (impact, irreversible) to one of three human-involvement levels."""
    score = impact + (2 if irreversible else 0)
    if score <= 2:
        return "out-of-the-loop"
    if score <= 4:
        return "on-the-loop"
    return "in-the-loop"


def refund_guardrail(amount: float) -> str:
    """Action rail for a refund: ALLOW / ESCALATE / DENY at action time."""
    if amount <= 0:
        return "DENY"          # invalid amount
    if amount <= 100:
        return "ALLOW"
    if amount <= 2000:
        return "ESCALATE"      # needs human approval
    return "DENY"              # out of operating domain


def can_run(switch: dict, agent_id: str) -> bool:
    """Fail-safe: only an explicit 'active' state lets the agent run."""
    # .get with a default that is NOT 'active' -> unknown agent cannot run.
    return switch.get(agent_id, "killed") == "active"


# === MEDIUM =================================================================
# Rolling-window budget, the combined autonomy gate, and an incident state
# machine with forbidden transitions. WHY: budgets cap cumulative risk; the
# gate chains kill-switch -> budget -> autonomy in a fixed order.


@dataclass
class RollingBudget:
    max_cost: float
    max_actions: int
    window_seconds: float
    _events: list[tuple[float, float]] = field(default_factory=list)  # (ts, cost)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._events = [(ts, c) for (ts, c) in self._events if ts >= cutoff]

    def would_exceed(self, cost: float, now: float) -> bool:
        self._prune(now)
        spent = sum(c for _, c in self._events)
        count = len(self._events)
        return (spent + cost > self.max_cost) or (count + 1 > self.max_actions)

    def record(self, cost: float, now: float) -> None:
        self._events.append((now, cost))

    def total(self, now: float) -> tuple[float, int]:
        self._prune(now)
        return sum(c for _, c in self._events), len(self._events)


def gate(agent_id: str, cost: float, impact: int, irreversible: bool,
         switch: dict, budget: RollingBudget, now: float) -> tuple[str, str]:
    """Single choke point: kill-switch -> budget -> autonomy, in that order."""
    # 1) Kill-switch FIRST.
    if not can_run(switch, agent_id):
        return ("DENY", "kill-switch")
    # 2) Budget (soft cap -> escalate, keep the service alive).
    if budget.would_exceed(cost, now):
        return ("ESCALATE", "budget")
    # 3) Autonomy calibrated on risk.
    score = impact + (2 if irreversible else 0)
    if score >= 5:
        return ("ESCALATE", "risk")
    budget.record(cost, now)
    return ("ALLOW", "ok")


class IncidentPhase(str, Enum):
    OPEN = "OPEN"
    CONTAINED = "CONTAINED"
    ERADICATED = "ERADICATED"
    RECOVERED = "RECOVERED"
    CLOSED = "CLOSED"


_FLOW = {
    IncidentPhase.OPEN: IncidentPhase.CONTAINED,
    IncidentPhase.CONTAINED: IncidentPhase.ERADICATED,
    IncidentPhase.ERADICATED: IncidentPhase.RECOVERED,
    IncidentPhase.RECOVERED: IncidentPhase.CLOSED,
}


class Incident:
    def __init__(self, agent_id: str, trigger: str) -> None:
        self.agent_id = agent_id
        self.phase = IncidentPhase.OPEN
        self.history: list[tuple[IncidentPhase, str]] = [(IncidentPhase.OPEN, trigger)]

    def advance(self, note: str) -> None:
        nxt = _FLOW.get(self.phase)
        if nxt is None:
            raise ValueError(f"no transition out of {self.phase.value}")
        self.phase = nxt
        self.history.append((nxt, note))


# === HARD ===================================================================
# Approval queue (soft cap with deferred accounting), incident orchestration
# wired to the kill-switch, and a decommission gate that refuses zombies.


class GateWithApprovals:
    """Autonomy gate that queues escalations for human approval."""

    def __init__(self, switch: dict, budgets: dict[str, RollingBudget]) -> None:
        self.switch = switch
        self.budgets = budgets
        self.pending: list[dict] = []
        self.audit: list[dict] = []
        self.hitl_forced: dict[str, bool] = {}

    def _log(self, agent_id: str, action: str, decision: str, reason: str) -> tuple[str, str]:
        self.audit.append({"agent": agent_id, "action": action,
                           "decision": decision, "reason": reason})
        return (decision, reason)

    def evaluate(self, agent_id: str, action: str, cost: float, impact: int,
                 irreversible: bool, now: float) -> tuple[str, str]:
        if not can_run(self.switch, agent_id):
            return self._log(agent_id, action, "DENY", "kill-switch")
        budget = self.budgets.get(agent_id)
        # HITL-forced (e.g. post-incident recovery): escalate any sensitive action.
        if self.hitl_forced.get(agent_id) and cost > 0:
            self.pending.append({"agent_id": agent_id, "action": action,
                                 "cost": cost, "risk": impact})
            return self._log(agent_id, action, "ESCALATE", "hitl-forced")
        if budget and budget.would_exceed(cost, now):
            self.pending.append({"agent_id": agent_id, "action": action,
                                 "cost": cost, "risk": impact})
            return self._log(agent_id, action, "ESCALATE", "budget")
        score = impact + (2 if irreversible else 0)
        if score >= 5:
            self.pending.append({"agent_id": agent_id, "action": action,
                                 "cost": cost, "risk": score})
            return self._log(agent_id, action, "ESCALATE", "risk")
        if budget:
            budget.record(cost, now)
        return self._log(agent_id, action, "ALLOW", "ok")

    def approve(self, index: int, now: float) -> tuple[str, str]:
        item = self.pending.pop(index)
        budget = self.budgets.get(item["agent_id"])
        if budget and budget.would_exceed(item["cost"], now):
            return self._log(item["agent_id"], item["action"], "DENY",
                             "budget still exceeded")
        if budget:
            budget.record(item["cost"], now)
        return self._log(item["agent_id"], item["action"], "ALLOW", "approved")

    def reject(self, index: int, reason: str) -> tuple[str, str]:
        item = self.pending.pop(index)
        return self._log(item["agent_id"], item["action"], "DENY", f"rejected: {reason}")


def handle_incident(gate_obj: GateWithApprovals, agent_id: str, trigger: str) -> Incident:
    """Orchestrate detect -> contain -> eradicate -> recover -> close, wired to
    the kill-switch. WHY: 'contain' must really stop the agent; 'recover' brings
    it back in degraded (HITL-forced) mode."""
    inc = Incident(agent_id, trigger)                         # detect (OPEN)

    gate_obj.switch[agent_id] = "killed"                      # contain: stop agent
    inc.advance("kill-switch engaged")
    # sanity check: the agent is now denied
    d, _ = gate_obj.evaluate(agent_id, "probe", 10.0, 2, False, now=0.0)
    assert d == "DENY", "containment must block actions"

    inc.advance("patched root cause")                         # eradicate

    gate_obj.switch[agent_id] = "active"                      # recover (degraded)
    gate_obj.hitl_forced[agent_id] = True
    inc.advance("re-activated HITL-forced")

    inc.advance("post-mortem filed")                          # close
    return inc


class LifecycleState(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DECOMMISSIONED = "DECOMMISSIONED"


def try_decommission(agent: dict) -> tuple[bool, list[str]]:
    """Allow DECOMMISSIONED only from ACTIVE/SUSPENDED with a fully green checklist."""
    missing: list[str] = []
    if agent.get("state") not in ("ACTIVE", "SUSPENDED"):
        missing.append(f"bad source state: {agent.get('state')}")
    if not agent.get("revoked_access"):
        missing.append("access/scopes not revoked (zombie identity risk)")
    if not agent.get("audit_archived"):
        missing.append("audit not archived (retention breach)")
    if not agent.get("owner_closed"):
        missing.append("owner not closed")
    if missing:
        return (False, missing)
    agent["state"] = LifecycleState.DECOMMISSIONED.value
    return (True, [])


# === SMOKE TEST =============================================================


def _smoke() -> None:
    # ---- EASY ----
    assert autonomy_level(1, False) == "out-of-the-loop"
    assert autonomy_level(3, False) == "on-the-loop"
    assert autonomy_level(4, True) == "in-the-loop"  # 4+2=6 -> in-the-loop
    assert refund_guardrail(-5) == "DENY"
    assert refund_guardrail(20) == "ALLOW"
    assert refund_guardrail(500) == "ESCALATE"
    assert refund_guardrail(5000) == "DENY"
    sw = {"refund-bot": "active"}
    assert can_run(sw, "refund-bot") is True
    sw["refund-bot"] = "killed"
    assert can_run(sw, "refund-bot") is False
    assert can_run(sw, "ghost-bot") is False  # unknown -> fail-safe
    print("[EASY]   autonomy levels, action rail, fail-safe kill-switch: OK")

    # ---- MEDIUM ----
    b = RollingBudget(max_cost=100.0, max_actions=3, window_seconds=60.0)
    for _ in range(3):
        b.record(30.0, now=0.0)
    assert b.would_exceed(30.0, now=10.0) is True   # action-count cap hit
    assert b.would_exceed(30.0, now=70.0) is False  # window rolled, events expired
    # cost cap: fresh budget, 2 actions of 60 -> 3rd of 60 breaks max_cost=150
    b2 = RollingBudget(max_cost=150.0, max_actions=10, window_seconds=60.0)
    b2.record(60.0, now=0.0)
    b2.record(60.0, now=0.0)
    assert b2.would_exceed(60.0, now=0.0) is True   # 180 > 150

    sw2 = {"bot": "active"}
    bud = RollingBudget(max_cost=500.0, max_actions=5, window_seconds=3600.0)
    assert gate("bot", 20.0, 2, False, sw2, bud, now=0.0) == ("ALLOW", "ok")
    assert gate("bot", 1800.0, 5, True, sw2, bud, now=0.0)[0] == "ESCALATE"  # risk
    sw2["bot"] = "killed"
    assert gate("bot", 20.0, 2, False, sw2, bud, now=0.0) == ("DENY", "kill-switch")
    # budget escalation path on a fresh, active bot
    sw3 = {"bot": "active"}
    tiny = RollingBudget(max_cost=10.0, max_actions=1, window_seconds=60.0)
    tiny.record(10.0, now=0.0)
    assert gate("bot", 5.0, 1, False, sw3, tiny, now=0.0) == ("ESCALATE", "budget")

    inc = Incident("bot", "anomaly")
    for note in ("contain", "eradicate", "recover", "close"):
        inc.advance(note)
    assert inc.phase == IncidentPhase.CLOSED
    try:
        inc.advance("again")
        raise AssertionError("should have raised on CLOSED")
    except ValueError:
        pass
    assert len(inc.history) == 5  # OPEN + 4 transitions
    print("[MEDIUM] rolling budget, combined gate, incident state machine: OK")

    # ---- HARD ----
    g = GateWithApprovals({"bot": "active"},
                          {"bot": RollingBudget(50.0, 10, 3600.0)})
    # big irreversible action -> escalate -> queued
    d, _ = g.evaluate("bot", "refund(1800)", 1800.0, 5, True, now=0.0)
    assert d == "ESCALATE" and len(g.pending) == 1
    before, _ = g.budgets["bot"].total(now=0.0)
    # approving fails here (1800 > budget 50) -> DENY, not counted
    assert g.approve(0, now=0.0)[0] == "DENY"
    after, _ = g.budgets["bot"].total(now=0.0)
    assert after == before  # rejected by budget, nothing recorded

    # an affordable escalation (by risk) that approve() can accept
    g.evaluate("bot", "refund(40)", 40.0, 5, False, now=0.0)  # risk=5 -> escalate
    assert len(g.pending) == 1
    spent_before, _ = g.budgets["bot"].total(now=0.0)
    assert g.approve(0, now=0.0)[0] == "ALLOW"
    spent_after, _ = g.budgets["bot"].total(now=0.0)
    assert spent_after == spent_before + 40.0  # approved AND counted

    # reject path leaves the budget untouched
    g.evaluate("bot", "refund(45)", 45.0, 5, False, now=0.0)  # would exceed 50 -> budget escalate
    s_b, _ = g.budgets["bot"].total(now=0.0)
    assert g.reject(0, "policy")[0] == "DENY"
    s_a, _ = g.budgets["bot"].total(now=0.0)
    assert s_a == s_b

    # incident orchestration
    g2 = GateWithApprovals({"bot": "active"}, {"bot": RollingBudget(500.0, 10, 3600.0)})
    inc2 = handle_incident(g2, "bot", "6 refunds same IBAN")
    assert inc2.phase == IncidentPhase.CLOSED
    # after recover: agent active again but HITL-forced -> sensitive action escalates
    d2, r2 = g2.evaluate("bot", "refund(10)", 10.0, 1, False, now=0.0)
    assert (d2, r2) == ("ESCALATE", "hitl-forced")

    # decommission gate
    ok, miss = try_decommission({"state": "SUSPENDED", "revoked_access": False,
                                 "audit_archived": True, "owner_closed": True})
    assert ok is False and any("zombie" in m for m in miss)
    ok, miss = try_decommission({"state": "PROPOSED", "revoked_access": True,
                                 "audit_archived": True, "owner_closed": True})
    assert ok is False and any("bad source state" in m for m in miss)
    agent = {"state": "SUSPENDED", "revoked_access": True,
             "audit_archived": True, "owner_closed": True}
    ok, miss = try_decommission(agent)
    assert ok is True and agent["state"] == "DECOMMISSIONED"
    print("[HARD]   approval queue, incident orchestration, decommission gate: OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()

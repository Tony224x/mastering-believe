"""
Day 10 — Autonomy, guardrails & operations: a runnable AUTONOMY GATE.

What this script demonstrates (governance mechanism of the day):
  Every action an agent wants to take is routed through ONE gate that, in order:
    1. checks a KILL-SWITCH  (external, fail-safe state the agent cannot override)
    2. enforces BUDGETS      (cost / action-count caps over a time window)
    3. calibrates AUTONOMY    (risk of the action -> ALLOW / ESCALATE / DENY)
  The gate returns a typed decision and never lets the agent decide its own limits.

It also models INCIDENT RESPONSE (detect -> contain -> eradicate -> recover) and a
clean DECOMMISSION as small state machines, because the day couples autonomy with
operations (incident + lifecycle).

Real tools this mimics in stdlib (cited, not imported):
  - Action guardrails        -> NVIDIA NeMo Guardrails / Guardrails AI (runtime rails)
  - Risk-calibrated controls -> EY "Six Steps to Enhance Agentic AI Governance" (2026)
  - "Humans remain ultimately accountable" -> IMDA Model AI Gov. for Agentic AI (2026)

# requires: stdlib only
Run:
    python 10-autonomie-gardefous.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# DECISIONS — the gate speaks a tiny, explicit vocabulary. WHY: a boolean
# allow/deny is too coarse; "ESCALATE" (route to a human) is the third option
# that keeps the service alive while raising supervision (soft cap, HITL).
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    ALLOW = "ALLOW"        # action proceeds as-is
    ESCALATE = "ESCALATE"  # action held for human approval (transition to HITL)
    DENY = "DENY"          # action blocked outright (and logged)


# ---------------------------------------------------------------------------
# KILL-SWITCH — an EXTERNAL state the agent reads but cannot write. WHY: if the
# switch were an attribute the agent could set, it would not be a kill-switch.
# Fail-safe: any unknown / unreadable status is treated as "killed" (default deny).
# ---------------------------------------------------------------------------


class AgentState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"  # temporarily on hold (e.g. during containment)
    KILLED = "killed"  # hard stop until a human re-activates


class KillSwitch:
    """Shared, external switch. The autonomy gate consults it FIRST."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}

    def register(self, agent_id: str) -> None:
        self._states.setdefault(agent_id, AgentState.ACTIVE)

    def set_state(self, agent_id: str, state: AgentState) -> None:
        # In a real system this is an audited admin action (who/when/why).
        self._states[agent_id] = state

    def can_run(self, agent_id: str) -> bool:
        # Fail-safe: unknown agent or non-ACTIVE state -> cannot run.
        return self._states.get(agent_id, AgentState.KILLED) is AgentState.ACTIVE


# ---------------------------------------------------------------------------
# BUDGETS — cap the CUMULATIVE risk that per-action guardrails never see.
# WHY: an incident is rarely one huge action; it is many small ones. We track
# spend and action count per agent over a rolling window.
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    max_cost: float          # currency cap over the window (e.g. 500.0 / day)
    max_actions: int         # number of sensitive actions over the window
    window_seconds: float = 86400.0  # rolling window length (default: 1 day)
    # internal counters (timestamped so the window can roll)
    _events: list[tuple[float, float]] = field(default_factory=list)  # (ts, cost)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._events = [(ts, c) for (ts, c) in self._events if ts >= cutoff]

    def would_exceed(self, cost: float, now: float | None = None) -> bool:
        """Check BEFORE acting whether adding this action breaks a cap."""
        now = time.time() if now is None else now
        self._prune(now)
        spent = sum(c for _, c in self._events)
        count = len(self._events)
        return (spent + cost > self.max_cost) or (count + 1 > self.max_actions)

    def record(self, cost: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._events.append((now, cost))

    def snapshot(self, now: float | None = None) -> tuple[float, int]:
        now = time.time() if now is None else now
        self._prune(now)
        return sum(c for _, c in self._events), len(self._events)


# ---------------------------------------------------------------------------
# AUTONOMY POLICY — maps an action's RISK to the required human involvement.
# WHY: autonomy is a per-ACTION dial, not a per-agent constant. The more
# irreversible / high-impact, the more the human must be upstream (in-the-loop).
# ---------------------------------------------------------------------------


@dataclass
class ProposedAction:
    agent_id: str
    name: str              # e.g. "issue_refund", "read_ticket"
    cost: float = 0.0      # monetary impact (for budget); 0 for read-only
    irreversible: bool = False
    impact: int = 1        # 1=low .. 5=high (see J4 risk scoring)


def risk_score(action: ProposedAction) -> int:
    """Tiny risk proxy: impact, bumped if the action is irreversible.

    Real version: likelihood x impact from the J4 risk register.
    """
    score = action.impact
    if action.irreversible:
        score += 2  # irreversibility is the multiplier that matters most
    return score


# Risk thresholds: low risk -> the agent may act alone (out-of-the-loop);
# medium -> on-the-loop is fine here too; high -> a human must approve (HITL).
ESCALATE_THRESHOLD = 5  # at/above this score, require human approval


@dataclass
class GateResult:
    decision: Decision
    reason: str


class AutonomyGate:
    """The single choke point every agent action must pass through."""

    def __init__(self, kill: KillSwitch, budgets: dict[str, Budget]) -> None:
        self.kill = kill
        self.budgets = budgets
        self.audit: list[dict] = []  # append-only trail (see J9 for tamper-evidence)

    def evaluate(self, action: ProposedAction, now: float | None = None) -> GateResult:
        now = time.time() if now is None else now
        aid = action.agent_id

        # 1) KILL-SWITCH FIRST — nothing else matters if the agent is stopped.
        if not self.kill.can_run(aid):
            return self._log(action, Decision.DENY, "kill-switch: agent not ACTIVE", now)

        # 2) BUDGET — cap the cumulative spend / action count.
        budget = self.budgets.get(aid)
        if budget and budget.would_exceed(action.cost, now):
            # Soft cap: escalate rather than hard-block, so the service survives
            # while supervision rises (EY: calibrate controls to risk).
            return self._log(action, Decision.ESCALATE, "budget exceeded -> human approval", now)

        # 3) AUTONOMY — calibrate to the action's risk.
        score = risk_score(action)
        if score >= ESCALATE_THRESHOLD:
            return self._log(action, Decision.ESCALATE, f"risk={score} >= {ESCALATE_THRESHOLD} -> HITL", now)

        # All checks passed: allow and record the spend against the budget.
        if budget:
            budget.record(action.cost, now)
        return self._log(action, Decision.ALLOW, f"risk={score} below threshold", now)

    def _log(self, action: ProposedAction, decision: Decision, reason: str, now: float) -> GateResult:
        self.audit.append(
            {"ts": now, "agent": action.agent_id, "action": action.name,
             "cost": action.cost, "decision": decision.value, "reason": reason}
        )
        return GateResult(decision, reason)


# ---------------------------------------------------------------------------
# INCIDENT RESPONSE — detect -> contain -> eradicate -> recover, as a state
# machine with FORBIDDEN transitions. WHY: you must CONTAIN before you eradicate
# (stop the bleeding first); you cannot jump OPEN -> CLOSED.
# ---------------------------------------------------------------------------


class IncidentPhase(str, Enum):
    OPEN = "OPEN"
    CONTAINED = "CONTAINED"
    ERADICATED = "ERADICATED"
    RECOVERED = "RECOVERED"
    CLOSED = "CLOSED"


# Allowed forward transitions only (no skipping containment).
_INCIDENT_FLOW: dict[IncidentPhase, IncidentPhase] = {
    IncidentPhase.OPEN: IncidentPhase.CONTAINED,
    IncidentPhase.CONTAINED: IncidentPhase.ERADICATED,
    IncidentPhase.ERADICATED: IncidentPhase.RECOVERED,
    IncidentPhase.RECOVERED: IncidentPhase.CLOSED,
}


class Incident:
    def __init__(self, agent_id: str, trigger: str) -> None:
        self.agent_id = agent_id
        self.trigger = trigger
        self.phase = IncidentPhase.OPEN
        self.history: list[tuple[IncidentPhase, str]] = [(IncidentPhase.OPEN, trigger)]

    def advance(self, note: str) -> None:
        nxt = _INCIDENT_FLOW.get(self.phase)
        if nxt is None:
            raise ValueError(f"no transition out of {self.phase.value}")
        self.phase = nxt
        self.history.append((nxt, note))


# ---------------------------------------------------------------------------
# DECOMMISSION — lifecycle end is a CHECKLIST, not just "turn it off". WHY: an
# inactive agent with live scopes is a zombie identity. We model the lifecycle
# state plus a checklist that must be fully satisfied to reach DECOMMISSIONED.
# ---------------------------------------------------------------------------


class LifecycleState(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DECOMMISSIONED = "DECOMMISSIONED"


def decommission_checklist(revoked_access: bool, audit_archived: bool,
                           owner_closed: bool) -> tuple[bool, list[str]]:
    """Return (ok, missing). All three must hold for a clean decommission."""
    missing = []
    if not revoked_access:
        missing.append("access/scopes/secrets not revoked (zombie identity risk)")
    if not audit_archived:
        missing.append("audit trail not archived (retention breach)")
    if not owner_closed:
        missing.append("owner not reassigned/closed")
    return (len(missing) == 0, missing)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------


def _print_decision(label: str, result: GateResult) -> None:
    print(f"  {label:<46} -> {result.decision.value:<8} ({result.reason})")


def main() -> None:
    print("=" * 70)
    print("AUTONOMY GATE — kill-switch + budget + risk-calibrated autonomy")
    print("=" * 70)

    kill = KillSwitch()
    kill.register("refund-bot")
    kill.register("risk-demo-bot")
    # Budget: 500 currency / 5 sensitive actions, over a short demo window.
    # A second agent gets a huge budget so step [3] isolates the RISK path
    # (no budget interference) — each demo step shows ONE mechanism.
    budgets = {
        "refund-bot": Budget(max_cost=500.0, max_actions=5, window_seconds=3600.0),
        "risk-demo-bot": Budget(max_cost=1_000_000.0, max_actions=1000, window_seconds=3600.0),
    }
    gate = AutonomyGate(kill, budgets)

    t0 = 1_000_000.0  # fixed clock so the demo is deterministic (no wall time)

    print("\n[1] Low-risk read action (out-of-the-loop is fine):")
    a = ProposedAction("refund-bot", "read_ticket", cost=0.0, impact=1)
    _print_decision("read_ticket (impact=1)", gate.evaluate(a, now=t0))

    print("\n[2] Small refund, reversible, low impact -> ALLOW:")
    a = ProposedAction("refund-bot", "issue_refund(20)", cost=20.0, impact=2)
    _print_decision("issue_refund 20 (impact=2)", gate.evaluate(a, now=t0))

    print("\n[3] Large irreversible refund -> high RISK -> ESCALATE to a human")
    print("    (budget is huge here, so the RISK path is what triggers it):")
    a = ProposedAction("risk-demo-bot", "issue_refund(1800)", cost=1800.0,
                       irreversible=True, impact=5)
    _print_decision("issue_refund 1800 (irreversible)", gate.evaluate(a, now=t0))

    print("\n[4] Budget cap — fire small refunds until the count cap (5) is hit:")
    for i in range(6):
        a = ProposedAction("refund-bot", f"issue_refund(10)#{i}", cost=10.0, impact=2)
        _print_decision(f"refund #{i}", gate.evaluate(a, now=t0))

    print("\n[5] KILL-SWITCH engaged — every action is denied, fail-safe:")
    kill.set_state("refund-bot", AgentState.KILLED)
    a = ProposedAction("refund-bot", "issue_refund(5)", cost=5.0, impact=1)
    _print_decision("issue_refund 5 while KILLED", gate.evaluate(a, now=t0))

    spent, count = budgets["refund-bot"].snapshot(now=t0)
    print(f"\n  Budget window snapshot: spent={spent:.0f} / 500, actions={count} / 5")

    # ---- Incident response -------------------------------------------------
    print("\n" + "=" * 70)
    print("INCIDENT RESPONSE — detect -> contain -> eradicate -> recover")
    print("=" * 70)
    inc = Incident("refund-bot", trigger="anomaly: 6 refunds to same IBAN")
    print(f"  detect : {inc.phase.value} ({inc.trigger})")
    inc.advance("kill-switch engaged, scopes revoked")   # contain
    print(f"  contain: {inc.phase.value}")
    inc.advance("patched prompt-injection in email parser")  # eradicate
    print(f"  eradicate: {inc.phase.value}")
    inc.advance("re-activated at HITL-forced for 48h")    # recover
    print(f"  recover: {inc.phase.value}")
    inc.advance("post-mortem filed, guardrail added")     # close
    print(f"  close  : {inc.phase.value}")

    print("\n  Adversarial probe — try to skip containment (OPEN -> CLOSED):")
    bad = Incident("x", "test")
    try:
        bad.phase = IncidentPhase.CLOSED  # force, then attempt an illegal advance
        bad.advance("illegal")
    except ValueError as exc:
        print(f"    blocked: {exc}")

    # ---- Decommission ------------------------------------------------------
    print("\n" + "=" * 70)
    print("DECOMMISSION — lifecycle end is a checklist, not just 'off'")
    print("=" * 70)
    ok, missing = decommission_checklist(revoked_access=False, audit_archived=True,
                                          owner_closed=True)
    print(f"  Attempt with access still live -> clean={ok}")
    for m in missing:
        print(f"    MISSING: {m}")
    ok, missing = decommission_checklist(revoked_access=True, audit_archived=True,
                                         owner_closed=True)
    print(f"  Attempt with everything revoked -> clean={ok} "
          f"-> state can move to {LifecycleState.DECOMMISSIONED.value}")

    print("\nDone. Autonomy is a dial; the human keeps the ultimate control.")


if __name__ == "__main__":
    main()

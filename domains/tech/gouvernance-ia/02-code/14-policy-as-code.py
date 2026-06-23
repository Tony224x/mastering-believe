"""
Day 14 -- Policy-as-code & runtime enforcement, from scratch in stdlib.

Demonstrates the governance mechanism of the day:
  1. A declarative policy engine (PDP -- Policy Decision Point): rules are
     pure functions (action, agent, context) -> Decision, composed with a
     SAFETY precedence  deny > oblige > allow  (mirrors OPA's `default allow=false`,
     re-implemented here in stdlib; the real tool is Open Policy Agent / Rego, CNCF).
  2. A PolicyEnforcementPoint (PEP) that intercepts every action BEFORE it runs,
     asks the PDP, and applies the verdict (allow / deny / oblige=suspend).
  3. An MCP-style permission gate: a `tools/call` dispatcher that requires
     explicit consent + scope before executing a tool -- modelling the MCP
     Specification "Security & Trust" section (consent, Tool Safety, per-tool
     permissions). Real MCP is JSON-RPC over stdio/HTTP; we keep the permission
     model, not the transport.

WHY this matters: an eval that passes (Day 13) proves nothing in production if
nothing stops the agent at the instant it acts. Policy-as-code is the runtime
guarantee: the rule is a declarative, versioned, testable artifact, evaluated
on EVERY action -- and no path to the action may bypass the PEP.

# requires: stdlib only

Run:
    python domains/tech/gouvernance-ia/02-code/14-policy-as-code.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable


# ===========================================================================
# 1. CORE TYPES -- the shared mini agent model (id/owner/scopes/risk_tier),
#    reused across the domain's fil-rouge.
# ===========================================================================

class Verdict(IntEnum):
    """Three verdicts, ordered by SAFETY precedence: higher wins on conflict.

    WHY an ordering: when several rules fire, we must collapse them into ONE
    decision deterministically. Safety-first means a single `deny` overrides
    everything, and any `oblige` overrides a plain `allow`.
    """
    ALLOW = 0
    OBLIGE = 1   # conditional: action suspended until obligation is satisfied
    DENY = 2


@dataclass(frozen=True)
class Agent:
    """The unit we govern. Minimal on purpose (the 4 pillars in spirit)."""
    id: str
    owner: str                       # named human accountable for the agent
    scopes: tuple[str, ...]          # least-privilege grants (cf. Day 8)
    risk_tier: str                   # "low" | "medium" | "high"


@dataclass(frozen=True)
class Action:
    """An action an agent wants to take: a tool call with parameters."""
    tool: str
    params: dict
    required_scope: str              # scope this tool needs to run


@dataclass
class Decision:
    """The PDP's answer for one rule (and, after merge, for the whole policy)."""
    verdict: Verdict
    rule: str                        # which rule produced this verdict
    reason: str
    obligation: str | None = None    # e.g. "human_approval" when verdict==OBLIGE


# A rule is a pure function. Returning None means "this rule does not apply".
Rule = Callable[[Action, Agent, dict], Decision | None]


# ===========================================================================
# 2. POLICY DECISION POINT (PDP) -- declarative rules + safety merge
# ===========================================================================

def rule_scope(action: Action, agent: Agent, ctx: dict) -> Decision | None:
    """Least privilege: the action's required scope must be granted to the agent.

    WHY deny (not oblige): missing a scope is a hard authorization failure,
    never something a human approval can paper over at call time.
    """
    if action.required_scope not in agent.scopes:
        return Decision(
            Verdict.DENY,
            "scope",
            f"agent {agent.id} lacks scope '{action.required_scope}'",
        )
    return None


def rule_refund_cap(action: Action, agent: Agent, ctx: dict) -> Decision | None:
    """Budget rule: refunds over 1000 require human approval (the concrete
    example from the theory). Note this is OBLIGE, not DENY: the action is
    legitimate, it just needs a second pair of (human) eyes.
    """
    if action.tool == "issue_refund" and action.params.get("amount", 0) > 1000:
        return Decision(
            Verdict.OBLIGE,
            "refund_cap",
            f"refund of {action.params['amount']} exceeds 1000 auto-limit",
            obligation="human_approval",
        )
    return None


def rule_high_risk_irreversible(action: Action, agent: Agent, ctx: dict) -> Decision | None:
    """Autonomy rule: a high-risk-tier agent performing an irreversible action
    must escalate to a human (cf. Days 4 & 10 -- autonomy calibrated on risk).
    """
    irreversible = ctx.get("irreversible_tools", set())
    if agent.risk_tier == "high" and action.tool in irreversible:
        return Decision(
            Verdict.OBLIGE,
            "high_risk_irreversible",
            f"high-tier agent attempting irreversible tool '{action.tool}'",
            obligation="human_approval",
        )
    return None


def rule_pii_egress(action: Action, agent: Agent, ctx: dict) -> Decision | None:
    """Data rule (cf. Day 6 / RGPD): block exporting personal data outside the
    allowed perimeter. Hard deny -- there is no acceptable auto-exfiltration.
    """
    if action.params.get("contains_pii") and action.params.get("destination") == "external":
        return Decision(
            Verdict.DENY,
            "pii_egress",
            "action would export personal data to an external destination",
        )
    return None


class PolicyDecisionPoint:
    """The 'brain': evaluates a versioned set of rules and merges them with
    SAFETY precedence (deny > oblige > allow), i.e. OPA's `default allow=false`.
    """

    def __init__(self, rules: list[Rule], version: str) -> None:
        self.rules = rules
        self.version = version

    def evaluate(self, action: Action, agent: Agent, ctx: dict) -> Decision:
        # Collect every rule that fired (returned a Decision, not None).
        fired = [d for r in self.rules if (d := r(action, agent, ctx)) is not None]
        if not fired:
            # Nothing forbade or constrained the action -> it is allowed.
            return Decision(Verdict.ALLOW, "default", "no rule objected")
        # Safety merge: the highest-severity verdict wins. max() works because
        # Verdict is an IntEnum ordered ALLOW < OBLIGE < DENY.
        return max(fired, key=lambda d: d.verdict)


# ===========================================================================
# 3. POLICY ENFORCEMENT POINT (PEP) -- intercepts and applies the verdict
# ===========================================================================

@dataclass
class EnforcementResult:
    executed: bool
    decision: Decision
    output: str | None = None


class PolicyEnforcementPoint:
    """The 'muscle': sits on the path of EVERY action. It calls the PDP and
    applies the verdict. The whole point is that the tool cannot run without
    passing through here -- no bypass path allowed.
    """

    def __init__(self, pdp: PolicyDecisionPoint) -> None:
        self.pdp = pdp
        self.audit: list[dict] = []   # append-only trace (cf. Day 9)

    def enforce(
        self,
        action: Action,
        agent: Agent,
        ctx: dict,
        run_tool: Callable[[Action], str],
        human_approves: bool = False,
    ) -> EnforcementResult:
        decision = self.pdp.evaluate(action, agent, ctx)
        executed = False
        output = None

        if decision.verdict == Verdict.ALLOW:
            output = run_tool(action)
            executed = True
        elif decision.verdict == Verdict.OBLIGE:
            # Obligation: the action is SUSPENDED until satisfied. Here the
            # obligation is human approval; in the demo we pass it explicitly.
            if decision.obligation == "human_approval" and human_approves:
                output = run_tool(action)
                executed = True
        # DENY (or unsatisfied OBLIGE) -> executed stays False, tool never runs.

        self.audit.append({
            "agent": agent.id,
            "tool": action.tool,
            "verdict": decision.verdict.name,
            "rule": decision.rule,
            "executed": executed,
            "policy_version": self.pdp.version,
        })
        return EnforcementResult(executed, decision, output)


# ===========================================================================
# 4. MCP-STYLE PERMISSION GATE -- consent + scope before tools/call
#    (models the MCP spec "Security & Trust" section; real MCP = JSON-RPC)
# ===========================================================================

class MCPPermissionGate:
    """A minimal 'tools/call' dispatcher that, like the MCP Security & Trust
    section, requires EXPLICIT consent for sensitive tools before executing.
    It wraps the PEP, so MCP becomes the single choke point for tool actions.
    """

    def __init__(self, pep: PolicyEnforcementPoint, consent_required: set[str]) -> None:
        self.pep = pep
        self.consent_required = consent_required   # tools needing explicit consent

    def call_tool(
        self,
        action: Action,
        agent: Agent,
        ctx: dict,
        run_tool: Callable[[Action], str],
        user_consented: bool = False,
        human_approves: bool = False,
    ) -> EnforcementResult:
        # MCP principle #1: explicit user consent before invoking a sensitive tool.
        if action.tool in self.consent_required and not user_consented:
            decision = Decision(
                Verdict.DENY, "mcp_consent",
                f"tool '{action.tool}' requires explicit user consent (not given)",
            )
            self.pep.audit.append({
                "agent": agent.id, "tool": action.tool,
                "verdict": "DENY", "rule": "mcp_consent",
                "executed": False, "policy_version": self.pep.pdp.version,
            })
            return EnforcementResult(False, decision)
        # Consent OK -> delegate to the PEP, which applies the policy.
        return self.pep.enforce(action, agent, ctx, run_tool, human_approves)


# ===========================================================================
# 5. DEMO
# ===========================================================================

def _fake_tool(action: Action) -> str:
    """Stand-in for a real side-effecting tool call (refund API, DB export...)."""
    return f"OK: executed {action.tool}({action.params})"


def _build_pep() -> PolicyEnforcementPoint:
    pdp = PolicyDecisionPoint(
        rules=[rule_scope, rule_refund_cap, rule_high_risk_irreversible, rule_pii_egress],
        version="policy-v1.3.0",
    )
    return PolicyEnforcementPoint(pdp)


def main() -> None:
    pep = _build_pep()
    gate = MCPPermissionGate(pep, consent_required={"issue_refund", "export_data"})

    support = Agent(id="agent-support-01", owner="alice@ops",
                    scopes=("refund:execute", "ticket:read"), risk_tier="high")
    ctx = {"irreversible_tools": {"issue_refund", "delete_account"}}

    print("=" * 72)
    print("J14 -- Policy-as-code & runtime enforcement (PDP / PEP / MCP gate)")
    print(f"Policy version: {pep.pdp.version}")
    print("=" * 72)

    scenarios = [
        ("Ticket lookup, in scope, non-sensitive -> ALLOW (clean pass)",
         Action("ticket_lookup", {}, "ticket:read"),
         dict(user_consented=True, human_approves=False)),
        ("Small refund (in scope) -> OBLIGE: high-tier agent + irreversible tool",
         Action("issue_refund", {"amount": 50}, "refund:execute"),
         dict(user_consented=True, human_approves=True)),
        ("Big refund 12000 -> OBLIGE human approval (denied, no approval)",
         Action("issue_refund", {"amount": 12000}, "refund:execute"),
         dict(user_consented=True, human_approves=False)),
        ("Big refund 12000 -> OBLIGE satisfied (human approves)",
         Action("issue_refund", {"amount": 12000}, "refund:execute"),
         dict(user_consented=True, human_approves=True)),
        ("Refund without MCP consent -> DENY at the gate",
         Action("issue_refund", {"amount": 50}, "refund:execute"),
         dict(user_consented=False, human_approves=False)),
        ("Tool missing required scope -> DENY",
         Action("delete_account", {}, "account:delete"),
         dict(user_consented=True, human_approves=True)),
        ("PII export to external destination -> DENY (RGPD)",
         Action("export_data", {"contains_pii": True, "destination": "external"}, "ticket:read"),
         dict(user_consented=True, human_approves=True)),
    ]

    for label, action, kwargs in scenarios:
        res = gate.call_tool(action, support, ctx, _fake_tool, **kwargs)
        d = res.decision
        status = "EXECUTED" if res.executed else "BLOCKED "
        oblig = f" (obligation: {d.obligation})" if d.obligation else ""
        print(f"\n* {label}")
        print(f"    verdict = {d.verdict.name:7s} rule = {d.rule}{oblig}")
        print(f"    -> {status} | {d.reason}")

    # Audit trail: append-only proof of every decision (machine evidence, J9).
    print("\n" + "-" * 72)
    print("AUDIT TRAIL (append-only -- one row per attempted action):")
    print(f"{'agent':18s} {'tool':16s} {'verdict':8s} {'rule':22s} executed")
    for row in pep.audit:
        print(f"{row['agent']:18s} {row['tool']:16s} {row['verdict']:8s} "
              f"{row['rule']:22s} {row['executed']}")

    # Governance metric: enforcement coverage = blocked / total risky attempts.
    blocked = sum(1 for r in pep.audit if not r["executed"])
    print("-" * 72)
    print(f"Attempts: {len(pep.audit)} | Blocked by policy: {blocked} | "
          f"Allowed: {len(pep.audit) - blocked}")


if __name__ == "__main__":
    main()

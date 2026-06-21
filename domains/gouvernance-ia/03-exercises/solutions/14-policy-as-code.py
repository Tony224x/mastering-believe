"""
Day 14 -- Solutions: policy-as-code & runtime enforcement (easy / medium / hard).

One file, three sections separated by  # === EASY / MEDIUM / HARD ===  and a
smoke test in __main__ that exercises all three.

# requires: stdlib only

Run:
    python domains/gouvernance-ia/03-exercises/solutions/14-policy-as-code.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable


# ===========================================================================
# === EASY ===
# Two pure rules returning "allow" / "deny" / "oblige", evaluated on a few
# actions. No engine yet -- just the building blocks.
# ===========================================================================

@dataclass(frozen=True)
class SimpleAgent:
    id: str
    scopes: tuple[str, ...]
    risk_tier: str


@dataclass(frozen=True)
class SimpleAction:
    tool: str
    params: dict
    required_scope: str


def rule_scope_easy(action: SimpleAction, agent: SimpleAgent) -> str:
    """Deny if the agent lacks the scope the action requires; else allow."""
    if action.required_scope not in agent.scopes:
        return "deny"
    return "allow"


def rule_refund_cap_easy(action: SimpleAction, agent: SimpleAgent) -> str:
    # WHY oblige and not deny: a large refund is a LEGITIMATE action; it merely
    # needs human approval. Deny would forbid a valid business operation.
    if action.tool == "issue_refund" and action.params.get("amount", 0) > 1000:
        return "oblige"
    return "allow"


def demo_easy() -> None:
    agent = SimpleAgent("agent-support-01", ("refund:execute", "ticket:read"), "high")
    actions = [
        SimpleAction("issue_refund", {"amount": 50}, "refund:execute"),     # allow/allow
        SimpleAction("issue_refund", {"amount": 12000}, "refund:execute"),  # allow/oblige
        SimpleAction("delete_account", {}, "account:delete"),               # deny/allow
        SimpleAction("ticket_lookup", {}, "ticket:read"),                   # allow/allow
    ]
    print("[EASY] rule_scope / rule_refund_cap verdicts:")
    for a in actions:
        print(f"  {a.tool:14s} amount={a.params.get('amount', '-'):>6} -> "
              f"scope={rule_scope_easy(a, agent):5s} cap={rule_refund_cap_easy(a, agent)}")


# ===========================================================================
# === MEDIUM ===
# A PDP merging rules with safety precedence (deny > oblige > allow) and a PEP
# that enforces the verdict and keeps an append-only audit log.
# ===========================================================================

class Verdict(IntEnum):
    ALLOW = 0
    OBLIGE = 1
    DENY = 2


@dataclass
class Decision:
    verdict: Verdict
    rule: str
    reason: str
    obligation: str | None = None


Rule = Callable[["SimpleAction", "SimpleAgent", dict], "Decision | None"]


def rule_scope(action: SimpleAction, agent: SimpleAgent, ctx: dict) -> Decision | None:
    if action.required_scope not in agent.scopes:
        return Decision(Verdict.DENY, "scope",
                        f"missing scope '{action.required_scope}'")
    return None


def rule_refund_cap(action: SimpleAction, agent: SimpleAgent, ctx: dict) -> Decision | None:
    if action.tool == "issue_refund" and action.params.get("amount", 0) > 1000:
        return Decision(Verdict.OBLIGE, "refund_cap",
                        f"refund {action.params['amount']} > 1000",
                        obligation="human_approval")
    return None


def rule_high_risk_irreversible(action: SimpleAction, agent: SimpleAgent, ctx: dict) -> Decision | None:
    if agent.risk_tier == "high" and action.tool in ctx.get("irreversible_tools", set()):
        return Decision(Verdict.OBLIGE, "high_risk_irreversible",
                        f"high-tier agent on irreversible '{action.tool}'",
                        obligation="human_approval")
    return None


class PolicyDecisionPoint:
    """The brain: collect fired rules, merge with safety precedence."""

    def __init__(self, rules: list[Rule], version: str = "v1") -> None:
        self.rules = rules
        self.version = version

    def evaluate(self, action: SimpleAction, agent: SimpleAgent, ctx: dict) -> Decision:
        fired = [d for r in self.rules if (d := r(action, agent, ctx)) is not None]
        if not fired:
            return Decision(Verdict.ALLOW, "default", "no rule objected")
        # max() works because Verdict is an IntEnum ordered ALLOW < OBLIGE < DENY.
        return max(fired, key=lambda d: d.verdict)


class PolicyEnforcementPoint:
    """The muscle: enforce the verdict and log every attempt (append-only)."""

    def __init__(self, pdp: PolicyDecisionPoint) -> None:
        self.pdp = pdp
        self.audit: list[dict] = []

    def enforce(self, action, agent, ctx, run_tool, human_approves=False):
        d = self.pdp.evaluate(action, agent, ctx)
        executed = False
        if d.verdict == Verdict.ALLOW:
            run_tool(action)
            executed = True
        elif d.verdict == Verdict.OBLIGE and human_approves:
            run_tool(action)
            executed = True
        self.audit.append({"agent": agent.id, "tool": action.tool,
                            "verdict": d.verdict.name, "rule": d.rule,
                            "executed": executed})
        return d, executed


def demo_medium() -> PolicyEnforcementPoint:
    pdp = PolicyDecisionPoint(
        [rule_scope, rule_refund_cap, rule_high_risk_irreversible], version="v1")
    pep = PolicyEnforcementPoint(pdp)
    agent = SimpleAgent("agent-support-01", ("refund:execute", "ticket:read"), "high")
    ctx = {"irreversible_tools": {"issue_refund", "delete_account"}}

    def run_tool(a):  # side-effecting stand-in
        return f"ran {a.tool}"

    cases = [
        (SimpleAction("issue_refund", {"amount": 50}, "refund:execute"), False),
        (SimpleAction("issue_refund", {"amount": 12000}, "refund:execute"), False),
        (SimpleAction("issue_refund", {"amount": 12000}, "refund:execute"), True),
        (SimpleAction("delete_account", {}, "account:delete"), True),
        (SimpleAction("ticket_lookup", {}, "ticket:read"), False),
    ]
    print("\n[MEDIUM] PDP/PEP enforcement:")
    for action, approves in cases:
        d, executed = pep.enforce(action, agent, ctx, run_tool, human_approves=approves)
        tag = "EXEC " if executed else "BLOCK"
        print(f"  {tag} {action.tool:14s} -> {d.verdict.name:7s} ({d.rule})")
    return pep


# ===========================================================================
# === HARD ===
# MCP-style permission gate (consent before tools/call) + policy test suite
# + drift detection between two policy versions.
# ===========================================================================

class MCPPermissionGate:
    """Models the MCP spec 'Security & Trust' section: explicit consent is
    required before invoking a sensitive tool. Wraps the PEP so MCP is the
    single choke point through which every tool action is governed.
    """

    def __init__(self, pep: PolicyEnforcementPoint, consent_required: set[str]) -> None:
        self.pep = pep
        self.consent_required = consent_required

    def call_tool(self, action, agent, ctx, run_tool,
                  user_consented=False, human_approves=False):
        if action.tool in self.consent_required and not user_consented:
            # MCP principle: no sensitive tool runs without explicit consent.
            self.pep.audit.append({"agent": agent.id, "tool": action.tool,
                                   "verdict": "DENY", "rule": "mcp_consent",
                                   "executed": False})
            return Decision(Verdict.DENY, "mcp_consent",
                            f"tool '{action.tool}' needs explicit consent"), False
        return self.pep.enforce(action, agent, ctx, run_tool, human_approves)


def run_policy_tests(pdp: PolicyDecisionPoint, cases: list[tuple]) -> tuple[int, int]:
    """Policy is code, so we test it: each case is (action, agent, ctx, expected
    verdict name). Returns (passed, failed) and prints any failing case.
    """
    passed = failed = 0
    for action, agent, ctx, expected in cases:
        got = pdp.evaluate(action, agent, ctx).verdict.name
        if got == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL {action.tool}: expected {expected}, got {got}")
    return passed, failed


def detect_drift(pdp_old: PolicyDecisionPoint, pdp_new: PolicyDecisionPoint,
                 actions: list[tuple]) -> list[dict]:
    """Flag any action whose verdict changes between two policy versions."""
    drift = []
    for action, agent, ctx in actions:
        old = pdp_old.evaluate(action, agent, ctx).verdict.name
        new = pdp_new.evaluate(action, agent, ctx).verdict.name
        if old != new:
            drift.append({"tool": action.tool, "amount": action.params.get("amount"),
                          "old": old, "new": new})
    return drift


def demo_hard() -> None:
    agent = SimpleAgent("agent-support-01", ("refund:execute", "ticket:read"), "high")
    ctx = {"irreversible_tools": {"issue_refund", "delete_account"}}
    pdp = PolicyDecisionPoint(
        [rule_scope, rule_refund_cap, rule_high_risk_irreversible], version="v1")
    pep = PolicyEnforcementPoint(pdp)
    gate = MCPPermissionGate(pep, consent_required={"issue_refund", "export_data"})

    def run_tool(a):
        return f"ran {a.tool}"

    print("\n[HARD] MCP permission gate:")
    # Sensitive tool without consent -> denied at the gate, tool never runs.
    d, executed = gate.call_tool(
        SimpleAction("issue_refund", {"amount": 50}, "refund:execute"),
        agent, ctx, run_tool, user_consented=False)
    print(f"  no consent  -> {d.verdict.name} ({d.rule}), executed={executed}")
    # With consent -> delegates to PEP (small refund auto-allowed).
    d, executed = gate.call_tool(
        SimpleAction("issue_refund", {"amount": 50}, "refund:execute"),
        agent, ctx, run_tool, user_consented=True)
    print(f"  consent     -> {d.verdict.name} ({d.rule}), executed={executed}")

    # Policy test suite (6 cases across allow/deny/oblige).
    cases = [
        (SimpleAction("ticket_lookup", {}, "ticket:read"), agent, ctx, "ALLOW"),
        (SimpleAction("issue_refund", {"amount": 50}, "refund:execute"), agent, ctx, "OBLIGE"),  # irreversible
        (SimpleAction("issue_refund", {"amount": 12000}, "refund:execute"), agent, ctx, "OBLIGE"),
        (SimpleAction("delete_account", {}, "account:delete"), agent, ctx, "DENY"),  # missing scope
        (SimpleAction("export_data", {}, "ticket:read"), agent, ctx, "ALLOW"),
        (SimpleAction("wire_transfer", {}, "wire:send"), agent, ctx, "DENY"),  # missing scope
    ]
    passed, failed = run_policy_tests(pdp, cases)
    print(f"  policy tests: {passed} passed, {failed} failed")

    # Drift: new policy lowers the refund cap from 1000 to 500.
    def rule_refund_cap_v2(action, agent, ctx):
        if action.tool == "issue_refund" and action.params.get("amount", 0) > 500:
            return Decision(Verdict.OBLIGE, "refund_cap", "refund > 500",
                            obligation="human_approval")
        return None

    # Use a non-irreversible tool ("micro_refund") so the cap rule is what drives
    # the verdict, making the drift visible without the autonomy rule masking it.
    def rule_refund_cap_micro(action, agent, ctx):
        if action.tool == "micro_refund" and action.params.get("amount", 0) > 1000:
            return Decision(Verdict.OBLIGE, "refund_cap", "refund > 1000",
                            obligation="human_approval")
        return None

    def rule_refund_cap_micro_v2(action, agent, ctx):
        if action.tool == "micro_refund" and action.params.get("amount", 0) > 500:
            return Decision(Verdict.OBLIGE, "refund_cap", "refund > 500",
                            obligation="human_approval")
        return None

    pdp_old = PolicyDecisionPoint([rule_scope, rule_refund_cap_micro], "v1")
    pdp_new = PolicyDecisionPoint([rule_scope, rule_refund_cap_micro_v2], "v2")
    drift_actions = [
        (SimpleAction("micro_refund", {"amount": 700}, "refund:execute"), agent, ctx),
        (SimpleAction("micro_refund", {"amount": 50}, "refund:execute"), agent, ctx),
    ]
    drift = detect_drift(pdp_old, pdp_new, drift_actions)
    print(f"  drift detected on {len(drift)} action(s):")
    for d_row in drift:
        print(f"    {d_row['tool']} amount={d_row['amount']}: "
              f"{d_row['old']} -> {d_row['new']}")


# ===========================================================================
# SMOKE TEST
# ===========================================================================

if __name__ == "__main__":
    demo_easy()
    demo_medium()
    demo_hard()

    # Minimal assertions so the smoke test actually verifies behaviour.
    a = SimpleAgent("a", ("refund:execute",), "low")
    pdp = PolicyDecisionPoint([rule_scope, rule_refund_cap], "v1")
    assert pdp.evaluate(SimpleAction("issue_refund", {"amount": 50}, "refund:execute"),
                        a, {}).verdict == Verdict.ALLOW
    assert pdp.evaluate(SimpleAction("issue_refund", {"amount": 5000}, "refund:execute"),
                        a, {}).verdict == Verdict.OBLIGE
    assert pdp.evaluate(SimpleAction("delete_account", {}, "account:delete"),
                        a, {}).verdict == Verdict.DENY
    print("\nAll smoke-test assertions passed.")

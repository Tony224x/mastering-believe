"""Day 15 — Solutions: capstone Agent Governance Toolkit.

One file covering the three exercise levels:

  # === EASY ===   ingest -> registry (4 pillars) -> coverage + orphans report
  # === MEDIUM === add ENFORCE (PDP/PEP) + LOG (hash-chained tamper-evident trail)
  # === HARD ===   full pipeline: MCP consent + SCORE + MAP (evidence-gated) + REPORT

Each level builds on the previous one. stdlib only.

# requires: stdlib only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Callable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# === EASY ===
# Ingest a raw fleet into governed agents; validate the 4 pillars; report
# coverage + orphans. The ossature of the toolkit (ingest -> report).

@dataclass
class GovernedAgent:
    agent_id: str
    owner: str | None
    scopes: tuple[str, ...]
    risk_tier: str = "low"
    autonomous: bool = False
    handles_irreversible: bool = False

    def is_fully_governed(self) -> tuple[bool, list[str]]:
        # Return the MISSING pillars so the report can name the gap, not just
        # assert a boolean.
        missing: list[str] = []
        if not self.agent_id:
            missing.append("identity")
        if not self.owner:
            missing.append("owner")        # orphan / shadow agent
        if not self.scopes:
            missing.append("permissions")
        return (len(missing) == 0, missing)


def ingest(raw_fleet: list[dict]) -> list[GovernedAgent]:
    # Tolerate missing keys (a real export is imperfect) but drop an agent with
    # no id at all (it cannot be addressed in the registry).
    agents: list[GovernedAgent] = []
    for raw in raw_fleet:
        agent_id = raw.get("agent_id", "")
        if not agent_id:
            continue
        agents.append(GovernedAgent(
            agent_id=agent_id,
            owner=raw.get("owner") or None,
            scopes=tuple(raw.get("scopes", ())),
            risk_tier=raw.get("risk_tier", "low"),
            autonomous=bool(raw.get("autonomous", False)),
            handles_irreversible=bool(raw.get("handles_irreversible", False)),
        ))
    return agents


def orphans(agents: list[GovernedAgent]) -> list[GovernedAgent]:
    return [a for a in agents if not a.owner]


def coverage(agents: list[GovernedAgent]) -> tuple[int, int, float]:
    total = len(agents)
    governed = sum(1 for a in agents if a.is_fully_governed()[0])
    pct = (governed / total * 100.0) if total else 0.0
    return governed, total, pct


def render_easy_report(org: str, agents: list[GovernedAgent]) -> str:
    g, t, pct = coverage(agents)
    orph = orphans(agents)
    lines = [f"GOVERNANCE COVERAGE — {org}",
             f"  agents: {t} | governed: {g} ({pct:.0f}%)",
             f"  orphans: {len(orph)} -> {[a.agent_id for a in orph]}"]
    return "\n".join(lines)


def _demo_easy() -> None:
    print("=== EASY: ingest -> coverage + orphans ===")
    fleet = [
        {"agent_id": "a1", "owner": "alice", "scopes": ["x:read"], "risk_tier": "low"},
        {"agent_id": "a2", "owner": "bob", "scopes": ["y:write"]},   # no risk_tier key
        {"agent_id": "a3", "owner": None, "scopes": ["z:fetch"]},    # orphan
        {"owner": "carol", "scopes": ["w:read"]},                    # no agent_id -> dropped
    ]
    agents = ingest(fleet)
    print(render_easy_report("Demo", agents))
    assert len(agents) == 3                       # the keyless one is dropped
    assert [a.agent_id for a in orphans(agents)] == ["a3"]
    g, t, pct = coverage(agents)
    assert (g, t) == (2, 3)                        # a3 is not fully governed
    ok, missing = agents[2].is_fully_governed()
    assert ok is False and "owner" in missing


# === MEDIUM ===
# Add ENFORCE (PDP/PEP, safety precedence) + LOG (hash-chained audit trail).

class Verdict(IntEnum):
    ALLOW = 0
    OBLIGE = 1
    DENY = 2


@dataclass(frozen=True)
class Action:
    tool: str
    params: dict
    required_scope: str


@dataclass
class Decision:
    verdict: Verdict
    rule: str
    reason: str
    obligation: str | None = None


Rule = Callable[[Action, GovernedAgent, dict], "Decision | None"]


def rule_scope(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    if action.required_scope not in agent.scopes:
        return Decision(Verdict.DENY, "scope",
                        f"agent {agent.agent_id} lacks '{action.required_scope}'")
    return None


def rule_budget(action: Action, agent: GovernedAgent, ctx: dict) -> Decision | None:
    limit = ctx.get("auto_amount_limit", 1000)
    if action.params.get("amount", 0) > limit:
        return Decision(Verdict.OBLIGE, "budget",
                        f"amount exceeds {limit}", obligation="human_approval")
    return None


def decide(rules: list[Rule], action: Action, agent: GovernedAgent, ctx: dict) -> Decision:
    fired = [d for r in rules if (d := r(action, agent, ctx)) is not None]
    if not fired:
        return Decision(Verdict.ALLOW, "default", "no rule objected")
    # Safety precedence: deny > oblige > allow (IntEnum order = severity).
    return max(fired, key=lambda d: d.verdict)


GENESIS = "GENESIS"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()


class AuditTrail:
    """Append-only hash chain: one silent edit is detected at its exact index."""

    def __init__(self) -> None:
        self._chain: list[dict] = []

    @property
    def head_hash(self) -> str:
        return self._chain[-1]["entry_hash"] if self._chain else GENESIS

    def record(self, payload: dict) -> None:
        prev = self.head_hash
        self._chain.append({
            "index": len(self._chain),
            "payload": payload,
            "prev_hash": prev,
            "entry_hash": _hash_entry(prev, payload),
        })

    def verify(self) -> tuple[bool, int | None]:
        prev = GENESIS
        for rec in self._chain:
            if rec["prev_hash"] != prev:
                return False, rec["index"]
            if _hash_entry(prev, rec["payload"]) != rec["entry_hash"]:
                return False, rec["index"]
            prev = rec["entry_hash"]
        return True, None

    def entries(self) -> list[dict]:
        return [rec["payload"] for rec in self._chain]


def enforce_all(rules: list[Rule], agents: list[GovernedAgent],
                attempts: list[dict], ctx: dict, trail: AuditTrail) -> int:
    by_id = {a.agent_id: a for a in agents}
    blocked = 0
    for att in attempts:
        agent = by_id.get(att["agent_id"])
        if agent is None:
            continue
        action = Action(att["tool"], att.get("params", {}), att["required_scope"])
        decision = decide(rules, action, agent, ctx)
        executed = (decision.verdict == Verdict.ALLOW) or (
            decision.verdict == Verdict.OBLIGE
            and decision.obligation == "human_approval"
            and att.get("human_approves", False))
        trail.record({
            "ts": _utc_now_iso(), "agent_id": agent.agent_id, "tool": action.tool,
            "verdict": decision.verdict.name, "rule": decision.rule, "executed": executed,
        })
        if not executed:
            blocked += 1
    return blocked


def _demo_medium() -> None:
    print("\n=== MEDIUM: enforce (PDP/PEP) + hash-chained audit ===")
    agents = ingest([
        {"agent_id": "fin", "owner": "alice",
         "scopes": ["payments:execute"], "risk_tier": "high"},
    ])
    rules = [rule_scope, rule_budget]
    ctx = {"auto_amount_limit": 1000}
    attempts = [
        {"agent_id": "fin", "tool": "transfer", "params": {"amount": 500},
         "required_scope": "payments:execute"},                       # ALLOW
        {"agent_id": "fin", "tool": "transfer", "params": {"amount": 40000},
         "required_scope": "payments:execute"},                       # OBLIGE, blocked
        {"agent_id": "fin", "tool": "transfer", "params": {"amount": 40000},
         "required_scope": "payments:execute", "human_approves": True},  # OBLIGE, runs
        {"agent_id": "fin", "tool": "delete", "params": {},
         "required_scope": "data:delete"},                            # DENY (scope)
    ]
    trail = AuditTrail()
    blocked = enforce_all(rules, agents, attempts, ctx, trail)
    for e in trail.entries():
        print(f"  {e['tool']:<9} {e['verdict']:<7} rule={e['rule']:<8} executed={e['executed']}")
    assert blocked == 2                                # the big-no-approval + the scope deny
    assert trail.verify() == (True, None)

    # Precedence check: deny AND oblige firing together -> deny wins.
    fin = agents[0]
    a = Action("transfer", {"amount": 40000}, "missing:scope")
    d = decide(rules, a, fin, ctx)
    assert d.verdict == Verdict.DENY and d.rule == "scope"

    # Tamper check: silently edit a stored verdict -> verify() catches it.
    trail._chain[3]["payload"]["verdict"] = "ALLOW"
    ok, broken = trail.verify()
    assert ok is False and broken == 3
    print(f"  tamper at index 3 -> verify = ({ok}, {broken})")


# === HARD ===
# Full pipeline: MCP consent gate + SCORE + MAP (evidence-gated) + REPORT.

class PolicyEngine:
    def __init__(self, rules: list[Rule], version: str, consent_required: set[str]) -> None:
        self.rules = rules
        self.version = version
        self.consent_required = consent_required

    def enforce(self, action: Action, agent: GovernedAgent, ctx: dict,
                user_consented: bool = False, human_approves: bool = False) -> tuple[Decision, bool]:
        # MCP Security & Trust: explicit consent before a sensitive tools/call.
        if action.tool in self.consent_required and not user_consented:
            return (Decision(Verdict.DENY, "mcp_consent",
                             f"tool '{action.tool}' needs explicit consent"), False)
        decision = decide(self.rules, action, agent, ctx)
        if decision.verdict == Verdict.ALLOW:
            return decision, True
        if (decision.verdict == Verdict.OBLIGE
                and decision.obligation == "human_approval" and human_approves):
            return decision, True
        return decision, False


# --- SCORE ---
LIKELIHOOD_BY_TIER = {"low": 2, "medium": 3, "high": 4}
IMPACT_BY_TIER = {"low": 2, "medium": 3, "high": 4}


@dataclass
class RiskScore:
    agent_id: str
    criticality: int
    decision: str
    rationale: list[str] = field(default_factory=list)


def score_agent(agent: GovernedAgent) -> RiskScore:
    likelihood = LIKELIHOOD_BY_TIER.get(agent.risk_tier, 2)
    impact = IMPACT_BY_TIER.get(agent.risk_tier, 2)
    rationale: list[str] = []
    if agent.handles_irreversible:
        impact = min(5, impact + 1)
        rationale.append("irreversible -> impact +1")
    if agent.autonomous:
        likelihood = min(5, likelihood + 1)
        rationale.append("autonomous -> likelihood +1")
    crit = likelihood * impact
    decision = "TREAT" if crit >= 12 else "MONITOR" if crit >= 6 else "ACCEPT"
    return RiskScore(agent.agent_id, crit, decision, rationale)


# --- MAP (crosswalk) ---
@dataclass(frozen=True)
class Requirement:
    framework: str
    ref: str
    label: str
    mandatory: bool


REQUIREMENTS: list[Requirement] = [
    Requirement("EU AI Act", "Art. 9", "Risk management", mandatory=True),
    Requirement("EU AI Act", "Art. 12", "Record-keeping", mandatory=True),
    Requirement("EU AI Act", "Art. 14", "Human oversight", mandatory=True),
    Requirement("NIST AI RMF", "Map", "Risk mapping", mandatory=False),
    Requirement("NIST AI RMF", "Manage", "Risk treatment", mandatory=False),
    Requirement("ISO/IEC 42001", "8.2", "Operational logging", mandatory=False),
]


def _key(r: Requirement) -> str:
    return f"{r.framework}::{r.ref}"


@dataclass
class Control:
    control_id: str
    covers: set[str] = field(default_factory=set)


def build_controls(audit_ok: bool, has_named_owner: bool) -> list[Control]:
    # Evidence-gated: a control only counts when its evidence holds live.
    controls = [Control("CTRL-RISK", {"EU AI Act::Art. 9", "NIST AI RMF::Map"})]
    if audit_ok:
        controls.append(Control("CTRL-AUDIT",
                                {"EU AI Act::Art. 12", "NIST AI RMF::Manage", "ISO/IEC 42001::8.2"}))
    if has_named_owner:
        controls.append(Control("CTRL-OWNER", {"EU AI Act::Art. 14"}))
    return controls


def crosswalk(controls: list[Control]) -> dict:
    covered: set[str] = set()
    for c in controls:
        covered |= c.covers
    by_fw: dict[str, list[int]] = {}
    for r in REQUIREMENTS:
        stat = by_fw.setdefault(r.framework, [0, 0])
        stat[1] += 1
        if _key(r) in covered:
            stat[0] += 1
    gaps = [r for r in REQUIREMENTS if _key(r) not in covered]
    return {"coverage": {fw: tuple(v) for fw, v in by_fw.items()},
            "gaps": gaps, "mandatory_gaps": [r for r in gaps if r.mandatory]}


# --- REPORT ---
def run_pipeline(org: str, fleet: list[dict], attempts: list[dict], ctx: dict) -> dict:
    agents = ingest(fleet)
    engine = PolicyEngine([rule_scope, rule_budget], "policy-v1",
                          ctx.get("consent_required", set()))
    trail = AuditTrail()
    by_id = {a.agent_id: a for a in agents}
    blocked = 0
    for att in attempts:
        agent = by_id.get(att["agent_id"])
        if agent is None:
            continue
        action = Action(att["tool"], att.get("params", {}), att["required_scope"])
        decision, executed = engine.enforce(
            action, agent, ctx,
            user_consented=att.get("user_consented", False),
            human_approves=att.get("human_approves", False))
        trail.record({"ts": _utc_now_iso(), "agent_id": agent.agent_id,
                      "tool": action.tool, "verdict": decision.verdict.name,
                      "rule": decision.rule, "executed": executed})
        if not executed:
            blocked += 1
    audit_ok, _ = trail.verify()
    has_owner = len(orphans(agents)) == 0
    cw = crosswalk(build_controls(audit_ok, has_owner))
    return {
        "org": org, "agents": agents, "trail": trail,
        "coverage": coverage(agents), "orphans": [a.agent_id for a in orphans(agents)],
        "audit_ok": audit_ok, "blocked": blocked, "attempts": len(trail.entries()),
        "scores": [score_agent(a) for a in agents], "crosswalk": cw,
    }


def render_markdown(res: dict) -> str:
    g, t, pct = res["coverage"]
    lines = [f"GOVERNANCE REPORT — {res['org']}",
             f"  coverage: {g}/{t} ({pct:.0f}%) | orphans: {res['orphans']}",
             f"  audit: {res['attempts']} entries, "
             f"{'VERIFIED' if res['audit_ok'] else 'BROKEN'}",
             f"  enforcement: attempts={res['attempts']} blocked={res['blocked']}",
             "  risk:"]
    for s in sorted(res["scores"], key=lambda x: x.criticality, reverse=True):
        lines.append(f"    {s.decision:<8} {s.agent_id:<14} crit={s.criticality}")
    lines.append("  compliance:")
    for fw, (c, tot) in sorted(res["crosswalk"]["coverage"].items()):
        lines.append(f"    {fw:<14} {c}/{tot}")
    n_mand = len(res["crosswalk"]["mandatory_gaps"])
    if not res["audit_ok"]:
        verdict = "audit BROKEN -> halt scale-up"
    elif n_mand:
        refs = ", ".join(f"{r.framework} {r.ref}" for r in res["crosswalk"]["mandatory_gaps"])
        verdict = f"{n_mand} mandatory gap(s) [{refs}] -> remediation required"
    elif res["orphans"]:
        verdict = "no legal gap, but orphans remain -> assign owners"
    else:
        verdict = "cleared to scale (re-run monthly)"
    lines.append(f"  VERDICT: {verdict}")
    return "\n".join(lines)


def render_json(res: dict) -> str:
    g, t, pct = res["coverage"]
    return json.dumps({
        "org": res["org"],
        "coverage": {"governed": g, "total": t, "pct": round(pct, 1)},
        "orphans": res["orphans"],
        "audit_verified": res["audit_ok"],
        "enforcement": {"attempts": res["attempts"], "blocked": res["blocked"]},
        "risk": [{"agent_id": s.agent_id, "criticality": s.criticality,
                  "decision": s.decision} for s in res["scores"]],
        "mandatory_gaps": [f"{r.framework} {r.ref}"
                           for r in res["crosswalk"]["mandatory_gaps"]],
    }, indent=2, sort_keys=True)


def _demo_hard() -> None:
    print("\n=== HARD: full pipeline -> board-ready report ===")
    ctx = {"auto_amount_limit": 1000, "consent_required": {"export_data", "issue_refund"}}
    fleet = [
        {"agent_id": "fin", "owner": "alice", "scopes": ["payments:execute"],
         "risk_tier": "high", "autonomous": True, "handles_irreversible": True},
        {"agent_id": "sup", "owner": "bob", "scopes": ["refund:execute"],
         "risk_tier": "medium"},
        {"agent_id": "scr", "owner": None, "scopes": ["web:fetch"],
         "risk_tier": "medium"},                                       # orphan
    ]
    attempts = [
        {"agent_id": "fin", "tool": "transfer", "params": {"amount": 500},
         "required_scope": "payments:execute"},                        # ALLOW
        {"agent_id": "sup", "tool": "issue_refund", "params": {"amount": 20},
         "required_scope": "refund:execute"},                          # DENY (consent)
        {"agent_id": "scr", "tool": "export_data",
         "params": {"contains_pii": True}, "required_scope": "web:fetch"},  # DENY (consent)
    ]
    res = run_pipeline("Acme", fleet, attempts, ctx)
    print(render_markdown(res))

    # (a) high+autonomous+irreversible -> crit 25, TREAT, both modulators present.
    fin_score = next(s for s in res["scores"] if s.agent_id == "fin")
    assert fin_score.criticality == 25 and fin_score.decision == "TREAT"
    assert len(fin_score.rationale) == 2

    # (b) orphan -> Art. 14 stays uncovered and is a MANDATORY gap.
    mand_refs = {f"{r.framework} {r.ref}" for r in res["crosswalk"]["mandatory_gaps"]}
    assert "EU AI Act Art. 14" in mand_refs

    # (c) JSON is valid + the sensitive tool without consent was blocked + logged.
    parsed = json.loads(render_json(res))
    assert parsed["enforcement"]["blocked"] >= 2
    assert any(e["rule"] == "mcp_consent" and not e["executed"]
               for e in res["trail"].entries())

    # (d) silent edit of an audit entry -> verify() flips to (False, index).
    res["trail"]._chain[-1]["payload"]["verdict"] = "ALLOW"
    ok, broken = res["trail"].verify()
    assert ok is False and broken == len(res["trail"]._chain) - 1
    print(f"  tamper probe -> verify = ({ok}, {broken})")


if __name__ == "__main__":
    _demo_easy()
    _demo_medium()
    _demo_hard()
    print("\nAll smoke tests passed.")

"""
Solutions -- Day 2: The 4 pillars of a governable agent.

Covers the easy / medium / hard exercises in one file, split by the
# === EASY/MEDIUM/HARD === markers. stdlib only; runs offline.

Run:  python 03-exercises/solutions/02-quatre-piliers.py
The `__main__` block is a smoke test exercising every solution.

Real-world anchors (re-implemented in miniature, stdlib only):
  - Non-Human Identity / agent IAM ............. CSA, Agentic AI IAM (2025)
  - ASI03 Identity & Privilege Abuse .......... OWASP Top 10 Agentic (2026)
  - "humans remain ultimately accountable" .... IMDA Agentic framework (2026)

# requires: stdlib only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Shared base: a minimal Agent Card (used across all three levels).
# ---------------------------------------------------------------------------
_HUMAN_ID_PREFIXES = ("user:", "human:", "person:", "employee:")
_NON_OWNERS = {"team", "it", "data team", "n/a", "tbd", "unknown", ""}


@dataclass
class GovernedAgent:
    agent_id: str
    owner: str
    permissions: list[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)  # bounded scopes (medium ex1)
    audit_log: list[dict] = field(default_factory=list)


def check_governance(agent: GovernedAgent) -> list[str]:
    """The 4-question smell test. Empty result => governable."""
    failures: list[str] = []
    if not agent.agent_id or not agent.agent_id.strip():
        failures.append("IDENTITY: missing unique agent id")
    elif agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES):
        failures.append("IDENTITY: agent uses a human's identity (impersonation)")
    owner = (agent.owner or "").strip()
    if owner.lower() in _NON_OWNERS:
        failures.append(f"OWNER: owner '{owner or '(empty)'}' is not a named person")
    if not agent.permissions:
        failures.append("PERMISSIONS: no explicit scopes (least privilege absent)")
    elif "*" in agent.permissions or "all" in agent.permissions:
        failures.append("PERMISSIONS: wildcard scope violates least privilege")
    if agent.audit_log is None:
        failures.append("AUDIT: no audit trail capability")
    return failures


def is_governable(agent: GovernedAgent) -> bool:
    return not check_governance(agent)


# ===========================================================================
# === EASY ===
# ===========================================================================
def easy_ex1_describe_agent() -> GovernedAgent:
    """Instantiate a complete agent and render its mini Agent Card."""
    agent = GovernedAgent(
        agent_id="agent://ticket-summarizer/a1",  # Pillar 1: unique, not human
        owner="Amina Diallo",  # Pillar 2: a named person
        permissions=["read:tickets", "write:ticket_summary"],  # Pillar 3: scoped
    )
    failures = check_governance(agent)
    print("[easy1] Agent Card")
    print(f"  Identite    : {agent.agent_id}")
    print(f"  Owner       : {agent.owner}")
    print(f"  Permissions : {agent.permissions}")
    print(f"  Audit       : {len(agent.audit_log)} entrees (trace activee)")
    print(f"  Smell test  : {'PASS (governable)' if not failures else failures}")
    return agent


def smell_test(agent: GovernedAgent) -> str:
    failures = check_governance(agent)
    if not failures:
        return "GOUVERNE"
    return "NON GOUVERNE: " + "; ".join(failures)


def easy_ex2_detect_ungoverned() -> None:
    """Three agents, each breaking exactly one pillar."""
    a = GovernedAgent("agent://x/1", owner="IT", permissions=["read:x"])  # owner
    b = GovernedAgent("agent://x/2", owner="Lea Roux", permissions=[])  # perms
    c = GovernedAgent("user:bob", owner="Lea Roux", permissions=["read:x"])  # id
    print("[easy2] smell test on 3 broken agents")
    for label, agent in (("A no-owner", a), ("B no-perms", b), ("C impersonate", c)):
        print(f"  {label}: {smell_test(agent)}")


def governance_coverage(agents: list[GovernedAgent]) -> float:
    """Share of the fleet that is fully governable (board KPI)."""
    if not agents:  # empty-fleet edge case -> no ZeroDivisionError
        return 0.0
    governed = sum(1 for a in agents if is_governable(a))
    return round(100.0 * governed / len(agents), 1)


def _is_orphan(agent: GovernedAgent) -> bool:
    return (agent.owner or "").strip().lower() in _NON_OWNERS


def easy_ex3_fleet_coverage() -> float:
    fleet = [
        GovernedAgent("agent://s/1", "Amina Diallo", ["read:tickets"]),
        GovernedAgent("agent://s/2", "Lea Roux", ["read:x", "write:y"]),
        GovernedAgent("agent://s/3", "IT", ["read:x"]),  # orphan
        GovernedAgent("user:carl", "Sam Park", ["read:x"]),  # impersonation
        GovernedAgent("agent://s/5", "Sam Park", []),  # no perms
    ]
    orphans = sum(1 for a in fleet if _is_orphan(a))
    cov = governance_coverage(fleet)
    print("[easy3] fleet report")
    print(f"  total agents       : {len(fleet)}")
    print(f"  governable agents  : {sum(1 for a in fleet if is_governable(a))}")
    print(f"  orphan agents      : {orphans}")
    print(f"  governance coverage: {cov} %")
    print(f"  empty fleet -> {governance_coverage([])} %  (no crash)")
    return cov


# ===========================================================================
# === MEDIUM ===
# ===========================================================================
def attempt(agent: GovernedAgent, action: str, scope_required: str, **context) -> dict:
    """Least-privilege enforcement + audit trace (medium ex1).

    Denies out-of-scope actions AND over-limit amounts; every attempt is
    recorded -- allowed or denied.
    """
    granted = scope_required in agent.permissions
    # Bounded scope: even with the scope, refuse amounts above the limit.
    if granted and "amount" in context:
        limit = agent.limits.get(scope_required)
        if limit is not None and context["amount"] > limit:
            granted = False
            context = {**context, "denied_reason": f"amount>{limit}"}
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent.agent_id,
        "action": action,
        "scope_required": scope_required,
        "result": "executed" if granted else "denied",
        "context": context,
    }
    agent.audit_log.append(record)  # append-only by convention
    return record


def medium_ex1_least_privilege() -> None:
    bot = GovernedAgent(
        "agent://refund-bot/7f3a",
        "Sofia Marchetti",
        ["read:disputes", "write:dispute_notes", "transfer:funds"],
        limits={"transfer:funds": 1000},  # bounded scope
    )
    r1 = attempt(bot, "read_dispute", "read:disputes", dispute="#88421")
    r2 = attempt(bot, "funds_transfer", "transfer:funds", amount=4200)  # over limit
    r3 = attempt(bot, "delete_account", "admin:delete")  # out of scope
    print("[medium1] least-privilege enforcement")
    print(f"  read_dispute (in scope)        -> {r1['result']}")
    print(f"  transfer 4200 (over 1000 limit)-> {r2['result']}")
    print(f"  delete (out of scope)          -> {r3['result']}")
    print(f"  audit entries recorded         : {len(bot.audit_log)}")


def to_card(agent: GovernedAgent) -> dict:
    return {
        "agent_id": agent.agent_id,
        "owner": agent.owner,
        "permissions": list(agent.permissions),
        "audit_entries": len(agent.audit_log),
    }


def from_card(data: dict) -> GovernedAgent:
    for pillar in ("agent_id", "owner", "permissions"):
        if pillar not in data:
            raise ValueError(f"missing pillar: {pillar}")
    agent = GovernedAgent(
        agent_id=data["agent_id"],
        owner=data["owner"],
        permissions=list(data["permissions"]),
    )
    # audit_entries is metadata only; the live log starts empty on reload.
    return agent


def medium_ex2_serialize_card() -> None:
    original = GovernedAgent(
        "agent://summ/9", "Amina Diallo", ["read:tickets", "write:ticket_summary"]
    )
    raw = json.dumps(to_card(original))  # serialize
    restored = from_card(json.loads(raw))  # round-trip
    print("[medium2] Agent Card round-trip (JSON)")
    print(f"  json            : {raw}")
    print(f"  restored govern : {is_governable(restored)}")
    try:
        from_card({"agent_id": "agent://x/1", "permissions": []})  # no owner
    except ValueError as e:
        print(f"  incomplete card -> ValueError: {e}")


# ASI03 split into two distinct detectors (medium ex3).
ROLE_ALLOWED_SCOPES = {
    "summarizer": {"read:tickets", "write:ticket_summary"},
}


def detect_impersonation(agent: GovernedAgent) -> bool:
    return agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES)


def detect_privilege_abuse(agent: GovernedAgent, role: str) -> bool:
    if "*" in agent.permissions or "all" in agent.permissions:
        return True
    allowed = ROLE_ALLOWED_SCOPES.get(role, set())
    return any(scope not in allowed for scope in agent.permissions)


def medium_ex3_asi03() -> None:
    fleet = [
        ("summarizer", GovernedAgent("agent://s/1", "Amina Diallo",
                                     ["read:tickets", "write:ticket_summary"])),
        ("summarizer", GovernedAgent("agent://s/2", "Lea Roux",
                                     ["read:tickets", "transfer:funds"])),  # abuse
        ("summarizer", GovernedAgent("user:carl", "Sam Park",
                                     ["read:tickets"])),  # impersonation
        ("summarizer", GovernedAgent("agent://s/4", "Sam Park", ["*"])),  # wildcard
    ]
    print("[medium3] ASI03 -- impersonation vs privilege abuse")
    asi03 = 0
    for role, agent in fleet:
        imp = detect_impersonation(agent)
        abuse = detect_privilege_abuse(agent, role)
        if imp or abuse:
            asi03 += 1
        print(f"  {agent.agent_id:<22} impersonation={imp!s:<5} abuse={abuse}")
    print(f"  agents flagged ASI03: {asi03}")


# ===========================================================================
# === HARD ===
# ===========================================================================
# Weights sum to 100. Permissions & Identity weigh most: an over-privileged
# or impersonating agent is the sharpest OWASP ASI03 risk; owner and audit,
# while essential, fail "more recoverably".
PILLAR_WEIGHTS = {"identity": 30, "owner": 20, "permissions": 35, "audit": 15}


def _pillar_credit(agent: GovernedAgent) -> dict[str, float]:
    """Per-pillar credit in [0,1]: 1 full, 0.5 partial, 0 absent."""
    c = {}
    # identity
    if not agent.agent_id or not agent.agent_id.strip():
        c["identity"] = 0.0
    elif agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES):
        c["identity"] = 0.0  # impersonation == absent identity
    else:
        c["identity"] = 1.0
    # owner: present-but-not-a-person => half credit
    owner = (agent.owner or "").strip().lower()
    if owner in {"", "n/a", "tbd", "unknown"}:
        c["owner"] = 0.0
    elif owner in {"team", "it", "data team"}:
        c["owner"] = 0.5
    else:
        c["owner"] = 1.0
    # permissions: wildcard => half (scoped but unbounded); empty => 0
    if not agent.permissions:
        c["permissions"] = 0.0
    elif "*" in agent.permissions or "all" in agent.permissions:
        c["permissions"] = 0.5
    else:
        c["permissions"] = 1.0
    # audit: capability present?
    c["audit"] = 0.0 if agent.audit_log is None else 1.0
    return c


def governance_score(agent: GovernedAgent) -> float:
    credit = _pillar_credit(agent)
    return round(sum(PILLAR_WEIGHTS[p] * credit[p] for p in PILLAR_WEIGHTS), 1)


def maturity_level(score: float) -> str:
    if score >= 90:
        return "GOVERNED"
    if score >= 60:
        return "MANAGED"
    if score >= 25:
        return "EMERGENT"
    return "ABSENT"


def hard_ex1_scoring() -> None:
    assert sum(PILLAR_WEIGHTS.values()) == 100  # weights sum to 100
    fleet = [
        GovernedAgent("agent://g/1", "Amina Diallo", ["read:x", "write:y"]),  # 100
        GovernedAgent("agent://g/2", "IT", ["read:x"]),  # partial owner
        GovernedAgent("agent://g/3", "Sam Park", ["*"]),  # wildcard
        GovernedAgent("user:bob", "Lea Roux", ["read:x"]),  # impersonation
        GovernedAgent("agent://g/5", "", []),  # nearly nothing
        GovernedAgent("agent://g/6", "Mia Chen", ["read:tickets"]),  # 100
    ]
    rows = sorted(((governance_score(a), a) for a in fleet), key=lambda t: t[0])
    print("[hard1] governance score & maturity (worst first)")
    print(f"  {'agent_id':<22} {'score':>5}  level")
    for score, a in rows:
        print(f"  {a.agent_id:<22} {score:>5}  {maturity_level(score)}")


def broken_chain(agent: GovernedAgent) -> list[str]:
    """The 4 pillars are a chain: a missing link voids dependent pillars."""
    has_identity = bool(agent.agent_id and agent.agent_id.strip()
                        and not agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES))
    has_owner = (agent.owner or "").strip().lower() not in _NON_OWNERS
    has_perms = bool(agent.permissions) and not (
        "*" in agent.permissions or "all" in agent.permissions
    )
    has_audit = agent.audit_log is not None
    broken = []
    if has_audit and not has_identity:
        broken.append("audit present but identity absent -> trace unattributable")
    if has_perms and not has_audit:
        broken.append("permissions present but audit absent -> usage unprovable")
    if has_identity and not has_owner:
        broken.append("identity present but owner absent -> nobody to escalate to")
    return broken


def hard_ex2_chain() -> None:
    print("[hard2] the 4 pillars are indissociable")
    complete = GovernedAgent("agent://c/1", "Amina Diallo", ["read:x"])
    # audit_log defaults to [] (capability present), so the chain holds.
    assert broken_chain(complete) == []  # a complete agent breaks nothing
    cases = {
        "no-identity": GovernedAgent("", "Amina Diallo", ["read:x"]),
        "no-owner": GovernedAgent("agent://c/2", "", ["read:x"]),
        "no-perms": GovernedAgent("agent://c/3", "Amina Diallo", []),
        "no-audit (id+perms)": GovernedAgent("agent://c/4", "Amina Diallo",
                                             ["read:x"]),
    }
    cases["no-audit (id+perms)"].audit_log = None  # simulate missing capability
    for label, agent in cases.items():
        chain = broken_chain(agent)
        print(f"  drop {label:<22}: {chain if chain else '(no dependent link here)'}")


def remediation_plan(agent: GovernedAgent) -> list[dict]:
    plan = []
    for issue in check_governance(agent):
        pillar = issue.split(":")[0].lower()
        if "wildcard" in issue or "impersonation" in issue:
            prio = "CRITICAL"  # sharpest ASI03 risk
        elif pillar == "owner":
            prio = "HIGH"
        else:
            prio = "MEDIUM"
        plan.append({
            "pillar": pillar,
            "issue": issue,
            "fix": _fix_for(pillar, issue),
            "priority": prio,
        })
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    return sorted(plan, key=lambda a: order[a["priority"]])


def _fix_for(pillar: str, issue: str) -> str:
    if "impersonation" in issue:
        return "issue a dedicated non-human identity (agent://...)"
    if "wildcard" in issue:
        return "replace '*' with explicit least-privilege scopes"
    return {
        "identity": "assign a unique stable agent id",
        "owner": "name a single accountable person as owner",
        "permissions": "grant explicit, bounded scopes",
        "audit": "enable an append-only audit trail",
    }.get(pillar, "review pillar")


def fleet_remediation(agents: list[GovernedAgent]) -> dict:
    by_prio = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
    worst_id, worst_crit = None, -1
    for a in agents:
        plan = remediation_plan(a)
        crit = sum(1 for x in plan if x["priority"] == "CRITICAL")
        for x in plan:
            by_prio[x["priority"]] += 1
        if crit > worst_crit:
            worst_crit, worst_id = crit, a.agent_id
    return {"by_priority": by_prio, "worst_agent": worst_id if worst_crit > 0 else None}


def remediation_report(agents: list[GovernedAgent]) -> str:
    lines = ["# Rapport de remediation -- gouvernance des agents", ""]
    cov = governance_coverage(agents)
    lines.append(f"Couverture de gouvernance : **{cov} %**")
    lines.append("")
    ungoverned = [a for a in agents if not is_governable(a)]
    if not ungoverned:
        lines.append("Aucune remediation requise.")
        return "\n".join(lines)
    for a in ungoverned:
        lines.append(f"## {a.agent_id}")
        for action in remediation_plan(a):
            lines.append(f"- [{action['priority']}] {action['pillar']}: {action['fix']}")
        lines.append("")
    return "\n".join(lines)


def hard_ex3_remediation() -> None:
    fleet = [
        GovernedAgent("agent://r/1", "Amina Diallo", ["read:x"]),  # clean
        GovernedAgent("user:bob", "IT", ["*"]),  # critical mess
        GovernedAgent("agent://r/3", "", ["read:x"]),  # no owner
    ]
    agg = fleet_remediation(fleet)
    print("[hard3] remediation")
    print(f"  by priority : {agg['by_priority']}")
    print(f"  worst agent : {agg['worst_agent']}")
    print("  --- markdown (truncated) ---")
    print("\n".join(remediation_report(fleet).splitlines()[:8]))
    # Edge case: a fully-governed fleet -> "Aucune remediation requise".
    clean = [GovernedAgent("agent://r/1", "Amina Diallo", ["read:x"])]
    assert "Aucune remediation requise" in remediation_report(clean)
    assert fleet_remediation(clean)["worst_agent"] is None
    print("  healthy fleet -> 'Aucune remediation requise' (no crash)")


# ===========================================================================
if __name__ == "__main__":
    print("===== EASY =====")
    easy_ex1_describe_agent()
    easy_ex2_detect_ungoverned()
    easy_ex3_fleet_coverage()
    print("\n===== MEDIUM =====")
    medium_ex1_least_privilege()
    medium_ex2_serialize_card()
    medium_ex3_asi03()
    print("\n===== HARD =====")
    hard_ex1_scoring()
    hard_ex2_chain()
    hard_ex3_remediation()
    print("\nAll smoke tests passed.")

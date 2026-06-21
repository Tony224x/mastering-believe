"""Solutions — Jour 1 : Pourquoi gouverner l'IA agentique.

One file covering the three difficulty levels of the day's exercises:
  - EASY   : inventory + orphan counting
  - MEDIUM : governance coverage + the "acting orphans" danger quadrant
  - HARD   : bounded risk score + board-ready report in text AND JSON, empty-fleet safe

# requires: stdlib only
Run with:  python 01-pourquoi-gouverner.py   (executes the smoke test at the bottom)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# Shared definition of side-effecting tools (an agent that *acts* on the world).
# WHY: irreversible side effects are the heart of agentic risk — they deserve priority.
SIDE_EFFECT_TOOLS = {"send_payment", "send_email", "post_social", "delete_record", "issue_credit"}


# === EASY ===
# Goal: build a tiny census and count orphan agents (no owner).
# Here we model agents as plain dicts, exactly as the easy exercise asks.

EASY_FLEET = [
    {"agent_id": "support-refund-bot", "owner": "alice.support", "tools": ["issue_credit"]},
    {"agent_id": "marketing-poster", "owner": "bob.mkt", "tools": ["post_social"]},
    {"agent_id": "weekend-db-report", "owner": None, "tools": ["query_db"]},        # orphan
    {"agent_id": "finance-payer", "owner": "", "tools": ["send_payment"]},          # orphan (blank)
    {"agent_id": "faq-answerer", "owner": "carol.ops", "tools": ["search_kb"]},
]


def is_orphan_dict(agent: dict) -> bool:
    """An agent (dict form) is orphan if owner is missing, None, or blank.
    WHY: an empty-string owner is just as ungoverned as a missing one."""
    owner = agent.get("owner")
    return owner is None or str(owner).strip() == ""


def count_orphans(agents: list[dict]) -> int:
    """Number of agents without a named owner."""
    return sum(1 for a in agents if is_orphan_dict(a))


def easy_demo() -> None:
    total = len(EASY_FLEET)
    orphans = [a["agent_id"] for a in EASY_FLEET if is_orphan_dict(a)]
    print("[EASY] Agent census")
    print(f"  Total agents   : {total}")
    print(f"  Orphan agents  : {count_orphans(EASY_FLEET)}")
    print(f"  Orphan ids     : {orphans}")


# === MEDIUM ===
# Goal: governance coverage + isolate agents that ACT but have NO owner.
# We switch to a dataclass for the richer attributes (audit flag).

@dataclass
class Agent:
    agent_id: str
    owner: str | None
    tools: list[str] = field(default_factory=list)
    audit_enabled: bool = False
    tier: str = "minimal"  # used by the HARD level; harmless default here


def is_orphan(agent: Agent) -> bool:
    return agent.owner is None or agent.owner.strip() == ""


def acts_on_world(agent: Agent) -> bool:
    """True if at least one granted tool has a real-world side effect."""
    return any(tool in SIDE_EFFECT_TOOLS for tool in agent.tools)


def is_governed(agent: Agent) -> bool:
    """Day-1 minimal bar: named owner AND audit trail enabled."""
    return not is_orphan(agent) and agent.audit_enabled


def governance_coverage(agents: list[Agent]) -> float:
    """Percentage of governed agents. Empty fleet -> 100.0 (no risk, no division by zero)."""
    if not agents:  # WHY: guard against ZeroDivisionError on an empty inventory.
        return 100.0
    governed = sum(1 for a in agents if is_governed(a))
    return round(governed / len(agents) * 100.0, 1)


def acting_orphans(agents: list[Agent]) -> list[Agent]:
    """The danger quadrant: agents that act on the world but have no owner."""
    return [a for a in agents if acts_on_world(a) and is_orphan(a)]


def _medium_fleet() -> list[Agent]:
    return [
        Agent("support-refund-bot", "alice.support", ["issue_credit", "send_email"], audit_enabled=True, tier="high"),
        Agent("marketing-poster", "bob.mkt", ["post_social"], audit_enabled=False, tier="limited"),
        Agent("weekend-db-report", None, ["query_db"], audit_enabled=False, tier="minimal"),
        Agent("finance-payer", None, ["send_payment"], audit_enabled=False, tier="high"),  # acts + orphan
        Agent("faq-answerer", "carol.ops", ["search_kb"], audit_enabled=True, tier="minimal"),
        Agent("invoice-emailer", "dan.fin", ["send_email"], audit_enabled=True, tier="limited"),
    ]


def medium_demo() -> None:
    fleet = _medium_fleet()
    cov = governance_coverage(fleet)
    risky = acting_orphans(fleet)
    print("[MEDIUM] Governance coverage")
    print(f"  Coverage (owner + audit) : {cov}%")
    print(f"  Acting orphans           : {len(risky)} -> {[a.agent_id for a in risky]}")


# === HARD ===
# Goal: bounded risk score + board-ready report in BOTH text and JSON, empty-fleet safe.

def risk_score(agent: Agent) -> int:
    """Defensible risk score in [0, 100], summing explicit, auditable penalties.
    WHY each penalty: no owner = unaccountable, acting = real-world blast radius,
    no audit = unprovable, high tier = regulatory weight (EU AI Act nod)."""
    score = 0
    if is_orphan(agent):
        score += 30
    if acts_on_world(agent):
        score += 25
    if not agent.audit_enabled:
        score += 25
    if agent.tier == "high":
        score += 20
    elif agent.tier == "limited":
        score += 10
    return min(100, score)  # bounded so the score stays interpretable


def build_report(agents: list[Agent]) -> dict:
    """Structured, serializable governance report. Safe on an empty fleet."""
    ranked = sorted(agents, key=risk_score, reverse=True)  # highest risk first
    return {
        "total": len(agents),
        "coverage_pct": governance_coverage(agents),
        "acting_orphans": [a.agent_id for a in acting_orphans(agents)],
        "agents_by_risk": [
            {"agent_id": a.agent_id, "risk_score": risk_score(a)} for a in ranked
        ],
    }


def render_text(report: dict) -> str:
    """Human-readable rendering of the exact same data carried by the JSON report."""
    lines = ["Governance report (board-ready)"]
    lines.append(f"  Total agents      : {report['total']}")
    lines.append(f"  Coverage          : {report['coverage_pct']}%")
    lines.append(f"  Acting orphans    : {report['acting_orphans'] or '(none)'}")
    lines.append("  Agents by risk (desc):")
    for entry in report["agents_by_risk"]:
        lines.append(f"    - {entry['agent_id']:<22} score={entry['risk_score']}")
    return "\n".join(lines)


def hard_demo() -> None:
    fleet = _medium_fleet()
    # Adversarial input: an agent with no tools at all must be handled gracefully.
    fleet.append(Agent("empty-tooled-agent", "ed.ops", tools=[], audit_enabled=False, tier="minimal"))

    report = build_report(fleet)
    print("[HARD] Text format")
    print(render_text(report))
    print("[HARD] JSON format")
    print(json.dumps(report, indent=2))

    # Adversarial probe: empty fleet must not crash and must read as fully covered.
    empty_report = build_report([])
    print("[HARD] Adversarial probe — empty fleet")
    print(f"  coverage_pct={empty_report['coverage_pct']} (expected 100.0), "
          f"agents_by_risk={empty_report['agents_by_risk']}")


if __name__ == "__main__":
    # Smoke test: run all three levels and assert the key invariants hold.
    easy_demo()
    print()
    medium_demo()
    print()
    hard_demo()

    # --- assertions (the smoke test proper) ---
    assert count_orphans(EASY_FLEET) == 2, "easy: expected 2 orphans (None + blank owner)"
    assert governance_coverage([]) == 100.0, "medium: empty fleet must be 100% covered"

    fleet = _medium_fleet()
    assert any(a.agent_id == "finance-payer" for a in acting_orphans(fleet)), \
        "medium: finance-payer acts on world and is orphaned"

    # finance-payer: orphan(30) + acts(25) + no audit(25) + high tier(20) = 100, capped at 100.
    payer = next(a for a in fleet if a.agent_id == "finance-payer")
    assert risk_score(payer) == 100, "hard: finance-payer must score 100"

    # Score must always stay within [0, 100].
    for a in fleet:
        assert 0 <= risk_score(a) <= 100, f"hard: score out of bounds for {a.agent_id}"

    # Ranking must be non-increasing.
    scores = [e["risk_score"] for e in build_report(fleet)["agents_by_risk"]]
    assert scores == sorted(scores, reverse=True), "hard: agents_by_risk must be sorted desc"

    # JSON round-trip must succeed (report is serializable).
    assert json.loads(json.dumps(build_report(fleet)))["total"] == len(fleet), "hard: JSON round-trip failed"

    print("\nAll smoke-test assertions passed.")

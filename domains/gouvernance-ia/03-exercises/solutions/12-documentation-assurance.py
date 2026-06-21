"""Solutions J12 — Documentation & assurance (preuve statique).

One file, three levels separated by banners. stdlib only.

- EASY   : agent card generator (Markdown), with visible flags for missing fields.
- MEDIUM : safety case skeleton (claim -> evidence -> gaps) + coverage / fragility.
- HARD   : risk-calibrated assurance gate (APPROVE / WITH_CONDITIONS / BLOCK).

References: Model Cards [Mitchell et al., 2019], GPT-4 System Card [OpenAI, 2023],
Safety Cases [Clymer et al., 2024].

# requires: stdlib only
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# WHY: make stdout robust to non-ASCII (e.g. Windows cp1252) so `python <file>`
# always runs to completion — no external dep, no env var required.
try:  # pragma: no cover - depends on the terminal
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# === EASY ===
# Goal: render an agent card from a dict; make missing required fields visible.

REQUIRED_CARD_FIELDS = ("agent_id", "owner", "purpose", "permissions")


def make_card(fields: dict) -> str:
    """Render an agent card to Markdown from a flat dict of fields.

    WHY a dict-driven renderer: the structure is the governance contract; missing
    required fields are flagged (⚠️) rather than silently dropped.
    """

    def flag(value, ok_render: str) -> str:
        # Empty / None / empty-list -> visible warning instead of omission.
        if value in (None, "", [], ()):
            return "MISSING ⚠️"
        return ok_render

    agent_id = fields.get("agent_id") or "MISSING ⚠️"
    owner = flag(fields.get("owner"), str(fields.get("owner")))
    status = fields.get("status") or "draft"
    purpose = flag(fields.get("purpose"), str(fields.get("purpose")))

    permissions = fields.get("permissions") or []
    perms_block = (
        "\n".join(f"- `{p}`" for p in permissions)
        if permissions
        else "- (none declared) ⚠️"
    )

    handles_pd = bool(fields.get("handles_personal_data"))
    if handles_pd:
        pd_line = f"yes — legal basis: {fields.get('legal_basis') or 'MISSING ⚠️'}"
    else:
        pd_line = "no"

    eval_line = fields.get("eval_summary") or "no evaluation recorded ⚠️"

    limitations = fields.get("known_limitations") or []
    lim_block = (
        "\n".join(f"- {x}" for x in limitations)
        if limitations
        else "- (none declared) ⚠️"
    )

    return "\n".join(
        [
            f"# Agent Card — {agent_id}",
            "",
            "## Identity & ownership",
            f"- **Agent ID**: {agent_id}",
            f"- **Owner**: {owner}",
            f"- **Status**: {status}",
            "",
            "## Intended use",
            f"- **Purpose**: {purpose}",
            "",
            "## Permissions",
            perms_block,
            "",
            "## Personal data",
            f"- {pd_line}",
            "",
            "## Evaluation",
            f"- {eval_line}",
            "",
            "## Known limitations",
            lim_block,
        ]
    )


def _easy_demo() -> str:
    triage_bot = {
        "agent_id": "triage-bot@v1.0",
        "owner": "Léa Martin (Support Lead)",
        "status": "production",
        "purpose": "Classify incoming support emails by priority (P1/P2/P3).",
        "permissions": ["read:email_subject", "read:email_body", "write:priority_tag"],
        "handles_personal_data": True,
        "legal_basis": "legitimate interest (GDPR art. 6.1.f)",
        "eval_summary": "88% correct triage on 300 emails (2026-04)",
        "known_limitations": ["under-classifies very short emails"],
    }
    return make_card(triage_bot)


# === MEDIUM ===
# Goal: safety case skeleton + coverage + fragility ranking.


class ArgumentType(Enum):
    """Safety-case argument types [Clymer et al., 2024], strongest first.

    value encodes robustness (1 = strongest = inability).
    """

    INABILITY = 1
    CONTROL = 2
    TRUSTWORTHINESS = 3
    DEFERENCE = 4


@dataclass
class Claim:
    statement: str
    argument_type: ArgumentType
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        # A claim with no evidence is an opinion, not an argument.
        return len(self.evidence) > 0


@dataclass
class SafetyCase:
    agent_id: str
    context: str
    claims: list[Claim] = field(default_factory=list)

    def coverage(self) -> float:
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.is_supported) / len(self.claims)

    def open_gaps(self) -> list[tuple[str, str]]:
        return [(c.statement, g) for c in self.claims for g in c.gaps]

    def weakest_arguments(self) -> list[Claim]:
        # Highest argument_type value first => most fragile first.
        return sorted(self.claims, key=lambda c: c.argument_type.value, reverse=True)


def _build_refund_case() -> SafetyCase:
    case = SafetyCase(
        agent_id="refund-agent@v2.3",
        context="production, EU, refunds < 200 EUR",
    )
    case.claims.append(
        Claim(
            statement="Cannot credit more than 200 EUR without a human.",
            argument_type=ArgumentType.INABILITY,
            evidence=[
                "policy engine hard-caps credit at 200 EUR",
                "200 adversarial over-limit attempts blocked (eval)",
            ],
            gaps=["not tested under concurrent load"],
        )
    )
    case.claims.append(
        Claim(
            statement="Cannot exfiltrate the customer database.",
            argument_type=ArgumentType.CONTROL,
            evidence=["scope limited to read:ticket + read:order_history"],
            gaps=["no independent review of scope enforcement"],
        )
    )
    # Deliberately unsupported + fragile type, to test detection.
    case.claims.append(
        Claim(
            statement="Does not produce discriminatory refusals.",
            argument_type=ArgumentType.TRUSTWORTHINESS,
            evidence=[],
            gaps=["fairness across segments not evaluated"],
        )
    )
    return case


def _medium_demo() -> SafetyCase:
    return _build_refund_case()


# === HARD ===
# Goal: risk-calibrated assurance gate over (card completeness, safety case).


@dataclass
class GovernedAgent:
    agent_id: str
    owner: str
    purpose: str
    permissions: list[str]
    risk_tier: str
    handles_personal_data: bool = False
    legal_basis: Optional[str] = None
    eval_summary: Optional[str] = None
    known_limitations: list[str] = field(default_factory=list)


def assess_card(agent: GovernedAgent) -> tuple[float, list[str]]:
    """Return (completeness 0..1, list of missing fields)."""
    issues: list[str] = []
    if not agent.owner:
        issues.append("no named owner")
    if not agent.permissions:
        issues.append("no declared permissions")
    if agent.handles_personal_data and not agent.legal_basis:
        issues.append("personal data without legal basis")
    if not agent.eval_summary:
        issues.append("no evaluation recorded")
    if not agent.known_limitations:
        issues.append("no known limitations declared")
    completeness = 1.0 - len(issues) / 5.0
    return completeness, issues


def assess_case(case: SafetyCase) -> dict:
    """Summarize a safety case for the gate."""
    # A claim is "weak & unsupported" if it leans on a fragile argument type
    # (trustworthiness/deference) AND has no evidence at all.
    weak_unsupported = [
        c
        for c in case.claims
        if c.argument_type
        in (ArgumentType.TRUSTWORTHINESS, ArgumentType.DEFERENCE)
        and not c.is_supported
    ]
    return {
        "coverage": case.coverage(),
        "n_open_gaps": len(case.open_gaps()),
        "relies_on_weak_argument": len(weak_unsupported) > 0,
        "unsupported_claims": [c.statement for c in case.claims if not c.is_supported],
    }


def assurance_gate(agent: GovernedAgent, case: SafetyCase, risk_tier: str) -> dict:
    """Decide APPROVE / APPROVE_WITH_CONDITIONS / BLOCK by risk tier.

    WHY tier-dependent: the same dossier may be fine for a minimal-risk agent and
    unacceptable for a high-risk one. The gate encodes that asymmetry.
    """
    completeness, missing = assess_card(agent)
    summary = assess_case(case)
    reasons: list[str] = []

    if risk_tier == "high":
        ok_card = completeness >= 0.8
        ok_cov = summary["coverage"] >= 1.0
        ok_weak = not summary["relies_on_weak_argument"]
        if ok_card and ok_cov and ok_weak:
            reasons.append("high-risk thresholds met (card>=0.8, coverage=100%, no weak unsupported claim)")
            return {"decision": "APPROVE", "reasons": reasons, "conditions": []}
        if not ok_card:
            reasons.append(f"card completeness {completeness:.0%} < 80% (missing: {missing})")
        if not ok_cov:
            reasons.append(
                f"safety-case coverage {summary['coverage']:.0%} < 100% "
                f"(unsupported: {summary['unsupported_claims']})"
            )
        if not ok_weak:
            reasons.append("a critical claim relies on a weak argument with no evidence")
        return {"decision": "BLOCK", "reasons": reasons, "conditions": []}

    if risk_tier == "limited":
        ok_card = completeness >= 0.6
        ok_cov = summary["coverage"] >= 0.7
        if ok_card and ok_cov:
            reasons.append("limited-risk thresholds met (card>=0.6, coverage>=0.7)")
            return {"decision": "APPROVE", "reasons": reasons, "conditions": []}
        conditions: list[str] = []
        if not ok_card:
            conditions += [f"complete card field: {m}" for m in missing]
        if not ok_cov:
            conditions += [f"add evidence to: {s}" for s in summary["unsupported_claims"]]
        reasons.append("below limited-risk thresholds; conditional approval")
        return {
            "decision": "APPROVE_WITH_CONDITIONS",
            "reasons": reasons,
            "conditions": conditions,
        }

    # minimal tier
    if agent.owner:
        reasons.append("minimal risk: a named owner is enough")
        return {"decision": "APPROVE", "reasons": reasons, "conditions": []}
    reasons.append("minimal risk but no named owner -> ungoverned")
    return {"decision": "BLOCK", "reasons": reasons, "conditions": []}


def _well_documented_high_agent() -> GovernedAgent:
    return GovernedAgent(
        agent_id="refund-agent@v2.3",
        owner="Camille Roux (Head of CX)",
        purpose="Decide refunds < 200 EUR.",
        permissions=["read:ticket", "credit:<=200_eur"],
        risk_tier="high",
        handles_personal_data=True,
        legal_basis="contract (GDPR art. 6.1.b)",
        eval_summary="94% correct on 500 tickets (2026-05)",
        known_limitations=["FR only"],
    )


def _solid_case() -> SafetyCase:
    case = SafetyCase(agent_id="refund-agent@v2.3", context="prod, EU")
    case.claims.append(
        Claim(
            "Cannot credit > 200 EUR without human.",
            ArgumentType.INABILITY,
            evidence=["policy hard-cap", "200 adversarial attempts blocked"],
            gaps=["not tested under load"],
        )
    )
    case.claims.append(
        Claim(
            "Cannot exfiltrate DB.",
            ArgumentType.CONTROL,
            evidence=["least-privilege scopes"],
            gaps=[],
        )
    )
    return case


def _broken_case() -> SafetyCase:
    # Same agent but the critical inability claim has NO evidence.
    case = SafetyCase(agent_id="refund-agent@v2.3", context="prod, EU")
    case.claims.append(
        Claim(
            "Cannot credit > 200 EUR without human.",
            ArgumentType.INABILITY,
            evidence=[],  # critical claim left unsupported
            gaps=["no test at all"],
        )
    )
    return case


def _hard_demo() -> list[dict]:
    results = []
    # (a) high-risk, well documented, solid case -> APPROVE
    results.append(
        assurance_gate(_well_documented_high_agent(), _solid_case(), "high")
    )
    # (b) high-risk, same agent, broken case -> BLOCK
    results.append(
        assurance_gate(_well_documented_high_agent(), _broken_case(), "high")
    )
    # (c) limited-risk, half-documented -> APPROVE_WITH_CONDITIONS
    half = GovernedAgent(
        agent_id="triage-bot@v1.0",
        owner="Léa Martin",
        purpose="Classify emails.",
        permissions=["read:email"],
        risk_tier="limited",
        handles_personal_data=True,
        legal_basis=None,  # missing -> card incomplete
        eval_summary=None,  # missing
        known_limitations=[],  # missing
    )
    weak_case = SafetyCase(agent_id="triage-bot@v1.0", context="prod")
    weak_case.claims.append(
        Claim("No external action.", ArgumentType.INABILITY, evidence=["no write scope"])
    )
    weak_case.claims.append(
        Claim("No bias.", ArgumentType.TRUSTWORTHINESS, evidence=[])  # unsupported
    )
    results.append(assurance_gate(half, weak_case, "limited"))
    return results


# === SMOKE TEST ===
if __name__ == "__main__":
    print("=== EASY ===")
    easy_card = _easy_demo()
    print(easy_card)
    assert "triage-bot@v1.0" in easy_card
    assert "legitimate interest" in easy_card
    assert "MISSING ⚠️" not in easy_card  # triage-bot is fully documented

    print("\n=== MEDIUM ===")
    case = _medium_demo()
    cov = case.coverage()
    print(f"coverage: {cov:.0%}")
    print(f"open gaps: {len(case.open_gaps())}")
    weakest = case.weakest_arguments()[0]
    print(f"most fragile: [{weakest.argument_type.name}] {weakest.statement}")
    # 2 of 3 claims supported -> ~67%
    assert abs(cov - 2 / 3) < 1e-9
    assert weakest.argument_type == ArgumentType.TRUSTWORTHINESS
    assert len(case.open_gaps()) == 3

    print("\n=== HARD ===")
    a, b, c = _hard_demo()
    for label, res in (("(a) high+solid", a), ("(b) high+broken", b), ("(c) limited+half", c)):
        print(f"{label}: {res['decision']}")
        for r in res["reasons"]:
            print(f"    - {r}")
        for cond in res["conditions"]:
            print(f"    condition: {cond}")
    assert a["decision"] == "APPROVE"
    assert b["decision"] == "BLOCK"
    assert c["decision"] == "APPROVE_WITH_CONDITIONS"
    assert c["conditions"], "conditional approval must list conditions"

    print("\nAll smoke tests passed.")

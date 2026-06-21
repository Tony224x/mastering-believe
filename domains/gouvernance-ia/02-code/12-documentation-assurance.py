"""J12 — Documentation & assurance (preuve statique).

This script demonstrates the *static evidence* side of AI governance: turning an
opaque agent into an inspectable artifact a human auditor can read.

It builds two things, in pure stdlib:

  1. An AGENT CARD generator: a structured agent description (the documentary
     form of the 4 governance pillars) rendered to portable Markdown. Inspired by
     Model Cards [Mitchell et al., 2019] and the GPT-4 System Card [OpenAI, 2023].

  2. A SAFETY CASE skeleton: a structured assurance argument decomposed into
     claim -> evidence -> gaps, with a coverage metric. Inspired by
     "Safety Cases" [Clymer et al., 2024], whose four argument types are
     (in decreasing robustness): inability > control > trustworthiness > deference.

WHY stdlib only: the point is to expose the *mechanism* of governance documentation
from the inside (no SaaS card generator, no policy SaaS), so it stays auditable and
reproducible anywhere. Real tools to look at: Hugging Face model-card toolkit,
GSN/CAE safety-case notations.

# requires: stdlib only
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# WHY: on some terminals (e.g. Windows cp1252) printing non-ASCII markers would
# crash with UnicodeEncodeError. Make stdout robust without any external dep or
# env var, so `python <file>` always runs to completion.
try:  # pragma: no cover - depends on the terminal
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Shared mini agent model (kept consistent across the domain's days:
# id / owner / permissions / risk_tier). Reused here, but self-contained.
# ---------------------------------------------------------------------------
@dataclass
class GovernedAgent:
    """The minimal governable agent: the 4 pillars in data form."""

    agent_id: str
    owner: str  # a NAMED human accountable for the agent (pillar 2)
    purpose: str
    permissions: list[str]  # what it is allowed to do (pillar 3 — least privilege)
    risk_tier: str  # e.g. "high" / "limited" / "minimal" (EU AI Act vocabulary)
    handles_personal_data: bool = False
    legal_basis: Optional[str] = None  # GDPR base if personal data is processed
    known_limitations: list[str] = field(default_factory=list)
    eval_summary: Optional[str] = None  # e.g. "94% correct on 500-ticket set, 2026-05"
    status: str = "draft"  # draft / staging / production / retired


# ---------------------------------------------------------------------------
# 1. AGENT CARD GENERATOR  (template structure -> Markdown)
# ---------------------------------------------------------------------------
def render_agent_card(agent: GovernedAgent) -> str:
    """Render an agent card to Markdown.

    WHY a card and not a free-form doc: a *standardized* structure forces the
    author to fill the fields that auditors actually ask about — including the
    uncomfortable ones (personal data, legal basis, known limitations). An empty
    required field becomes visible instead of silently absent.
    """
    # WHY normalize personal-data line: an auditor must never have to guess.
    if agent.handles_personal_data:
        pd_line = f"yes — legal basis: {agent.legal_basis or 'MISSING ⚠️'}"
    else:
        pd_line = "no"

    lines = [
        f"# Agent Card — {agent.agent_id}",
        "",
        "## Identity & ownership",
        f"- **Agent ID**: {agent.agent_id}",
        f"- **Owner (accountable human)**: {agent.owner}",
        f"- **Status**: {agent.status}",
        f"- **Risk tier**: {agent.risk_tier}",
        "",
        "## Intended use",
        f"- **Purpose**: {agent.purpose}",
        "",
        "## Permissions (least privilege)",
    ]
    # WHY list permissions explicitly: "what it is allowed to do" is the surface
    # an auditor checks against the safety case's `control` arguments.
    if agent.permissions:
        lines += [f"- `{perm}`" for perm in agent.permissions]
    else:
        lines.append("- (none declared) ⚠️")

    lines += [
        "",
        "## Personal data",
        f"- {pd_line}",
        "",
        "## Evaluation",
        f"- {agent.eval_summary or 'no evaluation recorded ⚠️'}",
        "",
        "## Known limitations",
    ]
    # WHY surface limitations as a first-class section: a card with no declared
    # limitation is a marketing brochure, not an audit artifact.
    if agent.known_limitations:
        lines += [f"- {lim}" for lim in agent.known_limitations]
    else:
        lines.append("- (none declared) ⚠️")

    return "\n".join(lines)


def card_completeness(agent: GovernedAgent) -> tuple[float, list[str]]:
    """Return (completeness ratio, list of missing/weak fields).

    WHY a completeness check: documentation that *looks* full but omits the
    awkward fields is the most common failure. We flag the gaps mechanically.
    """
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
    # 5 checks total; completeness = fraction passed.
    completeness = 1.0 - len(issues) / 5.0
    return completeness, issues


# ---------------------------------------------------------------------------
# 2. SAFETY CASE SKELETON  (claim -> evidence -> gaps)
# ---------------------------------------------------------------------------
class ArgumentType(Enum):
    """The 4 safety-case argument types [Clymer et al., 2024], strongest first.

    The integer value encodes robustness (lower = stronger). We use it to nudge
    authors toward verifiable arguments ("it *cannot*") over wagered ones
    ("it *would not*").
    """

    INABILITY = 1  # it cannot cause the harm (no capability)
    CONTROL = 2  # guardrails prevent it (sandbox, budget, kill-switch)
    TRUSTWORTHINESS = 3  # we believe it would not (evals, alignment)
    DEFERENCE = 4  # a credible authority attests it is safe


@dataclass
class Claim:
    """One assurance claim and its supporting structure."""

    statement: str
    argument_type: ArgumentType
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        # WHY: a claim with zero evidence is an opinion, not an argument.
        return len(self.evidence) > 0


@dataclass
class SafetyCase:
    """A structured assurance argument for one agent in one context."""

    agent_id: str
    context: str  # e.g. "production, EU, refunds < 200 EUR"
    claims: list[Claim] = field(default_factory=list)

    def coverage(self) -> float:
        """Fraction of claims backed by at least one piece of evidence."""
        if not self.claims:
            return 0.0
        supported = sum(1 for c in self.claims if c.is_supported)
        return supported / len(self.claims)

    def open_gaps(self) -> list[tuple[str, str]]:
        """All declared gaps, as (claim statement, gap)."""
        return [(c.statement, g) for c in self.claims for g in c.gaps]

    def weakest_arguments(self) -> list[Claim]:
        """Claims relying on the most fragile argument types (sorted).

        WHY: surfacing trustworthiness/deference-based claims tells a reviewer
        where the case is *wagered* rather than *verified* — the riskiest spots.
        """
        return sorted(
            self.claims,
            key=lambda c: c.argument_type.value,
            reverse=True,
        )

    def render(self) -> str:
        """Render the safety case to Markdown, gaps included.

        WHY render gaps prominently: a safety case that hides its lacunae is not
        reassuring, it is non-credible. Honesty about gaps is what makes it
        auditable.
        """
        cov = self.coverage()
        lines = [
            f"# Safety Case — {self.agent_id}",
            f"_Context: {self.context}_",
            "",
            f"**Claim coverage (≥1 evidence): {cov:.0%}**",
            "",
        ]
        for i, c in enumerate(self.claims, start=1):
            badge = "OK" if c.is_supported else "UNSUPPORTED ⚠️"
            lines += [
                f"## Claim {i} — [{c.argument_type.name}] {badge}",
                f"> {c.statement}",
                "",
                "**Evidence:**",
            ]
            lines += (
                [f"- {e}" for e in c.evidence]
                if c.evidence
                else ["- (none) ⚠️ — this claim is currently an opinion"]
            )
            lines += ["", "**Gaps:**"]
            lines += (
                [f"- {g}" for g in c.gaps]
                if c.gaps
                else ["- (none declared) ⚠️ — a case with no gaps is suspect"]
            )
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo() -> None:
    agent = GovernedAgent(
        agent_id="refund-agent@v2.3",
        owner="Camille Roux (Head of CX)",
        purpose="Decide refund requests under 200 EUR from SAV tickets.",
        permissions=["read:ticket", "read:order_history", "credit:<=200_eur"],
        risk_tier="limited",
        handles_personal_data=True,
        legal_basis="performance of the contract (GDPR art. 6.1.b)",
        known_limitations=[
            "French language only",
            "possible bias on aggressive tickets",
        ],
        eval_summary="94% correct classification on 500-ticket set (2026-05)",
        status="production",
    )

    print("=" * 64)
    print("AGENT CARD (rendered Markdown)")
    print("=" * 64)
    print(render_agent_card(agent))

    ratio, issues = card_completeness(agent)
    print()
    print(f"Card completeness: {ratio:.0%}")
    print("Open card issues:", issues or "none")

    # Adversarial probe: an under-documented agent should expose its gaps.
    sloppy = GovernedAgent(
        agent_id="mystery-bot@v0.1",
        owner="",  # missing owner -> ungoverned
        purpose="???",
        permissions=[],  # no declared permissions
        risk_tier="unknown",
        handles_personal_data=True,
        legal_basis=None,  # personal data with no legal basis
    )
    s_ratio, s_issues = card_completeness(sloppy)
    print()
    print(f"Sloppy card completeness: {s_ratio:.0%}")
    print("Sloppy card issues:", s_issues)

    print()
    print("=" * 64)
    print("SAFETY CASE (rendered Markdown)")
    print("=" * 64)
    case = SafetyCase(
        agent_id="refund-agent@v2.3",
        context="production, EU, refunds < 200 EUR",
    )
    case.claims.append(
        Claim(
            statement="The agent cannot credit more than 200 EUR without a human.",
            argument_type=ArgumentType.INABILITY,
            evidence=[
                "policy engine hard-caps credit at 200 EUR (config v2.3)",
                "200 adversarial over-limit attempts blocked (eval J13, 2026-05)",
            ],
            gaps=["not tested under concurrent load"],
        )
    )
    case.claims.append(
        Claim(
            statement="The agent cannot exfiltrate the customer database.",
            argument_type=ArgumentType.CONTROL,
            evidence=["scope limited to read:ticket + read:order_history (least privilege)"],
            gaps=["no independent review of the scope enforcement layer"],
        )
    )
    # WHY include an unsupported claim: to show the tool flags opinions, and that
    # a trustworthiness argument (weaker) is surfaced as such.
    case.claims.append(
        Claim(
            statement="The agent does not produce discriminatory refusals.",
            argument_type=ArgumentType.TRUSTWORTHINESS,
            evidence=[],  # deliberately empty -> opinion, not argument
            gaps=["fairness across customer segments not yet evaluated"],
        )
    )
    print(case.render())

    print(f"Overall claim coverage: {case.coverage():.0%}")
    print(f"Open gaps: {len(case.open_gaps())}")
    weakest = case.weakest_arguments()[0]
    print(
        "Most fragile argument:",
        f"[{weakest.argument_type.name}] {weakest.statement}",
    )


if __name__ == "__main__":
    _demo()

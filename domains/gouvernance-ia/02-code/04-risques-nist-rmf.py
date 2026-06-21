"""
Day 4 -- Risk taxonomy & NIST AI RMF: a runnable risk register + scorer.

Demonstrates the governance mechanism of the day:
  1. A typed Risk record carrying its taxonomy coordinates (causal axes +
     domain), its RMF function, and likelihood/impact on anchored 1..5 scales.
  2. A scorer that computes `criticality = likelihood * impact` AND applies the
     two AGENTIC MODULATORS from the theory (irreversible action raises impact,
     full autonomy raises likelihood) -- because the same logical risk is worse
     on an autonomous + irreversible agent.
  3. A treatment-threshold rule (Measure -> Manage) that turns a score into a
     decision: TREAT / MONITOR / ACCEPT.
  4. A risk register that sorts by criticality and aggregates coverage per RMF
     function -- the board-ready artifact described in section 6.

Real tools this mirrors (re-implemented in miniature, stdlib only):
  - NIST AI RMF 1.0 (AI 100-1, 2023): the Govern/Map/Measure/Manage functions.
  - MIT AI Risk Repository (Slattery et al., 2024): the causal + domain taxonomy.
A production setup would persist this register (a DB), feed Measure from real
evals, and map Manage decisions to tickets. Here everything is in-memory.

# requires: stdlib only
Run:
    python 04-risques-nist-rmf.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# 1. TAXONOMY -- stable vocabulary so two people name the same risk identically.
#    Mirrors the MIT AI Risk Repository's two taxonomies.
# ---------------------------------------------------------------------------


class RMFFunction(str, Enum):
    """The 4 NIST AI RMF functions. GOVERN is transversal (wraps the others)."""

    GOVERN = "GOVERN"   # who is responsible, which rules, which culture
    MAP = "MAP"         # which risks, in which context
    MEASURE = "MEASURE"  # how severe, quantified / tested
    MANAGE = "MANAGE"   # what we do: treat / accept / transfer


class CausalEntity(str, Enum):
    HUMAN = "human"
    AI = "ai"


class CausalIntent(str, Enum):
    INTENTIONAL = "intentional"
    UNINTENTIONAL = "unintentional"


class CausalTiming(str, Enum):
    PRE_DEPLOYMENT = "pre-deployment"
    POST_DEPLOYMENT = "post-deployment"


# Anchored scales (1..5). We store the human-readable anchors so a score can be
# EXPLAINED, not just asserted. This is what makes a score "defendable".
LIKELIHOOD_ANCHORS = {
    1: "rare (not seen in a year)",
    2: "unlikely",
    3: "possible (a few times/year)",
    4: "likely (monthly)",
    5: "near-certain (continuous)",
}
IMPACT_ANCHORS = {
    1: "negligible (minor annoyance)",
    2: "limited (recoverable, low cost)",
    3: "serious (notable loss, manual fix)",
    4: "grave (financial/legal, hard to undo)",
    5: "critical (irreversible, safety, sanction)",
}


@dataclass
class Risk:
    """One governable risk: cause -> effect -> impact, named, classed, scored."""

    risk_id: str
    agent_id: str
    title: str
    # Causal taxonomy coordinates (MIT AI Risk Repository).
    entity: CausalEntity
    intent: CausalIntent
    timing: CausalTiming
    domain: str  # e.g. "security & system failures"
    rmf_function: RMFFunction
    # Raw (un-modulated) anchored scores, 1..5.
    likelihood: int
    impact: int
    # Agentic context -> drives the modulators.
    irreversible: bool = False   # action cannot be undone (transfer, delete)
    autonomous: bool = False     # acts with NO human in the loop

    def __post_init__(self) -> None:
        # Adversarial guardrail: refuse out-of-range scores rather than score
        # nonsense. A register that silently accepts a 7/5 is worse than one
        # that raises -- garbage scores destroy the artifact's credibility.
        for name, value in (("likelihood", self.likelihood), ("impact", self.impact)):
            if value not in range(1, 6):
                raise ValueError(
                    f"{name}={value!r} for risk {self.risk_id!r} is out of the "
                    f"anchored 1..5 scale"
                )


# ---------------------------------------------------------------------------
# 2. SCORER -- criticality with agentic modulators.
# ---------------------------------------------------------------------------

TREAT_THRESHOLD = 12   # criticality >= 12 -> must be treated (Manage)
MONITOR_THRESHOLD = 6  # 6..11 -> monitor; < 6 -> accept + document


@dataclass
class Scored:
    """Result of scoring a Risk: effective scores, criticality, decision."""

    risk: Risk
    eff_likelihood: int
    eff_impact: int
    criticality: int
    decision: str          # TREAT | MONITOR | ACCEPT
    rationale: list[str] = field(default_factory=list)


def score_risk(risk: Risk) -> Scored:
    """
    WHY modulators: the same logical risk is worse on an autonomous + irreversible
    agent. Ignoring that makes the score lie (theory section 5.2).
      - irreversible action  -> impact +1 (nothing to roll back)
      - full autonomy        -> likelihood +1 (no human net to intercept)
    We cap effective scores at 5 so the scale stays meaningful.
    """
    rationale: list[str] = []
    eff_likelihood = risk.likelihood
    eff_impact = risk.impact

    if risk.irreversible:
        eff_impact = min(5, eff_impact + 1)
        rationale.append("irreversible action -> impact +1")
    if risk.autonomous:
        eff_likelihood = min(5, eff_likelihood + 1)
        rationale.append("no human in the loop -> likelihood +1")

    criticality = eff_likelihood * eff_impact

    # Measure -> Manage: the score becomes a decision via an assumed threshold.
    if criticality >= TREAT_THRESHOLD:
        decision = "TREAT"
    elif criticality >= MONITOR_THRESHOLD:
        decision = "MONITOR"
    else:
        decision = "ACCEPT"

    return Scored(
        risk=risk,
        eff_likelihood=eff_likelihood,
        eff_impact=eff_impact,
        criticality=criticality,
        decision=decision,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# 3. RISK REGISTER -- the board-ready artifact: sorted + aggregated.
# ---------------------------------------------------------------------------


class RiskRegister:
    """In-memory risk register. Real life: a DB or a GRC tool."""

    def __init__(self) -> None:
        self._risks: list[Risk] = []

    def add(self, risk: Risk) -> None:
        if any(r.risk_id == risk.risk_id for r in self._risks):
            raise ValueError(f"duplicate risk_id {risk.risk_id!r}")
        self._risks.append(risk)

    def scored_sorted(self) -> list[Scored]:
        """All risks scored, sorted by criticality DESC (worst first)."""
        scored = [score_risk(r) for r in self._risks]
        scored.sort(key=lambda s: s.criticality, reverse=True)
        return scored

    def coverage_by_function(self) -> dict[str, int]:
        """How many risks are tagged to each RMF function (coverage check)."""
        counts = {f.value: 0 for f in RMFFunction}
        for r in self._risks:
            counts[r.rmf_function.value] += 1
        return counts

    def to_decisions(self) -> dict[str, int]:
        """Tally of TREAT / MONITOR / ACCEPT across the register."""
        tally = {"TREAT": 0, "MONITOR": 0, "ACCEPT": 0}
        for s in self.scored_sorted():
            tally[s.decision] += 1
        return tally


# ---------------------------------------------------------------------------
# 4. DEMO DATA -- a small agent fleet, including the invoice-reconciler.
# ---------------------------------------------------------------------------


def _demo_register() -> RiskRegister:
    reg = RiskRegister()
    reg.add(Risk(
        risk_id="R-001",
        agent_id="invoice-reconciler",
        title="Phishing email (prompt injection) -> fraudulent wire transfer",
        entity=CausalEntity.AI, intent=CausalIntent.UNINTENTIONAL,
        timing=CausalTiming.POST_DEPLOYMENT,
        domain="security & malicious actors",
        rmf_function=RMFFunction.MAP,
        likelihood=3, impact=4,           # impact 4 raw; irreversible bumps to 5
        irreversible=True, autonomous=True,
    ))
    reg.add(Risk(
        risk_id="R-002",
        agent_id="support-bot",
        title="Hallucinated refund policy quoted to a customer",
        entity=CausalEntity.AI, intent=CausalIntent.UNINTENTIONAL,
        timing=CausalTiming.POST_DEPLOYMENT,
        domain="misinformation",
        rmf_function=RMFFunction.MEASURE,
        likelihood=4, impact=2,
        irreversible=False, autonomous=False,  # human-in-the-loop, reversible
    ))
    reg.add(Risk(
        risk_id="R-003",
        agent_id="data-cleaner",
        title="Tool misuse: bulk DELETE on a production table",
        entity=CausalEntity.AI, intent=CausalIntent.UNINTENTIONAL,
        timing=CausalTiming.POST_DEPLOYMENT,
        domain="security & system failures",
        rmf_function=RMFFunction.MANAGE,
        likelihood=2, impact=5,
        irreversible=True, autonomous=True,
    ))
    reg.add(Risk(
        risk_id="R-004",
        agent_id="report-writer",
        title="Reads PII into a summary without legal basis",
        entity=CausalEntity.AI, intent=CausalIntent.UNINTENTIONAL,
        timing=CausalTiming.PRE_DEPLOYMENT,
        domain="privacy & data",
        rmf_function=RMFFunction.GOVERN,
        likelihood=2, impact=3,
        irreversible=False, autonomous=False,
    ))
    return reg


# ---------------------------------------------------------------------------
# 5. RENDER -- a readable report a board member could read.
# ---------------------------------------------------------------------------


def render_report(reg: RiskRegister) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("AGENT FLEET RISK REGISTER  (NIST AI RMF -- Map/Measure/Manage)")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"{'RISK':<6} {'AGENT':<20} {'L':>2} {'I':>2} {'CRIT':>5} "
                 f"{'DECISION':<8} RMF")
    lines.append("-" * 68)
    for s in reg.scored_sorted():
        lines.append(
            f"{s.risk.risk_id:<6} {s.risk.agent_id:<20} "
            f"{s.eff_likelihood:>2} {s.eff_impact:>2} {s.criticality:>5} "
            f"{s.decision:<8} {s.risk.rmf_function.value}"
        )
    lines.append("")
    lines.append("Worst risk in detail:")
    worst = reg.scored_sorted()[0]
    lines.append(f"  {worst.risk.risk_id} -- {worst.risk.title}")
    lines.append(f"  causal: {worst.risk.entity.value} / "
                 f"{worst.risk.intent.value} / {worst.risk.timing.value}")
    lines.append(f"  domain: {worst.risk.domain}")
    lines.append(f"  raw L*I = {worst.risk.likelihood}*{worst.risk.impact}; "
                 f"effective = {worst.eff_likelihood}*{worst.eff_impact} "
                 f"= {worst.criticality}")
    for r in worst.rationale:
        lines.append(f"    modulator: {r}")
    lines.append("")
    lines.append("RMF function coverage (how many risks per function):")
    for fn, n in reg.coverage_by_function().items():
        lines.append(f"  {fn:<8}: {n}")
    lines.append("")
    lines.append(f"Treatment decisions: {reg.to_decisions()}")
    lines.append(f"(threshold: TREAT >= {TREAT_THRESHOLD}, "
                 f"MONITOR {MONITOR_THRESHOLD}..{TREAT_THRESHOLD - 1}, "
                 f"ACCEPT < {MONITOR_THRESHOLD})")
    return "\n".join(lines)


if __name__ == "__main__":
    reg = _demo_register()
    print(render_report(reg))

    # Adversarial probe: an out-of-range score must be refused, not stored.
    print("\nAdversarial probe (likelihood=7 should be rejected):")
    try:
        Risk(
            risk_id="R-BAD", agent_id="x", title="bad score",
            entity=CausalEntity.AI, intent=CausalIntent.UNINTENTIONAL,
            timing=CausalTiming.POST_DEPLOYMENT, domain="x",
            rmf_function=RMFFunction.MAP, likelihood=7, impact=3,
        )
        print("  ERROR: out-of-range score was accepted (bug!)")
    except ValueError as exc:
        print(f"  OK, rejected: {exc}")

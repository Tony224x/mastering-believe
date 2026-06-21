"""
Day 4 -- Solutions: risk taxonomy & NIST AI RMF.

One file, three levels separated by markers. stdlib only.
Mirrors NIST AI RMF 1.0 (AI 100-1, 2023) functions and the MIT AI Risk
Repository (Slattery et al., 2024) causal taxonomy -- re-implemented in
miniature (no external deps).

# requires: stdlib only
Run:
    python 04-risques-nist-rmf.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# =========================================================================
# === EASY ===
# Classify and score ONE risk: name it, give causal coords, RMF function,
# anchored likelihood/impact, and criticality. Here we encode the *answer*
# for the calendar-assistant scenario as a structured, checkable record.
# =========================================================================

EASY_ANSWER = {
    # cause -> effect -> impact, in one phrase
    "one_line": (
        "A spoofed HR email (prompt injection) -> the agent cancels legitimate "
        "meetings -> lost coordination, hard to fully restore (impact)"
    ),
    # 3 causal coordinates (MIT AI Risk Repository causal taxonomy)
    "entity": "ai",                 # the agent performs the deletion
    "intent": "unintentional",      # the agent does not "want" to harm
    "timing": "post-deployment",    # happens in operation
    "domain": "security & system failures",
    # RMF function + justification
    "rmf_function": "MAP",
    "rmf_justification": "We are identifying/contextualizing the risk -> Map.",
    # Anchored scores
    "likelihood": 3,  # possible: spoofed internal emails happen a few times/yr
    "impact": 3,      # serious: notable disruption, manual restore needed
}


def easy_criticality(answer: dict) -> int:
    """Compute likelihood * impact for the easy answer."""
    return int(answer["likelihood"]) * int(answer["impact"])


def easy_demo() -> None:
    a = EASY_ANSWER
    print("[EASY] risk:", a["one_line"])
    print(f"[EASY] causal: {a['entity']} / {a['intent']} / {a['timing']}")
    print(f"[EASY] domain: {a['domain']}")
    print(f"[EASY] RMF: {a['rmf_function']} -- {a['rmf_justification']}")
    print(f"[EASY] likelihood={a['likelihood']} (possible), "
          f"impact={a['impact']} (serious) -> "
          f"criticality={easy_criticality(a)}")


# =========================================================================
# === MEDIUM ===
# A scorer that applies the agentic modulators, plus a decision threshold.
# Show that the SAME raw risk scores higher when autonomous + irreversible.
# =========================================================================

def score(likelihood: int, impact: int, irreversible: bool,
          autonomous: bool) -> dict:
    """Score with agentic modulators (theory section 5.2). Caps at 5."""
    if likelihood not in range(1, 6):
        raise ValueError(f"likelihood={likelihood!r} out of 1..5 scale")
    if impact not in range(1, 6):
        raise ValueError(f"impact={impact!r} out of 1..5 scale")

    eff_likelihood = likelihood
    eff_impact = impact
    rationale: list[str] = []

    if irreversible:
        eff_impact = min(5, eff_impact + 1)       # nothing to roll back
        rationale.append("irreversible -> impact +1")
    if autonomous:
        eff_likelihood = min(5, eff_likelihood + 1)  # no human net
        rationale.append("autonomous -> likelihood +1")

    return {
        "eff_likelihood": eff_likelihood,
        "eff_impact": eff_impact,
        "criticality": eff_likelihood * eff_impact,
        "rationale": rationale,
    }


def decision(criticality: int) -> str:
    """Measure -> Manage: turn a score into a treatment decision."""
    if criticality >= 12:
        return "TREAT"
    if criticality >= 6:
        return "MONITOR"
    return "ACCEPT"


def medium_demo() -> None:
    # Same raw risk, two agentic contexts.
    case_a = score(3, 4, irreversible=False, autonomous=False)
    case_b = score(3, 4, irreversible=True, autonomous=True)
    print(f"[MEDIUM] case A (hitl, reversible):  crit={case_a['criticality']} "
          f"-> {decision(case_a['criticality'])}")
    print(f"[MEDIUM] case B (auto, irreversible): crit={case_b['criticality']} "
          f"-> {decision(case_b['criticality'])}")
    print(f"[MEDIUM] B rationale: {case_b['rationale']}")
    assert case_b["criticality"] > case_a["criticality"], "B must exceed A"


# =========================================================================
# === HARD ===
# A board-ready risk register: typed risks, sorting, RMF coverage, gap
# detection, and a rendered report with a treatment tally.
# =========================================================================

class RMFFunction(str, Enum):
    GOVERN = "GOVERN"
    MAP = "MAP"
    MEASURE = "MEASURE"
    MANAGE = "MANAGE"


@dataclass
class Risk:
    risk_id: str
    agent_id: str
    title: str
    entity: str          # "human" | "ai"
    intent: str          # "intentional" | "unintentional"
    timing: str          # "pre-deployment" | "post-deployment"
    domain: str
    rmf_function: RMFFunction
    likelihood: int
    impact: int
    irreversible: bool = False
    autonomous: bool = False

    def __post_init__(self) -> None:
        for name, value in (("likelihood", self.likelihood),
                            ("impact", self.impact)):
            if value not in range(1, 6):
                raise ValueError(f"{name}={value!r} out of 1..5 for {self.risk_id}")


@dataclass
class Scored:
    risk: Risk
    eff_likelihood: int
    eff_impact: int
    criticality: int
    decision: str
    rationale: list[str] = field(default_factory=list)


def _score_risk(risk: Risk) -> Scored:
    res = score(risk.likelihood, risk.impact, risk.irreversible, risk.autonomous)
    return Scored(
        risk=risk,
        eff_likelihood=res["eff_likelihood"],
        eff_impact=res["eff_impact"],
        criticality=res["criticality"],
        decision=decision(res["criticality"]),
        rationale=res["rationale"],
    )


class RiskRegister:
    def __init__(self) -> None:
        self._risks: list[Risk] = []

    def add(self, risk: Risk) -> None:
        if any(r.risk_id == risk.risk_id for r in self._risks):
            raise ValueError(f"duplicate risk_id {risk.risk_id!r}")
        self._risks.append(risk)

    def scored_sorted(self) -> list[Scored]:
        scored = [_score_risk(r) for r in self._risks]
        scored.sort(key=lambda s: s.criticality, reverse=True)
        return scored

    def coverage_by_function(self) -> dict[str, int]:
        counts = {f.value: 0 for f in RMFFunction}
        for r in self._risks:
            counts[r.rmf_function.value] += 1
        return counts

    def missing_functions(self) -> list[str]:
        """RMF functions with zero risks -- a governance gap to flag."""
        return [fn for fn, n in self.coverage_by_function().items() if n == 0]

    def decisions_tally(self) -> dict[str, int]:
        tally = {"TREAT": 0, "MONITOR": 0, "ACCEPT": 0}
        for s in self.scored_sorted():
            tally[s.decision] += 1
        return tally


def _hard_register() -> RiskRegister:
    reg = RiskRegister()
    reg.add(Risk("R-001", "invoice-reconciler",
                 "Prompt injection -> fraudulent wire transfer",
                 "ai", "unintentional", "post-deployment",
                 "security & malicious actors", RMFFunction.MAP,
                 likelihood=3, impact=4, irreversible=True, autonomous=True))
    reg.add(Risk("R-002", "support-bot",
                 "Hallucinated refund policy quoted to a customer",
                 "ai", "unintentional", "post-deployment",
                 "misinformation", RMFFunction.MEASURE,
                 likelihood=4, impact=2, irreversible=False, autonomous=False))
    reg.add(Risk("R-003", "data-cleaner",
                 "Tool misuse: bulk DELETE on production table",
                 "ai", "unintentional", "post-deployment",
                 "security & system failures", RMFFunction.MANAGE,
                 likelihood=2, impact=5, irreversible=True, autonomous=True))
    reg.add(Risk("R-004", "report-writer",
                 "Reads PII into a summary without legal basis",
                 "ai", "unintentional", "pre-deployment",
                 "privacy & data", RMFFunction.MANAGE,
                 likelihood=2, impact=3, irreversible=False, autonomous=False))
    return reg


def render_report(reg: RiskRegister) -> str:
    lines: list[str] = []
    lines.append(f"{'RISK':<6} {'AGENT':<20} {'L':>2} {'I':>2} {'CRIT':>5} "
                 f"{'DECISION':<8} RMF")
    lines.append("-" * 60)
    for s in reg.scored_sorted():
        lines.append(f"{s.risk.risk_id:<6} {s.risk.agent_id:<20} "
                     f"{s.eff_likelihood:>2} {s.eff_impact:>2} "
                     f"{s.criticality:>5} {s.decision:<8} "
                     f"{s.risk.rmf_function.value}")
    worst = reg.scored_sorted()[0]
    lines.append("")
    lines.append(f"Worst: {worst.risk.risk_id} -- {worst.risk.title}")
    lines.append(f"  causal: {worst.risk.entity} / {worst.risk.intent} / "
                 f"{worst.risk.timing}")
    lines.append(f"  modulators: {worst.rationale or ['none']}")
    lines.append(f"Coverage: {reg.coverage_by_function()}")
    missing = reg.missing_functions()
    if missing:
        lines.append(f"  WARNING: RMF functions with no risk mapped: {missing}")
    lines.append(f"Decisions: {reg.decisions_tally()}")
    return "\n".join(lines)


def hard_demo() -> None:
    reg = _hard_register()
    print("[HARD]")
    print(render_report(reg))


# =========================================================================
# SMOKE TEST
# =========================================================================

if __name__ == "__main__":
    easy_demo()
    print()
    medium_demo()
    print()
    hard_demo()

    # --- assertions: prove the mechanisms, not just print ---
    # EASY: criticality is the product.
    assert easy_criticality(EASY_ANSWER) == 9

    # MEDIUM: modulators raise the score; out-of-range rejected; caps at 5.
    a = score(3, 4, False, False)
    b = score(3, 4, True, True)
    assert b["criticality"] > a["criticality"]
    assert b["eff_impact"] == 5 and b["eff_likelihood"] == 4
    capped = score(5, 5, True, True)  # already maxed -> stays 5/5
    assert capped["eff_likelihood"] == 5 and capped["eff_impact"] == 5
    for bad in (0, 6, 7):
        try:
            score(bad, 3, False, False)
            raise AssertionError("out-of-range likelihood not rejected")
        except ValueError:
            pass
    assert decision(15) == "TREAT"
    assert decision(8) == "MONITOR"
    assert decision(4) == "ACCEPT"

    # HARD: register invariants.
    reg = _hard_register()
    crits = [s.criticality for s in reg.scored_sorted()]
    assert crits == sorted(crits, reverse=True), "must be sorted desc"
    try:
        reg.add(Risk("R-001", "dup", "dup", "ai", "unintentional",
                     "post-deployment", "x", RMFFunction.MAP, 1, 1))
        raise AssertionError("duplicate risk_id not rejected")
    except ValueError:
        pass
    # GOVERN has no risk in our demo -> must be flagged as missing.
    assert "GOVERN" in reg.missing_functions()
    # R-001 worst: raw 3*4=12, irreversible+autonomous -> 4*5=20.
    worst = reg.scored_sorted()[0]
    assert worst.risk.risk_id == "R-001" and worst.criticality == 20

    print("\nAll smoke-test assertions passed.")

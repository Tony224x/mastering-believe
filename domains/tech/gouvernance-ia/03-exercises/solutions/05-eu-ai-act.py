"""
Solutions — Day 5: EU AI Act & third-party / GPAI governance.

Covers the three exercise levels. Self-contained: the few primitives from
02-code/05-eu-ai-act.py (Tier, SystemProfile, classify, due_diligence,
SupplierComponent, DEADLINES) are re-declared here so the file runs standalone
without importing another day's module.

All dates / tiers follow Regulation (EU) 2024/1689 (see REFERENCES.md).

# requires: stdlib only

Run:  python 05-eu-ai-act.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


# ---------------------------------------------------------------------------
# Shared primitives (mirrors 02-code/05-eu-ai-act.py) so this file is standalone.
# ---------------------------------------------------------------------------
class Tier(Enum):
    UNACCEPTABLE = 4
    HIGH = 3
    LIMITED = 2
    MINIMAL = 1


DEADLINES = {
    "prohibitions": "2025-02-02",
    "gpai": "2025-08-02",
    "high_risk_annex_iii": "2026-08-02",
    "high_risk_annex_i": "2027-08-02",
    "limited": "2026-08-02",
    "minimal": "n/a",
}


@dataclass
class SystemProfile:
    name: str
    is_social_scoring_by_authority: bool = False
    uses_subliminal_manipulation: bool = False
    does_untargeted_face_scraping: bool = False
    is_annex_iii_use: bool = False
    is_annex_i_safety_component: bool = False
    interacts_with_humans_as_ai: bool = False
    builds_on_third_party_gpai: bool = False
    agent_takes_actions: bool = False


@dataclass
class Classification:
    name: str
    tier: Tier
    reason: str
    obligations: list[str] = field(default_factory=list)
    deadline: str = "n/a"
    notes: list[str] = field(default_factory=list)


def classify(profile: SystemProfile) -> Classification:
    p = profile
    if p.is_social_scoring_by_authority or p.uses_subliminal_manipulation or p.does_untargeted_face_scraping:
        return Classification(
            name=p.name,
            tier=Tier.UNACCEPTABLE,
            reason="Art. 5 prohibited practice",
            obligations=["PROHIBITED — do not deploy."],
            deadline=DEADLINES["prohibitions"],
        )
    if p.is_annex_i_safety_component or p.is_annex_iii_use:
        if p.is_annex_iii_use:
            annex, deadline = "Annex III", DEADLINES["high_risk_annex_iii"]
        else:
            annex, deadline = "Annex I", DEADLINES["high_risk_annex_i"]
        obligations = [
            "Risk management (Art. 9)",
            "Data governance (Art. 10)",
            "Technical documentation (Art. 11)",
            "Record-keeping (Art. 12)",
            "Transparency / instructions (Art. 13)",
            "Human oversight (Art. 14)",
            "Accuracy, robustness, cybersecurity (Art. 15)",
        ]
        c = Classification(p.name, Tier.HIGH, f"High risk via {annex}", obligations, deadline)
        if p.agent_takes_actions:
            c.notes.append("Agent acts: human oversight (Art. 14) is the critical control.")
        return c
    if p.interacts_with_humans_as_ai:
        return Classification(
            name=p.name,
            tier=Tier.LIMITED,
            reason="Transparency duty (Art. 50)",
            obligations=["Inform the person they interact with an AI (Art. 50)"],
            deadline=DEADLINES["limited"],
        )
    return Classification(
        name=p.name,
        tier=Tier.MINIMAL,
        reason="No trigger matched",
        obligations=["None imposed"],
        deadline=DEADLINES["minimal"],
    )


@dataclass
class SupplierComponent:
    name: str
    supplier: str
    your_tier: Tier
    is_or_uses_gpai: bool
    vendor_provides_gpai_docs: bool
    vendor_provides_usage_docs: bool
    responsibilities_contractualised: bool


def due_diligence(c: SupplierComponent) -> tuple[bool, list[str]]:
    gaps: list[str] = []
    if c.your_tier == Tier.UNACCEPTABLE:
        gaps.append("Prohibited use — cannot deploy.")
    if c.is_or_uses_gpai and not c.vendor_provides_gpai_docs:
        gaps.append("GPAI component without vendor GPAI documentation.")
    if not c.vendor_provides_usage_docs:
        gaps.append("No usage docs for downstream duties.")
    if not c.responsibilities_contractualised:
        gaps.append("Responsibility split not contractualised.")
    if c.your_tier == Tier.HIGH and not c.vendor_provides_usage_docs:
        gaps.append("High-risk use without vendor usage docs: cannot meet Art. 11/13.")
    return (len(gaps) == 0, gaps)


# ==========================================================================
# EASY — Classify five uses into the right tier and name the deadline.
# ==========================================================================
def easy_classify_five():
    profiles = [
        SystemProfile(name="a) Internal spam filter"),
        SystemProfile(name="b) Conversational AI assistant", interacts_with_humans_as_ai=True),
        SystemProfile(name="c) CV-screening for recruitment", is_annex_iii_use=True),
        SystemProfile(name="d) Generalised social scoring by an authority",
                      is_social_scoring_by_authority=True),
        SystemProfile(name="e) AI safety component of a medical device",
                      is_annex_i_safety_component=True),
    ]
    results = []
    for p in profiles:
        c = classify(p)
        results.append((c.name, c.tier.name, c.deadline))
    return results


# ==========================================================================
# MEDIUM — Tier -> obligations + days remaining before the deadline.
# ==========================================================================
def days_until_deadline(classification: Classification, today: date):
    """Days from `today` to the deadline; None when deadline is 'n/a'."""
    if classification.deadline == "n/a":
        return None
    target = date.fromisoformat(classification.deadline)
    return (target - today).days


def compliance_brief(profile: SystemProfile, today: date) -> dict:
    c = classify(profile)
    return {
        "name": c.name,
        "tier": c.tier.name,
        "deadline": c.deadline,
        "days_left": days_until_deadline(c, today),
        "obligations": c.obligations,
    }


def medium_briefs(today: date):
    profiles = [
        SystemProfile(name="Credit scoring (Annex III)", is_annex_iii_use=True),
        SystemProfile(name="Medical device component (Annex I)", is_annex_i_safety_component=True),
        SystemProfile(name="Support chatbot", interacts_with_humans_as_ai=True),
        SystemProfile(name="Spam filter"),
    ]
    briefs = [compliance_brief(p, today) for p in profiles]
    # Most urgent first; None (no deadline) goes last via a large sentinel key.
    briefs.sort(key=lambda b: (b["days_left"] is None, b["days_left"] if b["days_left"] is not None else 0))
    return briefs


# ==========================================================================
# HARD — Third-party governance register with due-diligence scoring.
# ==========================================================================
@dataclass
class RegisteredComponent:
    component_id: str
    owner: str                  # named human owner (one of the 4 pillars)
    usage: SystemProfile        # what WE use it for -> drives the tier
    supplier: SupplierComponent  # vendor diligence inputs


def assess(component: RegisteredComponent) -> dict:
    c = classify(component.usage)
    passes, gaps = due_diligence(component.supplier)
    # Risk score: tier weight amplified by the number of due-diligence gaps.
    # A high-risk use with open gaps must dominate the ranking.
    tier_weight = c.tier.value  # 4/3/2/1
    compliance_risk_score = tier_weight * (1 + len(gaps))
    return {
        "component_id": component.component_id,
        "owner": component.owner,
        "name": component.supplier.name,
        "tier": c.tier.name,
        "tier_enum": c.tier,
        "deadline": c.deadline,
        "passes": passes,
        "gaps": gaps,
        "compliance_risk_score": compliance_risk_score,
    }


def portfolio_report(components: list[RegisteredComponent], today: date) -> dict:
    assessed = [assess(comp) for comp in components]
    # BLOCKED for any unacceptable use, regardless of score.
    for a in assessed:
        a["status"] = "BLOCKED" if a["tier_enum"] == Tier.UNACCEPTABLE else "deployable"
    # Rank DEPLOYABLE components by compliance risk (worst first). BLOCKED
    # components are a separate category: they are not deployed at all, so they
    # do not compete for the "top risk to remediate" slot — they sink to the
    # bottom. WHY: the high-risk use you must actually fix should top the list,
    # not the one you have already rejected.
    assessed.sort(key=lambda a: (a["status"] == "BLOCKED", -a["compliance_risk_score"]))

    # Nearest deadline among deployable components, ignoring 'n/a'.
    upcoming = []
    for a in assessed:
        if a["status"] == "BLOCKED":
            continue
        if a["deadline"] != "n/a":
            upcoming.append(a["deadline"])
    nearest_deadline = min(upcoming) if upcoming else None  # ISO dates sort chronologically
    days_to_nearest = (date.fromisoformat(nearest_deadline) - today).days if nearest_deadline else None

    return {
        "generated_on": today.isoformat(),
        "n_components": len(assessed),
        "n_blocked": sum(1 for a in assessed if a["status"] == "BLOCKED"),
        "nearest_deadline": nearest_deadline,
        "days_to_nearest_deadline": days_to_nearest,
        "components": assessed,
    }


def _hard_sample_portfolio() -> list[RegisteredComponent]:
    return [
        RegisteredComponent(
            component_id="cmp-001", owner="alice@bank.eu",
            usage=SystemProfile(name="Credit engine", is_annex_iii_use=True, agent_takes_actions=True),
            supplier=SupplierComponent("Credit engine", "VendorCorp", Tier.HIGH,
                                       True, True, True, True),  # compliant
        ),
        RegisteredComponent(
            component_id="cmp-002", owner="bob@bank.eu",
            usage=SystemProfile(name="Hiring filter", is_annex_iii_use=True),
            supplier=SupplierComponent("Hiring filter", "HRTech", Tier.HIGH,
                                       True, False, False, False),  # gaps -> high score
        ),
        RegisteredComponent(
            component_id="cmp-003", owner="carol@bank.eu",
            usage=SystemProfile(name="Chatbot", interacts_with_humans_as_ai=True),
            supplier=SupplierComponent("Chatbot", "ChatVendor", Tier.LIMITED,
                                       True, True, True, True),  # compliant limited
        ),
        RegisteredComponent(
            component_id="cmp-004", owner="dan@bank.eu",
            usage=SystemProfile(name="Social scoring", is_social_scoring_by_authority=True),
            supplier=SupplierComponent("Social scoring", "ShadyCo", Tier.UNACCEPTABLE,
                                       False, False, False, False),  # blocked
        ),
    ]


# ==========================================================================
# SMOKE TEST
# ==========================================================================
if __name__ == "__main__":
    today = date(2026, 6, 21)

    print("=" * 66)
    print("EASY - classify five uses")
    print("=" * 66)
    easy = easy_classify_five()
    for name, tier, deadline in easy:
        print(f"  {tier:13s} {deadline:11s}  {name}")
    # Adversarial-ish checks on the easy expectations.
    expected = {
        "a) Internal spam filter": "MINIMAL",
        "b) Conversational AI assistant": "LIMITED",
        "c) CV-screening for recruitment": "HIGH",
        "d) Generalised social scoring by an authority": "UNACCEPTABLE",
        "e) AI safety component of a medical device": "HIGH",
    }
    got = {name: tier for name, tier, _ in easy}
    assert got == expected, f"easy mismatch: {got}"
    # Annex III vs Annex I deadlines differ.
    deadlines = {name: dl for name, _, dl in easy}
    assert deadlines["c) CV-screening for recruitment"] == "2026-08-02"
    assert deadlines["e) AI safety component of a medical device"] == "2027-08-02"

    print()
    print("=" * 66)
    print("MEDIUM - briefs sorted by urgency (today = 2026-06-21)")
    print("=" * 66)
    briefs = medium_briefs(today)
    for b in briefs:
        dl = b["days_left"]
        dl_str = "n/a" if dl is None else f"{dl} days"
        print(f"  {b['tier']:12s} deadline={b['deadline']:11s} days_left={dl_str:9s} {b['name']}")
    # Annex III brief should be ~42 days out and positive; minimal -> None.
    annex_iii = next(b for b in briefs if "Annex III" in b["name"])
    assert annex_iii["days_left"] is not None and annex_iii["days_left"] > 0
    assert annex_iii["days_left"] == (date(2026, 8, 2) - today).days  # 42
    spam = next(b for b in briefs if "Spam" in b["name"])
    assert spam["days_left"] is None
    # None goes last.
    assert briefs[-1]["days_left"] is None

    print()
    print("=" * 66)
    print("HARD - third-party governance portfolio report")
    print("=" * 66)
    report = portfolio_report(_hard_sample_portfolio(), today)
    print(f"  generated_on : {report['generated_on']}")
    print(f"  components   : {report['n_components']}  (blocked: {report['n_blocked']})")
    print(f"  nearest deadline : {report['nearest_deadline']} "
          f"({report['days_to_nearest_deadline']} days)")
    print("  ranked by compliance_risk_score (desc):")
    for a in report["components"]:
        status = a["status"]
        print(f"    [{a['compliance_risk_score']:>2}] {a['tier']:12s} {status:10s} "
              f"{a['name']}  owner={a['owner']}  gaps={len(a['gaps'])}")

    # Highest-scoring deployable should be the high-risk use WITH gaps (Hiring filter).
    assert report["components"][0]["name"] == "Hiring filter"
    # Unacceptable use is BLOCKED and excluded from nearest-deadline math.
    blocked = [a for a in report["components"] if a["status"] == "BLOCKED"]
    assert len(blocked) == 1 and blocked[0]["name"] == "Social scoring"
    # Nearest deadline = 2026-08-02 (the Annex III uses), not the chatbot's same date — still correct.
    assert report["nearest_deadline"] == "2026-08-02"

    # --- Adversarial probe: a high-risk use with FAILING due diligence must
    #     rank first and must not crash the report. ---
    probe = _hard_sample_portfolio() + [
        RegisteredComponent(
            component_id="cmp-bad", owner="eve@bank.eu",
            usage=SystemProfile(name="Worst high-risk", is_annex_iii_use=True),
            supplier=SupplierComponent("Worst high-risk", "NoDocsCo", Tier.HIGH,
                                       True, False, False, False),  # max gaps
        )
    ]
    probe_report = portfolio_report(probe, today)
    top = probe_report["components"][0]
    # Invariant (not a specific name — two failing high-risk components can tie
    # on score): the top-ranked item is a DEPLOYABLE high-risk use with open
    # gaps, i.e. the worst thing you actually have to remediate.
    assert top["status"] == "deployable"
    assert top["tier"] == "HIGH"
    assert len(top["gaps"]) > 0
    print()
    print(f"  adversarial probe: top component = "
          f"{top['name']} "
          f"(score {top['compliance_risk_score']}, gaps {len(top['gaps'])}) -> OK")

    print()
    print("All smoke-test assertions passed.")

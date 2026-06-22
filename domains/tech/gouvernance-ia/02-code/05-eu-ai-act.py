"""
Day 5 — EU AI Act tier classifier (rule-based) + supplier due-diligence.

What this script demonstrates
-----------------------------
The EU AI Act (Regulation (EU) 2024/1689) is a *risk-based* law: it does not
regulate "AI" uniformly, it sorts each AI **use** into one of four tiers, then
attaches obligations and a compliance deadline to that tier. On top of that, a
separate regime applies to general-purpose AI models (GPAI).

This module reimplements, in pure stdlib, the governance mechanism of the day:
a small rule engine that turns a structured questionnaire about a system into
(tier -> obligations -> applicable deadline), plus a minimal supplier
due-diligence check for AI components you *buy* rather than build.

Real tools this mimics
----------------------
A production setup would encode these rules in a policy engine such as Open
Policy Agent / Rego (see Day 14). Here the "policy" is a list of ordered Python
rules evaluated top-to-bottom — the same mental model, no dependency.

All dates and tier definitions are taken from REFERENCES.md (Reglement (UE)
2024/1689, Art. 5 / Annexe III / Annexe I / Art. 113 calendar) — they are NOT
invented. Tier classification here is an illustrative triage aid, not legal
advice.

# requires: stdlib only

Run:
    python 05-eu-ai-act.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# The four risk tiers. WHY an Enum: a system lands in EXACTLY one tier, and we
# want an ordered notion of severity (UNACCEPTABLE highest) to break ties and
# to sort outputs from most to least constrained.
# ---------------------------------------------------------------------------
class Tier(Enum):
    UNACCEPTABLE = 4  # Art. 5 — prohibited, cannot be deployed at all
    HIGH = 3          # Annex I (product safety) or Annex III (sensitive use)
    LIMITED = 2       # Art. 50 — transparency duty (e.g. chatbots, deepfakes)
    MINIMAL = 1       # everything else — no specific obligation


# ---------------------------------------------------------------------------
# Application calendar (Art. 113). Source of truth: REFERENCES.md.
# WHY a constant map: the single most common mistake is confusing "entered into
# force" (1 Aug 2024) with "becomes applicable". Obligations bite in waves; the
# deadline that concerns you depends on your tier AND your annex.
# ---------------------------------------------------------------------------
ENTRY_INTO_FORCE = "2024-08-01"

DEADLINES = {
    "prohibitions": "2025-02-02",   # Art. 5 bans + AI literacy
    "gpai": "2025-08-02",           # general-purpose AI obligations
    "high_risk_annex_iii": "2026-08-02",  # general application + Annex III
    "high_risk_annex_i": "2027-08-02",    # Annex I (product safety component)
    "limited": "2026-08-02",        # transparency duties (general application)
    "minimal": "n/a",               # no imposed obligation
}


@dataclass
class SystemProfile:
    """
    Structured answers about ONE AI system/use.

    The questionnaire is intentionally minimal: each boolean maps to a legal
    trigger. A real assessment has many more questions, but these capture the
    decisive branch points for triage.
    """
    name: str
    # --- Unacceptable triggers (Art. 5) ---
    is_social_scoring_by_authority: bool = False
    uses_subliminal_manipulation: bool = False
    does_untargeted_face_scraping: bool = False
    # --- High-risk triggers ---
    is_annex_iii_use: bool = False   # employment, credit, education, justice...
    is_annex_i_safety_component: bool = False  # safety part of a regulated product
    # --- Limited-risk trigger (Art. 50) ---
    interacts_with_humans_as_ai: bool = False  # chatbot / generated content
    # --- Cross-cutting context (does not change tier, informs obligations) ---
    builds_on_third_party_gpai: bool = False
    agent_takes_actions: bool = False  # acts (vs merely suggests) -> raises stakes


@dataclass
class Classification:
    name: str
    tier: Tier
    reason: str
    obligations: list[str] = field(default_factory=list)
    deadline: str = "n/a"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The rule engine. WHY ordered top-down: tiers are mutually exclusive and the
# law is a priority ladder — a prohibited use is prohibited even if it would
# also be "high risk". So we test from most severe to least severe and STOP at
# the first match. This is exactly how a Rego policy with deny-first ordering
# would behave.
# ---------------------------------------------------------------------------
def classify(profile: SystemProfile) -> Classification:
    p = profile

    # --- Rule 1: Unacceptable (Art. 5) — highest priority, no mitigation ---
    if p.is_social_scoring_by_authority or p.uses_subliminal_manipulation or p.does_untargeted_face_scraping:
        triggers = []
        if p.is_social_scoring_by_authority:
            triggers.append("social scoring by authorities")
        if p.uses_subliminal_manipulation:
            triggers.append("subliminal manipulation")
        if p.does_untargeted_face_scraping:
            triggers.append("untargeted facial-image scraping")
        return Classification(
            name=p.name,
            tier=Tier.UNACCEPTABLE,
            reason="Art. 5 prohibited practice: " + ", ".join(triggers),
            obligations=["PROHIBITED - do not deploy. No path to compliance."],
            deadline=DEADLINES["prohibitions"],
        )

    # --- Rule 2: High risk (Annex I or Annex III) ---
    if p.is_annex_i_safety_component or p.is_annex_iii_use:
        # WHY two deadlines: Annex III bites 2026-08-02, Annex I one year later.
        if p.is_annex_iii_use:
            annex = "Annex III (listed sensitive use)"
            deadline = DEADLINES["high_risk_annex_iii"]
        else:
            annex = "Annex I (safety component of a regulated product)"
            deadline = DEADLINES["high_risk_annex_i"]
        obligations = [
            "Risk management system (Art. 9)",
            "Data & data governance / quality (Art. 10)",
            "Technical documentation (Art. 11)",
            "Record-keeping / logging (Art. 12)",
            "Transparency & instructions for use (Art. 13)",
            "Human oversight (Art. 14)",
            "Accuracy, robustness, cybersecurity (Art. 15)",
        ]
        c = Classification(
            name=p.name,
            tier=Tier.HIGH,
            reason=f"High risk via {annex}",
            obligations=obligations,
            deadline=deadline,
        )
        if p.agent_takes_actions:
            # An agent that ACTS (not just suggests) makes human oversight
            # (Art. 14) the decisive control — flag it explicitly.
            c.notes.append("Agent takes actions: human oversight (Art. 14) is the critical control.")
        return c

    # --- Rule 3: Limited risk (Art. 50 transparency) ---
    if p.interacts_with_humans_as_ai:
        return Classification(
            name=p.name,
            tier=Tier.LIMITED,
            reason="Interacts with humans / generates content (Art. 50)",
            obligations=["Transparency: inform the person they interact with an AI / see AI-generated content (Art. 50)"],
            deadline=DEADLINES["limited"],
        )

    # --- Rule 4: Minimal — the default, no imposed obligation ---
    return Classification(
        name=p.name,
        tier=Tier.MINIMAL,
        reason="No prohibited / high-risk / limited-risk trigger matched",
        obligations=["None imposed (voluntary codes of conduct only)"],
        deadline=DEADLINES["minimal"],
    )


def gpai_obligations(profile: SystemProfile) -> list[str]:
    """
    GPAI is a PARALLEL regime, not one of the 4 tiers. If you build on a
    third-party general-purpose model, you depend on your supplier meeting
    these — and you still owe your own tier's obligations on top.
    """
    if not profile.builds_on_third_party_gpai:
        return []
    return [
        "Supplier GPAI duties (model side): technical documentation, "
        "info to downstream providers, copyright policy, training-data summary "
        f"(applicable since {DEADLINES['gpai']}).",
        "You (downstream) still owe the obligations of YOUR use's tier.",
    ]


# ---------------------------------------------------------------------------
# Supplier due diligence — the third-party governance angle of the day.
# For any AI component you BUY, compliance crosses the supply chain.
# ---------------------------------------------------------------------------
@dataclass
class SupplierComponent:
    name: str
    supplier: str
    your_tier: Tier               # the tier of YOUR use, not the vendor's claim
    is_or_uses_gpai: bool
    vendor_provides_gpai_docs: bool       # tech doc, training summary, copyright
    vendor_provides_usage_docs: bool      # instructions, known limits, logs
    responsibilities_contractualised: bool


def due_diligence(c: SupplierComponent) -> tuple[bool, list[str]]:
    """
    Return (passes, gaps). A gap = a missing piece that blocks proving
    compliance. WHY return gaps not just a bool: governance is about producing
    the evidence; an opaque "fail" is useless to a board.
    """
    gaps: list[str] = []
    # Q1: classified by OUR use?
    if c.your_tier == Tier.UNACCEPTABLE:
        gaps.append("Your use is a PROHIBITED practice — component cannot be deployed.")
    # Q2/Q3: GPAI documentation transmissible?
    if c.is_or_uses_gpai and not c.vendor_provides_gpai_docs:
        gaps.append("Component is/uses GPAI but vendor does not provide GPAI documentation.")
    # Q3: downstream usage documentation?
    if not c.vendor_provides_usage_docs:
        gaps.append("No usage docs (instructions/limits/logs) to support your downstream duties.")
    # Q4: responsibilities split contractualised?
    if not c.responsibilities_contractualised:
        gaps.append("Provider/deployer responsibility split not contractualised.")
    # High-risk use demands more: the vendor docs are mandatory inputs.
    if c.your_tier == Tier.HIGH and not c.vendor_provides_usage_docs:
        gaps.append("High-risk use without vendor usage docs: cannot meet Art. 11/13.")
    return (len(gaps) == 0, gaps)


def _print_classification(c: Classification) -> None:
    print(f"  System   : {c.name}")
    print(f"  Tier     : {c.tier.name}")
    print(f"  Reason   : {c.reason}")
    print(f"  Deadline : {c.deadline}")
    print("  Obligations:")
    for o in c.obligations:
        print(f"    - {o}")
    for n in c.notes:
        print(f"  Note: {n}")


if __name__ == "__main__":
    print("=" * 70)
    print("EU AI Act tier classifier  (Regulation (EU) 2024/1689)")
    print(f"Entered into force: {ENTRY_INTO_FORCE}  |  Obligations apply in waves (Art. 113)")
    print("=" * 70)

    # Three systems from the theory's concrete case (bank).
    systems = [
        SystemProfile(
            name="(C) Social scoring of citizens by a public authority",
            is_social_scoring_by_authority=True,
        ),
        SystemProfile(
            name="(A) Credit-scoring agent deciding loan eligibility",
            is_annex_iii_use=True,
            agent_takes_actions=True,
            builds_on_third_party_gpai=True,
        ),
        SystemProfile(
            name="(B) Billing-support chatbot",
            interacts_with_humans_as_ai=True,
            builds_on_third_party_gpai=True,
        ),
        SystemProfile(
            name="(D) Spam filter on internal email",
        ),
    ]

    # Sort most-constrained first so a reviewer sees the dangerous ones on top.
    results = sorted((classify(s) for s in systems), key=lambda c: c.tier.value, reverse=True)
    for c in results:
        print()
        _print_classification(c)

    print()
    print("-" * 70)
    print("GPAI parallel regime (applies on top of the tier):")
    for s in systems:
        gp = gpai_obligations(s)
        if gp:
            print(f"\n  {s.name}")
            for line in gp:
                print(f"    - {line}")

    print()
    print("-" * 70)
    print("Supplier due diligence (third-party governance):")

    components = [
        SupplierComponent(
            name="Credit-scoring engine (bought)",
            supplier="VendorCorp",
            your_tier=Tier.HIGH,
            is_or_uses_gpai=True,
            vendor_provides_gpai_docs=True,
            vendor_provides_usage_docs=True,
            responsibilities_contractualised=True,
        ),
        SupplierComponent(
            name="Chatbot SaaS (bought)",
            supplier="ChatVendor",
            your_tier=Tier.LIMITED,
            is_or_uses_gpai=True,
            vendor_provides_gpai_docs=False,   # missing -> a gap
            vendor_provides_usage_docs=True,
            responsibilities_contractualised=False,  # missing -> a gap
        ),
    ]
    for comp in components:
        ok, gaps = due_diligence(comp)
        verdict = "PASS" if ok else "FAIL"
        print(f"\n  {comp.name}  (supplier: {comp.supplier}, your tier: {comp.your_tier.name})  -> {verdict}")
        for g in gaps:
            print(f"    ! {g}")

    print()
    print("Done. Tiers, obligations and deadlines reflect Reg. (EU) 2024/1689.")
    print("This is a triage aid, not legal advice.")

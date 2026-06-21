"""Mini data-flow / DPIA assessor for an AI agent (RGPD / GDPR, EU 2016/679).

What this script demonstrates
-----------------------------
Day 6 governance mechanism: given the declared data fields an agent touches,
its purpose, its legal basis and a few risk signals, automatically:

  1. detect whether the agent processes PERSONAL DATA (and which fields are
     "special category" / Art. 9 sensitive data),
  2. validate the LEGAL BASIS (Art. 6) is declared and recognised, and that a
     legitimate-interest balancing test is present when that basis is chosen,
  3. enforce DATA MINIMISATION (Art. 5(1)(c)): flag declared fields that are
     not justified by the stated purpose,
  4. require a bounded RETENTION (Art. 5(1)(e)) for any persistent agent memory,
  5. decide whether a DPIA / AIPD is REQUIRED (Art. 35 + CNIL "two-criteria" rule).

This is a stdlib re-implementation of what a real data-governance / privacy
review (e.g. OneTrust, a CNIL AIPD template, or an internal RoPA) would encode.
No external tool, no network, no model call.

Sources (see REFERENCES.md):
  - GDPR / RGPD, Regulation (EU) 2016/679 (Art. 4, 5, 6, 9, 35).
  - EDPB Opinion 28/2024 (legitimate interest balancing test for AI models).
  - CNIL, IA & RGPD (DPIA criteria: a DPIA is expected once two criteria meet).
"""

# requires: stdlib only

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --- Reference data ----------------------------------------------------------
# WHY: encoding the legal taxonomy as data (not scattered if/else) keeps the
# rule engine auditable — a reviewer can read exactly which fields count as
# personal / sensitive, just like a real data classification policy would.

# Art. 6(1) legal bases. We key on short codes used in agent declarations.
LEGAL_BASES = {
    "consent": "Art. 6(1)(a) - consent",
    "contract": "Art. 6(1)(b) - performance of a contract",
    "legal_obligation": "Art. 6(1)(c) - legal obligation",
    "vital_interests": "Art. 6(1)(d) - vital interests",
    "public_task": "Art. 6(1)(e) - public interest task",
    "legitimate_interest": "Art. 6(1)(f) - legitimate interest",
}

# Fields that, by themselves, relate to an identified/identifiable person
# (Art. 4(1)). Non-exhaustive — a real classifier would be richer.
PERSONAL_FIELDS = {
    "name", "email", "phone", "address", "ip_address",
    "customer_id", "ticket_history", "location", "device_id",
}

# Art. 9 "special category" data — triggers a much higher bar (and a DPIA).
SENSITIVE_FIELDS = {
    "health", "biometric", "genetic", "religion",
    "political_opinion", "sexual_orientation", "ethnicity",
    "trade_union", "criminal_record",  # criminal data: Art. 10, treated as high-risk here
}


class DpiaVerdict(Enum):
    REQUIRED = "DPIA REQUIRED"
    RECOMMENDED = "DPIA RECOMMENDED"
    NOT_REQUIRED = "DPIA not required on these signals"


@dataclass
class AgentDataProfile:
    """Declaration an agent owner files for a given processing purpose.

    Mirrors the cross-day mini agent model (id/owner/...) but focused on the
    data-governance facet: what an agent touches and why.
    """

    agent_id: str
    owner: str
    purpose: str                       # Art. 5(1)(b): must be specific & legitimate
    declared_fields: list[str]         # every data field the agent reads/stores
    fields_needed_for_purpose: list[str]  # subset truly required (minimisation)
    legal_basis: str | None = None     # one of LEGAL_BASES keys
    has_li_balancing_test: bool = False  # required iff legal_basis == legitimate_interest
    persistent_memory: bool = False    # does it keep state across sessions?
    retention_days: int | None = None  # bounded retention for that memory
    # Risk signals feeding the DPIA decision (CNIL-style criteria):
    automated_decision: bool = False   # solely automated decision w/ significant effect (Art. 22)
    large_scale: bool = False
    profiling: bool = False
    vulnerable_subjects: bool = False  # children, patients, employees...
    innovative_use: bool = True        # agentic AI is, by default, an emerging tech


@dataclass
class AssessmentResult:
    findings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)  # licéité issues -> must fix
    dpia: DpiaVerdict = DpiaVerdict.NOT_REQUIRED
    dpia_reasons: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        # WHY: a missing legal basis or a minimisation breach makes the
        # processing UNLAWFUL — we surface that as a hard blocker, not a hint.
        return not self.blockers


# --- The assessor ------------------------------------------------------------

def assess(profile: AgentDataProfile) -> AssessmentResult:
    r = AssessmentResult()

    declared = set(profile.declared_fields)
    personal = declared & PERSONAL_FIELDS
    sensitive = declared & SENSITIVE_FIELDS
    processes_personal = bool(personal or sensitive)

    if not processes_personal:
        r.findings.append("No personal data detected -> GDPR core obligations not triggered.")
        # Even so, we keep going: minimisation is good hygiene regardless.
    else:
        r.findings.append(
            f"Personal data detected: {sorted(personal | sensitive)} "
            f"(sensitive/Art.9: {sorted(sensitive) or 'none'})."
        )

    # 1) Legal basis (Art. 6) — only required when personal data is processed.
    if processes_personal:
        if profile.legal_basis is None:
            r.blockers.append("No legal basis declared (Art. 6) -> processing is UNLAWFUL.")
        elif profile.legal_basis not in LEGAL_BASES:
            r.blockers.append(f"Unknown legal basis '{profile.legal_basis}' (not in Art. 6).")
        else:
            r.findings.append(f"Legal basis: {LEGAL_BASES[profile.legal_basis]}.")
            # Legitimate interest needs a documented balancing test (EDPB 28/2024).
            if profile.legal_basis == "legitimate_interest" and not profile.has_li_balancing_test:
                r.blockers.append(
                    "Legitimate interest declared without a balancing test "
                    "(required, EDPB Opinion 28/2024)."
                )

    # 2) Data minimisation (Art. 5(1)(c)) — flag fields not needed for purpose.
    # Only personal/sensitive excess matters for a minimisation finding.
    needed = set(profile.fields_needed_for_purpose)
    excess = (personal | sensitive) - needed
    if excess:
        r.blockers.append(
            f"Minimisation breach (Art. 5(1)(c)): collects personal fields not "
            f"justified by purpose: {sorted(excess)}."
        )
    elif processes_personal:
        r.findings.append("Minimisation OK: all personal fields map to the purpose.")

    # 3) Retention (Art. 5(1)(e)) — persistent memory MUST be time-bounded.
    if profile.persistent_memory:
        if profile.retention_days is None or profile.retention_days <= 0:
            r.blockers.append(
                "Persistent agent memory without a bounded retention "
                "(Art. 5(1)(e) storage limitation)."
            )
        else:
            r.findings.append(f"Retention bounded at {profile.retention_days} days.")

    # 4) DPIA decision (Art. 35 + CNIL two-criteria heuristic).
    r.dpia, r.dpia_reasons = _decide_dpia(profile, sensitive)

    return r


def _decide_dpia(profile: AgentDataProfile, sensitive: set[str]) -> tuple[DpiaVerdict, list[str]]:
    # Hard triggers (Art. 35(3)): a single one already forces a DPIA.
    hard = []
    if sensitive and profile.large_scale:
        hard.append("Large-scale processing of special-category data (Art. 35(3)(b)).")
    if profile.automated_decision and profile.profiling:
        hard.append(
            "Systematic automated evaluation with legal/similar effect (Art. 35(3)(a))."
        )
    if hard:
        return DpiaVerdict.REQUIRED, hard

    # CNIL heuristic: count criteria; >= 2 -> DPIA expected.
    criteria = {
        "sensitive data": bool(sensitive),
        "automated decision (Art. 22)": profile.automated_decision,
        "large scale": profile.large_scale,
        "profiling": profile.profiling,
        "vulnerable subjects": profile.vulnerable_subjects,
        "innovative technological use": profile.innovative_use,
    }
    met = [name for name, on in criteria.items() if on]
    if len(met) >= 2:
        return DpiaVerdict.REQUIRED, [f"CNIL criteria met ({len(met)}): {met}."]
    if len(met) == 1:
        return DpiaVerdict.RECOMMENDED, [f"One CNIL criterion met: {met}. Borderline."]
    return DpiaVerdict.NOT_REQUIRED, ["No high-risk criterion met on these signals."]


# --- Demo --------------------------------------------------------------------

def _print_report(profile: AgentDataProfile, r: AssessmentResult) -> None:
    print(f"=== Agent: {profile.agent_id}  (owner: {profile.owner}) ===")
    print(f"Purpose: {profile.purpose}")
    for f_ in r.findings:
        print(f"  [info]    {f_}")
    for b in r.blockers:
        print(f"  [BLOCKER] {b}")
    print(f"  -> Lawful on these signals? {'YES' if r.compliant else 'NO'}")
    print(f"  -> {r.dpia.value}")
    for reason in r.dpia_reasons:
        print(f"     - {reason}")
    print()


if __name__ == "__main__":
    # Case A: the "innocent" support agent from the theory module.
    # It hoards a sensitive field (health) it does not need, has memory but no
    # retention, and relies on legitimate interest without a balancing test.
    support_agent = AgentDataProfile(
        agent_id="support-bot-01",
        owner="alice@corp",
        purpose="Personalise customer support replies",
        declared_fields=["name", "email", "ticket_history", "health"],
        fields_needed_for_purpose=["name", "ticket_history"],
        legal_basis="legitimate_interest",
        has_li_balancing_test=False,
        persistent_memory=True,
        retention_days=None,
        automated_decision=False,
        large_scale=True,
        profiling=False,
        vulnerable_subjects=False,
        innovative_use=True,
    )

    # Case B: a well-governed order agent. Contract basis, minimal fields,
    # bounded retention, no sensitive data, no automated decision.
    order_agent = AgentDataProfile(
        agent_id="order-bot-02",
        owner="bob@corp",
        purpose="Process customer orders",
        declared_fields=["name", "address", "customer_id"],
        fields_needed_for_purpose=["name", "address", "customer_id"],
        legal_basis="contract",
        persistent_memory=True,
        retention_days=90,
        innovative_use=False,
    )

    # Case C: an internal analytics agent touching NO personal data.
    analytics_agent = AgentDataProfile(
        agent_id="metrics-bot-03",
        owner="carol@corp",
        purpose="Aggregate anonymous server metrics",
        declared_fields=["cpu_load", "request_count"],
        fields_needed_for_purpose=["cpu_load", "request_count"],
        legal_basis=None,
        innovative_use=False,
    )

    for p in (support_agent, order_agent, analytics_agent):
        _print_report(p, assess(p))

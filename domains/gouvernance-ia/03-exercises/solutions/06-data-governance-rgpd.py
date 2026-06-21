"""Solutions — Day 6: Data governance & RGPD for agentic AI.

Covers the three exercise levels in one file (stdlib only):
  - EASY   : personal / sensitive data detector (Art. 4(1), Art. 9).
  - MEDIUM : legal-basis + minimisation validator (Art. 6, Art. 5(1)(c), EDPB 28/2024).
  - HARD   : DPIA decider (Art. 35 + CNIL "two-criteria") + propagable erasure (Art. 17).

Run directly for a smoke test of all three:  python 06-data-governance-rgpd.py

Sources: GDPR / RGPD (EU 2016/679); EDPB Opinion 28/2024; CNIL, IA & RGPD.
"""

# requires: stdlib only

from __future__ import annotations

from dataclasses import dataclass


PERSONAL_FIELDS = {
    "name", "email", "phone", "address", "ip_address",
    "customer_id", "ticket_history", "location", "device_id",
}
SENSITIVE_FIELDS = {
    "health", "biometric", "genetic", "religion",
    "political_opinion", "sexual_orientation", "ethnicity",
    "trade_union", "criminal_record",
}
LEGAL_BASES = {
    "consent", "contract", "legal_obligation",
    "vital_interests", "public_task", "legitimate_interest",
}


# === EASY ===
# Detect personal / sensitive data among the fields an agent declares.

def classify(declared_fields: list[str]) -> dict:
    declared = set(declared_fields)
    personal = sorted(declared & PERSONAL_FIELDS)
    sensitive = sorted(declared & SENSITIVE_FIELDS)
    return {
        "personal": personal,
        "sensitive": sensitive,
        # Sensitive data is also personal data -> both feed the trigger.
        "processes_personal": bool(personal or sensitive),
    }


# === MEDIUM ===
# Validate legal basis (Art. 6) + data minimisation (Art. 5(1)(c)).

def validate(
    declared_fields: list[str],
    fields_needed: list[str],
    legal_basis: str | None,
    has_li_balancing_test: bool = False,
) -> dict:
    declared = set(declared_fields)
    personal_present = declared & PERSONAL_FIELDS
    sensitive_present = declared & SENSITIVE_FIELDS
    processes_personal = bool(personal_present or sensitive_present)

    blockers: list[str] = []

    if processes_personal:
        # Legal basis (Art. 6): mandatory and must be one of the six.
        if legal_basis is None:
            blockers.append("No legal basis declared (Art. 6) -> processing is UNLAWFUL.")
        elif legal_basis not in LEGAL_BASES:
            blockers.append(f"Unknown legal basis '{legal_basis}' (not in Art. 6).")
        elif legal_basis == "legitimate_interest" and not has_li_balancing_test:
            blockers.append(
                "Legitimate interest without a balancing test (EDPB Opinion 28/2024)."
            )

        # Minimisation (Art. 5(1)(c)): no personal/sensitive field beyond what
        # the purpose needs.
        needed = set(fields_needed)
        excess = sorted((personal_present | sensitive_present) - needed)
        if excess:
            blockers.append(f"Minimisation breach (Art. 5(1)(c)): unneeded fields {excess}.")

    return {"blockers": blockers, "compliant": not blockers}


# === HARD ===
# DPIA decider (Art. 35 + CNIL heuristic) and propagable erasure (Art. 17).

@dataclass
class DpiaProfile:
    sensitive: bool = False
    automated_decision: bool = False
    large_scale: bool = False
    profiling: bool = False
    vulnerable_subjects: bool = False
    innovative_use: bool = True  # agentic AI defaults to "emerging tech".


def decide_dpia(p: DpiaProfile) -> tuple[str, list[str]]:
    # Hard triggers (Art. 35(3)): a single one forces a DPIA.
    if p.sensitive and p.large_scale:
        return "REQUIRED", ["Large-scale special-category data (Art. 35(3)(b))."]
    if p.automated_decision and p.profiling:
        return "REQUIRED", ["Systematic automated evaluation w/ effect (Art. 35(3)(a))."]

    # CNIL heuristic: count criteria; >=2 -> REQUIRED, ==1 -> RECOMMENDED.
    criteria = {
        "sensitive": p.sensitive,
        "automated_decision": p.automated_decision,
        "large_scale": p.large_scale,
        "profiling": p.profiling,
        "vulnerable_subjects": p.vulnerable_subjects,
        "innovative_use": p.innovative_use,
    }
    met = [name for name, on in criteria.items() if on]
    if len(met) >= 2:
        return "REQUIRED", [f"CNIL criteria met ({len(met)}): {met}."]
    if len(met) == 1:
        return "RECOMMENDED", [f"One CNIL criterion met: {met}."]
    return "NOT_REQUIRED", ["No high-risk criterion met."]


class AgentStores:
    """Three places an agent may hold personal data about a subject.

    WHY: Art. 17 erasure must PROPAGATE. Deleting from main_db but leaving the
    subject in agent_memory or audit_logs is not erasure.
    """

    def __init__(self) -> None:
        self.main_db: list[dict] = []
        self.agent_memory: list[dict] = []
        self.audit_logs: list[dict] = []

    def _stores(self) -> dict[str, list[dict]]:
        return {
            "main_db": self.main_db,
            "agent_memory": self.agent_memory,
            "audit_logs": self.audit_logs,
        }


def forget(stores: AgentStores, subject: str) -> dict:
    """Remove every record of `subject` from all stores. Return per-store counts."""
    counts = {}
    for name, store in stores._stores().items():
        before = len(store)
        store[:] = [rec for rec in store if rec.get("subject") != subject]
        counts[name] = before - len(store)
    return counts


def assert_erased(stores: AgentStores, subject: str) -> bool:
    """True iff no trace of `subject` remains ANYWHERE. Adversarial check."""
    return all(
        rec.get("subject") != subject
        for store in stores._stores().values()
        for rec in store
    )


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # EASY ----------------------------------------------------------------
    support = classify(["name", "email", "ticket_history", "health"])
    metrics = classify(["cpu_load", "request_count"])
    assert support["processes_personal"] is True
    assert "health" in support["sensitive"]
    assert metrics["processes_personal"] is False
    print("EASY   ok:", support, "|", metrics)

    # MEDIUM --------------------------------------------------------------
    bad = validate(
        declared_fields=["name", "email", "health"],
        fields_needed=["name"],
        legal_basis="legitimate_interest",
        has_li_balancing_test=False,
    )
    good = validate(
        declared_fields=["name", "address", "customer_id"],
        fields_needed=["name", "address", "customer_id"],
        legal_basis="contract",
    )
    no_basis = validate(["name"], ["name"], legal_basis=None)
    assert bad["compliant"] is False and len(bad["blockers"]) >= 2  # LI + minimisation
    assert good["compliant"] is True and good["blockers"] == []
    assert no_basis["compliant"] is False
    print("MEDIUM ok:", bad["blockers"])
    print("           ", good)

    # HARD ----------------------------------------------------------------
    v1, r1 = decide_dpia(DpiaProfile(large_scale=True, innovative_use=True))   # 2 criteria
    v2, r2 = decide_dpia(DpiaProfile(sensitive=True, large_scale=True))        # hard trigger
    v3, r3 = decide_dpia(DpiaProfile(innovative_use=False))                    # 0 criteria
    assert v1 == "REQUIRED", v1
    assert v2 == "REQUIRED", v2
    assert v3 == "NOT_REQUIRED", v3
    print("HARD   DPIA:", v1, r1, "|", v2, "|", v3)

    # Propagable erasure + adversarial probe.
    s = AgentStores()
    s.main_db.append({"subject": "user42", "data": "name"})
    s.agent_memory.append({"subject": "user42", "data": "preferences"})
    s.audit_logs.append({"subject": "user42", "data": "action log"})
    s.main_db.append({"subject": "user99", "data": "name"})

    counts = forget(s, "user42")
    assert counts == {"main_db": 1, "agent_memory": 1, "audit_logs": 1}, counts
    assert assert_erased(s, "user42") is True
    assert forget(s, "ghost") == {"main_db": 0, "agent_memory": 0, "audit_logs": 0}
    print("HARD   erasure ok:", counts)

    # Adversarial: a residual trace left in ONE store must be caught.
    s.audit_logs.append({"subject": "user42", "data": "leftover"})
    assert assert_erased(s, "user42") is False
    print("HARD   adversarial probe ok: residual trace detected")

    print("\nAll smoke tests passed.")

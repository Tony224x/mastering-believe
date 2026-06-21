"""Solutions J7 — Normes & systemes de management (AIMS).

Un seul fichier couvrant les 3 niveaux (easy / medium / hard). stdlib pure.
Theme commun : referentiels de gouvernance IA, AIMS / PDCA, et CROSSWALK
(un controle interne -> N referentiels) avec calcul de couverture et de trous.

Faits/dates : EU AI Act = Reglement (UE) 2024/1689 (obligatoire) ; RGPD = (UE)
2016/679 (obligatoire) ; ISO/IEC 42001:2023, NIST AI RMF 1.0, OECD AI Principles
(maj 2024), IMDA Agentic (22 janv. 2026) = volontaires.

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =========================================================================
# === EASY ===
# Objectif : distinguer referentiel obligatoire (loi) vs volontaire (norme),
# et restituer la boucle PDCA d'un AIMS.
# =========================================================================

# WHY un set d'obligatoires : EU AI Act et RGPD sont des LOIS UE ; les autres
# referentiels sont des normes/cadres VOLONTAIRES. Le set rend is_mandatory O(1).
_MANDATORY_FRAMEWORKS = {"EU AI Act", "RGPD"}

EASY_FRAMEWORKS: list[tuple[str, bool]] = [
    ("EU AI Act", True),
    ("RGPD", True),
    ("ISO/IEC 42001", False),
    ("NIST AI RMF", False),
    ("OECD AI Principles", False),
]


def is_mandatory(name: str) -> bool:
    """True si le referentiel est une obligation legale (loi UE)."""
    return name in _MANDATORY_FRAMEWORKS


def pdca_steps() -> list[str]:
    """Les 4 etapes de la boucle d'amelioration continue d'un AIMS."""
    return ["Plan", "Do", "Check", "Act"]


def easy_report() -> str:
    lines = ["[EASY] Referentiels — obligatoire vs volontaire :"]
    for name, _ in EASY_FRAMEWORKS:
        tag = "OBLIGATOIRE" if is_mandatory(name) else "volontaire"
        lines.append(f"  - {name:<20} {tag}")
    lines.append(f"  PDCA : {' -> '.join(pdca_steps())}")
    return "\n".join(lines)


# =========================================================================
# === MEDIUM ===
# Objectif : construire un crosswalk (controle -> exigences), calculer la
# couverture par referentiel et lister les trous.
# =========================================================================


@dataclass(frozen=True)
class Requirement:
    framework: str
    ref: str
    label: str
    mandatory: bool


@dataclass
class Control:
    control_id: str
    label: str
    covers: set[str] = field(default_factory=set)


def req_key(r: Requirement) -> str:
    return f"{r.framework}::{r.ref}"


# Catalogue (>=6 exigences sur 3 referentiels). EU AI Act obligatoire ; le reste volontaire.
MED_REQUIREMENTS: list[Requirement] = [
    Requirement("EU AI Act", "Art. 9", "Systeme de gestion des risques", True),
    Requirement("EU AI Act", "Art. 12", "Tenue de registres / logging", True),
    Requirement("EU AI Act", "Art. 14", "Supervision humaine", True),
    Requirement("NIST AI RMF", "Govern", "Roles & responsabilites", False),
    Requirement("NIST AI RMF", "Measure", "Mesure & tracabilite", False),
    Requirement("ISO/IEC 42001", "8.2", "Suivi operationnel & journalisation", False),
]


def med_controls() -> list[Control]:
    """2 controles ; on laisse volontairement des exigences non couvertes (trous)."""
    audit = Control(
        "CTRL-AUDIT",
        "Audit trail des actions d'agent",
        covers={"EU AI Act::Art. 12", "NIST AI RMF::Measure", "ISO/IEC 42001::8.2"},
    )
    owner = Control(
        "CTRL-OWNER",
        "Owner humain nomme",
        covers={"EU AI Act::Art. 14", "NIST AI RMF::Govern"},
    )
    return [audit, owner]


def _covered_keys(controls: list[Control]) -> set[str]:
    out: set[str] = set()
    for c in controls:
        out |= c.covers
    return out


def coverage_by_framework(
    controls: list[Control], requirements: list[Requirement] = MED_REQUIREMENTS
) -> dict[str, tuple[int, int]]:
    """Par referentiel -> (nb couvertes, nb total)."""
    covered = _covered_keys(controls)
    acc: dict[str, list[int]] = {}
    for r in requirements:
        slot = acc.setdefault(r.framework, [0, 0])
        slot[1] += 1
        if req_key(r) in covered:
            slot[0] += 1
    return {fw: (c, t) for fw, (c, t) in acc.items()}


def gaps(
    controls: list[Control], requirements: list[Requirement] = MED_REQUIREMENTS
) -> list[Requirement]:
    """Exigences couvertes par aucun controle."""
    covered = _covered_keys(controls)
    return [r for r in requirements if req_key(r) not in covered]


def medium_report() -> str:
    controls = med_controls()
    lines = ["[MEDIUM] Crosswalk — couverture & trous :"]
    for fw, (cov, tot) in sorted(coverage_by_framework(controls).items()):
        pct = (cov / tot * 100) if tot else 0.0
        lines.append(f"  {fw:<16} {cov}/{tot} ({pct:.0f}%)")
    open_gaps = gaps(controls)
    lines.append(f"  Trous : {len(open_gaps)}")
    for r in sorted(open_gaps, key=req_key):  # tri stable = sortie deterministe
        tag = "OBLIGATOIRE" if r.mandatory else "volontaire"
        lines.append(f"    ! {r.framework} {r.ref} ({tag}) : {r.label}")
    return "\n".join(lines)


# =========================================================================
# === HARD ===
# Objectif : verdict de conformite (obligatoire vs volontaire) + prochaine
# etape Act du PDCA + probe adversariale (obligation injectee non couverte).
# =========================================================================

# Poids de priorite : une obligation legale pese plus qu'un alignement volontaire.
_PRIORITY = {True: 3, False: 1}


def verdict(controls: list[Control], requirements: list[Requirement]) -> str:
    open_gaps = gaps(controls, requirements)
    mandatory_gaps = [r for r in open_gaps if r.mandatory]
    if mandatory_gaps:
        return (
            f"NON-CONFORMITE LEGALE : {len(mandatory_gaps)} obligation(s) non couverte(s)."
        )
    return (
        "CONFORME AU LEGAL : aucune obligation non couverte ; "
        f"{len(open_gaps)} trou(s) volontaire(s) restant(s)."
    )


def next_action(controls: list[Control], requirements: list[Requirement]) -> dict:
    """Prochaine etape Act : trou prioritaire a couvrir (obligatoire d'abord)."""
    open_gaps = gaps(controls, requirements)
    if not open_gaps:
        return {"action": "none", "reason": "Couverture complete sur le catalogue."}
    # Tri : poids decroissant (obligatoire d'abord), puis cle stable -> deterministe.
    target = sorted(open_gaps, key=lambda r: (-_PRIORITY[r.mandatory], req_key(r)))[0]
    reason = (
        "Trou OBLIGATOIRE (loi) — priorite max."
        if target.mandatory
        else "Trou volontaire — amelioration continue."
    )
    return {
        "action": "implement_control",
        "target": req_key(target),
        "label": target.label,
        "mandatory": target.mandatory,
        "reason": reason,
    }


def hard_report() -> str:
    # On copie le catalogue medium pour pouvoir l'etendre sans muter l'original.
    requirements = list(MED_REQUIREMENTS)
    controls = med_controls()

    lines = ["[HARD] Verdict & next_action :"]
    lines.append(f"  Verdict : {verdict(controls, requirements)}")
    act = next_action(controls, requirements)
    lines.append(f"  Next Act : {act['action']} -> {act.get('target')} ({act['reason']})")

    # --- Probe adversariale : on injecte une obligation legale NON couverte.
    lines.append("  --- Probe : injection d'une obligation non couverte ---")
    requirements.append(
        Requirement("EU AI Act", "Art. 13", "Transparence & instructions d'usage", True)
    )
    lines.append(f"  Verdict : {verdict(controls, requirements)}")
    act2 = next_action(controls, requirements)
    lines.append(f"  Next Act : {act2['action']} -> {act2.get('target')} ({act2['reason']})")
    return "\n".join(lines)


# =========================================================================
# Smoke test
# =========================================================================

if __name__ == "__main__":
    print(easy_report())
    print()
    print(medium_report())
    print()
    print(hard_report())

    # --- Assertions de non-regression (le smoke test doit prouver le comportement) ---
    assert is_mandatory("EU AI Act") and is_mandatory("RGPD")
    assert not is_mandatory("ISO/IEC 42001")
    assert pdca_steps() == ["Plan", "Do", "Check", "Act"]

    ctrls = med_controls()
    cov = coverage_by_framework(ctrls)
    # EU AI Act : 2 des 3 exigences couvertes (Art. 9 reste un trou).
    assert cov["EU AI Act"] == (2, 3), cov["EU AI Act"]
    open_keys = {req_key(r) for r in gaps(ctrls)}
    assert "EU AI Act::Art. 9" in open_keys

    # Sans la probe : Art. 9 (obligatoire) non couvert -> non-conformite, next_action le cible.
    reqs = list(MED_REQUIREMENTS)
    assert verdict(ctrls, reqs).startswith("NON-CONFORMITE")
    assert next_action(ctrls, reqs)["mandatory"] is True

    print("\nOK: all smoke assertions passed.")

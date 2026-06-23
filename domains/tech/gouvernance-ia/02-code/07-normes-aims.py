"""Control crosswalk mapper — un controle interne -> N referentiels.

Demontre le mecanisme de gouvernance du jour 7 (Normes & AIMS) : comment un
*meme* controle interne satisfait plusieurs referentiels a la fois (EU AI Act,
NIST AI RMF, ISO/IEC 42001), et comment calculer la COUVERTURE de conformite
et reperer les TROUS (exigences non couvertes).

Idee pedagogique : on ne construit pas N systemes de conformite ; on construit
UN systeme de management (AIMS, logique ISO/IEC 42001) et on le mappe vers les N
referentiels via un *crosswalk*. Cela rejoint la boucle PDCA (Plan-Do-Check-Act) :
le rapport de couverture est l'etape "Check" qui declenche un "Act" (combler les
trous).

Outils reels evoques (ici re-implementes en miniature, stdlib pure) :
  - ISO/IEC 42001:2023 (AI Management System / AIMS)
  - NIST AI RMF 1.0 (fonctions Govern / Map / Measure / Manage)
  - EU AI Act (Reglement (UE) 2024/1689 — articles haut risque)

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# WHY an Enum: les fonctions du NIST AI RMF 1.0 sont un ensemble FERME et stable
# (Govern transversale, puis Map / Measure / Manage). Un Enum interdit les fautes
# de frappe silencieuses ("Manag") qui fausseraient un calcul de couverture.
class RmfFunction(str, Enum):
    GOVERN = "Govern"
    MAP = "Map"
    MEASURE = "Measure"
    MANAGE = "Manage"


@dataclass(frozen=True)
class Requirement:
    """Une exigence d'un referentiel (frozen: une exigence est immuable une fois posee)."""

    framework: str  # "EU AI Act" | "NIST AI RMF" | "ISO/IEC 42001"
    ref: str        # ex. "Art. 12", "Measure", "8.2"
    label: str      # description courte
    mandatory: bool  # True = obligation legale, False = referentiel volontaire


@dataclass
class Control:
    """Un controle interne implemente par l'organisation (un pilier de gouvernance)."""

    control_id: str
    label: str
    # WHY un set de ref-keys: un controle peut couvrir plusieurs exigences ; le set
    # evite les doublons et rend l'intersection avec les exigences triviale.
    covers: set[str] = field(default_factory=set)


def req_key(r: Requirement) -> str:
    """Cle stable d'une exigence (sert de pont entre Control.covers et le catalogue)."""
    return f"{r.framework}::{r.ref}"


# --- Catalogue d'exigences (sous-ensemble pedagogique, dates/refs verifiees) -------
# EU AI Act = obligatoire (Reglement (UE) 2024/1689) ; NIST/ISO = volontaires.
REQUIREMENTS: list[Requirement] = [
    # EU AI Act — systemes haut risque (obligatoire)
    Requirement("EU AI Act", "Art. 9", "Systeme de gestion des risques", mandatory=True),
    Requirement("EU AI Act", "Art. 12", "Tenue de registres / logging", mandatory=True),
    Requirement("EU AI Act", "Art. 14", "Supervision humaine", mandatory=True),
    # NIST AI RMF 1.0 — volontaire (les 4 fonctions coeur)
    Requirement("NIST AI RMF", RmfFunction.GOVERN.value, "Roles & responsabilites", mandatory=False),
    Requirement("NIST AI RMF", RmfFunction.MAP.value, "Cartographie du risque", mandatory=False),
    Requirement("NIST AI RMF", RmfFunction.MEASURE.value, "Mesure & tracabilite", mandatory=False),
    Requirement("NIST AI RMF", RmfFunction.MANAGE.value, "Traitement du risque", mandatory=False),
    # ISO/IEC 42001:2023 — volontaire (clauses AIMS, refs indicatives)
    Requirement("ISO/IEC 42001", "6.1", "AI risk assessment", mandatory=False),
    Requirement("ISO/IEC 42001", "8.2", "Suivi operationnel & journalisation", mandatory=False),
    Requirement("ISO/IEC 42001", "5.3", "Roles & responsabilites de l'AIMS", mandatory=False),
]

_REQ_BY_KEY = {req_key(r): r for r in REQUIREMENTS}


def build_demo_controls() -> list[Control]:
    """Trois controles internes = les piliers de gouvernance vus dans le parcours.

    WHY ces trois : chacun illustre qu'UN controle satisfait PLUSIEURS referentiels
    simultanement (le coeur de la lecon "crosswalk"). On laisse volontairement des
    exigences NON couvertes pour que le rapport montre des TROUS reels.
    """
    audit_trail = Control(
        "CTRL-AUDIT",
        "Audit trail inviolable des actions d'agent",
        covers={
            "EU AI Act::Art. 12",
            "NIST AI RMF::Measure",
            "ISO/IEC 42001::8.2",
        },
    )
    named_owner = Control(
        "CTRL-OWNER",
        "Owner humain nomme par agent",
        covers={
            "EU AI Act::Art. 14",
            "NIST AI RMF::Govern",
            "ISO/IEC 42001::5.3",
        },
    )
    risk_assess = Control(
        "CTRL-RISK",
        "Evaluation de risque par agent",
        covers={
            "EU AI Act::Art. 9",
            "NIST AI RMF::Map",
            "ISO/IEC 42001::6.1",
        },
    )
    return [audit_trail, named_owner, risk_assess]


def covered_keys(controls: list[Control]) -> set[str]:
    """Union des exigences couvertes par au moins un controle."""
    out: set[str] = set()
    for c in controls:
        out |= c.covers
    return out


def coverage_by_framework(controls: list[Control]) -> dict[str, tuple[int, int]]:
    """Pour chaque referentiel -> (nb exigences couvertes, nb total)."""
    covered = covered_keys(controls)
    stats: dict[str, list[int]] = {}
    for r in REQUIREMENTS:
        total_covered = stats.setdefault(r.framework, [0, 0])
        total_covered[1] += 1  # total
        if req_key(r) in covered:
            total_covered[0] += 1  # covered
    return {fw: (c, t) for fw, (c, t) in stats.items()}


def gaps(controls: list[Control]) -> list[Requirement]:
    """Exigences qu'AUCUN controle ne couvre (l'etape 'Act' du PDCA travaille ici)."""
    covered = covered_keys(controls)
    return [r for r in REQUIREMENTS if req_key(r) not in covered]


def render_report(controls: list[Control]) -> str:
    """Rapport board-ready lisible (markdown-ish) du crosswalk + couverture + trous."""
    lines: list[str] = []
    lines.append("=== CONTROL CROSSWALK REPORT (J7 — Normes & AIMS) ===\n")

    lines.append("Crosswalk (controle interne -> exigences satisfaites) :")
    for c in controls:
        lines.append(f"\n  [{c.control_id}] {c.label}")
        # WHY trier: sortie deterministe (reproductible d'un run a l'autre).
        for key in sorted(c.covers):
            r = _REQ_BY_KEY.get(key)
            if r is None:
                continue  # garde-fou: cle orpheline ignoree
            tag = "OBLIGATOIRE" if r.mandatory else "volontaire"
            lines.append(f"      -> {r.framework} {r.ref} ({tag}) : {r.label}")

    lines.append("\nCouverture par referentiel :")
    for fw, (cov, tot) in sorted(coverage_by_framework(controls).items()):
        pct = (cov / tot * 100) if tot else 0.0
        lines.append(f"  - {fw:<16} {cov}/{tot} couvert(es)  ({pct:.0f}%)")

    open_gaps = gaps(controls)
    lines.append(f"\nTrous de conformite (exigences non couvertes) : {len(open_gaps)}")
    if open_gaps:
        for r in open_gaps:
            tag = "OBLIGATOIRE" if r.mandatory else "volontaire"
            lines.append(f"  ! {r.framework} {r.ref} ({tag}) : {r.label}")
    else:
        lines.append("  (aucun — couverture complete sur le catalogue)")

    # WHY ce verdict: separer obligatoire de volontaire est LA distinction du jour.
    mandatory_gaps = [r for r in open_gaps if r.mandatory]
    lines.append("")
    if mandatory_gaps:
        lines.append(f"VERDICT : {len(mandatory_gaps)} trou(s) OBLIGATOIRE(S) -> "
                     "non-conformite legale a corriger (Act).")
    else:
        lines.append("VERDICT : aucune obligation legale non couverte. "
                     "Trous restants = referentiels volontaires (amelioration continue).")
    return "\n".join(lines)


if __name__ == "__main__":
    controls = build_demo_controls()
    print(render_report(controls))

    # --- Probe : on ajoute une exigence obligatoire SANS controle pour la couvrir.
    # WHY: montrer que le mapper detecte bien un trou OBLIGATOIRE et bascule le verdict.
    print("\n--- Probe : nouvelle obligation introduite, non couverte ---")
    REQUIREMENTS.append(
        Requirement("EU AI Act", "Art. 13", "Transparence & instructions d'usage", mandatory=True)
    )
    _REQ_BY_KEY[req_key(REQUIREMENTS[-1])] = REQUIREMENTS[-1]
    print(render_report(controls))

"""Projet guide 03 (gouvernance-ia) — Rapport de gouvernance board-ready + crosswalk.

Mission FleetSim / LogiSim
--------------------------
Un client LogiSim audite ISO 9001 / SOC 2 exploite une flotte d'agents FleetSim
(aide a la decision OCC, EOD automatique, classification d'events, coordination
de flottes externes). Avant un passage a l'echelle, il doit presenter a son
comite un etat de la gouvernance ou **chaque chiffre est derive d'un mecanisme**
(pas saisi a la main) — exigence directe d'un environnement audite (tracabilite).

Cet outil produit ce rapport. Il fait trois choses, dans cet ordre :

    1. SCORE    : criticite = vraisemblance x impact, sur echelles ancrees 1..5,
                  MODULEE par le contexte agentique (action irreversible -> impact +1 ;
                  autonomie totale out-of-the-loop -> vraisemblance +1). Le score est
                  EXPLICABLE : on conserve les ancres et les modulateurs appliques.
                  Sortie : classement TREAT / MONITOR / ACCEPT.        [theorie J4 — NIST AI RMF]

    2. CROSSWALK: une table de controles internes (audit trail inviolable, owner
                  nomme, kill-switch teste, revue humaine des actions high-risk...)
                  mappes vers {EU AI Act (article), NIST AI RMF (fonction),
                  ISO/IEC 42001 (clause)}. On calcule la couverture par referentiel
                  et surtout on ISOLE les trous OBLIGATOIRES (loi, EU AI Act) des
                  trous volontaires (norme).                            [theorie J7 — AIMS]

    3. REPORT   : rapport board-ready en DEUX formats — markdown (lisible/signable
                  par un humain) ET JSON (rejouable/archivable/diffable) — qui se
                  TERMINE sur un verdict actionnable, pas sur un tableau brut.

Posture (cf. IMDA Agentic 2026) : les humains restent ultimement responsables ;
l'outil produit la PREUVE qui eclaire la decision du comite, il ne decide pas.

Run:
    python domains/tech/gouvernance-ia/05-projets-guides/03-rapport-gouvernance-board/solution/governance_report.py

# requires: stdlib only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

# WHY une date figee en constante (et non datetime.now) : un client audite a
# besoin d'un rapport DETERMINISTE — deux executions sur la meme flotte doivent
# produire le meme JSON, octet pour octet, pour pouvoir etre diffees d'un mois
# sur l'autre et signees sans ambiguite. L'horodatage "live" casserait cela.
REPORT_DATE = "2026-06-21"


# ===========================================================================
# MODELE — l'agent FleetSim que l'on gouverne.
# On garde une forme unique, passee telle quelle d'une etape a l'autre.
# ===========================================================================

@dataclass(frozen=True)
class FleetAgent:
    """Un agent de la flotte FleetSim a gouverner.

    Les deux derniers champs sont le CONTEXTE AGENTIQUE qui module le risque :
    un agent qui *agit* de facon irreversible et/ou sans humain dans la boucle
    est plus dangereux a risque "logique" egal (cf. J4, modulateurs agentiques)."""
    agent_id: str
    owner: str | None          # owner humain nomme (None = agent orphelin / shadow)
    role: str                  # description metier courte (pour le rapport)
    risk_tier: str             # "low" | "medium" | "high" — ancre la criticite brute
    autonomous: bool = False           # agit sans validation humaine (out-of-the-loop)
    handles_irreversible: bool = False  # peut declencher une action non annulable
    external_fleet: bool = False        # pilote une flotte tierce (sous-traitant)


# ===========================================================================
# (1) SCORE — NIST AI RMF : criticite = vraisemblance x impact, modulee.
# ===========================================================================

# Echelles ANCREES 1..5 (cf. theorie J4, section 5.1). Le piege a eviter est
# d'utiliser des mots ("eleve", "moyen") sans definition : ici, chaque tier
# d'agent fixe une vraisemblance et un impact BRUTS reproductibles, sur lesquels
# on applique ensuite les modulateurs. C'est ce qui rend le score defendable
# devant un comite : un tiers qui rejoue le calcul obtient le meme chiffre.
LIKELIHOOD_BY_TIER = {"low": 2, "medium": 3, "high": 4}
IMPACT_BY_TIER = {"low": 2, "medium": 3, "high": 4}

# Seuils de traitement (decision de GOUVERNANCE, pas chiffre magique — J4 §5.3).
TREAT_THRESHOLD = 12    # criticite >= 12 -> TREAT (traitement obligatoire)
MONITOR_THRESHOLD = 6   # 6..11 -> MONITOR ; < 6 -> ACCEPT


@dataclass
class RiskScore:
    """Score EXPLICABLE : on ne garde pas que le chiffre final, mais aussi les
    ancres (L/I bruts) et la trace des modulateurs appliques. Sans cela, le
    comite ne peut pas auditer *pourquoi* tel agent sort en TREAT."""
    agent_id: str
    base_likelihood: int
    base_impact: int
    eff_likelihood: int
    eff_impact: int
    criticality: int
    decision: str               # TREAT | MONITOR | ACCEPT
    modulators: list[str] = field(default_factory=list)


def score_agent(agent: FleetAgent) -> RiskScore:
    """Calcule la criticite d'un agent et son classement TREAT/MONITOR/ACCEPT.

    WHY des modulateurs agentiques : le MEME risque logique est plus grave sur
    un agent autonome + irreversible. Irreversible -> impact +1 (rien a annuler) ;
    autonomie totale -> vraisemblance +1 (aucun filet humain pour intercepter).
    Le cap a 5 garde l'echelle coherente. Un score qui ignore ces deux facteurs
    *ment* sur le vrai niveau de risque d'un agent qui agit."""
    base_l = LIKELIHOOD_BY_TIER.get(agent.risk_tier, 2)
    base_i = IMPACT_BY_TIER.get(agent.risk_tier, 2)
    eff_l, eff_i = base_l, base_i
    modulators: list[str] = []

    if agent.handles_irreversible:
        eff_i = min(5, eff_i + 1)
        modulators.append("action irreversible -> impact +1")
    if agent.autonomous:
        eff_l = min(5, eff_l + 1)
        modulators.append("autonomie totale (out-of-the-loop) -> vraisemblance +1")

    crit = eff_l * eff_i
    decision = (
        "TREAT" if crit >= TREAT_THRESHOLD
        else "MONITOR" if crit >= MONITOR_THRESHOLD
        else "ACCEPT"
    )
    return RiskScore(
        agent_id=agent.agent_id,
        base_likelihood=base_l,
        base_impact=base_i,
        eff_likelihood=eff_l,
        eff_impact=eff_i,
        criticality=crit,
        decision=decision,
        modulators=modulators,
    )


# ===========================================================================
# (2) CROSSWALK — un controle interne -> N referentiels ; isoler les trous OBLIGATOIRES.
# ===========================================================================

@dataclass(frozen=True)
class Requirement:
    """Une exigence d'un referentiel.

    Le champ `mandatory` est le COEUR du verdict : il distingue une obligation
    LEGALE (EU AI Act, RGPD — non-respect = sanction) d'un alignement VOLONTAIRE
    (NIST RMF, ISO 42001 — choix qui sert de moyen de preuve). On ne peut pas
    traiter les deux pareil : un trou legal bloque le passage a l'echelle, un
    trou volontaire est une dette a planifier (cf. J7 §5 et J5)."""
    framework: str
    ref: str            # article / fonction / clause
    label: str
    mandatory: bool     # True = obligation legale (loi) ; False = norme volontaire


# Sous-ensemble pedagogique, refs verifiees contre la theorie du domaine
# (J5 EU AI Act : Art. 9/12/14 ; J4 NIST AI RMF : Govern/Map/Measure/Manage ;
#  J7 ISO/IEC 42001 : clauses citees dans le crosswalk de reference).
# Seul EU AI Act est `mandatory` : c'est la seule LOI ici. NIST et ISO sont
# volontaires (mais servent de moyen de preuve de l'obligation legale).
REQUIREMENTS: list[Requirement] = [
    # --- EU AI Act : OBLIGATOIRE (loi) ---
    Requirement("EU AI Act", "Art. 9", "Systeme de gestion des risques", mandatory=True),
    Requirement("EU AI Act", "Art. 12", "Tenue de registres / logging", mandatory=True),
    Requirement("EU AI Act", "Art. 14", "Supervision humaine", mandatory=True),
    # --- NIST AI RMF : VOLONTAIRE (cadre) ---
    Requirement("NIST AI RMF", "Govern", "Roles & responsabilites", mandatory=False),
    Requirement("NIST AI RMF", "Map", "Cartographie du risque", mandatory=False),
    Requirement("NIST AI RMF", "Measure", "Mesure & tracabilite", mandatory=False),
    Requirement("NIST AI RMF", "Manage", "Traitement du risque", mandatory=False),
    # --- ISO/IEC 42001 : VOLONTAIRE (norme certifiable) ---
    Requirement("ISO/IEC 42001", "6.1", "Evaluation des risques IA", mandatory=False),
    Requirement("ISO/IEC 42001", "8.2", "Suivi operationnel & journalisation", mandatory=False),
    Requirement("ISO/IEC 42001", "5.3", "Roles & responsabilites de l'AIMS", mandatory=False),
]


def _req_key(r: Requirement) -> str:
    # Cle stable framework::ref, utilisee pour relier un controle a ses exigences.
    return f"{r.framework}::{r.ref}"


@dataclass(frozen=True)
class Control:
    """Un controle interne implemente par le client.

    `covers` liste les exigences (par cle framework::ref) que ce SEUL controle
    satisfait. C'est l'idee directrice du crosswalk : on implemente un controle
    UNE fois, on prouve la conformite N fois (J7 §5). `implemented` permet de
    declarer honnetement qu'un controle est prevu mais pas encore en place — un
    controle absent ne couvre RIEN (sinon le crosswalk devient du theatre)."""
    control_id: str
    label: str
    implemented: bool
    covers: frozenset[str]


def build_controls(controls_state: dict[str, bool]) -> list[Control]:
    """Construit la table de controles internes du client.

    WHY un dict d'etat passe en parametre : l'etat reel des controles est une
    DONNEE d'audit, pas une constante. Le client declare ce qui est en place ;
    le crosswalk en derive mecaniquement la couverture et les trous. Changer un
    booleen ici (ex. kill-switch non teste) doit faire bouger le verdict — c'est
    ce qui rend chaque chiffre du rapport tracable jusqu'a un fait."""
    catalogue = [
        Control(
            "CTRL-RISK", "Evaluation de risque par agent (scoring NIST)",
            implemented=controls_state.get("CTRL-RISK", False),
            covers=frozenset({
                "EU AI Act::Art. 9",
                "NIST AI RMF::Map", "NIST AI RMF::Measure",
                "ISO/IEC 42001::6.1",
            }),
        ),
        Control(
            "CTRL-AUDIT", "Audit trail inviolable (hash-chaine) des actions",
            implemented=controls_state.get("CTRL-AUDIT", False),
            covers=frozenset({
                "EU AI Act::Art. 12",
                "NIST AI RMF::Manage",
                "ISO/IEC 42001::8.2",
            }),
        ),
        Control(
            "CTRL-OWNER", "Owner humain nomme par agent",
            implemented=controls_state.get("CTRL-OWNER", False),
            # L'owner nomme = roles & responsabilites (Govern cote NIST, 5.3 cote
            # ISO). On NE le mappe PAS sur Art. 14 : nommer un responsable n'est
            # pas la meme exigence que pouvoir reprendre la main en temps reel.
            covers=frozenset({
                "NIST AI RMF::Govern",
                "ISO/IEC 42001::5.3",
            }),
        ),
        Control(
            "CTRL-HITL", "Revue humaine des actions a haut risque (HITL)",
            implemented=controls_state.get("CTRL-HITL", False),
            # La revue humaine en amont des actions sensibles releve du traitement
            # du risque (Manage). Necessaire mais pas suffisante pour Art. 14.
            covers=frozenset({
                "NIST AI RMF::Manage",
            }),
        ),
        Control(
            "CTRL-KILL", "Kill-switch teste (arret d'urgence verifie)",
            implemented=controls_state.get("CTRL-KILL", False),
            # WHY le kill-switch porte Art. 14 (supervision humaine) : l'EU AI Act
            # exige que l'humain puisse INTERROMPRE le systeme. Un kill-switch non
            # TESTE ne prouve rien (d'ou le booleen implemented) -> il laisse un
            # trou sur une exigence LEGALE, ce que le verdict doit faire remonter
            # comme bloquant. C'est l'illustration centrale du projet.
            covers=frozenset({
                "EU AI Act::Art. 14",
                "NIST AI RMF::Manage",
            }),
        ),
    ]
    return catalogue


@dataclass
class CrosswalkResult:
    """Resultat du crosswalk : couverture par referentiel + trous tries par
    obligation. On separe `mandatory_gaps` de `voluntary_gaps` parce que le
    verdict ne les traite pas pareil (loi vs norme)."""
    coverage: dict[str, tuple[int, int]]   # framework -> (couverts, total)
    mandatory_gaps: list[Requirement]
    voluntary_gaps: list[Requirement]


def compute_crosswalk(controls: list[Control]) -> CrosswalkResult:
    """Croise les controles IMPLEMENTES contre toutes les exigences.

    WHY ne compter que les controles implementes : un controle prevu mais absent
    ne prouve aucune conformite. C'est ce qui fait remonter un kill-switch non
    teste comme un trou (potentiellement legal, via Art. 14) plutot que de le
    masquer."""
    covered: set[str] = set()
    for c in controls:
        if c.implemented:
            covered |= c.covers

    coverage: dict[str, tuple[int, int]] = {}
    for r in REQUIREMENTS:
        cov, tot = coverage.get(r.framework, (0, 0))
        tot += 1
        if _req_key(r) in covered:
            cov += 1
        coverage[r.framework] = (cov, tot)

    gaps = [r for r in REQUIREMENTS if _req_key(r) not in covered]
    # Le tri obligatoire/volontaire est le coeur du verdict : on isole ce qui
    # bloque legalement (EU AI Act) de ce qui est une dette d'alignement (norme).
    mandatory_gaps = [r for r in gaps if r.mandatory]
    voluntary_gaps = [r for r in gaps if not r.mandatory]
    return CrosswalkResult(coverage, mandatory_gaps, voluntary_gaps)


# ===========================================================================
# (3) REPORT — board-ready, en markdown ET JSON, finissant sur un VERDICT.
# ===========================================================================

@dataclass
class GovernanceReport:
    """Tout ce que le rapport doit presenter, rassemble par le pipeline."""
    client: str
    report_date: str
    fleet_size: int
    orphans: list[str]                 # agents sans owner nomme
    scores: list[RiskScore]
    crosswalk: CrosswalkResult
    controls: list[Control]


def _verdict(report: GovernanceReport) -> str:
    """Le verdict actionnable — la seule ligne que le board lit vraiment.

    WHY finir sur un verdict et non un tableau : un comite DECIDE, il ne lit pas
    des stats brutes. L'ordre de priorite encode la doctrine : un trou LEGAL
    (obligatoire) prime tout — il bloque le passage a l'echelle — avant les
    trous volontaires (dette planifiable) et les agents orphelins."""
    n_mand = len(report.crosswalk.mandatory_gaps)
    n_vol = len(report.crosswalk.voluntary_gaps)
    if n_mand:
        refs = ", ".join(f"{g.framework} {g.ref}" for g in report.crosswalk.mandatory_gaps)
        plur = "s" if n_mand > 1 else ""
        return (f"{n_mand} trou{plur} legal{plur} OBLIGATOIRE ({refs}) "
                f"-> remediation requise AVANT passage a l'echelle.")
    if report.orphans:
        return (f"Aucun trou legal, mais {len(report.orphans)} agent(s) orphelin(s) "
                f"({', '.join(report.orphans)}) -> nommer un owner avant passage a l'echelle.")
    if n_vol:
        return (f"Aucun trou legal obligatoire ; {n_vol} ecart(s) volontaire(s) restant(s) "
                f"-> dette d'alignement a planifier, n'empeche pas le passage a l'echelle.")
    return ("Aucun trou legal ni volontaire, flotte gouvernee "
            "-> passage a l'echelle autorise (re-executer le rapport chaque mois).")


def render_markdown(report: GovernanceReport) -> str:
    """Format humain : lisible et signable par un membre du comite."""
    lines: list[str] = []
    bar = "=" * 70
    lines.append(bar)
    lines.append(f"RAPPORT DE GOUVERNANCE DES AGENTS — {report.client} — {report.report_date}")
    lines.append(bar)
    lines.append(f"Flotte FleetSim : {report.fleet_size} agents")
    orphan_note = ", ".join(report.orphans) if report.orphans else "aucun"
    lines.append(f"Agents orphelins (sans owner nomme) : {orphan_note}")
    lines.append("")

    # --- Posture de risque (NIST AI RMF) ---
    lines.append("1. POSTURE DE RISQUE (NIST AI RMF — criticite = vraisemblance x impact)")
    lines.append("-" * 70)
    for s in sorted(report.scores, key=lambda x: x.criticality, reverse=True):
        why = "; ".join(s.modulators) if s.modulators else "aucun modulateur"
        lines.append(
            f"  [{s.decision:<7}] {s.agent_id:<22} "
            f"crit={s.criticality:>2}  (L{s.eff_likelihood} x I{s.eff_impact}"
            f", base L{s.base_likelihood}/I{s.base_impact})"
        )
        lines.append(f"            modulateurs : {why}")
    n_treat = sum(1 for s in report.scores if s.decision == "TREAT")
    lines.append("")
    lines.append(f"  -> {n_treat} agent(s) en TREAT (traitement prioritaire requis)")
    lines.append("")

    # --- Crosswalk de conformite ---
    lines.append("2. CONFORMITE — COUVERTURE PAR REFERENTIEL (crosswalk)")
    lines.append("-" * 70)
    for fw in sorted(report.crosswalk.coverage):
        cov, tot = report.crosswalk.coverage[fw]
        pct = (cov / tot * 100) if tot else 0
        tag = "OBLIGATOIRE (loi)" if any(
            r.framework == fw and r.mandatory for r in REQUIREMENTS
        ) else "volontaire (norme)"
        lines.append(f"  {fw:<16} {cov}/{tot} ({pct:>3.0f}%)   [{tag}]")
    lines.append("")

    # --- Trous, separes obligatoire / volontaire (le coeur du rapport) ---
    lines.append("3. TROUS DE CONFORMITE (separes par nature)")
    lines.append("-" * 70)
    if report.crosswalk.mandatory_gaps:
        lines.append("  TROUS OBLIGATOIRES (loi — bloquants) :")
        for g in report.crosswalk.mandatory_gaps:
            lines.append(f"    ! {g.framework} {g.ref} — {g.label}")
    else:
        lines.append("  TROUS OBLIGATOIRES (loi) : aucun.")
    if report.crosswalk.voluntary_gaps:
        lines.append("  Trous volontaires (norme — dette a planifier) :")
        for g in report.crosswalk.voluntary_gaps:
            lines.append(f"    - {g.framework} {g.ref} — {g.label}")
    else:
        lines.append("  Trous volontaires (norme) : aucun.")
    lines.append("")

    # --- Verdict actionnable (la ligne que le board signe) ---
    lines.append("4. VERDICT")
    lines.append("-" * 70)
    lines.append(f"  {_verdict(report)}")
    lines.append("")
    lines.append("  Note : les humains restent ultimement responsables. Ce rapport")
    lines.append("  produit la preuve qui eclaire la decision du comite ; il ne decide pas.")
    lines.append(bar)
    return "\n".join(lines)


def render_json(report: GovernanceReport) -> str:
    """Format machine : rejouable, archivable, diffable d'un mois sur l'autre.

    WHY deux formats : le markdown sert a la lecture/signature humaine du comite ;
    le JSON sert a la TRACABILITE (archiver le rapport du trimestre, le differ
    avec le suivant pour prouver la progression a un auditeur SOC 2 / ISO 9001).
    Le meme contenu, deux usages. `sort_keys` garantit un diff stable."""
    payload = {
        "client": report.client,
        "report_date": report.report_date,
        "fleet_size": report.fleet_size,
        "orphans": report.orphans,
        "risk": [
            {
                "agent_id": s.agent_id,
                "base": {"likelihood": s.base_likelihood, "impact": s.base_impact},
                "effective": {"likelihood": s.eff_likelihood, "impact": s.eff_impact},
                "criticality": s.criticality,
                "decision": s.decision,
                "modulators": s.modulators,
            }
            for s in sorted(report.scores, key=lambda x: x.criticality, reverse=True)
        ],
        "controls": [
            {"control_id": c.control_id, "label": c.label, "implemented": c.implemented}
            for c in report.controls
        ],
        "compliance": {
            "coverage": {
                fw: {"covered": cov, "total": tot}
                for fw, (cov, tot) in report.crosswalk.coverage.items()
            },
            "mandatory_gaps": [
                f"{g.framework} {g.ref}" for g in report.crosswalk.mandatory_gaps
            ],
            "voluntary_gaps": [
                f"{g.framework} {g.ref}" for g in report.crosswalk.voluntary_gaps
            ],
        },
        "verdict": _verdict(report),
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


# ===========================================================================
# PIPELINE — derive le rapport a partir de la flotte + l'etat des controles.
# ===========================================================================

def build_report(client: str, fleet: Iterable[FleetAgent],
                 controls_state: dict[str, bool]) -> GovernanceReport:
    """Assemble le rapport : chaque champ est DERIVE, jamais saisi a la main.

    Note de tracabilite : la presence d'un owner sur chaque agent conditionne le
    controle CTRL-OWNER (Art. 14, supervision humaine). Un agent orphelin fait
    donc mecaniquement tomber un controle qui couvre une exigence LEGALE — c'est
    ainsi qu'un owner manquant devient un trou obligatoire dans le verdict."""
    fleet = list(fleet)
    orphans = [a.agent_id for a in fleet if not a.owner]

    # Le controle "owner nomme" n'est reellement implemente que si AUCUN agent
    # n'est orphelin : on relie l'etat declare au fait observe sur la flotte.
    effective_state = dict(controls_state)
    if orphans:
        effective_state["CTRL-OWNER"] = False

    scores = [score_agent(a) for a in fleet]
    controls = build_controls(effective_state)
    crosswalk = compute_crosswalk(controls)

    return GovernanceReport(
        client=client,
        report_date=REPORT_DATE,
        fleet_size=len(fleet),
        orphans=orphans,
        scores=scores,
        crosswalk=crosswalk,
        controls=controls,
    )


# ===========================================================================
# DEMO — une flotte FleetSim de >=4 agents varies.
# ===========================================================================

def _demo_fleet() -> list[FleetAgent]:
    """Flotte FleetSim representative : un finance/irreversible HIGH, un support
    MED, un classifieur LOW, un agent de flotte externe."""
    return [
        # Agent qui declenche des paiements fournisseurs transport -> irreversible,
        # autonome, HIGH : le pire cas, doit ressortir en tete du classement.
        FleetAgent(
            agent_id="fleet-finance-billing-01",
            owner="a.dupont",
            role="Reglement automatique des transporteurs (paiements)",
            risk_tier="high",
            autonomous=True,
            handles_irreversible=True,
        ),
        # Agent de support OCC : repond aux operateurs, peut rouvrir un work order.
        FleetAgent(
            agent_id="occ-support-assistant-02",
            owner="m.martin",
            role="Assistant OCC (aide a la decision, reouverture de work orders)",
            risk_tier="medium",
            autonomous=False,
            handles_irreversible=False,
        ),
        # Classifieur d'events de shift (collision / fault) : lecture seule, LOW.
        FleetAgent(
            agent_id="shift-event-classifier-03",
            owner="c.bernard",
            role="Classification des events de shift (EOD)",
            risk_tier="low",
            autonomous=False,
            handles_irreversible=False,
        ),
        # Coordinateur de flotte EXTERNE (sous-traitants last-mile) : autonome,
        # MED, et orphelin -> illustre comment un owner manquant cree un trou legal.
        FleetAgent(
            agent_id="external-fleet-coordinator-04",
            owner=None,
            role="Coordination de la flotte externe (livreurs tiers)",
            risk_tier="medium",
            autonomous=True,
            handles_irreversible=False,
            external_fleet=True,
        ),
    ]


def main() -> None:
    fleet = _demo_fleet()

    # Etat declare des controles internes du client. Volontairement realiste :
    # presque tout est en place, SAUF le kill-switch teste (CTRL-KILL=False).
    # Comme le kill-switch est le seul controle qui porte Art. 14 (supervision
    # humaine, EU AI Act), son absence ouvre un trou sur une exigence LEGALE
    # OBLIGATOIRE -> c'est ce qui doit bloquer le passage a l'echelle.
    # En parallele, l'agent orphelin (external-fleet-coordinator-04) fait tomber
    # CTRL-OWNER, ce qui cree des trous VOLONTAIRES (Govern, ISO 5.3). Le rapport
    # montre les deux natures de trous, et le verdict priorise le legal.
    controls_state = {
        "CTRL-RISK": True,
        "CTRL-AUDIT": True,
        "CTRL-OWNER": True,    # sera force a False par le pipeline (agent orphelin)
        "CTRL-HITL": True,
        "CTRL-KILL": False,    # kill-switch pas encore teste -> trou legal Art. 14
    }

    report = build_report("LogiSim Client — Plateforme Nord", fleet, controls_state)

    # --- Rapport humain (markdown, lu et signe par le comite) ---
    print(render_markdown(report))

    # --- Rapport machine (JSON, archive / diffe / rejoue) ---
    print("\nArtefact JSON (rejouable / archivable / diffable) :")
    print(render_json(report))

    # --- Verdict final, repete seul pour ne pas se perdre dans le rapport ---
    print("\n" + "#" * 70)
    print("VERDICT FINAL :", _verdict(report))
    print("#" * 70)


if __name__ == "__main__":
    main()

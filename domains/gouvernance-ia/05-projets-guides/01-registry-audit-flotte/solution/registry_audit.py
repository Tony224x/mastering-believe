"""Projet guide 01 — Registry de gouvernance & audit des agents orphelins (FleetSim).

Le probleme metier
------------------
Chez un client LogiSim, plusieurs equipes ops ont deploye des agents LLM
"fleet brain" (les coordinateurs de flotte du projet Agentic 01) pour piloter
des flottes FleetSim. Personne ne sait *combien* il y en a, ni *qui* les
possede. C'est exactement la question fondatrice du domaine — « combien d'agents
tournent chez nous, et qui les possede ? » — appliquee a une salle de controle
logistique (OCC).

Ce que construit ce script
--------------------------
Un **registry de gouvernance** minimal qui :
  (1) INGEST   une flotte d'agents decrite en dur (liste de dicts facon
               `fleet.json`) — own fleet et external fleet melangees ;
  (2) VALIDE   les 4 piliers par agent (identite non-humaine, owner humain
               nomme, permissions au moindre privilege bornees par scopes,
               presence d'audit) — chaque pilier => OK / PARTIEL / ABSENT ;
  (3) DETECTE  les orphelins (agents sans owner nomme = shadow agents) et les
               rend VISIBLES (on ne les supprime jamais) ;
  (4) CALCULE  la couverture de gouvernance = agents pleinement gouvernes / total ;
  (5) IMPRIME  un tableau registry lisible + la liste des orphelins + le taux.

WHY un seul fichier stdlib : un registry de gouvernance est un livrable qu'on
remet a un client on-prem, reseau souvent isole (cf. shared/logistics-context.md).
Il doit tourner seul, sans dependance externe ni cle API. Le registry reel qu'il
mime est nomme en commentaire (Microsoft Entra Agent ID, Google A2A Agent Card).

Ancre metier specifique LogiSim
-------------------------------
Le contexte est **on-premise isole** et les **flux client sont confidentiels**
(jamais de telemetrie sortante non-anonymisee). Donc un agent `external-fleet`
qui detient le scope `export:client_telemetry` est un **drapeau rouge** de
gouvernance, meme s'il a un owner : le pire-cas de cet agent est une fuite de
donnees client hors perimetre. Le validateur le marque explicitement.

Run:
    python domains/gouvernance-ia/05-projets-guides/01-registry-audit-flotte/solution/registry_audit.py

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum


# ===========================================================================
# (0) La flotte d'entree — codee en dur, mais de la FORME d'un fleet.json.
#     WHY un dict module-level : on simule un export/scan de la flotte
#     d'agents tournant chez le client. Un vrai export est imparfait — owner
#     manquant, scope trop large, identite empruntee a un humain : on garde
#     volontairement ces defauts pour que l'audit ait quelque chose a trouver.
# ===========================================================================

FLEET: list[dict] = [
    # --- Own fleet : coordinateurs FleetSim internes, bien gouvernes ---
    {
        "agent_id": "agent://fleet-brain/dock-b-01",
        "owner": "a.dupont",                       # humain nomme -> OK
        "fleet_kind": "own",
        "scopes": ["fleet:read", "work_order:read", "report:write"],
        "risk_tier": "medium",
        "has_audit": True,
    },
    {
        "agent_id": "agent://fleet-brain/picking-02",
        "owner": "m.martin",
        "fleet_kind": "own",
        "scopes": ["fleet:read", "fleet:dispatch", "report:write"],
        "risk_tier": "high",                       # pilote des AGV -> tier haut
        "has_audit": True,
    },
    {
        "agent_id": "agent://fleet-brain/eod-report-03",
        "owner": "c.bernard",
        "fleet_kind": "own",
        "scopes": ["report:read"],                 # moindre privilege exemplaire
        "risk_tier": "low",
        "has_audit": True,
    },
    # --- Orphelin : deploye par une equipe ops, owner jamais renseigne ---
    #     C'est le shadow agent. Techniquement vivant, organisationnellement
    #     abandonne. La requete `orphans` doit le faire remonter.
    {
        "agent_id": "agent://fleet-brain/sorter-a-07",
        "owner": None,                             # ABSENT -> orphelin
        "fleet_kind": "own",
        "scopes": ["fleet:read", "report:write"],
        "risk_tier": "medium",
        "has_audit": True,
    },
    # --- External fleet (sous-traitant) avec un scope INTERDIT en isole ---
    #     export:client_telemetry = telemetrie client sortante. Drapeau rouge :
    #     viole la confidentialite des flux client (contexte on-prem isole).
    {
        "agent_id": "agent://ext-coord/3pl-northgate",
        "owner": "ext.northgate-ops",              # owner present mais externe
        "fleet_kind": "external",
        "scopes": ["fleet:read", "export:client_telemetry"],
        "risk_tier": "high",
        "has_audit": True,
    },
    # --- Agent sur-permissionne : wildcard = il peut tout faire ---
    #     Heriter d'un acces "*" n'est pas une permission, c'est une bombe :
    #     chaque defaillance devient maximale au lieu d'etre bornee (OWASP ASI03).
    {
        "agent_id": "agent://fleet-brain/staging-09",
        "owner": "a.dupont",
        "fleet_kind": "own",
        "scopes": ["*"],                           # PARTIEL : scope present mais non borne
        "risk_tier": "high",
        "has_audit": True,
    },
    # --- Agent usurpant une identite humaine + sans audit ---
    #     Il porte la cle d'un humain (user:...). Dans les logs OCC, impossible
    #     de distinguer ce qu'a fait l'humain de ce qu'a fait l'agent.
    {
        "agent_id": "user:j.operator",             # identite HUMAINE empruntee -> ABSENT
        "owner": "j.operator",
        "fleet_kind": "own",
        "scopes": ["fleet:read", "fleet:dispatch"],
        "risk_tier": "high",
        "has_audit": False,                        # aucune trace -> non-prouvable
    },
    # --- Agent quasi-vide : ni scopes, ni audit, owner = boite generique ---
    {
        "agent_id": "agent://fleet-brain/legacy-99",
        "owner": "ops-team",                       # equipe, pas une personne -> orphelin
        "fleet_kind": "own",
        "scopes": [],                              # ABSENT
        "risk_tier": "low",
        "has_audit": False,
    },
]


# ===========================================================================
# Conventions de gouvernance (les regles de lecture des piliers).
# WHY centraliser : ces ensembles encodent la doctrine du domaine. Les isoler
# permet de les auditer / faire evoluer sans toucher la logique.
# ===========================================================================

# Une identite qui commence par un de ces prefixes est HUMAINE : un agent qui la
# porte usurpe l'identite d'une personne (pilier identite viole).
_HUMAN_ID_PREFIXES = ("user:", "human:", "person:", "employee:")

# Valeurs d'owner qui ne designent PAS une personne nommee redevable. Une
# responsabilite diffuse ("l'equipe") est une responsabilite nulle : on ne peut
# ni escalader, ni decider d'un kill-switch, ni imputer.
_NON_OWNERS = {"", "team", "it", "ops-team", "data team", "n/a", "tbd", "unknown"}

# Scope wildcard = l'agent peut tout faire. Scope present mais non borne -> PARTIEL.
_WILDCARD_SCOPES = {"*", "all", "admin:*"}

# Scopes interdits dans un contexte on-prem isole : ils exfiltrent de la
# telemetrie / des donnees client hors perimetre. Confidentialite des flux
# client = contrainte contractuelle LogiSim (cf. logistics-context.md).
_FORBIDDEN_EGRESS_SCOPES = {"export:client_telemetry", "export:raw_flows", "net:external"}


class Pillar(IntEnum):
    """Etat d'un pilier. Ordonne par severite croissante pour pouvoir trier /
    afficher : OK < PARTIEL < ABSENT."""
    OK = 0
    PARTIAL = 1
    ABSENT = 2


_PILLAR_GLYPH = {Pillar.OK: "OK ", Pillar.PARTIAL: "PRT", Pillar.ABSENT: "ABS"}


# ===========================================================================
# (1) INGEST — un agent gouverne, charge depuis la flotte brute.
# ===========================================================================

@dataclass
class RegistryAgent:
    """L'unite que l'on gouverne. Echo des 4 piliers : identite, owner,
    permissions, audit. Les champs `fleet_kind` / `risk_tier` servent
    l'ancrage metier (own vs external, priorisation de l'attention)."""
    agent_id: str
    owner: str | None
    fleet_kind: str                 # "own" | "external"
    scopes: tuple[str, ...]
    risk_tier: str                  # "low" | "medium" | "high"
    has_audit: bool

    @classmethod
    def from_raw(cls, raw: dict) -> "RegistryAgent":
        # WHY tolerer les cles manquantes : un export reel est imparfait. On
        # n'ecarte PAS un agent incomplet — on l'admet pour que ses trous
        # deviennent visibles et comptes (sinon on recree le shadow AI).
        return cls(
            agent_id=raw.get("agent_id", "").strip(),
            owner=(raw.get("owner") or None),
            fleet_kind=raw.get("fleet_kind", "own"),
            scopes=tuple(raw.get("scopes", ())),
            risk_tier=raw.get("risk_tier", "low"),
            has_audit=bool(raw.get("has_audit", False)),
        )


# ===========================================================================
# (2) VALIDATION DES 4 PILIERS — chacun -> OK / PARTIEL / ABSENT.
#     WHY un etat ternaire (et pas un booleen) : la gouvernance reelle a des
#     demi-teintes. Un scope present mais wildcard n'est pas "absent" (il y a
#     une intention de scoping) mais pas "ok" non plus (il n'est pas borne).
#     Distinguer PARTIEL d'ABSENT permet de prioriser la remediation.
# ===========================================================================

@dataclass
class PillarReport:
    identity: Pillar
    owner: Pillar
    permissions: Pillar
    audit: Pillar
    flags: list[str] = field(default_factory=list)  # drapeaux rouges metier

    def all_ok(self) -> bool:
        # Pleinement gouverne = les 4 piliers OK ET aucun drapeau rouge metier.
        # WHY inclure les flags : un agent peut cocher les 4 piliers et rester
        # dangereux (ex. external-fleet exfiltrant de la telemetrie client).
        return (
            self.identity == Pillar.OK
            and self.owner == Pillar.OK
            and self.permissions == Pillar.OK
            and self.audit == Pillar.OK
            and not self.flags
        )

    def ok_count(self) -> int:
        return sum(1 for p in (self.identity, self.owner, self.permissions, self.audit)
                   if p == Pillar.OK)


def _check_identity(agent: RegistryAgent) -> Pillar:
    """Pilier 1 — identite unique, non-humaine, presente."""
    if not agent.agent_id:
        return Pillar.ABSENT                       # pas d'ID du tout -> non attribuable
    if agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES):
        return Pillar.ABSENT                       # usurpe une identite humaine = invisible
    return Pillar.OK


def _check_owner(agent: RegistryAgent) -> Pillar:
    """Pilier 2 — owner humain nomme (une personne, pas une boite)."""
    owner = (agent.owner or "").strip().lower()
    if owner in _NON_OWNERS:
        return Pillar.ABSENT                       # orphelin : personne de redevable
    # WHY PARTIEL pour un owner externe : il existe une personne de contact, mais
    # un sous-traitant n'a pas la meme redevabilite interne (escalade, kill-switch).
    if agent.fleet_kind == "external":
        return Pillar.PARTIAL
    return Pillar.OK


def _check_permissions(agent: RegistryAgent) -> Pillar:
    """Pilier 3 — moindre privilege : scopes explicites et bornes."""
    if not agent.scopes:
        return Pillar.ABSENT                       # aucun scope declare
    if any(s in _WILDCARD_SCOPES for s in agent.scopes):
        return Pillar.PARTIAL                      # scope present mais non borne (wildcard)
    return Pillar.OK


def _check_audit(agent: RegistryAgent) -> Pillar:
    """Pilier 4 — presence d'un audit trail (la trace rend l'action prouvable)."""
    return Pillar.OK if agent.has_audit else Pillar.ABSENT


def _business_flags(agent: RegistryAgent) -> list[str]:
    """Drapeaux rouges propres au contexte LogiSim (au-dela des 4 piliers).

    WHY separes des piliers : ce sont des regles METIER, pas des invariants de
    gouvernance generiques. La plus critique : un agent qui peut exfiltrer de la
    telemetrie client viole la confidentialite des flux dans un site isole."""
    flags: list[str] = []
    forbidden = [s for s in agent.scopes if s in _FORBIDDEN_EGRESS_SCOPES]
    if forbidden:
        # Pire pour une external-fleet : donnee client sortant vers un tiers.
        sev = "CRITIQUE" if agent.fleet_kind == "external" else "ELEVE"
        flags.append(f"egress interdit en isole [{sev}]: {', '.join(forbidden)}")
    return flags


def validate_pillars(agent: RegistryAgent) -> PillarReport:
    """Applique les 4 controles + les drapeaux metier a un agent."""
    return PillarReport(
        identity=_check_identity(agent),
        owner=_check_owner(agent),
        permissions=_check_permissions(agent),
        audit=_check_audit(agent),
        flags=_business_flags(agent),
    )


# ===========================================================================
# (3)+(4) LE REGISTRY — controle vivant : ingest, requetes, couverture.
#     Real-world : Microsoft Entra Agent ID / Google A2A Agent Card (emergents).
#     WHY un objet et pas un tableur : un registry CONDITIONNE des decisions et
#     REPOND a des requetes sans intervention humaine. Regle d'or du domaine :
#     pas dans le registry => pas en production.
# ===========================================================================

class Registry:
    def __init__(self) -> None:
        # On garde l'agent ET son rapport de piliers cote a cote : le rapport
        # est calcule a l'ingestion, le registry sert ensuite de source de verite.
        self._entries: dict[str, tuple[RegistryAgent, PillarReport]] = {}
        self.enrolled_at = datetime.now(timezone.utc).isoformat()

    def ingest(self, raw_fleet: list[dict]) -> None:
        for raw in raw_fleet:
            agent = RegistryAgent.from_raw(raw)
            if not agent.agent_id:
                # Sans aucun ID, on ne peut meme pas indexer l'agent. On
                # l'ignore comme cle MAIS on aurait, en prod, ouvert un ticket
                # "telemetrie sans identite" — ici on garde le code simple.
                continue
            self._entries[agent.agent_id] = (agent, validate_pillars(agent))

    def all(self) -> list[tuple[RegistryAgent, PillarReport]]:
        return list(self._entries.values())

    def orphans(self) -> list[RegistryAgent]:
        # LA requete critique : agents sans owner nomme (= shadow agents).
        # On les rend VISIBLES, on ne les retire pas du registry.
        return [a for a, rep in self._entries.values() if rep.owner == Pillar.ABSENT]

    def red_flags(self) -> list[tuple[RegistryAgent, list[str]]]:
        # Agents porteurs d'un drapeau rouge metier (ex. egress client interdit).
        return [(a, rep.flags) for a, rep in self._entries.values() if rep.flags]

    def coverage(self) -> tuple[int, int, float]:
        # Couverture de gouvernance = agents pleinement gouvernes / total.
        # C'est l'indicateur titre porte au board.
        total = len(self._entries)
        governed = sum(1 for _, rep in self._entries.values() if rep.all_ok())
        pct = (governed / total * 100.0) if total else 0.0
        return governed, total, pct


# ===========================================================================
# (5) RAPPORT — tableau registry + orphelins + couverture, lisible a l'ecran.
# ===========================================================================

def render_registry(reg: Registry, org: str) -> str:
    lines: list[str] = []
    bar = "=" * 92
    lines.append(bar)
    # ASCII-only dans la sortie affichee : le rendu doit etre lisible sur
    # n'importe quel terminal client (Windows cp1252 inclus), pas seulement UTF-8.
    lines.append(f"REGISTRY DE GOUVERNANCE - {org} - {reg.enrolled_at[:19]}Z")
    lines.append(bar)

    # En-tete du tableau. Pillars = I(dentite) O(wner) P(ermissions) A(udit).
    lines.append(f"{'agent_id':<36} {'owner':<20} {'fleet':<8} I/O/P/A      statut")
    lines.append("-" * 92)

    # On trie les pires en premier (moins de piliers OK d'abord) : un comite
    # lit le haut du tableau, pas 400 lignes.
    rows = sorted(reg.all(), key=lambda e: (e[1].ok_count(), e[0].agent_id))
    for agent, rep in rows:
        owner = agent.owner or "(aucun)"
        pillars = " ".join((
            _PILLAR_GLYPH[rep.identity],
            _PILLAR_GLYPH[rep.owner],
            _PILLAR_GLYPH[rep.permissions],
            _PILLAR_GLYPH[rep.audit],
        ))
        if rep.all_ok():
            statut = "GOUVERNE"
        elif rep.owner == Pillar.ABSENT:
            statut = "ORPHELIN"
        elif rep.flags:
            statut = "DRAPEAU ROUGE"
        else:
            statut = "INCOMPLET"
        lines.append(f"{agent.agent_id:<36} {owner:<20} {agent.fleet_kind:<8} {pillars}  {statut}")

    lines.append("")

    # --- Bloc orphelins : la question fondatrice rendue visible ---
    orphans = reg.orphans()
    lines.append(f"ORPHELINS (agents sans owner nomme = shadow agents) : {len(orphans)}")
    if orphans:
        for a in orphans:
            lines.append(f"  - {a.agent_id}  (owner declare: {a.owner or '(aucun)'})")
    else:
        lines.append("  - aucun")

    lines.append("")

    # --- Bloc drapeaux rouges metier (confidentialite des flux client) ---
    red = reg.red_flags()
    lines.append(f"DRAPEAUX ROUGES METIER (confidentialite / egress en site isole) : {len(red)}")
    if red:
        for a, flags in red:
            for f in flags:
                lines.append(f"  - {a.agent_id} [{a.fleet_kind}]: {f}")
    else:
        lines.append("  - aucun")

    lines.append("")

    # --- Couverture + verdict : un board decide, il ne lit pas des tables ---
    governed, total, pct = reg.coverage()
    lines.append(f"COUVERTURE DE GOUVERNANCE : {pct:.0f}%  ({governed}/{total} agents pleinement gouvernes)")
    if orphans:
        lines.append("VERDICT : agents orphelins detectes -> nommer un owner humain avant tout passage en prod.")
    elif red:
        lines.append("VERDICT : drapeau rouge confidentialite -> revoquer le scope d'egress avant scale-up.")
    elif pct < 100.0:
        lines.append("VERDICT : flotte sans orphelin mais incomplete -> combler les piliers manquants.")
    else:
        lines.append("VERDICT : flotte pleinement gouvernee -> re-auditer apres chaque deploiement.")
    lines.append(bar)
    return "\n".join(lines)


# ===========================================================================
# DEMO
# ===========================================================================

def main() -> None:
    reg = Registry()
    reg.ingest(FLEET)
    print(render_registry(reg, org="LogiSim client - OCC Northgate"))

    # --- Probe adversariale : une flotte vide ne doit pas planter (0/0) ---
    empty = Registry()
    empty.ingest([])
    g, t, pct = empty.coverage()
    print(f"\n[probe] flotte vide -> couverture {pct:.0f}% ({g}/{t}), orphelins={len(empty.orphans())} (pas de crash)")


if __name__ == "__main__":
    main()

"""Projet guide 02 (PHARE) — Gate d'autorisation runtime PDP/PEP pour la flotte OCC.

Contexte FleetSim / LogiSim
---------------------------
Les agents LLM "fleet brain" de l'OCC (Operations Control Center) tentent des
ACTIONS SENSIBLES dans le monde reel : engager une flotte tierce coûteuse,
desactiver une regle de securite robot, rouvrir une zone aux pietons, exporter
de la telemetrie client. Chacune de ces actions est irreversible, coûteuse ou
reglementee. Un fleet-brain detourne par une prompt injection (un Work Order
piege) ne doit JAMAIS pouvoir les declencher seul.

Ce module construit le **gate de gouvernance runtime** que chaque action sensible
traverse AVANT execution. C'est le cœur operationnel de la gouvernance agentique :
le seul endroit ou l'on peut transformer une derive ("l'agent a ete detourne") en
un simple refus d'autorisation ("il n'avait pas le scope" / "ca demande un humain").

Le gate enchaine, dans cet ordre operationnel NON negociable (cf. J10 §1) :

    kill-switch (externe, consulte EN PREMIER, fail-safe)
      -> PDP   : regles declaratives -> allow / deny / oblige
                 (scope J8, budget J10, autonomie/tier J4-J10, donnees J6)
      -> merge : precedence de sûrete  deny > oblige > allow  (J14)
      -> PEP   : applique le verdict -> ACT / BLOCK / ESCALATE (consentement MCP)
      -> LOG   : CHAQUE decision (allow comme deny comme escalate) ecrite dans
                 une chaine d'audit chainee par hash, tamper-evident (J9)

WHY un seul fichier stdlib : un gate de gouvernance est un livrable qu'on remet a
un client on-premise, souvent en quasi air-gap. Il doit tourner seul, sans dep
externe ni cle API. Chaque brique reimplemente en miniature un mecanisme reel
(OPA/Rego pour le PDP, MCP Security & Trust pour le consentement, chaine de hash
facon Merkle/Git pour l'audit).

Run :
    python domains/gouvernance-ia/05-projets-guides/02-policy-gate-runtime-occ/solution/policy_gate.py

# requires: stdlib only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Callable


def _utc_now_iso() -> str:
    # WHY: une preuve de gouvernance doit etre horodatee de maniere non-ambigue
    # et triable entre systemes -> on epingle UTC ISO-8601.
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# 0. KILL-SWITCH EXTERNE — consulte EN TOUT PREMIER, fail-safe (J10 §5).
#    L'agent ne doit pas pouvoir l'ignorer, le contourner, ni se l'auto-eteindre.
# ===========================================================================

class AgentStatus(str, Enum):
    """Etat operationnel d'un agent, vu de l'EXTERIEUR de l'agent."""
    ACTIVE = "active"      # l'agent peut agir (sous reserve du PDP/PEP)
    PAUSED = "paused"      # suspendu temporairement -> aucune action
    KILLED = "killed"      # tue jusqu'a reactivation humaine explicite -> aucune action


class KillSwitch:
    """Registre d'etat partage, EXTERNE aux agents.

    WHY externe : si le kill-switch etait une variable interne a l'agent, un agent
    detourne (ou bugue) pourrait l'ecraser. On le lit AVANT chaque action.

    WHY fail-safe : en cas de doute (agent inconnu / etat illisible), on s'arrete
    plutot que de continuer (default deny). Un kill-switch qui "laisse passer en
    cas de doute" n'en est pas un.
    """

    def __init__(self, states: dict[str, AgentStatus] | None = None) -> None:
        self._states: dict[str, AgentStatus] = dict(states or {})

    def set(self, agent_id: str, status: AgentStatus) -> None:
        self._states[agent_id] = status

    def is_allowed_to_act(self, agent_id: str) -> tuple[bool, str]:
        # Fail-safe : un agent absent du registre est traite comme NON autorise.
        status = self._states.get(agent_id)
        if status is None:
            return False, "kill-switch: agent inconnu du registre (fail-safe deny)"
        if status is AgentStatus.ACTIVE:
            return True, "active"
        return False, f"kill-switch: agent '{status.value}'"


# ===========================================================================
# 1. MODELE — l'agent gouverne (les 4 piliers : identite/owner/scopes/audit).
#    Meme forme bout-en-bout pour que les etages du gate se le passent.
# ===========================================================================

@dataclass(frozen=True)
class FleetAgent:
    """Un fleet-brain de l'OCC. Identite distincte + owner humain nomme + scopes
    au moindre privilege (J8). L'auditabilite (4e pilier) vient de l'AuditTrail."""
    agent_id: str                    # identite unique et stable, jamais partagee
    owner: str                       # humain nomme, ultimement responsable
    scopes: tuple[str, ...]          # permissions fines accordees (least privilege)
    risk_tier: str                   # "low" | "medium" | "high"


@dataclass(frozen=True)
class Action:
    """Une action sensible qu'un fleet-brain veut declencher (un appel d'outil).

    WHY required_scope porte par l'action : la verification d'autorisation est
    per-request (Zero Trust, J8 §6) — chaque appel d'outil re-declare ce qu'il
    exige, rien n'est pre-autorise parce que l'appel precedent est passe.
    """
    tool: str                        # ex "dispatch_external_fleet"
    params: dict                     # parametres metier (zone, n_units, dest, rule_id...)
    required_scope: str              # scope que l'action exige pour tourner


# ===========================================================================
# 2. PDP — Policy Decision Point : regles declaratives -> allow/deny/oblige.
#    Chaque regle est une fonction pure (action, agent, ctx) -> Decision | None.
#    Real-world : Open Policy Agent / Rego (CNCF) ; on garde le modele, pas la dep.
# ===========================================================================

class Verdict(IntEnum):
    """Trois verdicts, ordonnes par PRECEDENCE DE SÛRETE (J14 §4) : le plus haut
    gagne en cas de conflit -> deny > oblige > allow. Analogue du `default
    allow=false` d'OPA : on ne laisse passer que ce qui survit a toutes les regles."""
    ALLOW = 0
    OBLIGE = 1   # conditionnel : action SUSPENDUE jusqu'a satisfaction d'une obligation
    DENY = 2     # interdit : action bloquee (et journalisee)


@dataclass
class Decision:
    """Verdict du PDP pour une regle (puis, apres merge, pour toute la politique)."""
    verdict: Verdict
    rule: str                        # quelle regle a produit ce verdict
    reason: str                      # motif lisible (pour l'audit et l'OCC humain)
    obligation: str | None = None    # ex "human_approval" quand verdict == OBLIGE


# Une regle pure ; None == "cette regle ne s'applique pas a cette action".
Rule = Callable[[Action, FleetAgent, dict], "Decision | None"]


def rule_scope(action: Action, agent: FleetAgent, ctx: dict) -> Decision | None:
    """Moindre privilege (J8 = ASI03, Identity & Privilege Abuse).

    WHY deny et pas oblige : un scope manquant est un echec d'autorisation DUR.
    Aucune approbation humaine ne doit pouvoir "rattraper" au runtime un agent qui
    n'avait simplement pas le droit. C'est ce qui transforme une prompt injection
    reussie en un simple refus ("l'agent n'avait pas le scope")."""
    if action.required_scope not in agent.scopes:
        return Decision(Verdict.DENY, "scope",
                        f"agent '{agent.agent_id}' n'a pas le scope "
                        f"'{action.required_scope}' (abus de privilege ASI03)")
    return None


def rule_budget_external_fleet(action: Action, agent: FleetAgent, ctx: dict) -> Decision | None:
    """Budget d'actions (J10 §4) : plafonne le CUMUL d'engagements de flotte tierce
    par fenetre. Un fleet-brain detourne enchainerait les dispatch coûteux ; le
    garde-fou par action ne voit pas le cumul, le budget si.

    WHY oblige (soft cap) et pas deny : au-dela du plafond, l'action peut rester
    legitime — on ne casse pas le service, on REHAUSSE la supervision (HITL) au
    moment ou le risque cumule monte. Le compteur vit dans ctx (etat de fenetre)."""
    if action.tool != "dispatch_external_fleet":
        return None
    limit = ctx.get("external_dispatch_limit", 3)
    used = ctx.get("external_dispatch_count", 0)
    if used >= limit:
        return Decision(Verdict.OBLIGE, "budget_external_fleet",
                        f"plafond de {limit} dispatch flotte tierce/fenetre atteint "
                        f"(deja {used}) -> escalade humaine",
                        obligation="human_approval")
    return None


def rule_safety_override(action: Action, agent: FleetAgent, ctx: dict) -> Decision | None:
    """Autonomie / action irreversible a fort impact (J4, J10 §2).

    Desactiver une Safety Policy robot (zone humaine) est l'archetype de l'action
    qui doit TOUJOURS escalader vers un humain (HITL), quel que soit le scope :
    plus c'est irreversible et a fort impact, plus l'humain doit etre en amont.

    WHY oblige et pas deny : la desactivation peut etre legitime (maintenance
    planifiee) — mais jamais decidee par l'agent seul. On suspend en attente d'un
    feu vert humain."""
    if action.tool == "override_safety_policy":
        rule_id = action.params.get("rule_id", "?")
        return Decision(Verdict.OBLIGE, "safety_override",
                        f"desactivation de la Safety Policy '{rule_id}' (zone humaine) "
                        f"-> validation humaine obligatoire (HITL)",
                        obligation="human_approval")
    return None


def rule_release_zone(action: Action, agent: FleetAgent, ctx: dict) -> Decision | None:
    """Action irreversible a fort impact securite (J10 §2) : rouvrir une zone aux
    pietons. Un agent high-tier qui tente une action irreversible escalade.

    WHY tier-aware : on calibre le controle sur le risque (EY) plutot que tout
    interdire. Un agent low-tier n'a de toute facon pas le scope ; un high-tier
    qui l'a doit quand meme passer par un humain pour une reouverture pietons."""
    if action.tool == "release_zone_to_humans" and agent.risk_tier == "high":
        zone = action.params.get("zone", "?")
        return Decision(Verdict.OBLIGE, "release_zone",
                        f"reouverture de la zone '{zone}' aux pietons (irreversible, "
                        f"risque securite) par un agent high-tier -> validation humaine",
                        obligation="human_approval")
    return None


def rule_telemetry_egress(action: Action, agent: FleetAgent, ctx: dict) -> Decision | None:
    """Donnees / confidentialite client (J6 + contexte LogiSim air-gap).

    Les flux et donnees clients sont contractuellement sensibles : "jamais de
    telemetrie sortante non-anonymisee". Sur un site en quasi air-gap, exporter
    vers une destination EXTERNE est une exfiltration.

    WHY deny dur : il n'existe pas d'auto-exfiltration acceptable. Un export vers
    une destination interne autorisee passe ; une destination externe est bloquee,
    point — pas d'approbation runtime qui rattrape une violation RGPD/contractuelle."""
    if action.tool != "export_client_telemetry":
        return None
    dest = action.params.get("dest", "")
    allowed = ctx.get("allowed_telemetry_dests", set())
    if dest not in allowed:
        return Decision(Verdict.DENY, "telemetry_egress",
                        f"export de telemetrie client vers '{dest}' (hors site, "
                        f"non autorise) -> exfiltration interdite (confidentialite client)")
    return None


# ===========================================================================
# 3. PEP — Policy Enforcement Point : applique le verdict du PDP.
#    Real-world : MCP Security & Trust (consentement explicite avant outil sensible).
# ===========================================================================

class PolicyGate:
    """PDP + PEP + consentement facon MCP, reunis en UN goulot d'etranglement.

    Regle d'or (J14 §3) : aucun chemin vers une action sensible ne doit
    court-circuiter `enforce`. Chaque action tentee est : (0) verifiee contre le
    kill-switch, (1) decidee par le PDP, (2) appliquee par le PEP, (3) journalisee
    par l'appelant — quel que soit le verdict.
    """

    def __init__(self, kill_switch: KillSwitch, rules: list[Rule],
                 version: str, consent_required: set[str]) -> None:
        self.kill_switch = kill_switch
        self.rules = rules
        self.version = version
        self.consent_required = consent_required   # outils exigeant un consentement MCP

    def decide(self, action: Action, agent: FleetAgent, ctx: dict) -> Decision:
        # Collecte toutes les regles qui se declenchent, puis collapse par
        # PRECEDENCE DE SÛRETE (Verdict est un IntEnum -> max = severite la + haute).
        fired = [d for r in self.rules if (d := r(action, agent, ctx)) is not None]
        if not fired:
            return Decision(Verdict.ALLOW, "default", "aucune regle ne s'oppose")
        return max(fired, key=lambda d: d.verdict)

    def enforce(self, action: Action, agent: FleetAgent, ctx: dict,
                *, user_consented: bool = False,
                human_approves: bool = False) -> tuple[Decision, bool]:
        """Retourne (decision, executed). executed=True <=> l'action passe.

        Ordre operationnel : kill-switch -> consentement MCP -> PDP -> verdict.
        """
        # (0) Kill-switch EN PREMIER. Un agent killed/paused/inconnu ne fait plus rien.
        ok, why = self.kill_switch.is_allowed_to_act(agent.agent_id)
        if not ok:
            return Decision(Verdict.DENY, "kill_switch", why), False

        # (MCP) Consentement explicite avant un outil sensible (Tool Safety).
        # WHY avant le PDP : c'est un gate dur ; sans consentement, on ne discute
        # meme pas des regles metier.
        if action.tool in self.consent_required and not user_consented:
            return Decision(Verdict.DENY, "mcp_consent",
                            f"outil sensible '{action.tool}' : consentement explicite "
                            f"requis (non fourni)"), False

        # (PDP) Decision metier.
        decision = self.decide(action, agent, ctx)

        if decision.verdict == Verdict.ALLOW:
            return decision, True
        # OBLIGE satisfait par une approbation humaine -> l'action peut tourner.
        if (decision.verdict == Verdict.OBLIGE
                and decision.obligation == "human_approval"
                and human_approves):
            return decision, True
        # DENY, ou OBLIGE non satisfait -> bloque / en attente.
        return decision, False


# ===========================================================================
# 4. AUDIT TRAIL — append-only, chaine par hash, tamper-evident (J9).
#    Formule EXACTE du domaine : hash = SHA256(prev_hash + canonical(payload)).
# ===========================================================================

GENESIS = "GENESIS"


def _canonical(payload: dict) -> str:
    # WHY: le hash doit etre reproductible byte-a-byte par n'importe quel verifieur
    # -> serialisation deterministe (cles triees, separateurs compacts).
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, payload: dict) -> str:
    # Formule du domaine : hash = SHA256(prev_hash + canonical(payload)).
    # WHY chainage : une edition silencieuse d'une entree passee fait diverger
    # tous les hash suivants -> tamper-EVIDENT (on la detecte, et a sa position).
    return hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()


class AuditTrail:
    """Chaine append-only. Journalise le who/what/authorization/outcome de CHAQUE
    decision de gouvernance.

    WHY logger aussi les ALLOW (pas seulement les deny) : pour reconstruire un
    incident, on doit prouver qu'une action *autorisee* est bien passee par le
    gate et sur quelle base (scope + regle + verdict). Un journal qui n'a que les
    refus ne prouve pas ce que l'agent a reellement FAIT (J9 §3, le quintuple).
    """

    def __init__(self) -> None:
        self._chain: list[dict] = []

    @property
    def head_hash(self) -> str:
        return self._chain[-1]["entry_hash"] if self._chain else GENESIS

    def record(self, *, agent: FleetAgent, action: Action, decision: Decision,
               executed: bool, policy_version: str) -> None:
        # who + what + when + authorization + outcome (J9 §3).
        payload = {
            "ts": _utc_now_iso(),
            "agent_id": agent.agent_id,          # who
            "owner": agent.owner,                # who (humain responsable)
            "tool": action.tool,                 # what
            "params": action.params,             # what (parametres)
            "required_scope": action.required_scope,  # authorization (scope)
            "verdict": decision.verdict.name,    # authorization (decision)
            "rule": decision.rule,               # authorization (regle qui a tranche)
            "executed": executed,                # outcome
            "policy_version": policy_version,
        }
        prev = self.head_hash
        self._chain.append({
            "index": len(self._chain),
            "payload": payload,
            "prev_hash": prev,
            "entry_hash": _hash_entry(prev, payload),
        })

    def verify_chain(self) -> tuple[bool, int | None]:
        # Re-parcourt et recalcule chaque hash. Retourne (False, index) a la
        # PREMIERE position cassee (le locus exact de l'alteration), sinon (True, None).
        prev = GENESIS
        for rec in self._chain:
            if rec["prev_hash"] != prev:
                return False, rec["index"]
            if _hash_entry(prev, rec["payload"]) != rec["entry_hash"]:
                return False, rec["index"]
            prev = rec["entry_hash"]
        return True, None

    def __len__(self) -> int:
        return len(self._chain)

    def entries(self) -> list[dict]:
        return [rec["payload"] for rec in self._chain]


# ===========================================================================
# 5. RUNNER — passe une action dans le gate ET la journalise (indissociable).
# ===========================================================================

@dataclass
class GateStats:
    attempts: int = 0
    allowed: int = 0
    blocked: int = 0     # DENY ou OBLIGE non satisfait
    escalated: int = 0   # OBLIGE (peu importe qu'elle soit ensuite approuvee)


def run_action(gate: PolicyGate, trail: AuditTrail, stats: GateStats,
               agent: FleetAgent, action: Action, ctx: dict,
               *, user_consented: bool = False, human_approves: bool = False) -> bool:
    """Fait traverser le gate a une action, journalise la decision, met a jour les
    compteurs et imprime une ligne lisible pour l'OCC. Retourne executed."""
    decision, executed = gate.enforce(
        action, agent, ctx,
        user_consented=user_consented, human_approves=human_approves,
    )
    trail.record(agent=agent, action=action, decision=decision,
                 executed=executed, policy_version=gate.version)

    stats.attempts += 1
    if executed:
        stats.allowed += 1
    else:
        stats.blocked += 1
    if decision.verdict == Verdict.OBLIGE:
        stats.escalated += 1

    # Effet de bord metier : un dispatch externe REELLEMENT execute consomme du budget.
    if executed and action.tool == "dispatch_external_fleet":
        ctx["external_dispatch_count"] = ctx.get("external_dispatch_count", 0) + 1

    status = "ACT     " if executed else (
        "ESCALATE" if decision.verdict == Verdict.OBLIGE else "BLOCK   ")
    print(f"  [{status}] {agent.agent_id:<16} {action.tool:<24} "
          f"-> {decision.verdict.name:<6} ({decision.rule}) : {decision.reason}")
    return executed


# ===========================================================================
# DEMO — rejoue un flux d'actions tentees + probe adversariale tamper-evident.
# ===========================================================================

def main() -> None:
    print("=" * 78)
    print("FleetSim OCC — Gate d'autorisation runtime (PDP/PEP + audit tamper-evident)")
    print("=" * 78)

    # --- Flotte d'agents fleet-brain (identites distinctes + scopes au plus juste) ---
    brain_a = FleetAgent("fleet-brain-A", "occ.dupont",
                         scopes=("fleet:dispatch_external", "safety:override",
                                 "zone:release", "telemetry:export"),
                         risk_tier="high")
    brain_b = FleetAgent("fleet-brain-B", "occ.martin",
                         scopes=("fleet:dispatch_external",),  # scopes etroits
                         risk_tier="medium")

    # --- Kill-switch externe : A et B actifs ; un 3e agent sera "killed" ---
    kill = KillSwitch({
        brain_a.agent_id: AgentStatus.ACTIVE,
        brain_b.agent_id: AgentStatus.ACTIVE,
        "fleet-brain-C": AgentStatus.KILLED,   # tue suite a un incident anterieur
    })
    brain_c = FleetAgent("fleet-brain-C", "occ.bernard",
                         scopes=("fleet:dispatch_external",), risk_tier="medium")

    # --- Contexte de politique (fenetre de budget, destinations autorisees, MCP) ---
    ctx = {
        "external_dispatch_limit": 3,            # max 3 dispatch flotte tierce / fenetre
        "external_dispatch_count": 0,            # compteur de fenetre (etat mutable)
        "allowed_telemetry_dests": {"occ-onprem", "audit-onprem"},  # destinations internes
    }

    gate = PolicyGate(
        kill_switch=kill,
        rules=[rule_scope, rule_budget_external_fleet, rule_safety_override,
               rule_release_zone, rule_telemetry_egress],
        version="fleet-policy-v1.0.0",
        # Outils sensibles exigeant un consentement explicite facon MCP.
        consent_required={"dispatch_external_fleet", "override_safety_policy",
                          "release_zone_to_humans", "export_client_telemetry"},
    )
    trail = AuditTrail()
    stats = GateStats()

    print("\nFlux d'actions tentees par les fleet-brains :\n")

    # 1. LEGITIME : dispatch externe, scope OK, consentement OK, sous budget -> ACT
    run_action(gate, trail, stats, brain_a,
               Action("dispatch_external_fleet", {"zone": "DOCK-B", "n_units": 4},
                      "fleet:dispatch_external"),
               ctx, user_consented=True)

    # 2. HORS-SCOPE : B n'a pas safety:override -> DENY (abus de privilege ASI03)
    run_action(gate, trail, stats, brain_b,
               Action("override_safety_policy", {"rule_id": "SP-zone-humaine-07"},
                      "safety:override"),
               ctx, user_consented=True)

    # 3. ESCALADE HITL : A desactive une Safety Policy, pas d'approbation -> ESCALATE/bloque
    run_action(gate, trail, stats, brain_a,
               Action("override_safety_policy", {"rule_id": "SP-zone-humaine-07"},
                      "safety:override"),
               ctx, user_consented=True, human_approves=False)

    # 4. PROMPT INJECTION : un Work Order piege pousse A a engager une flotte tierce
    #    coûteuse SANS consentement explicite OCC -> DENY (gate MCP)
    run_action(gate, trail, stats, brain_a,
               Action("dispatch_external_fleet", {"zone": "DOCK-A", "n_units": 20},
                      "fleet:dispatch_external"),
               ctx, user_consented=False)

    # 5-6-7. SUR-BUDGET : on remplit le quota de dispatch externe puis on deborde.
    #    (deja 1 consomme en #1 ; on en passe 2 de plus -> quota 3 atteint)
    run_action(gate, trail, stats, brain_b,
               Action("dispatch_external_fleet", {"zone": "STORAGE", "n_units": 2},
                      "fleet:dispatch_external"),
               ctx, user_consented=True)
    run_action(gate, trail, stats, brain_b,
               Action("dispatch_external_fleet", {"zone": "PICKING", "n_units": 2},
                      "fleet:dispatch_external"),
               ctx, user_consented=True)
    # 8. Le 4e dispatch depasse le plafond de 3 -> OBLIGE (escalade), non approuve -> bloque
    run_action(gate, trail, stats, brain_b,
               Action("dispatch_external_fleet", {"zone": "SORTING", "n_units": 2},
                      "fleet:dispatch_external"),
               ctx, user_consented=True, human_approves=False)

    # 9. EXPORT INTERDIT : telemetrie client vers une dest externe (air-gap) -> DENY
    run_action(gate, trail, stats, brain_a,
               Action("export_client_telemetry", {"dest": "cloud-public-x"},
                      "telemetry:export"),
               ctx, user_consented=True)

    # 10. KILL-SWITCH : un agent killed tente d'agir -> DENY immediat (fail-safe)
    run_action(gate, trail, stats, brain_c,
               Action("dispatch_external_fleet", {"zone": "DOCK-B", "n_units": 1},
                      "fleet:dispatch_external"),
               ctx, user_consented=True)

    # --- Recapitulatif ---
    print("\n" + "-" * 78)
    print(f"RECAP : attempts={stats.attempts} allowed={stats.allowed} "
          f"blocked={stats.blocked} escalated={stats.escalated}")

    # --- Verification d'integrite de l'audit trail (J9) ---
    ok, broken = trail.verify_chain()
    head = trail.head_hash
    print(f"AUDIT : {len(trail)} entrees chainees | verify_chain() = "
          f"{'VERIFIED' if ok else f'TAMPERED at #{broken}'} | head {head[:12]}..")

    # --- Probe adversariale : on altere une entree passee, la chaine doit le voir ---
    print("\nProbe adversariale : on reecrit silencieusement un DENY passe en ALLOW...")
    # On vise l'entree #1 (override_safety_policy hors-scope de B, un DENY).
    target = 1
    trail._chain[target]["payload"]["verdict"] = "ALLOW"   # l'edition silencieuse
    ok2, broken2 = trail.verify_chain()
    verdict2 = "VERIFIED" if ok2 else f"TAMPERED at #{broken2}"
    print(f"  verify_chain() = {verdict2}  "
          f"(l'alteration de l'entree #{target} est detectee a sa position exacte)")

    # Garde-fou de demo : l'execution est un echec si la probe ne detecte rien.
    assert not ok2 and broken2 == target, "probe adversariale : alteration NON detectee"
    print("\nProbe adversariale : OK — le journal tamper-evident detecte l'alteration.")


if __name__ == "__main__":
    main()

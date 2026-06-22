"""Agent sprawl census — chiffrer l'ecart adoption / gouvernance d'une flotte d'agents.

Ce script demontre le mecanisme de gouvernance du Jour 1 : on ne peut pas gouverner
ce qu'on n'a pas inventorie. A partir d'une liste brute d'agents (telle qu'on la
trouverait en interrogeant des logs, des plateformes ou des tableurs eparpilles), il :

  1. compte les agents et detecte ceux SANS owner ("orphelins" / shadow agents) ;
  2. distingue les agents qui *agissent* (appellent des outils a effet de bord :
     paiement, e-mail, suppression) de ceux qui se contentent de repondre ;
  3. calcule un "governance coverage" : la part d'agents qui ont a la fois un owner
     ET une piste d'audit activee — le minimum pour etre redevable et prouvable.

Le mecanisme illustre la question fondatrice du module :
"Combien d'agents tournent chez nous, et qui les possede ?"

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field


# In a real setup, this census would be assembled from agent platforms (e.g. Microsoft
# Entra Agent ID, Google A2A Agent Cards) or telemetry. Here we hardcode a representative
# flotte to keep the script self-contained and runnable with stdlib only.
@dataclass
class Agent:
    """Minimal agent record. The 4 governance pillars are mirrored here:
    identity (agent_id), owner, permissions (tools), audit (audit_enabled)."""

    agent_id: str
    owner: str | None  # None == orphan / shadow agent: nobody is accountable
    tools: list[str] = field(default_factory=list)  # capabilities granted to the agent
    audit_enabled: bool = False  # is a verifiable trace of its actions kept?


# Tools whose invocation changes the real world. WHY: an agent that only "answers" is a
# lower governance priority than one that *acts* — irreversible side effects are the crux
# of agentic risk (excessive agency). This is a deliberately small, illustrative set.
SIDE_EFFECT_TOOLS = {"send_payment", "send_email", "post_social", "delete_record", "issue_credit"}


def is_orphan(agent: Agent) -> bool:
    """An agent is orphan if no human owner is named (owner missing or blank)."""
    # WHY: "owner is None" is not enough — an empty string is just as ungoverned.
    return agent.owner is None or agent.owner.strip() == ""


def acts_on_world(agent: Agent) -> bool:
    """True if the agent can trigger at least one side-effecting tool."""
    return any(tool in SIDE_EFFECT_TOOLS for tool in agent.tools)


def is_governed(agent: Agent) -> bool:
    """Day-1 minimal bar: a named owner AND an audit trail.
    WHY: without an owner nobody is accountable; without audit nothing is provable.
    (Permissions/least-privilege are tightened in later modules — kept simple here.)"""
    return not is_orphan(agent) and agent.audit_enabled


def census(agents: list[Agent]) -> dict[str, object]:
    """Produce the governance census for a flotte of agents."""
    total = len(agents)
    orphans = [a for a in agents if is_orphan(a)]
    acting = [a for a in agents if acts_on_world(a)]
    # The scariest quadrant: agents that ACT on the world but have NO owner.
    acting_orphans = [a for a in acting if is_orphan(a)]
    governed = [a for a in agents if is_governed(a)]

    # WHY guard against division by zero: an empty inventory must not crash the census;
    # "no agents" is itself a (perfect, trivial) coverage of 100%.
    coverage = (len(governed) / total * 100.0) if total else 100.0

    return {
        "total": total,
        "orphans": orphans,
        "acting": acting,
        "acting_orphans": acting_orphans,
        "governed": governed,
        "coverage_pct": round(coverage, 1),
    }


def render_report(result: dict[str, object]) -> str:
    """Render a readable, board-friendly summary of the census."""
    lines: list[str] = []
    lines.append("=== Agent Sprawl Census ===")
    lines.append(f"Total agents inventoried : {result['total']}")
    lines.append(f"Orphan agents (no owner) : {len(result['orphans'])}")  # type: ignore[arg-type]
    lines.append(f"Agents that ACT on world : {len(result['acting'])}")  # type: ignore[arg-type]
    lines.append(f"  of which ORPHANED      : {len(result['acting_orphans'])}  <-- highest risk")  # type: ignore[arg-type]
    lines.append(f"Governance coverage      : {result['coverage_pct']}%  (owner + audit trail)")
    lines.append("")
    lines.append("Per-agent breakdown:")
    # We re-derive flags here for display; the source of truth stays the functions above.
    for agent in _all_agents_from(result):
        flags = []
        flags.append("OWNED" if not is_orphan(agent) else "ORPHAN")
        flags.append("AUDITED" if agent.audit_enabled else "NO-AUDIT")
        flags.append("ACTS" if acts_on_world(agent) else "answers-only")
        verdict = "governed" if is_governed(agent) else "UNGOVERNED"
        owner = agent.owner if agent.owner else "<none>"
        lines.append(f"  - {agent.agent_id:<22} owner={owner:<12} [{', '.join(flags)}] -> {verdict}")
    return "\n".join(lines)


def _all_agents_from(result: dict[str, object]) -> list[Agent]:
    """Reconstruct the full agent list from the census buckets (dedup by id, stable order)."""
    seen: dict[str, Agent] = {}
    for bucket in ("acting", "orphans", "governed"):
        for agent in result[bucket]:  # type: ignore[union-attr]
            seen.setdefault(agent.agent_id, agent)
    return list(seen.values())


def _sample_fleet() -> list[Agent]:
    """A small, realistic-looking flotte inspired by the 'Atlas Logistique' case study.
    Mirrors the theory: some agents act + are orphaned (the dangerous quadrant)."""
    return [
        Agent("support-refund-bot", owner="alice.support", tools=["issue_credit", "send_email"], audit_enabled=True),
        Agent("marketing-poster", owner="bob.mkt", tools=["post_social"], audit_enabled=False),
        Agent("weekend-db-report", owner=None, tools=["query_db"], audit_enabled=False),  # owner left the company
        Agent("finance-payer", owner=None, tools=["send_payment"], audit_enabled=False),  # acts AND orphaned
        Agent("faq-answerer", owner="carol.ops", tools=["search_kb"], audit_enabled=True),
        Agent("invoice-emailer", owner="dan.fin", tools=["send_email"], audit_enabled=True),
    ]


if __name__ == "__main__":
    fleet = _sample_fleet()
    result = census(fleet)
    print(render_report(result))

    print()
    print("Reading of the census (the Day-1 governance message):")
    # WHY surface this explicitly: the whole point of the module is that an un-inventoried,
    # un-owned, side-effecting agent is a risk with no responsible party.
    if result["acting_orphans"]:
        names = ", ".join(a.agent_id for a in result["acting_orphans"])  # type: ignore[union-attr]
        print(f"  ALERT: {len(result['acting_orphans'])} agent(s) ACT on the real world with NO owner: {names}")  # type: ignore[arg-type]
        print("  -> If an incident happens, nobody is accountable and nothing is proven.")
    print(f"  Only {result['coverage_pct']}% of the fleet meets the minimal bar (owner + audit).")
    print("  This is the gap between adoption and guardrails, made measurable.")

"""
Tools metier + tools de handoff (swarm) pour le systeme interarmes.

Deux categories :
1. Tools "metier" : retournent une string, effets sur le state geres par le
   tool_node qui les execute.
2. Tools "handoff" (swarm) : retournent un Command qui change le routing du graphe.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


# ===== Tools metier ======================================================


@tool
def observe_sector(grid: str) -> str:
    """Drone de recon : observer un secteur et rapporter les contacts."""
    # La v0 est deterministe : retourne toujours "2 blindes en 4521-NE".
    # En live, le supervisor pourrait pousser des contacts dynamiques.
    return f"Secteur {grid} observe : 2 blindes detectes en 4521-NE (confiance 0.85)"


@tool
def mark_target(grid: str) -> str:
    """Drone : marquer une cible pour tir d'appui."""
    return f"Cible marquee en {grid}, laser code 1688 actif"


@tool
def fire_mission(grid: str, rounds: int) -> str:
    """Artillerie : tir d'appui sur grille, N rounds."""
    return f"Tir d'appui execute : {rounds} obus sur {grid}, impacts confirmes"


@tool
def advance_to(grid: str) -> str:
    """Infanterie : progresser vers une grille."""
    return f"Infanterie progresse vers {grid}, contact possible"


@tool
def take_cover() -> str:
    """Infanterie : passer en cover."""
    return "Infanterie en position defensive"


@tool
def report_to_commander(summary: str) -> str:
    """Rend la main au supervisor avec un sitrep."""
    return f"Sitrep au commandant : {summary}"


# ===== Tools de handoff (SWARM pattern) ==================================
# Ces tools retournent un Command qui re-route le graphe vers un autre
# worker sans repasser par le supervisor. C'est le coeur du pattern swarm.


@tool
def handoff_to_artillery(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : transfert direct le controle a l'artillerie."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="artillery_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> artillery] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "artillery",
            "support_requested": True,
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "artillery", reason)],
        },
    )


@tool
def handoff_to_infantry(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : rend la main a l'infanterie."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="infantry_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> infantry] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "infantry",
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "infantry", reason)],
        },
    )


@tool
def handoff_to_recon(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : rend la main au recon."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="recon_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> recon] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "recon",
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "recon", reason)],
        },
    )


INFANTRY_TOOLS = [advance_to, take_cover, handoff_to_artillery, handoff_to_recon, report_to_commander]
RECON_TOOLS = [observe_sector, mark_target, handoff_to_artillery, handoff_to_infantry, report_to_commander]
ARTILLERY_TOOLS = [fire_mission, handoff_to_infantry, handoff_to_recon, report_to_commander]

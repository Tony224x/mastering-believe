"""
Tools metier + tools de handoff (swarm) pour le systeme multi-flotte LogiSim.

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
def scan_sector(zone: str) -> str:
    """Drone d'inventaire : scanner une zone et reporter les contacts (colis, palettes)."""
    # V0 deterministe : retourne toujours "2 palettes lourdes en B-12-NE".
    # En live, le superviseur pourrait pousser des contacts dynamiques.
    return f"Zone {zone} scannee : 2 palettes lourdes detectees en B-12-NE (confiance 0.85)"


@tool
def mark_parcel(zone: str) -> str:
    """Drone : marquer un colis comme prioritaire pour pickup AGV."""
    return f"Colis marque en {zone}, beacon code 1688 actif"


@tool
def dispatch_pickup(zone: str, units: int) -> str:
    """AGV : envoyer N AGV pour un pickup en zone donnee."""
    return f"Pickup execute : {units} AGV dispatches en {zone}, colis embarques"


@tool
def move_to(zone: str) -> str:
    """Sorting : deplacer la flotte de tri vers une zone."""
    return f"Flotte de tri en mouvement vers {zone}"


@tool
def hold_position() -> str:
    """Sorting : tenir la position courante."""
    return "Flotte de tri en position de stabilisation"


@tool
def report_to_coordinator(summary: str) -> str:
    """Rend la main au supervisor avec un report d'etat."""
    return f"Report au coordinator : {summary}"


# ===== Tools de handoff (SWARM pattern) ==================================
# Ces tools retournent un Command qui re-route le graphe vers un autre
# worker sans repasser par le supervisor. C'est le coeur du pattern swarm.


@tool
def handoff_to_agv(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : transfere directement le controle a l'AGV."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="agv_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> agv] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "agv",
            "pickup_requested": True,
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "agv", reason)],
        },
    )


@tool
def handoff_to_sorting(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : rend la main au sorting."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="sorting_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> sorting] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "sorting",
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "sorting", reason)],
        },
    )


@tool
def handoff_to_drone(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """SWARM handoff : rend la main au drone."""
    from_agent = state.get("active_agent", "unknown")
    return Command(
        goto="drone_lead",
        update={
            "messages": [
                ToolMessage(
                    content=f"[SWARM HANDOFF {from_agent} -> drone] {reason}",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": "drone",
            "handoff_log": state.get("handoff_log", []) + [(from_agent, "drone", reason)],
        },
    )


SORTING_TOOLS = [move_to, hold_position, handoff_to_agv, handoff_to_drone, report_to_coordinator]
DRONE_TOOLS = [scan_sector, mark_parcel, handoff_to_agv, handoff_to_sorting, report_to_coordinator]
AGV_TOOLS = [dispatch_pickup, handoff_to_sorting, handoff_to_drone, report_to_coordinator]

"""
Assemblage du graphe LangGraph : supervisor + 3 workers.

Topologie :
- START -> supervisor
- supervisor -> routing vers worker ou FINISH
- worker -> si handoff (swarm), le Command du tool redirige ; sinon retour supervisor
- finish -> END

Cle : le tool_node execute les tools metier ET les tools de handoff. Les tools
de handoff retournent des Command(goto=...) que LangGraph interprete comme
"passe au node X sans repasser par le supervisor".
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents import (
    agv_node,
    sorting_node,
    drone_node,
    supervisor_node,
)
from state import ShiftState
from tools import (
    AGV_TOOLS,
    SORTING_TOOLS,
    DRONE_TOOLS,
)


# Toolnodes pour chaque worker — execute les tools du worker
sorting_tools_node = ToolNode(SORTING_TOOLS)
drone_tools_node = ToolNode(DRONE_TOOLS)
agv_tools_node = ToolNode(AGV_TOOLS)


def route_from_supervisor(state: ShiftState) -> str:
    next_agent = state.get("active_agent", "FINISH").lower()
    if next_agent == "drone":
        return "drone_lead"
    if next_agent == "agv":
        return "agv_lead"
    if next_agent == "sorting":
        return "sorting_lead"
    return END


def route_from_worker(state: ShiftState, worker_tools_node: str, fallback: str = "supervisor") -> str:
    """Decide si le dernier AIMessage du worker appelle un tool (-> tools_node) ou non (-> supervisor)."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return worker_tools_node
    return fallback


def route_after_tools(state: ShiftState) -> str:
    """Apres execution des tools, on regarde le dernier message :
    - si c'etait un handoff (ToolMessage contenant "[SWARM HANDOFF"), le state.active_agent a deja ete
      mis a jour et le graph routing va la. Sinon, retour au worker qui etait actif.
    """
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and "[SWARM HANDOFF" in last.content:
        # Le Command du tool a deja set active_agent et goto. Rien a faire ici.
        # Mais LangGraph va utiliser le goto du Command, pas ce retour.
        return state["active_agent"].lower() + "_lead"

    # Pas de handoff : le worker a execute un tool metier, il reprend la main
    active = state.get("active_agent", "supervisor")
    if active == "supervisor":
        return "supervisor"
    return f"{active}_lead"


def build_graph():
    g = StateGraph(ShiftState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("sorting_lead", sorting_node)
    g.add_node("drone_lead", drone_node)
    g.add_node("agv_lead", agv_node)
    g.add_node("sorting_tools", sorting_tools_node)
    g.add_node("drone_tools", drone_tools_node)
    g.add_node("agv_tools", agv_tools_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "sorting_lead": "sorting_lead",
            "drone_lead": "drone_lead",
            "agv_lead": "agv_lead",
            END: END,
        },
    )

    # Chaque worker : si l'AIMessage a des tool_calls -> tools node, sinon retour supervisor
    g.add_conditional_edges(
        "sorting_lead",
        lambda s: route_from_worker(s, "sorting_tools"),
        {"sorting_tools": "sorting_tools", "supervisor": "supervisor"},
    )
    g.add_conditional_edges(
        "drone_lead",
        lambda s: route_from_worker(s, "drone_tools"),
        {"drone_tools": "drone_tools", "supervisor": "supervisor"},
    )
    g.add_conditional_edges(
        "agv_lead",
        lambda s: route_from_worker(s, "agv_tools"),
        {"agv_tools": "agv_tools", "supervisor": "supervisor"},
    )

    # Apres execution des tools, on retourne au worker courant (sauf si swarm handoff
    # via Command, auquel cas LangGraph suit le goto du Command automatiquement).
    g.add_edge("sorting_tools", "supervisor")
    g.add_edge("drone_tools", "supervisor")
    g.add_edge("agv_tools", "supervisor")

    return g.compile()

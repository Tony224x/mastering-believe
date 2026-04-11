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
    artillery_node,
    infantry_node,
    recon_node,
    supervisor_node,
)
from state import BrigadeState
from tools import (
    ARTILLERY_TOOLS,
    INFANTRY_TOOLS,
    RECON_TOOLS,
)


# Toolnodes pour chaque worker — execute les tools du worker
infantry_tools_node = ToolNode(INFANTRY_TOOLS)
recon_tools_node = ToolNode(RECON_TOOLS)
artillery_tools_node = ToolNode(ARTILLERY_TOOLS)


def route_from_supervisor(state: BrigadeState) -> str:
    next_agent = state.get("active_agent", "FINISH").lower()
    if next_agent == "recon":
        return "recon_lead"
    if next_agent == "artillery":
        return "artillery_lead"
    if next_agent == "infantry":
        return "infantry_lead"
    return END


def route_from_worker(state: BrigadeState, worker_tools_node: str, fallback: str = "supervisor") -> str:
    """Decide si le dernier AIMessage du worker appelle un tool (-> tools_node) ou non (-> supervisor)."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return worker_tools_node
    return fallback


def route_after_tools(state: BrigadeState) -> str:
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
    g = StateGraph(BrigadeState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("infantry_lead", infantry_node)
    g.add_node("recon_lead", recon_node)
    g.add_node("artillery_lead", artillery_node)
    g.add_node("infantry_tools", infantry_tools_node)
    g.add_node("recon_tools", recon_tools_node)
    g.add_node("artillery_tools", artillery_tools_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "infantry_lead": "infantry_lead",
            "recon_lead": "recon_lead",
            "artillery_lead": "artillery_lead",
            END: END,
        },
    )

    # Chaque worker : si l'AIMessage a des tool_calls -> tools node, sinon retour supervisor
    g.add_conditional_edges(
        "infantry_lead",
        lambda s: route_from_worker(s, "infantry_tools"),
        {"infantry_tools": "infantry_tools", "supervisor": "supervisor"},
    )
    g.add_conditional_edges(
        "recon_lead",
        lambda s: route_from_worker(s, "recon_tools"),
        {"recon_tools": "recon_tools", "supervisor": "supervisor"},
    )
    g.add_conditional_edges(
        "artillery_lead",
        lambda s: route_from_worker(s, "artillery_tools"),
        {"artillery_tools": "artillery_tools", "supervisor": "supervisor"},
    )

    # Apres execution des tools, on retourne au worker courant (sauf si swarm handoff
    # via Command, auquel cas LangGraph suit le goto du Command automatiquement).
    g.add_edge("infantry_tools", "supervisor")
    g.add_edge("recon_tools", "supervisor")
    g.add_edge("artillery_tools", "supervisor")

    return g.compile()

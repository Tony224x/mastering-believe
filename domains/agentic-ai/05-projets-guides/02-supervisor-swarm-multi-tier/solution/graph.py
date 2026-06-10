"""
Assemblage du graphe LangGraph : supervisor + 3 workers.

Topologie :
- START -> supervisor
- supervisor -> conditional edge vers un worker ou END
- worker -> si l'AIMessage contient des tool_calls -> son tools node ; sinon supervisor
- tools node -> conditional edge (route_after_tools) :
    * handoff swarm  -> le Command(goto=...) du tool redirige vers le worker cible
    * report         -> retour au supervisor
    * tool metier    -> retour au worker courant

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
    """Apres execution des tools, route selon le dernier ToolMessage :
    - handoff swarm : le tool a renvoye un Command(goto=...) qui re-route deja le
      graphe et a mis a jour active_agent. On renvoie la MEME cible : LangGraph
      fusionne les deux destinations identiques en une seule tache (pas de
      double invocation).
    - report au coordinator : retour au supervisor.
    - tool metier classique : le worker actif reprend la main pour decider de
      sa prochaine action (continuer, reporter, ou handoff).
    """
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        if "[SWARM HANDOFF" in last.content:
            # active_agent vient d'etre mis a jour par le Command du tool de
            # handoff -> meme cible que le goto, deduplique par LangGraph.
            return state["active_agent"].lower() + "_lead"
        if "Report au coordinator" in last.content:
            return "supervisor"

    # Pas de handoff ni de report : le worker a execute un tool metier,
    # il reprend la main.
    active = state.get("active_agent", "supervisor")
    if active in ("sorting", "drone", "agv"):
        return f"{active}_lead"
    return "supervisor"


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

    # Apres execution des tools : conditional edge via route_after_tools.
    # - tool metier  -> retour au worker courant (il decide de la suite)
    # - report       -> retour au supervisor
    # - handoff swarm -> le Command(goto=...) du tool re-route deja le graphe ;
    #   route_after_tools renvoie la meme cible, LangGraph deduplique.
    # NB : un add_edge inconditionnel ici serait un bug — il s'executerait EN PLUS
    # du goto du Command (branches paralleles : double invocation des workers,
    # ToolNodes executes a vide).
    after_tools_targets = {
        "supervisor": "supervisor",
        "sorting_lead": "sorting_lead",
        "drone_lead": "drone_lead",
        "agv_lead": "agv_lead",
    }
    g.add_conditional_edges("sorting_tools", route_after_tools, after_tools_targets)
    g.add_conditional_edges("drone_tools", route_after_tools, after_tools_targets)
    g.add_conditional_edges("agv_tools", route_after_tools, after_tools_targets)

    return g.compile()

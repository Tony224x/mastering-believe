"""
Agent commandant de peloton — correction.

Structure :
- State : PlatoonState avec messages + etat simu
- Tools : 5 tools qui modifient le state via Command
- Graph : agent node + tool node, boucle ReAct
- Stop condition : hook dans l'agent node qui check health/steps/mission

Le mode "stub" (sans cle API) permet de tester la structure : on remplace
le LLM par une fonction deterministe qui cycle a travers une sequence
predefinie de tool calls. Utile pour les CI et pour comprendre le graphe
sans payer de tokens.
"""
from __future__ import annotations

import os
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# --------- State ----------------------------------------------------------


class PlatoonState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    position: tuple[float, float]
    health: float
    ammo: float
    last_order: str
    enemy_contact: dict[str, Any] | None
    steps_remaining: int
    mission_done: bool


# --------- Tools ----------------------------------------------------------
# Les tools ici sont "pures" : ils retournent juste une string descriptive.
# La mutation de state est faite dans le graphe via le ToolNode custom.


@tool
def move_to(x: float, y: float) -> str:
    """Deplacer le peloton vers (x, y)."""
    return f"Moved to ({x:.1f}, {y:.1f})"


@tool
def take_cover() -> str:
    """Passer en posture defensive sous couvert."""
    return "Peloton en cover defensif"


@tool
def engage(target_id: str) -> str:
    """Engager une cible identifiee."""
    return f"Engaging target {target_id}"


@tool
def report(summary: str) -> str:
    """Envoyer un sitrep au commandant."""
    return f"Report sent: {summary}"


@tool
def withdraw(direction: str) -> str:
    """Battre en retraite dans une direction (north/south/east/west)."""
    return f"Withdrawing {direction}"


TOOLS = [move_to, take_cover, engage, report, withdraw]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """Tu es le commandant d'un peloton d'infanterie en exercice SWORD.

Contraintes :
- Tu executes l'OPORD recu en entree
- ROE defensif par defaut : engager seulement si menace directe
- Sante < 20% ou munitions < 10% -> withdraw immediat
- Rapport toutes les 15 minutes ou sur changement de situation
- Decisions breves, style militaire, francais

Pour chaque step, tu evalues la situation et tu choisis UNE action via un tool.
Explique en une phrase pourquoi tu choisis cette action avant de l'appeler.
Quand l'objectif est atteint, appelle `report` avec "MISSION COMPLETE" et stop."""


# --------- Agent node -----------------------------------------------------


class _StubLLM:
    """LLM deterministe : cycle sur une sequence de tool calls.

    Attention : ce stub est *module-level* (instance unique) pour que le
    compteur progresse entre les appels successifs de agent_node. Sinon,
    creer un nouveau stub a chaque agent_node() reset le plan et l'agent
    tourne en boucle sur le premier step.
    """

    def __init__(self) -> None:
        self._steps = 0
        self._plan = [
            ("move_to", {"x": 45.0, "y": 21.0}, "Avance vers la ligne de crete."),
            ("take_cover", {}, "Position atteinte, couvert."),
            ("report", {"summary": "Peloton en position 4521, en cover"}, "Sitrep."),
            ("engage", {"target_id": "OPFOR-3"}, "Cible detectee, engagement autorise."),
            ("report", {"summary": "MISSION COMPLETE crete tenue"}, "Mission finie."),
        ]

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        idx = min(self._steps, len(self._plan) - 1)
        name, args, rationale = self._plan[idx]
        self._steps += 1
        tool_call = {
            "name": name,
            "args": args,
            "id": f"stub-{idx}",
            "type": "tool_call",
        }
        return AIMessage(content=rationale, tool_calls=[tool_call])


# Instance module-level — persiste entre les appels a agent_node
_STUB_LLM_SINGLETON = _StubLLM()


def _make_llm():
    """Renvoie un LLM configure, ou un stub deterministe si pas de cle."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
        return llm.bind_tools(TOOLS)
    return _STUB_LLM_SINGLETON


def agent_node(state: PlatoonState) -> dict:
    # Stop conditions dures : on ne laisse pas le LLM decider ici.
    if state["health"] < 0.2 and state.get("last_order", "") != "withdrawing":
        return {
            "messages": [AIMessage(content="Sante critique -> withdraw auto", tool_calls=[
                {"name": "withdraw", "args": {"direction": "south"}, "id": "auto-withdraw", "type": "tool_call"}
            ])],
            "last_order": "withdrawing",
        }
    if state["steps_remaining"] <= 0 or state["mission_done"]:
        return {"messages": [AIMessage(content="Fin de mission")]}

    llm = _make_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    return {
        "messages": [response],
        "steps_remaining": state["steps_remaining"] - 1,
    }


def tool_node(state: PlatoonState) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    tool_messages = []
    updates: dict[str, Any] = {}
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        result = TOOLS_BY_NAME[name].invoke(args)
        tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        # Effets sur le state simu
        if name == "move_to":
            updates["position"] = (args["x"], args["y"])
        elif name == "engage":
            updates["ammo"] = max(0.0, state["ammo"] - 0.1)
        elif name == "report" and "MISSION COMPLETE" in args.get("summary", ""):
            updates["mission_done"] = True
        elif name == "withdraw":
            updates["last_order"] = "withdrawing"

    return {"messages": tool_messages, **updates}


def should_continue(state: PlatoonState) -> str:
    last = state["messages"][-1]
    if state["mission_done"] or state["steps_remaining"] <= 0:
        return END
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(PlatoonState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# --------- Demo -----------------------------------------------------------


if __name__ == "__main__":
    OPORD = (
        "Tenir la ligne de crete 4521 jusqu'a 0800 locale. "
        "ROE defensif. Rapport toutes les 15 minutes. "
        "Point de depart : (40.0, 18.0). Sante pleine, munitions pleines."
    )

    initial_state: PlatoonState = {
        "messages": [HumanMessage(content=OPORD)],
        "position": (40.0, 18.0),
        "health": 1.0,
        "ammo": 1.0,
        "last_order": "hold",
        "enemy_contact": None,
        "steps_remaining": 8,
        "mission_done": False,
    }

    app = build_graph()
    final = app.invoke(initial_state)

    print("=" * 60)
    print("TRACE DE L'EXERCICE")
    print("=" * 60)
    for msg in final["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"[OPORD] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"[AGENT] {msg.content}")
            for tc in msg.tool_calls:
                print(f"    -> call {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"[TOOL ] {msg.content}")
    print("=" * 60)
    print(f"Position finale : {final['position']}")
    print(f"Sante : {final['health']:.0%} | Munitions : {final['ammo']:.0%}")
    print(f"Mission accomplie : {final['mission_done']}")

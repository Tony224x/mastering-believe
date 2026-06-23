"""
Agent Fleet Coordinator (LogiSim) — solution corrigee.

Structure :
- State : FleetState avec messages + etat operationnel (position, batterie, charge utile, etc.)
- Tools : 5 tools qui modifient le state via Command
- Graph : agent node + tool node, boucle ReAct
- Stop condition : hook dans l'agent node qui check batterie / steps / mission

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


class FleetState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    position: tuple[float, float]   # coords entrepot (x, y) en metres
    battery: float                  # 0..1 batterie cumulee de la flotte
    payload: float                  # 0..1 capacite de charge restante
    last_order: str
    blocking_event: dict[str, Any] | None  # ex: collision detectee, fault
    steps_remaining: int
    mission_done: bool


# --------- Tools ----------------------------------------------------------
# Les tools ici sont "purs" : ils retournent juste une string descriptive.
# La mutation de state est faite dans le graphe via le ToolNode custom.


@tool
def move_to(x: float, y: float) -> str:
    """Deplacer la flotte vers (x, y) en coords entrepot."""
    return f"Moved to ({x:.1f}, {y:.1f})"


@tool
def hold_position() -> str:
    """Stabiliser la flotte sur place (mode hold-and-scan)."""
    return "Flotte en position de stabilisation"


@tool
def pickup(parcel_id: str) -> str:
    """Embarquer un colis identifie."""
    return f"Pickup parcel {parcel_id}"


@tool
def report(summary: str) -> str:
    """Envoyer un report d'etat au coordinator OCC."""
    return f"Report sent: {summary}"


@tool
def return_to_dock(direction: str) -> str:
    """Retour au dock dans une direction (north/south/east/west) — typiquement batterie basse."""
    return f"Returning to dock {direction}"


TOOLS = [move_to, hold_position, pickup, report, return_to_dock]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """Tu es le Fleet Coordinator d'une flotte d'AGV dans un entrepot FleetSim.

Contraintes :
- Tu executes le Work Order Plan recu en entree
- Mode operationnel par defaut : pickup-and-deliver, eviter les zones humaines
- Batterie < 20% ou capacite < 10% -> return_to_dock immediat
- Report toutes les 15 minutes ou sur changement de situation (collision, fault)
- Decisions breves, vocabulaire operationnel logistique, francais

Pour chaque step, tu evalues la situation et tu choisis UNE action via un tool.
Explique en une phrase pourquoi tu choisis cette action avant de l'appeler.
Quand l'objectif est atteint, appelle `report` avec "SHIFT COMPLETE" et stop."""


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
            ("move_to", {"x": 45.0, "y": 21.0}, "Avance vers la zone B-12 (pickup point)."),
            ("hold_position", {}, "Position atteinte, scan environnement."),
            ("report", {"summary": "Flotte en position B-12, en hold"}, "Status report."),
            ("pickup", {"parcel_id": "PCL-1688"}, "Colis prioritaire localise, pickup."),
            ("report", {"summary": "SHIFT COMPLETE - colis livre"}, "Mission finie."),
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


def agent_node(state: FleetState) -> dict:
    # Stop conditions dures : on ne laisse pas le LLM decider ici.
    if state["battery"] < 0.2 and state.get("last_order", "") != "returning":
        return {
            "messages": [AIMessage(content="Batterie critique -> return_to_dock auto", tool_calls=[
                {"name": "return_to_dock", "args": {"direction": "south"}, "id": "auto-return", "type": "tool_call"}
            ])],
            "last_order": "returning",
        }
    if state["steps_remaining"] <= 0 or state["mission_done"]:
        return {"messages": [AIMessage(content="Fin de shift")]}

    llm = _make_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    return {
        "messages": [response],
        "steps_remaining": state["steps_remaining"] - 1,
    }


def tool_node(state: FleetState) -> dict:
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

        # Effets sur le state operationnel
        if name == "move_to":
            updates["position"] = (args["x"], args["y"])
        elif name == "pickup":
            updates["payload"] = max(0.0, state["payload"] - 0.1)
        elif name == "report" and "SHIFT COMPLETE" in args.get("summary", ""):
            updates["mission_done"] = True
        elif name == "return_to_dock":
            updates["last_order"] = "returning"

    return {"messages": tool_messages, **updates}


def should_continue(state: FleetState) -> str:
    last = state["messages"][-1]
    if state["mission_done"] or state["steps_remaining"] <= 0:
        return END
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(FleetState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# --------- Demo -----------------------------------------------------------


if __name__ == "__main__":
    work_order_plan = (
        "Couvrir la zone B-12 du quai 4 jusqu'a 0800 locale. "
        "Pickup des colis prioritaires. Report toutes les 15 minutes. "
        "Point de depart : (40.0, 18.0). Batterie pleine, capacite pleine."
    )

    initial_state: FleetState = {
        "messages": [HumanMessage(content=work_order_plan)],
        "position": (40.0, 18.0),
        "battery": 1.0,
        "payload": 1.0,
        "last_order": "hold",
        "blocking_event": None,
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
            print(f"[Work Order Plan] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"[AGENT] {msg.content}")
            for tc in msg.tool_calls:
                print(f"    -> call {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"[TOOL ] {msg.content}")
    print("=" * 60)
    print(f"Position finale : {final['position']}")
    print(f"Batterie : {final['battery']:.0%} | Capacite : {final['payload']:.0%}")
    print(f"Shift complete : {final['mission_done']}")

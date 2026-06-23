"""
Demo du systeme multi-flotte supervisor + swarm (LogiSim).

Scenario : decharger la zone B-12 du quai 4 avec support AGV et drones d'inventaire.
Trace : affiche chaque step du graphe avec l'agent actif et l'action prise.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import build_graph
from state import ShiftState

WORK_ORDER_PLAN = """Work Order Plan shift :
- Mission : decharger et trier la zone B-12 du quai 4 avant 14h
- Anomalies suspectees : 2 palettes lourdes mal etiquetees, possibles colis fragiles
- Moyens : 1 flotte sorting, 1 escouade AGV de transport, 1 patrouille drones d'inventaire
- Delai : H+2
- Contraintes : minimiser collisions inter-flotte (zone partagee avec operateurs humains)
"""


def format_message(msg) -> str:
    if isinstance(msg, HumanMessage):
        return f"[work order ] {msg.content[:80]}..."
    if isinstance(msg, AIMessage):
        name = getattr(msg, "name", None) or "agent"
        line = f"[{name:<9}] {msg.content[:120]}"
        for tc in msg.tool_calls:
            line += f"\n             -> {tc['name']}({tc['args']})"
        return line
    if isinstance(msg, ToolMessage):
        return f"[tool    ] {msg.content[:120]}"
    return str(msg)[:120]


def run_demo() -> None:
    app = build_graph()

    initial: ShiftState = {
        "messages": [HumanMessage(content=WORK_ORDER_PLAN)],
        "active_agent": "supervisor",
        "shift_phase": "PLAN",
        "parcels_observed": [],
        "pickup_requested": False,
        "parcel_marked": None,
        "shift_complete": False,
        "handoff_log": [],
    }

    print("=" * 70)
    print("DEMO — Supervisor + Swarm multi-flotte (decharger zone B-12 quai 4)")
    print("=" * 70)

    step_count = 0
    for event in app.stream(initial, {"recursion_limit": 40}, stream_mode="updates"):
        for node_name, node_output in event.items():
            step_count += 1
            print(f"\n--- Step {step_count} [{node_name}] ---")
            if "messages" in node_output and node_output["messages"]:
                for m in node_output["messages"]:
                    print(format_message(m))
            if "active_agent" in node_output:
                print(f"             active_agent -> {node_output['active_agent']}")
            if "shift_phase" in node_output:
                print(f"             phase        -> {node_output['shift_phase']}")

    # Puis un invoke pour recuperer le state final complet (handoff_log, etc.)
    final_state = app.invoke(initial, {"recursion_limit": 40})

    print("\n" + "=" * 70)
    print("HANDOFF LOG (trace des coordinations swarm)")
    print("=" * 70)
    if final_state and "handoff_log" in final_state:
        for i, (fr, to, reason) in enumerate(final_state["handoff_log"], start=1):
            print(f"  {i}. {fr:<10} -> {to:<10} : {reason}")
    else:
        print("  (handoff_log non disponible dans ce final_state)")


if __name__ == "__main__":
    run_demo()

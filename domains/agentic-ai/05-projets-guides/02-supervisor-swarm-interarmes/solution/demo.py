"""
Demo du systeme interarmes supervisor + swarm.

Scenario : prendre le village 4521 avec support artillerie et reco drone.
Trace : affiche chaque step du graphe avec l'agent actif et l'action prise.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import build_graph
from state import BrigadeState

OPORD = """OPORD Brigade :
- Mission : prendre et tenir le village de 4521
- Ennemi suspecte : 2 blindes legers + infanterie legere
- Moyens : 1 peloton infanterie, 1 section artillerie, 1 patrouille drone
- Delai : H+2
- ROE : reponse graduee, minimiser pertes civiles (village habite)
"""


def format_message(msg) -> str:
    if isinstance(msg, HumanMessage):
        return f"[OPORD   ] {msg.content[:80]}..."
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

    initial: BrigadeState = {
        "messages": [HumanMessage(content=OPORD)],
        "active_agent": "supervisor",
        "mission_phase": "PLAN",
        "enemy_observed": [],
        "support_requested": False,
        "target_marked": None,
        "objective_taken": False,
        "handoff_log": [],
    }

    print("=" * 70)
    print("DEMO — Supervisor + Swarm interarmes (mission prise village 4521)")
    print("=" * 70)

    # Stream en mode "updates" pour voir chaque node successivement
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
            if "mission_phase" in node_output:
                print(f"             phase        -> {node_output['mission_phase']}")

    # Puis un invoke pour recuperer le state final complet (handoff_log, etc.)
    final_state = app.invoke(initial, {"recursion_limit": 40})

    print("\n" + "=" * 70)
    print("HANDOFF LOG (trace des coordinations swarm)")
    print("=" * 70)
    # Reconstruire le handoff log via le final state (il est accumule)
    # Note : dans un vrai run, ca se recupere via app.get_state()
    if final_state and "handoff_log" in final_state:
        for i, (fr, to, reason) in enumerate(final_state["handoff_log"], start=1):
            print(f"  {i}. {fr:<10} -> {to:<10} : {reason}")
    else:
        print("  (handoff_log non disponible dans ce final_state)")


if __name__ == "__main__":
    run_demo()

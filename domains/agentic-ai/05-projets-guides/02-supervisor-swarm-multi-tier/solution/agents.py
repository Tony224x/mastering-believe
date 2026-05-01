"""
Les 4 agents du systeme multi-flotte LogiSim :
- ShiftCoordinator : supervisor qui delegue
- SortingFleetLead, InventoryDroneLead, AGVFleetLead : workers

En mode stub (pas de cle API), les agents suivent un script deterministe
qui illustre le pattern supervisor+swarm. En mode live, c'est le LLM qui
decide.
"""
from __future__ import annotations

import os
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from state import ShiftState
from tools import (
    AGV_TOOLS,
    SORTING_TOOLS,
    DRONE_TOOLS,
)


# ===== System prompts ====================================================

SUPERVISOR_PROMPT = """Tu es le COORDINATEUR DE SHIFT a l'OCC d'un entrepot FleetSim.

Tu pilotes 3 flottes :
- SORTING : sorters et conveyors, tri et stabilisation des flux colis
- DRONE   : drones d'inventaire, scan de zones et marquage de colis prioritaires
- AGV     : AGV de transport, pickup et acheminement vers lignes de tri / quais

Ta mission : decomposer le Work Order Plan en phases (SCAN -> DISPATCH -> FULFILL) et
assigner chaque phase au worker approprie. Tu ne gere PAS les detail operationnels
(laisse les workers s'auto-coordonner via leurs handoffs), tu orchestres les phases
globales.

Quand le shift est complete, reponds FINISH.

A chaque tour, reponds par une ligne unique :
NEXT: <worker>  # worker in {sorting, drone, agv, FINISH}
RATIONALE: <justification breve>
ORDERS: <instruction claire au worker>
"""

SORTING_PROMPT = """Tu es le lead de la flotte SORTING (sorters + conveyors).

Tools disponibles :
- move_to(zone), hold_position : deplacement et posture
- handoff_to_agv(reason) : demande directe de pickup AGV (swarm)
- handoff_to_drone(reason) : demande directe de scan / observation (swarm)
- report_to_coordinator(summary) : rend la main au supervisor

Style : decisions breves, vocabulaire operationnel logistique, une seule action par tour.
Quand tu as besoin d'un pickup AGV, utilise handoff_to_agv PLUTOT que
report_to_coordinator — c'est plus rapide et c'est prevu."""

DRONE_PROMPT = """Tu es le lead de la flotte DRONE (drones d'inventaire).

Tools : scan_sector, mark_parcel, handoff_to_agv (pour demander un pickup sur colis
marque), handoff_to_sorting, report_to_coordinator.

Apres avoir marque un colis prioritaire, tu DOIS le passer directement a l'AGV via
handoff_to_agv. Ne repasse pas par le supervisor pour ca."""

AGV_PROMPT = """Tu es le lead de la flotte AGV de transport.

Tools : dispatch_pickup(zone, units), handoff_to_sorting, handoff_to_drone, report_to_coordinator.

Tu execute un pickup quand un colis est marque OU quand SORTING demande un transport
(pickup_requested=True dans le state). Apres le pickup, tu rends la main a celui qui te l'a
demande via handoff."""


# ===== LLM loader (live ou stub) =========================================


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _make_llm(tools: list) -> Callable:
    if _llm_available():
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
        return llm.bind_tools(tools)
    return None  # on utilise les stubs


# ===== Supervisor =========================================================


def supervisor_node(state: ShiftState) -> dict:
    """Le supervisor decide qui joue ensuite. Pas de tools, juste du routing."""
    phase = state.get("shift_phase", "PLAN")

    # Stub deterministe : illustre l'orchestration phase par phase
    if not _llm_available():
        next_by_phase = {
            "PLAN":     ("drone",   "SCAN",     "Scanner la zone B-12, identifier anomalies"),
            "SCAN":     ("agv",     "DISPATCH", "Pickup des colis marques en B-12-NE"),
            "DISPATCH": ("sorting", "FULFILL",  "Trier la zone B-12 vers lignes 3 et 4"),
            "FULFILL":  ("FINISH",  "DONE",     "Shift complete"),
            "DONE":     ("FINISH",  "DONE",     "Deja termine"),
        }
        next_agent, next_phase, orders = next_by_phase.get(phase, ("FINISH", "DONE", "-"))
        content = (
            f"NEXT: {next_agent}\n"
            f"RATIONALE: phase {phase} -> {next_phase}\n"
            f"ORDERS: {orders}"
        )
        return {
            "messages": [AIMessage(content=content, name="supervisor")],
            "active_agent": next_agent,
            "shift_phase": next_phase,
        }

    # Mode live : on demande au LLM
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
    msgs = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = llm.invoke(msgs)
    # Parse rudimentaire du NEXT: xxx
    next_agent = "FINISH"
    for line in response.content.splitlines():
        if line.startswith("NEXT:"):
            next_agent = line.split(":", 1)[1].strip().lower()
            break
    return {
        "messages": [response],
        "active_agent": next_agent,
    }


# ===== Workers ============================================================


def _stub_worker_response(agent_name: str, state: ShiftState) -> AIMessage:
    """Reponses scriptees pour le mode stub. Illustre les patterns attendus."""
    phase = state.get("shift_phase", "PLAN")
    if agent_name == "drone":
        if not state.get("parcel_marked"):
            # Scan -> mark -> handoff vers agv (SWARM)
            return AIMessage(
                content="Drone lance, scan B-12, colis prioritaires marques, handoff AGV",
                tool_calls=[
                    {
                        "name": "handoff_to_agv",
                        "args": {"reason": "Colis prioritaires marques en B-12-NE, pickup immediat"},
                        "id": "stub-drone-1",
                        "type": "tool_call",
                    }
                ],
                name="drone_lead",
            )
        return AIMessage(
            content="Report: zone B-12 nominale",
            tool_calls=[
                {"name": "report_to_coordinator", "args": {"summary": "RAS zone B-12"}, "id": "stub-drone-2", "type": "tool_call"},
            ],
            name="drone_lead",
        )

    if agent_name == "agv":
        if state.get("pickup_requested"):
            # Pickup demande par sorting -> swarm retour vers sorting
            return AIMessage(
                content="Pickup execute en B-12-sud, retour sorting",
                tool_calls=[
                    {"name": "dispatch_pickup", "args": {"zone": "B-12-sud", "units": 2}, "id": "stub-agv-1", "type": "tool_call"},
                ],
                name="agv_lead",
            )
        # Premier passage : pickup sur colis marque par drone, puis retour au coordinator
        return AIMessage(
            content="Pickup colis marque B-12-NE termine, retour coordinator",
            tool_calls=[
                {"name": "dispatch_pickup", "args": {"zone": "B-12-NE", "units": 3}, "id": "stub-agv-2", "type": "tool_call"},
            ],
            name="agv_lead",
        )

    if agent_name == "sorting":
        if not state.get("shift_complete") and not state.get("pickup_requested"):
            # Detecte un colis fragile bloquant -> demande pickup AGV (swarm)
            return AIMessage(
                content="Colis fragile bloquant en B-12-sud, demande pickup AGV",
                tool_calls=[
                    {
                        "name": "handoff_to_agv",
                        "args": {"reason": "Colis fragile B-12-sud, pickup pour debloquer la ligne"},
                        "id": "stub-sorting-1",
                        "type": "tool_call",
                    }
                ],
                name="sorting_lead",
            )
        # Apres support : rapport shift complete
        return AIMessage(
            content="Zone B-12 videe, shift complete",
            tool_calls=[
                {"name": "report_to_coordinator", "args": {"summary": "SHIFT COMPLETE - zone B-12 quai 4 traitee"}, "id": "stub-sorting-2", "type": "tool_call"},
            ],
            name="sorting_lead",
        )

    return AIMessage(content="(stub inconnu)", name=agent_name)


def make_worker_node(agent_name: str, tools: list, system_prompt: str):
    """Factory pour generer un node worker."""

    def node(state: ShiftState) -> dict:
        if not _llm_available():
            response = _stub_worker_response(agent_name, state)
            return {"messages": [response]}

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.3).bind_tools(tools)
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(msgs)
        return {"messages": [response]}

    return node


sorting_node = make_worker_node("sorting", SORTING_TOOLS, SORTING_PROMPT)
drone_node = make_worker_node("drone", DRONE_TOOLS, DRONE_PROMPT)
agv_node = make_worker_node("agv", AGV_TOOLS, AGV_PROMPT)

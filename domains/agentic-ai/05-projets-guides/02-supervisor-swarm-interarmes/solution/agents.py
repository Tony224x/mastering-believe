"""
Les 4 agents du systeme interarmes :
- BrigadeCommander : supervisor qui delegue
- InfantryLead, ReconLead, ArtilleryLead : workers

En mode stub (pas de cle API), les agents suivent un script deterministe
qui illustre le pattern supervisor+swarm. En mode live, c'est le LLM qui
decide.
"""
from __future__ import annotations

import os
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from state import BrigadeState
from tools import (
    ARTILLERY_TOOLS,
    INFANTRY_TOOLS,
    RECON_TOOLS,
)


# ===== System prompts ====================================================

SUPERVISOR_PROMPT = """Tu es le COMMANDANT DE BRIGADE dans un exercice SWORD.

Tu diriges 3 subordonnes :
- INFANTRY : peloton d'infanterie, assaut et tenue de terrain
- RECON : drones de reconnaissance, observation et marquage cibles
- ARTILLERY : appui feu indirect

Ta mission : decomposer l'OPORD en phases (RECON -> STRIKE -> ASSAULT) et assigner
chaque phase au subordonne approprie. Tu ne commandes pas les gestes tactiques (laisse
les workers s'auto-coordonner via leurs handoffs), tu orchestres les phases globales.

Quand la mission est accomplie, reponds FINISH.

A chaque tour, reponds par une ligne unique :
NEXT: <worker>  # worker in {infantry, recon, artillery, FINISH}
RATIONALE: <justification breve>
ORDERS: <instruction claire au subordonne>
"""

INFANTRY_PROMPT = """Tu es le chef d'un peloton d'INFANTERIE.

Tools disponibles :
- advance_to(grid), take_cover : deplacement et posture
- handoff_to_artillery(reason) : demande directe de tir d'appui (swarm)
- handoff_to_recon(reason) : demande directe d'observation (swarm)
- report_to_commander(summary) : rend la main au supervisor

Style : decisions breves, vocabulaire militaire francais, une seule action par tour.
Quand tu as besoin d'un tir d'appui, utilise handoff_to_artillery PLUTOT que
report_to_commander — c'est plus rapide et c'est prevu."""

RECON_PROMPT = """Tu es le chef de l'unite RECON (drones).

Tools : observe_sector, mark_target, handoff_to_artillery (pour declencher un tir sur cible
marquee), handoff_to_infantry, report_to_commander.

Apres avoir marque une cible, tu DOIS la passer directement a l'artillerie via
handoff_to_artillery. Ne repasse pas par le supervisor pour ca."""

ARTILLERY_PROMPT = """Tu es le chef de l'ARTILLERIE d'appui.

Tools : fire_mission(grid, rounds), handoff_to_infantry, handoff_to_recon, report_to_commander.

Tu execute un tir quand une cible est marquee OU quand l'infanterie demande de l'appui
(support_requested=True dans le state). Apres le tir, tu rends la main a celui qui te l'a
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


def supervisor_node(state: BrigadeState) -> dict:
    """Le supervisor decide qui joue ensuite. Pas de tools, juste du routing."""
    phase = state.get("mission_phase", "PLAN")

    # Stub deterministe : illustre l'orchestration phase par phase
    if not _llm_available():
        next_by_phase = {
            "PLAN": ("recon", "RECON", "Marque les positions OPFOR en 4521"),
            "RECON": ("artillery", "STRIKE", "Feu d'appui sur cibles marquees"),
            "STRIKE": ("infantry", "ASSAULT", "Prends le village 4521"),
            "ASSAULT": ("FINISH", "DONE", "Mission accomplie"),
            "DONE": ("FINISH", "DONE", "Deja termine"),
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
            "mission_phase": next_phase,
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


def _stub_worker_response(agent_name: str, state: BrigadeState) -> AIMessage:
    """Reponses scriptees pour le mode stub. Illustre les patterns attendus."""
    phase = state.get("mission_phase", "PLAN")
    if agent_name == "recon":
        if not state.get("target_marked"):
            # Observe puis marque puis handoff vers artillery (SWARM)
            return AIMessage(
                content="Drone lance, observation 4521, cibles marquees, handoff artillerie",
                tool_calls=[
                    {
                        "name": "handoff_to_artillery",
                        "args": {"reason": "Cibles marquees en 4521-NE, tir immediat"},
                        "id": "stub-recon-1",
                        "type": "tool_call",
                    }
                ],
                name="recon_lead",
            )
        return AIMessage(content="Report: RAS", tool_calls=[
            {"name": "report_to_commander", "args": {"summary": "RAS secteur 4521"}, "id": "stub-recon-2", "type": "tool_call"},
        ], name="recon_lead")

    if agent_name == "artillery":
        if state.get("support_requested"):
            # Tir puis retour infanterie (swarm vers celui qui a demande)
            return AIMessage(
                content="Tir execute sur 4521-sud, appui infanterie",
                tool_calls=[
                    {"name": "fire_mission", "args": {"grid": "4521-sud", "rounds": 6}, "id": "stub-art-1", "type": "tool_call"},
                ],
                name="artillery_lead",
            )
        # Premier passage : tir sur cible marquee, puis retour au commander
        return AIMessage(
            content="Tir sur cible marquee, mission d'appui finie, retour commander",
            tool_calls=[
                {"name": "fire_mission", "args": {"grid": "4521-NE", "rounds": 8}, "id": "stub-art-2", "type": "tool_call"},
            ],
            name="artillery_lead",
        )

    if agent_name == "infantry":
        if not state.get("objective_taken") and not state.get("support_requested"):
            # Demande d'appui artillerie en plein assaut (swarm)
            return AIMessage(
                content="Contact OPFOR 4521-sud, demande appui feu",
                tool_calls=[
                    {
                        "name": "handoff_to_artillery",
                        "args": {"reason": "Contact ennemi 4521-sud, demande tir d'appui"},
                        "id": "stub-inf-1",
                        "type": "tool_call",
                    }
                ],
                name="infantry_lead",
            )
        # Apres appui : rapport mission complete
        return AIMessage(
            content="Village pris, mission accomplie",
            tool_calls=[
                {"name": "report_to_commander", "args": {"summary": "MISSION COMPLETE - village 4521 tenu"}, "id": "stub-inf-2", "type": "tool_call"},
            ],
            name="infantry_lead",
        )

    return AIMessage(content="(stub inconnu)", name=agent_name)


def make_worker_node(agent_name: str, tools: list, system_prompt: str):
    """Factory pour generer un node worker."""

    def node(state: BrigadeState) -> dict:
        if not _llm_available():
            response = _stub_worker_response(agent_name, state)
            return {"messages": [response]}

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.3).bind_tools(tools)
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(msgs)
        return {"messages": [response]}

    return node


infantry_node = make_worker_node("infantry", INFANTRY_TOOLS, INFANTRY_PROMPT)
recon_node = make_worker_node("recon", RECON_TOOLS, RECON_PROMPT)
artillery_node = make_worker_node("artillery", ARTILLERY_TOOLS, ARTILLERY_PROMPT)

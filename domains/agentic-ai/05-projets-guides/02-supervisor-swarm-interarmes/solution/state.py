"""
State partage par tous les agents du systeme interarmes.

Cle : le state est UN SEUL objet que tous les agents lisent et ecrivent.
C'est ce qui permet la coordination — chaque agent voit ce que les autres
ont fait.
"""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


MissionPhase = Literal["PLAN", "RECON", "STRIKE", "ASSAULT", "DONE"]


class EnemyObservation(TypedDict):
    unit_id: str
    grid: str
    kind: str   # "armor", "infantry", "unknown"
    confidence: float


class BrigadeState(TypedDict):
    # Messages : accumule tous les echanges entre agents
    messages: Annotated[list[BaseMessage], add_messages]

    # Qui a la main ? Utilise par les conditional edges pour router.
    active_agent: str  # "supervisor" | "infantry" | "recon" | "artillery"

    # Phase globale de la mission, maintenue par le supervisor
    mission_phase: MissionPhase

    # Etat operationnel partage
    enemy_observed: list[EnemyObservation]
    support_requested: bool   # set par handoff infantry -> artillery
    target_marked: str | None  # grid coord si un drone a marque une cible
    objective_taken: bool

    # Trace des handoffs (from, to, reason) pour debug / AAR
    handoff_log: list[tuple[str, str, str]]

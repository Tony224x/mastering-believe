"""
State partage par tous les agents du systeme multi-flotte LogiSim.

Cle : le state est UN SEUL objet que tous les agents lisent et ecrivent.
C'est ce qui permet la coordination — chaque agent voit ce que les autres
ont fait.
"""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


ShiftPhase = Literal["PLAN", "SCAN", "DISPATCH", "FULFILL", "DONE"]


class ParcelObservation(TypedDict):
    unit_id: str           # ex: "Drone-3"
    zone: str              # ex: "B-12-NE"
    kind: str              # "fragile", "heavy", "standard", "anomaly"
    confidence: float


class ShiftState(TypedDict):
    # Messages : accumule tous les echanges entre agents
    messages: Annotated[list[BaseMessage], add_messages]

    # Qui a la main ? Utilise par les conditional edges pour router.
    active_agent: str  # "supervisor" | "sorting" | "drone" | "agv"

    # Phase globale du shift, maintenue par le supervisor
    shift_phase: ShiftPhase

    # Etat operationnel partage
    parcels_observed: list[ParcelObservation]
    pickup_requested: bool       # set par handoff sorting -> agv
    parcel_marked: str | None    # zone si un drone a marque un colis prioritaire
    shift_complete: bool

    # Trace des handoffs (from, to, reason) pour debug / EOD Review
    handoff_log: list[tuple[str, str, str]]

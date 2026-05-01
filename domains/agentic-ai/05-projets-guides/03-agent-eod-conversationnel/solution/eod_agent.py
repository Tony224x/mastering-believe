"""
Agent EOD conversationnel LogiSim — correction.

Architecture :
- Un log d'events en memoire (fixture)
- 4 tools qui queryent le log
- Un agent ReAct via langgraph.prebuilt.create_react_agent
- Garde-fous : citations obligatoires, refus hors-scope

Mode stub : si pas de cle API, on simule un LLM qui suit une logique simple
"pour chaque question, choisir le tool le plus pertinent une fois, puis
repondre". Ca illustre le flux sans LLM.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


# ===== Fixture log =======================================================
# Schema canonique LogiSim : id, t_sim, kind, unit_id, fleet, payload
# kind in {MOVE, DETECT, ORDER, PICKUP, DROPOFF, COLLISION, FAULT, REPORT, MARK, HANDOFF}

DEMO_EVENTS: list[dict[str, Any]] = [
    {"id": 101, "t_sim": 100.0, "kind": "MOVE",     "unit_id": "AGV-Alpha-2",  "fleet": "own_fleet",      "payload": {"to": "B-12-N"}},
    {"id": 102, "t_sim": 110.0, "kind": "DETECT",   "unit_id": "AGV-Alpha-2",  "fleet": "own_fleet",      "payload": {"target": "PCL-3", "confidence": 0.65}},
    {"id": 103, "t_sim": 115.0, "kind": "PICKUP",   "unit_id": "AGV-Alpha-2",  "fleet": "own_fleet",      "payload": {"mode": "preemptive", "parcel_id": "PCL-3"}},
    {"id": 104, "t_sim": 116.0, "kind": "COLLISION","unit_id": "AGV-Alpha-2",  "fleet": "own_fleet",      "payload": {"with_unit": "Sorter-7", "severity": 0.3}},
    {"id": 105, "t_sim": 120.0, "kind": "FAULT",    "unit_id": "AGV-Alpha-2",  "fleet": "own_fleet",      "payload": {"code": "BATTERY_LOW", "severity": "minor"}},
    {"id": 106, "t_sim": 200.0, "kind": "MOVE",     "unit_id": "AGV-Bravo-1",  "fleet": "own_fleet",      "payload": {"to": "B-12-S"}},
    {"id": 107, "t_sim": 220.0, "kind": "DETECT",   "unit_id": "AGV-Bravo-1",  "fleet": "own_fleet",      "payload": {"target": "PCL-7", "confidence": 0.92}},
    {"id": 108, "t_sim": 250.0, "kind": "PICKUP",   "unit_id": "AGV-Bravo-1",  "fleet": "own_fleet",      "payload": {"mode": "ordered",   "parcel_id": "PCL-7"}},
    {"id": 109, "t_sim": 260.0, "kind": "FAULT",    "unit_id": "AGV-Bravo-1",  "fleet": "own_fleet",      "payload": {"code": "PAYLOAD_OVERLOAD", "severity": "major"}},
    {"id": 110, "t_sim": 270.0, "kind": "DROPOFF",  "unit_id": "AGV-Bravo-1",  "fleet": "own_fleet",      "payload": {"parcel_id": "PCL-7", "to_slot": "L4-3", "ok": True}},
]


# ===== Tools =============================================================


@tool
def list_units(fleet: str) -> list[str]:
    """Liste les unites d'une flotte ('own_fleet' ou 'external_fleet')."""
    return sorted({e["unit_id"] for e in DEMO_EVENTS if e["fleet"] == fleet.lower()})


@tool
def search_events(unit_id: str | None = None, kind: str | None = None,
                  t_start: float | None = None, t_end: float | None = None) -> list[dict]:
    """Recherche events avec filtres optionnels."""
    results = []
    for e in DEMO_EVENTS:
        if unit_id and e["unit_id"] != unit_id:
            continue
        if kind and e["kind"] != kind:
            continue
        if t_start is not None and e["t_sim"] < t_start:
            continue
        if t_end is not None and e["t_sim"] > t_end:
            continue
        results.append(e)
    return results


@tool
def get_unit_timeline(unit_id: str, t_start: float = 0.0, t_end: float = 1e9) -> list[dict]:
    """Tous les events d'une unite dans une fenetre temporelle."""
    return [e for e in DEMO_EVENTS if e["unit_id"] == unit_id and t_start <= e["t_sim"] <= t_end]


@tool
def aggregate_stats(unit_id: str, metric: str) -> dict:
    """Aggrege une metrique pour une unite. metric in {faults, pickups, moves, collisions}."""
    events = [e for e in DEMO_EVENTS if e["unit_id"] == unit_id]
    if metric == "faults":
        faults = [e["id"] for e in events if e["kind"] == "FAULT"]
        return {"unit_id": unit_id, "faults": len(faults), "source_events": faults}
    if metric == "pickups":
        pickups = [e["id"] for e in events if e["kind"] == "PICKUP"]
        return {"unit_id": unit_id, "pickups": len(pickups), "source_events": pickups}
    if metric == "moves":
        moves = [e["id"] for e in events if e["kind"] == "MOVE"]
        return {"unit_id": unit_id, "moves": len(moves), "source_events": moves}
    if metric == "collisions":
        collisions = [e["id"] for e in events if e["kind"] == "COLLISION"]
        return {"unit_id": unit_id, "collisions": len(collisions), "source_events": collisions}
    return {"error": f"metric {metric} inconnu"}


TOOLS = [list_units, search_events, get_unit_timeline, aggregate_stats]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """Tu es un assistant EOD (End-of-Day Review) pour un superviseur OCC d'entrepot.

Regles :
1. Base tes reponses EXCLUSIVEMENT sur les events retournes par les tools.
2. Chaque affirmation factuelle doit etre suivie de citations au format [ev:id] ou [ev:id1, ev:id2].
3. Si une question est hors-scope (meteo, marche, autre site), refuse poliment.
4. Si les events ne contiennent pas l'information, ecris "non documente" plutot que d'inventer.
5. Style : neutre, concis, vocabulaire operationnel logistique en francais.

Pour repondre, utilise les tools pour recuperer les events pertinents, puis redige la reponse
avec citations."""


# ===== Agent =============================================================


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _stub_answer(question: str) -> str:
    """Reponses scriptees pour demo sans cle API."""
    q = question.lower()
    if "alpha" in q and ("pourquoi" in q or "pickup" in q):
        return (
            "L'AGV Alpha-2 a tente un pickup a t=115s en mode preemptif [ev:103] "
            "apres avoir detecte le colis PCL-3 a t=110s avec une confiance moderee (0.65) [ev:102]. "
            "Le pickup a coincide avec une collision de severite 0.3 contre Sorter-7 [ev:104], "
            "puis Alpha-2 a remonte un FAULT batterie a t=120s [ev:105]. "
            "Le feu vert WMS pour le pickup preemptif n'apparait pas dans le log — non documente."
        )
    if "resume" in q or "bilan" in q:
        return (
            "Le shift comporte 10 events principaux. Alpha-2 a detecte puis tente le pickup de PCL-3 "
            "vers t=100-120s avec une collision et un FAULT batterie [ev:101, ev:102, ev:103, ev:104, ev:105]. "
            "Bravo-1 a recupere PCL-7 vers t=200-270s avec un FAULT overload mais dropoff reussi [ev:106-110]."
        )
    if "meteo" in q or "marche" in q or "politique" in q:
        return "Desole, cette question est hors scope de l'EOD Review. Je ne peux repondre que sur les events du shift."
    return "Je n'ai pas assez d'elements pour repondre, pourrais-tu reformuler ta question ?"


def answer_question(question: str) -> str:
    """Point d'entree principal de l'agent."""
    if not _llm_available():
        return _stub_answer(question)

    # Mode live avec create_react_agent
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


# ===== Demo ==============================================================

DEMO_QUESTIONS = [
    "Pourquoi Alpha-2 a tente un pickup avant le feu vert WMS ?",
    "Fais-moi un resume du shift.",
    "Quel temps faisait-il pendant le shift ?",
]


if __name__ == "__main__":
    print("=" * 70)
    print("AGENT EOD CONVERSATIONNEL")
    print("=" * 70)
    for q in DEMO_QUESTIONS:
        print(f"\nQ : {q}")
        print(f"R : {answer_question(q)}")

"""
Agent AAR conversationnel — correction.

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

DEMO_EVENTS: list[dict[str, Any]] = [
    {"id": 101, "t_sim": 100.0, "kind": "MOVE", "unit_id": "Alpha-2", "side": "BLUFOR", "payload": {"to": "4521-N"}},
    {"id": 102, "t_sim": 110.0, "kind": "DETECT", "unit_id": "Alpha-2", "side": "BLUFOR", "payload": {"target": "OPFOR-3"}},
    {"id": 103, "t_sim": 115.0, "kind": "FIRE", "unit_id": "Alpha-2", "side": "BLUFOR", "payload": {"mode": "preemptive"}},
    {"id": 104, "t_sim": 116.0, "kind": "IMPACT", "unit_id": "Alpha-2", "side": "BLUFOR", "payload": {"hit": True}},
    {"id": 105, "t_sim": 120.0, "kind": "DAMAGE", "unit_id": "Alpha-2", "side": "BLUFOR", "payload": {"casualties": 2}},
    {"id": 106, "t_sim": 200.0, "kind": "MOVE", "unit_id": "Bravo-1", "side": "BLUFOR", "payload": {"to": "4521-S"}},
    {"id": 107, "t_sim": 220.0, "kind": "DETECT", "unit_id": "Bravo-1", "side": "BLUFOR", "payload": {"target": "OPFOR-7"}},
    {"id": 108, "t_sim": 250.0, "kind": "FIRE", "unit_id": "Bravo-1", "side": "BLUFOR", "payload": {"mode": "ordered"}},
    {"id": 109, "t_sim": 260.0, "kind": "DAMAGE", "unit_id": "Bravo-1", "side": "BLUFOR", "payload": {"casualties": 5}},
    {"id": 110, "t_sim": 270.0, "kind": "NEUTRALIZED", "unit_id": "OPFOR-7", "side": "OPFOR", "payload": {}},
]


# ===== Tools =============================================================


@tool
def list_units(side: str) -> list[str]:
    """Liste les unites d'un camp ('BLUFOR' ou 'OPFOR')."""
    return sorted({e["unit_id"] for e in DEMO_EVENTS if e["side"] == side.upper()})


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
    """Aggrege une metrique pour une unite. metric in {casualties, fires, moves}."""
    events = [e for e in DEMO_EVENTS if e["unit_id"] == unit_id]
    if metric == "casualties":
        total = sum(e["payload"].get("casualties", 0) for e in events if e["kind"] == "DAMAGE")
        source_ids = [e["id"] for e in events if e["kind"] == "DAMAGE"]
        return {"unit_id": unit_id, "casualties": total, "source_events": source_ids}
    if metric == "fires":
        fires = [e["id"] for e in events if e["kind"] == "FIRE"]
        return {"unit_id": unit_id, "fires": len(fires), "source_events": fires}
    if metric == "moves":
        moves = [e["id"] for e in events if e["kind"] == "MOVE"]
        return {"unit_id": unit_id, "moves": len(moves), "source_events": moves}
    return {"error": f"metric {metric} inconnu"}


TOOLS = [list_units, search_events, get_unit_timeline, aggregate_stats]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """Tu es un assistant AAR (After-Action Review) pour un formateur militaire.

Regles :
1. Base tes reponses EXCLUSIVEMENT sur les events retournes par les tools.
2. Chaque affirmation factuelle doit etre suivie de citations au format [ev:id] ou [ev:id1, ev:id2].
3. Si une question est hors-scope (meteo, politique, autre exercice), refuse poliment.
4. Si les events ne contiennent pas l'information, ecris "non documente" plutot que d'inventer.
5. Style : neutre, concis, vocabulaire militaire OTAN en francais.

Pour repondre, utilise les tools pour recuperer les events pertinents, puis redige la reponse
avec citations."""


# ===== Agent =============================================================


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _stub_answer(question: str) -> str:
    """Reponses scriptees pour demo sans cle API."""
    q = question.lower()
    if "alpha" in q and ("pourquoi" in q or "feu" in q):
        return (
            "Le peloton Alpha-2 a ouvert le feu a t=115s en mode preemptif [ev:103] "
            "apres avoir detecte OPFOR-3 a t=110s [ev:102]. Le tir a touche [ev:104], "
            "puis Alpha-2 a subi 2 pertes a t=120s [ev:105]. Le feu preemptif n'est pas "
            "documente comme ordre prealable — non documente du cote decisionnel."
        )
    if "resume" in q or "bilan" in q:
        return (
            "L'exercice comporte 10 events principaux. Alpha-2 a detecte et engage "
            "OPFOR-3 vers t=100-120s (2 pertes BLUFOR) [ev:101, ev:102, ev:103, ev:105]. "
            "Bravo-1 a neutralise OPFOR-7 vers t=200-270s mais a subi 5 pertes [ev:106, ev:108, ev:109, ev:110]."
        )
    if "meteo" in q or "politique" in q:
        return "Desole, cette question est hors scope de l'AAR. Je ne peux repondre que sur les events de l'exercice."
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
    "Pourquoi Alpha-2 a ouvert le feu avant l'ordre ?",
    "Fais-moi un resume de l'exercice.",
    "Quel temps faisait-il pendant l'exercice ?",
]


if __name__ == "__main__":
    print("=" * 70)
    print("AGENT AAR CONVERSATIONNEL")
    print("=" * 70)
    for q in DEMO_QUESTIONS:
        print(f"\nQ : {q}")
        print(f"R : {answer_question(q)}")

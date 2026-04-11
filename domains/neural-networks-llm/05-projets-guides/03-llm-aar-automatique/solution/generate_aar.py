"""
Pipeline complet events -> rapport AAR markdown.

Pour la demo, on utilise l'API Anthropic (claude-haiku-4-5) qui est la plus
proche en format de ce qu'un Mistral/Llama instruct local fait. En prod
MASA ca serait un modele on-premise.

Usage :
    ANTHROPIC_API_KEY=... python generate_aar.py exercise_events.json

Le pipeline :
1. Parse les events
2. Extract les moments cles (heuristique v0)
3. Pour chaque moment, construit le prompt avec contexte
4. Appelle le LLM (temperature basse, max_tokens cappe)
5. Post-check : parse les citations, verifie que chaque event_id existe
6. Assemble le rapport markdown
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Import local — a adapter si relance depuis un autre repertoire
sys.path.insert(0, str(Path(__file__).parent))
from aar_prompt import SYSTEM_PROMPT, build_user_prompt, format_event_line
from extract_key_moments import KeyMoment, context_window, extract_key_moments

CITATION_RE = re.compile(r"\[ev:([\d, ]+)\]")


def call_llm(system: str, user: str) -> str:
    """Appelle l'API Anthropic. A swap en prod contre un client local (Ollama, vLLM)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        # Fallback : retourne un stub pour tester le pipeline sans cle API
        return (
            "**Contexte**\n"
            "Alpha-2 en observation. [ev:42871]\n\n"
            "**Deroulement**\n"
            "Contact detecte a 600m. [ev:42903] Ordre de cover emis. [ev:42905] "
            "Engagement preemptif. [ev:42931] Deux neutralisations OPFOR. [ev:42942] "
            "Un blesse cote BLUFOR. [ev:42951]\n\n"
            "**Recommandation**\n"
            "Revoir la regle d'engagement preemptif : conforme aux ROE mais a "
            "documenter dans le rapport de maneuver."
        )

    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def check_citations(paragraph: str, valid_event_ids: set[int]) -> tuple[bool, list[int]]:
    """Verifie que chaque event cite existe. Retourne (all_valid, missing_ids)."""
    missing: list[int] = []
    for match in CITATION_RE.finditer(paragraph):
        ids = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        for ev_id in ids:
            if ev_id not in valid_event_ids:
                missing.append(ev_id)
    return not missing, missing


def generate_aar(events: list[dict]) -> str:
    events.sort(key=lambda e: e["t_sim"])
    valid_ids = {e["id"] for e in events}

    moments = extract_key_moments(events)
    if not moments:
        return "# AAR\n\nAucun moment cle detecte dans cet exercice.\n"

    report_parts = ["# After-Action Review\n"]
    report_parts.append(f"Exercice : {len(events)} events, {len(moments)} moments cles.\n\n---\n")

    for idx, moment in enumerate(moments, start=1):
        h = int(moment.t_start // 3600)
        m = int((moment.t_start % 3600) // 60)
        desc = f"Engagement a {h}h{m:02d} ({moment.intensity} events de combat)"
        ctx_events = context_window(events, moment)
        event_lines = [format_event_line(e) for e in ctx_events]

        user_prompt = build_user_prompt(
            moment_description=desc,
            event_lines=event_lines,
            focus_unit=moment.center_unit_id,
        )
        paragraph = call_llm(SYSTEM_PROMPT, user_prompt)

        # Garde-fou citations
        ok, missing = check_citations(paragraph, valid_ids)
        if not ok:
            warning = f"\n> [WARN] Citations invalides detectees : {missing}\n"
            paragraph = paragraph + warning

        report_parts.append(f"## Moment {idx} : {desc}\n\n{paragraph}\n\n---\n")

    return "\n".join(report_parts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Demo avec events synthetiques
        demo = [
            {"id": 42871, "t_sim": 2810.0, "kind": "MOVE", "unit_id": "Alpha-2", "payload": {"to": "304-512"}},
            {"id": 42903, "t_sim": 2832.0, "kind": "DETECT", "unit_id": "Alpha-2", "payload": {"target": "OPFOR-5", "dist": "600m"}},
            {"id": 42905, "t_sim": 2833.0, "kind": "ORDER", "unit_id": "Alpha-2", "payload": {"from": "Bravo-HQ", "order": "cover_report"}},
            {"id": 42931, "t_sim": 2865.0, "kind": "FIRE", "unit_id": "Alpha-2"},
            {"id": 42942, "t_sim": 2878.0, "kind": "DAMAGE", "unit_id": "OPFOR-5", "payload": {"neutralized": 2}},
            {"id": 42951, "t_sim": 2885.0, "kind": "DAMAGE", "unit_id": "Alpha-2", "payload": {"wounded": 1}},
        ]
        print(generate_aar(demo))
    else:
        events = json.loads(Path(sys.argv[1]).read_text())
        print(generate_aar(events))

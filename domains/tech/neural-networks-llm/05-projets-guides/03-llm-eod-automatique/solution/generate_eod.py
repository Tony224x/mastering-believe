"""
Full pipeline events -> markdown EOD Review report (LogiSim).

For the demo we use the Anthropic API (claude-haiku-4-5), which is the
closest in format to what a local Mistral/Llama instruct model does. In
LogiSim production it would be an on-premise model.

Usage:
    ANTHROPIC_API_KEY=... python generate_eod.py shift_events.json

The pipeline:
1. Parse the events
2. Extract the key moments (v0 heuristic)
3. For each moment, build the prompt with context
4. Call the LLM (low temperature, capped max_tokens)
5. Post-check: parse the citations, verify every event_id exists
6. Assemble the markdown report

Note: the generated report (and the LLM stub) is in French on purpose —
the EOD report is a French-speaking business deliverable.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Local import — adapt if re-run from another directory
sys.path.insert(0, str(Path(__file__).parent))
from eod_prompt import SYSTEM_PROMPT, build_user_prompt, format_event_line
from extract_key_moments import KeyMoment, context_window, extract_key_moments

CITATION_RE = re.compile(r"\[ev:([\d, ]+)\]")


def call_llm(system: str, user: str) -> str:
    """Calls the Anthropic API. Swap in prod for a local client (Ollama, vLLM)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        # Fallback: returns a stub to test the pipeline without an API key
        return (
            "**Contexte**\n"
            "AGV-Alpha-2 en patrouille. [ev:42871]\n\n"
            "**Deroulement**\n"
            "Detection colis prioritaire en B-12-NE. [ev:42903] Ordre OCC de pickup conditionnel. [ev:42905] "
            "Pickup execute en mode preemptif. [ev:42931] Collision de severite 0.3 contre Sorter-7. [ev:42942] "
            "FAULT batterie minor remonte. [ev:42951]\n\n"
            "**Recommandation**\n"
            "Le pickup preemptif est conforme au SOP mais a coincide avec une collision : "
            "investiguer le sequencement avec la flotte Sorting dans la meme zone."
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
    """Verifies every cited event exists. Returns (all_valid, missing_ids)."""
    missing: list[int] = []
    for match in CITATION_RE.finditer(paragraph):
        ids = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        for ev_id in ids:
            if ev_id not in valid_event_ids:
                missing.append(ev_id)
    return not missing, missing


def generate_eod(events: list[dict]) -> str:
    events.sort(key=lambda e: e["t_sim"])
    valid_ids = {e["id"] for e in events}

    moments = extract_key_moments(events)
    if not moments:
        return "# EOD Review\n\nAucun moment cle detecte dans ce shift.\n"

    report_parts = ["# End-of-Day Review\n"]
    report_parts.append(f"Shift : {len(events)} events, {len(moments)} moments cles.\n\n---\n")

    for idx, moment in enumerate(moments, start=1):
        h = int(moment.t_start // 3600)
        m = int((moment.t_start % 3600) // 60)
        desc = f"Operation a {h}h{m:02d} ({moment.intensity} events sur la fenetre)"
        ctx_events = context_window(events, moment)
        event_lines = [format_event_line(e) for e in ctx_events]

        user_prompt = build_user_prompt(
            moment_description=desc,
            event_lines=event_lines,
            focus_unit=moment.center_unit_id,
        )
        paragraph = call_llm(SYSTEM_PROMPT, user_prompt)

        # Citation guardrail
        ok, missing = check_citations(paragraph, valid_ids)
        if not ok:
            warning = f"\n> [WARN] Citations invalides detectees : {missing}\n"
            paragraph = paragraph + warning

        report_parts.append(f"## Moment {idx} : {desc}\n\n{paragraph}\n\n---\n")

    return "\n".join(report_parts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Demo with synthetic events
        demo = [
            {"id": 42871, "t_sim": 2810.0, "kind": "MOVE",      "unit_id": "AGV-Alpha-2", "payload": {"to": "B-12-N"}},
            {"id": 42903, "t_sim": 2832.0, "kind": "DETECT",    "unit_id": "AGV-Alpha-2", "payload": {"target": "PCL-3", "dist_m": 4.0}},
            {"id": 42905, "t_sim": 2833.0, "kind": "ORDER",     "unit_id": "AGV-Alpha-2", "payload": {"from": "OCC", "order": "pickup_if_slot_free"}},
            {"id": 42920, "t_sim": 2860.0, "kind": "DETECT",    "unit_id": "Sorter-7",    "payload": {"target": "AGV-Alpha-2", "dist_m": 1.2}},
            {"id": 42931, "t_sim": 2865.0, "kind": "PICKUP",    "unit_id": "AGV-Alpha-2", "payload": {"parcel_id": "PCL-3"}},
            {"id": 42942, "t_sim": 2878.0, "kind": "COLLISION", "unit_id": "AGV-Alpha-2", "payload": {"with_unit": "Sorter-7", "severity": 0.3}},
            {"id": 42951, "t_sim": 2885.0, "kind": "FAULT",     "unit_id": "AGV-Alpha-2", "payload": {"code": "BATTERY_LOW", "severity": "minor"}},
            {"id": 42958, "t_sim": 2890.0, "kind": "DROPOFF",   "unit_id": "AGV-Alpha-2", "payload": {"parcel_id": "PCL-3"}},
        ]
        print(generate_eod(demo))
    else:
        events = json.loads(Path(sys.argv[1]).read_text())
        print(generate_eod(events))

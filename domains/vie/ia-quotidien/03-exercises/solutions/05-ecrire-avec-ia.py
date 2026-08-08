#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solutions guide — 05-ecrire-avec-ia (Ecrire avec l'IA).

This file does not call ChatGPT. It prints a structured answer key the learner
can compare against their prompts and artifacts. requires: stdlib only
"""
from __future__ import annotations

MODULE = "05-ecrire-avec-ia"
TITLE = "Ecrire avec l'IA"
REF = '[Mollick, 2023] ; [OpenAI Prompting Guide]'

# === EASY ===
EASY = {
    "goal": "One short ChatGPT task + personal judgment (2 sentences).",
    "must_include": [
        "Visible prompt text",
        "Personal note (not raw paste only)",
        "No real sensitive data",
    ],
    "sample_prompt_stub": (
        "Role: coach clair pour debutante. Contexte: module "
        + MODULE
        + ". Tache: aide-moi sur UN point. Format: 5 puces. "
        "Contraintes: pas de donnees inventees presentees comme reelles."
    ),
}

# === MEDIUM ===
MEDIUM = {
    "goal": "Reusable artifact + 2 prompt iterations + Risks section.",
    "iterations_min": 2,
    "risks_examples": [
        "Hallucinated citation or statistic",
        "Generic advice not tied to my context",
        "Wrong Excel locale (EN formulas on FR Excel)",
    ],
    "artifact_by_phase": {
        "foundations": "Structured notes 15-25 lines",
        "excel": "Formulas table with test values",
        "pptx": "Slide titles + max 3 bullets each",
    },
}

# === HARD ===
HARD = {
    "goal": "Mini-spec + deliverable tied to Budget PME Demo and/or HEC deck + AI journal.",
    "deliverables": [
        "5-bullet mini-spec",
        "Concrete file or written deliverable",
        "AI journal 8-12 lines",
        "Self-score /10 with 3 sentences",
    ],
    "ethics": "Fictional data only; never real NGO/client data.",
    "capstone_link": "J13-J14 HEC deck 8-12 slides with ChatGPT as primary tool",
}


def easy_solution() -> dict:
    return EASY


def medium_solution() -> dict:
    return MEDIUM


def hard_solution() -> dict:
    return HARD


def smoke() -> None:
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    assert e["must_include"], "easy must_include"
    assert m["iterations_min"] >= 2
    assert "HEC" in h["capstone_link"] or "deck" in h["capstone_link"]
    print(f"OK {MODULE} | {TITLE}")
    print(f"  REF: {REF}")
    print(f"  EASY sample prompt: {e['sample_prompt_stub'][:80]}...")
    print(f"  MEDIUM risks: {len(m['risks_examples'])} examples")
    print(f"  HARD ethics: {h['ethics']}")


if __name__ == "__main__":
    smoke()

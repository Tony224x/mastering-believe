#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 01-ia-sans-panique (IA sans panique).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "01-ia-sans-panique"
TITLE = 'IA sans panique'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "what_to_expect": "L'IA invente souvent des % et des titres de rapports. Aucune source ne doit etre citee sans ouverture reelle.",
    "sample_row": {
      "affirmation": "42 % des PME quebecoises utilisent l'IA en 2025",
      "source_ia": "Rapport Invente Inc. 2025",
      "ouvrable": "non/incertain",
      "decision": "refuser jusqu'a verification StatCan / ISQ"
    },
    "pass_if": [
      "tableau 3 lignes",
      "≥1 refus",
      "2 phrases de reflexion"
    ]
  },
  "medium_key": {
    "example_classification": [
      [
        "Preparer un plan de slides HEC",
        "A"
      ],
      [
        "Coller un extrait bancaire client",
        "B/C interdit"
      ],
      [
        "Demander une formule SOMME.SI",
        "A"
      ],
      [
        "Decider seule de demissionner sur conseil IA",
        "B"
      ],
      [
        "Resumer mes notes de cours",
        "A avec relecture"
      ]
    ],
    "personal_rule_example": "Je ne colle jamais de noms de donateurs, montants reels ONG, ni pieces d'identite."
  },
  "hard_key": {
    "charter_sections": [
      "Buts (3 usages max)",
      "Interdits donnees",
      "Verification V-A-I-R",
      "HEC: reecriture obligatoire",
      "Travail: accord avant usage client"
    ],
    "bad_good_prompt_examples": [
      [
        "Aide-moi",
        "Role coach HEC... Tache: outline 8 slides... Contraintes: fictif"
      ],
      [
        "Budget de mon ONG [vrais chiffres]",
        "Budget PME Demo fictif, colonnes Date|..."
      ],
      [
        "Ecris mon devoir entier",
        "Propose un plan ; je redige la section 2"
      ]
    ]
  }
}''')


def easy_solution() -> dict:
    return dict(SOL["easy_key"])


def medium_solution() -> dict:
    return dict(SOL["medium_key"])


def hard_solution() -> dict:
    return dict(SOL["hard_key"])


def _nonempty_structure(obj: Any, min_items: int = 1) -> None:
    assert obj is not None
    if isinstance(obj, dict):
        assert len(obj) >= min_items, f"dict too small: {obj!r}"
    elif isinstance(obj, (list, tuple, str)):
        assert len(obj) >= min_items, f"seq too small: {obj!r}"


def smoke() -> None:
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    _nonempty_structure(e)
    _nonempty_structure(m)
    _nonempty_structure(h)
    blob = json.dumps(SOL, ensure_ascii=False)
    assert len(blob) > 80, "solution payload too thin"
    # Day-specific asserts

    e, m, h = easy_solution(), medium_solution(), hard_solution()
    assert "sample_row" in e and e["sample_row"]["decision"]
    assert len(m["example_classification"]) >= 5
    assert len(h["bad_good_prompt_examples"]) == 3

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

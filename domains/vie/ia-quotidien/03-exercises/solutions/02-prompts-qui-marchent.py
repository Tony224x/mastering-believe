#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 02-prompts-qui-marchent (Prompts qui marchent).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "02-prompts-qui-marchent"
TITLE = 'Prompts qui marchent'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "fuzzy": "Ameliore mon Excel.",
    "structured_example": "Role: formateur Excel Microsoft 365 pour non-developpeurs.\nContexte: tableau A1:E20 colonnes Date|Libelle|Categorie|Montant|Type(entree/sortie), ligne 1 en-tetes.\nTache: propose 3 ameliorations concretes (formules ou structure) pour obtenir solde mensuel.\nFormat: liste numerotee | action | formule FR si besoin | benefice en 1 phrase.\nContraintes: pas de VBA ; donnees fictives ; Excel FR-CA.\n",
    "why_better": "Le modele recoit le schema + format de sortie + limites techniques."
  },
  "medium_key": {
    "three_templates_head": [
      "coach carriere socratique",
      "SOMME.SI entrees/sorties",
      "outline 10 slides HEC"
    ],
    "iteration_examples": [
      "Raccourcis de 30 %",
      "Supprime le jargon",
      "Donne un cas de test numerique"
    ]
  },
  "hard_key": {
    "library_min": 6,
    "few_shot_min": 2,
    "anti_patterns": [
      "Prompt d'un seul mot",
      "Coller des donnees reelles",
      "Demander un devoir entier sans reecriture",
      "Accepter stats sans source ouvrable"
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
    assert "RCCFC" in e["structured_example"] or "Role:" in e["structured_example"]
    assert len(m["three_templates_head"]) == 3
    assert h["library_min"] == 6 and h["few_shot_min"] == 2

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

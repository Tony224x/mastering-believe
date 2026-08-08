#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 08-nettoyer-analyser (Nettoyer, analyser, visualiser).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "08-nettoyer-analyser"
TITLE = 'Nettoyer, analyser, visualiser'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "problems": [
      "formats de dates heterogenes",
      "espaces autour des libelles",
      "casse Type inconsistante (sortie/Sortie)",
      "separateur decimal virgule vs point possible",
      "doublon loyer potentiel",
      "separateur champs ; vs colonnes Excel"
    ]
  },
  "medium_key": {
    "standards": {
      "dates": "YYYY-MM-DD",
      "type": "entree|sortie minuscules",
      "categories": "Title Case",
      "montants": "nombre pur"
    }
  },
  "hard_key": {
    "chart": "barres categories vs total sorties",
    "title_example": "Le loyer concentre plus de la moitie des sorties",
    "insights_min": 3
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
    assert len(e["problems"]) >= 5
    assert m["standards"]["dates"] == "YYYY-MM-DD"
    assert h["chart"].startswith("barres")

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

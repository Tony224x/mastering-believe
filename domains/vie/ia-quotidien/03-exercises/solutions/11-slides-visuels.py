#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 11-slides-visuels (Slides & design sobre).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "11-slides-visuels"
TITLE = 'Slides & design sobre'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "before_title": "Solution",
    "after_example": {
      "title": "Un solde clair chaque lundi en 10 minutes",
      "bullets": [
        "Tableau unique entrees/sorties",
        "3 formules seulement",
        "Routine IA pour expliquer les ecarts"
      ],
      "visual": "schema 3 blocs Lundi → Tableau → Decision"
    }
  },
  "medium_key": {
    "n_slides": 4,
    "max_bullets": 3,
    "iterations_min": 1
  },
  "hard_key": {
    "style_rules_min": 5,
    "slides_min": 6,
    "fixes_min": 3
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
    assert len(e["after_example"]["bullets"]) <= 3
    assert m["max_bullets"] == 3
    assert h["slides_min"] >= 6

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

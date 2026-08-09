#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 07-formules-tableaux (Formules, tableaux & modeles).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "07-formules-tableaux"
TITLE = 'Formules, tableaux & modeles'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "sens_formula_fr": "=SI(E2=\"entree\";D2;-D2)",
    "check": "SOMME(Sens) == total entrees - total sorties"
  },
  "medium_key": {
    "resume_cells": [
      "B2 entrees",
      "B3 sorties",
      "B4 solde",
      "B5 ratio"
    ],
    "div_guard_fr": "=SI(B2=0;\"n/a\";B3/B2)",
    "copy_errors": [
      "mauvaise plage",
      "formule EN",
      "entetes inclus dans SOMME"
    ]
  },
  "hard_key": {
    "min_categories": 4,
    "example_categories": [
      "loyer",
      "salaires",
      "marketing",
      "fournitures"
    ],
    "validation": "recalcul manuel 1 categorie complete"
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
    assert "SI(" in e["sens_formula_fr"]
    assert "DIV" in m["div_guard_fr"] or "n/a" in m["div_guard_fr"]
    assert h["min_categories"] >= 4

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

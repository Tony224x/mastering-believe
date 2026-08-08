#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 09-projet-tresorerie (Projet tresorerie / budget PME).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "09-projet-tresorerie"
TITLE = 'Projet tresorerie / budget PME'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "csv_header": "date,libelle,categorie,montant,type",
    "n_min": 15,
    "audit": [
      "pas de vrais noms",
      "mix entrees/sorties",
      "categories stables"
    ]
  },
  "medium_key": {
    "sheets": [
      "Transactions",
      "Resume",
      "Readme"
    ],
    "n_min": 20,
    "resume_formulas": [
      "SOMME.SI entrees",
      "SOMME.SI sorties",
      "solde"
    ]
  },
  "hard_key": {
    "sheets": [
      "Transactions",
      "Resume",
      "Scenarios",
      "Readme"
    ],
    "scenarios": {
      "base": [
        1.0,
        1.0
      ],
      "optimiste": [
        1.1,
        1.0
      ],
      "pessimiste": [
        0.9,
        1.05
      ]
    },
    "python_helper": "domains/vie/ia-quotidien/02-code/09-projet-tresorerie.py",
    "example_math": "base 10000/8000 -> solde 2000; pessimiste 9000/8400 -> solde 600"
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
    assert e["n_min"] == 15
    assert m["n_min"] == 20
    assert h["scenarios"]["pessimiste"] == [0.9, 1.05] or h["scenarios"]["pessimiste"] == (0.9, 1.05)
    # scenario math from helper docstring
    base_e, base_s = 10000.0, 8000.0
    pe, ps = 0.9, 1.05
    assert abs((base_e*pe) - (base_s*ps) - 600.0) < 1e-6

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

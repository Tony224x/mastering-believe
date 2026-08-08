#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 06-excel-bases-ia (Excel + IA : les bases).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "06-excel-bases-ia"
TITLE = 'Excel + IA : les bases'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "sample_data": [
      [
        "2026-01-02",
        "Vente A",
        120,
        "entree"
      ],
      [
        "2026-01-03",
        "Vente B",
        80,
        "entree"
      ],
      [
        "2026-01-04",
        "Vente C",
        50,
        "entree"
      ],
      [
        "2026-01-05",
        "Loyer",
        100,
        "sortie"
      ],
      [
        "2026-01-06",
        "Fournitures",
        30,
        "sortie"
      ],
      [
        "2026-01-07",
        "Pub",
        20,
        "sortie"
      ]
    ],
    "formula_fr": "=SOMME.SI(D2:D7;\"entree\";C2:C7)",
    "formula_fr_alt": "selon ordre colonnes Montant/Type — adapter plages",
    "expected_entrees": 250
  },
  "medium_key": {
    "formulas": {
      "entrees": "SOMME.SI sur Type=entree",
      "sorties": "SOMME.SI sur Type=sortie",
      "solde": "entrees - sorties"
    },
    "error_drill": "reduire la plage d'une ligne puis lire # éventuel / total faux"
  },
  "hard_key": {
    "sheets": [
      "Transactions",
      "Resume"
    ],
    "indicators": [
      "total entrees",
      "total sorties",
      "solde",
      "NB transactions"
    ],
    "locale_trap": "#NOM? si SUM au lieu de SOMME"
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
    assert e["expected_entrees"] == 250
    assert "SOMME.SI" in e["formula_fr"]
    assert "solde" in m["formulas"]
    assert "Transactions" in h["sheets"]
    # arithmetic of sample data
    rows = e["sample_data"]
    total_in = sum(r[2] for r in rows if r[3] == "entree")
    assert total_in == 250

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

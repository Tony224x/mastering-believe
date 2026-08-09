#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 10-structure-pitch (Structure d'un pitch qui tient).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "10-structure-pitch"
TITLE = "Structure d'un pitch qui tient"
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "problem_example": "Les petits cafes perdent du temps chaque semaine a suivre stock et tresorerie a la main.",
    "promise_example": "Un tableau simple + routines IA les aide a voir leur solde en 10 minutes le lundi."
  },
  "medium_key": {
    "n_slides": 10,
    "must_have": [
      "accroche probleme",
      "solution",
      "comment ca marche",
      "demande finale"
    ],
    "rewrite_min": 4
  },
  "hard_key": {
    "range": [
      8,
      12
    ],
    "fictional_numbers_label": "exemple fictif",
    "success_map_min": 3
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
    assert "phrase" in e["problem_example"] or len(e["problem_example"]) > 20
    assert m["n_slides"] == 10 and m["rewrite_min"] >= 4
    assert h["range"] == [8, 12] or h["range"] == (8, 12)

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

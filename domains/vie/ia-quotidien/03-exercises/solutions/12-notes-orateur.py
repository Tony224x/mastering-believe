#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 12-notes-orateur (Notes orateur & repetition).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "12-notes-orateur"
TITLE = 'Notes orateur & repetition'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "format": [
      "ouverture 10s",
      "2 details hors slide",
      "transition",
      "filet"
    ],
    "opening_example": "Imaginez perdre chaque lundi une heure a retrouver vos entrees dans trois cahiers differents."
  },
  "medium_key": {
    "slides": [
      "accroche",
      "solution",
      "chiffres ou demande"
    ],
    "questions_min": 5,
    "filter": "pas de chiffre absent du deck"
  },
  "hard_key": {
    "duration_target_min": 7,
    "passes": 2,
    "scores": [
      "fluidite/5",
      "clarte/5"
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
    assert len(e["format"]) == 4
    assert m["questions_min"] == 5
    assert h["passes"] == 2 and h["duration_target_min"] == 7

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

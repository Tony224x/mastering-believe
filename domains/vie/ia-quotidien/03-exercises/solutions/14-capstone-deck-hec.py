#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 14-capstone-deck-hec (Capstone final : deck HEC).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "14-capstone-deck-hec"
TITLE = 'Capstone final : deck HEC'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "cut_target_pct": 20,
    "slides_to_trim": 5,
    "before_after_min": 2
  },
  "medium_key": {
    "audit": "chaque chiffre a une source ou est retire",
    "notes_slides": 4,
    "rubric_max": 20
  },
  "hard_key": {
    "deliverables": [
      "Pitch-HEC-Final.pptx",
      "notes orateur",
      "journal-ia.md",
      "Budget-PME-Demo.xlsx (optionnel)"
    ],
    "oral_min": 6,
    "oral_max": 8,
    "score_min": 14,
    "primary_tool": "ChatGPT",
    "validator": "02-code/14-capstone-deck-hec.py",
    "outline_rules": {
      "min_slides": 8,
      "max_slides": 12,
      "max_bullets": 3
    }
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
    assert e["cut_target_pct"] == 20
    assert m["rubric_max"] == 20
    assert h["primary_tool"] == "ChatGPT"
    assert h["score_min"] == 14
    assert "Pitch-HEC-Final.pptx" in h["deliverables"]
    assert h["outline_rules"]["min_slides"] == 8

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

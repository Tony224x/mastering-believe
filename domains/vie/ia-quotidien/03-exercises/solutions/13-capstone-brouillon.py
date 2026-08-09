#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 13-capstone-brouillon (Capstone brouillon deck pitch).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "13-capstone-brouillon"
TITLE = 'Capstone brouillon deck pitch'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "n_titles": 10,
    "story_seconds": 60,
    "gaps_min": 3
  },
  "medium_key": {
    "range": [
      8,
      12
    ],
    "checklist": [
      "pas d'etude inventee",
      "pas de data employeur",
      "titres conclusions",
      "fil 60s",
      "label fictif"
    ]
  },
  "hard_key": {
    "critique_parts": [
      "note/10",
      "5 faiblesses",
      "3 accroches"
    ],
    "polish_tasks_max": 7,
    "journal_lines": 12
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
    assert e["n_titles"] == 10 and e["story_seconds"] == 60
    assert "pas d'etude inventee" in h.get("checklist", m.get("checklist", [])) or "pas d'etude inventee" in m["checklist"]
    assert h["polish_tasks_max"] == 7

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

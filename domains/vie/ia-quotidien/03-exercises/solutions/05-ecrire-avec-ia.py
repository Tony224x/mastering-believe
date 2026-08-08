#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 05-ecrire-avec-ia (Ecrire avec l'IA).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "05-ecrire-avec-ia"
TITLE = "Ecrire avec l'IA"
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "outline_example": [
      "Intro enjeu PME",
      "Diagnostic",
      "Options d'innovation",
      "Plan 30 jours",
      "Risques & mesures"
    ],
    "forbidden": "devoir entier en un prompt"
  },
  "medium_key": {
    "max_words": 400,
    "rewrite_min_pct": 30,
    "marker": "[A_VERIFIER]"
  },
  "hard_key": {
    "structure": [
      "intro",
      "2 sections",
      "conclusion",
      "journal IA"
    ],
    "topic": "test innovation 30 jours PME (fictif)"
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
    assert len(e["outline_example"]) == 5
    assert m["rewrite_min_pct"] >= 30
    assert "journal IA" in " ".join(h["structure"])

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

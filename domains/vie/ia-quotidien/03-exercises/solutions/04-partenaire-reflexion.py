#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 04-partenaire-reflexion (Partenaire de reflexion).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "04-partenaire-reflexion"
TITLE = 'Partenaire de reflexion'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "prompt_core": "une question a la fois ; pas de conseil avant Q5",
    "success": "resume corrige manuellement ; pas de plan de vie impose par l'IA"
  },
  "medium_key": {
    "seven_axes": [
      "marche",
      "temps",
      "competences",
      "argent",
      "ethique",
      "execution",
      "clarte"
    ],
    "response_labels": [
      "accepte",
      "mitige",
      "rejette"
    ]
  },
  "hard_key": {
    "protocol": [
      "cadre 5 lignes",
      "8+ questions",
      "resume IA",
      "synthese humaine",
      "metriques"
    ],
    "anti_pattern": "accepter un plan de carriere tout fait a la 1re reponse"
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
    assert "question" in e["prompt_core"]
    assert len(m["seven_axes"]) == 7
    assert "8+ questions" in h["protocol"] or any("8" in x for x in h["protocol"])

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

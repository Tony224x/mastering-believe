#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Answer key — 03-hallucinations-verification (Hallucinations & verification).

Per-level keys for the day-specific exercises. Runnable smoke tests encode
the critical constraints of each level (not a generic shell).
requires: stdlib only
"""
from __future__ import annotations

import json
from typing import Any

MODULE = "03-hallucinations-verification"
TITLE = 'Hallucinations & verification'
SOL: dict[str, Any] = json.loads(r'''{
  "easy_key": {
    "likely_outcome": "Numero d'article invente ou loi confondue — decision typique: jeter ou reformuler sans citation.",
    "vair_example": {
      "V": "non (source non ouverte)",
      "A": "non (pas fournie par l'utilisateur)",
      "I": "oui (precision excessive)",
      "R": "eleve si utilise en devoir/travail"
    }
  },
  "medium_key": {
    "cleaning_moves": [
      "retirer % non sources",
      "remplacer par 'souvent'/'parfois'",
      "marquer [A_VERIFIER]",
      "garder le raisonnement qualitatif"
    ],
    "before_after_required": true
  },
  "hard_key": {
    "sections": [
      "Interdits collage",
      "V-A-I-R",
      "formation",
      "Travail employeur",
      "Exemples"
    ],
    "case_actions": {
      "stats_pitch": "exiger source ouvrable ou retirer le chiffre",
      "formule": "tester sur 3 lignes connues dans Excel",
      "donnee_sensible": "anonymiser / jeu fictif / ne pas coller"
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
    assert set(e["vair_example"]) >= {"V", "A", "I", "R"}
    assert m.get("before_after_required") is True
    assert "stats_pitch" in h["case_actions"]

    # re-bind after extra (extra may reassign)
    e, m, h = easy_solution(), medium_solution(), hard_solution()
    print(f"OK {MODULE} | {TITLE}")
    print(f"  easy keys: {list(e)[:6]}")
    print(f"  medium keys: {list(m)[:6]}")
    print(f"  hard keys: {list(h)[:6]}")
    return None


if __name__ == "__main__":
    smoke()

#!/usr/bin/env python3
"""Demo: validate a tiny cash ledger with pure Python (stdlib).
Mirrors what learner builds in Excel with ChatGPT-suggested formulas.
requires: stdlib only
"""
from __future__ import annotations

def total_by_type(rows: list[dict], type_value: str) -> float:
    return sum(r["montant"] for r in rows if r["type"] == type_value)

def solde(rows: list[dict]) -> float:
    return total_by_type(rows, "entree") - total_by_type(rows, "sortie")

if __name__ == "__main__":
    demo = [
        {"libelle": "Vente", "montant": 500.0, "type": "entree"},
        {"libelle": "Loyer", "montant": 200.0, "type": "sortie"},
        {"libelle": "Cafe", "montant": 15.0, "type": "sortie"},
    ]
    print("entrees", total_by_type(demo, "entree"))
    print("sorties", total_by_type(demo, "sortie"))
    print("solde", solde(demo))
    assert solde(demo) == 285.0
    print("OK excel-bases demo")

#!/usr/bin/env python3
"""Scenario engine for fictional PME budget (stdlib).
Learner still builds the real workbook in Excel; this shows scenario math.
requires: stdlib only
"""
from __future__ import annotations

def apply_scenario(entrees: float, sorties: float, entrees_factor: float, sorties_factor: float) -> dict:
    e = entrees * entrees_factor
    s = sorties * sorties_factor
    return {"entrees": e, "sorties": s, "solde": e - s}

if __name__ == "__main__":
    base_e, base_s = 10000.0, 8000.0
    scenarios = {
        "base": (1.0, 1.0),
        "optimiste": (1.1, 1.0),
        "pessimiste": (0.9, 1.05),
    }
    for name, (ef, sf) in scenarios.items():
        r = apply_scenario(base_e, base_s, ef, sf)
        print(f"{name}: solde={r['solde']:.2f}")
    assert apply_scenario(base_e, base_s, 1.0, 1.0)["solde"] == 2000.0
    print("OK tresorerie scenarios")

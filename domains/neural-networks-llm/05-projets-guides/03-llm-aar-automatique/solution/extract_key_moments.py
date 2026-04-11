"""
Extraction de moments cles depuis un log d'events d'exercice.

Heuristique simple v0 : on detecte les "bursts" de densite d'events lies
au combat (FIRE, DAMAGE, NEUTRALIZED, DETECT) dans une fenetre glissante.
Un burst au-dessus d'un seuil devient un "moment cle".

Pour le vrai produit, on aurait probablement un classifier ML entraine
sur des moments cles annotes par des formateurs. v0 suffit pour une baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

COMBAT_KINDS = {"FIRE", "DAMAGE", "NEUTRALIZED", "DETECT", "HIT", "IMPACT"}


@dataclass
class KeyMoment:
    t_start: float
    t_end: float
    center_unit_id: str
    event_ids: list[int]
    intensity: int  # nombre d'events de combat dans la fenetre


def extract_key_moments(
    events: list[dict],
    window_sec: float = 30.0,
    min_intensity: int = 5,
) -> list[KeyMoment]:
    """
    events : liste dict avec au moins les cles {id, t_sim, kind, unit_id}, triee par t_sim.

    Retourne une liste de moments cles non-overlapping : quand deux fenetres
    intenses se chevauchent, on les fusionne.
    """
    if not events:
        return []

    # Fenetre glissante sur les events de combat
    combat_events = [e for e in events if e["kind"] in COMBAT_KINDS]
    moments: list[KeyMoment] = []

    i = 0
    n = len(combat_events)
    while i < n:
        t0 = combat_events[i]["t_sim"]
        j = i
        while j < n and combat_events[j]["t_sim"] - t0 <= window_sec:
            j += 1
        intensity = j - i
        if intensity >= min_intensity:
            # Identifier l'unite pivot : celle qui apparait le plus dans la fenetre
            counts: dict[str, int] = {}
            for e in combat_events[i:j]:
                counts[e["unit_id"]] = counts.get(e["unit_id"], 0) + 1
            center_unit = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

            m = KeyMoment(
                t_start=t0,
                t_end=combat_events[j - 1]["t_sim"],
                center_unit_id=center_unit,
                event_ids=[e["id"] for e in combat_events[i:j]],
                intensity=intensity,
            )
            # Fusionner avec le moment precedent si overlap
            if moments and m.t_start <= moments[-1].t_end + 5.0:
                prev = moments[-1]
                merged_ids = list(dict.fromkeys(prev.event_ids + m.event_ids))
                moments[-1] = KeyMoment(
                    t_start=prev.t_start,
                    t_end=max(prev.t_end, m.t_end),
                    center_unit_id=prev.center_unit_id,  # garde le pivot de la premiere fenetre
                    event_ids=merged_ids,
                    intensity=len(merged_ids),
                )
            else:
                moments.append(m)
            i = j  # skip the window
        else:
            i += 1

    return moments


def context_window(events: list[dict], moment: KeyMoment, pre_sec: float = 30.0, post_sec: float = 60.0) -> list[dict]:
    """Retourne les events dans une fenetre autour d'un moment cle (pour le contexte LLM)."""
    t_min = moment.t_start - pre_sec
    t_max = moment.t_end + post_sec
    return [e for e in events if t_min <= e["t_sim"] <= t_max]


if __name__ == "__main__":
    # Mini test synthetique
    demo_events = [
        {"id": 1, "t_sim": 100.0, "kind": "MOVE", "unit_id": "A1"},
        {"id": 2, "t_sim": 110.0, "kind": "DETECT", "unit_id": "A1"},
        {"id": 3, "t_sim": 111.0, "kind": "FIRE", "unit_id": "A1"},
        {"id": 4, "t_sim": 112.0, "kind": "IMPACT", "unit_id": "A1"},
        {"id": 5, "t_sim": 113.0, "kind": "DAMAGE", "unit_id": "A1"},
        {"id": 6, "t_sim": 115.0, "kind": "FIRE", "unit_id": "A1"},
        {"id": 7, "t_sim": 116.0, "kind": "NEUTRALIZED", "unit_id": "E2"},
        {"id": 8, "t_sim": 200.0, "kind": "MOVE", "unit_id": "A1"},
    ]
    moments = extract_key_moments(demo_events, min_intensity=3)
    for m in moments:
        print(m)

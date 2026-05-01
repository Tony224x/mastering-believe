"""
Event loop deterministe pour simulation discrete-event LogiSim — correction.

Cles de design :
- Heap (t, seq, event) : seq est le tie-breaker, garantit FIFO sur t egaux
- _now monotone : levee si on tente de scheduler dans le passe
- Pas de lazy cancel dans cette version : les events annules restent dans le
  heap jusqu'a leur t. Suffit pour la majorite des cas, on complexifie que
  si on mesure un probleme.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    handler: Callable[["EventLoop", "Event"], None] | None = None

    def apply(self, loop: "EventLoop") -> None:
        if self.handler is not None:
            self.handler(loop, self)


class EventLoop:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Event]] = []
        self._seq = 0
        self._now = 0.0
        self.log: list[tuple[float, str]] = []  # traces pour EOD Review

    def now(self) -> float:
        return self._now

    def schedule(self, event: Event, at: float) -> None:
        if at < self._now:
            # Violation d'invariant : on ne schedule jamais dans le passe.
            # Lever plutot que "corriger silencieusement" — un bug ici
            # casse completement le determinisme.
            raise ValueError(f"Cannot schedule in the past: at={at} < now={self._now}")
        self._seq += 1
        heapq.heappush(self._heap, (at, self._seq, event))

    def run_until(self, t_end: float) -> int:
        """Traite tous les events jusqu'a t_end inclus. Retourne le nombre traite."""
        count = 0
        while self._heap:
            # Peek sans pop pour verifier la condition
            t_next = self._heap[0][0]
            if t_next > t_end:
                break
            t, _seq, event = heapq.heappop(self._heap)
            # Avancer le temps simule AVANT d'appliquer, pour que event.apply()
            # puisse lire loop.now() et planifier des sous-events corrects.
            self._now = t
            self.log.append((t, event.kind))
            event.apply(self)
            count += 1
        # Apres la boucle, avancer le "now" a t_end si on s'arrete pour rien
        # (utile pour que run_until(10) suivi de run_until(20) ait un now correct).
        if self._now < t_end:
            self._now = t_end
        return count


# ---------- Scenario de demonstration : duel pickup -------------------------

def _duel_pickup_scenario() -> EventLoop:
    """Deux AGV, A et B essaient de prendre un colis prioritaire en zone B-12.
    Les 2 premiers tentatives de A echouent (slot occupe), la 3e reussit.
    Chaque tentative prend 1.0 s de positionnement + 0.5 s de pickup.
    """
    loop = EventLoop()
    state = {"slot_free": True, "attempts": 0, "pickups": 0}

    def on_attempt(lp: EventLoop, ev: Event) -> None:
        if state["pickups"] > 0:
            return
        state["attempts"] += 1
        # Pickup en cours : finalisation a t + 0.5
        lp.schedule(Event("PICKUP_TRY", {"attempt": state["attempts"]}, on_pickup_try), lp.now() + 0.5)
        # Repositionnement pour la prochaine tentative : 1.5 s plus tard
        if state["attempts"] < 3:
            lp.schedule(Event("MOVE_RETRY", {}, on_attempt), lp.now() + 1.5)

    def on_pickup_try(lp: EventLoop, ev: Event) -> None:
        # Deterministe : les 2 premieres tentatives echouent (slot occupe), la 3e reussit.
        # Dans un vrai moteur : tirage seed sur un PRNG / etat reel du WMS.
        attempt_no = ev.payload["attempt"]
        if attempt_no == 3:
            state["pickups"] += 1
            lp.schedule(Event("PICKUP_DONE", {"unit": "AGV-A"}), lp.now() + 0.0)

    loop.schedule(Event("MOVE_RETRY", {}, on_attempt), at=0.0)
    loop.run_until(10.0)
    return loop


if __name__ == "__main__":
    loop = _duel_pickup_scenario()
    for t, kind in loop.log:
        print(f"t={t:5.2f}  {kind}")
    # Run deux fois, verifier que les logs sont identiques -> determinisme
    l1 = _duel_pickup_scenario()
    l2 = _duel_pickup_scenario()
    assert l1.log == l2.log, "Non-deterministe !"
    print("Determinisme OK")

"""
Deterministic event loop for LogiSim discrete-event simulation — solution.

Design keys:
- Heap (t, seq, event): seq is the tie-breaker, guarantees FIFO on equal t
- Monotonic _now: raises if we try to schedule in the past
- No lazy cancel in this version: cancelled events stay in the heap until
  their t. Good enough for most cases — we only add complexity once a
  problem is actually measured.
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
        self.log: list[tuple[float, str]] = []  # traces for EOD Review

    def now(self) -> float:
        return self._now

    def schedule(self, event: Event, at: float) -> None:
        if at < self._now:
            # Invariant violation: we never schedule in the past.
            # Raise rather than "silently fix" — a bug here completely
            # breaks determinism.
            raise ValueError(f"Cannot schedule in the past: at={at} < now={self._now}")
        self._seq += 1
        heapq.heappush(self._heap, (at, self._seq, event))

    def run_until(self, t_end: float) -> int:
        """Process all events up to t_end inclusive. Returns the count processed."""
        count = 0
        while self._heap:
            # Peek without popping to check the condition
            t_next = self._heap[0][0]
            if t_next > t_end:
                break
            t, _seq, event = heapq.heappop(self._heap)
            # Advance simulated time BEFORE applying, so that event.apply()
            # can read loop.now() and schedule correct sub-events.
            self._now = t
            self.log.append((t, event.kind))
            event.apply(self)
            count += 1
        # After the loop, advance "now" to t_end if we stopped with nothing left
        # (useful so that run_until(10) followed by run_until(20) has a correct now).
        if self._now < t_end:
            self._now = t_end
        return count


# ---------- Demonstration scenario: pickup duel ------------------------------

def _duel_pickup_scenario() -> EventLoop:
    """Two AGVs, A and B, try to grab a priority parcel in zone B-12.
    A's first 2 attempts fail (slot occupied), the 3rd succeeds.
    Each attempt takes 1.0 s of positioning + 0.5 s of pickup.
    """
    loop = EventLoop()
    state = {"slot_free": True, "attempts": 0, "pickups": 0}

    def on_attempt(lp: EventLoop, ev: Event) -> None:
        if state["pickups"] > 0:
            return
        state["attempts"] += 1
        # Pickup in progress: completion at t + 0.5
        lp.schedule(Event("PICKUP_TRY", {"attempt": state["attempts"]}, on_pickup_try), lp.now() + 0.5)
        # Repositioning for the next attempt: 1.5 s later
        if state["attempts"] < 3:
            lp.schedule(Event("MOVE_RETRY", {}, on_attempt), lp.now() + 1.5)

    def on_pickup_try(lp: EventLoop, ev: Event) -> None:
        # Deterministic: the first 2 attempts fail (slot occupied), the 3rd succeeds.
        # In a real engine: seeded PRNG draw / actual WMS state.
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
    # Run twice, check that the logs are identical -> determinism
    l1 = _duel_pickup_scenario()
    l2 = _duel_pickup_scenario()
    assert l1.log == l2.log, "Non-deterministe !"
    print("Determinisme OK")

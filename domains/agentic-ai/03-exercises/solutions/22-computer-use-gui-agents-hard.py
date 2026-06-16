"""
Solutions -- Day 22 (HARD): Computer use & GUI agents

Contains solutions for:
  - Hard Ex 1: RobustGUIAgent -- retries + step budget + loop detection +
               HITL gate before irreversible actions + graceful failure.
               Three scenarios: success / loop-aborted / HITL-blocked.
  - Hard Ex 2: Mini WebArena-style multi-page harness with cumulative-error
               propagation + an ablation showing SoM grounding beats raw-pixel.

Self-contained & offline: the needed primitives from
02-code/22-computer-use-gui-agents.py (BoundingBox, UIElement, VirtualScreen)
are embedded below -- nothing is imported from the day's code. The "vision LLM"
is a deterministic MockLLM. stdlib only, no network, no API key. All randomness
is seeded via random.Random (no global state).

Run:  python 03-exercises/solutions/22-computer-use-gui-agents-hard.py
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable


# ==========================================================================
# EMBEDDED PRIMITIVES (minimal copy of 02-code/22-computer-use-gui-agents.py)
# ==========================================================================


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass
class UIElement:
    kind: str
    label: str
    bbox: BoundingBox
    value: str = ""
    focusable: bool = True
    dom_id: str = ""


class VirtualScreen:
    WIDTH = 50
    HEIGHT = 14

    def __init__(self, elements: list[UIElement]) -> None:
        self.elements = elements
        self.focused: UIElement | None = None
        self.last_activated: str | None = None

    def set_of_marks(self) -> dict[int, UIElement]:
        marks: dict[int, UIElement] = {}
        mid = 0
        for elem in self.elements:
            mid += 1
            if elem.focusable:
                marks[mid] = elem
        return marks

    def click(self, x: int, y: int) -> str:
        for elem in self.elements:
            if elem.focusable and elem.bbox.contains(x, y):
                self.focused = elem
                if elem.kind == "button":
                    self.last_activated = elem.label
                    return f"button '{elem.label}' activated"
                if elem.kind == "link":
                    return f"link '{elem.label}' followed"
                if elem.kind == "input":
                    return f"input '{elem.label}' focused"
        return f"miss at ({x},{y})"

    def type_text(self, text: str) -> str:
        if self.focused is None:
            return "error: no element focused"
        if self.focused.kind != "input":
            return f"error: '{self.focused.label}' is not an input"
        self.focused.value += text
        return f"typed '{text}' into '{self.focused.label}'"

    def state_signature(self) -> str:
        """Deterministic snapshot of the meaningful state (input values + last button)."""
        parts = [f"{e.dom_id or e.label}={e.value}" for e in self.elements if e.kind == "input"]
        parts.append(f"activated={self.last_activated}")
        blob = "|".join(parts)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_login_screen() -> VirtualScreen:
    return VirtualScreen([
        UIElement("label", "  Login to FleetSim", BoundingBox(1, 1, 48, 2),
                  focusable=False, dom_id="title"),
        UIElement("input", "Username", BoundingBox(5, 4, 38, 2), dom_id="inp-user"),
        UIElement("input", "Password", BoundingBox(5, 7, 38, 2), dom_id="inp-pass"),
        UIElement("button", "Submit", BoundingBox(8, 10, 14, 2), dom_id="btn-submit"),
        UIElement("button", "Cancel", BoundingBox(26, 10, 14, 2), dom_id="btn-cancel"),
    ])


# ==========================================================================
# HARD EXERCISE 1 -- Robust GUI agent runner
# ==========================================================================

IRREVERSIBLE = {"Submit", "Delete", "Buy"}


@dataclass
class Intent:
    action: str                       # "click" | "type"
    target_label: str | None = None
    text: str | None = None


class MockLLM:
    """
    Deterministic stand-in for a vision LLM. It does NOT actually 'think': it
    replays a scripted plan (one Intent per call), which is exactly the contract
    a real Claude computer-use loop would fulfill -- look at the SoM screenshot,
    emit the next Intent. Returns None when the plan is exhausted (== 'done').
    """

    def __init__(self, plan: list[Intent]) -> None:
        self._plan = list(plan)
        self._i = 0

    def next_intent(self, som_view: dict[int, UIElement], goal: str) -> Intent | None:
        if self._i >= len(self._plan):
            return None
        intent = self._plan[self._i]
        self._i += 1
        return intent


class RobustGUIAgent:
    """
    Production-shaped runner. Guarantees:
      - step_budget is never exceeded,
      - loop detection aborts when the same screen state recurs loop_threshold times,
      - clicks re-perceive (set_of_marks) and retry up to max_retries,
      - HITL gate is consulted before any IRREVERSIBLE action,
      - run() NEVER raises: it always returns a structured dict.
    """

    def __init__(self, screen: VirtualScreen, llm: MockLLM,
                 confirm: Callable[[Intent], bool],
                 step_budget: int = 12, max_retries: int = 2,
                 loop_threshold: int = 3) -> None:
        self.screen = screen
        self.llm = llm
        self.confirm = confirm
        self.step_budget = step_budget
        self.max_retries = max_retries
        self.loop_threshold = loop_threshold
        self.log: list[str] = []

    def _log(self, msg: str) -> None:
        self.log.append(msg)

    def _click_once(self, target: str) -> dict:
        """Re-perceive, resolve by label, validate, click."""
        marks = self.screen.set_of_marks()
        resolved = next((m for m, e in marks.items() if e.label == target), None)
        if resolved is None:
            return {"ok": False, "reason": "label_not_found"}
        elem = marks[resolved]
        if elem.label != target:
            return {"ok": False, "reason": "stale_mark"}
        res = self.screen.click(*elem.bbox.center())
        return {"ok": "miss" not in res, "reason": None if "miss" not in res else "miss",
                "result": res}

    def run(self, goal: str) -> dict:
        steps = 0
        seen_counts: dict[str, int] = {}

        try:
            while steps < self.step_budget:
                # Loop detection on the *current* state, before acting.
                sig = self.screen.state_signature()
                seen_counts[sig] = seen_counts.get(sig, 0) + 1
                if seen_counts[sig] >= self.loop_threshold:
                    self._log(f"loop detected (state x{seen_counts[sig]}) -> abort")
                    return self._result("loop_aborted", steps)

                marks = self.screen.set_of_marks()
                intent = self.llm.next_intent(marks, goal)
                if intent is None:
                    return self._result("success", steps)

                steps += 1

                # HITL gate before irreversible clicks.
                if intent.action == "click" and intent.target_label in IRREVERSIBLE:
                    if not self.confirm(intent):
                        self._log(f"HITL refused '{intent.target_label}' -> block")
                        return self._result("hitl_blocked", steps)

                if intent.action == "type":
                    res = self.screen.type_text(intent.text or "")
                    self._log(f"type '{intent.text}' -> {res}")
                    continue

                if intent.action == "click":
                    attempt, outcome = 0, self._click_once(intent.target_label)
                    while not outcome["ok"] and attempt < self.max_retries:
                        attempt += 1
                        outcome = self._click_once(intent.target_label)  # re-perceives
                    self._log(f"click '{intent.target_label}' ok={outcome['ok']} "
                              f"attempts={attempt + 1}")
                    if not outcome["ok"]:
                        return self._result("failed", steps)
                    continue

            # Budget exhausted without finishing the plan.
            return self._result("budget_exhausted", steps)

        except Exception as exc:  # graceful failure: never propagate
            self._log(f"unexpected error: {exc!r}")
            return self._result("error", steps)

    def _result(self, status: str, steps: int) -> dict:
        return {"status": status, "steps": steps, "log": list(self.log),
                "last_screen": self.screen}


def hard_ex1_robust_runner() -> None:
    print("=" * 60)
    print("HARD Ex1 -- Robust GUI agent (retries/budget/loop/HITL)")
    print("=" * 60)

    login_plan = [
        Intent("click", "Username"),
        Intent("type", text="alice"),
        Intent("click", "Password"),
        Intent("type", text="s3cr3t!"),
        Intent("click", "Submit"),
    ]

    # --- Scenario A: success (HITL says yes on Submit). ---
    sA = make_login_screen()
    agentA = RobustGUIAgent(sA, MockLLM(login_plan), confirm=lambda i: True)
    rA = agentA.run("Log in to FleetSim")
    print(f"\n  Scenario A (success): status={rA['status']} steps={rA['steps']}")
    assert rA["status"] == "success", rA["status"]
    assert next(e for e in sA.elements if e.dom_id == "inp-user").value == "alice"
    assert next(e for e in sA.elements if e.dom_id == "inp-pass").value == "s3cr3t!"
    assert sA.last_activated == "Submit"

    # --- Scenario B: loop detected (re-click same input, state never changes). ---
    loop_plan = [Intent("click", "Username")] * 8   # no typing -> state is identical
    sB = make_login_screen()
    agentB = RobustGUIAgent(sB, MockLLM(loop_plan), confirm=lambda i: True,
                            step_budget=12, loop_threshold=3)
    rB = agentB.run("spin")
    print(f"  Scenario B (loop)   : status={rB['status']} steps={rB['steps']}")
    assert rB["status"] == "loop_aborted", rB["status"]
    assert rB["steps"] < agentB.step_budget, "must abort before exhausting the budget"

    # --- Scenario C: HITL blocks Submit (confirm says no). ---
    sC = make_login_screen()
    confirm_no = lambda i: False
    agentC = RobustGUIAgent(sC, MockLLM(login_plan), confirm=confirm_no)
    rC = agentC.run("Log in to FleetSim")
    print(f"  Scenario C (HITL)   : status={rC['status']} steps={rC['steps']}")
    assert rC["status"] == "hitl_blocked", rC["status"]
    assert sC.last_activated != "Submit", "Submit must NEVER be activated when HITL refuses"
    # The inputs were still filled before the blocked Submit.
    assert next(e for e in sC.elements if e.dom_id == "inp-user").value == "alice"

    # Every run returned a structured dict (graceful, no exception escaped).
    for r in (rA, rB, rC):
        assert set(r) >= {"status", "steps", "log", "last_screen"}

    print("\n  PASS -- success / loop_aborted / hitl_blocked all handled gracefully.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Mini WebArena harness + cumulative error + ablation
# ==========================================================================


@dataclass
class Page:
    """One page of a multi-page task. Clicking goal_label advances to the next page."""
    name: str
    screen: VirtualScreen
    goal_label: str


def make_journey() -> list[Page]:
    """A 4-page e-commerce flow: Search -> Product -> Cart -> Checkout."""
    def page(name: str, target: str, distractor: str) -> Page:
        scr = VirtualScreen([
            UIElement("button", target, BoundingBox(5, 4, 18, 2), dom_id=f"btn-{target}"),
            UIElement("button", distractor, BoundingBox(26, 4, 18, 2), dom_id=f"btn-{distractor}"),
        ])
        return Page(name, scr, target)

    return [
        page("Search", "Search", "Filters"),
        page("Product", "AddToCart", "Reviews"),
        page("Cart", "GoToCheckout", "ContinueShopping"),
        page("Checkout", "PlaceOrder", "EditAddress"),
    ]


class RawPixelGrounding:
    """
    Pixel grounding: with probability error_rate, clicks a NEIGHBOR element
    instead of the target (imprecise coordinate estimation, theory 6).
    """
    name = "pixels"

    def __init__(self, error_rate: float) -> None:
        self.error_rate = error_rate

    def choose(self, page: Page, rng: random.Random) -> str:
        if rng.random() < self.error_rate:
            others = [e.label for e in page.screen.elements if e.label != page.goal_label]
            return rng.choice(others) if others else page.goal_label
        return page.goal_label


class SoMGrounding:
    """
    Set-of-marks grounding: choosing a DISCRETE id is far more reliable, so the
    error_rate is strictly lower than the pixel strategy.
    """
    name = "set-of-marks"

    def __init__(self, error_rate: float) -> None:
        self.error_rate = error_rate

    def choose(self, page: Page, rng: random.Random) -> str:
        if rng.random() < self.error_rate:
            others = [e.label for e in page.screen.elements if e.label != page.goal_label]
            return rng.choice(others) if others else page.goal_label
        return page.goal_label


def run_task(journey: list[Page], grounding, rng: random.Random) -> bool:
    """
    Cumulative error: one wrong click on any page fails the WHOLE task (the agent
    cannot recover the expected next page). Success requires all clicks correct.
    """
    for page in journey:
        clicked = grounding.choose(page, rng)
        if clicked != page.goal_label:
            return False   # error propagates: task is lost here
    return True


def success_rate(grounding_factory, n_trials: int) -> float:
    wins = 0
    for seed in range(n_trials):
        rng = random.Random(seed)               # deterministic, per-trial seed
        journey = make_journey()
        if run_task(journey, grounding_factory(), rng):
            wins += 1
    return wins / n_trials


def hard_ex2_webarena_ablation() -> None:
    print("=" * 60)
    print("HARD Ex2 -- Mini WebArena: cumulative error + SoM ablation")
    print("=" * 60)

    n_trials = 200
    pixel_err, som_err = 0.25, 0.05
    assert som_err < pixel_err, "SoM must be more reliable than raw pixels"

    sr_pixel = success_rate(lambda: RawPixelGrounding(pixel_err), n_trials)
    sr_som = success_rate(lambda: SoMGrounding(som_err), n_trials)

    print(f"\n  4-page task, {n_trials} trials:")
    print(f"    raw-pixel   (err={pixel_err}) success_rate = {sr_pixel:.3f}")
    print(f"    set-of-marks(err={som_err}) success_rate = {sr_som:.3f}")

    # Ablation result: SoM strictly beats pixel grounding.
    assert sr_som > sr_pixel, "SoM grounding must raise success vs raw pixels"

    # Cumulative-error curve: empirical ~ (1 - err) ** n_pages.
    print("\n  Cumulative-error curve (success ~ (1-err)^n_pages):")
    print(f"    {'n_pages':>8} | {'pixel(emp)':>11} {'pixel(theo)':>12} | "
          f"{'som(emp)':>9} {'som(theo)':>10}")
    for n_pages in (1, 2, 4, 8):
        sr_p = _success_rate_npages(RawPixelGrounding, pixel_err, n_pages, n_trials)
        sr_s = _success_rate_npages(SoMGrounding, som_err, n_pages, n_trials)
        theo_p = (1 - pixel_err) ** n_pages
        theo_s = (1 - som_err) ** n_pages
        print(f"    {n_pages:>8} | {sr_p:>11.3f} {theo_p:>12.3f} | "
              f"{sr_s:>9.3f} {theo_s:>10.3f}")
        # Empirical tracks theory within a tolerance.
        assert abs(sr_p - theo_p) < 0.08, (n_pages, sr_p, theo_p)
        assert abs(sr_s - theo_s) < 0.08, (n_pages, sr_s, theo_s)

    # Longer tasks crush the weaker strategy (the WebArena gap, theory 7.1).
    sr_p8 = _success_rate_npages(RawPixelGrounding, pixel_err, 8, n_trials)
    assert sr_p8 < 0.2, "8-step pixel task should be mostly failing (cumulative error)"

    print("\n  PASS -- SoM > pixels, and success decays as (1-err)^n_pages.\n")


def _make_journey_n(n_pages: int) -> list[Page]:
    base = make_journey()
    pages: list[Page] = []
    for i in range(n_pages):
        src = base[i % len(base)]
        # Fresh screen each time so trials don't share mutable state.
        scr = VirtualScreen([
            UIElement("button", e.label, BoundingBox(e.bbox.x, e.bbox.y, e.bbox.width,
                                                     e.bbox.height), dom_id=e.dom_id)
            for e in src.screen.elements
        ])
        pages.append(Page(f"{src.name}-{i}", scr, src.goal_label))
    return pages


def _success_rate_npages(grounding_cls, err: float, n_pages: int, n_trials: int) -> float:
    wins = 0
    for seed in range(n_trials):
        rng = random.Random(seed * 1000 + n_pages)   # distinct stream per (seed, n_pages)
        journey = _make_journey_n(n_pages)
        if run_task(journey, grounding_cls(err), rng):
            wins += 1
    return wins / n_trials


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 22 HARD Solutions -- Computer use & GUI agents")
    print("#" * 60 + "\n")

    hard_ex1_robust_runner()
    hard_ex2_webarena_ablation()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

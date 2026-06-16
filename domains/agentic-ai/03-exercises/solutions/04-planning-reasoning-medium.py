"""
Solutions -- Day 4 (MEDIUM): Planning & Reasoning

Contains solutions for:
  - Medium Ex 1: Plan-and-execute with replanning on new information
  - Medium Ex 2: Self-consistency with robust extraction + tie-breaking
  - Medium Ex 3: Cost-aware reasoning-strategy router

Everything uses a deterministic MockLLM so the file RUNS OFFLINE with zero
dependencies and zero API keys (same convention as 02-code/04-planning-reasoning.py).

Run:  python 03-exercises/solutions/04-planning-reasoning-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


# ==========================================================================
# SHARED -- Plan parsing (same shape as the day-4 code)
# ==========================================================================

@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: str) -> "Plan":
        steps = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("STEP"):
                _, _, rest = line.partition(":")
                if rest.strip():
                    steps.append(rest.strip())
        return cls(steps=steps)


# ==========================================================================
# MEDIUM EXERCISE 1 -- Plan-and-execute with replanning
# ==========================================================================
#
# The key idea of the module: a linear plan is brittle. When a step reveals
# information that invalidates the plan, a robust agent REPLANS instead of
# pushing a doomed plan to the end.

class ReplanMockLLM:
    """
    Deterministic mock. Produces an INITIAL plan and a REVISED plan for the
    same task, plus executor/synthesizer responses. The flight search returns
    REPLAN_NEEDED the first time (budget exceeded) and a hit the second time
    (flexible dates), which is what forces the single replan in the scenario.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        p = prompt.lower()

        # --- Planner: initial plan (no history of a failed attempt yet) ---
        if "planner" in p and "history" not in p:
            return (
                "STEP 1: Search flights Paris-Tokyo on fixed dates.\n"
                "STEP 2: Pick the cheapest under 800 EUR.\n"
                "STEP 3: Format the booking confirmation."
            )

        # --- Replanner: revised plan AFTER a REPLAN_NEEDED observation ---
        if ("replanner" in p) or ("planner" in p and "history" in p):
            return (
                "STEP 1: Search flights Paris-Tokyo with FLEXIBLE dates.\n"
                "STEP 2: Pick the cheapest under 800 EUR.\n"
                "STEP 3: Format the booking confirmation."
            )

        # --- Executor steps ---
        if "execute step" in p:
            if "fixed dates" in p:
                # Budget blown on fixed dates -> ask for a replan
                return "RESULT: REPLAN_NEEDED: cheapest fixed-date flight is 950 EUR (over 800 EUR budget)"
            if "flexible dates" in p:
                return "RESULT: found CDG->HND 690 EUR departing Tue (flexible dates)"
            if "pick the cheapest" in p:
                return "RESULT: selected CDG->HND 690 EUR"
            if "format the booking" in p:
                return "RESULT: Booking ready: Paris(CDG)->Tokyo(HND) for 690 EUR, flexible dates."
            return "RESULT: step done"

        # --- Synthesizer ---
        if "synthesizer" in p:
            return ("Vol reserve : Paris (CDG) -> Tokyo (HND) pour 690 EUR avec "
                    "dates flexibles (sous le budget de 800 EUR).")

        return "MOCK: unknown prompt"


def execute_step_replan(llm: Callable, step: str, scratchpad: dict) -> str:
    ctx = ", ".join(f"{k}={v}" for k, v in scratchpad.items())
    raw = llm(f"Execute step: {step}\nScratchpad so far: {ctx}")
    # Strip a leading RESULT: tag for readability
    result = raw.partition("RESULT:")[2].strip() or raw
    print(f"    {step[:45]:45s} -> {result[:60]}")
    return result


def plan_execute_replan(llm: Callable, question: str, max_replans: int = 2) -> dict:
    """
    Plan -> execute. If a step returns REPLAN_NEEDED, stop, send the history
    to the replanner, and restart execution on the NEW plan. Returns a report.
    """
    print("\n[Planner] initial plan")
    plan = Plan.parse(llm(f"You are a planner. Plan: {question}"))
    for i, s in enumerate(plan.steps, 1):
        print(f"    {i}. {s}")

    replan_count = 0
    history: list[str] = []

    while True:
        scratchpad: dict = {}
        results: list[str] = []
        replan_triggered = False

        print(f"\n[Executor] running plan (replan #{replan_count})")
        for step in plan.steps:
            result = execute_step_replan(llm, step, scratchpad)
            results.append(result)
            history.append(result)
            if "REPLAN_NEEDED" in result:
                replan_triggered = True
                break  # Abandon the rest of this doomed plan

        if not replan_triggered:
            break  # Plan completed without invalidation

        if replan_count >= max_replans:
            print(f"[Replan] WARNING: max_replans={max_replans} reached, "
                  f"returning best effort.")
            return {"answer": "Could not satisfy the task within replan budget.",
                    "replan_count": replan_count, "success": False}

        replan_count += 1
        print(f"\n[Replanner] replan triggered (#{replan_count}) -- history-aware")
        plan = Plan.parse(llm(
            f"You are a replanner. Plan: {question}\nHistory of failed steps: {history}"))
        for i, s in enumerate(plan.steps, 1):
            print(f"    {i}. {s}")

    print("\n[Synthesizer]")
    answer = llm(f"You are a synthesizer. Question: {question}\nResults: {results}")
    print(f"    {answer}")
    return {"answer": answer, "replan_count": replan_count, "success": True}


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Plan-and-execute with replanning")
    print("=" * 70)

    llm = ReplanMockLLM()
    report = plan_execute_replan(
        llm, "Reserve un vol Paris-Tokyo sous 800 EUR.", max_replans=2)

    # The scenario must trigger exactly ONE replan and end successfully.
    assert report["success"], "Task should succeed after replanning"
    assert report["replan_count"] == 1, f"Expected 1 replan, got {report['replan_count']}"
    assert "690" in report["answer"], "Answer must come from the revised plan"
    print("\n[Verification] PASS -- 1 replan, final answer from the revised plan")

    # Sanity: a task that succeeds first try should trigger ZERO replans.
    class NoFailLLM(ReplanMockLLM):
        def __call__(self, prompt, temperature=0.0):
            if "execute step" in prompt.lower() and "fixed dates" in prompt.lower():
                return "RESULT: found CDG->HND 690 EUR fixed dates"
            return super().__call__(prompt, temperature)

    r2 = plan_execute_replan(NoFailLLM(), "Reserve un vol Paris-Tokyo sous 800 EUR.")
    assert r2["replan_count"] == 0, "No replan expected when first plan succeeds"
    print("[Verification] PASS -- happy path triggers zero replans")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Robust self-consistency vote
# ==========================================================================

_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def extract_answer(text: str) -> str | None:
    """
    Robust extraction across several formats. Returns None on abstention.
    Tries, in order: 'Reponse/Answer/Final: X', '= X', a trailing word number,
    then the last bare number in the text.
    """
    t = text.strip()
    if not t:
        return None

    def _first_token(s: str) -> str:
        """Take the first answer-like token: a number or a spelled-out number."""
        s = s.strip()
        mm = re.match(r"-?\d+(?:\.\d+)?", s)
        if mm:
            return mm.group(0)
        first_word = s.split()[0] if s.split() else s
        return first_word

    # 1. Labelled answer (FR/EN) -- keep only the first answer token
    m = re.search(r"(?:reponse|answer|final|result)\s*[:=]\s*([^\n.]+)", t, re.I)
    if m:
        return _first_token(m.group(1))

    # 2. Equation form '... = X' -- first token after '='
    m = re.search(r"=\s*([^\n.]+?)\s*$", t)
    if m:
        return _first_token(m.group(1))

    # 3. A spelled-out number anywhere
    for word in _WORD_NUMBERS:
        if re.search(rf"\b{word}\b", t, re.I):
            return word

    # 4. Last bare number
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if nums:
        return nums[-1]

    return None  # Abstention


def normalize_answer(ans: str | None) -> str | None:
    """Canonicalise: strip punctuation, map words->digits, drop trailing .0."""
    if ans is None:
        return None
    a = ans.strip().strip(".").strip().lower()
    if a in _WORD_NUMBERS:
        a = _WORD_NUMBERS[a]
    # 13.0 -> 13 ; keep non-integer floats as-is
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def vote(responses: list[str], tie_break: str = "confidence") -> dict:
    """
    Majority vote over normalized answers, ignoring abstentions.
    tie_break in {"first", "confidence", "abstain"}.
    """
    extracted = [(r, normalize_answer(extract_answer(r))) for r in responses]
    valid = [(r, a) for r, a in extracted if a is not None]
    abstentions = len(extracted) - len(valid)

    counts = Counter(a for _, a in valid)
    if not counts:
        return {"winner": None, "votes": {}, "abstentions": abstentions, "confidence": 0.0}

    top = counts.most_common()
    best_n = top[0][1]
    leaders = [ans for ans, n in top if n == best_n]

    if len(leaders) == 1:
        winner = leaders[0]
    elif tie_break == "first":
        # First leader to appear in the original order
        order = [a for _, a in valid]
        winner = next(a for a in order if a in leaders)
    elif tie_break == "confidence":
        # Longest average reasoning among tied answers = highest "effort"
        def avg_len(ans):
            lens = [len(r) for r, a in valid if a == ans]
            return sum(lens) / len(lens)
        winner = max(leaders, key=avg_len)
    elif tie_break == "abstain":
        winner = None
    else:
        raise ValueError(f"Unknown tie_break: {tie_break}")

    confidence = (best_n / len(valid)) if winner is not None and len(valid) else 0.0
    return {
        "winner": winner,
        "votes": dict(counts),
        "abstentions": abstentions,
        "confidence": round(confidence, 3),
    }


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Robust self-consistency vote")
    print("=" * 70)

    # 7 heterogeneous responses, clear majority for 13
    responses = [
        "Step 1... Step 2... Reponse : 13",
        "I computed it carefully and the Answer: thirteen",
        "blah blah = 13.0",
        "Final: 12",
        "I am not sure how to answer.",   # abstention (None)
        "...the last number I trust is 13",
        "",                                # abstention (None)
    ]
    result = vote(responses, tie_break="confidence")
    print(f"  votes={result['votes']} abstentions={result['abstentions']} "
          f"confidence={result['confidence']}")
    print(f"  winner={result['winner']}")
    assert result["winner"] == "13", result
    assert result["abstentions"] == 2, result
    # 4 valid responses resolve to 13 (digit, word, =13.0, trailing number)
    assert result["votes"]["13"] == 4, result
    print("[Verification] PASS -- formats unified, abstentions excluded")

    # Tie scenario: 13 vs 12 (one each among the valid), test tie-breaks
    tie = [
        "Reponse : 13 because step1 step2 step3 long reasoning here indeed",  # long
        "Answer: 12",                                                          # short
    ]
    conf = vote(tie, tie_break="confidence")
    assert conf["winner"] == "13", f"confidence tie-break should pick the long one: {conf}"
    first = vote(tie, tie_break="first")
    assert first["winner"] == "13", first  # 13 appears first
    abstain = vote(tie, tie_break="abstain")
    assert abstain["winner"] is None, abstain
    print("[Verification] PASS -- tie-breaks: confidence/first pick a winner, abstain returns None")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Cost-aware reasoning-strategy router
# ==========================================================================

_CRITICAL_MARKERS = ("medical", "medicament", "dose", "legal", "budget",
                     "irreversible", "diagnos", "contrat")
_MULTI_STEP_MARKERS = (" et ", " puis ", " then ", " and then ")


class RouterMockLLM:
    """Mock that answers regardless of strategy (we only test routing here)."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        return "Reponse : 42"


class ReasoningRouter:
    """Pick a reasoning strategy from the question, respecting a token budget."""

    COST = {"direct": 1, "cot": 1, "self_consistency": 5, "plan_execute": 5}
    # Cheaper fallback for each expensive strategy
    DOWNGRADE = {"self_consistency": "cot", "plan_execute": "cot"}

    def __init__(self, llm: Callable):
        self.llm = llm

    def classify(self, question: str) -> str:
        q = question.lower()
        if any(m in q for m in _CRITICAL_MARKERS):
            return "critical"
        if any(m in q for m in _MULTI_STEP_MARKERS) or len(q.split()) > 25:
            return "multi_step"
        if re.search(r"\d", q) and re.search(r"[+\-*/x%]|fois|plus|moins|combien", q):
            return "arithmetic"
        return "factual"

    @staticmethod
    def _default_strategy(category: str) -> str:
        return {
            "factual": "direct",
            "arithmetic": "cot",
            "critical": "self_consistency",
            "multi_step": "plan_execute",
        }[category]

    def estimate_cost(self, strategy: str) -> int:
        return self.COST[strategy]

    def route(self, question: str, token_budget: int) -> dict:
        category = self.classify(question)
        strategy = self._default_strategy(category)
        fallback = False
        # token_budget here is a proxy "calls budget" for simplicity
        if self.estimate_cost(strategy) > token_budget and strategy in self.DOWNGRADE:
            strategy = self.DOWNGRADE[strategy]
            fallback = True
        return {
            "category": category,
            "chosen_strategy": strategy,
            "fallback_applied": fallback,
            "estimated_calls": self.estimate_cost(strategy),
        }

    def run(self, question: str, token_budget: int) -> dict:
        routing = self.route(question, token_budget)
        strat = routing["chosen_strategy"]
        # Execute the chosen strategy for real (mock LLM under the hood)
        n_calls = self.COST[strat]
        answers = [self.llm(f"[{strat}] {question}") for _ in range(n_calls)]
        answer = Counter(answers).most_common(1)[0][0]
        return {**routing, "answer": answer}


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Cost-aware reasoning-strategy router")
    print("=" * 70)

    router = ReasoningRouter(RouterMockLLM())
    cases = [
        ("Quelle est la capitale du Japon ?", 10, "factual", "direct"),
        ("Combien font 17 fois 23 ?", 10, "arithmetic", "cot"),
        ("Quelle dose de ce medicament est sans danger ?", 10, "critical", "self_consistency"),
        ("Cherche la population puis calcule la densite et formate le resultat", 10, "multi_step", "plan_execute"),
        # Tight budget forces downgrade of the critical (self_consistency) case
        ("Decision legale irreversible a prendre", 2, "critical", "cot"),
    ]
    for question, budget, exp_cat, exp_strat in cases:
        r = router.run(question, budget)
        print(f"  [{r['category']:11s}] budget={budget} -> {r['chosen_strategy']:16s} "
              f"(fallback={r['fallback_applied']}, calls={r['estimated_calls']})")
        assert r["category"] == exp_cat, f"{question}: cat {r['category']} != {exp_cat}"
        assert r["chosen_strategy"] == exp_strat, f"{question}: strat {r['chosen_strategy']} != {exp_strat}"

    # The last case must have downgraded (fallback_applied True)
    downgraded = router.route("Decision legale irreversible a prendre", 2)
    assert downgraded["fallback_applied"] is True
    # An affordable critical case must NOT downgrade
    afforded = router.route("Decision legale irreversible a prendre", 10)
    assert afforded["fallback_applied"] is False
    print("[Verification] PASS -- categories, strategies and budget downgrade correct")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 4 MEDIUM Solutions -- Planning & Reasoning")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

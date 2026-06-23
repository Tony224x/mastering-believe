"""
Solutions -- Day 4 (HARD): Planning & Reasoning

Contains solutions for:
  - Hard Ex 1: Tree-of-Thought with beam search + pruning (Game of 24)
  - Hard Ex 2: Multi-strategy self-evaluated reasoning pipeline
               (direct -> CoT + self-consistency -> grounded reflexion)

Everything runs OFFLINE with zero dependencies. The "LLM evaluator" in ToT
and the noisy sampler in the pipeline are deterministic mocks -- in production
those would be LLM calls, and the code marks exactly where.

Run:  python 03-exercises/solutions/04-planning-reasoning-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable


# ==========================================================================
# HARD EXERCISE 1 -- Tree-of-Thought with beam search (Game of 24)
# ==========================================================================
#
# ToT mapped onto Game of 24:
#   - State            = (remaining numbers, list of operation strings)
#   - Thought gen      = pick 2 numbers + 1 op, replace them by the result
#   - State evaluation = heuristic score (proxy for an LLM "is this promising?")
#   - Search           = beam search keeping the top `beam_width` per depth
#   - Pruning          = everything outside the beam is discarded

EPS = 1e-6
TARGET = 24.0


@dataclass(frozen=True)
class ToTState:
    numbers: tuple          # remaining numbers (floats)
    ops: tuple = ()         # list of human-readable operation strings

    def is_solution(self) -> bool:
        return len(self.numbers) == 1 and abs(self.numbers[0] - TARGET) < EPS


def generate_children(state: ToTState) -> list[ToTState]:
    """All states reachable by combining one PAIR with one of +,-,*,/."""
    children = []
    nums = state.numbers
    n = len(nums)
    for i, j in combinations(range(n), 2):
        a, b = nums[i], nums[j]
        rest = tuple(nums[k] for k in range(n) if k not in (i, j))
        # Both orderings matter for - and /
        candidates = [
            (a + b, f"{a:g}+{b:g}={a + b:g}"),
            (a * b, f"{a:g}*{b:g}={a * b:g}"),
            (a - b, f"{a:g}-{b:g}={a - b:g}"),
            (b - a, f"{b:g}-{a:g}={b - a:g}"),
        ]
        if abs(b) > EPS:
            candidates.append((a / b, f"{a:g}/{b:g}={a / b:g}"))
        if abs(a) > EPS:
            candidates.append((b / a, f"{b:g}/{a:g}={b / a:g}"))
        for value, label in candidates:
            children.append(ToTState(rest + (value,), state.ops + (label,)))
    return children


def _closeness(value: float) -> float:
    """Cheap heuristic: how close a single number is to 24 or a useful factor."""
    useful = [24, 12, 8, 6, 4, 3, 2, 1]
    return max(1.0 / (1.0 + abs(value - u)) for u in useful)


def evaluate_cheap(state: ToTState) -> float:
    """
    A WEAK evaluator (proxy for a cheap/low-effort LLM judge). It only looks at
    how close the current numbers are to useful values -- it has no lookahead,
    so it readily prunes the winning path. A greedy beam relying on it FAILS.
    """
    if state.is_solution():
        return 1.0
    return max((_closeness(x) for x in state.numbers), default=0.0)


def evaluate_strong(state: ToTState) -> float:
    """
    A STRONG evaluator (proxy for an expensive/high-effort LLM judge). It does a
    bounded recursive solvability probe: 1.0 if 24 is reachable from this state,
    else it falls back to the cheap closeness score. With a strong judge, a
    modest beam keeps the winning path alive.
    """
    if _reachable_24(state.numbers):
        return 1.0
    return 0.5 * max((_closeness(x) for x in state.numbers), default=0.0)


def _reachable_24(nums: tuple) -> bool:
    """Exact recursive solver over <=4 numbers -- cheap, runs offline."""
    if len(nums) == 1:
        return abs(nums[0] - TARGET) < EPS
    n = len(nums)
    for i, j in combinations(range(n), 2):
        a, b = nums[i], nums[j]
        rest = tuple(nums[k] for k in range(n) if k not in (i, j))
        results = [a + b, a * b, a - b, b - a]
        if abs(b) > EPS:
            results.append(a / b)
        if abs(a) > EPS:
            results.append(b / a)
        for r in results:
            if _reachable_24(rest + (r,)):
                return True
    return False


def tot_solve(start_numbers: tuple, beam_width: int = 5,
              evaluator: Callable[[ToTState], float] = evaluate_strong) -> dict:
    """
    Beam search over the ToT. Returns the solution path (if found), the number
    of explored nodes, and whether a solution was reached.

    `evaluator` is the pluggable state-evaluation function (the 'LLM judge').
    """
    start = ToTState(tuple(float(x) for x in start_numbers))
    beam = [start]
    explored = 0

    # Depth = number of combinations needed to collapse to a single number
    for _depth in range(len(start_numbers) - 1):
        next_level: list[ToTState] = []
        for node in beam:
            children = generate_children(node)
            explored += len(children)
            next_level.extend(children)

        # Check for a solution at this level
        for node in next_level:
            if node.is_solution():
                return {"found": True, "path": list(node.ops),
                        "explored": explored, "beam_width": beam_width}

        # Prune: keep only the top `beam_width` by evaluation score
        next_level.sort(key=evaluator, reverse=True)
        beam = next_level[:beam_width]

    return {"found": False, "path": [], "explored": explored,
            "beam_width": beam_width}


def verify_path(start_numbers: tuple, path: list[str]) -> bool:
    """
    Replay the operations against the actual multiset of numbers and confirm
    the final value is 24. Each label is 'a op b=result'; we consume a and b
    from the available pool and push the result, ensuring the path is valid.
    """
    import re as _re
    pool = [float(x) for x in start_numbers]
    for label in path:
        lhs, _, result = label.partition("=")
        result = float(result)
        # Parse the two operands from the left-hand side (a OP b)
        m = _re.match(r"(-?\d+(?:\.\d+)?)[+\-*/](-?\d+(?:\.\d+)?)", lhs)
        if not m:
            return False
        a, b = float(m.group(1)), float(m.group(2))
        # Consume a and b from the pool
        for v in (a, b):
            matched = next((x for x in pool if abs(x - v) < 1e-4), None)
            if matched is None:
                return False
            pool.remove(matched)
        pool.append(result)
    return len(pool) == 1 and abs(pool[0] - TARGET) < EPS


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Tree-of-Thought with beam search (Game of 24)")
    print("=" * 70)
    print("  Two knobs decide success: the JUDGE quality and the BEAM width.")

    games = [(4, 6, 8, 2), (1, 3, 4, 6)]  # second needs 6/(1-3/4)=24

    for game in games:
        # Weak judge + greedy beam: prunes the winning path -> fails.
        weak = tot_solve(game, beam_width=1, evaluator=evaluate_cheap)
        # Strong judge + modest beam: keeps the winning path -> solves.
        strong = tot_solve(game, beam_width=5, evaluator=evaluate_strong)
        print(f"\n  Game {game}:")
        print(f"    weak judge,  beam=1: found={str(weak['found']):<5} "
              f"explored={weak['explored']}")
        print(f"    strong judge, beam=5: found={str(strong['found']):<5} "
              f"explored={strong['explored']}")
        if strong["found"]:
            print(f"    solution path: {' | '.join(strong['path'])}")
            assert verify_path(game, strong["path"]), "Solution path does not reach 24!"
        # The contrast must hold: weak/greedy fails where strong/wide succeeds.
        assert strong["found"], f"strong judge + beam=5 must solve {game}"
        assert not weak["found"], f"weak judge + greedy should fail {game}"

    # Pruning really bounds exploration: a wider beam explores more nodes
    # (more survivors per level -> more children generated next level).
    narrow = tot_solve((9, 9, 9, 9), beam_width=1, evaluator=evaluate_strong)
    broad = tot_solve((9, 9, 9, 9), beam_width=8, evaluator=evaluate_strong)
    assert broad["explored"] >= narrow["explored"], "Wider beam explores more"
    print(f"\n  Pruning effect on (9,9,9,9): beam=1 explored {narrow['explored']}, "
          f"beam=8 explored {broad['explored']}")
    print("[Verification] PASS -- judge quality + beam width drive success; "
          "pruning bounds exploration")


# ==========================================================================
# HARD EXERCISE 2 -- Multi-strategy self-evaluated reasoning pipeline
# ==========================================================================
#
# A mini-benchmark with LOCALLY VERIFIABLE ground truth lets us MEASURE the
# accuracy gain of each stage without any real LLM:
#   Stage 1: direct baseline
#   Stage 2: CoT + self-consistency (vote over noisy samples)
#   Stage 3: grounded reflexion (execution-based critique) on remaining fails

BENCHMARK = [
    ("3 shirts cost 45, budget 200, how many shirts?", 13),
    ("15 apples, 3 baskets of 5, how many left over?", 0),
    ("Train at 80km/h for 3 hours, distance?", 240),
    ("A is taller than B, who is shorter?", "b"),
    ("7 times 6 equals?", 42),
]


def verify(answer, ground_truth) -> bool:
    """Deterministic local judge (this is the 'execution-based' critic)."""
    if isinstance(ground_truth, str):
        return str(answer).strip().lower() == ground_truth
    try:
        return abs(float(answer) - float(ground_truth)) < EPS
    except (ValueError, TypeError):
        return False


class NoisyMockLLM:
    """
    Deterministic-but-noisy 'LLM'. With a per-call seed it returns the correct
    answer with probability `p_correct`, otherwise a plausible wrong one.
    CoT/self-consistency exploits this: voting over samples beats one shot.
    """

    def __init__(self, p_correct: float = 0.6, seed: int = 0):
        self.p_correct = p_correct
        self.rng = random.Random(seed)
        self.call_count = 0

    def answer(self, problem_idx: int, mode: str) -> object:
        self.call_count += 1
        _, truth = BENCHMARK[problem_idx]
        # 'direct' is noisier than 'cot' (reasoning helps) -- models the module.
        p = self.p_correct if mode == "cot" else self.p_correct - 0.25
        if self.rng.random() < p:
            return truth
        # Wrong answer
        if isinstance(truth, str):
            return "a"
        return float(truth) + self.rng.choice([-2, -1, 1, 2, 10])


def stage_direct(llm: NoisyMockLLM) -> list:
    return [llm.answer(i, "direct") for i in range(len(BENCHMARK))]


def stage_self_consistency(llm: NoisyMockLLM, n: int = 5) -> list:
    answers = []
    for i in range(len(BENCHMARK)):
        samples = [llm.answer(i, "cot") for _ in range(n)]
        # Majority vote (stringified for hashability)
        from collections import Counter
        winner = Counter(str(s) for s in samples).most_common(1)[0][0]
        answers.append(winner)
    return answers


def stage_reflexion(llm: NoisyMockLLM, current: list, max_retry: int = 2) -> list:
    """
    Grounded reflexion: only retry the problems that are STILL wrong, using the
    verifiable judge as the critic. Each retry is a fresh self-consistency pass.
    """
    refined = list(current)
    for i in range(len(BENCHMARK)):
        _, truth = BENCHMARK[i]
        retry = 0
        while not verify(refined[i], truth) and retry < max_retry:
            samples = [llm.answer(i, "cot") for _ in range(5)]
            from collections import Counter
            refined[i] = Counter(str(s) for s in samples).most_common(1)[0][0]
            retry += 1
    return refined


def accuracy(answers: list) -> float:
    correct = sum(verify(a, BENCHMARK[i][1]) for i, a in enumerate(answers))
    return correct / len(BENCHMARK)


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- Multi-strategy self-evaluated reasoning pipeline")
    print("=" * 70)

    # One shared, seeded LLM so the whole run is reproducible.
    llm = NoisyMockLLM(p_correct=0.6, seed=42)

    calls_0 = llm.call_count
    direct = stage_direct(llm)
    acc_direct = accuracy(direct)
    calls_direct = llm.call_count - calls_0

    calls_1 = llm.call_count
    sc = stage_self_consistency(llm, n=5)
    acc_sc = accuracy(sc)
    calls_sc = llm.call_count - calls_1

    calls_2 = llm.call_count
    final = stage_reflexion(llm, sc, max_retry=2)
    acc_final = accuracy(final)
    calls_refl = llm.call_count - calls_2

    print(f"\n  {'Stage':<28} {'Accuracy':>10} {'LLM calls':>10}")
    print("  " + "-" * 50)
    print(f"  {'1. direct baseline':<28} {acc_direct:>9.0%} {calls_direct:>10}")
    print(f"  {'2. CoT + self-consistency':<28} {acc_sc:>9.0%} {calls_sc:>10}")
    print(f"  {'3. grounded reflexion':<28} {acc_final:>9.0%} {calls_refl:>10}")

    total_calls = llm.call_count
    gained = (acc_final - acc_direct) * len(BENCHMARK)  # problems gained
    efficiency = gained / total_calls if total_calls else 0.0
    print(f"\n  Total LLM calls: {total_calls}")
    print(f"  Marginal efficiency: {gained:.1f} problems gained over "
          f"{total_calls} calls = {efficiency:.3f} problem/call")

    # Core criterion: more compute never hurts here -- final >= baseline.
    assert acc_sc >= acc_direct, "Self-consistency should not be worse than direct"
    assert acc_final >= acc_direct, "Pipeline must not regress below baseline"
    assert acc_final == 1.0, f"Grounded reflexion should reach 100%, got {acc_final:.0%}"
    print("[Verification] PASS -- each stage helps, final accuracy 100%, deterministic")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 4 HARD Solutions -- Planning & Reasoning")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

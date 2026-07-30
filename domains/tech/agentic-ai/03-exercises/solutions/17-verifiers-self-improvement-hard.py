"""
Solutions -- Day 17 (HARD): Verifiers & self-improvement

Contains solutions for:
  - Hard Ex 1: Reward hacking -- a gaming generator fools a NAIVE verifier but
               is caught by a HARDENED verifier (internal-consistency, derivation
               trace, adversarial re-execution). Best-of-N contrast included.
  - Hard Ex 2: Full generate -> verify -> refine pipeline with an EXECUTION-BASED
               verifier (real in-memory unit tests). Proves convergence on a
               solvable problem and clean rejection of an unsolvable one.
  - Hard Ex 3: Autonomous Execution Tasks (Kimi K3, 2026) -- reward read off the
               ENVIRONMENT STATE rather than the agent's report, public vs
               hidden verifiers, and submission budgets. Three independent
               defenses, each proved by assertion.

stdlib only, fully offline, deterministic. The execution-based verifier uses
exec() in an isolated namespace and captures every exception so the loop never
crashes on a bad candidate.

Run:  python 03-exercises/solutions/17-verifiers-self-improvement-hard.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Tiny shared helpers (re-embedded -- file is self-contained)
# ---------------------------------------------------------------------------

def _safe_eval(expr: str) -> float | None:
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def _extract_answer(candidate: str) -> str:
    for line in candidate.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    lines = [ln for ln in candidate.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ==========================================================================
# HARD EXERCISE 1 -- Reward hacking: naive vs hardened verifier
# ==========================================================================

# A reasoning step is written "a op b = c". This regex pulls (a, op, b, c).
_STEP_RE = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)")


class NaiveVerifier:
    """
    Outcome-only verifier: 1.0 if 'answer: X' equals the target. It NEVER looks
    at the steps -- which is exactly the hole a gaming generator exploits.
    """

    def __init__(self, target: float) -> None:
        self.target = target

    def score(self, candidate: str) -> float:
        val = _safe_eval(_extract_answer(candidate))
        if val is None:
            return 0.0
        return 1.0 if math.isclose(val, self.target, rel_tol=1e-9) else 0.0


class HardenedVerifier:
    """
    Process-aware verifier with three anti-gaming controls:
      (1) internal consistency  -- every 'a op b = c' must actually evaluate to c
      (2) derivation trace      -- the final answer must appear as a step result,
                                   not materialize out of nowhere
      (3) adversarial re-exec    -- independently recompute each claimed result;
                                   a single mismatch invalidates the candidate
    """

    def __init__(self, target: float) -> None:
        self.target = target

    def _steps(self, candidate: str) -> list[tuple[int, str, int, int]]:
        out = []
        for m in _STEP_RE.finditer(candidate):
            a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            out.append((a, op, b, c))
        return out

    def score(self, candidate: str) -> float:
        steps = self._steps(candidate)
        # Control (2): there must be real derivation steps at all.
        if not steps:
            return 0.0

        # Controls (1) + (3): re-execute each step independently.
        ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}
        for a, op, b, c in steps:
            real = ops[op](a, b)
            if real != c:                      # claimed result is a lie
                return 0.0                     # one bad step invalidates everything

        # Final answer must be correct AND match a verified step result.
        ans = _safe_eval(_extract_answer(candidate))
        if ans is None or not math.isclose(ans, self.target, rel_tol=1e-9):
            return 0.0
        step_results = {c for *_, c in steps}
        # Control (2): the answer must be backed by a derivation step.
        if int(ans) not in step_results:
            return 0.3                         # correct number but underived -> suspicious
        return 1.0                             # consistent, derived, correct


def gaming_generator(target: float) -> str:
    """Cheats: incoherent steps + the right number copied into the answer."""
    t = int(target)
    return (
        f"step1: 2 * 2 = 5\n"          # blatant lie
        f"step2: 7 + 7 = 100\n"        # another lie
        f"answer: {t}"                 # but the final number is right
    )


def honest_generator(target: float) -> str:
    """Honestly derives the target with internally-consistent steps."""
    t = int(target)
    half = t // 2
    return (
        f"step1: {half} + {t - half} = {t}\n"
        f"step2: verify {half} + {t - half} = {t}\n"
        f"answer: {t}"
    )


def best_of_n(generator_pool: list[str], verifier) -> tuple[str, float]:
    """Generic best-of-N over a fixed candidate pool, scored by `verifier`."""
    scores = [verifier.score(c) for c in generator_pool]
    best_idx = max(range(len(generator_pool)), key=lambda i: scores[i])
    return generator_pool[best_idx], scores[best_idx]


def hard_ex1_reward_hacking() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Reward hacking: naive verifier fooled, hardened resists")
    print("=" * 70)

    target = 24.0
    naive = NaiveVerifier(target)
    hardened = HardenedVerifier(target)

    cheat = gaming_generator(target)
    honest = honest_generator(target)

    n_cheat, n_honest = naive.score(cheat), naive.score(honest)
    h_cheat, h_honest = hardened.score(cheat), hardened.score(honest)

    print(f"  NAIVE    -> cheat={n_cheat:.2f}  honest={n_honest:.2f}")
    print(f"  HARDENED -> cheat={h_cheat:.2f}  honest={h_honest:.2f}")

    # The naive verifier cannot tell the cheater from the honest solver.
    assert n_cheat == 1.0 and n_honest == 1.0, (n_cheat, n_honest)
    # The hardened verifier rewards the honest one and punishes the cheater.
    assert h_honest >= 0.9, h_honest
    assert h_cheat <= 0.1, h_cheat
    assert (h_honest - h_cheat) >= 0.8, (h_honest, h_cheat)  # clear separation

    # Best-of-N over a MIXED pool: hardened picks an honest candidate; naive may
    # pick a cheater (here the cheater sits first, so argmax ties go to it).
    pool = [cheat, honest, gaming_generator(target)]
    pick_naive, s_naive = best_of_n(pool, naive)
    pick_hard, s_hard = best_of_n(pool, hardened)

    print(f"  best-of-N (naive)    -> chose {'CHEAT' if pick_naive == cheat or pick_naive == pool[2] else 'HONEST'} (score {s_naive:.2f})")
    print(f"  best-of-N (hardened) -> chose {'HONEST' if pick_hard == honest else 'CHEAT'} (score {s_hard:.2f})")

    # Hardened best-of-N selects the honest candidate from the mixed pool.
    assert pick_hard == honest, "hardened best-of-N should pick the honest candidate"
    # Naive best-of-N elects a cheater (its argmax is fooled).
    assert pick_naive in (cheat, pool[2]), "naive best-of-N was expected to pick a cheater"

    print("[Verification] PASS -- gaming fools naive, hardened separates + selects honest")


# ==========================================================================
# HARD EXERCISE 2 -- generate->verify->refine with execution-based verifier
# ==========================================================================

def run_tests(func: Callable, cases: list[tuple[tuple, object]]) -> dict:
    """
    Run `cases` against `func`, capturing EVERY exception so the loop never
    crashes. Returns {passed, total, failures}.
    """
    passed = 0
    failures: list[dict] = []
    for args, expected in cases:
        try:
            got = func(*args)
            if got == expected:
                passed += 1
            else:
                failures.append({"args": args, "expected": expected, "got": got})
        except Exception as e:  # noqa: BLE001
            failures.append({"args": args, "expected": expected, "error": repr(e)})
    return {"passed": passed, "total": len(cases), "failures": failures}


def load_func(code: str, name: str = "f") -> Callable | None:
    """exec() the candidate in an isolated namespace; return the callable or None."""
    ns: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, ns)  # noqa: S102
    except Exception:
        return None
    fn = ns.get(name)
    return fn if callable(fn) else None


@dataclass
class Problem:
    name: str
    cases: list[tuple[tuple, object]]
    generator: Callable[[int], str]            # attempt -> candidate code
    refiner: Callable[[list[dict], int], str]  # (failures, attempt) -> next code


class ExecutionRefineLoop:
    """generate -> verify (run tests) -> refine, until all pass or budget spent."""

    def run(self, problem: Problem, max_attempts: int = 4) -> dict:
        score_history: list[float] = []
        code = problem.generator(0)
        last_failures: list[dict] = []
        final_code = code

        for attempt in range(max_attempts):
            if attempt > 0:
                code = problem.refiner(last_failures, attempt)
            final_code = code
            fn = load_func(code)
            if fn is None:
                # Unloadable candidate -> score 0, keep iterating within budget.
                score_history.append(0.0)
                last_failures = [{"error": "uncompilable / no function"}]
                continue
            result = run_tests(fn, problem.cases)
            score = result["passed"] / result["total"] if result["total"] else 1.0
            score_history.append(score)
            last_failures = result["failures"]
            if result["passed"] == result["total"]:   # accepted
                return {"converged": True, "attempts_used": attempt + 1,
                        "score_history": score_history, "final_code": final_code}

        return {"converged": False, "attempts_used": max_attempts,
                "score_history": score_history, "final_code": final_code}


# -- A SOLVABLE problem: sum_positive, buggy on the empty list at first --------

def _solvable_generator(attempt: int) -> str:
    # attempt 0: crashes on empty list (max of empty seq) -> some tests fail
    return (
        'def f(xs):\n'
        '    return sum(x for x in xs if x > 0) + (0 * max(xs))\n'
    )


def _solvable_refiner(failures: list[dict], attempt: int) -> str:
    # The fix: guard the empty list, drop the bogus max() term.
    return (
        'def f(xs):\n'
        '    if not xs:\n'
        '        return 0\n'
        '    return sum(x for x in xs if x > 0)\n'
    )


# -- An UNSOLVABLE-by-the-stub problem: the generator never reaches the spec ---

def _unsolvable_generator(attempt: int) -> str:
    # Spec wants sorted output, but the stub stubbornly returns the input as-is.
    return 'def f(xs):\n    return list(xs)\n'


def _unsolvable_refiner(failures: list[dict], attempt: int) -> str:
    # The "refiner" cannot actually fix it (models its limitation): keep failing.
    return 'def f(xs):\n    return list(xs)  # still not sorting\n'


def hard_ex2_execution_refine() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- generate->verify->refine with execution-based verifier")
    print("=" * 70)

    loop = ExecutionRefineLoop()

    # --- Solvable problem: must converge, history non-decreasing up to 1.0 ----
    solvable = Problem(
        name="sum_positive",
        cases=[(([],), 0), (([1, -2, 3],), 4), (([5],), 5), (([-1, -2],), 0)],
        generator=_solvable_generator,
        refiner=_solvable_refiner,
    )
    r1 = loop.run(solvable, max_attempts=4)
    print(f"  [solvable]   converged={r1['converged']} "
          f"attempts={r1['attempts_used']} history={[round(s,2) for s in r1['score_history']]}")

    assert r1["converged"], "solvable problem must converge"
    assert r1["attempts_used"] <= 4
    hist = r1["score_history"]
    assert all(hist[i] <= hist[i + 1] + 1e-12 for i in range(len(hist) - 1)), hist  # non-decreasing
    assert math.isclose(hist[-1], 1.0), hist
    # The returned final code really passes all tests.
    fn = load_func(r1["final_code"])
    assert fn is not None and run_tests(fn, solvable.cases)["failures"] == []

    # --- Unsolvable problem: must NOT fake success, no exception leaks --------
    unsolvable = Problem(
        name="sort_xs",
        cases=[(([3, 1, 2],), [1, 2, 3]), (([2, 1],), [1, 2])],
        generator=_unsolvable_generator,
        refiner=_unsolvable_refiner,
    )
    r2 = loop.run(unsolvable, max_attempts=4)
    print(f"  [unsolvable] converged={r2['converged']} "
          f"attempts={r2['attempts_used']} history={[round(s,2) for s in r2['score_history']]}")

    assert not r2["converged"], "must not fake success on an unsolvable spec"
    assert r2["attempts_used"] == 4, "budget must be fully spent"
    assert all(s < 1.0 for s in r2["score_history"]), r2["score_history"]  # never accepted

    # --- A candidate that crashes must be rejected cleanly (no loop crash) ----
    crasher = Problem(
        name="crash",
        cases=[(([],), 0)],
        generator=lambda a: 'def f(xs):\n    return xs[0]\n',  # IndexError on []
        refiner=lambda f, a: 'def f(xs):\n    return xs[0]\n',
    )
    r3 = loop.run(crasher, max_attempts=2)
    assert not r3["converged"], "crashing candidate must never be accepted"
    print(f"  [crasher]    converged={r3['converged']} "
          f"(exception captured, loop survived)")

    print("[Verification] PASS -- convergence, clean rejection, no crash on bad code")


# ==========================================================================
# HARD EX 3 -- Autonomous Execution Tasks: state, hidden verifier, budget
# ==========================================================================
# Ex 1 hardened the CONTENT of a verifier. This one is about the PROTOCOL
# around it: where the reward comes from, what the agent is allowed to see,
# and how many times it may try. Three independent defenses.


class HiddenSystem:
    """Black box. The agent sees outputs, never the rule. Counts its calls."""

    def __init__(self) -> None:
        self.n_queries = 0

    def __call__(self, n: int) -> str:
        self.n_queries += 1
        if n % 15 == 0:
            return "both"
        if n % 3 == 0:
            return "three"
        if n % 5 == 0:
            return "five"
        return "none"


def generalizing_agent(oracle, probes):
    """Queries, infers a general rule, returns a replica valid everywhere."""
    observations = {n: oracle(n) for n in probes}
    # A real agent searches a hypothesis space; the mock jumps to the answer,
    # because the exercise is about the EVALUATION, not about induction.
    assert len(observations) == len(probes)

    def replica(n: int) -> str:
        if n % 15 == 0:
            return "both"
        if n % 3 == 0:
            return "three"
        if n % 5 == 0:
            return "five"
        return "none"

    return replica


def memorizing_agent(oracle, probes):
    """Records what it was shown, guesses the majority label elsewhere.

    Not a strawman: when the only signal is a fixed, visible set of checks,
    reproducing the answers is strictly easier than finding the rule.
    """
    table = {n: oracle(n) for n in probes}

    def replica(n: int) -> str:
        return table.get(n, "none")

    return replica


def make_verifier(cases, ground_truth):
    """Returns a scorer. The case list stays captured in the closure -- the
    caller receives a score, never the criteria."""

    def verify(replica) -> float:
        return sum(1 for n in cases if replica(n) == ground_truth(n)) / len(cases)

    return verify


def textual_judge(report: str) -> bool:
    """What a purely text-based judge can actually check: plausibility."""
    claims_success = any(k in report.lower()
                         for k in ("tous les cas passent", "replique", "reussi"))
    return claims_success and len(report) > 20


def run_with_submission_budget(attempt_factory, verifier,
                               max_submissions=3, penalty=0.15):
    """reward = best score - penalty * (submissions beyond the first)."""
    best, used = 0.0, 0
    for i in range(max_submissions):
        used = i + 1
        best = max(best, verifier(attempt_factory(i)))
        if best >= 1.0:
            break
    return {"submissions": used, "raw_score": best,
            "reward": best - penalty * (used - 1)}


def hard_ex3_autonomous_execution():
    print("\n" + "=" * 70)
    print("  HARD EX 3 -- Autonomous Execution Tasks (Kimi K3, 2026)")
    print("=" * 70)

    truth = HiddenSystem()
    probes = [1, 3, 5, 9, 15, 20]                 # what the agent may see
    held_out = [7, 12, 25, 30, 33, 45, 50, 99]    # what it may not

    # --- (a) state vs self-report -----------------------------------------
    print("\n  (a) Verifier l'ETAT, pas le compte-rendu")
    cheat = memorizing_agent(HiddenSystem(), probes)
    report = "J'ai replique le systeme, tous les cas passent."

    state_score = sum(1 for n in held_out if cheat(n) == truth(n)) / len(held_out)
    print(f"      compte-rendu           : {report!r}")
    print(f"      juge textuel           : {'ACCEPTE' if textual_judge(report) else 'REJETTE'}")
    print(f"      verifier d'etat        : {state_score:.0%} sur les cas tenus a l'ecart")

    assert textual_judge(report) is True, "le juge textuel doit accepter la phrase"
    assert state_score < 0.5, "le verifier d'etat doit rejeter la replique"
    print("      [OK] la phrase est plausible ; le fait est faux. Seul l'etat est un signal.")

    # --- (b) public vs hidden verifier ------------------------------------
    print("\n  (b) Verifier public + verifier cache")
    public_verifier = make_verifier(probes, truth)
    hidden_verifier = make_verifier(held_out, truth)

    agents = {"generalise la regle": generalizing_agent,
              "memorise les cas vus": memorizing_agent}
    scores = {}
    print(f"      {'agent':<24}{'public':>10}{'cache':>10}")
    for name, factory in agents.items():
        replica = factory(HiddenSystem(), probes)
        scores[name] = (public_verifier(replica), hidden_verifier(replica))
        print(f"      {name:<24}{scores[name][0]:>10.0%}{scores[name][1]:>10.0%}")

    gap_public = abs(scores["generalise la regle"][0] - scores["memorise les cas vus"][0])
    gap_hidden = abs(scores["generalise la regle"][1] - scores["memorise les cas vus"][1])
    print(f"\n      ecart sur le public : {gap_public:.0%}   "
          f"ecart sur le cache : {gap_hidden:.0%}")

    assert gap_public < 0.05, "les deux agents doivent etre indiscernables sur le public"
    assert gap_hidden > 0.5, "le verifier cache doit les separer nettement"
    print("      [OK] les deux agents ne deviennent distinguables QUE grace au cache.")
    print("           C'est le split train/test, applique aux verifiers eux-memes.")

    # --- (c) submission budget --------------------------------------------
    print("\n  (c) Budget de soumissions")

    def guessing(i):        # memorise un peu plus a chaque essai
        return memorizing_agent(HiddenSystem(), probes[: 2 * (i + 1)])

    def reasoning(i):       # raisonne une fois, soumet une fois
        return generalizing_agent(HiddenSystem(), probes)

    print(f"      {'strategie':<24}{'soumissions':>13}{'brut':>8}{'recompense':>13}")
    results = {}
    for name, factory in [("essais successifs", guessing),
                          ("raisonne puis soumet", reasoning)]:
        r = run_with_submission_budget(factory, hidden_verifier)
        results[name] = r
        print(f"      {name:<24}{r['submissions']:>13}{r['raw_score']:>8.0%}"
              f"{r['reward']:>13.2f}")

    assert results["essais successifs"]["submissions"] == 3
    assert results["raisonne puis soumet"]["submissions"] == 1
    assert (results["raisonne puis soumet"]["reward"]
            > results["essais successifs"]["reward"])
    print("      [OK] la penalite ne punit pas l'erreur : elle rend le brute-force")
    print("           couteux. Sans elle, reessayer est gratuit et devient la")
    print("           strategie optimale -- on entraine a deviner, pas a raisonner.")

    # --- (6) synthese : les defenses sont independantes --------------------
    print("\n  (6) Retirer deux defenses sur trois : quel exploit redevient rentable ?")
    print("      sans (a) ni (b) -> il suffit d'AFFIRMER la reussite ; rien ne")
    print("                         confronte la declaration a l'environnement.")
    print("      sans (a) ni (c) -> meme chose, en boucle : on repete l'affirmation")
    print("                         jusqu'a ce qu'un juge textuel la gobe.")
    print("      sans (b) ni (c) -> l'etat est bien mesure, mais toujours sur les")
    print("                         memes cas visibles : on les memorise, en")
    print("                         autant de tentatives qu'il faut.")
    print("      -> aucune des trois ne couvre les angles morts des deux autres.")

    print("\n  (7) Transposition a une boucle de self-refine en production :")
    print("      mesurer  -> l'etat (tests relances, base interrogee, artefact relu),")
    print("                  jamais le resume que l'agent fait de son travail ;")
    print("      cacher   -> une partie des criteres, sinon la boucle converge vers")
    print("                  'passer les checks montres' et non vers 'resoudre' ;")
    print("      borner   -> le nombre de tentatives, pour que reessayer soit une")
    print("                  decision couteuse -- ce qu'elle est reellement.")

    print("\n  [Verification] PASS -- etat vs rapport, public vs cache, budget")

# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  Day 17 HARD Solutions -- Verifiers & self-improvement")
    print("  (stdlib only -- runs fully offline, no API key)")
    print("#" * 70)

    hard_ex1_reward_hacking()
    hard_ex2_execution_refine()
    hard_ex3_autonomous_execution()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70 + "\n")

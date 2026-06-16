"""
Solutions -- Day 28 (HARD): Capstone build & eval extensions

Contains solutions for:
  - Hard Ex 1: golden regression gate -- baseline vs several candidates; the
               gate BLOCKS a candidate that improves the mean but regresses a
               golden case >=0.10, APPROVES a strictly-better one, and an
               ablation of the gate proves the bad candidate would otherwise ship
  - Hard Ex 2: durability under eval -- crash-resume combined with the eval
               harness; across many seeds & crash points, no finished step is
               re-executed (side-effect counter == 1) AND the post-resume verdict
               matches the no-crash verdict (durability preserves correctness)

Self-contained & offline: embeds a faithful mini-DeepOpsAgent + pass^k harness +
DurableEngine/SQLiteCheckpointer (deterministic mock LLM with an injectable
error_rate + seeded random.Random; the coder fix outcome is SIMULATED, not
spawned in a subprocess). It does NOT import 02-code/28-capstone-build-eval.py.

Run:  python 03-exercises/solutions/28-capstone-build-eval-hard.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ==========================================================================
# EMBEDDED MINI-CAPSTONE (offline stand-in for the J28 capstone)
# ==========================================================================


@dataclass
class AgentResult:
    output: str
    trajectory: list[str] = field(default_factory=list)
    steps: int = 0


class DeepOpsAgent:
    """Planner orchestrating isolated sub-agents to fix a planted bug, with a
    stochastic failure (error_rate) so reliability (pass^k) is measurable.

    `golden_killer` lets a candidate be reliable in general but selectively
    degrade on one specific task substring (used to forge a golden regression).
    """

    def __init__(self, label: str = "deep-ops", error_rate: float = 0.0,
                 seed: int = 0, golden_killer: str | None = None,
                 golden_error_rate: float = 1.0) -> None:
        self.label = label
        self.error_rate = max(0.0, min(1.0, error_rate))
        self.golden_killer = golden_killer
        self.golden_error_rate = max(0.0, min(1.0, golden_error_rate))
        self._rng = random.Random(seed)

    def _effective_error_rate(self, task: str) -> float:
        if self.golden_killer and self.golden_killer in task:
            return self.golden_error_rate
        return self.error_rate

    def solve(self, task: str) -> AgentResult:
        traj: list[str] = ["plan", "research", "code", "search"]
        will_fix = self._rng.random() >= self._effective_error_rate(task)
        if will_fix:
            traj.append("edit")
        traj.append("run_tests")
        traj.append("verify")
        output = ("SUCCESS: bug fixed and verified" if will_fix
                  else "FAILURE: FAILED")
        return AgentResult(output=output, trajectory=traj, steps=len(traj))


@dataclass
class EvalCase:
    id: str
    task: str
    expected: str
    required_steps: list[str] = field(default_factory=list)
    max_steps: int = 12
    tags: list[str] = field(default_factory=list)


def score(result: AgentResult, case: EvalCase) -> bool:
    final_ok = case.expected.lower() in result.output.lower()
    traj_ok = all(s in result.trajectory for s in case.required_steps)
    budget_ok = result.steps <= case.max_steps
    return final_ok and traj_ok and budget_ok


@dataclass
class CaseReport:
    case_id: str
    successes: int
    k: int
    tags: list[str] = field(default_factory=list)

    @property
    def p_hat(self) -> float:
        return self.successes / self.k if self.k else 0.0

    @property
    def pass_k(self) -> float:
        return self.p_hat ** self.k


def run_suite(agent: DeepOpsAgent, cases: list[EvalCase], k: int = 5) -> dict:
    reports: list[CaseReport] = []
    for case in cases:
        successes = sum(1 for _ in range(k) if score(agent.solve(case.task), case))
        reports.append(CaseReport(case.id, successes, k, tags=list(case.tags)))
    mean_pass_k = sum(r.pass_k for r in reports) / len(reports) if reports else 0.0
    return {"label": agent.label, "k": k, "reports": reports, "mean_pass_k": mean_pass_k}


# ==========================================================================
# HARD EXERCISE 1 -- golden regression gate (matrix + ablation)
# ==========================================================================

CASES: list[EvalCase] = [
    EvalCase("fix-add", "fix the add bug in calc",
             expected="SUCCESS",
             required_steps=["plan", "research", "code", "edit", "run_tests", "verify"],
             max_steps=12, tags=["golden"]),
    EvalCase("fix-add-2", "repair calc.add so tests pass",
             expected="SUCCESS",
             required_steps=["code", "run_tests", "verify"],
             max_steps=12, tags=["golden"]),
    EvalCase("budget", "fix the bug within budget",
             expected="SUCCESS", required_steps=["plan"], max_steps=12, tags=[]),
]


def gate_verdict(baseline: dict, candidate: dict, *, golden_gate: bool = True,
                 golden_drop: float = 0.10) -> str:
    """Mirrors regression_report's verdict logic.

    With golden_gate=True, any golden case dropping >= golden_drop BLOCKS,
    regardless of overall mean. With golden_gate=False (ablation) only the mean
    matters -- which is exactly what lets a regressive candidate slip through.
    """
    b = {r.case_id: r for r in baseline["reports"]}
    c = {r.case_id: r for r in candidate["reports"]}
    blocking = False
    for cid, rb in b.items():
        if cid not in c:
            continue
        delta = c[cid].pass_k - rb.pass_k
        if golden_gate and "golden" in rb.tags and delta <= -golden_drop:
            blocking = True
    if blocking:
        return "BLOCKED"
    if candidate["mean_pass_k"] > baseline["mean_pass_k"]:
        return "APPROVED"
    return "NEUTRAL"


def golden_deltas(baseline: dict, candidate: dict) -> dict[str, float]:
    b = {r.case_id: r for r in baseline["reports"]}
    c = {r.case_id: r for r in candidate["reports"]}
    return {cid: c[cid].pass_k - b[cid].pass_k
            for cid in b if cid in c and "golden" in b[cid].tags}


def hard_ex1_regression_gate() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: golden regression gate -- candidate x verdict matrix")
    print("=" * 60)

    k = 8
    baseline = run_suite(DeepOpsAgent(label="baseline", error_rate=0.30, seed=100),
                         CASES, k=k)

    # cand_better: strictly more reliable everywhere.
    cand_better = run_suite(DeepOpsAgent(label="better", error_rate=0.05, seed=100),
                            CASES, k=k)

    # cand_golden_regress: better on the non-golden 'budget' case and on fix-add-2,
    # but selectively WORSE on the golden 'fix-add' task -> golden regression even
    # though the mean improves vs baseline.
    cand_golden_regress = run_suite(
        DeepOpsAgent(label="golden-regress", error_rate=0.02, seed=100,
                     golden_killer="add bug", golden_error_rate=0.95),
        CASES, k=k)

    # cand_neutral: ~equivalent to baseline.
    cand_neutral = run_suite(DeepOpsAgent(label="neutral", error_rate=0.30, seed=100),
                             CASES, k=k)

    candidates = {
        "cand_better": (cand_better, "APPROVED"),
        "cand_golden_regress": (cand_golden_regress, "BLOCKED"),
        "cand_neutral": (cand_neutral, "NEUTRAL"),
    }

    print(f"\n  baseline mean pass^k = {baseline['mean_pass_k']:.3f}  (k={k})")
    print(f"\n  {'candidate':>20}{'mean p^k':>10}{'verdict':>11}{'expected':>11}")
    print("  " + "-" * 52)
    for name, (cand, expected) in candidates.items():
        verdict = gate_verdict(baseline, cand)
        print(f"  {name:>20}{cand['mean_pass_k']:>10.3f}{verdict:>11}{expected:>11}")
        assert verdict == expected, (name, verdict, expected)

    # Show the golden case actually regressed by >= 0.10 for the bad candidate.
    deltas = golden_deltas(baseline, cand_golden_regress)
    print(f"\n  golden deltas (golden-regress vs baseline): "
          f"{ {cid: round(d, 3) for cid, d in deltas.items()} }")
    assert any(d <= -0.10 for d in deltas.values()), deltas
    # And it really did improve the mean (that's why the gate is the only thing
    # stopping it).
    assert cand_golden_regress["mean_pass_k"] > baseline["mean_pass_k"], (
        cand_golden_regress["mean_pass_k"], baseline["mean_pass_k"])

    # ABLATION: drop the golden rule -> the bad candidate would ship (APPROVED).
    ablated = gate_verdict(baseline, cand_golden_regress, golden_gate=False)
    print(f"\n  ABLATION (no golden rule): cand_golden_regress -> {ablated}")
    assert ablated == "APPROVED", ablated

    print("\n  PASS -- gate blocks the golden-regressing candidate; without the")
    print("         golden rule it would have shipped. Gate proven necessary.\n")


# ==========================================================================
# HARD EXERCISE 2 -- durability under eval (crash-resume vs verdict)
# ==========================================================================
# Minimal durable engine + sqlite checkpointer (mirrors J27), plus a global
# side-effect counter so we can prove idempotency: a resumed step runs exactly
# once across the whole crash+resume lifecycle.


EXEC_COUNTS: Counter = Counter()


class SQLiteCheckpointer:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "  run_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
            "  PRIMARY KEY (run_id, key))")
        self._conn.commit()

    def put(self, run_id: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, json.dumps(value)))
        self._conn.commit()

    def get(self, run_id: str, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value FROM checkpoints WHERE run_id=? AND key=?",
            (run_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()


@dataclass
class Step:
    name: str
    fn: Callable[[dict], Any]


class CrashSignal(Exception):
    pass


class DurableEngine:
    def __init__(self, checkpointer: SQLiteCheckpointer) -> None:
        self.cp = checkpointer
        self.executed: list[str] = []
        self.skipped: list[str] = []

    def run(self, run_id: str, steps: list[Step],
            crash_before: str | None = None) -> dict:
        self.executed = []
        self.skipped = []
        ctx: dict = self.cp.get(run_id, "__ctx__") or {}
        for step in steps:
            journal_key = f"step::{step.name}"
            cached = self.cp.get(run_id, journal_key)
            if cached is not None:
                ctx[step.name] = cached          # resume: load, do NOT re-run
                self.skipped.append(step.name)
                continue
            if crash_before is not None and step.name == crash_before:
                raise CrashSignal(f"crash before step '{step.name}'")
            result = step.fn(ctx)                 # the ONLY place a side effect fires
            ctx[step.name] = result
            self.cp.put(run_id, journal_key, result)
            self.cp.put(run_id, "__ctx__", ctx)
            self.executed.append(step.name)
        return ctx


class DurableDeepOpsAgent:
    """Same step sequence as the eval path, runnable under the DurableEngine.

    Every step increments EXEC_COUNTS on REAL execution only (skips don't),
    giving a clean idempotency proof. `will_fix` is decided once per run_id from
    the seed, so the no-crash and crash+resume runs share the same verdict.
    """

    def __init__(self, seed: int = 0, error_rate: float = 0.0) -> None:
        self._will_fix = random.Random(seed).random() >= max(0.0, min(1.0, error_rate))

    def _steps(self) -> list[Step]:
        def plan(ctx):
            EXEC_COUNTS["plan"] += 1
            return ["research", "code", "verify"]

        def research(ctx):
            EXEC_COUNTS["research"] += 1
            return "bug is in calc.add"

        def code(ctx):
            EXEC_COUNTS["code"] += 1
            return "fixed" if self._will_fix else "FAILED"

        def verify(ctx):
            EXEC_COUNTS["verify"] += 1
            return "verified-ok" if ctx.get("code") == "fixed" else "verify-failed"

        return [Step("plan", plan), Step("research", research),
                Step("code", code), Step("verify", verify)]

    def run_durable(self, run_id: str, cp: SQLiteCheckpointer,
                    crash_before: str | None = None) -> dict:
        engine = DurableEngine(cp)
        ctx = engine.run(run_id, self._steps(), crash_before=crash_before)
        ctx["__executed__"] = engine.executed
        ctx["__skipped__"] = engine.skipped
        return ctx

    @staticmethod
    def verdict_of(ctx: dict) -> str:
        return "SUCCESS" if ctx.get("verify") == "verified-ok" else "FAILURE"


STEP_ORDER = ["plan", "research", "code", "verify"]


def hard_ex2_durability_under_eval() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: durability under eval -- crash-resume vs verdict")
    print("=" * 60)

    seeds = [1, 2, 3, 4, 5, 6]
    crash_points = ["research", "code", "verify"]
    error_rate = 0.5     # mix of SUCCESS/FAILURE verdicts across seeds
    tmpdir = Path(tempfile.mkdtemp(prefix="j28_durable_"))
    checked = 0

    print(f"\n  seeds={seeds} crash_points={crash_points} error_rate={error_rate}")
    print(f"  {'seed':>5}{'crash@':>9}{'no_crash':>10}{'resumed':>10}{'match':>7}")
    print("  " + "-" * 41)

    try:
        for seed in seeds:
            # Reference verdict: a clean durable run, its own run_id/db.
            EXEC_COUNTS.clear()
            cp_ref = SQLiteCheckpointer(str(tmpdir / f"ref_{seed}.db"))
            ctx_ref = DurableDeepOpsAgent(seed=seed, error_rate=error_rate).run_durable(
                f"ref-{seed}", cp_ref)
            cp_ref.close()
            verdict_no_crash = DurableDeepOpsAgent.verdict_of(ctx_ref)
            # Clean run executed every step exactly once.
            assert all(EXEC_COUNTS[s] == 1 for s in STEP_ORDER), dict(EXEC_COUNTS)

            for crash_before in crash_points:
                EXEC_COUNTS.clear()
                db = str(tmpdir / f"run_{seed}_{crash_before}.db")
                run_id = f"run-{seed}-{crash_before}"

                # Attempt 1: crash before `crash_before`; process "dies".
                agent1 = DurableDeepOpsAgent(seed=seed, error_rate=error_rate)
                cp = SQLiteCheckpointer(db)
                try:
                    agent1.run_durable(run_id, cp, crash_before=crash_before)
                except CrashSignal:
                    pass
                cp.close()

                # Attempt 2: fresh process, SAME durable file -> resume.
                agent2 = DurableDeepOpsAgent(seed=seed, error_rate=error_rate)
                cp2 = SQLiteCheckpointer(db)
                ctx = agent2.run_durable(run_id, cp2)
                cp2.close()

                idx = STEP_ORDER.index(crash_before)
                expected_skipped = STEP_ORDER[:idx]
                expected_executed = STEP_ORDER[idx:]

                # Steps before the crash are skipped on resume...
                assert ctx["__skipped__"] == expected_skipped, (
                    seed, crash_before, ctx["__skipped__"])
                # ...and steps from the crash point onward are executed on resume.
                assert ctx["__executed__"] == expected_executed, (
                    seed, crash_before, ctx["__executed__"])

                # IDEMPOTENCY: every step ran EXACTLY once across crash+resume
                # (finished steps were never re-executed).
                for s in STEP_ORDER:
                    assert EXEC_COUNTS[s] == 1, (seed, crash_before, s, dict(EXEC_COUNTS))
                assert sum(EXEC_COUNTS.values()) == len(STEP_ORDER), dict(EXEC_COUNTS)

                # CORRECTNESS: resumed verdict == no-crash verdict.
                verdict_resumed = DurableDeepOpsAgent.verdict_of(ctx)
                match = verdict_no_crash == verdict_resumed
                print(f"  {seed:>5}{crash_before:>9}{verdict_no_crash:>10}"
                      f"{verdict_resumed:>10}{('yes' if match else 'NO'):>7}")
                assert match, (seed, crash_before, verdict_no_crash, verdict_resumed)
                checked += 1
    finally:
        for f in tmpdir.glob("*.db"):
            f.unlink()
        tmpdir.rmdir()

    assert checked == len(seeds) * len(crash_points)
    print(f"\n  checked {checked} (seed x crash-point) combinations")
    print("\n  PASS -- no finished step re-executed (each ran exactly once) and")
    print("         durability never changed the verdict.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 28 HARD Solutions -- Capstone build & eval extensions")
    print("#" * 60)

    hard_ex1_regression_gate()
    hard_ex2_durability_under_eval()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

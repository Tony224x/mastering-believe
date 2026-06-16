"""
Solutions -- Day 27 (MEDIUM): Capstone architecture extensions

Contains solutions for:
  - Medium Ex 1: DurableEngine with retry-with-backoff, journaling only on success
  - Medium Ex 2: ModelRouter with hard budget cap + downgrade-under-pressure
  - Medium Ex 3: VirtualFS <-> checkpointer integrity audit

These EXTEND the J27 durable building blocks. To run fully offline & standalone,
this file embeds a FAITHFUL mini-version of the J27 blocks (VirtualFS,
SQLiteCheckpointer, DurableEngine, Step, CrashSignal, ModelRouter) -- we do NOT
import 02-code/27-capstone-architecture.py (numeric module name, and we want the
files decoupled). Only stdlib is used (sqlite3 + tempfile).

Run:  python 03-exercises/solutions/27-capstone-architecture-medium.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ==========================================================================
# EMBEDDED MINI J27 BUILDING BLOCKS (offline stand-in for 02-code/27)
# Faithful copies of VirtualFS / SQLiteCheckpointer / DurableEngine / Step /
# CrashSignal / ModelRouter -- kept minimal but behaviorally identical.
# ==========================================================================


class VirtualFS:
    """Disk-backed key->text scratchpad (context offloading, J15)."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="capstone_fs_"))
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str) -> None:
        (self.root / name).write_text(content, encoding="utf-8")

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def remove(self, name: str) -> None:
        (self.root / name).unlink(missing_ok=True)

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())


class SQLiteCheckpointer:
    """Durable key/value store backed by sqlite3 (file survives process death)."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints ("
            "  run_id TEXT NOT NULL,"
            "  key    TEXT NOT NULL,"
            "  value  TEXT NOT NULL,"
            "  PRIMARY KEY (run_id, key)"
            ")"
        )
        self._conn.commit()

    def put(self, run_id: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, json.dumps(value)),
        )
        self._conn.commit()

    def put_raw(self, run_id: str, key: str, raw: str) -> None:
        """Write a RAW (possibly invalid-JSON) value -- used to simulate corruption."""
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, raw),
        )
        self._conn.commit()

    def get(self, run_id: str, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value FROM checkpoints WHERE run_id=? AND key=?", (run_id, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def keys(self, run_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT key FROM checkpoints WHERE run_id=? ORDER BY key", (run_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def delete(self, run_id: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM checkpoints WHERE run_id=? AND key=?", (run_id, key)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


@dataclass
class Step:
    """A named, idempotent unit of work: fn(ctx) -> result."""
    name: str
    fn: Callable[[dict], Any]


class CrashSignal(Exception):
    """Raised to simulate a process crash mid-workflow."""


class DurableEngine:
    """Runs Steps, journaling each completed step; resumes after crash."""

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
                ctx[step.name] = cached
                self.skipped.append(step.name)
                continue
            if crash_before is not None and step.name == crash_before:
                raise CrashSignal(f"crash before step '{step.name}'")
            result = step.fn(ctx)
            ctx[step.name] = result
            self.cp.put(run_id, journal_key, result)
            self.cp.put(run_id, "__ctx__", ctx)
            self.executed.append(step.name)
        return ctx


@dataclass
class ModelRouter:
    """Cost-aware weak/strong routing (J24), tracks cumulative cost."""
    threshold: int = 12
    cost_weak: float = 1.0
    cost_strong: float = 8.0
    total_cost: float = 0.0
    routed: dict = field(default_factory=lambda: {"weak": 0, "strong": 0})

    def route(self, task: str) -> str:
        tier = "strong" if len(task.split()) >= self.threshold else "weak"
        self.routed[tier] += 1
        self.total_cost += self.cost_strong if tier == "strong" else self.cost_weak
        return tier


def _tmp_db() -> str:
    """Return a fresh temp sqlite path that survives close/reopen (crash sim)."""
    fd, path = tempfile.mkstemp(prefix="capstone_cp_", suffix=".db")
    os.close(fd)
    os.unlink(path)  # let sqlite create it; we just want a unique path
    return path


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ==========================================================================
# MEDIUM EXERCISE 1 -- Retry-with-backoff, journal only on success
# ==========================================================================

class RetryingDurableEngine(DurableEngine):
    """DurableEngine that retries transient step failures with a BOUNDED,
    logical backoff, and journals a step ONLY after a successful call.

    A step that raises is retried up to `max_attempts`. The backoff is logical
    (no real sleep): we accumulate `backoff_base * 2**(attempt-1)` so tests stay
    instant and deterministic. If all attempts fail, the last exception is
    propagated and the step stays UN-journaled (so a later resume can retry it).
    """

    def __init__(self, checkpointer: SQLiteCheckpointer,
                 max_attempts: int = 3, backoff_base: float = 0.0) -> None:
        super().__init__(checkpointer)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.attempts: dict[str, int] = {}
        self.total_backoff: float = 0.0

    def run(self, run_id: str, steps: list[Step],
            crash_before: str | None = None) -> dict:
        self.executed = []
        self.skipped = []
        ctx: dict = self.cp.get(run_id, "__ctx__") or {}
        for step in steps:
            journal_key = f"step::{step.name}"
            cached = self.cp.get(run_id, journal_key)
            if cached is not None:
                ctx[step.name] = cached
                self.skipped.append(step.name)
                continue
            if crash_before is not None and step.name == crash_before:
                raise CrashSignal(f"crash before step '{step.name}'")

            result = self._run_with_retry(step, ctx)
            ctx[step.name] = result
            # Journal ONLY after success (atomic-enough for the demo).
            self.cp.put(run_id, journal_key, result)
            self.cp.put(run_id, "__ctx__", ctx)
            self.executed.append(step.name)
        return ctx

    def _run_with_retry(self, step: Step, ctx: dict) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.attempts[step.name] = attempt
            try:
                return step.fn(ctx)
            except Exception as exc:  # transient failure -> retry
                last_exc = exc
                if attempt < self.max_attempts:
                    # logical backoff, no real sleep
                    self.total_backoff += self.backoff_base * (2 ** (attempt - 1))
        assert last_exc is not None
        raise last_exc


def medium_ex1_retry_backoff() -> None:
    _banner("MEDIUM Ex1 -- DurableEngine: retry-with-backoff, journal on success")
    db = _tmp_db()
    run_id = "retry-run"

    # Instrument real side-effects to prove a step's fn is not replayed on resume.
    side_effects: dict[str, int] = {"gather": 0, "flaky": 0, "report": 0}
    flaky_state = {"calls": 0}

    def gather(ctx):
        side_effects["gather"] += 1
        return "facts: 3 items"

    def flaky(ctx):
        # Fails on the first TWO calls, succeeds on the third (transient).
        side_effects["flaky"] += 1
        flaky_state["calls"] += 1
        if flaky_state["calls"] < 3:
            raise RuntimeError(f"transient failure #{flaky_state['calls']}")
        return f"processed [{ctx['gather']}]"

    def report(ctx):
        side_effects["report"] += 1
        return f"REPORT from {ctx['flaky']}"

    steps = [Step("gather", gather), Step("flaky", flaky), Step("report", report)]

    cp = SQLiteCheckpointer(db)
    eng = RetryingDurableEngine(cp, max_attempts=3, backoff_base=0.5)
    final = eng.run(run_id, steps)
    cp.close()

    print(f"  attempts        : {eng.attempts}")
    print(f"  total_backoff   : {eng.total_backoff}")
    print(f"  executed        : {eng.executed}")
    print(f"  side_effects    : {side_effects}")

    assert eng.attempts["flaky"] == 3, "flaky must take exactly 3 attempts"
    assert eng.attempts["gather"] == 1 and eng.attempts["report"] == 1
    # Logical backoff = base*2^0 + base*2^1 over the 2 failed attempts = 0.5 + 1.0
    assert eng.total_backoff == 0.5 + 1.0, "backoff must accumulate logically (no sleep)"
    assert side_effects["flaky"] == 3, "fn called 3x (2 fail + 1 success)"
    assert final["flaky"] == "processed [facts: 3 items]"
    assert eng.executed == ["gather", "flaky", "report"]

    # --- Resume: a fresh engine on the SAME db must SKIP the journaled flaky ---
    cp2 = SQLiteCheckpointer(db)
    eng2 = RetryingDurableEngine(cp2, max_attempts=3)
    final2 = eng2.run(run_id, steps)
    cp2.close()
    print(f"  [resume] skipped: {eng2.skipped}, executed: {eng2.executed}")
    assert eng2.skipped == ["gather", "flaky", "report"], "all journaled -> skip"
    assert eng2.executed == []
    # CRITICAL: flaky.fn must NOT be replayed -> side-effect counter unchanged.
    assert side_effects["flaky"] == 3, "journaled step must not re-run its fn on resume"
    assert final2 == final

    # --- A step that ALWAYS fails: exception propagated, step stays un-journaled ---
    db2 = _tmp_db()

    def always_fail(ctx):
        raise RuntimeError("permanent failure")

    steps_bad = [Step("gather", gather), Step("doomed", always_fail)]
    cp3 = SQLiteCheckpointer(db2)
    eng3 = RetryingDurableEngine(cp3, max_attempts=2)
    raised = False
    try:
        eng3.run("bad-run", steps_bad)
    except RuntimeError as exc:
        raised = True
        print(f"  doomed step propagated: {exc}")
    assert raised, "all-fail step must propagate after max_attempts"
    assert cp3.get("bad-run", "step::doomed") is None, "failed step must NOT be journaled"
    assert cp3.get("bad-run", "step::gather") is not None, "prior step stays journaled"
    cp3.close()

    os.unlink(db)
    os.unlink(db2)
    print("  OK: retries bounded, journaled on success only, resume skips fn.")


# ==========================================================================
# MEDIUM EXERCISE 2 -- ModelRouter: budget cap + downgrade under pressure
# ==========================================================================

class BudgetExceeded(Exception):
    """Raised when even the cheapest tier would exceed the hard budget cap."""


@dataclass
class CappedRouter(ModelRouter):
    """ModelRouter with a HARD budget cap and a downgrade-under-pressure policy.

    - desired tier is computed like ModelRouter (complexity by word count);
    - if desired == strong but it would breach the cap, DOWNGRADE to weak;
    - if even weak would breach the cap, REFUSE (return None), leaving cost intact.

    Invariant: total_cost <= budget at all times.
    """
    budget: float = 0.0

    def route(self, task: str) -> str | None:
        desired = "strong" if len(task.split()) >= self.threshold else "weak"

        # Try desired tier first; downgrade strong->weak under budget pressure.
        if desired == "strong":
            if self.total_cost + self.cost_strong <= self.budget:
                return self._commit("strong")
            self.downgrades += 1  # we wanted strong but can't afford it
            desired = "weak"

        if desired == "weak":
            if self.total_cost + self.cost_weak <= self.budget:
                return self._commit("weak")
            return None  # even the cheapest tier breaks the cap -> refuse

        return None

    def _commit(self, tier: str) -> str:
        self.routed[tier] += 1
        self.total_cost += self.cost_strong if tier == "strong" else self.cost_weak
        assert self.total_cost <= self.budget, "invariant: cost must never exceed budget"
        return tier

    # `downgrades` is not a dataclass field of the parent; declare via __post_init__.
    def __post_init__(self) -> None:  # type: ignore[override]
        self.downgrades = 0


COMPLEX = "summarize this very long multi step analysis task that needs the strong model"
SIMPLE = "format date"


def medium_ex2_budget_downgrade() -> None:
    _banner("MEDIUM Ex2 -- CappedRouter: hard budget + downgrade under pressure")
    assert len(COMPLEX.split()) >= 12 and len(SIMPLE.split()) < 12

    # (a) Large budget: complex task routes strong.
    r = CappedRouter(budget=100.0)
    tier = r.route(COMPLEX)
    print(f"  large budget   : complex -> {tier}, cost={r.total_cost}")
    assert tier == "strong" and r.total_cost == 8.0 and r.downgrades == 0

    # (b) Tight budget: complex task can't afford strong -> downgrade to weak.
    r2 = CappedRouter(budget=3.0)  # strong (8) impossible, weak (1) ok
    tier2 = r2.route(COMPLEX)
    print(f"  tight budget   : complex -> {tier2}, downgrades={r2.downgrades}")
    assert tier2 == "weak", "complex must downgrade strong->weak under pressure"
    assert r2.downgrades == 1
    assert r2.total_cost == 1.0 <= 3.0

    # (c) Burst of mixed tasks under a cap: invariant total_cost <= budget holds.
    r3 = CappedRouter(budget=20.0)
    tasks = [COMPLEX, SIMPLE, COMPLEX, COMPLEX, SIMPLE, COMPLEX, COMPLEX]
    decisions = [r3.route(t) for t in tasks]
    strong_count = decisions.count("strong")
    print(f"  burst (cap=20) : decisions={decisions}")
    print(f"  burst result   : cost={r3.total_cost}, downgrades={r3.downgrades}, routed={r3.routed}")
    assert r3.total_cost <= 20.0, "invariant must hold across the whole burst"
    assert strong_count >= 1, "complex tasks still routed strong while affordable"
    assert r3.downgrades >= 1, "later complex tasks downgrade once budget tightens"

    # (d) Exhausted budget: next task refused cleanly, cost untouched.
    r4 = CappedRouter(budget=8.0)
    assert r4.route(COMPLEX) == "strong"  # spends the whole budget
    before = r4.total_cost
    refused = r4.route(SIMPLE)  # even weak (1.0) breaks cap (8.0)
    print(f"  exhausted      : refused={refused}, cost stable={r4.total_cost == before}")
    assert refused is None, "must refuse when even weak breaks the cap"
    assert r4.total_cost == before, "refused task must not corrupt total_cost"
    assert r4.total_cost <= 8.0

    print("  OK: cap never breached, downgrade observed, clean refusal.")


# ==========================================================================
# MEDIUM EXERCISE 3 -- VirtualFS <-> checkpointer integrity audit
# ==========================================================================

EXPECTED_ARTIFACTS = {"plan": "todo.md", "research": "research.md", "code": "report.md"}


def audit_integrity(checkpointer: SQLiteCheckpointer, fs: VirtualFS,
                    run_id: str, expected: dict[str, str]) -> dict:
    """Detect desync between the journal (finished steps) and FS artifacts.

    Returns journaled steps, journaled steps whose artifact is missing on disk,
    and artifacts on disk whose step is not journaled (orphans).
    """
    journal_keys = [k for k in checkpointer.keys(run_id) if k.startswith("step::")]
    journaled = sorted(k[len("step::"):] for k in journal_keys)

    missing_artifacts = [
        step for step in journaled
        if step in expected and not fs.exists(expected[step])
    ]

    artifact_to_step = {v: k for k, v in expected.items()}
    fs_files = set(fs.list())
    orphan_artifacts = sorted(
        fname for fname in fs_files
        if fname in artifact_to_step and artifact_to_step[fname] not in journaled
    )

    consistent = not missing_artifacts and not orphan_artifacts
    return {
        "journaled": journaled,
        "missing_artifacts": sorted(missing_artifacts),
        "orphan_artifacts": orphan_artifacts,
        "consistent": consistent,
    }


def _run_with_artifacts(cp: SQLiteCheckpointer, fs: VirtualFS, run_id: str) -> dict:
    """Run an engine whose steps ALSO write their artifact to the VirtualFS."""

    def plan(ctx):
        fs.write("todo.md", "- [ ] research\n- [ ] code")
        return "plan ok"

    def research(ctx):
        fs.write("research.md", "findings: 3 facts")
        return "research ok"

    def code(ctx):
        fs.write("report.md", "REPORT body")
        return "code ok"

    steps = [Step("plan", plan), Step("research", research), Step("code", code)]
    eng = DurableEngine(cp)
    return eng.run(run_id, steps)


def medium_ex3_integrity_audit() -> None:
    _banner("MEDIUM Ex3 -- VirtualFS <-> checkpointer integrity audit")

    # (a) Full coherent run -> consistent True.
    db = _tmp_db()
    fs = VirtualFS()
    cp = SQLiteCheckpointer(db)
    _run_with_artifacts(cp, fs, "audit-run")
    rep_ok = audit_integrity(cp, fs, "audit-run", EXPECTED_ARTIFACTS)
    print(f"  coherent       : {rep_ok}")
    assert rep_ok["journaled"] == ["code", "plan", "research"]
    assert rep_ok["consistent"] is True
    assert rep_ok["missing_artifacts"] == [] and rep_ok["orphan_artifacts"] == []

    # (b) Delete report.md while 'code' is journaled -> missing_artifacts=['code'].
    fs.remove("report.md")
    rep_missing = audit_integrity(cp, fs, "audit-run", EXPECTED_ARTIFACTS)
    print(f"  missing artifact: {rep_missing}")
    assert rep_missing["missing_artifacts"] == ["code"]
    assert rep_missing["consistent"] is False
    cp.close()

    # (c) Orphan artifact: artifact on FS but its step not journaled.
    db2 = _tmp_db()
    fs2 = VirtualFS()
    cp2 = SQLiteCheckpointer(db2)
    # Journal only 'plan'; write research.md by hand (orphan, no 'research' step).
    cp2.put("orphan-run", "step::plan", "plan ok")
    fs2.write("todo.md", "- [ ] research")
    fs2.write("research.md", "orphan finding written without a journaled step")
    rep_orphan = audit_integrity(cp2, fs2, "orphan-run", EXPECTED_ARTIFACTS)
    print(f"  orphan artifact: {rep_orphan}")
    assert rep_orphan["journaled"] == ["plan"]
    assert rep_orphan["orphan_artifacts"] == ["research.md"]
    assert rep_orphan["consistent"] is False
    cp2.close()

    os.unlink(db)
    os.unlink(db2)
    print("  OK: coherent / missing-artifact / orphan all detected correctly.")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    medium_ex1_retry_backoff()
    medium_ex2_budget_downgrade()
    medium_ex3_integrity_audit()
    print("\n" + "=" * 68)
    print("ALL MEDIUM (J27) SOLUTIONS PASSED")
    print("=" * 68)

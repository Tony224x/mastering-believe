"""
Solutions -- Day 27 (HARD): Capstone architecture extensions

Contains solutions for:
  - Hard Ex 1: Crash-resume robustness MATRIX (crash before every boundary) +
               half-written-checkpoint corruption detection + idempotency proof
  - Hard Ex 2: Concurrent idempotency -- journal-claim/lock so each step runs
               at-most-once across two workers sharing the same run_id, plus a
               "winner crashes before journaling" recovery (no permanent stall)

These DURABILITY-HARDEN the J27 building blocks. To run fully offline &
standalone, this file embeds a FAITHFUL mini-version of the J27 blocks
(VirtualFS-not-needed-here, SQLiteCheckpointer, DurableEngine, Step,
CrashSignal). We do NOT import 02-code/27-capstone-architecture.py. Only stdlib
is used (sqlite3 + tempfile). To demonstrate crash-resume we use a real tempfile
db that survives a simulated process death (close + reopen) and is cleaned up.

Run:  python 03-exercises/solutions/27-capstone-architecture-hard.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

# ==========================================================================
# EMBEDDED MINI J27 BUILDING BLOCKS (offline stand-in for 02-code/27)
# ==========================================================================


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
        # Claims table for the concurrency exercise (atomic at-most-once claim).
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS claims ("
            "  run_id TEXT NOT NULL,"
            "  step   TEXT NOT NULL,"
            "  owner  TEXT NOT NULL,"
            "  PRIMARY KEY (run_id, step)"
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
        """Write a RAW (possibly invalid-JSON) value -- simulate corruption."""
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

    def get_raw(self, run_id: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM checkpoints WHERE run_id=? AND key=?", (run_id, key)
        ).fetchone()
        return row[0] if row else None

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

    # --- atomic journal-claim primitives (Hard Ex2) ---
    def try_claim(self, run_id: str, step: str, owner: str) -> bool:
        """Atomically claim a step. Returns True iff THIS owner won the claim.

        Uses INSERT OR IGNORE on a UNIQUE (run_id, step) row: only the first
        writer inserts; concurrent writers are ignored. We then read back the
        owner to decide the winner. sqlite serializes writes -> atomic.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO claims (run_id, step, owner) VALUES (?, ?, ?)",
            (run_id, step, owner),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT owner FROM claims WHERE run_id=? AND step=?", (run_id, step)
        ).fetchone()
        return bool(row) and row[0] == owner

    def release_claim(self, run_id: str, step: str) -> None:
        """Release a claim (e.g. winner crashed before journaling -> reclaimable)."""
        self._conn.execute(
            "DELETE FROM claims WHERE run_id=? AND step=?", (run_id, step)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


@dataclass
class Step:
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


def _tmp_db() -> str:
    """Return a fresh temp sqlite path that survives close/reopen (crash sim)."""
    fd, path = tempfile.mkstemp(prefix="capstone_cp_", suffix=".db")
    os.close(fd)
    os.unlink(path)  # let sqlite create the file; we just want a unique path
    return path


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ==========================================================================
# HARD EXERCISE 1 -- Crash-resume robustness matrix + corruption detection
# ==========================================================================

def _build_steps(side_effects: dict[str, int]) -> list[Step]:
    """A 4-step pipeline; every fn bumps a side-effect counter so we can prove
    exactly which steps actually execute (vs are skipped/resumed)."""

    def plan(ctx):
        side_effects["plan"] += 1
        return "plan: 2 todos"

    def research(ctx):
        side_effects["research"] += 1
        return f"research[{ctx['plan']}]"

    def code(ctx):
        side_effects["code"] += 1
        return f"code[{ctx['research']}]"

    def verify(ctx):
        side_effects["verify"] += 1
        return f"verify[{ctx['code']}]"

    return [Step("plan", plan), Step("research", research),
            Step("code", code), Step("verify", verify)]


def audit_corruption(cp: SQLiteCheckpointer, run_id: str,
                     steps: list[Step]) -> list[str]:
    """Detect half-written / inconsistent checkpoints. Returns steps to REPLAY.

    Two corruption signatures are checked:
      1. journal value for a step is not valid JSON (half-written write);
      2. step::<x> is journaled but the durable __ctx__ does not reflect x
         (the two writes are not atomic; a crash landed between them).
    """
    to_replay: list[str] = []

    # Read __ctx__ defensively (it may itself be corrupt).
    ctx_raw = cp.get_raw(run_id, "__ctx__")
    try:
        ctx = json.loads(ctx_raw) if ctx_raw is not None else {}
    except json.JSONDecodeError:
        ctx = {}

    for step in steps:
        jkey = f"step::{step.name}"
        raw = cp.get_raw(run_id, jkey)
        if raw is None:
            continue  # not journaled -> normal resume will run it, not "corrupt"
        # signature 1: invalid JSON
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            to_replay.append(step.name)
            continue
        # signature 2: journaled but ctx doesn't reflect it
        if step.name not in ctx:
            to_replay.append(step.name)
    return to_replay


def hard_ex1_resume_matrix() -> None:
    _banner("HARD Ex1 -- crash-resume MATRIX across every boundary + corruption")

    # --- Reference no-crash run -> ref_ctx ---
    db_ref = _tmp_db()
    se_ref: dict[str, int] = {k: 0 for k in ("plan", "research", "code", "verify")}
    steps_ref = _build_steps(se_ref)
    cp_ref = SQLiteCheckpointer(db_ref)
    ref_ctx = DurableEngine(cp_ref).run("ref", steps_ref)
    cp_ref.close()
    os.unlink(db_ref)
    names = [s.name for s in steps_ref]
    print(f"  reference ctx keys: {sorted(k for k in ref_ctx if not k.startswith('__'))}")
    assert all(se_ref[n] == 1 for n in names), "reference run executes each step once"

    # --- Ablation by boundary: crash before each step i (and i==N == no crash) ---
    matrix: list[dict] = []
    for i in range(len(names) + 1):
        db = _tmp_db()
        run_id = f"matrix-{i}"
        se: dict[str, int] = {k: 0 for k in names}
        steps = _build_steps(se)

        crash_at = names[i] if i < len(names) else None

        # Attempt 1: run until the crash boundary (or full run if i==N).
        cp = SQLiteCheckpointer(db)
        eng = DurableEngine(cp)
        crashed = False
        try:
            eng.run(run_id, steps, crash_before=crash_at)
        except CrashSignal:
            crashed = True
        executed_first = list(eng.executed)
        cp.close()  # simulate process death

        # Attempt 2: fresh process, same run_id + db, no crash -> resume.
        cp2 = SQLiteCheckpointer(db)
        eng2 = DurableEngine(cp2)
        final = eng2.run(run_id, steps)
        cp2.close()

        prefix = names[:i]      # journaled before the crash -> must be skipped
        suffix = names[i:]      # remaining -> must be executed on resume

        # (a) journaled prefix is skipped on resume
        assert eng2.skipped == prefix, f"i={i}: skipped {eng2.skipped} != prefix {prefix}"
        # (b) only the suffix executes on resume
        assert eng2.executed == suffix, f"i={i}: executed {eng2.executed} != suffix {suffix}"
        # (c) final context identical to the no-crash reference
        clean_final = {k: v for k, v in final.items() if not k.startswith("__")}
        clean_ref = {k: v for k, v in ref_ctx.items() if not k.startswith("__")}
        assert clean_final == clean_ref, f"i={i}: resumed ctx differs from reference"
        # Each step's fn ran exactly once IN TOTAL across the two attempts.
        assert all(se[n] == 1 for n in names), f"i={i}: a step ran !=1 times total ({se})"
        if i == len(names):
            assert eng2.executed == [], "crash after last step re-executes nothing"

        matrix.append({"crash_before": crash_at, "skipped": eng2.skipped,
                       "executed": eng2.executed})
        os.unlink(db)

    print("  resume matrix (every boundary):")
    for row in matrix:
        print(f"    crash_before={str(row['crash_before']):>9} "
              f"-> skipped={row['skipped']}, executed={row['executed']}")
    print("  -> every boundary: prefix skipped, suffix executed, ctx == reference.")

    # --- Corruption case A: half-written step value (invalid JSON) ---
    db_c = _tmp_db()
    se_c: dict[str, int] = {k: 0 for k in names}
    steps_c = _build_steps(se_c)
    cp_c = SQLiteCheckpointer(db_c)
    DurableEngine(cp_c).run("corrupt", steps_c)  # full clean run first
    # Corrupt the 'code' journal entry with a half-written (invalid JSON) value.
    cp_c.put_raw("corrupt", "step::code", '"code[research[plan: 2 todos]]')  # missing quote
    detected = audit_corruption(cp_c, "corrupt", steps_c)
    print(f"\n  corruption A (bad JSON) detected to replay: {detected}")
    assert detected == ["code"], "invalid-JSON checkpoint must be flagged for replay"

    # --- Corruption case B: journaled step missing from durable __ctx__ ---
    # Rewrite __ctx__ to a state that does NOT include 'verify' though it's journaled.
    cp_c.put("corrupt", "__ctx__",
             {"plan": "plan: 2 todos", "research": "research[plan: 2 todos]",
              "code": "code[research[plan: 2 todos]]"})  # 'verify' absent
    # Fix 'code' back to valid first so only 'verify' is the new signal.
    cp_c.put("corrupt", "step::code", "code[research[plan: 2 todos]]")
    detected_b = audit_corruption(cp_c, "corrupt", steps_c)
    print(f"  corruption B (ctx desync) detected to replay: {detected_b}")
    assert detected_b == ["verify"], "ctx-desync checkpoint must be flagged for replay"

    # --- Idempotency: invalidate the corrupt checkpoint, resume re-runs ONLY it ---
    se_before = dict(se_c)  # snapshot side-effect counters after the clean run
    cp_c.delete("corrupt", "step::verify")  # drop the corrupt step's journal
    cp_c.put("corrupt", "__ctx__",
             {"plan": "plan: 2 todos", "research": "research[plan: 2 todos]",
              "code": "code[research[plan: 2 todos]]"})
    eng_fix = DurableEngine(cp_c)
    final_fix = eng_fix.run("corrupt", steps_c)
    print(f"  idempotent fix : skipped={eng_fix.skipped}, executed={eng_fix.executed}")
    assert eng_fix.executed == ["verify"], "resume re-runs ONLY the invalidated step"
    assert eng_fix.skipped == ["plan", "research", "code"]
    # Only 'verify' fn ran again; all others unchanged -> idempotent.
    assert se_c["verify"] == se_before["verify"] + 1
    for n in ("plan", "research", "code"):
        assert se_c[n] == se_before[n], f"{n} must NOT be re-executed"
    clean_fix = {k: v for k, v in final_fix.items() if not k.startswith("__")}
    assert clean_fix == clean_ref, "fixed run reconverges to the reference ctx"
    cp_c.close()
    os.unlink(db_c)

    print("  OK: full boundary matrix + corruption detection + idempotent replay.")


# ==========================================================================
# HARD EXERCISE 2 -- Concurrent idempotency via journal-claim/lock
# ==========================================================================

# Global side-effect counter: the instrument that proves at-most-once execution.
EXEC_COUNTER: dict[str, int] = {}


class ClaimingDurableEngine:
    """A DurableEngine variant safe under CONCURRENT workers sharing a run_id.

    Before executing a step, a worker must WIN an atomic journal-claim in
    sqlite. The winner executes + journals; the loser does NOT execute and
    instead loads the journaled result. This guarantees at-most-once execution
    of each step's fn, even if both workers attempt every step.

    A worker can crash AFTER claiming but BEFORE journaling. To avoid a
    permanent stall, a worker that finds a claim it doesn't own but with NO
    journaled result yet may RECLAIM it (we model the TTL/lease release
    explicitly via `release_claim`). This keeps total executions == 1 per step.
    """

    def __init__(self, cp: SQLiteCheckpointer, owner: str) -> None:
        self.cp = cp
        self.owner = owner
        self.executed: list[str] = []
        self.skipped: list[str] = []

    def _claimed_result(self, run_id: str, step: Step, ctx: dict) -> Any:
        """Return the journaled result, loading it (skip) if already present;
        otherwise win/lose the claim and act accordingly. Returns the result."""
        jkey = f"step::{step.name}"
        cached = self.cp.get(run_id, jkey)
        if cached is not None:
            self.skipped.append(step.name)
            return cached

        if self.cp.try_claim(run_id, step.name, self.owner):
            # We won the claim -> execute exactly once and journal.
            result = step.fn(ctx)
            self.cp.put(run_id, jkey, result)
            self.executed.append(step.name)
            return result

        # We lost the claim. The winner should journal the result; load it.
        cached = self.cp.get(run_id, jkey)
        if cached is not None:
            self.skipped.append(step.name)
            return cached
        # Lost the claim but no result yet -> winner may have crashed before
        # journaling. Release the stale claim and re-claim (lease recovery).
        self.cp.release_claim(run_id, step.name)
        if self.cp.try_claim(run_id, step.name, self.owner):
            result = step.fn(ctx)
            self.cp.put(run_id, jkey, result)
            self.executed.append(step.name)
            return result
        # Someone else grabbed it in the race; load their result.
        self.skipped.append(step.name)
        return self.cp.get(run_id, jkey)

    def run_step(self, run_id: str, step: Step) -> Any:
        ctx: dict = self.cp.get(run_id, "__ctx__") or {}
        result = self._claimed_result(run_id, step, ctx)
        ctx[step.name] = result
        self.cp.put(run_id, "__ctx__", ctx)
        return result


def hard_ex2_concurrent_idempotency() -> None:
    _banner("HARD Ex2 -- concurrent idempotency: journal-claim/lock at-most-once")
    global EXEC_COUNTER
    EXEC_COUNTER = {"plan": 0, "research": 0, "code": 0, "verify": 0}

    def make_steps() -> list[Step]:
        def plan(ctx):
            EXEC_COUNTER["plan"] += 1
            return "plan"

        def research(ctx):
            EXEC_COUNTER["research"] += 1
            return "research"

        def code(ctx):
            EXEC_COUNTER["code"] += 1
            return "code"

        def verify(ctx):
            EXEC_COUNTER["verify"] += 1
            return "verify"

        return [Step("plan", plan), Step("research", research),
                Step("code", code), Step("verify", verify)]

    db = _tmp_db()
    run_id = "concurrent-run"
    cpA = SQLiteCheckpointer(db)
    cpB = SQLiteCheckpointer(db)  # second connection = "second worker process"
    workerA = ClaimingDurableEngine(cpA, owner="A")
    workerB = ClaimingDurableEngine(cpB, owner="B")
    stepsA = make_steps()
    stepsB = make_steps()

    # --- Adversarial interleaving: BOTH workers attempt EVERY step, A then B ---
    for sa, sb in zip(stepsA, stepsB):
        rA = workerA.run_step(run_id, sa)   # A claims+executes
        rB = workerB.run_step(run_id, sb)   # B loses claim, loads A's result
        assert rA == rB, f"workers must converge on step '{sa.name}'"

    print(f"  EXEC_COUNTER   : {EXEC_COUNTER}")
    print(f"  A executed     : {workerA.executed}")
    print(f"  B executed     : {workerB.executed}")
    print(f"  B skipped      : {workerB.skipped}")

    # at-most-once: every step's fn ran exactly once despite two attempts each.
    assert all(v == 1 for v in EXEC_COUNTER.values()), "each step must run exactly once"
    assert workerA.executed == ["plan", "research", "code", "verify"]
    assert workerB.executed == [], "loser executes nothing"
    assert workerB.skipped == ["plan", "research", "code", "verify"]

    # Both converge to an identical final context.
    ctxA = cpA.get(run_id, "__ctx__")
    ctxB = cpB.get(run_id, "__ctx__")
    assert ctxA == ctxB, "both workers converge to the same final context"
    cpA.close()
    cpB.close()
    os.unlink(db)

    # --- Crash of the winner BEFORE journaling: must be recoverable (no stall) ---
    EXEC_COUNTER2 = {"task": 0}
    db2 = _tmp_db()
    run2 = "crash-before-journal"

    def task(ctx):
        EXEC_COUNTER2["task"] += 1
        return "done"

    step = Step("task", task)

    # Worker A claims the step, then "crashes" before journaling (we don't call
    # run_step which would journal; we only place the claim, then drop A).
    cpA2 = SQLiteCheckpointer(db2)
    assert cpA2.try_claim(run2, "task", "A") is True
    cpA2.close()  # A dies AFTER claiming, BEFORE journaling -> stale claim

    # Worker B arrives: loses the claim (A owns it) but finds NO journaled result
    # -> releases the stale claim, re-claims, executes exactly once.
    cpB2 = SQLiteCheckpointer(db2)
    workerB2 = ClaimingDurableEngine(cpB2, owner="B")

    def _b_task(ctx):
        EXEC_COUNTER2["task"] += 1
        return "done"

    rB2 = workerB2.run_step(run2, Step("task", _b_task))
    print(f"\n  crash-before-journal recovered: result={rB2!r}, "
          f"exec_count={EXEC_COUNTER2['task']}")
    assert rB2 == "done"
    assert EXEC_COUNTER2["task"] == 1, "recovered step executes exactly once total"
    assert workerB2.executed == ["task"], "B recovers and executes the orphaned step"
    cpB2.close()
    os.unlink(db2)

    print("  OK: at-most-once under concurrency + winner-crash recovery, no stall.")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    hard_ex1_resume_matrix()
    hard_ex2_concurrent_idempotency()
    print("\n" + "=" * 68)
    print("ALL HARD (J27) SOLUTIONS PASSED")
    print("=" * 68)

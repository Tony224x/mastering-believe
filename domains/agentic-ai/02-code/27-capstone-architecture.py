"""
Day 27 -- Capstone (architecture & setup): the durable building blocks.

This module defines the reusable foundations that the Day 28 capstone agent
assembles. Each block is small, runnable, and independently testable:

  1. VirtualFS          -- local-disk scratchpad / context offloading (J15)
  2. SQLiteCheckpointer -- durable key/value checkpoint store (J20/J25)
  3. DurableEngine      -- run a sequence of idempotent steps, journal each
                           completed step, and RESUME after a crash without
                           re-executing already-finished work (J20)
  4. ModelRouter        -- mock cost-aware weak/strong routing (J24)
  5. SubAgent           -- base class for a context-ISOLATED sub-agent (J15/J9)

All stdlib (sqlite3 included), no API key, no network. The "LLM" is a
deterministic mock so the focus stays on the durable architecture.

Run:
    python domains/agentic-ai/02-code/27-capstone-architecture.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ===========================================================================
# 1. VIRTUAL FILESYSTEM -- context offloading / scratchpad
# ===========================================================================

class VirtualFS:
    """A tiny disk-backed key->text store used to offload context out of the
    prompt window (J15). Sub-agents write notes/artifacts here instead of
    keeping everything in the LLM context.
    """

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="capstone_fs_"))
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str) -> None:
        (self.root / name).write_text(content, encoding="utf-8")

    def read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def list(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())


# ===========================================================================
# 2. SQLITE CHECKPOINTER -- durable state
# ===========================================================================

class SQLiteCheckpointer:
    """Durable checkpoint store backed by sqlite3 (file or :memory:).

    Survives process restarts when a real file path is used: that is the
    property the capstone relies on to resume after a crash.
    """

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

    def close(self) -> None:
        self._conn.close()


# ===========================================================================
# 3. DURABLE ENGINE -- crash-resumable sequence of steps
# ===========================================================================

@dataclass
class Step:
    """A named, idempotent unit of work.

    `fn(ctx) -> result` must be deterministic given ctx so that replaying it
    (or skipping it after a crash) is safe.
    """
    name: str
    fn: Callable[[dict], Any]


class CrashSignal(Exception):
    """Raised to simulate a process crash mid-workflow (for the demo)."""


class DurableEngine:
    """Runs a list of Steps, journaling each completed step's result to the
    checkpointer. If the process crashes mid-run, a second invocation with the
    same run_id RESUMES: journaled steps are loaded (not re-executed), and the
    engine continues from the first unfinished step.
    """

    def __init__(self, checkpointer: SQLiteCheckpointer) -> None:
        self.cp = checkpointer
        self.executed: list[str] = []   # steps actually run this invocation
        self.skipped: list[str] = []    # steps loaded from the journal

    def run(self, run_id: str, steps: list[Step],
            crash_before: str | None = None) -> dict:
        self.executed = []
        self.skipped = []
        ctx: dict = self.cp.get(run_id, "__ctx__") or {}

        for step in steps:
            journal_key = f"step::{step.name}"
            cached = self.cp.get(run_id, journal_key)
            if cached is not None:
                # Already completed in a previous (crashed) run -> resume.
                ctx[step.name] = cached
                self.skipped.append(step.name)
                continue

            # Simulate a crash that happens BEFORE this step commits.
            if crash_before is not None and step.name == crash_before:
                raise CrashSignal(f"crash before step '{step.name}'")

            result = step.fn(ctx)
            ctx[step.name] = result
            # Persist the step result AND the running context atomically enough
            # for the demo (sqlite autocommit per put).
            self.cp.put(run_id, journal_key, result)
            self.cp.put(run_id, "__ctx__", ctx)
            self.executed.append(step.name)

        return ctx


# ===========================================================================
# 4. MODEL ROUTER -- cost-aware weak/strong routing (mock)
# ===========================================================================

@dataclass
class ModelRouter:
    """Routes a task to a cheap or expensive mock model by complexity, and
    tracks cumulative cost so the capstone can report savings (J24).
    """
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


# ===========================================================================
# 5. SUBAGENT -- context-isolated worker
# ===========================================================================

class SubAgent:
    """Base class for a sub-agent with its OWN isolated context.

    Context isolation (J15): each sub-agent keeps a private message buffer so
    one agent's verbose history never pollutes another's context window. Only
    the final, compact result is returned to the orchestrator.
    """

    role: str = "base"

    def __init__(self, fs: VirtualFS, router: ModelRouter) -> None:
        self.fs = fs
        self.router = router
        self._context: list[str] = []   # private, never shared verbatim

    def observe(self, note: str) -> None:
        self._context.append(note)

    def context_size(self) -> int:
        return sum(len(c) for c in self._context)

    def run(self, task: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


# ===========================================================================
# DEMO
# ===========================================================================

def _demo() -> None:
    print("=" * 64)
    print("Day 27 -- Durable building blocks")
    print("=" * 64)

    # --- VirtualFS ---
    fs = VirtualFS()
    fs.write("todo.md", "- [ ] gather facts\n- [ ] write report")
    print(f"\n[VirtualFS] root={fs.root}")
    print(f"[VirtualFS] files={fs.list()}")

    # --- Durable crash + resume demo on a real sqlite FILE ---
    db_path = str(Path(fs.root) / "state.db")
    run_id = "demo-run-1"

    def gather(ctx):   return "facts: 3 items"
    def analyze(ctx):  return f"analysis of [{ctx['gather']}]"
    def report(ctx):   return f"REPORT based on {ctx['analyze']}"

    steps = [Step("gather", gather), Step("analyze", analyze), Step("report", report)]

    print("\n[DurableEngine] attempt 1 (crash before 'report')")
    cp = SQLiteCheckpointer(db_path)
    engine = DurableEngine(cp)
    try:
        engine.run(run_id, steps, crash_before="report")
    except CrashSignal as exc:
        print(f"  CRASH: {exc}")
        print(f"  executed so far : {engine.executed}")
    cp.close()  # simulate the process dying (connection lost)

    print("\n[DurableEngine] attempt 2 (fresh process, same run_id + db)")
    cp2 = SQLiteCheckpointer(db_path)   # re-open the SAME durable file
    engine2 = DurableEngine(cp2)
    final = engine2.run(run_id, steps)   # no crash this time
    print(f"  skipped (resumed) : {engine2.skipped}")
    print(f"  executed (new)    : {engine2.executed}")
    print(f"  final report      : {final['report']!r}")
    assert engine2.skipped == ["gather", "analyze"], "should resume journaled steps"
    assert engine2.executed == ["report"], "should only run the unfinished step"
    cp2.close()

    # --- Router ---
    router = ModelRouter()
    for t in ["format date", "summarize this very long multi step analysis task "
              "that clearly needs the strong model to handle properly here"]:
        print(f"\n[ModelRouter] {router.route(t):>6} <- {t[:40]}...")
    print(f"[ModelRouter] total_cost={router.total_cost}, routed={router.routed}")

    print("\nDemo complete: crash-resume works, no finished step re-executed.")


if __name__ == "__main__":
    _demo()

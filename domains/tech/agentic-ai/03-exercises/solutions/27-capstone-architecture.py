"""
Day 27 -- Solutions to the capstone-architecture exercises.

Run the whole file to execute every solution.

    python domains/tech/agentic-ai/03-exercises/solutions/27-capstone-architecture.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day27 = import_module("27-capstone-architecture")
VirtualFS = day27.VirtualFS
SQLiteCheckpointer = day27.SQLiteCheckpointer
DurableEngine = day27.DurableEngine
Step = day27.Step
CrashSignal = day27.CrashSignal
ModelRouter = day27.ModelRouter


# ===========================================================================
# SOLUTION 1 -- CompactingVirtualFS
# ===========================================================================

class CompactingVirtualFS(VirtualFS):
    def __init__(self, root=None, max_lines: int = 20) -> None:
        super().__init__(root)
        self.max_lines = max_lines

    def append(self, name: str, line: str) -> None:
        lines = self.read(name).splitlines() if self.exists(name) else []
        lines.append(line)
        if len(lines) > self.max_lines:
            dropped = len(lines) - 10
            lines = lines[:5] + [f"... [compacted {dropped} lines] ..."] + lines[-5:]
        self.write(name, "\n".join(lines))


def solution_1() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 1 -- CompactingVirtualFS")
    print("#" * 60)
    fs = CompactingVirtualFS(max_lines=20)
    for i in range(50):
        fs.append("log.md", f"event {i}")
    final = fs.read("log.md").splitlines()
    print(f"final line count: {len(final)}")
    print("\n".join(final))
    # Compaction caps the file at max_lines; right after a compaction it is 11
    # lines (5 head + marker + 5 tail), then it can grow back up to max_lines.
    assert len(final) <= fs.max_lines
    assert any("[compacted" in ln for ln in final)
    assert final[0] == "event 0" and final[-1] == "event 49"
    print("  [check] compaction capped file, kept head+tail with marker -> OK")


# ===========================================================================
# SOLUTION 2 -- AuditedDurableEngine
# ===========================================================================

class AuditedDurableEngine(DurableEngine):
    def __init__(self, checkpointer) -> None:
        super().__init__(checkpointer)
        self.audit: list[dict] = []

    def run(self, run_id, steps, crash_before=None):
        self.executed, self.skipped = [], []
        ctx = self.cp.get(run_id, "__ctx__") or {}
        ts = 0
        for step in steps:
            ts += 1
            cached = self.cp.get(run_id, f"step::{step.name}")
            if cached is not None:
                ctx[step.name] = cached
                self.skipped.append(step.name)
                self.audit.append({"step": step.name, "action": "skipped", "ts": ts})
                continue
            if crash_before is not None and step.name == crash_before:
                raise CrashSignal(f"crash before step '{step.name}'")
            result = step.fn(ctx)
            ctx[step.name] = result
            self.cp.put(run_id, f"step::{step.name}", result)
            self.cp.put(run_id, "__ctx__", ctx)
            self.executed.append(step.name)
            self.audit.append({"step": step.name, "action": "executed", "ts": ts})
        return ctx


def solution_2() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 2 -- AuditedDurableEngine")
    print("#" * 60)
    fs = VirtualFS()
    db = str(Path(fs.root) / "audit.db")
    rid = "audit-run"
    steps = [Step("gather", lambda c: "g"), Step("analyze", lambda c: "a"),
             Step("report", lambda c: "r")]

    cp = SQLiteCheckpointer(db)
    eng = AuditedDurableEngine(cp)
    try:
        eng.run(rid, steps, crash_before="report")
    except CrashSignal as exc:
        print(f"  crash: {exc}")
    cp.close()

    cp2 = SQLiteCheckpointer(db)
    eng2 = AuditedDurableEngine(cp2)
    eng2.run(rid, steps)
    print("  audit (run 2):", eng2.audit)
    actions = [e["action"] for e in eng2.audit]
    assert actions == ["skipped", "skipped", "executed"]
    assert [e["ts"] for e in eng2.audit] == [1, 2, 3]
    cp2.close()
    print("  [check] audit shows 2 skipped + 1 executed on resume -> OK")


# ===========================================================================
# SOLUTION 3 -- BudgetedRouter
# ===========================================================================

class BudgetExceeded(Exception):
    pass


class BudgetedRouter:
    TIERS = [("strong", 8.0), ("weak", 1.0), ("nano", 0.2)]
    COST = dict(TIERS)

    def __init__(self, budget: float) -> None:
        self.budget = budget
        self.total_cost = 0.0
        self.routed: list[str] = []

    def _natural_tier(self, task: str) -> str:
        n = len(task.split())
        if n < 6:
            return "nano"
        if n < 12:
            return "weak"
        return "strong"

    def route(self, task: str) -> str:
        # Order of degradation: strong -> weak -> nano
        order = ["strong", "weak", "nano"]
        start = order.index(self._natural_tier(task))
        for tier in order[start:]:
            if self.total_cost + self.COST[tier] <= self.budget:
                self.total_cost += self.COST[tier]
                self.routed.append(tier)
                return tier
        raise BudgetExceeded(f"budget {self.budget} exhausted (used {self.total_cost})")


def solution_3() -> None:
    print("\n" + "#" * 60)
    print("# SOLUTION 3 -- BudgetedRouter")
    print("#" * 60)
    # Budget forces degradation: a complex query would cost 8 (strong) but only
    # ~1.2 is left after the first call.
    r = BudgetedRouter(budget=2.0)
    t1 = r.route("short task")                                   # nano (0.2)
    t2 = r.route("analyse and design a complex multi step plan for the system")  # wants strong(8) -> degrade
    print(f"  routed: {r.routed} total_cost={r.total_cost}")
    assert t1 == "nano"
    assert t2 in ("weak", "nano"), "complex query must degrade under tight budget"
    assert r.total_cost <= r.budget

    # Exhaustion: zero budget on a complex query -> exception
    r2 = BudgetedRouter(budget=0.0)
    try:
        r2.route("analyse and design a complex multi step plan for the whole system end to end")
        raise AssertionError("should have raised BudgetExceeded")
    except BudgetExceeded as exc:
        print(f"  BudgetExceeded raised as expected: {exc}")
    print("  [check] degradation under budget + exception on exhaustion -> OK")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
    print("\nAll Day 27 solutions ran successfully.")

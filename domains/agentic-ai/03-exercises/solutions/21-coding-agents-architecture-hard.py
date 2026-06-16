"""
Solutions -- Day 21 (HARD): Coding agents architecture

Contains solutions for:
  - Hard Ex 1: a mini SWE-bench-like coding agent that LOCALIZES a buggy
               function from a failing test, PROPOSES an edit, applies it,
               reruns tests, and ITERATES until green or budget exhausted --
               and stops cleanly on an unfixable bug.
  - Hard Ex 2: an atomic edit TRANSACTION system (stage -> lint+test gate ->
               commit/rollback) over a multi-file mock repo, proving ATOMICITY
               (a partial failure leaves the repo strictly unchanged).

Self-contained: pure stdlib, no network, no API key. The in-process test runner
uses `exec` on TRUSTED hand-written source strings only (a pedagogical mock, not
a sandbox for untrusted code).

Run:  python 03-exercises/solutions/21-coding-agents-architecture-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ==========================================================================
# Shared ACI-style primitives (edit tool + in-process test runner)
# ==========================================================================

@dataclass
class EditResult:
    ok: bool
    new_source: Optional[str]
    reason: str = ""


def apply_edit(source: str, find: str, replace: str) -> EditResult:
    """Unique-match search/replace: reject missing or ambiguous `find`."""
    count = source.count(find)
    if count == 0:
        return EditResult(False, None, f"not found: {find!r}")
    if count > 1:
        return EditResult(False, None, f"ambiguous: {count} matches")
    return EditResult(True, source.replace(find, replace, 1), "")


def run_tests(repo: dict[str, str], tests: list[tuple[str, tuple, Any]]) -> dict:
    """
    In-process test runner over a mock repo (dict filename -> source).
    Compiles all files in a fresh namespace; a broken module is reported as
    `error`, a wrong result / raised exception as a `failure`.

    `exec` runs TRUSTED hand-written strings only (pedagogical mock).
    """
    ns: dict[str, Any] = {}
    try:
        for fname, src in repo.items():
            exec(compile(src, fname, "exec"), ns)
    except (SyntaxError, NameError) as e:
        return {"passed": 0, "failed": 0, "failures": [], "error": f"{type(e).__name__}: {e}"}

    passed, failed, failures = 0, 0, []
    for fn_name, args, expected in tests:
        try:
            result = ns[fn_name](*args)
            if result == expected:
                passed += 1
            else:
                failed += 1
                failures.append({"fn": fn_name, "args": args, "got": result, "expected": expected})
        except Exception as e:
            failed += 1
            failures.append({"fn": fn_name, "args": args, "error": f"{type(e).__name__}: {e}"})
    return {"passed": passed, "failed": failed, "failures": failures, "error": None}


# ==========================================================================
# HARD EXERCISE 1 -- mini SWE-bench-like coding agent
# ==========================================================================

def localize(repo: dict[str, str], failing_fn: str) -> Optional[tuple[str, int]]:
    """
    Localize a function definition from the name of a failing test's target.
    Searches every file for `def <fn>(` and returns (filename, 1-indexed line).
    """
    needle = f"def {failing_fn}("
    for fname, src in repo.items():
        for i, line in enumerate(src.splitlines(), start=1):
            if needle in line:
                return (fname, i)
    return None


# A tiny "fix bank": maps a known buggy operator (per function) to its fix.
# Stored as (bad_op, good_op); the edit block is built with the function's
# signature line so the `find` is UNIQUE within the file (two functions can
# share the same broken body otherwise).
_FIX_BANK: dict[str, tuple[str, str]] = {
    "add": ("-", "+"),
    "mul": ("+", "*"),
    "sub": ("+", "-"),
}


def propose_edit(repo: dict[str, str], filename: str, fn_name: str) -> Optional[tuple[str, str]]:
    """
    Propose a plausible (find, replace) edit block for a localized function.

    The find block includes the `def` header so it targets exactly ONE function,
    even if another function shares the same buggy body. Returns None if no
    known fix matches the actual source -> "cannot fix".
    """
    candidate = _FIX_BANK.get(fn_name)
    if candidate is None:
        return None
    bad_op, good_op = candidate
    src = repo.get(filename, "")
    # Build a find block anchored on the function header so it's unambiguous.
    find = f"def {fn_name}(a, b):\n    return a {bad_op} b"
    if find not in src:
        return None  # the known buggy pattern is not present -> don't guess
    replace = f"def {fn_name}(a, b):\n    return a {good_op} b"
    return (find, replace)


def mini_swe_agent(repo: dict[str, str], tests: list[tuple[str, tuple, Any]],
                   max_iters: int = 12) -> dict:
    """
    Full loop: run_tests -> pick a failing test -> localize -> propose_edit ->
    apply (unique-match) -> rerun -> keep if progress else rollback -> iterate.
    Stops at green, when no fix is proposable, or at the iteration budget.
    """
    current = dict(repo)
    trajectory: list[dict] = []
    fixed: list[str] = []
    res = run_tests(current, tests)
    iters = 0
    tried: set[tuple[str, str, str]] = set()  # avoid retrying the same dead-end edit

    while iters < max_iters:
        if res["failed"] == 0 and res["error"] is None:
            break  # green

        # Pick the first failing test as the localization signal.
        if not res["failures"]:
            break
        failing_fn = res["failures"][0]["fn"]
        loc = localize(current, failing_fn)
        if loc is None:
            trajectory.append({"fn": failing_fn, "step": "localize", "result": "not found"})
            break  # cannot even find it -> stop, don't loop

        filename, lineno = loc
        proposal = propose_edit(current, filename, failing_fn)
        if proposal is None:
            trajectory.append({"fn": failing_fn, "loc": (filename, lineno),
                               "step": "propose", "result": "no known fix"})
            break  # unfixable bug -> stop cleanly

        find, replace = proposal
        signature = (filename, find, replace)
        if signature in tried:
            trajectory.append({"fn": failing_fn, "step": "edit", "result": "already tried"})
            break
        tried.add(signature)

        iters += 1
        before = res["passed"]
        edited = apply_edit(current[filename], find, replace)
        if not edited.ok:
            trajectory.append({"fn": failing_fn, "step": "edit", "result": edited.reason})
            break

        candidate = dict(current)
        candidate[filename] = edited.new_source  # type: ignore[assignment]
        new_res = run_tests(candidate, tests)
        progressed = new_res["error"] is None and new_res["passed"] > before
        if progressed:
            current = candidate
            res = new_res
            fixed.append(failing_fn)
            trajectory.append({"fn": failing_fn, "loc": (filename, lineno),
                               "step": "edit", "result": "kept", "passed": new_res["passed"]})
        else:
            # Rollback: the edit didn't help -> stop (no other fix known here).
            trajectory.append({"fn": failing_fn, "step": "edit",
                               "result": "rolled back (no progress)"})
            break

    green = res["failed"] == 0 and res["error"] is None
    return {"green": green, "iters": iters, "fixed": fixed,
            "final_repo": current, "trajectory": trajectory}


def hard_ex1_mini_swe_agent() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: mini SWE-bench-like coding agent")
    print("=" * 60)

    # Repo with TWO seeded bugs across one file.
    repo = {
        "mathlib.py": (
            "def add(a, b):\n"
            "    return a - b\n"      # BUG: should be +
            "\n"
            "def mul(a, b):\n"
            "    return a + b\n"      # BUG: should be *
        ),
    }
    tests = [
        ("add", (2, 3), 5),
        ("add", (-1, 1), 0),
        ("mul", (3, 4), 12),
        ("mul", (-2, 5), -10),
    ]

    result = mini_swe_agent(repo, tests, max_iters=12)
    print(f"\n  green={result['green']} iters={result['iters']} fixed={result['fixed']}")
    for step in result["trajectory"]:
        print(f"    {step}")

    # Both bugs fixed, within budget.
    assert result["green"] is True, result
    assert set(result["fixed"]) == {"add", "mul"}, result["fixed"]
    assert result["iters"] <= 12

    # Every KEPT edit strictly increased the passing count (monotone progress).
    kept = [s for s in result["trajectory"] if s.get("result") == "kept"]
    passes = [s["passed"] for s in kept]
    assert passes == sorted(passes) and len(set(passes)) == len(passes), passes

    # Localization sanity check.
    loc = localize(repo, "mul")
    assert loc == ("mathlib.py", 4), loc

    # --- Unfixable bug: no fix bank entry matches -> stop cleanly, repo intact.
    print("\n  Unfixable-bug scenario:")
    repo_u = {"weird.py": "def power(a, b):\n    return a + b\n"}  # not in fix bank
    tests_u = [("power", (2, 3), 8)]
    result_u = mini_swe_agent(repo_u, tests_u, max_iters=12)
    print(f"    green={result_u['green']} iters={result_u['iters']} "
          f"trajectory_last={result_u['trajectory'][-1] if result_u['trajectory'] else None}")
    assert result_u["green"] is False
    assert result_u["iters"] <= 12  # bounded, no infinite loop
    # The part it cannot fix is left untouched.
    assert result_u["final_repo"]["weird.py"] == repo_u["weird.py"]

    print("\n  PASS -- localize+fix iterates to green, stops cleanly on unfixable.\n")


# ==========================================================================
# HARD EXERCISE 2 -- atomic edit transaction (stage -> gate -> commit/rollback)
# ==========================================================================

class StageError(Exception):
    """Raised when a staged edit cannot be applied (absent/ambiguous find)."""


class EditTransaction:
    """
    All-or-nothing edits across a multi-file mock repo.

    - snapshot taken at open; the REAL repo is never touched before commit
    - stage(): apply unique-match edits to a working copy
    - gate(): lint (compilable) + run tests on the working copy
    - commit(): copy working -> real repo ONLY if the gate passes; else no-op
    - rollback(): discard the working copy
    """

    def __init__(self, repo: dict[str, str]) -> None:
        self.repo = repo                       # the real repo (mutated only on commit)
        self.snapshot = dict(repo)             # immutable reference of the start state
        self.working = dict(repo)              # candidate state being edited
        self.staged: list[tuple[str, str, str]] = []

    def stage(self, filename: str, find: str, replace: str) -> None:
        """Apply an edit to the working copy with the uniqueness rule."""
        if filename not in self.working:
            raise StageError(f"unknown file: {filename}")
        res = apply_edit(self.working[filename], find, replace)
        if not res.ok:
            raise StageError(f"stage rejected for {filename}: {res.reason}")
        self.working[filename] = res.new_source  # type: ignore[assignment]
        self.staged.append((filename, find, replace))

    def gate(self, tests: list[tuple[str, tuple, Any]]) -> tuple[bool, dict]:
        """Lint (each file compiles) then run tests on the working copy."""
        for fname, src in self.working.items():
            try:
                compile(src, fname, "exec")
            except SyntaxError as e:
                return False, {"stage": "lint", "file": fname, "error": f"SyntaxError: {e}"}
        res = run_tests(self.working, tests)
        if res["error"] is not None:
            return False, {"stage": "lint/import", "error": res["error"]}
        if res["failed"] > 0:
            return False, {"stage": "test", "failures": res["failures"]}
        return True, {"stage": "ok", "passed": res["passed"]}

    def commit(self, tests: list[tuple[str, tuple, Any]]) -> bool:
        """Commit atomically iff the gate passes; otherwise leave repo untouched."""
        ok, report = self.gate(tests)
        if not ok:
            self.last_report = report
            return False
        self.repo.clear()
        self.repo.update(self.working)   # atomic publish of all staged edits
        self.last_report = report
        return True

    def rollback(self) -> None:
        """Throw away the working copy; the real repo stays at the snapshot."""
        self.working = dict(self.snapshot)
        self.staged.clear()


def hard_ex2_edit_transaction() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: atomic edit transaction (stage -> gate -> commit)")
    print("=" * 60)

    # --- (A) Successful commit: 2 edits across 2 files make tests green. ---
    repo = {
        "ops.py": "def add(a, b):\n    return a - b\n",   # bug
        "util.py": "def dbl(x):\n    return x + 1\n",      # bug
    }
    tests = [("add", (2, 3), 5), ("dbl", (4,), 8)]

    print(f"\n  (A) initial: {repo}")
    tx = EditTransaction(repo)
    tx.stage("ops.py", "return a - b", "return a + b")
    tx.stage("util.py", "return x + 1", "return x * 2")
    committed = tx.commit(tests)
    print(f"      commit -> {committed}; repo now: {repo}")
    assert committed is True
    assert repo["ops.py"] == "def add(a, b):\n    return a + b\n"
    assert repo["util.py"] == "def dbl(x):\n    return x * 2\n"

    # --- (B) Atomic failure by TEST: one edit causes a regression. ---
    repo_b = {
        "ops.py": "def add(a, b):\n    return a - b\n",
        "util.py": "def dbl(x):\n    return x * 2\n",
    }
    snapshot_b = dict(repo_b)
    tests_b = [("add", (2, 3), 5), ("dbl", (4,), 8)]

    print(f"\n  (B) initial: {repo_b}")
    tx_b = EditTransaction(repo_b)
    tx_b.stage("ops.py", "return a - b", "return a + b")   # good
    tx_b.stage("util.py", "return x * 2", "return x * 3")  # BAD: breaks dbl(4)==8
    committed_b = tx_b.commit(tests_b)
    print(f"      commit -> {committed_b}; report={tx_b.last_report['stage']}")
    print(f"      repo unchanged? {repo_b == snapshot_b}")
    assert committed_b is False
    # ATOMICITY: neither edit was applied -> repo strictly equals the snapshot.
    assert repo_b == snapshot_b, repo_b

    # --- (C) Atomic failure by LINT: an edit introduces a SyntaxError. ---
    repo_c = {"ops.py": "def add(a, b):\n    return a - b\n"}
    snapshot_c = dict(repo_c)
    tests_c = [("add", (2, 3), 5)]

    print(f"\n  (C) initial: {repo_c}")
    tx_c = EditTransaction(repo_c)
    tx_c.stage("ops.py", "return a - b", "return a +")  # broken syntax
    committed_c = tx_c.commit(tests_c)
    print(f"      commit -> {committed_c}; report={tx_c.last_report['stage']}")
    print(f"      repo unchanged? {repo_c == snapshot_c}")
    assert committed_c is False
    assert tx_c.last_report["stage"] == "lint"
    assert repo_c == snapshot_c, repo_c

    # --- Staging an ambiguous/absent find is rejected, working copy untouched. ---
    print("\n  (D) staging guard (ambiguous/absent find):")
    repo_d = {"f.py": "x = 1\ny = 1\n"}
    tx_d = EditTransaction(repo_d)
    try:
        tx_d.stage("f.py", "= 1", "= 2")  # appears twice -> ambiguous
        raise AssertionError("ambiguous stage should be rejected")
    except StageError as e:
        print(f"      rejected: {e}")
        assert "ambiguous" in str(e)
    assert tx_d.working == repo_d, "rejected stage must not alter the working copy"

    print("\n  PASS -- commit publishes all edits; failures leave repo atomic.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 21 HARD Solutions -- Coding agents architecture")
    print("  (offline, stdlib only -- no network, no API key)")
    print("#" * 60)

    hard_ex1_mini_swe_agent()
    hard_ex2_edit_transaction()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

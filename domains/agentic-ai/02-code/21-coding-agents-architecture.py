"""
Day 21 -- Coding Agents Architecture: ACI, edit/search/run loop, SWE-bench.

Demonstrates a minimal Agent-Computer Interface (ACI) operating on a toy
repository created inside a temporary directory.  A scripted CodingAgent
(no LLM key required -- the "model" is a deterministic policy) receives an
issue description, searches for the bug, edits the fix, and re-runs tests
until they turn green.

Key components:
  - ACI          : open_file, search, edit, run_tests -- tools designed for a
                   "new kind of user" (the LM), not for a human IDE
  - CodingAgent  : React-style loop (Thought -> Action -> Observation)
  - ToyRepo      : a self-contained broken Python project in a tempdir
  - Trajectory   : full step-by-step log printed at the end

Run:
    python domains/agentic-ai/02-code/21-coding-agents-architecture.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional


# ===========================================================================
# 1. TOY REPOSITORY -- a small broken Python project
# ===========================================================================

def create_toy_repo(root: Path) -> None:
    """
    Write a minimal Python project into *root*.

    Structure:
        mathlib/
            __init__.py
            operations.py   <-- contains TWO intentional bugs
        tests/
            test_operations.py   <-- unittest.TestCase so stdlib runner works
    """
    (root / "mathlib").mkdir()

    # __init__.py -- correct, just re-exports
    (root / "mathlib" / "__init__.py").write_text(
        "from .operations import add, multiply\n"
    )

    # operations.py -- BUG 1: add uses subtraction; BUG 2: multiply uses addition
    (root / "mathlib" / "operations.py").write_text(
        textwrap.dedent("""\
            def add(a, b):
                \"\"\"Return the sum of a and b.\"\"\"
                return a - b   # BUG: should be a + b


            def multiply(a, b):
                \"\"\"Return the product of a and b.\"\"\"
                return a + b   # BUG: should be a * b
        """)
    )

    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")

    # Test suite -- uses unittest.TestCase so stdlib discovery works without pytest
    (root / "tests" / "test_operations.py").write_text(
        textwrap.dedent("""\
            import unittest
            import sys
            import os
            # Make the repo root importable when run from the repo root
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from mathlib import add, multiply


            class TestAdd(unittest.TestCase):
                def test_positive(self):
                    self.assertEqual(add(2, 3), 5)

                def test_zero(self):
                    self.assertEqual(add(0, 0), 0)

                def test_negative(self):
                    self.assertEqual(add(-1, 1), 0)


            class TestMultiply(unittest.TestCase):
                def test_positive(self):
                    self.assertEqual(multiply(3, 4), 12)

                def test_negative(self):
                    self.assertEqual(multiply(-2, 5), -10)

                def test_zero(self):
                    self.assertEqual(multiply(0, 99), 0)


            if __name__ == "__main__":
                unittest.main()
        """)
    )


# ===========================================================================
# 2. AGENT-COMPUTER INTERFACE (ACI)
#    Each tool returns a structured observation string, not raw bytes.
# ===========================================================================

@dataclass
class ACI:
    """
    Minimal ACI on top of the toy repo.

    Tools mirror SWE-agent's design:
      - open_file  : show a slice of a file with line numbers
      - search     : grep for a pattern across all .py files
      - edit       : replace an exact snippet (fails loudly if not found)
      - run_tests  : execute pytest and return a compact summary
    """

    root: Path
    # Track files already opened for audit
    _opened: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # open_file
    # -----------------------------------------------------------------------
    def open_file(self, rel_path: str, start: int = 1, window: int = 40) -> str:
        """
        Return lines [start, start+window) of *rel_path*, with line numbers.
        start is 1-indexed.  Returns an error string if the file is missing.
        """
        p = self.root / rel_path
        if not p.exists():
            return f"[ERROR] File not found: {rel_path}"
        lines = p.read_text().splitlines()
        end = min(start - 1 + window, len(lines))
        selected = lines[start - 1 : end]
        numbered = "\n".join(
            f"{start + i:4d} | {line}" for i, line in enumerate(selected)
        )
        self._opened.append(rel_path)
        return f"[open_file] {rel_path} lines {start}-{end}:\n{numbered}"

    # -----------------------------------------------------------------------
    # search
    # -----------------------------------------------------------------------
    def search(self, pattern: str, path: str = ".") -> str:
        """
        Grep *pattern* (literal string) in all .py files under *path*.
        Returns matching file:line:content triples, or a no-match message.
        """
        base = self.root / path
        matches: list[str] = []
        for py_file in sorted(base.rglob("*.py")):
            rel = py_file.relative_to(self.root)
            for lineno, text in enumerate(py_file.read_text().splitlines(), 1):
                if pattern in text:
                    matches.append(f"{rel}:{lineno}: {text.rstrip()}")
        if not matches:
            return f"[search] No match for pattern: {pattern!r}"
        return "[search] Results:\n" + "\n".join(matches)

    # -----------------------------------------------------------------------
    # edit
    # -----------------------------------------------------------------------
    def edit(self, rel_path: str, old_text: str, new_text: str) -> str:
        """
        Replace the first occurrence of *old_text* with *new_text* in *rel_path*.

        Design decision (mirroring SWE-agent): if *old_text* is not found, the
        edit is REJECTED with an error.  This prevents silent no-ops caused by
        hallucinated snippets.
        """
        p = self.root / rel_path
        if not p.exists():
            return f"[ERROR] File not found: {rel_path}"
        content = p.read_text()
        if old_text not in content:
            return (
                f"[ERROR] edit rejected -- pattern not found in {rel_path}.\n"
                f"Pattern was:\n{old_text}"
            )
        updated = content.replace(old_text, new_text, 1)  # only first occurrence
        p.write_text(updated)
        return f"[edit] Applied to {rel_path}. Changed:\n  - {old_text.strip()!r}\n  + {new_text.strip()!r}"

    # -----------------------------------------------------------------------
    # run_tests
    # -----------------------------------------------------------------------
    def run_tests(self, flags: str = "") -> str:
        """
        Run the test suite in *root* using unittest discovery (stdlib only --
        no pytest dependency required).

        Falls back gracefully: any ImportError or assertion error is reported
        as a FAILED observation so the agent can iterate.
        """
        # Use Python's built-in unittest runner via subprocess so that the
        # toy repo's sys.path is isolated and the tests run cleanly.
        cmd = [
            sys.executable, "-m", "unittest", "discover",
            "-s", "tests",       # discover tests/ folder
            "-p", "test_*.py",   # standard naming convention
            "-v",                # verbose: each test name + pass/fail
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.root),
        )
        # unittest writes results to stderr; stdout may be empty
        output = (result.stderr + result.stdout).strip()
        status = "PASSED" if result.returncode == 0 else "FAILED"
        # Limit observation to last 30 lines to avoid context explosion
        lines = output.splitlines()
        snippet = "\n".join(lines[-30:]) if len(lines) > 30 else output
        return f"[run_tests] Status: {status}\n{snippet}"


# ===========================================================================
# 3. SCRIPTED CODING AGENT
#    A deterministic policy that mimics the Thought -> Action -> Observation
#    loop of a real LLM-based coding agent.
# ===========================================================================

@dataclass
class Step:
    """One step in the agent trajectory."""

    iteration: int
    thought: str
    action: str        # human-readable description of the tool call
    observation: str   # raw tool return value


@dataclass
class CodingAgent:
    """
    A rule-based coding agent that resolves a fixed bug issue.

    In a real system (Claude Code, SWE-agent) the "policy" is an LLM.
    Here we hard-code the reasoning to show the *shape* of the loop
    without requiring any API key.

    The agent follows these phases:
      Phase 1  -- run tests to see what fails (gather initial observation)
      Phase 2  -- search for the first bug (add)
      Phase 3  -- open the file for context
      Phase 4  -- edit the first bug
      Phase 5  -- run tests again (should have one failure remaining)
      Phase 6  -- search for the second bug (multiply)
      Phase 7  -- edit the second bug
      Phase 8  -- run tests again (all green)
    """

    aci: ACI
    issue: str
    max_iterations: int = 15
    trajectory: list[Step] = field(default_factory=list)

    def _record(
        self, iteration: int, thought: str, action: str, observation: str
    ) -> str:
        step = Step(iteration, thought, action, observation)
        self.trajectory.append(step)
        return observation

    def run(self) -> bool:
        """
        Execute the edit/search/run loop.
        Returns True if all tests pass at the end.
        """
        print("\n" + "=" * 70)
        print(f"ISSUE: {self.issue}")
        print("=" * 70)

        # ------------------------------------------------------------------
        # Step 1: Initial test run -- what is actually broken?
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=1,
            thought=(
                "I received the issue. First I must understand the current state "
                "of the test suite to know what is failing and where."
            ),
            action="run_tests()",
            observation=self.aci.run_tests(),
        )
        self._print_step(self.trajectory[-1])

        if "PASSED" in obs and "failed" not in obs.lower():
            print("[Agent] Tests already green -- nothing to fix.")
            return True

        # ------------------------------------------------------------------
        # Step 2: Search for the first bug keyword from the issue
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=2,
            thought=(
                "Tests fail.  The issue mentions 'add' returns wrong results. "
                "I'll search for 'def add' to locate the implementation."
            ),
            action="search('def add')",
            observation=self.aci.search("def add"),
        )
        self._print_step(self.trajectory[-1])

        # ------------------------------------------------------------------
        # Step 3: Open the file for context
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=3,
            thought=(
                "Found 'def add' in mathlib/operations.py. "
                "I'll open the file to see the full implementation and understand "
                "the bug before editing."
            ),
            action="open_file('mathlib/operations.py', start=1, window=20)",
            observation=self.aci.open_file("mathlib/operations.py", start=1, window=20),
        )
        self._print_step(self.trajectory[-1])

        # ------------------------------------------------------------------
        # Step 4: Fix the first bug (add: subtraction -> addition)
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=4,
            thought=(
                "Line 3 uses 'a - b' for add(), but the function should sum its "
                "arguments.  I'll replace 'return a - b' with 'return a + b'."
            ),
            action="edit('mathlib/operations.py', old='return a - b   # BUG: should be a + b', new='return a + b   # fixed')",
            observation=self.aci.edit(
                "mathlib/operations.py",
                old_text="    return a - b   # BUG: should be a + b",
                new_text="    return a + b   # fixed",
            ),
        )
        self._print_step(self.trajectory[-1])

        # ------------------------------------------------------------------
        # Step 5: Re-run tests -- first bug should be fixed
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=5,
            thought=(
                "First edit applied. Running tests to verify test_add now passes "
                "and to see if test_multiply still fails."
            ),
            action="run_tests()",
            observation=self.aci.run_tests(),
        )
        self._print_step(self.trajectory[-1])

        if "PASSED" in obs and "failed" not in obs.lower():
            print("[Agent] All tests green after first fix!")
            return True

        # ------------------------------------------------------------------
        # Step 6: Search for the second bug (multiply)
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=6,
            thought=(
                "test_multiply still fails.  The issue also mentions 'multiply'. "
                "I'll search for its implementation."
            ),
            action="search('def multiply')",
            observation=self.aci.search("def multiply"),
        )
        self._print_step(self.trajectory[-1])

        # ------------------------------------------------------------------
        # Step 7: Fix the second bug (multiply: addition -> multiplication)
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=7,
            thought=(
                "multiply() uses 'a + b' which is wrong -- it should multiply. "
                "I'll replace 'return a + b' with 'return a * b'."
            ),
            action="edit('mathlib/operations.py', old='return a + b   # BUG: should be a * b', new='return a * b   # fixed')",
            observation=self.aci.edit(
                "mathlib/operations.py",
                old_text="    return a + b   # BUG: should be a * b",
                new_text="    return a * b   # fixed",
            ),
        )
        self._print_step(self.trajectory[-1])

        # ------------------------------------------------------------------
        # Step 8: Final test run -- everything should be green
        # ------------------------------------------------------------------
        obs = self._record(
            iteration=8,
            thought=(
                "Both bugs fixed.  Running the full test suite one final time "
                "to confirm all tests pass before declaring success."
            ),
            action="run_tests()",
            observation=self.aci.run_tests(),
        )
        self._print_step(self.trajectory[-1])

        success = "PASSED" in obs and "failed" not in obs.lower()
        return success

    # -----------------------------------------------------------------------
    # Pretty-printer for console output
    # -----------------------------------------------------------------------
    @staticmethod
    def _print_step(step: Step) -> None:
        width = 70
        print(f"\n{'─' * width}")
        print(f"  Step {step.iteration}")
        print(f"{'─' * width}")
        print(f"  [Thought]     {step.thought}")
        print(f"  [Action]      {step.action}")
        print(f"  [Observation]")
        for line in step.observation.splitlines():
            print(f"    {line}")


# ===========================================================================
# 4. TRAJECTORY SUMMARY
# ===========================================================================

def print_trajectory_summary(agent: CodingAgent) -> None:
    """Print a compact trajectory table, like a .traj file in SWE-agent."""
    print("\n" + "=" * 70)
    print("TRAJECTORY SUMMARY")
    print("=" * 70)
    print(f"{'Step':>4}  {'Action':<50}  {'Result'}")
    print("-" * 70)
    for step in agent.trajectory:
        # Shorten observation to first non-empty line
        first_line = next(
            (ln.strip() for ln in step.observation.splitlines() if ln.strip()),
            "",
        )
        # Truncate for readability
        action_short = step.action[:48]
        result_short = first_line[:35]
        print(f"{step.iteration:>4}  {action_short:<50}  {result_short}")


# ===========================================================================
# 5. SHOW FINAL STATE OF THE FIXED FILE
# ===========================================================================

def print_fixed_file(aci: ACI) -> None:
    print("\n" + "=" * 70)
    print("FINAL STATE OF mathlib/operations.py")
    print("=" * 70)
    print(aci.open_file("mathlib/operations.py"))


# ===========================================================================
# 6. MAIN DEMO
# ===========================================================================

if __name__ == "__main__":
    issue = (
        "Bug report: mathlib.add(2, 3) returns -1 instead of 5. "
        "Also mathlib.multiply(3, 4) returns 7 instead of 12. "
        "Please locate and fix both bugs."
    )

    with TemporaryDirectory(prefix="toy_repo_") as tmpdir:
        root = Path(tmpdir)

        # --- Setup: create the broken toy repository -----------------------
        print("Setting up toy repository with two bugs ...")
        create_toy_repo(root)
        print(f"Repo created at: {root}")

        # --- Run the coding agent ------------------------------------------
        aci = ACI(root=root)
        agent = CodingAgent(aci=aci, issue=issue)

        try:
            success = agent.run()
        except Exception:
            traceback.print_exc()
            success = False

        # --- Print trajectory summary and final state ----------------------
        print_trajectory_summary(agent)
        print_fixed_file(aci)

        # --- Verdict -------------------------------------------------------
        print("\n" + "=" * 70)
        if success:
            print("RESULT: Agent resolved the issue -- all tests GREEN.")
        else:
            print("RESULT: Agent could NOT resolve the issue.")
        print("=" * 70)

        # Exit code reflects success so CI can detect failures
        sys.exit(0 if success else 1)

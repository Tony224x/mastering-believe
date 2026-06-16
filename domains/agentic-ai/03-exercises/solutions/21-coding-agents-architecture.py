"""
Day 21 -- Solutions to the easy exercises for coding agents architecture.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/21-coding-agents-architecture.py
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

# ---------------------------------------------------------------------------
# Import the day-21 module (ACI + create_toy_repo) without renaming the file
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location(
    "day21",
    SRC / "21-coding-agents-architecture.py",
)
day21 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
# Register before exec so that @dataclass can resolve cls.__module__
sys.modules["day21"] = day21
_spec.loader.exec_module(day21)  # type: ignore[union-attr]

ACI = day21.ACI
create_toy_repo = day21.create_toy_repo


# ===========================================================================
# SOLUTION 1 -- list_dir
# ===========================================================================

def add_list_dir(aci_instance: ACI) -> None:
    """
    Monkey-patch a list_dir method onto an existing ACI instance.

    In a real implementation this would be a proper method on the class.
    We keep it as a standalone function here so the solution is self-contained.
    """

    def list_dir(rel_path: str = ".", max_depth: int = 2) -> str:
        base = aci_instance.root / rel_path
        if not base.exists():
            return f"[ERROR] Path not found: {rel_path}"

        lines: list[str] = [f"[list_dir] {rel_path}"]

        def _recurse(directory: Path, depth: int) -> None:
            if depth > max_depth:
                return
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
            for entry in entries:
                indent = "  " * depth
                if entry.is_dir():
                    lines.append(f"{indent}{entry.name}/")
                    _recurse(entry, depth + 1)
                else:
                    lines.append(f"{indent}{entry.name}")

        _recurse(base, 1)
        return "\n".join(lines)

    # Attach as a bound-like callable to the instance
    import types
    aci_instance.list_dir = types.MethodType(  # type: ignore[attr-defined]
        lambda self, rel_path=".", max_depth=2: list_dir(rel_path, max_depth),
        aci_instance,
    )


def run_solution_1() -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("SOLUTION 1 -- list_dir ACI tool")
    print(sep)

    with TemporaryDirectory(prefix="sol1_") as tmpdir:
        root = Path(tmpdir)
        create_toy_repo(root)
        aci = ACI(root=root)
        add_list_dir(aci)

        # Case 1: root listing
        print("\n--- list_dir('.') ---")
        print(aci.list_dir("."))  # type: ignore[attr-defined]

        # Case 2: sub-folder listing
        print("\n--- list_dir('mathlib') ---")
        print(aci.list_dir("mathlib"))  # type: ignore[attr-defined]

        # Case 3: missing folder
        print("\n--- list_dir('missing_folder') ---")
        result = aci.list_dir("missing_folder")  # type: ignore[attr-defined]
        print(result)
        assert result.startswith("[ERROR]"), "Must return an error for missing paths"

    print("\n[OK] Solution 1 passed all assertions.")


# ===========================================================================
# SOLUTION 2 -- snapshot / restore (backtracking)
# ===========================================================================

class ACIWithBacktrack(ACI):
    """
    Extended ACI that supports file snapshots for backtracking.

    snapshot(rel_path) -> key : saves current content, returns opaque key
    restore(key)       -> str : restores file to snapshot content
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root=root)
        self._snapshots: dict[str, tuple[str, str]] = {}
        self._counter: int = 0

    def snapshot(self, rel_path: str) -> str:
        """
        Save the current content of *rel_path*.
        Returns an opaque key (e.g. 'snap_0') used to restore later.
        """
        p = self.root / rel_path
        if not p.exists():
            return f"[ERROR] Cannot snapshot -- file not found: {rel_path}"
        key = f"snap_{self._counter}"
        self._counter += 1
        self._snapshots[key] = (rel_path, p.read_text())
        return key

    def restore(self, key: str) -> str:
        """
        Restore the file saved under *key* to its snapshotted content.
        """
        if key not in self._snapshots:
            return f"[ERROR] Unknown snapshot key: {key!r}"
        rel_path, content = self._snapshots[key]
        p = self.root / rel_path
        p.write_text(content)
        return f"[restore] {rel_path} restored from snapshot {key!r}."


def run_solution_2() -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("SOLUTION 2 -- snapshot / restore (backtracking)")
    print(sep)

    with TemporaryDirectory(prefix="sol2_") as tmpdir:
        root = Path(tmpdir)
        create_toy_repo(root)
        aci = ACIWithBacktrack(root=root)

        target = "mathlib/operations.py"
        original_content = (root / target).read_text()

        # (a) Snapshot before the bad edit
        key = aci.snapshot(target)
        print(f"\n[snapshot] key = {key!r}")

        # (b) Apply an incorrect edit (still wrong, but different wrong)
        bad_edit_result = aci.edit(
            target,
            old_text="    return a - b   # BUG: should be a + b",
            new_text="    return a ** b  # wrong: power instead of sum",
        )
        print(f"\n[bad edit] {bad_edit_result}")

        # (c) Confirm tests still fail
        test_result = aci.run_tests()
        print(f"\n[run_tests after bad edit]\n{test_result}")
        assert "FAILED" in test_result, "Tests should still fail after bad edit"

        # (d) Restore snapshot
        restore_result = aci.restore(key)
        print(f"\n[restore] {restore_result}")

        # (e) Verify original content is back
        restored_content = (root / target).read_text()
        if restored_content == original_content:
            print("\nBacktrack successful -- file content matches original.")
        else:
            print("\nBacktrack FAILED -- content mismatch!")
            sys.exit(1)

    print("\n[OK] Solution 2 passed all assertions.")


# ===========================================================================
# SOLUTION 3 -- mini repo-map via ast
# ===========================================================================

def build_repo_map(root: Path) -> str:
    """
    Generate a compact map of all Python symbols in *root*.

    For each .py file: lists top-level functions, classes, and their methods.
    Uses ast.parse -- no regex.

    Example output:
        mathlib/operations.py
          def add(a, b)
          def multiply(a, b)
    """
    lines: list[str] = []

    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        try:
            tree = ast.parse(py_file.read_text(), filename=str(rel))
        except SyntaxError:
            lines.append(str(rel))
            lines.append("  [SyntaxError]")
            continue

        file_symbols: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Top-level function: extract argument names
                args = ", ".join(a.arg for a in node.args.args)
                file_symbols.append(f"  def {node.name}({args})")
            elif isinstance(node, ast.ClassDef):
                file_symbols.append(f"  class {node.name}")
                # Direct methods (one level deep)
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        args = ", ".join(a.arg for a in child.args.args)
                        file_symbols.append(f"    def {child.name}({args})")

        if file_symbols:
            lines.append(str(rel))
            lines.extend(file_symbols)

    return "\n".join(lines)


def run_solution_3() -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("SOLUTION 3 -- mini repo-map via ast")
    print(sep)

    with TemporaryDirectory(prefix="sol3_") as tmpdir:
        root = Path(tmpdir)
        create_toy_repo(root)

        # Also add a file with a syntax error to test graceful handling
        broken = root / "broken_module.py"
        broken.write_text("def oops(\n    # syntax error: unterminated\n")

        repo_map = build_repo_map(root)
        print("\n" + repo_map)

        # Count symbols: lines starting with "  def" or "  class" or "    def"
        symbol_lines = [ln for ln in repo_map.splitlines() if ln.strip().startswith(("def ", "class "))]
        symbol_count = len(symbol_lines)
        print(f"\nTotal symbols found: {symbol_count}")
        assert symbol_count >= 4, (
            f"Expected at least 4 symbols (add, multiply, test_add, test_multiply), "
            f"got {symbol_count}"
        )
        # Verify the broken file was handled gracefully
        assert "[SyntaxError]" in repo_map, "SyntaxError files must be noted, not skipped silently"

    print("\n[OK] Solution 3 passed all assertions.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    run_solution_1()
    run_solution_2()
    run_solution_3()

    print("\n" + "=" * 60)
    print("ALL SOLUTIONS PASSED")
    print("=" * 60)

"""
Solutions -- Day 21 (MEDIUM): Coding agents architecture (ACI & edit/run loop)

Contains solutions for:
  - Medium Ex 1: robust search/replace `edit` tool with a UNIQUENESS constraint
                 (reject a find block that is missing OR ambiguous)
  - Medium Ex 2: a unified-diff / patch applier over an in-memory file, with
                 CONFLICT detection (context lines must match) and hunk atomicity
  - Medium Ex 3: a plan-edit-test loop on a mock repo (dict filename -> source)
                 that edits a function, runs in-process tests, and rolls back
                 any edit that does not make progress

Self-contained: pure stdlib, no network, no API key. The mock test runner uses
`exec` on TRUSTED, hand-written source strings only (a pedagogical mock, not a
sandbox for untrusted code).

Run:  python 03-exercises/solutions/21-coding-agents-architecture-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ==========================================================================
# MEDIUM EXERCISE 1 -- robust search/replace edit with uniqueness constraint
# ==========================================================================

@dataclass
class EditBlock:
    """A structured edit: replace the `find` block with the `replace` block."""
    find: str
    replace: str


@dataclass
class EditResult:
    """Outcome of applying an edit. `new_source` is None when rejected."""
    ok: bool
    new_source: Optional[str]
    reason: str = ""


def apply_edit(source: str, block: EditBlock) -> EditResult:
    """
    Apply a single search/replace edit with a UNIQUENESS guarantee.

    - 0 matches  -> reject ("not found"): the snippet was likely hallucinated.
    - 2+ matches -> reject ("ambiguous: N matches"): refuse to guess WHICH one.
    - exactly 1  -> apply (the only safe case).
    """
    count = source.count(block.find)
    if count == 0:
        return EditResult(False, None, f"not found: {block.find!r}")
    if count > 1:
        return EditResult(False, None, f"ambiguous: {count} matches for {block.find!r}")
    new_source = source.replace(block.find, block.replace, 1)
    return EditResult(True, new_source, "")


def apply_edits(source: str, blocks: list[EditBlock]) -> EditResult:
    """
    Apply a SEQUENCE of edits, each on the result of the previous one.
    Stops at the first failing block and returns the partial source (for
    inspection) with ok=False.
    """
    current = source
    for i, block in enumerate(blocks):
        res = apply_edit(current, block)
        if not res.ok:
            return EditResult(False, current, f"block {i} failed: {res.reason}")
        current = res.new_source  # type: ignore[assignment]
    return EditResult(True, current, "")


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- robust search/replace edit with uniqueness constraint")
    print("=" * 70)

    src = "def add(a, b):\n    return a - b   # bug\n"

    # 1) Valid edit (exactly one match).
    ok = apply_edit(src, EditBlock("return a - b", "return a + b"))
    print(f"  valid edit -> ok={ok.ok}")
    assert ok.ok and ok.new_source == "def add(a, b):\n    return a + b   # bug\n"

    # 2) find absent -> rejected with "not found".
    miss = apply_edit(src, EditBlock("return a * b", "return a + b"))
    print(f"  missing find -> ok={miss.ok} reason={miss.reason!r}")
    assert not miss.ok and miss.new_source is None
    assert miss.reason.startswith("not found")

    # 3) find present twice -> rejected as "ambiguous: 2 matches".
    dup = "x = foo\ny = foo\n"
    amb = apply_edit(dup, EditBlock("foo", "bar"))
    print(f"  ambiguous find -> ok={amb.ok} reason={amb.reason!r}")
    assert not amb.ok and amb.new_source is None
    assert amb.reason.startswith("ambiguous") and "2 matches" in amb.reason

    # 4) Sequence of 2 valid edits chains correctly.
    multi = "a = 1\nb = 2\n"
    seq_ok = apply_edits(multi, [EditBlock("a = 1", "a = 10"), EditBlock("b = 2", "b = 20")])
    print(f"  valid sequence -> ok={seq_ok.ok}")
    assert seq_ok.ok and seq_ok.new_source == "a = 10\nb = 20\n"

    # 5) Sequence with an invalid 2nd block stops at the right place.
    seq_bad = apply_edits(multi, [EditBlock("a = 1", "a = 10"), EditBlock("zzz", "qqq")])
    print(f"  bad sequence -> ok={seq_bad.ok} reason={seq_bad.reason!r}")
    assert not seq_bad.ok
    assert "block 1 failed" in seq_bad.reason and "not found" in seq_bad.reason
    # The first (valid) edit IS reflected in the partial source returned for inspection.
    assert seq_bad.new_source == "a = 10\nb = 2\n"

    print("[Verification] PASS -- unique-only edits, rejection cases, sequencing")


# ==========================================================================
# MEDIUM EXERCISE 2 -- unified-diff applier with conflict detection
# ==========================================================================

@dataclass
class Hunk:
    """A parsed diff hunk: where it starts in the OLD file + its tagged lines."""
    old_start: int                       # 1-indexed line in the original source
    lines: list[tuple[str, str]] = field(default_factory=list)  # (tag, text)


def parse_hunks(diff: str) -> list[Hunk]:
    """
    Parse a simplified unified diff. We only need:
      - the @@ header's old_start
      - the tagged body lines: ' ' context, '-' delete, '+' add
    File headers (---, +++) are ignored.
    """
    hunks: list[Hunk] = []
    current: Optional[Hunk] = None
    for raw in diff.splitlines():
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        if raw.startswith("@@"):
            # Format: @@ -old_start,old_count +new_start,new_count @@
            # Extract old_start (the number right after the '-').
            try:
                minus = raw.split("-", 1)[1]
                old_start = int(minus.split(",", 1)[0].split(" ", 1)[0])
            except (IndexError, ValueError):
                old_start = 1
            current = Hunk(old_start=old_start)
            hunks.append(current)
        elif raw[:1] in (" ", "-", "+") and current is not None:
            current.lines.append((raw[0], raw[1:]))
    return hunks


def apply_patch(source: str, diff: str) -> tuple[bool, str, str]:
    """
    Apply a unified diff to *source*.

    Returns (ok, new_source, reason). Context (' ') and delete ('-') lines must
    match the real source EXACTLY at the hunk's position; otherwise it's a
    CONFLICT (the file drifted) and nothing is modified.
    """
    src_lines = source.splitlines()
    out: list[str] = []
    cursor = 0  # 0-indexed position in src_lines already copied to `out`

    for hunk in parse_hunks(diff):
        start = hunk.old_start - 1  # convert to 0-indexed
        if start < cursor or start > len(src_lines):
            return False, source, f"conflict at line {hunk.old_start}: out of range"
        # Copy untouched lines between the previous hunk and this one.
        out.extend(src_lines[cursor:start])
        cursor = start
        for tag, text in hunk.lines:
            if tag in (" ", "-"):
                # This line must match the real source line at the cursor.
                if cursor >= len(src_lines) or src_lines[cursor] != text:
                    real = src_lines[cursor] if cursor < len(src_lines) else "<EOF>"
                    return (False, source,
                            f"conflict at line {cursor + 1}: expected {text!r}, found {real!r}")
                if tag == " ":
                    out.append(text)  # context: keep it
                cursor += 1           # both ' ' and '-' consume an original line
            elif tag == "+":
                out.append(text)      # addition: no original line consumed

    out.extend(src_lines[cursor:])    # trailing untouched lines
    trailing_nl = "\n" if source.endswith("\n") else ""
    return True, "\n".join(out) + trailing_nl, ""


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- unified-diff applier with conflict detection")
    print("=" * 70)

    source = "def add(a, b):\n    return a - b\n"
    diff = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )

    hunks = parse_hunks(diff)
    print(f"  parsed {len(hunks)} hunk(s), first old_start={hunks[0].old_start}")
    assert len(hunks) == 1 and hunks[0].old_start == 1

    # 1) Clean apply.
    ok, new_src, reason = apply_patch(source, diff)
    print(f"  clean apply -> ok={ok}")
    assert ok and new_src == "def add(a, b):\n    return a + b\n", repr(new_src)

    # 2) Same patch on a DRIFTED source -> conflict (context no longer matches).
    drifted = "def add(a, b):\n    return a * b   # already changed elsewhere\n"
    ok2, new_src2, reason2 = apply_patch(drifted, diff)
    print(f"  drifted apply -> ok={ok2} reason={reason2!r}")
    assert not ok2 and "conflict" in reason2
    # 3) After a conflict, the source is untouched (hunk-level atomicity).
    assert new_src2 == drifted

    print("[Verification] PASS -- clean apply, conflict detected, source untouched")


# ==========================================================================
# MEDIUM EXERCISE 3 -- plan-edit-test loop on a mock repo
# ==========================================================================

def run_tests(repo: dict[str, str], tests: list[tuple[str, tuple, Any]]) -> dict:
    """
    In-process test runner over a mock repo (dict filename -> source).

    Each test is (fn_name, args, expected). Compiles ALL files into one fresh
    namespace, then calls each function and compares to expected.

    NOTE: `exec` is used on TRUSTED hand-written strings -- a pedagogical mock,
    not a sandbox for untrusted code.
    """
    ns: dict[str, Any] = {}
    try:
        for fname, src in repo.items():
            code = compile(src, fname, "exec")
            exec(code, ns)
    except (SyntaxError, NameError) as e:
        # A broken module: the edit damaged the code, not a logical test failure.
        return {"passed": 0, "failed": 0, "failures": [], "error": f"{type(e).__name__}: {e}"}

    passed, failed, failures = 0, 0, []
    for fn_name, args, expected in tests:
        try:
            fn = ns[fn_name]
            result = fn(*args)
            if result == expected:
                passed += 1
            else:
                failed += 1
                failures.append({"fn": fn_name, "args": args, "got": result, "expected": expected})
        except Exception as e:  # ZeroDivisionError, AssertionError, KeyError, ...
            failed += 1
            failures.append({"fn": fn_name, "args": args, "error": f"{type(e).__name__}: {e}"})
    return {"passed": passed, "failed": failed, "failures": failures, "error": None}


def _apply_to_repo(repo: dict[str, str], filename: str, find: str, replace: str) -> Optional[dict]:
    """Apply a unique-match edit to one file of a repo copy. None if rejected."""
    if filename not in repo:
        return None
    res = apply_edit(repo[filename], EditBlock(find, replace))
    if not res.ok:
        return None
    new_repo = dict(repo)
    new_repo[filename] = res.new_source  # type: ignore[assignment]
    return new_repo


def plan_edit_test_loop(
    repo: dict[str, str],
    tests: list[tuple[str, tuple, Any]],
    candidate_edits: list[tuple[str, str, str]],
    max_iters: int = 10,
) -> dict:
    """
    search -> edit -> run_tests -> iterate.

    Tries candidate edits in order. Keeps an edit ONLY if it strictly increases
    the number of passing tests; otherwise rolls it back. Stops when green or
    when candidates / max_iters are exhausted.
    """
    current = dict(repo)
    history: list[dict] = []
    res = run_tests(current, tests)
    iters = 0

    for (filename, find, replace) in candidate_edits:
        if iters >= max_iters:
            break
        if res["failed"] == 0 and res["error"] is None:
            break  # already green

        before = res["passed"]
        iters += 1
        candidate = _apply_to_repo(current, filename, find, replace)
        if candidate is None:
            history.append({"edit": (filename, find), "applied": False, "reason": "find absent"})
            continue

        new_res = run_tests(candidate, tests)
        # Progress = strictly more passing tests AND no compilation error.
        progressed = new_res["error"] is None and new_res["passed"] > before
        if progressed:
            current = candidate
            res = new_res
            history.append({"edit": (filename, find), "applied": True, "passed": new_res["passed"]})
        else:
            # Rollback: discard this candidate, keep `current` as-is.
            history.append({"edit": (filename, find), "applied": False,
                            "reason": "no progress", "passed_after": new_res["passed"]})

    green = res["failed"] == 0 and res["error"] is None
    return {"green": green, "iters": iters, "final_repo": current, "history": history}


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- plan-edit-test loop on a mock repo")
    print("=" * 70)

    repo = {"calc.py": "def add(a, b):\n    return a - b\n"}
    tests = [("add", (2, 3), 5), ("add", (0, 0), 0), ("add", (-1, 1), 0)]

    candidates = [
        # A useless edit first (a no-op rename that does NOT fix the bug).
        ("calc.py", "# nope", "# still nope"),       # find absent -> rejected
        ("calc.py", "return a - b", "return a * b"),  # wrong fix -> no progress -> rollback
        ("calc.py", "return a * b", "return a + b"),  # would only apply if previous stuck
        ("calc.py", "return a - b", "return a + b"),  # the real fix
    ]

    result = plan_edit_test_loop(repo, tests, candidates, max_iters=10)
    print(f"  green={result['green']} iters={result['iters']}")
    for h in result["history"]:
        print(f"    {h}")

    # Bug fixed.
    assert result["green"] is True
    # The wrong fix ("a * b") was rolled back: it must NOT be in the final repo.
    assert "return a * b" not in result["final_repo"]["calc.py"]
    assert "return a + b" in result["final_repo"]["calc.py"]

    # Unfixable bug: no candidate corrects it -> terminates green=False, no crash.
    repo2 = {"calc.py": "def div(a, b):\n    return a / b\n"}
    tests2 = [("div", (1, 0), 0)]  # ZeroDivisionError -> never passes
    candidates2 = [("calc.py", "return a / b", "return a // b")]  # still divides by zero
    result2 = plan_edit_test_loop(repo2, tests2, candidates2, max_iters=10)
    print(f"  unfixable -> green={result2['green']} iters={result2['iters']}")
    assert result2["green"] is False
    assert result2["iters"] <= 10  # bounded, no infinite loop

    print("[Verification] PASS -- progress-only edits, rollback, green + unfixable")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 21 MEDIUM Solutions -- Coding agents architecture")
    print("  (offline, stdlib only -- no network, no API key)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

"""Run every solution and code example in the repo and report failures.

Usage:
    python scripts/test_all_solutions.py [--domain <name>] [--timeout <seconds>]

Behavior:
- Runs each domains/*/03-exercises/solutions/*.py and domains/*/02-code/*.py
  as a standalone script (the repo convention).
- Forces matplotlib into headless mode (MPLBACKEND=Agg) so plt.show() never blocks CI.
- A script that exits because an optional dependency (torch, mujoco, gymnasium,
  langgraph...) is missing is reported as SKIP, not FAIL, since domains are
  designed with graceful-import fallbacks or documented optional deps.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OPTIONAL_DEPS = ("torch", "mujoco", "gymnasium", "langgraph", "langchain", "lerobot", "scipy", "matplotlib")


def classify_failure(stderr: str) -> str:
    # WHY: missing optional deps are an environment property, not a content bug.
    if "ModuleNotFoundError" in stderr and any(dep in stderr for dep in OPTIONAL_DEPS):
        return "SKIP"
    return "FAIL"


def run_one(script: Path, timeout: int) -> tuple[str, str]:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"exceeded {timeout}s"
    if proc.returncode == 0:
        return "PASS", ""
    status = classify_failure(proc.stderr)
    return status, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="restrict to one domain folder name")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    pattern = args.domain or "*"
    scripts = sorted(
        list(REPO_ROOT.glob(f"domains/{pattern}/03-exercises/solutions/*.py"))
        + list(REPO_ROOT.glob(f"domains/{pattern}/02-code/*.py"))
        + list(REPO_ROOT.glob(f"domains/{pattern}/05-projets-guides/*/solution/*.py"))
    )
    if not scripts:
        print("No scripts found.")
        return 1

    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0, "TIMEOUT": 0}
    failures: list[tuple[Path, str]] = []
    for script in scripts:
        status, detail = run_one(script, args.timeout)
        counts[status] += 1
        rel = script.relative_to(REPO_ROOT)
        print(f"[{status:7}] {rel}" + (f" — {detail}" if detail and status != "PASS" else ""))
        if status in ("FAIL", "TIMEOUT"):
            failures.append((rel, detail))

    print(f"\nTotal: {len(scripts)} | PASS {counts['PASS']} | SKIP {counts['SKIP']} "
          f"| FAIL {counts['FAIL']} | TIMEOUT {counts['TIMEOUT']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

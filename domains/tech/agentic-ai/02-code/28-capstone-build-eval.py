"""
Day 28 -- Capstone (build & eval): a durable, observable, self-evaluated deep ops agent.

This assembles the advanced patterns of the track into ONE runnable agent:

  - deep agent     : planner + todo/scratchpad on a VirtualFS (J15)
  - sub-agents     : 2-3 context-ISOLATED workers (research / code / verify) (J15/J9)
  - coding tool    : edit/search/run on a toy repo in a subprocess (J21/J23)
  - durability     : SQLite checkpoint + RESUME after a (simulated) crash (J20)
  - model routing  : mock cost-aware weak/strong routing (J24)
  - eval harness   : pass^k + regression report on the agent itself (J11/J26)

Everything is stdlib + deterministic mocks: it runs WITHOUT an API key and
WITHOUT network. The building blocks come from the Day 27 module.

Run:
    python domains/tech/agentic-ai/02-code/28-capstone-build-eval.py
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

# Reuse the Day 27 building blocks (import by numeric module name).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
day27 = import_module("27-capstone-architecture")
VirtualFS = day27.VirtualFS
SQLiteCheckpointer = day27.SQLiteCheckpointer
DurableEngine = day27.DurableEngine
Step = day27.Step
CrashSignal = day27.CrashSignal
ModelRouter = day27.ModelRouter
SubAgent = day27.SubAgent


# ===========================================================================
# 1. SUB-AGENTS (context-isolated)
# ===========================================================================

class ResearchSubAgent(SubAgent):
    role = "research"

    def run(self, task: str) -> str:
        self.router.route(task)
        self.observe(f"researching: {task}")            # private context
        facts = "bug is in calc.add (uses '-' instead of '+')"
        self.fs.write("research.md", facts)              # offload to scratchpad
        return facts


class CoderSubAgent(SubAgent):
    """Fixes a planted bug in a toy repo using an edit/search/run loop."""
    role = "code"

    def __init__(self, fs, router, will_fix: bool = True) -> None:
        super().__init__(fs, router)
        self.will_fix = will_fix
        self.trajectory: list[str] = []

    def _make_toy_repo(self) -> Path:
        repo = Path(self.fs.root) / "repo"
        repo.mkdir(exist_ok=True)
        (repo / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG\n",
                                      encoding="utf-8")
        (repo / "test_calc.py").write_text(
            "from calc import add\n"
            "assert add(2, 3) == 5, 'add is broken'\n"
            "print('TESTS OK')\n", encoding="utf-8")
        return repo

    def _run_tests(self, repo: Path) -> tuple[bool, str]:
        # Clear cached bytecode: an edit in the same second as the previous run
        # can leave a stale .pyc that Python would reuse, masking the fix.
        cache = repo / "__pycache__"
        if cache.is_dir():
            for pyc in cache.glob("*.pyc"):
                pyc.unlink()
        proc = subprocess.run(
            [sys.executable, "-B", "test_calc.py"], cwd=str(repo),
            capture_output=True, text=True, timeout=15)
        self.trajectory.append("run_tests")
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()

    def run(self, task: str) -> str:
        self.router.route(task)
        repo = self._make_toy_repo()

        ok, _ = self._run_tests(repo)               # red
        if ok:
            return "tests already green"

        # search for the buggy line
        self.trajectory.append("search")
        src = (repo / "calc.py").read_text(encoding="utf-8")

        if self.will_fix and "a - b" in src:
            self.trajectory.append("edit")
            (repo / "calc.py").write_text(src.replace("a - b", "a + b"),
                                          encoding="utf-8")

        ok, out = self._run_tests(repo)             # green (or still red)
        return "fixed" if ok else f"FAILED: {out}"


class VerifierSubAgent(SubAgent):
    role = "verify"

    def run(self, task: str) -> str:
        self.router.route(task)
        # Reads the coder's reported result from context (passed as task).
        return "verified-ok" if "fixed" in task or "green" in task else "verify-failed"


# ===========================================================================
# 2. DEEP OPS AGENT
# ===========================================================================

@dataclass
class AgentResult:
    output: str
    trajectory: list[str] = field(default_factory=list)
    steps: int = 0


class DeepOpsAgent:
    """Planner that orchestrates isolated sub-agents to fix a bug, with an
    optional stochastic failure (error_rate) so reliability (pass^k) is
    measurable. The same step sequence can run under the DurableEngine to
    demonstrate crash-resume.
    """

    def __init__(self, label: str = "deep-ops", error_rate: float = 0.0,
                 seed: int = 0) -> None:
        self.label = label
        self.error_rate = max(0.0, min(1.0, error_rate))
        self._rng = random.Random(seed)

    # ---- planning ----
    def _plan(self, fs: VirtualFS, task: str) -> list[str]:
        todos = ["research the bug", "fix the code", "verify the fix"]
        fs.write("todo.md", "\n".join(f"- [ ] {t}" for t in todos))
        return todos

    # ---- one full solve (used by the eval harness) ----
    def solve(self, task: str) -> AgentResult:
        fs = VirtualFS()
        router = ModelRouter()
        traj: list[str] = ["plan"]
        self._plan(fs, task)

        # The agent occasionally "loses focus" and the coder gives up (no fix).
        will_fix = self._rng.random() >= self.error_rate

        research = ResearchSubAgent(fs, router)
        traj.append("research")
        research.run(task)

        coder = CoderSubAgent(fs, router, will_fix=will_fix)
        traj.append("code")
        code_result = coder.run(task)
        traj.extend(coder.trajectory)             # search/edit/run_tests

        verifier = VerifierSubAgent(fs, router)
        traj.append("verify")
        verdict = verifier.run(code_result)

        fs.write("report.md", f"task={task}\nresult={code_result}\nverdict={verdict}")
        output = "SUCCESS: bug fixed and verified" if verdict == "verified-ok" \
            else f"FAILURE: {code_result}"
        return AgentResult(output=output, trajectory=traj, steps=len(traj))

    # ---- durable run (used by the crash-resume demo) ----
    def run_durable(self, run_id: str, cp: SQLiteCheckpointer,
                    crash_before: str | None = None) -> dict:
        fs = VirtualFS()
        router = ModelRouter()
        coder = CoderSubAgent(fs, router, will_fix=True)

        steps = [
            Step("plan", lambda ctx: self._plan(fs, "fix calc.add")),
            Step("research", lambda ctx: ResearchSubAgent(fs, router).run("bug?")),
            Step("code", lambda ctx: coder.run("fix calc.add")),
            Step("verify", lambda ctx: VerifierSubAgent(fs, router).run(ctx["code"])),
        ]
        engine = DurableEngine(cp)
        ctx = engine.run(run_id, steps, crash_before=crash_before)
        ctx["__executed__"] = engine.executed
        ctx["__skipped__"] = engine.skipped
        return ctx


# ===========================================================================
# 3. EVAL HARNESS (pass^k + regression)
# ===========================================================================

@dataclass
class EvalCase:
    id: str
    task: str
    expected: str                 # substring expected in output
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
    tags: list[str]

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
        reports.append(CaseReport(case.id, successes, k, case.tags))
    mean_pass_k = sum(r.pass_k for r in reports) / len(reports) if reports else 0.0
    return {"label": agent.label, "k": k, "reports": reports, "mean_pass_k": mean_pass_k}


def regression_report(baseline: dict, candidate: dict) -> str:
    print("\n" + "=" * 64)
    print(f"REGRESSION REPORT (pass^{baseline['k']})  "
          f"{baseline['label']} -> {candidate['label']}")
    print("=" * 64)
    b = {r.case_id: r for r in baseline["reports"]}
    c = {r.case_id: r for r in candidate["reports"]}
    blocking = False
    print(f"{'case':<16}{'base p^k':>10}{'cand p^k':>10}{'delta':>9}  flags")
    print("-" * 60)
    for cid in sorted(b):
        if cid not in c:
            continue  # case absent from the candidate suite -> skip, don't KeyError
        delta = c[cid].pass_k - b[cid].pass_k
        flag = ""
        if "golden" in b[cid].tags and delta <= -0.10:
            flag = "GOLDEN REGRESSION -- BLOCK"
            blocking = True
        print(f"{cid:<16}{b[cid].pass_k:>10.3f}{c[cid].pass_k:>10.3f}"
              f"{delta:>+9.3f}  {flag}")
    print("-" * 60)
    print(f"mean pass^k : {baseline['mean_pass_k']:.3f} -> {candidate['mean_pass_k']:.3f}")
    if blocking:
        verdict = "BLOCKED -- golden regression"
    elif candidate["mean_pass_k"] > baseline["mean_pass_k"]:
        verdict = "APPROVED -- candidate is better"
    else:
        verdict = "NEUTRAL"
    print(f"VERDICT     : {verdict}")
    return verdict


DEMO_CASES: list[EvalCase] = [
    EvalCase("fix-add", "fix the add bug in calc",
             expected="SUCCESS",
             required_steps=["plan", "research", "code", "edit", "run_tests", "verify"],
             max_steps=12, tags=["golden"]),
    EvalCase("fix-add-2", "repair calc.add so tests pass",
             expected="SUCCESS",
             required_steps=["code", "run_tests", "verify"],
             max_steps=12, tags=["golden"]),
    EvalCase("budget", "fix the bug within budget",
             expected="SUCCESS", required_steps=["plan"], max_steps=12),
]


# ===========================================================================
# 4. MAIN DEMO
# ===========================================================================

def _demo_full_run() -> None:
    print("=" * 64)
    print("PART A -- one full agent run")
    print("=" * 64)
    agent = DeepOpsAgent(error_rate=0.0, seed=1)
    res = agent.solve("fix the add bug in calc")
    print(f"output     : {res.output}")
    print(f"trajectory : {res.trajectory}")
    print(f"steps      : {res.steps}")
    assert "SUCCESS" in res.output


def _demo_crash_resume() -> None:
    print("\n" + "=" * 64)
    print("PART B -- durable crash + resume")
    print("=" * 64)
    fs = VirtualFS()
    db = str(Path(fs.root) / "capstone.db")
    run_id = "ops-run-1"
    agent = DeepOpsAgent(seed=2)

    cp = SQLiteCheckpointer(db)
    try:
        agent.run_durable(run_id, cp, crash_before="verify")
    except CrashSignal as exc:
        print(f"  CRASH: {exc}")
    cp.close()  # process dies

    cp2 = SQLiteCheckpointer(db)            # new process, same durable file
    ctx = agent.run_durable(run_id, cp2)
    print(f"  resumed (skipped) : {ctx['__skipped__']}")
    print(f"  executed (new)    : {ctx['__executed__']}")
    print(f"  verify result     : {ctx['verify']!r}")
    assert ctx["__skipped__"] == ["plan", "research", "code"]
    assert ctx["__executed__"] == ["verify"]
    cp2.close()


def _demo_eval() -> None:
    print("\n" + "=" * 64)
    print("PART C -- self-evaluation (pass^k + regression)")
    print("=" * 64)
    K = 5
    baseline = run_suite(DeepOpsAgent(label="v1.0", error_rate=0.40, seed=10), DEMO_CASES, k=K)
    candidate = run_suite(DeepOpsAgent(label="v1.1", error_rate=0.05, seed=10), DEMO_CASES, k=K)

    print(f"\n{'case':<16}{'base p^k':>10}{'cand p^k':>10}")
    print("-" * 36)
    for rb, rc in zip(baseline["reports"], candidate["reports"]):
        print(f"{rb.case_id:<16}{rb.pass_k:>10.3f}{rc.pass_k:>10.3f}")
    verdict = regression_report(baseline, candidate)
    assert candidate["mean_pass_k"] >= baseline["mean_pass_k"]
    assert verdict.startswith("APPROVED")


if __name__ == "__main__":
    _demo_full_run()
    _demo_crash_resume()
    _demo_eval()
    print("\nCapstone complete: deep agent + durability + routing + self-eval, "
          "all local, no API key.")

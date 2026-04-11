"""
Day 9 -- Multi-agent patterns: supervisor and swarm, side by side.

Demonstrates:
  1. MockLLM with role-specific skills (researcher, coder, reviewer, supervisor)
  2. SupervisorPattern  -- central supervisor delegates to 3 specialists
  3. SwarmPattern       -- specialists hand off to each other without a chief
  4. A direct comparison on the same task so you can see the tradeoffs

Dependencies: stdlib only. Optional: langchain / langgraph if installed,
but the MockLLM fallback guarantees the demo runs offline.

Run:
    python domains/agentic-ai/02-code/09-multi-agent-patterns.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Optional bindings -- fall back to MockLLM if not available
# ---------------------------------------------------------------------------

HAS_LANGGRAPH = False
try:
    import langgraph  # noqa: F401
    HAS_LANGGRAPH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1. MOCK LLM WITH ROLE-SPECIFIC SKILLS
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic LLM stub that behaves differently based on the "role" asked.
    Each role uses a small hardcoded knowledge base / template so the demo
    produces realistic-looking output without any API key.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.call_log: list[tuple[str, str]] = []

    def __call__(self, role: str, task: str, context: str = "") -> str:
        self.call_count += 1
        self.call_log.append((role, task))
        if role == "supervisor":
            return self._supervisor(task, context)
        if role == "researcher":
            return self._researcher(task, context)
        if role == "coder":
            return self._coder(task, context)
        if role == "reviewer":
            return self._reviewer(task, context)
        if role == "triage":
            return self._triage(task, context)
        raise ValueError(f"Unknown role: {role}")

    # --- role implementations --------------------------------------------

    def _supervisor(self, task: str, context: str) -> str:
        """
        Two jobs:
          - plan:      decompose the task into role-specific steps
          - synthesize: merge the worker outputs into a final answer
        We detect the job from the task prefix.
        """
        if task.startswith("plan:"):
            # Return a deterministic plan as JSON for easy parsing
            plan = [
                {"role": "researcher", "instruction": "Find existing sorting algorithms in Python stdlib"},
                {"role": "coder", "instruction": "Implement a toy quicksort in Python"},
                {"role": "reviewer", "instruction": "Review the code for bugs and style"},
            ]
            return json.dumps(plan)
        if task.startswith("synthesize:"):
            return (
                "Final report (synthesized by supervisor):\n"
                f"{context}\n"
                "Conclusion: the task was completed by coordinating researcher, "
                "coder and reviewer. The code is production-ready after review."
            )
        return "supervisor: unknown task"

    def _researcher(self, task: str, context: str) -> str:
        return (
            "[Researcher output]\n"
            "Python's stdlib provides sorted() and list.sort(), both based on Timsort. "
            "Timsort is stable, O(n log n) worst case. A toy quicksort is useful for "
            "teaching but not for production."
        )

    def _coder(self, task: str, context: str) -> str:
        return (
            "[Coder output]\n"
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        )

    def _reviewer(self, task: str, context: str) -> str:
        """Very simple heuristic review based on keywords in the context."""
        issues: list[str] = []
        if "def " in context:
            issues.append("OK: function definition present")
        if "return" not in context:
            issues.append("BUG: no return statement")
        if "pivot = arr[len(arr) // 2]" in context:
            issues.append("NOTE: middle-element pivot is fine for a toy impl")
        if not issues:
            issues.append("NOTE: nothing caught -- looks acceptable")
        return "[Reviewer output]\n" + "\n".join(f"- {i}" for i in issues)

    def _triage(self, task: str, context: str) -> str:
        """
        Triage agent used by the swarm entry point. It decides which
        specialist should handle the request first. Returns the role name.
        """
        low = task.lower()
        if any(w in low for w in ["algorithm", "stdlib", "find", "research"]):
            return "researcher"
        if any(w in low for w in ["code", "implement", "python", "write"]):
            return "coder"
        if any(w in low for w in ["review", "bug", "quality"]):
            return "reviewer"
        return "researcher"


# ===========================================================================
# 2. SUPERVISOR PATTERN
# ===========================================================================

@dataclass
class SupervisorRun:
    """Observability record for one supervisor run."""
    task: str
    plan: list[dict] = field(default_factory=list)
    worker_outputs: dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    llm_calls: int = 0


class SupervisorPattern:
    """
    Central supervisor decomposes the task, delegates to named workers,
    collects results, synthesizes a final answer.
    """

    def __init__(self, llm: MockLLM, workers: dict[str, str]) -> None:
        self.llm = llm
        # workers: mapping from role name to description (for logging).
        # In a real system, each worker would be its own agent with its own prompt.
        self.workers = workers

    def run(self, task: str, verbose: bool = True) -> SupervisorRun:
        run = SupervisorRun(task=task)
        before = self.llm.call_count

        # 1. Plan
        plan_json = self.llm("supervisor", f"plan: {task}")
        run.plan = json.loads(plan_json)
        if verbose:
            print(f"[Supervisor] plan: {[s['role'] for s in run.plan]}")

        # 2. Delegate each step to the appropriate worker
        collected_context = ""
        for step in run.plan:
            role = step["role"]
            instruction = step["instruction"]
            if verbose:
                print(f"[Supervisor] -> {role}: {instruction}")
            output = self.llm(role, instruction, context=collected_context)
            run.worker_outputs[role] = output
            collected_context += "\n" + output

        # 3. Synthesize final answer
        run.final_answer = self.llm(
            "supervisor", f"synthesize: {task}", context=collected_context
        )
        run.llm_calls = self.llm.call_count - before
        return run


# ===========================================================================
# 3. SWARM PATTERN
# ===========================================================================

@dataclass
class SwarmRun:
    """Observability record for one swarm run."""
    task: str
    trajectory: list[tuple[str, str]] = field(default_factory=list)
    final_answer: str = ""
    llm_calls: int = 0


class SwarmPattern:
    """
    Flat network of specialists. Each agent decides whether to answer
    or to hand off to another agent. No central supervisor.

    A handoff table is consulted to avoid infinite loops: once an agent
    has spoken, it cannot speak again in the same run unless explicitly
    re-invited.
    """

    def __init__(self, llm: MockLLM, agents: list[str], max_hops: int = 5) -> None:
        self.llm = llm
        self.agents = agents
        self.max_hops = max_hops

    def run(self, task: str, verbose: bool = True) -> SwarmRun:
        run = SwarmRun(task=task)
        before = self.llm.call_count

        # 1. Triage: decide who speaks first
        first = self.llm("triage", task)
        if verbose:
            print(f"[Swarm] triage -> {first}")
        current = first
        visited: set[str] = set()
        collected: list[str] = []

        for hop in range(self.max_hops):
            if current in visited:
                if verbose:
                    print(f"[Swarm] cycle detected on {current}, stopping")
                break
            visited.add(current)
            output = self.llm(current, task, context="\n".join(collected))
            collected.append(output)
            run.trajectory.append((current, output[:60] + "..."))
            if verbose:
                print(f"[Swarm] {current}: {output.splitlines()[0][:70]}")

            # Decide next agent via a simple handoff rule.
            # In a real swarm, the LLM itself would call a transfer_to(...) tool.
            next_agent = self._handoff(current, visited)
            if next_agent is None:
                break
            if verbose:
                print(f"[Swarm] {current} -> handoff {next_agent}")
            current = next_agent

        run.final_answer = "\n".join(collected)
        run.llm_calls = self.llm.call_count - before
        return run

    def _handoff(self, current: str, visited: set[str]) -> str | None:
        """
        Deterministic handoff rule:
          researcher -> coder
          coder      -> reviewer
          reviewer   -> None (final)
        Skip agents already visited to avoid loops.
        """
        chain = {"researcher": "coder", "coder": "reviewer", "reviewer": None}
        nxt = chain.get(current)
        if nxt is None or nxt in visited:
            return None
        return nxt


# ===========================================================================
# 4. COMPARISON DEMO
# ===========================================================================

def demo() -> None:
    task = "Research sorting algorithms, implement a toy quicksort, and review the code."

    print("=" * 70)
    print(f"Backends available: langgraph={HAS_LANGGRAPH} -- using MockLLM")
    print("=" * 70)

    llm = MockLLM()
    print("\n--- SUPERVISOR PATTERN ---\n")
    sup = SupervisorPattern(
        llm,
        workers={
            "researcher": "Finds facts",
            "coder": "Writes code",
            "reviewer": "Reviews code",
        },
    )
    sup_run = sup.run(task)
    print("\n=== Final answer (supervisor) ===")
    print(sup_run.final_answer)
    print(f"LLM calls: {sup_run.llm_calls}")

    # Reset LLM state for a fair comparison
    llm.call_count = 0
    llm.call_log.clear()

    print("\n--- SWARM PATTERN ---\n")
    swarm = SwarmPattern(llm, agents=["researcher", "coder", "reviewer"], max_hops=5)
    swarm_run = swarm.run(task)
    print("\n=== Final answer (swarm) ===")
    print(swarm_run.final_answer)
    print(f"LLM calls: {swarm_run.llm_calls}")

    print("\n--- TRADE-OFF SUMMARY ---")
    print(
        f"Supervisor: {sup_run.llm_calls} calls, centralized control, "
        f"clean synthesis ({len(sup_run.final_answer)} chars)"
    )
    print(
        f"Swarm:      {swarm_run.llm_calls} calls, flat handoffs, "
        f"no synthesis ({len(swarm_run.final_answer)} chars)"
    )
    print(
        "\nTake-away: supervisor is better when you need a coherent synthesized "
        "output. Swarm is better when the right agent is not known upfront and "
        "you trust each specialist to know when to hand off."
    )


if __name__ == "__main__":
    demo()

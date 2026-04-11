"""
Day 9 -- Solutions to the easy exercises for multi-agent patterns.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/09-multi-agent-patterns.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day9 = import_module("09-multi-agent-patterns")
MockLLM = day9.MockLLM
SupervisorPattern = day9.SupervisorPattern
SwarmPattern = day9.SwarmPattern


# ===========================================================================
# SOLUTION 1 -- Add a 4th specialist (documenter)
# ===========================================================================

class DocMockLLM(MockLLM):
    """Extend MockLLM with a documenter role and update the supervisor plan."""

    def __call__(self, role: str, task: str, context: str = "") -> str:  # type: ignore[override]
        if role == "documenter":
            self.call_count += 1
            self.call_log.append((role, task))
            return self._documenter(task, context)
        if role == "supervisor" and task.startswith("plan:"):
            self.call_count += 1
            self.call_log.append((role, task))
            plan = [
                {"role": "researcher", "instruction": "Find sorting algorithms in Python stdlib"},
                {"role": "coder", "instruction": "Implement a toy quicksort in Python"},
                {"role": "reviewer", "instruction": "Review the code for bugs and style"},
                {"role": "documenter", "instruction": "Write a docstring for the code"},
            ]
            return json.dumps(plan)
        return super().__call__(role, task, context)

    def _documenter(self, task: str, context: str) -> str:
        return (
            "[Documenter output]\n"
            '"""\n'
            "Toy quicksort.\n"
            "\n"
            ":param arr: list of comparable elements to sort\n"
            ":type arr: list\n"
            ":returns: a new list sorted in ascending order\n"
            ":rtype: list\n"
            '"""'
        )


class DocSwarmPattern(SwarmPattern):
    """Swarm variant with documenter in the handoff chain."""

    def _handoff(self, current: str, visited):  # type: ignore[override]
        chain = {
            "researcher": "coder",
            "coder": "reviewer",
            "reviewer": "documenter",
            "documenter": None,
        }
        nxt = chain.get(current)
        if nxt is None or nxt in visited:
            return None
        return nxt


def solution_1() -> None:
    print("\n=== Solution 1: add a 4th specialist (documenter) ===")
    llm = DocMockLLM()
    task = "Research sorting, implement quicksort, review it, then document it."

    sup = SupervisorPattern(
        llm,
        workers={
            "researcher": "Facts",
            "coder": "Code",
            "reviewer": "Review",
            "documenter": "Doc",
        },
    )
    run = sup.run(task, verbose=False)
    print("Supervisor plan roles:", [s["role"] for s in run.plan])
    print("Supervisor outputs collected for:", list(run.worker_outputs.keys()))
    assert "documenter" in run.worker_outputs

    llm.call_count = 0
    swarm = DocSwarmPattern(
        llm, agents=["researcher", "coder", "reviewer", "documenter"], max_hops=6
    )
    srun = swarm.run(task, verbose=False)
    print("Swarm trajectory agents:", [t[0] for t in srun.trajectory])
    assert any(t[0] == "documenter" for t in srun.trajectory)


# ===========================================================================
# SOLUTION 2 -- Budget and tight-loop protection in the swarm
# ===========================================================================

class BudgetedSwarm(SwarmPattern):
    """Swarm with a hard call budget and tight-loop detection."""

    def __init__(self, llm, agents, max_hops: int = 5, max_llm_calls: int = 10) -> None:
        super().__init__(llm, agents, max_hops)
        self.max_llm_calls = max_llm_calls

    def run(self, task: str, verbose: bool = True):  # type: ignore[override]
        from importlib import import_module as _imp
        _day9 = _imp("09-multi-agent-patterns")
        run = _day9.SwarmRun(task=task)
        before = self.llm.call_count
        local_calls = 0

        first = self.llm("triage", task)
        local_calls += 1
        current = first
        previous = None
        visited: set[str] = set()
        collected: list[str] = []

        for hop in range(self.max_hops):
            if current == previous:
                raise RuntimeError("Detected tight loop")
            if local_calls >= self.max_llm_calls:
                run.trajectory.append(("system", "[Swarm] budget reached"))
                if verbose:
                    print("[Swarm] budget reached")
                break
            if current in visited:
                break
            visited.add(current)
            output = self.llm(current, task, context="\n".join(collected))
            local_calls += 1
            collected.append(output)
            run.trajectory.append((current, output[:60] + "..."))
            previous = current
            nxt = self._handoff(current, visited)
            if nxt is None:
                break
            current = nxt

        run.final_answer = "\n".join(collected)
        run.llm_calls = self.llm.call_count - before
        return run


def solution_2() -> None:
    print("\n=== Solution 2: budget + tight-loop protection ===")
    llm = MockLLM()

    swarm_normal = BudgetedSwarm(
        llm, agents=["researcher", "coder", "reviewer"], max_llm_calls=10
    )
    run_normal = swarm_normal.run("task that should fit in budget", verbose=False)
    print(f"  normal: {run_normal.llm_calls} LLM calls, "
          f"trajectory={[t[0] for t in run_normal.trajectory]}")

    llm.call_count = 0
    swarm_tight = BudgetedSwarm(
        llm, agents=["researcher", "coder", "reviewer"], max_llm_calls=2
    )
    run_tight = swarm_tight.run("task with tight budget", verbose=False)
    final_agents = [t[0] for t in run_tight.trajectory]
    print(f"  tight:  trajectory={final_agents}")
    assert "system" in final_agents, "expected budget message"


# ===========================================================================
# SOLUTION 3 -- Debate pattern with scoring
# ===========================================================================

class DebateMockLLM(MockLLM):
    """Extend MockLLM with deterministic scores per agent."""

    SCORES = {"researcher": 7, "coder": 8, "reviewer": 5}

    def __call__(self, role: str, task: str, context: str = "") -> str:  # type: ignore[override]
        if role in self.SCORES and task.startswith("score:"):
            self.call_count += 1
            self.call_log.append((role, task))
            score = self.SCORES[role]
            justification = {
                "researcher": "Solid claim, facts check out, minor concerns on sources",
                "coder": "Technically sound and easy to implement",
                "reviewer": "Worries about edge cases and maintainability",
            }[role]
            return json.dumps({"score": score, "justification": justification})
        if role == "moderator" and task.startswith("decide:"):
            self.call_count += 1
            self.call_log.append((role, task))
            # The context contains the JSON of scores, we parse it
            scores_obj = json.loads(context)
            avg = sum(s["score"] for s in scores_obj) / max(1, len(scores_obj))
            if avg >= 6:
                verdict = "accept"
            elif avg < 4:
                verdict = "reject"
            else:
                verdict = "debate"
            return json.dumps({"verdict": verdict, "average": avg})
        return super().__call__(role, task, context)


@dataclass
class DebatePattern:
    llm: DebateMockLLM
    agents: list[str]
    moderator: str = "moderator"

    def decide(self, proposal: str) -> dict:
        scores = []
        for agent in self.agents:
            raw = self.llm(agent, f"score: {proposal}")
            scores.append({**json.loads(raw), "agent": agent})
        mod_raw = self.llm(self.moderator, f"decide: {proposal}", context=json.dumps(scores))
        mod = json.loads(mod_raw)
        return {
            "proposal": proposal,
            "scores": scores,
            "average": mod["average"],
            "verdict": mod["verdict"],
        }


def solution_3() -> None:
    print("\n=== Solution 3: debate with scoring ===")
    llm = DebateMockLLM()
    debate = DebatePattern(llm=llm, agents=["researcher", "coder", "reviewer"])
    proposals = [
        "Adopt TypeScript for the frontend",
        "Rewrite the backend in Rust next quarter",
        "Rename the project from Kalira to K.I.R.A.",
    ]
    for p in proposals:
        result = debate.decide(p)
        print(f"\nProposal: {p}")
        for s in result["scores"]:
            print(f"  {s['agent']:10} {s['score']}/10 -- {s['justification']}")
        print(f"  average={result['average']:.1f}  verdict={result['verdict']}")


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

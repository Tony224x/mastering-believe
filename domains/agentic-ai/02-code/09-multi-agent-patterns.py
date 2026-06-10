"""
Day 9 -- Multi-agent patterns: supervisor, swarm, hierarchical, debate.

Demonstrates:
  1. MockLLM with role-specific skills (researcher, coder, reviewer, supervisor,
     CEO/managers for hierarchy, debaters + judge for debate)
  2. SupervisorPattern    -- central supervisor delegates to 3 specialists
  3. SwarmPattern         -- specialists hand off to each other without a chief
  4. HierarchicalPattern  -- supervisor of supervisors: CEO -> 2 team managers -> workers
  5. DebatePattern        -- 2 agents argue over 2 rounds, a judge settles it
  6. A direct comparison on the same task so you can see the tradeoffs

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
        if role == "ceo":
            return self._ceo(task, context)
        if role in ("research_manager", "eng_manager"):
            return self._manager(role, task, context)
        if role in ("optimist", "skeptic"):
            return self._debater(role, task, context)
        if role == "judge":
            return self._judge(task, context)
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

    # --- hierarchical roles (CEO + team managers) --------------------------

    def _ceo(self, task: str, context: str) -> str:
        """
        Top-level supervisor. Like the regular supervisor it plans and
        synthesizes -- but it delegates to TEAMS (via their managers), never
        directly to workers. That indirection is the whole hierarchical idea.
        """
        if task.startswith("plan:"):
            # The CEO thinks in team-level objectives, not worker steps
            plan = [
                {"team": "research", "objective": "Survey existing sorting solutions and their tradeoffs"},
                {"team": "engineering", "objective": "Implement a toy quicksort and get it reviewed"},
            ]
            return json.dumps(plan)
        if task.startswith("synthesize:"):
            return (
                "Final report (synthesized by CEO):\n"
                f"{context}\n"
                "Conclusion: research and engineering each delivered through their "
                "own manager; the CEO only saw 2 condensed team reports, not the "
                "raw worker output."
            )
        return "ceo: unknown task"

    def _manager(self, role: str, task: str, context: str) -> str:
        """
        Mid-level supervisor: plans worker steps for its OWN team, then
        condenses the workers' raw output into a short report for the CEO.
        Compressing upward is what keeps hierarchy from exploding context.
        """
        if task.startswith("plan:"):
            if role == "research_manager":
                plan = [{"role": "researcher", "instruction": "Find existing sorting algorithms in Python stdlib"}]
            else:  # eng_manager owns both build and QA
                plan = [
                    {"role": "coder", "instruction": "Implement a toy quicksort in Python"},
                    {"role": "reviewer", "instruction": "Review the code for bugs and style"},
                ]
            return json.dumps(plan)
        if task.startswith("report:"):
            # Deliberately short: the manager DOES NOT forward raw worker
            # output -- it summarizes (anti context-explosion defense).
            if role == "research_manager":
                return (
                    "[Research team report] stdlib sorted()/list.sort() use Timsort "
                    "(stable, O(n log n)); a custom quicksort is pedagogical only. "
                    f"(condensed from {len(context)} chars of worker output)"
                )
            return (
                "[Engineering team report] Toy quicksort implemented; review found "
                "no blocking bugs (middle-element pivot acceptable for a toy). "
                f"(condensed from {len(context)} chars of worker output)"
            )
        return f"{role}: unknown task"

    # --- debate roles (2 debaters + judge) ----------------------------------

    def _debater(self, role: str, task: str, context: str) -> str:
        """
        Each debater takes a fixed stance. Round detection: if the context
        contains the opponent's argument ('argued:'), this is a revision
        round -- the debater updates its position instead of repeating it.
        """
        revising = "argued:" in context
        if role == "optimist":
            if not revising:
                return (
                    "[optimist] Ship the toy quicksort: it is correct, readable, "
                    "and fast enough for our small demo datasets."
                )
            return (
                "[optimist] REVISED: I concede the O(n^2) worst case and the "
                "maintenance cost. Compromise: keep quicksort in the teaching "
                "module only, and call sorted() on production paths."
            )
        # skeptic
        if not revising:
            return (
                "[skeptic] Do NOT ship it: stdlib sorted() is battle-tested "
                "Timsort, stable, and faster; a custom quicksort adds an O(n^2) "
                "worst case and maintenance risk for zero benefit."
            )
        return (
            "[skeptic] REVISED: still against production use, but I accept the "
            "optimist's compromise -- the toy quicksort is fine as documented "
            "teaching material."
        )

    def _judge(self, task: str, context: str) -> str:
        """
        Moderator: reads the full debate transcript and either certifies the
        consensus or breaks the tie. Here both debaters converged in round 2,
        so the judge certifies the compromise.
        """
        consensus = "REVISED" in context and "teaching" in context
        if consensus:
            return (
                "[judge] VERDICT: consensus certified. Use sorted() (Timsort) on "
                "all production code paths; keep the toy quicksort only in the "
                "teaching module. Both debaters converged on this split in round 2."
            )
        return "[judge] VERDICT: no consensus -- tie broken in favor of stdlib sorted()."


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
# 4. HIERARCHICAL PATTERN (supervisor of supervisors)
# ===========================================================================

@dataclass
class HierarchicalRun:
    """Observability record for one hierarchical run."""
    task: str
    ceo_plan: list[dict] = field(default_factory=list)
    team_reports: dict[str, str] = field(default_factory=dict)
    final_answer: str = ""
    llm_calls: int = 0


class HierarchicalPattern:
    """
    Two-level supervision: CEO -> team managers -> workers.

    The CEO never talks to a worker directly. Each manager runs a mini
    supervisor loop inside its team, then sends a CONDENSED report up.
    This trades extra LLM calls and latency for scalability: the CEO's
    context stays small no matter how many workers each team has.
    """

    def __init__(self, llm: MockLLM, teams: dict[str, dict]) -> None:
        self.llm = llm
        # teams: {"research": {"manager": "research_manager", "workers": [...]}, ...}
        # Workers are listed for documentation; the manager's plan decides
        # which of them actually run (same contract as SupervisorPattern).
        self.teams = teams

    def run(self, task: str, verbose: bool = True) -> HierarchicalRun:
        run = HierarchicalRun(task=task)
        before = self.llm.call_count

        # 1. CEO plans at TEAM granularity (objectives, not worker steps)
        run.ceo_plan = json.loads(self.llm("ceo", f"plan: {task}"))
        if verbose:
            print(f"[CEO] plan: {[(a['team'], a['objective'][:40] + '...') for a in run.ceo_plan]}")

        # 2. Each team runs its own internal supervisor loop
        for assignment in run.ceo_plan:
            team, objective = assignment["team"], assignment["objective"]
            manager = self.teams[team]["manager"]
            if verbose:
                print(f"[CEO] -> {manager}: {objective}")

            # 2a. The manager decomposes its objective into worker steps
            worker_plan = json.loads(self.llm(manager, f"plan: {objective}"))
            team_context = ""
            for step in worker_plan:
                if verbose:
                    print(f"  [{manager}] -> {step['role']}: {step['instruction']}")
                output = self.llm(step["role"], step["instruction"], context=team_context)
                team_context += "\n" + output

            # 2b. The manager condenses raw worker output before reporting up.
            # The CEO will only ever see this summary -- not the workers' text.
            report = self.llm(manager, f"report: {objective}", context=team_context)
            run.team_reports[team] = report
            if verbose:
                print(f"  [{manager}] report up: {report.splitlines()[0][:70]}")

        # 3. CEO synthesizes from the 2 team reports only
        ceo_context = "\n".join(run.team_reports.values())
        run.final_answer = self.llm("ceo", f"synthesize: {task}", context=ceo_context)
        run.llm_calls = self.llm.call_count - before
        return run


# ===========================================================================
# 5. DEBATE PATTERN (2 debaters + judge, 2 rounds)
# ===========================================================================

@dataclass
class DebateRun:
    """Observability record for one debate run."""
    topic: str
    rounds: list[dict[str, str]] = field(default_factory=list)
    verdict: str = ""
    llm_calls: int = 0


class DebatePattern:
    """
    Collaborative debate: every debater answers the same question, then sees
    the opponents' arguments and revises. After N rounds a judge certifies
    the consensus or breaks the tie.

    Cost warning baked into the structure: each round costs len(debaters)
    LLM calls -- this pattern is reserved for decisions worth the spend.
    """

    def __init__(self, llm: MockLLM, debaters: list[str], n_rounds: int = 2) -> None:
        self.llm = llm
        self.debaters = debaters
        self.n_rounds = n_rounds

    def run(self, topic: str, verbose: bool = True) -> DebateRun:
        run = DebateRun(topic=topic)
        before = self.llm.call_count
        latest: dict[str, str] = {d: "" for d in self.debaters}

        for round_no in range(1, self.n_rounds + 1):
            if verbose:
                print(f"[Debate] --- round {round_no} ---")
            this_round: dict[str, str] = {}
            for debater in self.debaters:
                # Each debater sees only the OPPONENTS' previous arguments;
                # showing its own would just waste context.
                opponents = "\n".join(
                    f"{other} argued: {latest[other]}"
                    for other in self.debaters
                    if other != debater and latest[other]
                )
                argument = self.llm(debater, topic, context=opponents)
                this_round[debater] = argument
                if verbose:
                    print(f"[Debate] {argument.splitlines()[0][:76]}")
            # Update AFTER the round so both debaters revised against the
            # same snapshot (simultaneous reveal, no first-mover advantage)
            latest.update(this_round)
            run.rounds.append(this_round)

        # The judge reads the whole transcript and settles the question
        transcript = "\n".join(arg for rnd in run.rounds for arg in rnd.values())
        run.verdict = self.llm("judge", topic, context=transcript)
        if verbose:
            print(f"[Debate] {run.verdict.splitlines()[0][:76]}")
        run.llm_calls = self.llm.call_count - before
        return run


# ===========================================================================
# 6. COMPARISON DEMO
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

    # Reset LLM state for a fair comparison
    llm.call_count = 0
    llm.call_log.clear()

    print("\n--- HIERARCHICAL PATTERN (CEO -> 2 team managers -> workers) ---\n")
    hier = HierarchicalPattern(
        llm,
        teams={
            "research": {"manager": "research_manager", "workers": ["researcher"]},
            "engineering": {"manager": "eng_manager", "workers": ["coder", "reviewer"]},
        },
    )
    hier_run = hier.run(task)
    print("\n=== Final answer (hierarchical) ===")
    print(hier_run.final_answer)
    print(f"LLM calls: {hier_run.llm_calls}")

    # Reset LLM state for a fair comparison
    llm.call_count = 0
    llm.call_log.clear()

    print("\n--- DEBATE PATTERN (optimist vs skeptic, judge, 2 rounds) ---\n")
    debate = DebatePattern(llm, debaters=["optimist", "skeptic"], n_rounds=2)
    debate_run = debate.run("Should we ship the toy quicksort to production?")
    print("\n=== Verdict (debate) ===")
    print(debate_run.verdict)
    print(f"LLM calls: {debate_run.llm_calls}")

    print("\n--- TRADE-OFF SUMMARY ---")
    print(
        f"Supervisor:   {sup_run.llm_calls} calls, centralized control, "
        f"clean synthesis ({len(sup_run.final_answer)} chars)"
    )
    print(
        f"Swarm:        {swarm_run.llm_calls} calls, flat handoffs, "
        f"no synthesis ({len(swarm_run.final_answer)} chars)"
    )
    print(
        f"Hierarchical: {hier_run.llm_calls} calls, 2 management layers, "
        f"CEO context kept small ({sum(len(r) for r in hier_run.team_reports.values())} chars of reports)"
    )
    print(
        f"Debate:       {debate_run.llm_calls} calls for ONE question "
        f"({debate_run.llm_calls - 1} arguments + 1 verdict) -- quality bought with cost"
    )
    print(
        "\nTake-away: supervisor is the default (coherent synthesis, easy to debug). "
        "Swarm fits when the right agent is unknown upfront. Hierarchical only pays "
        "off on very large projects (extra calls buy a small CEO context). Debate is "
        "the most expensive per question -- reserve it for decisions worth the spend."
    )


if __name__ == "__main__":
    demo()

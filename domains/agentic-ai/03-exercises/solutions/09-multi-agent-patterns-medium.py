"""
Solutions -- Day 9 (MEDIUM): Multi-agent patterns

Contains solutions for:
  - Medium Ex 1: Supervisor that ROUTES each subtask to the right specialist,
                 with a generalist fallback (routing pattern + synthesis)
  - Medium Ex 2: Swarm with an EXPLICIT handoff payload -- proves control
                 transfers AND context is preserved across 2 handoffs
  - Medium Ex 3: Shared BLACKBOARD where data-driven agents read/write and an
                 orchestrator detects task completion

Self-contained: embeds a deterministic MockLLM and plain agent callables, so
the file RUNS OFFLINE with zero dependencies (no langgraph, no API key).
langgraph is referenced behind a try/except only to mirror the course code.

Run:  python 03-exercises/solutions/09-multi-agent-patterns-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Optional binding -- the MockLLM fallback guarantees offline execution.
HAS_LANGGRAPH = False
try:  # pragma: no cover - environment dependent
    import langgraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    pass


# ==========================================================================
# MEDIUM EXERCISE 1 -- Supervisor routing with fallback
# ==========================================================================

class RoutingMockLLM:
    """
    Deterministic stub exposing 3 specialists + a generalist fallback.
    Each role tags its output so routing is verifiable downstream.
    """

    def __call__(self, role: str, task: str, context: str = "") -> str:
        if role == "math_agent":
            return f"[math] computed result for: {task}"
        if role == "code_agent":
            return f"[code] def solution():  # for: {task}\n    return 42"
        if role == "text_agent":
            return f"[text] polished prose for: {task}"
        if role == "generalist":
            return f"[generalist] best-effort answer for: {task}"
        if role == "supervisor":
            # Synthesis: REWRITE in one unified voice, not a raw concat.
            agents = sorted({line.split("]")[0].strip("[") for line in context.splitlines() if "]" in line})
            return (
                "Rapport unifie du supervisor :\n"
                f"- {len([l for l in context.splitlines() if l.strip()])} sous-taches traitees\n"
                f"- contributions agregees de : {', '.join(agents)}\n"
                "- toutes les sorties ont ete reecrites dans un style homogene."
            )
        raise ValueError(f"Unknown role: {role}")


# Keyword tables for routing -- pure data, easy to audit/extend.
_MATH_HINTS = ("+", "-", "*", "/", "=", "calcul", "somme", "moyenne", "produit")
_CODE_HINTS = ("code", "fonction", "python", "bug", "implemente")
_TEXT_HINTS = ("resume", "redige", "traduis", "corrige", "reformule")


@dataclass
class RoutingSupervisor:
    llm: RoutingMockLLM

    def route(self, subtask: str) -> str:
        """Total function: always returns a valid agent name, never raises."""
        low = subtask.lower()
        if any(h in low for h in _MATH_HINTS):
            return "math_agent"
        if any(h in low for h in _CODE_HINTS):
            return "code_agent"
        if any(h in low for h in _TEXT_HINTS):
            return "text_agent"
        return "generalist"  # fallback -- never an exception

    def run(self, subtasks: list[str]) -> dict:
        trace: list[tuple[str, str, str]] = []
        for sub in subtasks:
            agent = self.route(sub)
            output = self.llm(agent, sub)
            trace.append((sub, agent, output))
        context = "\n".join(out for _, _, out in trace)
        final = self.llm("supervisor", "synthesize", context=context)
        return {"trace": trace, "final_answer": final}


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Supervisor routing with fallback")
    print("=" * 70)

    sup = RoutingSupervisor(RoutingMockLLM())
    subtasks = [
        "Calcule la moyenne de ces 12 mesures",          # math
        "Implemente une fonction python de tri",          # code
        "Redige un resume de 3 lignes du rapport",         # text
        "Organise un brainstorm sur la strategie 2026",    # -> fallback
        "Traduis le paragraphe d'introduction",            # text
    ]
    result = sup.run(subtasks)

    for sub, agent, out in result["trace"]:
        print(f"  {agent:12} <- {sub[:40]}")
    print("  --- synthesis ---")
    print("  " + result["final_answer"].replace("\n", "\n  "))

    routed = {sub: agent for sub, agent, _ in result["trace"]}
    # Each category routed to the correct specialist
    assert routed["Calcule la moyenne de ces 12 mesures"] == "math_agent"
    assert routed["Implemente une fonction python de tri"] == "code_agent"
    assert routed["Redige un resume de 3 lignes du rapport"] == "text_agent"
    assert routed["Traduis le paragraphe d'introduction"] == "text_agent"
    # At least one subtask hit the fallback generalist
    assert "generalist" in routed.values(), "fallback agent was never used"
    # route() is total: arbitrary garbage still returns a valid name
    assert sup.route("???") == "generalist"
    assert sup.route("") == "generalist"
    # Synthesis is a rewrite, not a raw join of worker outputs
    assert "Rapport unifie" in result["final_answer"]
    assert "contributions agregees" in result["final_answer"]
    print("[Verification] PASS -- routing correct, fallback used, total route()")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Swarm with explicit handoff + context carry-over
# ==========================================================================

@dataclass
class Handoff:
    """Structured control transfer between swarm agents."""
    to_agent: str
    reason: str
    payload: dict = field(default_factory=dict)


# Each agent: (task, inbox) -> (output, handoff | None)

def triage_agent(task: str, inbox: dict) -> tuple[str, Optional[Handoff]]:
    spec = f"build feature for: {task}"
    return ("triage: classified as a coding task", Handoff(
        to_agent="coder", reason="this is a coding task",
        payload={"spec": spec}))


def coder_agent(task: str, inbox: dict) -> tuple[str, Optional[Handoff]]:
    spec = inbox["spec"]  # context received from triage
    code = f"def feature():  # {spec}\n    return 'done'"
    # Forward BOTH what we received and what we produced (no context loss).
    return (f"coder: wrote code from spec={spec!r}", Handoff(
        to_agent="qa", reason="code ready, needs QA",
        payload={"spec": spec, "code": code}))


def qa_agent(task: str, inbox: dict) -> tuple[str, Optional[Handoff]]:
    code, spec = inbox["code"], inbox["spec"]
    assert "def feature" in code
    return (f"qa: validated code for spec={spec!r}", None)  # terminal: no handoff


AGENTS: dict[str, Callable[[str, dict], tuple[str, Optional[Handoff]]]] = {
    "triage": triage_agent,
    "coder": coder_agent,
    "qa": qa_agent,
}


def run_swarm(start_agent: str, task: str, agents: dict, max_hops: int = 8) -> dict:
    control = start_agent
    inbox: dict = {}
    control_seq: list[str] = []
    outputs: list[str] = []
    last_inbox = inbox

    for hop in range(max_hops + 1):
        control_seq.append(control)
        output, handoff = agents[control](task, inbox)
        outputs.append(output)
        if handoff is None:
            last_inbox = inbox  # inbox seen by the terminal agent
            return {"control_seq": control_seq, "outputs": outputs,
                    "final_inbox": last_inbox, "hops": hop}
        # Accumulate context: the next agent receives this payload as inbox.
        last_inbox = handoff.payload
        inbox = handoff.payload
        control = handoff.to_agent
    raise RuntimeError("hop budget exceeded -- possible infinite handoff loop")


def _ping(task, inbox):  # agents that bounce forever (loop-guard test)
    return ("ping", Handoff(to_agent="pong", reason="x", payload={}))


def _pong(task, inbox):
    return ("pong", Handoff(to_agent="ping", reason="x", payload={}))


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Swarm with explicit handoff + context carry-over")
    print("=" * 70)

    result = run_swarm("triage", "user login form", AGENTS)
    print(f"  control sequence: {result['control_seq']}")
    print(f"  qa inbox keys:    {sorted(result['final_inbox'].keys())}")

    # Control really transferred triage -> coder -> qa
    assert result["control_seq"] == ["triage", "coder", "qa"], result["control_seq"]
    # Context survived 2 handoffs: qa saw the original spec AND the coder's code
    final_inbox = result["final_inbox"]
    assert "spec" in final_inbox and "code" in final_inbox, final_inbox
    assert "build feature" in final_inbox["spec"]
    assert "def feature" in final_inbox["code"]
    # Run terminated cleanly (last agent returned handoff=None)
    assert result["outputs"][-1].startswith("qa:")

    # Loop guard: two agents that bounce forever must raise, not hang
    loopers = {"ping": _ping, "pong": _pong}
    try:
        run_swarm("ping", "loop", loopers, max_hops=6)
        assert False, "expected RuntimeError on infinite handoff"
    except RuntimeError as e:
        assert "hop budget" in str(e)
    print("[Verification] PASS -- control transferred, context preserved, loop guarded")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Shared blackboard with completion detection
# ==========================================================================

# Each agent: (precondition, writer). Writer returns ONLY the keys it writes.

def researcher_pre(b: dict) -> bool:
    return b["research"] is None


def researcher_write(b: dict) -> dict:
    return {"research": "facts: Timsort is stable, O(n log n)"}


def coder_pre(b: dict) -> bool:
    return b["research"] is not None and b["code"] is None


def coder_write(b: dict) -> dict:
    return {"code": f"def f(): pass  # informed by {b['research'][:15]}..."}


def reviewer_pre(b: dict) -> bool:
    return b["code"] is not None and b["review"] is None


def reviewer_write(b: dict) -> dict:
    return {"review": f"reviewed: {b['code'][:10]}... LGTM"}


def is_complete(board: dict) -> bool:
    return all(board[k] is not None for k in ("research", "code", "review"))


def run_blackboard(board: dict, agents: list[tuple], max_rounds: int = 10) -> dict:
    """
    agents: list of (name, precondition, writer). Data-driven: only agents
    whose precondition holds run this round. Stops on completion.
    """
    run_count: dict[str, int] = {name: 0 for name, _, _ in agents}
    write_order: list[str] = []

    for _ in range(max_rounds):
        if is_complete(board):
            return {"board": board, "run_count": run_count, "write_order": write_order}
        progressed = False
        for name, pre, writer in agents:
            if pre(board):
                board.update(writer(board))  # merge the patch
                run_count[name] += 1
                write_order.append(name)
                progressed = True
        if not progressed:
            break  # no eligible agent -> deadlock, handled below
    if is_complete(board):
        return {"board": board, "run_count": run_count, "write_order": write_order}
    raise RuntimeError("blackboard deadlock: task could not be completed")


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Shared blackboard with completion detection")
    print("=" * 70)

    board = {"task": "sort + review", "research": None, "code": None, "review": None}
    # Deliberately SHUFFLED order: dependencies must still be respected.
    agents = [
        ("reviewer", reviewer_pre, reviewer_write),
        ("coder", coder_pre, coder_write),
        ("researcher", researcher_pre, researcher_write),
    ]
    result = run_blackboard(board, agents)

    print(f"  write order: {result['write_order']}")
    print(f"  run counts:  {result['run_count']}")
    print(f"  complete:    {is_complete(result['board'])}")

    # Dependency order respected despite shuffled agent list
    wo = result["write_order"]
    assert wo.index("researcher") < wo.index("coder") < wo.index("reviewer"), wo
    # Each agent ran exactly once (idempotence via preconditions)
    assert all(c == 1 for c in result["run_count"].values()), result["run_count"]
    # Completion detected
    assert is_complete(result["board"])

    # Missing-agent board must deadlock (RuntimeError), not loop forever
    broken_board = {"task": "x", "research": None, "code": None, "review": None}
    broken_agents = [  # no coder -> 'code' never gets written
        ("researcher", researcher_pre, researcher_write),
        ("reviewer", reviewer_pre, reviewer_write),
    ]
    try:
        run_blackboard(broken_board, broken_agents)
        assert False, "expected deadlock RuntimeError"
    except RuntimeError as e:
        assert "deadlock" in str(e)
    print("[Verification] PASS -- data-driven order, run-once, completion + deadlock")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 9 MEDIUM Solutions -- Multi-agent patterns")
    print(f"  (langgraph available: {HAS_LANGGRAPH} -- running offline either way)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

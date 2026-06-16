"""
Solutions -- Day 5 (HARD): LangGraph fondamentaux

Contains solutions for:
  - Hard Ex 1: Human-in-the-loop -- interrupt_before + checkpointer + resume
  - Hard Ex 2: Full ReAct agent with cycle budget + stuck detector

Self-contained: embeds an EXTENDED MiniLangGraph stub (interrupt + memory
checkpointer) so the file RUNS OFFLINE with zero dependencies.

Run:  python 03-exercises/solutions/05-langgraph-fondamentaux-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

import copy
import re
from operator import add
from typing import Annotated, Callable, TypedDict


MINI_START = "__start__"
MINI_END = "__end__"


# ==========================================================================
# EXTENDED MINI LANGGRAPH STUB -- interrupt_before + checkpointer
# ==========================================================================

class MemoryCheckpointer:
    """In-memory checkpoint store, keyed by thread_id (deterministic)."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def put(self, thread_id: str, snapshot: dict) -> None:
        # snapshot = {"state": ..., "next": node_name}
        self._store[thread_id] = copy.deepcopy(snapshot)

    def get(self, thread_id: str) -> dict | None:
        snap = self._store.get(thread_id)
        return copy.deepcopy(snap) if snap else None


class MiniStateGraph:
    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple] = {}
        self._reducers: dict[str, Callable] = {}
        hints = getattr(state_schema, "__annotations__", {})
        for key, tp in hints.items():
            meta = getattr(tp, "__metadata__", None)
            if meta:
                for m in meta:
                    if callable(m):
                        self._reducers[key] = m
                        break

    def add_node(self, name, fn): self._nodes[name] = fn
    def add_edge(self, src, dst): self._edges[src] = dst
    def add_conditional_edges(self, src, decider, mapping):
        self._conditional[src] = (decider, mapping)

    def compile(self, checkpointer=None, interrupt_before=None):
        return MiniCompiledGraph(self, checkpointer, interrupt_before or [])


class MiniCompiledGraph:
    def __init__(self, graph, checkpointer=None, interrupt_before=None):
        self.graph = graph
        self.checkpointer = checkpointer
        self.interrupt_before = list(interrupt_before or [])

    def _merge(self, state, updates):
        new_state = dict(state)
        for key, value in updates.items():
            reducer = self.graph._reducers.get(key)
            if reducer is not None and key in new_state:
                new_state[key] = reducer(new_state[key], value)
            else:
                new_state[key] = value
        return new_state

    def _next(self, current, state):
        if current in self.graph._conditional:
            decider, mapping = self.graph._conditional[current]
            return mapping[decider(state)]
        return self.graph._edges.get(current, MINI_END)

    def invoke(self, initial_state, config=None, max_steps: int = 50):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")

        # Resume path: initial_state is None -> load checkpoint and run the
        # pending node WITHOUT re-triggering its own interrupt (we already
        # paused there once and the human has decided).
        if initial_state is None:
            assert self.checkpointer and thread_id, "resume needs checkpointer+thread_id"
            snap = self.checkpointer.get(thread_id)
            assert snap is not None, f"no checkpoint for thread {thread_id}"
            state, current = snap["state"], snap["next"]
            resuming = True
        else:
            state = dict(initial_state)
            current = self._next(MINI_START, state)
            resuming = False

        for _ in range(max_steps):
            if current == MINI_END:
                return {**state, "status": "done"}
            # Pause BEFORE executing an interrupt node -- but not for the node
            # we are explicitly resuming into.
            if current in self.interrupt_before and not resuming:
                if self.checkpointer and thread_id:
                    self.checkpointer.put(thread_id, {"state": state, "next": current})
                return {**state, "status": "interrupted", "pending_node": current}
            resuming = False  # only the first resumed node is exempt
            updates = self.graph._nodes[current](state) or {}
            state = self._merge(state, updates)
            current = self._next(current, state)
        raise RuntimeError("Graph did not terminate")


StateGraph = MiniStateGraph
START = MINI_START
END = MINI_END


# ==========================================================================
# HARD EXERCISE 1 -- Human-in-the-loop: interrupt + checkpoint + resume
# ==========================================================================

class EmailState(TypedDict):
    messages: Annotated[list, add]
    draft: str
    approved: bool
    sent: bool


def draft_node(state: EmailState) -> dict:
    body = "Bonjour, voici le rapport demande. Cordialement."
    return {"draft": body,
            "messages": [{"role": "assistant", "content": f"Draft: {body}"}]}


def send_node(state: EmailState) -> dict:
    # The sensitive action: only proceed if a human approved.
    if state.get("approved"):
        return {"sent": True,
                "messages": [{"role": "system", "content": "EMAIL SENT"}]}
    return {"sent": False,
            "messages": [{"role": "system", "content": "EMAIL REJECTED by human"}]}


def build_email_graph(checkpointer):
    graph = StateGraph(EmailState)
    graph.add_node("draft", draft_node)
    graph.add_node("send", send_node)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "send")
    graph.add_edge("send", END)
    # Pause before the sensitive "send" node for human validation.
    return graph.compile(checkpointer=checkpointer, interrupt_before=["send"])


def run_email_flow(thread_id: str, human_approves: bool, cp: MemoryCheckpointer) -> dict:
    app = build_email_graph(cp)
    config = {"configurable": {"thread_id": thread_id}}

    # 1st invoke: runs draft, then pauses before send
    paused = app.invoke(
        {"messages": [{"role": "user", "content": "Envoie le rapport"}],
         "draft": "", "approved": False, "sent": False},
        config)
    assert paused["status"] == "interrupted"
    assert paused["pending_node"] == "send"
    assert paused["draft"]  # draft was produced before the interrupt

    # Human reviews the checkpointed draft and edits the approval flag
    snap = cp.get(thread_id)
    snap["state"]["approved"] = human_approves
    cp.put(thread_id, snap)

    # 2nd invoke: resume from checkpoint, runs send to END
    final = app.invoke(None, config)
    return final


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Human-in-the-loop: interrupt + checkpoint + resume")
    print("=" * 70)

    cp = MemoryCheckpointer()

    # Thread A: human approves -> email sent
    final_a = run_email_flow("user_A", human_approves=True, cp=cp)
    print(f"  thread A (approved): status={final_a['status']} sent={final_a['sent']}")
    assert final_a["status"] == "done" and final_a["sent"] is True

    # Thread B: human rejects -> email not sent
    final_b = run_email_flow("user_B", human_approves=False, cp=cp)
    print(f"  thread B (rejected): status={final_b['status']} sent={final_b['sent']}")
    assert final_b["status"] == "done" and final_b["sent"] is False

    # Checkpoints are isolated per thread
    assert cp.get("user_A")["state"]["approved"] is True
    assert cp.get("user_B")["state"]["approved"] is False
    assert cp.get("user_A") is not cp.get("user_B")
    print("[Verification] PASS -- interrupt, checkpointed resume, per-thread isolation")


# ==========================================================================
# HARD EXERCISE 2 -- Full ReAct agent with cycle budget + stuck detector
# ==========================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add]
    facts: Annotated[list, add]      # collected facts (progress signal)
    tools_called: Annotated[list, add]
    cycles: int
    no_progress: int                 # consecutive cycles without new facts
    stop_reason: str


def mock_search(query: str) -> str:
    q = query.lower()
    if "population" in q and "paris" in q:
        return "population=2161000"
    if ("area" in q or "superficie" in q) and "paris" in q:
        return "area=105"
    return "no_result"


def mock_calculator(expr: str) -> str:
    if not re.fullmatch(r"[\d\s+\-*/.()]+", expr):
        return "ERROR"
    try:
        return str(round(eval(expr)))  # safe: digits/operators only
    except Exception:
        return "ERROR"


class DensityMockLLM:
    """Correct agent: search population, search area, compute, then answer."""

    def __call__(self, state: AgentState) -> dict:
        facts = dict(f.split("=") for f in state["facts"] if "=" in f)
        if "population" not in facts:
            return {"role": "assistant", "content": "need population",
                    "tool_call": {"name": "search", "args": {"q": "population paris"}}}
        if "area" not in facts:
            return {"role": "assistant", "content": "need area",
                    "tool_call": {"name": "search", "args": {"q": "area paris"}}}
        if "density" not in facts:
            expr = f"{facts['population']}/{facts['area']}"
            return {"role": "assistant", "content": "compute density",
                    "tool_call": {"name": "calculator", "args": {"expr": expr}}}
        return {"role": "assistant",
                "content": f"La densite de Paris est ~{facts['density']} hab/km2."}


class StuckMockLLM:
    """Degenerate agent: forever asks the same fruitless search."""

    def __call__(self, state: AgentState) -> dict:
        return {"role": "assistant", "content": "searching again",
                "tool_call": {"name": "search", "args": {"q": "unknown thing"}}}


def build_react_agent(llm: Callable, cycle_budget: int = 8):
    def agent_node(state: AgentState) -> dict:
        return {"messages": [llm(state)], "cycles": state.get("cycles", 0) + 1}

    def tool_node(state: AgentState) -> dict:
        tc = state["messages"][-1]["tool_call"]
        if tc["name"] == "search":
            result = mock_search(tc["args"]["q"])
        else:
            result = mock_calculator(tc["args"]["expr"])

        new_facts = []
        if "=" in result:
            new_facts.append(result)
        elif tc["name"] == "calculator" and result != "ERROR":
            new_facts.append(f"density={result}")

        # Track progress: did we learn a new fact this cycle?
        progress = bool(new_facts) and any(
            f not in state["facts"] for f in new_facts)
        return {
            "messages": [{"role": "tool", "content": result}],
            "facts": new_facts,
            "tools_called": [tc["name"]],
            "no_progress": 0 if progress else state.get("no_progress", 0) + 1,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not last.get("tool_call"):
            return "final"
        if state.get("no_progress", 0) >= 2:
            return "stuck"
        if state.get("cycles", 0) >= cycle_budget:
            return "budget"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("stop_final", lambda s: {"stop_reason": "final_answer"})
    graph.add_node("stop_stuck", lambda s: {"stop_reason": "stuck"})
    graph.add_node("stop_budget", lambda s: {"stop_reason": "budget_exhausted"})
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools", "final": "stop_final",
        "stuck": "stop_stuck", "budget": "stop_budget",
    })
    graph.add_edge("tools", "agent")
    for stop in ("stop_final", "stop_stuck", "stop_budget"):
        graph.add_edge(stop, END)
    return graph.compile()


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- Full ReAct agent: cycle budget + stuck detector")
    print("=" * 70)

    base = {"messages": [{"role": "user", "content": "densite de Paris ?"}],
            "facts": [], "tools_called": [], "cycles": 0,
            "no_progress": 0, "stop_reason": ""}

    # Success scenario: chains search -> search -> calculator -> answer
    app_ok = build_react_agent(DensityMockLLM(), cycle_budget=8)
    final_ok = app_ok.invoke(copy.deepcopy(base))
    answer = final_ok["messages"][-1]["content"]
    print(f"  success: stop_reason={final_ok['stop_reason']} "
          f"cycles={final_ok['cycles']} tools={final_ok['tools_called']}")
    print(f"           answer: {answer}")
    assert final_ok["stop_reason"] == "final_answer"
    assert "20581" in answer or "20577" in answer, answer  # 2161000/105 = 20581
    assert final_ok["tools_called"] == ["search", "search", "calculator"]

    # Degenerate scenario: stuck detector fires before the budget
    app_stuck = build_react_agent(StuckMockLLM(), cycle_budget=8)
    final_stuck = app_stuck.invoke(copy.deepcopy(base))
    print(f"  stuck:   stop_reason={final_stuck['stop_reason']} "
          f"cycles={final_stuck['cycles']}")
    assert final_stuck["stop_reason"] == "stuck"
    assert final_stuck["cycles"] < 8, "stuck must fire before exhausting budget"
    print("[Verification] PASS -- chained tools, density correct, stuck < budget")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 5 HARD Solutions -- LangGraph fondamentaux")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

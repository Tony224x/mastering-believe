"""
Solutions -- Day 5: LangGraph fondamentaux

Contains:
  - Easy Ex 1: Add a greeter node before the agent
  - Easy Ex 2: Conditional edge with 3 destinations
  - Easy Ex 3: State with a custom reducer (merge_findings)
  - Medium Ex 1: agent -> tools loop with an iteration guard
  - Medium Ex 2: Validation node with a bounded retry-loop
  - Medium Ex 3: Human-in-the-loop interrupt before a sensitive node

Self-contained: embeds a tiny MiniLangGraph stub so no langgraph install
is needed. Uses the same API as the real library on the methods we need.

Run:  python 03-exercises/solutions/05-langgraph-fondamentaux.py
"""

from operator import add
from typing import Annotated, Callable, TypedDict


# ==========================================================================
# MINI LANGGRAPH STUB -- same API as langgraph.graph.StateGraph
# ==========================================================================

MINI_START = "__start__"
MINI_END = "__end__"


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

    def add_node(self, name: str, fn: Callable) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._edges[src] = dst

    def add_conditional_edges(self, src: str, decider: Callable, mapping: dict) -> None:
        self._conditional[src] = (decider, mapping)

    def compile(self) -> "MiniCompiledGraph":
        return MiniCompiledGraph(self)


class MiniCompiledGraph:
    def __init__(self, graph: MiniStateGraph):
        self.graph = graph

    def _merge(self, state: dict, updates: dict) -> dict:
        new_state = dict(state)
        for key, value in updates.items():
            reducer = self.graph._reducers.get(key)
            if reducer is not None and key in new_state:
                new_state[key] = reducer(new_state[key], value)
            else:
                new_state[key] = value
        return new_state

    def _next(self, current: str, state: dict) -> str:
        if current in self.graph._conditional:
            decider, mapping = self.graph._conditional[current]
            return mapping[decider(state)]
        return self.graph._edges.get(current, MINI_END)

    def invoke(self, initial_state: dict, max_steps: int = 50) -> dict:
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        for _ in range(max_steps):
            if current == MINI_END:
                return state
            updates = self.graph._nodes[current](state) or {}
            state = self._merge(state, updates)
            current = self._next(current, state)
        raise RuntimeError("Graph did not terminate")


StateGraph = MiniStateGraph
START = MINI_START
END = MINI_END


# ==========================================================================
# EASY EXERCISE 1 -- Add a greeter node before the agent
# ==========================================================================

class AgentState(TypedDict):
    messages: Annotated[list, add]
    step_count: int


def greeter_node(state: AgentState) -> dict:
    """Inject a friendly greeting that mentions the user's query."""
    first_user = next((m for m in state["messages"] if m.get("role") == "user"), None)
    query = first_user["content"] if first_user else "..."
    msg = {
        "role": "assistant",
        "content": f"Bonjour ! Je vais m'occuper de votre demande: {query}",
    }
    return {
        "messages": [msg],
        "step_count": state.get("step_count", 0) + 1,
    }


def simple_agent_node(state: AgentState) -> dict:
    """A tiny agent that answers based on the last user message."""
    first_user = next((m for m in state["messages"] if m.get("role") == "user"), None)
    text = (first_user or {}).get("content", "")
    # Very naive math: detect "5 + 7" etc.
    import re
    m = re.search(r"(\d+)\s*\+\s*(\d+)", text)
    if m:
        result = int(m.group(1)) + int(m.group(2))
        content = f"The result is {result}."
    else:
        content = "I am here to help."
    return {
        "messages": [{"role": "assistant", "content": content}],
        "step_count": state.get("step_count", 0) + 1,
    }


def solve_ex1() -> None:
    print("\n" + "=" * 70)
    print("EX1 -- Add a greeter node before the agent")
    print("=" * 70)

    graph = StateGraph(AgentState)
    graph.add_node("greeter", greeter_node)
    graph.add_node("agent", simple_agent_node)
    graph.add_edge(START, "greeter")
    graph.add_edge("greeter", "agent")
    graph.add_edge("agent", END)
    app = graph.compile()

    initial = {
        "messages": [{"role": "user", "content": "Compute 5 + 7"}],
        "step_count": 0,
    }
    final = app.invoke(initial)
    print(f"step_count: {final['step_count']}")
    for m in final["messages"]:
        print(f"  [{m['role']:9}] {m['content']}")

    assert len(final["messages"]) >= 3
    assert final["step_count"] >= 2
    print("\n[Verification] PASS")


# ==========================================================================
# EASY EXERCISE 2 -- Conditional edge with 3 destinations
# ==========================================================================

class ClassifierState(TypedDict):
    messages: Annotated[list, add]
    category: str


def classifier_node(state: ClassifierState) -> dict:
    """Classify the user question into one of three categories."""
    first_user = next((m for m in state["messages"] if m.get("role") == "user"), None)
    text = ((first_user or {}).get("content", "")).lower()
    import re
    if re.search(r"\d", text):
        cat = "math"
    elif "meteo" in text or "weather" in text:
        cat = "weather"
    else:
        cat = "default"
    return {"category": cat}


def math_node(state: ClassifierState) -> dict:
    return {"messages": [{"role": "assistant", "content": "Je vois une question math"}]}


def weather_node(state: ClassifierState) -> dict:
    return {"messages": [{"role": "assistant", "content": "Je vois une question meteo"}]}


def default_node(state: ClassifierState) -> dict:
    return {"messages": [{"role": "assistant", "content": "Je vais faire de mon mieux"}]}


def route_from_classifier(state: ClassifierState) -> str:
    return state["category"]


def solve_ex2() -> None:
    print("\n" + "=" * 70)
    print("EX2 -- Conditional edge with 3 destinations")
    print("=" * 70)

    graph = StateGraph(ClassifierState)
    graph.add_node("classifier", classifier_node)
    graph.add_node("math", math_node)
    graph.add_node("weather", weather_node)
    graph.add_node("default", default_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {"math": "math", "weather": "weather", "default": "default"},
    )
    graph.add_edge("math", END)
    graph.add_edge("weather", END)
    graph.add_edge("default", END)
    app = graph.compile()

    cases = [
        ("Compute 42 * 2", "math"),
        ("Quelle est la meteo a Paris ?", "weather"),
        ("Raconte-moi une blague", "default"),
    ]
    for question, expected in cases:
        final = app.invoke({
            "messages": [{"role": "user", "content": question}],
            "category": "",
        })
        last_content = final["messages"][-1]["content"].lower()
        print(f"  question: {question}")
        print(f"    -> category={final['category']}, msg={last_content}")
        assert final["category"] == expected, f"expected {expected}"
        assert expected in last_content or {
            "math": "math",
            "weather": "meteo",
            "default": "mieux",
        }[expected] in last_content

    print("\n[Verification] PASS -- all 3 routes behave correctly")


# ==========================================================================
# EASY EXERCISE 3 -- State with a custom reducer (merge_findings)
# ==========================================================================

def merge_findings(existing: dict, new: dict) -> dict:
    """
    Pure merge of two dicts of lists:
      - Shared keys: concatenate values (lists)
      - New-only keys: add to output
      - Existing-only keys: keep in output
    """
    result = {k: list(v) for k, v in existing.items()}
    for k, v in new.items():
        if k in result:
            result[k] = result[k] + list(v)
        else:
            result[k] = list(v)
    return result


class ResearchState(TypedDict):
    findings: Annotated[dict, merge_findings]
    step_count: int


def searcher_node(state: ResearchState) -> dict:
    return {
        "findings": {"sources": ["src_1", "src_2"]},
        "step_count": state.get("step_count", 0) + 1,
    }


def analyzer_node(state: ResearchState) -> dict:
    return {
        "findings": {"sources": ["src_3"], "keywords": ["ia"]},
        "step_count": state.get("step_count", 0) + 1,
    }


def solve_ex3() -> None:
    print("\n" + "=" * 70)
    print("EX3 -- State with a custom reducer")
    print("=" * 70)

    graph = StateGraph(ResearchState)
    graph.add_node("searcher", searcher_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_edge(START, "searcher")
    graph.add_edge("searcher", "analyzer")
    graph.add_edge("analyzer", END)
    app = graph.compile()

    final = app.invoke({"findings": {}, "step_count": 0})
    print(f"  final findings: {final['findings']}")
    print(f"  step_count:     {final['step_count']}")

    expected_sources = ["src_1", "src_2", "src_3"]
    assert final["findings"]["sources"] == expected_sources
    assert final["findings"]["keywords"] == ["ia"]
    print("\n[Verification] PASS -- custom reducer merged dicts correctly")


# ==========================================================================
# MEDIUM EXERCISE 1 -- agent -> tools loop with an iteration guard
# ==========================================================================

import json
import re


class LoopAgentState(TypedDict):
    messages: Annotated[list, add]
    iterations: int
    pending_tool: str | None


def make_agent_node(broken: bool = False):
    """Mock agent. When broken=True it asks for the same tool forever."""

    def agent_node(state: LoopAgentState) -> dict:
        iterations = state.get("iterations", 0) + 1
        tool_msgs = [m for m in state["messages"] if m.get("role") == "tool"]

        if broken or not tool_msgs:
            # No tool result yet (or broken agent): request the calculator
            return {"pending_tool": "calculator", "iterations": iterations}

        # A tool result exists -> produce the final answer
        result = tool_msgs[-1]["content"]
        return {
            "messages": [{"role": "assistant", "content": f"The result is {result}."}],
            "pending_tool": None,
            "iterations": iterations,
        }

    return agent_node


def tools_node(state: LoopAgentState) -> dict:
    """Execute the pending tool (calculator extracts 'a * b' from the question)."""
    first_user = next(m for m in state["messages"] if m["role"] == "user")
    m = re.search(r"(\d+)\s*\*\s*(\d+)", first_user["content"])
    result = str(int(m.group(1)) * int(m.group(2))) if m else "no expression found"
    return {"messages": [{"role": "tool", "content": result}]}


def route_agent(state: LoopAgentState) -> str:
    # Guard FIRST: a stuck agent must terminate whatever it asks for
    if state.get("iterations", 0) >= 5:
        return "max_iter"
    return "tools" if state.get("pending_tool") else "end"


def max_iter_node(state: LoopAgentState) -> dict:
    return {"messages": [{"role": "assistant", "content": "[MAX ITERATIONS]"}]}


def build_loop_graph(broken: bool = False):
    g = StateGraph(LoopAgentState)
    g.add_node("agent", make_agent_node(broken))
    g.add_node("tools", tools_node)
    g.add_node("max_iter", max_iter_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_agent,
                            {"tools": "tools", "end": END, "max_iter": "max_iter"})
    g.add_edge("tools", "agent")   # The cycle
    g.add_edge("max_iter", END)
    return g.compile()


def solve_medium1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- agent -> tools loop with an iteration guard")
    print("=" * 70)

    app = build_loop_graph()
    final = app.invoke({
        "messages": [{"role": "user", "content": "Compute 12 * 34 then stop"}],
        "iterations": 0, "pending_tool": None,
    })
    for m in final["messages"]:
        print(f"  [{m['role']:9}] {m['content']}")
    print(f"  iterations: {final['iterations']}")
    assert final["iterations"] == 2          # 2 agent passes
    assert any(m["role"] == "tool" and m["content"] == "408" for m in final["messages"])
    assert "408" in final["messages"][-1]["content"]
    print("[Verification] PASS -- 2 agent passes, 1 tool pass, correct answer")

    # Broken agent: keeps asking for the same tool -> guard fires
    broken_app = build_loop_graph(broken=True)
    final2 = broken_app.invoke({
        "messages": [{"role": "user", "content": "Compute 12 * 34"}],
        "iterations": 0, "pending_tool": None,
    })
    assert final2["messages"][-1]["content"] == "[MAX ITERATIONS]"
    assert final2["iterations"] == 5
    print("[Verification] PASS -- broken agent stopped by the guard at 5 iterations")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Validation node with a bounded retry-loop
# ==========================================================================

class DraftState(TypedDict):
    draft: str
    errors: Annotated[list, add]
    retries: int
    validated: bool


def make_generator(always_broken: bool = False):
    def generator(state: DraftState) -> dict:
        # First pass: invalid draft (missing 'summary'). After a fix: valid.
        if always_broken or state.get("retries", 0) == 0:
            return {"draft": '{"confidence": "high"}'}    # 2 violations
        return {"draft": '{"summary": "ok", "confidence": 0.9}'}
    return generator


def validator(state: DraftState) -> dict:
    """3 purely local rules -- no LLM involved."""
    violations: list[str] = []
    try:
        obj = json.loads(state["draft"])
    except json.JSONDecodeError:
        return {"errors": ["draft is not parseable JSON"], "validated": False}

    for key in ("summary", "confidence"):
        if key not in obj:
            violations.append(f"missing key '{key}'")
    conf = obj.get("confidence")
    if "confidence" in obj and not (isinstance(conf, float) and 0.0 <= conf <= 1.0):
        violations.append("confidence must be a float between 0 and 1")

    return {"errors": violations, "validated": not violations}


def make_fixer(always_broken: bool = False):
    def fixer(state: DraftState) -> dict:
        # Mock LLM fix: a correction prompt would carry state['errors']
        retries = state.get("retries", 0) + 1
        if always_broken:
            return {"draft": '{"confidence": "still wrong"}', "retries": retries}
        return {"draft": '{"summary": "ok", "confidence": 0.9}', "retries": retries}
    return fixer


def route_validation(state: DraftState) -> str:
    if state["validated"]:
        return "ok"
    if state.get("retries", 0) >= 2:
        return "give_up"
    return "fix"


def give_up(state: DraftState) -> dict:
    return {"draft": '{"summary": "VALIDATION FAILED", "confidence": 0.0}'}


def build_validation_graph(first_try_valid=False, always_broken=False):
    g = StateGraph(DraftState)
    if first_try_valid:
        g.add_node("generator", lambda s: {"draft": '{"summary": "ok", "confidence": 0.8}'})
    else:
        g.add_node("generator", make_generator(always_broken))
    g.add_node("validator", validator)
    g.add_node("fixer", make_fixer(always_broken))
    g.add_node("give_up", give_up)
    g.add_edge(START, "generator")
    g.add_edge("generator", "validator")
    g.add_conditional_edges("validator", route_validation,
                            {"ok": END, "fix": "fixer", "give_up": "give_up"})
    g.add_edge("fixer", "validator")
    g.add_edge("give_up", END)
    return g.compile()


def solve_medium2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Validation node with a bounded retry-loop")
    print("=" * 70)

    init = {"draft": "", "errors": [], "retries": 0, "validated": False}

    # Path 1: valid on the first try
    f1 = build_validation_graph(first_try_valid=True).invoke(dict(init))
    print(f"  first-try-valid : retries={f1['retries']}, errors={f1['errors']}")
    assert f1["validated"] and f1["retries"] == 0 and not f1["errors"]

    # Path 2: invalid then fixed
    f2 = build_validation_graph().invoke(dict(init))
    print(f"  fixed-after-1   : retries={f2['retries']}, errors={f2['errors']}")
    assert f2["validated"] and f2["retries"] == 1
    assert "missing key 'summary'" in f2["errors"]
    assert "confidence must be a float between 0 and 1" in f2["errors"]

    # Path 3: never valid -> give up at 2 retries
    f3 = build_validation_graph(always_broken=True).invoke(dict(init))
    print(f"  never-valid     : retries={f3['retries']}, draft={f3['draft']}")
    assert not f3["validated"] and f3["retries"] == 2
    assert "VALIDATION FAILED" in f3["draft"]

    print("\n[Verification] PASS -- all 3 paths behave correctly")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Human-in-the-loop interrupt before a sensitive node
# ==========================================================================

class InterruptibleGraph(MiniCompiledGraph):
    """MiniCompiledGraph extension supporting compile(interrupt_before=[...])."""

    def __init__(self, graph: MiniStateGraph, interrupt_before: list[str]):
        super().__init__(graph)
        self.interrupt_before = set(interrupt_before)

    def _run_from(self, state: dict, current: str, skip_interrupt_on: str | None,
                  max_steps: int = 50) -> dict:
        for _ in range(max_steps):
            if current == MINI_END:
                return state
            if current in self.interrupt_before and current != skip_interrupt_on:
                # Pause BEFORE executing the sensitive node
                return {**state, "__interrupted__": current}
            updates = self.graph._nodes[current](state) or {}
            state = self._merge(state, updates)
            current = self._next(current, state)
        raise RuntimeError("Graph did not terminate")

    def invoke(self, initial_state: dict, max_steps: int = 50) -> dict:
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        return self._run_from(state, current, skip_interrupt_on=None, max_steps=max_steps)

    def resume(self, state: dict, max_steps: int = 50) -> dict:
        node = state.get("__interrupted__")
        if node is None:
            raise ValueError("state is not an interrupted state")
        clean = {k: v for k, v in state.items() if k != "__interrupted__"}
        # Resume AT the interrupted node, executing it this time
        return self._run_from(clean, node, skip_interrupt_on=node, max_steps=max_steps)


class EmailState(TypedDict):
    email_to: str
    email_body: str
    sent: bool
    sent_to: str


def draft_email(state: EmailState) -> dict:
    return {"email_to": "boss@corp.com",
            "email_body": "Quarterly report attached."}


def send_email(state: EmailState) -> dict:
    return {"sent": True, "sent_to": state["email_to"]}


def build_email_graph() -> InterruptibleGraph:
    g = MiniStateGraph(EmailState)
    g.add_node("draft_email", draft_email)
    g.add_node("sensitive", send_email)
    g.add_edge(MINI_START, "draft_email")
    g.add_edge("draft_email", "sensitive")
    g.add_edge("sensitive", MINI_END)
    return InterruptibleGraph(g, interrupt_before=["sensitive"])


def solve_medium3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- HITL interrupt before a sensitive node")
    print("=" * 70)

    init = {"email_to": "", "email_body": "", "sent": False, "sent_to": ""}
    app = build_email_graph()

    # Scenario A: approve as-is
    paused = app.invoke(dict(init))
    print(f"  A: paused at '{paused['__interrupted__']}', to={paused['email_to']}")
    assert paused["__interrupted__"] == "sensitive"
    assert paused["sent"] is False                     # NOT executed yet
    final_a = app.resume(paused)
    print(f"  A: resumed -> sent={final_a['sent']} to {final_a['sent_to']}")
    assert final_a["sent"] is True and final_a["sent_to"] == "boss@corp.com"

    # Scenario B: human edits the state before resuming
    paused_b = app.invoke(dict(init))
    paused_b["email_to"] = "assistant@corp.com"        # Human intervention
    final_b = app.resume(paused_b)
    print(f"  B: human redirected -> sent to {final_b['sent_to']}")
    assert final_b["sent_to"] == "assistant@corp.com"

    # Scenario C: human rejects -- never resumes
    paused_c = app.invoke(dict(init))
    paused_c["aborted"] = True
    print(f"  C: rejected -> sent={paused_c['sent']}, aborted={paused_c['aborted']}")
    assert paused_c["sent"] is False

    print("\n[Verification] PASS -- approve / edit / reject all behave correctly")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    solve_ex1()
    solve_ex2()
    solve_ex3()

    solve_medium1()
    solve_medium2()
    solve_medium3()

    # Hard exercises are substantial projects -- key hints:
    #
    # Hard Ex 1 (superstep engine):
    #   - _edges becomes dict[str, list[str]] -- several targets per source
    #   - Run all active nodes on the SAME snapshot, collect updates, then
    #     apply reducers; two writers on a reducer-less key -> InvalidUpdateError
    #   - Deduplicate the next frontier so 'join' runs once per superstep
    #
    # Hard Ex 2 (declarative graph compiler):
    #   - validate_spec collects ALL errors before compile (no early return)
    #   - Reachability: BFS from START over edges (including conditional maps)
    #   - Dead-end check: reverse BFS from END
    #   - export_mermaid: 'graph TD' + 'a --> b' lines + 'a -->|cond| b'

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

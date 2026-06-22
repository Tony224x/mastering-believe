"""
Solutions -- Day 5 (MEDIUM): LangGraph fondamentaux

Contains solutions for:
  - Medium Ex 1: ReAct loop with iteration guard + loop detection
  - Medium Ex 2: Implement stream modes "values" and "updates"
  - Medium Ex 3: Custom add_messages reducer that dedups by id

Self-contained: embeds a tiny MiniLangGraph stub (same API as
langgraph.graph.StateGraph) so the file RUNS OFFLINE with zero dependencies.

Run:  python 03-exercises/solutions/05-langgraph-fondamentaux-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

import copy
import hashlib
from operator import add
from typing import Annotated, Callable, TypedDict


# ==========================================================================
# MINI LANGGRAPH STUB -- same API as langgraph.graph.StateGraph
# (extended in Ex 2 with stream_mode)
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
        raise RuntimeError(f"Graph did not terminate after {max_steps} steps")

    def stream(self, initial_state: dict, stream_mode: str = "updates", max_steps: int = 50):
        """
        Stream per-step events.
          - "updates": yield {node_name: updates} (the delta only)
          - "values":  yield a COPY of the full merged state after each step
        """
        if stream_mode not in ("updates", "values"):
            raise ValueError(f"Unknown stream_mode: {stream_mode!r}")
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        for _ in range(max_steps):
            if current == MINI_END:
                return
            updates = self.graph._nodes[current](state) or {}
            state = self._merge(state, updates)
            if stream_mode == "updates":
                yield {current: updates}
            else:  # values: deep copy so consumers can't corrupt internal state
                yield copy.deepcopy(state)
            current = self._next(current, state)


StateGraph = MiniStateGraph
START = MINI_START
END = MINI_END


# ==========================================================================
# MEDIUM EXERCISE 1 -- ReAct loop with iteration guard + loop detection
# ==========================================================================

class ReactState(TypedDict):
    messages: Annotated[list, add]
    tool_history: Annotated[list, add]   # list of (name, args_repr) already run
    iterations: int
    stop_reason: str


class LoopingMockLLM:
    """Buggy LLM: always re-requests the SAME tool call, even after a result."""

    def __call__(self, messages: list) -> dict:
        return {"role": "assistant", "content": "Let me search again",
                "tool_call": {"name": "search", "args": {"q": "paris"}}}


class GoodMockLLM:
    """Correct LLM: one tool call, then a final answer once it has a result."""

    def __call__(self, messages: list) -> dict:
        if any(m.get("role") == "tool" for m in messages):
            return {"role": "assistant", "content": "Paris has ~2.1M inhabitants."}
        return {"role": "assistant", "content": "Searching",
                "tool_call": {"name": "search", "args": {"q": "paris"}}}


def _tool_key(tool_call: dict) -> tuple:
    return (tool_call["name"], repr(sorted(tool_call.get("args", {}).items())))


def make_react_graph(llm: Callable, max_iterations: int = 5):
    def agent_node(state: ReactState) -> dict:
        response = llm(state["messages"])
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def tool_node(state: ReactState) -> dict:
        last = state["messages"][-1]
        tc = last["tool_call"]
        # Execute (mock) and record the call in history
        result = f"result for {tc['name']}({tc['args']})"
        return {
            "messages": [{"role": "tool", "name": tc["name"], "content": result}],
            "tool_history": [_tool_key(tc)],
        }

    def should_continue(state: ReactState) -> str:
        last = state["messages"][-1]
        # Final answer: no tool call requested
        if not (isinstance(last, dict) and last.get("tool_call")):
            return "final"
        # Iteration cap
        if state.get("iterations", 0) >= max_iterations:
            return "max"
        # Loop detection: the requested call was already executed
        if _tool_key(last["tool_call"]) in state.get("tool_history", []):
            return "loop"
        return "tools"

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("stop_final", lambda s: {"stop_reason": "final_answer"})
    graph.add_node("stop_max", lambda s: {"stop_reason": "max_iterations"})
    graph.add_node("stop_loop", lambda s: {"stop_reason": "loop_detected"})
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools", "final": "stop_final",
        "max": "stop_max", "loop": "stop_loop",
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("stop_final", END)
    graph.add_edge("stop_max", END)
    graph.add_edge("stop_loop", END)
    return graph.compile()


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- ReAct loop with iteration guard + loop detection")
    print("=" * 70)

    base = {"messages": [{"role": "user", "content": "population de Paris ?"}],
            "tool_history": [], "iterations": 0, "stop_reason": ""}

    # Buggy LLM -> should stop via loop detection (not a crash, not raw max)
    app_bad = make_react_graph(LoopingMockLLM(), max_iterations=5)
    final_bad = app_bad.invoke(copy.deepcopy(base))
    print(f"  buggy LLM  -> stop_reason={final_bad['stop_reason']} "
          f"iterations={final_bad['iterations']}")
    assert final_bad["stop_reason"] == "loop_detected", final_bad
    assert final_bad["iterations"] <= 5
    # The duplicate tool call must NOT have been executed twice
    assert len(final_bad["tool_history"]) == 1, final_bad["tool_history"]

    # Good LLM -> should stop via final answer
    app_good = make_react_graph(GoodMockLLM(), max_iterations=5)
    final_good = app_good.invoke(copy.deepcopy(base))
    print(f"  good LLM   -> stop_reason={final_good['stop_reason']} "
          f"iterations={final_good['iterations']}")
    assert final_good["stop_reason"] == "final_answer", final_good
    print("[Verification] PASS -- loop detection + iteration guard + clean exit")


# ==========================================================================
# MEDIUM EXERCISE 2 -- stream modes "values" vs "updates"
# ==========================================================================

class CounterState(TypedDict):
    messages: Annotated[list, add]
    step_count: int


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- stream modes: values vs updates")
    print("=" * 70)

    def n1(s): return {"messages": [{"role": "a", "content": "1"}],
                       "step_count": s.get("step_count", 0) + 1}
    def n2(s): return {"messages": [{"role": "a", "content": "2"}],
                       "step_count": s.get("step_count", 0) + 1}
    def n3(s): return {"messages": [{"role": "a", "content": "3"}],
                       "step_count": s.get("step_count", 0) + 1}

    graph = StateGraph(CounterState)
    for name, fn in (("n1", n1), ("n2", n2), ("n3", n3)):
        graph.add_node(name, fn)
    graph.add_edge(START, "n1")
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    graph.add_edge("n3", END)
    app = graph.compile()

    initial = {"messages": [{"role": "user", "content": "go"}], "step_count": 0}

    updates_events = list(app.stream(copy.deepcopy(initial), stream_mode="updates"))
    values_events = list(app.stream(copy.deepcopy(initial), stream_mode="values"))

    # updates: each event carries ONLY the node delta (1 new message)
    for ev in updates_events:
        (node, upd), = ev.items()
        assert len(upd["messages"]) == 1, "updates must be deltas"
    print(f"  updates: {len(updates_events)} events, each delta has 1 message")

    # values: each event has the FULL cumulative state, strictly growing
    sizes = [len(ev["messages"]) for ev in values_events]
    print(f"  values:  message counts per step = {sizes}")
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), "must grow"

    # last 'values' event == invoke result
    final = app.invoke(copy.deepcopy(initial))
    assert values_events[-1]["messages"] == final["messages"]
    assert values_events[-1]["step_count"] == final["step_count"]

    # unknown mode raises
    try:
        list(app.stream(initial, stream_mode="bananas"))
        assert False, "should have raised"
    except ValueError:
        pass

    # values events are copies: mutating one does not corrupt re-execution
    values_events[0]["messages"].append({"role": "x", "content": "tamper"})
    final2 = app.invoke(copy.deepcopy(initial))
    assert len(final2["messages"]) == len(final["messages"]), "no shared refs"
    print("[Verification] PASS -- modes differ, values grows, copies are safe")


# ==========================================================================
# MEDIUM EXERCISE 3 -- add_messages reducer that dedups by id
# ==========================================================================

def _stable_id(content: str) -> str:
    """Deterministic id from content (no random uuid -> testable)."""
    return "auto-" + hashlib.md5(content.encode()).hexdigest()[:8]


def add_messages(existing: list, new: list) -> list:
    """
    Pure reducer: concatenate, but messages sharing an id UPDATE in place
    (keep original position, no duplicates). Messages without id get a stable
    content-derived id.
    """
    result = [dict(m) for m in existing]          # copy, do not mutate existing
    index = {m.get("id"): i for i, m in enumerate(result)}

    for raw in new:
        msg = dict(raw)                            # copy, do not mutate new
        if "id" not in msg or msg["id"] is None:
            msg["id"] = _stable_id(msg.get("content", ""))
        mid = msg["id"]
        if mid in index:
            result[index[mid]] = msg               # update in place
        else:
            index[mid] = len(result)
            result.append(msg)
    return result


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- add_messages reducer (dedup by id)")
    print("=" * 70)

    def draft_node(s):
        return {"messages": [{"role": "assistant", "id": "a1", "content": "brouillon"}]}

    def revise_node(s):
        return {"messages": [
            {"role": "assistant", "id": "a1", "content": "version finale"},  # update
            {"role": "assistant", "id": "a2", "content": "note de bas"},      # new
        ]}

    graph = StateGraph(ChatState)
    graph.add_node("draft", draft_node)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "revise")
    graph.add_edge("revise", END)
    app = graph.compile()

    final = app.invoke({"messages": [{"role": "user", "id": "u1", "content": "salut"}]})
    ids = [m["id"] for m in final["messages"]]
    print(f"  final ids: {ids}")
    for m in final["messages"]:
        print(f"    [{m['id']}] {m['role']}: {m['content']}")

    # No duplicate ids; a1 was updated, not duplicated
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
    a1 = next(m for m in final["messages"] if m["id"] == "a1")
    assert a1["content"] == "version finale", a1
    assert ids == ["u1", "a1", "a2"], f"order must be preserved: {ids}"

    # Two id-less identical-content messages dedup to the same auto id
    deduped = add_messages([], [
        {"role": "assistant", "content": "same"},
        {"role": "assistant", "content": "same"},
    ])
    assert len(deduped) == 1, "identical content -> same stable id -> dedup"
    print("[Verification] PASS -- updates by id, preserves order, dedups id-less")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 5 MEDIUM Solutions -- LangGraph fondamentaux")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

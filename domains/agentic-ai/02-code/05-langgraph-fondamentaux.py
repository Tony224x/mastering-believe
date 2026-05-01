"""
Day 5 -- LangGraph fondamentaux: StateGraph, nodes, edges, conditional routing

Two modes:
  - REAL mode: if `langgraph` is installed, build a real StateGraph with a
    chat agent + tool node + conditional routing.
  - STUB mode: if `langgraph` is not installed, use a tiny MiniLangGraph
    implemented in pure Python (~80 lines) that mimics the core API:
      StateGraph, add_node, add_edge, add_conditional_edges, compile,
      invoke, stream.

Both modes run the same demo: a mock chat agent that answers a question,
optionally calls a calculator tool, and then synthesizes the final answer.

The LLM is always the MockLLM (no API key required) so the demo is
deterministic and reproducible.

Run:
    python 02-code/05-langgraph-fondamentaux.py
"""

import re
from operator import add
from typing import Annotated, Callable, TypedDict


# ===========================================================================
# OPTIONAL LANGGRAPH IMPORT
# ===========================================================================

try:
    from langgraph.graph import StateGraph as RealStateGraph  # type: ignore
    from langgraph.graph import START as REAL_START  # type: ignore
    from langgraph.graph import END as REAL_END  # type: ignore
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


# ===========================================================================
# MINI LANGGRAPH STUB -- pure Python, ~80 lines, mimics the core API
# ===========================================================================

# Sentinel node names that mirror the real library
MINI_START = "__start__"
MINI_END = "__end__"


class MiniStateGraph:
    """
    A minimal drop-in replacement for langgraph.graph.StateGraph.

    Supports:
      - add_node(name, fn)
      - add_edge(src, dst)
      - add_conditional_edges(src, decider_fn, mapping)
      - compile() -> CompiledGraph
    """

    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple] = {}  # src -> (decider, mapping)
        # Extract reducer hints from Annotated fields in the TypedDict
        self._reducers: dict[str, Callable] = {}
        hints = getattr(state_schema, "__annotations__", {})
        for key, tp in hints.items():
            # For Annotated[X, reducer_fn], __metadata__ holds the reducer
            meta = getattr(tp, "__metadata__", None)
            if meta:
                # Take the first callable from the metadata
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
    """The compiled, runnable form of a MiniStateGraph."""

    def __init__(self, graph: MiniStateGraph):
        self.graph = graph

    def _merge(self, state: dict, updates: dict) -> dict:
        """Merge updates into state using reducers when declared."""
        new_state = dict(state)
        for key, value in updates.items():
            reducer = self.graph._reducers.get(key)
            if reducer is not None and key in new_state:
                new_state[key] = reducer(new_state[key], value)
            else:
                new_state[key] = value
        return new_state

    def _next(self, current: str, state: dict) -> str:
        """Given the current node and state, compute the next node name."""
        if current in self.graph._conditional:
            decider, mapping = self.graph._conditional[current]
            choice = decider(state)
            return mapping[choice]
        return self.graph._edges.get(current, MINI_END)

    def invoke(self, initial_state: dict, max_steps: int = 50) -> dict:
        """Run the graph to completion and return the final state."""
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        for _ in range(max_steps):
            if current == MINI_END:
                return state
            fn = self.graph._nodes[current]
            updates = fn(state) or {}
            state = self._merge(state, updates)
            current = self._next(current, state)
        raise RuntimeError(f"Graph did not terminate after {max_steps} steps")

    def stream(self, initial_state: dict, max_steps: int = 50):
        """Run step-by-step, yielding {node_name: updates} per step."""
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        for _ in range(max_steps):
            if current == MINI_END:
                return
            fn = self.graph._nodes[current]
            updates = fn(state) or {}
            yield {current: updates}
            state = self._merge(state, updates)
            current = self._next(current, state)


# ===========================================================================
# SELECT REAL OR STUB
# ===========================================================================

if _HAS_LANGGRAPH:
    StateGraph = RealStateGraph
    START = REAL_START
    END = REAL_END
    print("[LangGraph] Using the REAL langgraph library")
else:
    StateGraph = MiniStateGraph  # type: ignore
    START = MINI_START
    END = MINI_END
    print("[LangGraph] langgraph not installed -- using MiniLangGraph stub")


# ===========================================================================
# MOCK LLM -- deterministic chat agent
# ===========================================================================

class MockLLM:
    """
    Tiny chat LLM. Looks at the last user message and decides whether to:
      - Call the calculator tool (if the message has a simple expression)
      - Give a final answer (otherwise)
    """

    def __call__(self, messages: list[dict]) -> dict:
        last_user = next((m for m in reversed(messages)
                          if m.get("role") == "user"), None)
        text = (last_user or {}).get("content", "").lower()

        # Detect a math expression
        m = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", text)
        if m and not any(msg.get("role") == "tool" for msg in messages):
            # First time we see the math expr -> request a tool call
            expr = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            return {
                "role": "assistant",
                "content": f"Let me compute {expr}",
                "tool_call": {"name": "calculator", "args": {"expression": expr}},
            }

        # If we already have a tool result, synthesize the final answer
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                return {
                    "role": "assistant",
                    "content": f"The result is {msg['content']}.",
                }

        # Otherwise, a generic response
        return {
            "role": "assistant",
            "content": "I am not sure. Could you ask a math question?",
        }


# ===========================================================================
# TOOLS
# ===========================================================================

def calculator_tool(expression: str) -> str:
    """Safely evaluate a very simple math expression: N op N."""
    m = re.match(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*$", expression)
    if not m:
        return "ERROR: unsupported expression"
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    try:
        return str({"+": a + b, "-": a - b, "*": a * b,
                    "/": a / b if b != 0 else "div_by_zero"}[op])
    except Exception as exc:
        return f"ERROR: {exc}"


# ===========================================================================
# GRAPH DEFINITION
# ===========================================================================

class AgentState(TypedDict):
    # Annotated[list, add] -> new message lists are CONCATENATED, not replaced
    messages: Annotated[list, add]
    step_count: int  # scalar, replaced on update


_llm = MockLLM()


def agent_node(state: AgentState) -> dict:
    """LLM node: reads messages, appends a new assistant message."""
    response = _llm(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


def tool_node(state: AgentState) -> dict:
    """
    Tool node: reads the last assistant message, finds tool_call, executes
    the tool, and appends the result as a 'tool' message.
    """
    last = state["messages"][-1]
    tool_call = last.get("tool_call") if isinstance(last, dict) else None
    if not tool_call:
        return {}
    result = calculator_tool(tool_call["args"]["expression"])
    return {
        "messages": [{"role": "tool", "name": tool_call["name"], "content": result}],
        "step_count": state.get("step_count", 0) + 1,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge: look at the last message, decide tools or end."""
    last = state["messages"][-1]
    if isinstance(last, dict) and last.get("tool_call"):
        return "tools"
    # If the last message is a tool result, go back through agent once more
    if isinstance(last, dict) and last.get("role") == "assistant" \
       and not last.get("tool_call"):
        return "end"
    return "end"


def build_graph():
    """Build and compile the graph. Works for both real and stub backends."""
    graph = StateGraph(AgentState)

    # Add two nodes: the LLM agent and the tool executor
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry: START -> agent
    graph.add_edge(START, "agent")

    # Conditional: after agent, either tools or END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # After tools, always go back to agent (to let it produce a final answer)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ===========================================================================
# DEMOS
# ===========================================================================

def demo_invoke():
    print("\n" + "=" * 70)
    print("DEMO 1 -- invoke: run the graph end-to-end, get final state")
    print("=" * 70)

    app = build_graph()
    initial_state = {
        "messages": [{"role": "user", "content": "Compute 17 * 23 for me"}],
        "step_count": 0,
    }
    final = app.invoke(initial_state)
    print(f"\nFinal step_count: {final['step_count']}")
    print("Final messages:")
    for m in final["messages"]:
        print(f"  [{m.get('role'):9}] {m.get('content')}")


def demo_stream():
    print("\n" + "=" * 70)
    print("DEMO 2 -- stream: see each node's update as it happens")
    print("=" * 70)

    app = build_graph()
    initial_state = {
        "messages": [{"role": "user", "content": "Compute 100 / 4"}],
        "step_count": 0,
    }
    for event in app.stream(initial_state):
        for node, updates in event.items():
            msgs = updates.get("messages", [])
            print(f"[step] node={node} messages_added={len(msgs)}")
            for m in msgs:
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                print(f"          [{m.get('role'):9}] {content[:60]}")


def demo_routing():
    print("\n" + "=" * 70)
    print("DEMO 3 -- conditional routing: non-math query skips tools")
    print("=" * 70)

    app = build_graph()
    initial_state = {
        "messages": [{"role": "user", "content": "Hello agent"}],
        "step_count": 0,
    }
    final = app.invoke(initial_state)
    print(f"\nFinal step_count: {final['step_count']}")
    for m in final["messages"]:
        print(f"  [{m.get('role'):9}] {m.get('content')}")
    # The non-math query should terminate after a single agent call
    assert final["step_count"] == 1, f"Expected 1 step, got {final['step_count']}"
    print("\n[Verification] PASS -- non-math query skipped the tool node.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_routing()

    print("\n" + "=" * 70)
    print("All demos complete.")
    print("=" * 70)

"""
Solutions -- Day 5: LangGraph fondamentaux

Contains:
  - Easy Ex 1: Add a greeter node before the agent
  - Easy Ex 2: Conditional edge with 3 destinations
  - Easy Ex 3: State with a custom reducer (merge_findings)

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
# MAIN
# ==========================================================================

if __name__ == "__main__":
    solve_ex1()
    solve_ex2()
    solve_ex3()

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

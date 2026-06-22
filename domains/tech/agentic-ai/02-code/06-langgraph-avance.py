"""
Day 6 -- LangGraph avance: subgraphs, parallel, streaming, persistence

Demonstrates:
  1. Subgraph composition: a research subgraph embedded in a parent graph
  2. Parallel fan-out / fan-in: multiple mock sources queried in parallel,
     results collected via a list reducer
  3. Streaming: iterate over per-step events
  4. Persistence: an in-memory checkpointer that saves each step and can
     replay or branch from past snapshots

Uses the real `langgraph` if installed, otherwise the MiniLangGraph stub
from day 5 extended with the extra features needed here.

Run:
    python 02-code/06-langgraph-avance.py
"""

import copy
import time
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
# EXTENDED MINI LANGGRAPH -- adds parallel, streaming, and checkpointer
# ===========================================================================

MINI_START = "__start__"
MINI_END = "__end__"


class MiniSend:
    """Mimic of langgraph.constants.Send for parallel fan-out."""

    def __init__(self, node: str, arg: dict):
        self.node = node
        self.arg = arg


class MiniCheckpointer:
    """In-memory checkpointer: stores each step's state for a thread_id."""

    def __init__(self):
        # thread_id -> list of (step_idx, state_snapshot)
        self._storage: dict[str, list[tuple[int, dict]]] = {}

    def save(self, thread_id: str, step: int, state: dict) -> None:
        self._storage.setdefault(thread_id, []).append((step, copy.deepcopy(state)))

    def load_latest(self, thread_id: str) -> dict | None:
        entries = self._storage.get(thread_id, [])
        return copy.deepcopy(entries[-1][1]) if entries else None

    def history(self, thread_id: str) -> list[tuple[int, dict]]:
        return copy.deepcopy(self._storage.get(thread_id, []))

    def load_at(self, thread_id: str, step: int) -> dict | None:
        for s, state in self._storage.get(thread_id, []):
            if s == step:
                return copy.deepcopy(state)
        return None


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

    def compile(self, checkpointer: MiniCheckpointer | None = None) -> "MiniCompiled":
        return MiniCompiled(self, checkpointer)


class MiniCompiled:
    def __init__(self, graph: MiniStateGraph, checkpointer: MiniCheckpointer | None):
        self.graph = graph
        self.checkpointer = checkpointer

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

    def _run_node(self, name: str, state: dict) -> dict:
        """Execute a node. If it returns a list of Send, fan out in parallel."""
        fn = self.graph._nodes[name]
        out = fn(state)
        if isinstance(out, list) and out and isinstance(out[0], MiniSend):
            # Parallel fan-out: run each Send and merge all resulting updates
            merged: dict = {}
            for send in out:
                target_fn = self.graph._nodes[send.node]
                branch_state = {**state, **send.arg}
                branch_updates = target_fn(branch_state) or {}
                # Apply reducers as we merge each branch's updates
                for key, value in branch_updates.items():
                    reducer = self.graph._reducers.get(key)
                    if key in merged and reducer is not None:
                        merged[key] = reducer(merged[key], value)
                    elif key in merged:
                        merged[key] = value
                    else:
                        merged[key] = value
            return merged
        return out or {}

    def invoke(
        self,
        initial_state: dict,
        config: dict | None = None,
        max_steps: int = 50,
    ) -> dict:
        thread_id = (config or {}).get("thread_id", "default")
        # If we have a checkpointer and a past state, resume from it
        if initial_state is None and self.checkpointer:
            state = self.checkpointer.load_latest(thread_id)
            if state is None:
                raise ValueError("No checkpoint to resume from")
        else:
            state = dict(initial_state)

        current = self._next(MINI_START, state)
        step_idx = 0
        while current != MINI_END:
            if step_idx >= max_steps:
                raise RuntimeError("Graph did not terminate")
            updates = self._run_node(current, state)
            state = self._merge(state, updates)
            step_idx += 1
            if self.checkpointer:
                self.checkpointer.save(thread_id, step_idx, state)
            current = self._next(current, state)
        return state

    def stream(self, initial_state: dict, max_steps: int = 50):
        """Yield {node_name: updates} at each step."""
        state = dict(initial_state)
        current = self._next(MINI_START, state)
        for _ in range(max_steps):
            if current == MINI_END:
                return
            updates = self._run_node(current, state)
            yield {current: updates}
            state = self._merge(state, updates)
            current = self._next(current, state)


StateGraph = MiniStateGraph
START = MINI_START
END = MINI_END
Send = MiniSend
Checkpointer = MiniCheckpointer


if _HAS_LANGGRAPH:
    print("[LangGraph] Real library detected, but this demo uses the stub "
          "for portability. The same API would work on the real library.")
print("[LangGraph] Running MiniLangGraph stub (no external dependency)")


# ===========================================================================
# DEMO 1 -- SUBGRAPH COMPOSITION
# ===========================================================================
#
# Build a research subgraph (3 nodes) and embed it as a single node in the
# parent graph. The subgraph is compiled and then wrapped in a Python
# function that invokes it with the parent's state.

class ResearchState(TypedDict):
    query: str
    findings: Annotated[list, add]


def research_search(state: ResearchState) -> dict:
    return {"findings": [f"raw_result_for({state['query']})"]}


def research_filter(state: ResearchState) -> dict:
    # Filter pass: add a "[FILTERED]" marker to each finding
    return {"findings": [f"[FILTERED] {f}" for f in state["findings"]]}


def research_rank(state: ResearchState) -> dict:
    return {"findings": [f"[RANKED_1] {state['findings'][-1]}"]}


def build_research_subgraph():
    g = StateGraph(ResearchState)
    g.add_node("search", research_search)
    g.add_node("filter", research_filter)
    g.add_node("rank", research_rank)
    g.add_edge(START, "search")
    g.add_edge("search", "filter")
    g.add_edge("filter", "rank")
    g.add_edge("rank", END)
    return g.compile()


class ParentState(TypedDict):
    query: str
    research_results: Annotated[list, add]
    final_answer: str


def demo_subgraph():
    print("\n" + "=" * 70)
    print("DEMO 1 -- Subgraph composition")
    print("=" * 70)

    research = build_research_subgraph()

    def research_wrapper(state: ParentState) -> dict:
        """Call the research subgraph and map its output to the parent state."""
        sub_initial = {"query": state["query"], "findings": []}
        sub_final = research.invoke(sub_initial)
        return {"research_results": list(sub_final["findings"])}

    def synthesize(state: ParentState) -> dict:
        joined = "; ".join(state["research_results"])
        return {"final_answer": f"Synthesis: {joined}"}

    parent = StateGraph(ParentState)
    parent.add_node("research", research_wrapper)
    parent.add_node("synthesize", synthesize)
    parent.add_edge(START, "research")
    parent.add_edge("research", "synthesize")
    parent.add_edge("synthesize", END)
    app = parent.compile()

    result = app.invoke({
        "query": "density of Paris",
        "research_results": [],
        "final_answer": "",
    })
    print(f"\nresearch_results: {result['research_results']}")
    print(f"final_answer:     {result['final_answer']}")


# ===========================================================================
# DEMO 2 -- PARALLEL FAN-OUT / FAN-IN
# ===========================================================================
#
# Fan out across 5 "sources", each returns a mock result. The reducer
# (operator.add) on the list field accumulates results as branches finish.

class FanoutState(TypedDict):
    query: str
    source: str                       # set by the Send arg per branch
    results: Annotated[list, add]     # accumulator


def fan_out_node(state: FanoutState) -> list[Send]:
    """Emit one Send per source, all running in parallel."""
    sources = ["src_A", "src_B", "src_C", "src_D", "src_E"]
    return [Send("search_one", {"source": s}) for s in sources]


def search_one(state: FanoutState) -> dict:
    # Simulate a tiny variable latency per source
    time.sleep(0.001)
    return {"results": [f"{state['source']}:data_for({state['query']})"]}


def aggregate(state: FanoutState) -> dict:
    print(f"  aggregated {len(state['results'])} results")
    return {}


def demo_parallel():
    print("\n" + "=" * 70)
    print("DEMO 2 -- Parallel fan-out / fan-in via Send API")
    print("=" * 70)

    g = StateGraph(FanoutState)
    g.add_node("fan_out", fan_out_node)
    g.add_node("search_one", search_one)
    g.add_node("aggregate", aggregate)
    g.add_edge(START, "fan_out")
    g.add_edge("fan_out", "aggregate")
    g.add_edge("aggregate", END)
    app = g.compile()

    result = app.invoke({
        "query": "climate",
        "source": "",
        "results": [],
    })
    print(f"\nall results ({len(result['results'])}):")
    for r in result["results"]:
        print(f"  - {r}")
    assert len(result["results"]) == 5
    print("\n[Verification] PASS -- 5 parallel branches all merged")


# ===========================================================================
# DEMO 3 -- STREAMING
# ===========================================================================

class StreamState(TypedDict):
    messages: Annotated[list, add]
    counter: int


def step_a(state: StreamState) -> dict:
    return {"messages": ["A was here"], "counter": state["counter"] + 1}


def step_b(state: StreamState) -> dict:
    return {"messages": ["B was here"], "counter": state["counter"] + 1}


def step_c(state: StreamState) -> dict:
    return {"messages": ["C was here"], "counter": state["counter"] + 1}


def demo_streaming():
    print("\n" + "=" * 70)
    print("DEMO 3 -- Streaming: see each step's updates as they happen")
    print("=" * 70)

    g = StateGraph(StreamState)
    g.add_node("a", step_a)
    g.add_node("b", step_b)
    g.add_node("c", step_c)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    app = g.compile()

    for event in app.stream({"messages": [], "counter": 0}):
        for node, updates in event.items():
            print(f"  [step] node={node:3} updates={updates}")


# ===========================================================================
# DEMO 4 -- PERSISTENCE + TIME-TRAVEL
# ===========================================================================

class ConvState(TypedDict):
    messages: Annotated[list, add]
    turn: int


def user_node(state: ConvState) -> dict:
    # In a real agent this would be the LLM; we hardcode for determinism
    turn = state["turn"] + 1
    return {"messages": [f"turn_{turn}_response"], "turn": turn}


def demo_persistence():
    print("\n" + "=" * 70)
    print("DEMO 4 -- Persistence + time-travel (in-memory checkpointer)")
    print("=" * 70)

    # Tiny graph with one node that runs exactly once per invoke
    g = StateGraph(ConvState)
    g.add_node("responder", user_node)
    g.add_edge(START, "responder")
    g.add_edge("responder", END)

    ckpt = Checkpointer()
    app = g.compile(checkpointer=ckpt)

    # First turn
    config = {"thread_id": "user_42"}
    app.invoke({"messages": ["hello"], "turn": 0}, config=config)

    # Second turn -- we pass new input but the checkpointer carries turn=1
    # In our stub we simulate multi-turn by explicitly loading+merging
    prev = ckpt.load_latest("user_42")
    state_for_turn_2 = {**prev, "messages": prev["messages"] + ["hi again"]}
    app.invoke(state_for_turn_2, config=config)

    # Third turn
    prev = ckpt.load_latest("user_42")
    state_for_turn_3 = {**prev, "messages": prev["messages"] + ["and again"]}
    app.invoke(state_for_turn_3, config=config)

    print("\nHistory for thread user_42:")
    for step, snap in ckpt.history("user_42"):
        print(f"  step {step}: turn={snap['turn']}, messages={snap['messages']}")

    # Time-travel: re-load state at step 1 and branch a new thread
    print("\nTime-travel: loading state at step 1...")
    step1_state = ckpt.load_at("user_42", 1)
    print(f"  loaded: {step1_state}")

    # Branch: fork into a new thread_id from that past state
    branched_input = {**step1_state,
                      "messages": step1_state["messages"] + ["branched_input"]}
    app.invoke(branched_input, config={"thread_id": "user_42_branch"})
    print("\nBranched thread history:")
    for step, snap in ckpt.history("user_42_branch"):
        print(f"  step {step}: turn={snap['turn']}, messages={snap['messages']}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    demo_subgraph()
    demo_parallel()
    demo_streaming()
    demo_persistence()

    print("\n" + "=" * 70)
    print("All demos complete.")
    print("=" * 70)

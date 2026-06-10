"""
Solutions -- Day 6: LangGraph avance

Contains:
  - Easy Ex 1: Compose two subgraphs (translator + summarizer) in a parent
  - Easy Ex 2: Fan-out on 3 tools via Send API
  - Easy Ex 3: Checkpoint + branching from a past state
  - Medium Ex 1: Multi-mode streaming (values / updates / events)
  - Medium Ex 2: Map-reduce over a dynamic list of documents via Send
  - Medium Ex 3: Multi-session persistence with conversation resume

Self-contained: embeds the mini stub from day 6 so the file runs standalone.

Run:  python 03-exercises/solutions/06-langgraph-avance.py
"""

import copy
from operator import add
from typing import Annotated, Callable, TypedDict


# ==========================================================================
# MINI LANGGRAPH STUB (with Send + Checkpointer)
# ==========================================================================

MINI_START = "__start__"
MINI_END = "__end__"


class Send:
    def __init__(self, node: str, arg: dict):
        self.node = node
        self.arg = arg


class Checkpointer:
    def __init__(self):
        self._storage: dict[str, list[tuple[int, dict]]] = {}

    def save(self, thread_id: str, step: int, state: dict) -> None:
        self._storage.setdefault(thread_id, []).append((step, copy.deepcopy(state)))

    def history(self, thread_id: str) -> list[tuple[int, dict]]:
        return copy.deepcopy(self._storage.get(thread_id, []))

    def load_at(self, thread_id: str, step: int) -> dict | None:
        for s, state in self._storage.get(thread_id, []):
            if s == step:
                return copy.deepcopy(state)
        return None


class StateGraph:
    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
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

    def compile(self, checkpointer: Checkpointer | None = None):
        return CompiledGraph(self, checkpointer)


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: Checkpointer | None):
        self.graph = graph
        self.checkpointer = checkpointer

    def _merge(self, state: dict, updates: dict) -> dict:
        new_state = dict(state)
        for k, v in updates.items():
            reducer = self.graph._reducers.get(k)
            if reducer is not None and k in new_state:
                new_state[k] = reducer(new_state[k], v)
            else:
                new_state[k] = v
        return new_state

    def _run_node(self, name: str, state: dict) -> dict:
        out = self.graph._nodes[name](state)
        if isinstance(out, list) and out and isinstance(out[0], Send):
            merged: dict = {}
            for send in out:
                target_fn = self.graph._nodes[send.node]
                branch_state = {**state, **send.arg}
                branch_updates = target_fn(branch_state) or {}
                for k, v in branch_updates.items():
                    reducer = self.graph._reducers.get(k)
                    if k in merged and reducer is not None:
                        merged[k] = reducer(merged[k], v)
                    elif k in merged:
                        merged[k] = v
                    else:
                        merged[k] = v
            return merged
        return out or {}

    def invoke(self, initial_state: dict, config: dict | None = None) -> dict:
        thread_id = (config or {}).get("thread_id", "default")
        state = dict(initial_state)
        current = self.graph._edges.get(MINI_START, MINI_END)
        step_idx = 0
        while current != MINI_END:
            updates = self._run_node(current, state)
            state = self._merge(state, updates)
            step_idx += 1
            if self.checkpointer:
                self.checkpointer.save(thread_id, step_idx, state)
            current = self.graph._edges.get(current, MINI_END)
        return state


START = MINI_START
END = MINI_END


# ==========================================================================
# EASY EXERCISE 1 -- Compose two subgraphs
# ==========================================================================

class TranslatorState(TypedDict):
    text: str
    lang: str
    translated: str


def detect_language(state: TranslatorState) -> dict:
    return {"lang": "fr"}


def translate(state: TranslatorState) -> dict:
    return {"translated": f"{state['text']}_translated_from_{state['lang']}"}


def build_translator():
    g = StateGraph(TranslatorState)
    g.add_node("detect", detect_language)
    g.add_node("translate", translate)
    g.add_edge(START, "detect")
    g.add_edge("detect", "translate")
    g.add_edge("translate", END)
    return g.compile()


class SummarizerState(TypedDict):
    text: str
    keywords: list
    summary: str


def extract_keywords(state: SummarizerState) -> dict:
    return {"keywords": ["a", "b", "c"]}


def build_summary(state: SummarizerState) -> dict:
    kw = ", ".join(state["keywords"])
    return {"summary": f"summary with keywords: {kw}"}


def build_summarizer():
    g = StateGraph(SummarizerState)
    g.add_node("extract", extract_keywords)
    g.add_node("build", build_summary)
    g.add_edge(START, "extract")
    g.add_edge("extract", "build")
    g.add_edge("build", END)
    return g.compile()


class ParentState(TypedDict):
    text: str
    lang: str
    translated: str
    keywords: list
    summary: str


def solve_ex1() -> None:
    print("\n" + "=" * 70)
    print("EX1 -- Compose two subgraphs")
    print("=" * 70)

    translator = build_translator()
    summarizer = build_summarizer()

    def translator_wrapper(state: ParentState) -> dict:
        sub_out = translator.invoke({"text": state["text"], "lang": "", "translated": ""})
        return {"lang": sub_out["lang"], "translated": sub_out["translated"]}

    def summarizer_wrapper(state: ParentState) -> dict:
        sub_out = summarizer.invoke({"text": state["translated"], "keywords": [], "summary": ""})
        return {"keywords": sub_out["keywords"], "summary": sub_out["summary"]}

    parent = StateGraph(ParentState)
    parent.add_node("translator", translator_wrapper)
    parent.add_node("summarizer", summarizer_wrapper)
    parent.add_edge(START, "translator")
    parent.add_edge("translator", "summarizer")
    parent.add_edge("summarizer", END)
    app = parent.compile()

    result = app.invoke({
        "text": "le chat",
        "lang": "", "translated": "",
        "keywords": [], "summary": "",
    })
    print(f"  lang:       {result['lang']}")
    print(f"  translated: {result['translated']}")
    print(f"  keywords:   {result['keywords']}")
    print(f"  summary:    {result['summary']}")

    assert result["lang"] == "fr"
    assert "translated_from_fr" in result["translated"]
    assert result["keywords"] == ["a", "b", "c"]
    assert "keywords" in result["summary"]
    print("\n[Verification] PASS -- all 4 fields populated")


# ==========================================================================
# EASY EXERCISE 2 -- Fan-out on 3 tools
# ==========================================================================

class FanoutState(TypedDict):
    query: str
    tool: str
    results: Annotated[list, add]


def web_search_tool(query: str) -> str:
    return f"WEB: results about {query}"


def wikipedia_tool(query: str) -> str:
    return f"WIKI: article about {query}"


def arxiv_tool(query: str) -> str:
    return f"ARXIV: paper about {query}"


TOOLS = {
    "web": web_search_tool,
    "wiki": wikipedia_tool,
    "arxiv": arxiv_tool,
}


def dispatcher(state: FanoutState) -> list[Send]:
    # One Send per tool -- all launched in parallel
    return [Send("run_tool", {"tool": name}) for name in TOOLS.keys()]


def run_tool(state: FanoutState) -> dict:
    tool_name = state["tool"]
    result = TOOLS[tool_name](state["query"])
    return {"results": [result]}


def aggregator(state: FanoutState) -> dict:
    return {}


def solve_ex2() -> None:
    print("\n" + "=" * 70)
    print("EX2 -- Fan-out on 3 tools")
    print("=" * 70)

    g = StateGraph(FanoutState)
    g.add_node("dispatcher", dispatcher)
    g.add_node("run_tool", run_tool)
    g.add_node("aggregator", aggregator)
    g.add_edge(START, "dispatcher")
    g.add_edge("dispatcher", "aggregator")
    g.add_edge("aggregator", END)
    app = g.compile()

    result = app.invoke({"query": "LangGraph", "tool": "", "results": []})
    print(f"  results ({len(result['results'])}):")
    for r in result["results"]:
        print(f"    - {r}")

    assert len(result["results"]) == 3
    assert any("WEB" in r for r in result["results"])
    assert any("WIKI" in r for r in result["results"])
    assert any("ARXIV" in r for r in result["results"])
    print("\n[Verification] PASS -- 3 tools fanned out and results collected")


# ==========================================================================
# EASY EXERCISE 3 -- Checkpoint + branching
# ==========================================================================

class ChainState(TypedDict):
    messages: Annotated[list, add]
    counter: int


def make_step(name: str):
    def step(state: ChainState) -> dict:
        return {"messages": [f"{name}_msg"], "counter": state["counter"] + 1}
    return step


def solve_ex3() -> None:
    print("\n" + "=" * 70)
    print("EX3 -- Checkpoint + branching")
    print("=" * 70)

    g = StateGraph(ChainState)
    g.add_node("a", make_step("a"))
    g.add_node("b", make_step("b"))
    g.add_node("c", make_step("c"))
    g.add_node("d", make_step("d"))
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    g.add_edge("d", END)

    ckpt = Checkpointer()
    app = g.compile(checkpointer=ckpt)

    # Run the original thread
    app.invoke({"messages": [], "counter": 0}, config={"thread_id": "original"})

    print("\nOriginal history:")
    for step, snap in ckpt.history("original"):
        print(f"  step {step}: counter={snap['counter']}, messages={snap['messages']}")

    # Load state at step 2 and branch
    step2_state = ckpt.load_at("original", 2)
    print(f"\nState at step 2: {step2_state}")

    branched = {
        "messages": step2_state["messages"] + ["[BRANCHED] I took a different path"],
        "counter": step2_state["counter"],
    }
    # Note: re-running full graph here; in real LangGraph you'd resume with update_state
    app.invoke(branched, config={"thread_id": "fork"})

    print("\nFork history:")
    for step, snap in ckpt.history("fork"):
        print(f"  step {step}: counter={snap['counter']}, messages={snap['messages']}")

    # Verify original is untouched
    orig_hist = ckpt.history("original")
    assert len(orig_hist) == 4
    assert "[BRANCHED]" not in " ".join(orig_hist[-1][1]["messages"])

    # Verify fork contains the injected message
    fork_hist = ckpt.history("fork")
    assert any("[BRANCHED]" in " ".join(snap["messages"]) for _, snap in fork_hist)
    print("\n[Verification] PASS -- original untouched, fork carries the branch")


# ==========================================================================
# MEDIUM EXERCISE 1 -- Multi-mode streaming (values / updates / events)
# ==========================================================================

class StreamingGraph(CompiledGraph):
    """CompiledGraph extension exposing the 3 LangGraph streaming modes."""

    def stream(self, initial_state: dict, mode: str = "values", max_steps: int = 50):
        state = dict(initial_state)
        current = self.graph._edges.get(MINI_START, MINI_END)
        for _ in range(max_steps):
            if current == MINI_END:
                if mode == "events":
                    yield {"event": "on_graph_end", "state": state}
                return
            if mode == "events":
                yield {"event": "on_node_start", "node": current}
            updates = self._run_node(current, state)
            state = self._merge(state, updates)
            if mode == "values":
                yield dict(state)                       # Full state per superstep
            elif mode == "updates":
                yield {current: updates}                # Only the delta
            elif mode == "events":
                yield {"event": "on_node_end", "node": current, "updates": updates}
            current = self.graph._edges.get(current, MINI_END)
        raise RuntimeError("did not terminate")


class PipelineState(TypedDict):
    messages: Annotated[list, add]
    progress: float


def build_pipeline() -> StreamingGraph:
    g = StateGraph(PipelineState)
    g.add_node("fetch", lambda s: {"messages": ["fetched 12 docs"], "progress": 0.33})
    g.add_node("analyze", lambda s: {"messages": ["analyzed docs"], "progress": 0.66})
    g.add_node("report", lambda s: {"messages": ["report ready"], "progress": 1.0})
    g.add_edge(START, "fetch")
    g.add_edge("fetch", "analyze")
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return StreamingGraph(g, checkpointer=None)


def progress_bar(stream) -> list[str]:
    """Consume an 'updates' stream and render a textual progress bar."""
    bars = []
    for chunk in stream:
        for node, updates in chunk.items():
            pct = updates.get("progress", 0.0)
            filled = int(pct * 20)
            bars.append(f"[{'#' * filled}{'.' * (20 - filled)}] {pct:.0%} ({node})")
    return bars


def solve_medium1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Multi-mode streaming")
    print("=" * 70)

    init = {"messages": [], "progress": 0.0}

    values = list(build_pipeline().stream(dict(init), mode="values"))
    print("\n  mode=values (full state per superstep):")
    for v in values:
        print(f"    progress={v['progress']:.2f}, messages={len(v['messages'])}")

    updates = list(build_pipeline().stream(dict(init), mode="updates"))
    print("\n  mode=updates (deltas only):")
    for u in updates:
        print(f"    {u}")

    events = list(build_pipeline().stream(dict(init), mode="events"))
    print("\n  mode=events:")
    for e in events:
        print(f"    {e['event']:14} {e.get('node', '')}")

    print("\n  progress bar from the updates stream:")
    for line in progress_bar(build_pipeline().stream(dict(init), mode="updates")):
        print(f"    {line}")

    assert len(values) == 3 and values[-1]["progress"] == 1.0
    assert len(updates) == 3 and all(len(u) == 1 for u in updates)
    assert len(events) == 7   # 3 starts + 3 ends + 1 graph_end
    assert values[0]["messages"] == ["fetched 12 docs"]   # cumulative
    assert list(updates[1].keys()) == ["analyze"]          # delta only
    print("\n[Verification] PASS -- 3 values, 3 updates, 7 events")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Map-reduce over a dynamic list of documents
# ==========================================================================

class MapReduceState(TypedDict):
    docs: list
    doc: str
    index: int
    summaries: Annotated[list, add]
    final_digest: str


def mr_dispatcher(state: MapReduceState) -> list[Send]:
    # One Send per document -- the graph never hardcodes the doc count
    return [Send("summarize_one", {"doc": doc, "index": i})
            for i, doc in enumerate(state["docs"])]


def summarize_one(state: MapReduceState) -> dict:
    doc, index = state["doc"], state["index"]
    if not doc.strip():
        return {"summaries": [(index, "[EMPTY DOC]")]}
    first_sentence = doc.split(".")[0].strip()
    return {"summaries": [(index, f"{first_sentence} [{len(doc)} chars]")]}


def mr_aggregate(state: MapReduceState) -> dict:
    # Sort by index: arrival order is NOT guaranteed in real parallel runs
    ordered = sorted(state["summaries"], key=lambda pair: pair[0])
    digest = "\n".join(f"  doc {i}: {s}" for i, s in ordered)
    return {"final_digest": digest}


def build_mapreduce():
    g = StateGraph(MapReduceState)
    g.add_node("dispatcher", mr_dispatcher)
    g.add_node("summarize_one", summarize_one)
    g.add_node("aggregate", mr_aggregate)
    g.add_edge(START, "dispatcher")
    g.add_edge("dispatcher", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


def solve_medium2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Map-reduce via Send")
    print("=" * 70)

    corpus_3 = [
        "LangGraph models agents as graphs. It supports cycles and state.",
        "Send enables dynamic fan-out. Each branch gets its own payload.",
        "Reducers merge concurrent updates. The add reducer concatenates lists.",
    ]
    corpus_5 = corpus_3 + [
        "",                                                   # Empty doc
        "Checkpointers persist state. Threads isolate conversations.",
    ]

    app = build_mapreduce()
    for label, corpus in (("3 docs", corpus_3), ("5 docs (1 empty)", corpus_5)):
        final = app.invoke({"docs": corpus, "doc": "", "index": -1,
                            "summaries": [], "final_digest": ""})
        print(f"\n  corpus {label}:")
        print(final["final_digest"])
        assert len(final["summaries"]) == len(corpus)
        for i in range(len(corpus)):
            assert f"doc {i}:" in final["final_digest"]
        if "" in corpus:
            assert "[EMPTY DOC]" in final["final_digest"]

    print("\n[Verification] PASS -- fan-out adapts to corpus size, empty doc handled")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Multi-session persistence with conversation resume
# ==========================================================================

class ChatState(TypedDict):
    messages: Annotated[list, add]
    turn_count: int


def recall_node(state: ChatState) -> dict:
    turns = sum(1 for m in state["messages"] if m["role"] == "user")
    return {"turn_count": turns}


def respond_node(state: ChatState) -> dict:
    """Mock assistant that uses earlier turns as context."""
    last_user = [m for m in state["messages"] if m["role"] == "user"][-1]["content"]
    history = " ".join(m["content"] for m in state["messages"])

    if "budget" in last_user.lower() and "?" in last_user:
        m = re.search(r"(\d+)\s*EUR", history)
        reply = (f"Your budget is {m.group(1)} EUR." if m
                 else "I do not know your budget yet.")
    elif "budget" in history.lower() and "leger" in last_user.lower():
        reply = "Noted: lightweight laptop, within your 500 EUR budget."
    elif "laptop" in last_user.lower():
        reply = "Sure, I will look for a laptop."
    else:
        reply = "Understood."
    return {"messages": [{"role": "assistant", "content": reply}]}


import re  # noqa: E402  (used by respond_node)


def build_chat_graph(ckpt: Checkpointer):
    g = StateGraph(ChatState)
    g.add_node("recall", recall_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "recall")
    g.add_edge("recall", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=ckpt)


def chat(app, ckpt: Checkpointer, thread_id: str, user_message: str) -> str:
    # Load the latest persisted state for this thread (or start fresh)
    history = ckpt.history(thread_id)
    state = history[-1][1] if history else {"messages": [], "turn_count": 0}
    state = dict(state)
    state["messages"] = state["messages"] + [{"role": "user", "content": user_message}]
    final = app.invoke(state, config={"thread_id": thread_id})
    return final["messages"][-1]["content"]


def solve_medium3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Multi-session persistence")
    print("=" * 70)

    ckpt = Checkpointer()

    # --- Session 1 (thread alice) ---
    app = build_chat_graph(ckpt)
    r1 = chat(app, ckpt, "alice", "Je cherche un laptop, budget 500 EUR")
    r2 = chat(app, ckpt, "alice", "Plutot leger si possible")
    print(f"  alice t1: {r1}")
    print(f"  alice t2: {r2}")

    # --- Simulated restart: new compiled graph, SAME checkpointer ---
    del app
    app2 = build_chat_graph(ckpt)

    # --- Session 2 (same thread) ---
    r3 = chat(app2, ckpt, "alice", "Quel etait mon budget deja ?")
    print(f"  alice t3 (after restart): {r3}")
    final_alice = ckpt.history("alice")[-1][1]
    assert "500" in r3
    assert final_alice["turn_count"] == 3

    # --- Session 3 (thread bob, isolated) ---
    r4 = chat(app2, ckpt, "bob", "Quel etait mon budget deja ?")
    print(f"  bob   t1: {r4}")
    assert "500" not in r4
    assert ckpt.history("bob")[-1][1]["turn_count"] == 1

    print("\n[Verification] PASS -- state survives restart, threads are isolated")


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
    # Hard Ex 1 (time-travel debugger):
    #   - Save (step, node, updates) metadata alongside each checkpoint
    #   - diff: compare fields, render lists as '+N elements'
    #   - fork: load_at + mutate + invoke on a new thread_id
    #   - replay: re-invoke from step 0 and compare with stored final state
    #
    # Hard Ex 2 (isolated-state subgraphs):
    #   - as_subgraph_node(sub_app, map_in, map_out, on_error, max_retries)
    #     returns a parent-node function that translates states both ways
    #   - on_error='retry': loop sub_app.invoke in try/except, then 'continue'
    #   - Assert no sub-state key leaks into the parent's final state

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

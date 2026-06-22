"""
Solutions -- Day 6: LangGraph avance

Contains:
  - Easy Ex 1: Compose two subgraphs (translator + summarizer) in a parent
  - Easy Ex 2: Fan-out on 3 tools via Send API
  - Easy Ex 3: Checkpoint + branching from a past state

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
# MAIN
# ==========================================================================

if __name__ == "__main__":
    solve_ex1()
    solve_ex2()
    solve_ex3()

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

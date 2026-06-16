"""
Solutions -- Day 6 (MEDIUM): LangGraph avance

Contains solutions for:
  - Medium Ex 1: Map-reduce with a CUSTOM reducer (dedup-best-by-key) and a
                 flaky branch that fails gracefully without killing the graph
  - Medium Ex 2: Checkpoint + RESUME mid-graph after a simulated crash
                 (already-run nodes are NOT replayed)
  - Medium Ex 3: DYNAMIC fan-out where the number of Send branches is computed
                 from the state (document chunking), with an ordering reducer

Everything runs OFFLINE with zero dependencies. We embed the MiniLangGraph
stub (Send + Checkpointer + custom reducers) adapted from the day-6 easy
solution. If the real `langgraph` is installed it is detected, but for
portability the demos always run on the stub.

Run:  python 03-exercises/solutions/06-langgraph-avance-medium.py
Each solution is self-contained and ends with assertions (self-test).
"""

import copy
from operator import add
from typing import Annotated, Callable, TypedDict


# ==========================================================================
# OPTIONAL REAL LANGGRAPH DETECTION (we still run on the stub for portability)
# ==========================================================================

try:
    import langgraph  # type: ignore  # noqa: F401
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


# ==========================================================================
# MINI LANGGRAPH STUB (Send + Checkpointer + custom reducers)
# ==========================================================================
#
# This is the same engine as the easy day-6 solution, with two additions
# needed for the medium exercises:
#   - invoke() accepts `start_node` and `start_step` so we can RESUME a graph
#     mid-way (Ex 2) instead of always starting from START.
#   - invoke() can swallow a node crash and report it (so the caller can
#     resume) -- controlled by the `safe` flag.

MINI_START = "__start__"
MINI_END = "__end__"


class Send:
    """Mimic of langgraph.constants.Send for parallel fan-out."""

    def __init__(self, node: str, arg: dict):
        self.node = node
        self.arg = arg


class NodeCrash(Exception):
    """Raised by a node to simulate a crash; carries the node name + step."""

    def __init__(self, node: str, step: int, original: Exception):
        super().__init__(f"node {node!r} crashed at step {step}: {original}")
        self.node = node
        self.step = step
        self.original = original


class Checkpointer:
    """In-memory checkpointer: stores each step's state per thread_id."""

    def __init__(self):
        self._storage: dict[str, list[tuple[int, str, dict]]] = {}

    def save(self, thread_id: str, step: int, node: str, state: dict) -> None:
        # We also remember which node produced this snapshot (for resume).
        self._storage.setdefault(thread_id, []).append(
            (step, node, copy.deepcopy(state)))

    def history(self, thread_id: str) -> list[tuple[int, str, dict]]:
        return copy.deepcopy(self._storage.get(thread_id, []))

    def latest(self, thread_id: str) -> tuple[int, str, dict] | None:
        entries = self._storage.get(thread_id, [])
        return copy.deepcopy(entries[-1]) if entries else None

    def load_at(self, thread_id: str, step: int) -> dict | None:
        for s, _node, state in self._storage.get(thread_id, []):
            if s == step:
                return copy.deepcopy(state)
        return None


class StateGraph:
    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple] = {}
        self._reducers: dict[str, Callable] = {}
        # Read Annotated[...] metadata to find per-field reducers.
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

    def compile(self, checkpointer: Checkpointer | None = None) -> "CompiledGraph":
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
        """Run a node. If it returns a list of Send, fan out (parallel)."""
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
                    else:
                        merged[k] = v
            return merged
        return out or {}

    def _next(self, current: str, state: dict) -> str:
        if current in self.graph._conditional:
            decider, mapping = self.graph._conditional[current]
            return mapping[decider(state)]
        return self.graph._edges.get(current, MINI_END)

    def invoke(
        self,
        initial_state: dict,
        config: dict | None = None,
        start_node: str | None = None,
        start_step: int = 0,
        safe: bool = False,
    ) -> dict:
        """
        Run the graph.

        - start_node/start_step let us RESUME mid-graph (Ex 2): execution begins
          at `start_node` with the step counter already at `start_step`.
        - safe=True: if a node raises, we keep all checkpoints written so far and
          re-raise a NodeCrash carrying enough info for the caller to resume.
        """
        thread_id = (config or {}).get("thread_id", "default")
        state = dict(initial_state)
        current = start_node if start_node is not None else self._next(MINI_START, state)
        step_idx = start_step
        while current != MINI_END:
            try:
                updates = self._run_node(current, state)
            except Exception as exc:  # noqa: BLE001 -- intentional crash boundary
                if safe:
                    # Re-raise with the node where it died so we can resume there.
                    raise NodeCrash(current, step_idx + 1, exc) from exc
                raise
            state = self._merge(state, updates)
            step_idx += 1
            if self.checkpointer:
                self.checkpointer.save(thread_id, step_idx, current, state)
            current = self._next(current, state)
        return state


START = MINI_START
END = MINI_END


# ==========================================================================
# MEDIUM EXERCISE 1 -- Map-reduce, custom reducer, graceful branch failure
# ==========================================================================
#
# Pattern: fan out one Send per source, each worker returns either an OK hit or
# an (internally captured) failure. The custom reducer dedups by source keeping
# the highest score, so duplicates and parallel arrival order are both handled.

def merge_best(left: list, right: list) -> list:
    """
    CUSTOM reducer: merge two result lists, deduplicating by `source` and
    keeping, for each source, the entry with the highest `score`. Failures
    (ok=False) are kept too but never displace an OK entry for the same source.
    """
    by_source: dict[str, dict] = {}
    for entry in [*left, *right]:
        src = entry["source"]
        prev = by_source.get(src)
        if prev is None:
            by_source[src] = entry
            continue
        # Prefer an OK entry over a failed one; among equals, the higher score.
        if entry.get("ok") and not prev.get("ok"):
            by_source[src] = entry
        elif entry.get("ok") == prev.get("ok") and entry["score"] > prev["score"]:
            by_source[src] = entry
    # Stable, deterministic output: sort by source name.
    return [by_source[s] for s in sorted(by_source)]


class MapReduceState(TypedDict):
    query: str
    sources: list
    results: Annotated[list, merge_best]   # custom reducer (not `add`)
    summary: dict


# A source name listed here will simulate an intermittent failure.
_FLAKY_SOURCES = {"src_C"}

# Deterministic per-source mock scores so the test is reproducible.
_SOURCE_SCORES = {"src_A": 0.9, "src_B": 0.6, "src_C": 0.5,
                  "src_D": 0.8, "src_E": 0.4}


def mr_fetch_one(state: MapReduceState) -> dict:
    """
    Worker: returns an OK hit OR fails GRACEFULLY (captured internally) for a
    flaky source. A failure NEVER propagates -- the branch returns ok=False.
    """
    source = state["source"]  # injected via the Send arg
    try:
        if source in _FLAKY_SOURCES:
            # Simulate a transient error inside the branch.
            raise ConnectionError(f"{source} timed out")
        score = _SOURCE_SCORES.get(source, 0.1)
        return {"results": [{"source": source, "score": score, "ok": True,
                             "doc": f"doc_from({source})"}]}
    except Exception as exc:  # noqa: BLE001 -- branch-local graceful handling
        return {"results": [{"source": source, "score": 0.0, "ok": False,
                             "error": str(exc)}]}


def mr_dispatch(state: MapReduceState) -> list:
    """Fan out one Send per source. Note: src_A is sent TWICE on purpose
    (with different scores) to prove the reducer dedups by source."""
    sends = [Send("fetch_one", {"source": s}) for s in state["sources"]]
    # Inject a duplicate src_A via a node that returns a deliberately lower
    # score, so merge_best must keep the higher one.
    sends.append(Send("fetch_one_low", {"source": "src_A"}))
    return sends


def mr_fetch_one_low(state: MapReduceState) -> dict:
    """A second worker for the SAME source returning a deliberately low score,
    used to prove deduplication keeps the best score."""
    return {"results": [{"source": state["source"], "score": 0.1, "ok": True,
                         "doc": "stale_duplicate"}]}


def mr_reduce(state: MapReduceState) -> dict:
    results = state["results"]
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    ok_sorted = sorted(ok, key=lambda r: r["score"], reverse=True)
    return {"summary": {
        "n_ok": len(ok),
        "n_fail": len(fail),
        "ok_sources": [r["source"] for r in ok_sorted],
        "failed_sources": [r["source"] for r in fail],
    }}


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Map-reduce, custom reducer, graceful failure")
    print("=" * 70)

    g = StateGraph(MapReduceState)
    g.add_node("dispatch", mr_dispatch)
    g.add_node("fetch_one", mr_fetch_one)
    g.add_node("fetch_one_low", mr_fetch_one_low)
    g.add_node("reduce", mr_reduce)
    g.add_edge(START, "dispatch")
    g.add_edge("dispatch", "reduce")
    g.add_edge("reduce", END)
    app = g.compile()

    result = app.invoke({
        "query": "vector databases",
        "sources": ["src_A", "src_B", "src_C", "src_D", "src_E"],
        "results": [],
        "summary": {},
    })
    summary = result["summary"]
    print(f"  n_ok={summary['n_ok']}  n_fail={summary['n_fail']}")
    print(f"  ok_sources (by score):  {summary['ok_sources']}")
    print(f"  failed_sources:         {summary['failed_sources']}")

    # src_C failed gracefully -> exactly 1 failure.
    assert summary["n_fail"] == 1, summary
    assert summary["failed_sources"] == ["src_C"], summary
    # A,B,D,E succeeded -> 4 OK, and src_A appears ONCE despite being sent twice.
    assert summary["n_ok"] == 4, summary
    assert summary["ok_sources"].count("src_A") == 1, "dedup by source failed"
    # The best score (0.9) won over the stale duplicate (0.1): src_A is first.
    assert summary["ok_sources"][0] == "src_A", summary
    # No exception ever reached us (we got a result back at all).
    print("[Verification] PASS -- dedup keeps best score, flaky branch isolated")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Checkpoint + resume mid-graph after a crash
# ==========================================================================
#
# We instrument call counts per node to PROVE that resuming does not replay
# nodes that already ran. step_3 crashes the FIRST time, then succeeds on the
# resume; step_1/step_2 must run exactly once across the whole run.

class ChainState(TypedDict):
    messages: Annotated[list, add]
    counter: int


# Linear order of the graph -- used to compute the resume point.
CHAIN_ORDER = ["step_1", "step_2", "step_3", "step_4", "step_5"]


def make_chain_node(name: str, call_counter: dict, crash_once: bool = False):
    """Factory for a chain node that records how many times it actually runs."""

    def node(state: ChainState) -> dict:
        call_counter[name] = call_counter.get(name, 0) + 1
        # Crash the FIRST time this node runs (to simulate a mid-graph crash).
        if crash_once and call_counter[name] == 1:
            raise RuntimeError(f"{name} crashed (transient)")
        return {"messages": [f"{name}_msg"], "counter": state["counter"] + 1}

    return node


def run_with_resume(app: CompiledGraph, ckpt: Checkpointer, initial: dict,
                    thread_id: str, max_resumes: int = 3) -> dict:
    """
    Run a graph; if a node crashes, reload the LATEST checkpoint and resume from
    the node that crashed -- WITHOUT replaying earlier nodes.
    """
    state = initial
    start_node: str | None = None
    start_step = 0
    for attempt in range(max_resumes + 1):
        try:
            return app.invoke(state, config={"thread_id": thread_id},
                              start_node=start_node, start_step=start_step,
                              safe=True)
        except NodeCrash as crash:
            print(f"  [crash] {crash.node} died at step {crash.step}; resuming...")
            latest = ckpt.latest(thread_id)
            if latest is None:
                # Crash before any checkpoint: restart from the crashed node,
                # carrying the original input forward.
                start_node = crash.node
                start_step = crash.step - 1
                state = initial
            else:
                done_step, done_node, snapshot = latest
                # Resume from the node AFTER the last successfully checkpointed
                # one -- which is exactly the node that crashed.
                idx = CHAIN_ORDER.index(done_node)
                start_node = CHAIN_ORDER[idx + 1]
                start_step = done_step
                state = snapshot
    raise RuntimeError("exceeded resume budget")


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Checkpoint + resume mid-graph after a crash")
    print("=" * 70)

    calls: dict[str, int] = {}
    g = StateGraph(ChainState)
    g.add_node("step_1", make_chain_node("step_1", calls))
    g.add_node("step_2", make_chain_node("step_2", calls))
    g.add_node("step_3", make_chain_node("step_3", calls, crash_once=True))
    g.add_node("step_4", make_chain_node("step_4", calls))
    g.add_node("step_5", make_chain_node("step_5", calls))
    g.add_edge(START, "step_1")
    g.add_edge("step_1", "step_2")
    g.add_edge("step_2", "step_3")
    g.add_edge("step_3", "step_4")
    g.add_edge("step_4", "step_5")
    g.add_edge("step_5", END)

    ckpt = Checkpointer()
    app = g.compile(checkpointer=ckpt)

    final = run_with_resume(app, ckpt,
                            {"messages": [], "counter": 0}, "job_1")

    print(f"\n  final messages: {final['messages']}")
    print(f"  final counter:  {final['counter']}")
    print(f"  per-node call counts: {calls}")

    # 5 distinct messages, in order, no duplicates.
    assert final["messages"] == [f"step_{i}_msg" for i in range(1, 6)], final
    assert len(set(final["messages"])) == 5, "duplicate messages -> replayed!"
    assert final["counter"] == 5, final
    # step_1 and step_2 ran ONCE each (not replayed on resume).
    assert calls["step_1"] == 1, f"step_1 replayed: {calls}"
    assert calls["step_2"] == 1, f"step_2 replayed: {calls}"
    # step_3 ran twice (crash + successful resume); 4 and 5 once.
    assert calls["step_3"] == 2, calls
    assert calls["step_4"] == 1 and calls["step_5"] == 1, calls
    print("[Verification] PASS -- resumed from crash, no replay of earlier nodes")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Dynamic fan-out (branch count from state)
# ==========================================================================
#
# The planner chunks the document into chunks of `chunk_size`, then emits ONE
# Send per chunk. Branch count therefore depends on the state. The reducer
# keeps annotations ordered by chunk_id despite non-deterministic arrival.

def order_by_chunk(left: list, right: list) -> list:
    """Custom reducer: merge and keep annotations sorted by chunk_id."""
    return sorted([*left, *right], key=lambda a: a["chunk_id"])


class ChunkState(TypedDict):
    document: list
    chunk_size: int
    chunk: list             # set per-branch via the Send arg
    chunk_id: int           # set per-branch via the Send arg
    annotations: Annotated[list, order_by_chunk]
    final: str
    coverage: int


def chunk_planner(state: ChunkState) -> list:
    """DYNAMIC fan-out: number of Send branches = ceil(len(doc)/chunk_size)."""
    doc = state["document"]
    size = state["chunk_size"]
    sends = []
    for chunk_id, start in enumerate(range(0, len(doc), size)):
        chunk = doc[start:start + size]   # last chunk may be shorter
        sends.append(Send("annotate_chunk",
                          {"chunk": chunk, "chunk_id": chunk_id}))
    return sends


def annotate_chunk(state: ChunkState) -> dict:
    chunk = state["chunk"]
    return {"annotations": [{
        "chunk_id": state["chunk_id"],
        "n": len(chunk),
        "text": " ".join(chunk).upper(),
    }]}


def assemble(state: ChunkState) -> dict:
    anns = state["annotations"]   # already ordered by the reducer
    final = " || ".join(a["text"] for a in anns)
    coverage = sum(a["n"] for a in anns)
    return {"final": final, "coverage": coverage}


def build_chunk_graph() -> CompiledGraph:
    g = StateGraph(ChunkState)
    g.add_node("planner", chunk_planner)
    g.add_node("annotate_chunk", annotate_chunk)
    g.add_node("assemble", assemble)
    g.add_edge(START, "planner")
    g.add_edge("planner", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


def run_chunking(document: list, chunk_size: int) -> dict:
    app = build_chunk_graph()
    return app.invoke({
        "document": document, "chunk_size": chunk_size,
        "chunk": [], "chunk_id": 0,
        "annotations": [], "final": "", "coverage": 0,
    })


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Dynamic fan-out (branch count from state)")
    print("=" * 70)

    doc = ["the", "cat", "sat", "on", "the", "mat", "today"]  # 7 sentences

    import math
    for chunk_size in (1, 2, 3):
        result = run_chunking(doc, chunk_size)
        n_branches = len(result["annotations"])
        expected_branches = math.ceil(len(doc) / chunk_size)
        chunk_ids = [a["chunk_id"] for a in result["annotations"]]
        print(f"\n  chunk_size={chunk_size}: branches={n_branches} "
              f"coverage={result['coverage']} ids={chunk_ids}")
        print(f"    final = {result['final']}")

        # Branch count is computed from state (varies with chunk_size).
        assert n_branches == expected_branches, (chunk_size, n_branches)
        # Reducer keeps chunk_ids ordered.
        assert chunk_ids == sorted(chunk_ids) == list(range(expected_branches))
        # Every sentence is covered exactly once.
        assert result["coverage"] == len(doc), result["coverage"]

    # Two different configs must yield a different number of branches.
    r1 = run_chunking(doc, 1)
    r3 = run_chunking(doc, 3)
    assert len(r1["annotations"]) != len(r3["annotations"]), "branch count fixed?"
    print("\n[Verification] PASS -- dynamic branch count, ordered, full coverage")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 6 MEDIUM Solutions -- LangGraph avance")
    print(f"  (real langgraph detected: {_HAS_LANGGRAPH}; running on stub)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

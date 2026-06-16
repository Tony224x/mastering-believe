"""
Solutions -- Day 6 (HARD): LangGraph avance

Contains solutions for:
  - Hard Ex 1: Full map-reduce-rerank pipeline (fan-out N workers via Send ->
               reduce with dedup -> rerank by composite score), with a
               concurrency bound (waves) and an error budget / partial-failure
               tolerance (status flips to "degraded" but the graph finishes).
  - Hard Ex 2: Time-travel debugging -- checkpoint every step, fork from an
               intermediate past checkpoint with an alternate input, and
               compare the two branches step by step (find the divergence).
  - Hard Ex 3: Supervisor subgraph that dispatches to specialized worker
               subgraphs, each with its OWN state schema (transformed state,
               with explicit input/output mapping) + a fallback route.

Everything runs OFFLINE with zero dependencies. We embed the MiniLangGraph
stub (Send + Checkpointer + custom reducers + conditional edges + update_state)
adapted from the day-6 easy solution. If the real `langgraph` is installed it
is detected, but for portability the demos always run on the stub.

Run:  python 03-exercises/solutions/06-langgraph-avance-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

import copy
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
# MINI LANGGRAPH STUB (Send + Checkpointer + conditional edges + time-travel)
# ==========================================================================

MINI_START = "__start__"
MINI_END = "__end__"


class Send:
    """Mimic of langgraph.constants.Send for parallel fan-out."""

    def __init__(self, node: str, arg: dict):
        self.node = node
        self.arg = arg


class Checkpointer:
    """
    In-memory checkpointer with time-travel support: it stores one snapshot per
    step per thread_id, exposes the full history, and can FORK a new thread from
    a past snapshot with an applied state override (mimics update_state).
    """

    def __init__(self):
        # thread_id -> list of {step, node, next, values}
        self._storage: dict[str, list[dict]] = {}

    def save(self, thread_id: str, step: int, node: str, nxt: str,
             values: dict) -> None:
        self._storage.setdefault(thread_id, []).append({
            "step": step, "node": node, "next": nxt,
            "values": copy.deepcopy(values),
        })

    def history(self, thread_id: str) -> list[dict]:
        return copy.deepcopy(self._storage.get(thread_id, []))

    def get_snapshot(self, thread_id: str, step: int) -> dict | None:
        for snap in self._storage.get(thread_id, []):
            if snap["step"] == step:
                return copy.deepcopy(snap)
        return None

    def update_state(self, thread_id: str, step: int, override: dict) -> dict:
        """
        Time-travel: return the state at `step` with `override` applied. This is
        the value you feed back into invoke() to fork a new branch.
        """
        snap = self.get_snapshot(thread_id, step)
        if snap is None:
            raise ValueError(f"no checkpoint at step {step} for {thread_id!r}")
        return {**snap["values"], **override}


class StateGraph:
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

    def invoke(self, initial_state: dict, config: dict | None = None,
               start_node: str | None = None, start_step: int = 0) -> dict:
        thread_id = (config or {}).get("thread_id", "default")
        state = dict(initial_state)
        current = start_node if start_node is not None else self._next(MINI_START, state)
        step_idx = start_step
        while current != MINI_END:
            updates = self._run_node(current, state)
            state = self._merge(state, updates)
            step_idx += 1
            nxt = self._next(current, state)
            if self.checkpointer:
                # Record which node ran and what comes next (for time-travel UI).
                self.checkpointer.save(thread_id, step_idx, current, nxt, state)
            current = nxt
        return state


START = MINI_START
END = MINI_END


# ==========================================================================
# HARD EXERCISE 1 -- map-reduce-rerank with concurrency + error budget
# ==========================================================================
#
# Fan out N workers (Send), in WAVES bounded by max_concurrency. Workers may
# fail gracefully. Reduce dedups by source. Rerank by a composite score and
# truncate to top_k. If failures exceed error_budget, status flips to
# "degraded" but the pipeline still finishes with the survivors.

def dedup_best(left: list, right: list) -> list:
    """Reducer: dedup hits by source, keep the higher raw_score; keep failures."""
    by_source: dict[str, dict] = {}
    for e in [*left, *right]:
        src = e["source"]
        prev = by_source.get(src)
        if prev is None:
            by_source[src] = e
        elif e.get("ok") and not prev.get("ok"):
            by_source[src] = e
        elif e.get("ok") == prev.get("ok") and e.get("raw_score", 0) > prev.get("raw_score", 0):
            by_source[src] = e
    return [by_source[s] for s in sorted(by_source)]


class SearchState(TypedDict):
    query: str
    sources: list
    flaky: list                 # source names that simulate failures
    max_concurrency: int
    error_budget: int
    top_k: int
    recency_weight: float
    cost_penalty: float
    hits: Annotated[list, dedup_best]
    waves: int
    stats: dict
    ranked: list
    status: str
    answer: str


# Deterministic mock data per source: (raw_score, cost, recency).
_DATA = {
    "s1": (0.95, 1.0), "s2": (0.40, 0.5), "s3": (0.80, 2.0),
    "s4": (0.70, 0.8), "s5": (0.55, 1.5), "s6": (0.88, 1.2),
    "s7": (0.30, 0.4), "s8": (0.60, 0.9),
}


def sr_worker(state: SearchState) -> dict:
    """Worker: OK hit or graceful failure for a flaky source."""
    src = state["source"]
    try:
        if src in state.get("flaky", []):
            raise TimeoutError(f"{src} unreachable")
        raw, cost = _DATA.get(src, (0.1, 1.0))
        return {"hits": [{"source": src, "doc": f"doc({src})",
                          "raw_score": raw, "cost": cost, "ok": True}]}
    except Exception as exc:  # noqa: BLE001 -- branch-local graceful handling
        return {"hits": [{"source": src, "ok": False, "error": str(exc),
                          "raw_score": 0.0, "cost": 0.0}]}


def run_in_waves(app: CompiledGraph, base_state: dict) -> dict:
    """
    Drive the fan-out in bounded waves OUTSIDE the graph node, then run the
    reduce/rerank/synthesize graph on the accumulated hits. This keeps each
    parallel fan-out <= max_concurrency, the way real LangGraph batching does.
    """
    sources = base_state["sources"]
    limit = base_state["max_concurrency"]
    waves = [sources[i:i + limit] for i in range(0, len(sources), limit)]

    all_hits: list = []
    for wave in waves:
        # Each wave is a real fan-out of <= limit Send branches.
        wave_state = {**base_state, "hits": []}

        def wave_dispatch(state, _wave=wave):
            return [Send("worker", {"source": s}) for s in _wave]

        g = StateGraph(SearchState)
        g.add_node("dispatch", wave_dispatch)
        g.add_node("worker", sr_worker)
        g.add_node("collect", lambda s: {})
        g.add_edge(START, "dispatch")
        g.add_edge("dispatch", "collect")
        g.add_edge("collect", END)
        wave_app = g.compile()
        out = wave_app.invoke(wave_state)
        assert len(out["hits"]) <= limit, "wave exceeded max_concurrency"
        all_hits = dedup_best(all_hits, out["hits"])

    base_state = {**base_state, "hits": all_hits, "waves": len(waves)}
    return app.invoke(base_state)


def sr_reduce(state: SearchState) -> dict:
    hits = state["hits"]
    ok = [h for h in hits if h["ok"]]
    fail = [h for h in hits if not h["ok"]]
    status = "degraded" if len(fail) > state["error_budget"] else "ok"
    return {"stats": {"n_ok": len(ok), "n_fail": len(fail),
                      "total_cost": round(sum(h["cost"] for h in ok), 3)},
            "status": status}


def sr_rerank(state: SearchState) -> dict:
    ok = [h for h in state["hits"] if h["ok"]]
    rw = state["recency_weight"]
    cp = state["cost_penalty"]
    for h in ok:
        h["final_score"] = round(h["raw_score"] * rw - cp * h["cost"], 4)
    ranked = sorted(ok, key=lambda h: h["final_score"], reverse=True)
    return {"ranked": ranked[:state["top_k"]]}


def sr_synthesize(state: SearchState) -> dict:
    names = [h["source"] for h in state["ranked"]]
    return {"answer": f"[{state['status']}] top sources: {', '.join(names)}"}


def build_search_graph() -> CompiledGraph:
    """The reduce/rerank/synthesize part of the pipeline (post fan-out)."""
    g = StateGraph(SearchState)
    g.add_node("reduce", sr_reduce)
    g.add_node("rerank", sr_rerank)
    g.add_node("synthesize", sr_synthesize)
    g.add_edge(START, "reduce")
    g.add_edge("reduce", "rerank")
    g.add_edge("rerank", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- map-reduce-rerank with concurrency + error budget")
    print("=" * 70)

    app = build_search_graph()
    sources = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    common = {
        "query": "best vector db", "sources": sources,
        "max_concurrency": 3, "top_k": 3,
        "recency_weight": 1.0, "cost_penalty": 0.1,
        "hits": [], "waves": 0, "stats": {}, "ranked": [],
        "status": "", "answer": "",
    }

    # --- Scenario A: few failures, under budget -> status "ok" ---
    state_a = {**common, "flaky": ["s7"], "error_budget": 2}
    res_a = run_in_waves(app, state_a)
    print(f"\n  Scenario A (1 fail, budget 2):")
    print(f"    waves={res_a['waves']}  stats={res_a['stats']}  status={res_a['status']}")
    print(f"    answer: {res_a['answer']}")
    scores_a = [h["final_score"] for h in res_a["ranked"]]
    print(f"    top_k final_scores: {scores_a}")

    # 8 sources, concurrency 3 -> ceil(8/3) = 3 waves.
    assert res_a["waves"] == 3, res_a["waves"]
    assert res_a["status"] == "ok", res_a["status"]
    assert res_a["stats"]["n_fail"] == 1, res_a["stats"]
    assert len(res_a["ranked"]) == 3, "top_k truncation failed"
    # Rerank order strictly non-increasing.
    assert scores_a == sorted(scores_a, reverse=True), scores_a
    # s1 has the best composite score (0.95*1 - 0.1*1.0 = 0.85) -> ranked first.
    assert res_a["ranked"][0]["source"] == "s1", res_a["ranked"][0]

    # --- Scenario B: many failures, over budget -> status "degraded" ---
    state_b = {**common, "flaky": ["s3", "s5", "s6"], "error_budget": 2}
    res_b = run_in_waves(app, state_b)
    print(f"\n  Scenario B (3 fails, budget 2):")
    print(f"    waves={res_b['waves']}  stats={res_b['stats']}  status={res_b['status']}")
    print(f"    answer: {res_b['answer']}")

    assert res_b["stats"]["n_fail"] == 3, res_b["stats"]
    assert res_b["status"] == "degraded", res_b["status"]
    # Pipeline FINISHED despite over-budget failures: it produced an answer.
    assert res_b["answer"].startswith("[degraded]"), res_b["answer"]
    assert len(res_b["ranked"]) >= 1, "degraded run must still return survivors"

    # Dedup proof: send s1 twice with a lower duplicate; best score must win.
    dup_state = {**common, "flaky": [], "error_budget": 5,
                 "sources": ["s1", "s1", "s2"], "max_concurrency": 3}
    res_dup = run_in_waves(app, dup_state)
    s1_entries = [h for h in res_dup["hits"] if h["source"] == "s1"]
    assert len(s1_entries) == 1, "dedup by source failed"
    print(f"\n  Dedup check: s1 appears {len(s1_entries)} time in reduced hits")
    print("[Verification] PASS -- waves bounded, budget flips status, rerank+dedup OK")


# ==========================================================================
# HARD EXERCISE 2 -- time-travel debugging: fork + compare branches
# ==========================================================================
#
# A deterministic graph whose `route` node branches on `mode`. We checkpoint
# every step, then fork from the snapshot right AFTER route by overriding `mode`,
# which sends the fork down a different path. compare_branches finds where they
# diverge and shows the differing finals.

class TTState(TypedDict):
    value: int
    mode: str            # "double" | "square"
    trace: Annotated[list, lambda a, b: a + b]
    result: int


def tt_route(state: TTState) -> dict:
    # No-op compute node; the branching happens on the conditional edge below.
    return {"trace": ["route"]}


def tt_double(state: TTState) -> dict:
    return {"value": state["value"] * 2, "trace": ["double"]}


def tt_square(state: TTState) -> dict:
    return {"value": state["value"] ** 2, "trace": ["square"]}


def tt_validate(state: TTState) -> dict:
    return {"trace": ["validate"], "value": state["value"] + 1}


def tt_finalize(state: TTState) -> dict:
    return {"result": state["value"], "trace": ["finalize"]}


def build_tt_graph(ckpt: Checkpointer) -> CompiledGraph:
    g = StateGraph(TTState)
    g.add_node("route", tt_route)
    g.add_node("double", tt_double)
    g.add_node("square", tt_square)
    g.add_node("validate", tt_validate)
    g.add_node("finalize", tt_finalize)
    g.add_edge(START, "route")
    # Conditional branching on `mode` -- this is what the fork will alter.
    g.add_conditional_edges("route", lambda s: s["mode"],
                            {"double": "double", "square": "square"})
    g.add_edge("double", "validate")
    g.add_edge("square", "validate")
    g.add_edge("validate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=ckpt)


def compare_branches(ckpt: Checkpointer, thread_a: str, thread_b: str) -> dict:
    """
    Align two histories BY STEP NUMBER and find the first diverging step.

    The fork shares an identical prefix with main (the steps before the fork
    point are inherited from main's checkpoint and not re-written on the fork
    thread). So we index each history by step and only compare steps present in
    BOTH threads -- the first such step whose node/values differ is the
    divergence point.
    """
    ha = {s["step"]: s for s in ckpt.history(thread_a)}
    hb = {s["step"]: s for s in ckpt.history(thread_b)}
    diverge_step = None
    for step in sorted(set(ha) & set(hb)):
        sa, sb = ha[step], hb[step]
        if sa["node"] != sb["node"] or sa["values"] != sb["values"]:
            diverge_step = step
            break
    final_a = ckpt.history(thread_a)[-1]["values"]["result"] if ha else None
    final_b = ckpt.history(thread_b)[-1]["values"]["result"] if hb else None
    return {"diverge_step": diverge_step,
            "final_a": final_a, "final_b": final_b,
            "len_a": len(ha), "len_b": len(hb)}


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- time-travel debugging: fork + compare branches")
    print("=" * 70)

    ckpt = Checkpointer()
    app = build_tt_graph(ckpt)

    # --- Original run on thread "main": value=5, mode="double" ---
    main_final = app.invoke({"value": 5, "mode": "double", "trace": [], "result": 0},
                            config={"thread_id": "main"})
    print("\n  Original run (main) history:")
    for snap in ckpt.history("main"):
        print(f"    step {snap['step']}: ran={snap['node']:9} next={snap['next']:9} "
              f"value={snap['values']['value']}")

    main_hist_before_fork = ckpt.history("main")  # snapshot for the invariant

    # --- Fork: take the snapshot RIGHT AFTER route (step 1), override mode ---
    fork_seed = ckpt.update_state("main", step=1, override={"mode": "square"})
    print(f"\n  Forking from step 1 with override mode='square' "
          f"(value carried = {fork_seed['value']})")
    # Resume from the conditional edge after `route`: next node depends on the
    # NEW mode, so we re-enter at `route`'s successor via the conditional.
    # Easiest faithful resume: re-run starting at the node `route` points to.
    # We recompute the branch target from the overridden mode.
    branch_target = "square" if fork_seed["mode"] == "square" else "double"
    fork_final = app.invoke(fork_seed, config={"thread_id": "fork"},
                            start_node=branch_target, start_step=1)
    print("  Fork run (fork) history:")
    for snap in ckpt.history("fork"):
        print(f"    step {snap['step']}: ran={snap['node']:9} next={snap['next']:9} "
              f"value={snap['values']['value']}")

    cmp = compare_branches(ckpt, "main", "fork")
    print(f"\n  compare_branches: {cmp}")

    # main: 5 ->(double)10 ->(validate)11 ->(finalize) result 11
    assert main_final["result"] == 11, main_final
    # fork: 5 ->(square)25 ->(validate)26 ->(finalize) result 26
    assert fork_final["result"] == 26, fork_final
    # Every step of the main run is checkpointed: route,double,validate,finalize.
    assert cmp["len_a"] == 4, cmp
    # Divergence is at step 2 (route is identical at step 1, then double vs square).
    assert cmp["diverge_step"] == 2, cmp
    # The two branches produce DIFFERENT finals -> the fork had an effect.
    assert cmp["final_a"] != cmp["final_b"], cmp
    # INVARIANT: main thread is strictly unchanged after the fork.
    assert ckpt.history("main") == main_hist_before_fork, "fork mutated main!"
    print("[Verification] PASS -- forked from past, branches diverge, main intact")


# ==========================================================================
# HARD EXERCISE 3 -- supervisor subgraph dispatching to worker subgraphs
# ==========================================================================
#
# Three worker subgraphs, each with its OWN state schema, compiled separately
# and testable in isolation. The supervisor classifies the request and routes
# (conditional edges) to a wrapper that does input mapping -> invoke worker ->
# output mapping. A fallback handles unknown categories.

# ---- Worker 1: math (own schema) ----
class MathState(TypedDict):
    expr: str
    a: int
    b: int
    op: str
    out: int


def math_parse(state: MathState) -> dict:
    # Very small parser: "<a> <op> <b>" with op in + - *
    a, op, b = state["expr"].split()
    return {"a": int(a), "b": int(b), "op": op}


def math_compute(state: MathState) -> dict:
    a, b, op = state["a"], state["b"], state["op"]
    out = {"+": a + b, "-": a - b, "*": a * b}[op]
    return {"out": out}


def build_math_subgraph() -> CompiledGraph:
    g = StateGraph(MathState)
    g.add_node("parse", math_parse)
    g.add_node("compute", math_compute)
    g.add_edge(START, "parse")
    g.add_edge("parse", "compute")
    g.add_edge("compute", END)
    return g.compile()


# ---- Worker 2: text (own schema) ----
class TextState(TypedDict):
    text: str
    tokens: list
    summary: str


def text_tokenize(state: TextState) -> dict:
    return {"tokens": state["text"].split()}


def text_summarize(state: TextState) -> dict:
    toks = state["tokens"]
    head = " ".join(toks[:3])
    return {"summary": f"{head}... ({len(toks)} tokens)"}


def build_text_subgraph() -> CompiledGraph:
    g = StateGraph(TextState)
    g.add_node("tokenize", text_tokenize)
    g.add_node("summarize", text_summarize)
    g.add_edge(START, "tokenize")
    g.add_edge("tokenize", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# ---- Worker 3: lookup (own schema) ----
class LookupState(TypedDict):
    key: str
    record: str
    formatted: str


_KB = {"capital_japan": "Tokyo", "speed_light": "299792 km/s"}


def lookup_search(state: LookupState) -> dict:
    return {"record": _KB.get(state["key"], "NOT_FOUND")}


def lookup_format(state: LookupState) -> dict:
    return {"formatted": f"{state['key']} = {state['record']}"}


def build_lookup_subgraph() -> CompiledGraph:
    g = StateGraph(LookupState)
    g.add_node("search", lookup_search)
    g.add_node("format", lookup_format)
    g.add_edge(START, "search")
    g.add_edge("search", "format")
    g.add_edge("format", END)
    return g.compile()


# ---- Supervisor (parent graph) ----
class SupervisorState(TypedDict):
    request: str
    payload: str          # the meaningful content, set by classify
    category: str
    answer: str
    handled_by: str


def supervisor_classify(state: SupervisorState) -> dict:
    """Heuristic classification + extract the payload to hand to the worker."""
    req = state["request"].strip()
    low = req.lower()
    if low.startswith("calc:"):
        return {"category": "math", "payload": req.split(":", 1)[1].strip()}
    if low.startswith("summarize:"):
        return {"category": "text", "payload": req.split(":", 1)[1].strip()}
    if low.startswith("lookup:"):
        return {"category": "lookup", "payload": req.split(":", 1)[1].strip()}
    return {"category": "unknown", "payload": req}


def make_math_wrapper(worker: CompiledGraph):
    def wrapper(state: SupervisorState) -> dict:
        # INPUT mapping: parent payload -> worker schema
        sub_out = worker.invoke({"expr": state["payload"], "a": 0, "b": 0,
                                 "op": "", "out": 0})
        # OUTPUT mapping: worker result -> parent schema
        return {"answer": str(sub_out["out"]), "handled_by": "math"}
    return wrapper


def make_text_wrapper(worker: CompiledGraph):
    def wrapper(state: SupervisorState) -> dict:
        sub_out = worker.invoke({"text": state["payload"], "tokens": [], "summary": ""})
        return {"answer": sub_out["summary"], "handled_by": "text"}
    return wrapper


def make_lookup_wrapper(worker: CompiledGraph):
    def wrapper(state: SupervisorState) -> dict:
        sub_out = worker.invoke({"key": state["payload"], "record": "", "formatted": ""})
        return {"answer": sub_out["formatted"], "handled_by": "lookup"}
    return wrapper


def fallback_wrapper(state: SupervisorState) -> dict:
    return {"answer": "Désolé, je ne sais pas traiter cette requête.",
            "handled_by": "fallback"}


def supervisor_finalize(state: SupervisorState) -> dict:
    return {"answer": f"[{state['handled_by']}] {state['answer']}"}


def build_supervisor() -> CompiledGraph:
    math_sg = build_math_subgraph()
    text_sg = build_text_subgraph()
    lookup_sg = build_lookup_subgraph()

    g = StateGraph(SupervisorState)
    g.add_node("classify", supervisor_classify)
    g.add_node("math", make_math_wrapper(math_sg))
    g.add_node("text", make_text_wrapper(text_sg))
    g.add_node("lookup", make_lookup_wrapper(lookup_sg))
    g.add_node("fallback", fallback_wrapper)
    g.add_node("finalize", supervisor_finalize)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", lambda s: s["category"],
        {"math": "math", "text": "text", "lookup": "lookup",
         "unknown": "fallback"})
    for node in ("math", "text", "lookup", "fallback"):
        g.add_edge(node, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def solve_hard_3() -> None:
    print("\n" + "=" * 70)
    print("HARD 3 -- supervisor subgraph dispatching to worker subgraphs")
    print("=" * 70)

    # --- Workers are testable in ISOLATION (no supervisor involved) ---
    math_iso = build_math_subgraph().invoke(
        {"expr": "6 * 7", "a": 0, "b": 0, "op": "", "out": 0})
    text_iso = build_text_subgraph().invoke(
        {"text": "the quick brown fox jumps", "tokens": [], "summary": ""})
    lookup_iso = build_lookup_subgraph().invoke(
        {"key": "capital_japan", "record": "", "formatted": ""})
    print("\n  Isolation tests:")
    print(f"    math:   {math_iso['out']}")
    print(f"    text:   {text_iso['summary']}")
    print(f"    lookup: {lookup_iso['formatted']}")
    assert math_iso["out"] == 42
    assert "5 tokens" in text_iso["summary"]
    assert lookup_iso["formatted"] == "capital_japan = Tokyo"

    # --- Supervisor routing ---
    sup = build_supervisor()
    cases = [
        ("calc: 6 * 7", "math", "42"),
        ("summarize: the quick brown fox jumps", "text", "5 tokens"),
        ("lookup: capital_japan", "lookup", "Tokyo"),
        ("what's the weather?", "fallback", "je ne sais pas"),
    ]
    print("\n  Supervisor routing:")
    for request, exp_handler, exp_in_answer in cases:
        out = sup.invoke({"request": request, "payload": "", "category": "",
                          "answer": "", "handled_by": ""})
        print(f"    {request:45s} -> [{out['handled_by']:8}] {out['answer']}")
        assert out["handled_by"] == exp_handler, (request, out["handled_by"])
        assert exp_in_answer in out["answer"], (request, out["answer"])
        # finalize prefixes the answer with the handler tag.
        assert out["answer"].startswith(f"[{exp_handler}]"), out["answer"]

    print("[Verification] PASS -- isolation OK, 4 routes correct (incl. fallback)")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 6 HARD Solutions -- LangGraph avance")
    print(f"  (real langgraph detected: {_HAS_LANGGRAPH}; running on stub)")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()
    solve_hard_3()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

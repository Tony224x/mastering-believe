"""
Day 7 -- Capstone: a complete research agent

Combines everything from week 1:
  - Tools (J2): mock_web_search, read_doc, summarize
  - Memory (J3): short-term scratchpad + long-term knowledge store
  - Planning (J4): planner + executor + synthesizer
  - LangGraph (J5-J6): StateGraph with nodes and conditional routing

The agent:
  1. Receives a question
  2. PLANNER decomposes it into steps
  3. EXECUTOR runs each step, consulting long-term memory first
  4. ANALYZER extracts clean facts from raw tool outputs
  5. SYNTHESIZER produces the final, sourced answer

Works without any external dependency. Uses a MockLLM and the MiniLangGraph
stub for full portability.

Run:
    python 02-code/07-agent-complet.py
"""

import re
from operator import add
from typing import Annotated, Callable, TypedDict


# ===========================================================================
# MINI LANGGRAPH STUB -- minimal version with Send + conditional edges
# ===========================================================================

MINI_START = "__start__"
MINI_END = "__end__"


class StateGraph:
    def __init__(self, state_schema: type):
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

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def _merge(self, state: dict, updates: dict) -> dict:
        new_state = dict(state)
        for k, v in updates.items():
            reducer = self.graph._reducers.get(k)
            if reducer is not None and k in new_state:
                new_state[k] = reducer(new_state[k], v)
            else:
                new_state[k] = v
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


START = MINI_START
END = MINI_END


# ===========================================================================
# TOOLS -- three mock tools the agent can call
# ===========================================================================

# Pre-populated "web index" -- simulates a search engine
SEARCH_INDEX = {
    "africa area": "Africa covers approximately 30,370,000 km2 of land area.",
    "africa population": "Africa's population is about 1,460,000,000 inhabitants in 2024.",
    "paris area": "Paris has an area of 105 km2.",
    "paris population": "Paris has 2,161,000 inhabitants in 2024.",
}

# Pre-populated "document store" -- simulates a PDF reader
DOCS = {
    "africa_report_2024.pdf": (
        "FULL REPORT: The African continent is the second largest by area "
        "at 30,370,000 km2 and by population at 1,460,000,000 inhabitants."
    ),
    "paris_stats.pdf": (
        "FULL STATS: Paris has 2,161,000 inhabitants spread across 105 km2, "
        "giving a density of approximately 20,581 hab/km2."
    ),
}


def mock_web_search(query: str) -> str:
    """Simple keyword-based mock search. Returns a snippet or a 'no result'."""
    q = query.lower()
    for key, snippet in SEARCH_INDEX.items():
        if all(word in q for word in key.split()):
            return snippet
    return f"NO_RESULT for query: {query}"


def read_doc(doc_name: str) -> str:
    """Fake PDF reader. Returns the canned text for a doc name."""
    return DOCS.get(doc_name, f"NO_SUCH_DOC: {doc_name}")


def summarize(text: str) -> str:
    """
    Extract the first numeric-with-unit value from a text.
    In a real system this would be an LLM call; here we use regex for
    determinism and zero cost.
    """
    m = re.search(
        r"([\d,]+(?:\.\d+)?)\s*(km2|inhabitants|hab/km2|%)",
        text,
    )
    if m:
        return f"KEY_FACT: {m.group(1).replace(',', '')} {m.group(2)}"
    return "KEY_FACT: none"


# ===========================================================================
# STATE
# ===========================================================================

class AgentState(TypedDict):
    # Input
    question: str
    # Plan
    plan: list
    current_step: int
    # Memories
    short_term: dict                            # scratchpad for this run
    long_term: Annotated[list, add]             # persistent facts (accumulates)
    # Tool outputs
    findings: Annotated[list, add]              # raw tool results
    # Output
    final_answer: str
    # Control flow
    stuck: bool


# ===========================================================================
# PLANNER NODE
# ===========================================================================

def planner_node(state: AgentState) -> dict:
    """
    Decompose the question into concrete steps. In a real agent this would
    be an LLM call; here we hardcode a simple rule-based decomposition.
    """
    q = state["question"].lower()
    steps: list[str] = []

    # Detect "density" / "ratio" questions
    if "density" in q or "densite" in q or "ratio" in q:
        if "africa" in q or "afrique" in q:
            steps = [
                "search:africa area",
                "search:africa population",
                "compute:density",
                "format:final_answer",
            ]
        elif "paris" in q:
            steps = [
                "search:paris area",
                "search:paris population",
                "compute:density",
                "format:final_answer",
            ]
        else:
            steps = ["search:generic", "compute:density", "format:final_answer"]
    else:
        # Simple info questions: one search + format
        if "africa" in q or "afrique" in q:
            steps = ["search:africa population", "format:final_answer"]
        else:
            steps = ["search:paris population", "format:final_answer"]

    print(f"[PLANNER] Decomposed question into {len(steps)} steps:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    return {"plan": steps, "current_step": 0}


# ===========================================================================
# EXECUTOR NODE
# ===========================================================================

def check_long_term(long_term: list, query: str) -> str | None:
    """
    Look for a cached fact that matches the query. Returns the cached
    snippet or None if not found.
    """
    q = query.lower()
    for entry in long_term:
        if all(word in entry["fact"].lower() for word in q.split() if word):
            return entry["fact"]
    return None


def executor_node(state: AgentState) -> dict:
    """
    Execute the current step. Consults long-term memory first (cache-first),
    otherwise calls the appropriate tool.
    """
    idx = state["current_step"]
    plan = state["plan"]

    if idx >= len(plan):
        return {"stuck": False, "current_step": idx}

    step = plan[idx]
    print(f"\n[EXECUTOR] Step {idx + 1}/{len(plan)}: {step}")

    # Parse the step instruction
    action, _, arg = step.partition(":")

    if action == "search":
        # 1. Check long-term memory first
        cached = check_long_term(state["long_term"], arg)
        if cached:
            print(f"  [cache HIT] using long-term fact: {cached[:60]}...")
            return {
                "findings": [cached],
                "current_step": idx + 1,
                "stuck": False,
            }

        # 2. Call the web search tool
        result = mock_web_search(arg)
        if result.startswith("NO_RESULT"):
            print(f"  [tool FAIL] {result}")
            # 3. Fallback: try the doc reader
            doc_name = f"{arg.split()[0]}_report_2024.pdf"
            result = read_doc(doc_name)
            if result.startswith("NO_SUCH_DOC"):
                print(f"  [fallback FAIL] agent is stuck")
                return {"stuck": True, "current_step": idx}
            print(f"  [fallback OK] read_doc({doc_name})")
        else:
            print(f"  [tool OK] web_search({arg}): {result[:60]}...")

        # Store the fact in long-term for future runs
        new_long_term = [{"fact": result, "source": "search", "confidence": 0.9}]
        return {
            "findings": [result],
            "long_term": new_long_term,
            "current_step": idx + 1,
            "stuck": False,
        }

    if action == "compute":
        # Compute a derived value. At this point the analyzer has NOT run
        # yet (it runs after all executor steps), so for derived values that
        # depend on clean facts we just mark the step as "deferred" and let
        # the synthesizer produce it after the analyzer has populated
        # short_term. This keeps the architecture clean: executor = raw
        # calls, analyzer = fact extraction, synthesizer = final derivation.
        print(f"  [compute] {arg} (deferred to synthesizer)")
        return {"current_step": idx + 1, "stuck": False}

    if action == "format":
        # Let the synthesizer node handle this; just advance the pointer
        print(f"  [format] deferred to synthesizer")
        return {"current_step": idx + 1, "stuck": False}

    print(f"  [UNKNOWN ACTION] {action}")
    return {"stuck": True, "current_step": idx}


# ===========================================================================
# ANALYZER NODE
# ===========================================================================

def analyzer_node(state: AgentState) -> dict:
    """
    Extract clean facts from raw findings and update short_term memory.
    This is where noisy text becomes structured data.
    """
    print("\n[ANALYZER] Extracting facts from findings...")
    short_term = dict(state["short_term"])

    for finding in state["findings"]:
        summary = summarize(finding)
        print(f"  finding: {finding[:60]}...")
        print(f"    -> {summary}")

        # Parse the KEY_FACT into typed short_term entries
        m = re.search(r"KEY_FACT:\s*([\d.]+)\s*(\w+)", summary)
        if not m:
            continue
        value = float(m.group(1))
        unit = m.group(2)

        if unit == "km2":
            short_term["area_km2"] = int(value)
        elif unit == "inhabitants":
            short_term["population"] = int(value)

    print(f"  short_term after analysis: {short_term}")
    return {"short_term": short_term}


# ===========================================================================
# SYNTHESIZER NODE
# ===========================================================================

def synthesizer_node(state: AgentState) -> dict:
    """
    Produce the final answer from short_term memory. Cites the tools used.
    If the question asked for a density, compute it here from the facts
    the analyzer just put into short_term memory.
    """
    print("\n[SYNTHESIZER] Building final answer...")
    st = dict(state["short_term"])
    question = state["question"]

    # If the question is about density and we have area+pop, compute now
    if ("density" in question.lower() or "densite" in question.lower()) \
            and "area_km2" in st and "population" in st:
        if st["area_km2"] > 0:
            st["density"] = round(st["population"] / st["area_km2"], 2)

    parts = []
    if "area_km2" in st:
        parts.append(f"area={st['area_km2']:,} km2")
    if "population" in st:
        parts.append(f"population={st['population']:,} inhabitants")
    if "density" in st:
        parts.append(f"density~{st['density']} hab/km2")

    if not parts:
        answer = ("I could not find enough information to answer the question: "
                  f"'{question}'. Please rephrase or provide more context.")
    else:
        sources = len(state["findings"])
        answer = (f"Answer to '{question}':\n"
                  f"  " + ", ".join(parts) + "\n"
                  f"  (derived from {sources} source(s))")

    print(f"  {answer}")
    return {"final_answer": answer}


# ===========================================================================
# ROUTING
# ===========================================================================

def route_from_executor(state: AgentState) -> str:
    """
    After executor: continue executing, go to analyzer, or give up if stuck.
    """
    if state.get("stuck"):
        return "end"
    if state["current_step"] >= len(state["plan"]):
        return "analyzer"
    return "executor"  # loop back for the next step


# ===========================================================================
# BUILD AND RUN
# ===========================================================================

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        route_from_executor,
        {"executor": "executor", "analyzer": "analyzer", "end": END},
    )
    graph.add_edge("analyzer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_sample_question(question: str) -> dict:
    print("\n" + "#" * 70)
    print(f"# QUESTION: {question}")
    print("#" * 70)

    app = build_agent()
    initial_state = {
        "question": question,
        "plan": [],
        "current_step": 0,
        "short_term": {},
        "long_term": [],
        "findings": [],
        "final_answer": "",
        "stuck": False,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    # Question 1: density of Africa
    result = run_sample_question("What is the population density of Africa?")
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(f"  short_term:   {result['short_term']}")
    print(f"  long_term:    {len(result['long_term'])} facts stored")
    print(f"  findings:     {len(result['findings'])} raw results")
    print(f"  final_answer: {result['final_answer']}")

    # Question 2: density of Paris (should re-use or fall back)
    result2 = run_sample_question("What is the population density of Paris?")
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(f"  short_term:   {result2['short_term']}")
    print(f"  final_answer: {result2['final_answer']}")

    print("\n" + "=" * 70)
    print("Capstone agent: DONE.")
    print("=" * 70)

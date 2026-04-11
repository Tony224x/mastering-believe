"""
Solutions -- Day 7: Capstone research agent

Contains:
  - Easy Ex 1: Add a 4th tool (compare_tool) to compare two densities
  - Easy Ex 2: Template-based planner with multiple question patterns
  - Easy Ex 3: Real-LLM planner with deterministic fallback

Each solution is self-contained. They share the MiniLangGraph stub and the
mock tools from the day-7 capstone.

Run:  python 03-exercises/solutions/07-agent-complet.py
"""

import os
import re
from operator import add
from typing import Annotated, Callable, TypedDict


# ==========================================================================
# MINI LANGGRAPH STUB
# ==========================================================================

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

    def compile(self):
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
        raise RuntimeError("did not terminate")


START = MINI_START
END = MINI_END


# ==========================================================================
# SHARED TOOLS AND DATA
# ==========================================================================

SEARCH_INDEX = {
    "africa area": "Africa covers approximately 30,370,000 km2 of land area.",
    "africa population": "Africa has about 1,460,000,000 inhabitants in 2024.",
    "paris area": "Paris has an area of 105 km2.",
    "paris population": "Paris has 2,161,000 inhabitants in 2024.",
}


def mock_web_search(query: str) -> str:
    q = query.lower()
    for key, snippet in SEARCH_INDEX.items():
        if all(word in q for word in key.split()):
            return snippet
    return f"NO_RESULT for query: {query}"


def summarize(text: str) -> str:
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(km2|inhabitants)", text)
    if m:
        return f"KEY_FACT: {m.group(1).replace(',', '')} {m.group(2)}"
    return "KEY_FACT: none"


class AgentState(TypedDict):
    question: str
    plan: list
    current_step: int
    short_term: dict
    entities: dict  # entity name -> short_term key prefix (ex1, ex2)
    long_term: Annotated[list, add]
    findings: Annotated[list, add]
    final_answer: str
    stuck: bool


# ==========================================================================
# EASY EXERCISE 1 -- Add a compare tool
# ==========================================================================

def compare_tool(a: float, b: float) -> str:
    """Compare two numbers and return verdict + percentage delta."""
    if a == 0 and b == 0:
        return "both zero"
    if a == b:
        return "a == b (identical)"
    larger, smaller = (a, b) if a > b else (b, a)
    pct = (larger - smaller) / smaller * 100 if smaller > 0 else float("inf")
    verdict = "a > b" if a > b else "a < b"
    return f"{verdict} (delta = {pct:.1f}%)"


def solve_ex1() -> None:
    print("\n" + "=" * 70)
    print("EX1 -- Compare tool")
    print("=" * 70)

    # Planner that handles the "compare" question
    def planner(state: AgentState) -> dict:
        q = state["question"].lower()
        if "compare" in q and "africa" in q and "paris" in q:
            plan = [
                "search:africa area",
                "search:africa population",
                "compute:density_africa",
                "search:paris area",
                "search:paris population",
                "compute:density_paris",
                "compare:density_africa_vs_density_paris",
                "format:final_answer",
            ]
        else:
            plan = ["search:africa area", "format:final_answer"]
        print(f"[PLANNER] {len(plan)} steps")
        for i, s in enumerate(plan, 1):
            print(f"  {i}. {s}")
        return {"plan": plan, "current_step": 0}

    def executor(state: AgentState) -> dict:
        idx = state["current_step"]
        if idx >= len(state["plan"]):
            return {"current_step": idx}
        step = state["plan"][idx]
        action, _, arg = step.partition(":")
        print(f"\n[EXECUTOR] {step}")
        short_term = dict(state["short_term"])

        if action == "search":
            result = mock_web_search(arg)
            # Analyze right away and store by entity prefix
            summary = summarize(result)
            m = re.search(r"KEY_FACT:\s*([\d.]+)\s*(\w+)", summary)
            if m:
                entity = arg.split()[0]  # "africa" or "paris"
                if m.group(2) == "km2":
                    short_term[f"{entity}_area"] = float(m.group(1))
                elif m.group(2) == "inhabitants":
                    short_term[f"{entity}_population"] = float(m.group(1))
            return {
                "findings": [result],
                "short_term": short_term,
                "current_step": idx + 1,
            }

        if action == "compute":
            if arg.startswith("density_"):
                entity = arg.split("_")[1]
                area = short_term.get(f"{entity}_area", 0)
                pop = short_term.get(f"{entity}_population", 0)
                if area > 0:
                    short_term[f"density_{entity}"] = round(pop / area, 2)
                    print(f"  density_{entity} = {short_term[f'density_{entity}']}")
            return {"short_term": short_term, "current_step": idx + 1}

        if action == "compare":
            a_key, _, b_key = arg.partition("_vs_")
            a_val = short_term.get(a_key, 0)
            b_val = short_term.get(b_key, 0)
            verdict = compare_tool(a_val, b_val)
            short_term[f"compare_{arg}"] = verdict
            print(f"  compare: {a_val} vs {b_val} -> {verdict}")
            return {"short_term": short_term, "current_step": idx + 1}

        return {"current_step": idx + 1}

    def synthesizer(state: AgentState) -> dict:
        st = state["short_term"]
        parts = []
        if "density_africa" in st:
            parts.append(f"Africa density: {st['density_africa']} hab/km2")
        if "density_paris" in st:
            parts.append(f"Paris density: {st['density_paris']} hab/km2")
        compare_keys = [k for k in st if k.startswith("compare_")]
        if compare_keys:
            parts.append(f"Verdict: {st[compare_keys[0]]}")
        answer = " | ".join(parts) if parts else "No comparison available."
        return {"final_answer": answer}

    def route(state: AgentState) -> str:
        if state["current_step"] >= len(state["plan"]):
            return "synth"
        return "executor"

    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("synth", synthesizer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", route,
                            {"executor": "executor", "synth": "synth"})
    g.add_edge("synth", END)
    app = g.compile()

    result = app.invoke({
        "question": "Compare the population density of Africa and Paris.",
        "plan": [], "current_step": 0,
        "short_term": {}, "entities": {},
        "long_term": [], "findings": [],
        "final_answer": "", "stuck": False,
    })
    print(f"\nFINAL ANSWER:\n  {result['final_answer']}")
    assert "Africa" in result["final_answer"]
    assert "Paris" in result["final_answer"]
    assert "Verdict" in result["final_answer"]
    print("\n[Verification] PASS")


# ==========================================================================
# EASY EXERCISE 2 -- Template-based planner
# ==========================================================================

PLAN_TEMPLATES = {
    "density_of_X": [
        "search:X area",
        "search:X population",
        "compute:density",
        "format:final_answer",
    ],
    "population_of_X": [
        "search:X population",
        "format:final_answer",
    ],
    "area_of_X": [
        "search:X area",
        "format:final_answer",
    ],
}


def detect_template(question: str) -> tuple[str, str] | None:
    """Return (template_name, entity) or None if no match."""
    q = question.lower()
    entity = None
    for candidate in ("africa", "paris"):
        if candidate in q:
            entity = candidate
            break
    if entity is None:
        return None

    if "density" in q or "densite" in q:
        return "density_of_X", entity
    if "area" in q or "superficie" in q:
        return "area_of_X", entity
    if "population" in q:
        return "population_of_X", entity
    return None


def solve_ex2() -> None:
    print("\n" + "=" * 70)
    print("EX2 -- Template-based planner")
    print("=" * 70)

    questions = [
        "What is the population density of Africa?",
        "What is the area of Africa?",
        "What is the population of Paris?",
    ]

    for q in questions:
        print(f"\n  question: {q}")
        match = detect_template(q)
        if match is None:
            print("    no template matched")
            continue
        template_name, entity = match
        template = PLAN_TEMPLATES[template_name]
        plan = [step.replace("X", entity) for step in template]
        print(f"    template:  {template_name}")
        print(f"    plan ({len(plan)} steps):")
        for i, step in enumerate(plan, 1):
            print(f"      {i}. {step}")

    # Verify that area_of_X has no compute step
    area_plan = [s.replace("X", "africa") for s in PLAN_TEMPLATES["area_of_X"]]
    assert not any(s.startswith("compute:") for s in area_plan)
    print("\n[Verification] PASS -- area_of_X has no compute step")


# ==========================================================================
# EASY EXERCISE 3 -- Real-LLM planner with fallback
# ==========================================================================

try:
    import anthropic  # type: ignore
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def deterministic_planner(question: str) -> list[str]:
    """The fallback planner from the main capstone code."""
    q = question.lower()
    if "density" in q and "africa" in q:
        return ["search:africa area", "search:africa population",
                "compute:density", "format:final_answer"]
    if "population" in q and "africa" in q:
        return ["search:africa population", "format:final_answer"]
    return ["search:africa area", "format:final_answer"]


def make_planner_llm() -> tuple[Callable[[str], list[str]], str]:
    """
    Return (planner_fn, mode) where mode is 'REAL' or 'MOCK'.

    If both anthropic is installed and ANTHROPIC_API_KEY is set, return a
    real-LLM planner. Otherwise return the deterministic fallback.
    """
    if _HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()

        def real_planner(question: str) -> list[str]:
            try:
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=512,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Decompose this question into 3-5 concrete steps. "
                            "Use the format 'action:target'. Actions available: "
                            "search, compute, format. One step per line.\n\n"
                            f"Question: {question}"
                        ),
                    }],
                )
                text = response.content[0].text  # type: ignore
                steps = [line.strip() for line in text.splitlines() if ":" in line]
                return steps or deterministic_planner(question)
            except Exception as exc:
                print(f"  [LLM FAIL] {exc} -- falling back to deterministic")
                return deterministic_planner(question)

        return real_planner, "REAL"

    return deterministic_planner, "MOCK"


def solve_ex3() -> None:
    print("\n" + "=" * 70)
    print("EX3 -- Real-LLM planner with fallback")
    print("=" * 70)

    planner_fn, mode = make_planner_llm()
    print(f"Planner mode: {mode}")

    question = "What is the population density of Africa?"
    plan = planner_fn(question)
    print(f"Plan produced ({len(plan)} steps):")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. {s}")

    assert len(plan) >= 1
    print("\n[Verification] PASS -- planner returns a non-empty plan")


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

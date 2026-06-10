"""
Solutions -- Day 7: Capstone research agent

Contains:
  - Easy Ex 1: Add a 4th tool (compare_tool) to compare two densities
  - Easy Ex 2: Template-based planner with multiple question patterns
  - Easy Ex 3: Real-LLM planner with deterministic fallback
  - Medium Ex 1: Long-term memory that short-circuits the plan
  - Medium Ex 2: Robust executor (retry, tool fallback, honest failure)
  - Medium Ex 3: Clarification node -- when to ask the user

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
# MEDIUM EXERCISE 1 -- Long-term memory that short-circuits the plan
# ==========================================================================

def normalize_question(question: str) -> str:
    return " ".join(sorted(re.findall(r"[a-z]+", question.lower())))


class MemoryAgent:
    """Plan-and-execute agent with a cross-run long-term store.

    The planner consults memory BEFORE planning:
      - full hit  -> single 'recall:final_answer' step
      - partial hit -> 'recall:<fact>' steps replace already-covered searches
    """

    def __init__(self):
        self.long_term: list[dict] = []
        self.stats: list[dict] = []

    # ---- memory helpers ----------------------------------------------
    def _find_entry(self, question: str) -> dict | None:
        key = normalize_question(question)
        for entry in self.long_term:
            if entry["question_pattern"] == key:
                return entry
        return None

    def _known_fact(self, fact_key: str) -> float | None:
        for entry in self.long_term:
            if fact_key in entry["facts"]:
                return entry["facts"][fact_key]
        return None

    # ---- planner ------------------------------------------------------
    def plan(self, question: str) -> tuple[list[str], dict]:
        q = question.lower()
        entry = self._find_entry(question)
        if entry is not None:                      # Full hit
            entry["hits"] += 1
            return ["recall:final_answer"], {"memory": "full_hit"}

        entity = "africa" if "africa" in q else "paris"
        if "density" in q:
            raw = [f"search:{entity} area", f"search:{entity} population",
                   "compute:density", "format:final_answer"]
        elif "population" in q:
            raw = [f"search:{entity} population", "format:final_answer"]
        else:
            raw = [f"search:{entity} area", "format:final_answer"]

        # Partial hit: swap searches whose fact is already known
        plan: list[str] = []
        hits = 0
        for step in raw:
            if step.startswith("search:"):
                fact_key = step.split(":", 1)[1].replace(" ", "_")
                if self._known_fact(fact_key) is not None:
                    plan.append(f"recall:{fact_key}")
                    hits += 1
                    continue
            plan.append(step)
        return plan, {"memory": f"partial_hit({hits})" if hits else "miss"}

    # ---- run ------------------------------------------------------------
    def run(self, question: str) -> str:
        plan, meta = self.plan(question)
        scratchpad: dict = {}
        tool_calls = 0
        recalls = 0

        for step in plan:
            action, _, arg = step.partition(":")
            if action == "recall":
                recalls += 1
                if arg == "final_answer":
                    answer = self._find_entry(question)["answer"]
                else:
                    scratchpad[arg] = self._known_fact(arg)
            elif action == "search":
                tool_calls += 1
                raw = mock_web_search(arg)
                m = re.search(r"([\d,]+)", raw)
                fact_key = arg.replace(" ", "_")
                scratchpad[fact_key] = float(m.group(1).replace(",", ""))
            elif action == "compute" and arg == "density":
                entity = "africa" if any("africa" in k for k in scratchpad) else "paris"
                area = scratchpad[f"{entity}_area"]
                pop = scratchpad[f"{entity}_population"]
                scratchpad["density"] = round(pop / area, 2)

        if plan != ["recall:final_answer"]:
            if "density" in scratchpad:
                answer = f"Density: {scratchpad['density']} hab/km2"
            else:
                key, val = next(iter(scratchpad.items()))
                answer = f"{key} = {val:,.0f}"
            # Persist the run outcome + intermediate facts for future runs
            self.long_term.append({
                "question_pattern": normalize_question(question),
                "answer": answer,
                "facts": dict(scratchpad),
                "hits": 0,
            })

        self.stats.append({"question": question, "steps": len(plan),
                           "tool_calls": tool_calls, "memory_hits": recalls,
                           "memory_mode": meta["memory"]})
        return answer


def solve_medium1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Long-term memory short-circuit")
    print("=" * 70)

    agent = MemoryAgent()
    q_density = "What is the population density of Africa?"
    q_pop = "What is the population of Africa?"

    a1 = agent.run(q_density)        # Run 1: cold, 2 searches
    a2 = agent.run(q_density)        # Run 2: full hit, 1 recall step
    a3 = agent.run(q_pop)            # Run 3: partial hit, 0 searches
    a4 = agent.run("What is the population of Paris?")   # Never seen: normal

    for s in agent.stats:
        print(f"  steps={s['steps']} tools={s['tool_calls']} "
              f"hits={s['memory_hits']} mode={s['memory_mode']:15} | {s['question']}")
    print(f"\n  run1: {a1}\n  run2: {a2}\n  run3: {a3}\n  run4: {a4}")

    assert agent.stats[1]["tool_calls"] == 0 and agent.stats[1]["steps"] == 1
    assert a1 == a2
    assert agent.stats[2]["tool_calls"] == 0          # Partial hit: fact reused
    assert agent.stats[2]["memory_mode"] == "partial_hit(1)"
    assert agent.stats[3]["tool_calls"] == 1          # No false hit on new question
    print("\n[Verification] PASS -- full hit, partial hit, no false hit")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Robust executor: retry, fallback, honest failure
# ==========================================================================

def make_flaky_search(fail_first_n: int):
    """Search tool that raises for the first N calls per query."""
    counts: dict[str, int] = {}

    def flaky(query: str) -> str:
        counts[query] = counts.get(query, 0) + 1
        if counts[query] <= fail_first_n:
            raise RuntimeError("search backend timeout")
        return mock_web_search(query)

    return flaky


DOC_LIBRARY = {
    "africa area": "Doc 'africa area': Africa covers 30,370,000 km2.",
}


def read_doc_fallback(query: str) -> str:
    doc = DOC_LIBRARY.get(query)
    if doc is None:
        raise RuntimeError(f"no doc found for '{query}'")
    return doc


def robust_execute(query: str, tool_chain: list, incidents: list,
                   max_attempts: int = 2) -> str | None:
    """Try each tool in the chain up to max_attempts times. None = all failed."""
    for tool_name, tool_fn in tool_chain:
        for attempt in range(1, max_attempts + 1):
            try:
                result = tool_fn(query)
                incidents.append({"step": query, "tool": tool_name,
                                  "attempt": attempt, "error": None})
                return result
            except RuntimeError as exc:
                incidents.append({"step": query, "tool": tool_name,
                                  "attempt": attempt, "error": str(exc)})
    return None


def solve_medium2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Robust executor")
    print("=" * 70)

    # (a) Transient failure: fails once, retry succeeds
    incidents_a: list = []
    flaky_once = make_flaky_search(fail_first_n=1)
    result_a = robust_execute("africa area",
                              [("web_search", flaky_once),
                               ("read_doc", read_doc_fallback)], incidents_a)
    print(f"\n  (a) transient: result={result_a[:50]}...")
    for i in incidents_a:
        print(f"      {i}")
    assert result_a is not None and "30,370,000" in result_a
    assert incidents_a[0]["error"] is not None and incidents_a[1]["error"] is None

    # (b) Permanent search failure: falls back to read_doc
    incidents_b: list = []
    always_down = make_flaky_search(fail_first_n=99)
    result_b = robust_execute("africa area",
                              [("web_search", always_down),
                               ("read_doc", read_doc_fallback)], incidents_b)
    print(f"\n  (b) fallback: result={result_b[:50]}...")
    assert result_b is not None and result_b.startswith("Doc")
    assert [i["tool"] for i in incidents_b] == ["web_search", "web_search", "read_doc"]

    # (c) Total failure: honest partial answer, no hallucination
    incidents_c: list = []
    result_c = robust_execute("atlantis population",
                              [("web_search", always_down),
                               ("read_doc", read_doc_fallback)], incidents_c)
    assert result_c is None
    answer_c = ("I could not retrieve 'atlantis population' (2 tools failed). "
                "Partial answer based on: nothing retrieved.")
    print(f"\n  (c) total failure -> {answer_c}")
    # 2 attempts per tool x 2 tools = 4 logged failures
    assert len(incidents_c) == 4 and all(i["error"] for i in incidents_c)
    assert not any(ch.isdigit() for ch in answer_c.split("(2 tools")[0])

    print("\n[Verification] PASS -- retry, fallback and honest failure all work")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Clarification node: when to ask the user
# ==========================================================================

KNOWN_PLACES = ("africa", "paris", "lyon")

# Mock user answers for the clarification round-trip
USER_ANSWERS = {
    "which place?": "Paris",
    "denser than what?": "Lyon",
    "area or population?": "area",
    "what does 'it' refer to?": "Africa",
}


def detect_ambiguity(question: str) -> list[str]:
    """Local, deterministic ambiguity rules. Empty list = clear question."""
    q = question.lower()
    issues: list[str] = []
    mentions_place = any(p in q for p in KNOWN_PLACES)

    if re.search(r"\b(its|it)\b", q) and not mentions_place:
        issues.append("what does 'it' refer to?")
    elif ("density" in q or "population" in q) and not mentions_place:
        issues.append("which place?")
    if "denser" in q and " than " not in q:
        issues.append("denser than what?")
    if "how big" in q:
        issues.append("area or population?")
    return issues


def run_with_clarification(question: str, max_rounds: int = 2) -> dict:
    """Clarify (up to max_rounds), then answer via the deterministic planner."""
    rounds = 0
    current = question
    clarifications: list[tuple[str, str]] = []

    while rounds < max_rounds:
        issues = detect_ambiguity(current)
        if not issues:
            break
        rounds += 1
        answer = USER_ANSWERS.get(issues[0], "")
        clarifications.append((issues[0], answer))
        current = f"{current} (clarified: {answer})" if answer else current
        if not answer:
            break

    issues_left = detect_ambiguity(current)
    if issues_left:
        final = (f"Assuming you mean Paris: I cannot fully disambiguate "
                 f"({', '.join(issues_left)}).")
    else:
        # Tiny answerer reusing the shared search index
        q = current.lower()
        entity = next((p for p in KNOWN_PLACES if p in q), "paris")
        if "density" in q:
            area_raw = mock_web_search(f"{entity} area")
            pop_raw = mock_web_search(f"{entity} population")
            area = float(re.search(r"([\d,]+)", area_raw).group(1).replace(",", ""))
            pop = float(re.search(r"([\d,]+)", pop_raw).group(1).replace(",", ""))
            final = f"The density of {entity.title()} is {round(pop / area)} hab/km2."
        elif "population" in q:
            final = mock_web_search(f"{entity} population")
        else:
            final = mock_web_search(f"{entity} area")

    return {"question": question, "resolved": current,
            "clarifications": clarifications, "rounds": rounds, "answer": final}


def solve_medium3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Clarification node")
    print("=" * 70)

    # Ambiguous: missing entity -> one clarification round
    r1 = run_with_clarification("What is the density?")
    print(f"\n  Q: {r1['question']}")
    print(f"     clarifications: {r1['clarifications']}")
    print(f"     resolved: {r1['resolved']}")
    print(f"     answer: {r1['answer']}")
    assert r1["rounds"] == 1
    assert r1["clarifications"] == [("which place?", "Paris")]
    assert "20581" in r1["answer"].replace(",", "")

    # Clear question: zero clarification
    r2 = run_with_clarification("What is the population of Africa?")
    print(f"\n  Q: {r2['question']} -> rounds={r2['rounds']}")
    assert r2["rounds"] == 0 and "1,460,000,000" in r2["answer"]

    # Guard: still ambiguous after max rounds -> explicit assumption
    r3 = run_with_clarification("How big is it and is it denser?", max_rounds=2)
    print(f"\n  Q: {r3['question']}")
    print(f"     rounds={r3['rounds']}, answer: {r3['answer']}")
    assert r3["rounds"] == 2
    print("\n[Verification] PASS -- clarify, passthrough and max-rounds guard")


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
    # Hard Ex 1 (session agent with anaphora):
    #   - Keep last_intent + last_entities in a Session object
    #   - resolve(): regex on 'and for X?', 'its <attr>', 'compare them'
    #   - Reuse MemoryAgent (medium 1) so 'compare them' costs 0 searches
    #
    # Hard Ex 2 (cost-aware planner):
    #   - COSTS table + estimate_plan_cost(plan) = sum of step costs
    #   - optimize: dedup -> memory substitution (recall=0) -> downgrade
    #   - Generate variants A/B for comparisons, pick the cheaper estimate

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

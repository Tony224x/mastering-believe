"""
Solutions -- Day 1 (HARD): Anatomy of an AI Agent

Contains solutions for:
  - Hard Ex 1: Self-correcting agent with a Reflexion loop (Shinn et al., 2023)
  - Hard Ex 2: Generic, configurable agent framework (what LangGraph/CrewAI
               do under the hood: config object, hooks, traces, retry, stops)

All LLM calls are mocked (scripted, deterministic) so the file runs offline
with no API key -- same convention as the other solution files. The TOOLS,
however, are real: the calculator really evaluates, the sandboxed python_exec
really executes the prime-summing code and produces the correct result.

Run:  python 03-exercises/solutions/01-anatomie-agent-hard.py
Each solution is self-contained.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ==========================================================================
# SHARED UTILS
# ==========================================================================

def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token) -- enough for budgeting demos."""
    return max(1, len(text) // 4)


# ==========================================================================
# HARD EXERCISE 1 -- Self-correcting agent (Reflexion loop)
# ==========================================================================
#
# Architecture (from the exercise statement):
#
#   User Question -> [ReAct attempt] -> [Evaluator: PASS/RETRY]
#                         ^                    |
#                         +---- reflexion -----+   (max 3 attempts)
#
# Why this works: the agent's first attempt often fails for a *diagnosable*
# reason. A second LLM call acting as a judge can spot the flaw, and feeding
# that critique back into the next attempt's context is enough to steer the
# agent toward a better strategy -- no fine-tuning, no extra tools.

# --- Tools (real implementations) -----------------------------------------

def tool_calculator(params: dict) -> str:
    """Evaluate a basic arithmetic expression. Whitelist chars to stay safe."""
    expression = params["expression"]
    if not all(c in "0123456789+-*/.() %" for c in expression):
        return f"ToolError: unsafe expression: {expression!r}"
    return str(eval(expression))


def tool_search(params: dict) -> str:
    """Mock web search -- returns canned snippets for the demo."""
    query = params["query"].lower()
    if "prime" in query:
        return ("Snippet: 'The 100th prime number is 541. Prime numbers become "
                "sparser as numbers grow (prime number theorem).'")
    return f"Snippet: mock result for '{params['query']}'"


# Sandbox: only these builtins are visible to executed code. No __import__,
# so `import` cannot work even if it slipped past the token filter.
SAFE_BUILTINS: dict[str, Any] = {
    "range": range, "len": len, "sum": sum, "min": min, "max": max,
    "abs": abs, "all": all, "any": any, "enumerate": enumerate,
    "int": int, "float": float, "str": str, "list": list, "bool": bool,
}

# Tokens that would let code escape the sandbox. We block them at the string
# level AND remove __import__ from builtins -- defense in depth.
FORBIDDEN_TOKENS = ["import", "open(", "__", "exec(", "eval(", "compile("]


def tool_python_exec(params: dict) -> str:
    """
    Execute Python code in an isolated namespace.
    Convention: the code must assign its output to a variable named `result`.
    """
    code = params["code"]
    for token in FORBIDDEN_TOKENS:
        if token in code:
            return f"SandboxError: forbidden token {token!r} in code"
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, namespace)  # noqa: S102 -- sandboxed namespace, demo only
    except Exception as e:
        return f"SandboxError: {type(e).__name__}: {e}"
    return str(namespace.get("result", "(no `result` variable set)"))


REFLEXION_TOOLS: dict[str, Callable[[dict], str]] = {
    "calculator": tool_calculator,
    "search": tool_search,
    "python_exec": tool_python_exec,
}


# --- Mock LLM: scripted ReAct decisions per attempt -----------------------

# Attempt 1 deliberately uses a WRONG strategy (arithmetic-mean shortcut on
# primes), so the evaluator has something real to catch. Attempt 2 follows
# the reflexion feedback and computes the sum exactly with python_exec.
PRIME_SUM_CODE = """
primes = []
n = 2
while len(primes) < 100:
    if all(n % p != 0 for p in primes):
        primes.append(n)
    n += 1
result = sum(primes)
"""

ATTEMPT_SCRIPTS: dict[int, list[dict]] = {
    1: [
        {
            "thought": ("I need the sum of the first 100 primes. I'll search to "
                        "recall the 100th prime, then approximate."),
            "action": "search",
            "action_input": {"query": "100th prime number"},
        },
        {
            "thought": ("The 100th prime is 541. I'll use the arithmetic-mean "
                        "shortcut: 100 * (first + last) / 2."),
            "action": "calculator",
            "action_input": {"expression": "100 * (2 + 541) / 2"},
        },
        {
            "thought": "The shortcut gives 27150. I'll answer with that.",
            "action": "finish",
            "action_input": "The sum of the first 100 prime numbers is 27150.",
        },
    ],
    2: [
        {
            "thought": ("The feedback says the arithmetic-mean shortcut is invalid "
                        "because primes are NOT evenly distributed. I must generate "
                        "the primes explicitly and sum them with python_exec."),
            "action": "python_exec",
            "action_input": {"code": PRIME_SUM_CODE},
        },
        {
            "thought": "Exact computation returned 24133. That is the answer.",
            "action": "finish",
            "action_input": "The sum of the first 100 prime numbers is 24133.",
        },
    ],
}


def mock_react_llm(attempt: int, step_index: int, context: str) -> dict:
    """
    Scripted stand-in for the ReAct LLM call. In production this would be
    one chat-completion per step, with `context` injected into the prompt.
    """
    return ATTEMPT_SCRIPTS[attempt][step_index]


def mock_evaluator(question: str, answer: str) -> tuple[str, str]:
    """
    Second LLM call acting as a judge. The mock 'knows' how to spot the flaw:
    in production the prompt would be:
      "Is this answer correct and complete? If not, what went wrong?
       Reply with PASS or RETRY plus an explanation."
    """
    if "24133" in answer:
        return ("PASS", "The answer was computed exactly by generating the primes; "
                        "24133 is the correct sum of the first 100 primes.")
    return ("RETRY", "The answer looks like an approximation. Primes are not "
                     "uniformly distributed, so the arithmetic-mean shortcut "
                     "(n * (first + last) / 2) is invalid. Generate the 100 primes "
                     "explicitly (e.g. with python_exec) and sum them.")


# --- The Reflexion agent ---------------------------------------------------

class ReflexionAgent:
    """Outer loop (attempts + reflexion) wrapped around an inner ReAct loop."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.attempts: list[dict] = []  # Full trace of every attempt

    def _react_attempt(self, question: str, attempt: int, context: str) -> tuple[str, list[dict]]:
        """Inner ReAct loop: Thought -> Action -> Observation until finish."""
        steps: list[dict] = []
        answer = ""
        for step_index in range(len(ATTEMPT_SCRIPTS[attempt])):
            decision = mock_react_llm(attempt, step_index, context)
            thought, action = decision["thought"], decision["action"]
            action_input = decision["action_input"]

            print(f"    Thought: {thought}")
            if action == "finish":
                answer = str(action_input)
                print(f"    Final Answer: {answer}")
                steps.append({"thought": thought, "action": "finish", "observation": None})
                break

            observation = REFLEXION_TOOLS[action](action_input)
            print(f"    Action: {action}({json.dumps(action_input)[:60]}...)"
                  if len(json.dumps(action_input)) > 60
                  else f"    Action: {action}({json.dumps(action_input)})")
            print(f"    Observation: {observation[:90]}")
            steps.append({"thought": thought, "action": action,
                          "action_input": action_input, "observation": observation})
        return answer, steps

    def run(self, question: str) -> dict:
        feedback_history: list[str] = []

        for attempt in range(1, self.max_attempts + 1):
            # Reflexion feedback from previous failures is injected into the
            # context the agent sees -- THIS is the self-correction mechanism.
            context = ""
            if feedback_history:
                context = "Previous attempt failed because: " + " | ".join(feedback_history)

            print(f"\n  --- Attempt {attempt}/{self.max_attempts} ---")
            if context:
                print(f"    [Reflexion context] {context[:100]}...")

            answer, steps = self._react_attempt(question, attempt, context)
            verdict, explanation = mock_evaluator(question, answer)
            print(f"    Evaluator: {verdict} -- {explanation[:80]}")

            self.attempts.append({
                "attempt": attempt, "context": context, "answer": answer,
                "steps": steps, "verdict": verdict, "explanation": explanation,
            })

            if verdict == "PASS":  # Early exit -- no wasted attempts
                return {"answer": answer, "attempts": self.attempts, "status": "PASS"}
            feedback_history.append(explanation)

        # 3 failures: return the best (here: last) attempt instead of nothing
        return {"answer": self.attempts[-1]["answer"],
                "attempts": self.attempts, "status": "EXHAUSTED"}


def hard_ex1_reflexion_agent():
    """
    Solution: ReAct agent + evaluator + reflexion outer loop (max 3 attempts).
    Key insight: self-correction = re-running with the critique IN the context.
    """
    print("\n" + "=" * 60)
    print("  Hard Ex 1 -- Self-Correcting Agent (Reflexion Loop)")
    print("=" * 60)

    question = "What is the sum of the first 100 prime numbers?"
    print(f"\n  Question: {question}")

    agent = ReflexionAgent(max_attempts=3)
    result = agent.run(question)

    # --- Verify the success criteria -------------------------------------
    a1, a2 = result["attempts"][0], result["attempts"][1]
    assert a1["verdict"] == "RETRY", "Attempt 1 should fail (wrong shortcut)"
    assert a2["verdict"] == "PASS", "Attempt 2 should pass after reflexion"
    assert a1["explanation"] in a2["context"], "Feedback must be in attempt 2's context"
    assert "24133" in result["answer"], "Final answer must be the exact prime sum"
    assert len(result["attempts"]) == 2, "Early exit on PASS: only 2 of 3 attempts used"

    # Sandbox check: python_exec must reject escape attempts
    blocked = tool_python_exec({"code": "import os\nresult = os.getcwd()"})
    assert "SandboxError" in blocked, "Sandbox must block 'import os'"
    blocked2 = tool_python_exec({"code": "().__class__.__bases__"})
    assert "SandboxError" in blocked2, "Sandbox must block dunder access"
    print(f"\n  Sandbox test: 'import os' -> {blocked}")
    print(f"  Sandbox test: dunder access -> {blocked2}")

    print(f"\n  Trace summary: attempt 1 = {a1['answer'][-6:-1]} (RETRY), "
          f"attempt 2 = {a2['answer'][-6:-1]} (PASS, early exit)")
    print("  PASS -- Reflexion loop: fail -> critique -> corrected retry.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Generic, configurable agent framework
# ==========================================================================
#
# A mini-framework with the 6 required features: config object, hooks,
# automatic traces, retry with backoff, multiple stopping conditions, and
# JSON-serialisable traces. The framework itself lives between the two
# sentinel comments below and is asserted to be < 300 lines.

# --- FRAMEWORK START -------------------------------------------------------

@dataclass
class Tool:
    """Encapsulates a tool: name, description, schema, implementation."""
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict], str]


class Memory:
    """Working memory: chronological entries with a cheap summarize()."""

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._entries.append({"role": role, "content": content})

    def get(self) -> list[dict[str, str]]:
        return list(self._entries)

    def summarize(self, keep_last: int = 3) -> str:
        """Compress old entries to a count, keep the tail verbatim."""
        old, recent = self._entries[:-keep_last], self._entries[-keep_last:]
        head = f"[{len(old)} earlier entries summarized] " if old else ""
        return head + " | ".join(e["content"][:60] for e in recent)


@dataclass
class AgentConfig:
    """All knobs in one object -- no scattered params."""
    model: str = "mock-react-1"
    temperature: float = 0.0
    max_iterations: int = 10
    max_tokens: int = 4000
    timeout_s: float = 30.0
    system_prompt: str = "You are a ReAct agent. Think, act, observe, repeat."
    llm_max_retries: int = 3
    retry_base_delay: float = 0.02  # Small for the demo; ~1s in production
    # Custom stopping condition: receives the trace, returns True to stop
    stop_predicate: Callable[["AgentTrace"], bool] | None = None


@dataclass
class TraceStep:
    """One Thought -> Action -> Observation cycle."""
    index: int
    thought: str
    action: str
    action_input: Any
    observation: str | None
    duration_s: float
    tokens: int


@dataclass
class AgentTrace:
    """Records every step: thoughts, actions, durations, token counts."""
    question: str
    model: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    total_duration_s: float = 0.0
    llm_retries: int = 0

    @property
    def tools_used(self) -> list[str]:
        return sorted({s.action for s in self.steps if s.action != "finish"})

    def to_json(self) -> str:
        """Serialize for persistence (log files, eval datasets, replay)."""
        return json.dumps({
            "question": self.question,
            "model": self.model,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "total_duration_s": round(self.total_duration_s, 4),
            "llm_retries": self.llm_retries,
            "tools_used": self.tools_used,
            "steps": [{
                "index": s.index, "thought": s.thought, "action": s.action,
                "action_input": s.action_input, "observation": s.observation,
                "duration_s": round(s.duration_s, 4), "tokens": s.tokens,
            } for s in self.steps],
        }, indent=2)


@dataclass
class AgentResult:
    """What run() returns: the answer + full trace + why we stopped."""
    answer: str
    trace: AgentTrace
    stopped_by: str  # finish | max_iterations | max_tokens | timeout | custom_predicate


class ReActAgent:
    """The main agent class. Configurable, observable, extensible.

    The LLM is injected as a callable (question, iteration) -> decision dict
    {"thought", "action", "action_input"} -- so tests can script it and
    production can swap in a real API client without touching the loop.
    """

    HOOK_EVENTS = ("before_llm_call", "after_tool_call", "on_error", "on_finish")

    def __init__(self, config: AgentConfig, llm: Callable[[str, int], dict]):
        self.config = config
        self.llm = llm
        self.tools: dict[str, Tool] = {}
        self.memory = Memory()
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def add_hook(self, event: str, callback: Callable[..., None]) -> None:
        if event not in self.HOOK_EVENTS:
            raise ValueError(f"Unknown event '{event}'. Valid: {self.HOOK_EVENTS}")
        self._hooks[event].append(callback)

    def _emit(self, event: str, **payload: Any) -> None:
        for cb in self._hooks[event]:
            cb(**payload)

    def _call_llm_with_retry(self, question: str, iteration: int, trace: AgentTrace) -> dict:
        """Retry transient LLM failures with exponential backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                return self.llm(question, iteration)
            except Exception as e:  # Rate limit / timeout / network in real life
                last_exc = e
                trace.llm_retries += 1
                self._emit("on_error", error=str(e), attempt=attempt)
                if attempt < self.config.llm_max_retries:
                    # Exponential backoff: base * 2^attempt
                    time.sleep(self.config.retry_base_delay * (2 ** attempt))
        raise RuntimeError(f"LLM failed after {self.config.llm_max_retries} retries") from last_exc

    def run(self, question: str) -> AgentResult:
        trace = AgentTrace(question=question, model=self.config.model)
        self.memory.add("user", question)
        start = time.time()
        answer, stopped_by = "", "max_iterations"

        for i in range(self.config.max_iterations):
            self._emit("before_llm_call", iteration=i, question=question)
            step_start = time.time()
            decision = self._call_llm_with_retry(question, i, trace)
            thought = decision.get("thought", "")
            action = decision.get("action", "finish")
            action_input = decision.get("action_input", {})

            if action == "finish":
                answer, stopped_by = str(action_input), "finish"
                trace.steps.append(TraceStep(i, thought, "finish", action_input, None,
                                             time.time() - step_start,
                                             estimate_tokens(thought + answer)))
                break

            if action not in self.tools:
                observation = f"ToolError: unknown tool '{action}'"
                self._emit("on_error", error=observation, attempt=0)
            else:
                try:
                    observation = self.tools[action].fn(action_input)
                except Exception as e:
                    observation = f"ToolError: {type(e).__name__}: {e}"
                    self._emit("on_error", error=observation, attempt=0)
            self._emit("after_tool_call", tool=action, observation=observation)

            tokens = estimate_tokens(thought + str(action_input) + observation)
            trace.steps.append(TraceStep(i, thought, action, action_input,
                                         observation, time.time() - step_start, tokens))
            trace.total_tokens += tokens
            self.memory.add("assistant", f"{action}({action_input}) -> {observation[:80]}")

            # --- Stopping conditions (checked after every step) ------------
            if trace.total_tokens > self.config.max_tokens:
                stopped_by = "max_tokens"
                break
            if time.time() - start > self.config.timeout_s:
                stopped_by = "timeout"
                break
            if self.config.stop_predicate and self.config.stop_predicate(trace):
                stopped_by = "custom_predicate"
                break

        trace.final_answer = answer
        trace.total_duration_s = time.time() - start
        self._emit("on_finish", answer=answer, stopped_by=stopped_by)
        return AgentResult(answer=answer, trace=trace, stopped_by=stopped_by)

# --- FRAMEWORK END ---------------------------------------------------------


# --- Scripted / flaky LLMs for the demos -----------------------------------

class ScriptedLLM:
    """Pops pre-written decisions -- deterministic stand-in for the real LLM."""

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self._cursor = 0

    def __call__(self, question: str, iteration: int) -> dict:
        decision = self._script[min(self._cursor, len(self._script) - 1)]
        self._cursor += 1
        return decision


class FlakyLLM:
    """Wraps an LLM and fails the first `fail_times` calls -- tests retry."""

    def __init__(self, inner: Callable[[str, int], dict], fail_times: int):
        self.inner = inner
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, question: str, iteration: int) -> dict:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError(f"Simulated rate limit (call {self.calls})")
        return self.inner(question, iteration)


def make_tool(name: str, description: str, fn: Callable[[dict], str]) -> Tool:
    """Helper: most demo tools take a single free-form params dict."""
    return Tool(name=name, description=description,
                parameters={"type": "object", "properties": {}}, fn=fn)


def hard_ex2_agent_framework():
    """
    Solution: 3 different agents built from the SAME framework, differing
    only by their AgentConfig + tool set + (scripted) LLM.
    """
    print("\n" + "=" * 60)
    print("  Hard Ex 2 -- Generic Agent Framework")
    print("=" * 60)

    # Hook callbacks record events so we can ASSERT they fired in order
    events: list[str] = []

    def on_llm(iteration: int, question: str) -> None:
        events.append(f"before_llm_call:{iteration}")

    def on_tool(tool: str, observation: str) -> None:
        events.append(f"after_tool_call:{tool}")
        print(f"      [hook] after_tool_call: {tool} -> {observation[:50]}")

    def on_error(error: str, attempt: int) -> None:
        events.append("on_error")
        print(f"      [hook] on_error (attempt {attempt}): {error[:60]}")

    def on_finish(answer: str, stopped_by: str) -> None:
        events.append(f"on_finish:{stopped_by}")

    # --- Agent A: calculator only (1 tool) --------------------------------
    print("\n  --- Agent A: calculator agent (1 tool) ---")
    llm_a = ScriptedLLM([
        {"thought": "Compute 17 * 23 with the calculator.",
         "action": "calculator", "action_input": {"expression": "17 * 23"}},
        {"thought": "Result is 391.", "action": "finish", "action_input": "17 * 23 = 391"},
    ])
    agent_a = ReActAgent(AgentConfig(max_iterations=5), llm_a)
    agent_a.add_tool(make_tool("calculator", "Evaluate math", tool_calculator))
    for ev, cb in [("before_llm_call", on_llm), ("after_tool_call", on_tool),
                   ("on_error", on_error), ("on_finish", on_finish)]:
        agent_a.add_hook(ev, cb)
    result_a = agent_a.run("What is 17 * 23?")
    print(f"    Answer: {result_a.answer} (stopped_by={result_a.stopped_by})")
    assert result_a.answer == "17 * 23 = 391" and result_a.stopped_by == "finish"
    assert events == ["before_llm_call:0", "after_tool_call:calculator",
                      "before_llm_call:1", "on_finish:finish"], f"Hook order wrong: {events}"
    print("    Hooks fired in order: " + " -> ".join(events))

    # --- Agent B: research agent (2 tools) + retry/backoff test -----------
    print("\n  --- Agent B: research agent (2 tools, flaky LLM x2 failures) ---")
    scripted_b = ScriptedLLM([
        {"thought": "Search for the 100th prime first.",
         "action": "search", "action_input": {"query": "100th prime number"}},
        {"thought": "Now double it to sanity-check the calculator.",
         "action": "calculator", "action_input": {"expression": "541 * 2"}},
        {"thought": "Done.", "action": "finish",
         "action_input": "The 100th prime is 541 (and 541 * 2 = 1082)."},
    ])
    flaky = FlakyLLM(scripted_b, fail_times=2)  # First 2 LLM calls raise
    agent_b = ReActAgent(AgentConfig(max_iterations=5, retry_base_delay=0.01), flaky)
    agent_b.add_tool(make_tool("search", "Web search", tool_search))
    agent_b.add_tool(make_tool("calculator", "Evaluate math", tool_calculator))
    agent_b.add_hook("on_error", on_error)
    t0 = time.time()
    result_b = agent_b.run("What is the 100th prime number?")
    elapsed = time.time() - t0
    print(f"    Answer: {result_b.answer}")
    print(f"    LLM retries: {result_b.trace.llm_retries} "
          f"(backoff delays included, total {elapsed:.3f}s)")
    assert result_b.trace.llm_retries == 2, "Retry should have fired exactly twice"
    assert "541" in result_b.answer

    # --- Agent C: multi-step agent with working memory (4 tools) ----------
    print("\n  --- Agent C: multi-step agent with working memory (4 tools) ---")
    scratch: dict[str, str] = {}
    llm_c = ScriptedLLM([
        {"thought": "Find the laptop price first.",
         "action": "search", "action_input": {"query": "Laptop Pro 16 price"}},
        {"thought": "Note the budget so I don't lose it.",
         "action": "note_to_self", "action_input": {"key": "budget", "value": "600"}},
        {"thought": "Compute 15% tax on a 520 EUR laptop.",
         "action": "calculator", "action_input": {"expression": "520 * 1.15"}},
        {"thought": "Recall the budget to compare.",
         "action": "recall", "action_input": {"key": "budget"}},
        {"thought": "598 <= 600: it fits.", "action": "finish",
         "action_input": "With 15% tax the laptop costs 598.0 EUR, within the 600 EUR budget."},
    ])
    agent_c = ReActAgent(AgentConfig(max_iterations=8), llm_c)
    agent_c.add_tool(make_tool("search", "Web search", tool_search))
    agent_c.add_tool(make_tool("calculator", "Evaluate math", tool_calculator))
    agent_c.add_tool(make_tool("note_to_self", "Save a note",
                               lambda p: scratch.update({p["key"]: p["value"]}) or f"Noted {p['key']}"))
    agent_c.add_tool(make_tool("recall", "Recall a note",
                               lambda p: f"{p['key']} = {scratch.get(p['key'], '(missing)')}"))
    result_c = agent_c.run("Does a 520 EUR laptop + 15% tax fit a 600 EUR budget?")
    print(f"    Answer: {result_c.answer}")
    assert result_c.trace.tools_used == ["calculator", "note_to_self", "recall", "search"]

    # --- Stopping conditions: max_iterations + custom predicate -----------
    print("\n  --- Stopping conditions ---")
    looping_llm = ScriptedLLM([  # Never finishes -- always the same action
        {"thought": "loop", "action": "calculator", "action_input": {"expression": "1 + 1"}},
    ])
    agent_d = ReActAgent(AgentConfig(max_iterations=3), looping_llm)
    agent_d.add_tool(make_tool("calculator", "Evaluate math", tool_calculator))
    result_d = agent_d.run("loop forever")
    print(f"    Looping agent stopped_by: {result_d.stopped_by} "
          f"after {len(result_d.trace.steps)} steps")
    assert result_d.stopped_by == "max_iterations"

    agent_e = ReActAgent(
        AgentConfig(max_iterations=50,
                    # Custom predicate: stop once any tool was used twice
                    stop_predicate=lambda tr: len(tr.steps) >= 2),
        ScriptedLLM([{"thought": "loop", "action": "calculator",
                      "action_input": {"expression": "2 + 2"}}]))
    agent_e.add_tool(make_tool("calculator", "Evaluate math", tool_calculator))
    result_e = agent_e.run("loop with custom stop")
    print(f"    Custom-predicate agent stopped_by: {result_e.stopped_by}")
    assert result_e.stopped_by == "custom_predicate"

    # --- AgentTrace.to_json(): valid, parseable JSON -----------------------
    print("\n  --- Trace serialisation ---")
    trace_json = result_c.trace.to_json()
    parsed = json.loads(trace_json)  # Raises if invalid -> implicit assert
    assert parsed["tools_used"] == ["calculator", "note_to_self", "recall", "search"]
    assert len(parsed["steps"]) == 5 and parsed["total_tokens"] > 0
    print(f"    to_json() OK: {len(trace_json)} chars, {len(parsed['steps'])} steps, "
          f"{parsed['total_tokens']} tokens, tools={parsed['tools_used']}")

    # --- Framework size: < 300 lines (complexity budget) -------------------
    source = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(source) if "FRAMEWORK START" in l)
    end = next(i for i, l in enumerate(source) if "FRAMEWORK END" in l)
    framework_lines = end - start - 1
    print(f"\n  Framework size: {framework_lines} lines (< 300 required)")
    assert framework_lines < 300, f"Framework too big: {framework_lines} lines"

    print("\n  PASS -- 3 agents, hooks, retry/backoff, 3 stop conditions, JSON trace.\n")


# ==========================================================================
# MAIN -- Run both hard solutions
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 1 HARD Solutions -- Anatomy of an AI Agent")
    print("#" * 60)

    hard_ex1_reflexion_agent()
    hard_ex2_agent_framework()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

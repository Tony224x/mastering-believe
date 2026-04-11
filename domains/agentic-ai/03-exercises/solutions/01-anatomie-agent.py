"""
Solutions — Day 1: Anatomy of an AI Agent

Contains solutions for:
  - Easy Ex 1: Weather tool
  - Easy Ex 2: Guard-rails (loop detection, token budget)
  - Easy Ex 3: Execution trace formatter
  - Medium Ex 1: Function calling (structured output)
  - Medium Ex 2: Working memory
  - Medium Ex 3: Router + Specialists (multi-agent)

Run:  python 03-exercises/solutions/01-anatomie-agent.py
Each solution is a self-contained function that can be run independently.
"""

import json
import re
import time
from datetime import datetime
from typing import Any, Callable

# ==========================================================================
# EASY EXERCISE 1 — Add a "weather" tool
# ==========================================================================

def easy_ex1_weather_tool():
    """
    Solution: define a weather tool and plug it into the agent.
    Key insight: a tool is just a dict (schema) + a function (implementation).
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 1 — Weather Tool")
    print("=" * 60)

    # 1. Define the tool schema
    weather_tool = {
        "name": "get_weather",
        "description": "Get current weather for a city. Use when the user asks about weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    }

    # 2. Implement the tool function (mock)
    def tool_get_weather(params: dict) -> str:
        city = params["city"]
        # Mock data — in production, call a weather API
        mock_weather = {
            "tokyo": "Tokyo: 22C, sunny with light clouds",
            "paris": "Paris: 18C, partly cloudy",
            "conakry": "Conakry: 31C, humid, chance of rain",
        }
        return mock_weather.get(city.lower(), f"{city}: 20C, clear skies (mock data)")

    # 3. Simulate the agent using this tool
    simulated_steps = [
        {
            "thought": "The user wants to know the weather in Tokyo. I'll use get_weather.",
            "action": "get_weather",
            "action_input": {"city": "Tokyo"},
        },
    ]

    # Execute the simulation
    for i, step in enumerate(simulated_steps):
        print(f"\n  Step {i + 1}:")
        print(f"  Thought: {step['thought']}")
        print(f"  Action: {step['action']}")
        print(f"  Input: {step['action_input']}")

        if step["action"] == "get_weather":
            obs = tool_get_weather(step["action_input"])
            print(f"  Observation: {obs}")

    final = "The weather in Tokyo is 22C and sunny with light clouds."
    print(f"\n  Final Answer: {final}")
    print(f"  PASS — Tool defined, called, and observation returned correctly.\n")


# ==========================================================================
# EASY EXERCISE 2 — Guard-rails: loop detection + token budget
# ==========================================================================

def easy_ex2_guardrails():
    """
    Solution: add safety mechanisms to prevent infinite loops and cost explosion.
    Key insight: ALWAYS add these in production. An unguarded agent is a liability.
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 2 — Guard-rails")
    print("=" * 60)

    # Track state for guard-rails
    max_iterations = 10
    max_tokens_budget = 5000
    tokens_used = 0
    action_history: list[tuple[str, Any]] = []  # (action_name, action_input) pairs

    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    def check_loop(action: str, action_input: Any) -> bool:
        """Detect if the last 2 actions are identical — sign of a stuck agent."""
        current = (action, json.dumps(action_input, sort_keys=True))
        if len(action_history) >= 1:
            last = action_history[-1]
            if current == last:
                return True  # Loop detected
        action_history.append(current)
        return False

    # Simulate an agent that gets stuck in a loop
    stuck_steps = [
        ("search", {"query": "GDP France"}),
        ("search", {"query": "GDP France"}),  # Same as previous — loop!
        ("search", {"query": "GDP France"}),
    ]

    print("\n  Testing loop detection with a stuck agent:")
    for i, (action, params) in enumerate(stuck_steps):
        step_text = f"Thought: ... Action: {action} Input: {json.dumps(params)}"
        tokens_used += estimate_tokens(step_text)

        print(f"\n  Step {i + 1}/{max_iterations} | Tokens: {tokens_used}/{max_tokens_budget}")

        # Check loop
        if check_loop(action, params):
            print(f"  LOOP DETECTED — Same action repeated: {action}({params})")
            print(f"  Agent stopped to prevent infinite loop.")
            break

        # Check token budget
        if tokens_used > max_tokens_budget:
            print(f"  TOKEN BUDGET EXCEEDED — {tokens_used} > {max_tokens_budget}")
            print(f"  Agent stopped to prevent cost explosion.")
            break

        print(f"  Action: {action}({params}) — OK")

    print(f"\n  PASS — Loop detected at step 2, agent stopped.\n")


# ==========================================================================
# EASY EXERCISE 3 — Execution trace formatter
# ==========================================================================

def easy_ex3_trace_formatter():
    """
    Solution: format the agent's execution history into a readable trace.
    Key insight: observability is not optional. You WILL need to debug agents.
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 3 — Execution Trace Formatter")
    print("=" * 60)

    def format_trace(question: str, steps: list[dict], final_answer: str) -> str:
        """
        Format agent execution steps into a human-readable trace.

        Args:
            question: The original question
            steps: List of dicts with keys: thought, action, action_input, observation, duration
            final_answer: The agent's final answer
        """
        total_duration = sum(s.get("duration", 0) for s in steps)
        tools_used = list({s["action"] for s in steps if s["action"] != "finish"})

        lines = [
            "=== Agent Trace ===",
            f"Question: {question}",
            f"Steps: {len(steps)} | Duration: {total_duration:.1f}s | Tools used: {', '.join(tools_used) or 'none'}",
            "",
        ]

        for i, step in enumerate(steps):
            action_str = step["action"]
            if step.get("action_input"):
                input_repr = json.dumps(step["action_input"]) if isinstance(step["action_input"], dict) else str(step["action_input"])
                action_str += f"({input_repr})"

            lines.append(f"  [{i + 1}] Thought: {step['thought']}")
            lines.append(f"      Action: {action_str}")
            if step.get("observation"):
                lines.append(f"      Observation: {step['observation']}")
            lines.append(f"      Duration: {step.get('duration', 0):.1f}s")
            lines.append("")

        lines.append(f"Final Answer: {final_answer}")
        lines.append("=" * 19)

        return "\n".join(lines)

    # Example trace data
    steps = [
        {
            "thought": "I need to calculate 25 * 47",
            "action": "calculator",
            "action_input": {"expression": "25 * 47"},
            "observation": "1175",
            "duration": 0.8,
        },
        {
            "thought": "Now I need the current time",
            "action": "get_current_time",
            "action_input": {},
            "observation": "2026-04-11 14:30:00",
            "duration": 0.3,
        },
        {
            "thought": "I have both answers, finishing",
            "action": "finish",
            "action_input": "25 * 47 = 1175, current time is 2026-04-11 14:30:00",
            "observation": None,
            "duration": 0.5,
        },
    ]

    trace = format_trace(
        "What is 25 * 47, and what time is it?",
        steps,
        "25 * 47 = 1175, current time is 2026-04-11 14:30:00"
    )
    print(f"\n{trace}")
    print(f"\n  PASS — Trace formatted with steps, durations, and tools.\n")


# ==========================================================================
# MEDIUM EXERCISE 1 — Function calling (structured output)
# ==========================================================================

def medium_ex1_function_calling():
    """
    Solution: use OpenAI's function calling API format instead of text parsing.
    Key insight: function calling eliminates the fragile regex parsing.
    In production, ALWAYS prefer structured output over text parsing.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 1 — Function Calling (Structured Output)")
    print("=" * 60)

    # Tools in OpenAI function calling format
    tools_openai_format = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
    ]

    # Simulated API responses in function calling format
    simulated_responses = [
        # Step 1: LLM returns a tool call (not text)
        {
            "role": "assistant",
            "content": None,  # No text content when calling a tool
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "25 * 47"}'
                }
            }]
        },
        # Step 2: LLM returns another tool call
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "arguments": "{}"
                }
            }]
        },
        # Step 3: LLM responds with text (no tool call = final answer)
        {
            "role": "assistant",
            "content": "25 * 47 = 1175, and the current time is 2026-04-11 14:30:00.",
            "tool_calls": None
        },
    ]

    # Tool implementations (same as before)
    tool_fns = {
        "calculator": lambda p: str(eval(p["expression"])),
        "get_current_time": lambda p: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # The agent loop using function calling
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 25 * 47, and what time is it?"},
    ]

    print("\n  Running agent with function calling format:\n")

    for i, response in enumerate(simulated_responses):
        tool_calls = response.get("tool_calls")

        if tool_calls:
            # LLM wants to call a tool — structured, no parsing needed
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                result = tool_fns[fn_name](fn_args)

                print(f"  Step {i + 1}: tool_call -> {fn_name}({fn_args})")
                print(f"           observation -> {result}")

                # Add tool call to history (assistant message)
                messages.append(response)
                # Add tool result with role "tool" and reference the call ID
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })
        else:
            # No tool call = final answer
            print(f"\n  Step {i + 1}: final answer -> {response['content']}")

    print(f"\n  PASS — No regex, structured tool calls, role='tool' for results.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 — Working memory
# ==========================================================================

def medium_ex2_working_memory():
    """
    Solution: agent with a scratchpad that persists between steps.
    Key insight: working memory lets the agent track intermediate results
    without relying on the LLM to "remember" across the growing context.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 2 — Working Memory")
    print("=" * 60)

    # Working memory store
    memory: dict[str, str] = {}

    def tool_note_to_self(params: dict) -> str:
        """Save a note to working memory."""
        key, value = params["key"], params["value"]
        memory[key] = value
        return f"Noted: {key} = {value}"

    def tool_recall(params: dict) -> str:
        """Recall a note from working memory."""
        key = params["key"]
        if key in memory:
            return f"{key} = {memory[key]}"
        return f"No note found for key '{key}'"

    def tool_calculator(params: dict) -> str:
        expression = params["expression"]
        return str(eval(expression))

    def format_memory() -> str:
        """Format working memory for injection into the prompt."""
        if not memory:
            return "Working Memory: (empty)"
        lines = ["Working Memory:"]
        for k, v in memory.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    # Simulate the agent solving: "Calculate 25*47, add 100, multiply by 2"
    steps = [
        # Step 1: Calculate 25 * 47
        {
            "thought": "First I need to calculate 25 * 47",
            "action": "calculator",
            "params": {"expression": "25 * 47"},
        },
        # Step 2: Save the result
        {
            "thought": "The result is 1175. I should save this for later.",
            "action": "note_to_self",
            "params": {"key": "step1_result", "value": "1175"},
        },
        # Step 3: Add 100
        {
            "thought": "Now I add 100 to the saved result (1175).",
            "action": "calculator",
            "params": {"expression": "1175 + 100"},
        },
        # Step 4: Save intermediate
        {
            "thought": "1175 + 100 = 1275. Saving this.",
            "action": "note_to_self",
            "params": {"key": "step2_result", "value": "1275"},
        },
        # Step 5: Multiply by 2
        {
            "thought": "Finally, multiply 1275 by 2.",
            "action": "calculator",
            "params": {"expression": "1275 * 2"},
        },
    ]

    tool_dispatch = {
        "calculator": tool_calculator,
        "note_to_self": tool_note_to_self,
        "recall": tool_recall,
    }

    print(f"\n  Question: Calculate 25*47, add 100, multiply by 2\n")

    for i, step in enumerate(steps):
        print(f"  [{format_memory()}]")
        print(f"  Step {i + 1}:")
        print(f"    Thought: {step['thought']}")
        print(f"    Action: {step['action']}({step['params']})")

        result = tool_dispatch[step["action"]](step["params"])
        print(f"    Observation: {result}")
        print()

    print(f"  [{format_memory()}]")
    print(f"  Final: (25 * 47 + 100) * 2 = 2550")
    print(f"\n  PASS — Working memory tracks intermediate results across steps.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 — Router + Specialists (multi-agent)
# ==========================================================================

def medium_ex3_router_specialists():
    """
    Solution: a router agent that delegates to specialist agents.
    Key insight: multi-agent is just function composition.
    The router's "tools" are other agents.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 3 — Router + Specialists")
    print("=" * 60)

    # --- Specialist agents (simplified: single-step for this demo) ---

    def math_agent(question: str) -> str:
        """Specialist agent for math. Extracts expression and calculates."""
        print(f"    [Math Agent] Received: {question}")
        # In a real implementation, this would be a full ReAct loop
        # For the demo, extract numbers and compute
        match = re.search(r'(\d+)\s*[\*x]\s*(\d+)', question)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            result = a * b
            print(f"    [Math Agent] Calculated: {a} * {b} = {result}")
            return f"{a} * {b} = {result}"
        return "Could not parse math expression"

    def research_agent(question: str) -> str:
        """Specialist agent for research. Searches and synthesizes."""
        print(f"    [Research Agent] Received: {question}")
        # Mock search results
        mock_answers = {
            "capital": "The capital of Guinea is Conakry.",
            "guinea": "The capital of Guinea is Conakry.",
            "population": "Guinea has approximately 14 million inhabitants.",
        }
        for keyword, answer in mock_answers.items():
            if keyword in question.lower():
                print(f"    [Research Agent] Found: {answer}")
                return answer
        return f"[Research Agent] No specific info found for: {question}"

    # --- Router logic ---

    def route_question(question: str) -> tuple[str, str]:
        """
        Determine which specialist should handle the question.
        Returns (agent_type, reasoning).

        In production, this would be an LLM call.
        Here we use keyword matching for the demo.
        """
        q_lower = question.lower()

        # Math indicators
        math_keywords = ["calculate", "compute", "what is", "sum", "multiply", "*", "+", "-", "/"]
        if any(k in q_lower for k in math_keywords) and any(c.isdigit() for c in question):
            return "math", "Question contains numbers and math keywords"

        # Research indicators
        research_keywords = ["capital", "who", "when", "where", "history", "population", "what is the"]
        if any(k in q_lower for k in research_keywords) and not any(c.isdigit() for c in question):
            return "research", "Question is factual, needs research"

        # Default: respond directly
        return "direct", "Simple greeting or general question"

    # --- Test the router ---

    test_questions = [
        "What is 123 * 456?",
        "What is the capital of Guinea?",
        "Hello, how are you?",
    ]

    specialists = {
        "math": math_agent,
        "research": research_agent,
        "direct": lambda q: "Hello! I'm an AI assistant. How can I help you today?",
    }

    for question in test_questions:
        print(f"\n  Question: {question}")
        agent_type, reasoning = route_question(question)
        print(f"  Router -> {agent_type} ({reasoning})")

        answer = specialists[agent_type](question)
        print(f"  Final Answer: {answer}")

    print(f"\n  PASS — Router correctly delegates to 3 different specialists.\n")


# ==========================================================================
# MAIN — Run all solutions
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 1 Solutions — Anatomy of an AI Agent")
    print("#" * 60)

    # Easy exercises
    easy_ex1_weather_tool()
    easy_ex2_guardrails()
    easy_ex3_trace_formatter()

    # Medium exercises
    medium_ex1_function_calling()
    medium_ex2_working_memory()
    medium_ex3_router_specialists()

    # Hard exercises are not included here because they are substantial
    # projects (100-300 lines each). The exercise descriptions provide
    # enough guidance to implement them. Key hints:
    #
    # Hard Ex 1 (Reflexion):
    #   - Wrap react_agent in an outer loop with max 3 retries
    #   - Use a second LLM call as evaluator: "Is this correct? PASS/RETRY"
    #   - On RETRY, append the evaluator's feedback to the agent's context
    #   - The agent sees: "Previous attempt failed because: <feedback>"
    #
    # Hard Ex 2 (Framework):
    #   - Use dataclasses for Tool, Memory, AgentConfig, AgentTrace
    #   - Hooks are just Dict[str, List[Callable]] -- event name -> callbacks
    #   - ReActAgent.run() is the same loop as 01-anatomie-agent.py
    #   - But wrapped in try/except with retry logic and hook dispatch
    #   - AgentTrace.to_json() uses dataclasses.asdict() + json.dumps()

    print("\n" + "#" * 60)
    print("  All solutions executed successfully.")
    print("#" * 60 + "\n")

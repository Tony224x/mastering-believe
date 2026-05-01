"""
Day 1 — Anatomy of an AI Agent: Minimal ReAct Agent from Scratch

A complete ReAct agent in ~150 lines using only httpx + stdlib.
Two modes:
  - LIVE mode:  Uses a real OpenAI-compatible API (set OPENAI_API_KEY env var)
  - SIMULATED mode: Works without any API key using hardcoded LLM responses

Run:
    python 02-code/01-anatomie-agent.py              # simulated mode (no API key needed)
    OPENAI_API_KEY=sk-... python 02-code/01-anatomie-agent.py   # live mode

The agent solves: "What is 25 * 47, and what time is it right now?"
This requires two tool calls — demonstrating the multi-step ReAct loop.
"""

import json
import os
import re
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# TOOLS — The agent's "arms". Each tool: name, description, schema, function.
# Design choice: tools are plain dicts + callables. No framework needed.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Use for any arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '25 * 47'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time. No parameters needed.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search",
        "description": "Search for information on a topic. Use when you need facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
]

# Tool implementation functions — kept deliberately simple
def tool_calculator(expression: str) -> str:
    """Evaluate math safely. In production, use a sandboxed evaluator."""
    try:
        # Only allow safe math characters
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return f"Error: unsafe expression '{expression}'"
        result = eval(expression)  # Safe here because we validated the input
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def tool_get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tool_search(query: str) -> str:
    """Mock search — in production, call a real search API (Tavily, SerpAPI, etc.)."""
    return f"[Mock result for '{query}': No real search configured. This is a demo.]"

# Map tool names to their implementations
TOOL_FUNCTIONS = {
    "calculator": lambda params: tool_calculator(params["expression"]),
    "get_current_time": lambda params: tool_get_current_time(),
    "search": lambda params: tool_search(params["query"]),
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — The "brain" configuration.
# This prompt defines the ReAct format the LLM must follow.
# Critical: the format must be parseable by our extract logic below.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step using tools.

You have access to these tools:
{tools_description}

For each step, you MUST use this exact format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON parameters for the tool>

When you have the final answer, use:

Thought: <your final reasoning>
Action: finish
Action Input: <your final answer as a string>

Rules:
- Always think before acting
- Use ONE tool per step
- Wait for the observation before your next thought
- When you have enough information, use the finish action"""

def build_tools_description(tools: list[dict]) -> str:
    """Format tool definitions for the system prompt. Human-readable, not JSON Schema."""
    lines = []
    for t in tools:
        params = ", ".join(
            f"{k}: {v.get('description', '')}"
            for k, v in t["parameters"].get("properties", {}).items()
        )
        lines.append(f"- {t['name']}: {t['description']} Parameters: {{{params}}}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# LLM CALLER — Abstracts the API call. Supports real API or simulation.
# ---------------------------------------------------------------------------

# Check if we have a real API key
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # Cheap and fast for demos
USE_SIMULATION = not API_KEY


def call_llm_live(messages: list[dict]) -> str:
    """Call a real OpenAI-compatible API using httpx. No SDK dependency."""
    import httpx

    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0,       # Deterministic for agents — less creative, more reliable
            "max_tokens": 512,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# Simulated responses — hardcoded to demonstrate the loop without an API key
SIMULATED_RESPONSES = [
    # Step 1: Agent decides to calculate 25 * 47
    """Thought: I need to solve two things: calculate 25 * 47, and get the current time. Let me start with the calculation.
Action: calculator
Action Input: {"expression": "25 * 47"}""",

    # Step 2: After seeing 1175, agent decides to get the time
    """Thought: The calculation result is 1175. Now I need to get the current time.
Action: get_current_time
Action Input: {}""",

    # Step 3: Agent has both answers, finishes
    """Thought: I now have both pieces of information. 25 * 47 = 1175, and I have the current time. Let me provide the final answer.
Action: finish
Action Input: 25 * 47 = 1175, and the current time is {time}.""",
]

_sim_step = 0

def call_llm_simulated(messages: list[dict]) -> str:
    """Return hardcoded responses to simulate a ReAct agent without an API."""
    global _sim_step
    if _sim_step >= len(SIMULATED_RESPONSES):
        return "Thought: I'm done.\nAction: finish\nAction Input: Task complete."

    response = SIMULATED_RESPONSES[_sim_step]
    _sim_step += 1

    # Inject real time into the simulated response
    return response.replace("{time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def call_llm(messages: list[dict]) -> str:
    """Route to live or simulated LLM."""
    if USE_SIMULATION:
        return call_llm_simulated(messages)
    return call_llm_live(messages)

# ---------------------------------------------------------------------------
# ReAct PARSER — Extracts Thought, Action, and Action Input from LLM output.
# This is the "glue" between the LLM's text output and our tool execution.
# ---------------------------------------------------------------------------

def parse_react_output(text: str) -> dict:
    """
    Parse the LLM's ReAct-formatted output into structured data.
    Returns: {"thought": str, "action": str, "action_input": str | dict}

    Design choice: regex-based parsing is fragile but transparent.
    In production, prefer structured output (function calling / tool_use API).
    """
    thought = ""
    action = ""
    action_input = ""

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract Action
    action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()

    # Extract Action Input
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if input_match:
        raw = input_match.group(1).strip()
        # Try to parse as JSON (for tool parameters), fallback to string
        try:
            action_input = json.loads(raw)
        except json.JSONDecodeError:
            action_input = raw

    return {"thought": thought, "action": action, "action_input": action_input}

# ---------------------------------------------------------------------------
# REACT AGENT — The main loop. This is the core ~30 lines that matter.
# ---------------------------------------------------------------------------

def react_agent(question: str, max_iterations: int = 10, verbose: bool = True) -> str:
    """
    Run a ReAct agent loop to answer a question.

    The loop:
    1. Send context to LLM → LLM produces Thought + Action
    2. Parse the action
    3. If action is "finish", return the answer
    4. Otherwise, execute the tool, append observation, loop back to 1

    Args:
        question: The user's question / objective
        max_iterations: Safety limit to prevent infinite loops (ALWAYS set this)
        verbose: Print the trace for debugging
    """
    # Build the system prompt with tool descriptions
    system = SYSTEM_PROMPT.format(tools_description=build_tools_description(TOOLS))

    # Message history — this IS the agent's short-term memory
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if verbose:
        mode = "SIMULATED" if USE_SIMULATION else f"LIVE ({MODEL})"
        print(f"\n{'='*60}")
        print(f"  ReAct Agent — Mode: {mode}")
        print(f"  Question: {question}")
        print(f"{'='*60}\n")

    for i in range(max_iterations):
        # Step 1: Call the LLM
        llm_output = call_llm(messages)

        # Step 2: Parse the output
        parsed = parse_react_output(llm_output)

        if verbose:
            print(f"--- Step {i + 1} ---")
            print(f"Thought: {parsed['thought']}")
            print(f"Action:  {parsed['action']}")
            print(f"Input:   {parsed['action_input']}")

        # Step 3: Check if the agent wants to finish
        if parsed["action"].lower() == "finish":
            final_answer = parsed["action_input"]
            if isinstance(final_answer, dict):
                final_answer = json.dumps(final_answer)
            if verbose:
                print(f"\n{'='*60}")
                print(f"  FINAL ANSWER: {final_answer}")
                print(f"  (Completed in {i + 1} steps)")
                print(f"{'='*60}\n")
            return str(final_answer)

        # Step 4: Execute the tool
        action_name = parsed["action"].lower().strip()
        if action_name not in TOOL_FUNCTIONS:
            observation = f"Error: Unknown tool '{action_name}'. Available: {list(TOOL_FUNCTIONS.keys())}"
        else:
            params = parsed["action_input"] if isinstance(parsed["action_input"], dict) else {}
            try:
                observation = TOOL_FUNCTIONS[action_name](params)
            except Exception as e:
                # Feed errors back to the LLM so it can self-correct
                observation = f"Error executing {action_name}: {e}"

        if verbose:
            print(f"Observation: {observation}\n")

        # Step 5: Append the LLM output + observation to history
        # This is how the agent "remembers" what happened
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    # Safety: max iterations reached — the agent is stuck
    return "[Agent stopped: max iterations reached. Likely stuck in a loop.]"


# ---------------------------------------------------------------------------
# MAIN — Run the agent on a multi-step problem
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This question requires 2 tool calls: calculator + get_current_time
    question = "What is 25 * 47, and what time is it right now?"

    answer = react_agent(question)

    # Bonus: show how easy it is to add a new tool and ask another question
    print("\n--- Bonus: Single-step question ---\n")

    # Reset simulation counter for the second run
    if USE_SIMULATION:
        _sim_step = 0
        SIMULATED_RESPONSES.clear()
        SIMULATED_RESPONSES.extend([
            """Thought: I need to search for information about the MCP protocol.
Action: search
Action Input: {"query": "Model Context Protocol MCP Anthropic"}""",
            """Thought: The search returned a mock result since this is a demo. In a real setup, I would get actual search results. Let me provide what I know.
Action: finish
Action Input: MCP (Model Context Protocol) is an open standard by Anthropic for connecting AI models to external data sources and tools. It provides a unified interface for AI agents to interact with the world."""
        ])

    answer2 = react_agent("What is MCP (Model Context Protocol)?")

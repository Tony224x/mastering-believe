"""
Day 10 -- Solutions to the easy exercises for MCP.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/10-mcp.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# pylint: disable=wrong-import-position
day10 = import_module("10-mcp")
MiniMCPServer = day10.MiniMCPServer
MiniMCPClient = day10.MiniMCPClient
jsonrpc_request = day10.jsonrpc_request
jsonrpc_response = day10.jsonrpc_response
jsonrpc_error = day10.jsonrpc_error


# ===========================================================================
# SOLUTION 1 -- multiply with strict argument validation
# ===========================================================================

def solution_1() -> None:
    print("\n=== Solution 1: multiply with validation ===")
    server = MiniMCPServer(name="math-server")

    @server.tool(
        name="multiply",
        description="Multiply two integers",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    def multiply(a: int, b: int) -> int:
        return a * b

    # Monkey-patch the server's tool call dispatch to do strict validation
    original_tools_call = server._tools_call

    def strict_tools_call(req_id, params):
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name == "multiply":
            if "a" not in args or "b" not in args:
                return jsonrpc_error(req_id, -32602, "multiply requires arguments a and b")
            if not (isinstance(args["a"], int) and isinstance(args["b"], int)):
                return jsonrpc_error(req_id, -32602, "multiply arguments must be integers")
            if isinstance(args["a"], bool) or isinstance(args["b"], bool):
                # bool is a subclass of int -- reject explicitly
                return jsonrpc_error(req_id, -32602, "multiply arguments must be integers")
        return original_tools_call(req_id, params)

    server._tools_call = strict_tools_call  # type: ignore

    client = MiniMCPClient(server=server)
    client.initialize()

    # Good case
    print("  multiply(6, 7) =", client.call_tool("multiply", {"a": 6, "b": 7}))

    # Bad type
    bad_type_resp = server.handle(
        jsonrpc_request("tools/call", {"name": "multiply", "arguments": {"a": "x", "b": 7}}, 100)
    )
    print("  multiply('x', 7) error:", bad_type_resp["error"]["code"], bad_type_resp["error"]["message"])

    # Missing arg
    missing_resp = server.handle(
        jsonrpc_request("tools/call", {"name": "multiply", "arguments": {"a": 6}}, 101)
    )
    print("  multiply(6) error:", missing_resp["error"]["code"], missing_resp["error"]["message"])


# ===========================================================================
# SOLUTION 2 -- dynamic resources (time://now, time://uptime)
# ===========================================================================

def solution_2() -> None:
    print("\n=== Solution 2: dynamic resources ===")
    server = MiniMCPServer(name="time-server")
    start_time = time.time()

    @server.resource(uri="time://now", name="current time", mime_type="text/plain")
    def now() -> str:
        # Recomputed every read -- lazy, no capture at registration time
        return datetime.now().isoformat()

    @server.resource(uri="time://uptime", name="server uptime seconds", mime_type="text/plain")
    def uptime() -> str:
        return f"{time.time() - start_time:.3f}"

    client = MiniMCPClient(server=server)
    client.initialize()

    t1 = client.read_resource("time://now")
    time.sleep(0.01)
    t2 = client.read_resource("time://now")
    assert t1 != t2, "time://now should be different on two reads"
    print(f"  time://now read 1: {t1}")
    print(f"  time://now read 2: {t2}")

    u1 = float(client.read_resource("time://uptime"))
    time.sleep(0.01)
    u2 = float(client.read_resource("time://uptime"))
    assert u2 > u1, "uptime should grow"
    print(f"  time://uptime 1: {u1:.3f}s, 2: {u2:.3f}s")


# ===========================================================================
# SOLUTION 3 -- parameterized code_review prompt
# ===========================================================================

def solution_3() -> None:
    print("\n=== Solution 3: parameterized prompt ===")
    server = MiniMCPServer(name="review-server")

    @server.prompt(
        name="code_review",
        description="Generate a code review prompt for a given language and focus",
        arguments=[
            {"name": "language", "required": True},
            {"name": "file_content", "required": True},
            {"name": "focus", "required": True},
        ],
    )
    def code_review(language: str, file_content: str, focus: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    f"You are a senior {language} reviewer. "
                    f"Focus on {focus}. Be specific and actionable."
                ),
            },
            {
                "role": "user",
                "content": f"Review this {language} code:\n\n{file_content}",
            },
        ]

    client = MiniMCPClient(server=server)
    client.initialize()

    m1 = client.get_prompt(
        "code_review",
        {"language": "python", "file_content": "def foo(): return 1", "focus": "bugs"},
    )
    m2 = client.get_prompt(
        "code_review",
        {"language": "typescript", "file_content": "const x = 1;", "focus": "style"},
    )

    for label, messages in [("python/bugs", m1), ("typescript/style", m2)]:
        print(f"\n  --- {label} ---")
        for m in messages:
            print(f"  [{m['role']}] {m['content']}")

    assert "python" in m1[0]["content"] and "bugs" in m1[0]["content"]
    assert "typescript" in m2[0]["content"] and "style" in m2[0]["content"]
    assert m1[1]["content"] != m2[1]["content"]


if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

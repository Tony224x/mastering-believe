"""
Day 10 -- MCP (Model Context Protocol): minimal server and client from scratch.

Demonstrates:
  1. A tiny MCP-shaped server implementing the protocol surface:
       - tool registration + dispatch
       - resource registration + read
       - prompt registration + get
       - JSON-RPC 2.0 message format
       - stdio-style message bus (mocked with a queue so the demo runs in-process)
  2. A MiniMCPClient that knows how to talk to the server
  3. An end-to-end demo: client discovers tools, calls them, reads a resource,
     instantiates a prompt
  4. Optional real MCP example if `mcp` is installed

The goal is pedagogical: you should finish this file understanding what actually
goes over the wire. Once you know that, the real SDK is just syntactic sugar.

Run:
    python domains/agentic-ai/02-code/10-mcp.py
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Optional: real MCP SDK -- show a native example if it is available
# ---------------------------------------------------------------------------

HAS_MCP = False
try:
    import mcp  # noqa: F401
    HAS_MCP = True
except ImportError:
    pass


# ===========================================================================
# 1. JSON-RPC 2.0 HELPERS
# ===========================================================================

def jsonrpc_request(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    """Build a JSON-RPC 2.0 request object."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_response(req_id: int, result: Any) -> dict:
    """Build a successful JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: int, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


# ===========================================================================
# 2. MINI MCP SERVER
# ===========================================================================

@dataclass
class ToolDef:
    """Description of a tool exposed by the server."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]


@dataclass
class ResourceDef:
    """Description of a resource exposed by the server."""
    uri: str
    name: str
    mime_type: str
    reader: Callable[[], str]


@dataclass
class PromptDef:
    """Description of a prompt template exposed by the server."""
    name: str
    description: str
    arguments: list[dict]
    builder: Callable[..., list[dict]]


class MiniMCPServer:
    """
    A minimal MCP-shaped server implementing the core of the protocol.

    Not feature-complete. Supports:
      - initialize / initialized
      - tools/list, tools/call
      - resources/list, resources/read
      - prompts/list, prompts/get

    Transport is abstracted as a simple "receive message, return message"
    interface so we can drive it from a queue in a unit test, from stdio in
    production, or from HTTP if you wrap it accordingly.
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, ToolDef] = {}
        self._resources: dict[str, ResourceDef] = {}
        self._prompts: dict[str, PromptDef] = {}
        self._initialized = False

    # --- registration API -------------------------------------------------

    def tool(self, name: str, description: str, input_schema: dict):
        """Decorator to register a tool handler."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=fn,
            )
            return fn
        return decorator

    def resource(self, uri: str, name: str, mime_type: str = "text/plain"):
        """Decorator to register a resource reader."""
        def decorator(fn: Callable[[], str]) -> Callable[[], str]:
            self._resources[uri] = ResourceDef(
                uri=uri, name=name, mime_type=mime_type, reader=fn
            )
            return fn
        return decorator

    def prompt(self, name: str, description: str, arguments: list[dict]):
        """Decorator to register a prompt template."""
        def decorator(fn: Callable[..., list[dict]]) -> Callable[..., list[dict]]:
            self._prompts[name] = PromptDef(
                name=name,
                description=description,
                arguments=arguments,
                builder=fn,
            )
            return fn
        return decorator

    # --- request handling -------------------------------------------------

    def handle(self, message: dict) -> dict:
        """
        Dispatch a single JSON-RPC request to the appropriate handler.
        Returns the JSON-RPC response dict.
        """
        method = message.get("method", "")
        req_id = message.get("id", 0)
        params = message.get("params", {}) or {}

        try:
            if method == "initialize":
                return self._initialize(req_id, params)
            if method == "initialized":
                # Notification -- no response
                self._initialized = True
                return {}
            if not self._initialized and method != "initialize":
                return jsonrpc_error(req_id, -32002, "server not initialized")
            if method == "tools/list":
                return self._tools_list(req_id)
            if method == "tools/call":
                return self._tools_call(req_id, params)
            if method == "resources/list":
                return self._resources_list(req_id)
            if method == "resources/read":
                return self._resources_read(req_id, params)
            if method == "prompts/list":
                return self._prompts_list(req_id)
            if method == "prompts/get":
                return self._prompts_get(req_id, params)
            return jsonrpc_error(req_id, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001
            return jsonrpc_error(req_id, -32603, f"internal error: {exc}")

    # --- individual method handlers ---------------------------------------

    def _initialize(self, req_id: int, params: dict) -> dict:
        return jsonrpc_response(
            req_id,
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "serverInfo": {"name": self.name, "version": "0.1.0"},
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
            },
        )

    def _tools_list(self, req_id: int) -> dict:
        return jsonrpc_response(
            req_id,
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in self._tools.values()
                ]
            },
        )

    def _tools_call(self, req_id: int, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        tool = self._tools.get(name or "")
        if tool is None:
            return jsonrpc_error(req_id, -32602, f"unknown tool: {name}")
        # Call the handler. In a real MCP server we would validate arguments
        # against tool.input_schema, but that is out of scope for this demo.
        result = tool.handler(**arguments)
        return jsonrpc_response(
            req_id,
            {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            },
        )

    def _resources_list(self, req_id: int) -> dict:
        return jsonrpc_response(
            req_id,
            {
                "resources": [
                    {"uri": r.uri, "name": r.name, "mimeType": r.mime_type}
                    for r in self._resources.values()
                ]
            },
        )

    def _resources_read(self, req_id: int, params: dict) -> dict:
        uri = params.get("uri")
        resource = self._resources.get(uri or "")
        if resource is None:
            return jsonrpc_error(req_id, -32602, f"unknown resource: {uri}")
        text = resource.reader()
        return jsonrpc_response(
            req_id,
            {
                "contents": [
                    {"uri": resource.uri, "mimeType": resource.mime_type, "text": text}
                ]
            },
        )

    def _prompts_list(self, req_id: int) -> dict:
        return jsonrpc_response(
            req_id,
            {
                "prompts": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "arguments": p.arguments,
                    }
                    for p in self._prompts.values()
                ]
            },
        )

    def _prompts_get(self, req_id: int, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        prompt = self._prompts.get(name or "")
        if prompt is None:
            return jsonrpc_error(req_id, -32602, f"unknown prompt: {name}")
        messages = prompt.builder(**arguments)
        return jsonrpc_response(
            req_id, {"description": prompt.description, "messages": messages}
        )


# ===========================================================================
# 3. MINI MCP CLIENT
# ===========================================================================

@dataclass
class MiniMCPClient:
    """
    A tiny client that drives a MiniMCPServer through a mocked stdio bus.
    In production this would read/write real stdin/stdout and buffer by line.
    """
    server: MiniMCPServer
    _req_id: int = 0
    inbox: deque = field(default_factory=deque)
    outbox: deque = field(default_factory=deque)

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send(self, message: dict) -> dict:
        """Send a request via the mocked bus and get the response."""
        # In a real setup we would write to stdin and read from stdout.
        # Here we just call the handler directly so you can see the shape.
        raw = json.dumps(message)
        self.outbox.append(raw)
        response_raw = json.dumps(self.server.handle(json.loads(raw)))
        self.inbox.append(response_raw)
        return json.loads(response_raw)

    # --- high-level convenience API ---------------------------------------

    def initialize(self) -> dict:
        resp = self._send(
            jsonrpc_request("initialize", {"protocolVersion": "2024-11-05"}, self._next_id())
        )
        # Send the "initialized" notification (no response expected)
        self._send(jsonrpc_request("initialized", None, self._next_id()))
        return resp["result"]

    def list_tools(self) -> list[dict]:
        resp = self._send(jsonrpc_request("tools/list", None, self._next_id()))
        return resp["result"]["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._send(
            jsonrpc_request("tools/call", {"name": name, "arguments": arguments}, self._next_id())
        )
        return resp["result"]["content"][0]["text"]

    def list_resources(self) -> list[dict]:
        resp = self._send(jsonrpc_request("resources/list", None, self._next_id()))
        return resp["result"]["resources"]

    def read_resource(self, uri: str) -> str:
        resp = self._send(
            jsonrpc_request("resources/read", {"uri": uri}, self._next_id())
        )
        return resp["result"]["contents"][0]["text"]

    def list_prompts(self) -> list[dict]:
        resp = self._send(jsonrpc_request("prompts/list", None, self._next_id()))
        return resp["result"]["prompts"]

    def get_prompt(self, name: str, arguments: dict | None = None) -> list[dict]:
        resp = self._send(
            jsonrpc_request(
                "prompts/get", {"name": name, "arguments": arguments or {}}, self._next_id()
            )
        )
        return resp["result"]["messages"]


# ===========================================================================
# 4. BUILD A CONCRETE SERVER
# ===========================================================================

def build_demo_server() -> MiniMCPServer:
    """Build a server with one tool, one resource and one prompt."""
    server = MiniMCPServer(name="demo-server")

    @server.tool(
        name="add",
        description="Add two integers and return the sum.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool(
        name="greet",
        description="Return a greeting for a given name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    # Resources are inert data that the client can pull into the LLM context.
    _NOTES = "Acme ships AI consulting and SaaS. Main product: acme-immo."

    @server.resource(
        uri="notes://acme",
        name="Acme notes",
        mime_type="text/plain",
    )
    def get_notes() -> str:
        return _NOTES

    @server.prompt(
        name="review",
        description="Review an object for quality",
        arguments=[{"name": "target", "required": True}],
    )
    def review_prompt(target: str) -> list[dict]:
        return [
            {"role": "system", "content": "You are a senior reviewer."},
            {"role": "user", "content": f"Review the following: {target}"},
        ]

    return server


# ===========================================================================
# 5. END-TO-END DEMO
# ===========================================================================

def demo() -> None:
    print("=" * 70)
    print(f"Backends available: mcp={HAS_MCP} -- using MiniMCPServer/Client")
    print("=" * 70)

    server = build_demo_server()
    client = MiniMCPClient(server=server)

    print("\n--- 1. initialize ---")
    info = client.initialize()
    print(json.dumps(info, indent=2))

    print("\n--- 2. list tools ---")
    tools = client.list_tools()
    for t in tools:
        print(f"  - {t['name']}: {t['description']}")

    print("\n--- 3. call tools ---")
    print("  add(3, 4) =", client.call_tool("add", {"a": 3, "b": 4}))
    print("  greet('Alex') =", client.call_tool("greet", {"name": "Alex"}))

    print("\n--- 4. list + read resources ---")
    resources = client.list_resources()
    for r in resources:
        print(f"  - {r['uri']} ({r['name']})")
    content = client.read_resource("notes://acme")
    print(f"  content of notes://acme -> {content}")

    print("\n--- 5. list + get prompts ---")
    prompts = client.list_prompts()
    for p in prompts:
        print(f"  - {p['name']}: {p['description']}")
    messages = client.get_prompt("review", {"target": "acme-immo business model"})
    for m in messages:
        print(f"  [{m['role']}] {m['content']}")

    print("\n--- 6. error path ---")
    resp = server.handle(jsonrpc_request("tools/call", {"name": "nope"}, 99))
    print(json.dumps(resp, indent=2))

    if HAS_MCP:
        print("\n--- 7. real MCP SDK is installed -- see docs for FastMCP example ---")
    else:
        print("\n(Install `pip install mcp` to also run a real MCP server.)")


if __name__ == "__main__":
    demo()

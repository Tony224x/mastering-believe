"""
Solutions -- Day 10 (MEDIUM): MCP (Model Context Protocol)

Contains solutions for:
  - Medium Ex 1: Generic JSON-Schema mini-validator wired into tools/call
  - Medium Ex 2: Sampling -- the server borrows the host's LLM
  - Medium Ex 3: Multi-server router (aggregating client + namespacing)

Everything runs OFFLINE: the "LLM" used by sampling is a deterministic mock,
the transport is in-process, and there are no network calls or API keys. The
protocol shapes (JSON-RPC 2.0, tools/list, tools/call) mirror 02-code/10-mcp.py.

Run:  python 03-exercises/solutions/10-mcp-medium.py
Each solution is self-contained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

# ==========================================================================
# SHARED -- minimal JSON-RPC + a tiny MCP server (same shape as 02-code)
# ==========================================================================


def jsonrpc_request(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_response(req_id: int, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]


@dataclass
class ResourceDef:
    uri: str
    name: str
    reader: Callable[[], str]
    mime_type: str = "text/plain"


# ==========================================================================
# MEDIUM EXERCISE 1 -- Generic JSON-Schema mini-validator
# ==========================================================================
#
# The shipped server calls handler(**arguments) with no validation, so a bad
# type explodes inside the handler with an opaque TypeError. A real MCP server
# validates against inputSchema FIRST and returns -32602 (invalid params).

# Map JSON Schema primitive types to Python checks. The bool/int trap matters:
# in Python `True` IS an int, so an integer field must explicitly reject bool.
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_against_schema(schema: dict, arguments: dict) -> list[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # 1. Required fields present
    for field_name in required:
        if field_name not in arguments:
            errors.append(f"missing required property: '{field_name}'")

    # 2. Each provided argument: known + correct type
    for key, value in arguments.items():
        if key not in properties:
            errors.append(f"unexpected property: '{key}'")
            continue
        expected_type = properties[key].get("type")
        check = _TYPE_CHECKS.get(expected_type)
        if check and not check(value):
            errors.append(
                f"property '{key}' must be {expected_type}, got "
                f"{type(value).__name__} ({value!r})"
            )
    return errors


class ValidatingServer:
    """Minimal server whose tools/call validates against inputSchema first."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def handle_tools_call(self, req_id: int, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        tool = self._tools.get(name or "")
        if tool is None:
            return jsonrpc_error(req_id, -32602, f"unknown tool: {name}")

        # Validate BEFORE touching the handler -- this is the whole point.
        errors = validate_against_schema(tool.input_schema, arguments)
        if errors:
            return jsonrpc_error(req_id, -32602, f"invalid params: {errors}")

        result = tool.handler(**arguments)
        return jsonrpc_response(
            req_id, {"content": [{"type": "text", "text": str(result)}], "isError": False}
        )


def medium_ex1_schema_validation() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Generic JSON-Schema validation")
    print("=" * 60)

    server = ValidatingServer()
    server.register(
        ToolDef(
            name="add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            handler=lambda a, b: a + b,
        )
    )

    cases = [
        ("valid", {"a": 3, "b": 4}, True),
        ("string type", {"a": "3", "b": 4}, False),
        ("missing required", {"a": 3}, False),
        ("unexpected prop", {"a": 3, "b": 4, "c": 9}, False),
        ("bool not integer", {"a": True, "b": 4}, False),
    ]
    print()
    for label, args, should_pass in cases:
        resp = server.handle_tools_call(1, {"name": "add", "arguments": args})
        ok = "result" in resp
        verdict = "OK   " if ok else "ERROR"
        detail = (
            resp["result"]["content"][0]["text"]
            if ok
            else resp["error"]["message"]
        )
        print(f"  {verdict} | {label:18s} | {detail}")
        assert ok == should_pass, f"{label}: expected pass={should_pass}, got {ok}"
        if not ok:
            assert resp["error"]["code"] == -32602

    print("\n  PASS -- handler never runs on invalid params; bool!=integer enforced.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Sampling: the server borrows the host's LLM
# ==========================================================================
#
# Sampling INVERTS the usual direction: normally client->server, but here the
# server asks the client to invoke its host LLM (sampling/createMessage). The
# server itself has zero generation logic -- it only orchestrates.


class SamplingServer:
    """Server that can ask the host (client) to run its LLM via sampling."""

    def __init__(self) -> None:
        self._resources: dict[str, ResourceDef] = {}
        self._tools: dict[str, ToolDef] = {}
        # Branched by the client at initialize() time. The server calls this
        # to perform sampling/createMessage instead of owning an LLM.
        self._on_sampling_request: Callable[[str], str] | None = None

    def register_resource(self, res: ResourceDef) -> None:
        self._resources[res.uri] = res

    def register_tool(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def set_sampling_handler(self, handler: Callable[[str], str]) -> None:
        self._on_sampling_request = handler

    def request_sampling(self, prompt: str) -> str:
        """server -> client : 'please run your LLM on this prompt'."""
        if self._on_sampling_request is None:
            raise RuntimeError("client did not expose a sampler (no sampling cap)")
        return self._on_sampling_request(prompt)

    def read_resource(self, uri: str) -> str:
        res = self._resources.get(uri)
        if res is None:
            raise KeyError(f"unknown resource: {uri}")
        return res.reader()

    def call_tool(self, name: str, arguments: dict) -> str:
        return str(self._tools[name].handler(**arguments))


@dataclass
class SamplingClient:
    """Host that owns the LLM (mock). It branches its sampler onto the server."""

    server: SamplingServer
    sampler: Callable[[str], str]
    sampler_calls: int = 0

    def initialize(self) -> None:
        # During the handshake the client advertises the 'sampling' capability
        # by giving the server a way to call back into the host LLM.
        def _wrapped(prompt: str) -> str:
            self.sampler_calls += 1
            return self.sampler(prompt)

        self.server.set_sampling_handler(_wrapped)

    def call_tool(self, name: str, arguments: dict | None = None) -> str:
        return self.server.call_tool(name, arguments or {})


def mock_host_llm(prompt: str) -> str:
    """Deterministic mock of the host LLM: summarize = first 5 words."""
    # In production this would be claude-sonnet on the host side.
    words = prompt.replace("Summarize this:", "").split()
    return "SUMMARY: " + " ".join(words[:5]) + "..."


def medium_ex2_sampling() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: Sampling -- server borrows the host LLM")
    print("=" * 60)

    server = SamplingServer()
    server.register_resource(
        ResourceDef(
            uri="notes://acme",
            name="Acme notes",
            reader=lambda: "Acme ships AI consulting and SaaS. Main product: acme-immo.",
        )
    )

    # The tool combines resource-read + sampling. It has NO generation logic.
    def summarize_notes() -> str:
        content = server.read_resource("notes://acme")
        return server.request_sampling(f"Summarize this: {content}")

    server.register_tool(
        ToolDef(
            name="summarize_notes",
            description="Summarize the internal notes using the host LLM",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=summarize_notes,
        )
    )

    client = SamplingClient(server=server, sampler=mock_host_llm)
    client.initialize()

    print("\n  Calling summarize_notes() (server has no LLM of its own)...")
    summary = client.call_tool("summarize_notes")
    print(f"  -> {summary}")

    assert summary.startswith("SUMMARY:"), summary
    assert "Acme" in summary, "summary should reflect the resource content"
    assert client.sampler_calls == 1, f"expected exactly 1 sampling call, got {client.sampler_calls}"
    print(f"\n  Sampler invoked {client.sampler_calls}x (server stayed LLM-free).")
    print("  PASS -- resource-read + sampling, no generation logic on the server.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Multi-server router (aggregating client)
# ==========================================================================
#
# A host typically connects to MANY servers. The router lists every server's
# tools, namespaces collisions as 'server.tool', keeps unique names callable
# bare, and refuses ambiguous bare calls.


class SimpleToolServer:
    """A tiny server exposing named tools (stand-in for a full MiniMCPServer)."""

    def __init__(self, server_id: str) -> None:
        self.server_id = server_id
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        return str(self._tools[name].handler(**arguments))


class MCPRouter:
    """Aggregates several servers, namespaces collisions, routes calls."""

    def __init__(self, servers: list[SimpleToolServer]) -> None:
        self._servers = {s.server_id: s for s in servers}
        # name -> set of server_ids exposing it (to detect collisions)
        self._owners: dict[str, set[str]] = {}
        for s in servers:
            for tool in s.list_tools():
                self._owners.setdefault(tool["name"], set()).add(s.server_id)

    def list_all_tools(self) -> list[dict]:
        out: list[dict] = []
        for server_id, server in self._servers.items():
            for tool in server.list_tools():
                bare = tool["name"]
                # Always reachable namespaced; bare only shown when unique.
                out.append(
                    {
                        "name": f"{server_id}.{bare}",
                        "bare_name": bare,
                        "server": server_id,
                        "unique": len(self._owners[bare]) == 1,
                    }
                )
        return out

    def call(self, tool_name: str, arguments: dict) -> str:
        if "." in tool_name:
            server_id, bare = tool_name.split(".", 1)
            if server_id not in self._servers:
                raise KeyError(f"unknown server: {server_id}")
            return self._servers[server_id].call(bare, arguments)

        # Bare name: only allowed if exactly one server owns it.
        owners = self._owners.get(tool_name)
        if not owners:
            raise KeyError(f"unknown tool: {tool_name}")
        if len(owners) > 1:
            raise ValueError(
                f"ambiguous tool '{tool_name}': owned by {sorted(owners)} -- "
                f"use a namespaced name like '{sorted(owners)[0]}.{tool_name}'"
            )
        return self._servers[next(iter(owners))].call(tool_name, arguments)


def medium_ex3_multi_server_router() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: Multi-server router (namespacing + collisions)")
    print("=" * 60)

    math_srv = SimpleToolServer("math")
    math_srv.register(ToolDef("add", "add", {}, lambda a, b: a + b))
    math_srv.register(ToolDef("search", "search formulas", {}, lambda q: f"math:{q}"))

    kb_srv = SimpleToolServer("kb")
    kb_srv.register(ToolDef("lookup", "kb lookup", {}, lambda key: f"kb-entry:{key}"))
    kb_srv.register(ToolDef("search", "search docs", {}, lambda q: f"kb:{q}"))

    router = MCPRouter([math_srv, kb_srv])

    print("\n  Aggregated tool view:")
    for t in router.list_all_tools():
        tag = "unique" if t["unique"] else "COLLISION"
        print(f"    {t['name']:14s} (server={t['server']}, bare-callable={t['unique']}) [{tag}]")

    print("\n  Routing:")
    print(f"    add(2,3)         -> {router.call('add', {'a': 2, 'b': 3})}")
    print(f"    math.search(x)   -> {router.call('math.search', {'q': 'derivative'})}")
    print(f"    kb.search(x)     -> {router.call('kb.search', {'q': 'webhook'})}")

    assert router.call("add", {"a": 2, "b": 3}) == "5"   # SimpleToolServer.call stringifies
    assert router.call("math.search", {"q": "z"}) == "math:z"
    assert router.call("kb.search", {"q": "z"}) == "kb:z"

    # Bare 'search' is ambiguous -> must raise.
    try:
        router.call("search", {"q": "z"})
        raise AssertionError("ambiguous bare call should have raised")
    except ValueError as exc:
        print(f"    search(x)        -> ValueError (as expected): {str(exc)[:55]}...")

    print("\n  PASS -- unique tool bare-callable, collision namespaced, ambiguity rejected.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 10 MEDIUM Solutions -- MCP")
    print("#" * 60)

    medium_ex1_schema_validation()
    medium_ex2_sampling()
    medium_ex3_multi_server_router()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")

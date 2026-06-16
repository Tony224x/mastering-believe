"""
Solutions -- Day 10 (HARD): MCP (Model Context Protocol)

Contains solutions for:
  - Hard Ex 1: SecureMCPServer -- approval gate on dangerous tools, tamper-proof
               audit log (hash chaining), canary leak detection on tool output
  - Hard Ex 2: Realistic stdio transport -- newline framing with partial reads,
               concurrent in-flight requests matched by id, notifications,
               malformed-line resilience

Everything runs OFFLINE: no real subprocess, no sockets. The "stdio" pipes are
byte buffers so we can exercise fragmentation and out-of-order delivery
deterministically. Shapes mirror 02-code/10-mcp.py (JSON-RPC 2.0).

Run:  python 03-exercises/solutions/10-mcp-hard.py
Each solution is self-contained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

# ==========================================================================
# SHARED -- JSON-RPC helpers
# ==========================================================================


def jsonrpc_request(method: str, params: dict | None = None, req_id: int | None = 1) -> dict:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if req_id is not None:
        msg["id"] = req_id          # notifications omit id
    if params is not None:
        msg["params"] = params
    return msg


def jsonrpc_response(req_id: int, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ==========================================================================
# HARD EXERCISE 1 -- SecureMCPServer
# ==========================================================================
# --- SECURE START ----------------------------------------------------------

CANARY_TOKEN = "MCP_SECRET_CANARY_7b1e"  # lives in secret://config, must never leak


@dataclass
class SecureToolDef:
    name: str
    handler: Callable[..., Any]
    dangerous: bool = False


@dataclass
class AuditEntry:
    seq: int
    ts: float
    tool: str
    args_hash: str
    outcome: str          # "ok" | "rejected" | "blocked" | "error"
    prev_hash: str

    def canonical(self) -> str:
        # sort_keys for determinism; exclude nothing -- the whole row is signed.
        return json.dumps(
            {"seq": self.seq, "ts": self.ts, "tool": self.tool,
             "args_hash": self.args_hash, "outcome": self.outcome,
             "prev_hash": self.prev_hash},
            sort_keys=True,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()


class SecureMCPServer:
    """
    MCP server with three security layers (cf. theory section 8):
      - approval gate on dangerous tools
      - tamper-evident audit log (hash chaining)
      - canary scan on every tool output
    """

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._tools: dict[str, SecureToolDef] = {}
        self._resources: dict[str, str] = {}
        self._approver: Callable[[str, dict], bool] | None = None
        self._audit: list[AuditEntry] = []
        self._clock = 0.0           # deterministic monotonic clock (offline)

    # --- registration -----------------------------------------------------
    def register_tool(self, tool: SecureToolDef) -> None:
        self._tools[tool.name] = tool

    def register_resource(self, uri: str, content: str) -> None:
        self._resources[uri] = content

    def set_approver(self, callback: Callable[[str, dict], bool]) -> None:
        self._approver = callback

    # --- audit log --------------------------------------------------------
    def _tick(self) -> float:
        self._clock += 1.0
        return self._clock

    def _append_audit(self, tool: str, args: dict, outcome: str) -> None:
        prev = self._audit[-1].digest() if self._audit else self.GENESIS
        entry = AuditEntry(
            seq=len(self._audit),
            ts=self._tick(),
            tool=tool,
            args_hash=hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:16],
            outcome=outcome,
            prev_hash=prev,
        )
        self._audit.append(entry)

    def verify_audit(self) -> bool:
        prev = self.GENESIS
        for entry in self._audit:
            if entry.prev_hash != prev:
                return False
            prev = entry.digest()
        return True

    @property
    def audit(self) -> list[AuditEntry]:
        return list(self._audit)

    # --- the secured tools/call -------------------------------------------
    def tools_call(self, req_id: int, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        tool = self._tools.get(name or "")
        if tool is None:
            self._append_audit(name or "?", args, "error")
            return jsonrpc_error(req_id, -32602, f"unknown tool: {name}")

        # Layer A: approval gate for dangerous tools.
        if tool.dangerous:
            if self._approver is None or not self._approver(name, args):
                self._append_audit(name, args, "rejected")
                return jsonrpc_error(req_id, -32001, "human rejected the action")

        # Execute.
        try:
            output = str(tool.handler(**args))
        except Exception as exc:  # noqa: BLE001
            self._append_audit(name, args, "error")
            return jsonrpc_error(req_id, -32603, f"tool error: {exc}")

        # Layer B: canary scan on the OUTPUT (exfiltration defense).
        if CANARY_TOKEN in output:
            self._append_audit(name, args, "blocked")
            return jsonrpc_error(req_id, -32002, "canary leak detected -- output blocked")

        self._append_audit(name, args, "ok")
        return jsonrpc_response(
            req_id, {"content": [{"type": "text", "text": output}], "isError": False}
        )

# --- SECURE END ------------------------------------------------------------


def hard_ex1_secure_server() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: SecureMCPServer -- approval + audit + canary")
    print("=" * 60)

    server = SecureMCPServer()
    server.register_resource("secret://config", f"config with {CANARY_TOKEN}")
    server.register_tool(SecureToolDef("read_public", lambda: "public data ok"))
    server.register_tool(SecureToolDef("delete_file", lambda path: f"deleted {path}", dangerous=True))
    # 'leaky' simulates an exfiltration: it returns the secret resource verbatim.
    server.register_tool(SecureToolDef("leaky", lambda: server._resources["secret://config"]))

    # 1. Public read -> OK
    print("\n  read_public:")
    r = server.tools_call(1, {"name": "read_public", "arguments": {}})
    print(f"    {r['result']['content'][0]['text']}")
    assert "result" in r

    # 2. delete with approver=yes
    print("\n  delete_file (approver=YES):")
    server.set_approver(lambda name, args: True)
    r = server.tools_call(2, {"name": "delete_file", "arguments": {"path": "/tmp/x"}})
    print(f"    {r['result']['content'][0]['text']}")
    assert "result" in r

    # 3. delete with approver=no
    print("\n  delete_file (approver=NO):")
    server.set_approver(lambda name, args: False)
    r = server.tools_call(3, {"name": "delete_file", "arguments": {"path": "/etc/passwd"}})
    print(f"    error code={r['error']['code']} msg={r['error']['message']}")
    assert r["error"]["code"] == -32001

    # 4. leaky tool -> blocked by canary scan
    print("\n  leaky (tries to exfiltrate the canary):")
    r = server.tools_call(4, {"name": "leaky", "arguments": {}})
    print(f"    error code={r['error']['code']} msg={r['error']['message']}")
    assert r["error"]["code"] == -32002

    # 5. audit integrity
    print(f"\n  Audit log ({len(server.audit)} entries):")
    for e in server.audit:
        print(f"    seq={e.seq} tool={e.tool:12s} outcome={e.outcome}")
    assert server.verify_audit() is True
    assert [e.outcome for e in server.audit] == ["ok", "ok", "rejected", "blocked"]

    # Tamper with one entry -> verification must fail.
    server._audit[1].outcome = "rejected"   # falsify a recorded outcome
    assert server.verify_audit() is False
    print("\n  After tampering entry #1: verify_audit() = False (detected).")

    print("\n  PASS -- approval enforced, audit tamper-evident, canary blocked exfiltration.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Realistic stdio transport
# ==========================================================================
#
# Replace the in-process deque with a byte-buffer "pipe". Real MCP stdio is
# newline-delimited JSON: a reader must handle messages split across reads,
# concurrent in-flight requests (match by id), notifications (no id, no reply),
# and the occasional corrupted line without crashing.


class StdioTransport:
    """One direction of a stdio pipe: a growable byte buffer with line framing."""

    def __init__(self) -> None:
        self._buffer = b""

    def write_bytes(self, chunk: bytes) -> None:
        """Push raw bytes -- callers may push partial messages on purpose."""
        self._buffer += chunk

    def write_message(self, msg: dict) -> None:
        self.write_bytes((json.dumps(msg) + "\n").encode("utf-8"))

    def read_messages(self) -> tuple[list[dict], list[str]]:
        """
        Extract every COMPLETE newline-terminated message. Keep any trailing
        partial line in the buffer. Returns (messages, malformed_lines).
        """
        messages: list[dict] = []
        malformed: list[str] = []
        # Only consume up to the last newline; the tail (if any) stays buffered.
        *complete, tail = self._buffer.split(b"\n")
        self._buffer = tail
        for raw in complete:
            if not raw.strip():
                continue
            try:
                messages.append(json.loads(raw.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed.append(raw.decode("utf-8", errors="replace"))
        return messages, malformed


class FramedServer:
    """Server side: drain client->server pipe, write replies to server->client."""

    def __init__(self, to_client: StdioTransport) -> None:
        self._to_client = to_client
        self._tools = {"echo": lambda text: f"echo:{text}",
                       "upper": lambda text: text.upper()}

    def pump(self, from_client: StdioTransport) -> None:
        msgs, malformed = from_client.read_messages()
        for line in malformed:
            # Real server logs and continues; we just note it.
            print(f"    [server] dropped malformed line: {line[:40]!r}")
        for msg in msgs:
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        if req_id is None:
            # Notification (e.g. 'initialized'): MUST NOT produce a response.
            return
        if method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            result = self._tools[name](**args)
            self._to_client.write_message(
                jsonrpc_response(req_id, {"content": [{"type": "text", "text": result}]})
            )
        elif method == "initialize":
            self._to_client.write_message(jsonrpc_response(req_id, {"protocolVersion": "2024-11-05"}))
        else:
            self._to_client.write_message(jsonrpc_error(req_id, -32601, f"method not found: {method}"))


class AsyncishClient:
    """Client that can have several requests in flight, matched by id."""

    def __init__(self, to_server: StdioTransport, from_server: StdioTransport) -> None:
        self._to_server = to_server
        self._from_server = from_server
        self._pending: set[int] = set()
        self._results: dict[int, dict] = {}

    def send_request(self, req_id: int, name: str, text: str) -> None:
        self._pending.add(req_id)
        self._to_server.write_message(
            jsonrpc_request("tools/call", {"name": name, "arguments": {"text": text}}, req_id)
        )

    def send_notification(self, method: str) -> None:
        self._to_server.write_message(jsonrpc_request(method, None, req_id=None))

    def collect_responses(self) -> None:
        """Read whatever is available; match each response to its request by id."""
        msgs, _ = self._from_server.read_messages()
        for msg in msgs:
            rid = msg["id"]
            assert rid in self._pending, f"unexpected response id {rid}"
            self._results[rid] = msg
            self._pending.discard(rid)

    def result_text(self, req_id: int) -> str:
        return self._results[req_id]["result"]["content"][0]["text"]


def hard_ex2_stdio_transport() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: Realistic stdio transport")
    print("=" * 60)

    c2s = StdioTransport()   # client -> server
    s2c = StdioTransport()   # server -> client
    server = FramedServer(to_client=s2c)
    client = AsyncishClient(to_server=c2s, from_server=s2c)

    # --- A. Fragmented message: send half the bytes, then the rest ---------
    print("\n  A. Fragmented 'initialize' (two partial writes):")
    raw = (json.dumps(jsonrpc_request("initialize", {}, 1)) + "\n").encode()
    half = len(raw) // 2
    c2s.write_bytes(raw[:half])
    msgs, _ = c2s.read_messages()
    assert msgs == [], "must not parse an incomplete line"
    print(f"    after first half: {len(msgs)} complete message(s) (buffered)")
    c2s.write_bytes(raw[half:])
    server.pump(c2s)               # now the line is complete
    init_resp, _ = s2c.read_messages()
    assert init_resp and init_resp[0]["id"] == 1
    print(f"    after second half: parsed initialize, replied id={init_resp[0]['id']}")

    # --- B. Concurrent in-flight requests, out-of-order replies -----------
    print("\n  B. 3 concurrent requests (ids 1,2,3) BEFORE reading any reply:")
    client.send_request(1, "echo", "alpha")
    client.send_request(2, "upper", "beta")
    client.send_request(3, "echo", "gamma")

    # Server processes them but we force OUT-OF-ORDER delivery (2, 1, 3) to
    # prove the client matches by id, not by arrival order.
    incoming, _ = c2s.read_messages()
    by_id = {m["id"]: m for m in incoming}
    for rid in (2, 1, 3):                       # scrambled processing order
        server._dispatch(by_id[rid])
    client.collect_responses()
    print(f"    id=1 -> {client.result_text(1)}")
    print(f"    id=2 -> {client.result_text(2)}")
    print(f"    id=3 -> {client.result_text(3)}")
    assert client.result_text(1) == "echo:alpha"
    assert client.result_text(2) == "BETA"
    assert client.result_text(3) == "echo:gamma"
    assert not client._pending, "all requests must be resolved"

    # --- C. Notification produces NO response -----------------------------
    print("\n  C. Notification 'initialized' (no id):")
    client.send_notification("initialized")
    server.pump(c2s)
    resp, _ = s2c.read_messages()
    assert resp == [], "a notification must not produce a response"
    print(f"    server replies: {len(resp)} (expected 0)")

    # --- D. Malformed line in the stream ----------------------------------
    print("\n  D. Corrupted line wedged between two valid messages:")
    c2s.write_message(jsonrpc_request("tools/call", {"name": "echo", "arguments": {"text": "before"}}, 10))
    c2s.write_bytes(b"{ this is not json }\n")
    c2s.write_message(jsonrpc_request("tools/call", {"name": "echo", "arguments": {"text": "after"}}, 11))
    server.pump(c2s)              # logs the malformed line, processes 10 and 11
    client._pending.update({10, 11})
    client.collect_responses()
    assert client.result_text(10) == "echo:before"
    assert client.result_text(11) == "echo:after"
    print(f"    id=10 -> {client.result_text(10)} | id=11 -> {client.result_text(11)} (corrupt line skipped)")

    print("\n  PASS -- framing, id-matching, notifications, malformed resilience.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 10 HARD Solutions -- MCP")
    print("#" * 60)

    hard_ex1_secure_server()
    hard_ex2_stdio_transport()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

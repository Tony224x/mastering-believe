"""
Day 19 -- Inter-agent protocols: mini A2A implementation in pure stdlib.

Demonstrates:
  1. AgentCard          -- declarative identity of an agent (skills, capabilities, auth)
  2. Task               -- unit of work with formal lifecycle (submitted→working→completed/failed)
  3. Message / Part     -- multimodal message format (text, data)
  4. AgentServer        -- in-memory A2A server: exposes agent-card, dispatches tasks,
                          streams status updates via a simple event queue
  5. AgentClient        -- discovers an agent via its card, sends tasks, polls results
  6. Two-agent demo     -- Orchestrator delegates to SpecialistAgent in-process

Everything runs in-memory. No network, no API keys required.
The JSON-RPC 2.0 envelope is faithfully reproduced so the code stays
close to the real A2A wire format.

Run:
    python domains/agentic-ai/02-code/19-protocoles-inter-agents.py
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ===========================================================================
# 1. PROTOCOL DATA STRUCTURES
# ===========================================================================

class TaskState(str, Enum):
    """Formal task lifecycle states defined by A2A spec."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"   # agent needs clarification
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Part:
    """A single content part in a message (text or structured data)."""
    type: str       # "text" | "data"
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {"type": "data", "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "Part":
        if d["type"] == "text":
            return cls(type="text", text=d.get("text", ""))
        return cls(type="data", data=d.get("data", {}))


@dataclass
class Message:
    """A2A message: a role + a list of content parts."""
    role: str           # "user" | "agent"
    parts: list[Part] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"role": self.role, "parts": [p.to_dict() for p in self.parts]}

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            parts=[Part.from_dict(p) for p in d.get("parts", [])],
        )


@dataclass
class TaskStatus:
    """Current status of a task, with optional human-readable message."""
    state: TaskState
    timestamp: str
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "timestamp": self.timestamp,
            "message": self.message,
        }


@dataclass
class Task:
    """
    An A2A Task — the fundamental unit of inter-agent work.

    Lifecycle:  submitted → working → completed (or failed / canceled)
    Artifacts are the final outputs produced by the agent.
    History records all status transitions for observability.
    """
    id: str
    session_id: str
    message: Message
    status: TaskStatus = field(
        default_factory=lambda: TaskStatus(
            state=TaskState.SUBMITTED,
            timestamp=_now(),
        )
    )
    artifacts: list[dict] = field(default_factory=list)
    history: list[TaskStatus] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "message": self.message.to_dict(),
            "status": self.status.to_dict(),
            "artifacts": self.artifacts,
            "history": [s.to_dict() for s in self.history],
        }


@dataclass
class Skill:
    """A capability declared in an Agent Card."""
    id: str
    name: str
    description: str
    input_modes: list[str] = field(default_factory=lambda: ["text"])
    output_modes: list[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "inputModes": self.input_modes,
            "outputModes": self.output_modes,
        }


@dataclass
class AgentCard:
    """
    The declarative identity of an A2A agent.

    Published at /.well-known/agent.json (here: agent_url + '/agent-card').
    Enables discovery without prior knowledge of the agent's implementation.
    """
    name: str
    description: str
    url: str                    # base URL of the agent's A2A endpoint
    version: str = "1.0.0"
    skills: list[Skill] = field(default_factory=list)
    streaming: bool = True      # does the agent support SSE push?
    auth_schemes: list[str] = field(default_factory=lambda: ["bearer"])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": {
                "streaming": self.streaming,
                "pushNotifications": False,
            },
            "skills": [s.to_dict() for s in self.skills],
            "authentication": {"schemes": self.auth_schemes},
        }


# ===========================================================================
# 2. JSON-RPC 2.0 HELPERS
# ===========================================================================

def make_request(method: str, params: dict, req_id: str | None = None) -> dict:
    """Build a JSON-RPC 2.0 request envelope."""
    return {
        "jsonrpc": "2.0",
        "id": req_id or str(uuid.uuid4()),
        "method": method,
        "params": params,
    }


def make_response(req_id: str, result: dict) -> dict:
    """Build a JSON-RPC 2.0 success response envelope."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id: str, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response envelope."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def serialize(obj: dict) -> str:
    """Serialize to compact JSON (mirrors wire format)."""
    return json.dumps(obj, separators=(",", ":"))


# ===========================================================================
# 3. AGENT SERVER
# ===========================================================================

class AgentServer:
    """
    In-memory A2A agent server.

    Exposes:
      - get_agent_card()    → AgentCard   (mimics GET /.well-known/agent.json)
      - dispatch(request)   → response dict  (mimics POST /a2a)

    Task processing runs in a background thread to simulate async execution.
    Status updates are pushed to an event queue so the client can poll them.
    """

    def __init__(self, card: AgentCard) -> None:
        self.card = card
        self._tasks: dict[str, Task] = {}           # task_id → Task
        self._events: dict[str, queue.Queue] = {}   # task_id → event queue
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_agent_card(self) -> dict:
        """Return the agent card as a JSON-serializable dict."""
        return self.card.to_dict()

    # ------------------------------------------------------------------
    # JSON-RPC dispatch (mimics an HTTP POST handler)
    # ------------------------------------------------------------------

    def dispatch(self, raw_request: str) -> str:
        """
        Receive a JSON-RPC 2.0 request string, dispatch to the right handler,
        return a JSON-RPC 2.0 response string.
        """
        try:
            req = json.loads(raw_request)
        except json.JSONDecodeError:
            return serialize(make_error("null", -32700, "Parse error"))

        req_id = req.get("id", "null")
        method = req.get("method", "")
        params = req.get("params", {})

        handlers = {
            "tasks/send": self._handle_tasks_send,
            "tasks/get": self._handle_tasks_get,
            "tasks/cancel": self._handle_tasks_cancel,
        }

        handler = handlers.get(method)
        if handler is None:
            return serialize(make_error(req_id, -32601, f"Method not found: {method}"))

        try:
            result = handler(params)
            return serialize(make_response(req_id, result))
        except Exception as exc:  # noqa: BLE001
            return serialize(make_error(req_id, -32000, str(exc)))

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    def _handle_tasks_send(self, params: dict) -> dict:
        """Create a new task (or resume an existing one) and start processing."""
        task_id = params.get("id") or str(uuid.uuid4())
        session_id = params.get("sessionId", str(uuid.uuid4()))
        msg_data = params.get("message", {})

        message = Message.from_dict(msg_data)

        task = Task(id=task_id, session_id=session_id, message=message)
        with self._lock:
            self._tasks[task_id] = task
            self._events[task_id] = queue.Queue()

        # Process asynchronously in a background thread
        t = threading.Thread(target=self._process_task, args=(task,), daemon=True)
        t.start()

        return task.to_dict()

    def _handle_tasks_get(self, params: dict) -> dict:
        """Return the current state of a task."""
        task_id = params.get("id")
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        return task.to_dict()

    def _handle_tasks_cancel(self, params: dict) -> dict:
        """Cancel a task if it is still running."""
        task_id = params.get("id")
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        self._transition(task, TaskState.CANCELED)
        return task.to_dict()

    # ------------------------------------------------------------------
    # Internal task processing (override in subclasses for custom logic)
    # ------------------------------------------------------------------

    def _process_task(self, task: Task) -> None:
        """
        Default processing: transition to WORKING, do the work, transition to COMPLETED.

        Subclasses override _execute() to implement real logic.
        """
        self._transition(task, TaskState.WORKING, message="Processing task…")
        try:
            artifacts = self._execute(task)
            with self._lock:
                task.artifacts = artifacts
            self._transition(task, TaskState.COMPLETED, message="Done.")
        except Exception as exc:  # noqa: BLE001
            self._transition(task, TaskState.FAILED, message=str(exc))

    def _execute(self, task: Task) -> list[dict]:
        """
        Override in subclasses. Return a list of artifact dicts.
        Default: echo the input text back, wrapped in a data artifact.
        """
        # Extract text from the first text part
        text = ""
        for part in task.message.parts:
            if part.type == "text":
                text = part.text
                break

        # Simulate work delay
        time.sleep(0.05)

        return [
            {
                "type": "text",
                "parts": [{"type": "text", "text": f"[echo] {text}"}],
            }
        ]

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, task: Task, new_state: TaskState, message: str = "") -> None:
        """
        Move a task to a new state, record in history,
        and push a status update to the event queue.
        """
        status = TaskStatus(state=new_state, timestamp=_now(), message=message)
        with self._lock:
            task.history.append(task.status)
            task.status = status
            if task.id in self._events:
                self._events[task.id].put(status.to_dict())

    # ------------------------------------------------------------------
    # SSE-style event polling (simulates SSE subscription)
    # ------------------------------------------------------------------

    def poll_events(self, task_id: str, timeout: float = 5.0) -> list[dict]:
        """
        Drain all pending status-update events for a task.
        In a real A2A server, these would be pushed via Server-Sent Events.
        Here we use a Queue to simulate the same push semantics.
        """
        events = []
        q = self._events.get(task_id)
        if q is None:
            return events

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = q.get(timeout=0.05)
                events.append(event)
                # Stop collecting once we reach a terminal state
                if event.get("state") in (
                    TaskState.COMPLETED.value,
                    TaskState.FAILED.value,
                    TaskState.CANCELED.value,
                ):
                    break
            except queue.Empty:
                # Keep polling if the task is still working
                with self._lock:
                    task = self._tasks.get(task_id)
                if task and task.status.state in (
                    TaskState.SUBMITTED,
                    TaskState.WORKING,
                    TaskState.INPUT_REQUIRED,
                ):
                    continue
                break

        return events


# ===========================================================================
# 4. AGENT CLIENT
# ===========================================================================

class AgentClient:
    """
    In-memory A2A client.

    In production, `discover()` would do:
        GET https://<base_url>/.well-known/agent.json

    Here it calls server.get_agent_card() directly to stay stdlib-only.
    """

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        self._card: dict | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> dict:
        """
        Fetch and cache the agent card.
        Returns the raw dict (mirrors what a GET /.well-known/agent.json would return).
        """
        self._card = self._server.get_agent_card()
        return self._card

    def has_skill(self, skill_id: str) -> bool:
        """Check if the discovered agent declares a given skill."""
        if self._card is None:
            self.discover()
        return any(s["id"] == skill_id for s in self._card.get("skills", []))

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def send_task(
        self,
        text: str,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """
        Send a text task to the agent server.
        Returns the initial task dict (status = submitted or working).
        """
        req = make_request(
            method="tasks/send",
            params={
                "id": task_id or str(uuid.uuid4()),
                "sessionId": session_id or str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": text}],
                },
            },
        )
        raw_resp = self._server.dispatch(serialize(req))
        resp = json.loads(raw_resp)
        if "error" in resp:
            raise RuntimeError(f"A2A error: {resp['error']}")
        return resp["result"]

    def get_task(self, task_id: str) -> dict:
        """Poll the current state of a task."""
        req = make_request(method="tasks/get", params={"id": task_id})
        raw_resp = self._server.dispatch(serialize(req))
        resp = json.loads(raw_resp)
        if "error" in resp:
            raise RuntimeError(f"A2A error: {resp['error']}")
        return resp["result"]

    def cancel_task(self, task_id: str) -> dict:
        """Cancel a running task."""
        req = make_request(method="tasks/cancel", params={"id": task_id})
        raw_resp = self._server.dispatch(serialize(req))
        resp = json.loads(raw_resp)
        if "error" in resp:
            raise RuntimeError(f"A2A error: {resp['error']}")
        return resp["result"]

    def send_and_wait(self, text: str) -> dict:
        """
        Send a task and wait for completion by polling SSE-style events.
        Returns the completed task dict.
        """
        initial = self.send_task(text)
        task_id = initial["id"]

        # Poll event queue (mirrors SSE subscription)
        events = self._server.poll_events(task_id, timeout=10.0)

        # Return the final task state
        return self.get_task(task_id)


# ===========================================================================
# 5. SPECIALIST AGENT — custom _execute logic
# ===========================================================================

class RouteOptimizerServer(AgentServer):
    """
    A specialist agent that pretends to optimize delivery routes.

    Skill: 'optimize_routes' — accepts a list of stops, returns a mock optimal order.
    """

    def __init__(self) -> None:
        card = AgentCard(
            name="Route Optimizer Agent",
            description=(
                "Calcule les tournees de livraison optimales pour une flotte de vehicules."
            ),
            url="http://localhost:8001",  # would be real in production
            skills=[
                Skill(
                    id="optimize_routes",
                    name="Optimize Routes",
                    description="Reordonne des stops pour minimiser la distance totale.",
                    input_modes=["text", "application/json"],
                    output_modes=["application/json"],
                ),
                Skill(
                    id="estimate_eta",
                    name="Estimate ETA",
                    description="Estime le temps de livraison pour un trajet donne.",
                    input_modes=["text"],
                    output_modes=["text"],
                ),
            ],
        )
        super().__init__(card)

    def _execute(self, task: Task) -> list[dict]:
        """
        Mock route optimization:
        - Parse stop names from the text part
        - Return them in alphabetical order (stands in for a real TSP solver)
        """
        text = ""
        for part in task.message.parts:
            if part.type == "text":
                text = part.text
                break

        # Simulate processing time
        time.sleep(0.1)

        # Extract comma-separated stops if present
        if ":" in text:
            _, stops_raw = text.split(":", 1)
            stops = [s.strip() for s in stops_raw.split(",") if s.strip()]
        else:
            # Fallback: split on spaces as dummy stops
            stops = text.split()

        # "Optimize": alphabetical order as a deterministic mock
        optimized = sorted(stops)

        return [
            {
                "type": "application/json",
                "parts": [
                    {
                        "type": "data",
                        "data": {
                            "original_order": stops,
                            "optimized_order": optimized,
                            "estimated_distance_km": len(stops) * 12.4,
                            "note": "Mock optimizer — real system would use TSP/VRP solver",
                        },
                    }
                ],
            }
        ]


class OrchestratorAgent:
    """
    High-level orchestrator that:
    1. Discovers the RouteOptimizer via its agent card
    2. Checks the agent has the required skill
    3. Delegates the optimization task via A2A
    4. Processes and prints the result
    """

    def __init__(self, specialist_server: AgentServer) -> None:
        self._client = AgentClient(specialist_server)

    def run(self, delivery_stops: list[str]) -> dict:
        """Orchestrate a route optimization for the given stops."""
        # Step 1 — Discovery
        card = self._client.discover()
        print(f"\n[Orchestrator] Discovered agent: '{card['name']}'")
        print(f"               Version : {card['version']}")
        print(f"               Skills  : {[s['id'] for s in card['skills']]}")
        print(f"               Streaming: {card['capabilities']['streaming']}")

        # Step 2 — Capability check
        required_skill = "optimize_routes"
        if not self._client.has_skill(required_skill):
            raise RuntimeError(
                f"Agent does not declare required skill: {required_skill}"
            )
        print(f"[Orchestrator] Skill '{required_skill}' confirmed. Sending task…")

        # Step 3 — Send task and wait
        stops_str = ", ".join(delivery_stops)
        prompt = f"Optimize delivery route for stops: {stops_str}"
        result = self._client.send_and_wait(prompt)

        # Step 4 — Process result
        print(f"[Orchestrator] Task {result['id'][:8]}… → state: {result['status']['state']}")
        return result


# ===========================================================================
# 6. UTILITY
# ===========================================================================

def _now() -> str:
    """Return current UTC time as ISO 8601 string (stdlib only)."""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_json(label: str, data: dict) -> None:
    """Pretty-print a dict as indented JSON with a label."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(json.dumps(data, indent=2))


# ===========================================================================
# 7. DEMO
# ===========================================================================

def demo_agent_card() -> None:
    """Show what a real A2A agent card looks like (wire format)."""
    print("\n" + "=" * 60)
    print("  DEMO 1 — Agent Card (GET /.well-known/agent.json)")
    print("=" * 60)

    server = RouteOptimizerServer()
    card = server.get_agent_card()
    _print_json("Agent Card (JSON)", card)


def demo_json_rpc_messages() -> None:
    """Show raw JSON-RPC 2.0 request/response pairs."""
    print("\n" + "=" * 60)
    print("  DEMO 2 — JSON-RPC 2.0 Wire Format")
    print("=" * 60)

    # Construct and display a tasks/send request
    req = make_request(
        method="tasks/send",
        params={
            "id": "task-demo-001",
            "sessionId": "session-xyz",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "Optimize route: Lyon, Marseille, Paris"}],
            },
        },
        req_id="req-001",
    )
    _print_json("Request: tasks/send", req)

    # Show what a response looks like
    resp = make_response(
        req_id="req-001",
        result={
            "id": "task-demo-001",
            "sessionId": "session-xyz",
            "status": {"state": "working", "timestamp": _now(), "message": "Processing…"},
            "artifacts": [],
            "history": [
                {"state": "submitted", "timestamp": _now(), "message": ""}
            ],
        },
    )
    _print_json("Response: tasks/send (initial)", resp)


def demo_task_lifecycle() -> None:
    """Show the complete task lifecycle via A2A in-memory."""
    print("\n" + "=" * 60)
    print("  DEMO 3 — Task Lifecycle (submitted → working → completed)")
    print("=" * 60)

    server = RouteOptimizerServer()
    client = AgentClient(server)

    stops = ["Marseille", "Lyon", "Bordeaux", "Nantes", "Toulouse", "Nice"]
    stops_str = ", ".join(stops)

    print(f"\n[Client] Sending task: optimize {len(stops)} stops")
    print(f"         Stops: {stops_str}")

    initial = client.send_task(f"Optimize delivery route for stops: {stops_str}")
    task_id = initial["id"]

    print(f"[Client] Task created: {task_id}")
    print(f"[Client] Initial state: {initial['status']['state']}")

    # Poll for SSE-style events
    print("[Client] Subscribing to status updates (SSE simulation)…")
    events = server.poll_events(task_id, timeout=5.0)
    for evt in events:
        print(f"[SSE]    state={evt['state']}  msg='{evt.get('message', '')}'")

    # Get final result
    final = client.get_task(task_id)
    print(f"\n[Client] Final state: {final['status']['state']}")
    print("[Client] Task history:")
    for h in final["history"]:
        print(f"         {h['state']:18s} @ {h['timestamp']}")

    if final["artifacts"]:
        artifact_data = final["artifacts"][0]["parts"][0]["data"]
        print("\n[Client] Result artifact:")
        print(f"         original  : {artifact_data['original_order']}")
        print(f"         optimized : {artifact_data['optimized_order']}")
        print(f"         distance  : {artifact_data['estimated_distance_km']:.1f} km")


def demo_two_agents_collaboration() -> None:
    """Show two agents collaborating: Orchestrator delegates to RouteOptimizer."""
    print("\n" + "=" * 60)
    print("  DEMO 4 — Two Agents Collaborating via A2A")
    print("=" * 60)
    print("""
  Architecture:
    OrchestratorAgent
      │
      │  [A2A: tasks/send / tasks/get]
      ▼
    RouteOptimizerServer  (has its own skills)
""")

    specialist = RouteOptimizerServer()
    orchestrator = OrchestratorAgent(specialist)

    delivery_stops = ["Grenoble", "Annecy", "Chambery", "Valence", "Gap"]
    result = orchestrator.run(delivery_stops)

    # Extract and display final artifacts
    if result.get("artifacts"):
        data = result["artifacts"][0]["parts"][0]["data"]
        print("\n[Orchestrator] Route optimization result:")
        print(f"               Before : {data['original_order']}")
        print(f"               After  : {data['optimized_order']}")
        print(f"               Est. distance: {data['estimated_distance_km']:.1f} km")
    else:
        print("[Orchestrator] No artifacts (task may have failed)")


def demo_error_handling() -> None:
    """Show JSON-RPC error responses (unknown method, missing task)."""
    print("\n" + "=" * 60)
    print("  DEMO 5 — Error Handling (JSON-RPC error responses)")
    print("=" * 60)

    server = RouteOptimizerServer()

    # Unknown method
    bad_method = make_request("tasks/unknown", {}, req_id="err-001")
    resp = json.loads(server.dispatch(serialize(bad_method)))
    print(f"\n[Error] Unknown method: {resp.get('error')}")

    # Task not found
    get_req = make_request("tasks/get", {"id": "nonexistent-task-id"}, req_id="err-002")
    resp = json.loads(server.dispatch(serialize(get_req)))
    print(f"[Error] Task not found: {resp.get('error')}")

    # Malformed JSON
    resp = json.loads(server.dispatch("not json at all{{{"))
    print(f"[Error] Parse error    : {resp.get('error')}")


def demo_mcp_vs_a2a() -> None:
    """Textual summary comparing MCP and A2A (no code needed)."""
    print("\n" + "=" * 60)
    print("  DEMO 6 — MCP vs A2A: Complementarity")
    print("=" * 60)
    summary = """
  MCP  (Model Context Protocol — Anthropic, 2024)
    role   : Agent ↔ Tools/Data
    what   : An LLM calls tools (filesystem, GitHub, DB…) via a standardized protocol
    who    : Anthropic — widely adopted (Claude Desktop, Cursor, VS Code…)
    when   : your agent needs to USE a tool or READ data
    format : JSON-RPC 2.0  |  transport: stdio or HTTP

  A2A  (Agent2Agent Protocol — Linux Foundation / Google, 2025)
    role   : Agent ↔ Agent
    what   : An agent delegates a TASK to another agent (peer-to-peer)
    who    : Google + 50+ vendors under Linux Foundation
    when   : your agent needs to DELEGATE to another autonomous agent
    format : JSON-RPC 2.0  |  transport: HTTP + SSE (streaming)

  Used together in the same system:
    ┌──────────────────────────────────────────────────┐
    │  Orchestrator Agent                              │
    │    │── [MCP] ──► Filesystem, GitHub, DB tools   │
    │    └── [A2A] ──► Specialist Agent B             │
    │                       │── [MCP] ──► Its tools   │
    └──────────────────────────────────────────────────┘

  Key rule: MCP = your agent's TOOLS; A2A = other AGENTS you delegate to.
"""
    print(summary)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    demo_agent_card()
    demo_json_rpc_messages()
    demo_task_lifecycle()
    demo_two_agents_collaboration()
    demo_error_handling()
    demo_mcp_vs_a2a()

    print("\n" + "=" * 60)
    print("  All demos completed successfully.")
    print("=" * 60)

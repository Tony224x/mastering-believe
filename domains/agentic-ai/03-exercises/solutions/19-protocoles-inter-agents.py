"""
Day 19 -- Solutions to the easy exercises on inter-agent protocols (A2A).

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/19-protocoles-inter-agents.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the day-19 module (stdlib importlib trick used in all day solutions)
# ---------------------------------------------------------------------------
from importlib import import_module

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# Module names with hyphens require import_module (not a regular import)
day19 = import_module("19-protocoles-inter-agents")

# Re-export what we need
AgentCard = day19.AgentCard
AgentClient = day19.AgentClient
AgentServer = day19.AgentServer
RouteOptimizerServer = day19.RouteOptimizerServer
Skill = day19.Skill
Task = day19.Task
TaskState = day19.TaskState
TaskStatus = day19.TaskStatus
Message = day19.Message
Part = day19.Part
_now = day19._now


# ===========================================================================
# SOLUTION 1 — Extend AgentCard with trust metadata
# ===========================================================================

def _compute_signature(name: str, url: str, issued_at: str) -> str:
    """Mock signature: SHA-256 of 'name|url|issued_at'."""
    payload = f"{name}|{url}|{issued_at}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TrustMetadata:
    """Trust fields added to an Agent Card for identity verification."""
    issuer: str
    issued_at: str      # ISO 8601
    expires_at: str     # ISO 8601
    signature: str      # mock: SHA-256(name|url|issued_at)

    def to_dict(self) -> dict:
        return {
            "issuer": self.issuer,
            "issuedAt": self.issued_at,
            "expiresAt": self.expires_at,
            "signature": self.signature,
        }


class TrustedAgentCard(AgentCard):
    """
    AgentCard extended with TrustMetadata.

    Adds:
      - trust_metadata  : issuer, issued_at, expires_at, signature
      - is_valid()      : checks expiry + signature
      - to_dict()       : includes trustMetadata key
    """

    def __init__(self, *args: Any, trust: TrustMetadata, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trust = trust

    def to_dict(self) -> dict:  # type: ignore[override]
        d = super().to_dict()
        d["trustMetadata"] = self.trust.to_dict()
        return d

    def is_valid(self) -> tuple[bool, str]:
        """
        Return (True, 'ok') or (False, reason).
        Checks:
          1. expires_at > now  (lexicographic ISO 8601 comparison is safe)
          2. signature matches expected SHA-256
        """
        now = _now()
        if self.trust.expires_at <= now:
            return False, f"expired: {self.trust.expires_at} <= {now}"

        expected = _compute_signature(self.name, self.url, self.trust.issued_at)
        if self.trust.signature != expected:
            return False, f"invalid signature (got {self.trust.signature[:12]}…)"

        return True, "ok"


def verify_trust(card_dict: dict) -> tuple[bool, str]:
    """
    Standalone verifier that works on a plain dict (as received over the wire).
    Re-computes the mock signature and checks expiry.
    """
    tm = card_dict.get("trustMetadata", {})
    if not tm:
        return False, "no trustMetadata field"

    expires_at = tm.get("expiresAt", "")
    now = _now()
    if expires_at <= now:
        return False, f"expired: {expires_at}"

    expected = _compute_signature(
        card_dict.get("name", ""),
        card_dict.get("url", ""),
        tm.get("issuedAt", ""),
    )
    if tm.get("signature") != expected:
        return False, "invalid signature"

    return True, "ok"


def _make_valid_card() -> TrustedAgentCard:
    issued = "2025-01-01T00:00:00Z"
    return TrustedAgentCard(
        name="Route Optimizer Agent",
        description="Optimise les tournees.",
        url="https://fleet.example.com/a2a",
        trust=TrustMetadata(
            issuer="acme-corp.example.com",
            issued_at=issued,
            expires_at="2099-12-31T23:59:59Z",   # far future
            signature=_compute_signature(
                "Route Optimizer Agent",
                "https://fleet.example.com/a2a",
                issued,
            ),
        ),
    )


def _make_expired_card() -> TrustedAgentCard:
    issued = "2020-01-01T00:00:00Z"
    return TrustedAgentCard(
        name="Old Agent",
        description="Expired.",
        url="https://old.example.com/a2a",
        trust=TrustMetadata(
            issuer="acme-corp.example.com",
            issued_at=issued,
            expires_at="2020-12-31T23:59:59Z",   # past
            signature=_compute_signature("Old Agent", "https://old.example.com/a2a", issued),
        ),
    )


def _make_tampered_card() -> TrustedAgentCard:
    issued = "2025-01-01T00:00:00Z"
    return TrustedAgentCard(
        name="Fake Agent",
        description="Attacker-controlled.",
        url="https://attacker.evil.com/a2a",
        trust=TrustMetadata(
            issuer="evil.com",
            issued_at=issued,
            expires_at="2099-12-31T23:59:59Z",
            signature="deadbeef" * 8,             # forged / wrong signature
        ),
    )


def solution_1() -> None:
    print("\n" + "=" * 60)
    print("  SOLUTION 1 — Trust Metadata on Agent Card")
    print("=" * 60)

    cards = [
        ("Valid card   ", _make_valid_card()),
        ("Expired card ", _make_expired_card()),
        ("Tampered card", _make_tampered_card()),
    ]

    for label, card in cards:
        ok, reason = card.is_valid()
        status = "VALID  " if ok else "INVALID"
        print(f"\n[{label}] → {status}  | reason: {reason}")

        # Also test the standalone dict-based verifier
        card_dict = card.to_dict()
        ok2, reason2 = verify_trust(card_dict)
        assert ok == ok2, "is_valid() and verify_trust() disagree!"

    print("\n[Solution 1] All 3 cards verified correctly.")


# ===========================================================================
# SOLUTION 2 — input-required state in the task lifecycle
# ===========================================================================

class ClarifyingAgentServer(AgentServer):
    """
    Agent that suspends execution when a task is 'ambiguous' and waits for
    the client to provide a clarification before completing the task.

    Uses a threading.Event per task to synchronize the pause/resume.
    """

    def __init__(self) -> None:
        card = AgentCard(
            name="Clarifying Agent",
            description="Demande des clarifications avant de traiter les taches ambigues.",
            url="http://localhost:8002",
            skills=[
                Skill(
                    id="process_task",
                    name="Process Task",
                    description="Traite une tache, avec clarification si necessaire.",
                ),
            ],
        )
        super().__init__(card)
        # task_id → (Event, clarification_text_holder)
        self._clarification_events: dict[str, tuple[threading.Event, list[str]]] = {}

    def provide_clarification(self, task_id: str, clarification: str) -> None:
        """Called by the client to unblock a suspended task."""
        entry = self._clarification_events.get(task_id)
        if entry is None:
            raise ValueError(f"No pending clarification for task {task_id}")
        event, holder = entry
        holder.append(clarification)
        event.set()

    def _execute(self, task: Task) -> list[dict]:
        text = ""
        for part in task.message.parts:
            if part.type == "text":
                text = part.text
                break

        if "ambiguous" in text.lower():
            # Suspend and wait for client input
            event = threading.Event()
            holder: list[str] = []
            self._clarification_events[task.id] = (event, holder)

            # Transition to input-required
            self._transition(
                task,
                TaskState.INPUT_REQUIRED,
                message="Veuillez preciser le nombre de vehicules disponibles.",
            )

            # Wait for clarification (up to 10 seconds)
            event.wait(timeout=10.0)

            clarification = holder[0] if holder else "(no clarification received)"

            # Resume working
            self._transition(task, TaskState.WORKING, message=f"Clarification recue: {clarification}")
            time.sleep(0.05)  # simulate work

            return [
                {
                    "type": "text",
                    "parts": [
                        {
                            "type": "text",
                            "text": (
                                f"Tache traitee avec clarification: '{clarification}'. "
                                f"Input original: '{text}'"
                            ),
                        }
                    ],
                }
            ]

        # Non-ambiguous task: normal processing
        time.sleep(0.05)
        return [
            {
                "type": "text",
                "parts": [{"type": "text", "text": f"Processed: {text}"}],
            }
        ]


def solution_2() -> None:
    print("\n" + "=" * 60)
    print("  SOLUTION 2 — input-required State in Task Lifecycle")
    print("=" * 60)

    server = ClarifyingAgentServer()
    client = AgentClient(server)

    # Send an ambiguous task
    ambiguous_text = "This request is ambiguous — I need more info."
    print(f"\n[Client] Sending ambiguous task: '{ambiguous_text}'")
    initial = client.send_task(ambiguous_text)
    task_id = initial["id"]
    print(f"[Client] Task created: {task_id[:8]}…  initial state: {initial['status']['state']}")

    # Wait briefly for the task to reach input-required
    time.sleep(0.2)

    # Poll once to see the input-required event
    current = client.get_task(task_id)
    print(f"[Client] Polled state: {current['status']['state']}")
    print(f"[Client] Agent message: '{current['status']['message']}'")

    assert current["status"]["state"] == TaskState.INPUT_REQUIRED.value, (
        f"Expected input-required, got {current['status']['state']}"
    )

    # Provide clarification
    clarification = "3 vehicules disponibles, rayon max 50 km"
    print(f"\n[Client] Providing clarification: '{clarification}'")
    server.provide_clarification(task_id, clarification)

    # Now wait for completion
    events = server.poll_events(task_id, timeout=5.0)
    for evt in events:
        print(f"[SSE]    state={evt['state']:20s}  msg='{evt.get('message', '')}'")

    final = client.get_task(task_id)
    print(f"\n[Client] Final state: {final['status']['state']}")
    print("[Client] Full history:")
    for h in final["history"]:
        print(f"         {h['state']:22s} | {h.get('message', '')}")

    # Verify lifecycle
    states = [h["state"] for h in final["history"]] + [final["status"]["state"]]
    assert TaskState.INPUT_REQUIRED.value in states, "input-required missing from history"
    assert final["status"]["state"] == TaskState.COMPLETED.value, "task not completed"

    print("\n[Solution 2] input-required lifecycle verified correctly.")


# ===========================================================================
# SOLUTION 3 — Agent registry with skill-based routing
# ===========================================================================

class AgentRegistry:
    """
    Central registry that indexes AgentServers by their declared skills.
    Supports dynamic registration and skill-based lookup.
    """

    def __init__(self) -> None:
        # skill_id → list of servers that declare it
        self._index: dict[str, list[AgentServer]] = {}
        self._servers: list[AgentServer] = []

    def register(self, server: AgentServer) -> None:
        """Register a server and index all its skills."""
        self._servers.append(server)
        for skill in server.card.skills:
            self._index.setdefault(skill.id, []).append(server)

    def find_by_skill(self, skill_id: str) -> list[AgentServer]:
        """Return all servers that declare the given skill."""
        return list(self._index.get(skill_id, []))

    def find_best(self, skill_id: str) -> AgentServer | None:
        """Return the first available server for a skill (simple round-robin placeholder)."""
        candidates = self.find_by_skill(skill_id)
        return candidates[0] if candidates else None

    def list_skills(self) -> list[str]:
        """Return all known skill ids."""
        return list(self._index.keys())


class WeatherAgentServer(AgentServer):
    """Specialist agent: returns mock weather data for a city."""

    def __init__(self) -> None:
        card = AgentCard(
            name="Weather Agent",
            description="Retourne les previsions meteo pour une ville.",
            url="http://localhost:8003",
            skills=[
                Skill(
                    id="get_weather",
                    name="Get Weather",
                    description="Retourne la meteo actuelle pour une ville donnee.",
                    input_modes=["text"],
                    output_modes=["application/json"],
                ),
            ],
        )
        super().__init__(card)

    def _execute(self, task: Task) -> list[dict]:
        text = ""
        for part in task.message.parts:
            if part.type == "text":
                text = part.text
                break

        time.sleep(0.05)

        # Extract city name: look for "for <city>" or just use the last word
        city = "Unknown"
        words = text.lower().split()
        if "for" in words:
            idx = words.index("for")
            if idx + 1 < len(words):
                city = words[idx + 1].capitalize()
        elif words:
            city = words[-1].capitalize()

        mock_data = {
            "city": city,
            "temperature_c": 18.5,
            "condition": "Partly cloudy",
            "wind_kmh": 22,
            "source": "mock — no real API called",
        }

        return [
            {
                "type": "application/json",
                "parts": [{"type": "data", "data": mock_data}],
            }
        ]


def _detect_skill(text: str) -> str | None:
    """
    Heuristic: detect the required skill from the task text.
    Returns a skill id or None if undecidable.
    """
    lower = text.lower()
    if any(kw in lower for kw in ("route", "optimize", "livraison", "stops", "tournee")):
        return "optimize_routes"
    if any(kw in lower for kw in ("weather", "meteo", "temperature", "forecast", "climat")):
        return "get_weather"
    return None


class SmartOrchestrator:
    """
    Orchestrator that routes tasks to the right agent via the AgentRegistry.
    Uses _detect_skill() as a simple keyword-based router.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def run(self, task_text: str) -> dict:
        print(f"\n[SmartOrchestrator] Input: '{task_text}'")

        skill_id = _detect_skill(task_text)
        if skill_id is None:
            msg = f"Cannot detect required skill from: '{task_text}'"
            print(f"[SmartOrchestrator] ERROR: {msg}")
            return {"error": msg, "available_skills": self._registry.list_skills()}

        print(f"[SmartOrchestrator] Detected skill: '{skill_id}'")

        agent = self._registry.find_best(skill_id)
        if agent is None:
            msg = f"No agent found for skill: '{skill_id}'"
            print(f"[SmartOrchestrator] ERROR: {msg}")
            return {"error": msg}

        print(f"[SmartOrchestrator] Routing to agent: '{agent.card.name}'")

        client = AgentClient(agent)
        result = client.send_and_wait(task_text)

        state = result["status"]["state"]
        print(f"[SmartOrchestrator] Task completed with state: {state}")
        return result


def solution_3() -> None:
    print("\n" + "=" * 60)
    print("  SOLUTION 3 — Agent Registry with Skill-Based Routing")
    print("=" * 60)

    # Build the registry
    registry = AgentRegistry()
    registry.register(RouteOptimizerServer())
    registry.register(WeatherAgentServer())

    print(f"\n[Registry] Registered skills: {registry.list_skills()}")

    orchestrator = SmartOrchestrator(registry)

    # Task 1 — routing / optimization
    result1 = orchestrator.run("Optimize route for stops: Lyon, Paris, Bordeaux, Nantes")
    if result1.get("artifacts"):
        data = result1["artifacts"][0]["parts"][0]["data"]
        print(f"  → Optimized: {data['optimized_order']}")
    assert result1["status"]["state"] == TaskState.COMPLETED.value

    # Task 2 — weather
    result2 = orchestrator.run("What is the weather forecast for Lyon?")
    if result2.get("artifacts"):
        data = result2["artifacts"][0]["parts"][0]["data"]
        print(f"  → Weather in {data['city']}: {data['temperature_c']}°C, {data['condition']}")
    assert result2["status"]["state"] == TaskState.COMPLETED.value

    # Task 3 — ambiguous (no skill detected)
    result3 = orchestrator.run("Tell me something interesting.")
    assert "error" in result3, "Expected error for ambiguous task"
    print(f"\n[SmartOrchestrator] Ambiguous task → error: '{result3['error']}'")

    print("\n[Solution 3] All routing scenarios handled correctly.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

    print("\n" + "=" * 60)
    print("  All exercise solutions completed successfully.")
    print("=" * 60)

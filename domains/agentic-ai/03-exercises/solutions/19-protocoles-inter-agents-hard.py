"""
Solutions -- Day 19 (HARD): Inter-agent protocols (A2A)

Contains solutions for:
  - Hard Ex 1: End-to-end A2A interop between two "frameworks". A client
               (AlphaClient, "framework A") drives an agent (BravoAgent,
               "framework B") PURELY via the protocol: Agent Card, versioned
               handshake, task lifecycle, structured artifacts. Includes
               version negotiation (highest common, '1.10' > '1.9') and a
               clean JSON-RPC error for unknown methods.
  - Hard Ex 2: Protocol-level trust layer -- HMAC-signed envelopes (stdlib
               hmac/hashlib), nonce/timestamp replay protection, and
               capability-scoped authorization. Tampered / replayed / expired
               / unauthorized / unknown-sender messages are rejected; a
               legitimate one passes.

stdlib only, fully offline, deterministic. No network, no API key.

Run:  python 03-exercises/solutions/19-protocoles-inter-agents-hard.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Callable


# ==========================================================================
# HARD EXERCISE 1 -- A2A interop between two "frameworks" + version negotiation
# ==========================================================================

# ---- Shared protocol layer (the "wire") ----------------------------------

JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_SERVER_ERROR = -32000


def make_request(method: str, params: dict, req_id: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


def make_response(req_id: str, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_error(req_id: str, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def version_key(v: str) -> tuple[int, ...]:
    """'1.10' -> (1, 10), so '1.10' > '1.9'."""
    return tuple(int(p) for p in v.split("."))


# ---- Agent "Framework B" (BeeAI/ACP flavour) -----------------------------

class BravoAgent:
    """
    A self-contained agent. Its ONLY public surface is the protocol:
      - get_card() -> dict
      - handle(request_dict) -> response_dict
    No shared code/state with the client beyond JSON-serialisable dicts.
    """

    def __init__(self, protocol_versions: list[str] | None = None) -> None:
        self._versions = protocol_versions or ["1.0", "1.1"]
        self._tasks: dict[str, dict] = {}
        self._counter = 0

    def get_card(self) -> dict:
        return {
            "name": "Bravo Route Optimizer",
            "protocol_versions": list(self._versions),
            "skills": {
                "optimize_routes": {
                    "input_modes": ["application/json"],
                    "output_modes": ["application/json"],
                },
            },
        }

    def handle(self, request: dict) -> dict:
        req_id = request.get("id", "null")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "tasks/send":
            return self._tasks_send(req_id, params)
        if method == "tasks/get":
            return self._tasks_get(req_id, params)
        # Unknown method -> clean JSON-RPC error (no exception escapes).
        return make_error(req_id, JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")

    def _tasks_send(self, req_id: str, params: dict) -> dict:
        skill = params.get("skill")
        if skill not in self.get_card()["skills"]:
            return make_error(req_id, JSONRPC_INVALID_PARAMS, f"unknown skill: {skill}")

        self._counter += 1
        task_id = f"bravo-task-{self._counter}"

        try:
            stops = params.get("input", {}).get("stops", [])
            optimized = sorted(stops)
            artifact = {
                "type": "data",
                "data": {
                    "optimized_order": optimized,
                    "distance_km": round(len(optimized) * 12.4, 1),
                    "solver": "bravo-mock-tsp",
                },
            }
            task = {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [artifact],
            }
        except Exception as exc:  # noqa: BLE001
            task = {"id": task_id, "status": {"state": "failed", "error": str(exc)},
                    "artifacts": []}

        self._tasks[task_id] = task
        return make_response(req_id, task)

    def _tasks_get(self, req_id: str, params: dict) -> dict:
        task = self._tasks.get(params.get("id"))
        if task is None:
            return make_error(req_id, JSONRPC_INVALID_PARAMS, "task not found")
        return make_response(req_id, task)


# ---- Client "Framework A" (LangGraph flavour) ----------------------------

class HandshakeError(RuntimeError):
    pass


class AlphaClient:
    """
    Knows ONLY the protocol. It discovers, negotiates a version, checks the
    skill is declared, then drives the task -- all via dicts. `transport` is a
    callable agent.handle, so the client is fully decoupled from the agent.
    """

    SUPPORTED_VERSIONS = ["1.0", "1.1"]

    def __init__(self, card: dict, transport: Callable[[dict], dict]) -> None:
        self._card = card
        self._transport = transport
        self._req_id = 0
        self.negotiated_version: str | None = None

    def _next_id(self) -> str:
        self._req_id += 1
        return f"alpha-req-{self._req_id}"

    def handshake(self) -> str:
        """Pick the highest common protocol version or fail cleanly."""
        common = set(self.SUPPORTED_VERSIONS) & set(self._card.get("protocol_versions", []))
        if not common:
            raise HandshakeError("version_mismatch")
        self.negotiated_version = max(common, key=version_key)
        return self.negotiated_version

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._card.get("skills", {})

    def optimize(self, stops: list[str]) -> dict:
        if self.negotiated_version is None:
            raise HandshakeError("handshake not performed")
        if not self.has_skill("optimize_routes"):
            raise HandshakeError("skill not declared: optimize_routes")

        req = make_request(
            "tasks/send",
            {"skill": "optimize_routes", "input": {"stops": stops}},
            self._next_id(),
        )
        resp = self._transport(req)
        if "error" in resp:
            raise RuntimeError(f"A2A error: {resp['error']}")
        return resp["result"]


def hard_ex1_interop() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: A2A interop between two 'frameworks' + version negotiation")
    print("=" * 60)

    # --- End-to-end interop: AlphaClient drives BravoAgent via protocol only.
    bravo = BravoAgent(protocol_versions=["1.0", "1.1"])
    alpha = AlphaClient(card=bravo.get_card(), transport=bravo.handle)

    version = alpha.handshake()
    print(f"\n  handshake -> negotiated protocol v{version}")
    assert version == "1.1", "client must pick the highest common version"
    assert alpha.has_skill("optimize_routes")

    task = alpha.optimize(["Lyon", "Paris", "Bordeaux", "Nice"])
    artifact = task["artifacts"][0]["data"]
    print(f"  task state: {task['status']['state']}")
    print(f"  optimized : {artifact['optimized_order']}")
    print(f"  distance  : {artifact['distance_km']} km (solver={artifact['solver']})")
    assert task["status"]["state"] == "completed"
    assert artifact["optimized_order"] == ["Bordeaux", "Lyon", "Nice", "Paris"]
    assert artifact["solver"] == "bravo-mock-tsp"

    # --- Version mismatch: agent only speaks 2.0; handshake fails BEFORE send.
    print("\n  version mismatch test:")
    old_bravo = BravoAgent(protocol_versions=["2.0"])

    class CountingTransport:
        def __init__(self, inner: Callable[[dict], dict]) -> None:
            self.inner = inner
            self.calls = 0

        def __call__(self, req: dict) -> dict:
            self.calls += 1
            return self.inner(req)

    counting = CountingTransport(old_bravo.handle)
    alpha_old = AlphaClient(card=old_bravo.get_card(), transport=counting)
    try:
        alpha_old.handshake()
        raise AssertionError("expected version_mismatch")
    except HandshakeError as e:
        print(f"    handshake refused: {e}")
        assert str(e) == "version_mismatch"
    assert counting.calls == 0, "no task must be sent when handshake fails"

    # --- '1.10' > '1.9' ordering and highest-common selection.
    semver_bravo = BravoAgent(protocol_versions=["1.9", "1.10"])

    class WideAlpha(AlphaClient):
        SUPPORTED_VERSIONS = ["1.9", "1.10"]

    alpha_semver = WideAlpha(card=semver_bravo.get_card(), transport=semver_bravo.handle)
    picked = alpha_semver.handshake()
    print(f"  semver pick -> v{picked} (1.10 preferred over 1.9)")
    assert picked == "1.10"
    assert version_key("1.10") > version_key("1.9")

    # --- Unknown method -> clean JSON-RPC -32601, no exception.
    print("\n  protocol error test (unknown method):")
    bad = bravo.handle(make_request("tasks/teleport", {}, "x1"))
    print(f"    response: {bad['error']}")
    assert "error" in bad and bad["error"]["code"] == JSONRPC_METHOD_NOT_FOUND

    print("\n  PASS -- cross-framework interop, version negotiation, clean errors.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Trust layer: HMAC signatures + replay + scoped authz
# ==========================================================================

def sign(payload: dict, secret: str) -> str:
    """HMAC-SHA256 over a CANONICAL JSON serialisation (sort_keys is essential)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(secret.encode(), canonical, hashlib.sha256).hexdigest()


def verify_signature(payload: dict, secret: str, sig: str) -> bool:
    """Constant-time comparison to avoid timing leaks."""
    return hmac.compare_digest(sign(payload, secret), sig)


def make_signed_envelope(sender: str, capability: str, body: dict, secret: str,
                         nonce: str, ts: float) -> dict:
    """Build an envelope and sign everything EXCEPT the 'sig' field itself."""
    env = {"sender": sender, "capability": capability, "body": body,
           "nonce": nonce, "ts": ts}
    env["sig"] = sign(env, secret)
    return env


@dataclass
class TrustGate:
    """
    Receiver-side verifier. Rejection reasons are checked in a fixed order so
    behaviour is predictable: unknown_sender -> bad_signature -> expired ->
    replay -> unauthorized -> accepted.
    """
    secrets: dict[str, str]                  # sender_id -> shared secret
    authz: dict[str, set[str]]               # sender_id -> allowed capabilities
    window_s: float = 30.0
    _seen_nonces: set[str] = field(default_factory=set)

    def accept(self, env: dict, now: float) -> tuple[bool, str]:
        sender = env.get("sender")
        secret = self.secrets.get(sender)
        if secret is None:
            return False, "unknown_sender"

        sig = env.get("sig", "")
        unsigned = {k: v for k, v in env.items() if k != "sig"}
        if not verify_signature(unsigned, secret, sig):
            return False, "bad_signature"  # body/capability/sender tampering caught here

        ts = env.get("ts", 0.0)
        if abs(now - ts) > self.window_s:
            return False, "expired"

        nonce = env.get("nonce")
        if nonce in self._seen_nonces:
            return False, "replay"

        if env.get("capability") not in self.authz.get(sender, set()):
            return False, "unauthorized"

        self._seen_nonces.add(nonce)  # consume only on full acceptance
        return True, "accepted"


def hard_ex2_trust_layer() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: Trust layer -- HMAC signatures + replay + scoped authz")
    print("=" * 60)

    secrets = {"orchestrator": "s3cr3t-orch", "intruder": "s3cr3t-intr"}
    authz = {
        "orchestrator": {"optimize_routes", "estimate_eta"},
        "intruder": {"ping"},  # legitimate but tightly scoped
    }
    gate = TrustGate(secrets=secrets, authz=authz, window_s=30.0)
    now = 1000.0

    # --- 1) Legitimate message -> accepted
    env = make_signed_envelope("orchestrator", "optimize_routes",
                               {"stops": ["Lyon", "Paris"]},
                               secrets["orchestrator"], nonce="n-001", ts=now)
    ok, reason = gate.accept(env, now=now)
    print(f"\n  legitimate         -> {reason}")
    assert ok and reason == "accepted"

    # --- 2) Tampered body after signing -> bad_signature
    tampered = make_signed_envelope("orchestrator", "optimize_routes",
                                    {"stops": ["Lyon"]},
                                    secrets["orchestrator"], nonce="n-002", ts=now)
    tampered["body"] = {"stops": ["EVIL"]}  # mutate after signature
    ok, reason = gate.accept(tampered, now=now)
    print(f"  tampered body      -> {reason}")
    assert not ok and reason == "bad_signature"

    # --- 3) Capability hijack after signing -> bad_signature (capability is signed)
    hijack = make_signed_envelope("orchestrator", "estimate_eta", {"x": 1},
                                  secrets["orchestrator"], nonce="n-003", ts=now)
    hijack["capability"] = "optimize_routes"  # swap the requested capability
    ok, reason = gate.accept(hijack, now=now)
    print(f"  capability hijack  -> {reason}")
    assert not ok and reason == "bad_signature"

    # --- 4) Replay: send the same valid envelope twice -> 2nd is replay
    replay_env = make_signed_envelope("orchestrator", "optimize_routes", {"stops": []},
                                      secrets["orchestrator"], nonce="n-004", ts=now)
    ok1, r1 = gate.accept(replay_env, now=now)
    ok2, r2 = gate.accept(replay_env, now=now)
    print(f"  replay (1st/2nd)   -> {r1} / {r2}")
    assert ok1 and r1 == "accepted"
    assert not ok2 and r2 == "replay"

    # --- 5) Expired: timestamp outside the window
    stale = make_signed_envelope("orchestrator", "optimize_routes", {"stops": []},
                                 secrets["orchestrator"], nonce="n-005", ts=now - 120.0)
    ok, reason = gate.accept(stale, now=now)
    print(f"  expired ts         -> {reason}")
    assert not ok and reason == "expired"

    # --- 6) Unauthorized: known sender, valid signature, capability out of scope
    unauth = make_signed_envelope("intruder", "optimize_routes", {"stops": []},
                                  secrets["intruder"], nonce="n-006", ts=now)
    ok, reason = gate.accept(unauth, now=now)
    print(f"  out-of-scope cap   -> {reason}")
    assert not ok and reason == "unauthorized"

    # --- 7) Unknown sender
    rogue_secret = "i-made-this-up"
    rogue = make_signed_envelope("ghost", "ping", {}, rogue_secret, nonce="n-007", ts=now)
    ok, reason = gate.accept(rogue, now=now)
    print(f"  unknown sender     -> {reason}")
    assert not ok and reason == "unknown_sender"

    # --- Sanity: signature helpers are stable and constant-time.
    assert verify_signature({"a": 1, "b": 2}, "k", sign({"b": 2, "a": 1}, "k")), \
        "canonical signing must be order-independent"

    print("\n  PASS -- HMAC integrity, replay/expiry defence, scoped authorization.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 19 HARD Solutions -- Inter-agent protocols (A2A)")
    print("#" * 60)

    hard_ex1_interop()
    hard_ex2_trust_layer()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

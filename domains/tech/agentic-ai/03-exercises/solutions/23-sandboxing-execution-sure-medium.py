"""
Solutions -- Day 23 (MEDIUM): Sandboxing & Safe Execution

Contains solutions for:
  - Medium Ex 1: Layered PolicyEngine (allowlist + denylist + capability model),
                 deny-first ordering. A denied/non-capable action is blocked, an
                 allowed one passes.
  - Medium Ex 2: MeteredExecutor -- simulated resource limits (CPU / wall-clock /
                 output size) that gracefully KILLS a runaway task without ever
                 running an infinite loop for real.
  - Medium Ex 3: EgressPolicy -- allowlist + exfiltration heuristic that alerts on
                 suspicious traffic toward an *allowed* host.

Self-contained & offline. No network, no real subprocess, no hostile commands.
The sandbox is modelled as a pure in-memory policy engine. The CapabilityToken
class is a minimal embed of 02-code/23-sandboxing-execution-sure.py (not imported).

Run:  python 03-exercises/solutions/23-sandboxing-execution-sure-medium.py
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


# ==========================================================================
# Shared minimal embed of CapabilityToken (from 02-code, trimmed)
# ==========================================================================

@dataclass
class CapabilityToken:
    """Scoped, time-limited authorisation for one tool/action (least-privilege)."""
    tool: str
    ttl: float = 60.0
    max_calls: int = 10
    _issued_at: float = field(default_factory=time.time, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)

    def is_valid(self) -> bool:
        return (time.time() - self._issued_at) < self.ttl and self._calls < self.max_calls

    def consume(self) -> bool:
        if not self.is_valid():
            return False
        self._calls += 1
        return True


# ==========================================================================
# MEDIUM EXERCISE 1 -- Layered allow/deny + capability PolicyEngine
# ==========================================================================

class PolicyEngine:
    """
    Layered policy: deny-first, then allowlist, then capability check.

    Decision order matters: the denylist always wins, even if the action is
    also allowlisted and a valid token is presented. This models the
    "least privilege + explicit kill-switch" posture of a sandbox runtime.
    """

    def __init__(self, allow: set[str], deny: set[str]) -> None:
        self.allow = set(allow)
        self.deny = set(deny)
        self.audit: list[dict[str, Any]] = []

    def classify(self, action: str, token: CapabilityToken | None) -> dict[str, Any]:
        # Layer A: denylist wins over everything else.
        if action in self.deny:
            return {"decision": "deny", "reason": "denylist", "layer": "denylist"}
        # Layer B: anything not explicitly allowed is denied (deny-by-default).
        if action not in self.allow:
            return {"decision": "deny", "reason": "not_in_allowlist", "layer": "allowlist"}
        # Layer C: capability must cover this exact action and still be valid.
        if token is None or token.tool != action or not token.is_valid():
            return {"decision": "deny", "reason": "capability_missing", "layer": "capability"}
        # All layers passed -> consume one call slot and allow.
        token.consume()
        return {"decision": "allow", "reason": "ok", "layer": "ok"}

    def enforce(self, action: str, token: CapabilityToken | None) -> bool:
        result = self.classify(action, token)
        self.audit.append({"action": action, **result})
        return result["decision"] == "allow"


def medium_ex1_policy_engine() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 1: Layered allow/deny + capability PolicyEngine")
    print("=" * 60)

    engine = PolicyEngine(
        allow={"read", "ptrace"},            # note: ptrace is also allowlisted...
        deny={"ptrace", "execve"},           # ...but denylist must override it.
    )

    # 1. Allowed action with a valid matching token -> allow.
    tok_read = CapabilityToken(tool="read", ttl=60, max_calls=3)
    d_read = engine.classify("read", tok_read)
    print(f"\n  read   (allow, valid token)   -> {d_read['decision']:5s} [{d_read['layer']}]")
    assert d_read["decision"] == "allow"
    assert tok_read._calls == 1, "valid allowed action must consume the token"

    # 2. Denylisted action -> blocked even WITH a valid token (deny-first).
    tok_ptrace = CapabilityToken(tool="ptrace", ttl=60, max_calls=3)
    d_ptrace = engine.classify("ptrace", tok_ptrace)
    print(f"  ptrace (deny+allow+token)     -> {d_ptrace['decision']:5s} [{d_ptrace['layer']}]")
    assert d_ptrace["decision"] == "deny" and d_ptrace["layer"] == "denylist"
    assert tok_ptrace._calls == 0, "denied action must NOT consume the token"

    # 3. Action absent from the allowlist -> blocked by allowlist.
    tok_socket = CapabilityToken(tool="socket", ttl=60, max_calls=3)
    d_socket = engine.classify("socket", tok_socket)
    print(f"  socket (not allowlisted)      -> {d_socket['decision']:5s} [{d_socket['layer']}]")
    assert d_socket["decision"] == "deny" and d_socket["layer"] == "allowlist"

    # 4. Allowlisted action but NO capability -> blocked by capability.
    d_nocap = engine.classify("read", None)
    print(f"  read   (no token)             -> {d_nocap['decision']:5s} [{d_nocap['layer']}]")
    assert d_nocap["decision"] == "deny" and d_nocap["layer"] == "capability"

    # enforce() must log every decision with its layer.
    engine.audit.clear()
    assert engine.enforce("read", CapabilityToken(tool="read")) is True
    assert engine.enforce("ptrace", CapabilityToken(tool="ptrace")) is False
    assert len(engine.audit) == 2
    assert {e["layer"] for e in engine.audit} == {"ok", "denylist"}
    print(f"\n  audit entries: {[ (e['action'], e['layer']) for e in engine.audit ]}")

    print("\n  PASS -- deny-first ordering proven; allowed action passes, others blocked.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- MeteredExecutor with resource limits
# ==========================================================================

@dataclass
class ResourceBudget:
    max_cpu_ticks: int
    max_wall_ticks: int
    max_output_bytes: int


@dataclass
class Op:
    cpu: int = 0     # CPU ticks consumed by this op
    wall: int = 0    # wall-clock ticks consumed by this op
    emit: int = 0    # output bytes produced by this op


class MeteredExecutor:
    """
    Simulated metered executor (no real process).

    Iterates a stream of Ops, accumulating cpu/wall/output. As soon as ANY
    budget is exceeded, it stops gracefully (a "kill") and returns a clean
    report -- it never iterates a (possibly infinite) op stream to completion.
    """

    def __init__(self, budget: ResourceBudget) -> None:
        self.budget = budget

    def run(self, ops: Iterable[Op]) -> dict[str, Any]:
        cpu_used = wall_used = output_bytes = ops_executed = 0
        killed_by: str | None = None

        for op in ops:
            # Apply this op's costs, then check budgets. A single op cannot
            # blow past the cap unnoticed -- we check right after applying it.
            cpu_used += op.cpu
            wall_used += op.wall
            output_bytes += op.emit
            ops_executed += 1

            if cpu_used > self.budget.max_cpu_ticks:
                killed_by = "cpu"
            elif wall_used > self.budget.max_wall_ticks:
                killed_by = "wall"
            elif output_bytes > self.budget.max_output_bytes:
                killed_by = "output"

            if killed_by is not None:
                break  # graceful kill -- stop consuming the (maybe infinite) stream

        status = "killed" if killed_by else "completed"
        return {
            "status": status,
            "killed_by": killed_by,
            "cpu_used": cpu_used,
            "wall_used": wall_used,
            "output_bytes": output_bytes,
            "ops_executed": ops_executed,
        }


def _infinite_cpu_ops() -> Iterator[Op]:
    """A never-ending generator (simulated infinite loop). MUST be bounded by the executor."""
    while True:
        yield Op(cpu=1, wall=1)


def medium_ex2_metered_executor() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 2: MeteredExecutor with resource limits")
    print("=" * 60)

    budget = ResourceBudget(max_cpu_ticks=100, max_wall_ticks=100, max_output_bytes=1024)
    ex = MeteredExecutor(budget)

    # 1. Short task under all budgets -> completed.
    short = [Op(cpu=5, wall=5, emit=10) for _ in range(5)]
    r_short = ex.run(short)
    print(f"\n  short task     -> {r_short['status']:9s} cpu={r_short['cpu_used']} "
          f"out={r_short['output_bytes']}B ops={r_short['ops_executed']}")
    assert r_short["status"] == "completed"
    assert r_short["cpu_used"] == 25 and r_short["output_bytes"] == 50
    assert r_short["ops_executed"] == 5

    # 2. (Quasi-)infinite loop -> killed by CPU, never hangs.
    r_loop = ex.run(_infinite_cpu_ops())
    print(f"  infinite loop  -> {r_loop['status']:9s} killed_by={r_loop['killed_by']} "
          f"cpu={r_loop['cpu_used']} ops={r_loop['ops_executed']}")
    assert r_loop["status"] == "killed"
    assert r_loop["killed_by"] in ("cpu", "wall")
    assert r_loop["ops_executed"] == 101  # 100 within budget + 1 that tips it over

    # 3. Output bomb -> killed by output size.
    bomb = [Op(emit=500) for _ in range(10)]   # 5000B total >> 1024B cap
    r_bomb = ex.run(bomb)
    print(f"  output bomb    -> {r_bomb['status']:9s} killed_by={r_bomb['killed_by']} "
          f"out={r_bomb['output_bytes']}B ops={r_bomb['ops_executed']}")
    assert r_bomb["status"] == "killed" and r_bomb["killed_by"] == "output"
    assert r_bomb["ops_executed"] < len(bomb), "killed task must run fewer ops than total"

    print("\n  PASS -- runaway tasks killed gracefully on cpu/wall/output budgets.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- EgressPolicy with exfiltration detection
# ==========================================================================

class EgressPolicy:
    """
    Egress allowlist + exfiltration heuristic.

    Two-stage decision:
      1. allowlist (exact + *.wildcard, mirroring 02-code) -> block unknown hosts
      2. exfiltration heuristic on ALLOWED hosts -> 'alert' on suspicious URL/payload
         (the egress filter alone is not enough: see theory section 6.2).
    """

    _B64ISH = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
    _SUSPECT_PARAMS = ("data=", "exfil", "dump", "leak=")

    def __init__(self, allowed_domains: set[str], max_payload_bytes: int = 100_000) -> None:
        self.allowed = set(allowed_domains)
        self.max_payload_bytes = max_payload_bytes
        self.allows = self.blocks = self.alerts = 0

    def _domain_of(self, url: str) -> str:
        without_scheme = url.split("://", 1)[-1]
        return without_scheme.split("/")[0].split(":")[0].lower().strip()

    def _is_allowed(self, domain: str) -> bool:
        for rule in self.allowed:
            if rule.startswith("*."):
                suffix = rule[2:]
                if domain == suffix or domain.endswith("." + suffix):
                    return True
            elif domain == rule:
                return True
        return False

    def _looks_like_exfil(self, url: str, payload_bytes: int) -> bool:
        low = url.lower()
        if any(p in low for p in self._SUSPECT_PARAMS):
            return True
        query = url.split("?", 1)[1] if "?" in url else ""
        if self._B64ISH.search(query):
            return True
        if payload_bytes > self.max_payload_bytes:
            return True
        return False

    def check(self, url: str, payload_bytes: int = 0) -> dict[str, Any]:
        domain = self._domain_of(url)
        if not self._is_allowed(domain):
            self.blocks += 1
            return {"action": "block", "reason": "domain_not_allowed", "domain": domain}
        if self._looks_like_exfil(url, payload_bytes):
            self.alerts += 1
            return {"action": "alert", "reason": "possible_exfiltration", "domain": domain}
        self.allows += 1
        return {"action": "allow", "reason": "ok", "domain": domain}

    def report(self) -> dict[str, int]:
        return {"allowed": self.allows, "blocked": self.blocks, "alerted": self.alerts}


def medium_ex3_egress_policy() -> None:
    print("\n" + "=" * 60)
    print("  MEDIUM 3: EgressPolicy with exfiltration detection")
    print("=" * 60)

    policy = EgressPolicy(
        allowed_domains={"api.anthropic.com", "*.wikipedia.org", "pypi.org"},
        max_payload_bytes=10_000,
    )

    long_b64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5"  # >= 40 chars
    cases = [
        # (label, url, payload_bytes, expected_action)
        ("legit api",        "https://api.anthropic.com/v1/messages", 200,   "allow"),
        ("wildcard wiki",    "https://en.wikipedia.org/wiki/GVisor",  100,   "allow"),
        ("unknown host",     "https://random-host.example.net/ping",  10,    "block"),
        ("evil domain",      "https://evil-c2.example.com/x",          10,    "block"),
        ("exfil via param",  f"https://api.anthropic.com/log?data={long_b64}", 50, "alert"),
        ("big payload",      "https://pypi.org/upload",               50_000, "alert"),
    ]

    print()
    results = {}
    for label, url, payload, expected in cases:
        r = policy.check(url, payload_bytes=payload)
        results[label] = r
        print(f"  {label:16s} -> {r['action']:5s} [{r['reason']}] ({r['domain']})")
        assert r["action"] == expected, f"{label}: got {r['action']}, expected {expected}"

    rep = policy.report()
    print(f"\n  report: {rep}")
    assert rep == {"allowed": 2, "blocked": 2, "alerted": 2}
    # Wildcard matching works; exfil toward an ALLOWED host is alerted, not silently passed.
    assert results["wildcard wiki"]["action"] == "allow"
    assert results["exfil via param"]["action"] == "alert"

    print("\n  PASS -- allowlist enforced, exfil toward allowed host detected.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 23 MEDIUM Solutions -- Sandboxing & Safe Execution")
    print("#" * 60)

    medium_ex1_policy_engine()
    medium_ex2_metered_executor()
    medium_ex3_egress_policy()

    print("\n" + "#" * 60)
    print("  All medium solutions executed successfully.")
    print("#" * 60 + "\n")

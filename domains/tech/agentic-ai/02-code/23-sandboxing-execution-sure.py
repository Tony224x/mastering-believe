"""
Day 23 -- Sandboxing & Safe Execution: infra-level isolation concepts in pure Python.

Demonstrates:
  1. CapabilityToken        -- capability-based tool access (least-privilege, TTL, scope)
  2. NetworkAllowlistProxy  -- mock egress filter: allows/blocks domains by allowlist
  3. ResourceLimiter        -- timeout + best-effort resource limits via subprocess
  4. run_in_subprocess      -- execute Python code in a child process with timeout,
                              optional stdout capture, and allowed-paths concept
  Shows:
    - Authorized case     : code runs within limits, allowed domain, valid capability
    - Blocked case 1      : capability missing -> tool call refused
    - Blocked case 2      : domain not in allowlist -> egress refused
    - Blocked case 3      : timeout exceeded -> process killed

Dependencies: stdlib only. No root, no Docker, no special kernel features required.
Best-effort resource limiting (resource module available on Linux/macOS).

Run:
    python domains/tech/agentic-ai/02-code/23-sandboxing-execution-sure.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Optional: resource module (POSIX only — Linux, macOS)
# ---------------------------------------------------------------------------
try:
    import resource as _resource  # noqa: F401
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False  # Windows: skip resource limits


# ===========================================================================
# 1. CAPABILITY TOKEN
# ===========================================================================

@dataclass
class CapabilityToken:
    """
    Represents a scoped, time-limited authorisation for one tool.

    Design: the token is a value object. The holder presents it to the tool
    registry; the registry validates scope, TTL, and max_calls before
    executing the tool. No ambient authority — if you don't have the token,
    you can't call the tool.
    """

    tool: str                       # Name of the tool this token grants access to
    ttl: float = 60.0               # Seconds until expiry
    scope: dict[str, Any] = field(default_factory=dict)  # Extra constraints
    max_calls: int = 10             # Maximum number of invocations
    _issued_at: float = field(default_factory=time.time, init=False, repr=False)
    _calls: int = field(default=0, init=False, repr=False)
    _token_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:8], init=False, repr=False
    )

    # ---- validation --------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if the token is still within TTL and call budget."""
        elapsed = time.time() - self._issued_at
        return elapsed < self.ttl and self._calls < self.max_calls

    def consume(self) -> bool:
        """
        Attempt to consume one call slot.
        Returns True on success, False if the token is exhausted or expired.
        """
        if not self.is_valid():
            return False
        self._calls += 1
        return True

    def remaining_ttl(self) -> float:
        """Seconds left before expiry (can be negative if expired)."""
        return self.ttl - (time.time() - self._issued_at)

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid() else "EXPIRED/EXHAUSTED"
        return (
            f"CapabilityToken(tool={self.tool!r}, id={self._token_id}, "
            f"calls={self._calls}/{self.max_calls}, "
            f"ttl_left={self.remaining_ttl():.1f}s, status={status})"
        )


class CapabilityRegistry:
    """
    Tool registry that enforces capability-based access.

    An agent receives only the tokens it needs for the current task.
    Calling a tool without a valid token is refused immediately.
    """

    def __init__(self) -> None:
        # Maps tool name -> callable
        self._tools: dict[str, Any] = {}

    def register(self, name: str, fn: Any) -> None:
        """Register a tool implementation."""
        self._tools[name] = fn

    def call(
        self,
        tool: str,
        token: CapabilityToken | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a tool if and only if the caller holds a valid capability token.

        The token is consumed on success. On failure, a structured error is
        returned — never an exception — so the agent can handle it gracefully.
        """
        # 1. No token at all
        if token is None:
            return {
                "error": f"capability_missing: no token provided for tool '{tool}'"
            }

        # 2. Token is for a different tool
        if token.tool != tool:
            return {
                "error": (
                    f"capability_mismatch: token is for '{token.tool}', "
                    f"not '{tool}'"
                )
            }

        # 3. Token expired or exhausted
        if not token.consume():
            return {
                "error": (
                    f"capability_invalid: token for '{tool}' is expired or "
                    f"call-budget exhausted"
                )
            }

        # 4. Tool not registered
        if tool not in self._tools:
            return {"error": f"tool_unknown: '{tool}' is not registered"}

        # 5. Execute
        try:
            result = self._tools[tool](**kwargs)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"tool_exception: {exc}"}


# ===========================================================================
# 2. NETWORK ALLOWLIST PROXY (mock)
# ===========================================================================

class NetworkAllowlistProxy:
    """
    Mock egress filter.

    In production this would be an actual HTTP proxy (e.g. using mitmproxy
    or a simple CONNECT-capable server). Here we simulate the allow/deny
    decision that the proxy makes for each outbound request.

    Architecture in sandbox-runtime (Anthropic):
      - The sandboxed process has HTTP_PROXY / HTTPS_PROXY set to a local proxy.
      - The proxy intercepts every TCP connection (HTTPS via CONNECT).
      - The proxy checks the target hostname against the allowlist.
      - Denied connections receive a 403 response; the attempt is logged.
    """

    def __init__(self, allowed_domains: set[str]) -> None:
        # The allowlist is a set of exact hostnames or *.wildcard prefixes.
        self._allowed = allowed_domains
        self._log: list[dict[str, Any]] = []

    def _is_allowed(self, domain: str) -> bool:
        """
        Check if the domain is covered by the allowlist.

        Supports exact matches ("api.anthropic.com") and wildcard prefix
        matches ("*.wikipedia.org" matches "en.wikipedia.org").
        """
        domain = domain.lower().strip()
        for rule in self._allowed:
            if rule.startswith("*."):
                # Wildcard: *.example.com matches sub.example.com
                suffix = rule[2:]  # "example.com"
                if domain == suffix or domain.endswith("." + suffix):
                    return True
            elif domain == rule:
                return True
        return False

    def request(self, url: str, method: str = "GET") -> dict[str, Any]:
        """
        Simulate an outbound HTTP request through the proxy.

        Returns a response dict with 'status' (200/403) and 'body'.
        Every attempt is logged (including denied ones).
        """
        # Extract hostname from URL (very simplified — no urllib to keep it minimal)
        try:
            # url like "https://api.anthropic.com/v1/messages"
            without_scheme = url.split("://", 1)[-1]      # "api.anthropic.com/v1/..."
            domain = without_scheme.split("/")[0].split(":")[0]  # "api.anthropic.com"
        except Exception:  # noqa: BLE001
            domain = "<invalid>"

        allowed = self._is_allowed(domain)
        entry = {
            "ts": time.time(),
            "method": method,
            "url": url,
            "domain": domain,
            "allowed": allowed,
        }
        self._log.append(entry)

        if allowed:
            return {
                "status": 200,
                "body": f"[mock] 200 OK from {domain}",
                "domain": domain,
            }
        return {
            "status": 403,
            "body": f"[proxy] 403 Forbidden — domain '{domain}' not in allowlist",
            "domain": domain,
        }

    def audit_log(self) -> list[dict[str, Any]]:
        """Return the full request log (allowed + denied)."""
        return list(self._log)


# ===========================================================================
# 3. RESOURCE LIMITER
# ===========================================================================

class ResourceLimiter:
    """
    Best-effort resource limits for child processes.

    Uses the POSIX 'resource' module (Linux/macOS) when available.
    Falls back to timeout-only on platforms that lack it (Windows, some CI).

    In production this would be complemented by cgroup v2 limits set by the
    sandbox runtime (memory.max, cpu.max) before spawning the child.
    """

    def __init__(
        self,
        max_cpu_seconds: int = 5,
        max_memory_bytes: int = 64 * 1024 * 1024,  # 64 MB
    ) -> None:
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_bytes = max_memory_bytes

    def _preexec(self) -> None:
        """
        Called in the child process (before exec) on POSIX systems.

        Sets RLIMIT_CPU  : hard CPU time limit (SIGKILL after N seconds).
        Sets RLIMIT_AS   : hard virtual address space limit.
        Both limits are best-effort: the kernel enforces them if supported.
        """
        if not HAS_RESOURCE:
            return  # Windows: no-op

        import resource  # noqa: PLC0415

        # CPU time: (soft, hard) — soft raises SIGXCPU, hard raises SIGKILL
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.max_cpu_seconds, self.max_cpu_seconds),
            )
        except (ValueError, resource.error):
            pass  # Some environments restrict setrlimit — best-effort

        # Virtual memory: protect against memory bombs
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.max_memory_bytes, self.max_memory_bytes),
            )
        except (ValueError, resource.error):
            pass

    def build_preexec(self):  # noqa: ANN201
        """Return the preexec_fn to pass to subprocess.Popen."""
        if sys.platform == "win32":
            return None  # preexec_fn unsupported on Windows
        return self._preexec


# ===========================================================================
# 4. run_in_subprocess — execute code in a restricted child process
# ===========================================================================

def run_in_subprocess(
    code: str,
    *,
    timeout: float = 5.0,
    allowed_paths: list[str] | None = None,
    limiter: ResourceLimiter | None = None,
    capture_stderr: bool = False,
) -> dict[str, Any]:
    """
    Execute *code* (a Python string) in a fresh subprocess.

    Safety measures applied:
      - Hard wall-clock timeout: the process is killed after *timeout* seconds.
      - Best-effort POSIX resource limits via *limiter* (CPU time, memory).
      - The child process environment is stripped to the minimum needed to run
        Python (no AWS_*, ANTHROPIC_API_KEY, DATABASE_URL, etc.).
      - allowed_paths is informational here (shown in the audit output); in a
        real sandbox it would be fed to bubblewrap --ro-bind / --bind args.

    Returns a dict with keys:
      'returncode': int
      'stdout': str
      'stderr': str (only if capture_stderr=True)
      'timed_out': bool
      'elapsed': float  (seconds)
    """
    allowed_paths = allowed_paths or []
    limiter = limiter or ResourceLimiter()

    # --- Build a minimal environment (strip secrets) -------------------------
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }

    # --- Launch child process -------------------------------------------------
    stderr_target = subprocess.PIPE if capture_stderr else subprocess.DEVNULL
    start = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            env=safe_env,
            preexec_fn=limiter.build_preexec(),
        )
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        returncode = proc.returncode

    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_bytes, stderr_bytes = proc.communicate()  # drain buffers
        returncode = -9
        timed_out = True

    elapsed = time.monotonic() - start

    result = {
        "returncode": returncode,
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "timed_out": timed_out,
        "elapsed": round(elapsed, 3),
    }
    if capture_stderr:
        result["stderr"] = (stderr_bytes or b"").decode("utf-8", errors="replace")

    return result


# ===========================================================================
# DEMO
# ===========================================================================

def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def demo_capability_tokens() -> None:
    """Show capability-based access: authorised call vs missing token."""
    _separator("1. Capability Tokens")

    registry = CapabilityRegistry()
    registry.register("web_search", lambda query: f"Search results for: {query}")
    registry.register("write_file", lambda path, content: f"Written {len(content)} bytes to {path}")

    # --- Authorised: agent has a valid web_search token
    token_search = CapabilityToken(tool="web_search", ttl=60, max_calls=5)
    print(f"\n[Token] {token_search}")

    res = registry.call("web_search", token_search, query="gVisor vs Firecracker")
    print(f"[OK]    web_search result: {res}")

    # --- Blocked: agent tries to write a file but has no write_file token
    res_no_token = registry.call("write_file", None, path="/etc/passwd", content="pwned")
    print(f"[BLOCK] write_file without token: {res_no_token}")

    # --- Blocked: wrong token (search token used for write_file)
    res_wrong = registry.call("write_file", token_search, path="/tmp/out.txt", content="data")
    print(f"[BLOCK] write_file with wrong token: {res_wrong}")

    # --- Exhaust the token
    print("\n[INFO]  Exhausting search token (max_calls=5, already used 1)...")
    for i in range(4):
        registry.call("web_search", token_search, query=f"q{i}")
    res_exhausted = registry.call("web_search", token_search, query="one more")
    print(f"[BLOCK] After exhaustion: {res_exhausted}")


def demo_network_proxy() -> None:
    """Show egress filtering: allowed domain vs blocked domain."""
    _separator("2. Network Allowlist Proxy (Egress Filtering)")

    ALLOWED = {
        "api.anthropic.com",
        "*.wikipedia.org",
        "pypi.org",
    }
    proxy = NetworkAllowlistProxy(allowed_domains=ALLOWED)

    requests = [
        ("https://api.anthropic.com/v1/messages", "POST"),
        ("https://en.wikipedia.org/wiki/GVisor", "GET"),
        ("https://evil-c2.example.com/exfil?data=secret", "POST"),
        ("https://pypi.org/simple/", "GET"),
        ("https://attacker.ngrok.io/callback", "GET"),
    ]

    for url, method in requests:
        resp = proxy.request(url, method=method)
        icon = "OK  " if resp["status"] == 200 else "DENY"
        print(f"[{icon}] {method:4s} {url[:55]:<55}  -> HTTP {resp['status']}")

    denied_count = sum(1 for e in proxy.audit_log() if not e["allowed"])
    print(f"\n[AUDIT] {denied_count}/{len(requests)} requests denied, logged for review.")


def demo_subprocess_timeout() -> None:
    """Show run_in_subprocess: normal run, timeout exceeded."""
    _separator("3. Subprocess Execution with Timeout & Resource Limits")

    limiter = ResourceLimiter(max_cpu_seconds=3, max_memory_bytes=64 * 1024 * 1024)

    # --- Authorised: quick computation
    code_ok = textwrap.dedent("""
        import math
        result = sum(math.sqrt(i) for i in range(1_000_000))
        print(f"Sum of sqrts: {result:.2f}")
    """).strip()

    print("\n[RUN]  Authorised computation (1M sqrt)...")
    res = run_in_subprocess(
        code_ok,
        timeout=10.0,
        allowed_paths=["/tmp"],
        limiter=limiter,
        capture_stderr=True,
    )
    icon = "OK  " if res["returncode"] == 0 else "FAIL"
    print(f"[{icon}] returncode={res['returncode']}, elapsed={res['elapsed']}s, "
          f"timed_out={res['timed_out']}")
    if res["stdout"]:
        print(f"       stdout: {res['stdout'].strip()}")

    # --- Blocked: infinite loop exceeds wall-clock timeout
    code_loop = textwrap.dedent("""
        import time
        print("Starting infinite loop...")
        while True:
            time.sleep(0.1)
    """).strip()

    print("\n[RUN]  Infinite loop (timeout=2s)...")
    res_loop = run_in_subprocess(
        code_loop,
        timeout=2.0,
        limiter=limiter,
        capture_stderr=True,
    )
    icon = "KILL" if res_loop["timed_out"] else "DONE"
    print(f"[{icon}] returncode={res_loop['returncode']}, "
          f"elapsed={res_loop['elapsed']}s, timed_out={res_loop['timed_out']}")
    if res_loop["stdout"]:
        print(f"       stdout: {res_loop['stdout'].strip()}")


def demo_combined() -> None:
    """
    Show the full defence-in-depth stack:
      1. Check capability token before executing anything
      2. Validate egress target through proxy
      3. Execute code in subprocess with timeout
    """
    _separator("4. Combined: Capability + Egress + Subprocess")

    registry = CapabilityRegistry()
    proxy = NetworkAllowlistProxy(allowed_domains={"api.anthropic.com", "pypi.org"})
    limiter = ResourceLimiter(max_cpu_seconds=3, max_memory_bytes=32 * 1024 * 1024)

    def safe_http_tool(url: str) -> str:
        resp = proxy.request(url)
        if resp["status"] != 200:
            raise PermissionError(resp["body"])
        return resp["body"]

    registry.register("http_get", safe_http_tool)

    # Token for http_get only
    http_token = CapabilityToken(tool="http_get", ttl=30, max_calls=3)

    scenarios = [
        # (label, token, url, code)
        (
            "Allowed API call",
            http_token,
            "https://api.anthropic.com/health",
            None,
        ),
        (
            "Blocked egress (evil domain)",
            http_token,
            "https://evil.com/steal",
            None,
        ),
        (
            "Blocked capability (no token)",
            None,
            "https://api.anthropic.com/health",
            None,
        ),
    ]

    for label, tok, url, _ in scenarios:
        res = registry.call("http_get", tok, url=url)
        if "ok" in res:
            print(f"[OK  ] {label}: {res['result']}")
        else:
            print(f"[DENY] {label}: {res['error']}")

    # Subprocess demo with resource limit
    print()
    code_fast = "print('Subprocess: hello from sandboxed child')"
    res = run_in_subprocess(code_fast, timeout=5.0, limiter=limiter)
    print(f"[SUBP] returncode={res['returncode']}, stdout={res['stdout'].strip()!r}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Day 23 — Sandboxing & Safe Execution: infra-level isolation (stdlib demo)")
    print("Note: this is a mock/conceptual demo — no root or kernel features required.\n")

    demo_capability_tokens()
    demo_network_proxy()
    demo_subprocess_timeout()
    demo_combined()

    print("\n" + "=" * 60)
    print("  Demo complete. All concepts illustrated without root or network.")
    print("=" * 60)

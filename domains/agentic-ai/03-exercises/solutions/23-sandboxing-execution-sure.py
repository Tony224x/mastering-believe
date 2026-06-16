"""
Day 23 -- Solutions to the easy exercises for sandboxing & safe execution.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/23-sandboxing-execution-sure.py
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import helpers from Day 23 code module
# ---------------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# importlib handles the hyphen in the module filename
day23 = importlib.import_module("23-sandboxing-execution-sure")

CapabilityToken = day23.CapabilityToken
CapabilityRegistry = day23.CapabilityRegistry
NetworkAllowlistProxy = day23.NetworkAllowlistProxy
ResourceLimiter = day23.ResourceLimiter
run_in_subprocess = day23.run_in_subprocess


# ===========================================================================
# SOLUTION 1 -- Revocation de CapabilityToken
# ===========================================================================

class CapabilityStore:
    """
    Central store for capability tokens with revocation support.

    The store holds the authoritative list of valid (non-revoked) tokens.
    Revoking a token is O(1) and immediate — no TTL modification needed.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, CapabilityToken] = {}  # token_id -> token
        self._revoked: set[str] = set()               # revoked token ids

    def issue(self, tool: str, ttl: float = 60.0, max_calls: int = 10) -> CapabilityToken:
        """Create, register, and return a new capability token."""
        token = CapabilityToken(tool=tool, ttl=ttl, max_calls=max_calls)
        self._tokens[token._token_id] = token
        return token

    def revoke(self, token_id: str) -> bool:
        """
        Revoke a token by ID.
        Returns True if the token existed and was revoked, False otherwise.
        """
        if token_id in self._tokens:
            self._revoked.add(token_id)
            return True
        return False

    def is_valid(self, token: CapabilityToken) -> bool:
        """
        Return True only if the token is:
          1. Registered in this store
          2. Not revoked
          3. Internally valid (TTL not expired, calls not exhausted)
        """
        if token._token_id not in self._tokens:
            return False
        if token._token_id in self._revoked:
            return False
        return token.is_valid()


class RevocableRegistry(CapabilityRegistry):
    """
    Extended registry that checks revocation status via a CapabilityStore.
    """

    def __init__(self, store: CapabilityStore) -> None:
        super().__init__()
        self._store = store

    def call(  # type: ignore[override]
        self,
        tool: str,
        token: CapabilityToken | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Extra check: revocation
        if token is not None and not self._store.is_valid(token):
            return {
                "error": (
                    f"capability_revoked_or_invalid: token '{token._token_id}' "
                    f"for tool '{tool}' has been revoked or is no longer valid"
                )
            }
        return super().call(tool, token, **kwargs)


def solution_1() -> None:
    print("\n" + "=" * 60)
    print("  SOLUTION 1 — Revocation de CapabilityToken")
    print("=" * 60)

    store = CapabilityStore()
    registry = RevocableRegistry(store=store)
    registry.register("send_email", lambda to, subject: f"Email sent to {to}: {subject}")

    # Issue a token
    token = store.issue(tool="send_email", ttl=60.0, max_calls=5)
    print(f"\n[TOKEN ] Issued: {token}")

    # Call 1: should succeed
    res1 = registry.call("send_email", token, to="alice@company.com", subject="Report")
    print(f"[CALL1 ] {res1}")

    # Revoke the token
    revoked = store.revoke(token._token_id)
    print(f"\n[REVOKE] store.revoke('{token._token_id}') -> {revoked}")

    # Call 2: should be refused
    res2 = registry.call("send_email", token, to="eve@evil.com", subject="Exfil")
    print(f"[CALL2 ] {res2}")

    assert "ok" in res1, "Call 1 should succeed"
    assert "revoked" in res2.get("error", "").lower(), "Call 2 should mention revocation"
    print("\n[PASS  ] Revocation works correctly.")


# ===========================================================================
# SOLUTION 2 -- Proxy avec journalisation structuree et rapport d'audit
# ===========================================================================

class AuditableProxy(NetworkAllowlistProxy):
    """
    Extends NetworkAllowlistProxy with:
      - session_id tracking
      - sequence numbering
      - URL SHA-256 fingerprinting
      - summary() and suspicious_sessions() analytics
    """

    def __init__(self, allowed_domains: set[str], session_id: str) -> None:
        super().__init__(allowed_domains)
        self._session_id = session_id
        self._seq = 0

    def request(self, url: str, method: str = "GET") -> dict[str, Any]:
        """Override to enrich the log entry with session metadata."""
        self._seq += 1
        response = super().request(url, method=method)

        # Enrich the last log entry added by the parent
        entry = self._log[-1]
        entry["session_id"] = self._session_id
        entry["seq"] = self._seq
        entry["url_hash"] = hashlib.sha256(url.encode()).hexdigest()

        return response

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics for this session."""
        total = len(self._log)
        denied = [e for e in self._log if not e["allowed"]]
        denied_count = len(denied)
        unique_domains_denied = list({e["domain"] for e in denied})
        rate = denied_count / total if total > 0 else 0.0

        return {
            "total_requests": total,
            "denied_requests": denied_count,
            "unique_domains_denied": unique_domains_denied,
            "denial_rate": round(rate, 4),
        }

    def suspicious_sessions(self, threshold: float = 0.5) -> bool:
        """Return True if the denial rate exceeds *threshold*."""
        s = self.summary()
        return s["denial_rate"] > threshold


def solution_2() -> None:
    print("\n" + "=" * 60)
    print("  SOLUTION 2 — Proxy avec journalisation et audit")
    print("=" * 60)

    ALLOWED = {"api.anthropic.com", "pypi.org", "docs.python.org"}
    proxy = AuditableProxy(allowed_domains=ALLOWED, session_id="sess-42")

    urls = [
        ("https://api.anthropic.com/v1/messages", "POST"),
        ("https://pypi.org/simple/requests/", "GET"),
        ("https://docs.python.org/3/library/subprocess.html", "GET"),
        ("https://evil.exfil.io/data", "POST"),
        ("https://attacker.ngrok.io/callback", "GET"),
        ("https://c2.darknet.xyz/beacon", "GET"),
        ("https://api.anthropic.com/v1/count_tokens", "POST"),
        ("https://malware-drop.ru/payload.sh", "GET"),
        ("https://botnet-c2.xyz/checkin", "GET"),   # extra denied -> 5/9 > 50%
    ]

    print()
    for url, method in urls:
        resp = proxy.request(url, method=method)
        icon = "OK  " if resp["status"] == 200 else "DENY"
        print(f"  [{icon}] {method:4s} {url[:52]:<52}  HTTP {resp['status']}")

    summary = proxy.summary()
    print(f"\n[SUMMARY] {summary}")

    is_suspicious = proxy.suspicious_sessions(threshold=0.5)
    print(f"[SUSPICIOUS?] denial_rate={summary['denial_rate']:.0%}, "
          f"threshold=50% -> {is_suspicious}")

    # Verify log enrichment
    entry = proxy.audit_log()[0]
    assert "session_id" in entry, "entry must have session_id"
    assert "seq" in entry, "entry must have seq"
    assert "url_hash" in entry and len(entry["url_hash"]) == 64, "entry must have sha256 hash"
    assert is_suspicious, "More than 50% denied -> should be suspicious"
    print("[PASS  ] Audit log and suspicious detection work correctly.")


# ===========================================================================
# SOLUTION 3 -- Sandbox multi-niveaux avec allowlist de chemins
# ===========================================================================

def validate_path(path: str, allowed_paths: list[str]) -> tuple[bool, str]:
    """
    Check that *path* is safely inside one of *allowed_paths*.

    Guards against:
      - Absolute paths outside allowed dirs
      - Path traversal via '..' components
    Returns (True, "") on success, (False, reason) on failure.
    """
    # Reject if the raw string contains '..' (belt-and-suspenders)
    if ".." in Path(path).parts:
        return False, f"path traversal detected in '{path}'"

    try:
        resolved = Path(path).resolve()
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot resolve path: {exc}"

    for allowed in allowed_paths:
        try:
            allowed_resolved = Path(allowed).resolve()
            # is_relative_to available Python 3.9+; fallback for 3.8
            try:
                if resolved.is_relative_to(allowed_resolved):
                    return True, ""
            except AttributeError:
                # Python < 3.9: use commonpath
                common = os.path.commonpath([str(resolved), str(allowed_resolved)])
                if common == str(allowed_resolved):
                    return True, ""
        except Exception:  # noqa: BLE001
            continue

    return False, f"'{path}' is outside allowed directories {allowed_paths}"


class FilesystemSandbox:
    """
    Simulates bubblewrap filesystem isolation:
    only paths under allowed_dirs can be read or written.
    """

    def __init__(self, allowed_dirs: list[str]) -> None:
        self._allowed = allowed_dirs

    def read(self, path: str) -> str:
        ok, reason = validate_path(path, self._allowed)
        if not ok:
            return f"[SANDBOX ERROR] read blocked: {reason}"
        try:
            return Path(path).read_text()
        except OSError as exc:
            return f"[IO ERROR] {exc}"

    def write(self, path: str, content: str) -> str:
        ok, reason = validate_path(path, self._allowed)
        if not ok:
            return f"[SANDBOX ERROR] write blocked: {reason}"
        try:
            Path(path).write_text(content)
            return f"[OK] wrote {len(content)} bytes to '{path}'"
        except OSError as exc:
            return f"[IO ERROR] {exc}"


def sandboxed_exec(
    code: str,
    work_dir: str,
    timeout: float,
) -> dict[str, Any]:
    """
    Execute *code* in a subprocess, but only if *work_dir* is under /tmp.

    Returns:
      'stdout', 'timed_out', 'returncode', 'path_validated'
    """
    ok, reason = validate_path(work_dir, ["/tmp"])
    if not ok:
        return {
            "stdout": "",
            "timed_out": False,
            "returncode": -1,
            "path_validated": False,
            "error": reason,
        }

    limiter = ResourceLimiter(max_cpu_seconds=int(timeout) + 1)
    result = run_in_subprocess(code, timeout=timeout, limiter=limiter)
    result["path_validated"] = True
    return result


def solution_3() -> None:
    import tempfile

    print("\n" + "=" * 60)
    print("  SOLUTION 3 — Sandbox filesystem + timeout")
    print("=" * 60)

    sandbox = FilesystemSandbox(allowed_dirs=["/tmp"])

    # Scenario A: write + read inside /tmp (authorised)
    with tempfile.NamedTemporaryFile(
        dir="/tmp", suffix=".txt", delete=False, mode="w"
    ) as tf:
        tf.write("hello sandbox")
        tmp_path = tf.name

    res_read = sandbox.read(tmp_path)
    print(f"\n[A - OK  ] read '/tmp/...': '{res_read}'")
    assert "hello sandbox" in res_read, "Should read the file content"

    # Scenario B: read /etc/passwd (blocked)
    res_blocked = sandbox.read("/etc/passwd")
    print(f"[B - DENY] read '/etc/passwd': '{res_blocked}'")
    assert "SANDBOX ERROR" in res_blocked, "Should be blocked"

    # Scenario C: path traversal attempt (blocked)
    res_traversal = sandbox.read("/tmp/../etc/passwd")
    print(f"[C - DENY] traversal '/tmp/../etc/passwd': '{res_traversal}'")
    # validate_path resolves the path, so /tmp/../etc/passwd -> /etc/passwd -> denied
    assert "SANDBOX ERROR" in res_traversal or "blocked" in res_traversal.lower()

    # Scenario D: sandboxed_exec — fast code, valid work_dir
    code_fast = "print('sandboxed child alive')"
    res_exec = sandboxed_exec(code_fast, work_dir="/tmp", timeout=5.0)
    print(f"\n[D - OK  ] sandboxed_exec fast: returncode={res_exec['returncode']}, "
          f"stdout={res_exec['stdout'].strip()!r}")
    assert res_exec["path_validated"] and res_exec["returncode"] == 0

    # Scenario E: sandboxed_exec — timeout
    code_slow = textwrap.dedent("""
        import time
        time.sleep(10)
    """).strip()
    res_timeout = sandboxed_exec(code_slow, work_dir="/tmp", timeout=1.0)
    print(f"[E - KILL] sandboxed_exec timeout: timed_out={res_timeout['timed_out']}, "
          f"returncode={res_timeout['returncode']}")
    assert res_timeout["timed_out"], "Should have timed out"

    # Scenario F: invalid work_dir (not under /tmp)
    res_bad_dir = sandboxed_exec("print('x')", work_dir="/home/user", timeout=5.0)
    print(f"[F - DENY] sandboxed_exec bad work_dir: path_validated={res_bad_dir['path_validated']}")
    assert not res_bad_dir["path_validated"]

    # Cleanup
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    print("\n[PASS  ] Filesystem sandbox and timeout work correctly.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print("Day 23 — Solutions: Sandboxing & Safe Execution")
    print("Running all 3 solutions...\n")

    solution_1()
    solution_2()
    solution_3()

    print("\n" + "=" * 60)
    print("  All solutions passed.")
    print("=" * 60)

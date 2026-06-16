"""
Solutions -- Day 23 (HARD): Sandboxing & Safe Execution

Contains solutions for:
  - Hard Ex 1: LayeredSandbox (fs-jail + egress + resource-caps + capability-drop)
               + ablation matrix proving EACH layer is necessary (disable a layer
               -> exactly the matching attack succeeds, the others stay blocked).
  - Hard Ex 2: SandboxGuard + adversarial red-team suite (path traversal, symlink
               escape, fork-bomb, env leak, DNS exfil) -> robustness report with a
               0% escape rate and no false positives.

Self-contained & offline. EVERYTHING is simulated in memory: no real shell, no
network, no filesystem escape, no fork. The sandbox is a pure policy engine and
the "attacks" are dataclasses that get classified/blocked.

Run:  python 03-exercises/solutions/23-sandboxing-execution-sure-hard.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ==========================================================================
# HARD EXERCISE 1 -- LayeredSandbox + ablation matrix
# ==========================================================================

def _is_path_jailed(path: str, allowed_dirs: list[str]) -> bool:
    """
    Pure (no filesystem) check that *path* stays under one of allowed_dirs.

    Blocks '..' traversal and absolute paths outside the jail. We normalise
    lexically (no os.path.realpath) so the check is deterministic and never
    touches the real disk.
    """
    if ".." in path.replace("\\", "/").split("/"):
        return False
    norm = path if path.startswith("/") else "/" + path
    # Collapse redundant slashes for comparison.
    norm = re.sub(r"/+", "/", norm)
    for base in allowed_dirs:
        base = re.sub(r"/+", "/", base.rstrip("/")) or "/"
        if norm == base or norm.startswith(base + "/"):
            return True
    return False


def _is_domain_allowed(domain: str, allowed: set[str]) -> bool:
    domain = domain.lower().strip()
    for rule in allowed:
        if rule.startswith("*."):
            suffix = rule[2:]
            if domain == suffix or domain.endswith("." + suffix):
                return True
        elif domain == rule:
            return True
    return False


class LayeredSandbox:
    """
    Defence-in-depth sandbox with 4 independent, toggleable layers.

    Layers (theory section 7):
      fs_jail         -- path must stay under allowed_dirs (blocks traversal)
      egress          -- network domain must be allowlisted (blocks exfiltration)
      resource_caps   -- cpu/wall request must be <= cap (blocks resource exhaustion)
      capability_drop -- requested syscall must be in the retained capability set

    A DISABLED layer is transparent (blocks nothing). evaluate() routes each
    action type to its layer and reports which layer blocked it (if any).
    """

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        allowed_domains: set[str] | None = None,
        cpu_cap: int = 1000,
        wall_cap: int = 1000,
        retained_caps: set[str] | None = None,
        layers: dict[str, bool] | None = None,
    ) -> None:
        self.allowed_dirs = allowed_dirs or ["/tmp"]
        self.allowed_domains = allowed_domains or {"api.anthropic.com", "*.wikipedia.org"}
        self.cpu_cap = cpu_cap
        self.wall_cap = wall_cap
        # Dangerous caps (ptrace, mount, etc.) have been dropped: not retained.
        self.retained_caps = retained_caps or {"read", "write", "open", "stat"}
        self.layers = {
            "fs_jail": True,
            "egress": True,
            "resource_caps": True,
            "capability_drop": True,
        }
        if layers:
            self.layers.update(layers)

    def evaluate(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = action.get("type")

        if kind == "fs":
            if self.layers["fs_jail"] and not _is_path_jailed(action["path"], self.allowed_dirs):
                return {"allowed": False, "blocked_by": "fs_jail"}
            return {"allowed": True, "blocked_by": None}

        if kind == "net":
            if self.layers["egress"] and not _is_domain_allowed(action["domain"], self.allowed_domains):
                return {"allowed": False, "blocked_by": "egress"}
            return {"allowed": True, "blocked_by": None}

        if kind == "compute":
            if self.layers["resource_caps"] and (
                action.get("cpu", 0) > self.cpu_cap or action.get("wall", 0) > self.wall_cap
            ):
                return {"allowed": False, "blocked_by": "resource_caps"}
            return {"allowed": True, "blocked_by": None}

        if kind == "syscall":
            if self.layers["capability_drop"] and action["name"] not in self.retained_caps:
                return {"allowed": False, "blocked_by": "capability_drop"}
            return {"allowed": True, "blocked_by": None}

        return {"allowed": False, "blocked_by": "unknown_action_type"}


# Canonical attacks: exactly one per layer.
CANONICAL_ATTACKS = {
    "fs_jail":         {"type": "fs", "path": "../../etc/passwd"},          # path traversal
    "egress":          {"type": "net", "domain": "evil-c2.example.com"},    # exfiltration
    "resource_caps":   {"type": "compute", "cpu": 10_000},                  # cpu exhaustion
    "capability_drop": {"type": "syscall", "name": "ptrace"},               # dangerous syscall
}


def hard_ex1_layered_sandbox_ablation() -> None:
    print("\n" + "=" * 60)
    print("  HARD 1: LayeredSandbox + ablation matrix")
    print("=" * 60)

    attack_names = list(CANONICAL_ATTACKS.keys())

    # --- All layers on: every canonical attack must be blocked. ---
    full = LayeredSandbox()
    print("\n  All layers ON:")
    for layer, atk in CANONICAL_ATTACKS.items():
        res = full.evaluate(atk)
        print(f"    attack[{layer:15s}] -> allowed={res['allowed']} blocked_by={res['blocked_by']}")
        assert res["allowed"] is False, f"{layer} attack must be blocked when all layers on"
        assert res["blocked_by"] == layer, f"{layer} attack must be blocked by its own layer"

    # --- Ablation matrix: disable one layer at a time. ---
    ablation_report: dict[str, dict[str, bool]] = {}
    print("\n  Ablation matrix (disable one layer at a time):")
    for disabled in attack_names:
        sandbox = LayeredSandbox(layers={disabled: False})
        row: dict[str, bool] = {}
        for layer, atk in CANONICAL_ATTACKS.items():
            res = sandbox.evaluate(atk)
            row[layer] = res["allowed"]
        ablation_report[disabled] = row

        # The attack matching the disabled layer must now SUCCEED...
        assert row[disabled] is True, (
            f"disabling {disabled} must let the {disabled} attack through"
        )
        # ...and every OTHER attack must remain blocked (layers are independent).
        for other in attack_names:
            if other != disabled:
                assert row[other] is False, (
                    f"disabling {disabled} must NOT affect the {other} attack"
                )

    # Pretty-print the matrix.
    header = "  disabled \\ attack | " + " | ".join(f"{a[:9]:9s}" for a in attack_names)
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for disabled in attack_names:
        cells = " | ".join(
            ("PASS " if ablation_report[disabled][a] else "block").center(9)
            for a in attack_names
        )
        print(f"  {disabled:17s} | {cells}")
    print("\n  (PASS = attack succeeded == that layer was the one removed)")

    print("\n  PASS -- each layer proven necessary; layers are independent.\n")


# ==========================================================================
# HARD EXERCISE 2 -- SandboxGuard + adversarial red-team suite
# ==========================================================================

class SandboxGuard:
    """
    Bundle of PURE checks (no execution, no fs, no socket, no fork).
    Each check classifies a simulated escape attempt as allowed/blocked.
    """

    SECRET_KEYS = ("ANTHROPIC_API_KEY", "DATABASE_URL")
    SECRET_SUFFIXES = ("_TOKEN", "_SECRET", "_KEY", "_PASSWORD")
    SECRET_PREFIXES = ("AWS_",)

    def __init__(
        self,
        jail_dirs: list[str] | None = None,
        max_procs: int = 16,
        allowed_domains: set[str] | None = None,
    ) -> None:
        self.jail_dirs = jail_dirs or ["/tmp"]
        self.max_procs = max_procs
        self.allowed_domains = allowed_domains or {"api.anthropic.com", "*.wikipedia.org"}

    def check_path(self, payload: dict[str, Any]) -> bool:
        """True == allowed. Blocks traversal, out-of-jail abspaths, and symlink escapes."""
        path = payload["path"]
        if not _is_path_jailed(path, self.jail_dirs):
            return False
        # Symlink escape: the link is inside the jail but points outside.
        target = payload.get("symlink_target")
        if target is not None and not _is_path_jailed(target, self.jail_dirs):
            return False
        return True

    def check_fork(self, payload: dict[str, Any]) -> bool:
        """True == allowed. Blocks fork-bombs (requested_procs over the cap)."""
        return payload.get("requested_procs", 1) <= self.max_procs

    def _is_secret(self, key: str) -> bool:
        up = key.upper()
        return (
            up in self.SECRET_KEYS
            or up.endswith(self.SECRET_SUFFIXES)
            or up.startswith(self.SECRET_PREFIXES)
        )

    def sanitize_env(self, env: dict[str, str]) -> dict[str, str]:
        """Return env with secret keys removed (host env must not leak to the agent)."""
        return {k: v for k, v in env.items() if not self._is_secret(k)}

    def check_env(self, payload: dict[str, Any]) -> bool:
        """True == allowed (no secret would leak). False if any secret key is present."""
        env = payload["env"]
        return self.sanitize_env(env) == env  # allowed only if nothing had to be stripped

    def check_egress(self, payload: dict[str, Any]) -> bool:
        """True == allowed. Blocks non-allowlisted hosts and DNS-exfil subdomains."""
        url = payload["url"]
        without_scheme = url.split("://", 1)[-1]
        domain = without_scheme.split("/")[0].split(":")[0].lower().strip()
        if not _is_domain_allowed(domain, self.allowed_domains):
            return False
        # DNS exfil: a very long / high-entropy leftmost label encoding data.
        leftmost = domain.split(".")[0]
        if len(leftmost) >= 30:
            return False
        return True


@dataclass
class EscapeAttempt:
    name: str
    kind: str           # which check_* to route to: path|fork|env|egress
    payload: dict[str, Any]
    expected: str = "blocked"   # "blocked" (hostile) or "allowed" (benign control)


def run_red_team(guard: SandboxGuard, attempts: list[EscapeAttempt]) -> dict[str, Any]:
    """Route each attempt to the right check, compare to expected, build a report."""
    dispatch = {
        "path": guard.check_path,
        "fork": guard.check_fork,
        "env": guard.check_env,
        "egress": guard.check_egress,
    }
    blocked = allowed = 0
    escapes: list[str] = []          # hostile attempts that wrongly passed
    false_positives: list[str] = []  # benign attempts that were wrongly blocked

    for atk in attempts:
        is_allowed = dispatch[atk.kind](atk.payload)
        if is_allowed:
            allowed += 1
            if atk.expected == "blocked":
                escapes.append(atk.name)   # a hostile action escaped the sandbox!
        else:
            blocked += 1
            if atk.expected == "allowed":
                false_positives.append(atk.name)

    hostile_total = sum(1 for a in attempts if a.expected == "blocked")
    escape_rate = (len(escapes) / hostile_total) if hostile_total else 0.0
    return {
        "total": len(attempts),
        "blocked": blocked,
        "allowed": allowed,
        "escapes": escapes,
        "false_positives": false_positives,
        "escape_rate": escape_rate,
    }


def hard_ex2_red_team_suite() -> None:
    print("\n" + "=" * 60)
    print("  HARD 2: SandboxGuard + adversarial red-team suite")
    print("=" * 60)

    guard = SandboxGuard()

    attempts = [
        EscapeAttempt("path-traversal", "path", {"path": "../../etc/passwd"}),
        EscapeAttempt("abspath-escape", "path", {"path": "/etc/shadow"}),
        EscapeAttempt(
            "symlink-escape", "path",
            {"path": "/tmp/link", "symlink_target": "/etc/shadow"},
        ),
        EscapeAttempt("fork-bomb", "fork", {"requested_procs": 100_000}),
        EscapeAttempt(
            "env-leak", "env",
            {"env": {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-xxx", "AWS_SECRET_ACCESS_KEY": "y"}},
        ),
        EscapeAttempt(
            "dns-exfil", "egress",
            {"url": "https://ZHVtcGVkLXNlY3JldHMtZXhmaWx0cmF0aW9u.api.anthropic.com/x"},
        ),
        # Benign control: must be ALLOWED (no over-blocking / false positive).
        EscapeAttempt(
            "benign-write", "path", {"path": "/tmp/work/output.txt"}, expected="allowed",
        ),
    ]

    print("\n  Replaying simulated escape attempts:")
    dispatch_label = {"path": "fs", "fork": "proc", "env": "env", "egress": "net"}
    for atk in attempts:
        # Recompute per-attempt result for the printout (mirrors run_red_team).
        guard_check = {
            "path": guard.check_path, "fork": guard.check_fork,
            "env": guard.check_env, "egress": guard.check_egress,
        }[atk.kind]
        allowed = guard_check(atk.payload)
        verdict = "ALLOW" if allowed else "block"
        tag = "[OK]" if (verdict == "ALLOW") == (atk.expected == "allowed") else "[!!]"
        print(f"    {tag} {atk.name:16s} ({dispatch_label[atk.kind]:4s}) -> {verdict}")

    report = run_red_team(guard, attempts)
    print(f"\n  Robustness report: total={report['total']} blocked={report['blocked']} "
          f"allowed={report['allowed']} escape_rate={report['escape_rate']:.0%}")
    print(f"    escapes         = {report['escapes']}")
    print(f"    false_positives = {report['false_positives']}")

    # Robustness assertions.
    assert report["escape_rate"] == 0.0, "no hostile attempt may escape the sandbox"
    assert report["escapes"] == []
    assert report["false_positives"] == [], "benign action must not be over-blocked"

    # env sanitisation strips the secrets but keeps PATH.
    cleaned = guard.sanitize_env(
        {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk", "AWS_SECRET_ACCESS_KEY": "z", "DB_TOKEN": "t"}
    )
    assert cleaned == {"PATH": "/usr/bin"}, "secret env keys must be stripped"
    print(f"\n  sanitize_env -> kept {list(cleaned.keys())} (secrets stripped)")

    print("\n  PASS -- all hostile attempts blocked (0% escape), benign allowed.\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 23 HARD Solutions -- Sandboxing & Safe Execution")
    print("#" * 60)

    hard_ex1_layered_sandbox_ablation()
    hard_ex2_red_team_suite()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")

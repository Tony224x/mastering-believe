"""
Day 8 — Solutions: agent identity & IAM (easy / medium / hard).

One file, three levels, separated by banner comments. A smoke test in
`if __name__ == "__main__":` runs all three and asserts the key outcomes.

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ===========================================================================
# === EASY ===
# Identity + least-privilege scope check.
# ===========================================================================
@dataclass
class EasyIdentity:
    agent_id: str          # unique, stable
    owner: str             # the named responsible human
    active: bool = True


def can(token_scopes: set[str], requested_scope: str) -> bool:
    """True iff the requested scope is in the granted set (least privilege)."""
    return requested_scope in token_scopes


def easy_demo() -> dict[str, bool]:
    recon = EasyIdentity("agent-invoice-recon-007", owner="alice@finance")
    # Strictly what the mission needs: read invoices, write the ledger. No payments.
    granted = {"invoices:read", "ledger:write"}

    results: dict[str, bool] = {}
    for scope in ("invoices:read", "ledger:write", "payments:execute"):
        ok = can(granted, scope)
        results[scope] = ok
        print(f"  {'ALLOW' if ok else 'DENY '}  {recon.agent_id} -> {scope}")
    # WHY the DENY on payments:execute is GOOD news: if a prompt injection ever
    # hijacks this agent into attempting a wire transfer, least privilege turns
    # the attack into a harmless authorization refusal.
    return results


# ===========================================================================
# === MEDIUM ===
# A per-request PDP: identity-active + expiry + scope, fail-closed.
# ===========================================================================
@dataclass
class Token:
    agent_id: str
    scopes: set[str]
    expires_at: float                                  # epoch seconds (ephemerality)
    delegated_by: tuple["Link", ...] = field(default_factory=tuple)  # used in HARD


def authorize_medium(
    registry: dict[str, EasyIdentity],
    token: Token,
    requested_scope: str,
    now: float,
) -> tuple[bool, str]:
    """Decide ONE request. Stops at the first failing check (fail-closed)."""
    # 1. identity known and active
    ident = registry.get(token.agent_id)
    if ident is None:
        return False, f"unknown identity: {token.agent_id}"
    if not ident.active:
        return False, f"identity deprovisioned: {token.agent_id}"
    # 2. token not expired (ephemerality bounds the blast radius of any leak)
    if now >= token.expires_at:
        return False, f"token expired for {token.agent_id}"
    # 3. requested scope granted (least privilege)
    if requested_scope not in token.scopes:
        return False, (
            f"scope '{requested_scope}' not granted "
            f"(has {sorted(token.scopes)})"
        )
    return True, f"{token.agent_id} may '{requested_scope}'"


def medium_demo() -> None:
    registry = {
        "agent-invoice-recon-007": EasyIdentity(
            "agent-invoice-recon-007", owner="alice@finance"
        )
    }
    now = 1_000_000.0
    token = Token(
        agent_id="agent-invoice-recon-007",
        scopes={"invoices:read", "ledger:write"},
        expires_at=now + 900,                          # 15-minute TTL
    )

    # legitimate -> allow
    ok, reason = authorize_medium(registry, token, "invoices:read", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert ok

    # same request, clock past TTL -> deny (expired)
    ok, reason = authorize_medium(registry, token, "invoices:read", now + 1000)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "expired" in reason

    # deprovision -> even a non-expired token is rejected
    registry["agent-invoice-recon-007"].active = False
    ok, reason = authorize_medium(registry, token, "invoices:read", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "deprovisioned" in reason

    # adversarial probe: unknown agent id -> deny, no exception
    ghost = Token("agent-ghost", {"invoices:read"}, expires_at=now + 900)
    ok, reason = authorize_medium(registry, ghost, "invoices:read", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "unknown" in reason


# ===========================================================================
# === HARD ===
# Add a delegation chain: human-rooted + privilege attenuation.
# ===========================================================================
@dataclass(frozen=True)
class Link:
    principal: str
    is_human: bool
    scopes: frozenset[str]


def authorize_hard(
    registry: dict[str, EasyIdentity],
    token: Token,
    requested_scope: str,
    now: float,
) -> tuple[bool, str]:
    """Medium checks + a 4th check: delegation chain validity."""
    # Reuse the first three checks (fail-closed ordering matters: an expired
    # token must be rejected for expiry even if its delegation is otherwise fine).
    ok, reason = authorize_medium(registry, token, requested_scope, now)
    if not ok:
        return False, reason

    # 4. delegation chain
    chain = token.delegated_by
    if not chain:
        # No delegation: allowed, but flag the missing human root for visibility.
        return True, f"{token.agent_id} may '{requested_scope}' (no delegation chain)"

    # 4a. must root at a human principal
    if not chain[0].is_human:
        return False, "delegation chain does not root at a human principal"

    # 4b. privilege attenuation: every link must already hold the scope.
    # Privileges can only shrink down the chain, never appear by magic.
    for link in chain:
        if requested_scope not in link.scopes:
            return False, (
                f"privilege attenuation violated: '{link.principal}' "
                f"did not hold '{requested_scope}'"
            )
    return True, (
        f"{token.agent_id} may '{requested_scope}' "
        f"(root={root_principal(token)})"
    )


def root_principal(token: Token) -> str:
    """Return the human root principal of the delegation chain."""
    if token.delegated_by:
        return token.delegated_by[0].principal
    return f"<self:{token.agent_id}>"


def hard_demo() -> None:
    registry = {
        "agent-orchestrator": EasyIdentity("agent-orchestrator", owner="bob@ops"),
        "agent-data-fetch": EasyIdentity("agent-data-fetch", owner="bob@ops"),
    }
    now = 1_000_000.0

    human = Link("alice@finance", True, frozenset({"report:build", "data:fetch"}))
    orchestrator = Link(
        "agent-orchestrator", False, frozenset({"report:build", "data:fetch"})
    )

    # Case 1 — valid chain human -> orchestrator -> data agent, scope held throughout
    valid = Token(
        "agent-data-fetch", {"data:fetch"}, expires_at=now + 600,
        delegated_by=(human, orchestrator),
    )
    ok, reason = authorize_hard(registry, valid, "data:fetch", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert ok and root_principal(valid) == "alice@finance"

    # Case 2 — chain whose first link is NOT human -> deny
    non_human_root = Token(
        "agent-data-fetch", {"data:fetch"}, expires_at=now + 600,
        delegated_by=(orchestrator,),  # first link is an agent, not a human
    )
    ok, reason = authorize_hard(registry, non_human_root, "data:fetch", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "human" in reason

    # Case 3 — 'magic' escalation: token grants payments:execute but no link held it
    escalating = Token(
        "agent-data-fetch", {"data:fetch", "payments:execute"}, expires_at=now + 600,
        delegated_by=(human, orchestrator),
    )
    ok, reason = authorize_hard(registry, escalating, "payments:execute", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "attenuation" in reason

    # Case 4 — regression: expired token with otherwise-valid delegation -> deny (expired)
    expired = Token(
        "agent-data-fetch", {"data:fetch"}, expires_at=now - 1,
        delegated_by=(human, orchestrator),
    )
    ok, reason = authorize_hard(registry, expired, "data:fetch", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "expired" in reason

    # Adversarial probe — intermediate agent holds the scope, but the HUMAN root
    # does not -> deny (an agent cannot exceed the human who mandated it).
    human_without_scope = Link("alice@finance", True, frozenset({"report:build"}))
    orchestrator_with_scope = Link(
        "agent-orchestrator", False, frozenset({"report:build", "data:fetch"})
    )
    probe = Token(
        "agent-data-fetch", {"data:fetch"}, expires_at=now + 600,
        delegated_by=(human_without_scope, orchestrator_with_scope),
    )
    ok, reason = authorize_hard(registry, probe, "data:fetch", now)
    print(f"  {'ALLOW' if ok else 'DENY '}  {reason}")
    assert not ok and "attenuation" in reason


if __name__ == "__main__":
    print("=== EASY ===")
    easy_results = easy_demo()
    assert easy_results == {
        "invoices:read": True,
        "ledger:write": True,
        "payments:execute": False,
    }

    print("\n=== MEDIUM ===")
    medium_demo()

    print("\n=== HARD ===")
    hard_demo()

    print("\nAll smoke-test assertions passed.")

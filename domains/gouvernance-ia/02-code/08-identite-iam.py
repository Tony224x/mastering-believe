"""
Day 8 — Agent identity & IAM: a scope-based access control engine (a mini PDP).

This script demonstrates the runtime mechanics of governing *what an agent is
allowed to do*, request by request — the engineer-side counterpart of the
"4 pillars" board view from Day 2.

It re-implements, in pure stdlib, the core of a Zero Trust Policy Decision Point
(PDP). For each agent action it answers allow/deny by checking, in order:

    1. the agent identity is known and active        (non-human identity)
    2. the token has not expired                      (ephemerality)
    3. the requested scope is granted                 (least privilege)
    4. the delegation chain is valid                  (on-behalf-of a human,
       with privilege attenuation: scopes only shrink down the chain)

Real-world anchors (re-implemented here in miniature, cited, NOT imported):
    - CSA "Agentic AI Identity and Access Management" (2025): NHI, ephemerality,
      delegation as first-class IAM concerns for agents.
    - NIST SP 800-207 "Zero Trust Architecture" (2020): per-request decision at a
      Policy Decision Point ("never trust, always verify").
    - OAuth 2.0 scopes + RFC 8693 token-exchange (on-behalf-of / delegation grant):
      the scope strings and the `delegated_by` field mimic these.
    - OWASP Top 10 for Agentic Applications 2026: "Identity & Privilege Abuse" is
      exactly the failure this engine is built to prevent.

# requires: stdlib only
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
# WHY a frozen-ish principal record: an agent must be a first-class identity,
# never a borrowed human/team token. `agent_id` is unique and stable; `owner`
# ties it to a responsible human; `active` lets us deprovision instantly.
@dataclass
class AgentIdentity:
    agent_id: str           # unique, stable, never reused for another agent
    owner: str              # the named human ultimately responsible
    active: bool = True     # deprovisioning flips this -> all access stops


# WHY a token separate from the identity: identity is "who", the token carries
# the *granted scopes* (least privilege), an *expiry* (ephemerality), and a
# *delegation pointer* (on-behalf-of). Tokens are short-lived; identities persist.
@dataclass
class AccessToken:
    agent_id: str
    scopes: frozenset[str]          # least-privilege grant, e.g. {"invoices:read"}
    expires_at: float               # epoch seconds; short TTL = ephemerality
    # delegation chain: who handed work to this agent. The root must be a human.
    # Each link records the principal id and the scopes it actually held, so we
    # can enforce privilege attenuation (a delegate can never gain new scopes).
    delegated_by: tuple["DelegationLink", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DelegationLink:
    principal: str                  # id of the delegating principal (human or agent)
    is_human: bool                  # True only for the human root
    scopes: frozenset[str]          # scopes that principal held when delegating


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str                     # human-readable WHY — this feeds the audit log

    def __str__(self) -> str:
        tag = "ALLOW" if self.allowed else "DENY "
        return f"[{tag}] {self.reason}"


# ---------------------------------------------------------------------------
# The Policy Decision Point (PDP)
# ---------------------------------------------------------------------------
class AccessEngine:
    """A per-request scope-based access control engine (a tiny Zero Trust PDP).

    WHY a registry of identities lives here: the engine must verify the identity
    is *known and active* on every call — a Zero Trust PDP trusts nothing it
    cannot currently resolve. In production this would be Entra Agent ID / an
    agent registry; here it's an in-memory dict.
    """

    def __init__(self) -> None:
        self._identities: dict[str, AgentIdentity] = {}

    def register(self, identity: AgentIdentity) -> None:
        # WHY refuse duplicate ids: one identity == one agent. Sharing breaks
        # both audit attribution and deprovisioning.
        if identity.agent_id in self._identities:
            raise ValueError(f"agent id already registered: {identity.agent_id}")
        self._identities[identity.agent_id] = identity

    def deprovision(self, agent_id: str) -> None:
        # WHY flip `active` instead of deleting: deprovisioning must invalidate
        # *all* of an agent's tokens at once, not wait for them to expire.
        ident = self._identities.get(agent_id)
        if ident is not None:
            ident.active = False

    def authorize(
        self,
        token: AccessToken,
        requested_scope: str,
        now: float | None = None,
    ) -> Decision:
        """Evaluate ONE request. Any single failing check -> deny (fail-closed)."""
        now = time.time() if now is None else now

        # Check 1 — identity known and active (non-human identity, deprovisioning).
        ident = self._identities.get(token.agent_id)
        if ident is None:
            return Decision(False, f"unknown identity: {token.agent_id}")
        if not ident.active:
            return Decision(False, f"identity deprovisioned: {token.agent_id}")

        # Check 2 — token not expired (ephemerality). Short TTL bounds any leak.
        if now >= token.expires_at:
            return Decision(False, f"token expired for {token.agent_id}")

        # Check 3 — requested scope granted (least privilege). The whole point:
        # a hijacked agent that lacks the scope simply gets denied here.
        if requested_scope not in token.scopes:
            return Decision(
                False,
                f"scope '{requested_scope}' not granted to {token.agent_id} "
                f"(has {sorted(token.scopes)})",
            )

        # Check 4 — delegation chain valid: roots at a human, privileges only
        # attenuate. Models "humans remain ultimately accountable" [IMDA, 2026].
        chain_ok, chain_reason = self._verify_delegation(token, requested_scope)
        if not chain_ok:
            return Decision(False, chain_reason)

        return Decision(
            True,
            f"{token.agent_id} may '{requested_scope}' "
            f"(owner={ident.owner}, principal={self._root_principal(token)})",
        )

    # -- delegation helpers --------------------------------------------------
    @staticmethod
    def _verify_delegation(token: AccessToken, requested_scope: str) -> tuple[bool, str]:
        """Privilege attenuation + human-rooted chain.

        WHY: a delegate must never hold a scope its delegant did not have, and
        the chain must trace back to a human. This stops "magic" privilege
        escalation through an agent-to-agent hop.
        """
        if not token.delegated_by:
            # No delegation declared: the agent acts on its own grant. Fine,
            # but flag the absence of a human root for governance visibility.
            return True, "no delegation chain (agent acts on its own grant)"

        root = token.delegated_by[0]
        if not root.is_human:
            return False, "delegation chain does not root at a human principal"

        # Every link must already hold the requested scope: privilege can only
        # shrink as it flows down the chain, never appear.
        for link in token.delegated_by:
            if requested_scope not in link.scopes:
                return False, (
                    f"privilege attenuation violated: principal "
                    f"'{link.principal}' did not hold '{requested_scope}'"
                )
        return True, "delegation chain valid"

    def _root_principal(self, token: AccessToken) -> str:
        if token.delegated_by:
            return token.delegated_by[0].principal
        return f"<self:{token.agent_id}>"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def _demo() -> None:
    engine = AccessEngine()

    # One identity per agent, each tied to a named human owner.
    engine.register(AgentIdentity("agent-invoice-recon-007", owner="alice@finance"))
    engine.register(AgentIdentity("agent-orchestrator", owner="bob@ops"))
    engine.register(AgentIdentity("agent-data-fetch", owner="bob@ops"))

    now = 1_000_000.0  # fixed clock so the demo is deterministic

    # Least-privilege token: recon agent can read invoices and write the ledger,
    # but CANNOT execute payments. This is the guardrail that defeats hijacking.
    recon_token = AccessToken(
        agent_id="agent-invoice-recon-007",
        scopes=frozenset({"invoices:read", "ledger:write"}),
        expires_at=now + 900,  # 15-minute TTL (ephemerality)
    )

    print("=" * 70)
    print("Scenario A — least privilege defeats a hijacked agent")
    print("=" * 70)
    # Legitimate action: granted.
    print(engine.authorize(recon_token, "invoices:read", now=now))
    # Prompt-injection makes it attempt a wire transfer: DENIED at the PDP,
    # because the scope was never granted (OWASP: Identity & Privilege Abuse).
    print(engine.authorize(recon_token, "payments:execute", now=now))

    print("\n" + "=" * 70)
    print("Scenario B — ephemerality: an expired token is worthless")
    print("=" * 70)
    # Same legitimate scope, but the clock has moved past the TTL.
    print(engine.authorize(recon_token, "invoices:read", now=now + 1000))

    print("\n" + "=" * 70)
    print("Scenario C — deprovisioning kills all access instantly")
    print("=" * 70)
    engine.deprovision("agent-invoice-recon-007")
    print(engine.authorize(recon_token, "invoices:read", now=now))  # token still 'valid'

    print("\n" + "=" * 70)
    print("Scenario D — delegation chain rooted at a human, with attenuation")
    print("=" * 70)
    # Alice (human) holds report:build & data:fetch. She delegates to the
    # orchestrator, which sub-delegates the fetch to a data agent.
    human = DelegationLink(
        "alice@finance", is_human=True,
        scopes=frozenset({"report:build", "data:fetch"}),
    )
    orchestrator = DelegationLink(
        "agent-orchestrator", is_human=False,
        scopes=frozenset({"report:build", "data:fetch"}),
    )
    data_token = AccessToken(
        agent_id="agent-data-fetch",
        scopes=frozenset({"data:fetch"}),       # attenuated to just what it needs
        expires_at=now + 600,
        delegated_by=(human, orchestrator),      # chain: human -> orchestrator
    )
    # Valid: every link held data:fetch, chain roots at a human.
    print(engine.authorize(data_token, "data:fetch", now=now))

    print("\n" + "=" * 70)
    print("Scenario E — privilege attenuation blocks 'magic' escalation")
    print("=" * 70)
    # The data agent's own token grants payments:execute, but NO link in its
    # delegation chain ever held that scope -> attenuation violation -> deny.
    escalating_token = AccessToken(
        agent_id="agent-data-fetch",
        scopes=frozenset({"data:fetch", "payments:execute"}),
        expires_at=now + 600,
        delegated_by=(human, orchestrator),
    )
    print(engine.authorize(escalating_token, "payments:execute", now=now))


if __name__ == "__main__":
    _demo()

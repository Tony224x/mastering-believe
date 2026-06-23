"""
Day 2 -- The 4 pillars of a governable agent.

What this script demonstrates
-----------------------------
The 4 pillars from the theory module, made executable:
  1. Identity     -- a unique, stable, verifiable ID (NOT a human's credential).
  2. Owner        -- a single, named, accountable human (a person, not a team).
  3. Permissions  -- explicit, bounded scopes (least privilege).
  4. Audit trail  -- a verifiable record linking identity -> action -> authorization.

We build a minimal `GovernedAgent` dataclass (a tiny "Agent Card") and a
governance-completeness validator that runs the "smell test": for any agent,
can we answer the 4 questions -- WHICH agent, WHO owns it, WHAT may it do,
and is WHAT-it-did traceable? An agent failing any check is *ungoverned*.

We then replay the opening incident (a refund bot that wired 4 200 EUR using a
human's API key) to show how the validator + a least-privilege check would have
caught the over-privileged action BEFORE it executed.

Real-world anchors (re-implemented in miniature here, stdlib only):
  - "Non-Human Identity" / agent IAM ............ CSA, Agentic AI IAM (2025)
  - ASI03 Identity & Privilege Abuse, least priv. OWASP Top 10 Agentic (2026)
  - "humans remain ultimately accountable" ...... IMDA Agentic framework (2026)
The "Agent Card" idea (Google A2A / Microsoft Entra Agent ID) is an *emerging*
practice, not a frozen standard -- we model only its durable core.

# requires: stdlib only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Pillar 1+2+3 -- the Agent Card: identity, owner, permissions in one record.
# A @dataclass keeps it a plain, serializable "card" (lisible par machine),
# which is exactly what real Agent Cards aim to be (A2A / Entra Agent ID).
# ---------------------------------------------------------------------------
@dataclass
class GovernedAgent:
    """Minimal governance record for ONE agent -- a tiny Agent Card.

    The four governance attributes map 1:1 to the four pillars. `audit_log`
    holds the 4th pillar's trace; the first three are static metadata.
    """

    agent_id: str  # Pillar 1: unique, stable URN-like id, NOT a human's id.
    owner: str  # Pillar 2: a single NAMED human (person), the accountable one.
    permissions: list[str] = field(
        default_factory=list
    )  # Pillar 3: explicit scopes (least privilege). Empty == ungoverned.
    # Pillar 4 lives here: an append-only-by-convention list of action records.
    audit_log: list[dict] = field(default_factory=list)

    def act(self, action: str, scope_required: str, **context) -> dict:
        """Perform an action ONLY if a matching scope is held, and trace it.

        WHY enforce scope here: this is where least privilege (Pillar 3) and
        the audit trail (Pillar 4) meet -- we both *deny* over-reach and
        *record* the decision linking identity -> action -> authorization.
        """
        granted = scope_required in self.permissions
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": self.agent_id,  # WHO (identity) -- never a human's id
            "action": action,  # WHAT
            "scope_required": scope_required,  # under WHICH authorization
            "result": "executed" if granted else "denied",
            "context": context,
        }
        # Append-only by convention (J9 adds hash-chaining to make it
        # tamper-EVIDENT; here the point is just: every action leaves a trace).
        self.audit_log.append(record)
        return record


# ---------------------------------------------------------------------------
# The governance-completeness validator == the "smell test" as code.
# Returns the list of FAILED pillars. Empty list == governable.
# WHY return reasons, not just a bool: governance is about *why* it fails,
# so an owner/auditor can fix the specific missing pillar.
# ---------------------------------------------------------------------------
# A human's id should never BE an agent's identity (the incident's root cause).
_HUMAN_ID_PREFIXES = ("user:", "human:", "person:", "employee:")


def check_governance(agent: GovernedAgent) -> list[str]:
    """Run the 4-question smell test. Empty result => agent is governable."""
    failures: list[str] = []

    # Pillar 1 -- Identity: present, non-empty, and NOT a human's identity.
    if not agent.agent_id or not agent.agent_id.strip():
        failures.append("IDENTITY: missing unique agent id")
    elif agent.agent_id.lower().startswith(_HUMAN_ID_PREFIXES):
        # Borrowing a human's identity is the bank-incident anti-pattern.
        failures.append("IDENTITY: agent uses a human's identity (impersonation)")

    # Pillar 2 -- Owner: a single NAMED person. Teams/queues are not owners.
    owner = (agent.owner or "").strip()
    if not owner:
        failures.append("OWNER: no named human owner")
    elif owner.lower() in {"team", "it", "data team", "n/a", "tbd", "unknown"}:
        # Diffuse ownership == no ownership (IMDA 2026: a human is accountable).
        failures.append(f"OWNER: owner '{owner}' is not a named person")

    # Pillar 3 -- Permissions: explicit and bounded. Empty == ungoverned;
    # a wildcard == privilege abuse (the opposite of least privilege).
    if not agent.permissions:
        failures.append("PERMISSIONS: no explicit scopes (least privilege absent)")
    elif "*" in agent.permissions or "all" in agent.permissions:
        failures.append("PERMISSIONS: wildcard scope violates least privilege")

    # Pillar 4 -- Audit trail: the *capability* exists (a list, not None).
    # (Content/integrity is J9; here we assert the trace mechanism is present.)
    if agent.audit_log is None:
        failures.append("AUDIT: no audit trail capability")

    return failures


def is_governable(agent: GovernedAgent) -> bool:
    """True iff the agent passes all 4 pillars."""
    return not check_governance(agent)


def governance_coverage(agents: list[GovernedAgent]) -> float:
    """Share of a fleet that is fully governable -- a board-ready KPI."""
    if not agents:
        return 0.0
    governed = sum(1 for a in agents if is_governable(a))
    return round(100.0 * governed / len(agents), 1)


# ---------------------------------------------------------------------------
# Demo / runnable proof of the mechanism.
# ---------------------------------------------------------------------------
def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    _banner("1. A WELL-GOVERNED agent passes the 4-question smell test")
    good = GovernedAgent(
        agent_id="agent://refund-bot/7f3a",  # Pillar 1: unique, not a human
        owner="Sofia Marchetti",  # Pillar 2: a named person
        permissions=["read:disputes", "write:dispute_notes"],  # Pillar 3: scoped
    )
    print(f"agent_id   : {good.agent_id}")
    print(f"owner      : {good.owner}")
    print(f"permissions: {good.permissions}")
    failures = check_governance(good)
    print(f"smell test : {'PASS (governable)' if not failures else failures}")

    _banner("2. The bank-incident agent FAILS the smell test")
    # The real root cause: it ran under a human's API key with all his rights.
    ungoverned = GovernedAgent(
        agent_id="user:chef-projet-42",  # Pillar 1 broken: a human's identity
        owner="IT",  # Pillar 2 broken: not a named person
        permissions=["*"],  # Pillar 3 broken: wildcard == abuse
    )
    for reason in check_governance(ungoverned):
        print(f"  FAIL -> {reason}")

    _banner("3. Least privilege STOPS the 4 200 EUR transfer at action time")
    # `good` has NO transfer:funds scope -> the dangerous action is denied
    # and the denial is recorded in the audit trail (Pillar 4).
    allowed = good.act(
        "read_dispute", "read:disputes", dispute="#88421"
    )  # within scope
    blocked = good.act(
        "funds_transfer", "transfer:funds", amount=4200, dispute="#88421"
    )  # NOT in scope -> denied, but TRACED
    print(f"read_dispute   -> {allowed['result']}")
    print(f"funds_transfer -> {blocked['result']}  (no 'transfer:funds' scope)")

    _banner("4. The audit trail makes every action provable (Pillar 4)")
    for entry in good.audit_log:
        print(json.dumps(entry, ensure_ascii=False))

    _banner("5. Fleet-level governance coverage (a board KPI)")
    fleet = [good, ungoverned]
    print(f"agents in fleet     : {len(fleet)}")
    print(f"governable agents   : {sum(1 for a in fleet if is_governable(a))}")
    print(f"governance coverage : {governance_coverage(fleet)} %")


if __name__ == "__main__":
    main()

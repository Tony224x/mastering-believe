"""Day 3 — Agent inventory & registry.

This script demonstrates the core governance mechanism of Day 3: a *live*,
queryable agent registry — NOT a frozen spreadsheet. It shows, end to end:

  1. Enrolling agents from declarative "Agent Cards" (the A2A-style JSON
     descriptor of identity + capabilities), with validation.
  2. Persisting the registry to a durable JSON file (round-trip load/save).
  3. The three founding governance queries: by_owner, orphans, by_risk.
  4. A governance-coverage metric (the % of agents with the 4 governance
     attributes filled), which answers the founding question of the domain:
     "how many agents run, and who owns them?".
  5. Lifecycle transitions: we change status, we do NOT delete the row —
     so traceability survives a decommission.

Why a registry and not a list: a spreadsheet is a *photo*; the fleet is a
*video*. The registry is a control — interrogeable and tied to the lifecycle.

Real-world tools this mini version stands in for (cited, not imported):
  - Microsoft Entra Agent ID (Preview, Dec 2025) — first-class agent identity.
  - Google A2A "Agent Card" (Apr 2025) — JSON capability/auth declaration.
  - CSA "Agentic AI Identity and Access Management" (2025) — lifecycle IAM.
All are EMERGENT / non-frozen: we reproduce the durable *principle* (a queryable
register + a declarative Agent Card), not a specific product.

# requires: stdlib only
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Risk tiers, lowest to highest. Ordering matters for "by_risk" sorting and for
# prioritising governance attention (we do not audit a note-summariser like a
# payment-triggering agent). Mirrors the spirit of the EU AI Act tiering seen J5.
RISK_ORDER = {"minimal": 0, "limited": 1, "high": 2, "unacceptable": 3}

# The four governance attributes (the J2 pillars made concrete) that an agent
# must have filled to count as "governed". Used by the coverage metric.
GOVERNANCE_FIELDS = ("agent_id", "owner", "permissions", "risk_tier")


def _now_iso() -> str:
    """UTC timestamp. WHY: lifecycle mutations must be horodated for the audit
    trail (Day 9). A naive local time would be ambiguous across machines."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Agent:
    """One registry entry. Carries the J2 governance pillars + lifecycle state.

    We keep an explicit `status` and never delete entries: decommissioning is a
    transition, not a deletion, so the trace of "who owned what, when" survives.
    """

    agent_id: str
    owner: str | None            # named human accountable; None => orphan candidate
    permissions: list[str] = field(default_factory=list)
    risk_tier: str = "minimal"
    status: str = "active"       # active | suspended | decommissioned
    enrolled_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def is_governed(self) -> bool:
        """True only if the 4 pillars are present AND non-empty.

        WHY non-empty: an agent with owner="" or permissions=[] is technically
        "filled" but ungoverned in practice. Coverage must reflect reality."""
        return bool(self.agent_id) and bool(self.owner) and bool(self.permissions) and bool(self.risk_tier)


class AgentRegistry:
    """A live, queryable agent registry with JSON persistence.

    This is the Day 3 control. It is deliberately small but exercises the real
    governance primitives: enrol (from an Agent Card), query, transition, persist.
    """

    def __init__(self) -> None:
        # Keyed by agent_id => O(1) lookup and natural uniqueness of identity.
        self._agents: dict[str, Agent] = {}

    # --- Enrolment -------------------------------------------------------

    @staticmethod
    def validate_agent_card(card: dict) -> list[str]:
        """Validate a declarative Agent Card before ingesting it.

        Returns the list of problems (empty => valid). WHY validate first: an
        Agent Card is the machine-readable enrolment form. In production it would
        also be cryptographically *verifiable* (signed / verifiable credentials,
        per CSA 2025); here we at least enforce the presence of critical fields so
        a malformed card cannot silently create a half-governed entry.
        """
        problems: list[str] = []
        agent_id = card.get("agent_id")
        if not agent_id or not isinstance(agent_id, str):
            problems.append("missing or non-string 'agent_id'")
        tier = card.get("risk_tier", "minimal")
        if tier not in RISK_ORDER:
            problems.append(f"unknown risk_tier '{tier}' (expected one of {list(RISK_ORDER)})")
        perms = card.get("permissions", [])
        if not isinstance(perms, list):
            problems.append("'permissions' must be a list")
        return problems

    def enrol_from_card(self, card: dict) -> Agent:
        """Ingest an Agent Card (push enrolment). Rejects invalid cards and
        duplicate identities — identity must stay unique in the registry."""
        problems = self.validate_agent_card(card)
        if problems:
            raise ValueError(f"invalid Agent Card: {'; '.join(problems)}")
        agent_id = card["agent_id"]
        if agent_id in self._agents:
            raise ValueError(f"agent '{agent_id}' already enrolled (identity must be unique)")
        agent = Agent(
            agent_id=agent_id,
            owner=card.get("owner"),
            permissions=list(card.get("permissions", [])),
            risk_tier=card.get("risk_tier", "minimal"),
        )
        self._agents[agent_id] = agent
        return agent

    # --- Lifecycle -------------------------------------------------------

    def transition(self, agent_id: str, *, status: str | None = None, owner: str | None = None) -> Agent:
        """Mutate an entry (status and/or owner) and horodate the change.

        WHY no delete: we transition status (e.g. -> 'decommissioned') so the
        traceability of past ownership/permissions survives for incident review.
        """
        if agent_id not in self._agents:
            raise KeyError(f"unknown agent '{agent_id}'")
        agent = self._agents[agent_id]
        if status is not None:
            if status not in {"active", "suspended", "decommissioned"}:
                raise ValueError(f"invalid status '{status}'")
            agent.status = status
        if owner is not None:
            agent.owner = owner
        agent.updated_at = _now_iso()
        return agent

    # --- Discovery / reconciliation -------------------------------------

    def reconcile(self, observed_ids: list[str]) -> list[str]:
        """Compare what is DECLARED (registry) with what ACTS (telemetry).

        `observed_ids` simulates discovery (telemetry/scan). The returned list is
        the gap: agents that act but were never enrolled — ghost/orphan candidates.
        This is the heart of pull-discovery: never trust the register alone.
        """
        return [oid for oid in observed_ids if oid not in self._agents]

    # --- Governance queries ---------------------------------------------

    def by_owner(self, owner: str) -> list[Agent]:
        """All agents owned by `owner` — accountability & ownership concentration."""
        return [a for a in self._agents.values() if a.owner == owner]

    def orphans(self) -> list[Agent]:
        """The critical governance query: agents with no owner.

        An orphan is unmanageable (nobody to supervise/decommission) and
        non-imputable (nobody accountable on incident)."""
        return [a for a in self._agents.values() if not a.owner]

    def by_risk(self, min_tier: str = "minimal") -> list[Agent]:
        """Agents at or above a risk tier, highest-risk first — for prioritising."""
        floor = RISK_ORDER[min_tier]
        selected = [a for a in self._agents.values() if RISK_ORDER.get(a.risk_tier, 0) >= floor]
        return sorted(selected, key=lambda a: RISK_ORDER.get(a.risk_tier, 0), reverse=True)

    def coverage(self) -> dict:
        """Governance-coverage metric: the board-ready answer to
        'how many agents, and who owns them?'."""
        total = len(self._agents)
        governed = sum(1 for a in self._agents.values() if a.is_governed())
        n_orphans = len(self.orphans())
        rate = round(100 * governed / total, 1) if total else 0.0
        return {
            "total_agents": total,
            "governed_agents": governed,
            "orphans": n_orphans,
            "coverage_pct": rate,
        }

    # --- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist to a durable JSON file (the live source of truth on disk).

        WHY a real file (atomic write): a registry that lives only in memory dies
        with the process. We write to a temp file then replace, so a crash mid-write
        cannot corrupt the existing registry."""
        path = Path(path)
        payload = {"agents": [asdict(a) for a in self._agents.values()]}
        # Atomic-ish write: serialise to a temp file in the same dir, then replace.
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
        try:
            with open(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "AgentRegistry":
        """Reload the registry from its JSON file — proving it is durable."""
        reg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for raw in data.get("agents", []):
            reg._agents[raw["agent_id"]] = Agent(**raw)
        return reg


# --- Sample data: a small fleet, deliberately including 1 orphan ------------

SAMPLE_CARDS = [
    {"agent_id": "agt-invoice-bot", "owner": "alice.finance", "permissions": ["read:invoices", "send:email"], "risk_tier": "high"},
    {"agent_id": "agt-note-summary", "owner": "bob.ops", "permissions": ["read:notes"], "risk_tier": "minimal"},
    {"agent_id": "agt-payment-run", "owner": "alice.finance", "permissions": ["read:invoices", "trigger:payment"], "risk_tier": "high"},
    {"agent_id": "agt-hr-screener", "owner": "carol.hr", "permissions": ["read:cv"], "risk_tier": "limited"},
    # An orphan: deployed by an intern, never assigned an owner. The very thing
    # the governance queries must surface.
    {"agent_id": "agt-shadow-x", "owner": None, "permissions": ["read:files"], "risk_tier": "limited"},
]


def _print_agents(label: str, agents: list[Agent]) -> None:
    print(f"  {label} ({len(agents)}):")
    for a in agents:
        print(f"    - {a.agent_id:<18} owner={str(a.owner):<16} tier={a.risk_tier:<8} status={a.status}")


if __name__ == "__main__":
    print("=== Day 3 - Agent inventory & registry (live, queryable) ===\n")

    registry = AgentRegistry()

    # 1) Enrol the fleet from Agent Cards (push enrolment, with validation).
    print("[1] Enrolling agents from Agent Cards...")
    for card in SAMPLE_CARDS:
        agent = registry.enrol_from_card(card)
        print(f"    enrolled {agent.agent_id} (owner={agent.owner})")

    # Adversarial probe: a malformed card must be rejected, not silently ingested.
    print("\n[2] Rejecting a malformed Agent Card (no agent_id, bad tier)...")
    try:
        registry.enrol_from_card({"owner": "dave", "risk_tier": "ultra"})
    except ValueError as exc:
        print(f"    rejected as expected -> {exc}")

    # 3) Governance queries.
    print("\n[3] Governance queries:")
    _print_agents("by_owner('alice.finance')", registry.by_owner("alice.finance"))
    _print_agents("orphans (the critical query)", registry.orphans())
    _print_agents("by_risk(min_tier='high')", registry.by_risk("high"))

    # 4) Discovery / reconciliation: telemetry sees an agent we never enrolled.
    print("\n[4] Reconciliation (declared vs observed):")
    observed = ["agt-invoice-bot", "agt-note-summary", "agt-ghost-42"]
    ghosts = registry.reconcile(observed)
    print(f"    observed acting: {observed}")
    print(f"    ghosts (act but unregistered) -> {ghosts}")

    # 5) Coverage metric — the board-ready answer.
    print("\n[5] Governance coverage:")
    for k, v in registry.coverage().items():
        print(f"    {k}: {v}")

    # 6) Lifecycle: reassign the orphan, then decommission an agent (no delete).
    print("\n[6] Lifecycle transitions:")
    registry.transition("agt-shadow-x", owner="erin.platform")
    print("    assigned owner to agt-shadow-x -> orphans now:", [a.agent_id for a in registry.orphans()])
    registry.transition("agt-hr-screener", status="decommissioned")
    decommissioned = registry._agents["agt-hr-screener"]
    print(f"    decommissioned agt-hr-screener -> status={decommissioned.status}, row kept (traceability preserved)")

    # 7) Persistence round-trip: prove the registry is durable, not in-memory only.
    print("\n[7] Persistence round-trip (JSON file):")
    out_path = Path(tempfile.gettempdir()) / "agent_registry_day3.json"
    registry.save(out_path)
    reloaded = AgentRegistry.load(out_path)
    print(f"    saved to {out_path}")
    print(f"    reloaded {len(reloaded._agents)} agents; coverage after fixes -> {reloaded.coverage()}")

    print("\nDone. A registry is a live control, not a spreadsheet.")

"""Solutions — Day 3, Inventory & agent registry (easy + medium + hard).

One file, three levels separated by markers. stdlib only. The __main__ block is a
smoke test that exercises every level and asserts the key invariants, so running
`python <file>` both demonstrates the solutions AND self-checks them.

# requires: stdlib only
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Shared across levels: valid risk tiers, lowest -> highest.
RISK_ORDER = {"minimal": 0, "limited": 1, "high": 2, "unacceptable": 3}


def _now_iso() -> str:
    """UTC ISO timestamp — mutations must be horodated for the audit trail (J9)."""
    return datetime.now(timezone.utc).isoformat()


# === EASY ===
# Goal: a minimal in-memory registry answering "how many agents, who owns them?".

@dataclass
class EasyAgent:
    agent_id: str
    owner: str | None
    permissions: list[str] = field(default_factory=list)
    risk_tier: str = "minimal"


class EasyRegistry:
    """In-memory registry keyed by agent_id (identity must be unique)."""

    def __init__(self) -> None:
        self._agents: dict[str, EasyAgent] = {}

    def add(self, agent: EasyAgent) -> None:
        # Reject duplicate identities: the registry is the source of truth for "who".
        if agent.agent_id in self._agents:
            raise ValueError(f"agent '{agent.agent_id}' already exists")
        self._agents[agent.agent_id] = agent

    def count(self) -> int:
        return len(self._agents)

    def by_owner(self, owner: str) -> list[EasyAgent]:
        return [a for a in self._agents.values() if a.owner == owner]

    def orphans(self) -> list[EasyAgent]:
        # Empty owner (None or "") => orphan, the critical governance gap.
        return [a for a in self._agents.values() if not a.owner]


def _demo_easy() -> EasyRegistry:
    reg = EasyRegistry()
    reg.add(EasyAgent("agt-invoice", "alice", ["read:invoices"], "high"))
    reg.add(EasyAgent("agt-notes", "bob", ["read:notes"], "minimal"))
    reg.add(EasyAgent("agt-payroll", "alice", ["trigger:payment"], "high"))
    reg.add(EasyAgent("agt-shadow", None, ["read:files"], "limited"))  # the orphan
    return reg


# === MEDIUM ===
# Goal: durable (JSON persistence), fed by validated Agent Cards, with reconciliation.

GOVERNANCE_FIELDS = ("agent_id", "owner", "permissions", "risk_tier")


@dataclass
class Agent:
    agent_id: str
    owner: str | None
    permissions: list[str] = field(default_factory=list)
    risk_tier: str = "minimal"
    status: str = "active"
    enrolled_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def is_governed(self) -> bool:
        # All 4 pillars present AND non-empty.
        return bool(self.agent_id) and bool(self.owner) and bool(self.permissions) and bool(self.risk_tier)


def validate_agent_card(card: dict) -> list[str]:
    """Return list of problems (empty => valid). Validate before ingesting."""
    problems: list[str] = []
    if not card.get("agent_id") or not isinstance(card.get("agent_id"), str):
        problems.append("missing or non-string 'agent_id'")
    tier = card.get("risk_tier", "minimal")
    if tier not in RISK_ORDER:
        problems.append(f"unknown risk_tier '{tier}'")
    if not isinstance(card.get("permissions", []), list):
        problems.append("'permissions' must be a list")
    return problems


class MediumRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def enrol_from_card(self, card: dict) -> Agent:
        problems = validate_agent_card(card)
        if problems:
            raise ValueError(f"invalid Agent Card: {'; '.join(problems)}")
        aid = card["agent_id"]
        if aid in self._agents:
            raise ValueError(f"agent '{aid}' already enrolled")
        agent = Agent(
            agent_id=aid,
            owner=card.get("owner"),
            permissions=list(card.get("permissions", [])),
            risk_tier=card.get("risk_tier", "minimal"),
        )
        self._agents[aid] = agent
        return agent

    def orphans(self) -> list[Agent]:
        return [a for a in self._agents.values() if not a.owner]

    def reconcile(self, observed_ids: list[str]) -> list[str]:
        # Declared (registry) vs acting (telemetry): the gap = ghosts.
        return [oid for oid in observed_ids if oid not in self._agents]

    def coverage(self) -> dict:
        total = len(self._agents)
        governed = sum(1 for a in self._agents.values() if a.is_governed())
        rate = round(100 * governed / total, 1) if total else 0.0
        return {
            "total_agents": total,
            "governed_agents": governed,
            "orphans": len(self.orphans()),
            "coverage_pct": rate,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {"agents": [asdict(a) for a in self._agents.values()]}
        # Atomic-ish write so a crash mid-write can't corrupt the live registry.
        tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent or ".", suffix=".tmp")
        try:
            with open(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            Path(tmp_name).replace(path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "MediumRegistry":
        reg = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for raw in data.get("agents", []):
            reg._agents[raw["agent_id"]] = Agent(**raw)
        return reg


def _demo_medium() -> MediumRegistry:
    reg = MediumRegistry()
    cards = [
        {"agent_id": "agt-invoice", "owner": "alice", "permissions": ["read:invoices"], "risk_tier": "high"},
        {"agent_id": "agt-notes", "owner": "bob", "permissions": ["read:notes"], "risk_tier": "minimal"},
        {"agent_id": "agt-payroll", "owner": "alice", "permissions": ["trigger:payment"], "risk_tier": "high"},
        {"agent_id": "agt-screener", "owner": "carol", "permissions": ["read:cv"], "risk_tier": "limited"},
        {"agent_id": "agt-shadow", "owner": None, "permissions": ["read:files"], "risk_tier": "limited"},
    ]
    for c in cards:
        reg.enrol_from_card(c)
    return reg


# === HARD ===
# Goal: auditable, versioned registry (append-only history, lifecycle, board report).

class HardRegistry:
    """Source of authority: every mutation is horodated and appended to an
    append-only history; nothing is ever destructively deleted."""

    # Thresholds that drive board alerts.
    MAX_AGENTS_PER_OWNER = 3

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self.history: list[dict] = []  # append-only audit journal

    def enrol_from_card(self, card: dict) -> Agent:
        problems = validate_agent_card(card)
        if problems:
            raise ValueError(f"invalid Agent Card: {'; '.join(problems)}")
        aid = card["agent_id"]
        if aid in self._agents:
            raise ValueError(f"agent '{aid}' already enrolled")
        agent = Agent(
            agent_id=aid,
            owner=card.get("owner"),
            permissions=list(card.get("permissions", [])),
            risk_tier=card.get("risk_tier", "minimal"),
        )
        self._agents[aid] = agent
        self.history.append({"timestamp": _now_iso(), "agent_id": aid, "change": "enrolled"})
        return agent

    def transition(self, agent_id: str, *, status: str | None = None, owner: str | None = None) -> Agent:
        if agent_id not in self._agents:
            raise KeyError(f"unknown agent '{agent_id}'")
        agent = self._agents[agent_id]
        change = {}
        if status is not None:
            if status not in {"active", "suspended", "decommissioned"}:
                raise ValueError(f"invalid status '{status}'")
            change["status"] = status
            agent.status = status
        if owner is not None:
            change["owner"] = owner
            agent.owner = owner
        agent.updated_at = _now_iso()
        # Append-only: we record the mutation, we never delete the row.
        self.history.append({"timestamp": agent.updated_at, "agent_id": agent_id, "change": change})
        return agent

    # --- Queries ---------------------------------------------------------

    def by_risk(self, min_tier: str = "minimal") -> list[Agent]:
        floor = RISK_ORDER[min_tier]
        sel = [a for a in self._agents.values() if RISK_ORDER.get(a.risk_tier, 0) >= floor]
        return sorted(sel, key=lambda a: RISK_ORDER.get(a.risk_tier, 0), reverse=True)

    def ownership_concentration(self) -> dict:
        counts: dict[str, int] = {}
        for a in self._agents.values():
            if a.owner:
                counts[a.owner] = counts.get(a.owner, 0) + 1
        return counts

    def active_orphans(self) -> list[Agent]:
        # Worst case: acts (active) and imputable to nobody (no owner).
        return [a for a in self._agents.values() if not a.owner and a.status == "active"]

    def _active(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.status == "active"]

    def build_report(self) -> dict:
        total = len(self._agents)
        governed = sum(1 for a in self._agents.values() if a.is_governed())
        coverage_pct = round(100 * governed / total, 1) if total else 0.0
        active = self._active()
        conc = self.ownership_concentration()
        active_orphans = self.active_orphans()
        high_risk_active = [a for a in active if a.risk_tier == "high"]

        alerts: list[str] = []
        if active_orphans:
            alerts.append(f"{len(active_orphans)} active orphan(s) acting without an owner")
        for owner, n in conc.items():
            if n > self.MAX_AGENTS_PER_OWNER:
                alerts.append(f"owner '{owner}' owns {n} agents (> {self.MAX_AGENTS_PER_OWNER}, accountability bottleneck)")

        return {
            "coverage_pct": coverage_pct,
            "total_active": len(active),
            "n_active_orphans": len(active_orphans),
            "ownership_concentration": conc,
            "high_risk_active": len(high_risk_active),
            "alerts": alerts,
        }


def _demo_hard() -> HardRegistry:
    reg = HardRegistry()
    cards = [
        {"agent_id": "agt-invoice", "owner": "alice", "permissions": ["read:invoices"], "risk_tier": "high"},
        {"agent_id": "agt-payroll", "owner": "alice", "permissions": ["trigger:payment"], "risk_tier": "high"},
        {"agent_id": "agt-recon", "owner": "alice", "permissions": ["read:ledger"], "risk_tier": "limited"},
        {"agent_id": "agt-audit", "owner": "alice", "permissions": ["read:logs"], "risk_tier": "limited"},  # alice -> 4 (bottleneck)
        {"agent_id": "agt-notes", "owner": "bob", "permissions": ["read:notes"], "risk_tier": "minimal"},
        {"agent_id": "agt-shadow", "owner": None, "permissions": ["read:files"], "risk_tier": "limited"},  # active orphan
    ]
    for c in cards:
        reg.enrol_from_card(c)
    return reg


# === SMOKE TEST ===

if __name__ == "__main__":
    print("=== Solutions Day 3 smoke test ===\n")

    # --- EASY ---
    print("[EASY]")
    easy = _demo_easy()
    assert easy.count() == 4, "easy: expected 4 agents"
    assert len(easy.by_owner("alice")) == 2, "easy: alice should own 2"
    assert len(easy.orphans()) == 1, "easy: exactly 1 orphan"
    try:
        easy.add(EasyAgent("agt-invoice", "x"))  # duplicate id
        raise AssertionError("easy: duplicate id should have raised")
    except ValueError:
        pass
    print(f"  total={easy.count()} alice_owns={len(easy.by_owner('alice'))} orphans={len(easy.orphans())} (duplicate rejected) OK")

    # --- MEDIUM ---
    print("\n[MEDIUM]")
    med = _demo_medium()
    cov = med.coverage()
    assert cov["total_agents"] == 5 and cov["governed_agents"] == 4, "medium: coverage counts"
    assert cov["coverage_pct"] == 80.0, "medium: 4/5 => 80.0"
    # Adversarial: invalid card must be rejected.
    try:
        med.enrol_from_card({"owner": "dave", "risk_tier": "ultra"})
        raise AssertionError("medium: invalid card should have raised")
    except ValueError:
        pass
    # Persistence round-trip.
    out = Path(tempfile.gettempdir()) / "solutions_day3_registry.json"
    med.save(out)
    reloaded = MediumRegistry.load(out)
    assert len(reloaded._agents) == 5, "medium: round-trip agent count"
    assert {a.owner for a in reloaded._agents.values()} == {a.owner for a in med._agents.values()}, "medium: owners preserved"
    # Reconciliation.
    ghosts = med.reconcile(["agt-invoice", "agt-ghost-1", "agt-ghost-2"])
    assert ghosts == ["agt-ghost-1", "agt-ghost-2"], "medium: ghosts detected"
    print(f"  coverage={cov} ghosts={ghosts} round-trip OK (invalid card rejected)")

    # --- HARD ---
    print("\n[HARD]")
    hard = _demo_hard()
    history_before = len(hard.history)
    # by_risk sorted highest-first.
    risky = hard.by_risk("limited")
    tiers = [a.risk_tier for a in risky]
    assert tiers == sorted(tiers, key=lambda t: RISK_ORDER[t], reverse=True), "hard: by_risk sort"
    # Lifecycle: decommission an agent (no delete).
    hard.transition("agt-notes", status="decommissioned")
    assert "agt-notes" in hard._agents, "hard: row must survive decommission"
    assert hard._agents["agt-notes"].status == "decommissioned", "hard: status changed"
    assert len(hard.history) == history_before + 1, "hard: history is append-only and grew"
    assert all(a.agent_id != "agt-notes" for a in hard._active()), "hard: decommissioned not in active"
    # Adversarial: invalid status rejected.
    try:
        hard.transition("agt-invoice", status="paused")
        raise AssertionError("hard: invalid status should have raised")
    except ValueError:
        pass
    # Board report: must surface the active orphan AND alice's bottleneck.
    report = hard.build_report()
    assert report["n_active_orphans"] == 1, "hard: 1 active orphan expected"
    assert any("orphan" in alert for alert in report["alerts"]), "hard: orphan alert"
    assert any("alice" in alert for alert in report["alerts"]), "hard: ownership bottleneck alert"
    # Probe invariant: updated_at changed on decommission.
    notes = hard._agents["agt-notes"]
    assert notes.updated_at >= notes.enrolled_at, "hard: updated_at moved forward"
    print(f"  by_risk_tiers={tiers}")
    print(f"  report={json.dumps(report, ensure_ascii=False)}")
    print("  decommission kept row + grew history + left active set OK")

    print("\nAll levels passed.")

"""Organizational ownership resolver for an agent fleet (Day 11).

This script demonstrates the *organizational* governance layer that sits on top
of the technical registry (Day 3): given a fleet of agents and an org of human
actors, it resolves the RACI accountability chain and maps each actor to a line
of the IIA Three Lines Model, then flags responsibility gaps a board would want
to know about.

Why this matters: a technical owner ("this agent belongs to team X") is NOT the
same as accountability ("a named human answers for this agent to the board").
The classic RACI rule is that there must be EXACTLY ONE Accountable per agent.
Zero accountable = nobody answers; two accountable = diffused responsibility =
nobody answers either. This resolver makes that rule machine-checkable.

Real-world anchors (re-implemented here in miniature, stdlib only):
- RACI matrix (Responsible / Accountable / Consulted / Informed) — classic PM tool.
- IIA Three Lines Model (2020): line 1 operates risk, line 2 supports/challenges,
  line 3 audits independently and reports to the governing body (board).
- HBR (2026) "agent managers": a line-1 operational role supervising a fleet.

# requires: stdlib only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --- The three "lines" of the IIA Three Lines Model. -----------------------
# WHY an Enum: a line is a closed set of legal values; encoding it as a string
# would let typos ("lien 1") slip through silently. The Enum makes an invalid
# line a hard error at construction time.
class Line(Enum):
    FIRST = "1-operational-management"   # operates the agent (e.g. agent manager)
    SECOND = "2-risk-and-compliance"     # supports & challenges (risk, security, DPO)
    THIRD = "3-internal-audit"           # independent assurance, reports to board
    BOARD = "board"                      # governing body, accountable in fine


# --- RACI roles. -----------------------------------------------------------
class Raci(Enum):
    RESPONSIBLE = "R"  # does the work (can be several people)
    ACCOUNTABLE = "A"  # answers for the result (MUST be exactly one)
    CONSULTED = "C"    # gives input before decisions
    INFORMED = "I"     # kept in the loop afterwards


@dataclass(frozen=True)
class Actor:
    """A human (or team) in the org chart, pinned to one Three-Lines line.

    WHY frozen: an actor's identity and line should not mutate underneath a
    resolution; if a reorg happens you create a new org snapshot, you don't
    silently edit the old one.
    """

    actor_id: str
    name: str
    line: Line


@dataclass
class Agent:
    """An agent as the *organizational* layer sees it.

    `raci` maps an actor_id -> RACI role for THIS agent. The technical fields
    (scopes, risk_tier) mirror the shared mini-model used across the domain,
    but Day 11 only cares about the human responsibility chain.
    """

    agent_id: str
    risk_tier: str  # "minimal" | "limited" | "high" — kept for continuity (J5)
    raci: dict[str, Raci] = field(default_factory=dict)


@dataclass
class GapReport:
    """A single responsibility gap found on one agent."""

    agent_id: str
    severity: str  # "critical" | "warning"
    issue: str


class OwnershipResolver:
    """Resolves accountability and Three-Lines coverage over a fleet.

    WHY a class: it bundles the org snapshot (who exists, in which line) with the
    queries, so a caller cannot accidentally resolve an agent against a stale or
    mismatched org chart.
    """

    def __init__(self, actors: list[Actor]) -> None:
        # Index actors by id for O(1) lookup during resolution.
        self._actors: dict[str, Actor] = {a.actor_id: a for a in actors}

    # -- Core RACI query ----------------------------------------------------
    def accountable_for(self, agent: Agent) -> Optional[Actor]:
        """Return the single Accountable actor, or None if absent/ambiguous.

        WHY None on ambiguity: zero OR multiple accountables both mean "nobody
        truly answers". We collapse both to None and surface the distinction in
        the gap report, so callers can't treat a broken chain as valid.
        """
        accountables = [
            self._actors.get(aid)
            for aid, role in agent.raci.items()
            if role is Raci.ACCOUNTABLE
        ]
        # Exactly one known actor flagged Accountable -> valid.
        known = [a for a in accountables if a is not None]
        if len(known) == 1 and len(accountables) == 1:
            return known[0]
        return None

    def responsibles_for(self, agent: Agent) -> list[Actor]:
        """All actors marked Responsible (the operators, e.g. agent managers)."""
        return [
            self._actors[aid]
            for aid, role in agent.raci.items()
            if role is Raci.RESPONSIBLE and aid in self._actors
        ]

    def lines_covered(self, agent: Agent) -> set[Line]:
        """Which Three-Lines lines have at least one actor on this agent."""
        return {
            self._actors[aid].line
            for aid in agent.raci
            if aid in self._actors
        }

    # -- Gap detection ------------------------------------------------------
    def find_gaps(self, agent: Agent) -> list[GapReport]:
        """Flag the responsibility holes a board/audit would care about."""
        gaps: list[GapReport] = []

        # Gap 1: accountability is the non-negotiable RACI rule.
        accountable_ids = [
            aid for aid, role in agent.raci.items() if role is Raci.ACCOUNTABLE
        ]
        if len(accountable_ids) == 0:
            gaps.append(GapReport(agent.agent_id, "critical",
                                  "no Accountable (A) - nobody answers for this agent"))
        elif len(accountable_ids) > 1:
            gaps.append(GapReport(agent.agent_id, "critical",
                                  f"{len(accountable_ids)} Accountables - diffused "
                                  "responsibility (RACI requires exactly one)"))

        # Gap 2: a RACI entry pointing to an actor who left the org chart.
        for aid in agent.raci:
            if aid not in self._actors:
                gaps.append(GapReport(agent.agent_id, "critical",
                                      f"actor '{aid}' is in the RACI but absent from "
                                      "the org chart (ghost responsibility)"))

        # Gap 3: no first line means no operator — the agent runs unsupervised.
        covered = self.lines_covered(agent)
        if Line.FIRST not in covered:
            gaps.append(GapReport(agent.agent_id, "warning",
                                  "no first line (operational management) - agent runs "
                                  "without a named operator / agent manager"))

        # Gap 4 (risk-scaled): high-risk agents need independent assurance (line 3).
        # WHY only high-risk: requiring internal audit on every minimal-risk agent
        # would not scale — oversight is calibrated to risk (board appetite, J10/J5).
        if agent.risk_tier == "high" and Line.THIRD not in covered:
            gaps.append(GapReport(agent.agent_id, "warning",
                                  "high-risk agent with no third line (internal audit) "
                                  "in scope - independent assurance missing"))

        return gaps

    # -- Fleet-level rollup -------------------------------------------------
    def board_summary(self, fleet: list[Agent]) -> dict[str, int]:
        """The numbers a board cares about: coverage and gap counts."""
        total = len(fleet)
        with_accountable = sum(1 for a in fleet if self.accountable_for(a) is not None)
        all_gaps = [g for a in fleet for g in self.find_gaps(a)]
        critical = sum(1 for g in all_gaps if g.severity == "critical")
        return {
            "agents_total": total,
            "agents_with_accountable": with_accountable,
            "agents_without_clear_accountable": total - with_accountable,
            "gaps_total": len(all_gaps),
            "gaps_critical": critical,
        }


def _demo() -> None:
    """Build a small org + fleet and show the resolver flagging real gaps."""
    actors = [
        Actor("alice", "Alice (agent manager, finance)", Line.FIRST),
        Actor("bob", "Bob (business owner, finance)", Line.FIRST),
        Actor("carol", "Carol (AI risk & compliance)", Line.SECOND),
        Actor("dave", "Dave (internal audit)", Line.THIRD),
        Actor("board", "Board / AI committee", Line.BOARD),
    ]
    resolver = OwnershipResolver(actors)

    fleet = [
        # A well-governed high-risk agent: one A, an operator, audit in scope.
        Agent("invoice-reconciler", "high", {
            "alice": Raci.RESPONSIBLE,
            "bob": Raci.ACCOUNTABLE,
            "carol": Raci.CONSULTED,
            "dave": Raci.INFORMED,
            "board": Raci.INFORMED,
        }),
        # The section-1 nightmare: nobody is Accountable.
        Agent("rogue-categorizer", "high", {
            "alice": Raci.RESPONSIBLE,
        }),
        # Two Accountables: diffused responsibility.
        Agent("forecast-bot", "limited", {
            "bob": Raci.ACCOUNTABLE,
            "carol": Raci.ACCOUNTABLE,
        }),
        # Ghost responsibility: 'eve' left the org but is still in the RACI.
        Agent("legacy-mailer", "minimal", {
            "eve": Raci.ACCOUNTABLE,
            "alice": Raci.RESPONSIBLE,
        }),
    ]

    print("=== Per-agent resolution ===")
    for agent in fleet:
        acc = resolver.accountable_for(agent)
        acc_label = acc.name if acc else "*** NONE / AMBIGUOUS ***"
        lines = sorted(line.value for line in resolver.lines_covered(agent))
        print(f"\n[{agent.agent_id}]  risk={agent.risk_tier}")
        print(f"  Accountable : {acc_label}")
        print(f"  Lines covered: {', '.join(lines) or '(none)'}")
        gaps = resolver.find_gaps(agent)
        if gaps:
            for g in gaps:
                print(f"  GAP ({g.severity}): {g.issue}")
        else:
            print("  GAP: none - responsibility chain is clean")

    print("\n=== Board summary (the numbers oversight cares about) ===")
    for key, value in resolver.board_summary(fleet).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _demo()

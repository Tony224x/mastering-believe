"""Solutions — Day 11 (organizational design & boardroom).

One file, three levels, separated by section banners. stdlib only.
Run `python 11-design-organisationnel.py` for the smoke test of all levels.

# requires: stdlib only
"""

from __future__ import annotations


# === EASY ===
# Goal: enforce the RACI golden rule — exactly one Accountable ("A") per agent.

EASY_FLEET: dict[str, dict[str, str]] = {
    "daily-report": {"sam": "R", "lea": "A", "kim": "I"},
    "price-watcher": {"sam": "R", "kim": "C"},          # no A
    "shipping-bot": {"lea": "A", "kim": "A"},           # two A
}


def count_accountables(raci: dict[str, str]) -> int:
    """How many actors hold the Accountable ('A') role for this agent."""
    return sum(1 for role in raci.values() if role == "A")


def is_clearly_owned(raci: dict[str, str]) -> bool:
    """True iff exactly one Accountable — the non-negotiable RACI rule."""
    return count_accountables(raci) == 1


def run_easy() -> None:
    owned = 0
    for agent_id, raci in EASY_FLEET.items():
        n = count_accountables(raci)
        if is_clearly_owned(raci):
            owned += 1
            print(f"{agent_id} : OK")
        else:
            print(f"{agent_id} : ORPHELIN ({n} accountables)")
    print(f"Clairement possedes : {owned}/{len(EASY_FLEET)}")


# === MEDIUM ===
# Goal: add the IIA Three Lines dimension and flag line-coverage gaps.

# actor_id -> line. Lines: "1" operational, "2" risk/compliance, "3" audit, "board".
MEDIUM_ACTORS: dict[str, str] = {
    "sam": "1", "lea": "1", "kim": "2", "max": "3", "board": "board",
}

# agent_id -> {"risk_tier": ..., "raci": {actor_id: role}}
MEDIUM_FLEET: dict[str, dict] = {
    "reconciler": {  # high, well covered (lines 1 and 3 present)
        "risk_tier": "high",
        "raci": {"sam": "R", "lea": "A", "kim": "C", "max": "I", "board": "I"},
    },
    "forecaster": {  # high, NO third line
        "risk_tier": "high",
        "raci": {"sam": "R", "lea": "A", "kim": "C"},
    },
    "newsletter": {  # minimal, NO first line
        "risk_tier": "minimal",
        "raci": {"kim": "A", "board": "I"},
    },
}


def lines_covered(agent: dict, actors: dict[str, str]) -> set[str]:
    """Set of Three-Lines lines present in the agent's RACI (unknown actors ignored)."""
    return {
        actors[aid]
        for aid in agent["raci"]
        if aid in actors
    }


def find_line_gaps(agent: dict, actors: dict[str, str]) -> list[str]:
    """Flag missing first line and (for high-risk only) missing third line."""
    gaps: list[str] = []
    covered = lines_covered(agent, actors)
    if "1" not in covered:
        gaps.append("no first line")
    # WHY high-risk only: independent assurance (line 3) is calibrated to risk,
    # mirroring board risk appetite — requiring it everywhere would not scale.
    if agent["risk_tier"] == "high" and "3" not in covered:
        gaps.append("high-risk without third line")
    return gaps


def run_medium() -> None:
    for agent_id, agent in MEDIUM_FLEET.items():
        covered = ", ".join(sorted(lines_covered(agent, MEDIUM_ACTORS)))
        gaps = find_line_gaps(agent, MEDIUM_ACTORS)
        gap_label = "; ".join(gaps) if gaps else "aucun"
        print(f"{agent_id} (risk={agent['risk_tier']}) lignes=[{covered}] gaps={gap_label}")


# === HARD ===
# Goal: assemble a board-ready oversight report with a governance score.

HARD_ACTORS: dict[str, str] = {
    "sam": "1", "lea": "1", "kim": "2", "max": "3", "board": "board",
}

HARD_FLEET: dict[str, dict] = {
    "reconciler": {  # healthy: one A, line 1 + line 3 present
        "risk_tier": "high",
        "raci": {"sam": "R", "lea": "A", "kim": "C", "max": "I"},
    },
    "ghost-mailer": {  # phantom responsibility: 'eve' is not in the org chart
        "risk_tier": "minimal",
        "raci": {"eve": "A", "sam": "R"},
    },
    "double-owner": {  # diffused: two Accountables
        "risk_tier": "limited",
        "raci": {"lea": "A", "kim": "A"},
    },
    "forecaster": {  # high without third line -> warning
        "risk_tier": "high",
        "raci": {"sam": "R", "lea": "A", "kim": "C"},
    },
}


def find_gaps(agent: dict, actors: dict[str, str]) -> list[tuple[str, str]]:
    """Return (severity, message) tuples. severity in {'critical', 'warning'}."""
    gaps: list[tuple[str, str]] = []
    raci = agent["raci"]

    accountable_ids = [aid for aid, role in raci.items() if role == "A"]
    if len(accountable_ids) == 0:
        gaps.append(("critical", "no Accountable"))
    elif len(accountable_ids) > 1:
        gaps.append(("critical", f"{len(accountable_ids)} Accountables (diffused)"))

    # Phantom responsibility: a RACI entry pointing at a non-existent actor.
    for aid in raci:
        if aid not in actors:
            gaps.append(("critical", f"phantom responsibility: '{aid}' not in org chart"))

    covered = {actors[aid] for aid in raci if aid in actors}
    if "1" not in covered:
        gaps.append(("warning", "no first line (operator)"))
    if agent["risk_tier"] == "high" and "3" not in covered:
        gaps.append(("warning", "high-risk without third line"))

    return gaps


def _has_single_known_accountable(agent: dict, actors: dict[str, str]) -> bool:
    """True iff exactly one Accountable AND that actor exists in the org chart."""
    accs = [aid for aid, role in agent["raci"].items() if role == "A"]
    return len(accs) == 1 and accs[0] in actors


def governance_score(fleet: dict[str, dict], actors: dict[str, str]) -> int:
    """0..100 coverage score, penalized by gap severity.

    Probe choice: an EMPTY fleet returns 100 (nothing to reproach) rather than
    0/undefined — there are no agents lacking an Accountable.
    """
    if not fleet:
        return 100
    with_acc = sum(
        1 for a in fleet.values() if _has_single_known_accountable(a, actors)
    )
    base = round(100 * with_acc / len(fleet))
    penalty = 0
    for agent in fleet.values():
        for severity, _msg in find_gaps(agent, actors):
            penalty += 10 if severity == "critical" else 3
    return max(0, base - penalty)


def _criticality_rank(agent: dict, actors: dict[str, str]) -> int:
    """Sort key: 0 = has a critical gap, 1 = only warnings, 2 = clean."""
    severities = {sev for sev, _ in find_gaps(agent, actors)}
    if "critical" in severities:
        return 0
    if "warning" in severities:
        return 1
    return 2


def rank_agents(fleet: dict[str, dict], actors: dict[str, str]) -> list[str]:
    """Remediation stack: most critical agents first, healthy ones last."""
    return sorted(fleet, key=lambda aid: _criticality_rank(fleet[aid], actors))


def build_report(fleet: dict[str, dict], actors: dict[str, str]) -> str:
    """Board-ready oversight report as markdown text."""
    score = governance_score(fleet, actors)
    without_acc = sum(
        1 for a in fleet.values() if not _has_single_known_accountable(a, actors)
    )
    all_gaps = [g for a in fleet.values() for g in find_gaps(a, actors)]
    n_critical = sum(1 for sev, _ in all_gaps if sev == "critical")
    n_warning = sum(1 for sev, _ in all_gaps if sev == "warning")

    lines = [
        "# Rapport d'oversight - flotte d'agents",
        f"- Score de gouvernance : {score}/100",
        f"- Agents sans Accountable clair : {without_acc}/{len(fleet)}",
        f"- Gaps : {n_critical} critical, {n_warning} warning",
        "",
        "## Pile de remediation (du plus critique au plus sain)",
    ]
    for agent_id in rank_agents(fleet, actors):
        gaps = find_gaps(fleet[agent_id], actors)
        if gaps:
            detail = "; ".join(f"[{sev}] {msg}" for sev, msg in gaps)
        else:
            detail = "OK - chaine de responsabilite propre"
        lines.append(f"- {agent_id} : {detail}")
    return "\n".join(lines)


def run_hard() -> None:
    print(build_report(HARD_FLEET, HARD_ACTORS))


# === SMOKE TEST ===
if __name__ == "__main__":
    print("----- EASY -----")
    run_easy()

    print("\n----- MEDIUM -----")
    run_medium()

    print("\n----- HARD -----")
    run_hard()

    # Adversarial probes — assert invariants hold on edge cases.
    print("\n----- ASSERTIONS -----")
    assert count_accountables({"a": "A", "b": "A"}) == 2
    assert is_clearly_owned({"a": "A"}) is True
    assert is_clearly_owned({"a": "R"}) is False
    # Unknown actor is ignored by line coverage.
    assert lines_covered({"raci": {"ghost": "A"}}, MEDIUM_ACTORS) == set()
    # High-risk without third line is flagged; minimal is not.
    assert find_line_gaps(MEDIUM_FLEET["forecaster"], MEDIUM_ACTORS) == [
        "high-risk without third line"
    ]
    # Empty fleet must not crash and scores 100 by convention.
    assert governance_score({}, HARD_ACTORS) == 100
    # Score is always bounded in [0, 100].
    s = governance_score(HARD_FLEET, HARD_ACTORS)
    assert 0 <= s <= 100, s
    # Phantom-responsibility agent must surface a critical gap.
    assert any(
        sev == "critical" for sev, _ in find_gaps(HARD_FLEET["ghost-mailer"], HARD_ACTORS)
    )
    # Critical agents must rank before healthy ones.
    order = rank_agents(HARD_FLEET, HARD_ACTORS)
    assert order[0] in {"ghost-mailer", "double-owner"}
    assert order[-1] == "reconciler"
    print("all assertions passed")

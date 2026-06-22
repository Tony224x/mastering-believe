"""
Solutions -- Day 9 (HARD): Multi-agent patterns

Contains solutions for:
  - Hard Ex 1: Hierarchical multi-team -- a CEO routes to sub-teams (each with
               its own sub-supervisor + workers); managers condense worker
               output; reports aggregate up to the CEO
  - Hard Ex 2: Supervisor + swarm HYBRID -- supervisor starts the task, agents
               hand off laterally, with a hop budget + tight-loop detector, and
               a final-answer assembler that proves >= 3 distinct contributors

Self-contained: deterministic agents + MockLLM, RUNS OFFLINE with zero
dependencies (no langgraph, no API key). langgraph is referenced behind a
try/except only to mirror the course code.

Run:  python 03-exercises/solutions/09-multi-agent-patterns-hard.py
Each solution is self-contained and ends with assertions (self-test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

HAS_LANGGRAPH = False
try:  # pragma: no cover - environment dependent
    import langgraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    pass


# ==========================================================================
# HARD EXERCISE 1 -- Hierarchical multi-team: routing + upward aggregation
# ==========================================================================

class HierMockLLM:
    """
    Deterministic stub for a 2-level hierarchy.
      - workers produce verbose raw output
      - managers CONDENSE worker output into a short team report
      - CEO synthesizes from team reports only
    """

    def __call__(self, role: str, task: str, context: str = "") -> str:
        # --- workers (verbose) -------------------------------------------
        if role == "sql_agent":
            return ("[sql_agent] SELECT region, SUM(rev) FROM sales GROUP BY region; "
                    "-> 12 rows, north leads with 1.2M, south 0.9M, east 0.7M, west 0.5M "
                    "(long verbose dump simulating a real query result set)")
        if role == "stats_agent":
            return ("[stats_agent] mean=825k std=290k, north is +45% vs mean, "
                    "trend +8% QoQ, seasonality detected, confidence 0.93 "
                    "(verbose statistical breakdown across many metrics)")
        if role == "writer_agent":
            return ("[writer_agent] Draft article: 'Regional growth accelerates...' "
                    "followed by 5 long paragraphs of prose simulating real copy "
                    "that a manager must compress before sending upward")
        if role == "translator_agent":
            return ("[translator_agent] FR translation of the full article, "
                    "another long block of text simulating a complete translation")
        # --- managers (condense) -----------------------------------------
        if role == "data_manager" and task.startswith("report:"):
            return "[data team] North leads regional revenue (1.2M, +45% vs mean)."
        if role == "content_manager" and task.startswith("report:"):
            return "[content team] Article drafted and translated to FR; ready to ship."
        # --- CEO (aggregate) ---------------------------------------------
        if role == "ceo" and task.startswith("synthesize:"):
            teams = [line for line in context.splitlines() if line.strip()]
            body = " ".join(teams)
            return ("Synthese executive (CEO) :\n"
                    f"{body}\n"
                    f"(agrege a partir de {len(teams)} rapport(s) d'equipe, "
                    "sans voir la sortie brute des workers).")
        raise ValueError(f"Unknown role/task: {role} / {task}")


# Team registry: each team has a manager and its workers.
TEAMS = {
    "data": {"manager": "data_manager", "workers": ["sql_agent", "stats_agent"]},
    "content": {"manager": "content_manager", "workers": ["writer_agent", "translator_agent"]},
}

_DATA_HINTS = ("donnees", "sql", "metrique", "stats", "chiffre", "revenu")
_CONTENT_HINTS = ("redige", "article", "traduis", "resume", "texte")


@dataclass
class HierarchicalCEO:
    llm: HierMockLLM
    teams: dict = field(default_factory=lambda: TEAMS)
    fallback: str = "content"

    def route_teams(self, task: str) -> list[str]:
        """Decide which sub-teams to activate. Never returns an empty list."""
        low = task.lower()
        activated: list[str] = []
        if any(h in low for h in _DATA_HINTS):
            activated.append("data")
        if any(h in low for h in _CONTENT_HINTS):
            activated.append("content")
        return activated or [self.fallback]

    def run(self, task: str) -> dict:
        activated = self.route_teams(task)
        team_reports: dict[str, str] = {}
        worker_calls = 0
        raw_by_team: dict[str, str] = {}

        for team in activated:
            manager = self.teams[team]["manager"]
            workers = self.teams[team]["workers"]
            # Manager runs its own mini-supervisor loop over its workers.
            raw_outputs = []
            for w in workers:
                raw_outputs.append(self.llm(w, f"do: {task}"))
                worker_calls += 1
            raw = "\n".join(raw_outputs)
            raw_by_team[team] = raw
            # Manager CONDENSES before reporting up (anti context-explosion).
            report = self.llm(manager, f"report: {task}", context=raw)
            team_reports[team] = report

        ceo_context = "\n".join(team_reports.values())
        final = self.llm("ceo", f"synthesize: {task}", context=ceo_context)
        return {
            "teams_activated": activated,
            "team_reports": team_reports,
            "raw_by_team": raw_by_team,
            "worker_calls": worker_calls,
            "final_answer": final,
        }


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Hierarchical multi-team: routing + upward aggregation")
    print("=" * 70)

    ceo = HierarchicalCEO(HierMockLLM())

    # Scenario A: pure data task
    a = ceo.run("Analyse les donnees de revenu par region (sql + stats)")
    print(f"  data task    -> teams={a['teams_activated']} worker_calls={a['worker_calls']}")
    assert a["teams_activated"] == ["data"]
    assert a["worker_calls"] == 2

    # Scenario B: pure content task
    b = ceo.run("Redige un article et traduis-le")
    print(f"  content task -> teams={b['teams_activated']} worker_calls={b['worker_calls']}")
    assert b["teams_activated"] == ["content"]
    assert b["worker_calls"] == 2

    # Scenario C: mixed task -> both teams
    c = ceo.run("Sors les chiffres de revenu puis redige un article de synthese")
    print(f"  mixed task   -> teams={c['teams_activated']} worker_calls={c['worker_calls']}")
    assert c["teams_activated"] == ["data", "content"]
    assert c["worker_calls"] == 4

    # Managers condense: each report strictly shorter than its raw worker dump
    for team, report in c["team_reports"].items():
        assert len(report) < len(c["raw_by_team"][team]), f"{team} report not condensed"

    # CEO aggregates ONLY team reports, never raw worker output.
    final = c["final_answer"]
    assert "sql_agent" not in final and "writer_agent" not in final, "raw leaked to CEO"
    # Final answer cites each activated team
    assert "[data team]" in final and "[content team]" in final, final

    # Fallback: a task matching nothing still activates one team, no exception.
    d = ceo.run("Fais quelque chose de vague et indefini")
    assert d["teams_activated"] == ["content"], d["teams_activated"]
    print("  fallback     -> teams=" + str(d["teams_activated"]))
    print("[Verification] PASS -- routing, condensation, upward aggregation, fallback")


# ==========================================================================
# HARD EXERCISE 2 -- Supervisor + swarm hybrid: loop guard + assembler
# ==========================================================================

@dataclass
class Handoff:
    to_agent: str
    reason: str
    payload: dict = field(default_factory=dict)


# Each agent: (state) -> (contribution: dict, handoff | None)
# contribution = {"agent": name, "artifact": text}; state carries 'contributions'.

def planner_agent(state: dict):
    contrib = {"agent": "planner", "artifact": "plan: research -> code -> review"}
    return contrib, Handoff("researcher", "need facts", {})


def researcher_agent(state: dict):
    contrib = {"agent": "researcher", "artifact": "facts: use Timsort, O(n log n)"}
    return contrib, Handoff("coder", "facts ready", {})


def coder_agent(state: dict):
    # If we came back from the reviewer once, mark the fix.
    fixes = sum(1 for c in state["contributions"] if c["agent"] == "coder")
    artifact = "code v2 (fixed)" if fixes else "code v1: def f(): ..."
    contrib = {"agent": "coder", "artifact": artifact}
    return contrib, Handoff("reviewer", "code ready", {})


def reviewer_agent(state: dict):
    # Legitimate single correction loop: first review sends back to coder once.
    reviews = sum(1 for c in state["contributions"] if c["agent"] == "reviewer")
    if reviews == 0:
        contrib = {"agent": "reviewer", "artifact": "review v1: needs a fix"}
        return contrib, Handoff("coder", "found a bug, fix it", {})
    contrib = {"agent": "reviewer", "artifact": "review v2: LGTM"}
    return contrib, None  # terminal


GOOD_AGENTS: dict[str, Callable] = {
    "planner": planner_agent,
    "researcher": researcher_agent,
    "coder": coder_agent,
    "reviewer": reviewer_agent,
}


# Degenerate agents that bounce forever (tight-loop test).
def bad_coder(state: dict):
    return {"agent": "coder", "artifact": "x"}, Handoff("reviewer", "loop", {})


def bad_reviewer(state: dict):
    return {"agent": "reviewer", "artifact": "y"}, Handoff("coder", "loop", {})


BAD_AGENTS: dict[str, Callable] = {
    "planner": planner_agent,
    "researcher": researcher_agent,
    "coder": bad_coder,
    "reviewer": bad_reviewer,
}


def run_hybrid(task: str, agents: dict, start: str = "planner", max_hops: int = 12) -> dict:
    """
    Supervisor kicks off via `start`; execution proceeds as a swarm of
    handoffs. Guards: hard hop budget + tight-loop detector (same (from,to)
    handoff edge repeated twice in a row).
    """
    state = {"task": task, "contributions": []}
    control = start
    control_seq: list[str] = []
    edges: list[tuple[str, str]] = []  # handoff edges, for cycle detection

    for hop in range(max_hops + 1):
        if hop == max_hops:
            raise RuntimeError("hop budget exceeded")
        control_seq.append(control)
        contrib, handoff = agents[control](state)
        state["contributions"].append(contrib)
        if handoff is None:
            return {"control_seq": control_seq, "contributions": state["contributions"]}
        edge = (control, handoff.to_agent)
        # Tight loop: the exact same edge twice in a row (A->B then A->B again).
        if len(edges) >= 1 and edges[-1] == edge:
            raise RuntimeError(f"tight loop {edge[0]}<->{edge[1]}")
        edges.append(edge)
        control = handoff.to_agent
    raise RuntimeError("hop budget exceeded")


def assemble_final(contributions: list[dict]) -> str:
    """Unified report; requires >= 3 distinct contributors; dedup by agent."""
    distinct = {c["agent"] for c in contributions}
    if len(distinct) < 3:
        raise ValueError("not enough distinct contributors")
    # Keep the LAST contribution per agent, in first-seen order.
    order: list[str] = []
    latest: dict[str, str] = {}
    for c in contributions:
        if c["agent"] not in order:
            order.append(c["agent"])
        latest[c["agent"]] = c["artifact"]
    lines = [f"- {agent}: {latest[agent]}" for agent in order]
    return "Rapport final assemble (style unifie) :\n" + "\n".join(lines)


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- Supervisor + swarm hybrid: loop guard + final assembler")
    print("=" * 70)

    # Nominal run: planner -> researcher -> coder -> reviewer -> coder -> reviewer
    result = run_hybrid("ship feature X", GOOD_AGENTS)
    seq = result["control_seq"]
    print(f"  control sequence: {seq}")
    assert seq == ["planner", "researcher", "coder", "reviewer", "coder", "reviewer"], seq
    # Exactly one correction pass; the legit loop did NOT trip the guard.
    assert seq.count("coder") == 2 and seq.count("reviewer") == 2

    final = assemble_final(result["contributions"])
    print("  --- final report ---")
    print("  " + final.replace("\n", "\n  "))
    distinct = {c["agent"] for c in result["contributions"]}
    assert len(distinct) >= 3, distinct
    assert distinct == {"planner", "researcher", "coder", "reviewer"}
    # Dedup by agent: 4 contributors -> 4 lines, code shows the FIXED version.
    assert final.count("\n- ") == 4
    assert "code v2 (fixed)" in final and "review v2: LGTM" in final

    # Degenerate run: infinite coder<->reviewer bounce must raise, not hang.
    try:
        run_hybrid("loop forever", BAD_AGENTS)
        assert False, "expected RuntimeError on tight loop"
    except RuntimeError as e:
        assert "tight loop" in str(e) or "hop budget" in str(e)
    print("  degenerate run -> guard raised cleanly")

    # Insufficient coverage: only 2 distinct contributors -> ValueError.
    too_few = [{"agent": "planner", "artifact": "p"},
               {"agent": "coder", "artifact": "c"}]
    try:
        assemble_final(too_few)
        assert False, "expected ValueError on <3 contributors"
    except ValueError as e:
        assert "distinct contributors" in str(e)
    print("[Verification] PASS -- hybrid sequence, loop guard, >=3 contributors")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 9 HARD Solutions -- Multi-agent patterns")
    print(f"  (langgraph available: {HAS_LANGGRAPH} -- running offline either way)")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

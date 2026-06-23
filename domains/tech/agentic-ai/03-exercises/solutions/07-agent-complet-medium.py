"""
Solutions -- Day 7 (MEDIUM): Complete research agent

Contains solutions for:
  - Medium Ex 1: Dynamic replan on step failure (bounded, no infinite loop)
  - Medium Ex 2: Long-term cache shared across questions in a session
  - Medium Ex 3: Synthesizer with source citations + citation verifier

Self-contained and OFFLINE: the tools are real local mocks (a keyword search
index and a doc store), all "LLM" decisions are scripted/deterministic. No API
key, no network. Same convention as the other medium/hard solution files.

Run:  python3 03-exercises/solutions/07-agent-complet-medium.py
Each solution ends with assertions (self-test).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ==========================================================================
# SHARED -- mock tools (the only "world" the agent can touch)
# ==========================================================================

# A search engine with a small pre-filled index. Some facts are ONLY in docs
# (not in the search index) so the replan/fallback path is exercised.
SEARCH_INDEX = {
    "africa area": "Africa covers approximately 30,370,000 km2 of land area.",
    "africa population": "Africa's population is about 1,460,000,000 inhabitants in 2024.",
    "paris population": "Paris has 2,161,000 inhabitants in 2024.",
    # NOTE: "paris area" is intentionally absent from the search index.
}

DOCS = {
    "paris_report.pdf": "FULL REPORT: Paris spans an area of 105 km2.",
    # "mars_report.pdf" intentionally does not exist -> total failure path.
}


def mock_web_search(query: str) -> str | None:
    """Keyword search. Returns a snippet, or None if nothing matches."""
    q = query.lower()
    for key, snippet in SEARCH_INDEX.items():
        if all(word in q for word in key.split()):
            return snippet
    return None


def read_doc(doc_name: str) -> str | None:
    """Fake PDF reader. Returns canned text or None if the doc is missing."""
    return DOCS.get(doc_name)


def extract_number_with_unit(text: str) -> tuple[float, str] | None:
    """Pull the first 'NUMBER unit' pair out of a snippet."""
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(km2|inhabitants)", text)
    if not m:
        return None
    return float(m.group(1).replace(",", "")), m.group(2)


# ==========================================================================
# MEDIUM EXERCISE 1 -- Dynamic replan on step failure
# ==========================================================================
#
# Static plan first; if a step fails (stuck), route to a replanner that
# swaps the failed search for a doc-read fallback and retries. A hard cap of
# max_replans guarantees termination even when the info exists nowhere.

@dataclass
class ReplanResult:
    answer: str
    verdict: str            # "ok" | "failed"
    replan_count: int
    short_term: dict = field(default_factory=dict)


def _default_plan(entity: str, want_density: bool) -> list[str]:
    if want_density:
        return [f"search:{entity} area", f"search:{entity} population",
                "compute:density", "format"]
    return [f"search:{entity} population", "format"]


def run_agent_with_replan(question: str, max_replans: int = 2,
                          verbose: bool = True) -> ReplanResult:
    q = question.lower()
    entity = "africa" if ("africa" in q or "afrique" in q) else "paris" if "paris" in q \
        else "mars"
    want_density = "density" in q or "densite" in q
    plan = _default_plan(entity, want_density)

    short_term: dict = {}
    replan_count = 0

    idx = 0
    while idx < len(plan):
        step = plan[idx]
        action, _, arg = step.partition(":")

        if action == "search":
            result = mock_web_search(arg)
            if result is None:
                # --- step failed: route to the replanner ------------------
                if replan_count >= max_replans:
                    if verbose:
                        print(f"  [replanner] cap reached ({max_replans}) -> giving up")
                    return ReplanResult("", "failed", replan_count, short_term)
                replan_count += 1
                doc = f"{arg.split()[0]}_report.pdf"
                if verbose:
                    print(f"  [replanner] '{arg}' failed -> swap step for read_doc:{doc}")
                # Build a NEW plan: replace the failed search by a doc read.
                plan = plan[:idx] + [f"read_doc:{doc}", *plan[idx + 1:]]
                continue  # retry the same index, now a read_doc step

            parsed = extract_number_with_unit(result)
            if parsed:
                short_term[_fact_key(arg, parsed[1])] = parsed[0]
            if verbose:
                print(f"  [search] {arg} -> {result[:50]}...")

        elif action == "read_doc":
            text = read_doc(arg)
            if text is None:
                if replan_count >= max_replans:
                    if verbose:
                        print(f"  [replanner] doc {arg} missing, cap reached -> give up")
                    return ReplanResult("", "failed", replan_count, short_term)
                replan_count += 1
                if verbose:
                    print(f"  [replanner] doc {arg} missing -> no other source -> give up next")
                # No further fallback exists: force a final give-up on next miss.
                plan = plan[:idx] + [f"read_doc:{arg}", *plan[idx + 1:]]
                # Make the cap bite immediately if it was the last allowance.
                if replan_count >= max_replans:
                    return ReplanResult("", "failed", replan_count, short_term)
                continue
            parsed = extract_number_with_unit(text)
            if parsed:
                # doc names like "paris_report.pdf" -> entity from prefix
                ent = arg.split("_")[0]
                short_term[_fact_key(f"{ent} {parsed[1]}", parsed[1])] = parsed[0]
            if verbose:
                print(f"  [read_doc] {arg} -> {text[:50]}...")

        elif action == "compute":
            area = short_term.get(f"{entity}_area_km2")
            pop = short_term.get(f"{entity}_population_inhabitants")
            if area and pop and area > 0:
                short_term[f"{entity}_density"] = round(pop / area, 2)

        idx += 1

    answer = _format_answer(entity, short_term, want_density)
    return ReplanResult(answer, "ok", replan_count, short_term)


def _fact_key(arg: str, unit: str) -> str:
    entity = arg.split()[0]
    unit_name = "area_km2" if unit == "km2" else "population_inhabitants"
    return f"{entity}_{unit_name}"


def _format_answer(entity: str, st: dict, want_density: bool) -> str:
    if want_density and f"{entity}_density" in st:
        return f"{entity.title()} density ~ {st[f'{entity}_density']} hab/km2"
    pop = st.get(f"{entity}_population_inhabitants")
    if pop:
        return f"{entity.title()} population ~ {int(pop):,} inhabitants"
    return "no answer"


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Dynamic replan on step failure")
    print("=" * 70)

    # Scenario A: passes first try (Africa: both facts in the search index)
    print("\n  Scenario A: Africa density (no failure)")
    a = run_agent_with_replan("What is the population density of Africa?")
    print(f"    -> verdict={a.verdict} replans={a.replan_count} answer={a.answer!r}")
    assert a.verdict == "ok", a
    assert a.replan_count == 0, a
    assert "48" in a.answer  # 1.46e9 / 30.37e6 = 48.07

    # Scenario B: 'paris area' missing from search -> 1 replan -> doc read OK
    print("\n  Scenario B: Paris density (search miss -> doc fallback)")
    b = run_agent_with_replan("What is the population density of Paris?")
    print(f"    -> verdict={b.verdict} replans={b.replan_count} answer={b.answer!r}")
    assert b.verdict == "ok", b
    assert b.replan_count == 1, b              # exactly one replan happened
    assert "density" in b.answer.lower() or "hab/km2" in b.answer

    # Scenario C: Mars: nothing anywhere -> bounded give-up
    print("\n  Scenario C: Mars population (info exists nowhere)")
    c = run_agent_with_replan("What is the population of Mars?", max_replans=2)
    print(f"    -> verdict={c.verdict} replans={c.replan_count}")
    assert c.verdict == "failed", c
    assert c.replan_count <= 2, c             # never exceeds the cap

    print("\n[Verification] PASS -- 0/1/2 replans, bounded, clean give-up")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Long-term cache shared across a session
# ==========================================================================
#
# A ResearchSession keeps long-term facts between questions. Before any real
# tool call, the agent checks the cache -> cache hits avoid network calls.

@dataclass
class QuestionStats:
    real_tool_calls: int = 0
    cache_hits: int = 0


class ResearchSession:
    """Persists long-term facts across successive ask() calls."""

    def __init__(self) -> None:
        # long_term: canonical_key -> {"value", "unit", "source"}
        self.long_term: dict[str, dict] = {}

    def _lookup(self, key: str) -> dict | None:
        return self.long_term.get(key)

    def _store(self, key: str, value: float, unit: str, source: str) -> None:
        # No duplicates: storing the same key twice just overwrites in place.
        self.long_term[key] = {"value": value, "unit": unit, "source": source}

    def ask(self, question: str, verbose: bool = True) -> tuple[str, QuestionStats]:
        q = question.lower()
        entity = "africa" if "africa" in q else "paris"
        want_density = "density" in q
        stats = QuestionStats()

        needed: list[tuple[str, str]] = []  # (canonical_key, search query)
        if want_density:
            needed = [(f"{entity}_area_km2", f"{entity} area"),
                      (f"{entity}_population_inhabitants", f"{entity} population")]
        elif "area" in q:
            needed = [(f"{entity}_area_km2", f"{entity} area")]
        else:
            needed = [(f"{entity}_population_inhabitants", f"{entity} population")]

        for key, search_query in needed:
            cached = self._lookup(key)
            if cached is not None:
                stats.cache_hits += 1
                if verbose:
                    print(f"    [cache HIT] {key} = {cached['value']}")
                continue
            # Cache miss -> real tool call
            result = mock_web_search(search_query)
            stats.real_tool_calls += 1
            if result is None:
                continue
            parsed = extract_number_with_unit(result)
            if parsed:
                self._store(key, parsed[0], parsed[1], f"mock_web_search('{search_query}')")
                if verbose:
                    print(f"    [tool CALL] {key} = {parsed[0]} (cached for later)")

        # Build the answer purely from the (now-populated) long-term cache.
        if want_density:
            area = self.long_term.get(f"{entity}_area_km2", {}).get("value")
            pop = self.long_term.get(f"{entity}_population_inhabitants", {}).get("value")
            if area and pop and area > 0:
                answer = f"{entity.title()} density ~ {round(pop / area, 2)} hab/km2"
            else:
                answer = "insufficient data"
        elif "area" in q:
            v = self.long_term.get(f"{entity}_area_km2", {}).get("value")
            answer = f"{entity.title()} area = {int(v):,} km2" if v else "no data"
        else:
            v = self.long_term.get(f"{entity}_population_inhabitants", {}).get("value")
            answer = f"{entity.title()} population = {int(v):,}" if v else "no data"
        return answer, stats


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Long-term cache shared across a session")
    print("=" * 70)

    session = ResearchSession()
    questions = [
        "What is the population of Africa?",
        "What is the area of Africa?",
        "What is the population density of Africa?",
    ]
    per_question: list[QuestionStats] = []
    for question in questions:
        print(f"\n  Q: {question}")
        answer, stats = session.ask(question)
        per_question.append(stats)
        print(f"    answer={answer!r}")
        print(f"    real_calls={stats.real_tool_calls} cache_hits={stats.cache_hits} "
              f"long_term_size={len(session.long_term)}")

    # Q1 fetches population (1 real call, 0 hits)
    assert per_question[0].real_tool_calls == 1 and per_question[0].cache_hits == 0
    # Q2 fetches area (1 real call, 0 hits)
    assert per_question[1].real_tool_calls == 1 and per_question[1].cache_hits == 0
    # Q3 (density) reuses BOTH facts -> 2 cache hits, 0 real calls
    assert per_question[2].cache_hits == 2, per_question[2]
    assert per_question[2].real_tool_calls == 0, per_question[2]
    # long_term stabilized at exactly 2 distinct facts (no duplicates)
    assert len(session.long_term) == 2, session.long_term

    print("\n[Verification] PASS -- 3rd question fully served from cache (2 hits, 0 calls)")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Synthesizer with source citations
# ==========================================================================
#
# Findings carry (tool, source). The analyzer threads the source into
# short_term; the synthesizer cites every numeric fact it uses.

@dataclass
class Finding:
    text: str
    tool: str       # "mock_web_search" | "read_doc"
    source: str     # the query string or doc name


def gather_findings(entity: str) -> list[Finding]:
    """Collect findings keeping provenance. Africa area comes from search,
    population from a doc -> two different tools/sources to cite."""
    findings: list[Finding] = []
    area_snip = mock_web_search(f"{entity} area")
    if area_snip:
        findings.append(Finding(area_snip, "mock_web_search", f"{entity} area"))
    # Force the population to come from a doc to exercise dual-source citing.
    DOCS["africa_report_2024.pdf"] = ("FULL REPORT: Africa has a population of "
                                      "1,460,000,000 inhabitants and 30,370,000 km2.")
    doc_text = read_doc("africa_report_2024.pdf")
    if doc_text:
        findings.append(Finding(doc_text, "read_doc", "africa_report_2024.pdf"))
    return findings


def analyze_with_sources(findings: list[Finding]) -> dict:
    """short_term[key] = {"value", "unit", "tool", "source"}."""
    short_term: dict[str, dict] = {}
    for f in findings:
        parsed = extract_number_with_unit(f.text)
        if not parsed:
            continue
        value, unit = parsed
        key = "area_km2" if unit == "km2" else "population"
        # Keep the first source per key (don't let the doc overwrite search).
        short_term.setdefault(key, {"value": value, "unit": unit,
                                    "tool": f.tool, "source": f.source})
    return short_term


def synthesize_with_citations(question: str, short_term: dict) -> str:
    st = dict(short_term)
    want_density = "density" in question.lower()
    if want_density and "area_km2" in st and "population" in st \
            and st["area_km2"]["value"] > 0:
        density = round(st["population"]["value"] / st["area_km2"]["value"], 2)
    else:
        density = None

    lines: list[str] = []
    if density is not None:
        lines.append(f"The population density of Africa is approximately {density} hab/km2.")
    for key, fact in st.items():
        unit = "km2" if key == "area_km2" else "inhabitants"
        lines.append(f"  - {key}: {int(fact['value']):,} {unit} "
                     f"(source: {fact['tool']} {fact['source']})")
    return "\n".join(lines)


def verify_citations(answer: str, short_term: dict) -> bool:
    """Every numeric fact in short_term must appear cited in the answer."""
    for fact in short_term.values():
        formatted = f"{int(fact['value']):,}"
        if formatted not in answer or fact["source"] not in answer:
            return False
    return True


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Synthesizer with source citations")
    print("=" * 70)

    findings = gather_findings("africa")
    short_term = analyze_with_sources(findings)
    answer = synthesize_with_citations("What is the population density of Africa?",
                                       short_term)
    print(answer)

    # Two distinct facts, each from a different tool
    assert set(short_term) == {"area_km2", "population"}, short_term
    tools_used = {f["tool"] for f in short_term.values()}
    assert tools_used == {"mock_web_search", "read_doc"}, tools_used
    # The answer cites both sources and both numbers
    assert verify_citations(answer, short_term), "every fact must be cited"
    assert "mock_web_search" in answer and "read_doc" in answer

    # Negative case: drop a citation -> verifier must catch it
    broken = answer.replace("read_doc africa_report_2024.pdf", "")
    assert not verify_citations(broken, short_term), "must detect missing citation"

    print("\n[Verification] PASS -- dual-source citations verified, negative case caught")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 7 MEDIUM Solutions -- Complete research agent")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

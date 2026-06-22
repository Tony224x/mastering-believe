"""
Solutions -- Day 7 (HARD): Complete research agent

Contains solutions for:
  - Hard Ex 1: Parallel executor with fan-out / fan-in (topological waves,
               ThreadPoolExecutor, correctness + speedup, cycle detection)
  - Hard Ex 2: Self-correcting agent with plausibility checks + re-extraction

Self-contained and OFFLINE: real local mock tools, scripted decisions, no API
key, no network. Same convention as the other hard solution files.

Run:  python3 03-exercises/solutions/07-agent-complet-hard.py
Each solution ends with assertions (self-test).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# ==========================================================================
# SHARED -- mock tools (I/O-bound: a small sleep makes parallelism visible)
# ==========================================================================

SEARCH_INDEX = {
    "africa area": (30_370_000, "km2"),
    "africa population": (1_460_000_000, "inhabitants"),
    "africa gdp": (2_980_000_000_000, "usd"),
}

TOOL_LATENCY_S = 0.05  # simulated I/O latency per tool call


def mock_web_search(query: str) -> tuple[float, str] | None:
    time.sleep(TOOL_LATENCY_S)  # simulate network round-trip
    q = query.lower()
    for key, value in SEARCH_INDEX.items():
        if all(word in q for word in key.split()):
            return value
    return None


# ==========================================================================
# HARD EXERCISE 1 -- Parallel executor with fan-out / fan-in
# ==========================================================================
#
# The plan carries a dependency graph. We compute topological "waves" (levels)
# and run each wave concurrently. Independent searches (area, population) land
# in wave 0 and run in parallel; compute/format depend on them.

@dataclass
class Step:
    id: str
    action: str          # "search:africa area" | "compute:density" | "format"
    depends_on: list[str] = field(default_factory=list)


def topological_waves(steps: list[Step]) -> list[list[str]]:
    """Group step ids into waves; raise on cycles. Each wave is independent."""
    by_id = {s.id: s for s in steps}
    remaining = set(by_id)
    done: set[str] = set()
    waves: list[list[str]] = []
    while remaining:
        ready = [sid for sid in remaining
                 if all(dep in done for dep in by_id[sid].depends_on)]
        if not ready:
            raise ValueError(f"Dependency cycle detected among: {sorted(remaining)}")
        ready.sort()  # deterministic ordering inside a wave
        waves.append(ready)
        done |= set(ready)
        remaining -= set(ready)
    return waves


def _run_step(step: Step, results: dict) -> tuple[str, object]:
    """Execute one step; reads upstream results from the shared dict."""
    action, _, arg = step.action.partition(":")
    if action == "search":
        return step.id, mock_web_search(arg)
    if action == "compute":  # density = population / area
        area = next((v[0] for v in results.values()
                     if isinstance(v, tuple) and v[1] == "km2"), None)
        pop = next((v[0] for v in results.values()
                    if isinstance(v, tuple) and v[1] == "inhabitants"), None)
        if area and pop and area > 0:
            return step.id, round(pop / area, 2)
        return step.id, None
    if action == "format":
        density = next((v for k, v in results.items()
                        if isinstance(v, float)), None)
        return step.id, f"Africa density ~ {density} hab/km2"
    return step.id, None


def execute_parallel(steps: list[Step]) -> tuple[dict, float]:
    """Run the plan wave by wave, parallel inside each wave."""
    waves = topological_waves(steps)
    by_id = {s.id: s for s in steps}
    results: dict[str, object] = {}
    start = time.perf_counter()
    for wave in waves:
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [pool.submit(_run_step, by_id[sid], dict(results)) for sid in wave]
            for fut in futures:
                sid, value = fut.result()
                results[sid] = value
    return results, time.perf_counter() - start


def execute_sequential(steps: list[Step]) -> tuple[dict, float]:
    """Reference sequential execution for correctness + timing comparison."""
    results: dict[str, object] = {}
    start = time.perf_counter()
    for s in steps:               # plan is already in dependency order here
        sid, value = _run_step(s, dict(results))
        results[sid] = value
    return results, time.perf_counter() - start


def solve_hard_1() -> None:
    print("\n" + "=" * 70)
    print("HARD 1 -- Parallel executor with fan-out / fan-in")
    print("=" * 70)

    plan = [
        Step("s1", "search:africa area", []),
        Step("s2", "search:africa population", []),
        Step("s3", "compute:density", ["s1", "s2"]),
        Step("s4", "format", ["s3"]),
    ]

    waves = topological_waves(plan)
    print(f"  waves = {waves}")
    assert waves == [["s1", "s2"], ["s3"], ["s4"]], waves

    par_results, par_time = execute_parallel(plan)
    seq_results, seq_time = execute_sequential(plan)
    print(f"  parallel result  : {par_results['s4']}  ({par_time * 1000:.0f} ms)")
    print(f"  sequential result: {seq_results['s4']}  ({seq_time * 1000:.0f} ms)")

    # Correctness: identical final answers
    assert par_results["s4"] == seq_results["s4"], (par_results, seq_results)
    assert par_results["s3"] == 48.07, par_results["s3"]

    # Speedup: parallel (~3 sequential waves of latency) < sequential (~4 calls)
    assert par_time < seq_time, (par_time, seq_time)

    # Generalizability: adding a 3rd independent search joins wave 0 automatically
    plan2 = plan[:2] + [Step("s5", "search:africa gdp", [])] + plan[2:]
    waves2 = topological_waves(plan2)
    assert set(waves2[0]) == {"s1", "s2", "s5"}, waves2

    # Cycle detection
    cyclic = [Step("a", "search:x", ["b"]), Step("b", "search:y", ["a"])]
    try:
        topological_waves(cyclic)
        raise AssertionError("cycle should have raised")
    except ValueError as e:
        print(f"  cycle correctly rejected: {e}")

    print("\n[Verification] PASS -- same result, real speedup, extensible, cycle caught")


# ==========================================================================
# HARD EXERCISE 2 -- Self-correcting agent with plausibility checks
# ==========================================================================
#
# Facts are validated against declarative ranges. An implausible value triggers
# a re-extraction via a more reliable source; derivations are checked for
# internal consistency; unfixable facts are flagged "unreliable" (no hallucination).

PLAUSIBILITY_RULES = {
    "area_km2":   {"min": 1, "max": 200_000_000},
    "population": {"min": 1, "max": 10_000_000_000},
    "density":    {"min": 0.01, "max": 50_000},
}

# Two sources per fact: a noisy "snippet" (may be wrong) and a reliable "doc".
NOISY_SOURCE = {
    "africa_area_km2": 0.3,             # WRONG: regex grabbed "0.3" (below min -> implausible)
    "africa_population": 1_460_000_000,  # correct
}
RELIABLE_SOURCE = {
    "africa_area_km2": 30_370_000,       # correct value from the detailed report
    "africa_population": 1_460_000_000,
    # "mars_*" absent -> unfixable -> flagged unreliable
}


@dataclass
class FactResult:
    value: float | None
    status: str          # "ok" | "unreliable"
    corrections: int = 0


def _plausible(fact_type: str, value: float | None) -> bool:
    if value is None:
        return False
    rule = PLAUSIBILITY_RULES[fact_type]
    return rule["min"] <= value <= rule["max"]


def extract_fact(entity: str, fact_type: str, verbose: bool = True) -> FactResult:
    """Try the noisy source; if implausible, re-extract from the reliable one."""
    key = f"{entity}_{fact_type}"
    corrections = 0
    value = NOISY_SOURCE.get(key)
    if not _plausible(fact_type, value):
        if verbose:
            print(f"    [check] {key}={value} implausible -> re-extract from reliable source")
        corrections += 1
        value = RELIABLE_SOURCE.get(key)
        if not _plausible(fact_type, value):
            if verbose:
                print(f"    [check] {key} still implausible after retry -> unreliable")
            return FactResult(None, "unreliable", corrections)
    return FactResult(value, "ok", corrections)


def derive_density(area: FactResult, pop: FactResult,
                   verbose: bool = True) -> FactResult:
    """Compute density and verify it is consistent with its own inputs (+/-5%)."""
    if area.status != "ok" or pop.status != "ok" or not area.value:
        return FactResult(None, "unreliable", 0)
    density = round(pop.value / area.value, 2)
    if not _plausible("density", density):
        return FactResult(None, "unreliable", 0)
    # Internal consistency check: density * area should reproduce population.
    reconstructed = density * area.value
    if abs(reconstructed - pop.value) / pop.value > 0.05:
        if verbose:
            print("    [check] density inconsistent with inputs -> recompute")
        density = round(pop.value / area.value, 2)  # recompute cleanly
    return FactResult(density, "ok", 0)


def research_with_correction(entity: str, verbose: bool = True) -> dict:
    area = extract_fact(entity, "area_km2", verbose)
    pop = extract_fact(entity, "population", verbose)
    density = derive_density(area, pop, verbose)
    corrections = area.corrections + pop.corrections + density.corrections
    return {"area": area, "population": pop, "density": density,
            "corrections": corrections}


def render_answer(entity: str, facts: dict) -> str:
    parts = [f"Report for {entity.title()}:"]
    for name in ("area", "population", "density"):
        f = facts[name]
        if f.status == "ok":
            parts.append(f"  {name}: {f.value} (reliable)")
        else:
            parts.append(f"  {name}: UNRELIABLE (no plausible source found)")
    return "\n".join(parts)


def solve_hard_2() -> None:
    print("\n" + "=" * 70)
    print("HARD 2 -- Self-correcting agent with plausibility checks")
    print("=" * 70)

    # Scenario A: nominal -- but the noisy area IS wrong, so 1 correction.
    print("\n  Scenario A: Africa (noisy area is wrong -> auto-corrected)")
    facts_a = research_with_correction("africa")
    print(render_answer("africa", facts_a))
    assert facts_a["area"].status == "ok"
    assert facts_a["area"].value == 30_370_000          # corrected to the real value
    assert facts_a["density"].status == "ok"
    assert facts_a["density"].value == 48.07, facts_a["density"].value
    assert facts_a["corrections"] == 1, facts_a["corrections"]

    # Scenario B: make the population also wrong AND reliable source missing
    print("\n  Scenario B: implausible everywhere -> unreliable, no hallucination")
    NOISY_SOURCE["mars_area_km2"] = 0           # implausible (min is 1)
    NOISY_SOURCE["mars_population"] = -5         # implausible
    facts_b = research_with_correction("mars")
    print(render_answer("mars", facts_b))
    assert facts_b["area"].status == "unreliable", facts_b["area"]
    assert facts_b["population"].status == "unreliable", facts_b["population"]
    assert facts_b["density"].status == "unreliable", facts_b["density"]
    # corrections were attempted (>=1) but no fabricated value is returned
    assert facts_b["corrections"] >= 1
    assert facts_b["density"].value is None

    print("\n[Verification] PASS -- aberrant facts corrected; unfixable flagged, not faked")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 7 HARD Solutions -- Complete research agent")
    print("#" * 70)

    solve_hard_1()
    solve_hard_2()

    print("\n" + "#" * 70)
    print("  All hard solutions executed successfully.")
    print("#" * 70)

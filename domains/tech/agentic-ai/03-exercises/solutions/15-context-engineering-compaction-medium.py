"""
Solutions -- Day 15 (MEDIUM): Context engineering & compaction

Contains solutions for:
  - Medium Ex 1: Hierarchical (thematic-block) compaction -- groups messages by
                 phase, summarizes one block per phase, keeps pinned messages.
  - Medium Ex 2: OffloadingManager -- auto-offloads any tool result above a
                 token threshold to a VirtualFS, keeps in-context tokens bounded.
  - Medium Ex 3: Proactive vs reactive compaction harness -- measures
                 compactions / overflows / peak tokens across the same run.

Self-contained: embeds a heuristic token estimator and a tiny VirtualFS, so the
file RUNS OFFLINE with zero dependencies (no langgraph/numpy, no API key).
Each solution ends with assertions (self-test) and prints PASS.

Run:  python 03-exercises/solutions/15-context-engineering-compaction-medium.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ==========================================================================
# Shared helpers (embedded so the file is self-contained / offline)
# ==========================================================================

def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~1 token per 4 chars (OpenAI rule of thumb)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class VirtualFS:
    """In-memory filesystem used as external agent memory (offload target)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        self._store[path] = content

    def read(self, path: str) -> Optional[str]:
        return self._store.get(path)

    def list_files(self) -> list[str]:
        return sorted(self._store.keys())

    def total_tokens(self) -> int:
        return sum(estimate_tokens(c) for c in self._store.values())


# ==========================================================================
# MEDIUM EXERCISE 1 -- Hierarchical (thematic-block) compaction
# ==========================================================================

def group_by_phase(messages: list[dict]) -> dict[str, list[dict]]:
    """
    Group messages by their "phase" tag, PRESERVING first-appearance order.

    A plain dict in Python 3.7+ keeps insertion order, so we just insert phases
    the first time we see them -> the resulting key order = order of appearance.
    """
    grouped: dict[str, list[dict]] = {}
    for m in messages:
        grouped.setdefault(m["phase"], []).append(m)
    return grouped


def summarize_block(phase: str, msgs: list[dict]) -> dict:
    """Condense one phase's messages into a single system summary message."""
    last = msgs[-1]["content"] if msgs else ""
    content = (
        f"[SUMMARY phase={phase}] {len(msgs)} messages condensed. "
        f"Last item: {last[:80]}"
    )
    return {"role": "system", "content": content, "phase": phase, "pinned": False}


def hierarchical_compact(messages: list[dict]) -> list[dict]:
    """
    Hierarchical compaction:
      1. Extract all pinned messages (never summarized), in original order.
      2. Summarize each non-pinned phase block into ONE message.
      3. Return [pinned...] + [one summary per non-pinned phase...].
    """
    pinned = [m for m in messages if m.get("pinned")]
    non_pinned = [m for m in messages if not m.get("pinned")]

    grouped = group_by_phase(non_pinned)  # phase order preserved
    summaries = [summarize_block(phase, msgs) for phase, msgs in grouped.items()]

    return pinned + summaries


def _total_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def solve_medium_1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Hierarchical (thematic-block) compaction")
    print("=" * 70)

    def msg(role, content, phase, pinned=False):
        return {"role": role, "content": content, "phase": phase, "pinned": pinned}

    messages = [
        # pinned goal + pinned constraint
        msg("user", "goal: audit the Flask repo for security vulnerabilities",
            "exploration", pinned=True),
        msg("system", "[CONSTRAINT] never run eval(); everything must be type-safe",
            "exploration", pinned=True),
        # exploration phase
        msg("assistant", "Listing the directory structure to map the project.",
            "exploration"),
        msg("tool", "Found app.py, models.py, routes.py, templates/, tests/.",
            "exploration"),
        msg("assistant", "Reading app.py as the main entry point.", "exploration"),
        msg("tool", "app.py: Flask init, SECRET_KEY from env, debug=False.",
            "exploration"),
        # implementation phase
        msg("assistant", "Adding input validation to the two unguarded routes.",
            "implementation"),
        msg("tool", "Patched routes.py: request.args now validated via a schema.",
            "implementation"),
        msg("assistant", "Refactoring the raw SQL query into a parameterized one.",
            "implementation"),
        msg("tool", "models.py: replaced string-format query with bound params.",
            "implementation"),
        # tests phase
        msg("assistant", "Writing regression tests for the patched routes.",
            "tests"),
        msg("tool", "pytest: 14 passed, 0 failed. Coverage on routes.py at 92%.",
            "tests"),
        msg("assistant", "All checks green; summarizing findings for the report.",
            "tests"),
    ]

    grouped = group_by_phase([m for m in messages if not m["pinned"]])
    phase_order = list(grouped.keys())
    print(f"\n  Phases (first-appearance order): {phase_order}")

    tokens_before = _total_tokens(messages)
    compacted = hierarchical_compact(messages)
    tokens_after = _total_tokens(compacted)

    print(f"  Messages: {len(messages)} -> {len(compacted)}")
    print(f"  Tokens:   {tokens_before} -> {tokens_after}")
    for m in compacted:
        tag = " (pinned)" if m.get("pinned") else ""
        print(f"    [{m['role']:9}|{m['phase']:14}]{tag} {m['content'][:50]}")

    # --- Assertions ---
    # Phase order = first-appearance, not alphabetical.
    assert phase_order == ["exploration", "implementation", "tests"], phase_order

    # All pinned messages survive unchanged.
    pinned_in = [m for m in messages if m["pinned"]]
    pinned_out = [m for m in compacted if m.get("pinned")]
    assert pinned_out == pinned_in, "pinned messages must survive unchanged"

    # Exactly one summary per distinct non-pinned phase.
    summaries = [m for m in compacted if not m.get("pinned")]
    assert len(summaries) == len(phase_order), (len(summaries), len(phase_order))

    # Final count = pinned + distinct phases.
    assert len(compacted) == len(pinned_in) + len(phase_order)

    # Token count strictly drops.
    assert tokens_after < tokens_before, (tokens_after, tokens_before)

    print("[Verification] PASS -- phase order kept, pinned survive, tokens dropped")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Automatic offloading to a VirtualFS
# ==========================================================================

class OffloadingManager:
    """
    Auto-offloads any tool result above a token threshold to a VirtualFS,
    replacing it in-context with a short placeholder. Small results stay inline.
    """

    PLACEHOLDER_TOKEN_CAP = 40  # placeholders must stay tiny

    def __init__(self, vfs: VirtualFS, token_threshold: int = 200) -> None:
        self.vfs = vfs
        self.token_threshold = token_threshold
        self._context: list[str] = []  # what is actually in the context window
        self._counter = 0

    def add_tool_result(self, tool_name: str, content: str) -> str:
        """Offload if large, else keep inline. Returns what entered the context."""
        if estimate_tokens(content) > self.token_threshold:
            self._counter += 1
            path = f"tool_results/{tool_name}_{self._counter}.txt"
            self.vfs.write(path, content)
            placeholder = (
                f"[OFFLOADED -> {path} ({estimate_tokens(content)} tokens on disk)]"
            )
            self._context.append(placeholder)
            return placeholder
        # Small enough: keep inline.
        self._context.append(content)
        return content

    def context_tokens(self) -> int:
        """Tokens currently held IN the context (placeholders + small results)."""
        return sum(estimate_tokens(c) for c in self._context)

    def retrieve(self, path: str) -> Optional[str]:
        """Pull an offloaded result back from disk on demand."""
        return self.vfs.read(path)


def solve_medium_2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Automatic offloading to a VirtualFS")
    print("=" * 70)

    vfs = VirtualFS()
    mgr = OffloadingManager(vfs, token_threshold=200)

    # 3 huge results, 2 small results, 1 medium-but-under-threshold.
    huge_html = "<div>" + ("scraped row of data; " * 2000) + "</div>"   # ~ 40k chars
    huge_json = '{"items": [' + ("{'id': 1, 'val': 'x'}, " * 1500) + "]}"
    huge_logs = "\n".join(f"2026-06-16 INFO line {i} processed" for i in range(2000))
    small_ok = "Tool returned OK."
    small_count = "Found 3 issues."
    medium = "summary: " + ("note. " * 20)  # under threshold -> inline

    results = [
        ("scrape_page", huge_html),
        ("query_db", huge_json),
        ("read_logs", huge_logs),
        ("ping", small_ok),
        ("count_issues", small_count),
        ("brief", medium),
    ]

    on_disk_paths: dict[str, str] = {}   # tool -> path, for the big ones
    print()
    for tool, content in results:
        entered = mgr.add_tool_result(tool, content)
        offloaded = entered.startswith("[OFFLOADED")
        if offloaded:
            # path is between "-> " and " ("
            path = entered.split("-> ", 1)[1].split(" (", 1)[0]
            on_disk_paths[tool] = path
        kind = "OFFLOAD" if offloaded else "inline "
        print(f"  {kind} {tool:14} content_tokens={estimate_tokens(content):6} "
              f"-> in-context={estimate_tokens(entered)}")

    print(f"\n  Context tokens (in window):  {mgr.context_tokens()}")
    print(f"  VFS tokens (on disk):        {vfs.total_tokens()}")
    print(f"  VFS files: {vfs.list_files()}")

    # --- Assertions ---
    # Context stays bounded despite tens of thousands of tokens on disk.
    assert vfs.total_tokens() > 10_000, "scenario should store a lot on disk"
    assert mgr.context_tokens() < 500, mgr.context_tokens()

    # Each huge result is retrievable byte-for-byte (round-trip).
    assert mgr.retrieve(on_disk_paths["scrape_page"]) == huge_html
    assert mgr.retrieve(on_disk_paths["query_db"]) == huge_json
    assert mgr.retrieve(on_disk_paths["read_logs"]) == huge_logs

    # Small results stayed inline (not offloaded, not in the VFS).
    assert small_ok in mgr._context and small_count in mgr._context
    assert medium in mgr._context

    # Placeholders stay tiny.
    for entry in mgr._context:
        if entry.startswith("[OFFLOADED"):
            assert estimate_tokens(entry) <= mgr.PLACEHOLDER_TOKEN_CAP, entry

    print("[Verification] PASS -- big results offloaded & retrievable, context bounded")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Proactive vs reactive compaction harness
# ==========================================================================

@dataclass
class CompactionRun:
    compactions: int
    overflows: int
    peak_tokens: int
    final_tokens: int

    def as_dict(self) -> dict:
        return {
            "compactions": self.compactions,
            "overflows": self.overflows,
            "peak_tokens": self.peak_tokens,
            "final_tokens": self.final_tokens,
        }


def run_reactive(turns: list[int], limit: int, keep_tail: int,
                 summary_tokens: int) -> CompactionRun:
    """
    Reactive: add the turn, THEN compact if over limit.
    An overflow is counted whenever, right after adding (before compaction),
    tokens > limit -- the model already saw an oversized context that turn.
    """
    history: list[int] = []   # token cost of each turn currently in context
    compactions = 0
    overflows = 0
    peak = 0

    for cost in turns:
        history.append(cost)
        total = sum(history)
        peak = max(peak, total)
        if total > limit:
            overflows += 1                     # damage already done this turn
            # Compact: keep the last keep_tail turns + a fixed-size summary.
            tail = history[-keep_tail:] if keep_tail else []
            history = [summary_tokens] + tail
            compactions += 1

    return CompactionRun(compactions, overflows, peak, sum(history))


def run_proactive(turns: list[int], limit: int, keep_tail: int,
                  summary_tokens: int) -> CompactionRun:
    """
    Proactive: BEFORE adding the turn, if current + estimated next > limit,
    compact first, THEN add. An overflow is counted only if, despite the
    preventive compaction, the context still exceeds the limit after adding.
    """
    history: list[int] = []
    compactions = 0
    overflows = 0
    peak = 0

    for cost in turns:
        # Preventive check: would adding this turn overflow?
        if sum(history) + cost > limit:
            tail = history[-keep_tail:] if keep_tail else []
            history = [summary_tokens] + tail
            compactions += 1
        history.append(cost)
        total = sum(history)
        peak = max(peak, total)
        if total > limit:
            overflows += 1     # only if compaction wasn't enough (huge turn)

    return CompactionRun(compactions, overflows, peak, sum(history))


def solve_medium_3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Proactive vs reactive compaction harness")
    print("=" * 70)

    # Deterministic scenario: each ReAct turn adds a steady ~80-token tool
    # result. The context grows until it crosses the limit. Same scenario for
    # both strategies (fair comparison). With keep_tail=2 + summary=50, a
    # compaction always brings the context well under the limit (50+2*80=210),
    # so the only difference is WHEN each strategy compacts.
    turns = [80] * 14
    limit = 500
    keep_tail = 2
    summary_tokens = 50

    reactive = run_reactive(turns, limit, keep_tail, summary_tokens)
    proactive = run_proactive(turns, limit, keep_tail, summary_tokens)

    print(f"\n  Scenario: {len(turns)} turns, limit={limit}, "
          f"keep_tail={keep_tail}, summary={summary_tokens}")
    print(f"\n  {'strategy':<10} {'compactions':>11} {'overflows':>10} "
          f"{'peak':>6} {'final':>6}")
    print(f"  {'-'*10} {'-'*11} {'-'*10} {'-'*6} {'-'*6}")
    for name, r in [("reactive", reactive), ("proactive", proactive)]:
        d = r.as_dict()
        print(f"  {name:<10} {d['compactions']:>11} {d['overflows']:>10} "
              f"{d['peak_tokens']:>6} {d['final_tokens']:>6}")

    # --- Assertions ---
    # Reactive suffers at least one overflow on this scenario.
    assert reactive.overflows >= 1, reactive.overflows
    # Proactive never has MORE overflows, and strictly fewer here.
    assert proactive.overflows <= reactive.overflows
    assert proactive.overflows < reactive.overflows, (
        proactive.overflows, reactive.overflows)
    # Both end below the limit (viable context).
    assert reactive.final_tokens <= limit, reactive.final_tokens
    assert proactive.final_tokens <= limit, proactive.final_tokens
    # Proactive doesn't compact an absurd number of times.
    assert proactive.compactions <= len(turns), proactive.compactions

    print("[Verification] PASS -- proactive avoids overflows the reactive subjects to")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("#" * 70)
    print("  Day 15 MEDIUM Solutions -- Context engineering & compaction")
    print("  (stdlib only -- running fully offline)")
    print("#" * 70)

    solve_medium_1()
    solve_medium_2()
    solve_medium_3()

    print("\n" + "#" * 70)
    print("  All medium solutions executed successfully.")
    print("#" * 70)

"""
Day 16 -- Solutions to the easy exercises for long-horizon memory.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/16-memoire-long-horizon.py
"""

from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the Day 16 module from 02-code/
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# importlib is needed because the filename starts with a digit
mod = import_module("16-memoire-long-horizon")

EpisodicMemory = mod.EpisodicMemory
SemanticMemory = mod.SemanticMemory
HierarchicalMemory = mod.HierarchicalMemory
MemoryEntry = mod.MemoryEntry
RelevanceScorer = mod.RelevanceScorer
fake_embed = mod.fake_embed
cosine_similarity = mod.cosine_similarity
MAX_MAIN_CONTEXT = mod.MAX_MAIN_CONTEXT


# ===========================================================================
# SOLUTION 1 -- Scoring de pertinence personnalise
# ===========================================================================

def solution_1() -> None:
    print("\n=== Solution 1: Scoring de pertinence personnalise ===")

    now = time.time()
    one_hour = 3600

    # Four contrasting entries
    entries = [
        MemoryEntry(
            "payment error 503 tool failed",       # hors-sujet par rapport a la query
            "episodic", importance=0.2,
            created_at=now - 0.25 * one_hour,     # very recent (15 min)
        ),
        MemoryEntry(
            "user CSV report format preference important critical",  # tres similaire
            "semantic", importance=0.9,
            created_at=now - 7 * 24 * one_hour,   # 7 days ago
        ),
        MemoryEntry(
            "user requested sales report CSV format export",   # moderement similaire
            "episodic", importance=0.6,
            created_at=now - 24 * one_hour,        # 1 day ago
        ),
        MemoryEntry(
            "agent started session greeting hello",  # hors-sujet
            "episodic", importance=0.3,
            created_at=now - 10 * 60,              # 10 min ago
        ),
    ]
    labels = [
        "recent low-imp off-topic",
        "old high-imp similar",
        "day-old mid-imp similar",
        "recent low-imp off-topic2",
    ]

    query = "user preference format CSV report"

    configs = {
        "A recency-heavy (0.5,0.1,0.4)": RelevanceScorer(weights=(0.5, 0.1, 0.4)),
        "B importance-heavy (0.1,0.7,0.2)": RelevanceScorer(weights=(0.1, 0.7, 0.2)),
        "C similarity-heavy (0.2,0.2,0.6)": RelevanceScorer(weights=(0.2, 0.2, 0.6)),
    }

    q_emb = fake_embed(query)

    # Build score table
    scores: dict[str, list[float]] = {cfg: [] for cfg in configs}
    for cfg_name, scorer in configs.items():
        for entry in entries:
            scores[cfg_name].append(scorer.score(entry, q_emb, now))

    # Print table
    col_w = 28
    label_w = 32
    header = f"  {'Entry':>{label_w}}"
    for cfg_name in configs:
        header += f"  {cfg_name[:col_w]:>{col_w}}"
    print(header)
    print("  " + "-" * (label_w + len(configs) * (col_w + 2)))

    for i, label in enumerate(labels):
        row = f"  {label:>{label_w}}"
        for cfg_name in configs:
            row += f"  {scores[cfg_name][i]:>{col_w}.4f}"
        print(row)

    print()
    for cfg_name in configs:
        winner_idx = scores[cfg_name].index(max(scores[cfg_name]))
        print(f"  {cfg_name} -> winner: '{labels[winner_idx]}'")

    # Assertions: each config should have a potentially different winner
    # A (recency): most recent entry should score well
    # B (importance): high-importance entry should score well
    # C (similarity): semantically similar entry should score well
    winner_a = labels[scores["A recency-heavy (0.5,0.1,0.4)"].index(
        max(scores["A recency-heavy (0.5,0.1,0.4)"]))]
    winner_b = labels[scores["B importance-heavy (0.1,0.7,0.2)"].index(
        max(scores["B importance-heavy (0.1,0.7,0.2)"]))]
    winner_c = labels[scores["C similarity-heavy (0.2,0.2,0.6)"].index(
        max(scores["C similarity-heavy (0.2,0.2,0.6)"]))]

    # Config B (importance-heavy) must pick the high-importance entry
    assert winner_b == "old high-imp similar", (
        f"Expected B to pick 'old high-imp similar', got '{winner_b}'"
    )
    # Config C (similarity-heavy) must pick one of the semantically similar entries
    assert winner_c in ("old high-imp similar", "day-old mid-imp similar"), (
        f"Expected C to pick a similar entry, got '{winner_c}'"
    )
    print("  [OK] Assertions passed")


# ===========================================================================
# SOLUTION 2 -- Consolidation manuelle episodique → semantique
# ===========================================================================

def manual_consolidate(
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    theme: str,
    min_count: int = 2,
) -> MemoryEntry | None:
    """
    Filter episodes containing `theme` in their content.
    If enough episodes exist, create a consolidated semantic fact.
    """
    # Case-insensitive filter
    matching = [e for e in episodic.all() if theme.lower() in e.content.lower()]
    if len(matching) < min_count:
        return None

    # Importance scales with the number of source episodes
    importance = min(1.0, 0.5 + 0.1 * len(matching))
    fact_content = (
        f"[{len(matching)} episodes] Theme '{theme}': "
        + matching[-1].content  # most recent episode as anchor
    )
    return semantic.add(
        key=f"consolidated_{theme}",
        content=fact_content,
        importance=importance,
        tags=["consolidated", theme],
    )


def solution_2() -> None:
    print("\n=== Solution 2: Consolidation manuelle episodique → semantique ===")

    episodic = EpisodicMemory()
    semantic = SemanticMemory()

    # 3 episodes about CSV
    episodic.add("user requested CSV report for Q3 sales", importance=0.8)
    episodic.add("CSV export completed successfully and emailed", importance=0.6)
    episodic.add("user thanked for the CSV report format", importance=0.5)

    # 3 episodes about payment
    episodic.add("payment service returned 503 error during checkout", importance=0.9)
    episodic.add("payment retry succeeded after 2 attempts", importance=0.7)
    episodic.add("user reported payment delay on Friday evening", importance=0.8)

    # 2 vague episodes (no clear theme)
    episodic.add("user said hello at session start", importance=0.2)
    episodic.add("agent initialized session context", importance=0.1)

    print(f"  Total episodes: {len(episodic)}")

    for theme in ["CSV", "payment", "hello"]:
        fact = manual_consolidate(episodic, semantic, theme, min_count=2)
        if fact is None:
            print(f"  Theme '{theme}' -> no fact (insufficient episodes)")
        else:
            print(f"  Theme '{theme}' -> fact created:")
            print(f"    content   : {fact.content[:80]}")
            print(f"    importance: {fact.importance:.2f}")

    # Assertions
    assert semantic.get("consolidated_CSV") is not None, "CSV fact should be created"
    assert semantic.get("consolidated_payment") is not None, "payment fact should be created"
    assert semantic.get("consolidated_hello") is None, "hello fact should NOT be created"

    csv_fact = semantic.get("consolidated_CSV")
    assert "3 episodes" in csv_fact.content, "CSV fact should mention 3 episodes"
    assert csv_fact.importance == min(1.0, 0.5 + 0.1 * 3)

    payment_fact = semantic.get("consolidated_payment")
    assert payment_fact.importance == min(1.0, 0.5 + 0.1 * 3)

    print("  [OK] Assertions passed")


# ===========================================================================
# SOLUTION 3 -- MemGPT paging — visualiser les evictions
# ===========================================================================

def solution_3() -> None:
    print("\n=== Solution 3: MemGPT paging — visualiser les evictions ===")

    hmem = HierarchicalMemory()

    observations = [
        ("user logged in successfully", 0.3),
        ("user asked for CSV sales report", 0.8),
        ("payment service 503 error critical", 0.9),
        ("agent generated Q1 CSV report", 0.7),
        ("user said thank you for the report", 0.4),
        ("payment retry failed again critical error", 0.9),
        ("user updated email address", 0.6),
        ("payment service still down urgent", 0.95),
    ]

    print(f"  Context limit = MAX_MAIN_CONTEXT = {MAX_MAIN_CONTEXT}")
    print()

    for i, (content, importance) in enumerate(observations, 1):
        hmem.observe(content, importance=importance)
        # Invariant: main context must never exceed the limit
        assert len(hmem._main_context) <= MAX_MAIN_CONTEXT, (
            f"Context overflow at observation {i}!"
        )
        slots_used = len(hmem._main_context)
        print(f"  After obs {i} ({slots_used}/{MAX_MAIN_CONTEXT} slots): "
              f"imp={importance:.2f} | {content[:45]}")

    print("\n  --- Main context BEFORE page_in ---")
    hmem.show_main_context()

    # Page in: bring payment-related entries back into context
    print("\n  Calling page_in('payment error critical', k=2)...")
    paged = hmem.page_in("payment error critical", k=2)
    print(f"  Paged in {len(paged)} entries:")
    for e in paged:
        print(f"    [{e.id}] {e.content[:60]}")

    # Invariant must still hold after page_in
    assert len(hmem._main_context) <= MAX_MAIN_CONTEXT, "Context overflow after page_in!"

    print("\n  --- Main context AFTER page_in ---")
    hmem.show_main_context()

    # Verify that at least one payment-related entry is now in context
    payment_in_ctx = any("payment" in e.content.lower() for e in hmem._main_context)
    assert payment_in_ctx, "Expected at least one payment entry in main context after page_in"

    print(f"\n  Final archival size: {len(hmem._archival)}")
    print(f"  Final semantic facts: {len(hmem.semantic)}")
    print("  [OK] Assertions passed")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()
    print("\n[All Day 16 solutions passed]\n")

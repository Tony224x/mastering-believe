"""
Solutions -- Day 15 (HARD): Context engineering & token budgeting

Contains solutions for:
  - Hard Ex 1: HierarchicalBudgetAllocator -- depth-decaying per-node budgets
               plus a GLOBAL circuit-breaker (veto_local / escalate_global /
               granted). Proves no budget invariant is ever violated.
  - Hard Ex 2: Full deep-agent loop combining auto-compaction (ContextWindow)
               + VFS offloading + per-subagent context isolation. Proves a long
               run completes without overflow AND that sub-agent isolation keeps
               the parent context tiny while sub-agents do the heavy lifting.

Self-contained: embeds a heuristic token estimator, a tiny VFS and a
ContextWindow, so the file RUNS OFFLINE with zero dependencies (no langgraph /
numpy, no API key). Each solution ends with assertions (self-test) and prints
PASS.

Run:  python 03-exercises/solutions/15-context-engineering-compaction-hard.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ==========================================================================
# Shared helpers (embedded so the file is self-contained / offline)
# ==========================================================================

def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~1 token per 4 chars."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ==========================================================================
# HARD EXERCISE 1 -- Hierarchical budget allocator + global circuit-breaker
# ==========================================================================

@dataclass
class BudgetVerdict:
    node_id: str
    depth: int
    tokens: int
    verdict: str   # "granted" | "veto_local" | "escalate_global"


class HierarchicalBudgetAllocator:
    """
    Per-depth token budgets that DECAY with depth, plus a global circuit-breaker.

    Invariants guaranteed at all times:
      - global_spent <= global_budget
      - local_spent[node] <= budget_at_depth(depth[node])

    A request is granted only if it fits BOTH the node's local budget AND the
    remaining global budget; otherwise it consumes nothing and is vetoed
    (local) or escalated (global).
    """

    def __init__(self, base_budget: int = 10_000, decay: float = 0.4,
                 global_budget: int = 30_000, max_depth: int = 3) -> None:
        self.base_budget = base_budget
        self.decay = decay
        self.global_budget = global_budget
        self.max_depth = max_depth

        self.global_spent = 0
        self.local_spent: dict[str, int] = {}
        self.depth_of: dict[str, int] = {}
        self.escalations: list[str] = []   # trace of escalate_global events

    def budget_at_depth(self, depth: int) -> int:
        """Depth-decaying per-node budget: base * decay**depth."""
        return int(self.base_budget * (self.decay ** depth))

    def spawn(self, parent_id: str, depth: int) -> bool:
        """Allow spawning a child only if depth stays within max_depth."""
        return depth <= self.max_depth

    def _register(self, node_id: str, depth: int) -> None:
        self.local_spent.setdefault(node_id, 0)
        self.depth_of.setdefault(node_id, depth)

    def request(self, node_id: str, depth: int, tokens: int) -> str:
        """Try to consume `tokens` for `node_id` at `depth`. Returns a verdict."""
        self._register(node_id, depth)
        local_cap = self.budget_at_depth(depth)

        # Local budget check first (per-node cap at this depth).
        if self.local_spent[node_id] + tokens > local_cap:
            return "veto_local"  # nothing consumed

        # Global circuit-breaker check.
        if self.global_spent + tokens > self.global_budget:
            self.escalations.append(
                f"{node_id}@d{depth} wanted {tokens}, global "
                f"{self.global_spent}/{self.global_budget}")
            return "escalate_global"  # nothing consumed

        # Both pass -> grant and record.
        self.local_spent[node_id] += tokens
        self.global_spent += tokens
        return "granted"

    def check_invariants(self) -> None:
        """Hard guarantee: no budget is ever exceeded."""
        assert self.global_spent <= self.global_budget, (
            self.global_spent, self.global_budget)
        for node, spent in self.local_spent.items():
            cap = self.budget_at_depth(self.depth_of[node])
            assert spent <= cap, (node, spent, cap)


def hard_ex1_hierarchical_budget() -> None:
    print("\n" + "=" * 64)
    print("  HARD 1: Hierarchical budget allocator + global circuit-breaker")
    print("=" * 64)

    alloc = HierarchicalBudgetAllocator(
        base_budget=10_000, decay=0.4, global_budget=18_000, max_depth=3)

    print("\n  Per-depth budgets:")
    for d in range(4):
        print(f"    depth {d}: {alloc.budget_at_depth(d)} tokens "
              f"(spawn allowed: {alloc.spawn('p', d)})")

    # Build a ReAct tree: 1 supervisor (d0), K children (d1), K^2 grandchildren (d2).
    supervisor = "sup"
    children = [f"sub_{i}" for i in range(2)]
    grandkids = [f"sub_{i}_{j}" for i in range(3) for j in range(2)]

    verdicts: list[BudgetVerdict] = []

    def do(node, depth, tokens):
        # Re-check invariants AFTER every request (granted or not).
        v = alloc.request(node, depth, tokens)
        alloc.check_invariants()
        verdicts.append(BudgetVerdict(node, depth, tokens, v))
        return v

    print("\n  Requests:")
    # Supervisor: budget_at_depth(0) = 10_000.
    do(supervisor, 0, 3_000)            # granted   (global = 3_000)
    do(supervisor, 0, 8_000)            # veto_local (3000+8000 > 10000) -> 0 consumed

    # Depth-1 children: budget = 4_000 each.
    do(children[0], 1, 3_500)           # granted   (global = 6_500)
    do(children[0], 1, 1_000)           # veto_local (3500+1000 > 4000) -> 0 consumed
    do(children[1], 1, 3_500)           # granted   (global = 10_000)

    # Depth-2 grandchildren: budget = 1_600 each.
    do(grandkids[0], 2, 1_500)          # granted   (global = 11_500)
    do(grandkids[1], 2, 1_500)          # granted   (global = 13_000)
    do(grandkids[2], 2, 1_500)          # granted   (global = 14_500)
    do(grandkids[3], 2, 1_400)          # granted   (global = 15_900)
    do(grandkids[4], 2, 1_400)          # granted   (global = 17_300, remaining = 700)
    # Fresh grandchild (local_spent=0, cap=1600) requests 1_000: fits LOCALLY
    # but 17_300 + 1_000 = 18_300 > 18_000 global -> circuit-breaker escalates.
    do(grandkids[5], 2, 1_000)          # escalate_global -> 0 consumed

    for v in verdicts:
        print(f"    {v.node_id:10} d{v.depth} req={v.tokens:>5} -> {v.verdict}")

    kinds = {v.verdict for v in verdicts}
    print(f"\n  Verdict kinds observed: {sorted(kinds)}")
    print(f"  global_spent: {alloc.global_spent} / {alloc.global_budget}")
    print(f"  escalations: {alloc.escalations}")

    # --- Assertions ---
    # Depth-decaying budgets.
    assert alloc.budget_at_depth(0) > alloc.budget_at_depth(1) > alloc.budget_at_depth(2)

    # All three verdicts observed at least once.
    assert "granted" in kinds
    assert "veto_local" in kinds
    assert "escalate_global" in kinds

    # Vetoed/escalated requests consumed nothing: reconstruct expected spent.
    granted = [v for v in verdicts if v.verdict == "granted"]
    expected_global = sum(v.tokens for v in granted)
    assert alloc.global_spent == expected_global, (alloc.global_spent, expected_global)

    # Per-node local spent matches only granted requests for that node.
    for node in set(v.node_id for v in verdicts):
        expected_local = sum(v.tokens for v in granted if v.node_id == node)
        assert alloc.local_spent.get(node, 0) == expected_local, node

    # Global invariant holds.
    assert alloc.global_spent <= alloc.global_budget
    alloc.check_invariants()

    # spawn refuses beyond max_depth.
    assert alloc.spawn("x", alloc.max_depth) is True
    assert alloc.spawn("x", alloc.max_depth + 1) is False

    print("\n  PASS -- 3 verdicts, no-consume on refusal, invariants hold, depth cap\n")


# ==========================================================================
# HARD EXERCISE 2 -- Full deep-agent loop (compaction + offload + isolation)
# ==========================================================================

class VFS:
    """In-memory filesystem used as external agent memory (offload target)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        self._store[path] = content

    def read(self, path: str) -> Optional[str]:
        return self._store.get(path)

    def list_files(self) -> list[str]:
        return sorted(self._store.keys())


class ContextWindow:
    """
    A bounded message buffer that AUTO-COMPACTS when it crosses a threshold,
    keeping the last `keep_tail` messages + a fixed-size summary of the rest.
    """

    def __init__(self, token_limit: int = 1_000, ratio: float = 0.75,
                 keep_tail: int = 3, summary_tokens: int = 30) -> None:
        self.token_limit = token_limit
        self.threshold = int(token_limit * ratio)
        self.keep_tail = keep_tail
        self.summary_tokens = summary_tokens
        self.messages: list[dict] = []
        self.compactions = 0

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if self.tokens() > self.threshold:
            self._compact()

    def tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    def text(self) -> str:
        return " ".join(m["content"] for m in self.messages)

    def _compact(self) -> None:
        if len(self.messages) <= self.keep_tail:
            return
        tail = self.messages[-self.keep_tail:]
        n_summarized = len(self.messages) - self.keep_tail
        summary = {
            "role": "system",
            # Fixed-size-ish summary; its token count is bounded by summary_tokens.
            "content": "S" * (self.summary_tokens * 4) +
                       f" [{n_summarized} msgs condensed]",
        }
        self.messages = [summary] + tail
        self.compactions += 1


class SubAgent:
    """
    A sub-agent with its OWN isolated context window. It receives a COMPACT
    task prompt (not the parent's history), offloads big inputs to the VFS,
    runs several internal turns, and returns a COMPACT result to the parent.
    """

    RESULT_TOKEN_CAP = 50

    def __init__(self, name: str, vfs: VFS) -> None:
        self.name = name
        self.vfs = vfs
        # Tight limit so the heavy internal work forces compactions even though
        # the big inputs themselves are offloaded (only reasoning turns remain).
        self.ctx = ContextWindow(token_limit=400, ratio=0.6,
                                  keep_tail=2, summary_tokens=20)
        self.processed_tokens = 0   # how much raw work this sub-agent did

    def run(self, task: str, big_inputs: list[str]) -> str:
        # Compact task prompt only -- no parent history.
        self.ctx.add("user", task)

        for idx, blob in enumerate(big_inputs):
            # Offload the big input to the VFS; keep only a placeholder inline.
            path = f"{self.name}/input_{idx}.txt"
            self.vfs.write(path, blob)
            self.processed_tokens += estimate_tokens(blob)
            self.ctx.add("tool", f"[OFFLOADED -> {path}]")

            # Several internal reasoning turns over the (referenced) input.
            # These accumulate in the sub-agent's OWN context and trigger ITS
            # auto-compaction -- the heavy lifting happens in isolation.
            for t in range(4):
                self.ctx.add(
                    "assistant",
                    f"turn {t} on {path}: analyzing chunk, extracting findings, "
                    f"cross-checking against prior observations, noting candidate "
                    f"issues and their severity for the consolidated report")
                self.ctx.add(
                    "tool",
                    f"observation t{t}: 2 findings recorded, 1 false positive "
                    f"dismissed, context updated with running tally")

        # Return a COMPACT result -- not the internal logs.
        result = f"[{self.name}] done: {len(big_inputs)} inputs, summary ready"
        return result


class ParentAgent:
    """
    Orchestrator with its OWN context window. For each sub-task it spawns a
    FRESH sub-agent (isolated context) and integrates ONLY the compact result.
    """

    def __init__(self, vfs: VFS) -> None:
        self.vfs = vfs
        self.ctx = ContextWindow(token_limit=2_000, ratio=0.8,
                                 keep_tail=4, summary_tokens=30)
        self.subagents: list[SubAgent] = []

    def orchestrate(self, subtasks: list[tuple[str, list[str]]]) -> dict:
        self.ctx.add("user", "orchestrate the security audit across modules")
        total_sub_processed = 0
        max_sub_compactions = 0

        for i, (task, big_inputs) in enumerate(subtasks):
            sub = SubAgent(f"sub_{i}", self.vfs)
            self.subagents.append(sub)
            # Delegate with a SHORT prompt; sub-agent does the heavy lifting.
            result = sub.run(task, big_inputs)
            total_sub_processed += sub.processed_tokens
            max_sub_compactions = max(max_sub_compactions, sub.ctx.compactions)
            # Parent integrates ONLY the compact result.
            self.ctx.add("assistant", f"integrated sub_{i}: {result}")

        return {
            "parent_tokens": self.ctx.tokens(),
            "total_sub_processed": total_sub_processed,
            "max_sub_compactions": max_sub_compactions,
        }


def hard_ex2_deep_agent_loop() -> None:
    print("\n" + "=" * 64)
    print("  HARD 2: Deep-agent loop (compaction + offload + isolation)")
    print("=" * 64)

    vfs = VFS()
    parent = ParentAgent(vfs)

    # Unique marker embedded in every big input -- it must NEVER reach the parent.
    MARKER = "ZZZ_BIG_INPUT_MARKER_QXR"

    def big(n_repeat: int) -> str:
        return MARKER + " " + ("scraped data row with details; " * n_repeat)

    # 5 sub-tasks, each with several huge inputs.
    subtasks = [
        ("audit auth module for injection", [big(400), big(500), big(450)]),
        ("audit payment module", [big(600), big(550)]),
        ("audit user-profile module", [big(500), big(400), big(420)]),
        ("audit admin module", [big(700), big(650)]),
        ("audit api gateway", [big(480), big(520), big(460)]),
    ]

    report = parent.orchestrate(subtasks)

    parent_tokens = report["parent_tokens"]
    sub_processed = report["total_sub_processed"]
    ratio = sub_processed / max(1, parent_tokens)

    print(f"\n  sub-agents spawned: {len(parent.subagents)} (each isolated context)")
    print(f"  total tokens processed by sub-agents: {sub_processed}")
    print(f"  parent context tokens (final):        {parent_tokens}")
    print(f"  isolation ratio (sub/parent):         {ratio:.1f}x")
    print(f"  max compactions in a sub-agent:       {report['max_sub_compactions']}")
    print(f"  parent compactions:                   {parent.ctx.compactions}")
    print(f"  VFS files stored: {len(vfs.list_files())}")

    # --- Assertions ---
    # (a) Run completed; NO context exceeds its limit.
    assert parent.ctx.tokens() <= parent.ctx.token_limit, parent.ctx.tokens()
    for sub in parent.subagents:
        assert sub.ctx.tokens() <= sub.ctx.token_limit, (sub.name, sub.ctx.tokens())

    # (b) Parent stays small: sub-agents did >10x the parent's token volume.
    assert ratio > 10.0, ratio

    # (c) Big inputs NEVER leaked into the parent context.
    assert MARKER not in parent.ctx.text(), "big input leaked into parent context!"

    # (d) Big inputs remain retrievable from the VFS (round-trip on a sample).
    sample_path = parent.subagents[0].name + "/input_0.txt"
    restored = vfs.read(sample_path)
    assert restored is not None and MARKER in restored, "offloaded input lost"

    # (e) At least one sub-agent auto-compacted (the run truly stressed context).
    assert report["max_sub_compactions"] >= 1, "no sub-agent compaction occurred"

    # Compact results only: every integrated result stays under the cap.
    for m in parent.ctx.messages:
        if m["content"].startswith("integrated sub_"):
            assert estimate_tokens(m["content"]) <= SubAgent.RESULT_TOKEN_CAP + 20

    print("\n  PASS -- no overflow, parent tiny (>10x), inputs offloaded & retrievable\n")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 64)
    print("  Day 15 HARD Solutions -- Context engineering & token budgeting")
    print("  (stdlib only -- running fully offline)")
    print("#" * 64)

    hard_ex1_hierarchical_budget()
    hard_ex2_deep_agent_loop()

    print("\n" + "#" * 64)
    print("  All hard solutions executed successfully.")
    print("#" * 64 + "\n")

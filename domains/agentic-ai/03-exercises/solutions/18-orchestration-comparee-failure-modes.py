"""
Day 18 -- Solutions to easy exercises: orchestration compared & failure modes.

Run:
    python domains/agentic-ai/03-exercises/solutions/18-orchestration-comparee-failure-modes.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Import from the day-18 code module
# ---------------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# We need importlib because the module name contains a hyphen
import importlib

day18 = importlib.import_module("18-orchestration-comparee-failure-modes")

ToyLangGraph = day18.ToyLangGraph
GraphState = day18.GraphState
node_research = day18.node_research
node_writer = day18.node_writer
node_reviewer = day18.node_reviewer
conditional_edge = day18.conditional_edge
mock_llm = day18.mock_llm
_CALL_LOG = day18._CALL_LOG
reset_log = day18.reset_log
tokens_used = day18.tokens_used


# ===========================================================================
# SOLUTION 1 -- Checkpointing + replay in ToyLangGraph
# ===========================================================================

class CheckpointedGraph(ToyLangGraph):
    """
    Extends ToyLangGraph with per-node checkpointing and replay.
    After each node execution, a deep copy of the state is appended to
    self.history so we can resume from any step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: list[GraphState] = []
        self._node_order: list[str] = []   # tracks execution order

    def add_edge(self, from_node: str, to_node) -> None:  # type: ignore[override]
        super().add_edge(from_node, to_node)

    def run(  # type: ignore[override]
        self,
        entry: str,
        state: GraphState,
        max_steps: int = 10,
        crash_at_step: int | None = None,
    ) -> GraphState:
        self.history.clear()
        self._node_order.clear()

        current = entry
        steps = 0
        while current and steps < max_steps:
            if current not in self._nodes:
                break

            state = self._nodes[current](state)

            # Save checkpoint (deep copy so future mutations don't affect it)
            self.history.append(deepcopy(state))
            self._node_order.append(current)

            print(f"  [checkpoint] step={state.step} node={current} saved")

            # Simulate crash after a specific step
            if crash_at_step is not None and state.step == crash_at_step:
                raise RuntimeError(
                    f"Simulated crash after node '{current}' (step={state.step})"
                )

            edge = self._edges.get(current)
            if edge is None:
                break
            current = edge(state) if callable(edge) else edge
            steps += 1

        return state

    def replay(self, from_step: int) -> GraphState:
        """
        Reload state from checkpoint at `from_step` (0-based index into
        self.history) and continue execution from the next node.
        """
        if from_step >= len(self.history):
            raise IndexError(f"No checkpoint at step index {from_step}")

        restored = deepcopy(self.history[from_step])
        # The node after from_step is the one following in _node_order
        next_node_idx = from_step + 1
        if next_node_idx >= len(self._node_order):
            # Already at the end -- nothing to replay
            return restored

        next_node = self._node_order[next_node_idx]
        print(f"  [replay] restoring state from step={restored.step}, resuming at node='{next_node}'")

        # Build a minimal re-run from next_node onward (no more crash)
        current = next_node
        state = restored
        steps = 0
        while current and steps < 10:
            if current not in self._nodes:
                break
            state = self._nodes[current](state)
            print(f"  [replay] executed node={current} step={state.step}")
            edge = self._edges.get(current)
            if edge is None:
                break
            current = edge(state) if callable(edge) else edge
            steps += 1
        return state


def solution_1() -> None:
    print("\n" + "=" * 60)
    print("Solution 1: Checkpointing + replay in ToyLangGraph")
    print("=" * 60)

    g = CheckpointedGraph()
    g.add_node("researcher", node_research)
    g.add_node("writer", node_writer)
    g.add_node("reviewer", node_reviewer)
    g.add_edge("researcher", "writer")
    g.add_edge("writer", conditional_edge)
    g.add_edge("reviewer", None)

    state = GraphState(task="checkpointing demo")

    # --- Run with crash at step 2 (after writer) ---
    print("\n  --- Run with crash at step=2 ---")
    try:
        g.run("researcher", state, crash_at_step=2)
    except RuntimeError as exc:
        print(f"  Caught: {exc}")

    # --- Replay from checkpoint at index 0 (step=1, after researcher) ---
    print("\n  --- Replay from checkpoint index 0 (step=1) ---")
    final = g.replay(from_step=0)

    print(f"\n  Final state after replay:")
    print(f"    task     : {final.task}")
    print(f"    research : {final.research_result[:60]}")
    print(f"    draft    : {final.draft[:60]}")
    print(f"    review   : {final.review[:60]}")

    # Assertions
    assert final.step == 3, f"Expected step=3, got {final.step}"
    assert final.review != "", "Review should be non-empty after replay"
    print("\n  [OK] Replay produced correct final state.")


# ===========================================================================
# SOLUTION 2 -- Budget guard
# ===========================================================================

class BudgetExceededError(Exception):
    """Raised when the token budget is exceeded."""


class BudgetGuard:
    """
    Monitors cumulative token-in consumption against a fixed budget.
    Raises BudgetExceededError if the budget is crossed.
    """

    def __init__(self, max_tokens_in: int) -> None:
        self.max_tokens_in = max_tokens_in

    def check(self) -> None:
        t_in, _ = tokens_used()
        if t_in > self.max_tokens_in:
            raise BudgetExceededError(
                f"Budget exceeded: {t_in} tokens used > limit {self.max_tokens_in}"
            )

    def reset(self) -> None:
        reset_log()


def solution_2() -> None:
    print("\n" + "=" * 60)
    print("Solution 2: Budget guard -- stop pipeline if tokens exceed limit")
    print("=" * 60)

    # Budget is set low (30 tokens) to guarantee a cut within the 5-agent loop.
    # In a real system the threshold would be in the thousands; we keep it
    # small here so the demo is deterministic without a real LLM.
    guard = BudgetGuard(max_tokens_in=30)
    guard.reset()

    context = "Task: analyse LangGraph checkpointing patterns. "
    cut_at = None

    for agent_idx in range(5):
        agent_name = f"agent_{agent_idx}"
        try:
            response = mock_llm(agent_name, context)
            context += response + " "
            guard.check()
            t_in, _ = tokens_used()
            print(f"  agent_{agent_idx}: OK  | tokens_in so far: {t_in}")
        except BudgetExceededError as exc:
            cut_at = agent_idx
            t_in, _ = tokens_used()
            print(f"  agent_{agent_idx}: BUDGET CUT | {exc}")
            break

    assert cut_at is not None, "Budget guard should have triggered"
    print(f"\n  Pipeline stopped at agent_{cut_at} -- budget guard worked correctly.")

    # Test reset
    guard.reset()
    t_in_after, _ = tokens_used()
    assert t_in_after == 0, "reset() should clear the token log"
    print("  [OK] reset() cleared the counter. Ready for next pipeline.")


# ===========================================================================
# SOLUTION 3 -- Tiebreaker agent
# ===========================================================================

@dataclass
class TiebreakerAgent:
    name: str = "arbitre"

    def resolve(self, position_a: str, position_b: str, criteria: str) -> str:
        """
        Call mock_llm once to pick the winning position.
        Returns a decision string.
        """
        prompt = (
            f"Arbitrate: position_A='{position_a[:40]}' vs "
            f"position_B='{position_b[:40]}'. Criteria: {criteria}. "
            f"Pick the best approach."
        )
        verdict = mock_llm(self.name, prompt)
        # Decide based on which position letter appears first in the verdict
        if "position_a" in verdict.lower() or "approach x" in verdict.lower():
            return "Position A retenue"
        return "Position B retenue"


def solution_3() -> None:
    print("\n" + "=" * 60)
    print("Solution 3: Tiebreaker -- resolve disagreement between two agents")
    print("=" * 60)

    MAX_DEBATE_TURNS = 3
    tiebreaker = TiebreakerAgent()

    # --- Run WITH tiebreaker ---
    reset_log()
    position_a = "approach X (LangGraph) is better because it is stateful"
    position_b = "approach Y (CrewAI) is better because it is simpler"

    print("\n  --- Debate with tiebreaker after 3 turns ---")
    for turn in range(MAX_DEBATE_TURNS):
        response_a = mock_llm("agent_A", f"disagree with '{position_b}', my approach is better")
        position_a = response_a
        response_b = mock_llm("agent_B", f"disagree with '{position_a}', my approach is better")
        position_b = response_b
        t_in, t_out = tokens_used()
        print(f"  turn {turn + 1:02d} | tokens_in: {t_in}")

    # Invoke tiebreaker
    decision = tiebreaker.resolve(
        position_a, position_b, criteria="production reliability and debuggability"
    )
    t_in_with, t_out_with = tokens_used()
    print(f"  Tiebreaker decision: {decision}")
    print(f"  Total tokens WITH tiebreaker: {t_in_with} in / {t_out_with} out")

    # --- Run WITHOUT tiebreaker (6 turns, no resolution) ---
    reset_log()
    position_a = "approach X (LangGraph) is better because it is stateful"
    position_b = "approach Y (CrewAI) is better because it is simpler"

    print("\n  --- Debate WITHOUT tiebreaker (6 turns, no resolution) ---")
    for turn in range(6):
        response_a = mock_llm("agent_A", f"disagree with '{position_b}', my approach is better")
        position_a = response_a
        response_b = mock_llm("agent_B", f"disagree with '{position_a}', my approach is better")
        position_b = response_b

    t_in_without, t_out_without = tokens_used()
    print(f"  Total tokens WITHOUT tiebreaker: {t_in_without} in / {t_out_without} out")

    savings = t_in_without - t_in_with
    print(f"\n  Token savings from tiebreaker: {savings} input tokens")
    assert t_in_with < t_in_without, "Tiebreaker should use fewer tokens than 6 unbounded turns"
    assert decision in ("Position A retenue", "Position B retenue")
    print("  [OK] Tiebreaker resolves debate cheaper than unbounded loop.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

    print("\n" + "=" * 60)
    print("All solutions ran successfully.")
    print("=" * 60)

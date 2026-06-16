"""
Day 15 -- Solutions to the easy exercises for context engineering & compaction.

Run the whole file to execute every solution.

    python domains/agentic-ai/03-exercises/solutions/15-context-engineering-compaction.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Import the day-15 module (same technique as J13 solutions)
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "02-code"
sys.path.insert(0, str(SRC))

# Module name contains hyphens, so we must use import_module (not plain import)
mod = import_module("15-context-engineering-compaction")

ContextManager = mod.ContextManager
TokenBudget = mod.TokenBudget
VirtualFS = mod.VirtualFS
estimate_tokens = mod.estimate_tokens
_mock_summarize = mod._mock_summarize


# ===========================================================================
# SOLUTION 1 -- RetentionPolicy + selective compaction
# ===========================================================================

class RetentionPolicy:
    """
    Decide which messages must survive compaction regardless of age.

    Two criteria:
      1. The very first user message (the original goal).
      2. Any message containing the [CRITICAL] marker.
    """

    def __init__(self) -> None:
        self._first_user_seen = False

    def should_keep(self, message: dict) -> bool:
        if message.get("role") == "user" and not self._first_user_seen:
            # The first user message = the goal -- never discard it
            self._first_user_seen = True
            return True
        if "[CRITICAL]" in message.get("content", ""):
            return True
        return False


class RetentionContextManager(ContextManager):
    """
    ContextManager extended with a RetentionPolicy.

    During compaction, messages flagged by the policy are extracted first,
    then the remainder is summarized, and the result is:
        [retained messages] + [summary] + [recent messages]
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._policy = RetentionPolicy()
        # Track which message indices are retained so we can re-apply
        # the policy after each compaction round.
        self._retained_indices: set[int] = set()

    def add_message(self, role: str, content: str) -> None:
        """Append and check retention before auto-compaction."""
        idx = len(self.messages)
        msg = {"role": role, "content": content}
        if self._policy.should_keep(msg):
            self._retained_indices.add(idx)
        self.messages.append(msg)
        # Trigger compaction if the threshold is exceeded (same logic as parent)
        if self.total_tokens() > self.threshold_tokens:
            self._compact_with_retention()

    def _compact_with_retention(self) -> None:
        """Compact while preserving retained messages."""
        n = len(self.messages)
        if n <= self.MIN_MESSAGES_BEFORE_COMPACT:
            return

        keep_tail = self.MIN_MESSAGES_BEFORE_COMPACT
        candidates = self.messages[:-keep_tail]
        recent = self.messages[-keep_tail:]

        # Split candidates into retained vs summarizable
        retained = [m for i, m in enumerate(candidates) if i in self._retained_indices]
        summarizable = [m for i, m in enumerate(candidates) if i not in self._retained_indices]

        summary_text = _mock_summarize(summarizable) if summarizable else ""

        new_messages = []
        if retained:
            new_messages.extend(retained)
        if summary_text:
            new_messages.append({"role": "system", "content": summary_text})
        new_messages.extend(recent)

        self.messages = new_messages
        # Reset retained indices -- they now live at positions 0..len(retained)-1
        self._retained_indices = set(range(len(retained)))
        self.compaction_count += 1


def solution_1() -> None:
    print("\n=== Solution 1: RetentionPolicy + selective compaction ===")

    # Small limit so compaction fires during the test
    ctx = RetentionContextManager(token_limit=200, compaction_threshold=0.70)

    messages = [
        ("user", "[CRITICAL] goal: never expose external API credentials in any output"),
        ("assistant", "Understood. Starting analysis of the Flask repo."),
        ("tool", "Found 45 Python files."),
        ("assistant", "Reading app.py."),
        ("tool", "app.py: Flask initialized, SECRET_KEY from env."),
        ("assistant", "[CRITICAL] Decision: use bandit for static analysis, avoid raw regex scanning."),
        ("tool", "bandit results: 3 medium issues found."),
        ("assistant", "Investigating medium issues."),
        ("tool", "Issue 1: B105 hardcoded password string in tests/fixtures.py line 12."),
        ("assistant", "Flagged as false positive — it is a test fixture, not production."),
        ("tool", "Issue 2: B608 SQL injection risk in admin/views.py line 78."),
        ("assistant", "This is a real finding. Adding to the security report."),
    ]

    print(f"\n  {'Message':55}  tokens_after  compact#")
    for role, content in messages:
        ctx.add_message(role, content)
        preview = f"[{role}] {content[:45]}..."
        print(f"  {preview:55}  {ctx.total_tokens():5}       {ctx.compaction_count}")

    # Verify retained messages are still present
    all_content = " ".join(m["content"] for m in ctx.messages)
    has_goal = "[CRITICAL] goal:" in all_content
    has_decision = "Decision: use bandit" in all_content
    print(f"\n  Goal message retained: {has_goal}")
    print(f"  Decision [CRITICAL] retained: {has_decision}")
    print(f"  Final message count: {len(ctx.messages)} (was {len(messages)})")
    print(f"  Total compactions: {ctx.compaction_count}")

    assert has_goal, "Goal message was lost during compaction"
    assert has_decision, "[CRITICAL] decision was lost during compaction"


# ===========================================================================
# SOLUTION 2 -- ContextRotMonitor
# ===========================================================================

_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou",
    "en", "a", "the", "of", "to", "in", "is", "it", "for", "on",
}


def coherence_score(goal: str, recent_action: str) -> float:
    """
    Proportion of significant goal words present in recent_action.

    Returns 1.0 if all goal words appear in the action, 0.0 if none do.
    """
    goal_words = {
        w.lower().strip(".,;:!?")
        for w in goal.split()
        if w.lower() not in _STOPWORDS and len(w) > 2
    }
    if not goal_words:
        return 1.0  # empty goal => trivially coherent

    action_lower = recent_action.lower()
    matching = sum(1 for w in goal_words if w in action_lower)
    return matching / len(goal_words)


class ContextRotMonitor:
    """
    Track agent coherence over a sliding window of recent actions.

    High coherence = the agent is still on task.
    Low coherence  = the agent is drifting (context rot).
    """

    def __init__(self, goal: str, window: int = 5) -> None:
        self.goal = goal
        self.window = window
        self._scores: list[float] = []

    def record(self, action: str) -> float:
        """Record an action and return its coherence score."""
        score = coherence_score(self.goal, action)
        self._scores.append(score)
        return score

    def rot_detected(self, threshold: float = 0.30) -> bool:
        """Return True if mean score over the window is below threshold."""
        recent = self._scores[-self.window :]
        if not recent:
            return False
        return (sum(recent) / len(recent)) < threshold

    def summary(self) -> dict:
        recent = self._scores[-self.window :]
        mean = sum(recent) / len(recent) if recent else 0.0
        return {
            "scores": [round(s, 2) for s in self._scores],
            "mean_last_window": round(mean, 2),
            "rot": self.rot_detected(),
        }


def solution_2() -> None:
    print("\n=== Solution 2: ContextRotMonitor ===")

    goal = "analyze Flask security vulnerabilities in the authentication module"
    monitor = ContextRotMonitor(goal, window=5)

    turns = [
        # On-task turns (should score high)
        "analyze Flask authentication routes for security issues",
        "scan Flask login function for vulnerability patterns",
        "review Flask security configuration in auth module",
        # Off-task turns (should score low -> rot detected)
        "listing all python packages installed in the environment",
        "checking git history for last merge commit",
        "reading the project README markdown file",
    ]

    print(f"\n  Goal: {goal}\n")
    print(f"  {'Turn':<4} {'Action':<55} {'Score':>5}  {'Rot?':>5}")
    print(f"  {'-'*4} {'-'*55} {'-'*5}  {'-'*5}")

    for i, action in enumerate(turns, start=1):
        score = monitor.record(action)
        rot = monitor.rot_detected()
        preview = action[:53]
        print(f"  {i:<4} {preview:<55} {score:>5.2f}  {str(rot):>5}")

    print(f"\n  Final summary: {monitor.summary()}")

    # Assertions
    # After on-task turns, no rot
    early_monitor = ContextRotMonitor(goal, window=5)
    for action in turns[:3]:
        early_monitor.record(action)
    assert not early_monitor.rot_detected(), "Should not detect rot after on-task turns"

    # After off-task turns, rot should be detected
    assert monitor.rot_detected(), "Should detect rot after off-task turns"
    print("\n  Assertions passed.")


# ===========================================================================
# SOLUTION 3 -- MonitoredBudget with event escalation
# ===========================================================================

@dataclass
class BudgetEvent:
    """A budget threshold crossing event."""
    agent: str
    kind: str          # "warning" | "critical" | "exhausted"
    remaining_pct: float
    message: str


class BudgetEventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[BudgetEvent], None]] = []

    def subscribe(self, handler: Callable[[BudgetEvent], None]) -> None:
        self._handlers.append(handler)

    def publish(self, event: BudgetEvent) -> None:
        for handler in self._handlers:
            handler(event)


class MonitoredBudget:
    """
    Wraps TokenBudget and publishes BudgetEvents on threshold crossings.

    Each threshold is fired at most once per agent to avoid repeated events.
    """

    # (remaining_fraction_ceiling, event_kind)
    _THRESHOLDS: list[tuple[float, str]] = [
        (0.30, "warning"),
        (0.10, "critical"),
        (0.00, "exhausted"),
    ]

    def __init__(self, budget: TokenBudget, bus: BudgetEventBus) -> None:
        self._budget = budget
        self._bus = bus
        # Track which thresholds have already been fired per agent
        self._fired: dict[str, set[str]] = {}

    def consume(self, agent: str, tokens: int) -> bool:
        in_budget = self._budget.consume(agent, tokens)
        frac = self._budget.remaining_fraction(agent)
        pct = frac * 100

        if agent not in self._fired:
            self._fired[agent] = set()

        # Check each threshold from most severe to least severe
        for threshold_frac, kind in reversed(self._THRESHOLDS):
            if frac <= threshold_frac and kind not in self._fired[agent]:
                self._fired[agent].add(kind)
                event = BudgetEvent(
                    agent=agent,
                    kind=kind,
                    remaining_pct=round(pct, 1),
                    message=(
                        f"Agent '{agent}' budget {kind}: "
                        f"{pct:.1f}% remaining ({tokens} tokens just consumed)"
                    ),
                )
                self._bus.publish(event)

        return in_budget

    def remaining_fraction(self, agent: str) -> float:
        return self._budget.remaining_fraction(agent)

    def recommended_mode(self, agent: str) -> str:
        return self._budget.recommended_mode(agent)


class OrchestratorLogger:
    """Receives budget events and logs them with a timestamp counter."""

    def __init__(self) -> None:
        self.events: list[BudgetEvent] = []
        self._tick = 0

    def handle(self, event: BudgetEvent) -> None:
        self._tick += 1
        self.events.append(event)
        icon = {"warning": "⚡", "critical": "🔴", "exhausted": "💀"}.get(event.kind, "?")
        # Use ASCII fallback to stay within repo convention
        icon_ascii = {"warning": "[!]", "critical": "[!!]", "exhausted": "[X]"}[event.kind]
        print(f"  [{self._tick:02d}] ORCHESTRATOR received {icon_ascii} {event.message}")


def solution_3() -> None:
    print("\n=== Solution 3: MonitoredBudget with event escalation ===")

    bus = BudgetEventBus()
    orchestrator = OrchestratorLogger()
    bus.subscribe(orchestrator.handle)

    inner_budget = TokenBudget(
        total=3_000,
        allocations={
            "fast_agent":  0.20,    # 600 tokens -- will exhaust quickly
            "main_agent":  0.50,    # 1500 tokens
            "slow_agent":  0.30,    # 900 tokens
        },
    )
    budget = MonitoredBudget(inner_budget, bus)

    print()
    # fast_agent consumes its 600-token budget rapidly
    consumptions = [
        ("fast_agent", 150),   # 75% remaining
        ("fast_agent", 120),   # 55% remaining
        ("fast_agent", 100),   # 38% remaining -- passes 30% warning? not yet
        ("fast_agent", 60),    # 28% remaining -> WARNING
        ("fast_agent", 100),   # 11% remaining -> still warning already fired
        ("fast_agent", 40),    # 5% remaining  -> CRITICAL
        ("fast_agent", 40),    # -1% remaining -> EXHAUSTED (over budget)
        # main_agent moderate consumption
        ("main_agent", 400),   # 73% remaining
        ("main_agent", 400),   # 47% remaining
        ("main_agent", 400),   # 20% remaining -> WARNING
        ("main_agent", 150),   # 10% remaining -> CRITICAL
        ("main_agent", 60),    # 6% remaining
        ("main_agent", 90),    # -0% remaining -> EXHAUSTED
        # slow_agent barely uses budget
        ("slow_agent", 100),   # 89% remaining
        ("slow_agent", 100),   # 78% remaining
        ("slow_agent", 400),   # 33% remaining
        ("slow_agent", 40),    # 29% remaining -> WARNING
        ("slow_agent", 180),   # 9% remaining  -> CRITICAL
        ("slow_agent", 100),   # over -> EXHAUSTED
    ]
    print(f"  {'Agent':<12} {'Consumed':>8}  {'Remaining%':>10}  {'Mode':<14}")
    print(f"  {'-'*12} {'-'*8}  {'-'*10}  {'-'*14}")
    for agent, tokens in consumptions:
        budget.consume(agent, tokens)
        pct = budget.remaining_fraction(agent) * 100
        mode = budget.recommended_mode(agent)
        print(f"  {agent:<12} {tokens:>8}  {pct:>10.1f}  {mode:<14}")

    print(f"\n  Total events received by orchestrator: {len(orchestrator.events)}")

    # Each agent should have emitted all 3 event types
    for agent in ["fast_agent", "main_agent", "slow_agent"]:
        agent_events = {e.kind for e in orchestrator.events if e.agent == agent}
        print(f"  {agent}: events emitted = {sorted(agent_events)}")
        assert "warning"   in agent_events, f"{agent} missing warning event"
        assert "critical"  in agent_events, f"{agent} missing critical event"
        assert "exhausted" in agent_events, f"{agent} missing exhausted event"

    print("\n  All assertions passed.")


# ===========================================================================
# ENTRYPOINT
# ===========================================================================

if __name__ == "__main__":
    solution_1()
    solution_2()
    solution_3()

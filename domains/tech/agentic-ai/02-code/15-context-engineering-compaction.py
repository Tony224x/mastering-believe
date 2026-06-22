"""
Day 15 -- Context engineering: compaction, offloading & token budgeting.

Demonstrates:
  1. TokenCounter     -- lightweight word/char-based token estimator (stdlib)
  2. ContextManager   -- manages a message history, auto-compacts when the
                         estimated token count crosses a configurable threshold,
                         offloads large blobs to a virtual filesystem (dict)
  3. VirtualFS        -- in-memory filesystem usable as external agent memory
                         (todo list, scratchpad, documents)
  4. TokenBudget      -- allocates a global token budget across named sub-agents,
                         tracks consumption, reports fraction remaining, and
                         maps remaining fraction to a mode recommendation
  5. BudgetedAgent    -- toy ReAct agent that respects a TokenBudget and
                         self-compacts its context when approaching the limit
  6. Deep-agent demo  -- end-to-end run showing context rot prevention and
                         budget enforcement across three simulated sub-agents

Dependencies: stdlib only. No LLM API key required -- the LLM is mocked.

Run:
    python domains/tech/agentic-ai/02-code/15-context-engineering-compaction.py
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# 1. TOKEN COUNTER -- heuristic estimator (no tiktoken dependency)
# ===========================================================================

def estimate_tokens(text: str) -> int:
    """
    Approximate token count for a piece of text.

    Rule of thumb: 1 token ≈ 4 chars (English, as per OpenAI guidance).
    Accurate enough for budget management; not suitable for billing.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a single {"role": ..., "content": ...} message."""
    # Role contributes a small fixed overhead (~4 tokens in real tokenizers)
    role_overhead = 4
    return role_overhead + estimate_tokens(message.get("content", ""))


# ===========================================================================
# 2. VIRTUAL FILESYSTEM -- external agent memory
# ===========================================================================

class VirtualFS:
    """
    In-memory filesystem for external agent memory.

    Why: an agent cannot keep every document in its context window.
    The VirtualFS lets it store structured notes (todo, scratchpad, docs)
    and retrieve them on demand via tool-like read/write methods.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        """Create or overwrite a file."""
        self._store[path] = content

    def read(self, path: str) -> str | None:
        """Return file content or None if not found."""
        return self._store.get(path)

    def list_files(self) -> list[str]:
        """Return sorted list of file paths."""
        return sorted(self._store.keys())

    def token_cost(self, path: str) -> int:
        """How many tokens would it cost to read this file into context."""
        content = self._store.get(path, "")
        return estimate_tokens(content)

    def delete(self, path: str) -> bool:
        """Remove a file. Returns True if it existed."""
        return self._store.pop(path, None) is not None

    def total_tokens(self) -> int:
        """Total tokens stored (not in context, but on disk)."""
        return sum(estimate_tokens(c) for c in self._store.values())


# ===========================================================================
# 3. CONTEXT MANAGER -- history + auto-compaction + offloading
# ===========================================================================

def _mock_summarize(messages: list[dict]) -> str:
    """
    Mock LLM summarization.

    In production: call the LLM with a summarization prompt.
    Here we extract key fields to show what a summary should contain.
    """
    goal = next(
        (m["content"] for m in messages if m.get("role") == "user"), "unknown goal"
    )
    num_turns = len(messages)
    assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
    last_action = assistant_msgs[-1] if assistant_msgs else "none"
    return (
        f"[COMPACTED SUMMARY — {num_turns} messages condensed]\n"
        f"Original goal: {goal[:120]}\n"
        f"Last action taken: {last_action[:120]}\n"
        f"Progress: {num_turns} steps completed."
    )


class ContextManager:
    """
    Manages the message history for a single agent.

    Key responsibilities:
      - Track estimated token usage
      - Trigger compaction when approaching the token limit
      - Offload large blobs to VirtualFS instead of keeping them inline
    """

    # Never compact below this many messages (keep recent context intact)
    MIN_MESSAGES_BEFORE_COMPACT = 4

    def __init__(
        self,
        token_limit: int = 8_000,
        compaction_threshold: float = 0.75,
        vfs: VirtualFS | None = None,
    ) -> None:
        self.token_limit = token_limit
        # Fire compaction when usage exceeds this fraction of the limit
        self.threshold_tokens = int(token_limit * compaction_threshold)
        self.messages: list[dict] = []
        self.vfs: VirtualFS = vfs or VirtualFS()
        self.compaction_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and auto-compact if needed."""
        self.messages.append({"role": role, "content": content})
        if self.total_tokens() > self.threshold_tokens:
            self._compact()

    def offload(self, key: str, content: str) -> str:
        """
        Store large content in the VirtualFS instead of inline.

        Returns a short placeholder string suitable for the context.
        The agent can call vfs.read(key) when it needs the full content.
        """
        self.vfs.write(key, content)
        cost = estimate_tokens(content)
        # The placeholder is tiny (~10 tokens) vs the full content
        return f"[OFFLOADED → {key} ({cost} tokens on disk)]"

    def total_tokens(self) -> int:
        """Estimated token count of the current in-context message list."""
        return sum(estimate_message_tokens(m) for m in self.messages)

    def utilization(self) -> float:
        """Fraction of token_limit used (0.0 .. 1.0+)."""
        return self.total_tokens() / self.token_limit

    def status(self) -> dict:
        return {
            "messages": len(self.messages),
            "tokens_used": self.total_tokens(),
            "token_limit": self.token_limit,
            "utilization_pct": round(self.utilization() * 100, 1),
            "compaction_count": self.compaction_count,
            "vfs_files": self.vfs.list_files(),
            "vfs_tokens_stored": self.vfs.total_tokens(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compact(self) -> None:
        """
        Summarize old messages, keeping only the most recent ones in context.

        Why keep the last N messages fresh? The most recent exchanges contain
        the immediate task state. Summarizing them would lose critical context.
        """
        n = len(self.messages)
        if n <= self.MIN_MESSAGES_BEFORE_COMPACT:
            return  # nothing to compact

        to_summarize = self.messages[: -self.MIN_MESSAGES_BEFORE_COMPACT]
        recent = self.messages[-self.MIN_MESSAGES_BEFORE_COMPACT :]

        summary_text = _mock_summarize(to_summarize)

        # Replace old messages with a single summary message
        self.messages = [
            {"role": "system", "content": summary_text},
            *recent,
        ]
        self.compaction_count += 1


# ===========================================================================
# 4. TOKEN BUDGET -- allocate & track per sub-agent
# ===========================================================================

# Mapping from remaining budget fraction to operational mode
_MODE_THRESHOLDS: list[tuple[float, str]] = [
    (0.30, "normal"),          # > 30% remaining → proceed normally
    (0.15, "economy"),         # 15-30% → lighter tool calls, shorter outputs
    (0.05, "finalize"),        # 5-15%  → wrap up current sub-task only
    (0.00, "return_partial"),  # < 5%   → return partial result immediately
]


def _budget_mode(remaining_fraction: float) -> str:
    """Map remaining budget fraction to a recommended operating mode."""
    for threshold, mode in _MODE_THRESHOLDS:
        if remaining_fraction > threshold:
            return mode
    return "return_partial"


class TokenBudget:
    """
    Allocate a global token budget across named sub-agents.

    Why explicit budgets? Without them a sub-agent can exhaust the global
    quota, leaving sibling agents unable to complete their tasks. Explicit
    allocation makes cost predictable and lets each agent adapt its behavior
    before hitting a hard limit.
    """

    def __init__(self, total: int, allocations: dict[str, float]) -> None:
        """
        Args:
            total:       Total token budget for the whole system.
            allocations: {agent_name: fraction} where fractions sum to ≈1.0.
        """
        assert abs(sum(allocations.values()) - 1.0) < 1e-6, (
            "Allocation fractions must sum to 1.0"
        )
        self.total = total
        self._allocated: dict[str, int] = {
            name: int(total * frac) for name, frac in allocations.items()
        }
        self._used: dict[str, int] = {name: 0 for name in allocations}

    def consume(self, agent: str, tokens: int) -> bool:
        """
        Record token consumption for an agent.

        Returns True if the agent still has budget, False if over-budget.
        The caller should check the return value and switch mode accordingly.
        """
        if agent not in self._used:
            self._used[agent] = 0
            self._allocated[agent] = 0
        self._used[agent] += tokens
        return self._used[agent] <= self._allocated[agent]

    def remaining_fraction(self, agent: str) -> float:
        """Fraction of the agent's allocation that is still available."""
        alloc = self._allocated.get(agent, 0)
        if alloc == 0:
            return 0.0
        used = self._used.get(agent, 0)
        return max(0.0, (alloc - used) / alloc)

    def recommended_mode(self, agent: str) -> str:
        """Return an operating mode string based on remaining budget."""
        return _budget_mode(self.remaining_fraction(agent))

    def status(self) -> dict[str, dict]:
        """Full status snapshot for all registered agents."""
        return {
            name: {
                "allocated": self._allocated[name],
                "used": self._used[name],
                "remaining": self._allocated[name] - self._used[name],
                "remaining_pct": round(self.remaining_fraction(name) * 100, 1),
                "mode": self.recommended_mode(name),
            }
            for name in self._allocated
        }


# ===========================================================================
# 5. BUDGETED AGENT -- ReAct loop with context management and budget checks
# ===========================================================================

@dataclass
class AgentTurn:
    """One ReAct cycle: reason → act → observe."""
    reasoning: str
    action: str
    observation: str
    tokens_this_turn: int


class BudgetedAgent:
    """
    Toy agent that runs a ReAct loop while respecting token budgets.

    Key behaviors:
      - Uses ContextManager to auto-compact its history
      - Calls TokenBudget.consume() after each turn
      - Adjusts verbosity based on the recommended mode
      - Stops gracefully when budget is exhausted
    """

    def __init__(
        self,
        name: str,
        context_manager: ContextManager,
        budget: TokenBudget,
        max_turns: int = 20,
    ) -> None:
        self.name = name
        self.ctx = context_manager
        self.budget = budget
        self.max_turns = max_turns
        self.turns: list[AgentTurn] = []

    def run(self, task: str) -> str:
        """Execute the task, return a result string."""
        self.ctx.add_message("user", task)

        for turn_idx in range(1, self.max_turns + 1):
            mode = self.budget.recommended_mode(self.name)

            if mode == "return_partial":
                # Budget nearly exhausted: stop and return what we have
                result = self._partial_result(turn_idx)
                self.ctx.add_message("assistant", result)
                return result

            # Simulate reasoning and action (mocked LLM output)
            reasoning, action, observation = self._mock_react_step(
                turn_idx, task, mode
            )

            # Estimate tokens for this turn
            turn_text = reasoning + action + observation
            turn_tokens = estimate_tokens(turn_text)

            # Record turn in context
            assistant_content = f"[Turn {turn_idx}] Reasoning: {reasoning}\nAction: {action}"
            self.ctx.add_message("assistant", assistant_content)
            self.ctx.add_message("tool", f"Observation: {observation}")

            # Deduct from budget -- check if we are still in budget
            in_budget = self.budget.consume(self.name, turn_tokens)

            self.turns.append(
                AgentTurn(
                    reasoning=reasoning,
                    action=action,
                    observation=observation,
                    tokens_this_turn=turn_tokens,
                )
            )

            if not in_budget:
                # Over budget: wrap up immediately
                result = self._partial_result(turn_idx)
                self.ctx.add_message("assistant", result)
                return result

            # Simulate task completion after enough turns
            if turn_idx >= 3 and "complete" in observation.lower():
                result = f"Task '{task}' completed in {turn_idx} turns."
                self.ctx.add_message("assistant", result)
                return result

        return f"Max turns reached for task '{task}'."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mock_react_step(
        self, turn: int, task: str, mode: str
    ) -> tuple[str, str, str]:
        """
        Mock one ReAct step. Adjust verbosity based on mode.

        In economy mode we generate shorter outputs to preserve budget.
        """
        if mode == "economy":
            # Shorter outputs to conserve tokens
            reasoning = f"[economy] step {turn}: analyzing..."
            action = "read_file(next_target)"
            observation = f"File {turn} read. 3 findings noted."
        elif mode == "finalize":
            reasoning = f"[finalize] wrapping up step {turn}"
            action = "write_summary()"
            observation = "Summary written. Task complete."
        else:
            # Normal mode: full verbosity
            reasoning = (
                f"Step {turn}: I need to {task}. "
                f"I will examine the relevant files and collect findings."
            )
            action = f"read_file(target_{turn}.py)"
            observation = (
                f"File target_{turn}.py read successfully. "
                f"Found 5 functions, 2 classes, no obvious issues in step {turn}. "
                f"Proceeding to next file."
            )

        return reasoning, action, observation

    def _partial_result(self, turn: int) -> str:
        return (
            f"[PARTIAL RESULT — budget exhausted at turn {turn}] "
            f"Completed {len(self.turns)} turns of '{self.name}' task. "
            f"Context compacted {self.ctx.compaction_count} time(s)."
        )


# ===========================================================================
# 6. DEMO
# ===========================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def demo_context_manager() -> None:
    _banner("Demo 1 — ContextManager: auto-compaction & offloading")

    # Tight limit so compaction triggers visibly in the demo.
    # 150 tokens @ 70% threshold = compact when we exceed 105 tokens.
    ctx = ContextManager(token_limit=150, compaction_threshold=0.70)

    print("\n[Adding messages until compaction triggers...]")
    messages = [
        ("user", "Analyze the security of the Flask application in the repo."),
        ("assistant", "I will start by reading the directory structure of the project."),
        ("tool", "Found: app.py, models.py, routes.py, templates/, tests/"),
        ("assistant", "Reading app.py first since it is the main entry point."),
        ("tool", "app.py content: Flask app initialized, SECRET_KEY loaded from env, 3 blueprints registered, debug=False."),
        ("assistant", "No hardcoded secrets in app.py. Moving to models.py now."),
        ("tool", "models.py content: SQLAlchemy models, User.query.filter_by used in 3 places."),
        ("assistant", "Potential SQL injection risk if raw queries mixed in. Checking routes.py."),
        ("tool", "routes.py content: 12 routes, 2 endpoints use request.args directly without validation."),
        ("assistant", "Found 2 unvalidated inputs in routes.py. This is a significant security finding."),
    ]
    prev_compactions = 0
    for role, content in messages:
        status_before = ctx.total_tokens()
        ctx.add_message(role, content)
        status_after = ctx.total_tokens()
        fired = ctx.compaction_count > prev_compactions
        prev_compactions = ctx.compaction_count
        tag = " ← COMPACTED" if fired else ""
        print(f"  [{role:9}] tokens: {status_before:4d} → {status_after:4d}{tag}")

    print(f"\n  Final context status: {ctx.status()}")

    # Demonstrate offloading a large blob
    large_doc = "# Security Report\n" + ("This is a detailed finding.\n" * 50)
    placeholder = ctx.offload("security_report.md", large_doc)
    print(f"\n[Offloaded large document]\n  Placeholder in context: {placeholder}")
    print(f"  VFS files: {ctx.vfs.list_files()}")
    print(f"  VFS total tokens stored: {ctx.vfs.total_tokens()}")

    # Show the agent can still retrieve the full content
    retrieved = ctx.vfs.read("security_report.md")
    print(f"  Retrieved first 80 chars: {retrieved[:80]!r}")


def demo_virtual_fs() -> None:
    _banner("Demo 2 — VirtualFS: todo list & scratchpad pattern")

    vfs = VirtualFS()

    # Agent writes its todo list at session start
    vfs.write("todo.md", textwrap.dedent("""\
        ## Objectif : auditer repo Flask
        - [x] Lire la structure
        - [x] Analyser app.py
        - [ ] Analyser models.py
        - [ ] Analyser routes.py
        - [ ] Lancer bandit
        - [ ] Rediger le rapport
    """))

    # Agent writes findings to scratchpad instead of keeping them in context
    vfs.write("scratchpad.md", textwrap.dedent("""\
        Hypothese : faille XSS dans templates Jinja2
        Verification : templates scannés → aucun filtre |safe abusif → OK
        Faille confirmee : request.args sans validation dans routes.py ligne 47
        Next : verifier si c'est aussi present dans l'API REST
    """))

    # Simulate the agent updating its todo list mid-task
    current_todo = vfs.read("todo.md")
    updated_todo = current_todo.replace(
        "- [ ] Analyser models.py", "- [x] Analyser models.py"
    )
    vfs.write("todo.md", updated_todo)

    print("\n[VirtualFS contents]")
    for path in vfs.list_files():
        content = vfs.read(path)
        print(f"\n  {path} ({vfs.token_cost(path)} tokens on disk):")
        for line in content.splitlines()[:6]:
            print(f"    {line}")

    print(f"\n  Total VFS storage: {vfs.total_tokens()} tokens")
    print(f"  (None of this is in the context window right now)")


def demo_token_budget() -> None:
    _banner("Demo 3 — TokenBudget: allocation & mode recommendations")

    budget = TokenBudget(
        total=100_000,
        allocations={
            "orchestrator":    0.10,   # 10 000 tokens — coordination only
            "file_analyzer":   0.35,   # 35 000 tokens — reads many files
            "security_scanner": 0.25,  # 25 000 tokens — analysis
            "report_writer":   0.30,   # 30 000 tokens — text generation
        },
    )

    print("\n[Initial budget allocation]")
    for agent, info in budget.status().items():
        print(f"  {agent:20} allocated={info['allocated']:>6}  mode={info['mode']}")

    # Simulate consumption
    print("\n[Simulating token consumption...]")
    consumptions = [
        ("file_analyzer",   20_000),   # 57% used → normal
        ("file_analyzer",   10_000),   # 86% used → economy
        ("security_scanner", 20_000),  # 80% used → economy
        ("report_writer",    28_500),  # 95% used → finalize
        ("orchestrator",      9_800),  # 98% used → return_partial
    ]
    for agent, tokens in consumptions:
        in_budget = budget.consume(agent, tokens)
        mode = budget.recommended_mode(agent)
        pct = round(budget.remaining_fraction(agent) * 100, 1)
        print(
            f"  {agent:20} consumed {tokens:>6} | "
            f"remaining={pct:5.1f}% | mode={mode} | in_budget={in_budget}"
        )

    print("\n[Final budget status]")
    for agent, info in budget.status().items():
        bar = "#" * int(info["remaining_pct"] / 5) + "." * (20 - int(info["remaining_pct"] / 5))
        print(
            f"  {agent:20} [{bar}] {info['remaining_pct']:5.1f}% left | mode={info['mode']}"
        )


def demo_budgeted_agent() -> None:
    _banner("Demo 4 — BudgetedAgent: end-to-end ReAct with auto-compaction")

    vfs = VirtualFS()
    budget = TokenBudget(
        total=10_000,
        allocations={
            "researcher": 0.60,
            "writer":     0.40,
        },
    )

    # Agent with a tight context limit to trigger compaction quickly.
    # With token_limit=300 and threshold=0.70 we compact around 210 tokens.
    ctx_researcher = ContextManager(
        token_limit=300,
        compaction_threshold=0.70,
        vfs=vfs,
    )

    researcher = BudgetedAgent(
        name="researcher",
        context_manager=ctx_researcher,
        budget=budget,
        max_turns=8,
    )

    print("\n[Running researcher agent (budget=6000 tokens)...]")
    result = researcher.run("Audit the Flask security vulnerabilities in the repo")
    print(f"\n  Result: {result}")
    print(f"  Turns completed: {len(researcher.turns)}")
    print(f"  Context compactions: {ctx_researcher.compaction_count}")
    remaining_r = budget.remaining_fraction("researcher") * 100
    print(f"  Budget remaining: {remaining_r:.1f}%")
    print(f"  Mode: {budget.recommended_mode('researcher')}")

    # Second agent using remaining budget — note tighter allocation (4000 tokens)
    ctx_writer = ContextManager(
        token_limit=400,
        compaction_threshold=0.75,
        vfs=vfs,
    )
    writer = BudgetedAgent(
        name="writer",
        context_manager=ctx_writer,
        budget=budget,
        max_turns=5,
    )
    print("\n[Running writer agent (budget=4000 tokens)...]")
    result2 = writer.run("Write a security report based on the researcher findings")
    print(f"\n  Result: {result2}")
    remaining_w = budget.remaining_fraction("writer") * 100
    print(f"  Budget remaining for writer: {remaining_w:.1f}%")

    print("\n[Global budget summary]")
    for agent, info in budget.status().items():
        print(
            f"  {agent:12} used={info['used']:>5} / {info['allocated']:>5} "
            f"({info['remaining_pct']}% left, mode={info['mode']})"
        )


if __name__ == "__main__":
    demo_context_manager()
    demo_virtual_fs()
    demo_token_budget()
    demo_budgeted_agent()

    print("\n" + "=" * 70)
    print("  All demos completed successfully.")
    print("=" * 70)

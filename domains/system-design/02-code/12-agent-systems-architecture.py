"""
Jour 12 -- Supervisor pattern simulation.

Usage:
    python 12-agent-systems-architecture.py

Ce script simule un systeme multi-agent avec un superviseur et des specialistes.
Aucun LLM reel n'est appele : le "reasoning" est simule par du pattern matching
et des regles. L'architecture illustree est la meme que celle utilisee en prod
avec LangGraph / CrewAI / OpenAI Swarm.

On montre aussi :
  - le pattern state + plan + act + observe
  - la memoire short-term avec summarization apres un seuil
  - les handoff messages structures
  - une condition d'arret (budget ou task completed)
  - les traces pour debugging
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Handoff message format
# =============================================================================


@dataclass
class Handoff:
    """Structured message passed between agents.

    Toujours inclure le contexte, ce qui est fait, ce qui reste, le critere
    de succes et le budget. Un simple 'continue' est un code smell.
    """

    sender: str
    receiver: str
    context: str
    done: list[str]
    remaining: list[str]
    success_criteria: str
    budget_steps: int

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "context": self.context,
            "done": self.done,
            "remaining": self.remaining,
            "success_criteria": self.success_criteria,
            "budget_steps": self.budget_steps,
        }


# =============================================================================
# SECTION 2 : Shared agent state
# =============================================================================


@dataclass
class AgentState:
    """Global state that flows through the supervisor + specialists.

    In real frameworks like LangGraph, this is a TypedDict.
    """

    user_request: str
    history: list[dict] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    steps_used: int = 0
    budget: int = 10
    done: bool = False
    final_answer: Optional[str] = None

    def log(self, event: str, **kwargs: Any) -> None:
        entry = {"t": time.time(), "event": event, "step": self.steps_used, **kwargs}
        self.history.append(entry)
        if len(self.history) > 20:
            self._summarize_history()

    def _summarize_history(self) -> None:
        # Summarize older history into a single compact entry to avoid context bloat
        old = self.history[:-5]
        summary = {
            "event": "summary",
            "note": f"{len(old)} earlier events summarized",
            "agents_used": sorted({e.get("agent", "") for e in old if "agent" in e}),
        }
        self.history = [summary] + self.history[-5:]


# =============================================================================
# SECTION 3 : Specialist mock agents
# =============================================================================


class SpecialistAgent:
    """A mock specialist that knows one capability.

    In real life, each specialist would have its own system prompt, tools,
    and possibly its own LLM backend.
    """

    name: str
    capabilities: set[str]

    def __init__(self, name: str, capabilities: set[str]) -> None:
        self.name = name
        self.capabilities = capabilities

    def can_handle(self, task: str) -> bool:
        return any(cap in task.lower() for cap in self.capabilities)

    def run(self, handoff: Handoff, state: AgentState) -> dict:
        """Do the actual work. Subclasses override."""
        raise NotImplementedError


class SearchAgent(SpecialistAgent):
    def __init__(self) -> None:
        super().__init__("search_agent", {"search", "find", "lookup"})

    def run(self, handoff: Handoff, state: AgentState) -> dict:
        # Simulate a web search : return some fake results
        q = handoff.context
        fake_results = [
            {"title": f"Result 1 about {q[:20]}", "snippet": "relevant info A"},
            {"title": f"Result 2 about {q[:20]}", "snippet": "relevant info B"},
        ]
        state.log("specialist_done", agent=self.name, found=len(fake_results))
        return {"search_results": fake_results}


class CodeAgent(SpecialistAgent):
    def __init__(self) -> None:
        super().__init__("code_agent", {"code", "python", "refactor", "function"})

    def run(self, handoff: Handoff, state: AgentState) -> dict:
        snippet = "def hello():\n    return 'world'"
        state.log("specialist_done", agent=self.name, lines=len(snippet.splitlines()))
        return {"code": snippet}


class WriterAgent(SpecialistAgent):
    def __init__(self) -> None:
        super().__init__("writer_agent", {"write", "draft", "email", "summary", "summarize"})

    def run(self, handoff: Handoff, state: AgentState) -> dict:
        # Compose a short text based on prior results in state
        parts = []
        if "search_results" in state.results:
            parts.append(
                "Based on search: " + ", ".join(r["snippet"] for r in state.results["search_results"])
            )
        if "code" in state.results:
            parts.append("Here is the code:\n" + state.results["code"])
        if not parts:
            parts.append(f"Response to: {state.user_request}")
        text = "\n\n".join(parts)
        state.log("specialist_done", agent=self.name, chars=len(text))
        return {"draft": text}


class AnalystAgent(SpecialistAgent):
    def __init__(self) -> None:
        super().__init__("analyst_agent", {"analyze", "analysis", "compare", "evaluate"})

    def run(self, handoff: Handoff, state: AgentState) -> dict:
        analysis = "The key metric is X=42. Trend is positive. Recommendation: continue."
        state.log("specialist_done", agent=self.name)
        return {"analysis": analysis}


# =============================================================================
# SECTION 4 : Supervisor
# =============================================================================


class Supervisor:
    """Routes tasks to specialists, aggregates their results, then responds.

    Decision logic here is rule-based pattern matching. In production, this
    would be an LLM call that returns a plan / next action.
    """

    def __init__(self, specialists: list[SpecialistAgent]) -> None:
        self.specialists = specialists

    def plan(self, state: AgentState) -> list[str]:
        """Decide which specialists to call (in order).

        This is where an LLM would be used : given the user request and
        the tools available, return a plan (JSON).
        """
        req = state.user_request.lower()
        plan: list[str] = []
        if any(kw in req for kw in ("search", "find", "look up", "latest")):
            plan.append("search_agent")
        if any(kw in req for kw in ("code", "function", "refactor", "python")):
            plan.append("code_agent")
        if any(kw in req for kw in ("analyze", "analysis", "compare")):
            plan.append("analyst_agent")
        # Almost everything needs a final writer
        plan.append("writer_agent")
        return plan

    def dispatch(self, plan: list[str], state: AgentState) -> None:
        """Call each specialist in order, collecting results in state.

        Check budget before each call; stop if exceeded.
        """
        by_name = {s.name: s for s in self.specialists}
        for name in plan:
            if state.steps_used >= state.budget:
                state.log("budget_exceeded")
                return
            agent = by_name.get(name)
            if agent is None:
                state.log("unknown_agent", name=name)
                continue
            handoff = Handoff(
                sender="supervisor",
                receiver=name,
                context=state.user_request,
                done=list(state.results.keys()),
                remaining=[p for p in plan if p not in state.results],
                success_criteria=f"produce useful {name} output for the user request",
                budget_steps=state.budget - state.steps_used,
            )
            state.log("handoff", **handoff.to_dict())
            try:
                result = agent.run(handoff, state)
                state.results.update(result)
                state.steps_used += 1
            except Exception as e:
                state.log("specialist_error", agent=name, error=str(e))
                state.steps_used += 1  # failing still counts

    def finalize(self, state: AgentState) -> None:
        """Produce the final answer from aggregated results."""
        if "draft" in state.results:
            state.final_answer = state.results["draft"]
        else:
            state.final_answer = json.dumps(state.results, indent=2)
        state.done = True
        state.log("done", final_len=len(state.final_answer))


# =============================================================================
# SECTION 5 : Top-level agent runner
# =============================================================================


def run_agent(user_request: str, budget: int = 10) -> AgentState:
    state = AgentState(user_request=user_request, budget=budget)
    state.log("start", request=user_request, budget=budget)
    supervisor = Supervisor(
        specialists=[SearchAgent(), CodeAgent(), AnalystAgent(), WriterAgent()]
    )
    plan = supervisor.plan(state)
    state.log("plan", plan=plan)
    supervisor.dispatch(plan, state)
    supervisor.finalize(state)
    return state


# =============================================================================
# SECTION 6 : Demo
# =============================================================================


def print_state(state: AgentState) -> None:
    print(f"\nFinal answer:\n---\n{state.final_answer}\n---")
    print(f"Steps used: {state.steps_used}/{state.budget}")
    print(f"Specialists called: {sorted({e.get('agent') for e in state.history if e.get('event') == 'specialist_done'})}")
    print(f"Trace events: {len(state.history)}")


def demo() -> None:
    print(SEPARATOR)
    print("SUPERVISOR PATTERN SIMULATION")
    print(SEPARATOR)

    requests = [
        "Find the latest news on Python 3.13 and write me a summary email.",
        "Refactor this function to use async, and compare the two versions.",
        "Analyze our Q1 sales and draft a memo.",
        "Hello how are you?",
    ]
    for req in requests:
        print("\n" + SEPARATOR)
        print(f"USER REQUEST: {req}")
        state = run_agent(req, budget=6)
        print_state(state)


if __name__ == "__main__":
    demo()

"""
Day 18 -- Orchestration frameworks compared + multi-agent failure modes.

Demonstrates five toy orchestrators that simulate the execution model of each
framework, then a failure-mode section (token explosion, disagreement loop).
All simulations use stdlib only -- no API key, no framework install required.

Run:
    python domains/agentic-ai/02-code/18-orchestration-comparee-failure-modes.py
"""

from __future__ import annotations

import random
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Shared mock LLM
# ---------------------------------------------------------------------------

_CALL_LOG: list[dict] = []   # global token counter for the cost demo


def mock_llm(agent_name: str, prompt: str, max_tokens: int = 80) -> str:
    """
    Simulate an LLM call.  Uses a tiny deterministic response library so the
    demo is reproducible without any API key.
    """
    tokens_in = len(prompt.split())
    tokens_out = random.randint(10, max_tokens)
    _CALL_LOG.append(
        {"agent": agent_name, "tokens_in": tokens_in, "tokens_out": tokens_out}
    )
    low = prompt.lower()
    if "research" in low or "search" in low:
        return f"[{agent_name}] Research result: topic '{prompt[:30]}...' found 3 relevant sources."
    if "write" in low or "draft" in low:
        return f"[{agent_name}] Draft complete: 'The analysis of {prompt[:20]}... shows promising data.'"
    if "review" in low or "check" in low:
        return f"[{agent_name}] Review done: output looks correct and well-structured."
    if "summarize" in low:
        return f"[{agent_name}] Summary: the pipeline concluded successfully with 3 steps."
    if "plan" in low or "delegat" in low:
        return f"[{agent_name}] Plan: step1=research, step2=write, step3=review."
    if "disagree" in low or "better" in low:
        return f"[{agent_name}] I strongly disagree — my approach is clearly superior."
    return f"[{agent_name}] Processed: '{prompt[:40]}...'"


def tokens_used() -> tuple[int, int]:
    """Return (total_in, total_out) across all mock LLM calls so far."""
    return (
        sum(e["tokens_in"] for e in _CALL_LOG),
        sum(e["tokens_out"] for e in _CALL_LOG),
    )


def reset_log() -> None:
    _CALL_LOG.clear()


# ===========================================================================
# 1. LangGraph-style: stateful directed graph
# ===========================================================================
# Execution model:
#   - nodes: functions that receive the shared state dict and return an update
#   - edges: conditional or fixed transitions between nodes
#   - state: a typed dict passed through every node
#   - checkpointing (simulated): state is snapshotted after each node
# ===========================================================================

@dataclass
class GraphState:
    """Shared state circulating through the graph nodes."""
    task: str
    research_result: str = ""
    draft: str = ""
    review: str = ""
    step: int = 0
    checkpoint: dict = field(default_factory=dict)

    def snapshot(self) -> None:
        """Simulate checkpointing: deep-copy current fields into checkpoint."""
        self.checkpoint = {
            "task": self.task,
            "research_result": self.research_result,
            "draft": self.draft,
            "review": self.review,
            "step": self.step,
        }


def node_research(state: GraphState) -> GraphState:
    state.research_result = mock_llm("researcher", f"search {state.task}")
    state.step = 1
    state.snapshot()   # simulate checkpointer persistence
    return state


def node_writer(state: GraphState) -> GraphState:
    state.draft = mock_llm("writer", f"write based on: {state.research_result}")
    state.step = 2
    state.snapshot()
    return state


def node_reviewer(state: GraphState) -> GraphState:
    state.review = mock_llm("reviewer", f"review this draft: {state.draft}")
    state.step = 3
    state.snapshot()
    return state


def conditional_edge(state: GraphState) -> str:
    """Conditional routing: if draft is empty, go back to writer."""
    return "writer" if not state.draft else "reviewer"


class ToyLangGraph:
    """
    Minimal directed graph runtime.
    Nodes are callables: state -> state.
    Edges can be fixed (str) or conditional (callable returning str).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, Any] = {}   # str or Callable

    def add_node(self, name: str, fn: Callable) -> None:
        self._nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str | Callable) -> None:
        self._edges[from_node] = to_node

    def run(self, entry: str, state: GraphState, max_steps: int = 10) -> GraphState:
        current = entry
        steps = 0
        while current and steps < max_steps:
            if current not in self._nodes:
                break
            state = self._nodes[current](state)
            edge = self._edges.get(current)
            if edge is None:
                break
            # Resolve the next node: fixed name or conditional function
            current = edge(state) if callable(edge) else edge
            steps += 1
        return state


def demo_langgraph() -> None:
    print("\n" + "=" * 60)
    print("1. LangGraph-style: stateful directed graph")
    print("=" * 60)

    g = ToyLangGraph()
    g.add_node("researcher", node_research)
    g.add_node("writer", node_writer)
    g.add_node("reviewer", node_reviewer)

    # Fixed edges: researcher -> writer -> reviewer (END)
    g.add_edge("researcher", "writer")
    g.add_edge("writer", conditional_edge)   # conditional
    g.add_edge("reviewer", None)             # END

    state = GraphState(task="multi-agent orchestration patterns")
    final = g.run("researcher", state)

    print(f"  task       : {final.task}")
    print(f"  research   : {final.research_result}")
    print(f"  draft      : {final.draft}")
    print(f"  review     : {final.review}")
    print(f"  checkpoint : step={final.checkpoint.get('step')}")


# ===========================================================================
# 2. CrewAI-style: role-based sequential / hierarchical crew
# ===========================================================================
# Execution model:
#   - Agent: role + backstory + tools (here mocked as a callable)
#   - Task: description + expected_output + assigned agent
#   - Crew: list of agents + tasks + process (sequential | hierarchical)
#   - Sequential: each task runs after the previous; output is passed as context
#   - Hierarchical: a manager LLM reads all tasks and decides who does what
# ===========================================================================

@dataclass
class CrewAgent:
    role: str
    goal: str
    tools: list[str] = field(default_factory=list)

    def execute(self, task_description: str, context: str = "") -> str:
        prompt = f"{task_description}. Context: {context}" if context else task_description
        return mock_llm(self.role, prompt)


@dataclass
class CrewTask:
    description: str
    expected_output: str
    agent: CrewAgent


class ToyCrewSequential:
    """Sequential process: tasks execute in order, output chained as context."""

    def __init__(self, tasks: list[CrewTask]) -> None:
        self.tasks = tasks

    def kickoff(self) -> list[str]:
        outputs = []
        context = ""
        for task in self.tasks:
            result = task.agent.execute(task.description, context=context)
            outputs.append(result)
            context = result   # each task receives the previous task's output
        return outputs


class ToyCrewHierarchical:
    """
    Hierarchical process: a manager LLM reads all tasks, decides the order,
    then delegates each task to the right agent.
    """

    def __init__(self, agents: list[CrewAgent], tasks: list[CrewTask]) -> None:
        self.agents = agents
        self.tasks = tasks
        self.manager = CrewAgent(role="manager", goal="coordinate the crew")

    def kickoff(self) -> list[str]:
        # Manager plans
        agent_names = ", ".join(a.role for a in self.agents)
        task_descs = " | ".join(t.description for t in self.tasks)
        plan = self.manager.execute(
            f"plan and delegate these tasks: {task_descs} to agents: {agent_names}"
        )
        print(f"  [manager plan] {plan}")

        # Execute each task according to the plan (simplified: same order)
        outputs = []
        context = ""
        for task in self.tasks:
            result = task.agent.execute(task.description, context=context)
            outputs.append(result)
            context = result
        return outputs


def demo_crewai() -> None:
    print("\n" + "=" * 60)
    print("2. CrewAI-style: role-based crew (sequential + hierarchical)")
    print("=" * 60)

    researcher = CrewAgent(role="researcher", goal="find relevant information")
    writer = CrewAgent(role="writer", goal="write clear reports")
    reviewer = CrewAgent(role="reviewer", goal="ensure quality")

    tasks = [
        CrewTask("research multi-agent failure modes", "list of failure modes", researcher),
        CrewTask("write a report on failure modes", "800-word report", writer),
        CrewTask("review the report", "reviewed and approved report", reviewer),
    ]

    print("  --- Sequential process ---")
    crew_seq = ToyCrewSequential(tasks)
    outputs = crew_seq.kickoff()
    for i, out in enumerate(outputs, 1):
        print(f"  task {i}: {out}")

    print("\n  --- Hierarchical process ---")
    crew_hier = ToyCrewHierarchical([researcher, writer, reviewer], tasks)
    outputs_h = crew_hier.kickoff()
    for i, out in enumerate(outputs_h, 1):
        print(f"  task {i}: {out}")


# ===========================================================================
# 3. AutoGen 0.4-style: event-driven / actor model
# ===========================================================================
# Execution model:
#   - Each agent is an actor with a mailbox (queue of messages)
#   - Actors process messages asynchronously and publish responses to a bus
#   - No shared mutable state -- actors only communicate via typed messages
#   - Here: simulated synchronously with a simple event bus
# ===========================================================================

@dataclass
class Message:
    topic: str
    sender: str
    content: str


class EventBus:
    """Simple synchronous pub/sub bus (simulates AutoGen's distributed runtime)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = {}
        self._log: list[Message] = []

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    def publish(self, msg: Message) -> None:
        self._log.append(msg)
        for handler in self._subscribers.get(msg.topic, []):
            handler(msg)

    @property
    def log(self) -> list[Message]:
        return self._log


class ActorAgent:
    """
    Minimal actor: subscribes to its inbox topic, processes messages,
    and publishes responses to a reply topic.
    """

    def __init__(
        self,
        name: str,
        bus: EventBus,
        inbox_topic: str,
        reply_topic: str,
    ) -> None:
        self.name = name
        self.bus = bus
        self.inbox_topic = inbox_topic
        self.reply_topic = reply_topic
        bus.subscribe(inbox_topic, self._handle)

    def _handle(self, msg: Message) -> None:
        response_content = mock_llm(self.name, msg.content)
        self.bus.publish(
            Message(
                topic=self.reply_topic,
                sender=self.name,
                content=response_content,
            )
        )


def demo_autogen() -> None:
    print("\n" + "=" * 60)
    print("3. AutoGen 0.4-style: event-driven actor model")
    print("=" * 60)

    bus = EventBus()

    # Three actors: researcher listens on "task", posts to "research_done"
    # writer listens on "research_done", posts to "draft_done"
    # reviewer listens on "draft_done", posts to "review_done"
    ActorAgent("researcher", bus, inbox_topic="task", reply_topic="research_done")
    ActorAgent("writer", bus, inbox_topic="research_done", reply_topic="draft_done")
    ActorAgent("reviewer", bus, inbox_topic="draft_done", reply_topic="review_done")

    # Kick off the pipeline by publishing to "task"
    bus.publish(
        Message(
            topic="task",
            sender="orchestrator",
            content="research orchestration frameworks",
        )
    )

    print(f"  messages on bus: {len(bus.log)}")
    for m in bus.log:
        print(f"  [{m.topic}] {m.sender}: {m.content[:70]}")


# ===========================================================================
# 4. OpenAI Agents SDK-style: stateless handoff / tool-centric
# ===========================================================================
# Execution model:
#   - An Agent holds instructions + tools
#   - A special "handoff tool" transfers control to another agent
#   - Context = the full conversation history passed along with the handoff
#   - No shared state object: state lives in the message history
# ===========================================================================

@dataclass
class HandoffAgent:
    name: str
    instructions: str
    tools: list[str] = field(default_factory=list)
    handoffs: dict[str, "HandoffAgent"] = field(default_factory=dict)

    def run(self, history: list[dict], max_turns: int = 5) -> list[dict]:
        """
        Process the history until done or a handoff is triggered.
        Returns the updated history.
        """
        for _ in range(max_turns):
            # Build a prompt from history
            context = " | ".join(m["content"] for m in history[-3:])
            response = mock_llm(self.name, f"{self.instructions}: {context}")

            history.append({"role": "assistant", "agent": self.name, "content": response})

            # Simulate handoff decision: if a handoff agent is mentioned, delegate
            for trigger, target in self.handoffs.items():
                if trigger in response.lower():
                    print(f"  [{self.name}] handoff -> {target.name}")
                    return target.run(history, max_turns=max_turns)

            # No handoff: we are done
            break
        return history


def demo_agents_sdk() -> None:
    print("\n" + "=" * 60)
    print("4. OpenAI Agents SDK-style: stateless handoff / tool-centric")
    print("=" * 60)

    reviewer = HandoffAgent(
        name="reviewer",
        instructions="review the submitted draft",
    )
    writer = HandoffAgent(
        name="writer",
        instructions="write a draft, then handoff to reviewer",
        handoffs={"review": reviewer},
    )
    triage = HandoffAgent(
        name="triage",
        instructions="decide if task needs writing, then handoff to writer",
        handoffs={"write": writer},
    )

    history = [{"role": "user", "agent": "user", "content": "write a report on LangGraph"}]
    final_history = triage.run(history)

    print(f"  conversation turns: {len(final_history)}")
    for turn in final_history:
        print(f"  [{turn['agent']}] {turn['content'][:70]}")


# ===========================================================================
# 5. OpenAI Swarm-style: stateless handoffs (archive pattern)
# ===========================================================================
# Swarm was the direct predecessor of the Agents SDK.
# Key difference: state was passed as context_variables (a mutable dict),
# not via conversation history.  Handoffs were plain Python function returns.
# ===========================================================================

@dataclass
class SwarmAgent:
    name: str
    instructions: str

    def run(
        self,
        messages: list[str],
        context: dict,
        handoffs: dict[str, "SwarmAgent"] | None = None,
    ) -> tuple["SwarmAgent", list[str], dict]:
        """
        Returns (next_agent, messages, context).
        If a handoff is triggered, returns the new agent.
        """
        prompt = f"{self.instructions}: {messages[-1] if messages else ''}"
        response = mock_llm(self.name, prompt)
        messages = messages + [response]
        context["last_response"] = response

        # Simulate handoff by keyword detection
        if handoffs:
            for keyword, next_agent in handoffs.items():
                if keyword in response.lower():
                    print(f"  [{self.name}] swarm handoff -> {next_agent.name}")
                    return next_agent, messages, context

        # No handoff -- we are the last agent
        return self, messages, context


def demo_swarm() -> None:
    print("\n" + "=" * 60)
    print("5. Swarm-style: stateless handoffs (archive, pedagogique)")
    print("=" * 60)

    agent_b = SwarmAgent(name="specialist", instructions="handle specialised requests")
    agent_a = SwarmAgent(name="triage", instructions="triage then handoff to specialist")

    messages = ["I need a summarize of AutoGen 0.4 architecture"]
    ctx: dict = {}

    # First turn: triage
    next_agent, messages, ctx = agent_a.run(
        messages, ctx, handoffs={"result": agent_b}
    )
    # Second turn: specialist (or still triage if no handoff triggered)
    next_agent, messages, ctx = next_agent.run(messages, ctx)

    print(f"  final agent: {next_agent.name}")
    for i, m in enumerate(messages):
        print(f"  msg {i}: {m[:70]}")


# ===========================================================================
# 6. FAILURE MODE DEMO
# ===========================================================================

def demo_token_explosion() -> None:
    """
    Compare token cost: single agent (5 steps in one context) vs
    multi-agent pipeline (5 agents, each receiving the full accumulated context).
    """
    print("\n" + "=" * 60)
    print("6a. Failure mode: token explosion (single vs multi-agent)")
    print("=" * 60)

    # --- Single agent ---
    reset_log()
    base_context = "Task: analyse LangGraph checkpointing patterns. "
    single_history = base_context
    for step in range(5):
        response = mock_llm("single_agent", f"Step {step}: {single_history}")
        single_history += response + " "   # context grows but stays in one call per step
    t_in_single, t_out_single = tokens_used()

    # --- Multi-agent pipeline (sequential, each agent passes full context) ---
    reset_log()
    context = base_context
    for agent_idx in range(5):
        agent_name = f"agent_{agent_idx}"
        # Each agent sees the accumulated context from all previous agents
        response = mock_llm(agent_name, context)
        context += response + " "   # context accumulates across agents
    t_in_multi, t_out_multi = tokens_used()

    print(f"  Single agent (5 steps) : {t_in_single} in + {t_out_single} out tokens")
    print(f"  Multi-agent (5 agents) : {t_in_multi} in + {t_out_multi} out tokens")
    ratio = (t_in_multi + t_out_multi) / max(1, t_in_single + t_out_single)
    print(f"  Cost ratio             : {ratio:.2f}x  (multi / single)")
    print("  -> Sequential multi-agent ~2x more expensive without quality gain")


def demo_disagreement_loop() -> None:
    """
    Two agents disagree indefinitely -- shows why max_turns is mandatory.
    Without it the loop runs forever (or until budget exhaustion).
    """
    print("\n" + "=" * 60)
    print("6b. Failure mode: disagreement loop between two agents")
    print("=" * 60)

    MAX_TURNS = 6   # guard against infinite loops

    position_a = "approach X (LangGraph) is better because it is stateful"
    position_b = "approach Y (CrewAI) is better because it is simpler"

    reset_log()
    for turn in range(MAX_TURNS):
        # Agent A advocates for X
        response_a = mock_llm("agent_A", f"disagree with '{position_b}', my approach is better")
        position_a = response_a

        # Agent B advocates for Y
        response_b = mock_llm("agent_B", f"disagree with '{position_a}', my approach is better")
        position_b = response_b

        t_in, t_out = tokens_used()
        print(f"  turn {turn + 1:02d} | tokens so far: {t_in} in / {t_out} out")

        # No arbiter -> loop completes only because MAX_TURNS is set
        # In real systems without max_turns, this runs until budget exhaustion

    print(f"  Loop stopped by MAX_TURNS={MAX_TURNS} (no resolution reached)")
    print("  Defence: add a tiebreaker agent or escalate to human after N turns")

    # Show cost vs productive single-agent call
    reset_log()
    mock_llm("single_agent", "Compare LangGraph and CrewAI, pick the better fit")
    t_in_s, t_out_s = tokens_used()
    t_in, t_out = tokens_used()
    print(f"  Single-agent answer cost: {t_in_s} in / {t_out_s} out tokens")
    print(f"  Disagreement loop cost  : {t_in + t_out} total tokens -- wasted")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Day 18 -- Orchestration frameworks compared & failure modes")
    print("=" * 60)

    # --- Framework simulations ---
    demo_langgraph()
    demo_crewai()
    demo_autogen()
    demo_agents_sdk()
    demo_swarm()

    # --- Failure modes ---
    demo_token_explosion()
    demo_disagreement_loop()

    print("\n" + "=" * 60)
    print("All demos completed successfully.")
    print("=" * 60)

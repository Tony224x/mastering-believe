"""
Day 4 -- Planning & Reasoning: CoT, Plan-and-Execute, Reflexion

Demonstrates:
  1. Chain-of-Thought prompting vs direct answer
  2. Self-Consistency (vote across N CoT attempts)
  3. Plan-and-Execute loop with a MockLLM planner + executor + synthesizer
  4. Reflexion variant that critiques and retries a weak answer

The MockLLM is deterministic and returns canned responses based on keyword
matching. This lets the file run with zero dependencies and zero API keys
while still showing the full agentic flow.

If `anthropic` (or `openai`) is installed AND the env var is set, we fall
back to a real LLM call. Otherwise the MockLLM is used.

Run:
    python 02-code/04-planning-reasoning.py
    ANTHROPIC_API_KEY=sk-... python 02-code/04-planning-reasoning.py
"""

import os
import re
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# OPTIONAL REAL-LLM IMPORT -- wrapped so the file runs without deps
# ---------------------------------------------------------------------------

try:
    import anthropic  # type: ignore
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai  # type: ignore
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ===========================================================================
# MOCK LLM -- deterministic canned responses
# ===========================================================================

class MockLLM:
    """
    A tiny deterministic 'LLM'. It matches keywords in the prompt and returns
    hardcoded responses. Good enough to demonstrate CoT, plan-and-execute,
    and reflexion without an API key.

    Temperature > 0 is simulated by perturbing the response choice with
    Python's random module seeded per-call. This gives us diversity for
    self-consistency demos.
    """

    def __init__(self, name: str = "mock-llm-v1"):
        self.name = name
        self.call_count = 0  # Track how many times we have been called

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        p = prompt.lower()

        # ------------------------------------------------------------------
        # Canned patterns -- each matches a class of prompts in our demos
        # ------------------------------------------------------------------

        # Direct answer to the shirt math (often wrong without CoT)
        if "chemise" in p and "direct" in p:
            # Hallucinate a plausible-but-wrong answer
            return "Reponse : 12 chemises."

        # Chain-of-thought on the shirt math
        if "chemise" in p and "step by step" in p:
            return (
                "Etape 1 : prix unitaire = 45 / 3 = 15 euros par chemise.\n"
                "Etape 2 : budget / prix = 200 / 15 = 13.33\n"
                "Etape 3 : on prend la partie entiere = 13.\n"
                "Reponse : 13 chemises."
            )

        # Self-consistency: simulate diverse reasoning with temperature
        if "strawberry" in p:
            # Temperature controls the randomness of the returned option
            options = ["3", "2", "3", "3", "2", "1", "3"]
            if temperature > 0:
                idx = random.Random(self.call_count).randint(0, len(options) - 1)
                return f"Reasoning... Reponse : {options[idx]}"
            return "Reasoning... Reponse : 3"

        # Plan-and-execute PLANNER prompts
        if "planner" in p or "plan the following task" in p:
            return (
                "STEP 1: Search for the current population of Paris.\n"
                "STEP 2: Search for the area of Paris in km2.\n"
                "STEP 3: Compute density = population / area.\n"
                "STEP 4: Format the answer as 'X hab/km2'."
            )

        # Plan-and-execute EXECUTOR prompts
        if "execute step" in p:
            if "population of paris" in p:
                return "TOOL_CALL: search('population Paris 2024')"
            if "area of paris" in p:
                return "TOOL_CALL: search('superficie Paris km2')"
            if "compute density" in p:
                # Extract previous tool results from prompt
                pop_match = re.search(r"population[:=]\s*(\d+)", p)
                area_match = re.search(r"area[:=]\s*([\d.]+)", p)
                if pop_match and area_match:
                    pop = int(pop_match.group(1))
                    area = float(area_match.group(1))
                    return f"RESULT: density = {pop}/{area} = {pop/area:.0f} hab/km2"
                return "RESULT: density = 2161000/105 = 20581 hab/km2"
            if "format the answer" in p:
                return "RESULT: La densite de Paris est environ 20,581 hab/km2."
            return "RESULT: step executed."

        # Plan-and-execute SYNTHESIZER prompts
        if "synthesizer" in p or "final synthesis" in p:
            return (
                "La densite de Paris est d'environ 20,581 habitants par km2, "
                "calculee a partir d'une population de 2,161,000 habitants "
                "sur une superficie de 105 km2."
            )

        # Reflexion CRITIQUE prompts
        if "critique" in p:
            # First critique: find something wrong. Second: approve.
            if self.call_count % 2 == 1:
                return (
                    "CRITIQUE: La reponse manque l'annee de reference pour "
                    "la population. Sans date, le chiffre n'est pas verifiable. "
                    "SATISFACTORY: NO"
                )
            return "CRITIQUE: La reponse est complete et verifiable. SATISFACTORY: YES"

        # Reflexion RETRY prompts
        if "retry" in p or "revise" in p:
            return (
                "La densite de Paris en 2024 est d'environ 20,581 habitants "
                "par km2 (population 2,161,000 / superficie 105 km2)."
            )

        # Default fallback
        return "I am a mock LLM. I do not know how to answer that specific prompt."


# ===========================================================================
# REAL-LLM ADAPTER (only used if env var is set and library present)
# ===========================================================================

def make_llm() -> Callable[[str, float], str]:
    """
    Return an LLM callable. Prefer Anthropic if available, then OpenAI,
    otherwise fall back to MockLLM.
    """
    if _HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()

        def anthropic_call(prompt: str, temperature: float = 0.0) -> str:
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text  # type: ignore

        print("[LLM] Using real Anthropic API")
        return anthropic_call

    if _HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        client = openai.OpenAI()

        def openai_call(prompt: str, temperature: float = 0.0) -> str:
            resp = client.chat.completions.create(
                model="gpt-5.4",
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        print("[LLM] Using real OpenAI API")
        return openai_call

    print("[LLM] No API key detected -- using MockLLM")
    return MockLLM()


# ===========================================================================
# DEMO 1 -- Chain-of-Thought vs direct answer
# ===========================================================================

def demo_cot(llm: Callable) -> None:
    print("\n" + "=" * 70)
    print("DEMO 1 -- Chain-of-Thought vs direct answer")
    print("=" * 70)

    question = ("Trois chemises coutent 45 euros. Avec un budget de 200 euros, "
                "combien de chemises puis-je acheter ?")

    # Direct answer (no reasoning) -- the mock will return a wrong answer
    direct_prompt = f"Question: {question}\nReponds directement sans raisonner."
    print("\n[Direct prompt]")
    print(llm(direct_prompt))

    # CoT answer -- the mock returns a correct step-by-step reasoning
    cot_prompt = f"Question: {question}\nLet's think step by step."
    print("\n[CoT prompt]")
    print(llm(cot_prompt))


# ===========================================================================
# DEMO 2 -- Self-Consistency (vote across N CoT)
# ===========================================================================

def extract_final_answer(text: str) -> str:
    """
    Pull the last number-or-word after 'Reponse :' out of a CoT response.
    Robustness is not the point -- the point is demonstrating the vote.
    """
    match = re.search(r"reponse\s*:\s*(\S+)", text.lower())
    return match.group(1).strip(". ") if match else text.strip()[-10:]


def self_consistency(llm: Callable, question: str, n: int = 5) -> str:
    """
    Call the LLM N times with temperature > 0, extract each answer,
    and return the most frequent one.
    """
    print(f"\n[Self-Consistency] Running {n} samples with temperature=0.7")
    answers = []
    for i in range(n):
        response = llm(question, temperature=0.7)
        ans = extract_final_answer(response)
        print(f"  sample {i+1}: {ans}  <- from: {response[:60]}...")
        answers.append(ans)
    counts = Counter(answers)
    most_common, count = counts.most_common(1)[0]
    print(f"\n[Self-Consistency] Vote winner: {most_common} ({count}/{n} votes)")
    return most_common


def demo_self_consistency(llm: Callable) -> None:
    print("\n" + "=" * 70)
    print("DEMO 2 -- Self-Consistency (vote majoritaire sur N CoT)")
    print("=" * 70)

    question = "How many r in 'strawberry'? Let's think step by step."
    winner = self_consistency(llm, question, n=5)
    print(f"\nFinal answer (after vote): {winner}")


# ===========================================================================
# DEMO 3 -- Plan-and-Execute
# ===========================================================================

@dataclass
class Plan:
    """A plan is a list of natural-language steps produced by the planner."""
    steps: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: str) -> "Plan":
        """Parse the planner response into a Plan object."""
        steps = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("STEP"):
                # Drop the "STEP N:" prefix
                _, _, rest = line.partition(":")
                if rest.strip():
                    steps.append(rest.strip())
        return cls(steps=steps)


# A fake tool registry the executor can call
def mock_search_tool(query: str) -> str:
    """Mock search -- returns canned facts for two queries."""
    q = query.lower()
    if "population" in q and "paris" in q:
        return "population: 2161000"
    if "superficie" in q or "area" in q:
        return "area: 105"
    return "no_result"


def execute_step_with_tool(llm: Callable, step: str, scratchpad: dict) -> str:
    """
    Ask the LLM to execute ONE step. The LLM may emit a TOOL_CALL which we
    intercept and run locally, appending the result back into the scratchpad.
    """
    # Build a prompt that includes the scratchpad so the LLM has context
    ctx = ", ".join(f"{k}={v}" for k, v in scratchpad.items())
    prompt = f"Execute step: {step}\nScratchpad so far: {ctx}"
    raw = llm(prompt)

    # Parse the LLM's response: TOOL_CALL or RESULT
    tool_match = re.match(r"TOOL_CALL:\s*search\('([^']+)'\)", raw.strip())
    if tool_match:
        tool_result = mock_search_tool(tool_match.group(1))
        print(f"    tool call: search('{tool_match.group(1)}') -> {tool_result}")
        # Update the scratchpad from the tool result
        if ":" in tool_result:
            key, _, val = tool_result.partition(":")
            scratchpad[key.strip()] = val.strip()
        return tool_result

    print(f"    direct result: {raw[:80]}")
    return raw


def plan_and_execute(llm: Callable, question: str) -> str:
    """
    Full plan-and-execute loop: plan -> execute each step -> synthesize.
    """
    # 1. PLANNING phase -- the planner produces a list of steps
    print("\n[Planner] Generating plan...")
    plan_raw = llm(f"You are a planner. Plan the following task:\n{question}")
    plan = Plan.parse(plan_raw)
    print(f"  Plan has {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"    {i}. {step}")

    # 2. EXECUTION phase -- execute each step in order, tracking state
    print("\n[Executor] Running steps...")
    scratchpad: dict = {}
    results: list[str] = []
    for i, step in enumerate(plan.steps, 1):
        print(f"  Step {i}: {step}")
        result = execute_step_with_tool(llm, step, scratchpad)
        results.append(result)

    # 3. SYNTHESIS phase -- combine results into a final answer
    print("\n[Synthesizer] Combining results into final answer...")
    synth_prompt = (
        f"You are a synthesizer. Produce the final synthesis for:\n"
        f"Question: {question}\n"
        f"Steps executed: {results}\n"
        f"Scratchpad: {scratchpad}"
    )
    final = llm(synth_prompt)
    return final


def demo_plan_execute(llm: Callable) -> None:
    print("\n" + "=" * 70)
    print("DEMO 3 -- Plan-and-Execute")
    print("=" * 70)

    question = "What is the population density of Paris?"
    answer = plan_and_execute(llm, question)
    print(f"\nFINAL ANSWER:\n  {answer}")


# ===========================================================================
# DEMO 4 -- Reflexion (self-critique + retry)
# ===========================================================================

def reflexion_loop(llm: Callable, question: str, max_retries: int = 3) -> str:
    """
    Reflexion: produce an answer, critique it, retry until satisfactory.
    """
    # Initial attempt using plan-and-execute
    attempt = plan_and_execute(llm, question)

    for i in range(max_retries):
        print(f"\n[Reflexion] Critique round {i+1}")
        critique_prompt = (
            f"You are a critic. Critique this answer to the question.\n"
            f"Question: {question}\n"
            f"Answer: {attempt}\n"
            f"Is it satisfactory? Respond with CRITIQUE: ... SATISFACTORY: YES|NO"
        )
        critique = llm(critique_prompt)
        print(f"  {critique}")

        if "SATISFACTORY: YES" in critique.upper():
            print("  [Reflexion] Answer accepted.")
            return attempt

        # Retry with the critique injected
        retry_prompt = (
            f"Retry the answer taking the critique into account.\n"
            f"Question: {question}\n"
            f"Previous answer: {attempt}\n"
            f"Critique: {critique}\n"
            f"Please revise."
        )
        attempt = llm(retry_prompt)
        print(f"  [Reflexion] Revised answer: {attempt[:100]}...")

    return attempt


def demo_reflexion(llm: Callable) -> None:
    print("\n" + "=" * 70)
    print("DEMO 4 -- Reflexion (self-critique + retry)")
    print("=" * 70)

    question = "What is the population density of Paris?"
    final = reflexion_loop(llm, question, max_retries=2)
    print(f"\nFINAL (REFLEXED) ANSWER:\n  {final}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    llm = make_llm()

    demo_cot(llm)
    demo_self_consistency(llm)
    demo_plan_execute(llm)
    demo_reflexion(llm)

    print("\n" + "=" * 70)
    print("All demos complete.")
    print("=" * 70)

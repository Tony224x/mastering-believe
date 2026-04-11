"""
Solutions -- Day 4: Planning & Reasoning

Contains solutions for the three easy exercises:
  - Easy Ex 1: CoT vs direct on 3 logic problems
  - Easy Ex 2: Plan-and-execute on a new question (Lyon density)
  - Easy Ex 3: Reflexion with concrete checklist

Each solution is self-contained and uses a MockLLM so it runs offline.

Run:  python 03-exercises/solutions/04-planning-reasoning.py
"""

import re
from dataclasses import dataclass, field
from typing import Callable


# ==========================================================================
# SHARED -- Mock LLM with an extensible response table
# ==========================================================================

class MockLLM:
    """
    Deterministic mock LLM. The response table maps keyword tuples to
    pre-written answers. Extending behaviour is just adding a rule.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        self.call_count += 1
        p = prompt.lower()

        # -- Exercise 1: three logic problems, with direct-vs-CoT answers --

        # Q1: arithmetic  ("15 pommes, 3 paniers, 5 dans chaque, combien de trop ?")
        if "15 pommes" in p:
            if "step by step" in p:
                return ("Etape 1: 3 paniers * 5 pommes = 15 pommes placees.\n"
                        "Etape 2: 15 - 15 = 0 pomme de trop.\n"
                        "Reponse : 0")
            return "Reponse : 2"   # Hallucinated direct answer

        # Q2: enigma  ("la mere de Jean a 3 enfants: Lundi, Mardi, ?")
        if "mere de jean" in p:
            if "step by step" in p:
                return ("Etape 1: la question dit 'la mere de JEAN a 3 enfants'.\n"
                        "Etape 2: si 2 enfants s'appellent Lundi et Mardi, le "
                        "3e est Jean (celui qu'on mentionne des le debut).\n"
                        "Reponse : Jean")
            return "Reponse : Mercredi"   # Classic trap, wrong answer

        # Q3: simple deduction  ("Alice est plus grande que Bob. Qui est le plus petit ?")
        if "alice est plus grande que bob" in p:
            # Deduction so simple that direct and CoT both get it right
            return "Reponse : Bob"

        # -- Exercise 2: Lyon plan-and-execute --
        if "lyon" in p and ("planner" in p or "plan the following" in p):
            return (
                "STEP 1: Search for the current population of Lyon.\n"
                "STEP 2: Search for the area of Lyon in km2.\n"
                "STEP 3: Compute surface_per_habitant = area_m2 / population.\n"
                "STEP 4: Format the answer as 'X m2 per habitant'."
            )
        if "execute step" in p and "population of lyon" in p:
            return "TOOL_CALL: search('population Lyon 2024')"
        if "execute step" in p and "area of lyon" in p:
            return "TOOL_CALL: search('superficie Lyon km2')"
        if "execute step" in p and "compute surface_per_habitant" in p:
            pop_match = re.search(r"population[:=]\s*(\d+)", p)
            area_match = re.search(r"area[:=]\s*([\d.]+)", p)
            if pop_match and area_match:
                pop = int(pop_match.group(1))
                area_km2 = float(area_match.group(1))
                area_m2 = area_km2 * 1_000_000
                ratio = area_m2 / pop
                return f"RESULT: ratio = {area_m2}/{pop} = {ratio:.1f} m2/hab"
            return "RESULT: ratio = 48000000/520000 = 92.3 m2/hab"
        if "execute step" in p and "format the answer" in p:
            return "RESULT: Environ 92 m2 par habitant a Lyon."
        if "synthesizer" in p and "lyon" in p:
            return ("Le ratio surface / habitant a Lyon est d'environ 92 m2 "
                    "par habitant (48 km2 pour 520,000 habitants).")

        # -- Exercise 3: Reflexion with checklist --
        # Poor initial answer (missing unit and date)
        if "densite de paris" in p and "retry" not in p and "revise" not in p:
            return "La densite de Paris est d'environ 20581."
        # Improved answer after critique
        if ("retry" in p or "revise" in p) and "paris" in p:
            return ("La densite de Paris en 2024 est d'environ 20,581 "
                    "habitants par km2.")

        return "MOCK: unknown prompt"


# ==========================================================================
# EASY EXERCISE 1 -- CoT vs direct on 3 logic problems
# ==========================================================================

def check_answer(response: str, expected: str) -> bool:
    """Case-insensitive substring check."""
    return expected.lower() in response.lower()


def solve_ex1() -> None:
    print("\n" + "=" * 70)
    print("EX1 -- CoT vs direct on 3 logic problems")
    print("=" * 70)

    llm = MockLLM()

    problems = [
        ("Q1 (arithmetique)", "J'ai 15 pommes et 3 paniers. Si je mets 5 pommes "
                              "dans chaque panier, combien de pommes sont en trop ?",
         "0"),
        ("Q2 (enigme)", "La mere de Jean a 3 enfants. Les deux premiers s'appellent "
                        "Lundi et Mardi. Comment s'appelle le troisieme ?",
         "jean"),
        ("Q3 (deduction)", "Alice est plus grande que Bob. Qui est le plus petit ?",
         "bob"),
    ]

    print(f"\n{'Question':20} | {'Direct':^8} | {'CoT':^8}")
    print("-" * 45)

    for name, q, expected in problems:
        direct = llm(f"Question: {q}\nReponds directement.")
        cot = llm(f"Question: {q}\nLet's think step by step.")
        direct_ok = "OK" if check_answer(direct, expected) else "FAUX"
        cot_ok = "OK" if check_answer(cot, expected) else "FAUX"
        print(f"{name:20} | {direct_ok:^8} | {cot_ok:^8}")

    print("\nCommentaire: Q3 n'a pas eu besoin de CoT car la deduction est en 1")
    print("seule etape (A > B donc B < A). Pour Q1 et Q2, le raisonnement en")
    print("plusieurs etapes evite les pieges et les hallucinations.")


# ==========================================================================
# EASY EXERCISE 2 -- Plan-and-execute on Lyon
# ==========================================================================

@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: str) -> "Plan":
        steps = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("STEP"):
                _, _, rest = line.partition(":")
                if rest.strip():
                    steps.append(rest.strip())
        return cls(steps=steps)


def mock_search_tool(query: str) -> str:
    q = query.lower()
    if "population" in q and "lyon" in q:
        return "population: 520000"
    if ("superficie" in q or "area" in q) and "lyon" in q:
        return "area: 48"
    return "no_result"


def execute_step(llm: Callable, step: str, scratchpad: dict) -> str:
    ctx = ", ".join(f"{k}={v}" for k, v in scratchpad.items())
    prompt = f"Execute step: {step}\nScratchpad so far: {ctx}"
    raw = llm(prompt)

    m = re.match(r"TOOL_CALL:\s*search\('([^']+)'\)", raw.strip())
    if m:
        result = mock_search_tool(m.group(1))
        print(f"    tool call: search('{m.group(1)}') -> {result}")
        if ":" in result:
            key, _, val = result.partition(":")
            scratchpad[key.strip()] = val.strip()
        return result

    print(f"    direct: {raw[:80]}")
    return raw


def plan_and_execute(llm: Callable, question: str) -> str:
    print("\n[Planner]")
    plan_raw = llm(f"You are a planner. Plan the following task:\n{question}")
    plan = Plan.parse(plan_raw)
    print(f"  {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"    {i}. {step}")

    print("\n[Executor]")
    scratchpad: dict = {}
    results: list[str] = []
    for i, step in enumerate(plan.steps, 1):
        print(f"  Step {i}: {step}")
        results.append(execute_step(llm, step, scratchpad))

    print("\n[Synthesizer]")
    synth_prompt = (f"You are a synthesizer. For question: {question}\n"
                    f"Scratchpad: {scratchpad}\nProduce the final synthesis.")
    final = llm(synth_prompt)
    print(f"  final: {final}")
    return final


def solve_ex2() -> None:
    print("\n" + "=" * 70)
    print("EX2 -- Plan-and-execute on a new question (Lyon)")
    print("=" * 70)

    llm = MockLLM()
    question = "What is the surface per habitant ratio of Lyon?"
    answer = plan_and_execute(llm, question)

    # Verification: the answer must contain the expected ratio
    assert "92" in answer, f"Expected '92' in the answer, got: {answer}"
    print("\n[Verification] PASS -- answer contains '92 m2/hab'")


# ==========================================================================
# EASY EXERCISE 3 -- Reflexion with concrete checklist
# ==========================================================================

def check_criterion(answer: str, criterion: str) -> bool:
    """
    Dispatch by criterion label. Each criterion has a local, deterministic
    check. No LLM call involved -- that is the point.
    """
    a = answer.lower()
    if "chiffre" in criterion.lower():
        return bool(re.search(r"\d", answer))
    if "unite" in criterion.lower() or "unit" in criterion.lower():
        units = ["hab/km2", "m2/hab", "habitants par km2", "m2 par habitant"]
        return any(u in a for u in units)
    if "temp" in criterion.lower() or "date" in criterion.lower() or "annee" in criterion.lower():
        return bool(re.search(r"\b20\d{2}\b", answer))
    # Default: substring match
    return criterion.lower() in a


def reflexion_with_checklist(
    llm: Callable,
    question: str,
    checklist: list[str],
    max_retries: int = 3,
) -> str:
    """
    Reflexion loop where the critique is a local checklist, not an LLM call.
    Returns the answer as soon as all criteria pass.
    """
    # Initial attempt
    attempt = llm(f"Answer: {question}")
    print(f"\n[Attempt 0] {attempt}")

    for i in range(max_retries):
        missing = [c for c in checklist if not check_criterion(attempt, c)]
        if not missing:
            print(f"[Reflexion] All {len(checklist)} criteria passed.")
            return attempt

        print(f"[Reflexion] Round {i+1}: missing {len(missing)} criteria: {missing}")
        critique = "The answer is missing: " + "; ".join(missing)
        retry_prompt = (
            f"Retry / revise the previous answer.\n"
            f"Question: {question}\n"
            f"Previous: {attempt}\n"
            f"Critique: {critique}"
        )
        attempt = llm(retry_prompt)
        print(f"[Attempt {i+1}] {attempt}")

    # Max retries reached without satisfying all criteria
    print(f"[Reflexion] WARNING: max_retries reached, returning best attempt.")
    return attempt


def solve_ex3() -> None:
    print("\n" + "=" * 70)
    print("EX3 -- Reflexion with concrete checklist")
    print("=" * 70)

    llm = MockLLM()
    question = "Quelle est la densite de Paris ?"
    checklist = [
        "Contient un chiffre",
        "Contient une unite (hab/km2, m2/hab)",
        "Contient une reference temporelle (annee)",
    ]

    final = reflexion_with_checklist(llm, question, checklist, max_retries=3)
    print(f"\nFINAL: {final}")

    # Verify the final answer matches all criteria
    for c in checklist:
        assert check_criterion(final, c), f"Criterion not met: {c}"
    print("\n[Verification] PASS -- all 3 criteria met on final answer")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    solve_ex1()
    solve_ex2()
    solve_ex3()

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

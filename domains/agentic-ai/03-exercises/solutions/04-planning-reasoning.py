"""
Solutions -- Day 4: Planning & Reasoning

Contains solutions for:
  - Easy Ex 1: CoT vs direct on 3 logic problems
  - Easy Ex 2: Plan-and-execute on a new question (Lyon density)
  - Easy Ex 3: Reflexion with concrete checklist
  - Medium Ex 1: Self-consistency (majority vote over N samples)
  - Medium Ex 2: Dynamic replanning after a failed step
  - Medium Ex 3: Task decomposition as a dependency DAG

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
# MEDIUM EXERCISE 1 -- Self-consistency (majority vote)
# ==========================================================================

# 5 pre-written CoT samples for the same multi-step arithmetic question.
# Sample with seed 2 is deliberately wrong (a typical sampling failure).
SC_QUESTION = "A shop sells 14 boxes of 12 eggs, 5 eggs break. How many intact eggs?"
SC_SAMPLES = {
    0: "Step 1: 14 * 12 = 168 eggs.\nStep 2: 168 - 5 = 163.\nReponse : 163",
    1: "Step 1: total = 14 boxes x 12 = 168.\nStep 2: minus 5 broken = 163.\nReponse : 163",
    2: "Step 1: 14 * 12 = 158 (miscalculated).\nStep 2: 158 - 5 = 153.\nReponse : 153",
    3: "Step 1: dozen = 12, so 12 * 14 = 168.\nStep 2: 168 - 5 = 163.\nReponse : 163",
    4: "Step 1: 14 * 12 = 168.\nStep 2: broken eggs: 168 - 5 = 163.\nReponse : 163",
}


class SamplingMockLLM:
    """Deterministic 'sampling': temperature>0 picks a pre-written variant by seed."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str, temperature: float = 0.0, seed: int = 0) -> str:
        self.call_count += 1
        if SC_QUESTION.lower() not in prompt.lower():
            return "MOCK: unknown prompt"
        if temperature == 0.0:
            return SC_SAMPLES[0]  # Greedy decoding: always the same reasoning
        # Deterministic variant selection -- no unseeded randomness
        return SC_SAMPLES[seed % len(SC_SAMPLES)]


def extract_final_answer(reasoning: str) -> str | None:
    """Pull the final answer from a 'Reponse : X' line."""
    m = re.search(r"Reponse\s*:\s*([^\n]+)", reasoning)
    return m.group(1).strip() if m else None


def self_consistency(llm: Callable, question: str, n_samples: int = 5) -> dict:
    """Sample n reasonings, extract answers, majority vote."""
    votes: dict[str, int] = {}
    for seed in range(n_samples):
        raw = llm(f"Question: {question}\nLet's think step by step.",
                  temperature=0.7, seed=seed)
        answer = extract_final_answer(raw)
        if answer is not None:
            votes[answer] = votes.get(answer, 0) + 1
    best = max(votes, key=lambda a: votes[a])
    return {
        "answer": best,
        "votes": votes,
        "confidence": votes[best] / n_samples,
    }


def solve_medium1() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 1 -- Self-consistency (majority vote)")
    print("=" * 70)

    llm = SamplingMockLLM()

    greedy = llm(f"Question: {SC_QUESTION}\nLet's think step by step.", temperature=0.0)
    print(f"\nGreedy (temperature=0) answer: {extract_final_answer(greedy)}")

    result = self_consistency(llm, SC_QUESTION, n_samples=5)
    print(f"Self-consistency votes: {result['votes']}")
    print(f"Final answer: {result['answer']} (confidence {result['confidence']:.0%})")

    # The wrong sample (seed 2) is outvoted 4 to 1
    assert result["answer"] == "163"
    assert result["votes"] == {"163": 4, "153": 1}
    assert result["confidence"] == 0.8
    print("\n[Verification] PASS -- 1 bad sample outvoted, confidence 0.8")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Dynamic replanning after a failed step
# ==========================================================================

# Bordeaux data: only reachable through the REPAIRED query
def bordeaux_search_tool(query: str) -> str:
    q = query.lower()
    if "population bordeaux 2024" in q:
        return "no_result"                       # The flaky query
    if "bordeaux" in q and ("habitants" in q or "insee" in q):
        return "population: 260000"              # The repaired query works
    if "superficie" in q and "bordeaux" in q:
        return "area: 49"
    return "no_result"


class ReplanMockLLM:
    """Planner that produces an initial plan, then a repaired plan on demand."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        p = prompt.lower()
        if "replan" in p and "bordeaux" in p:
            # The repaired plan only swaps the failed step (the executor keeps
            # already-completed steps, so we re-emit the remaining ones)
            return ("STEP 1: search('Bordeaux nombre habitants INSEE')\n"
                    "STEP 2: search('superficie Bordeaux km2')\n"
                    "STEP 3: compute density")
        if "plan" in p and "bordeaux" in p:
            return ("STEP 1: search('population Bordeaux 2024')\n"
                    "STEP 2: search('superficie Bordeaux km2')\n"
                    "STEP 3: compute density")
        return "MOCK: unknown prompt"


def parse_steps(raw: str) -> list[str]:
    steps = []
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("STEP"):
            _, _, rest = line.partition(":")
            if rest.strip():
                steps.append(rest.strip())
    return steps


def execute_plan_step(step: str, scratchpad: dict) -> str:
    """Run one step. Returns 'OK' or 'FAILED'. Never stores garbage."""
    m = re.match(r"search\('([^']+)'\)", step)
    if m:
        result = bordeaux_search_tool(m.group(1))
        if result == "no_result":
            return "FAILED"
        key, _, val = result.partition(":")
        scratchpad[key.strip()] = float(val.strip())
        return "OK"
    if "compute density" in step:
        if "population" in scratchpad and "area" in scratchpad:
            scratchpad["density"] = round(scratchpad["population"] / scratchpad["area"], 1)
            return "OK"
        return "FAILED"
    return "FAILED"


def plan_execute_with_replan(llm: Callable, question: str, max_replans: int = 2) -> dict:
    raw = llm(f"You are a planner. plan: {question}")
    plan = parse_steps(raw)
    scratchpad: dict = {}
    trace: list[str] = [f"initial plan: {plan}"]
    replans = 0
    i = 0

    while i < len(plan):
        step = plan[i]
        status = execute_plan_step(step, scratchpad)
        trace.append(f"step {i + 1}: {step} -> {status}")

        if status == "FAILED":
            if replans >= max_replans:
                trace.append("max replans reached -- giving up")
                return {"answer": "FAILED: could not complete the plan",
                        "scratchpad": scratchpad, "trace": trace, "replans": replans}
            replans += 1
            # The replan prompt carries: question, failed step, scratchpad
            replan_raw = llm(
                f"replan: {question}\nfailed step: {step}\nscratchpad: {scratchpad}"
            )
            new_steps = parse_steps(replan_raw)
            # Resume FROM the repaired step: keep what already succeeded,
            # replace the remainder of the plan with the repaired tail.
            plan = plan[:i] + new_steps[i:] if len(new_steps) > i else plan[:i] + new_steps
            trace.append(f"replan #{replans}: new tail = {plan[i:]}")
            continue  # Retry at the same index with the repaired step

        i += 1

    answer = f"Bordeaux density is about {scratchpad.get('density')} hab/km2"
    return {"answer": answer, "scratchpad": scratchpad, "trace": trace, "replans": replans}


def solve_medium2() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 2 -- Dynamic replanning after a failed step")
    print("=" * 70)

    llm = ReplanMockLLM()
    result = plan_execute_with_replan(llm, "Quelle est la densite de Bordeaux ?")
    for line in result["trace"]:
        print(f"  {line}")
    print(f"\nFINAL: {result['answer']}")

    assert result["replans"] == 1
    assert "no_result" not in str(result["scratchpad"])
    assert result["scratchpad"]["density"] == round(260000 / 49, 1)
    # The successful area step must not be re-executed: trace shows it once
    area_runs = [t for t in result["trace"] if "superficie" in t and "-> OK" in t]
    assert len(area_runs) == 1
    print("\n[Verification] PASS -- 1 replan, clean scratchpad, no re-execution")

    # Max replans guard: a query that always fails
    always_fail_llm = ReplanMockLLM()
    bad = plan_execute_with_replan(
        # Trick: ask about a city the tool does not know -> every search fails
        lambda prompt: ("STEP 1: search('population Nantes 2024')"
                        if "replan" not in prompt.lower()
                        else "STEP 1: search('population Nantes 2024')"),
        "Densite de Nantes ?", max_replans=2,
    )
    assert "FAILED" in bad["answer"] and bad["replans"] == 2
    print("[Verification] PASS -- replan limit enforced on persistent failure")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Task decomposition as a dependency DAG
# ==========================================================================

@dataclass
class SubTask:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"   # pending | done | failed
    result: str | None = None


DAG_FACTS = {
    "population paris": 2_161_000,
    "surface paris": 105,
    "population lyon": 520_000,
    "surface lyon": 48,
}


def decompose_compare_density(question: str) -> list[SubTask]:
    """Mock decomposition for 'Compare la densite de Paris et de Lyon'."""
    return [
        SubTask("t1", "population paris"),
        SubTask("t2", "surface paris"),
        SubTask("t3", "population lyon"),
        SubTask("t4", "surface lyon"),
        SubTask("t5", "compare densities", depends_on=["t1", "t2", "t3", "t4"]),
    ]


def dag_executor(task: SubTask, by_id: dict[str, SubTask]) -> str:
    """Execute one subtask. The compare task reads its dependencies' results."""
    if task.id == "t5":
        pop_p = float(by_id["t1"].result)
        sur_p = float(by_id["t2"].result)
        pop_l = float(by_id["t3"].result)
        sur_l = float(by_id["t4"].result)
        d_paris = round(pop_p / sur_p)
        d_lyon = round(pop_l / sur_l)
        winner = "Paris" if d_paris > d_lyon else "Lyon"
        return f"Paris={d_paris} hab/km2, Lyon={d_lyon} hab/km2 -> {winner} is denser"
    return str(DAG_FACTS[task.description])


def run_dag(tasks: list[SubTask], executor: Callable) -> dict:
    """Wave-based scheduler: each wave runs all tasks whose deps are done."""
    by_id = {t.id: t for t in tasks}
    waves: list[list[str]] = []

    while any(t.status == "pending" for t in tasks):
        runnable = [
            t for t in tasks
            if t.status == "pending"
            and all(by_id[d].status == "done" for d in t.depends_on)
        ]
        if not runnable:
            raise RuntimeError("Cycle or unsatisfiable dependency")
        wave_ids = [t.id for t in runnable]
        waves.append(wave_ids)
        print(f"  wave {len(waves)}: {wave_ids}")
        for t in runnable:  # Parallel-eligible; executed sequentially here
            t.result = executor(t, by_id)
            t.status = "done"

    return {"waves": waves, "final": by_id["t5"].result if "t5" in by_id else None}


def solve_medium3() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM 3 -- Task decomposition as a dependency DAG")
    print("=" * 70)

    tasks = decompose_compare_density("Compare la densite de Paris et de Lyon")
    print(f"\nDAG: {[(t.id, t.depends_on) for t in tasks]}")
    out = run_dag(tasks, dag_executor)
    print(f"\nFINAL: {out['final']}")

    assert out["waves"] == [["t1", "t2", "t3", "t4"], ["t5"]]
    assert "Paris is denser" in out["final"]
    print("\n[Verification] PASS -- 2 waves, Paris denser than Lyon")

    # Cycle detection
    cyclic = [
        SubTask("t1", "population paris", depends_on=["t2"]),
        SubTask("t2", "surface paris", depends_on=["t1"]),
    ]
    try:
        run_dag(cyclic, dag_executor)
        raise AssertionError("cycle not detected")
    except RuntimeError as exc:
        print(f"[Verification] PASS -- cycle detected: {exc}")


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    solve_ex1()
    solve_ex2()
    solve_ex3()

    solve_medium1()
    solve_medium2()
    solve_medium3()

    # Hard exercises are substantial projects -- key hints:
    #
    # Hard Ex 1 (Tree-of-Thought beam search):
    #   - A "thought" = one operation between two remaining numbers
    #   - generate(level) -> all (a op b) pairs; evaluate -> heuristic 0..1
    #   - Keep beam_width best per level, recurse, backtrack on dead ends
    #   - Compare with greedy: greedy takes argmax at each level and can miss 24
    #
    # Hard Ex 2 (Meta-controller):
    #   - 4 Strategy classes sharing run(llm, question) -> StrategyResult
    #   - Classifier: count entities, detect "compare"/"ratio", digits, length
    #   - Budget check BEFORE running: downgrade plan-and-execute -> CoT -> direct
    #   - Track llm_calls per run and assert the global total

    print("\n" + "=" * 70)
    print("All solutions complete.")
    print("=" * 70)

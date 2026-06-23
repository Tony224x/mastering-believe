"""
Jour 13 — Emergent abilities & reasoning: CoT demo with a mock LLM
====================================================================
Pure Python. No external dependencies.

Covers:
  1. A rule-based "mock LLM" that can answer simple problems
  2. Direct answer mode vs chain-of-thought mode
  3. Synthetic reasoning task where direct fails but step-by-step works
  4. Self-consistency demo (majority vote)

Run: python 02-code/13-emergent-abilities-reasoning.py
"""

import sys
import io
import random
import re
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

random.seed(42)


# ============================================================================
# PART 1: A tiny rule-based "mock LLM"
# ============================================================================

print("=" * 70)
print("PART 1: A mock LLM that illustrates CoT vs direct")
print("=" * 70)


class MockLLM:
    """
    A rule-based mock LLM that simulates two modes:
      - direct: answer a problem in one step (prone to errors)
      - cot: answer step by step (more reliable)

    The "reasoning" is done by explicit Python logic that mimics what a
    big LLM would do internally. We use this to illustrate the pedagogy,
    not to be a real model.
    """

    def __init__(self, error_rate_direct=0.5, error_rate_cot=0.1, seed=0):
        self.error_rate_direct = error_rate_direct
        self.error_rate_cot = error_rate_cot
        self.rng = random.Random(seed)

    def direct_answer(self, problem):
        """
        Answer without thinking step by step.
        Simulates a big model 'guessing' a direct answer: often wrong on
        complex problems because it cannot plan.
        """
        correct = self._solve_correctly(problem)
        if self.rng.random() < self.error_rate_direct:
            # Return a plausible but wrong answer
            wrong = correct + self.rng.choice([-3, -1, 1, 3, 5])
            return wrong, []
        return correct, []

    def cot_answer(self, problem):
        """
        Answer step by step. Each step is a natural-language sentence
        that manipulates the problem. Much more reliable.
        """
        steps = []
        correct = self._solve_correctly(problem, verbose_steps=steps)
        # Even CoT can make mistakes occasionally
        if self.rng.random() < self.error_rate_cot:
            wrong = correct + self.rng.choice([-1, 1])
            steps.append(f"So the answer is {wrong}.  [wrong]")
            return wrong, steps
        steps.append(f"So the answer is {correct}.")
        return correct, steps

    def _solve_correctly(self, problem, verbose_steps=None):
        """
        The "golden" solver. For the demo, problems are formatted as
        a small arithmetic word problem:
          (start, op, count, op2, count2, ...)
        """
        start = problem['start']
        if verbose_steps is not None:
            verbose_steps.append(f"We start with {start}.")
        result = start
        for op, n in problem['ops']:
            if op == '+':
                new_result = result + n
                if verbose_steps is not None:
                    verbose_steps.append(f"We add {n}: {result} + {n} = {new_result}.")
                result = new_result
            elif op == '-':
                new_result = result - n
                if verbose_steps is not None:
                    verbose_steps.append(f"We subtract {n}: {result} - {n} = {new_result}.")
                result = new_result
            elif op == 'x':
                new_result = result * n
                if verbose_steps is not None:
                    verbose_steps.append(f"We multiply by {n}: {result} * {n} = {new_result}.")
                result = new_result
        return result


def format_problem_as_text(problem):
    """Render a problem as a natural-language description for display."""
    parts = [f"Start with {problem['start']}."]
    for op, n in problem['ops']:
        if op == '+':
            parts.append(f"Add {n}.")
        elif op == '-':
            parts.append(f"Subtract {n}.")
        elif op == 'x':
            parts.append(f"Multiply by {n}.")
    parts.append("What is the result?")
    return " ".join(parts)


# Create a few problems of varying difficulty
problems = [
    {'start': 10, 'ops': [('+', 5), ('-', 3)]},  # easy
    {'start': 7, 'ops': [('x', 3), ('+', 4), ('-', 2)]},  # medium
    {'start': 15, 'ops': [('-', 2), ('x', 2), ('+', 10), ('-', 5)]},  # hard
    {'start': 20, 'ops': [('+', 15), ('-', 8), ('x', 2), ('-', 4), ('+', 6)]},
]

llm = MockLLM(error_rate_direct=0.5, error_rate_cot=0.05, seed=42)

for i, p in enumerate(problems, 1):
    print(f"\n--- Probleme {i} ---")
    print(f"Enonce: {format_problem_as_text(p)}")

    # True answer
    true_answer = llm._solve_correctly(p)
    print(f"Vraie reponse: {true_answer}")

    # Direct mode
    direct, _ = llm.direct_answer(p)
    print(f"Direct answer:  {direct}  {'[OK]' if direct == true_answer else '[FAUX]'}")

    # CoT mode
    cot, steps = llm.cot_answer(p)
    print(f"CoT answer:     {cot}  {'[OK]' if cot == true_answer else '[FAUX]'}")
    for step in steps:
        print(f"    {step}")


# ============================================================================
# PART 2: Aggregate accuracy over many problems
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Accuracy on 100 random problems")
print("=" * 70)


def generate_problem(n_ops=4, max_val=20):
    """Generate a random arithmetic problem."""
    start = random.randint(1, max_val)
    ops = []
    for _ in range(n_ops):
        op = random.choice(['+', '-', 'x'])
        if op == 'x':
            n = random.randint(2, 4)  # multipliers stay small
        else:
            n = random.randint(1, max_val)
        ops.append((op, n))
    return {'start': start, 'ops': ops}


# Evaluate direct vs CoT on 100 problems
N = 100
direct_correct = 0
cot_correct = 0

for _ in range(N):
    p = generate_problem(n_ops=5)
    true = llm._solve_correctly(p)
    d, _ = llm.direct_answer(p)
    c, _ = llm.cot_answer(p)
    direct_correct += (d == true)
    cot_correct += (c == true)

print(f"\nResultats sur {N} problemes:")
print(f"  Direct answer:  {direct_correct}/{N} = {direct_correct}%")
print(f"  CoT answer:     {cot_correct}/{N} = {cot_correct}%")
print(f"  Gain avec CoT:  +{cot_correct - direct_correct}%")


# ============================================================================
# PART 3: Self-consistency — majority vote over multiple CoT samples
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Self-consistency (majority vote)")
print("=" * 70)


class NoisyCoTLLM:
    """
    A mock LLM whose CoT is noisy: about 60% correct, 40% different wrong
    answers. Self-consistency should still get the right one by majority.
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def cot_answer(self, true_answer):
        """Sample a noisy answer around the true one."""
        if self.rng.random() < 0.6:
            return true_answer
        # Wrong answer: random offset
        return true_answer + self.rng.choice([-3, -2, -1, 1, 2, 3, 4])


noisy = NoisyCoTLLM(seed=1)


def self_consistency(llm, true_answer, n_samples=5):
    """Run n CoT samples and return the majority vote."""
    samples = [llm.cot_answer(true_answer) for _ in range(n_samples)]
    votes = Counter(samples)
    return votes.most_common(1)[0][0], samples


# Test on 100 problems
n_problems = 100
single_correct = 0
sc5_correct = 0
sc10_correct = 0

for _ in range(n_problems):
    true = random.randint(1, 100)

    # Single sample
    single = noisy.cot_answer(true)
    single_correct += (single == true)

    # Self-consistency with 5 samples
    vote5, _ = self_consistency(noisy, true, n_samples=5)
    sc5_correct += (vote5 == true)

    # Self-consistency with 10 samples
    vote10, _ = self_consistency(noisy, true, n_samples=10)
    sc10_correct += (vote10 == true)

print(f"\nResultats sur {n_problems} problemes:")
print(f"  Single CoT (1 sample):   {single_correct}%")
print(f"  Self-consistency (5):    {sc5_correct}%")
print(f"  Self-consistency (10):   {sc10_correct}%")

# Show a concrete example
print("\nExemple detaille:")
true = 42
vote, samples = self_consistency(noisy, true, n_samples=8)
print(f"  Vraie reponse: {true}")
print(f"  Samples: {samples}")
print(f"  Vote majoritaire: {vote}  ({'OK' if vote == true else 'FAUX'})")

print("""
Observation: plus on a de samples, plus la majorite se stabilise sur
la bonne reponse. Meme avec un modele qui a seulement 60% de chance
individuelle d'etre correct, la majorite de 10 samples atteint ~85%+.
""")


# ============================================================================
# PART 4: The 'Tous meurent sauf N' trick problem
# ============================================================================

print("=" * 70)
print("PART 4: Le probleme piege classique")
print("=" * 70)

print("""
Q: Un fermier a 15 moutons. Tous meurent sauf 8. Combien en reste-t-il ?

Direct (modele qui 'devine'):
  Calcul rapide: 15 - (15 - 8) = 8?
  Ou: 'tous sauf 8' est traduit en 'tous - 8 = 7'
  -> Reponse: 7 (FAUX — piege linguistique)

CoT (step by step):
  Etape 1: Le fermier avait 15 moutons au depart.
  Etape 2: 'Tous meurent sauf 8' signifie: 8 moutons SURVIVENT.
  Etape 3: Donc, le nombre de moutons restants est 8.
  Reponse: 8 (CORRECT)

C'est exactement le type de probleme ou CoT aide: pas parce que c'est
mathematiquement difficile, mais parce que le modele doit se souvenir
d'interpreter le langage correctement. Ecrire etape par etape force
le modele a verbaliser son interpretation.
""")

print("=" * 70)
print("Fin — CoT + self-consistency = les meta-techniques fondamentales.")
print("=" * 70)

"""
Jour 15 — Test-time compute & reasoning models
================================================
Pure Python, pas de dependance externe.

Objectif pedagogique : comprendre concretement pourquoi le test-time compute
marche, comment GRPO met a jour une policy, et comment router entre un LLM
classique et un reasoning model dans un vrai systeme de production.

Contenu :
  1. Self-consistency (majority vote) sur un solver de sommes modulo
  2. Simulation naive de GRPO (group-relative advantage) sur un bandit
  3. Router planner/executor : quand appeler o3 vs Haiku
  4. Budget de thinking tokens et overthinking

Run : python 02-code/15-test-time-compute-reasoning-models.py
"""

from __future__ import annotations
import sys
import io
import random
import math
import statistics
from collections import Counter
from dataclasses import dataclass

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(7)

# ============================================================================
# PART 1 — Self-consistency : pourquoi generer N reponses et voter marche
# ============================================================================
print("=" * 70)
print("PART 1 : Self-consistency / majority vote")
print("=" * 70)


def noisy_solver(problem: tuple[int, int, int], noise: float = 0.4) -> int:
    """
    Simule un LLM qui repond (a + b) % m avec une probabilite `noise`
    de produire une reponse plausible mais fausse. C'est le setting typique
    d'un modele qui fait des erreurs de calcul en one-shot.
    """
    a, b, m = problem
    correct = (a + b) % m
    if random.random() < noise:
        # Erreur plausible : off-by-one, oubli du modulo, inversion de signe.
        return random.choice([correct - 1, correct + 1, (a - b) % m, (a + b)])
    return correct


def self_consistency(problem, k: int) -> int:
    """Majority vote sur K echantillons : on parie qu'une reponse fausse
    est rarement repetee de la meme facon."""
    votes = [noisy_solver(problem) for _ in range(k)]
    return Counter(votes).most_common(1)[0][0]


problems = [(random.randint(1, 1000), random.randint(1, 1000), 97) for _ in range(500)]
for k in [1, 5, 10, 40]:
    correct_k = sum(
        1 for p in problems if self_consistency(p, k) == (p[0] + p[1]) % p[2]
    )
    print(f"  K={k:>2}  accuracy = {correct_k / len(problems):.3f}")
print("  → meme solver, accuracy monte de ~0.6 (k=1) a ~0.95 (k=40) par simple vote.\n")


# ============================================================================
# PART 2 — GRPO simule : group relative advantage sur un bandit
# ============================================================================
print("=" * 70)
print("PART 2 : GRPO en version pedagogique (bandit 4 bras)")
print("=" * 70)

# 4 "strategies de reasoning" avec reward moyenne cachee.
# Objectif : apprendre la distribution qui maximise la reward, SANS critic,
# juste en utilisant la moyenne du groupe comme baseline.
TRUE_REWARDS = [0.1, 0.3, 0.8, 0.5]  # le bras 2 est le meilleur.


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exp = [math.exp(l - m) for l in logits]
    s = sum(exp)
    return [e / s for e in exp]


def sample_action(probs: list[float]) -> int:
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return i
    return len(probs) - 1


def reward_fn(action: int) -> float:
    # Reward stochastique autour de la moyenne cachee.
    return TRUE_REWARDS[action] + random.gauss(0, 0.05)


logits = [0.0, 0.0, 0.0, 0.0]   # policy uniforme au depart
lr = 0.1
G = 16  # taille du groupe GRPO

for step in range(200):
    probs = softmax(logits)
    # 1) Echantillonner G trajectoires depuis la meme policy
    actions = [sample_action(probs) for _ in range(G)]
    rewards = [reward_fn(a) for a in actions]

    # 2) Advantage = (r - mean) / std : "a quel point cette trajectoire
    #    est-elle meilleure que la moyenne du groupe ?". Pas de critic.
    mean_r = statistics.mean(rewards)
    std_r = statistics.pstdev(rewards) or 1e-6
    advantages = [(r - mean_r) / std_r for r in rewards]

    # 3) Policy gradient : augmenter le logit des actions a advantage positive.
    for a, adv in zip(actions, advantages):
        for i in range(len(logits)):
            grad = (1.0 if i == a else 0.0) - probs[i]
            logits[i] += lr * adv * grad

    if step % 40 == 0:
        print(f"  step {step:>3}  probs = {[round(p, 3) for p in softmax(logits)]}")

print(f"  final probs = {[round(p, 3) for p in softmax(logits)]}")
print("  → la policy converge vers le bras 2 (vraie reward 0.8) sans value function.\n")


# ============================================================================
# PART 3 — Router planner-executor : quand appeler un reasoning model ?
# ============================================================================
print("=" * 70)
print("PART 3 : Router cout/latence planner vs executor")
print("=" * 70)


@dataclass
class ModelProfile:
    name: str
    cost_per_1k_out: float   # USD / 1k tokens output
    latency_per_1k_out: float  # seconds / 1k tokens output
    quality_reasoning: float   # 0..1 sur taches verifiables
    quality_extraction: float  # 0..1 sur taches extractives


PROFILES = {
    "haiku": ModelProfile("claude-haiku-4-5", 0.004, 0.2, 0.55, 0.90),
    "sonnet": ModelProfile("claude-sonnet-4-6", 0.015, 0.4, 0.72, 0.95),
    "opus_thinking": ModelProfile("claude-opus-4-7-thinking", 0.075, 3.0, 0.94, 0.95),
}


def route(task_type: str, quality_target: float, latency_budget_s: float,
          output_tokens: int = 800):
    """
    Heuristique minimaliste mais realiste pour un routeur de prod :
      - filtre par latence budget
      - filtre par qualite attendue
      - choisit le moins cher parmi les candidats
    C'est *cette* fonction qui determine 80% de la facture OpenAI/Anthropic
    d'un produit LLM, pas le prompt.
    """
    candidates = []
    for p in PROFILES.values():
        latency = p.latency_per_1k_out * output_tokens / 1000
        if latency > latency_budget_s:
            continue
        q = p.quality_reasoning if task_type == "reasoning" else p.quality_extraction
        if q < quality_target:
            continue
        cost = p.cost_per_1k_out * output_tokens / 1000
        candidates.append((cost, p, q, latency))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


scenarios = [
    ("reasoning", 0.90, 60.0, 2000, "Resoudre une preuve math complexe"),
    ("extraction", 0.85, 3.0, 500, "Extraire le nom du contact d'un email"),
    ("reasoning", 0.70, 5.0, 1500, "Debug un bug de logique dans du Python"),
    ("reasoning", 0.92, 2.0, 1000, "Math dur ET user-facing (impossible)"),
]

for task_type, quality, latency, tokens, label in scenarios:
    r = route(task_type, quality, latency, tokens)
    if r is None:
        print(f"  [{label}] → aucun modele ne satisfait les contraintes")
    else:
        cost, profile, q, l = r
        print(
            f"  [{label}]\n"
            f"      → {profile.name}  cost=${cost:.4f}  lat={l:.2f}s  quality={q:.2f}"
        )
print("  Lecon : si les contraintes sont incompatibles, changer le produit")
print("  (async UX, explain-then-act) plutot que chercher un modele magique.\n")


# ============================================================================
# PART 4 — Overthinking : quand scaler le thinking budget devient negatif
# ============================================================================
print("=" * 70)
print("PART 4 : Overthinking — la courbe de rendement")
print("=" * 70)


def reasoning_accuracy(budget_tokens: int, problem_difficulty: float) -> float:
    """
    Modele simplifie : la precision croit en log(budget) jusqu'a un plateau
    difficulte-dependant, puis DESCEND legerement (le modele 'se perd' dans
    trop de reflexion sur les problemes faciles). Inspire des courbes
    observees sur GSM8K, MATH500, AIME avec o1/R1.
    """
    saturation = 0.55 + 0.40 * problem_difficulty
    # croissance log, plafonnee puis legere decroissance au-dela du sweet spot
    sweet_spot = 2000 * (0.5 + problem_difficulty)
    if budget_tokens <= sweet_spot:
        return min(saturation, 0.3 + 0.3 * math.log10(max(budget_tokens, 1)))
    over = (budget_tokens - sweet_spot) / sweet_spot
    return max(0.3, saturation - 0.05 * over)


print(f"  {'budget':>8} | {'facile':>8} | {'dur':>8}")
for budget in [100, 500, 2000, 8000, 32000]:
    easy = reasoning_accuracy(budget, 0.1)
    hard = reasoning_accuracy(budget, 0.9)
    print(f"  {budget:>8} | {easy:>8.3f} | {hard:>8.3f}")
print("  → sur les problemes faciles, le sweet spot est ~2000 tokens ;")
print("    au-dela, la precision baisse. Sur les dur, plateau plus haut.\n")

print("FIN. Retenir :")
print("  - Self-consistency = test-time compute pauvre mais tres efficace.")
print("  - GRPO enleve le critic, advantage = (r - mean_groupe) / std.")
print("  - Le routeur planner/executor est la piece la plus rentable d'un LLM product.")
print("  - Un thinking budget mal regle coute cher ET degrade sur les taches faciles.")

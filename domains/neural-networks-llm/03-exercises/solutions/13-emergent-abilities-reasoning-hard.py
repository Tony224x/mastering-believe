"""
Solutions HARD — Jour 13 : Emergent abilities & reasoning
==========================================================
Exercices 7, 8 (hard).

NumPy + stdlib. Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/13-emergent-abilities-reasoning-hard.py
"""

import sys
import io
import random
from collections import Counter
from itertools import combinations
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

random.seed(42)
np.random.seed(42)


# ============================================================================
# EXERCICE 7 : Best-of-N — majority vote vs ORM vs PRM
# ============================================================================

print("=" * 70)
print("EXERCICE 7 : Test-time compute — majority vs ORM vs PRM")
print("=" * 70)


def gen_chain(k, p_step, true_answer, rng):
    """
    Une chaine de raisonnement: k etapes, chacune correcte avec proba p_step.
    Reponse finale = true_answer si TOUTES correctes, sinon une valeur erronee.
    Renvoie (steps_correct (liste bool), final_answer).
    """
    steps = [rng.random() < p_step for _ in range(k)]
    if all(steps):
        final = true_answer
    else:
        # erreur plausible : true +/- petit offset (les erreurs se dispersent)
        final = true_answer + rng.choice([-3, -2, -1, 1, 2, 3])
    return steps, final


def majority_vote(chains):
    finals = [c[1] for c in chains]
    return Counter(finals).most_common(1)[0][0]


def best_of_n_orm(chains, true_answer, rng):
    """ORM: note la reponse FINALE (1 si correcte else 0) + bruit gaussien."""
    best_score, best_ans = -np.inf, None
    for steps, final in chains:
        score = (1.0 if final == true_answer else 0.0) + rng.gauss(0, 0.6)
        if score > best_score:
            best_score, best_ans = score, final
    return best_ans


def best_of_n_prm(chains, true_answer, rng):
    """
    PRM: note CHAQUE etape (etape correcte -> score moyen plus eleve) + bruit.
    Score de la chaine = somme des scores d'etapes. Le PRM 'voit' ou ca derape,
    donc il discrimine mieux les chaines partiellement fausses.
    """
    best_score, best_ans = -np.inf, None
    for steps, final in chains:
        s = sum((0.9 if ok else 0.1) + rng.gauss(0, 0.3) for ok in steps)
        if s > best_score:
            best_score, best_ans = s, final
    return best_ans


def evaluate(k, p_step, N, n_problems=2000, seed=0):
    rng = random.Random(seed)
    acc_maj = acc_orm = acc_prm = 0
    for _ in range(n_problems):
        true = rng.randint(10, 99)
        chains = [gen_chain(k, p_step, true, rng) for _ in range(N)]
        acc_maj += (majority_vote(chains) == true)
        acc_orm += (best_of_n_orm(chains, true, rng) == true)
        acc_prm += (best_of_n_prm(chains, true, rng) == true)
    return acc_maj / n_problems, acc_orm / n_problems, acc_prm / n_problems


k, p_step = 6, 0.82  # chaine a 6 etapes, 82% par etape -> ~30% de chaines parfaites
print(f"\nProbleme a k={k} etapes, p_step={p_step} "
      f"(P(chaine parfaite)={p_step**k:.2f})")
print(f"\n{'N':>4s} {'majority':>10s} {'ORM':>8s} {'PRM':>8s}")
for N in [1, 2, 4, 8, 16, 32, 64]:
    am, ao, ap = evaluate(k, p_step, N, seed=N)
    print(f"{N:>4d} {am:>10.3f} {ao:>8.3f} {ap:>8.3f}")

print("""
Analyse:
- Les 3 strategies montent avec N : c'est le test-time compute scaling (plus de calcul
  a l'inference -> meilleure precision), la 'nouvelle scaling law' de o1/R1.
- PRM >= ORM >= majority sur les chaines longues : le PRM note CHAQUE etape, il sait
  reperer ou une chaine derape et favorise les chaines coherentes de bout en bout ;
  l'ORM ne voit que la reponse finale (signal plus pauvre) ; la majority vote echoue si
  le modele a un biais systematique (toutes les chaines font la MEME erreur -> elle gagne).
- Difference avec le scaling de pre-training : ici on ne change pas le modele, on depense
  plus de FLOPs A L'INFERENCE. C'est exactement ce que font o1 / DeepSeek R1.
""")


# ============================================================================
# EXERCICE 8 : Tree-of-Thought — recherche (variante Game of 24)
# ============================================================================

print("=" * 70)
print("EXERCICE 8 : Tree-of-Thought vs CoT lineaire (recherche)")
print("=" * 70)


def expand(numbers):
    """
    Genere les etats successeurs: combiner 2 nombres par une operation.
    Renvoie liste de (nouveaux_nombres, expression).
    """
    succ = []
    n = len(numbers)
    for i, j in combinations(range(n), 2):
        a, b = numbers[i], numbers[j]
        rest = [numbers[t] for t in range(n) if t != i and t != j]
        candidates = [(a + b, f"({a}+{b})"), (a * b, f"({a}*{b})"),
                      (a - b, f"({a}-{b})"), (b - a, f"({b}-{a})")]
        for val, expr in candidates:
            succ.append((rest + [val], expr))
    return succ


def reachable_heuristic(numbers, target):
    """
    Heuristique d'evaluation (sure/maybe/impossible -> score).
    Simplifie: si un nombre == target et c'est le dernier -> sure ;
    si tous les nombres sont deja > 2*target+10 (et pas de soustraction utile) -> peu probable.
    """
    if len(numbers) == 1:
        return 1.0 if numbers[0] == target else -1.0
    # maybe : on garde tant que c'est plausible
    if all(x > 2 * target + 20 for x in numbers):
        return 0.0  # improbable mais on n'elague pas brutalement
    return 0.5


def solve_linear_cot(numbers, target, rng, max_tries=1):
    """CoT lineaire: a chaque etape, choisir UNE continuation au hasard. Pas de backtrack."""
    for _ in range(max_tries):
        nums = list(numbers)
        while len(nums) > 1:
            succ = expand(nums)
            if not succ:
                break
            nums, _ = rng.choice(succ)  # un seul choix, pas de retour arriere
        if len(nums) == 1 and nums[0] == target:
            return True
    return False


def solve_tot_bfs(numbers, target, beam=3):
    """
    Tree-of-Thought, BFS avec beam: a chaque niveau, garder les `beam` meilleurs
    etats selon l'heuristique, elaguer les 'impossible'. Renvoie (resolu?, n_explored).
    """
    frontier = [(list(numbers), "")]
    explored = 0
    while frontier:
        next_frontier = []
        for nums, expr in frontier:
            explored += 1
            if len(nums) == 1:
                if nums[0] == target:
                    return True, explored
                continue
            for new_nums, op in expand(nums):
                h = reachable_heuristic(new_nums, target)
                if h < 0 and len(new_nums) > 1:
                    continue  # elagage
                next_frontier.append((new_nums, expr + op))
        # garder les beam meilleurs (par heuristique)
        next_frontier.sort(key=lambda s: -reachable_heuristic(s[0], target))
        frontier = next_frontier[:beam]
        if not frontier:
            break
    return False, explored


# Jeu de problemes (4 nombres, cible 24) — certains solubles
problems = [
    ([4, 6, 8, 2], 24),
    ([2, 3, 4, 5], 24),
    ([1, 1, 1, 1], 24),   # insoluble
    ([6, 6, 6, 6], 24),
    ([3, 3, 8, 8], 24),
    ([5, 5, 5, 1], 24),
]

rng = random.Random(0)
print("\nCoT lineaire (1 essai) vs ToT (BFS beam=5):")
print(f"{'nombres':>16s} {'cible':>5s} {'CoT':>6s} {'ToT':>6s} {'ToT_explored':>13s}")
cot_solved = tot_solved = 0
for nums, target in problems:
    cot = any(solve_linear_cot(nums, target, rng) for _ in range(1))
    tot, expl = solve_tot_bfs(nums, target, beam=5)
    cot_solved += cot
    tot_solved += tot
    print(f"{str(nums):>16s} {target:>5d} {str(cot):>6s} {str(tot):>6s} {expl:>13d}")
print(f"\nResolus : CoT lineaire={cot_solved}/{len(problems)}  ToT={tot_solved}/{len(problems)}")

# Effet de la largeur du beam
print("\nEffet de la largeur du beam (total resolus / etats explores):")
for b in [1, 3, 5, 10]:
    solved = total_expl = 0
    for nums, target in problems:
        ok, expl = solve_tot_bfs(nums, target, beam=b)
        solved += ok
        total_expl += expl
    print(f"  beam={b:>2d} : resolus={solved}/{len(problems)}  etats_explores={total_expl}")

print("""
Analyse:
- ToT bat le CoT lineaire car il EXPLORE plusieurs branches et ELAGUE les impasses :
  le CoT greedy s'engage sur un chemin et ne revient jamais en arriere -> il rate les
  problemes ou le bon chemin n'est pas le premier choix.
- Cout : ToT explore beaucoup plus d'etats (= beaucoup plus d'appels au modele en vrai).
- beam=1 ~ CoT greedy. Augmenter le beam resout plus mais coute plus.
- ToT n'apporte rien sur les problemes a 1 chemin evident (pas besoin d'explorer).
""")

print("=" * 70)
print("Fin solutions hard Jour 13.")
print("=" * 70)

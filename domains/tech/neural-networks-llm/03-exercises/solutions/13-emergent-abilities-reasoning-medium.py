"""
Solutions MEDIUM — Jour 13 : Emergent abilities & reasoning
============================================================
Exercices 4, 5, 6 (medium).

NumPy + stdlib (le 02-code de J13 est pur Python ; on utilise NumPy pour les
stats / regression, plus parlant). Chaque etape commentee avec le POURQUOI.

Run: python 03-exercises/solutions/13-emergent-abilities-reasoning-medium.py
"""

import sys
import io
import math
import random
from collections import Counter
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

random.seed(42)
np.random.seed(42)


# ============================================================================
# EXERCICE 4 : Self-consistency — binomiale exacte + simulation
# ============================================================================

print("=" * 70)
print("EXERCICE 4 : Self-consistency (binomiale exacte + Monte-Carlo)")
print("=" * 70)


def p_majority(p, n):
    """
    Borne BASSE (conservatrice) : proba que la bonne reponse soit en MAJORITE
    STRICTE (count(correct) > n/2). Loi binomiale. C'est la formule classique
    (Wang et al. 2022) : si correct a la majorite absolue, il gagne forcement.
    """
    need = n // 2 + 1  # strictement majoritaire pour n impair
    total = 0.0
    for k in range(need, n + 1):
        total += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return total


print("\nProba majorite correcte STRICTE (p=0.6) :")
for n in [1, 3, 5, 11, 25]:
    print(f"  n={n:>3d} : p_majority = {p_majority(0.6, n):.4f}")


def simulate_self_consistency(p, n, m=20, trials=50000, mode="strict"):
    """
    Simulation reelle : tirer n samples, voter.
      mode='strict'    -> compte un succes si count(correct) > n/2 (== la formule)
      mode='plurality' -> compte un succes si correct a le PLUS de votes (vote reel)
    WHY deux modes: la formule modelise la majorite STRICTE ; le vote reel est une
    PLURALITE (le plus de votes), bien plus facile a gagner quand les mauvaises
    reponses se dispersent sur m labels -> plurality >> strict.
    """
    rng = random.Random(123)
    success = 0
    for _ in range(trials):
        votes = []
        for _ in range(n):
            if rng.random() < p:
                votes.append("CORRECT")
            else:
                votes.append(f"wrong_{rng.randint(0, m - 1)}")  # erreur uniforme sur m
        counts = Counter(votes)
        n_correct = counts["CORRECT"]
        if mode == "strict":
            success += (n_correct > n / 2)
        else:  # plurality : correct a strictement le plus de votes
            max_wrong = max((c for k, c in counts.items() if k != "CORRECT"),
                            default=0)
            success += (n_correct > max_wrong)
    return success / trials


print("\nFormule (majorite stricte) vs simulation 'strict' (doit matcher):")
for n in [3, 5, 11]:
    f = p_majority(0.6, n)
    s = simulate_self_consistency(0.6, n, mode="strict")
    print(f"  n={n:>3d} : formule={f:.4f}  simu_strict={s:.4f}  ecart={abs(f-s):.4f}")

print("\nVote reel = PLURALITE (m=20 mauvaises reponses dispersees) : encore meilleur :")
for n in [3, 5, 11]:
    sp = simulate_self_consistency(0.6, n, mode="plurality")
    print(f"  n={n:>3d} : simu_plurality={sp:.4f} (> majorite stricte : "
          f"les erreurs dispersees ne s'accordent pas)")

# Nombre de samples pour 95%
print("\nNombre de samples (impair) pour atteindre 95% :")
for p in [0.55, 0.6, 0.7, 0.8]:
    n = 1
    while p_majority(p, n) < 0.95:
        n += 2
        if n > 999:
            break
    print(f"  p={p:.2f} : n={n}")

# Rendements decroissants
print("\nGain marginal (p=0.6):")
g_5_7 = p_majority(0.6, 7) - p_majority(0.6, 5)
g_21_23 = p_majority(0.6, 23) - p_majority(0.6, 21)
print(f"  5 -> 7   samples : +{g_5_7:.4f}")
print(f"  21 -> 23 samples : +{g_21_23:.4f}  (gain marginal bien plus faible)")
print("""
Analyse cout/benefice: cout lineaire en n, mais gain de fiabilite log-concave.
Self-consistency vaut le coup pour les taches critiques (math, code, medical) ou
une erreur coute cher ; inutile pour les questions a 1 etape ou la generation creative.
""")


# ============================================================================
# EXERCICE 5 : CoT vs direct — modele d'erreur par etape
# ============================================================================

print("=" * 70)
print("EXERCICE 5 : CoT vs direct (modele d'erreur par etape)")
print("=" * 70)


def solve(start, ops):
    r = start
    for op, n in ops:
        r = {'+': r + n, '-': r - n, 'x': r * n}[op]
    return r


def gen_problem(k, rng):
    start = rng.randint(1, 20)
    ops = []
    for _ in range(k):
        op = rng.choice(['+', '-', 'x'])
        n = rng.randint(2, 4) if op == 'x' else rng.randint(1, 20)
        ops.append((op, n))
    return start, ops


def answer(start, ops, e_step, rng):
    """
    Resout en propageant une erreur de proba e_step a CHAQUE etape.
    WHY: en mode direct, e_step est eleve (le modele ne 'voit' pas les
    intermediaires) ; en CoT, e_step est faible (il les ecrit).
    """
    r = start
    for op, n in ops:
        r = {'+': r + n, '-': r - n, 'x': r * n}[op]
        if rng.random() < e_step:
            r += rng.choice([-2, -1, 1, 2])  # erreur qui se propage
    return r


def accuracy(k, e_step, n_problems=2000, seed=0):
    rng = random.Random(seed)
    correct = 0
    for _ in range(n_problems):
        start, ops = gen_problem(k, rng)
        true = solve(start, ops)
        if answer(start, ops, e_step, rng) == true:
            correct += 1
    return correct / n_problems


e_direct, e_cot = 0.25, 0.03
print(f"\ne_direct={e_direct}, e_cot={e_cot}")
print(f"{'etapes':>7s} {'direct':>9s} {'CoT':>9s} {'theorie_dir':>12s} {'theorie_cot':>12s}")
for k in [1, 3, 5, 8, 12]:
    ad = accuracy(k, e_direct, seed=k)
    ac = accuracy(k, e_cot, seed=k + 100)
    print(f"{k:>7d} {ad:>9.3f} {ac:>9.3f} {(1-e_direct)**k:>12.3f} "
          f"{(1-e_cot)**k:>12.3f}")

print("""
-> Direct s'effondre exponentiellement: (1-0.25)^k. CoT decroit lentement: (1-0.03)^k.
   L'ecart devient enorme des ~5-8 etapes.
""")

# Self-consistency sur le CoT
def accuracy_sc(k, e_step, n_vote=5, n_problems=2000, seed=0):
    rng = random.Random(seed)
    correct = 0
    for _ in range(n_problems):
        start, ops = gen_problem(k, rng)
        true = solve(start, ops)
        votes = [answer(start, ops, e_step, rng) for _ in range(n_vote)]
        if Counter(votes).most_common(1)[0][0] == true:
            correct += 1
    return correct / n_problems


print("Self-consistency (5 votes) sur CoT :")
for k in [5, 8, 12]:
    ac = accuracy(k, e_cot, seed=k + 100)
    asc = accuracy_sc(k, e_cot, n_vote=5, seed=k + 200)
    print(f"  k={k:>2d} : CoT seul={ac:.3f}  CoT+SC5={asc:.3f}  gain=+{asc-ac:.3f}")

print("""
Analyse: le CoT aide surtout sur les problemes LONGS car l'erreur direct s'accumule
multiplicativement ((1-e)^k). Sur 1 etape, direct et CoT sont quasi equivalents.
Plus il y a d'etapes, plus 'verbaliser' chaque resultat intermediaire paie.
""")


# ============================================================================
# EXERCICE 6 : In-context learning — modele jouet (regression sur le prompt)
# ============================================================================

print("=" * 70)
print("EXERCICE 6 : In-context learning (modele jouet)")
print("=" * 70)


def make_task(d, rng):
    """Une 'tache' = fonction lineaire inconnue y = w.x + b."""
    w = rng.standard_normal(d)
    b = rng.standard_normal()
    return w, b


def in_context_predict(X_demos, y_demos, x_query):
    """
    'Apprend' la fonction depuis les demos par moindres carres (avec biais),
    puis predit y_query. Analogue de ce que fait l'attention dans le forward.
    """
    # Ajouter une colonne de 1 pour le biais
    A = np.hstack([X_demos, np.ones((X_demos.shape[0], 1))])
    # moindres carres : theta = argmin ||A theta - y||^2
    theta, *_ = np.linalg.lstsq(A, y_demos, rcond=None)
    aq = np.append(x_query, 1.0)
    return aq @ theta


d = 4
rng = np.random.default_rng(0)
print("\nErreur de prediction vs nombre de demos (few-shot scaling):")
print(f"{'k_demos':>8s} {'err_moy':>10s}")
for k in [1, 2, 5, 10, 20]:
    errs = []
    for _ in range(500):
        w, b = make_task(d, rng)
        X = rng.standard_normal((k, d))
        y = X @ w + b
        xq = rng.standard_normal(d)
        yq_true = xq @ w + b
        yq_pred = in_context_predict(X, y, xq)
        errs.append(abs(yq_pred - yq_true))
    print(f"{k:>8d} {np.mean(errs):>10.4f}")
print("  -> plus de demos = erreur plus faible (zero/one/few-shot du cours).")

# Robustesse au bruit
print("\nAvec bruit sur les y des demos (sigma=0.5):")
print(f"{'k_demos':>8s} {'err_moy':>10s}")
for k in [2, 5, 10, 20, 50]:
    errs = []
    for _ in range(500):
        w, b = make_task(d, rng)
        X = rng.standard_normal((k, d))
        y = X @ w + b + rng.standard_normal(k) * 0.5  # bruit
        xq = rng.standard_normal(d)
        yq_true = xq @ w + b
        yq_pred = in_context_predict(X, y, xq)
        errs.append(abs(yq_pred - yq_true))
    print(f"{k:>8d} {np.mean(errs):>10.4f}")
print("""
Analyse:
- Avec du bruit, il faut PLUS de demos pour 'moyenner' le bruit et bien estimer w.
- Ce modele jouet capture l'idee de l'ICL : inferer la regle a partir des exemples
  DANS le prompt, sans mise a jour de poids. Limite : un vrai LLM fait de l'ICL sur
  des taches non lineaires, du raisonnement, du format... pas juste de la regression.
""")

print("=" * 70)
print("Fin solutions medium Jour 13.")
print("=" * 70)

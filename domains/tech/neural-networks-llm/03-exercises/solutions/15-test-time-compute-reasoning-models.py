"""
Solutions EASY — Jour 15 : Test-time compute & reasoning models
================================================================
Exercices 1, 2, 3 (faciles).

Pur Python + numpy (comme 02-code/15-test-time-compute-reasoning-models.py,
qui est lui-meme stdlib). Aucun framework. Chaque etape non triviale est
commentee avec le POURQUOI.

Run: python 03-exercises/solutions/15-test-time-compute-reasoning-models.py
"""

import sys
import io
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# EXERCISE 1: Self-consistency a la main (majority vote)
# ============================================================================
print("=" * 70)
print("EXERCISE 1: Self-consistency / majority vote")
print("=" * 70)


def self_consistency(answers):
    """
    Majority vote. Tie-break : la plus PETITE valeur gagne.
    POURQUOI un tie-break deterministe : sans regle fixe, deux modes a egalite
    rendraient l'eval non reproductible (depend de l'ordre d'insertion).
    """
    counts = Counter(answers)
    best = max(counts.values())
    # Parmi les valeurs ayant le compte max, on prend la plus petite.
    winners = [v for v, c in counts.items() if c == best]
    return min(winners)


problems = {
    "A": ([56, 56, 49, 56, 64], 56),
    "B": ([49, 64, 56, 48, 49], 56),
    "C": ([56, 14, 56, 56, 56], 56),
}

one_shot_correct = 0
vote_correct = 0
for name, (answers, truth) in problems.items():
    one_shot = answers[0]                 # 1) one-shot = la 1ere reponse
    vote = self_consistency(answers)      # 2) majority vote sur K=5
    one_shot_correct += int(one_shot == truth)
    vote_correct += int(vote == truth)
    print(f"  Probleme {name}: one-shot={one_shot} (truth {truth}) | "
          f"vote={vote} {'OK' if vote == truth else 'FAUX'}")

n = len(problems)
print(f"\n  Accuracy one-shot   = {one_shot_correct}/{n} = {one_shot_correct / n:.3f}")
print(f"  Accuracy vote (K=5) = {vote_correct}/{n} = {vote_correct / n:.3f}")
# Q3 : sur B, la bonne reponse (56) est MINORITAIRE (1 vote) face a 49 (2 votes).
# Les erreurs sont assez concordantes pour battre la verite -> le vote elit 49.
print("  Q3 : sur B le vote echoue car 56 est minoritaire et l'erreur 49")
print("       est repetee 2x. Le vote n'aide que si la verite est le MODE.")
# Q4 : condition. Si chaque mode d'erreur est moins frequent que la bonne
# reponse, voter concentre la masse sur la verite quand K grandit. Si le
# modele se trompe plus souvent (et de facon concordante) qu'il n'a raison,
# le vote AMPLIFIE l'erreur.
print("  Q4 : le vote aide ssi la bonne reponse est, en esperance, le mode")
print("       (proba de la verite > proba de chaque mode d'erreur).\n")


# ============================================================================
# EXERCISE 2: Advantage GRPO d'un groupe
# ============================================================================
print("=" * 70)
print("EXERCISE 2: GRPO advantage = (r - mean) / std")
print("=" * 70)


def grpo_advantages(rewards):
    """
    Advantage group-relative. std = ecart-type de POPULATION (diviseur N),
    comme dans le papier GRPO (statistique du groupe, pas estimateur non biaise).
    On clippe std a 1e-6 : si tout le groupe a la meme reward, std=0 -> div/0.
    """
    g = len(rewards)
    mean = sum(rewards) / g
    var = sum((r - mean) ** 2 for r in rewards) / g          # population
    std = var ** 0.5
    std_safe = max(std, 1e-6)                                 # garde-fou div/0
    adv = [(r - mean) / std_safe for r in rewards]
    return mean, std, adv


rewards = [1, 0, 1, 1, 0, 0]
mean, std, adv = grpo_advantages(rewards)
print(f"  rewards = {rewards}")
print(f"  mean = {mean}  std = {std}")
print(f"  advantages = {[round(a, 3) for a in adv]}")
# Signe : advantage > 0 => on AUGMENTE la proba de cette reponse (elle bat la
# moyenne du groupe) ; advantage < 0 => on la DIMINUE.
pos = [i for i, a in enumerate(adv) if a > 0]
neg = [i for i, a in enumerate(adv) if a < 0]
print(f"  advantage > 0 (proba augmentee) : indices {pos}  (reponses correctes)")
print(f"  advantage < 0 (proba diminuee)  : indices {neg}  (reponses fausses)")

# Cas degenere : tout correct.
mean2, std2, adv2 = grpo_advantages([1, 1, 1, 1, 1, 1])
print(f"\n  Cas degenere rewards=[1]*6 : std={std2} -> clip a 1e-6")
print(f"  advantages = {[round(a, 3) for a in adv2]}  (~0 : aucun signal relatif)")
print("  -> un groupe homogene n'apprend RIEN (tout le monde pareil).")
# Q5 : GRPO supprime le critic (value net) de PPO, un 2e reseau de la taille
# de la policy. Baseline = moyenne du groupe -> ~2x moins de memoire.
print("  Q5 : GRPO remplace le critic de PPO par mean(groupe) -> ~2x moins")
print("       de memoire (pas de value network a la taille de la policy).\n")


# ============================================================================
# EXERCISE 3: Router LLM classique vs reasoning model
# ============================================================================
print("=" * 70)
print("EXERCISE 3: Router cout/latence/qualite")
print("=" * 70)

PROFILES = {
    "haiku":         {"cost": 0.004, "lat": 0.2, "q_reason": 0.55, "q_extract": 0.90},
    "sonnet":        {"cost": 0.015, "lat": 0.4, "q_reason": 0.72, "q_extract": 0.95},
    "opus_thinking": {"cost": 0.075, "lat": 3.0, "q_reason": 0.94, "q_extract": 0.95},
}


def route(task_type, quality_target, latency_budget_s, output_tokens):
    """
    1) filtre latence  2) filtre qualite  3) min cout.
    C'est CETTE fonction qui determine ~80% de la facture d'un produit LLM,
    bien plus que le prompt : router une extraction vers un reasoning model
    coute x10-20 pour zero gain.
    """
    candidates = []
    for name, p in PROFILES.items():
        latency = p["lat"] * output_tokens / 1000              # latence reelle
        if latency > latency_budget_s:                          # 1) filtre latence
            continue
        q = p["q_reason"] if task_type == "reasoning" else p["q_extract"]
        if q < quality_target:                                 # 2) filtre qualite
            continue
        cost = p["cost"] * output_tokens / 1000
        candidates.append((cost, name, q, latency))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])                        # 3) min cout
    return candidates[0]


scenarios = [
    ("extraction", 0.85, 3.0, 500, "Extraire un nom de contact"),
    ("reasoning", 0.90, 60.0, 2000, "Preuve math complexe"),
    ("reasoning", 0.92, 2.0, 1000, "Math dur ET user-facing"),
]
for task_type, q, lat, tok, label in scenarios:
    r = route(task_type, q, lat, tok)
    if r is None:
        print(f"  [{label}] -> AUCUN modele ne satisfait les contraintes")
    else:
        cost, name, qual, latency = r
        print(f"  [{label}] -> {name}  cost=${cost:.4f}  lat={latency:.2f}s  q={qual:.2f}")

print("  Q3 cas 3 : opus_thinking depasse le budget latence (3.0s/1k -> 3.0s")
print("       pour 1000 tok > 2.0s), et haiku/sonnet n'atteignent pas q_reason=0.92.")
print("       -> changer le PRODUIT (UX async, decoupage) pas chercher un modele magique.")
print("  Q4 : a qualite egale on prend le moins cher/rapide ; reasoning sur de")
print("       l'extraction = gaspillage pur (overkill).")
print()
print("FIN. Retenir :")
print("  - Self-consistency = test-time compute pauvre mais efficace (vote).")
print("  - GRPO advantage = (r - mean_groupe) / std, pas de critic.")
print("  - Le routeur est la piece la plus rentable d'un produit LLM.")

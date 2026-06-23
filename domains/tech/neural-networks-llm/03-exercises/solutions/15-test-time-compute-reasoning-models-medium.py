"""
Solutions MEDIUM — Jour 15 : Test-time compute & reasoning models
=================================================================
Exercices 4, 5, 6 (medium).

Pur NumPy. Seed fixe pour des resultats deterministes. Chaque etape non
triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/15-test-time-compute-reasoning-models-medium.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

rng = np.random.default_rng(7)


# ============================================================================
# EXERCISE 4: Courbe de scaling de la self-consistency
# ============================================================================
print("=" * 70)
print("EXERCISE 4: Self-consistency accuracy vs K")
print("=" * 70)


def noisy_solver(a, b, m, noise):
    """
    Repond (a+b)%m mais, avec proba `noise`, produit une erreur PLAUSIBLE.
    POURQUOI plausible et non uniforme : un vrai LLM fait des erreurs
    structurees (off-by-one, oubli du modulo) ; le majority vote marche
    justement parce que ces erreurs sont DIVERSES et donc rarement majoritaires.
    """
    correct = (a + b) % m
    if rng.random() < noise:
        return int(rng.choice([correct - 1, correct + 1, (a - b) % m, (a + b)]))
    return correct


def self_consistency(a, b, m, k, noise):
    """Majority vote sur k echantillons (numpy.bincount sur des entiers >=0)."""
    votes = [noisy_solver(a, b, m, noise) % (10 * m) for _ in range(k)]
    # bincount + argmax = mode. On a borne les valeurs pour rester >=0.
    return int(np.argmax(np.bincount(votes)))


def measure(noise, ks, n_problems=500, m=97):
    probs = [(int(rng.integers(1, 1000)), int(rng.integers(1, 1000)), m)
             for _ in range(n_problems)]
    accs = []
    for k in ks:
        correct = 0
        for a, b, mm in probs:
            pred = self_consistency(a, b, mm, k, noise)
            if pred == (a + b) % mm:
                correct += 1
        accs.append(correct / n_problems)
    return accs


KS = [1, 3, 5, 10, 20, 40, 80]
accs = measure(0.4, KS)
print(f"  noise=0.4   {'K':>4} | {'accuracy':>9}")
for k, acc in zip(KS, accs):
    print(f"              {k:>4} | {acc:>9.3f}")

# Gains marginaux + cout : le gain par token depense s'effondre.
gain_1_5 = accs[2] - accs[0]
gain_40_80 = accs[-1] - accs[-2]
print(f"\n  Gain k=1 -> k=5   : {gain_1_5:+.3f}  (cout x5)")
print(f"  Gain k=40 -> k=80 : {gain_40_80:+.3f}  (cout x2, pour quasi rien)")
print(f"  -> plateau ~k=20-40. Sweet spot cout/qualite autour de k=5-15.")

# Q6 : noise eleve -> la masse des erreurs concordantes peut depasser la verite.
print("\n  Effet du noise (k=40) :")
for noise in [0.2, 0.4, 0.6]:
    acc = measure(noise, [40])[0]
    print(f"    noise={noise}  accuracy(k=40) = {acc:.3f}")
print("  -> a noise tres eleve, le vote n'aide plus : trop d'erreurs, parfois")
print("     concordantes, noient la bonne reponse.\n")


# ============================================================================
# EXERCISE 5: GRPO sur bandit, avec/sans normalisation par std
# ============================================================================
print("=" * 70)
print("EXERCISE 5: GRPO bandit — impact de la normalisation par std")
print("=" * 70)

TRUE_REWARDS = np.array([0.1, 0.3, 0.8, 0.5])   # bras 2 optimal
N_ARMS = len(TRUE_REWARDS)


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def run_grpo(variant, sigma, steps=300, G=16, lr=0.1, seed=0):
    """
    variant: 'raw' (r-mean), 'norm' ((r-mean)/std), 'none' (r, REINFORCE).
    Retourne la trajectoire de proba du bras optimal (bras 2).
    """
    local = np.random.default_rng(seed)
    logits = np.zeros(N_ARMS)
    traj = []
    for _ in range(steps):
        probs = softmax(logits)
        # 1) echantillonner un groupe de G actions depuis la meme policy
        actions = local.choice(N_ARMS, size=G, p=probs)
        rewards = TRUE_REWARDS[actions] + local.normal(0, sigma, size=G)

        # 2) advantage selon la variante
        if variant == "none":
            adv = rewards                                   # pas de baseline
        else:
            adv = rewards - rewards.mean()                 # baseline = mean groupe
            if variant == "norm":
                adv = adv / max(rewards.std(), 1e-6)       # z-score -> robuste a l'echelle

        # 3) policy gradient : grad logit_i = (1[i==a] - probs[i])
        grad = np.zeros(N_ARMS)
        for a, ad in zip(actions, adv):
            onehot = np.zeros(N_ARMS)
            onehot[a] = 1.0
            grad += ad * (onehot - probs)
        logits += lr * grad / G
        traj.append(softmax(logits)[2])                    # proba du bras optimal
    return np.array(traj)


for sigma in [0.05, 0.3, 0.8]:
    print(f"\n  sigma={sigma}  (proba du bras optimal a la fin)")
    for variant in ["none", "raw", "norm"]:
        traj = run_grpo(variant, sigma, seed=1)
        # variance des deltas comme proxy de stabilite des updates
        instability = float(np.std(np.diff(traj)))
        print(f"    variant={variant:>4} | p_final={traj[-1]:.3f} | "
              f"instabilite(std des deltas)={instability:.4f}")
print("\n  -> 'none' (sans baseline) : plus haute variance, convergence lente.")
print("  -> 'norm' (std-normalise) : le plus stable, surtout a sigma eleve.")
print("  Q6 : diviser par std transforme l'advantage en z-score -> le pas")
print("       d'update est invariant a l'echelle des rewards ([0,1] ou [0,100]).\n")


# ============================================================================
# EXERCISE 6: Budget de thinking tokens & overthinking
# ============================================================================
print("=" * 70)
print("EXERCISE 6: Thinking budget — courbe de rendement & routeur cout-conscient")
print("=" * 70)


def reasoning_accuracy(budget_tokens, difficulty):
    """
    Croissance log(budget) jusqu'a un sweet spot dependant de la difficulte,
    plateau a `saturation`, puis LEGERE decroissance (overthinking sur facile).
    Inspire des courbes GSM8K/MATH/AIME avec o1/R1.
    """
    saturation = 0.55 + 0.40 * difficulty
    sweet_spot = 2000 * (0.5 + difficulty)
    if budget_tokens <= sweet_spot:
        return min(saturation, 0.3 + 0.3 * np.log10(max(budget_tokens, 1)))
    over = (budget_tokens - sweet_spot) / sweet_spot
    return max(0.3, saturation - 0.05 * over)               # declin au-dela du pic


BUDGETS = [100, 500, 2000, 8000, 32000]
print(f"  {'budget':>8} | {'facile':>8} | {'moyen':>8} | {'dur':>8}")
for b in BUDGETS:
    print(f"  {b:>8} | {reasoning_accuracy(b, 0.1):>8.3f} | "
          f"{reasoning_accuracy(b, 0.5):>8.3f} | {reasoning_accuracy(b, 0.9):>8.3f}")
print("  -> sur facile, pic tot (~2000) puis declin ; sur dur, plateau haut/tardif.")


def optimal_budget(difficulty, candidate_budgets):
    """Budget maximisant l'accuracy pure (ignore le cout)."""
    return max(candidate_budgets, key=lambda b: reasoning_accuracy(b, difficulty))


print("\n  optimal_budget (accuracy pure) :")
for diff in [0.1, 0.5, 0.9]:
    print(f"    difficulty={diff} -> budget={optimal_budget(diff, BUDGETS)}")


def cost_aware_budget(difficulty, candidate_budgets, lam, price=1e-5):
    """Maximise accuracy - lam * cout. Plus lam est grand, plus on rogne le budget."""
    return max(candidate_budgets,
               key=lambda b: reasoning_accuracy(b, difficulty) - lam * (b * price))


print("\n  Routeur cout-conscient (difficulty=0.5) :")
for lam in [0.0, 1.0, 5.0, 20.0]:
    b = cost_aware_budget(0.5, BUDGETS, lam)
    print(f"    lambda={lam:>5} -> budget={b}")
print("  -> le budget choisi DIMINUE quand lambda (sensibilite cout) augmente.")

# Economie d'un routeur adaptatif vs budget fixe a 32000 pour tout.
# Mix : 70% faciles (diff 0.1), 30% durs (diff 0.9), 1000 requetes.
n_easy, n_hard = 700, 300
adaptive_tokens = n_easy * optimal_budget(0.1, BUDGETS) + n_hard * optimal_budget(0.9, BUDGETS)
fixed_tokens = (n_easy + n_hard) * 32000
acc_adaptive = (n_easy * reasoning_accuracy(optimal_budget(0.1, BUDGETS), 0.1)
                + n_hard * reasoning_accuracy(optimal_budget(0.9, BUDGETS), 0.9)) / 1000
acc_fixed = (n_easy * reasoning_accuracy(32000, 0.1)
             + n_hard * reasoning_accuracy(32000, 0.9)) / 1000
print(f"\n  Mix 70% facile / 30% dur (1000 requetes) :")
print(f"    Adaptatif : {adaptive_tokens:>10} tokens | accuracy {acc_adaptive:.3f}")
print(f"    Budget fixe 32k : {fixed_tokens:>10} tokens | accuracy {acc_fixed:.3f}")
print(f"    Economie tokens : {1 - adaptive_tokens / fixed_tokens:.1%} "
      f"(accuracy comparable, voire meilleure car overthinking evite sur facile)")
print("\n  Anti-overthinking en prod :")
print("    - cap `budget_tokens` (Claude) par requete")
print("    - routeur/classifieur de difficulte EN AMONT (choix modele/budget)")
print("\nFIN MEDIUM.")

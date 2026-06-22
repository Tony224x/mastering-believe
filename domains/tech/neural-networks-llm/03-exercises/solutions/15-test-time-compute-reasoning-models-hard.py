"""
Solutions HARD — Jour 15 : Test-time compute & reasoning models
===============================================================
Exercices 7, 8 (hard).

Pur NumPy. Seed fixe -> resultats deterministes. Commentaires WHY.

Run: python 03-exercises/solutions/15-test-time-compute-reasoning-models-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# EXERCISE 7: Best-of-N (reward model bruite) vs self-consistency
# ============================================================================
print("=" * 70)
print("EXERCISE 7: Best-of-N vs self-consistency")
print("=" * 70)


def sample_solutions(rng, N, M_distractors=8):
    """
    Genere N solutions candidates pour UN probleme.
      q          : qualite latente ~ Beta(2,2) (bimodal-ish autour de 0.5)
      correct    : q > 0.5
      final_ans  : 0 (la "vraie" reponse) si correct, sinon un distracteur 1..M
    POURQUOI ce modele : best-of-N exploite q via un RM ; self-consistency
    exploite la CONCORDANCE des final_ans (les bonnes convergent vers 0).
    """
    q = rng.beta(2.0, 2.0, size=N)
    correct = q > 0.5
    final = np.where(correct, 0, rng.integers(1, M_distractors + 1, size=N))
    return q, correct, final


def best_of_n(rng, q, correct, rm_noise):
    """RM score = q + bruit. On retient la solution de score max ; succes = elle est correcte."""
    rm_scores = q + rng.normal(0, rm_noise, size=len(q))
    pick = int(np.argmax(rm_scores))
    return bool(correct[pick])


def self_consistency_pick(correct, final):
    """Majority vote sur final_ans ; succes = le mode est la vraie reponse (0)."""
    mode = np.bincount(final).argmax()
    return mode == 0


def oracle(correct):
    """Borne sup : succes ssi AU MOINS une solution correcte existe dans le lot."""
    return bool(correct.any())


def experiment(N, rm_noise, n_problems=1000, M_distractors=8, seed=0):
    rng = np.random.default_rng(seed)
    bon = sc = orc = 0
    for _ in range(n_problems):
        q, correct, final = sample_solutions(rng, N, M_distractors)
        bon += best_of_n(rng, q, correct, rm_noise)
        sc += self_consistency_pick(correct, final)
        orc += oracle(correct)
    return bon / n_problems, sc / n_problems, orc / n_problems


print(f"  {'N':>4} {'rm_noise':>9} | {'best-of-N':>10} {'self-cons':>10} {'oracle':>8}")
for N in [1, 4, 8, 16, 32]:
    for rm_noise in [0.0, 0.2, 0.5, 1.0]:
        bon, sc, orc = experiment(N, rm_noise)
        print(f"  {N:>4} {rm_noise:>9} | {bon:>10.3f} {sc:>10.3f} {orc:>8.3f}")
    print()

# Analyse : a rm_noise=0 best-of-N approche l'oracle ; il s'effondre vers du
# random pick quand rm_noise grandit. self-consistency est independant du RM.
bon0, sc0, orc0 = experiment(16, 0.0)
bon1, sc1, orc1 = experiment(16, 1.0)
print(f"  N=16 rm_noise=0 : best-of-N={bon0:.3f} ~ oracle={orc0:.3f} (RM parfait).")
print(f"  N=16 rm_noise=1 : best-of-N={bon1:.3f} < self-cons={sc1:.3f} "
      f"(RM bruite -> SC gagne).")

# Contre-exemple : peu de distracteurs -> les erreurs CONCORDENT par hasard,
# self-consistency peut elire une mauvaise reponse alors qu'un RM (meme moyen)
# saurait reperer une solution correcte.
print("\n  Contre-exemple SC echoue (peu de distracteurs, M=1) :")
bon_ce, sc_ce, orc_ce = experiment(16, 0.2, M_distractors=1)
print(f"    M=1 : best-of-N={bon_ce:.3f}  self-cons={sc_ce:.3f}  oracle={orc_ce:.3f}")
print("    -> avec M=1, toutes les erreurs disent la meme chose : si les fausses")
print("       sont majoritaires, SC elit l'erreur ; best-of-N reste protege par q.")

print("\n  Synthese decisionnelle :")
print("    RM de qualite dispo            -> best-of-N (approche l'oracle)")
print("    pas de RM, reponse extractible -> self-consistency (gratuit)")
print("    erreurs concordantes probables -> eviter SC pur, prefere RM/verif")
print("    reponse non verifiable/diffuse -> ni l'un ni l'autre fiable\n")


# ============================================================================
# EXERCISE 8: Pipeline RL reasoning (GRPO) + reward hacking + mitigation
# ============================================================================
print("=" * 70)
print("EXERCISE 8: GRPO sur tache verifiable + reward hacking")
print("=" * 70)

# p_correct cachee par strategie. Strategie 2 est la meilleure (0.9).
P_CORRECT = np.array([0.2, 0.4, 0.9, 0.6, 0.1])
N_STRAT = len(P_CORRECT)


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def rollout(rng, strat, cheat_strat_id=None, robust_verify=True):
    """
    Simule une generation : la strategie produit une reponse correcte avec
    proba p_correct, et un format <think>..</think><answer>..</answer> bien
    forme (proba qui monte avec l'entrainement -> ici on suppose bien forme).

    cheat_strat_id : si fourni, cette strategie est la "triche" : elle ECRIT
      la bonne reponse dans le <think> sans resoudre dans <answer>.

    robust_verify=False  : verif NAIVE (la sortie CONTIENT la reponse n'importe ou)
                           -> la triche passe.
    robust_verify=True   : on extrait UNIQUEMENT le <answer> -> la triche echoue.
    """
    format_bonus = 0.1                                  # balises bien formees
    if cheat_strat_id is not None and strat == cheat_strat_id:
        # La triche met la reponse dans <think>, <answer> est vide/faux.
        contains_answer_anywhere = True
        answer_block_correct = False                   # n'a pas vraiment resolu
    else:
        solved = rng.random() < P_CORRECT[strat]
        contains_answer_anywhere = solved
        answer_block_correct = solved

    is_correct = answer_block_correct if robust_verify else contains_answer_anywhere
    return 1.0 * is_correct + format_bonus, answer_block_correct


def train_grpo(rng, cheat_strat_id=None, robust_verify=True,
               steps=400, G=16, lr=0.2):
    """Boucle GRPO. Retourne (trajectoire proba meilleure strat reelle,
    trajectoire reward moyen, proba finale de la triche si elle existe)."""
    n = N_STRAT + (1 if cheat_strat_id is not None else 0)
    logits = np.zeros(n)
    p_best_traj, reward_traj = [], []
    for _ in range(steps):
        probs = softmax(logits)
        actions = rng.choice(n, size=G, p=probs)
        rewards = np.array([
            rollout(rng, a if a < N_STRAT else cheat_strat_id,
                    cheat_strat_id=cheat_strat_id, robust_verify=robust_verify)[0]
            for a in actions
        ])
        adv = (rewards - rewards.mean()) / max(rewards.std(), 1e-6)   # GRPO
        grad = np.zeros(n)
        for a, ad in zip(actions, adv):
            onehot = np.zeros(n)
            onehot[a] = 1.0
            grad += ad * (onehot - probs)
        logits += lr * grad / G
        p = softmax(logits)
        p_best_traj.append(p[2])                         # vraie meilleure strat (id 2)
        reward_traj.append(rewards.mean())
    final = softmax(logits)
    cheat_p = final[-1] if cheat_strat_id is not None else 0.0
    return np.array(p_best_traj), np.array(reward_traj), float(cheat_p)


# (1)-(3) Entrainement propre : converge vers la strategie 2.
rng = np.random.default_rng(0)
p_best, rew, _ = train_grpo(rng, cheat_strat_id=None)
print(f"  [Propre] proba strat optimale : {p_best[0]:.3f} -> {p_best[-1]:.3f}")
print(f"           reward moyen         : {rew[0]:.3f} -> {rew[-1]:.3f}")
print("           -> la policy decouvre la meilleure strategie verifiable.")

# (4) Reward hacking : on ajoute une strategie 'triche' (id = N_STRAT) et une
# verification NAIVE. La policy converge vers la triche (reward facile).
rng = np.random.default_rng(0)
cheat_id = N_STRAT
p_best_h, rew_h, cheat_p_h = train_grpo(rng, cheat_strat_id=cheat_id,
                                        robust_verify=False)
print(f"\n  [Verif NAIVE] proba de la TRICHE finale : {cheat_p_h:.3f}")
print(f"                proba strat 2 finale       : {p_best_h[-1]:.3f}")
print("                -> reward hacking : la triche capte 'contains answer' partout.")

# (5) Mitigation : verification ROBUSTE (extraction du seul <answer>, match exact).
rng = np.random.default_rng(0)
p_best_m, rew_m, cheat_p_m = train_grpo(rng, cheat_strat_id=cheat_id,
                                        robust_verify=True)
print(f"\n  [Verif ROBUSTE] proba de la TRICHE finale : {cheat_p_m:.3f}")
print(f"                  proba strat 2 finale       : {p_best_m[-1]:.3f}")
print("                  -> la triche n'a plus que le format_bonus, la vraie")
print("                     strategie reprend le dessus.")

# (6) Analyse.
print("\n  Analyse :")
print("    - format_bonus seul peut detourner l'optim si la reward de")
print("      correction est rare/bruitee : le modele maximise le terme facile.")
print("    - Autres reward hacking (cours) :")
print("        * corrompre les tests unitaires -> parade : tests caches/immuables")
print("        * imprimer la reponse attendue dans le CoT -> parade : juge LLM")
print("          + verif croisee (extraction stricte du bloc final).")
print("    - Distillation > RL pour petits modeles : un 7B n'a pas la base de")
print("      connaissances pour DECOUVRIR les strategies reasoning par")
print("      exploration RL ; distiller transmet les patterns deja trouves.")
print("\nFIN HARD.")

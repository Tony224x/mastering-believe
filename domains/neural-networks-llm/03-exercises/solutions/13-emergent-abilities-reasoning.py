"""
Solutions — Jour 13 : Emergent abilities & reasoning

Run: python 03-exercises/solutions/13-emergent-abilities-reasoning.py
"""

import sys
import io
from math import comb

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — CoT on trick problems
# ============================================================================

print("=" * 70)
print("Exercice 1: CoT sur des problemes piege")
print("=" * 70)

print("""
Probleme 1: 5 personnes, 2 bras chacune.
  Direct: 10 bras (correct, trivial)
  CoT:
    1. Il y a 5 personnes.
    2. Chaque personne a 2 bras.
    3. Total = 5 * 2 = 10.
    4. Reponse: 10 bras.

Probleme 2: Marie a 3x plus de pommes que Pierre, total 16.
  Direct: souvent 12 et 4 (correct) mais peut dire '3+1 = 4, 16/4 = 4'
          et s'embrouiller sur qui a quoi.
  CoT:
    1. Soit p = nombre de pommes de Pierre.
    2. Marie a 3p pommes.
    3. Ensemble: p + 3p = 4p = 16.
    4. Donc p = 4. Pierre a 4 pommes.
    5. Marie a 3 * 4 = 12 pommes.
    Reponse: Pierre=4, Marie=12.

Probleme 3: Train de Paris a 8h 100 km/h. Train de Lyon a 9h 120 km/h.
            Distance = 460 km. Quand se croisent-ils ?
  Direct: impossible a deviner sans calculer.
  CoT:
    1. A 8h, train A part de Paris.
    2. De 8h a 9h, train A parcourt 100 km. Donc a 9h, train A est a
       100 km de Paris, soit a 360 km de Lyon.
    3. A 9h, train B part de Lyon. Les deux se rapprochent a
       100 + 120 = 220 km/h.
    4. Distance restante entre eux: 360 km.
    5. Temps pour se croiser: 360 / 220 = 1.636 h = 1h 38min
    6. Heure: 9h + 1h38 = 10h38.
    Reponse: 10h38 (environ).

Probleme 4 (Kahneman): Batte + balle = 1.10. Batte = balle + 1.00.
                      Combien coute la balle ?
  Direct: 0.10 (REPONSE INTUITIVE MAIS FAUSSE)
  CoT:
    1. Soit b = prix de la balle. La batte coute b + 1.00.
    2. Ensemble: b + (b + 1.00) = 1.10.
    3. Donc 2b + 1.00 = 1.10, ce qui donne 2b = 0.10, soit b = 0.05.
    4. La balle coute 0.05 euro (5 centimes).
    5. Verification: batte = 0.05 + 1.00 = 1.05. Total = 0.05 + 1.05 = 1.10. OK.
    Reponse: la balle coute 0.05 euro.

Observation cle: dans P4, le modele 'direct' tombe dans le piege linguistique.
Le CoT force a poser les equations et evite le raccourci intuitif.
""")


# ============================================================================
# Exercice 2 — Self-consistency probabilities
# ============================================================================

print("=" * 70)
print("Exercice 2: Self-consistency — calcul des probabilites")
print("=" * 70)

p = 0.6  # probability of a single sample being correct


def majority_prob(n, p):
    """
    Probability that a majority (>= n/2 + 1) of n independent Bernoulli(p)
    trials are correct.
    """
    total = 0.0
    threshold = n // 2 + 1
    for k in range(threshold, n + 1):
        total += comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return total


print(f"\nHypothese: p(1 sample correct) = {p}")
print("\nP(majorite correcte) pour differents N:")
for n in [1, 3, 5, 7, 11, 21, 51, 101]:
    prob = majority_prob(n, p) if n >= 3 else p
    print(f"  N={n:4d}: {prob:.4f}  ({prob * 100:.2f}%)")

# Find the smallest N such that majority_prob >= 0.95
for n in range(1, 500, 2):
    if majority_prob(n, p) >= 0.95:
        print(f"\nPlus petit N pour >= 95%: {n}")
        break

print("""
Observation: la loi des grands nombres garantit la convergence, mais c'est
LENT. Passer de 60% a 95% demande ~25+ samples.

6) Cout: si 1 sample = 1 cent
  - 5 samples = 5 cents (raisonnable)
  - 20 samples = 20 cents (cher pour une requete standard)
  - 100 samples = 1 USD (seulement pour des taches critiques)

Vaut la peine pour:
  - Math/code ou la bonne reponse est critique
  - Decisions medicales ou legales
  - Benchmarks (on veut la performance max, pas la vitesse)
Pas la peine pour:
  - Chat casual
  - Generation creative (il n'y a pas "une" bonne reponse)
""")


# ============================================================================
# Exercice 3 — When does CoT help?
# ============================================================================

print("=" * 70)
print("Exercice 3: Quand CoT aide-t-il ?")
print("=" * 70)

tasks = [
    ("1. Capitale de la France ?", "-",
     "Fait direct, reponse en memoire"),
    ("2. Somme 1+2+...+100", "+",
     "Calcul multi-etapes, CoT force a poser la formule"),
    ("3. Sentiment 'Ce film etait super ennuyeux'", "-",
     "Classification directe, CoT ajoute du bruit"),
    ("4. 7eme lettre de l'alphabet", "-",
     "Fait direct (g), CoT ne sert a rien (sauf si compter)"),
    ("5. Fonction Python tri", "+/-",
     "Beneficie d'une planification mais pas d'arithmetique"),
    ("6. 3 pommes 1eur + 2 oranges 1.5eur + banane 0.8eur, 10eur - total",
     "+", "Calcul multi-etapes, CoT indispensable"),
    ("7. Haiku sur l'automne", "-",
     "Creation creative, le CoT ajoute du meta-verbiage inutile"),
    ("8. Debuguer code", "+/-",
     "Aide a structurer mais pas critique si le bug est evident"),
    ("9. 5 plus gros pays Europe par population", "-",
     "Liste de faits, reponse directe"),
    ("10. Resoudre 2x + 5 = 15", "+",
     "Calcul, CoT aide a poser soustraction puis division"),
]

print(f"\n{'Tache':<60s} {'CoT':>5s}  {'Pourquoi':<40s}")
print("-" * 110)
for task, cot, why in tasks:
    print(f"{task:<60s} {cot:>5s}  {why}")

print("""
Regle generale:
  CoT AIDE     -> problemes multi-etapes, calcul, logique, planification
  CoT N'AIDE PAS -> faits directs, creativite, classification simple
  CoT PEUT NUIRE -> petits modeles (<60B) qui se perdent dans leur reasoning

Bonus: meme si CoT ne "change pas la reponse", l'ecrire permet
  - d'etre plus verifiable (on voit le raisonnement)
  - d'etre plus calibre (le modele "sait" quand il est incertain)
""")

"""
Calculateur d'interets composes — Module 01 Finance Personnelle
================================================================
Ce script est EDUCATIF uniquement. Il ne constitue pas un conseil
financier personnalise. Les rendements illustratifs ne sont pas garantis.
Les performances passees ne prejudgent pas des performances futures.

Utilisation :
    python 01-interets-composes.py

Contenu :
    1. Calculateur de base (capital initial + versements reguliers)
    2. Effet du temps — tableau de croissance annuelle
    3. Comparaison de scenarios (Alice vs Bob)
    4. Impact de la frequence de capitalisation
    5. Regle des 72 (approximation rapide)

Dependances : stdlib uniquement (math, decimal) — aucune installation requise.
"""

import math


# ---------------------------------------------------------------------------
# Fonctions de calcul
# ---------------------------------------------------------------------------

def capital_final(
    capital_initial: float,
    taux_annuel: float,
    annees: float,
    versement_mensuel: float = 0.0,
    freq_capitalisation: int = 12,
) -> float:
    """
    Calcule le montant final apres capitalisation.

    Formule :
        - Capital seul   : A = P * (1 + r/n)^(n*t)
        - Avec versements : A += M * [((1 + r/n)^(n*t) - 1) / (r/n)]

    Parametres
    ----------
    capital_initial     : montant place au depart (€)
    taux_annuel         : taux annuel en decimal (ex. 0.07 pour 7 %)
    annees              : duree en annees
    versement_mensuel   : versement regulier en fin de periode (€/mois)
    freq_capitalisation : nombre de capitalisations par an (12 = mensuelle)

    Retour
    ------
    Montant final en euros (flottant).

    Note pedagogique : les versements sont traites comme des annuites
    ordinaires (versement en fin de chaque sous-periode).
    """
    r = taux_annuel
    n = freq_capitalisation
    t = annees

    # Facteur de croissance du capital initial
    facteur = (1 + r / n) ** (n * t)
    total = capital_initial * facteur

    # Contribution des versements reguliers (si present)
    if versement_mensuel > 0:
        n_v = 12  # versements mensuels
        if r > 0:
            # Valeur future d'une annuite ordinaire (formule fermee).
            facteur_v = (1 + r / n_v) ** (n_v * t)
            total += versement_mensuel * ((facteur_v - 1) / (r / n_v))
        else:
            # Cas limite r == 0 : pas d'interets, on additionne simplement les
            # versements (sinon la formule ci-dessus divise par zero / serait ignoree).
            total += versement_mensuel * n_v * t

    return total


def tableau_croissance(
    capital_initial: float,
    taux_annuel: float,
    annees_max: int,
    versement_mensuel: float = 0.0,
) -> list[tuple[int, float, float]]:
    """
    Retourne une liste de tuples (annee, montant, gain cumulatif)
    pour visualiser la croissance an par an.
    """
    rows = []
    for a in range(1, annees_max + 1):
        montant = capital_final(capital_initial, taux_annuel, a, versement_mensuel)
        verse = capital_initial + versement_mensuel * 12 * a
        gain = montant - verse
        rows.append((a, montant, gain))
    return rows


def duree_doublement(taux_annuel: float) -> float:
    """
    Regle des 72 : estimation en annees pour doubler le capital.
    72 / taux_en_pourcent
    """
    if taux_annuel <= 0:
        return float("inf")
    return 72 / (taux_annuel * 100)


def duree_doublement_exact(taux_annuel: float, n: int = 1) -> float:
    """
    Calcul exact via ln(2) / (n * ln(1 + r/n)).
    """
    if taux_annuel <= 0:
        return float("inf")
    return math.log(2) / (n * math.log(1 + taux_annuel / n))


# ---------------------------------------------------------------------------
# Fonctions d'affichage
# ---------------------------------------------------------------------------

SEPARATEUR = "-" * 60


def afficher_titre(titre: str) -> None:
    print()
    print("=" * 60)
    print(f"  {titre}")
    print("=" * 60)


def formater_eur(montant: float) -> str:
    """Formate un montant en euros avec separateurs de milliers."""
    return f"{montant:>12,.0f} €".replace(",", " ")


# ---------------------------------------------------------------------------
# Demo principale
# ---------------------------------------------------------------------------

def demo_calcul_de_base() -> None:
    """Scenario de base : 10 000 € + 200 €/mois a 7 % sur 30 ans."""
    afficher_titre("1. Calcul de base")

    capital = 10_000
    versement = 200
    taux = 0.07
    duree = 30

    resultat = capital_final(capital, taux, duree, versement)
    verses_total = capital + versement * 12 * duree

    print(f"  Capital initial      : {formater_eur(capital)}")
    print(f"  Versement mensuel    : {formater_eur(versement)}")
    print(f"  Taux annuel          : {taux * 100:.1f} %")
    print(f"  Duree                : {duree} ans")
    print(SEPARATEUR)
    print(f"  Total verse          : {formater_eur(verses_total)}")
    print(f"  Capital final        : {formater_eur(resultat)}")
    print(f"  Gain (interets)      : {formater_eur(resultat - verses_total)}")
    facteur = resultat / verses_total
    print(f"  Multiplicateur       : x{facteur:.1f} vs sommes versees")


def demo_effet_du_temps() -> None:
    """Montre comment le capital croît differemment selon la duree."""
    afficher_titre("2. Effet du temps (1 000 € initial, 7 %/an)")

    capital = 1_000
    taux = 0.07
    checkpoints = [5, 10, 15, 20, 30, 40]

    print(f"  {'Duree':>8}  {'Capital final':>15}  {'Multiplicateur':>14}")
    print(f"  {'-'*8}  {'-'*15}  {'-'*14}")
    for annees in checkpoints:
        montant = capital_final(capital, taux, annees)
        mult = montant / capital
        print(f"  {annees:>6} ans  {formater_eur(montant)}  {mult:>11.1f} x")


def demo_alice_vs_bob() -> None:
    """
    Scenario classique : Alice commence tot, Bob commence tard.
    Illustre le cout de l'attente.
    """
    afficher_titre("3. Alice vs Bob — Le cout de l'attente")

    versement = 200
    taux = 0.07
    retraite = 65

    # Alice : 25 -> 35 ans (10 ans), puis laisse fructifier jusqu'a 65 ans
    alice_actif = capital_final(0, taux, 10, versement)  # capital a 35 ans
    alice_final = capital_final(alice_actif, taux, 30)   # croissance de 35 a 65 ans
    alice_verse = versement * 12 * 10

    # Bob : 35 -> 65 ans (30 ans)
    bob_final = capital_final(0, taux, 30, versement)
    bob_verse = versement * 12 * 30

    print(f"  Versement mensuel    : {formater_eur(versement)}")
    print(f"  Taux illustratif     : {taux * 100:.0f} % / an")
    print()
    print(f"  {'':40} {'Alice':>12}  {'Bob':>12}")
    print(f"  {'Age de depart':40} {'25 ans':>12}  {'35 ans':>12}")
    print(f"  {'Age d arret':40} {'35 ans':>12}  {'65 ans':>12}")
    print(f"  {'Annees de versement':40} {'10 ans':>12}  {'30 ans':>12}")
    print(f"  {'Total verse':40} {formater_eur(alice_verse)}  {formater_eur(bob_verse)}")
    print(f"  {'Capital a 65 ans':40} {formater_eur(alice_final)}  {formater_eur(bob_final)}")
    print()

    # Cout de l'attente de 10 ans pour quelqu'un qui commence a 35 vs 25 ans
    bob_si_25_ans = capital_final(0, taux, 40, versement)
    gain_10_ans = bob_si_25_ans - bob_final
    print(f"  Cout de l'attente de 10 ans (35 vs 25) : {formater_eur(gain_10_ans)}")
    print()
    print("  Note : taux illustratif, non garanti. Les performances")
    print("  passees ne prejudgent pas des performances futures.")


def demo_frequence_capitalisation() -> None:
    """Compare l'impact de la frequence de capitalisation."""
    afficher_titre("4. Impact de la frequence de capitalisation (10 000 €, 6 %, 10 ans)")

    capital = 10_000
    taux = 0.06
    duree = 10

    frequences = [
        (1, "Annuelle"),
        (4, "Trimestrielle"),
        (12, "Mensuelle"),
        (365, "Journaliere"),
    ]

    print(f"  {'Frequence':20}  {'Capital final':>15}")
    print(f"  {'-'*20}  {'-'*15}")
    for n, label in frequences:
        montant = capital_final(capital, taux, duree, freq_capitalisation=n)
        print(f"  {label:20}  {formater_eur(montant)}")

    print()
    print("  Conclusion : la frequence a un impact reel mais modeste.")
    print("  Priorite = commencer tot et verser regulierement.")


def demo_regle_72() -> None:
    """Illustre la regle des 72 vs calcul exact."""
    afficher_titre("5. Regle des 72 — Duree pour doubler le capital")

    taux_liste = [0.02, 0.04, 0.06, 0.08, 0.10]

    print(f"  {'Taux':>8}  {'Regle des 72':>14}  {'Calcul exact':>14}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}")
    for taux in taux_liste:
        approx = duree_doublement(taux)
        exact = duree_doublement_exact(taux)
        print(f"  {taux*100:>6.0f} %  {approx:>11.1f} ans  {exact:>11.1f} ans")


def demo_comparaison_scenarios() -> None:
    """Compare 3 scenarios d'epargne sur 20 ans."""
    afficher_titre("6. Comparaison de 3 scenarios d'epargne (20 ans, 6 %/an)")

    taux = 0.06
    duree = 20

    scenarios = [
        (5_000, 0, "5 000 € initial, pas de versement"),
        (0, 100, "Pas de capital, 100 €/mois"),
        (2_000, 200, "2 000 € initial + 200 €/mois"),
    ]

    print(f"  {'Scenario':40}  {'Verse':>10}  {'Final':>12}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*12}")
    for capital, versement, label in scenarios:
        montant = capital_final(capital, taux, duree, versement)
        verses = capital + versement * 12 * duree
        print(f"  {label:40}  {formater_eur(verses)}  {formater_eur(montant)}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  CALCULATEUR D'INTERETS COMPOSES")
    print("  Domaine : Finance Personnelle — Module 01")
    print("=" * 60)
    print()
    print("  AVERTISSEMENT : Ce script est EDUCATIF uniquement.")
    print("  Il ne constitue pas un conseil financier personnalise.")
    print("  Les taux utilises sont illustratifs. Les performances")
    print("  passees ne prejudgent pas des performances futures.")

    demo_calcul_de_base()
    demo_effet_du_temps()
    demo_alice_vs_bob()
    demo_frequence_capitalisation()
    demo_regle_72()
    demo_comparaison_scenarios()

    print()
    print("=" * 60)
    print("  Fin du calculateur.")
    print("  Pour modifier les parametres, editez ce script.")
    print("=" * 60)
    print()

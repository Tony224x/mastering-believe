"""
Module 05 — Bases de l'investissement : risque, rendement, diversification
==========================================================================
Ce script illustre, par simulation Monte-Carlo simple (random walk), DEUX idees
du module :
  1. Pourquoi une volatilite plus elevee = des resultats plus disperses
     (meme rendement espere, mais une fourchette de fins de course bien plus large).
  2. Pourquoi diversifier (combiner des actifs peu correles) REDUIT la variance
     du portefeuille a rendement espere comparable — le "repas gratuit" de Markowitz.

Usage : python 05-bases-investissement.py
Stdlib uniquement (`random`, `math`) — aucune dependance externe.

POURQUOI une simulation plutot qu'une formule ? Parce que voir la DISPERSION des
trajectoires (et pas juste une moyenne) rend tangible la notion de risque : deux
placements de meme rendement moyen ne se "vivent" pas du tout pareil.

/!\\ Contenu educatif uniquement. Pas un conseil financier personnalise.
    Rendements hypothetiques — les marches reels ne garantissent aucun resultat.
"""

from __future__ import annotations

import math
import random

# Graine fixe : POURQUOI ? Pour que la sortie soit reproductible d'une execution
# a l'autre (utile en pedagogie : l'apprenant retrouve les memes chiffres).
random.seed(42)


# ---------------------------------------------------------------------------
# 1. Brique de base : simuler UNE trajectoire annuelle d'un placement
# ---------------------------------------------------------------------------

def simuler_trajectoire(
    capital_initial: float,
    rendement_espere: float,
    volatilite: float,
    annees: int,
) -> float:
    """
    Simule la valeur finale d'un placement apres `annees`, annee par annee.

    Chaque annee, le rendement est tire d'une loi normale de moyenne
    `rendement_espere` et d'ecart-type `volatilite`. C'est le coeur du modele :
    la VOLATILITE est precisement l'ecart-type des rendements annuels (cf. theorie).

    POURQUOI multiplier le capital ? Les rendements se composent : une annee a
    -20 % suivie d'une annee a +20 % ne revient PAS au point de depart
    (0,8 * 1,2 = 0,96). C'est l'effet de la composition sur des rendements volatils.
    """
    capital = capital_initial
    for _ in range(annees):
        # random.gauss(mu, sigma) tire un rendement annuel aleatoire.
        rendement_annuel = random.gauss(rendement_espere, volatilite)
        # On borne a -95 % pour eviter un capital negatif aberrant (un actif ne
        # peut pas perdre plus que sa valeur). POURQUOI : garder le modele realiste.
        rendement_annuel = max(rendement_annuel, -0.95)
        capital *= (1 + rendement_annuel)
    return capital


def stats_simulation(
    capital_initial: float,
    rendement_espere: float,
    volatilite: float,
    annees: int,
    n_simulations: int = 5000,
) -> dict:
    """
    Lance `n_simulations` trajectoires et resume la DISPERSION des resultats.

    Renvoie moyenne, mediane, percentiles 10/90 et le pire/meilleur cas.
    POURQUOI des percentiles ? La moyenne seule cache le risque ; ce qui compte
    pour l'epargnant, c'est l'eventail des fins possibles (surtout le bas).
    """
    resultats = [
        simuler_trajectoire(capital_initial, rendement_espere, volatilite, annees)
        for _ in range(n_simulations)
    ]
    resultats.sort()
    n = len(resultats)
    moyenne = sum(resultats) / n
    return {
        "moyenne": moyenne,
        "mediane": resultats[n // 2],
        "p10": resultats[int(0.10 * n)],   # 10 % des cas font moins bien que ca
        "p90": resultats[int(0.90 * n)],   # 10 % des cas font mieux que ca
        "min": resultats[0],
        "max": resultats[-1],
    }


# ---------------------------------------------------------------------------
# 2. Demo 1 — Meme rendement espere, volatilites differentes
# ---------------------------------------------------------------------------

def demo_volatilite(
    capital_initial: float = 10_000,
    rendement_espere: float = 0.06,
    annees: int = 20,
) -> None:
    """
    Compare trois placements de MEME rendement espere mais de volatilites
    croissantes. Objectif pedagogique : montrer que la fourchette des resultats
    (p10 -> p90) s'elargit fortement quand la volatilite monte, alors meme que
    la moyenne reste proche. C'est ca, le "risque".
    """
    print("=" * 68)
    print("DEMO 1 — Meme rendement espere, volatilites differentes")
    print("=" * 68)
    print(f"  Capital initial : {capital_initial:,.0f} €   "
          f"Rendement espere : {rendement_espere*100:.0f} %/an   "
          f"Horizon : {annees} ans")
    print(f"  (Chiffres illustratifs, 5000 simulations par scenario)\n")

    scenarios = [
        (0.03, "Faible volatilite (~obligations)"),
        (0.10, "Volatilite moyenne (~mixte)"),
        (0.20, "Forte volatilite (~actions)"),
    ]

    print(f"  {'Volatilite':<12}{'Mediane':>12}{'Pire 10% (p10)':>16}"
          f"{'Meilleur 10% (p90)':>20}")
    print("  " + "-" * 60)
    for vol, label in scenarios:
        s = stats_simulation(capital_initial, rendement_espere, vol, annees)
        # L'amplitude p90 - p10 mesure la dispersion : plus elle est large,
        # plus le resultat est incertain.
        print(f"  {vol*100:>4.0f} %      "
              f"{s['mediane']:>11,.0f} €"
              f"{s['p10']:>15,.0f} €"
              f"{s['p90']:>19,.0f} €"
              f"   {label}")

    print()
    print("  => Meme rendement espere, mais la fourchette de resultats s'elargit")
    print("     fortement avec la volatilite. Une mediane proche cache un risque")
    print("     de baisse (p10) bien plus profond pour le placement volatil.\n")


# ---------------------------------------------------------------------------
# 3. Demo 2 — Diversification : combiner des actifs peu correles
# ---------------------------------------------------------------------------

def simuler_portefeuille_diversifie(
    capital_initial: float,
    rendement_espere: float,
    volatilite_actif: float,
    annees: int,
    n_actifs: int,
    correlation: float,
) -> float:
    """
    Simule un portefeuille equipondere de `n_actifs`, chacun de meme rendement
    espere et meme volatilite, mais correles entre eux a hauteur de `correlation`.

    MODELE simple a un facteur (POURQUOI ce choix ? il capture l'essentiel sans
    matrice de covariance) :
      rendement_actif = sqrt(rho) * choc_commun + sqrt(1-rho) * choc_propre
    - `choc_commun` : ce qui bouge TOUS les actifs ensemble (= risque de marche,
       non diversifiable).
    - `choc_propre` : le risque SPECIFIQUE a chaque actif (= diversifiable :
       en moyennant beaucoup d'actifs, ces chocs independants s'annulent).
    """
    capital = capital_initial
    racine_rho = math.sqrt(correlation)
    racine_1_moins_rho = math.sqrt(1 - correlation)

    for _ in range(annees):
        # Un seul choc commun par an, partage par tous les actifs.
        choc_commun = random.gauss(0, 1)
        rendements = []
        for _ in range(n_actifs):
            choc_propre = random.gauss(0, 1)
            # On reconstruit un rendement de moyenne `rendement_espere` et
            # d'ecart-type `volatilite_actif`, avec la correlation voulue.
            z = racine_rho * choc_commun + racine_1_moins_rho * choc_propre
            rendements.append(rendement_espere + volatilite_actif * z)
        # Portefeuille equipondere = moyenne des rendements des actifs.
        rendement_portefeuille = sum(rendements) / n_actifs
        rendement_portefeuille = max(rendement_portefeuille, -0.95)
        capital *= (1 + rendement_portefeuille)
    return capital


def demo_diversification(
    capital_initial: float = 10_000,
    rendement_espere: float = 0.07,
    volatilite_actif: float = 0.25,
    annees: int = 20,
    n_simulations: int = 5000,
) -> None:
    """
    Montre l'effet de la diversification : a rendement espere IDENTIQUE, passer
    de 1 actif a 30 actifs peu correles reduit nettement la dispersion (le risque).
    On compare aussi un cas tres correle (rho=0,8) pour montrer la limite :
    quand tout bouge ensemble, diversifier aide beaucoup moins.
    """
    print("=" * 68)
    print("DEMO 2 — La diversification reduit le risque (Markowitz)")
    print("=" * 68)
    print(f"  Chaque actif : rendement espere {rendement_espere*100:.0f} %/an, "
          f"volatilite {volatilite_actif*100:.0f} %/an, horizon {annees} ans")
    print(f"  (Le rendement espere ne change JAMAIS ; seul le risque varie)\n")

    configs = [
        (1, 0.0, "1 seul actif (aucune diversification)"),
        (5, 0.2, "5 actifs, faible correlation"),
        (30, 0.2, "30 actifs, faible correlation"),
        (30, 0.8, "30 actifs, MAIS forte correlation (0,8)"),
    ]

    print(f"  {'Config':<38}{'Mediane':>11}{'p10':>11}{'p90':>11}")
    print("  " + "-" * 64)
    for n_actifs, rho, label in configs:
        resultats = [
            simuler_portefeuille_diversifie(
                capital_initial, rendement_espere, volatilite_actif,
                annees, n_actifs, rho,
            )
            for _ in range(n_simulations)
        ]
        resultats.sort()
        m = len(resultats)
        mediane = resultats[m // 2]
        p10 = resultats[int(0.10 * m)]
        p90 = resultats[int(0.90 * m)]
        print(f"  {label:<38}{mediane:>10,.0f} €{p10:>10,.0f} €{p90:>10,.0f} €")

    print()
    print("  => De 1 a 30 actifs peu correles : la mediane bouge peu, mais la")
    print("     fourchette (p10..p90) se RESSERRE => moins de risque pour un")
    print("     rendement espere comparable. C'est le 'repas gratuit'.")
    print("  => Avec une forte correlation (0,8), la diversification aide bien")
    print("     moins : le risque de marche (choc commun) reste, non diversifiable.\n")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_volatilite()
    demo_diversification()
    print("Rappel : simulation educative. Aucun rendement n'est garanti ;")
    print("tout investissement comporte un risque de perte en capital.")

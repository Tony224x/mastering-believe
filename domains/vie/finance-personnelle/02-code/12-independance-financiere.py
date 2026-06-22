"""
Simulateur d'independance financiere — Module 12 Finance Personnelle
===================================================================
Ce script est EDUCATIF uniquement. Il ne constitue pas un conseil
financier personnalise. Les rendements et hypotheses sont illustratifs
et ne sont pas garantis. Les performances passees ne prejugent pas des
performances futures. Tout investissement comporte un risque de perte
en capital.

Idee centrale
-------------
A revenu donne, c'est le TAUX D'EPARGNE qui decide en combien de temps
on atteint l'independance financiere. Le taux d'epargne agit sur DEUX
leviers a la fois :
  1. il fixe combien on investit chaque mois (accumulation) ;
  2. il fixe combien on depense, donc la TAILLE de la cible a atteindre
     (cible = depenses annuelles / taux de retrait).
C'est pourquoi un taux d'epargne eleve raccourcit l'horizon bien plus
vite qu'une simple regle de trois.

Utilisation :
    python 12-independance-financiere.py

Contenu :
    1. Annees jusqu'a l'independance pour un taux d'epargne donne
    2. Tableau comparatif sur une gamme de taux d'epargne
    3. Montant de retrait annuel/mensuel a partir d'un capital (regle 4 %)
    4. Rappel explicite des LIMITES (sequence des rendements, horizon 30 ans)

Dependances : stdlib uniquement (aucune installation requise).
"""


# ---------------------------------------------------------------------------
# Coeur du calcul
# ---------------------------------------------------------------------------

def annees_jusqu_independance(
    taux_epargne: float,
    rendement_reel: float = 0.05,
    taux_retrait: float = 0.04,
    revenu_net_annuel: float = 36_000.0,
    capital_initial: float = 0.0,
) -> float:
    """
    Estime le nombre d'annees pour atteindre l'independance financiere.

    On raisonne en termes REELS (rendement reel = net d'inflation) parce
    que la cible et le retrait sont penses en pouvoir d'achat constant :
    inutile alors de modeliser separement l'inflation, elle est deja
    "retiree" du rendement (voir Module 01, rendement reel).

    Parametres
    ----------
    taux_epargne     : fraction du revenu net epargnee+investie (0.50 = 50 %)
    rendement_reel   : rendement annuel reel du portefeuille (0.05 = 5 %)
    taux_retrait     : taux de retrait soutenable vise (0.04 = regle des 4 %)
    revenu_net_annuel: revenu net annuel (€). N'influence PAS la duree quand
                       on part de zero : il s'annule (voir note plus bas),
                       mais il sert a chiffrer epargne et cible en euros.
    capital_initial  : patrimoine deja investi au depart (€).

    Retour : nombre d'annees (float). Renvoie 0.0 si la cible est deja
             atteinte, et float('inf') si le taux d'epargne est nul ou
             negatif (on n'investit rien : on n'arrive jamais).

    Pourquoi le revenu s'annule (en partant de zero)
    -------------------------------------------------
    Epargne annuelle = revenu * taux_epargne.
    Depenses annuelles = revenu * (1 - taux_epargne).
    Cible = depenses / taux_retrait = revenu * (1 - taux_epargne) / taux_retrait.
    Dans la formule de duree, le "revenu" apparait au numerateur (cible) ET
    au denominateur (versement annuel) : il se simplifie. C'est la raison
    profonde pour laquelle "le salaire ne decide pas la duree, le taux
    d'epargne si" (cf. exemple Sophie/Theo du cours).
    """
    if taux_epargne <= 0:
        # On n'investit rien : la cible n'est jamais atteinte.
        return float("inf")
    if rendement_reel <= 0:
        # Cas degenere hors scope pedagogique : on evite la division par zero
        # de la formule fermee et on signale l'hypothese irrealiste.
        raise ValueError("rendement_reel doit etre > 0 pour ce modele simplifie")

    depenses_annuelles = revenu_net_annuel * (1.0 - taux_epargne)
    versement_annuel = revenu_net_annuel * taux_epargne
    cible = depenses_annuelles / taux_retrait  # = depenses x (1/taux_retrait)

    if capital_initial >= cible:
        return 0.0

    # Simulation annuelle (plus lisible et plus honnete qu'une formule fermee :
    # on VOIT le capital franchir la cible). Versement en fin d'annee.
    capital = capital_initial
    annees = 0
    # Garde-fou : on plafonne a 200 ans pour ne jamais boucler a l'infini
    # si une combinaison de parametres rend la cible quasi inatteignable.
    while capital < cible and annees < 200:
        capital = capital * (1.0 + rendement_reel) + versement_annuel
        annees += 1

    if annees >= 200 and capital < cible:
        return float("inf")

    # Interpolation lineaire sur la derniere annee pour une estimation plus fine
    # (le capital a saute au-dessus de la cible "entre" deux fins d'annee).
    capital_avant = (capital - versement_annuel) / (1.0 + rendement_reel)
    if capital > capital_avant:
        fraction = (cible - capital_avant) / (capital - capital_avant)
        return (annees - 1) + fraction
    return float(annees)


def montant_retrait(capital: float, taux_retrait: float = 0.04) -> dict:
    """
    Montant qu'on peut retirer selon un taux de retrait soutenable.

    La regle des 4 % : retrait annuel = capital x 0,04, indexe ensuite sur
    l'inflation. Ce script en donne la PREMIERE annee, en euros d'aujourd'hui.

    Retour : dict avec retrait annuel et mensuel.
    """
    annuel = capital * taux_retrait
    return {"annuel": annuel, "mensuel": annuel / 12.0}


def capital_cible(depenses_annuelles: float, taux_retrait: float = 0.04) -> float:
    """
    Capital cible = depenses annuelles / taux de retrait.
    A 4 %, cela revient a "depenses annuelles x 25" (car 1 / 0,04 = 25).
    """
    return depenses_annuelles / taux_retrait


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_un_taux() -> None:
    """Cas unique : un taux d'epargne, en partant de zero."""
    print("=" * 64)
    print("1) Annees jusqu'a l'independance — un taux d'epargne")
    print("=" * 64)
    revenu = 36_000.0           # 3 000 €/mois net (illustratif)
    taux_epargne = 0.40         # 40 %
    rendement = 0.05            # 5 % reel (illustratif, non garanti)
    retrait = 0.04              # regle des 4 %

    depenses = revenu * (1 - taux_epargne)
    cible = capital_cible(depenses, retrait)
    annees = annees_jusqu_independance(taux_epargne, rendement, retrait, revenu)

    print(f"Revenu net annuel        : {revenu:,.0f} €")
    print(f"Taux d'epargne           : {taux_epargne:.0%}")
    print(f"Depenses annuelles       : {depenses:,.0f} €")
    print(f"Rendement reel suppose   : {rendement:.0%}")
    print(f"Taux de retrait vise     : {retrait:.0%}")
    print(f"-> Capital cible         : {cible:,.0f} €  (depenses x {1/retrait:.0f})")
    print(f"-> Annees (depart a zero): ~{annees:.1f} ans")
    print()


def demo_tableau_taux() -> None:
    """Comparatif : l'effet (non lineaire) du taux d'epargne sur l'horizon."""
    print("=" * 64)
    print("2) Effet du taux d'epargne sur l'horizon (rendement reel 5 %)")
    print("=" * 64)
    rendement = 0.05
    retrait = 0.04
    revenu = 36_000.0
    print(f"{'Taux epargne':>13} | {'Annees jusqu independance':>26}")
    print("-" * 44)
    for te in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        annees = annees_jusqu_independance(te, rendement, retrait, revenu)
        print(f"{te:>12.0%} | {annees:>24.1f} ans")
    print()
    print("Lecture : doubler le taux d'epargne fait BIEN PLUS que diviser")
    print("l'horizon par deux (double effet : on accumule plus + cible plus basse).")
    print()


def demo_retrait() -> None:
    """A partir d'un capital, ce que la regle des 4 % autorise a retirer."""
    print("=" * 64)
    print("3) Montant de retrait soutenable (regle des 4 %)")
    print("=" * 64)
    for capital in (250_000.0, 500_000.0, 750_000.0, 1_000_000.0):
        r = montant_retrait(capital, 0.04)
        print(
            f"Capital {capital:>11,.0f} €  ->  "
            f"retrait {r['annuel']:>8,.0f} €/an  "
            f"({r['mensuel']:,.0f} €/mois)"
        )
    print()
    print("Rappel : ces montants sont indexes sur l'inflation chaque annee,")
    print("et exprimes ici en euros d'aujourd'hui (pouvoir d'achat constant).")
    print()


def rappel_limites() -> None:
    """Les limites doivent toujours accompagner le chiffre (cf. cours)."""
    print("=" * 64)
    print("4) LIMITES a garder en tete (ne jamais utiliser le chiffre seul)")
    print("=" * 64)
    print(
        "- Sequence des rendements : a moyenne EGALE, un krach en debut de\n"
        "  retrait est bien plus destructeur qu'en fin de retrait.\n"
        "- Hypothese 30 ans : la regle des 4 % a ete calibree pour ~30 ans.\n"
        "  Pour un horizon de 50 ans (independance precoce), viser plutot\n"
        "  3 % a 3,5 % selon plusieurs chercheurs.\n"
        "- Donnees US uniquement ; fiscalite et frais NON inclus.\n"
        "- Modele simplifie : rendement reel CONSTANT et depenses STABLES,\n"
        "  ce qui n'arrive pas dans la vraie vie. Ceci illustre un PRINCIPE,\n"
        "  pas une projection a l'euro pres."
    )
    print()


def main() -> None:
    print()
    print("SIMULATEUR INDEPENDANCE FINANCIERE — educatif, pas un conseil.")
    print("Hypotheses illustratives, rendements non garantis.\n")
    demo_un_taux()
    demo_tableau_taux()
    demo_retrait()
    rappel_limites()
    print("Modifiez les parametres des fonctions pour tester VOS chiffres.")
    print("Disclaimer : contenu educatif, pas un conseil financier personnalise.")


if __name__ == "__main__":
    main()

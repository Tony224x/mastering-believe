"""
Module 06 — Fonds indiciels et allocation : mise en oeuvre
==========================================================
Ce script chiffre DEUX idees concretes du module :
  1. L'IMPACT COMPOSE DES FRAIS : 0,1 % vs 1 % de frais annuels sur 30 ans.
     (La "tyrannie des couts composes" de Bogle, opposee a la magie des
      rendements composes.)
  2. La REPARTITION d'une allocation "3 fonds" : a partir d'un capital donne,
     combien va dans chaque bloc et quel rendement espere pondere en resulte.

Usage : python 06-fonds-indiciels-allocation.py
Stdlib uniquement — aucune dependance externe.

/!\\ Contenu educatif uniquement. Pas un conseil financier personnalise.
    Aucun produit, emetteur ou ticker n'est recommande. Rendements hypothetiques :
    les marches reels ne garantissent aucun resultat.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Brique de base : capital final avec versements mensuels
# ---------------------------------------------------------------------------

def capital_final_mensuel(
    capital_initial: float,
    versement_mensuel: float,
    taux_annuel_net: float,
    duree_annees: int,
) -> float:
    """
    Capital final avec capitalisation mensuelle et versements reguliers.

    POURQUOI un taux mensuel equivalent et non `taux/12` ? Parce qu'on veut que
    la capitalisation mensuelle composee redonne EXACTEMENT le taux annuel vise :
        (1 + r_mensuel)^12 = 1 + taux_annuel
    C'est plus rigoureux que l'approximation taux/12 utilisee par certaines banques.
    """
    r_mensuel = (1 + taux_annuel_net) ** (1 / 12) - 1
    n_mois = duree_annees * 12

    croissance_capital = capital_initial * (1 + r_mensuel) ** n_mois
    if r_mensuel == 0:
        valeur_versements = versement_mensuel * n_mois
    else:
        valeur_versements = (
            versement_mensuel * ((1 + r_mensuel) ** n_mois - 1) / r_mensuel
        )
    return croissance_capital + valeur_versements


def total_verse(capital_initial: float, versement_mensuel: float, duree_annees: int) -> float:
    """Somme reellement sortie de votre poche (capital initial + versements)."""
    return capital_initial + versement_mensuel * duree_annees * 12


# ---------------------------------------------------------------------------
# 2. Demo 1 — Impact compose des frais : 0,1 % vs 1 % (et plus) sur 30 ans
# ---------------------------------------------------------------------------

def demo_impact_frais(
    capital_initial: float = 10_000,
    versement_mensuel: float = 200,
    taux_brut: float = 0.07,
    duree_annees: int = 30,
) -> None:
    """
    Compare plusieurs niveaux de frais (TER) sur un meme portefeuille.

    POURQUOI soustraire les frais au taux brut ? Les frais courants se prelevent
    chaque annee sur tout le capital : economiquement, ils REDUISENT le rendement
    net annuel. Net = brut - frais. Comme tout se compose, l'ecart explose sur 30 ans.
    """
    print("=" * 66)
    print("DEMO 1 — Impact compose des frais sur 30 ans")
    print("=" * 66)
    verse = total_verse(capital_initial, versement_mensuel, duree_annees)
    print(f"  Capital initial   : {capital_initial:,.0f} €")
    print(f"  Versement mensuel : {versement_mensuel:,.0f} €")
    print(f"  Rendement brut/an : {taux_brut*100:.1f} %   Duree : {duree_annees} ans")
    print(f"  Total sorti de votre poche : {verse:,.0f} €\n")

    # Reference : le plus bas niveau de frais (cas ETF a tres bas cout).
    niveaux = [
        (0.001, "0,1 % — ETF actions monde a bas frais"),
        (0.005, "0,5 % — fonds intermediaire"),
        (0.010, "1,0 % — fonds aux frais eleves"),
        (0.018, "1,8 % — fonds tres charge"),
    ]

    capital_ref = capital_final_mensuel(
        capital_initial, versement_mensuel, taux_brut - niveaux[0][0], duree_annees
    )

    print(f"  {'TER':<7}{'Capital net':>14}{'Cout vs 0,1%':>15}{'Cout %':>9}")
    print("  " + "-" * 56)
    for frais, label in niveaux:
        taux_net = taux_brut - frais
        cap = capital_final_mensuel(capital_initial, versement_mensuel, taux_net, duree_annees)
        cout = capital_ref - cap            # Manque a gagner vs le cas 0,1 %
        cout_pct = cout / capital_ref * 100
        print(f"  {frais*100:>4.1f} %{cap:>13,.0f} €{cout:>14,.0f} €{cout_pct:>8.1f} %"
              f"   {label}")

    # Chiffre cle demande par le module : ecart precis 0,1 % vs 1 %.
    cap_01 = capital_final_mensuel(capital_initial, versement_mensuel, taux_brut - 0.001, duree_annees)
    cap_1 = capital_final_mensuel(capital_initial, versement_mensuel, taux_brut - 0.010, duree_annees)
    print()
    print(f"  => Passer de 0,1 % a 1,0 % de frais coute ~{cap_01 - cap_1:,.0f} € "
          f"sur {duree_annees} ans,")
    print("     soit l'equivalent de plusieurs annees de versements. Les frais")
    print("     sont le levier le plus directement sous votre controle.\n")


# ---------------------------------------------------------------------------
# 3. Demo 2 — Repartition d'une allocation "3 fonds"
# ---------------------------------------------------------------------------

def demo_allocation_3_fonds(
    capital: float = 20_000,
    poids_actions_dom: float = 0.40,   # actions domestiques
    poids_actions_int: float = 0.40,   # actions internationales
    poids_obligations: float = 0.20,   # obligations
    rdt_actions_dom: float = 0.07,     # rendements ESPERES bruts hypothetiques
    rdt_actions_int: float = 0.07,
    rdt_obligations: float = 0.03,
    ter: float = 0.0015,               # frais courants moyens ponderes (0,15 %)
) -> None:
    """
    Repartit un capital donne entre les 3 blocs et calcule le rendement espere
    pondere du portefeuille, net de frais.

    NB pedagogique : les poids ci-dessous sont des PARAMETRES a ajuster par
    l'apprenant selon SON horizon/sa tolerance au risque. Le module ne prescrit
    aucune allocation ; ce sont des chiffres d'illustration.
    """
    print("=" * 66)
    print("DEMO 2 — Repartition d'une allocation '3 fonds'")
    print("=" * 66)

    # Garde-fou : une allocation doit sommer a 100 %.
    total = poids_actions_dom + poids_actions_int + poids_obligations
    assert abs(total - 1.0) < 1e-9, "Les poids doivent sommer a 100 %"

    blocs = [
        ("Actions domestiques (ETF bas frais)", poids_actions_dom, rdt_actions_dom),
        ("Actions internationales (ETF bas frais)", poids_actions_int, rdt_actions_int),
        ("Obligations (ETF bas frais)", poids_obligations, rdt_obligations),
    ]

    print(f"  Capital a repartir : {capital:,.0f} €\n")
    print(f"  {'Bloc':<42}{'Poids':>7}{'Montant':>12}")
    print("  " + "-" * 60)
    rdt_brut_pondere = 0.0
    for label, poids, rdt in blocs:
        montant = capital * poids
        rdt_brut_pondere += poids * rdt   # moyenne ponderee des rendements esperes
        print(f"  {label:<42}{poids*100:>5.0f} %{montant:>11,.0f} €")

    rdt_net = rdt_brut_pondere - ter
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<42}{'100 %':>7}{capital:>11,.0f} €\n")
    print(f"  Rendement espere brut pondere : {rdt_brut_pondere*100:.2f} %/an")
    print(f"  Frais courants moyens (TER)   : {ter*100:.2f} %/an")
    print(f"  Rendement espere net          : {rdt_net*100:.2f} %/an\n")

    print("  => La part en actions tire le rendement espere vers le haut ;")
    print("     les obligations amortissent. Ajustez les poids selon VOTRE")
    print("     horizon et tolerance au risque (cf. Module 05).\n")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_impact_frais()
    demo_allocation_3_fonds()
    print("Rappel : contenu educatif, aucun produit recommande. Rendements")
    print("hypothetiques ; tout investissement comporte un risque de perte en capital.")

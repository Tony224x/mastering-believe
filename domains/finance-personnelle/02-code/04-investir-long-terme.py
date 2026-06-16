"""
Module 04 — Investir simplement et sur le long terme
=====================================================
Ce script démontre deux concepts clés du module :
1. L'impact des frais annuels sur la croissance d'un portefeuille (30 ans)
2. La projection d'une allocation "3 fonds" avec versements mensuels

Usage : python 04-investir-long-terme.py
Stdlib uniquement — pas de dépendance externe.

⚠️  Contenu éducatif uniquement. Pas un conseil financier personnalisé.
    Performances hypothétiques — les marchés réels ne garantissent aucun rendement.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Fonctions utilitaires
# ---------------------------------------------------------------------------

def capital_final_mensuel(
    capital_initial: float,
    versement_mensuel: float,
    taux_annuel_net: float,
    duree_annees: int,
) -> float:
    """
    Calcule le capital final avec capitalisation mensuelle.

    Formule :
      C_final = C0 * (1 + r_m)^n_m
              + V_m * [(1 + r_m)^n_m - 1] / r_m

    Avec :
      r_m = taux mensuel = (1 + taux_annuel)^(1/12) - 1
      n_m = durée en mois
    """
    # Convertir le taux annuel en taux mensuel équivalent
    r_mensuel = (1 + taux_annuel_net) ** (1 / 12) - 1
    n_mois = duree_annees * 12

    # Croissance du capital initial
    croissance_capital = capital_initial * (1 + r_mensuel) ** n_mois

    # Valeur future des versements mensuels réguliers
    if r_mensuel == 0:
        # Cas limite : taux nul
        valeur_versements = versement_mensuel * n_mois
    else:
        valeur_versements = (
            versement_mensuel
            * ((1 + r_mensuel) ** n_mois - 1)
            / r_mensuel
        )

    return croissance_capital + valeur_versements


def total_verse(versement_mensuel: float, duree_annees: int) -> float:
    """Calcule la somme totale versée (hors capital initial)."""
    return versement_mensuel * duree_annees * 12


# ---------------------------------------------------------------------------
# 2. Démonstration 1 — Impact des frais sur 30 ans
# ---------------------------------------------------------------------------

def demo_impact_frais(
    capital_initial: float = 10_000,
    versement_mensuel: float = 200,
    taux_brut: float = 0.07,
    duree_annees: int = 30,
) -> None:
    """
    Compare plusieurs niveaux de frais annuels sur un même portefeuille.
    Illustre pourquoi les frais sont "l'ennemi silencieux" de l'épargnant.
    """
    print("=" * 60)
    print("DÉMO 1 — Impact des frais annuels sur 30 ans")
    print("=" * 60)
    print(f"  Capital initial     : {capital_initial:>10,.0f} €")
    print(f"  Versement mensuel   : {versement_mensuel:>10,.0f} €")
    print(f"  Rendement brut/an   : {taux_brut * 100:>9.2f} %")
    print(f"  Durée               : {duree_annees:>10} ans")
    print(f"  Total versé (hors C0): {total_verse(versement_mensuel, duree_annees):>10,.0f} €")
    print()

    # Niveaux de frais à comparer (TER typiques sur le marché)
    niveaux_frais = [
        (0.0005, "ETF indiciel très bas coût (0,05 %)"),
        (0.0020, "ETF indiciel classique (0,20 %)"),
        (0.0100, "Fonds actif d'entrée de gamme (1,00 %)"),
        (0.0180, "Fonds actif courant (1,80 %)"),
        (0.0250, "Fonds actif coûteux (2,50 %)"),
    ]

    # Référence : premier scénario (frais les plus bas)
    reference_frais = niveaux_frais[0][0]
    capital_reference = capital_final_mensuel(
        capital_initial, versement_mensuel,
        taux_brut - reference_frais, duree_annees
    )

    print(f"{'Frais':<10} {'Capital final':>14} {'Perte vs réf.':>14} {'Perte %':>9}")
    print("-" * 55)

    for frais, label in niveaux_frais:
        taux_net = taux_brut - frais            # Rendement net de frais
        capital = capital_final_mensuel(
            capital_initial, versement_mensuel, taux_net, duree_annees
        )
        perte = capital_reference - capital     # Euros perdus vs référence
        perte_pct = perte / capital_reference * 100

        print(
            f"{frais * 100:>5.2f} %    "
            f"{capital:>14,.0f} €"
            f"{perte:>13,.0f} €"
            f"{perte_pct:>8.1f} %"
            f"   {label}"
        )

    print()
    print(
        "💡 Conclusion : 1,80 point de frais annuels en plus sur 30 ans "
        "peut coûter\n   des dizaines de milliers d'euros — "
        "sans aucun avantage de performance prouvé."
    )
    print()


# ---------------------------------------------------------------------------
# 3. Démonstration 2 — Projection d'une allocation "3 fonds"
# ---------------------------------------------------------------------------

def demo_allocation_3_fonds(
    capital_initial: float = 8_000,
    versement_mensuel: float = 300,
    duree_annees: int = 25,
    allocation_actions_dev: float = 0.60,   # Actions monde développé
    allocation_actions_em: float = 0.20,    # Actions marchés émergents
    allocation_obligations: float = 0.20,   # Obligations
    ter_moyen: float = 0.0015,              # TER moyen pondéré (0,15 %)
    taux_actions_dev: float = 0.07,         # Rendement brut hypothétique actions dev
    taux_actions_em: float = 0.08,          # Rendement brut hypothétique EM
    taux_obligations: float = 0.03,         # Rendement brut hypothétique obligations
) -> None:
    """
    Projette la croissance d'un portefeuille "3 fonds" avec allocation pondérée.
    Calcule le rendement pondéré global, puis la projection totale.
    """
    print("=" * 60)
    print("DÉMO 2 — Projection allocation '3 fonds'")
    print("=" * 60)

    # Vérifier que l'allocation somme à 100 %
    total_alloc = allocation_actions_dev + allocation_actions_em + allocation_obligations
    assert abs(total_alloc - 1.0) < 1e-6, "L'allocation doit sommer à 100 %"

    # Rendement pondéré brut de l'allocation
    taux_brut_pondere = (
        allocation_actions_dev * taux_actions_dev
        + allocation_actions_em * taux_actions_em
        + allocation_obligations * taux_obligations
    )
    # Rendement net (après frais moyens pondérés)
    taux_net = taux_brut_pondere - ter_moyen

    print(f"  Capital initial     : {capital_initial:>10,.0f} €")
    print(f"  Versement mensuel   : {versement_mensuel:>10,.0f} €")
    print(f"  Durée               : {duree_annees:>10} ans")
    print()
    print("  Allocation cible :")
    print(f"    Actions monde développé : {allocation_actions_dev * 100:.0f} %  "
          f"(rendement brut hypothétique : {taux_actions_dev * 100:.1f} %)")
    print(f"    Actions marchés émerg.  : {allocation_actions_em * 100:.0f} %  "
          f"(rendement brut hypothétique : {taux_actions_em * 100:.1f} %)")
    print(f"    Obligations             : {allocation_obligations * 100:.0f} %  "
          f"(rendement brut hypothétique : {taux_obligations * 100:.1f} %)")
    print()
    print(f"  Rendement brut pondéré  : {taux_brut_pondere * 100:.2f} %/an")
    print(f"  TER moyen pondéré       : {ter_moyen * 100:.2f} %/an")
    print(f"  Rendement net estimé    : {taux_net * 100:.2f} %/an")
    print()

    # Projection jalonnée
    print(f"  {'Horizon':<10} {'Capital estimé':>16} {'Total versé':>14} {'Gains':>14}")
    print("  " + "-" * 58)

    jalons = [5, 10, 15, 20, duree_annees]
    for n in jalons:
        capital = capital_final_mensuel(capital_initial, versement_mensuel, taux_net, n)
        verse = capital_initial + total_verse(versement_mensuel, n)
        gains = capital - verse
        print(
            f"  {n} ans       "
            f"{capital:>15,.0f} €"
            f"{verse:>13,.0f} €"
            f"{gains:>13,.0f} €"
        )

    print()
    capital_final = capital_final_mensuel(
        capital_initial, versement_mensuel, taux_net, duree_annees
    )
    verse_total = capital_initial + total_verse(versement_mensuel, duree_annees)
    print(
        f"  ➜ Sur {duree_annees} ans, {verse_total:,.0f} € versés "
        f"deviennent {capital_final:,.0f} € "
        f"(×{capital_final / verse_total:.1f})."
    )
    print()
    print(
        "⚠️  Ces chiffres sont illustratifs. Les rendements réels varient "
        "et peuvent être\n   négatifs certaines années. Les performances passées "
        "ne préjugent pas du futur."
    )
    print()


# ---------------------------------------------------------------------------
# 4. Démonstration 3 — Comparaison "maintenant" vs "attendre 2 ans"
# ---------------------------------------------------------------------------

def demo_cout_attente(
    versement_mensuel: float = 300,
    taux_net: float = 0.05,
    duree_totale_annees: int = 25,
) -> None:
    """
    Montre le coût d'attendre 2 ans avant d'investir.
    Illustre le principe 'le temps dans le marché bat le timing du marché'.
    """
    print("=" * 60)
    print("DÉMO 3 — Coût d'attendre 2 ans avant d'investir")
    print("=" * 60)

    # Investir maintenant (25 ans)
    capital_maintenant = capital_final_mensuel(0, versement_mensuel, taux_net, duree_totale_annees)

    # Attendre 2 ans, puis investir (23 ans seulement)
    capital_apres_attente = capital_final_mensuel(0, versement_mensuel, taux_net, duree_totale_annees - 2)

    print(f"  Versement mensuel : {versement_mensuel:.0f} €")
    print(f"  Rendement net     : {taux_net * 100:.1f} %/an")
    print()
    print(f"  Investir maintenant ({duree_totale_annees} ans)  : {capital_maintenant:>12,.0f} €")
    print(f"  Attendre 2 ans ({duree_totale_annees - 2} ans)      : {capital_apres_attente:>12,.0f} €")
    print(f"  Coût de l'attente                   : {capital_maintenant - capital_apres_attente:>12,.0f} €")
    print()
    print(
        "💡 2 ans d'attente peuvent coûter plusieurs milliers d'euros "
        "en gains composés perdus."
    )
    print()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_impact_frais(
        capital_initial=10_000,
        versement_mensuel=200,
        taux_brut=0.07,
        duree_annees=30,
    )

    demo_allocation_3_fonds(
        capital_initial=8_000,
        versement_mensuel=300,
        duree_annees=25,
        allocation_actions_dev=0.60,
        allocation_actions_em=0.20,
        allocation_obligations=0.20,
        ter_moyen=0.0015,
    )

    demo_cout_attente(
        versement_mensuel=300,
        taux_net=0.05,
        duree_totale_annees=25,
    )

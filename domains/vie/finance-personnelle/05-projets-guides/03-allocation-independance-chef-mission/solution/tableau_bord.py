"""
Tableau de bord patrimonial — Projet guide 03 (Finance Personnelle)
===================================================================
Contexte : Karim est chef de mission chez LogiSim. Il pilote des deploiements
FleetSim chez les clients, gagne correctement sa vie, et veut enfin structurer
son patrimoine sur le long terme : quelle allocation, quel cout des frais, a
quel horizon il pourrait viser l'independance financiere, et quel risque le
guette au moment de vivre de son capital.

Ce script est EDUCATIF uniquement. Il ne constitue pas un conseil financier,
fiscal ou en investissement personnalise. Les rendements sont illustratifs et
non garantis. Tout investissement comporte un risque de perte en capital. La
"regle des 4 %" est une heuristique americaine historique, pas une garantie.

Modules couverts : 04 (investir long terme), 05 (frais), 06 (independance
financiere), 01 (interets composes).

Utilisation :
    python tableau_bord.py

Dependances : stdlib uniquement (dataclasses, statistics). Aucune installation.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Allocation par horizon
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """Repartition actions / obligations et rendement attendu illustratif."""

    horizon_ans: int
    pct_actions: int

    @property
    def pct_obligations(self) -> int:
        return 100 - self.pct_actions

    def rendement_attendu(self, r_actions: float = 0.07, r_obligations: float = 0.03) -> float:
        """Rendement moyen pondere (illustratif, brut de frais et d'inflation)."""
        return (self.pct_actions / 100) * r_actions + (self.pct_obligations / 100) * r_obligations


def allocation_par_horizon(horizon_ans: int) -> Allocation:
    """Regle d'horizon simple : plus l'echeance est lointaine, plus on peut
    supporter de volatilite (donc d'actions). Heuristique pedagogique, PAS
    une prescription : 100 % - 4 points d'obligations par annee qui manque
    sous 20 ans, bornee a [30 %, 90 %] d'actions.

    Le vrai determinant n'est pas l'age mais l'HORIZON et la tolerance au
    risque (capacite a ne pas vendre en pleine baisse — cf. module 05).
    """
    if horizon_ans >= 20:
        actions = 90
    else:
        actions = max(30, 90 - (20 - horizon_ans) * 4)
    return Allocation(horizon_ans=horizon_ans, pct_actions=actions)


def demo_allocation() -> None:
    titre("1. Allocation suggeree selon l'horizon")
    print(f"  {'Horizon':>10} {'Actions':>9} {'Obligations':>12} {'Rdt attendu*':>13}")
    print(f"  {'-'*10} {'-'*9} {'-'*12} {'-'*13}")
    for h in (3, 8, 15, 25, 35):
        a = allocation_par_horizon(h)
        print(f"  {h:>7} ans {a.pct_actions:>7} % {a.pct_obligations:>10} % "
              f"{a.rendement_attendu()*100:>10.1f} %")
    print()
    print("  * Rendement attendu illustratif, brut de frais et d'inflation,")
    print("    non garanti. Une allocation 90/10 peut perdre 30-40 % une annee.")
    print("  Le but de l'horizon : avoir le temps de digerer ces baisses.")


# ---------------------------------------------------------------------------
# 2. Actif vs passif, net de frais (cadrage SPIVA)
# ---------------------------------------------------------------------------

def capital_apres_frais(
    versement_annuel: float, rendement_brut: float, frais: float, annees: int
) -> float:
    """Capital final d'un versement annuel, NET de frais.

    Le rendement net = brut - frais s'applique chaque annee a l'encours.
    """
    r_net = rendement_brut - frais
    capital = 0.0
    for _ in range(annees):
        capital = capital * (1 + r_net) + versement_annuel
    return capital


def demo_actif_vs_passif() -> None:
    titre("2. Gestion active vs fonds indiciel passif (net de frais)")
    versement = 6_000   # 500 EUR/mois
    annees = 30
    rendement_brut = 0.07

    # Hypothese honnete fondee sur les rapports SPIVA : sur le long terme, la
    # majorite des fonds actifs ne battent PAS leur indice AVANT frais, et
    # encore moins APRES frais. On modelise donc le cas median realiste :
    # meme rendement brut, mais l'actif coute beaucoup plus cher.
    passif = capital_apres_frais(versement, rendement_brut, 0.002, annees)   # ETF 0,2 %
    actif = capital_apres_frais(versement, rendement_brut, 0.018, annees)    # fonds actif 1,8 %

    ecart = passif - actif
    pct = ecart / passif * 100
    print(f"  Versement : {euros(versement)}/an pendant {annees} ans, brut {rendement_brut*100:.0f} %")
    print()
    print(f"  Fonds indiciel passif (0,2 % frais) : {euros(passif)}")
    print(f"  Fonds gere activement (1,8 % frais) : {euros(actif)}")
    print(f"  Manque a gagner du a 1,6 pt de frais : {euros(ecart)} ({pct:.0f} %)")
    print()
    print("  Cadrage SPIVA : sur 15-20 ans, ~85-95 % des fonds actifs sous-")
    print("  performent leur indice net de frais. Payer plus cher pour 'battre")
    print("  le marche' est, en moyenne et statistiquement, un pari perdant.")


# ---------------------------------------------------------------------------
# 3. Independance financiere (FI) — nombre cible et horizon
# ---------------------------------------------------------------------------

def numero_fi(depenses_annuelles: float, taux_retrait: float = 0.04) -> float:
    """Capital cible = depenses annuelles / taux de retrait soutenable.

    A 4 %, c'est 25x les depenses annuelles. La 'regle des 4 %' vient de
    l'etude Trinity (marche US, horizon 30 ans) : heuristique, pas une loi.
    Un taux plus prudent (3,5 %) -> ~28,6x ; plus agressif (5 %) -> 20x.
    """
    return depenses_annuelles / taux_retrait


def annees_jusqu_a_fi(
    capital_actuel: float,
    epargne_annuelle: float,
    cible: float,
    rendement_net: float = 0.05,
) -> int | None:
    """Nombre d'annees pour atteindre la cible FI, par simulation annuelle.

    Retourne None si la cible n'est pas atteinte en 100 ans (garde-fou).
    """
    capital = capital_actuel
    for an in range(1, 101):
        capital = capital * (1 + rendement_net) + epargne_annuelle
        if capital >= cible:
            return an
    return None


def demo_fi() -> None:
    titre("3. Independance financiere — la cible de Karim")
    depenses = 30_000   # depenses annuelles visees
    capital_actuel = 40_000
    epargne_annuelle = 18_000  # taux d'epargne eleve mais atteignable

    print(f"  Depenses annuelles visees : {euros(depenses)}")
    print(f"  Capital actuel            : {euros(capital_actuel)}")
    print(f"  Epargne annuelle          : {euros(epargne_annuelle)}")
    print()
    print(f"  {'Taux de retrait':>16} {'Capital cible':>16} {'Annees pour y arriver':>24}")
    print(f"  {'-'*16} {'-'*16} {'-'*24}")
    for swr in (0.05, 0.04, 0.035):
        cible = numero_fi(depenses, swr)
        ans = annees_jusqu_a_fi(capital_actuel, epargne_annuelle, cible)
        ans_txt = f"{ans} ans" if ans is not None else "> 100 ans"
        mult = cible / depenses
        print(f"  {swr*100:>14.1f} % {euros(cible):>16} {ans_txt:>20} (x{mult:.0f})")
    print()
    print("  Lecon : le taux de retrait choisi change radicalement la cible.")
    print("  Plus prudent = plus gros capital = plus d'annees. Le taux d'EPARGNE")
    print("  (pas le revenu) est le principal accelerateur du delai (module 06).")


# ---------------------------------------------------------------------------
# 4. Risque de sequence des rendements (le piege du debut de retraite)
# ---------------------------------------------------------------------------

def simuler_retraite(capital_initial: float, retrait_annuel: float, rendements: list[float]) -> float:
    """Simule un retrait annuel en debut d'annee, puis applique le rendement.

    Retourne le capital restant a la fin de la sequence (0 si epuise).
    Modelise le 'risque de sequence' : l'ordre des rendements compte enormement
    quand on RETIRE de l'argent, alors qu'il est neutre quand on accumule.
    """
    capital = capital_initial
    for r in rendements:
        capital -= retrait_annuel       # on vit d'abord (retrait en debut d'annee)
        if capital <= 0:
            return 0.0
        capital *= 1 + r                # puis le marche fait son oeuvre
    return capital


def demo_sequence() -> None:
    titre("4. Risque de sequence — le meme rendement moyen, deux destins")
    capital = 750_000
    retrait = 30_000  # 4 % de 750 000

    # Deux sequences avec EXACTEMENT les memes rendements, dans l'ordre inverse.
    # 'krach_a_la_fin' place les deux mauvaises annees (-10 %, -20 %) tout a la
    # fin ; en l'inversant, le krach tombe au TOUT DEBUT de la retraite. Memes
    # rendements, meme moyenne -> deux destins tres differents.
    krach_a_la_fin = [0.10, 0.12, 0.08, 0.10, 0.15, 0.06, 0.10, 0.05, -0.10, -0.20]
    krach_au_debut = list(reversed(krach_a_la_fin))

    moy = statistics.mean(krach_a_la_fin) * 100
    fin_krach_fin = simuler_retraite(capital, retrait, krach_a_la_fin)
    fin_krach_debut = simuler_retraite(capital, retrait, krach_au_debut)

    print(f"  Capital de depart : {euros(capital)}, retrait {euros(retrait)}/an")
    print(f"  Memes 10 rendements (moyenne {moy:.1f} %), ordre different :")
    print()
    print(f"    Krach en FIN de retraite   -> capital final : {euros(fin_krach_fin)}")
    print(f"    Krach en DEBUT de retraite -> capital final : {euros(fin_krach_debut)}")
    print()
    ecart = fin_krach_fin - fin_krach_debut
    print(f"  Ecart : {euros(ecart)} pour une moyenne de rendement IDENTIQUE.")
    print("  C'est le risque de sequence : subir une grosse baisse juste apres")
    print("  avoir commence a retirer epuise le capital bien plus vite. Parades :")
    print("  matelas de cash 1-2 ans, taux de retrait flexible, ne pas etre 100 %")
    print("  actions a la veille de vivre de son capital (module 04 + 06).")


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def euros(montant: float) -> str:
    return f"{montant:,.0f} EUR".replace(",", " ")


def titre(t: str) -> None:
    print()
    print("=" * 64)
    print(f"  {t}")
    print("=" * 64)


def main() -> None:
    print()
    print("=" * 64)
    print("  TABLEAU DE BORD PATRIMONIAL — UN CHEF DE MISSION LOGISIM")
    print("  Finance Personnelle / Projet guide 03")
    print("=" * 64)
    print()
    print("  AVERTISSEMENT : script EDUCATIF. Pas un conseil financier ni")
    print("  fiscal. Rendements illustratifs, non garantis. Risque de perte.")

    demo_allocation()
    demo_actif_vs_passif()
    demo_fi()
    demo_sequence()

    print()
    print("=" * 64)
    print("  Fin. Remplace les chiffres par les tiens pour ton propre tableau.")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()

"""
Simulateur d'epargne d'equipe — Projet guide 01 (Finance Personnelle)
=====================================================================
Contexte : quatre operateurs de l'OCC (Operations Control Center) d'un
client FleetSim comparent leurs strategies d'epargne. Meme salaire, memes
annees devant eux, mais des comportements differents : qui commence tot,
qui paie des frais eleves, qui profite de l'abondement employeur.

Ce script est EDUCATIF uniquement. Il ne constitue pas un conseil financier
personnalise. Les taux sont illustratifs. Les performances passees ne
prejugent pas des performances futures. Tout investissement comporte un
risque de perte en capital.

Modules couverts : 01 (interets composes), 02 (epargne automatique),
05 (impact des frais).

Utilisation :
    python epargne_simulator.py

Dependances : stdlib uniquement (dataclasses, typing). Aucune installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Modele de donnees
# ---------------------------------------------------------------------------

@dataclass
class Operateur:
    """Un operateur OCC et sa strategie d'epargne.

    Tous les montants sont en euros, les taux en decimal (0.06 = 6 %).
    """

    nom: str
    versement_mensuel: float          # ce que la personne met de sa poche
    annees_versement: int             # combien d'annees elle verse
    annees_totales: int               # horizon total (versement + croissance passive)
    annee_debut: int = 0              # delai avant de commencer a verser (en annees)
    abondement_employeur: float = 0.0  # % du versement ajoute par l'employeur (0.5 = +50 %)
    frais_annuels: float = 0.0        # frais de gestion annuels (0.02 = 2 %)
    rendement_brut: float = 0.06      # rendement annuel AVANT frais

    # Champs calcules (remplis par simuler)
    capital_final: float = field(default=0.0, init=False)
    total_verse_perso: float = field(default=0.0, init=False)
    total_abonde: float = field(default=0.0, init=False)


# ---------------------------------------------------------------------------
# Moteur de simulation
# ---------------------------------------------------------------------------

def simuler(op: Operateur) -> Operateur:
    """Simule mois par mois le capital d'un operateur.

    Pourquoi une boucle mensuelle plutot que la formule fermee d'annuite ?
    Parce qu'on veut modeliser proprement deux choses que la formule
    standard gere mal ensemble : (1) l'arret des versements apres N annees
    suivi d'une phase de croissance passive, et (2) les frais de gestion
    preleves sur l'encours (pas sur les versements). La boucle rend ces
    deux mecanismes explicites et faciles a auditer.
    """
    # Le rendement NET de frais est ce qui fait reellement croitre l'encours.
    # C'est le coeur du module 05 : 2 % de frais sur un brut de 6 % ampute
    # un tiers de la performance, et l'effet se compose sur des decennies.
    rendement_net = op.rendement_brut - op.frais_annuels
    taux_mensuel = rendement_net / 12.0

    capital = 0.0
    total_perso = 0.0
    total_abonde = 0.0

    debut = op.annee_debut * 12
    fin_versement = (op.annee_debut + op.annees_versement) * 12

    for mois in range(op.annees_totales * 12):
        # 1) Croissance de l'encours existant (interets composes mensuels).
        capital *= 1.0 + taux_mensuel

        # 2) Versement du mois — uniquement pendant la fenetre de versement
        #    [annee_debut, annee_debut + annees_versement[.
        if debut <= mois < fin_versement:
            verse = op.versement_mensuel
            abonde = verse * op.abondement_employeur
            capital += verse + abonde
            total_perso += verse
            total_abonde += abonde

    op.capital_final = capital
    op.total_verse_perso = total_perso
    op.total_abonde = total_abonde
    return op


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def euros(montant: float) -> str:
    """Format humain : '123 456 EUR' (espace insecable fine evitee pour la console)."""
    return f"{montant:>12,.0f} EUR".replace(",", " ")


def titre(t: str) -> None:
    print()
    print("=" * 64)
    print(f"  {t}")
    print("=" * 64)


def afficher_classement(operateurs: list[Operateur]) -> None:
    """Tableau de classement par capital final decroissant."""
    titre("Classement final (capital a horizon)")
    print(f"  {'Operateur':22} {'Capital final':>16} {'Verse perso':>14}")
    print(f"  {'-'*22} {'-'*16} {'-'*14}")
    for op in sorted(operateurs, key=lambda o: o.capital_final, reverse=True):
        print(f"  {op.nom:22} {euros(op.capital_final)} {euros(op.total_verse_perso)}")


def afficher_detail(op: Operateur) -> None:
    """Detail pedagogique d'un operateur : ce qu'il a mis vs ce qu'il obtient."""
    verse_total = op.total_verse_perso + op.total_abonde
    gain = op.capital_final - verse_total
    mult = op.capital_final / op.total_verse_perso if op.total_verse_perso else 0.0
    print()
    print(f"  --- {op.nom} ---")
    print(f"    Versement mensuel perso : {euros(op.versement_mensuel)}")
    print(f"    Annees de versement     : {op.annees_versement} (horizon {op.annees_totales} ans)")
    print(f"    Rendement brut / frais  : {op.rendement_brut*100:.1f} % / {op.frais_annuels*100:.1f} %")
    print(f"    Abondement employeur    : +{op.abondement_employeur*100:.0f} % des versements")
    print(f"    Total verse (perso)     : {euros(op.total_verse_perso)}")
    if op.total_abonde:
        print(f"    Total abonde (employeur): {euros(op.total_abonde)}")
    print(f"    Capital final           : {euros(op.capital_final)}")
    print(f"    Gain (interets)         : {euros(gain)}")
    print(f"    Multiplicateur          : x{mult:.1f} vs ce que j'ai sorti de ma poche")


# ---------------------------------------------------------------------------
# Scenarios — les 4 operateurs OCC
# ---------------------------------------------------------------------------

def construire_operateurs() -> list[Operateur]:
    """Quatre profils, meme horizon de 35 ans, meme effort de base.

    Principe de conception : AMINA est la baseline. Chaque autre operateur
    ne change qu'UNE seule variable par rapport a elle, pour que l'apprenant
    attribue chaque ecart a sa vraie cause (et pas a un melange de facteurs).

        - Amina  : baseline (verse 10 ans, des le depart, frais bas, sans abondement)
        - Bruno  : SEULE difference = commence 10 ans plus tard      -> effet TEMPS
        - Carla  : SEULE difference = frais 2 % au lieu de 0,4 %      -> effet FRAIS
        - Diallo : SEULE difference = abondement employeur +50 %       -> effet ABONDEMENT

    Tous versent 200 EUR/mois pendant 10 ans (24 000 EUR de leur poche).
    """
    base = dict(versement_mensuel=200, annees_versement=10, annees_totales=35,
                frais_annuels=0.004, rendement_brut=0.06)
    return [
        Operateur(nom="Amina (baseline)", **base),
        # Bruno : identique a Amina, mais commence 10 ans plus tard.
        Operateur(nom="Bruno (debut +10 ans)", **{**base, "annee_debut": 10}),
        # Carla : identique a Amina, mais frais de gestion eleves.
        Operateur(nom="Carla (frais 2 %)", **{**base, "frais_annuels": 0.02}),
        # Diallo : identique a Amina, mais l'employeur abonde +50 %.
        Operateur(nom="Diallo (abondement +50%)", **{**base, "abondement_employeur": 0.5}),
    ]


def comparaisons_cles(ops: dict[str, Operateur]) -> None:
    """Met en evidence les trois lecons chiffrees, chacune isolee vs la baseline."""
    titre("Lecons chiffrees (chacune isole UNE variable vs Amina)")

    amina = ops["Amina (baseline)"]
    bruno = ops["Bruno (debut +10 ans)"]
    carla = ops["Carla (frais 2 %)"]
    diallo = ops["Diallo (abondement +50%)"]

    # Lecon 1 : le temps. Amina et Bruno versent le MEME montant total (24 000),
    # Bruno commence juste 10 ans plus tard. Tout le reste est identique.
    cout_retard = amina.capital_final - bruno.capital_final
    pct1 = cout_retard / amina.capital_final * 100
    print()
    print("  1) Le temps est le levier n.1 :")
    print(f"     Amina et Bruno versent exactement la meme somme ({euros(amina.total_verse_perso)}).")
    print(f"     Bruno commence 10 ans plus tard et finit avec {euros(cout_retard)} de moins")
    print(f"     ({pct1:.0f} % du capital d'Amina). Le retard ne se rattrape pas.")

    # Lecon 2 : les frais. Amina vs Carla, seule difference = les frais.
    cout_frais = amina.capital_final - carla.capital_final
    pct2 = cout_frais / amina.capital_final * 100
    print()
    print("  2) Les frais sont un impot silencieux :")
    print("     Carla = Amina, mais 2 % de frais au lieu de 0,4 %. Resultat :")
    print(f"     {euros(cout_frais)} envoles, soit {pct2:.0f} % du capital d'Amina.")

    # Lecon 3 : l'abondement. Amina vs Diallo.
    bonus = diallo.capital_final - amina.capital_final
    print()
    print("  3) L'abondement employeur, c'est du rendement immediat :")
    print("     Diallo = Amina, mais l'employeur abonde +50 %. Il finit avec")
    print(f"     {euros(bonus)} de plus, pour le meme effort d'epargne personnel.")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 64)
    print("  SIMULATEUR D'EPARGNE — EQUIPE OCC (FleetSim)")
    print("  Finance Personnelle / Projet guide 01")
    print("=" * 64)
    print()
    print("  AVERTISSEMENT : script EDUCATIF. Pas un conseil financier.")
    print("  Taux illustratifs, non garantis. Risque de perte en capital.")

    operateurs = [simuler(op) for op in construire_operateurs()]
    par_nom = {op.nom: op for op in operateurs}

    afficher_classement(operateurs)

    titre("Detail par operateur")
    for op in operateurs:
        afficher_detail(op)

    comparaisons_cles(par_nom)

    print()
    print("=" * 64)
    print("  Fin. Modifie construire_operateurs() pour tester tes hypotheses.")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()

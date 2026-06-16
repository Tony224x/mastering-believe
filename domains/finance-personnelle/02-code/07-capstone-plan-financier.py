"""
Module 07 — Capstone : Mini-simulateur de plan financier personnel
==================================================================
Ce simulateur interactif (mode CLI) construit un plan financier complet :
  1. Budget → taux d'épargne
  2. Fonds d'urgence → délai pour l'atteindre
  3. Allocation d'investissement simulée (3 fonds)
  4. Projection composée sur 20 et 30 ans avec jalons

Usage : python 07-capstone-plan-financier.py
Stdlib uniquement — pas de dépendance externe.

⚠️  Contenu éducatif uniquement. Pas un conseil financier personnalisé.
    Les projections reposent sur des hypothèses qui ne garantissent pas
    les résultats futurs. Tout investissement comporte un risque de perte.
"""

from __future__ import annotations
import sys


# ---------------------------------------------------------------------------
# 1. Utilitaires de calcul financier
# ---------------------------------------------------------------------------

def capital_final_mensuel(
    capital_initial: float,
    versement_mensuel: float,
    taux_annuel_net: float,
    duree_annees: int,
) -> float:
    """
    Calcule le capital final avec capitalisation mensuelle.
    Formule : C0*(1+r_m)^n_m + V_m*[(1+r_m)^n_m - 1]/r_m
    Avec r_m = taux mensuel équivalent.
    """
    r_m = (1 + taux_annuel_net) ** (1 / 12) - 1
    n_m = duree_annees * 12

    croissance_capital = capital_initial * (1 + r_m) ** n_m

    if r_m == 0:
        valeur_versements = versement_mensuel * n_m
    else:
        valeur_versements = versement_mensuel * ((1 + r_m) ** n_m - 1) / r_m

    return croissance_capital + valeur_versements


def mois_pour_atteindre(
    objectif: float,
    capital_actuel: float,
    epargne_mensuelle: float,
) -> int:
    """
    Calcule le nombre de mois pour atteindre un objectif d'épargne,
    sans rendement (fonds d'urgence = compte sécurisé).
    """
    if capital_actuel >= objectif:
        return 0
    if epargne_mensuelle <= 0:
        return -1  # Impossible
    return int((objectif - capital_actuel) / epargne_mensuelle) + 1


def rendement_pondere(
    pct_actions_dev: float,
    pct_actions_em: float,
    pct_obligations: float,
    taux_actions_dev: float = 0.07,
    taux_actions_em: float = 0.08,
    taux_obligations: float = 0.03,
    ter: float = 0.002,
) -> float:
    """
    Calcule le rendement annuel net pondéré d'une allocation 3 fonds.
    Les taux bruts hypothétiques sont des valeurs indicatives — pas des garanties.
    """
    brut = (
        pct_actions_dev * taux_actions_dev
        + pct_actions_em * taux_actions_em
        + pct_obligations * taux_obligations
    )
    return brut - ter


# ---------------------------------------------------------------------------
# 2. Collecte des données utilisateur (mode non-interactif possible)
# ---------------------------------------------------------------------------

def demander_float(question: str, defaut: float) -> float:
    """
    Demande un nombre flottant à l'utilisateur.
    En mode non-interactif (ex. tests), retourne directement la valeur par défaut.
    """
    if not sys.stdin.isatty():
        # Entrée non-interactive : utiliser la valeur par défaut
        print(f"{question} [par défaut : {defaut}] → {defaut} (mode non-interactif)")
        return defaut
    try:
        reponse = input(f"{question} [défaut : {defaut}] : ").strip()
        return float(reponse) if reponse else defaut
    except ValueError:
        print(f"  ↳ Valeur invalide, utilisation de {defaut}.")
        return defaut


def demander_int(question: str, defaut: int) -> int:
    """Demande un entier à l'utilisateur."""
    if not sys.stdin.isatty():
        print(f"{question} [par défaut : {defaut}] → {defaut} (mode non-interactif)")
        return defaut
    try:
        reponse = input(f"{question} [défaut : {defaut}] : ").strip()
        return int(reponse) if reponse else defaut
    except ValueError:
        print(f"  ↳ Valeur invalide, utilisation de {defaut}.")
        return defaut


# ---------------------------------------------------------------------------
# 3. Sections du plan
# ---------------------------------------------------------------------------

def section_budget(revenus_nets: float, depenses_essentielles: float,
                   depenses_non_essentielles: float, remboursements_dettes: float
                   ) -> dict:
    """
    Calcule le budget et le taux d'épargne.
    Retourne un dict avec les métriques clés.
    """
    total_depenses = depenses_essentielles + depenses_non_essentielles + remboursements_dettes
    epargne_mensuelle = revenus_nets - total_depenses
    taux_epargne = epargne_mensuelle / revenus_nets * 100 if revenus_nets > 0 else 0.0

    return {
        "revenus_nets": revenus_nets,
        "depenses_essentielles": depenses_essentielles,
        "depenses_non_essentielles": depenses_non_essentielles,
        "remboursements_dettes": remboursements_dettes,
        "total_depenses": total_depenses,
        "epargne_mensuelle": epargne_mensuelle,
        "taux_epargne": taux_epargne,
    }


def section_fonds_urgence(depenses_mensuelles_essentielles: float,
                          fonds_urgence_actuel: float,
                          mois_objectif: int = 3) -> dict:
    """
    Calcule l'objectif de fonds d'urgence et le délai pour l'atteindre.
    """
    objectif = depenses_mensuelles_essentielles * mois_objectif
    ecart = max(0.0, objectif - fonds_urgence_actuel)
    return {
        "objectif": objectif,
        "actuel": fonds_urgence_actuel,
        "ecart": ecart,
        "mois_objectif": mois_objectif,
        "complete": ecart <= 0,
    }


def section_projection(
    capital_initial: float,
    versement_mensuel: float,
    taux_net: float,
    depenses_annuelles: float,
    jalons_ans: list[int] | None = None,
) -> list[dict]:
    """
    Calcule la projection du capital à plusieurs jalons.
    Retourne une liste de dict par jalon.
    """
    if jalons_ans is None:
        jalons_ans = [5, 10, 15, 20, 25, 30]

    resultats = []
    for n in jalons_ans:
        capital = capital_final_mensuel(capital_initial, versement_mensuel, taux_net, n)
        verse = capital_initial + versement_mensuel * 12 * n
        gains = capital - verse
        multiple_depenses = capital / depenses_annuelles if depenses_annuelles > 0 else 0
        resultats.append({
            "annees": n,
            "capital": capital,
            "total_verse": verse,
            "gains_composes": gains,
            "multiple_depenses": multiple_depenses,
        })
    return resultats


# ---------------------------------------------------------------------------
# 4. Affichage du plan complet
# ---------------------------------------------------------------------------

def afficher_plan(budget: dict, fonds: dict, allocation: dict,
                  projection: list[dict], depenses_annuelles: float) -> None:
    """Affiche le plan financier synthétique dans la console."""

    separateur = "=" * 65

    print()
    print(separateur)
    print("         PLAN FINANCIER PERSONNEL — Synthèse")
    print(separateur)
    print("  ⚠️  Document éducatif — pas un conseil financier personnalisé.")
    print()

    # BLOC 1 — Budget
    print("── BLOC 1 : BUDGET ──────────────────────────────────────────")
    print(f"  Revenus nets             : {budget['revenus_nets']:>10,.0f} €/mois")
    print(f"  Dépenses essentielles    : {budget['depenses_essentielles']:>10,.0f} €/mois")
    print(f"  Dépenses non essentielles: {budget['depenses_non_essentielles']:>10,.0f} €/mois")
    print(f"  Remboursements de dettes : {budget['remboursements_dettes']:>10,.0f} €/mois")
    print(f"  ────────────────────────────────────────────────────────────")
    print(f"  Épargne mensuelle        : {budget['epargne_mensuelle']:>10,.0f} €/mois")
    print(f"  Taux d'épargne           : {budget['taux_epargne']:>9.1f} %")

    # Évaluation qualitative du taux d'épargne
    if budget['taux_epargne'] < 10:
        note = "Faible — chercher à augmenter (objectif : 15-20 % minimum)"
    elif budget['taux_epargne'] < 20:
        note = "Moyen — bien, viser 20 %+ pour accélérer"
    elif budget['taux_epargne'] < 40:
        note = "Bon — solide, continuez"
    else:
        note = "Excellent — trajectoire d'indépendance financière rapide"
    print(f"  Évaluation               : {note}")
    print()

    # BLOC 2 — Fonds d'urgence
    print("── BLOC 2 : FONDS D'URGENCE ─────────────────────────────────")
    print(f"  Objectif ({fonds['mois_objectif']} mois)         : {fonds['objectif']:>10,.0f} €")
    print(f"  Fonds actuel             : {fonds['actuel']:>10,.0f} €")
    if fonds['complete']:
        print(f"  Statut                   : ✓ Complet !")
    else:
        print(f"  Manque                   : {fonds['ecart']:>10,.0f} €")
        # Estimation du délai si on y consacre 1/4 de l'épargne
        epargne_dediee = max(1.0, budget['epargne_mensuelle'] * 0.5)
        mois = mois_pour_atteindre(fonds['objectif'], fonds['actuel'], epargne_dediee)
        if mois > 0:
            print(f"  Délai estimé (50 % épargne): {mois} mois à {epargne_dediee:.0f} €/mois")
        else:
            print("  Délai estimé             : impossible avec épargne actuelle — augmenter les revenus ou réduire les dépenses")
    print()

    # BLOC 3 — Allocation
    print("── BLOC 3 : ALLOCATION D'INVESTISSEMENT ─────────────────────")
    print(f"  Versement investi/mois   : {allocation['versement_mensuel']:>10,.0f} €")
    print(f"  Actions monde développé  : {allocation['pct_actions_dev'] * 100:>9.0f} %")
    print(f"  Actions marchés émergents: {allocation['pct_actions_em'] * 100:>9.0f} %")
    print(f"  Obligations              : {allocation['pct_obligations'] * 100:>9.0f} %")
    print(f"  TER moyen cible          : {allocation['ter'] * 100:>9.2f} %/an")
    print(f"  Rendement net estimé     : {allocation['taux_net'] * 100:>9.2f} %/an  (hypothèse)")
    print()

    # BLOC 4 — Projection
    print("── BLOC 4 : PROJECTION (hypothèse, pas une garantie) ────────")
    print(f"  {'Horizon':<10} {'Capital':>13} {'Total versé':>13} {'Gains':>12} {'× dép. ann.':>12}")
    print(f"  {'-'*10} {'-'*13} {'-'*13} {'-'*12} {'-'*12}")

    for p in projection:
        # Tag seulement quand le seuil est REELLEMENT atteint (>= 10× / >= 25×),
        # pas dans une bande symetrique qui labelliserait une ligne sous le seuil.
        jalon_fi_partielle = "← 10× dépenses (IF partielle)" if 0 <= p['multiple_depenses'] - 10 < 1.5 else ""
        jalon_fi_totale = "← 25× dépenses (IF totale)" if 0 <= p['multiple_depenses'] - 25 < 2.5 else ""
        jalon = jalon_fi_partielle or jalon_fi_totale
        print(
            f"  {p['annees']:<3} ans    "
            f"{p['capital']:>12,.0f} €"
            f"{p['total_verse']:>12,.0f} €"
            f"{p['gains_composes']:>11,.0f} €"
            f"{p['multiple_depenses']:>11.1f} ×"
            f"  {jalon}"
        )
    print()

    # Jalons d'indépendance financière
    print("── JALONS D'INDÉPENDANCE FINANCIÈRE ─────────────────────────")
    capital_fi_partielle = depenses_annuelles * 10
    capital_fi_totale = depenses_annuelles * 25
    print(f"  IF partielle (10×) cible : {capital_fi_partielle:>10,.0f} €")
    print(f"  IF totale (25×) cible    : {capital_fi_totale:>10,.0f} €")

    # Recherche de l'horizon pour chaque jalon
    for label, cible in [("IF partielle (10×)", capital_fi_partielle),
                          ("IF totale (25×)", capital_fi_totale)]:
        # Cherche l'horizon en incréments d'1 an
        for n in range(1, 61):
            c = capital_final_mensuel(
                allocation['capital_initial'], allocation['versement_mensuel'],
                allocation['taux_net'], n
            )
            if c >= cible:
                print(f"  → {label} : atteinte en ~{n} ans")
                break
        else:
            print(f"  → {label} : non atteinte en 60 ans avec ces paramètres")

    print()
    print(separateur)
    print("  Prochaine révision annuelle : dans 12 mois")
    print("  Rééquilibrage : 1 fois/an si écart > 5 points vs cible")
    print()
    print("  ⚠️  Les rendements hypothétiques ne garantissent pas les résultats futurs.")
    print("      Consultez un conseiller agréé pour toute décision réelle.")
    print(separateur)
    print()


# ---------------------------------------------------------------------------
# 5. Point d'entrée — mode interactif ou démo avec valeurs par défaut
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Collecte les informations et génère le plan financier.
    En mode non-interactif (stdin non-tty), utilise des valeurs par défaut
    correspondant au profil "Alex" du capstone.
    """
    print()
    print("=" * 65)
    print("   SIMULATEUR DE PLAN FINANCIER PERSONNEL")
    print("   Module 07 — Domaine Finance Personnelle")
    print("=" * 65)
    print("  Appuyez sur Entrée pour utiliser les valeurs par défaut.")
    print("  Les valeurs par défaut correspondent au profil 'Alex' (capstone).")
    print()

    # ── Bloc Budget ──
    print("[ BUDGET ]")
    revenus = demander_float("Revenus nets mensuels (€)", 3200.0)
    dep_ess = demander_float("Dépenses essentielles mensuelles (€)", 1370.0)
    dep_non_ess = demander_float("Dépenses non essentielles mensuelles (€)", 320.0)
    rembt_dettes = demander_float("Remboursements de dettes mensuels (€)", 230.0)

    budget = section_budget(revenus, dep_ess, dep_non_ess, rembt_dettes)
    print(f"  → Épargne disponible : {budget['epargne_mensuelle']:.0f} €/mois "
          f"(taux d'épargne : {budget['taux_epargne']:.1f} %)")
    print()

    # ── Bloc Fonds d'urgence ──
    print("[ FONDS D'URGENCE ]")
    fonds_actuel = demander_float("Fonds d'urgence actuel (€)", 1200.0)
    mois_obj = demander_int("Objectif : combien de mois de dépenses essentielles ?", 3)

    fonds = section_fonds_urgence(dep_ess, fonds_actuel, mois_obj)
    print(f"  → Objectif : {fonds['objectif']:.0f} € | "
          f"Manque : {fonds['ecart']:.0f} €")
    print()

    # ── Bloc Allocation ──
    print("[ ALLOCATION D'INVESTISSEMENT ]")
    print("  (Saisir les pourcentages sous forme décimale, ex. 60 pour 60 %)")
    versement_invest = demander_float("Versement mensuel à investir (€)", 280.0)
    capital_init = demander_float("Capital déjà investi (€)", 0.0)
    pct_dev = demander_float("% Actions monde développé (ex. 60)", 60.0) / 100
    pct_em = demander_float("% Actions marchés émergents (ex. 20)", 20.0) / 100
    pct_oblig = demander_float("% Obligations (ex. 20)", 20.0) / 100
    ter = demander_float("TER moyen cible (%, ex. 0.20)", 0.20) / 100

    # Normaliser si nécessaire
    total_pct = pct_dev + pct_em + pct_oblig
    if abs(total_pct - 1.0) > 0.01:
        print(f"  ⚠️  Les proportions ({total_pct * 100:.0f} %) ne somment pas à 100 %. Normalisation automatique.")
        pct_dev /= total_pct
        pct_em /= total_pct
        pct_oblig /= total_pct

    taux_net = rendement_pondere(pct_dev, pct_em, pct_oblig, ter=ter)
    allocation = {
        "capital_initial": capital_init,
        "versement_mensuel": versement_invest,
        "pct_actions_dev": pct_dev,
        "pct_actions_em": pct_em,
        "pct_obligations": pct_oblig,
        "ter": ter,
        "taux_net": taux_net,
    }
    print(f"  → Rendement net estimé : {taux_net * 100:.2f} %/an (hypothèse)")
    print()

    # ── Bloc Projection ──
    print("[ PROJECTION ]")
    depenses_annuelles = (dep_ess + dep_non_ess) * 12  # Hors remboursement dette
    print(f"  Dépenses annuelles (hors dette) : {depenses_annuelles:,.0f} €")

    duree_max = demander_int("Durée de projection maximale (années, ex. 30)", 30)
    jalons = [j for j in [5, 10, 15, 20, 25, 30] if j <= duree_max]
    if duree_max not in jalons:
        jalons.append(duree_max)
    jalons.sort()

    projection = section_projection(
        capital_init, versement_invest, taux_net, depenses_annuelles, jalons
    )

    # ── Affichage du plan complet ──
    afficher_plan(budget, fonds, allocation, projection, depenses_annuelles)


if __name__ == "__main__":
    main()

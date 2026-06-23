"""
Module 14 — Capstone : gabarit parametrable de plan financier
=============================================================
Ce programme est un GABARIT : vous entrez VOS chiffres dans le bloc
"=== VOS CHIFFRES ===" ci-dessous, vous relancez, et le programme affiche
les CONSEQUENCES de vos hypotheses. Il ne recommande RIEN.

Blocs couverts :
  1. Budget -> taux d'epargne
  2. Fonds d'urgence cible (en mois de depenses)
  3. Plan de dette (priorise les taux eleves)
  4. Allocation simulee (VOS proportions -> rendement net hypothetique)
  5. Projection composee 20-30 ans : NOMINAL vs REEL (ajuste de l'inflation)
  6. Cible FIRE optionnelle (regle des 4 %) avec ses limites

Usage : python 14-capstone-plan-financier.py
Stdlib uniquement — pas de dependance externe.

/!\  Contenu EDUCATIF. Gabarit aux hypotheses ajustables, PAS une
     recommandation d'investissement, fiscale ou juridique. Le programme
     n'indique aucune allocation : il calcule a partir de vos entrees.
     Tout investissement comporte un risque de perte en capital.
"""

from __future__ import annotations


# ===========================================================================
# === VOS CHIFFRES =========================================================
# Modifiez ces variables avec VOS donnees, puis relancez le programme.
# (Le profil ci-dessous est un exemple FICTIF de demonstration.)
# ===========================================================================

# --- Bloc 1 : budget mensuel (en euros) ---
REVENUS_NETS_MENSUELS          = 3200.0
DEPENSES_ESSENTIELLES          = 1370.0   # logement, alimentation, transport, sante
DEPENSES_NON_ESSENTIELLES      = 320.0    # loisirs, envies
REMBOURSEMENTS_DETTES_MENSUELS = 230.0    # total des mensualites de dette

# --- Bloc 2 : fonds d'urgence ---
FONDS_URGENCE_ACTUEL           = 1200.0
FONDS_URGENCE_CIBLE_MOIS       = 4        # cible en MOIS de depenses essentielles

# --- Bloc 3 : dettes en cours ---
# Liste de (libelle, capital_restant_du, taux_annuel_en_decimal)
DETTES = [
    ("Credit conso",     4200.0, 0.14),   # 14 %/an -> taux eleve
    ("Pret etudiant",    8000.0, 0.02),   # 2 %/an  -> taux faible
]
SEUIL_TAUX_ELEVE = 0.07   # au-dessus, on considere la dette "a traiter en priorite"

# --- Bloc 4 : allocation d'investissement (VOS proportions, somme ~1.0) ---
CAPITAL_DEJA_INVESTI           = 0.0
VERSEMENT_MENSUEL_INVESTI      = 280.0
PROPORTIONS = {                            # vos % (en decimal), a vous de choisir
    "Poche A": 0.60,
    "Poche B": 0.20,
    "Poche C": 0.20,
}
# Hypotheses de rendement BRUT par poche (en decimal) — ajustables, NON garanties.
RENDEMENTS_BRUTS_HYP = {
    "Poche A": 0.07,
    "Poche B": 0.08,
    "Poche C": 0.03,
}
TER_ANNUEL                     = 0.0020    # frais annuels (0.20 %) — Module 06

# --- Bloc 5 : projection ---
HORIZONS_ANS                   = [10, 20, 30]   # jalons a projeter
INFLATION_ANNUELLE_HYP         = 0.02           # pour le calcul du REEL (Module 01)

# --- Bloc 6 : cible FIRE (optionnelle) ---
ACTIVER_CIBLE_FIRE             = True
TAUX_RETRAIT_FIRE              = 0.04            # regle des 4 % -> cible = depenses/0.04

# ===========================================================================
# === FIN DE VOS CHIFFRES — le reste est de la logique de calcul ============
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Utilitaires de calcul
# ---------------------------------------------------------------------------

def capital_final_mensuel(capital_initial: float, versement_mensuel: float,
                          taux_annuel: float, duree_annees: int) -> float:
    """Capital final avec capitalisation mensuelle (versements en fin de mois)."""
    r_m = (1 + taux_annuel) ** (1 / 12) - 1
    n_m = duree_annees * 12
    croissance = capital_initial * (1 + r_m) ** n_m
    if r_m == 0:
        versements = versement_mensuel * n_m
    else:
        versements = versement_mensuel * ((1 + r_m) ** n_m - 1) / r_m
    return croissance + versements


def taux_reel(nominal: float, inflation: float) -> float:
    """Taux reel exact : (1+nominal)/(1+inflation) - 1."""
    return (1 + nominal) / (1 + inflation) - 1


def pouvoir_achat_reel(montant_nominal: float, inflation: float,
                       annees: int) -> float:
    """Convertit un montant futur en euros d'aujourd'hui (pouvoir d'achat)."""
    return montant_nominal / ((1 + inflation) ** annees)


def rendement_net_pondere(proportions: dict, rendements_bruts: dict,
                          ter: float) -> float:
    """Rendement net hypothetique = somme(part * rendement brut) - TER."""
    brut = sum(proportions[k] * rendements_bruts[k] for k in proportions)
    return brut - ter


# ---------------------------------------------------------------------------
# 2. Affichage des blocs
# ---------------------------------------------------------------------------

SEP = "=" * 68


def imprimer_entete() -> None:
    print()
    print(SEP)
    print("   GABARIT DE PLAN FINANCIER — Module 14 (capstone)")
    print(SEP)
    print("  /!\\  PROJECTION EDUCATIVE — hypotheses ajustables.")
    print("       Ce gabarit calcule les consequences de VOS chiffres.")
    print("       Il ne recommande AUCUNE allocation et n'est PAS un conseil.")
    print(SEP)
    print()


def bloc1_budget() -> dict:
    total_dep = (DEPENSES_ESSENTIELLES + DEPENSES_NON_ESSENTIELLES
                 + REMBOURSEMENTS_DETTES_MENSUELS)
    epargne = REVENUS_NETS_MENSUELS - total_dep
    taux = (epargne / REVENUS_NETS_MENSUELS * 100
            if REVENUS_NETS_MENSUELS > 0 else 0.0)

    print("-- BLOC 1 : BUDGET ET TAUX D'EPARGNE --------------------------")
    print(f"  Revenus nets              : {REVENUS_NETS_MENSUELS:>10,.0f} EUR/mois")
    print(f"  Depenses essentielles     : {DEPENSES_ESSENTIELLES:>10,.0f} EUR/mois")
    print(f"  Depenses non essentielles : {DEPENSES_NON_ESSENTIELLES:>10,.0f} EUR/mois")
    print(f"  Remboursements de dettes  : {REMBOURSEMENTS_DETTES_MENSUELS:>10,.0f} EUR/mois")
    print(f"  Epargne mensuelle         : {epargne:>10,.0f} EUR/mois")
    print(f"  Taux d'epargne            : {taux:>9.1f} %")
    print("  (Chiffre brut, a vous de l'interpreter — pas de note imposee.)")
    print()
    return {"epargne_mensuelle": epargne, "taux_epargne": taux}


def bloc2_fonds_urgence() -> None:
    cible = DEPENSES_ESSENTIELLES * FONDS_URGENCE_CIBLE_MOIS
    ecart = max(0.0, cible - FONDS_URGENCE_ACTUEL)
    print("-- BLOC 2 : FONDS D'URGENCE -----------------------------------")
    print(f"  Cible ({FONDS_URGENCE_CIBLE_MOIS} mois de dep. ess.) : {cible:>10,.0f} EUR")
    print(f"  Reserve actuelle          : {FONDS_URGENCE_ACTUEL:>10,.0f} EUR")
    if ecart <= 0:
        print("  Statut                    : cible atteinte")
    else:
        print(f"  Manque                    : {ecart:>10,.0f} EUR")
    print("  Rappel : cible exprimee EN MOIS (suit l'inflation), pas en")
    print("           montant fige. A constituer AVANT d'investir.")
    print()


def bloc3_dettes() -> None:
    print("-- BLOC 3 : PLAN DE DETTE -------------------------------------")
    if not DETTES:
        print("  Aucune dette saisie.")
        print()
        return
    print(f"  {'Libelle':<18}{'Capital':>12}{'Taux':>9}   Priorite")
    print(f"  {'-'*18}{'-'*12}{'-'*9}   {'-'*8}")
    for libelle, capital, taux in sorted(DETTES, key=lambda d: -d[2]):
        prio = "ELEVEE" if taux >= SEUIL_TAUX_ELEVE else "faible"
        print(f"  {libelle:<18}{capital:>11,.0f}E{taux*100:>7.1f} %   {prio}")
    print(f"  Seuil 'taux eleve' retenu : {SEUIL_TAUX_ELEVE*100:.0f} %")
    print("  Methode : traiter d'abord les taux eleves (avalanche, Module 04).")
    print("  Arbitrage rembourser/investir : comparez le taux de la dette a")
    print("  votre hypothese de rendement net. Le gabarit ne tranche pas.")
    print()


def bloc4_allocation() -> dict:
    total_pct = sum(PROPORTIONS.values())
    proportions = dict(PROPORTIONS)
    print("-- BLOC 4 : ALLOCATION SIMULEE (VOS proportions) --------------")
    if abs(total_pct - 1.0) > 0.01:
        print(f"  /!\\ Vos proportions somment a {total_pct*100:.0f} % "
              f"-> normalisation auto a 100 %.")
        proportions = {k: v / total_pct for k, v in proportions.items()}
    for k in proportions:
        print(f"  {k:<22}: {proportions[k]*100:>6.1f} %   "
              f"(rdt brut hyp. {RENDEMENTS_BRUTS_HYP[k]*100:.1f} %)")
    taux_net = rendement_net_pondere(proportions, RENDEMENTS_BRUTS_HYP, TER_ANNUEL)
    print(f"  TER annuel                : {TER_ANNUEL*100:>6.2f} %")
    print(f"  Rendement NET hypothetique: {taux_net*100:>6.2f} %/an  (hypothese)")
    print("  Ces % sont VOS entrees, pas une suggestion du programme.")
    print()
    return {"taux_net": taux_net}


def bloc5_projection(versement: float, taux_net: float) -> None:
    print("-- BLOC 5 : PROJECTION NOMINAL vs REEL ------------------------")
    t_reel = taux_reel(taux_net, INFLATION_ANNUELLE_HYP)
    print(f"  Hypotheses : rendement net {taux_net*100:.2f} %/an, "
          f"inflation {INFLATION_ANNUELLE_HYP*100:.1f} %/an")
    print(f"  -> rendement REEL ~= {t_reel*100:.2f} %/an "
          f"(nominal - inflation, Module 01)")
    print()
    print(f"  {'Horizon':<9}{'Capital NOMINAL':>18}{'Capital REEL':>18}")
    print(f"  {'-'*9}{'-'*18}{'-'*18}")
    for n in sorted(HORIZONS_ANS):
        nominal = capital_final_mensuel(CAPITAL_DEJA_INVESTI, versement, taux_net, n)
        reel = pouvoir_achat_reel(nominal, INFLATION_ANNUELLE_HYP, n)
        print(f"  {n:>3} ans  {nominal:>16,.0f}E {reel:>16,.0f}E")
    print("  Le REEL = pouvoir d'achat en euros d'aujourd'hui. Sur 30 ans,")
    print("  l'ecart est enorme : raisonner en nominal SURESTIME le resultat.")
    print()


def bloc6_fire(versement: float, taux_net: float) -> None:
    if not ACTIVER_CIBLE_FIRE:
        return
    depenses_annuelles = (DEPENSES_ESSENTIELLES + DEPENSES_NON_ESSENTIELLES) * 12
    cible = depenses_annuelles / TAUX_RETRAIT_FIRE
    print("-- BLOC 6 : CIBLE FIRE (optionnelle) --------------------------")
    print(f"  Depenses annuelles (hors dette) : {depenses_annuelles:>10,.0f} EUR")
    print(f"  Cible (regle des {TAUX_RETRAIT_FIRE*100:.0f} %)            : "
          f"{cible:>10,.0f} EUR  (= depenses / taux)")
    horizon = None
    for n in range(1, 61):
        if capital_final_mensuel(CAPITAL_DEJA_INVESTI, versement, taux_net, n) >= cible:
            horizon = n
            break
    if horizon:
        print(f"  Horizon estime (hypotheses ci-dessus) : ~{horizon} ans")
    else:
        print("  Horizon estime : non atteinte en 60 ans avec ces parametres")
    print("  LIMITES : la regle des 4 % suppose un horizon ~30 ans et un")
    print("  retrait indexe sur l'inflation ; elle NE couvre PAS le risque de")
    print("  sequence de rendements. C'est un repere, pas une promesse (Module 12).")
    print()


def imprimer_pied() -> None:
    print(SEP)
    print("  Revision : relancer ce gabarit 1 fois/an (ou apres un evenement")
    print("  majeur), mettre a jour vos chiffres et vos hypotheses.")
    print()
    print("  /!\\  PROJECTION EDUCATIVE. Hypotheses ajustables, NON garanties.")
    print("       Ceci N'EST PAS une recommandation d'allocation ni un conseil")
    print("       financier. Tout investissement comporte un risque de perte.")
    print("       Consultez un professionnel agree pour toute decision reelle.")
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# 3. Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    imprimer_entete()
    budget = bloc1_budget()
    bloc2_fonds_urgence()
    bloc3_dettes()
    alloc = bloc4_allocation()
    bloc5_projection(VERSEMENT_MENSUEL_INVESTI, alloc["taux_net"])
    bloc6_fire(VERSEMENT_MENSUEL_INVESTI, alloc["taux_net"])
    imprimer_pied()


if __name__ == "__main__":
    main()

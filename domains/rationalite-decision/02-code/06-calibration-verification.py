"""
Module 06 -- Calibration : score de Brier et courbe de calibration.

Outil du domaine `rationalite-decision`. Permet de SCORER un journal de
predictions : etes-vous bien calibre (quand vous dites 70%, est-ce vrai ~70%
du temps) ? Stdlib pur, aucune dependance.

Le score de Brier = moyenne de (proba - resultat)^2, ou resultat vaut 1 (l'evenement
s'est produit) ou 0 (non). Plus c'est BAS, mieux c'est. Reperes :
  - 0.0  = parfait
  - 0.25 = ce qu'on obtient en disant toujours "50%"
  - 1.0  = parfaitement, et confiantement, faux

Exemples 100% neutres (meteo, sport, predictions perso) -- aucune prise de position.

Run :
    python domains/rationalite-decision/02-code/06-calibration-verification.py
"""

from __future__ import annotations

from dataclasses import dataclass


# ===========================================================================
# 1. SCORE DE BRIER
# ===========================================================================

@dataclass
class Prediction:
    """Une prediction : un enonce, la proba annoncee, et le resultat observe."""
    enonce: str
    proba: float        # proba annoncee dans [0, 1]
    resultat: int       # 1 si l'evenement s'est produit, 0 sinon


def brier_score(predictions: list[Prediction]) -> float:
    """Score de Brier = moyenne de (proba - resultat)^2. Plus bas = mieux."""
    if not predictions:
        return 0.0
    # (p - o)^2 penalise d'autant plus qu'on etait confiant ET faux.
    total = sum((p.proba - p.resultat) ** 2 for p in predictions)
    return total / len(predictions)


def brier_reference_naive(predictions: list[Prediction]) -> float:
    """Score d'un predicteur qui dirait toujours 50% : sert de point de repere.

    Si votre score de Brier est >= a celui-ci, vous n'apportez aucune
    information par rapport au pur "je ne sais pas (50/50)".
    """
    if not predictions:
        return 0.0
    return sum((0.5 - p.resultat) ** 2 for p in predictions) / len(predictions)


# ===========================================================================
# 2. COURBE DE CALIBRATION (par tranches de confiance)
# ===========================================================================

def calibration_table(predictions: list[Prediction], n_bins: int = 5) -> list[dict]:
    """Regroupe les predictions par tranche de proba et compare proba moyenne
    annoncee vs frequence reelle observee. Un calibre parfait a annonce ~= reel.
    """
    # Bornes des tranches : [0,0.2), [0.2,0.4), ... [0.8,1.0].
    bins: list[list[Prediction]] = [[] for _ in range(n_bins)]
    for p in predictions:
        # min(...) pour que proba == 1.0 tombe dans la derniere tranche, pas hors borne.
        idx = min(int(p.proba * n_bins), n_bins - 1)
        bins[idx].append(p)

    rows: list[dict] = []
    for i, group in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not group:
            rows.append({"tranche": f"[{lo:.0%}-{hi:.0%})", "n": 0,
                         "proba_moy": None, "freq_reelle": None, "ecart": None})
            continue
        proba_moy = sum(p.proba for p in group) / len(group)
        freq_reelle = sum(p.resultat for p in group) / len(group)
        rows.append({
            "tranche": f"[{lo:.0%}-{hi:.0%})", "n": len(group),
            "proba_moy": proba_moy, "freq_reelle": freq_reelle,
            "ecart": freq_reelle - proba_moy,
        })
    return rows


def print_calibration(predictions: list[Prediction], n_bins: int = 5) -> None:
    """Affiche la table de calibration + une barre ASCII annonce vs reel."""
    print(f"{'Tranche':<12}{'n':>4}{'Annonce':>10}{'Reel':>9}{'Ecart':>9}   calibration")
    print("-" * 70)
    for r in calibration_table(predictions, n_bins):
        if r["n"] == 0:
            print(f"{r['tranche']:<12}{0:>4}{'-':>10}{'-':>9}{'-':>9}")
            continue
        # Une fleche montre si on est sur-confiant (reel < annonce) ou sous-confiant.
        signe = "ok" if abs(r["ecart"]) < 0.1 else ("sur-confiant" if r["ecart"] < 0 else "sous-confiant")
        print(f"{r['tranche']:<12}{r['n']:>4}{r['proba_moy']:>10.0%}"
              f"{r['freq_reelle']:>9.0%}{r['ecart']:>+9.0%}   {signe}")


# ===========================================================================
# 3. DEMO
# ===========================================================================

# Journal de predictions jouet (exemples neutres : meteo, sport, vie quotidienne).
DEMO: list[Prediction] = [
    Prediction("Il pleuvra demain", 0.80, 1),
    Prediction("L'equipe A gagne", 0.60, 0),
    Prediction("Le colis arrive aujourd'hui", 0.90, 1),
    Prediction("Le bus sera en retard", 0.30, 0),
    Prediction("Il fera > 25 degres", 0.70, 1),
    Prediction("Le film depasse 2h", 0.50, 0),
    Prediction("La reunion finit a l'heure", 0.40, 0),
    Prediction("Le magasin est ouvert dimanche", 0.85, 1),
    Prediction("Mon equipe gagne le quiz", 0.65, 1),
    Prediction("Il y aura une greve", 0.20, 0),
]


if __name__ == "__main__":
    print("=" * 70)
    print("Module 06 -- Calibration : score de Brier")
    print("=" * 70)

    bs = brier_score(DEMO)
    ref = brier_reference_naive(DEMO)
    print(f"\nPredictions evaluees : {len(DEMO)}")
    print(f"Score de Brier       : {bs:.3f}  (plus bas = mieux ; 0 = parfait)")
    print(f"Repere 'toujours 50%' : {ref:.3f}")
    verdict = "vous battez le 50/50 -> vos probas apportent de l'info" if bs < ref \
        else "vous ne battez PAS le 50/50 -> calibration a revoir"
    print(f"Verdict              : {verdict}")

    print("\n" + "=" * 70)
    print("Courbe de calibration (annonce vs reel par tranche)")
    print("=" * 70)
    print_calibration(DEMO)

    # Illustration : deux profils extremes pour montrer la metrique.
    print("\n" + "=" * 70)
    print("Reperes : predicteur parfait vs predicteur sur-confiant")
    print("=" * 70)
    parfait = [Prediction("e", 1.0, 1), Prediction("e", 0.0, 0), Prediction("e", 1.0, 1)]
    surconfiant = [Prediction("e", 1.0, 0), Prediction("e", 1.0, 0), Prediction("e", 0.0, 1)]
    print(f"Parfait      (toujours sur et juste) : Brier = {brier_score(parfait):.3f}")
    print(f"Sur-confiant (toujours sur et faux)  : Brier = {brier_score(surconfiant):.3f}")

    print("\nA retenir : se tromper en etant tres confiant coute cher (terme au carre).")
    print("Tenir un journal de predictions chiffrees + le scorer = la facon la plus")
    print("honnete de savoir si votre jugement vaut quelque chose.")

"""
Module 08 -- Calibration & Forecasting : score de Brier et courbe de calibration.

Outil du domaine `rationalite-decision`. Permet de SCORER un journal de
predictions : etes-vous bien calibre (quand vous dites 70%, est-ce vrai ~70%
du temps) ? Stdlib pur, aucune dependance externe.

Le score de Brier = moyenne de (proba - resultat)^2, ou resultat vaut 1
(l'evenement s'est produit) ou 0 (non). Plus c'est BAS, mieux c'est. Reperes :
  - 0.0  = parfait (impossible en pratique)
  - 0.25 = baseline "toujours 50%"
  - 1.0  = parfaitement et confiantement faux

Exemples 100% neutres (meteo, sport) -- aucune prise de position.

Run :
    python domains/vie/rationalite-decision/02-code/08-calibration-forecasting.py
"""

from __future__ import annotations

from dataclasses import dataclass


# ===========================================================================
# 1. STRUCTURES DE DONNEES
# ===========================================================================

@dataclass
class Prediction:
    """Une prediction : un enonce, la proba annoncee, et le resultat observe.

    Args:
        enonce   : description concise de l'evenement predit
        proba    : probabilite annoncee dans [0.0, 1.0]
        resultat : 1 si l'evenement s'est produit, 0 sinon
    """
    enonce: str
    proba: float
    resultat: int


# ===========================================================================
# 2. SCORE DE BRIER
# ===========================================================================

def brier_score(predictions: list[Prediction]) -> float:
    """Calcule le score de Brier sur une liste de predictions.

    Formule : Brier = (1/N) * sum((p_i - o_i)^2)
    - p_i : proba annoncee pour la prediction i (dans [0, 1])
    - o_i : resultat observe pour la prediction i (0 ou 1)

    Le carre penalise d'autant plus les erreurs commises avec forte confiance :
    dire "90%" et se tromper coute (0.9)^2 = 0.81 sur cette entree ;
    dire "55%" et se tromper ne coute que (0.55)^2 ~ 0.30.

    Returns:
        float : score de Brier (0.0 = parfait, 0.25 = baseline, 1.0 = pire cas)
    """
    if not predictions:
        return 0.0
    total = sum((p.proba - p.resultat) ** 2 for p in predictions)
    return total / len(predictions)


def brier_baseline(predictions: list[Prediction]) -> float:
    """Score de Brier d'un predicteur "toujours 50%" -- point de repere.

    Si votre score >= cette valeur, vos probas n'apportent pas plus d'information
    que de dire "je ne sais pas (50/50)" a chaque fois.

    Returns:
        float : score baseline (vaut 0.25 quand le jeu est equilibre)
    """
    if not predictions:
        return 0.0
    # Predicteur naif : dit toujours 0.5, independamment de l'evenement.
    return sum((0.5 - p.resultat) ** 2 for p in predictions) / len(predictions)


# ===========================================================================
# 3. COURBE DE CALIBRATION (table ASCII par tranches de confiance)
# ===========================================================================

def calibration_table(predictions: list[Prediction], n_bins: int = 5) -> list[dict]:
    """Regroupe les predictions par tranche de probabilite annoncee.

    Pour chaque tranche [lo, hi), on compare :
    - proba_moy  : moyenne des probas annoncees dans la tranche
    - freq_reelle : proportion d'evenements survenus dans la tranche
    Un calibreur parfait a proba_moy == freq_reelle pour chaque tranche.

    Args:
        predictions : liste de Prediction
        n_bins      : nombre de tranches (5 par defaut -> [0-20%), [20-40%), ...)

    Returns:
        list[dict] avec les champs : tranche, n, proba_moy, freq_reelle, ecart
    """
    bins: list[list[Prediction]] = [[] for _ in range(n_bins)]
    for p in predictions:
        # min(...) : proba == 1.0 tombe dans la derniere tranche, pas hors borne.
        idx = min(int(p.proba * n_bins), n_bins - 1)
        bins[idx].append(p)

    rows: list[dict] = []
    for i, group in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not group:
            rows.append({
                "tranche": f"[{lo:.0%}-{hi:.0%})",
                "n": 0,
                "proba_moy": None,
                "freq_reelle": None,
                "ecart": None,
            })
            continue
        proba_moy = sum(p.proba for p in group) / len(group)
        freq_reelle = sum(p.resultat for p in group) / len(group)
        rows.append({
            "tranche": f"[{lo:.0%}-{hi:.0%})",
            "n": len(group),
            "proba_moy": proba_moy,
            "freq_reelle": freq_reelle,
            # ecart > 0 : sous-confiant (on annonce moins que ce qui se produit)
            # ecart < 0 : sur-confiant  (on annonce plus que ce qui se produit)
            "ecart": freq_reelle - proba_moy,
        })
    return rows


def print_calibration(predictions: list[Prediction], n_bins: int = 5) -> None:
    """Affiche la table de calibration en ASCII avec diagnostic par tranche."""
    header = f"{'Tranche':<12} {'n':>4} {'Annonce':>9} {'Reel':>8} {'Ecart':>8}   Diagnostic"
    print(header)
    print("-" * 65)
    for r in calibration_table(predictions, n_bins):
        if r["n"] == 0:
            print(f"{r['tranche']:<12} {0:>4} {'—':>9} {'—':>8} {'—':>8}")
            continue
        # Diagnostic : seuil d'ecart tolere = 10 points de % (0.10)
        if abs(r["ecart"]) < 0.10:
            diag = "calibre"
        elif r["ecart"] < 0:
            diag = "sur-confiant"  # on annonce plus que ce qui arrive
        else:
            diag = "sous-confiant"  # on annonce moins que ce qui arrive
        print(
            f"{r['tranche']:<12} {r['n']:>4} {r['proba_moy']:>9.0%}"
            f" {r['freq_reelle']:>8.0%} {r['ecart']:>+8.0%}   {diag}"
        )


# ===========================================================================
# 4. DEMO
# ===========================================================================

# Journal de predictions jouet (exemples neutres : meteo, sport).
# Chaque entree : (description, proba_annoncee, resultat_observe)
JOURNAL_DEMO: list[Prediction] = [
    # -- Meteo --
    Prediction("Pluie demain matin avant 12h",      0.80, 1),  # Il a plu
    Prediction("Temperature > 28 degres cet apres-midi", 0.65, 1),  # Il a fait 30 degres
    Prediction("Rafales > 50 km/h ce soir",         0.40, 0),  # Pas de vent fort
    Prediction("Orage dans les 48h",                0.55, 1),  # Orage survenu
    Prediction("Journee ensoleilee sans nuages",     0.20, 0),  # Nuageux
    # -- Sport --
    Prediction("L'equipe locale gagne le match",    0.60, 0),  # Defaite
    Prediction("Match termine en moins de 90 min (pas de prolongation)", 0.75, 1),  # Oui
    Prediction("Au moins 3 buts dans le match",     0.50, 1),  # 4 buts
    Prediction("Le favori gagne le tournoi",        0.70, 1),  # Oui
    Prediction("Record du stade battu ce week-end", 0.15, 0),  # Non
    # -- Logistique quotidienne neutre --
    Prediction("Le bus arrive avec moins de 5 min de retard", 0.55, 0),  # En retard
    Prediction("La reunion se termine a l'heure",   0.45, 0),  # Depasse
]


def demo_profiles_extremes() -> None:
    """Montre deux profils extremes pour ancrer l'interpretation du score."""
    parfait = [
        Prediction("", 1.0, 1),
        Prediction("", 0.0, 0),
        Prediction("", 1.0, 1),
        Prediction("", 0.0, 0),
    ]
    surconfiant = [
        Prediction("", 0.95, 0),  # tres confiant, faux
        Prediction("", 0.90, 0),
        Prediction("", 0.05, 1),  # tres confiant dans la negation, faux
    ]
    print(f"  Parfait      (confiant ET juste)  : Brier = {brier_score(parfait):.3f}")
    print(f"  Sur-confiant (confiant ET faux)   : Brier = {brier_score(surconfiant):.3f}")
    print(f"  Baseline     (toujours 50%)        : Brier = 0.250")


if __name__ == "__main__":
    sep = "=" * 65

    print(sep)
    print("Module 08 -- Calibration & Forecasting")
    print(sep)

    # ---- Score de Brier global ----
    bs = brier_score(JOURNAL_DEMO)
    ref = brier_baseline(JOURNAL_DEMO)
    n = len(JOURNAL_DEMO)
    print(f"\nPredictions evaluees  : {n}")
    print(f"Score de Brier        : {bs:.3f}  (plus bas = mieux ; 0.0 = parfait)")
    print(f"Baseline 'toujours 50%' : {ref:.3f}")
    if bs < ref:
        print("Verdict : vous BATTEZ le 50/50 -> vos probas apportent de l'info")
    else:
        print("Verdict : vous ne battez PAS le 50/50 -> calibration a revoir")

    # ---- Courbe de calibration ----
    print(f"\n{sep}")
    print("Courbe de calibration (proba annoncee vs frequence reelle par tranche)")
    print(sep)
    print("  Ecart > 0 -> sous-confiant (annonce moins que ce qui se produit)")
    print("  Ecart < 0 -> sur-confiant  (annonce plus que ce qui se produit)\n")
    print_calibration(JOURNAL_DEMO, n_bins=5)

    # ---- Profils extremes pour ancrer l'interpretation ----
    print(f"\n{sep}")
    print("Reperes : deux profils extremes")
    print(sep)
    demo_profiles_extremes()

    # ---- Lecon sur la penalisation de la sur-confiance ----
    print(f"\n{sep}")
    print("Illustration : cout de la sur-confiance")
    print(sep)
    cas = [
        Prediction("predit a 90%, resultat = 0 (faux)",   0.90, 0),
        Prediction("predit a 55%, resultat = 0 (faux)",   0.55, 0),
        Prediction("predit a 90%, resultat = 1 (juste)",  0.90, 1),
    ]
    for c in cas:
        cout = (c.proba - c.resultat) ** 2
        print(f"  {c.enonce:<45} -> contribution Brier = {cout:.4f}")
    print("\nConclusion : se tromper en etant tres confiant coute tres cher.")
    print("Tenir un journal de predictions chiffrees + le scorer = la facon la plus")
    print("honnete de savoir si votre jugement vaut quelque chose.")

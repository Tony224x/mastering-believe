"""
Module 06 — Calibration & Vérification
Scoreur de calibration : score de Brier + courbe de calibration en texte.

Stdlib pur (Python 3.11+). Aucune dépendance externe.

Usage:
    python domains/rationalite-decision/02-code/06-calibration-verification.py

Concepts:
    - Score de Brier : (1/N) * sum((p_i - o_i)^2)
        p_i : probabilité prédite (entre 0 et 1)
        o_i : outcome réel (0 ou 1)
        Parfait = 0.0 | Baseline (50 % partout) = 0.25 | Pire = 1.0
    - Courbe de calibration : pour chaque bin de probabilités (ex. 0-10 %, 10-20 %...),
        comparer la probabilité moyenne prédite à la fréquence réelle des outcomes.
        Un prévisionniste parfaitement calibré a des points sur la diagonale y = x.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Types de base
# ---------------------------------------------------------------------------

# Une prédiction est un tuple (probabilité_prédite, outcome_réel)
# probabilité_prédite : float dans [0, 1]
# outcome_réel        : int 0 ou 1
Prediction = tuple[float, int]


# ---------------------------------------------------------------------------
# Score de Brier
# ---------------------------------------------------------------------------

def brier_score(predictions: Sequence[Prediction]) -> float:
    """Calcule le score de Brier moyen sur un ensemble de prédictions.

    Args:
        predictions: liste de (p, o) où p est la probabilité prédite (0-1)
                     et o est l'outcome réel (0 ou 1).

    Returns:
        Score de Brier (float). Plus bas = meilleur.
        Retourne float('nan') si la liste est vide.

    >>> brier_score([(1.0, 1), (0.0, 0)])  # prédictions parfaites
    0.0
    >>> round(brier_score([(0.5, 1), (0.5, 0)]), 4)  # baseline
    0.25
    """
    if not predictions:
        return float("nan")

    # Somme des erreurs quadratiques : (p - o)^2 pour chaque prédiction
    total = sum((p - o) ** 2 for p, o in predictions)
    return total / len(predictions)


# ---------------------------------------------------------------------------
# Courbe de calibration (bins)
# ---------------------------------------------------------------------------

def calibration_curve(
    predictions: Sequence[Prediction],
    n_bins: int = 10,
) -> list[dict]:
    """Calcule la courbe de calibration par intervalles (bins).

    Divise les prédictions en n_bins intervalles de probabilité [0, 1/n],
    [1/n, 2/n], etc. Pour chaque bin non vide :
        - p_mean : probabilité moyenne prédite dans le bin
        - freq   : fréquence réelle des outcomes (proportion d'outcomes = 1)
        - count  : nombre de prédictions dans le bin

    Un prévisionniste parfaitement calibré a p_mean ≈ freq dans chaque bin.

    Args:
        predictions: liste de (p, o).
        n_bins: nombre d'intervalles (défaut 10, soit tranches de 10 %).

    Returns:
        Liste de dicts triée par p_mean.
    """
    # Initialiser les bins : chaque bin collecte (liste de p, liste de o)
    bins: list[list[list]] = [[[], []] for _ in range(n_bins)]

    for p, o in predictions:
        # Clamp pour gérer les cas limites p=1.0
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx][0].append(p)  # probabilités prédites
        bins[idx][1].append(o)  # outcomes

    results = []
    for i, (ps, os) in enumerate(bins):
        if not ps:
            continue  # bin vide, on ignore

        # Borne basse et haute du bin (pour l'affichage)
        low = i / n_bins
        high = (i + 1) / n_bins

        p_mean = sum(ps) / len(ps)         # probabilité moyenne dans le bin
        freq = sum(os) / len(os)           # fréquence réelle des outcomes
        count = len(ps)

        results.append({
            "bin": f"{low:.0%}–{high:.0%}",
            "p_mean": p_mean,
            "freq": freq,
            "count": count,
            "gap": p_mean - freq,          # écart (positif = sur-confiance)
        })

    return results


# ---------------------------------------------------------------------------
# Affichage texte de la courbe de calibration
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20, char: str = "█") -> str:
    """Génère une barre de progression ASCII."""
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


def print_calibration_report(
    predictions: Sequence[Prediction],
    title: str = "Rapport de calibration",
) -> None:
    """Affiche un rapport de calibration complet en texte."""
    n = len(predictions)
    if n == 0:
        print("Aucune prédiction à analyser.")
        return

    score = brier_score(predictions)
    baseline = 0.25  # score d'un prévisionniste qui dit toujours 50 %

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Nombre de prédictions : {n}")
    print(f"  Score de Brier        : {score:.4f}")
    print(f"  Baseline (50% partout): {baseline:.4f}")

    if not math.isnan(score):
        if score < baseline:
            gain = (baseline - score) / baseline * 100
            print(f"  → Meilleur que le hasard de {gain:.1f} %")
        else:
            loss = (score - baseline) / baseline * 100
            print(f"  → Moins bon que le hasard de {loss:.1f} %")

    print(f"\n{'─' * 60}")
    print("  Courbe de calibration (prédit vs réel)")
    print(f"  {'Bin':<10} {'Prédit':>7} {'Réel':>7} {'N':>4}  Barre (réel)")
    print(f"  {'───':<10} {'──────':>7} {'──────':>7} {'─':>4}  {'─' * 20}")

    curve = calibration_curve(predictions)
    for row in curve:
        bar = _bar(row["freq"])
        gap_str = ""
        if abs(row["gap"]) > 0.10:
            # Signaler les écarts significatifs
            direction = "↑sur-confiance" if row["gap"] > 0 else "↓sous-confiance"
            gap_str = f"  {direction}"
        print(
            f"  {row['bin']:<10} "
            f"{row['p_mean']:>6.1%} "
            f"{row['freq']:>7.1%} "
            f"{row['count']:>4}  "
            f"{bar}"
            f"{gap_str}"
        )

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Démo — données d'exemple
# ---------------------------------------------------------------------------

DEMO_PREDICTIONS: list[Prediction] = [
    # (probabilité prédite, outcome réel)
    # Prévisions météo (pluie)
    (0.80, 1),  # prédit 80 % → il a plu
    (0.30, 0),  # prédit 30 % → pas de pluie
    (0.70, 1),  # prédit 70 % → pluie
    (0.20, 0),  # prédit 20 % → pas de pluie
    (0.90, 1),  # prédit 90 % → pluie
    # Prévisions sportives (victoire équipe locale)
    (0.60, 0),  # prédit 60 % → défaite (sur-confiance)
    (0.55, 1),  # prédit 55 % → victoire
    (0.45, 0),  # prédit 45 % → défaite
    (0.75, 1),  # prédit 75 % → victoire
    (0.40, 0),  # prédit 40 % → défaite
    # Prévisions de délais / tâches
    (0.85, 1),  # prédit 85 % → livraison dans les temps
    (0.50, 1),  # prédit 50 % → tâche finie à l'heure
    (0.65, 0),  # prédit 65 % → retard (sur-confiance)
    (0.10, 0),  # prédit 10 % → pas d'annulation (bien calibré)
    (0.95, 1),  # prédit 95 % → succès (confiance justifiée)
    # Prévisions de résultats personnels
    (0.30, 1),  # prédit 30 % → ça s'est quand même produit (sous-confiance)
    (0.80, 0),  # prédit 80 % → non réalisé (sur-confiance)
    (0.50, 0),  # prédit 50 % → non réalisé
    (0.15, 0),  # prédit 15 % → bien (faible probabilité, pas réalisé)
    (0.70, 1),  # prédit 70 % → réalisé
]


if __name__ == "__main__":
    # Rapport principal sur les données de démo
    print_calibration_report(DEMO_PREDICTIONS, title="Démo — 20 prévisions mixtes")

    # Exemple minimaliste pour illustrer les cas extrêmes
    perfect = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
    worst = [(0.0, 1), (1.0, 0), (0.0, 1), (1.0, 0)]
    baseline_preds = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]

    print("Valeurs de référence :")
    print(f"  Prédictions parfaites    → Brier = {brier_score(perfect):.4f}")
    print(f"  Baseline (50 % partout)  → Brier = {brier_score(baseline_preds):.4f}")
    print(f"  Prédictions inversées    → Brier = {brier_score(worst):.4f}")
    print()

    # Exemple : scorer vos propres prédictions
    # Remplacez my_predictions par vos données réelles
    my_predictions: list[Prediction] = [
        # (0.70, 1),   # décommentez et ajoutez vos entrées ici
        # (0.40, 0),
    ]

    if my_predictions:
        print_calibration_report(my_predictions, title="Mes prédictions personnelles")
    else:
        print("Astuce : ajoutez vos prédictions dans `my_predictions` pour vous scorer.")

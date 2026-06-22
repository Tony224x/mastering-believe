"""
Module 10 — Lire une étude & stats trompeuses
Simulation de p-hacking : tester N hypothèses sur du bruit pur.

Objectif pédagogique :
  - Montrer qu'en testant de nombreuses hypothèses indépendantes sur des données
    aléatoires, on obtient mécaniquement des résultats "significatifs" (faux positifs).
  - Visualiser comment le taux de faux positifs explose avec le nombre de tests.

Stdlib uniquement. Aucune clé API ni accès réseau.
"""

import random
import math


# ---------------------------------------------------------------------------
# Utilitaires statistiques (stdlib uniquement)
# ---------------------------------------------------------------------------

def mean(data: list[float]) -> float:
    """Moyenne arithmétique."""
    return sum(data) / len(data)


def variance(data: list[float]) -> float:
    """Variance non biaisée (diviseur N-1)."""
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def pooled_std(group_a: list[float], group_b: list[float]) -> float:
    """Écart-type poolé pour le test t de Welch simplifié."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = variance(group_a), variance(group_b)
    # Erreur standard de la différence des moyennes
    return math.sqrt(var_a / n_a + var_b / n_b)


def t_statistic(group_a: list[float], group_b: list[float]) -> float:
    """Statistique t pour deux groupes indépendants (approximation de Welch)."""
    diff = mean(group_a) - mean(group_b)
    se = pooled_std(group_a, group_b)
    if se == 0:
        return 0.0
    return diff / se


def approx_p_value_two_tailed(t: float, df: int) -> float:
    """
    Approximation de la p-value bilatérale pour le test t via la distribution
    normale standard (valide pour df > 30).

    Utilise la fonction de survie de la loi normale : 2 * P(Z > |t|).
    Approximation de Abramowitz & Stegun pour la CDF normale.
    """
    abs_t = abs(t)

    # CDF de la loi normale standard par approximation polynomiale (AS formula 26.2.17)
    # précision ~1e-5 pour |z| < 5
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p_coef = 0.2316419

    t_pos = abs_t
    k = 1.0 / (1.0 + p_coef * t_pos)
    poly = k * (b1 + k * (b2 + k * (b3 + k * (b4 + k * b5))))
    # Densité de la normale standard
    phi = math.exp(-0.5 * t_pos ** 2) / math.sqrt(2 * math.pi)
    # Probabilité de la queue droite
    p_one_tail = phi * poly
    # Bilatérale
    return min(2 * p_one_tail, 1.0)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """
    Taille d'effet d de Cohen.
    d = (μ_A - μ_B) / s_pooled  où s_pooled = √((s_A² + s_B²) / 2).
    """
    var_a = variance(group_a)
    var_b = variance(group_b)
    s_pooled = math.sqrt((var_a + var_b) / 2)
    if s_pooled == 0:
        return 0.0
    return (mean(group_a) - mean(group_b)) / s_pooled


# ---------------------------------------------------------------------------
# Génération de données aléatoires (bruit pur, H₀ vraie par construction)
# ---------------------------------------------------------------------------

def generate_noise_groups(n: int, rng: random.Random) -> tuple[list[float], list[float]]:
    """
    Génère deux groupes de N observations tirées de la même distribution
    normale (μ=0, σ=1) via Box-Muller.  H₀ est vraie par construction :
    aucun effet réel n'existe.
    """
    def normal_sample(size: int) -> list[float]:
        samples = []
        for _ in range(size):
            # Box-Muller transform : deux uniformes → une normale
            u1 = rng.random()
            u2 = rng.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            samples.append(z)
        return samples

    return normal_sample(n), normal_sample(n)


# ---------------------------------------------------------------------------
# Simulation principale
# ---------------------------------------------------------------------------

def simulate_p_hacking(
    num_hypotheses: int,
    n_per_group: int,
    alpha: float,
    seed: int,
) -> dict:
    """
    Simule un chercheur qui teste `num_hypotheses` hypothèses indépendantes
    sur du bruit pur et ne retient que celles avec p < alpha.

    Paramètres
    ----------
    num_hypotheses : nombre de tests réalisés (= researcher degrees of freedom)
    n_per_group    : taille de chaque groupe dans chaque test
    alpha          : seuil de significativité (typiquement 0.05)
    seed           : graine pour reproductibilité

    Retourne un dict avec les résultats agrégés.
    """
    rng = random.Random(seed)
    results = []

    for i in range(num_hypotheses):
        group_a, group_b = generate_noise_groups(n_per_group, rng)
        t = t_statistic(group_a, group_b)
        # df approximatif = 2*(n-1) pour groupes égaux
        df = 2 * (n_per_group - 1)
        p = approx_p_value_two_tailed(t, df)
        d = cohens_d(group_a, group_b)
        results.append({"hypothesis": i + 1, "t": t, "p": p, "d": d})

    false_positives = [r for r in results if r["p"] < alpha]
    return {
        "num_hypotheses": num_hypotheses,
        "n_per_group": n_per_group,
        "alpha": alpha,
        "num_false_positives": len(false_positives),
        "false_positive_rate": len(false_positives) / num_hypotheses,
        "expected_false_positives": num_hypotheses * alpha,
        "false_positives": false_positives,
        "all_results": results,
    }


def print_separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Démonstration pédagogique
# ---------------------------------------------------------------------------

def demo_single_scenario(
    num_hypotheses: int,
    n_per_group: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> None:
    """Affiche les résultats d'une simulation de p-hacking."""
    sim = simulate_p_hacking(num_hypotheses, n_per_group, alpha, seed)

    print(f"\n{'='*60}")
    print(f"  SCÉNARIO : {num_hypotheses} hypothèses testées sur du BRUIT PUR")
    print(f"  Taille d'échantillon : {n_per_group} par groupe | α = {alpha}")
    print(f"{'='*60}")
    print(f"  Faux positifs observés : {sim['num_false_positives']}")
    print(f"  Taux de faux positifs  : {sim['false_positive_rate']:.1%}")
    print(f"  Faux positifs attendus : {sim['expected_false_positives']:.1f} "
          f"(= {num_hypotheses} × {alpha})")

    if sim["false_positives"]:
        print(f"\n  → Détail des {sim['num_false_positives']} résultat(s) 'significatif(s)' :")
        for fp in sim["false_positives"]:
            print(f"    Hypothèse #{fp['hypothesis']:3d} | "
                  f"p = {fp['p']:.4f} | "
                  f"d de Cohen = {fp['d']:.3f}  (taille d'effet)")
        print()
        print("  RAPPEL : ces effets sont 100 % aléatoires — H₀ est vraie par")
        print("  construction. Un lecteur ne voyant que le 'résultat significatif'")
        print("  ignorerait tous les autres tests effectués.")
    else:
        print("  → Aucun faux positif cette fois (chance !)")


def demo_false_positive_rate_vs_num_tests(
    n_per_group: int = 30,
    alpha: float = 0.05,
    num_simulations: int = 500,
    seed: int = 99,
) -> None:
    """
    Montre comment le taux de faux positifs augmente avec le nombre de tests,
    en moyennant sur plusieurs simulations.
    """
    print(f"\n{'='*60}")
    print("  TAUX DE FAUX POSITIFS vs NOMBRE DE TESTS (moyenne sur"
          f" {num_simulations} simulations)")
    print(f"  α = {alpha} | n = {n_per_group} par groupe")
    print(f"{'='*60}")
    print(f"  {'Tests':>6} | {'FP moyen':>10} | {'Taux moyen':>12} | "
          f"{'Prob. ≥1 FP':>12}")
    print_separator()

    rng_outer = random.Random(seed)

    for k in [1, 5, 10, 20, 50, 100]:
        fp_counts = []
        at_least_one = 0

        for _ in range(num_simulations):
            sim = simulate_p_hacking(
                num_hypotheses=k,
                n_per_group=n_per_group,
                alpha=alpha,
                seed=rng_outer.randint(0, 2**31),
            )
            fp_counts.append(sim["num_false_positives"])
            if sim["num_false_positives"] > 0:
                at_least_one += 1

        avg_fp = mean(fp_counts)
        avg_rate = avg_fp / k
        prob_at_least_one = at_least_one / num_simulations
        # Probabilité théorique d'au moins 1 FP : 1 - (1-α)^k
        theoretical = 1 - (1 - alpha) ** k

        print(f"  {k:>6} | {avg_fp:>10.2f} | {avg_rate:>11.1%} | "
              f"{prob_at_least_one:>11.1%}  (théo: {theoretical:.1%})")

    print()
    print("  Interprétation : avec 20 tests, la probabilité de trouver au")
    print("  moins un 'résultat significatif' par hasard avoisine 64 %.")
    print("  Avec 100 tests, elle dépasse 99 %.")


def demo_effect_size_lesson(seed: int = 7) -> None:
    """
    Montre qu'un grand N peut rendre statistiquement significative
    une différence pratiquement nulle (p-value ≠ importance).
    """
    print(f"\n{'='*60}")
    print("  LEÇON : p-value ≠ taille d'effet")
    print(f"{'='*60}")
    print(f"  {'N/groupe':>8} | {'p-value':>10} | {'d Cohen':>9} | {'Significatif':>13}")
    print_separator()

    rng = random.Random(seed)

    # On injecte un très petit effet réel (δ = 0.05 σ) pour illustrer
    delta = 0.05  # effet minuscule mais réel

    for n in [30, 100, 500, 2000, 10_000]:
        # Groupe A : N(0, 1) ; groupe B : N(delta, 1)
        group_a = []
        group_b = []
        for _ in range(n):
            u1 = rng.random()
            u2 = rng.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            group_a.append(z)
            group_b.append(z + delta)  # décalage artificiel minime

        t = t_statistic(group_a, group_b)
        df = 2 * (n - 1)
        p = approx_p_value_two_tailed(t, df)
        d = cohens_d(group_a, group_b)
        sig = "OUI  ← attention" if p < 0.05 else "non"

        print(f"  {n:>8} | {p:>10.4f} | {d:>9.3f} | {sig:>13}")

    print()
    print(f"  Effet injecté : δ = {delta} σ (quasi imperceptible en pratique).")
    print("  Avec N=10 000, cet effet devient 'significatif' — mais d ≈ 0.05")
    print("  correspond à une taille d'effet négligeable (seuil usuel : 0.20).")
    print("  → Toujours lire la taille d'effet, pas seulement la p-value.")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 10 — Simulation de p-hacking")
    print("  Toutes les données sont du bruit aléatoire pur (H₀ vraie).")
    print("=" * 60)

    # --- Scénario 1 : 20 hypothèses testées, on publie le « résultat » ---
    demo_single_scenario(
        num_hypotheses=20,
        n_per_group=30,
        alpha=0.05,
        seed=42,
    )

    # --- Scénario 2 : 100 hypothèses (chercheur très flexible) ---
    demo_single_scenario(
        num_hypotheses=100,
        n_per_group=30,
        alpha=0.05,
        seed=123,
    )

    # --- Vue d'ensemble : taux de faux positifs vs nombre de tests ---
    demo_false_positive_rate_vs_num_tests(
        n_per_group=30,
        alpha=0.05,
        num_simulations=500,
        seed=99,
    )

    # --- Leçon p-value vs taille d'effet ---
    demo_effect_size_lesson(seed=7)

    print("\n" + "=" * 60)
    print("  CONCLUSION")
    print("=" * 60)
    print("  1. Tester de nombreuses hypothèses sans correction (Bonferroni,")
    print("     FDR) produit mécaniquement des faux positifs.")
    print("  2. p < 0.05 ≠ vrai ; p < 0.05 ≠ important.")
    print("  3. La taille d'effet (d de Cohen) mesure l'importance pratique.")
    print("  4. Le pré-enregistrement et la transparence des analyses")
    print("     sont les principaux antidotes au p-hacking.")
    print("=" * 60)

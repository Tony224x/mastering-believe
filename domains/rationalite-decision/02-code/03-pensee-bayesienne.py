"""
Module 03 — Pensee bayesienne : mise a jour prior/likelihood -> posterior
=========================================================================
Illustre le theoreme de Bayes sous trois formes :
  1. Forme probabiliste classique  P(H|E) = P(E|H)*P(H) / P(E)
  2. Forme odds (rapport de vraisemblance)
  3. Mise a jour sequentielle (chaque posterior devient le prior suivant)

Utilisation : python 03-pensee-bayesienne.py
Stdlib pur — aucune dependance externe.
"""

import math


# ---------------------------------------------------------------------------
# Primitives bayesiennes
# ---------------------------------------------------------------------------

def bayes_posterior(prior: float, vraisemblance_h: float,
                    vraisemblance_non_h: float) -> float:
    """
    Calcule le posterior P(H|E) via le theoreme de Bayes.

    Parametres
    ----------
    prior : float
        P(H) — probabilite de l'hypothese avant d'observer la preuve.
    vraisemblance_h : float
        P(E|H) — probabilite de la preuve si H est vraie.
    vraisemblance_non_h : float
        P(E|non-H) — probabilite de la preuve si H est fausse.

    Retourne
    --------
    float : P(H|E) — probabilite de H apres avoir observe la preuve.
    """
    # Probabilite totale de la preuve (denominateur de Bayes)
    p_evidence = vraisemblance_h * prior + vraisemblance_non_h * (1 - prior)

    if p_evidence == 0:
        raise ValueError("P(E) = 0 : la preuve est impossible quelle que soit l'hypothese.")

    # Theoreme de Bayes : numerateur / denominateur
    return (vraisemblance_h * prior) / p_evidence


def rapport_de_vraisemblance(vraisemblance_h: float,
                              vraisemblance_non_h: float) -> float:
    """
    Calcule le rapport de vraisemblance (LR) = P(E|H) / P(E|non-H).

    LR > 1 : la preuve soutient H.
    LR < 1 : la preuve affaiblit H.
    LR = 1 : la preuve n'est pas informative.
    """
    if vraisemblance_non_h == 0:
        return float("inf")  # preuve impossible si H fausse => preuve parfaite pour H
    return vraisemblance_h / vraisemblance_non_h


def prob_vers_odds(prob: float) -> float:
    """Convertit une probabilite en odds. P -> P/(1-P)."""
    if prob >= 1.0:
        return float("inf")
    return prob / (1 - prob)


def odds_vers_prob(odds: float) -> float:
    """Convertit des odds en probabilite. O -> O/(1+O)."""
    if math.isinf(odds):
        return 1.0
    return odds / (1 + odds)


def bayes_odds(prior: float, lr: float) -> float:
    """
    Mise a jour bayesienne sous forme odds.

    Odds posterieur = Odds prieur × LR
    Puis reconversion en probabilite.

    Parametres
    ----------
    prior : float  — P(H) avant la preuve
    lr    : float  — rapport de vraisemblance P(E|H) / P(E|non-H)

    Retourne
    --------
    float : P(H|E)
    """
    odds_prieur = prob_vers_odds(prior)
    odds_posterieur = odds_prieur * lr
    return odds_vers_prob(odds_posterieur)


def mise_a_jour_sequentielle(prior: float,
                              preuves: list[tuple[float, float]]) -> list[float]:
    """
    Applique une sequence de mises a jour bayesiennes.

    Chaque element de `preuves` est un tuple (vraisemblance_h, vraisemblance_non_h).
    Le posterior de chaque etape devient le prior de la suivante.

    Retourne la liste de tous les posteriors (un par preuve).
    """
    posteriors = []
    p_courante = prior  # on part du prior initial

    for i, (lh, lnh) in enumerate(preuves):
        p_courante = bayes_posterior(p_courante, lh, lnh)
        posteriors.append(p_courante)

    return posteriors


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_urne():
    """
    Exemple classique : deux urnes, on tire une bille et on met a jour.

    Urne A : 70 % rouge, 30 % bleue
    Urne B : 40 % rouge, 60 % bleue
    Prior : 50/50
    On tire une bille rouge.
    """
    print("\n" + "="*60)
    print("  DEMO 1 : Probleme des urnes")
    print("="*60)
    print("  Urne A : 70 % rouge | Urne B : 40 % rouge")
    print("  Prior P(Urne A) = 50 %")
    print("  Observation : on tire une bille ROUGE")

    prior = 0.50
    # P(rouge | Urne A) = 0.70 ; P(rouge | Urne B) = 0.40
    posterior = bayes_posterior(prior, vraisemblance_h=0.70, vraisemblance_non_h=0.40)

    print(f"\n  Prior      P(A)        = {prior*100:.1f} %")
    print(f"  P(rouge|A)              = 70,0 %")
    print(f"  P(rouge|B)              = 40,0 %")
    print(f"  LR                      = {rapport_de_vraisemblance(0.70, 0.40):.2f}")
    print(f"  Posterior  P(A|rouge)  = {posterior*100:.1f} %")
    print()
    print("  Interpretation : la bille rouge augmente la confiance")
    print(f"  en l'urne A de 50 % a {posterior*100:.1f} %.")


def demo_test_medical():
    """
    Reprise de l'exemple du module 02 avec la forme Bayes explicite.
    Prevalence 1 %, sensibilite 90 %, specificite 95 %.
    """
    print("\n" + "="*60)
    print("  DEMO 2 : Test medical (reprise module 02)")
    print("="*60)
    print("  Prevalence 1 %, sensibilite 90 %, specificite 95 %")

    prior = 0.01           # taux de base de la maladie
    vrais_pos = 0.90       # P(test+ | malade) = sensibilite
    faux_pos = 1 - 0.95    # P(test+ | sain)   = 1 - specificite

    posterior = bayes_posterior(prior, vrais_pos, faux_pos)
    lr = rapport_de_vraisemblance(vrais_pos, faux_pos)

    print(f"\n  Prior P(malade)          = {prior*100:.1f} %")
    print(f"  P(test+ | malade)         = {vrais_pos*100:.1f} %  (sensibilite)")
    print(f"  P(test+ | sain)           = {faux_pos*100:.1f} %  (1 - specificite)")
    print(f"  Rapport de vraisemblance  = {lr:.1f}")
    print(f"  Posterior P(malade|test+) = {posterior*100:.1f} %")
    print()
    print("  Verification via la forme odds :")
    posterior_odds = bayes_odds(prior, lr)
    print(f"  Odds prieur = {prob_vers_odds(prior):.4f}")
    print(f"  Odds post.  = {prob_vers_odds(prior)*lr:.4f}")
    print(f"  => P(malade|test+) = {posterior_odds*100:.1f} % (identique)")


def demo_sequentielle():
    """
    Deux tests positifs independants successifs sur la meme maladie.
    Illustre la mise a jour sequentielle.
    """
    print("\n" + "="*60)
    print("  DEMO 3 : Mise a jour sequentielle — 2 tests positifs")
    print("="*60)
    print("  Meme maladie : prevalence 1 %, sensibilite 90 %, specificite 95 %")
    print("  On effectue 2 tests independants, tous deux positifs.")

    prior = 0.01
    # Chaque test a les memes caracteristiques
    preuves = [
        (0.90, 0.05),  # 1er test positif : P(test+|malade)=0.90, P(test+|sain)=0.05
        (0.90, 0.05),  # 2eme test positif (independant)
    ]

    posteriors = mise_a_jour_sequentielle(prior, preuves)

    print(f"\n  Prior                       = {prior*100:.1f} %")
    for i, p in enumerate(posteriors):
        print(f"  Apres test {i+1} positif       = {p*100:.1f} %")

    print()
    print("  Interpretation : chaque test positif accumule les preuves.")
    print(f"  On passe de {prior*100:.1f} % -> {posteriors[0]*100:.1f} % -> {posteriors[1]*100:.1f} %")
    print("  meme si la maladie est initialement rare.")


def demo_rapport_vraisemblance():
    """
    Compare la force informationnelle de plusieurs preuves via leur LR.
    """
    print("\n" + "="*60)
    print("  DEMO 4 : Comparer des preuves par leur rapport de vraisemblance")
    print("="*60)

    preuves_examples = [
        ("Test tres fort (LR=18)",        0.90, 0.05),
        ("Test modere (LR=2,33)",         0.70, 0.30),
        ("Preuve faible (LR=1,20)",       0.60, 0.50),
        ("Preuve non informative (LR=1)", 0.50, 0.50),
        ("Preuve contre H (LR=0.5)",      0.20, 0.40),
    ]

    prior = 0.20  # prior arbitraire de 20 %

    print(f"\n  Prior P(H) = {prior*100:.0f} % pour tous les cas\n")
    print(f"  {'Preuve':35s} {'LR':>6s}  {'Posterior':>10s}")
    print(f"  {'-'*55}")

    for label, lh, lnh in preuves_examples:
        lr = rapport_de_vraisemblance(lh, lnh)
        post = bayes_posterior(prior, lh, lnh)
        print(f"  {label:35s} {lr:>6.2f}  {post*100:>9.1f} %")

    print()
    print("  LR >> 1 : preuve forte pour H.")
    print("  LR ≈ 1  : preuve non informative.")
    print("  LR << 1 : preuve contre H.")


if __name__ == "__main__":
    print("\nModule 03 — Pensee bayesienne : demonstrations")
    print("Theoreme de Bayes : P(H|E) = P(E|H) × P(H) / P(E)")

    demo_urne()
    demo_test_medical()
    demo_sequentielle()
    demo_rapport_vraisemblance()

    print("\n" + "="*60)
    print("  Pour experimenter avec vos propres valeurs :")
    print("  from 03-pensee-bayesienne import bayes_posterior, mise_a_jour_sequentielle")
    print()
    print("  # Exemple : prior 30 %, preuve avec LR = 5")
    print("  post = bayes_posterior(prior=0.30, vraisemblance_h=0.70, vraisemblance_non_h=0.14)")
    print(f"  # => {bayes_posterior(0.30, 0.70, 0.14)*100:.1f} %")
    print("="*60)

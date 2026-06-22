"""
Module 04 — Pensee bayesienne : mise a jour sequentielle prior -> posterior
===========================================================================
Valeur ajoutee vs module 03 :
  - Notation formelle explicitee dans les docstrings
  - Focus sur la mise a jour ITERATIVE (le posterior devient le prior suivant)
  - Trois scenarios neutres distincts :
      1. Urne et billes (tirage avec remise)
      2. Lumiere oubliee (3 hypotheses)
      3. Controle qualite (machine defectueuse)

Utilisation : python 04-pensee-bayesienne.py
Stdlib pur — aucune dependance externe. Exit 0.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Primitives bayesiennes (re-exportees pour import externe)
# ---------------------------------------------------------------------------

def bayes_posterior(prior: float,
                    vraisemblance_h: float,
                    vraisemblance_non_h: float) -> float:
    """
    Calcule P(H|E) via le theoreme de Bayes (forme developpee).

    Formule :
        P(H|E) = P(E|H) * P(H)
                 -----------------------------------------
                 P(E|H) * P(H) + P(E|non-H) * (1 - P(H))

    Parametres
    ----------
    prior            : P(H)       — probabilite de H avant la preuve
    vraisemblance_h  : P(E|H)     — probabilite de la preuve si H vraie
    vraisemblance_non_h : P(E|nH) — probabilite de la preuve si H fausse

    Retourne
    --------
    float : P(H|E), le posterior.
    """
    # Probabilite totale de la preuve (denominateur de Bayes)
    p_e = vraisemblance_h * prior + vraisemblance_non_h * (1.0 - prior)

    if p_e == 0.0:
        raise ValueError(
            "P(E) = 0 : la preuve est impossible quelle que soit l'hypothese."
        )

    # Numerateur = P(E|H) * P(H)
    return (vraisemblance_h * prior) / p_e


def rapport_de_vraisemblance(vraisemblance_h: float,
                              vraisemblance_non_h: float) -> float:
    """
    LR = P(E|H) / P(E|non-H).

    LR > 1  : la preuve soutient H.
    LR < 1  : la preuve affaiblit H.
    LR = 1  : la preuve est non informative.
    """
    if vraisemblance_non_h == 0.0:
        return float("inf")
    return vraisemblance_h / vraisemblance_non_h


def prob_vers_odds(prob: float) -> float:
    """Probabilite -> odds : O = P / (1 - P)."""
    if prob >= 1.0:
        return float("inf")
    if prob <= 0.0:
        return 0.0
    return prob / (1.0 - prob)


def odds_vers_prob(odds: float) -> float:
    """Odds -> probabilite : P = O / (1 + O)."""
    if odds == float("inf"):
        return 1.0
    return odds / (1.0 + odds)


def mise_a_jour_sequentielle(
    prior: float,
    preuves: list[tuple[float, float]],
    labels: list[str] | None = None,
    verbose: bool = True
) -> list[float]:
    """
    Applique une sequence de mises a jour bayesiennes.

    Parametres
    ----------
    prior   : P(H) initial
    preuves : liste de (vraisemblance_h, vraisemblance_non_h) pour chaque preuve
    labels  : noms optionnels des preuves (pour l'affichage)
    verbose : si True, affiche chaque etape

    Retourne
    --------
    list[float] : posteriors apres chaque preuve (meme longueur que `preuves`)

    Principe cle : le posterior de l'etape i devient le prior de l'etape i+1.
    Toutes les preuves passees sont ainsi "compressees" dans le prior courant.
    """
    if labels is None:
        labels = [f"preuve {i+1}" for i in range(len(preuves))]

    posteriors: list[float] = []
    p_courante = prior  # point de depart : le prior initial

    if verbose:
        print(f"  Prior initial        = {p_courante*100:.2f} %")

    for i, (lh, lnh) in enumerate(preuves):
        lr = rapport_de_vraisemblance(lh, lnh)
        # Le prior de cette iteration = posterior de l'iteration precedente
        p_courante = bayes_posterior(p_courante, lh, lnh)
        posteriors.append(p_courante)

        if verbose:
            print(f"  Apres {labels[i]:20s}  LR={lr:5.2f}  => {p_courante*100:.2f} %")

    return posteriors


# ---------------------------------------------------------------------------
# Demo 1 : urne et billes — mise a jour sequentielle
# ---------------------------------------------------------------------------

def demo_urne_sequentielle():
    """
    Urne A : 70 % rouge, 30 % bleue.
    Urne B : 40 % rouge, 60 % bleue.
    Prior 50/50. On tire 4 billes (toutes rouges) une par une.

    Montre comment la confiance en l'urne A monte a chaque tirage.
    """
    print("\n" + "=" * 60)
    print("  DEMO 1 : Urne — mise a jour sequentielle (4 billes rouges)")
    print("=" * 60)
    print("  Urne A : 70 % rouge | Urne B : 40 % rouge")
    print("  Hypothese H = 'la bille vient de l'urne A'")
    print()

    prior = 0.50
    # Chaque tirage d'une bille rouge est une preuve :
    # P(rouge|A) = 0.70, P(rouge|B) = 0.40
    preuves_rouges = [(0.70, 0.40)] * 4
    labels = [f"bille rouge #{i+1}" for i in range(4)]

    mise_a_jour_sequentielle(prior, preuves_rouges, labels)

    print()
    print("  Interpretation : chaque bille rouge renforce la these 'urne A'.")
    print("  Apres 4 billes rouges, la confiance depasse 90 %.")


# ---------------------------------------------------------------------------
# Demo 2 : lumiere oubliee — 3 hypotheses
# ---------------------------------------------------------------------------

def bayes_3_hypotheses(priors: list[float],
                        vraisemblances: list[float]) -> list[float]:
    """
    Mise a jour bayesienne pour 3 hypotheses mutuellement exclusives et exhaustives.

    Parametres
    ----------
    priors        : [P(H1), P(H2), P(H3)] — doivent sommer a 1
    vraisemblances: [P(E|H1), P(E|H2), P(E|H3)]

    Retourne
    --------
    list[float] : [P(H1|E), P(H2|E), P(H3|E)]
    """
    # Probabilite totale de la preuve (denominateur)
    p_e = sum(p * lh for p, lh in zip(priors, vraisemblances))

    if p_e == 0.0:
        raise ValueError("P(E) = 0 : aucune hypothese ne peut produire cette preuve.")

    # Posterior de chaque hypothese : P(Hi|E) = P(E|Hi)*P(Hi) / P(E)
    return [(lh * p) / p_e for p, lh in zip(priors, vraisemblances)]


def demo_lumiere():
    """
    Trois colocataires A, B, C. La lumiere du salon est restee allumee.
    On accumule deux preuves successives.

    Preuve 1 : A dormait chez ses parents (P(lumiere_allumee|A) tres faible).
    Preuve 2 : C a eteint sa lampe de chevet a 22h (indice neutre d'heure de coucher).
    """
    print("\n" + "=" * 60)
    print("  DEMO 2 : Lumiere oubliee — 3 colocataires, 2 preuves")
    print("=" * 60)
    print("  Hypotheses : H_A='A a oublie', H_B='B a oublie', H_C='C a oublie'")

    # Prior uniforme : on ne sait pas qui a ete le dernier
    priors = [1/3, 1/3, 1/3]
    noms = ["A", "B", "C"]

    print(f"\n  Prior initial : {', '.join(f'P(H_{n})={p:.3f}' for n, p in zip(noms, priors))}")

    # Preuve 1 : A a passe la nuit chez ses parents.
    # P(lumiere allumee | A) est quasi-nulle (il n'etait pas la)
    # P(lumiere allumee | B) = 0.45 (plausible)
    # P(lumiere allumee | C) = 0.45 (plausible)
    vr1 = [0.02, 0.49, 0.49]
    posteriors1 = bayes_3_hypotheses(priors, vr1)
    print(f"\n  Apres preuve 1 (A absent cette nuit) :")
    for n, p in zip(noms, posteriors1):
        print(f"    P(H_{n}|E1) = {p*100:.1f} %")

    # Preuve 2 : C a envoye un message a 23h depuis son telephone
    # (donc il etait eveille — un peu plus susceptible d'etre le dernier debout)
    # P(message 23h | B) = 0.30 (B se couche tot)
    # P(message 23h | C) = 0.70 (C etait eveille)
    vr2 = [0.01, 0.30, 0.69]  # A toujours quasiment exclu
    posteriors2 = bayes_3_hypotheses(posteriors1, vr2)
    print(f"\n  Apres preuve 2 (C a envoye un message a 23h) :")
    for n, p in zip(noms, posteriors2):
        print(f"    P(H_{n}|E1,E2) = {p*100:.1f} %")

    print()
    print("  Interpretation : les deux preuves accumulent pour designer C.")
    print("  Le raisonnement reste probabiliste, pas une certitude.")


# ---------------------------------------------------------------------------
# Demo 3 : controle qualite — machine defectueuse
# ---------------------------------------------------------------------------

def demo_controle_qualite():
    """
    Une chaine de production a deux machines M1 et M2.
    M1 produit 60 % des pieces, M2 produit 40 %.
    M1 a un taux de defaut de 3 %, M2 a un taux de defaut de 8 %.
    On tire une piece defectueuse. De quelle machine vient-elle ?

    Ensuite : on tire une 2eme piece defectueuse de la meme machine (inconnue).
    Mise a jour sequentielle.
    """
    print("\n" + "=" * 60)
    print("  DEMO 3 : Controle qualite — piece defectueuse")
    print("=" * 60)
    print("  M1 : 60 % de la production, 3 % de defauts")
    print("  M2 : 40 % de la production, 8 % de defauts")
    print("  Hypothese H = 'la piece vient de M1'")
    print()

    # Prior : proportion de production de M1
    prior = 0.60

    # Preuve : la piece est defectueuse
    # P(defaut | M1) = 0.03
    # P(defaut | M2) = 0.08
    preuves = [
        (0.03, 0.08),   # 1ere piece defectueuse
        (0.03, 0.08),   # 2eme piece defectueuse (tirage independant)
        (0.03, 0.08),   # 3eme piece defectueuse
    ]
    labels = [f"defaut #{i+1}" for i in range(3)]

    print("  Mise a jour apres chaque piece defectueuse :")
    posteriors = mise_a_jour_sequentielle(prior, preuves, labels)

    print()
    lr = rapport_de_vraisemblance(0.03, 0.08)
    print(f"  LR(defaut) = P(defaut|M1)/P(defaut|M2) = {lr:.3f}")
    print("  LR < 1 : un defaut est plus probable pour M2.")
    print("  => Chaque piece defectueuse deplace la credence vers M2 (P(H=M1) baisse).")


# ---------------------------------------------------------------------------
# Demo 4 : forme odds — verification croisee
# ---------------------------------------------------------------------------

def demo_forme_odds():
    """
    Verifie que la forme probabiliste et la forme odds donnent
    le meme resultat sur l'exemple du controle qualite.
    """
    print("\n" + "=" * 60)
    print("  DEMO 4 : Equivalence forme probabiliste et forme odds")
    print("=" * 60)

    prior = 0.60
    lh, lnh = 0.03, 0.08
    lr = rapport_de_vraisemblance(lh, lnh)

    # Forme probabiliste
    post_prob = bayes_posterior(prior, lh, lnh)

    # Forme odds
    odds_prieur = prob_vers_odds(prior)
    odds_posterieur = odds_prieur * lr
    post_odds = odds_vers_prob(odds_posterieur)

    print(f"\n  Prior P(M1)       = {prior*100:.1f} %")
    print(f"  LR(defaut)        = {lr:.4f}")
    print()
    print(f"  Via forme probabiliste : P(M1|defaut) = {post_prob*100:.4f} %")
    print(f"  Via forme odds         : P(M1|defaut) = {post_odds*100:.4f} %")

    diff = abs(post_prob - post_odds)
    assert diff < 1e-10, f"Ecart inattendu : {diff}"
    print(f"\n  Les deux formes sont equivalentes (ecart = {diff:.2e}).")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nModule 04 — Pensee bayesienne : mise a jour sequentielle")
    print("Formule : P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|nH)*(1-P(H))]")
    print("Principe : posterior_i = prior_{i+1} (chaque preuve s'accumule)")

    demo_urne_sequentielle()
    demo_lumiere()
    demo_controle_qualite()
    demo_forme_odds()

    print("\n" + "=" * 60)
    print("  Fin des demonstrations — toutes les verifications passent.")
    print("=" * 60)

# Exit 0 implicite (aucune exception non geree).

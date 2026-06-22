"""
Module 07 — Décision sous incertitude
======================================
Démo stdlib : calcul d'espérance mathématique + utilité espérée + arbre de décision.

Concepts illustrés :
  - Espérance mathématique  E[X] = Σ pᵢ × xᵢ
  - Utilité espérée          EU   = Σ pᵢ × U(xᵢ)
  - Arbre de décision jouet  (choix d'assurance illustratif)
  - Aversion au risque       comparaison U linéaire vs U racine carrée

Aucune dépendance externe — stdlib uniquement.
"""

import math


# ---------------------------------------------------------------------------
# 1. Espérance mathématique
# ---------------------------------------------------------------------------

def esperance(outcomes: list[tuple[float, float]]) -> float:
    """Calcule E[X] = Σ pᵢ × xᵢ.

    Args:
        outcomes: liste de (probabilité, valeur)

    Returns:
        Espérance mathématique (float).
    """
    # Vérification : les probabilités doivent sommer à 1 (tolérance flottante)
    total_p = sum(p for p, _ in outcomes)
    assert abs(total_p - 1.0) < 1e-9, f"Somme des probabilités = {total_p} ≠ 1"
    return sum(p * x for p, x in outcomes)


# ---------------------------------------------------------------------------
# 2. Utilité espérée avec différents profils
# ---------------------------------------------------------------------------

def utilite_esperee(
    outcomes: list[tuple[float, float]],
    util_fn=None,
) -> float:
    """Calcule EU = Σ pᵢ × U(xᵢ).

    Args:
        outcomes : liste de (probabilité, valeur monétaire)
        util_fn  : fonction U(x) → utilité. Défaut = neutre au risque U(x)=x.

    Returns:
        Utilité espérée (float).
    """
    if util_fn is None:
        util_fn = lambda x: x          # neutre au risque
    return sum(p * util_fn(x) for p, x in outcomes)


def u_neutre(x: float) -> float:
    """U(x) = x  — profil neutre au risque."""
    return x


def u_averse(x: float) -> float:
    """U(x) = √x  — profil averse au risque (concave, valeurs ≥ 0)."""
    # Pour les valeurs négatives on utilise -√(-x) pour conserver la monotonie
    if x >= 0:
        return math.sqrt(x)
    return -math.sqrt(-x)


def u_chercheur(x: float) -> float:
    """U(x) = x²  — profil chercheur de risque (convexe)."""
    return x ** 2


# ---------------------------------------------------------------------------
# 3. Arbre de décision — choix d'assurance illustratif
# ---------------------------------------------------------------------------

def noeud_hasard(branches: list[tuple[float, float]]) -> float:
    """Valeur d'un nœud de hasard = espérance de ses branches.

    Args:
        branches: liste de (probabilité, valeur_terminale)
    """
    return esperance(branches)


def resoudre_arbre_assurance(
    valeur_bien: float = 500.0,
    prime_annuelle: float = 45.0,
    proba_sinistre: float = 0.10,
) -> dict:
    """
    Arbre de décision — vélo ou objet de valeur 'valeur_bien' €.

    Nœud racine (décision) :
    ├── Option A : Sans assurance
    │   ○ Nœud de hasard
    │   ├── Sinistre (vol/perte) avec proba p  → −valeur_bien €
    │   └── Pas de sinistre         avec proba (1−p) → 0 €
    │
    └── Option B : Avec assurance
        Coût certain = −prime_annuelle €

    Retourne un dict avec les espérances de chaque option.
    """
    # Option A : sans assurance
    branches_sans = [
        (proba_sinistre,       -valeur_bien),
        (1 - proba_sinistre,    0.0),
    ]
    e_sans = noeud_hasard(branches_sans)

    # Option B : avec assurance (coût certain)
    e_avec = -prime_annuelle

    # Décision : choisir l'option à espérance maximale (moins négative = moins coûteuse)
    choix = "sans assurance" if e_sans > e_avec else "avec assurance"

    return {
        "E[sans assurance]": round(e_sans, 2),
        "E[avec assurance]": round(e_avec, 2),
        "choix_esperance_max": choix,
    }


# ---------------------------------------------------------------------------
# 4. Comparaison utilité espérée selon le profil de risque
# ---------------------------------------------------------------------------

def comparer_options_utilite(
    option_a: list[tuple[float, float]],
    option_b: list[tuple[float, float]],
    label_a: str = "Option A",
    label_b: str = "Option B",
) -> None:
    """Affiche l'espérance monétaire et l'utilité espérée (3 profils) pour deux options."""
    profils = [
        ("Neutre au risque  (U=x)", u_neutre),
        ("Averse au risque  (U=√x)", u_averse),
        ("Chercheur de risque (U=x²)", u_chercheur),
    ]

    print(f"\n  {'':30s} {label_a:>22s}   {label_b:>22s}")
    print(f"  {'Espérance monétaire':30s} {esperance(option_a):>22.2f}   {esperance(option_b):>22.2f}")
    for nom_profil, fn in profils:
        eu_a = utilite_esperee(option_a, fn)
        eu_b = utilite_esperee(option_b, fn)
        preferee = label_a if eu_a >= eu_b else label_b
        print(f"  {nom_profil:30s} EU={eu_a:>10.3f}           EU={eu_b:>10.3f}  → {preferee}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("MODULE 07 — Décision sous incertitude")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Exemple 1 : pari simple — dé à 6 faces
    # ------------------------------------------------------------------
    print("\n--- 1. Pari au dé (gain 12 € sur 6, perte 2 € sinon) ---")
    de = [
        (1 / 6,   12.0),   # face 6
        (5 / 6,   -2.0),   # toute autre face
    ]
    e_de = esperance(de)
    print(f"  Espérance = {e_de:.4f} €  ({'favorable' if e_de > 0 else 'défavorable'})")

    # ------------------------------------------------------------------
    # Exemple 2 : pari équitable — pile ou face
    # ------------------------------------------------------------------
    print("\n--- 2. Pari équitable — pile ou face (±5 €) ---")
    pile_face = [
        (0.5,  5.0),
        (0.5, -5.0),
    ]
    print(f"  Espérance = {esperance(pile_face):.2f} €  (équitable)")

    # ------------------------------------------------------------------
    # Exemple 3 : comparaison de deux options sous différentes utilités
    # ------------------------------------------------------------------
    print("\n--- 3. Option certaine vs pari (aversion au risque) ---")
    # Option A : 300 € certains
    option_a = [(1.0, 300.0)]
    # Option B : 50 % de 700 €, 50 % de 0 €  → E = 350 €
    option_b = [(0.5, 700.0), (0.5, 0.0)]
    comparer_options_utilite(option_a, option_b, "A (300 € certains)", "B (50%×700 €)")
    print()
    print("  → L'individu neutre choisit B (espérance > A).")
    print("  → L'individu averse au risque peut préférer A (EU[A] > EU[B] avec U=√x).")

    # ------------------------------------------------------------------
    # Exemple 4 : arbre de décision — assurance
    # ------------------------------------------------------------------
    print("\n--- 4. Arbre de décision — choix d'assurance illustratif ---")
    print("  Objet : 500 € | Prime : 45 €/an | P(sinistre) = 10 %")
    resultat = resoudre_arbre_assurance(
        valeur_bien=500.0,
        prime_annuelle=45.0,
        proba_sinistre=0.10,
    )
    for cle, val in resultat.items():
        print(f"  {cle:30s} : {val}")

    # ------------------------------------------------------------------
    # Exemple 5 : arbre — paramètres modifiés (prime plus élevée)
    # ------------------------------------------------------------------
    print("\n--- 5. Même arbre avec prime plus élevée (60 €) ---")
    print("  Objet : 500 € | Prime : 60 €/an | P(sinistre) = 10 %")
    resultat2 = resoudre_arbre_assurance(
        valeur_bien=500.0,
        prime_annuelle=60.0,
        proba_sinistre=0.10,
    )
    for cle, val in resultat2.items():
        print(f"  {cle:30s} : {val}")
    print("  → Quand la prime > espérance du sinistre, l'espérance seule dit 'non'.")
    print("    Mais l'aversion au risque peut quand même justifier de payer.")

    # ------------------------------------------------------------------
    # Rappel paradoxes
    # ------------------------------------------------------------------
    print("\n--- 6. Rappel — Paradoxes d'Allais & Ellsberg ---")
    print("  Allais  : la certitude a une valeur spéciale → viole l'axiome d'indépendance.")
    print("  Ellsberg: risque (p connu) ≠ ambiguïté (p inconnu) → aversion à l'ambiguïté.")

    print("\n" + "=" * 65)
    print("Fin de la démo. Sortie normale (exit 0).")
    print("=" * 65)

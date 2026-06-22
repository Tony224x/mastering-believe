"""
Module 03 — Probabilités en fréquences naturelles
===================================================
Calcule les valeurs prédictives (VPP et VPN) à partir d'un tableau de
fréquences naturelles (méthode Gigerenzer). Entrées :
  - sensibilite : taux de vrais positifs parmi les malades  [0–1]
  - specificite  : taux de vrais négatifs parmi les sains    [0–1]
  - prevalence   : taux de base de la maladie dans la pop.   [0–1]
  - population   : taille de la cohorte virtuelle (défaut 1 000)

Aucune formule de Bayes n'est utilisée ici — uniquement des effectifs.
Le Module 04 introduira la notation bayésienne formelle.

Réutilise et étend 02-probabilites-utiles.py (même noyau calculer_vpp_vpn,
nouveaux scénarios pédagogiques centrés sur les fréquences naturelles).

Utilisation : python 03-probabilites-frequences.py
Stdlib pur — aucune dépendance externe.
"""


# ---------------------------------------------------------------------------
# Noyau de calcul : fréquences naturelles → tableau 2×2 → VPP / VPN
# ---------------------------------------------------------------------------

def calculer_frequences(
    sensibilite: float,
    specificite: float,
    prevalence: float,
    population: int = 1_000,
) -> dict:
    """
    Construit le tableau des fréquences naturelles et dérive VPP et VPN.

    Paramètres
    ----------
    sensibilite : float
        P(test+ | malade) — proportion de malades correctement détectés.
        Ex. 0.90 → 90 % des malades ont un test positif.
    specificite : float
        P(test- | sain) — proportion de sains correctement rejetés.
        Ex. 0.95 → 95 % des sains ont un test négatif.
    prevalence : float
        Proportion de la population réellement malade (taux de base).
        Ex. 0.01 → 1 % de malades dans la population.
    population : int
        Taille de la cohorte fictive pour les effectifs (défaut 1 000).

    Retourne
    --------
    dict avec les effectifs (vp, fp, fn, vn), les totaux et les valeurs
    prédictives (vpp, vpn).
    """
    # Validation des plages
    for nom, val in [
        ("sensibilite", sensibilite),
        ("specificite", specificite),
        ("prevalence", prevalence),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{nom} doit être entre 0 et 1, reçu : {val}")

    # --- Étape 1 : répartir la population selon le taux de base ---
    n_malades = population * prevalence          # effectif réellement malade
    n_sains   = population * (1 - prevalence)   # effectif réellement sain

    # --- Étape 2 : appliquer la sensibilité aux malades ---
    vp = n_malades * sensibilite          # vrais positifs  (malades détectés)
    fn = n_malades * (1 - sensibilite)    # faux négatifs   (malades ratés)

    # --- Étape 3 : appliquer la spécificité aux sains ---
    vn = n_sains * specificite            # vrais négatifs  (sains rejetés)
    fp = n_sains * (1 - specificite)      # faux positifs   (alarmes inutiles)

    # --- Étape 4 : lire les valeurs prédictives dans le tableau ---
    total_positifs = vp + fp
    total_negatifs = vn + fn

    # VPP : sur tous les tests positifs, quelle fraction est vraiment malade ?
    vpp = vp / total_positifs if total_positifs > 0 else 0.0

    # VPN : sur tous les tests négatifs, quelle fraction est vraiment saine ?
    vpn = vn / total_negatifs if total_negatifs > 0 else 0.0

    return {
        "population"     : population,
        "n_malades"      : n_malades,
        "n_sains"        : n_sains,
        "vp"             : vp,
        "fp"             : fp,
        "fn"             : fn,
        "vn"             : vn,
        "total_positifs" : total_positifs,
        "total_negatifs" : total_negatifs,
        "vpp"            : vpp,
        "vpn"            : vpn,
    }


# ---------------------------------------------------------------------------
# Affichage lisible du tableau des fréquences naturelles
# ---------------------------------------------------------------------------

def afficher_tableau(
    label: str,
    sensibilite: float,
    specificite: float,
    prevalence: float,
    population: int = 1_000,
) -> None:
    """Affiche le tableau 2×2 et les valeurs prédictives pour un scénario."""
    r = calculer_frequences(sensibilite, specificite, prevalence, population)

    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Taux de base (prévalence) : {prevalence*100:.2f} %")
    print(f"  Sensibilité               : {sensibilite*100:.1f} %")
    print(f"  Spécificité               : {specificite*100:.1f} %")
    print(f"  Population fictive        : {population:,}")
    print()

    # En-tête du tableau
    col = 12
    print(f"  {'':22s} {'Malade':>{col}} {'Sain':>{col}} {'Total':>{col}}")
    print(f"  {'-'*60}")
    print(
        f"  {'Test positif':22s}"
        f" {r['vp']:>{col}.1f}"
        f" {r['fp']:>{col}.1f}"
        f" {r['total_positifs']:>{col}.1f}"
    )
    print(
        f"  {'Test négatif':22s}"
        f" {r['fn']:>{col}.1f}"
        f" {r['vn']:>{col}.1f}"
        f" {r['total_negatifs']:>{col}.1f}"
    )
    print(f"  {'-'*60}")
    print(
        f"  {'Total':22s}"
        f" {r['n_malades']:>{col}.1f}"
        f" {r['n_sains']:>{col}.1f}"
        f" {r['population']:>{col},}"
    )
    print()

    # Valeurs prédictives
    print(f"  VPP (test+ → vraiment malade) : {r['vpp']*100:.1f} %")
    print(f"  VPN (test- → vraiment sain)   : {r['vpn']*100:.1f} %")

    # Alerte si la majorité des positifs sont des faux positifs
    if r["vpp"] < 0.5:
        fp_pct = (r["fp"] / r["total_positifs"] * 100) if r["total_positifs"] > 0 else 0
        print(
            f"\n  >> Attention : {fp_pct:.0f} % des tests positifs sont des FAUX positifs."
        )
        print(f"     Effet du faible taux de base — un test de confirmation est justifié.")


# ---------------------------------------------------------------------------
# Démonstration principale : impact du taux de base sur la VPP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Démonstration : fréquences naturelles et valeur prédictive positive")
    print("Méthode Gigerenzer — effectifs concrets sur 1 000 personnes")

    # --- Scénario 1 : dépistage médical, maladie rare ---
    # Contexte : dépistage systématique d'une maladie touchant 1 % de la pop.
    # Question pédagogique : pourquoi un résultat positif ne signifie pas 90 % de malades ?
    afficher_tableau(
        label      = "Scénario A — Dépistage, maladie rare (prévalence 1 %)",
        sensibilite = 0.90,
        specificite = 0.95,
        prevalence  = 0.01,
    )

    # --- Scénario 2 : même test, population à risque élevé ---
    # Contexte : consultation spécialisée, patients déjà sélectionnés (prévalence 20 %)
    # Montre comment le taux de base change tout — même test, autre interprétation
    afficher_tableau(
        label      = "Scénario B — Consultation spécialisée (prévalence 20 %)",
        sensibilite = 0.90,
        specificite = 0.95,
        prevalence  = 0.20,
    )

    # --- Scénario 3 : test quasi-parfait, maladie ultra-rare ---
    # Intuition contre-intuitive : même avec sensibilité et spécificité à 99 %,
    # une maladie à 0.1 % génère encore plus de faux positifs que de vrais positifs
    afficher_tableau(
        label      = "Scénario C — Test quasi-parfait, maladie ultra-rare (0.1 %)",
        sensibilite = 0.99,
        specificite = 0.99,
        prevalence  = 0.001,
        population  = 10_000,   # population plus grande pour avoir des effectifs lisibles
    )

    # --- Scénario 4 : contrôle qualité industriel ---
    # Contexte : chaîne de production, 2 % de pièces défectueuses
    # Montre que le raisonnement par fréquences s'applique hors médecine
    afficher_tableau(
        label      = "Scénario D — Contrôle qualité (prévalence 2 %)",
        sensibilite = 0.95,
        specificite = 0.90,
        prevalence  = 0.02,
    )

    # --- Synthèse comparative : même test, quatre taux de base ---
    print("\n" + "="*62)
    print("  Synthèse : VPP selon le taux de base")
    print("  (Test fixe : sensibilité 90 %, spécificité 95 %)")
    print("="*62)
    print(f"  {'Prévalence':>12}  {'VP':>6}  {'FP':>6}  {'Total+':>8}  {'VPP':>8}")
    print(f"  {'-'*50}")

    for prev in [0.001, 0.01, 0.05, 0.10, 0.30, 0.50]:
        r = calculer_frequences(0.90, 0.95, prev, population=10_000)
        print(
            f"  {prev*100:>11.1f} %"
            f"  {r['vp']:>6.0f}"
            f"  {r['fp']:>6.0f}"
            f"  {r['total_positifs']:>8.0f}"
            f"  {r['vpp']*100:>7.1f} %"
        )

    print()
    print("Lecture : quand la prévalence monte de 0.1 % à 50 %, la VPP passe")
    print("de ~2 % à ~95 % — avec exactement le même test.")
    print()
    print("Pour tester vos propres valeurs :")
    print("  from 03_probabilites_frequences import calculer_frequences")
    print("  r = calculer_frequences(sensibilite=0.80, specificite=0.90, prevalence=0.05)")
    print("  print(f\"VPP = {r['vpp']*100:.1f} %\")")

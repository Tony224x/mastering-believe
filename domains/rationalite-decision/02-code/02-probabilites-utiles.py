"""
Module 02 — Probabilites utiles : calculateur taux de base / faux positifs
===========================================================================
Calcule les valeurs predictives (VPP et VPN) a partir de :
  - la sensibilite du test (taux de vrais positifs parmi les malades)
  - la specificite du test (taux de vrais negatifs parmi les sains)
  - la prevalence (taux de base de la maladie dans la population)

Formules :
  VPP = VP / (VP + FP)  — probabilite d'etre malade sachant test positif
  VPN = VN / (VN + FN)  — probabilite d'etre sain sachant test negatif

Utilisation : python 02-probabilites-utiles.py
Stdlib pur — aucune dependance externe.
"""


def calculer_vpp_vpn(sensibilite: float, specificite: float, prevalence: float,
                     population: int = 10_000) -> dict:
    """
    Calcule VPP, VPN et le tableau des frequences naturelles.

    Parametres
    ----------
    sensibilite : float
        P(test+ | malade), entre 0 et 1.
        Ex : 0.90 pour 90 % de detection des vrais malades.
    specificite : float
        P(test- | sain), entre 0 et 1.
        Ex : 0.95 pour 95 % de rejet correct des sains.
    prevalence : float
        Proportion de la population reellement malade (taux de base), entre 0 et 1.
        Ex : 0.01 pour 1 % de prevalence.
    population : int
        Taille de la cohorte virtuelle pour les frequences naturelles (defaut 10 000).

    Retourne
    --------
    dict avec cles : vp, fp, fn, vn, vpp, vpn, taux_faux_positifs_parmi_positifs
    """
    # Validation basique des entrees
    for nom, val in [("sensibilite", sensibilite), ("specificite", specificite),
                     ("prevalence", prevalence)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{nom} doit etre entre 0 et 1, recu : {val}")

    # --- Effectifs de base ---
    n_malades = population * prevalence          # personnes reellement malades
    n_sains   = population * (1 - prevalence)   # personnes reellement saines

    # --- Tableau 2x2 ---
    # Vrais positifs : malades correctement detectes
    vp = n_malades * sensibilite
    # Faux negatifs : malades rates par le test
    fn = n_malades * (1 - sensibilite)
    # Vrais negatifs : sains correctement rejetes
    vn = n_sains * specificite
    # Faux positifs : sains incorrectement detectes (alarmes inutiles)
    fp = n_sains * (1 - specificite)

    # --- Valeurs predictives ---
    # VPP : parmi tous les tests positifs, quelle fraction est vraiment malade ?
    total_positifs = vp + fp
    vpp = vp / total_positifs if total_positifs > 0 else 0.0

    # VPN : parmi tous les tests negatifs, quelle fraction est vraiment saine ?
    total_negatifs = vn + fn
    vpn = vn / total_negatifs if total_negatifs > 0 else 0.0

    return {
        "population": population,
        "n_malades": n_malades,
        "n_sains": n_sains,
        "vp": vp,   # vrais positifs
        "fp": fp,   # faux positifs (alarmes inutiles)
        "fn": fn,   # faux negatifs (malades rates)
        "vn": vn,   # vrais negatifs
        "vpp": vpp,
        "vpn": vpn,
    }


def afficher_resultats(label: str, sensibilite: float, specificite: float,
                       prevalence: float, population: int = 10_000) -> None:
    """Affiche le tableau et les valeurs predictives pour un scenario donne."""
    r = calculer_vpp_vpn(sensibilite, specificite, prevalence, population)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sensibilite  : {sensibilite*100:.1f} %")
    print(f"  Specificite  : {specificite*100:.1f} %")
    print(f"  Prevalence   : {prevalence*100:.2f} %")
    print(f"  Population   : {population:,}")
    print()

    # Tableau des frequences naturelles
    print(f"  {'':20s} {'Malade':>10s} {'Sain':>10s} {'Total':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Test positif':20s} {r['vp']:>10.1f} {r['fp']:>10.1f} {r['vp']+r['fp']:>10.1f}")
    print(f"  {'Test negatif':20s} {r['fn']:>10.1f} {r['vn']:>10.1f} {r['fn']+r['vn']:>10.1f}")
    print(f"  {'-'*52}")
    print(f"  {'Total':20s} {r['n_malades']:>10.1f} {r['n_sains']:>10.1f} {population:>10,}")
    print()
    print(f"  VPP (test+ => vraiment malade) : {r['vpp']*100:.1f} %")
    print(f"  VPN (test- => vraiment sain)   : {r['vpn']*100:.1f} %")
    print()

    # Mise en evidence si la VPP est surprenante
    if r["vpp"] < 0.5:
        print(f"  >> Attention : VPP < 50 % — plus de la moitie des positifs")
        print(f"     sont des FAUX positifs (effet du faible taux de base).")


if __name__ == "__main__":
    print("\nDemonstration : impact du taux de base sur la valeur predictive")
    print("Test fixe : sensibilite = 95 %, specificite = 95 %\n")

    # --- Scenario 1 : depistage medical classique (maladie rare) ---
    afficher_resultats(
        label="Scenario A — Maladie rare (prevalence 1 %)",
        sensibilite=0.95,
        specificite=0.95,
        prevalence=0.01,
    )

    # --- Scenario 2 : maladie moderement repandue ---
    afficher_resultats(
        label="Scenario B — Maladie moderee (prevalence 10 %)",
        sensibilite=0.95,
        specificite=0.95,
        prevalence=0.10,
    )

    # --- Scenario 3 : maladie commune ---
    afficher_resultats(
        label="Scenario C — Maladie commune (prevalence 50 %)",
        sensibilite=0.95,
        specificite=0.95,
        prevalence=0.50,
    )

    # --- Scenario 4 : reprise de l'exemple du module (depistage 1 %) ---
    print("\n" + "="*60)
    print("  Exemple du module : sensibilite=90 %, specificite=95 %, prevalence=1 %")
    print("="*60)
    afficher_resultats(
        label="Test du module 02",
        sensibilite=0.90,
        specificite=0.95,
        prevalence=0.01,
    )

    # --- Scenario 5 : test de qualite industrielle (exercice 5 du module) ---
    afficher_resultats(
        label="Scenario D — Controle qualite (prevalence 2 %)",
        sensibilite=0.95,
        specificite=0.90,
        prevalence=0.02,
    )

    print("\nPour tester vos propres valeurs, importez calculer_vpp_vpn() :")
    print("  from 02-probabilites-utiles import calculer_vpp_vpn")
    print("  r = calculer_vpp_vpn(sensibilite=0.80, specificite=0.90, prevalence=0.05)")
    print("  print(f\"VPP = {r['vpp']*100:.1f} %\")")

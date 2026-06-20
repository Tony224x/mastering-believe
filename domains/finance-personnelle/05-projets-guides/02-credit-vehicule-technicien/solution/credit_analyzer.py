"""
Analyseur de credit — Projet guide 02 (Finance Personnelle)
===========================================================
Contexte : Sofia est technicienne terrain pour un integrateur FleetSim. Elle
se deplace entre les entrepots clients pour installer et calibrer les robots,
et a besoin d'un vehicule fiable (8 000 EUR). Elle hesite entre deux offres
de credit, se demande si elle ne devrait pas payer comptant, et traine par
ailleurs deux petites dettes. Ce script l'aide a decider chiffres en main.

Ce script est EDUCATIF uniquement. Il ne constitue pas un conseil financier
personnalise. Les taux sont illustratifs. Tout credit engage l'emprunteur ;
verifiez les conditions reelles (TAEG, assurance, frais) avant de signer.

Modules couverts : 03 (dette et credit), 01 (interets composes / cout d'opportunite).

Utilisation :
    python credit_analyzer.py

Dependances : stdlib uniquement (dataclasses). Aucune installation.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Mensualite et amortissement d'un pret
# ---------------------------------------------------------------------------

def mensualite(capital: float, taux_annuel: float, mois: int) -> float:
    """Mensualite constante d'un pret amortissable (formule fermee).

        M = P * r / (1 - (1 + r)^-n)

    avec r = taux mensuel (taux_annuel / 12) et n = nombre de mensualites.

    Cas limite r == 0 (pret a taux zero) : on rembourse simplement P / n,
    sinon la formule diviserait par zero.
    """
    r = taux_annuel / 12.0
    if r == 0:
        return capital / mois
    return capital * r / (1 - (1 + r) ** -mois)


@dataclass
class Offre:
    """Une offre de credit a la consommation."""

    nom: str
    capital: float
    taux_annuel: float   # TAEG approxime, en decimal (0.049 = 4,9 %)
    mois: int

    @property
    def mensualite(self) -> float:
        return mensualite(self.capital, self.taux_annuel, self.mois)

    @property
    def total_rembourse(self) -> float:
        return self.mensualite * self.mois

    @property
    def cout_du_credit(self) -> float:
        """Total des interets payes = ce que le credit coute reellement."""
        return self.total_rembourse - self.capital


def tableau_amortissement(offre: Offre, lignes_max: int = 6) -> list[tuple[int, float, float, float]]:
    """Retourne (mois, interet, principal, solde restant) pour les premiers mois.

    But pedagogique : montrer qu'au debut d'un credit, la mensualite paie
    surtout des INTERETS, et tres peu de capital. C'est contre-intuitif et
    explique pourquoi rembourser par anticipation tot fait gagner gros.
    """
    r = offre.taux_annuel / 12.0
    m = offre.mensualite
    solde = offre.capital
    rows = []
    for mois in range(1, min(lignes_max, offre.mois) + 1):
        interet = solde * r
        principal = m - interet
        solde -= principal
        rows.append((mois, interet, principal, max(solde, 0.0)))
    return rows


# ---------------------------------------------------------------------------
# 2. Comptant vs credit + placement de la difference (cout d'opportunite)
# ---------------------------------------------------------------------------

def comptant_vs_credit(
    offre: Offre,
    rendement_placement: float,
    epargne_disponible: float,
) -> dict[str, float]:
    """Compare deux strategies sur la duree du credit, a effort egal.

    Strategie A (comptant) : Sofia sort `capital` de son epargne maintenant.
       Elle n'a plus de mensualite, donc elle PLACE chaque mois l'equivalent
       de la mensualite qu'elle n'a pas a payer.
    Strategie B (credit)   : Sofia garde son epargne placee, et paie la
       mensualite avec son salaire.

    On compare le patrimoine net a la fin (placements - dette restante).
    Hypothese clef rendue explicite : le verdict depend entierement du signe
    de (rendement_placement - taux_credit). Si le placement rapporte plus que
    le credit ne coute, le credit "intelligent" gagne ; sinon le comptant gagne.
    """
    n = offre.mois
    r_place = rendement_placement / 12.0

    # Strategie A : payer comptant, puis placer la mensualite economisee.
    reste_epargne_A = epargne_disponible - offre.capital
    place_A = reste_epargne_A
    for _ in range(n):
        place_A = place_A * (1 + r_place) + offre.mensualite
    patrimoine_A = place_A  # aucune dette

    # Strategie B : garder l'epargne placee, payer la mensualite a part.
    # (La mensualite vient du salaire dans les deux cas -> effort egal.)
    place_B = epargne_disponible
    for _ in range(n):
        place_B = place_B * (1 + r_place)
    patrimoine_B = place_B  # dette entierement remboursee a la fin

    return {
        "patrimoine_comptant": patrimoine_A,
        "patrimoine_credit": patrimoine_B,
        "ecart": patrimoine_B - patrimoine_A,
    }


# ---------------------------------------------------------------------------
# 3. Avalanche vs boule de neige (plusieurs dettes, budget fixe)
# ---------------------------------------------------------------------------

@dataclass
class Dette:
    nom: str
    solde: float
    taux_annuel: float
    mensualite_min: float


def rembourser(dettes: list[Dette], budget_total: float, strategie: str) -> dict[str, float]:
    """Simule le remboursement de plusieurs dettes avec un budget mensuel fixe.

    - "avalanche"  : on concentre le surplus sur la dette au TAUX le plus eleve
                     (mathematiquement optimal : minimise les interets).
    - "boule"      : on concentre le surplus sur le plus PETIT SOLDE
                     (boule de neige : moins optimal en interets, mais effet
                     psychologique de "victoires rapides" — cf. module 03/05).

    Retourne le nombre de mois et le total d'interets payes.
    """
    # Copie de travail (on ne mute pas les dettes d'entree).
    actives = [Dette(d.nom, d.solde, d.taux_annuel, d.mensualite_min) for d in dettes]
    total_interets = 0.0
    mois = 0
    garde_fou = 1000  # evite toute boucle infinie si budget < interets

    while actives and mois < garde_fou:
        mois += 1
        # 1) Interets du mois sur chaque dette.
        for d in actives:
            interet = d.solde * d.taux_annuel / 12.0
            d.solde += interet
            total_interets += interet

        # 2) Paiements minimums obligatoires.
        budget_restant = budget_total
        for d in actives:
            paye = min(d.mensualite_min, d.solde)
            d.solde -= paye
            budget_restant -= paye

        # 3) Surplus dirige selon la strategie, sur la dette cible.
        if budget_restant > 0 and actives:
            if strategie == "avalanche":
                cible = max(actives, key=lambda d: d.taux_annuel)
            else:  # boule de neige
                cible = min(actives, key=lambda d: d.solde)
            paye = min(budget_restant, cible.solde)
            cible.solde -= paye

        # 4) On retire les dettes soldees.
        actives = [d for d in actives if d.solde > 0.005]

    return {"mois": float(mois), "total_interets": total_interets}


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def euros(montant: float) -> str:
    return f"{montant:,.0f} EUR".replace(",", " ")


def euros2(montant: float) -> str:
    return f"{montant:,.2f} EUR".replace(",", " ")


def titre(t: str) -> None:
    print()
    print("=" * 64)
    print(f"  {t}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_offres() -> tuple[Offre, Offre]:
    """Deux offres pour financer le vehicule de Sofia (8 000 EUR)."""
    titre("1. Deux offres de credit pour le vehicule (8 000 EUR)")

    offre_a = Offre("Offre A — 4,9 % / 36 mois", 8_000, 0.049, 36)
    offre_b = Offre("Offre B — 4,5 % / 72 mois", 8_000, 0.045, 72)

    print(f"  {'Offre':30} {'Mensualite':>12} {'Cout credit':>14} {'Total':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*14} {'-'*12}")
    for o in (offre_a, offre_b):
        print(f"  {o.nom:30} {euros2(o.mensualite):>12} "
              f"{euros(o.cout_du_credit):>14} {euros(o.total_rembourse):>12}")

    print()
    print("  Piege classique : l'offre B a une mensualite presque deux fois plus")
    print("  FAIBLE (elle rassure le budget mensuel) mais coute pres de 2x plus")
    print("  cher AU TOTAL, car on paie des interets sur 72 mois au lieu de 36 —")
    print("  malgre un taux affiche plus bas. 'Petite mensualite' n'est pas")
    print("  synonyme de 'pas cher'.")
    return offre_a, offre_b


def demo_amortissement(offre: Offre) -> None:
    titre(f"2. Amortissement — {offre.nom}")
    print(f"  {'Mois':>5} {'Interet':>12} {'Capital':>12} {'Solde restant':>16}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*16}")
    for mois, interet, principal, solde in tableau_amortissement(offre):
        print(f"  {mois:>5} {euros2(interet):>12} {euros2(principal):>12} {euros(solde):>16}")
    print()
    print("  Lecture : la 1re mensualite est surtout des interets. Rembourser")
    print("  par anticipation tot attaque directement le capital -> gros gain.")


def demo_comptant_vs_credit(offre: Offre) -> None:
    titre("3. Payer comptant ou garder l'epargne placee ?")
    epargne = 15_000  # Sofia a 15 000 EUR de cote

    for rendement in (0.02, 0.07):
        res = comptant_vs_credit(offre, rendement, epargne)
        gagnant = "CREDIT" if res["ecart"] > 0 else "COMPTANT"
        print()
        print(f"  Si le placement rapporte {rendement*100:.0f} % / an "
              f"(credit a {offre.taux_annuel*100:.1f} %) :")
        print(f"    Patrimoine net si comptant : {euros(res['patrimoine_comptant'])}")
        print(f"    Patrimoine net si credit   : {euros(res['patrimoine_credit'])}")
        print(f"    -> Avantage {gagnant} de {euros(abs(res['ecart']))}")
    print()
    print("  Regle : le credit ne 'rapporte' que si le placement bat son taux,")
    print(f"  net d'impot et de risque. A 2 % vs credit a {offre.taux_annuel*100:.1f} %, le comptant")
    print("  gagne. (Et payer comptant est un rendement SANS risque ; le placement, non.)")


def demo_avalanche_vs_boule() -> None:
    titre("4. Avalanche vs boule de neige (3 dettes, budget 400 EUR/mois)")

    dettes = [
        Dette("Carte de credit", 1_500, 0.18, 45),       # petit solde, taux tres eleve
        Dette("Credit telephone", 600, 0.05, 30),         # tout petit solde, taux faible
        Dette("Pret etudiant", 4_000, 0.03, 120),         # gros solde, taux faible
    ]
    budget = 400.0

    res_av = rembourser(dettes, budget, "avalanche")
    res_bo = rembourser(dettes, budget, "boule")

    print(f"  {'Strategie':24} {'Duree':>10} {'Total interets':>16}")
    print(f"  {'-'*24} {'-'*10} {'-'*16}")
    print(f"  {'Avalanche (taux max)':24} {int(res_av['mois']):>7} mois {euros2(res_av['total_interets']):>16}")
    print(f"  {'Boule de neige (petit)':24} {int(res_bo['mois']):>7} mois {euros2(res_bo['total_interets']):>16}")

    economie = res_bo["total_interets"] - res_av["total_interets"]
    print()
    print(f"  L'avalanche economise {euros2(economie)} d'interets ici (elle tue")
    print("  d'abord la carte a 18 %). La boule de neige solde le credit telephone")
    print("  en premier : moins optimal, mais une 'victoire' rapide qui motive.")
    print("  Choix honnete : avalanche si tu tiens sur la duree, boule si tu as")
    print("  besoin de momentum pour ne pas abandonner (module 05 : le facteur humain).")


def main() -> None:
    print()
    print("=" * 64)
    print("  ANALYSEUR DE CREDIT — VEHICULE D'UNE TECHNICIENNE TERRAIN")
    print("  Finance Personnelle / Projet guide 02")
    print("=" * 64)
    print()
    print("  AVERTISSEMENT : script EDUCATIF. Pas un conseil financier.")
    print("  Verifiez TAEG, assurance et frais reels avant de signer.")

    offre_a, offre_b = demo_offres()
    demo_amortissement(offre_a)
    demo_comptant_vs_credit(offre_b)
    demo_avalanche_vs_boule()

    print()
    print("=" * 64)
    print("  Fin. Edite les Offre(...) et Dette(...) pour ta situation.")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()

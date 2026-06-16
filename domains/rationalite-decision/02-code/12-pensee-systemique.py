"""
Module 12 — Pensée systémique
Simulation de boucles de rétroaction avec stocks et flux (stdlib pur).

Deux démonstrations :
  A) Thermostat — boucle équilibrante (E)
  B) File d'attente logistique — boucle renforçante (R) puis saturation

Aucune dépendance externe — stdlib Python 3.8+.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Utilitaires d'affichage
# ---------------------------------------------------------------------------

def barre(valeur: float, maximum: float, largeur: int = 40, symbole: str = "█") -> str:
    """Retourne une barre proportionnelle pour affichage terminal."""
    remplissage = int(round(largeur * min(valeur, maximum) / maximum))
    return symbole * remplissage + "░" * (largeur - remplissage)


def ligne(etiquette: str, valeur: float, maximum: float = 100.0) -> None:
    print(f"  {etiquette:30s} {valeur:6.1f}  [{barre(valeur, maximum)}]")


# ---------------------------------------------------------------------------
# A) Thermostat — boucle équilibrante avec délai
# ---------------------------------------------------------------------------

def simuler_thermostat(
    temp_initiale: float = 15.0,   # °C au démarrage
    consigne: float = 20.0,         # objectif du thermostat
    gain: float = 0.4,              # sensibilité de la boucle E (0 < gain ≤ 1)
    delai_chaudiere: int = 2,       # nombre de pas avant que la chaleur arrive
    n_pas: int = 30,
) -> None:
    """
    Modèle discret d'un thermostat.

    À chaque pas de temps (≈ 1 minute) :
      - flux_chauffage = gain × max(écart, 0)  [chaudière ne refroidit pas]
      - La chaleur produite arrive DELAI_CHAUDIERE pas plus tard (boucle E avec délai).
    """
    print("\n" + "=" * 60)
    print("A) THERMOSTAT — boucle équilibrante (E) avec délai")
    print(f"   Consigne={consigne}°C | Temp. initiale={temp_initiale}°C | "
          f"Gain={gain} | Délai chaudière={delai_chaudiere} pas")
    print("=" * 60)

    temp = temp_initiale
    # File de chaleur en transit (délai de chaudière)
    chaleur_en_route: list[float] = [0.0] * (delai_chaudiere + 1)

    for t in range(n_pas):
        ecart = consigne - temp                      # négatif si trop chaud
        flux_demande = gain * max(ecart, 0.0)        # boucle E : correction proportionnelle

        # Enqueue la chaleur produite (arrivera dans DELAI_CHAUDIERE pas)
        chaleur_en_route.append(flux_demande)
        chaleur_arrivee = chaleur_en_route.pop(0)    # chaleur commandée il y a DELAI pas

        temp = temp + chaleur_arrivee                # mise à jour du stock (température)

        if t % 3 == 0 or t < 5:                      # affichage allégé
            print(f"  Pas {t:02d} | Temp={temp:5.2f}°C | Écart={ecart:+.2f} | "
                  f"Flux demandé={flux_demande:.2f} | Arrivé={chaleur_arrivee:.2f}")

    print(f"\n  → Température finale : {temp:.2f}°C (consigne : {consigne}°C)")
    print("  → La boucle E converge, avec oscillations dues au délai si gain trop élevé.\n")


# ---------------------------------------------------------------------------
# B) File d'attente logistique — boucle renforçante + saturation naturelle
# ---------------------------------------------------------------------------

def simuler_file_attente(
    taux_arrivee_base: float = 10.0,   # commandes/heure en temps normal
    capacite_traitement: float = 12.0, # max commandes/heure (capacité)
    n_heures: int = 24,
    surge_debut: int = 4,              # heure de début d'un pic de demande
    surge_fin: int = 8,                # heure de fin du pic
    surge_facteur: float = 2.0,        # multiplicateur de la demande pendant le pic
) -> None:
    """
    Modèle discret d'une file d'attente logistique.

    Boucle renforçante : plus la file est longue, plus les agents sont ralentis
    (surcharge cognitive → temps de traitement allongé → file grandit encore).
    Saturation naturelle : capacité bornée.

    Stock = commandes en attente dans la file.
    Flux entrant = arrivées (externe, avec pic).
    Flux sortant = traitements (capacité réduite quand file > seuil de surcharge).
    """
    print("=" * 60)
    print("B) FILE D'ATTENTE LOGISTIQUE — boucle R + saturation")
    print(f"   Taux normal={taux_arrivee_base}/h | Capacité={capacite_traitement}/h | "
          f"Pic ×{surge_facteur} de h{surge_debut} à h{surge_fin}")
    print("=" * 60)

    file: float = 0.0          # stock initial : file vide
    total_traite: float = 0.0
    total_arrive: float = 0.0

    SEUIL_SURCHARGE = 20.0     # commandes : au-delà, les agents ralentissent
    PENALITE_SURCHARGE = 0.6   # facteur de réduction du débit sous surcharge

    print(f"\n  {'Heure':>5} | {'File':>6} | {'Arrivées':>9} | {'Traités':>8} | Statut")
    print("  " + "-" * 52)

    for h in range(n_heures):
        # --- Flux entrant : arrivées de commandes ---
        if surge_debut <= h < surge_fin:
            arrivees = taux_arrivee_base * surge_facteur   # pic de demande
        else:
            arrivees = taux_arrivee_base

        # --- Flux sortant : traitement (réduit si surcharge = boucle R) ---
        if file > SEUIL_SURCHARGE:
            # Boucle renforçante : surcharge → ralentissement → file grossit encore
            debit_reel = capacite_traitement * PENALITE_SURCHARGE
            statut = "⚠ SURCHARGE"
        else:
            debit_reel = min(capacite_traitement, file + arrivees)
            statut = "OK"

        # --- Mise à jour du stock ---
        file = max(0.0, file + arrivees - debit_reel)
        total_arrivee_h = arrivees
        total_traite += debit_reel
        total_arrive += arrivees

        # Affichage barre pour la file (max = 80 pour lisibilité)
        barre_file = barre(file, 80.0, largeur=20)
        print(f"  h{h:02d}   | {file:6.1f} | {arrivees:9.1f} | {debit_reel:8.1f} | "
              f"{barre_file} {statut}")

    print(f"\n  → Commandes arrivées : {total_arrive:.0f}")
    print(f"  → Commandes traitées : {total_traite:.0f}")
    print(f"  → Stock final en file : {file:.1f}")
    print("  → La boucle R amplifie la congestion ; la saturation la borne.\n")


# ---------------------------------------------------------------------------
# C) Effets de second ordre — simulation de remise commerciale
# ---------------------------------------------------------------------------

def simuler_remise_commerciale(
    stock_initial: float = 500.0,   # unités en entrepôt
    demande_normale: float = 50.0,  # unités/jour
    remise_jour: int = 3,            # jour où la remise est déclenchée
    boost_demande: float = 2.5,      # pic de demande (×2.5 pendant 2 jours)
    delai_reappro: int = 5,          # jours pour réapprovisionner
    capacite_reappro: float = 80.0,  # unités/commande de réapprovisionnement
    n_jours: int = 20,
) -> None:
    """
    Illustre les effets de 1er, 2e et 3e ordre d'une remise commerciale.

    Stock = unités disponibles.
    Flux entrant = réapprovisionnements (commandés quand stock < seuil).
    Flux sortant = demande clients (multipliée pendant la remise).
    """
    print("=" * 60)
    print("C) REMISE COMMERCIALE — effets de second ordre")
    print(f"   Stock initial={stock_initial} | Demande normale={demande_normale}/j | "
          f"Remise J{remise_jour} (×{boost_demande} pendant 2j) | "
          f"Délai réappro={delai_reappro}j")
    print("=" * 60)

    stock = stock_initial
    reappros_en_route: list[float] = [0.0] * (delai_reappro + 1)  # file de livraisons
    SEUIL_REAPPRO = 150.0   # commande si stock < seuil
    commande_en_cours = False
    ruptures_cumulees = 0.0

    print(f"\n  {'Jour':>4} | {'Stock':>6} | {'Demande':>8} | {'Réappro':>8} | Note")
    print("  " + "-" * 55)

    for j in range(n_jours):
        # Flux sortant : demande
        if remise_jour <= j < remise_jour + 2:
            demande_j = demande_normale * boost_demande   # pic 1er ordre
            note = "🏷 REMISE"
        else:
            demande_j = demande_normale
            note = ""

        # Commande de réapprovisionnement si stock bas
        if stock < SEUIL_REAPPRO and not commande_en_cours:
            reappros_en_route.append(capacite_reappro)
            commande_en_cours = True
            note += " [commande lancée]"
        else:
            reappros_en_route.append(0.0)

        # Livraison du réappro (après délai)
        livraison = reappros_en_route.pop(0)
        if livraison > 0:
            commande_en_cours = False
            note += " [livraison reçue]"

        # Mise à jour stock
        ventes_reelles = min(demande_j, stock + livraison)   # pas de vente si rupture
        rupture = max(0.0, demande_j - (stock + livraison))
        stock = max(0.0, stock + livraison - demande_j)
        ruptures_cumulees += rupture

        print(f"  J{j:02d}  | {stock:6.1f} | {demande_j:8.1f} | {livraison:8.1f} | {note}")

    print(f"\n  → Ruptures cumulées (ventes perdues) : {ruptures_cumulees:.1f} unités")
    print("  → 2e ordre : le pic a vidé le stock ; le délai de réappro a causé des ruptures.")
    print("  → 3e ordre : clients non servis peuvent chercher un fournisseur concurrent.\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "━" * 60)
    print("  MODULE 12 — PENSÉE SYSTÉMIQUE : 3 simulations")
    print("━" * 60)

    # A) Thermostat : boucle équilibrante, gain modéré → convergence
    simuler_thermostat(
        temp_initiale=15.0, consigne=20.0, gain=0.4, delai_chaudiere=2, n_pas=30
    )

    # Même thermostat, gain trop élevé → oscillations (délai + gain excessif)
    print("\n--- Thermostat avec gain trop élevé (oscillations) ---")
    simuler_thermostat(
        temp_initiale=15.0, consigne=20.0, gain=0.9, delai_chaudiere=3, n_pas=30
    )

    # B) File d'attente : pic de demande → boucle renforçante → saturation
    simuler_file_attente(
        taux_arrivee_base=10.0, capacite_traitement=12.0,
        surge_debut=4, surge_fin=8, surge_facteur=2.0, n_heures=24
    )

    # C) Remise commerciale : effets de 1er, 2e, 3e ordre
    simuler_remise_commerciale()

    print("━" * 60)
    print("  Toutes les simulations terminées — exit 0")
    print("━" * 60)

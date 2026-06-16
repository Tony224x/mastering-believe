"""
Planificateur de revisions SM-2 (SuperMemo 2) — stdlib pur
===========================================================
Source : Wozniak, P. (1987-1990). SuperMemo Method.
         https://www.supermemo.com/en/supermemo-method

Principe :
- Chaque carte a un "facteur d'aisance" (EF, defaut 2.5) et un intervalle en jours.
- Apres chaque revision, l'utilisateur note son rappel de 0 a 5.
- L'intervalle suivant et l'EF sont mis a jour selon l'algorithme SM-2.
- Les intervalles croissants = spaced repetition — base scientifique : Cepeda et al. (2006, 2008).

Formules SM-2 :
  EF_new = EF + (0.1 - (5 - note) * (0.08 + (5 - note) * 0.02))
  EF_new = max(1.3, EF_new)  # plancher a 1.3

  Si note >= 3 (succes) :
    - 1re revision reussie  -> intervalle = 1 jour
    - 2e revision reussie   -> intervalle = 6 jours
    - Suivantes             -> intervalle = round(intervalle_precedent * EF)
  Si note < 3 (echec) :
    - Intervalle remis a 1 jour (cycle recommence)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Constantes SM-2
# ---------------------------------------------------------------------------
EF_DEFAULT = 2.5      # facteur d'aisance initial
EF_MIN = 1.3          # plancher : une carte tres difficile ne descend pas sous ca
INTERVAL_FIRST = 1    # 1re revision reussie -> 1 jour
INTERVAL_SECOND = 6   # 2e revision reussie -> 6 jours


# ---------------------------------------------------------------------------
# Structures de donnees
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """Represente une flashcard avec son etat SM-2."""
    question: str
    answer: str

    # Etat SM-2
    ef: float = EF_DEFAULT            # facteur d'aisance
    interval: int = 0                 # intervalle courant en jours
    repetitions: int = 0              # nombre de revisions reussies consecutives
    next_review: datetime.date = field(
        default_factory=datetime.date.today  # premiere revision aujourd'hui
    )

    def __post_init__(self) -> None:
        # S'assurer que next_review est un objet date (pas datetime)
        if isinstance(self.next_review, datetime.datetime):
            self.next_review = self.next_review.date()


@dataclass
class ReviewResult:
    """Resultat d'une session de revision pour une carte."""
    card_question: str
    note: int                   # 0-5
    ef_before: float
    ef_after: float
    interval_before: int
    interval_after: int
    next_review: datetime.date


# ---------------------------------------------------------------------------
# Algorithme SM-2
# ---------------------------------------------------------------------------

def update_card(card: Card, note: int, reviewed_on: datetime.date | None = None) -> ReviewResult:
    """
    Applique l'algorithme SM-2 a une carte apres une revision.

    Parametres
    ----------
    card : Card
        La carte a mettre a jour (modifiee en place).
    note : int
        Qualite du rappel, de 0 (echec total) a 5 (parfait).
    reviewed_on : date optionnelle
        Date de la revision (defaut : aujourd'hui). Utile pour les simulations.

    Retour
    ------
    ReviewResult contenant les valeurs avant/apres pour audit.
    """
    if not 0 <= note <= 5:
        raise ValueError(f"La note doit etre entre 0 et 5 (recue : {note})")

    today = reviewed_on or datetime.date.today()

    ef_before = card.ef
    interval_before = card.interval

    # --- Mise a jour du facteur d'aisance ---
    # Formule SM-2 officielle
    new_ef = card.ef + (0.1 - (5 - note) * (0.08 + (5 - note) * 0.02))
    card.ef = max(EF_MIN, new_ef)  # plancher

    # --- Calcul du prochain intervalle ---
    if note >= 3:
        # Succes : incrementer le compteur de repetitions reussies
        card.repetitions += 1
        if card.repetitions == 1:
            card.interval = INTERVAL_FIRST   # 1 jour
        elif card.repetitions == 2:
            card.interval = INTERVAL_SECOND  # 6 jours
        else:
            # Appliquer le facteur d'aisance (EF avant mise a jour, conventionnel SM-2)
            # Certaines implementations utilisent l'EF apres ; on utilise l'EF apres ici
            # comme dans Anki.
            card.interval = max(1, round(interval_before * card.ef))
    else:
        # Echec : recommencer le cycle depuis 1 jour
        card.repetitions = 0
        card.interval = 1  # pas 0 — l'espacement minimal d'1 jour reste benefique

    # --- Planifier la prochaine revision ---
    card.next_review = today + datetime.timedelta(days=card.interval)

    return ReviewResult(
        card_question=card.question,
        note=note,
        ef_before=ef_before,
        ef_after=card.ef,
        interval_before=interval_before,
        interval_after=card.interval,
        next_review=card.next_review,
    )


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def cards_due_today(deck: List[Card], today: datetime.date | None = None) -> List[Card]:
    """Retourne les cartes dont la revision est due aujourd'hui ou en retard."""
    today = today or datetime.date.today()
    return [c for c in deck if c.next_review <= today]


def simulate_sessions(
    card: Card,
    notes: List[int],
    start_date: datetime.date | None = None,
) -> List[ReviewResult]:
    """
    Simule plusieurs sessions de revision consecutives sur une carte.

    Parametres
    ----------
    card : Card
        La carte a simuler (modifiee en place).
    notes : list[int]
        Liste de notes dans l'ordre chronologique.
    start_date : date optionnelle
        Date de depart (defaut : aujourd'hui).

    Retour
    ------
    Liste de ReviewResult dans l'ordre chronologique.
    """
    results = []
    current_date = start_date or datetime.date.today()

    for note in notes:
        result = update_card(card, note, reviewed_on=current_date)
        results.append(result)
        # La prochaine session a lieu a la date de prochaine revision calculee
        current_date = card.next_review

    return results


def print_simulation_table(results: List[ReviewResult], start_date: datetime.date) -> None:
    """Affiche les resultats de simulation sous forme de tableau."""
    header = (
        f"{'Session':^7} | {'Note':^4} | {'EF avant':^8} | {'EF apres':^8} | "
        f"{'Intv. avant':^11} | {'Intv. apres':^11} | {'Prochaine revision':^18}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    cumulative_days = 0
    for i, r in enumerate(results, start=1):
        # Calculer les jours cumules depuis le debut
        days_from_start = (r.next_review - start_date).days
        print(
            f"  {i:^5}  | {r.note:^4} | {r.ef_before:^8.2f} | {r.ef_after:^8.2f} | "
            f"  {r.interval_before:^9} | "
            f"  {r.interval_after:^9} | "
            f"  {r.next_review} (J+{days_from_start})"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Demo principale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Planificateur SM-2 — Demo")
    print("Sources : Wozniak 1987-1990 ; Cepeda et al. 2006, 2008")
    print("=" * 65)

    # --- Deck exemple avec 3 cartes ---
    deck = [
        Card(
            question="Quel chiffre de rappel obtient le groupe 'test' dans Roediger & Karpicke (2006) apres 1 semaine ?",
            answer="61 % (vs 40 % pour le groupe relecture).",
        ),
        Card(
            question="Quelle est la formule de l'intervalle optimal selon Cepeda et al. (2008) ?",
            answer="Gap optimal ≈ 10-20 % du delai avant le prochain test.",
        ),
        Card(
            question="Que se passe-t-il dans SM-2 quand la note est < 3 ?",
            answer="L'intervalle repart a 1 jour et le compteur de repetitions reussies est remis a 0.",
        ),
    ]

    today = datetime.date.today()

    print(f"\nDate de debut de simulation : {today}")
    print(f"Nombre de cartes dans le deck : {len(deck)}\n")

    # --- Simulation sur la premiere carte avec des notes variees ---
    test_notes = [4, 3, 5, 2, 4, 4]
    print(f"Simulation de {len(test_notes)} sessions sur la carte 1")
    print(f"Question : \"{deck[0].question}\"")
    print(f"Notes appliquees : {test_notes}")
    print()

    # On travaille sur une copie pour ne pas alterer le deck original
    import copy
    card_sim = copy.deepcopy(deck[0])
    results = simulate_sessions(card_sim, test_notes, start_date=today)
    print_simulation_table(results, start_date=today)

    # --- Cartes dues aujourd'hui (toutes, puisque next_review = today au depart) ---
    print("\n--- Cartes dues aujourd'hui ---")
    due = cards_due_today(deck, today=today)
    for card in due:
        print(f"  * {card.question[:70]}...")

    # --- Simulation rapide pour les 2 autres cartes (notes parfaites) ---
    print("\n--- Simulation cartes 2 et 3 (notes parfaites : 5, 5, 5) ---")
    for i, card in enumerate(deck[1:], start=2):
        c = copy.deepcopy(card)
        res = simulate_sessions(c, [5, 5, 5], start_date=today)
        last = res[-1]
        print(
            f"  Carte {i} — apres 3 revisions parfaites : "
            f"EF={last.ef_after:.2f}, prochain intervalle={last.interval_after} jours "
            f"(revision le {last.next_review})"
        )

    print("\nConclusion pedagogique :")
    print("  - Notes elevees -> EF monte -> intervalles croissent rapidement.")
    print("  - Note < 3 -> reset a 1 jour -> le systeme force une nouvelle consolidation.")
    print("  - Sans SM-2 : on revise trop tot (gaspille) ou on oublie (trop tard).")
    print("  Voir : Cepeda et al. (2008), Psychological Science 19(11), 1095-1102.")
    print()

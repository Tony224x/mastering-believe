"""
Module 10 — Motivation, habitudes, énergie & apprendre sous pression
Tracker d'habitudes léger : série (streak), taux de complétion sur N jours.

Usage :
    python 10-motivation-habitudes.py

stdlib pur — aucune dépendance externe.
"""

import json
import os
import sys
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Fichier de persistance (JSON, dans le même répertoire que ce script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "10-habits-data.json")

# ---------------------------------------------------------------------------
# Chargement / sauvegarde
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Charge le fichier JSON. Retourne un dict vide si inexistant."""
    if not os.path.exists(DATA_FILE):
        return {"habits": {}}
    with open(DATA_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_data(data: dict) -> None:
    """Persiste le dict dans le fichier JSON."""
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Logique métier
# ---------------------------------------------------------------------------

def add_habit(data: dict, name: str) -> str:
    """Ajoute une nouvelle habitude (clé = nom). Retourne un message."""
    habits = data["habits"]
    if name in habits:
        return f"[info] L'habitude '{name}' existe déjà."
    # Chaque habitude est un dictionnaire {date_iso: bool}
    habits[name] = {}
    save_data(data)
    return f"[ok] Habitude '{name}' ajoutée."


def check_in(data: dict, name: str, target_date: date = None) -> str:
    """Marque l'habitude comme accomplie pour la date donnée (défaut: aujourd'hui)."""
    habits = data["habits"]
    if name not in habits:
        return f"[erreur] Habitude '{name}' inconnue. Ajoutez-la d'abord."
    if target_date is None:
        target_date = date.today()
    date_key = target_date.isoformat()
    habits[name][date_key] = True
    save_data(data)
    return f"[ok] '{name}' cochée pour le {date_key}."


def compute_streak(completions: dict) -> int:
    """
    Calcule la série courante (streak) : nombre de jours consécutifs
    terminés jusqu'à aujourd'hui (ou hier si aujourd'hui n'est pas coché).

    Paramètre :
        completions — dict {date_iso: bool}, seules les entrées True comptent.

    Retourne le nombre de jours de la série en cours.
    """
    done_dates = {
        date.fromisoformat(k)
        for k, v in completions.items()
        if v  # seulement les jours marqués True
    }

    today = date.today()
    # Si aujourd'hui est déjà coché, on part d'aujourd'hui ; sinon d'hier.
    current = today if today in done_dates else today - timedelta(days=1)

    streak = 0
    while current in done_dates:
        streak += 1
        current -= timedelta(days=1)  # on remonte dans le temps

    return streak


def compute_completion_rate(completions: dict, window: int = 30) -> float:
    """
    Taux de complétion sur les N derniers jours (défaut 30).
    = nombre de jours cochés / window * 100 (en %)

    Paramètre :
        completions — dict {date_iso: bool}
        window      — nombre de jours à considérer
    """
    today = date.today()
    done = 0
    for offset in range(window):
        day = today - timedelta(days=offset)
        if completions.get(day.isoformat(), False):
            done += 1
    # On évite la division par zéro (window est toujours >= 1 ici)
    return done / window * 100


def show_stats(data: dict, window: int = 30) -> None:
    """Affiche les statistiques de toutes les habitudes."""
    habits = data["habits"]
    if not habits:
        print("Aucune habitude enregistrée. Utilisez l'option 'add' pour en créer une.")
        return

    print(f"\n{'Habitude':<30} {'Streak':>8} {'Taux %':>10} (sur {window}j)")
    print("-" * 55)
    for name, completions in sorted(habits.items()):
        streak = compute_streak(completions)
        rate = compute_completion_rate(completions, window=window)
        # Barre de progression visuelle (sur 20 caractères)
        filled = int(rate / 5)  # 1 bloc = 5 %
        bar = "█" * filled + "░" * (20 - filled)
        print(f"{name:<30} {streak:>7}j  {rate:>6.1f}%  {bar}")
    print()


def list_habits(data: dict) -> None:
    """Liste toutes les habitudes enregistrées."""
    habits = data["habits"]
    if not habits:
        print("Aucune habitude enregistrée.")
        return
    print("\nHabitudes enregistrées :")
    for name in sorted(habits.keys()):
        print(f"  • {name}")
    print()


# ---------------------------------------------------------------------------
# Démonstration autonome (mode --demo)
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Démo sans interaction : crée des données fictives en mémoire,
    calcule les métriques, affiche les résultats.
    Aucun fichier n'est écrit sur le disque.
    """
    print("=" * 55)
    print("  DÉMONSTRATION — Tracker d'habitudes (données fictives)")
    print("=" * 55)

    today = date.today()

    # Habitude 1 : Anki (30 jours complets sauf les 5 derniers)
    anki_completions = {
        (today - timedelta(days=i)).isoformat(): True
        for i in range(6, 31)  # jours 6 à 30 (en partant d'aujourd'hui)
    }

    # Habitude 2 : Deep work (tous les 2 jours environ)
    deep_work_completions = {
        (today - timedelta(days=i)).isoformat(): True
        for i in range(0, 30, 2)  # j0, j2, j4 ... j28
    }

    # Habitude 3 : Lecture (streak actif de 7 jours)
    lecture_completions = {
        (today - timedelta(days=i)).isoformat(): True
        for i in range(0, 7)  # j0 à j6
    }

    demo_habits = {
        "Anki (15 cartes/jour)": anki_completions,
        "Deep work (45 min)":    deep_work_completions,
        "Lecture (20 min)":      lecture_completions,
    }

    window = 30
    print(f"\n{'Habitude':<30} {'Streak':>8} {'Taux %':>10} (sur {window}j)")
    print("-" * 55)
    for name, completions in demo_habits.items():
        streak = compute_streak(completions)
        rate = compute_completion_rate(completions, window=window)
        filled = int(rate / 5)
        bar = "█" * filled + "░" * (20 - filled)
        print(f"{name:<30} {streak:>7}j  {rate:>6.1f}%  {bar}")

    print()

    # Explication pédagogique du lien avec le Module 10
    print("─" * 55)
    print("Lien Module 10 — Motivation, habitudes & énergie")
    print("─" * 55)
    print("""
  • 'Streak' = renforcement de la boucle signal→routine→récompense
    (Wood & Rünger 2016) : voir sa série progresser est une
    récompense immédiate qui stabilise l'habitude.

  • 'Taux sur 30j' = vue Steel (2007) : un taux < 50 % signale
    souvent un problème de contexte (signal instable) ou
    d'aversion de la tâche — levier à corriger.

  • Recommandation : planifier les sessions à heure fixe,
    dans un lieu dédié — la stabilité du contexte prime sur
    la motivation du moment.
""")


# ---------------------------------------------------------------------------
# Interface en ligne de commande (CLI) minimale
# ---------------------------------------------------------------------------

HELP = """
Tracker d'habitudes — Module 10 (apprendre-a-apprendre)

Commandes :
  --demo                   Lance une démonstration avec données fictives
  --add <nom>              Ajoute une nouvelle habitude
  --check <nom>            Marque l'habitude comme accomplie aujourd'hui
  --stats [<n_jours>]      Affiche les stats (défaut : 30 jours)
  --list                   Liste les habitudes
  --help / -h              Affiche cette aide

Exemples :
  python 10-motivation-habitudes.py --demo
  python 10-motivation-habitudes.py --add "Anki (15 cartes)"
  python 10-motivation-habitudes.py --check "Anki (15 cartes)"
  python 10-motivation-habitudes.py --stats 7
"""


def main() -> int:
    """
    Point d'entrée CLI.
    Retourne 0 si tout s'est bien passé, 1 en cas d'erreur.
    """
    args = sys.argv[1:]

    # Sans argument : afficher l'aide
    if not args or args[0] in ("--help", "-h"):
        print(HELP)
        return 0

    # Mode démonstration (aucun fichier écrit)
    if args[0] == "--demo":
        run_demo()
        return 0

    # Mode interactif avec persistance
    data = load_data()

    if args[0] == "--add":
        if len(args) < 2:
            print("[erreur] Usage : --add <nom>")
            return 1
        name = " ".join(args[1:])
        print(add_habit(data, name))
        return 0

    if args[0] == "--check":
        if len(args) < 2:
            print("[erreur] Usage : --check <nom>")
            return 1
        name = " ".join(args[1:])
        print(check_in(data, name))
        return 0

    if args[0] == "--stats":
        window = 30
        if len(args) >= 2:
            try:
                window = int(args[1])
                if window <= 0:
                    raise ValueError
            except ValueError:
                print("[erreur] <n_jours> doit être un entier positif.")
                return 1
        show_stats(data, window=window)
        return 0

    if args[0] == "--list":
        list_habits(data)
        return 0

    # Commande inconnue
    print(f"[erreur] Commande inconnue : {args[0]}\n{HELP}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

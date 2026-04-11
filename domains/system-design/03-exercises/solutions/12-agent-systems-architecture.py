"""
Solutions -- Jour 12 : Agent Systems Architecture
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Single-agent ou multi-agent ?

    +-----------------------------+---------------+------------------------------+
    | Produit                     | Pattern       | Justification                |
    +-----------------------------+---------------+------------------------------+
    | Coding assistant (Cursor)   | Single-agent  | Taches homogenes (code), low |
    |                             |               | latence critique, < 20 tools.|
    |                             |               | Un bon ReAct loop suffit.    |
    +-----------------------------+---------------+------------------------------+
    | Recherche financiere        | Supervisor    | Taches heterogenes (read,    |
    | + memo                      | multi-agent   | compare, write), contexte    |
    |                             |               | long, besoin de specialistes.|
    +-----------------------------+---------------+------------------------------+
    | Bot Slack doc interne       | Single-agent  | C'est un RAG + 2-3 tools.    |
    |                             |               | Multi-agent serait over-     |
    |                             |               | engineered.                  |
    +-----------------------------+---------------+------------------------------+
    | Automatisation marketing    | Supervisor    | Segment + write + send +     |
    |                             | multi-agent   | analyze. Specialistes        |
    |                             |               | permettent d'optimiser chaque|
    |                             |               | etape independamment.        |
    +-----------------------------+---------------+------------------------------+
    | Deep research report        | Hierarchical  | Phases (planning / research /|
    |                             |               | drafting / review) avec sous |
    |                             |               | agents par phase. Budget     |
    |                             |               | eleve, latence non critique. |
    +-----------------------------+---------------+------------------------------+
    | Correcteur grammaire IDE    | Single-agent  | Latence < 100 ms obligatoire.|
    |                             | (local !)     | Aucun tool, juste le modele. |
    |                             |               | Pas de place pour un multi.  |
    +-----------------------------+---------------+------------------------------+

    Principe sous-jacent : **start with the simplest thing that works**.
    Commence single-agent. Passe a multi quand tu as la preuve que c'est
    necessaire (contexte trop gros, specialisations distinctes, ou
    bottleneck clair).

    Latence mentionnee pour :
    - Coding assistant : doit repondre < 2 s -> single
    - Correcteur grammaire : < 100 ms -> single local
    - Bot Slack : < 5 s acceptable -> single
    - Deep research : plusieurs minutes OK -> multi-agent hierarchical
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- State et conditions d'arret d'un assistant voyage.

    1) State minimum (pseudo-dataclass) :

        @dataclass
        class TravelState:
            # user input
            user_id: str
            session_id: str
            user_request: str
            destination: Optional[str]
            date_start: Optional[str]
            date_end: Optional[str]
            budget_eur: Optional[float]
            preferences: dict        # {airline, hotel_class, vegetarian...}

            # agent working memory
            candidate_flights: list[dict]
            candidate_hotels: list[dict]
            itinerary: Optional[dict]
            validation_status: Literal["pending", "approved", "rejected"]

            # conversation
            messages: list[dict]      # chronological dialog

            # control flow
            steps_used: int
            max_steps: int
            last_agent: Optional[str]
            waiting_for_user: bool
            done: bool
            error: Optional[str]

    2) Conditions d'arret :
       a) `done == True AND validation_status == "approved"` : tache complete
       b) `steps_used >= max_steps` : budget epuise
       c) `error is not None` : erreur non recuperable (API down, auth KO)
       d) `waiting_for_user == True` : main rendue a l'utilisateur, on sort
          de la boucle jusqu'a reveil
       e) Detection d'une boucle (meme action 3 fois de suite) : stop
          defensif

    3) Si budget depasse avant completion :
       - L'agent doit retourner un *checkpoint* au user : "voici ce que
         j'ai trouve jusqu'ici, continue ou arrete ?"
       - Ne PAS crasher. Ne PAS halluciner pour completer.
       - Enregistrer les partial results dans le state.

    4) Human-in-the-loop sans boucle :
       - L'agent positionne `waiting_for_user = True` et retourne un
         message structure ("je propose itineraire X, valides-tu ?").
       - Le state est *persistee* (checkpoint DB).
       - L'agent ne tourne plus. Quand le user repond, on recharge le
         state et on reprend au point suivant.
       - Frameworks : LangGraph `interrupt()` + `Command(resume=...)`.
         CrewAI a aussi ce pattern.

    5) Persistence entre sessions :
       - **State structure** : DB relationnelle (PostgreSQL) pour les
         donnees critiques (budget restant, reservations, auth). Cle =
         session_id.
       - **Checkpoint de l'agent** : store cle-valeur (Redis, DynamoDB)
         serialise JSON, indexe par session_id.
       - **Long-term preferences** : store separe par user_id. Peut
         etre un vector store + une DB (ex: "aime les hotels 4 etoiles
         avec piscine, pas de vol apres 22h").
       - Toujours respecter le RGPD : TTL sur les PII, consent explicite.
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Multi-agent qui hallucine.

    Hypotheses, tests, mitigations :

    H1 -- **Handoff messages trop pauvres**
      - Test : inspecter les messages passes d'un agent a l'autre. S'ils
        contiennent juste "continue" ou "do the thing", on a trouve.
      - Mitigation : formaliser un schema Handoff {context, done,
        remaining, success_criteria, budget}. Refuser les handoffs qui
        ne respectent pas le schema.
      - Impact/effort : HIGH / LOW -> **PRIORITE 1**

    H2 -- **Context bleed / reuse sans verification**
      - Le writer cite les resultats du search sans verifier qu'ils
        existent vraiment dans la memoire.
      - Test : inspecter les cites du writer et comparer au log du
        search_agent.
      - Mitigation : groundedness check automatique : chaque citation
        du writer doit pointer vers un chunk_id qui existe dans le
        state.results. Si pas trouve -> flag invalid.
      - Impact/effort : HIGH / MEDIUM -> **PRIORITE 2**

    H3 -- **Memoire stale reutilisee**
      - Des donnees d'un step precedent sont reutilisees alors qu'elles
        ne sont plus valides (ex: snapshot de prix 30 min plus tard).
      - Test : timestamper toutes les entrees de state.results et
        verifier la difference temporelle au moment de l'utilisation.
      - Mitigation : TTL sur les entrees de memoire, et invalidation
        explicite par etape.
      - Impact/effort : MEDIUM / MEDIUM -> PRIORITE 4

    H4 -- **Critic non-independant (meme modele que les agents)**
      - Si le critic utilise le meme LLM que les executors, il valide
        les memes hallucinations qu'il est cense detecter.
      - Test : faire passer le critic sur 50 exemples avec erreurs
        connues et mesurer le recall.
      - Mitigation : utiliser un modele different pour le critic
        (autre provider, ou plus fort, ou plus strict via prompt).
        Idealement combine avec des verifications deterministes
        (schema, groundedness) AVANT d'invoquer le LLM critic.
      - Impact/effort : HIGH / MEDIUM -> **PRIORITE 3**

    H5 -- **Pas de stopping criteria explicite / supervisor trop
          permissif**
      - Le supervisor declare "done" alors qu'une partie du plan n'est
        pas faite parce qu'il n'a pas de checklist.
      - Test : comparer le plan initial avec les steps reellement
        executes.
      - Mitigation : le supervisor maintient une checklist, et ne
        peut declarer "done" qu'apres avoir verifie que chaque item
        est coche (avec proof -> chaque item est lie a un result dans
        le state).
      - Impact/effort : HIGH / LOW -> **PRIORITE 1bis**

    Bonus H6 -- Pas d'observability : si on n'a pas de trace LangSmith
    / Langfuse, debugger est impossible. Installer en priorite zero.

    Plan de priorisation (impact/effort) :
      1. Handoff messages structures (H1)
      2. Checklist obligatoire du supervisor (H5)
      3. Groundedness check automatique (H2)
      4. Critic modele independant (H4)
      5. Memoire timestampee et invalide-able (H3)

    Transverse : metrique globale = taux d'hallucination mesure sur
    un gold set de 50 taches. Suivre apres chaque mitigation.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)

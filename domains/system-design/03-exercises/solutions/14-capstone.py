"""
Solutions -- Jour 14 : Capstone

Les 3 exercices capstone sont longs. Les solutions sont fournies comme
walkthroughs complets dans des docstrings. Chaque solution suit le
framework en 6 etapes : clarifier -> estimer -> HL design -> deep dive
-> bottlenecks -> extensions.
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Spotify Recommendations.

    === CLARIFIER ===
    Functional :
      - Home feed personnalisee (mix morceaux + playlists)
      - Discover Weekly : 30 morceaux nouveaux chaque lundi
      - Radios personnalisees a partir d'un titre / artiste
      - Feedback (like/dislike/skip) alimente les recos
      - Cold start pour nouveaux users

    Non-functional :
      - 500M MAU, 200M DAU, peak 50-80M concurrent
      - Latence home feed < 500 ms p95
      - Recos mises a jour en "near real-time" (apres un like, l'effet
        doit etre visible dans les minutes qui suivent)
      - Cost efficace : 50 recos/user/jour = 10 G recos/jour

    === CAPACITY ===
      Home requests : 200M users * 10 visites / jour * 50 items = 100 G items/jour
      Peak factor 3x -> 3.5M items/sec. Mais les items sont batches
      cote server -> on compte plutot en "home requests" :
      200M * 10 = 2 G home requests / jour = 23K rps avg, ~70K peak.
      Storage recos materialisees : 200M users * 50 reco_slots * 20 B (ids)
      = 200 GB. Rien.
      Feature store : 200M users * ~50 features = 10 G cells.
      Vector embeddings tracks : 100M * 256 dims * 4 bytes = 100 GB.
      Ingestion d'events (plays, skips, likes) : 200M users * 50 events/jour
      = 10 G events/jour, stockes pour retraining.

    === HIGH LEVEL DESIGN ===
                                          +-----------+
        client app ---> LB ---> API ---+->| Recos svc |---+
                                       |  +-----------+   |
                                       v                  v
                                 +-----------+      +----------+
                                 | Home      |      | Radios   |
                                 | Composer  |      | svc      |
                                 +-----+-----+      +----------+
                                       |
                          fetch cached     fetch online scores
                          recos (Redis)    (feature store online)
                                       |
                     +-----------------+------------------+
                     |                 |                  |
                     v                 v                  v
              +------------+    +-------------+    +-------------+
              | Candidate  |    | Ranker      |    | Diversifier |
              | generator  |    | (XGBoost /  |    | + freshness |
              |            |    |  2-tower NN)|    |             |
              +------------+    +-------------+    +-------------+
                     ^
                     |
              +------+------+
              | Feature     |
              | store       |
              | (offline +  |
              |  online)    |
              +-------------+
                     ^
                     |
              +-------------+      +-------------+
              | Batch jobs  |<-----| Event stream|
              | Spark/Flink |      | Kafka       |
              | (Discover,  |      | (plays,     |
              |  playlists) |      |  skips...)  |
              +-------------+      +-------------+

    === DEEP DIVE ===

    1) Feature store
      - Offline (BigQuery / Parquet) : historiques par user (avg skip rate,
        diversity index, genre preferences 30d, listening time by time-of-day)
      - Online (Redis / Dynamo) : latest materialised features, cle =
        user_id, valeur = pack de features necessaires au ranker
      - Materialisation : chaque feature batch/stream a son pipeline

    2) Batch Discover Weekly
      - Job Spark hebdo qui :
        a) joint les user embeddings (appris par 2-tower) avec les
           track embeddings
        b) calcule top-500 candidats par user par ANN dans l'espace
           embedding
        c) applique des filtres (pas deja ecoutes, diversity, freshness)
        d) materialise 30 tracks par user dans Redis avec TTL 1 semaine
      - Volume : 200M users * 30 = 6G ecritures -> faisable en quelques
        heures sur un cluster Spark dimensionne

    3) Real-time home scoring
      - Candidats : mix de 3 sources
        a) Discover Weekly precompute (poids 40%)
        b) Collaborative filtering batch (poids 30%)
        c) Context-aware online (poids 30%) : temps de la journee,
           device, derniere ecoute
      - Ranker online : XGBoost ou 2-tower leger, features lues du
        online store, latence < 30 ms
      - Re-ranking final : diversity constraint (pas 3 morceaux du
        meme artiste en top 5), freshness boost, A/B test variants
      - Tout ceci tourne en < 200 ms. Avec cache Redis pour les users
        qui reviennent, < 50 ms sur cache hit.

    === BOTTLENECKS ===
      B1. Feature store online : cle hot (top users) -> cache local
          sur les ranker pods + replicas sharde par user_id.
      B2. Ranker CPU : le peak QPS sature les ranker pods -> autoscale
          HPA sur CPU + queue depth, ranker batch en micro-batch de 10.
      B3. Event ingestion : Kafka -> Flink -> feature store, lag si
          burst de likes -> buffer + retries + monitoring lag.

    === EXTENSIONS ===
      - Real-time personnalisation + updates : apres chaque skip/like,
        ajuster les features online sous 1 s.
      - Generation de descriptions par LLM ("Parce que vous ecoutez
        X, voici Y") : LLM utilise UNIQUEMENT pour le texte humain,
        jamais pour scorer.
      - Cold-start pour nouveaux users : onboarding avec quelques
        preferences explicites + content-based sur la premiere
        session.
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- GitHub Copilot backend.

    === CLARIFIER ===
    Functional :
      - Inline code completion pendant le user tape
      - Multi-line suggestions
      - Context-aware : sait ce qui est dans le fichier courant + fichiers
        voisins
      - Streaming : tokens arrivent un par un
      - Cancellation : nouvelle touche user -> annule la completion en cours
      - Multi-langage (TS, Py, Go, Rust...)

    Non-functional :
      - 10M devs / jour
      - Latence p95 < 300 ms end-to-end (TTFT)
      - Uptime 99.9%+
      - Privacy : enterprise peut demander "don't send code out of VPC"
      - Cost : budget hyper serre ($0.001 - $0.003 par completion)

    === CAPACITY ===
      Completions / jour : 10M * 30/hr * 8 hr = 2.4 G completions / jour
      -> ~28K rps avg, peak 80K rps.
      Tokens par call : ~2500 in + 80 out
      -> 6T tokens in / jour. Gigantesque.
      Cost estime (code-specialized model hosted) : $0.0015 / completion
      -> $3.6M / jour de cost brut.
      -> evidemment pas tenable sans routing + caching + smaller models.
      Marge : chaque dev paye $10-20/mois -> $100M/mois revenu.
      Cible infra : < 20% du revenu = $20M/mois budget infra.

    === HIGH LEVEL DESIGN ===

        IDE client ----> edge router (geo) ----+
                              |                |
                      cancellation              v
                      handler          +----------------+
                                       | Context engine |
                                       | - open buffer  |
                                       | - nearby files |
                                       | - symbol index |
                                       +--------+-------+
                                                |
                                                v
                                     +----------------------+
                                     | Prefix cache /       |
                                     | KV sharing           |
                                     +-----------+----------+
                                                 |
                                                 v
                                     +----------------------+
                                     | Routing layer        |
                                     | - small model 70% req|
                                     | - big model 25% req  |
                                     | - giant model  5% req|
                                     +-----------+----------+
                                                 |
                         +-----------+-----------+------------+
                         |           |           |            |
                         v           v           v            v
                    +---------+ +---------+ +---------+  +--------+
                    | Small   | | Medium  | | Frontier|  | Enterp.|
                    | LLM     | | LLM     | | (rare)  |  | isolated
                    | (A100)  | | (H100)  | | (H100)  |  | deploy |
                    +---------+ +---------+ +---------+  +--------+

    === DEEP DIVE ===

    1) Comment tenir < 300 ms p95 ?
      - Modele principal quantise (int8) tournant sur GPUs aggressivement
        batched (vLLM continuous batching)
      - Streaming : renvoyer les premiers tokens des ~100 ms (TTFT)
      - Prefix cache : si le prompt commence par le meme code que la
        precedente requete, KV cache reutilise (gain 2-3x)
      - Routing : 70% des requetes vont vers le small model (~100 ms
        inference) et ne sont escaladees que si besoin

    2) Cache strategy
      - Completion-level cache : cle = hash(buffer prefix). Si la meme
        completion a deja ete servie, renvoyer directement. Hit rate
        reel : 5-10% (beaucoup de variation) mais gratuit.
      - KV prefix cache sur le modele : si les premiers 2000 tokens du
        prompt sont les memes, le KV est reutilise (inferieur au modele,
        gere par vLLM).
      - Embedding-based cache : pour les tournures courantes ("def
        process_" "return ..."), un cache semantique peut capter.

    3) Routing par tier
      - Small (code-llama-7b quantise, hosted) : 70% des requetes
        (code simple, completion 1 ligne)
      - Medium (starcoder-15b ou dedie) : 25% (multi-ligne, logique)
      - Frontier (GPT-5.4) : 5% (cas ambigus, ou "explain-this")
      - Enterprise : deployed dans VPC client, modele dedie

    === CHALLENGES SPECIFIQUES ===

    Streaming token-by-token :
      - L'API doit etre SSE (Server-Sent Events) ou HTTP streaming
      - L'IDE lit au fur et a mesure et met a jour le preview
      - Pas de buffering cote CDN !

    Cancellation :
      - Chaque requete a un request_id
      - Si le user tape une nouvelle touche, l'IDE envoie
        POST /cancel/{request_id}
      - Cote serving : le job en cours est interrompu, le GPU passe au
        suivant (vLLM le supporte avec abort_request)
      - Critique pour eviter de gacher du GPU sur des requetes abandonnees

    Privacy enterprise :
      - Deux deploiements distincts :
        a) Public : multi-tenant, modele centralise
        b) Enterprise : VPC isolated, soit self-hosted, soit peered
           avec le VPC client, logs chiffres, pas de donnees conservees
      - Enterprise paye plus cher (2-3x) pour ce confort

    === BOTTLENECKS ===
      - GPU capacity : autoscale aggressif + reservation guarantee pour
        enterprise
      - Context engine : indexation des fichiers doit etre rapide
        (tree-sitter + embedding leger cote client)
      - Latence reseau : deploiement multi-region (US, EU, Asia)

    === EXTENSIONS ===
      - Copilot Chat : meme infra mais avec un long context et des outils
      - Multi-file editing : agentic mode qui edit plusieurs fichiers
      - Inline tests : generer test au cote de la fonction
      - Fine-tuning tenant : adapter le modele au code propre d'un client
      - Telemetry pour fine-tune futurs modeles (avec consent)
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Autonomous Research Agent.

    === CLARIFIER ===
    Functional :
      - Input : un sujet ("etat de l'art batteries 2026")
      - Output : rapport markdown 10-20 pages, citations verifiables
      - Progress tracking en temps reel
      - Interrupt / resume

    Non-functional :
      - Run : 5-30 minutes, async
      - Budget : $5-20 par run en LLM cost
      - 100-500 search queries max
      - Robustesse aux sites en panne

    Definition de "done" :
      - Critic agent valide que le rapport couvre tous les sous-sujets
      - Budget non depasse
      - Completeness score > 80%
      - OU user interrompt

    === ARCHITECTURE MULTI-AGENT (hierarchical) ===

                              +-----------------+
                              | Orchestrator    |
                              | (master plan)   |
                              +--------+--------+
                                       |
              +------------------------+-----------------------+
              |                        |                       |
              v                        v                       v
        +------------+           +-------------+         +------------+
        | Planner    |           | Executor    |         | Reviewer   |
        | - decompose|           | supervisor  |         | - critic   |
        | - subquest |           +------+------+         | - rewrite  |
        +------------+                  |                +------------+
                                        |
                        +---------------+---------------+
                        |               |               |
                        v               v               v
                 +-----------+   +-----------+   +-----------+
                 | Searcher  |   | Reader    |   | Summarizer|
                 | agent     |   | agent     |   | agent     |
                 +-----------+   +-----------+   +-----------+
                        |               |
                        v               v
                 +-----------+   +-----------+
                 | Web search|   | Content   |
                 | tool      |   | extractor |
                 +-----------+   +-----------+

    === DEEP DIVE ===

    1) Planning
      - Le planner utilise un LLM fort pour decomposer la question en
        5-10 sous-questions couvrantes (coverage check)
      - Chaque sous-question a : objectif, priorite, budget estime
      - Le plan est versionne : on peut le modifier en cours de run

    2) Source evaluation
      - Chaque source reçoit un score de credibilite :
        - Domain reputation (whitelist scientifique, blacklist low-qual)
        - Date de publication
        - Citations presentes
        - LLM-as-a-judge : est-ce une source primaire ?
      - Score < seuil -> rejeter
      - Diversite : eviter d'utiliser 5 sources du meme site

    3) Citation verification
      - Chaque claim dans le rapport est lie a un chunk source
      - Avant d'ecrire le rapport, le writer doit fournir les chunks
      - Apres generation, un check exact-match verifie que les quotes
        existent dans les sources. Si non : alerter et regenerer.
      - L'URL + le quote exact sont stockes pour permettre au user de
        verifier.

    4) Memory management
      - Working memory (contexte courant) : plan courant + top 5
        sources actives + resume des recherches recentes
      - Long-term memory : vector store indexe toutes les sources
        lues dans ce run (permet de retrouver plus tard)
      - Quand le contexte s'approche du max, summariser les plus
        anciennes sources et les archiver dans le long-term

    === FAILURE MODES ===

    F1. Boucle infinie (l'agent cherche les memes choses)
      - Detection : si le meme sous-objectif est retrye > 3 fois
        sans progres -> skip + note
      - Cap dur : max_iterations = 50

    F2. Budget explosion
      - Chaque call LLM et search est compte vs budget
      - Warning a 70%, hard stop a 100%
      - Le rapport peut etre "partiel" avec un disclaimer

    F3. Source hallucination
      - Groundedness check avant ecriture : chaque citation est verifiee
      - Un critic independant (modele different) re-lit et flag les
        citations douteuses
      - Si hallucination detectee -> regenerer la section OU flagger
        "unverified" dans le rapport

    === OBSERVABILITY ===

    - Langfuse trace par run, avec sub-spans par agent et par step
    - Live progress : une table "run_id -> status" est updated a chaque
      step. Le frontend polle ou utilise SSE pour afficher "30/100
      searches done, 5/10 sub-questions covered"
    - Budget tracker visible en live
    - Un mode "debug" permet de voir le prompt et la reponse de chaque
      sous-agent

    === EXTENSIONS ===

    - Human interrupt : le user peut mettre en pause, rediriger
      ("concentre-toi sur la chimie, pas sur les constructeurs"),
      reprendre
    - Collaboration entre agents : un agent peut demander de l'aide
      a un autre agent s'il est bloque
    - Reutilisation : les recherches deja faites sont cache-ables. Si
      un autre user pose la meme question, on reutilise.
    - Verification automatique : avant de livrer le rapport, un agent
      "fact-checker" repete une partie des queries pour verifier
      que les faits sont toujours vrais.
    - Output formats multiples : markdown, PDF, slides
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)

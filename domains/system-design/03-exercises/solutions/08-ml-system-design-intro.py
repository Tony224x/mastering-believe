"""
Solutions -- Jour 8 : ML System Design Intro

Les solutions sont fournies sous forme de docstrings detaillees, comme
dans le reste du domaine. Ce fichier est standalone et executable pour
verification syntaxique (python 08-ml-system-design-intro.py).
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Diagnostic training-serving skew.

    +-----+----------------------------+------------------------------------+
    | Cas | Cause                      | Mitigation                         |
    +-----+----------------------------+------------------------------------+
    |  1  | Code divergent :           | Feature store unique. La meme      |
    |     | Python au training, SQL    | fonction Python sert offline et    |
    |     | au serving -> formules     | online. Si SQL obligatoire,        |
    |     | differentes (rounding,     | generer la SQL depuis la meme      |
    |     | NaN, cast types).          | definition (DSL / DBT / Feast).    |
    +-----+----------------------------+------------------------------------+
    |  2  | Feature drift ou concept   | Monitoring continu (PSI, KL) sur   |
    |     | drift non detecte. Le      | les distributions des inputs et    |
    |     | monde a change, le modele  | des outputs. Retraining automatise |
    |     | a date gele.               | declenche par le monitoring.       |
    |     |                            | A minima : retraining planifie     |
    |     |                            | (hebdomadaire/mensuel).            |
    +-----+----------------------------+------------------------------------+
    |  3  | Data handling different.   | Implementer la logique NaN         |
    |     | Les NaN sont traitees en   | dans la feature definition elle-   |
    |     | training (fillna mean)     | meme. Pas dans le notebook. Ce     |
    |     | mais pas en serving.       | qu'on fait au training = ce qu'on  |
    |     |                            | fait au serving.                   |
    +-----+----------------------------+------------------------------------+
    |  4  | Data leakage / absence de  | Point-in-time correctness :        |
    |     | point-in-time correctness. | quand on construit le dataset,     |
    |     | Le modele a vu le futur.   | chaque exemple voit les features   |
    |     |                            | telles qu'elles etaient a          |
    |     |                            | l'instant de l'evenement. Les      |
    |     |                            | reviews posterieures ne doivent    |
    |     |                            | PAS etre incluses dans la moyenne. |
    |     |                            | C'est LA raison d'etre du feature  |
    |     |                            | store cote offline.                |
    +-----+----------------------------+------------------------------------+

    Note : dans le monde reel, les 4 problemes cohabitent souvent. Un feature
    store bien concu en regle 3 sur 4. Le 4e (drift) demande du monitoring.
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- Batch vs real-time.

    +-----------------+----------+---------------+-------------------------------+
    | Produit         | Archi    | Latence cible | Justification                 |
    +-----------------+----------+---------------+-------------------------------+
    | Netflix homepage| Hybride  | < 500 ms      | Batch quotidien pour les      |
    |                 |          | (re-ranking)  | candidats (1M+ items).        |
    |                 |          |               | Re-ranking online par         |
    |                 |          |               | contexte (device, heure).     |
    +-----------------+----------+---------------+-------------------------------+
    | Anti-fraude     | Real-time| p99 < 100 ms  | Refus bloquant = latence      |
    | paiement        |          |               | critique. 1000 tx/s = infra   |
    |                 |          |               | dimensionnee en consequence   |
    |                 |          |               | (GPU + dynamic batching).     |
    +-----------------+----------+---------------+-------------------------------+
    | Resume emails   | Batch    | Plusieurs     | Generation offline pendant    |
    | quotidiens      |          | heures OK     | la nuit. Lu le matin depuis   |
    |                 |          |               | une DB. Cout divise par 10+.  |
    +-----------------+----------+---------------+-------------------------------+
    | Pricing Uber    | Real-time| p95 < 500 ms  | Depend de l'offre/demande     |
    |                 |          |               | locale en temps reel.         |
    |                 |          |               | Features frequemment mises    |
    |                 |          |               | a jour.                       |
    +-----------------+----------+---------------+-------------------------------+
    | IoT anomalies   | Hybride  | 1-60 sec      | Micro-batch toutes les 10-60s |
    | (5M capteurs)   | (micro)  |               | pour accumuler les signaux.   |
    |                 |          |               | Volume trop eleve pour du     |
    |                 |          |               | vrai real-time par capteur.   |
    +-----------------+----------+---------------+-------------------------------+
    | Correcteur IDE  | Real-time| < 50 ms       | UX exige la reactivite.       |
    |                 | (local)  |               | Souvent modele distille qui   |
    |                 |          |               | tourne localement dans l'IDE. |
    +-----------------+----------+---------------+-------------------------------+

    Points cles :
    - Le volume et la frequence de changement du contexte dictent le choix.
    - Real-time n'est pas "mieux" -- il est 10x plus cher en infra.
    - Hybride est la norme a grande echelle : batch pour le lourd,
      online pour la personnalisation finale.
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Feature store pour un e-commerce de mode.

    +----------------------------+--------+--------------+----------------+---------+
    | Feature                    | Source | Offline      | Online         | TTL     |
    +----------------------------+--------+--------------+----------------+---------+
    | user_avg_order_value_30d   | batch  | Parquet/BQ   | Redis/Dynamo   | 24h     |
    | user_fav_category          | batch  | Parquet/BQ   | Redis/Dynamo   | 24h     |
    | user_current_cart_size     | stream | Kafka topic  | Redis (stream) | session |
    |                            |        | + archived   |                |         |
    | product_sales_rank_7d      | batch  | BigQuery     | Redis cluster  | 1h      |
    |                            |        |              | (hot items)    |         |
    | user_device_type           | on-    | --           | passee dans    | --      |
    |                            | demand |              | la requete     |         |
    +----------------------------+--------+--------------+----------------+---------+

    Schema ASCII:

        +-------------+      +------------+     +---------------+
        | Click/order |----->| Kafka topic|---->| Flink stream  |
        | events      |      +-----+------+     | processor     |
        +-------------+            |            +-------+-------+
                                   |                    |
                                   v                    v (cart_size)
                          +---------------+      +-------------+
                          | S3 / Parquet  |      | Redis online|
                          | (offline)     |      | store       |
                          +-------+-------+      +------+------+
                                  |                     ^
                                  v                     |
                          +---------------+             |
                          | Spark batch   |-------------+
                          | jobs          | materialize
                          | (daily)       |
                          +-------+-------+
                                  |
                                  v
                          +---------------+
                          | BigQuery      | <-- training datasets
                          | (offline store)|    point-in-time joins
                          +---------------+

        Model serving flow:
            request ---> FastAPI ---> Redis get (online features)
                                      + device_type from request
                                      ---> model.predict ---> response

    Q3 -- Point-in-time le plus delicat :
    C'est `product_sales_rank_7d`. Il evolue tres vite (un item viral change
    de rang en heures). Quand on construit le dataset d'entrainement, chaque
    ligne doit contenir le rang du produit TEL QU'IL ETAIT au moment de
    l'evenement, pas le rang actuel. Sinon le modele apprend que "les items
    populaires aujourd'hui etaient populaires hier", ce qui est du data leak.

    Mitigation : le feature store offline doit versionner les valeurs de
    rang par timestamp. La generation du dataset fait un point-in-time join
    sur la plus grande valeur <= timestamp de l'event.

    Q4 -- Degradation si Redis online down :
    Deux strategies combinables :
    1. Fallback features : si Redis unreachable, utiliser des valeurs par
       defaut (moyenne globale, 0, etc.) et logger un warning. Le modele
       continuera a repondre, avec une precision degradee.
    2. Fallback model : router vers un modele plus simple qui n'a pas
       besoin des features manquantes (ex: "top sellers" rule-based).
       C'est un pattern "graceful degradation" : on accepte un modele
       moins bon plutot que pas de reponse.

    En tout cas : NE JAMAIS renvoyer une erreur 500. Une recommandation
    degradee vaut mieux qu'une page vide.
    """


if __name__ == "__main__":
    print("Jour 8 -- Solutions : voir les docstrings.")
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)

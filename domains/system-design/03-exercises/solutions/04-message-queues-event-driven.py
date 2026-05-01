"""
Solutions -- Exercices Jour 4 : Message Queues & Event-Driven

Ce fichier contient les solutions et raisonnements pour les exercices Easy.
Chaque solution est expliquee etape par etape.

Usage:
    python 04-message-queues-event-driven.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Queue ou pas queue ?
# =============================================================================


def easy_1_queue_or_not():
    """Solution pour l'identification du besoin en queue."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Queue ou pas queue ?")
    print(SEPARATOR)

    decisions = [
        (
            "POST /signup + email de bienvenue",
            "QUEUE -- Point-to-point -- SQS ou RabbitMQ",
            "Le SMTP prend 2-5s, pas acceptable dans la requete HTTP. "
            "On push un job 'SendWelcomeEmail' dans une queue, le worker "
            "traite en arriere-plan. Point-to-point car on veut UN seul "
            "email envoye. SQS si AWS, RabbitMQ sinon (simple, fiable)."
        ),
        (
            "GET /users/:id depuis Postgres",
            "PAS DE QUEUE -- Requete synchrone directe",
            "Une query de lecture qui doit repondre immediatement. "
            "Ajouter une queue serait absurde : on ne peut pas attendre "
            "un traitement asynchrone pour un GET. Ici on ajoute plutot "
            "un cache Redis pour accelerer."
        ),
        (
            "IoT 500K events/sec vers 5 equipes",
            "QUEUE -- Pub/sub -- Kafka",
            "Throughput enorme (500K/sec) = Kafka obligatoire (RabbitMQ "
            "sature a ~100K). Pub/sub car 5 consommateurs independants "
            "(chacun un consumer group). Kafka permet aussi le replay "
            "pour onboarder une 6e equipe plus tard sans perdre l'historique."
        ),
        (
            "Ordre de bourse en < 10 us",
            "PAS DE QUEUE -- Chemin critique en memoire",
            "Meme Kafka fait 2-5 ms de latence. Pour < 10 us (microsecondes), "
            "on est dans le HFT : tout en memoire, zero reseau entre le "
            "matching engine et l'ordre. Une queue ajouterait 1000x la "
            "latence autorisee. On utilise des ring buffers in-memory "
            "(type LMAX Disruptor)."
        ),
        (
            "Redimensionnement d'images (5-30s)",
            "QUEUE -- Point-to-point -- SQS, RabbitMQ ou Celery",
            "Traitement long, CPU/IO intensif. On push 'ResizeImage{url}' "
            "dans une queue, une pool de workers traite en parallele. "
            "Point-to-point (chaque image traitee une fois). DLQ obligatoire "
            "pour les images corrompues. C'est le cas d'ecole pour SQS/Celery."
        ),
        (
            "GET /search en < 100 ms",
            "PAS DE QUEUE -- Query directe vers Elasticsearch",
            "Encore un read synchrone. Le search doit repondre dans la "
            "requete HTTP. On n'utilise pas de queue pour servir les reads. "
            "En revanche, l'indexation (quand un document est cree) peut "
            "passer par une queue en amont d'Elasticsearch."
        ),
    ]

    for i, (system, verdict, reason) in enumerate(decisions, 1):
        print(f"\n  {i}. {system}")
        print(f"     Verdict : {verdict}")
        print(f"     Raison  : {reason}")


# =============================================================================
# EASY -- Exercice 2 : Idempotence d'un consommateur
# =============================================================================


def easy_2_idempotent_consumer():
    """Solution : worker at-least-once idempotent via INSERT ON CONFLICT."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Idempotence d'un consommateur Kafka")
    print(SEPARATOR)

    schema = """
    CREATE TABLE payments (
        payment_id   TEXT PRIMARY KEY,          -- UNIQUE naturelle, fournie par le producer
        user_id      TEXT NOT NULL,
        amount_cents INTEGER NOT NULL,
        status       TEXT NOT NULL,             -- 'processed', 'failed', ...
        processed_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    print("\n  1. Schema SQL :")
    print(schema)

    worker_code = '''
    def process_payment_event(event: dict):
        """
        At-least-once + idempotence = effectively exactly-once.

        WHY INSERT ON CONFLICT : c'est atomique cote Postgres. Si deux
        processus tentent la meme insertion en parallele, un seul gagne,
        l'autre recoit un no-op. Pas de race condition possible.
        """
        payment_id = event["payment_id"]
        user_id = event["user_id"]
        amount = event["amount_cents"]

        # Primitive atomique : tente l insert, ignore si existe deja
        result = db.execute("""
            INSERT INTO payments (payment_id, user_id, amount_cents, status)
            VALUES (%s, %s, %s, 'processed')
            ON CONFLICT (payment_id) DO NOTHING
            RETURNING payment_id
        """, (payment_id, user_id, amount))

        if result.rowcount == 0:
            # Deja traite precedemment : NO-OP silencieux
            log.info(f"payment {payment_id} already processed, skipping")
            return

        # Premier traitement : effet de bord (debiter le compte)
        debit_account(user_id, amount)
        # Le commit Kafka se fait apres le return (consumer loop)
    '''
    print("  2. Code du worker :")
    print(worker_code)

    print("\n  3. Scenario : le message est recu 3 fois")
    print("""
     - Reception 1 : INSERT reussit (rowcount=1) -> debit_account() -> commit Kafka
     - Reception 2 : INSERT est no-op (conflict sur payment_id) -> skip silencieux
     - Reception 3 : idem, no-op silencieux

     Resultat : le compte est debite EXACTEMENT 1 fois, meme si le message
     a ete vu 3 fois par le worker.
    """)

    print("  4. Pourquoi 'SELECT puis INSERT' ne marche pas (race condition) :")
    print("""
     Sequence :
       Worker A : SELECT ... WHERE payment_id = 'X'  -> not found
       Worker B : SELECT ... WHERE payment_id = 'X'  -> not found
       Worker A : INSERT ...                          -> OK
       Worker B : INSERT ...                          -> OK (ou erreur tardive)
       Worker A : debit_account(amount)
       Worker B : debit_account(amount)   <- DOUBLE DEBIT !

     Le SELECT et l INSERT sont deux operations distinctes. Entre les deux,
     un autre processus peut inserer. La verification est stale. La seule
     solution correcte est une primitive ATOMIQUE (INSERT ... ON CONFLICT,
     ou UNIQUE constraint + try/except IntegrityError, ou SELECT FOR UPDATE).
    """)

    print("  5. Primitives SQL qui resolvent le probleme :")
    print("""
     - INSERT ... ON CONFLICT (payment_id) DO NOTHING      (Postgres, recommande)
     - INSERT ... ON DUPLICATE KEY UPDATE                  (MySQL)
     - MERGE INTO ...                                      (SQL Server, Oracle)
     - UNIQUE constraint + try/except IntegrityError       (portable)
     - SELECT FOR UPDATE dans une transaction              (verrou plus lourd)
    """)


# =============================================================================
# EASY -- Exercice 3 : Dimensionnement Kafka
# =============================================================================


def easy_3_kafka_sizing():
    """Solution : capacity planning pour le topic ride-sharing."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Dimensionnement Kafka (ride-sharing)")
    print(SEPARATOR)

    # Donnees
    drivers = 200_000
    interval_sec = 4
    event_bytes = 400
    peak_factor = 3
    retention_days = 7
    replication = 3

    # 1. Throughput moyen
    events_per_sec_avg = drivers / interval_sec  # 50 000
    bytes_per_sec_avg = events_per_sec_avg * event_bytes
    mb_per_sec_avg = bytes_per_sec_avg / (1024 * 1024)

    print(f"\n  1. Throughput moyen :")
    print(f"     events/sec = {drivers} drivers / {interval_sec}s = {events_per_sec_avg:,.0f}")
    print(f"     bytes/sec  = {events_per_sec_avg:,.0f} * {event_bytes} B = {bytes_per_sec_avg/1e6:.1f} MB/s")
    print(f"     soit ~{mb_per_sec_avg:.1f} MiB/s")

    # 2. Throughput peak
    events_per_sec_peak = events_per_sec_avg * peak_factor
    mb_per_sec_peak = mb_per_sec_avg * peak_factor
    print(f"\n  2. Throughput peak (x{peak_factor}) :")
    print(f"     events/sec = {events_per_sec_peak:,.0f}")
    print(f"     MiB/s      = {mb_per_sec_peak:.1f}")

    # 3. Partitions
    # Regle : on dimensionne sur le peak. Un consumer traite ~10 MB/s confortablement.
    # On vise aussi le parallelisme souhaite (ex : matching algorithm scale a 30 pods).
    consumer_throughput_mb = 10
    partitions_by_throughput = max(1, int(mb_per_sec_peak / consumer_throughput_mb) + 1)
    # On arrondit au multiple superieur pour laisser de la marge
    recommended_partitions = 60
    print(f"\n  3. Partitions :")
    print(f"     Peak {mb_per_sec_peak:.1f} MiB/s / {consumer_throughput_mb} MiB/s par consumer "
          f"= {partitions_by_throughput} partitions minimum")
    print(f"     Recommande : {recommended_partitions} partitions")
    print(f"     Pourquoi arrondir au-dessus : anticiper la croissance et le parallelisme")
    print(f"     du matching algorithm (30+ pods). Changer le nombre de partitions apres")
    print(f"     coup est douloureux (casse l'ordre par cle).")

    # 4. Stockage brut
    seconds_per_day = 86400
    bytes_per_day = bytes_per_sec_avg * seconds_per_day
    # On utilise la MOYENNE pour le stockage (pas le peak) car le peak est transitoire
    total_bytes_7d = bytes_per_day * retention_days
    total_gb_7d = total_bytes_7d / 1e9
    total_tb_7d = total_bytes_7d / 1e12
    print(f"\n  4. Stockage brut 7 jours :")
    print(f"     bytes/jour = {bytes_per_sec_avg/1e6:.1f} MB/s * 86400 = {bytes_per_day/1e9:.0f} GB/jour")
    print(f"     sur 7j    = {total_gb_7d:,.0f} GB = {total_tb_7d:.1f} TB")

    # 5. Avec replication
    replicated_tb = total_tb_7d * replication
    print(f"\n  5. Stockage avec replication factor {replication} :")
    print(f"     {total_tb_7d:.1f} TB x {replication} = {replicated_tb:.1f} TB (cluster total)")

    # 6. Brokers
    broker_capacity_tb = 2
    # On ne remplit jamais un disque a 100% : viser 60-70% max
    usable_per_broker = broker_capacity_tb * 0.65
    min_brokers = int(replicated_tb / usable_per_broker) + 1
    recommended_brokers = max(min_brokers, 6)  # minimum 6 pour HA sur 3 AZ
    print(f"\n  6. Nombre de brokers :")
    print(f"     {replicated_tb:.1f} TB / ({broker_capacity_tb} TB * 65%) = {min_brokers} brokers min")
    print(f"     Recommande : {recommended_brokers} brokers (minimum pour HA, marge CPU/IO)")
    print(f"     Repartir sur 3 availability zones pour survivre a la perte d'une AZ.")

    print("\n  Remarques finales :")
    print("     - Utiliser 'driver_id' comme cle de partition pour garder l'ordre par driver")
    print("     - Retention=7d permet de rejouer 1 semaine d'events pour debug/backfill")
    print("     - Monitorer le lag du consumer group 'matching' : lag > 10s = alerte")


def main():
    easy_1_queue_or_not()
    easy_2_idempotent_consumer()
    easy_3_kafka_sizing()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 4.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

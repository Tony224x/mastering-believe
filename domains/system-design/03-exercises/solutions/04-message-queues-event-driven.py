"""
Solutions -- Exercices Jour 4 : Message Queues & Event-Driven

Ce fichier contient les solutions et raisonnements pour les exercices
Easy, Medium et Hard. Chaque solution est expliquee etape par etape.

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


# =============================================================================
# MEDIUM -- Exercice 1 : Saga e-commerce
# =============================================================================


def medium_1_saga():
    """Solution : saga choreographed puis orchestrated avec compensations."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Saga pour une commande e-commerce")
    print(SEPARATOR)

    print("""
  1. CHOREOGRAPHED -- flux nominal (events)
     Order Svc     : publie  order.created
     Inventory Svc : consomme order.created   -> reserve  -> publie stock.reserved
     Payment Svc   : consomme stock.reserved  -> debite   -> publie payment.succeeded
     Shipping Svc  : consomme payment.succeeded -> etiquette -> publie shipment.created
     Order Svc     : consomme shipment.created -> statut COMPLETED

  2. COMPENSATIONS (une par etape, en sens inverse)
     shipment rate     -> shipping.cancelled  (annuler l'etiquette)
     paiement rate     -> stock.release       (liberer la reservation)
     reservation ratee -> order.cancelled     (annuler la commande)
     NB : une compensation n'est PAS un rollback ACID, c'est une action
     metier inverse, elle-meme idempotente.

  3. SCENARIO : paiement refuse APRES reservation du stock
     a) order.created          (Order)
     b) stock.reserved         (Inventory)
     c) payment.failed         (Payment -- carte refusee)
     d) Inventory consomme payment.failed -> libere -> stock.released
     e) Order consomme payment.failed (ou stock.released) -> order.cancelled
     f) Notification au client : commande annulee, aucun debit.
     La compensation remonte la chaine dans l'ordre inverse des etapes.

  4. ORCHESTRATED -- machine a etats de l'orchestrateur
     PENDING -> STOCK_RESERVED -> PAID -> SHIPPED -> COMPLETED
        |             |            |
        v             v            v
      FAILED <- COMPENSATING <-----+   (echec a n'importe quelle etape)
     L'orchestrateur appelle chaque service (commande/reponse) et
     enregistre l'etat apres chaque transition (persistence obligatoire :
     il doit survivre a son propre crash et reprendre la saga).

  5. QUAND BASCULER EN ORCHESTRATED ?
     - > 4-5 etapes : le graphe d'events choreographed devient illisible
     - Besoin de visualiser/requeter l'etat d'une saga (support client)
     - Logique conditionnelle (si stock partiel alors...) difficile a
       exprimer en pur event-driven
     Cout accepte : un composant de plus a operer (l'orchestrateur),
     couplage logique centralise.""")


# =============================================================================
# MEDIUM -- Exercice 2 : Backpressure et consumer lag
# =============================================================================


def medium_2_consumer_lag():
    """Solution calculee : throughput, lag accumule, resorption."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Backpressure et consumer lag")
    print(SEPARATOR)

    avg_rate = 20_000          # events/sec en moyenne
    peak_rate = 40_000         # events/sec en pic
    peak_hours = 2
    partitions = 24
    consumers = 8
    per_event_ms = 2           # traitement sequentiel par consumer

    # 1. Throughput max actuel
    per_consumer = 1000 / per_event_ms          # 500 events/sec
    group_max = consumers * per_consumer        # 4 000 events/sec
    print(f"\n  1. Throughput max actuel")
    print(f"     1 consumer = 1000ms / {per_event_ms}ms = {per_consumer:.0f} events/s")
    print(f"     8 consumers = {group_max:,.0f} events/s")
    print(f"     Charge moyenne {avg_rate:,} ev/s -> INSUFFISANT meme hors pic !")
    print(f"     (le systeme accumule du lag en permanence)")

    # Pour la suite on suppose le fix minimal : 24 consumers (1/partition)
    max_consumers = partitions
    group_fixed = max_consumers * per_consumer  # 12 000 events/s
    print(f"\n     Fix minimal : scaler a {max_consumers} consumers (1 par partition)")
    print(f"     -> {group_fixed:,.0f} events/s : OK en moyenne, KO en pic ({peak_rate:,})")

    # 2. Lag accumule pendant le pic (avec 24 consumers)
    deficit = peak_rate - group_fixed
    lag = deficit * peak_hours * 3600
    print(f"\n  2. Lag accumule pendant le pic (avec 24 consumers)")
    print(f"     Deficit = {peak_rate:,} - {group_fixed:,.0f} = {deficit:,.0f} events/s")
    print(f"     Lag fin de pic = {deficit:,.0f} * {peak_hours}h * 3600 = {lag:,.0f} events")

    # 3. Temps de resorption
    surplus = group_fixed - avg_rate
    catchup_hours = lag / surplus / 3600
    print(f"\n  3. Resorption une fois revenu a la moyenne")
    print(f"     Surplus = {group_fixed:,.0f} - {avg_rate:,} = {surplus:,.0f} events/s")
    print(f"     Temps = {lag:,.0f} / {surplus:,.0f} = {lag/surplus:,.0f}s "
          f"= {catchup_hours:.1f} heures")

    print("""
  4. Trois solutions pour tenir le pic
     a) Scaling horizontal : plafonne a 24 consumers (1 par partition).
        Au-dela, les consumers supplementaires restent idle. Pour aller
        plus loin il faudrait re-partitionner (operation lourde).
     b) Batching des appels DB : traiter 100 events par batch divise le
        cout per-event (2ms -> ~0.2ms). Limite : latence de constitution
        du batch + complexite du retry partiel.
     c) Parallelisme intra-consumer (thread pool par consumer) :
        multiplie le debit MAIS l'ordre par cle de partition est perdu
        si deux events de la meme cle partent sur deux threads.
        Mitigation : paralleliser PAR CLE (un worker par cle hashee).

  5. Alerte SLA 30 secondes
     Metrique : consumer lag exprime en TEMPS (lag_events / debit consume),
     pas seulement en events.
     Alerte : lag_seconds > 15s pendant 2 min (warning), > 25s (critical).
     On alerte AVANT 30s pour laisser le temps a l'autoscaling d'agir.""")


# =============================================================================
# MEDIUM -- Exercice 3 : Outbox pattern
# =============================================================================


def medium_3_outbox():
    """Solution : transactional outbox, garanties, polling vs CDC."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Outbox pattern")
    print(SEPARATOR)

    print("""
  1. DEUX SCENARIOS D'INCOHERENCE (dual-write)
     a) Crash apres db.insert, avant kafka.publish :
        la commande existe en DB mais AUCUN event n'est emis.
        Les consommateurs (facturation, stock) ne sauront jamais.
     b) Kafka indisponible (timeout) apres le commit DB : pareil,
        et le retry applicatif n'est pas garanti si le process meurt.

  2. INVERSER L'ORDRE NE RESOUT RIEN
     publish puis insert : crash entre les deux -> event SANS ligne en DB.
     Les consommateurs traitent une commande qui n'existe pas.
     Le probleme de fond : deux systemes, pas de transaction commune.

  3. TRANSACTIONAL OUTBOX
     Schema :
       CREATE TABLE outbox (
           id            BIGSERIAL PRIMARY KEY,
           aggregate_id  TEXT NOT NULL,        -- order_id
           event_type    TEXT NOT NULL,        -- 'order.created'
           payload       JSONB NOT NULL,
           created_at    TIMESTAMPTZ DEFAULT NOW(),
           published_at  TIMESTAMPTZ           -- NULL = pas encore publie
       );

     Write (atomique, UNE transaction) :
       BEGIN;
         INSERT INTO orders (...) VALUES (...);
         INSERT INTO outbox (aggregate_id, event_type, payload)
                VALUES (:order_id, 'order.created', :json);
       COMMIT;

     Relay (process separe, en boucle) :
       rows = SELECT * FROM outbox
              WHERE published_at IS NULL
              ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED;
       for row in rows:
           kafka.publish(row.event_type, row.payload, key=row.aggregate_id)
       UPDATE outbox SET published_at = NOW() WHERE id IN (...);

  4. GARANTIE OBTENUE
     Le relay peut crasher ENTRE publish et UPDATE -> l'event repart au
     prochain cycle : livraison AT-LEAST-ONCE.
     Le consommateur DOIT etre idempotent (cle unique = outbox.id ou
     aggregate_id + event_type + version). C'est le duo standard :
     at-least-once + idempotence = exactly-once EFFECTIF.

  5. POLLING vs CDC (Debezium)
     Polling : simple (un SELECT periodique), latence = intervalle de
       polling (souvent 0.5-5s), charge DB constante, aucune infra en plus.
     CDC : lit le WAL Postgres -> latence quasi temps reel (<100ms),
       zero charge de polling, MAIS infra Kafka Connect/Debezium a operer,
       gestion des slots de replication, montee en competence.
     Regle pratique : polling pour demarrer, CDC quand la latence ou le
     volume l'exigent.""")


# =============================================================================
# HARD -- Exercice 1 : Event backbone d'une plateforme de paiement
# =============================================================================


def hard_1_payment_backbone():
    """Solution chiffree : topologie Kafka, garanties, stockage, failure drill."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Event backbone paiement")
    print(SEPARATOR)

    peak_tps = 30_000
    avg_tps = 4_000
    event_kb = 1
    consumers = 6
    retention_days = 30
    rf = 3

    # 1. Topologie
    peak_mb_s = peak_tps * event_kb / 1024
    min_partitions_throughput = int(peak_mb_s / 10) + 1
    print(f"\n  1. Topologie Kafka")
    print(f"     Throughput pic = {peak_tps:,} ev/s * {event_kb} Ko = {peak_mb_s:.0f} Mo/s")
    print(f"     Minimum par le throughput : ~{min_partitions_throughput} partitions (10 Mo/s/partition)")
    print(f"     MAIS le parallelisme consumer dicte le choix : 60-100 partitions")
    print(f"     (6 consumer groups, le plus lent -- ledger -- doit pouvoir scaler)")
    print(f"     Cle de partition : account_id (ordre garanti par compte)")
    print(f"     RF=3, min.insync.replicas=2, acks=all, enable.idempotence=true")

    print("""
  2. CHAINE DE GARANTIES
     Producer : outbox pattern cote payment-api (la transaction est
       d'abord un fait en DB) + producer idempotent (pas de doublon broker
       en cas de retry reseau).
     Broker   : acks=all + min.insync=2 -> un event acke survit a la
       perte d'un broker.
     Consumer : commit d'offset APRES traitement + traitement idempotent
       (le ledger insere avec une cle unique transaction_id).
     Frontiere : exactly-once Kafka (transactions) seulement pour les
     pipelines Kafka->Kafka internes. Des qu'on touche une DB externe,
     la regle est at-least-once + idempotence. C'est elle qui garantit
     "aucun double debit VISIBLE".

  3. ORDERING ET HOT ACCOUNT
     Cle = account_id : tous les events d'un compte vont dans la meme
     partition -> ordre garanti authorization -> capture -> refund.
     Hot account (un marchand = 5% du trafic = 1 500 ev/s pic) :
     une seule partition le porte -> elle sature (1.5 Mo/s ca passe,
     mais le consumer du ledger sur cette partition devient le goulot).
     Mitigations : sous-cle account_id#bucket pour les comptes marchands
     (l'ordre par paiement individuel suffit, pas par marchand entier),
     ou un topic dedie aux comptes a fort volume.""")

    # 4. Stockage et cout
    avg_gb_day = avg_tps * event_kb * 86_400 / (1024**2)
    online_tb = avg_gb_day * retention_days * rf / 1024
    archive_tb_year = avg_gb_day * 365 / 1024
    print(f"  4. Stockage et cout")
    print(f"     Moyenne/jour = {avg_tps:,} * 1 Ko * 86400 = {avg_gb_day:,.0f} Go/jour")
    print(f"     En ligne 30j x RF3 = {online_tb:,.1f} To sur le cluster")
    print(f"     Archive : ~{archive_tb_year:,.0f} To/an (avant compression) -> S3/tiered storage")
    print(f"     Hypotheses de cout (a poser explicitement) :")
    print(f"       - 9 brokers (HA 3 AZ, marge pic) ~ 9 * 1 200 $/mois = 10 800 $")
    print(f"       - Stockage NVMe inclus + S3 archive ~ 120 To/an * 23 $/To = ~2 800 $/mois")
    print(f"       - Reseau inter-AZ + monitoring ~ 3 000 $/mois")
    print(f"     Total ~ 17 000 $/mois < budget 25 000 $ : OK avec marge")

    print("""
  5. FAILURE DRILL
     a) Broker mort en plein pic :
        - election d'un nouveau leader par partition (quelques secondes)
        - aucun event acke n'est perdu (min.insync=2)
        - le cluster passe en under-replicated -> alerte, re-replication
        - producer : retries automatiques pendant l'election
     b) Consumer ledger : 20 min de lag :
        - SLO 2s de bout en bout = VIOLE -> incident
        - alerte des lag > 10s (leading indicator : lag en secondes)
        - actions auto : scale-out du consumer group (jusqu'a 1/partition)
        - actions humaines : verifier la DB ledger (souvent le vrai goulot),
          activer le batching d'inserts, prioriser ledger sur analytics

  6. TROIS TRADEOFFS EXPLICITES
     1) Cle par account_id plutot que round-robin : ordre garanti par
        compte, MAIS hot partitions possibles (mitigation ci-dessus).
     2) at-least-once + idempotence plutot que transactions exactly-once
        partout : plus simple et plus rapide, MAIS chaque consumer doit
        implementer l'idempotence (discipline d'equipe).
     3) Tiered storage S3 pour les 7 ans plutot que tout sur les brokers :
        cout divise par ~10, MAIS replay des donnees anciennes plus lent
        (heures au lieu de minutes). Acceptable : le replay 30j couvre
        l'operationnel, l'archive sert l'audit.""")


# =============================================================================
# HARD -- Exercice 2 : Event sourcing -- ou pas
# =============================================================================


def hard_2_event_sourcing_migration():
    """Solution : decision argumentee, architecture, plan strangler."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Migration event sourcing -- ou pas")
    print(SEPARATOR)

    print("""
  1. DECISION ARGUMENTEE
     Options croisees avec les contraintes :
     a) Event sourcing COMPLET : repond a tout sur le papier, MAIS
        equipe novice + 200 requetes SQL a preserver + 6 mois de deadline
        -> risque projet maximal. REJETE.
     b) Event sourcing sur le sous-domaine livraisons : interessant a
        moyen terme (5K writes/s, besoin du "statut a 14h mardi").
        Mais ne tient pas seul la deadline de 6 mois.
     c) Audit log via CDC/outbox : capture TOUTES les mutations en events
        immuables SANS toucher au monolithe. Audit trail fiable en
        2-4 mois. Ne donne pas le decouplage ni le replay d'etat complet.
     DECISION : (c) maintenant pour la deadline client, puis (b) sur le
     sous-domaine livraisons une fois l'equipe montee en competence sur
     les events. Le full ES n'a pas de justification metier suffisante
     pour les 7 autres modules.

  2. ARCHITECTURE CIBLE (sous-domaine livraisons, phase 2)
     Event store : Postgres append-only (table events) au debut --
       l'equipe le connait ; EventStoreDB/Kafka si le besoin de debit
       explose. 5K writes/s tiennent sur Postgres partitionne.
     Schema d'un event :
       event_id (uuid), aggregate_id (delivery_id), version (int,
       unique par aggregate), event_type, schema_version, occurred_at,
       actor, payload (jsonb)
       UNIQUE (aggregate_id, version)  <- optimistic concurrency
     Snapshots : toutes les ~200 versions d'un aggregate (une livraison
       depasse rarement 50 events ; le snapshot sert surtout les vehicules,
       aggregates long-lived).
     Projections : consumers asynchrones qui materialisent les read models.

  3. PLAN STRANGLER (sans interruption, SLA 99.9%)
     Phase 0 : CDC sur les tables livraisons -> topic 'delivery-changes'.
               Verite = CRUD. Les events sont DERIVES. (audit OK ici)
     Phase 1 : nouveau service livraisons ecrit en double : il recoit les
               commandes, ecrit des events, ET met a jour l'ancienne table
               (double-write transactionnel via outbox).
               Verite = encore le CRUD ; comparaison continue events vs
               tables (reconciliation job) pour valider.
     Phase 2 : bascule de la verite : l'event store devient la source,
               les tables legacy deviennent une PROJECTION.
               POINT DE NON-RETOUR : quand des workflows s'appuient sur
               des invariants impossibles a reconstruire depuis le CRUD
               (ex : compensation basee sur l'historique fin).
     Phase 3 : extinction des ecritures legacy directes, module par module.

  4. LES 200 REQUETES SQL
     On ne les reecrit PAS : les projections materialisent les MEMES
     tables/colonnes que le schema actuel (read model de compatibilite).
     Le reporting continue de taper les memes vues. Cout accepte :
     latence de projection (secondes) -- acceptable pour du reporting.

  5. REGLES DE SCHEMA EVOLUTION (a poser DES MAINTENANT)
     - Un event publie est IMMUABLE : on ne corrige jamais le passe.
     - Chaque event porte schema_version.
     - Changement additif : nouveaux champs optionnels seulement.
     - Changement structurel : nouveau type d'event (V2) + UPCASTER a la
       lecture (V1 -> V2 a la volee), jamais de migration destructive.
     - Les consumers doivent tolerer les champs inconnus.

  6. CHIFFRAGE (dev-mois, equipe de 12 dont ~4 mobilisables)
     CDC/outbox audit trail     : 2-4 dev-mois   -> tient dans 6 mois  OK
     ES sous-domaine livraisons : 8-12 dev-mois  -> 6-9 mois apres
     ES complet (8 modules)     : 24-40 dev-mois -> 18+ mois, risque max
     La deadline de 6 mois ne laisse qu'une option serieuse : (c) puis (b).""")


def main():
    easy_1_queue_or_not()
    easy_2_idempotent_consumer()
    easy_3_kafka_sizing()
    medium_1_saga()
    medium_2_consumer_lag()
    medium_3_outbox()
    hard_1_payment_backbone()
    hard_2_event_sourcing_migration()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 4.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

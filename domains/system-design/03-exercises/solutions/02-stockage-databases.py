"""
Solutions -- Exercices Jour 2 : Stockage & Databases

Ce fichier contient les solutions calculees pour les exercices Easy, Medium, et Hard.
Chaque solution montre le raisonnement etape par etape.

Usage:
    python 02-stockage-databases.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : SQL ou NoSQL
# =============================================================================

def easy_1_sql_or_nosql():
    """Solution pour le choix SQL vs NoSQL par use case."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : SQL ou NoSQL — Choisis ta DB")
    print(SEPARATOR)

    choices = [
        (
            "Gestion de paie (salaires, bulletins, cotisations)",
            "SQL (PostgreSQL)",
            "Transactions ACID obligatoires : un bulletin de paie implique des calculs "
            "sur salaire brut, cotisations, net a payer. Integrite referentielle entre "
            "employe, contrat, et bulletin. Erreur = consequence legale et financiere."
        ),
        (
            "Cache de resultats de recherche (TTL 5 min)",
            "NoSQL Key-Value (Redis)",
            "Access pattern = GET/SET par cle de recherche. TTL natif. "
            "Latence sub-ms requise. Perte tolerable (c'est un cache). "
            "Aucun besoin de jointures ou de transactions."
        ),
        (
            "Catalogue de recettes (attributs variables)",
            "NoSQL Document (MongoDB)",
            "Schema flexible : chaque recette a des attributs differents "
            "(temps de cuisson, allergenes variables, ingredients en quantites differentes). "
            "Un document JSON par recette contient tout. Pas de jointures necessaires."
        ),
        (
            "Recommandation 'les acheteurs de X ont aussi achete Y'",
            "NoSQL Graph (Neo4j)",
            "Le modele est un graphe : noeuds = utilisateurs + produits, aretes = achats. "
            "La query 'acheteurs de X -> autres achats -> produits' est un traversal "
            "de profondeur 2-3. En SQL, c'est des jointures recursives tres couteuses."
        ),
        (
            "Metriques IoT (200K events/sec, 50K capteurs)",
            "NoSQL Column-family (Cassandra) ou TimescaleDB",
            "Write throughput massif (200K/sec). Time-series data. "
            "Cassandra : LSM-Tree, scaling lineaire, partition par sensor_id. "
            "TimescaleDB si besoin de SQL. Tradeoff : Cassandra scale mieux, "
            "mais TimescaleDB offre des queries SQL et des fonctions d'aggregation natives."
        ),
        (
            "Backoffice gestion de contrats d'assurance",
            "SQL (PostgreSQL)",
            "Integrite referentielle complexe : contrat -> client -> clauses -> sinistres. "
            "Transactions ACID pour les modifications de contrat. "
            "Queries ad hoc frequentes (reporting, recherche). Volume modere."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Choix : {choice}")
        print(f"     Raison : {justification}")


# =============================================================================
# EASY -- Exercice 2 : Index ou pas index ?
# =============================================================================

def easy_2_index_or_not():
    """Solution pour les decisions d'indexation."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Index ou pas index ?")
    print(SEPARATOR)

    decisions = [
        {
            "situation": "Table users (50M lignes), colonne email, WHERE email = ? a chaque login",
            "decision": "OUI — Index unique sur email",
            "justification": (
                "50M lignes et une query executee a chaque login = critique. "
                "Sans index : full scan de 50M lignes a chaque login (~secondes). "
                "Avec index B-Tree : lookup en O(log 50M) = ~26 comparaisons (~ms). "
                "Bonus : index UNIQUE garantit l'unicite de l'email."
            ),
        },
        {
            "situation": "Table logs (500M lignes), colonne level (3 valeurs), 10K inserts/sec",
            "decision": "NON — Pas d'index sur level seul",
            "justification": (
                "Cardinalite = 3 (INFO, WARN, ERROR). L'index ne filtre que 33% en moyenne. "
                "Le query planner preferera un full scan a un index aussi peu selectif. "
                "De plus, 10K inserts/sec signifie que chaque insert doit aussi "
                "mettre a jour l'index -> overhead significatif pour un benefice nul. "
                "Alternative : si on cherche seulement les ERROR (1% du total), "
                "un partial index 'WHERE level = ERROR' serait utile et petit."
            ),
        },
        {
            "situation": "Table orders (10M lignes), WHERE customer_id = ? ORDER BY created_at DESC",
            "decision": "OUI — Index composite (customer_id, created_at DESC)",
            "justification": (
                "Composite index parfait : filtre par customer_id puis trie par created_at. "
                "Le moteur SQL traverse l'index dans l'ordre du clustering key "
                "sans tri supplementaire (index-ordered scan). "
                "Un index sur customer_id seul obligerait un sort sur created_at apres le filtre."
            ),
        },
        {
            "situation": "Table config (50 lignes), colonne key, WHERE key = ?",
            "decision": "NON — Pas necessaire",
            "justification": (
                "50 lignes tiennent dans une seule page disque (~8 Ko). "
                "Un full scan de 50 lignes est plus rapide que de traverser un B-Tree. "
                "L'overhead de maintenance de l'index est inutile pour si peu de donnees."
            ),
        },
        {
            "situation": "Table events (1B lignes), insert-only, 1 batch job de lecture par nuit",
            "decision": "NON pour les index permanents — OUI pour un index temporaire si necessaire",
            "justification": (
                "Insert-only + 1 lecture/nuit = workload 99.99% ecriture. "
                "Chaque index permanent ralentit chacun des milliards d'inserts. "
                "Mieux : creer un index juste avant le batch job nocturne, "
                "puis le supprimer apres (ou utiliser un partitioning par date "
                "pour limiter le scan au jour necessaire)."
            ),
        },
    ]

    for i, d in enumerate(decisions, 1):
        print(f"\n  {i}. {d['situation']}")
        print(f"     Decision : {d['decision']}")
        print(f"     Raison : {d['justification']}")


# =============================================================================
# EASY -- Exercice 3 : Replication — Qui lit quoi ?
# =============================================================================

def easy_3_replication():
    """Solution pour le probleme de replication lag."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Replication — Qui lit quoi ?")
    print(SEPARATOR)

    print(f"""
  1. QUE VOIT L'UTILISATEUR ?
  {'-'*50}
  L'utilisateur voit son ANCIEN nom de profil.
  Pourquoi : il a ecrit sur le leader, mais le follower n'a pas encore
  replique la modification. Le lag moyen est de 200ms, et la lecture
  arrive 100ms apres l'ecriture -> le follower a ~50% de chance
  de ne pas avoir la donnee a jour (et au pic de lag = 2s, c'est certain).

  C'est le probleme classique du "read-your-writes".

  2. SOLUTION SANS FORCER TOUTES LES LECTURES SUR LE LEADER
  {'-'*50}
  Principe du 'read-your-writes consistency' :
  - Quand un utilisateur ECRIT, on enregistre un marqueur
    (timestamp de la derniere ecriture) dans sa session ou un cookie.
  - Quand ce MEME utilisateur LIT, on verifie le marqueur :
    - Si l'ecriture est recente (< seuil de lag max), lire depuis le LEADER
    - Sinon, lire depuis un follower normalement.
  - Seules les lectures de l'utilisateur qui vient d'ecrire vont au leader.
    Les autres utilisateurs lisent toujours depuis les followers.

  3. PSEUDO-CODE
  {'-'*50}""")

    print("""  def handle_read(user_id, key):
      # Recuperer le timestamp de la derniere ecriture de cet utilisateur
      last_write_ts = session.get(f"last_write:{user_id}")
      max_lag = 2000  # 2 secondes = pic de lag observe

      if last_write_ts and (now() - last_write_ts) < max_lag:
          # Ecriture recente : lire depuis le leader pour garantir la fraicheur
          return leader.read(key)
      else:
          # Pas d'ecriture recente : follower OK
          return random_follower.read(key)

  def handle_write(user_id, key, value):
      leader.write(key, value)
      # Marquer la derniere ecriture pour cet utilisateur
      session.set(f"last_write:{user_id}", now())""")

    print(f"""
  4. COUT EN CHARGE SUR LE LEADER
  {'-'*50}
  Si 5% des utilisateurs ecrivent dans un intervalle de 2 secondes,
  alors 5% des reads sont rediriges vers le leader au lieu des followers.

  Exemple concret :
  - 100K QPS total en lecture
  - 5% redirige vers le leader = 5K QPS supplementaires sur le leader
  - Le leader gere deja les writes (~2K QPS) -> total = 7K QPS
  - PostgreSQL gere facilement 50K QPS -> marge largement suffisante

  Le cout est faible car seule une petite fraction des lectures
  (celles des utilisateurs qui viennent d'ecrire) va au leader.
  C'est beaucoup mieux que de router TOUT le trafic de lecture au leader.""")


# =============================================================================
# MEDIUM -- Exercice 1 : Sharding pour une messagerie
# =============================================================================

def medium_1_messaging_sharding():
    """Solution pour le schema de sharding d'une messagerie."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Schema de sharding — Messagerie")
    print(SEPARATOR)

    # 1. Estimation de stockage
    users = 100_000_000
    msgs_per_user_per_day = 50
    msg_size = 500  # octets
    retention_days = 3 * 365  # 3 ans

    total_msgs_per_day = users * msgs_per_user_per_day
    storage_per_day_gb = total_msgs_per_day * msg_size / (1024 ** 3)
    storage_total_tb = storage_per_day_gb * retention_days / 1024
    storage_total_pb = storage_total_tb / 1024

    print(f"\n  1. ESTIMATION DE STOCKAGE")
    print(f"  {'-'*50}")
    print(f"  Messages/jour = {users:,} * {msgs_per_user_per_day} = {total_msgs_per_day:,.0f}")
    print(f"  Stockage/jour = {total_msgs_per_day:,.0f} * {msg_size} B = {storage_per_day_gb:,.0f} Go")
    print(f"  Stockage 3 ans = {storage_per_day_gb:,.0f} Go * {retention_days} = {storage_total_tb:,.0f} To = {storage_total_pb:,.1f} Po")

    print(f"""
  2. PARTITION KEY
  {'-'*50}
  Choix : conversation_id

  Pourquoi conversation_id :
  - L'access pattern principal est "50 derniers messages d'une conversation"
    -> TOUTES les donnees de la query sont dans UNE seule partition
    -> 1 seul noeud interroge, pas de scatter-gather

  Pourquoi PAS user_id :
  - Un message dans un groupe de 500 personnes devrait etre stocke 500 fois
    (une par user_id), ou bien un scatter-gather sur 500 partitions.
  - Les conversations 1-to-1 seraient split entre 2 partitions.

  Pourquoi PAS message_id :
  - Chaque message serait sur une partition differente.
  - "50 derniers messages" = scatter-gather sur 50 partitions -> terrible.

  3. CLUSTERING KEY
  {'-'*50}
  Clustering key : message_id (time-based, type Snowflake ID)
  Ou : created_at DESC + message_id DESC (pour gerer les egalites de timestamp)

  Resultat : les messages d'une conversation sont stockes tries par date.
  "SELECT * WHERE conversation_id = ? ORDER BY message_id DESC LIMIT 50"
  -> lecture sequentielle dans une seule partition, sans tri.

  4. HOT SPOT — GROUPE DE 500 PERSONNES
  {'-'*50}
  Probleme : une conversation tres active (1000 messages/sec) surcharge
  une seule partition.

  Solutions :
  a) Sub-partitioning : partition key = (conversation_id, bucket)
     Ou bucket = message_id % 10 (10 sous-partitions par conversation)
     Cout : "50 derniers messages" = scatter-gather sur 10 buckets.

  b) Write buffering : les messages sont ecrits dans un buffer (Redis/Kafka)
     puis batch-inseres dans Cassandra toutes les 100ms.
     Reduit la charge sur la partition.

  c) En pratique : les groupes de 500+ personnes sont rares (<0.1% des conversations).
     On peut les traiter comme un cas special (shard dediee).

  5. RECHERCHE PAR MOT-CLE
  {'-'*50}
  Cassandra n'est PAS fait pour la recherche full-text.

  Solution : Elasticsearch en async.
  - Chaque message est indexe dans Elasticsearch via un pipeline async (Kafka).
  - Schema ES : conversation_id, message_text, sender_id, timestamp
  - Query : recherche full-text filtree par conversation_id
  - ES retourne les message_ids, puis on les fetch dans Cassandra.

  6. SCHEMA DE TABLE (Cassandra CQL)
  {'-'*50}""")

    print("""  CREATE TABLE messages (
      conversation_id UUID,
      message_id      TIMEUUID,    -- Snowflake-like, inclut le timestamp
      sender_id       UUID,
      content         TEXT,
      content_type    TEXT,         -- 'text', 'image', 'file'
      metadata        MAP<TEXT, TEXT>,
      created_at      TIMESTAMP,
      PRIMARY KEY (conversation_id, message_id)
  ) WITH CLUSTERING ORDER BY (message_id DESC);

  -- Index secondaire pour lister les conversations d'un user
  CREATE TABLE user_conversations (
      user_id          UUID,
      last_activity_at TIMESTAMP,
      conversation_id  UUID,
      PRIMARY KEY (user_id, last_activity_at)
  ) WITH CLUSTERING ORDER BY (last_activity_at DESC);""")


# =============================================================================
# MEDIUM -- Exercice 2 : Migration SQL vers NoSQL
# =============================================================================

def medium_2_migration_analysis():
    """Solution pour l'analyse de migration PostgreSQL -> DynamoDB."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Migration SQL vers NoSQL — Analyse")
    print(SEPARATOR)

    print(f"""
  1. AVANTAGES CONCRETS DE DYNAMODB POUR CE CAS
  {'-'*50}
  a) Write scaling natif : DynamoDB auto-scale les writes.
     Les 95% de capacite write de PostgreSQL ne seront plus un probleme.
  b) Replication lag inexistant pour les reads eventually consistent
     (DynamoDB replique nativement sur 3 AZs).
  c) Latence previsible : single-digit ms quel que soit le volume.
  d) Pas de maintenance serveur (managed service) : plus de vacuum,
     plus d'autovacuum qui bloque les writes.

  2. COUTS CACHES DE LA MIGRATION (au moins 5)
  {'-'*50}
  a) Rewrite des queries : tous les SQL complexes (jointures, subqueries,
     GROUP BY) doivent etre recris en access patterns DynamoDB.
     Chaque jointure = une denormalisation ou un GSI (Global Secondary Index).

  b) Migration des donnees : 2 To et 100M de lignes a migrer sans downtime.
     Dual-write pendant la transition. Validation de coherence.
     Duree estimee : 2-4 semaines de migration + 2 semaines de dual-run.

  c) Perte de jointures : "orders JOIN order_items JOIN products"
     -> il faut soit denormaliser (dupliquer products dans chaque order_item),
     soit faire 3 requetes separees cote application.

  d) Formation equipe : l'equipe connait SQL/PostgreSQL.
     DynamoDB a un modele mental different (single-table design, GSI, LSI).
     Courbe d'apprentissage de 1-2 mois pour l'equipe.

  e) Cout financier : DynamoDB facture les reads/writes (WCU/RCU).
     A 10K TPS peak, le cout peut etre > $10K/mois.
     PostgreSQL sur EC2/RDS est souvent moins cher pour un workload previsible.

  f) Lock-in AWS : DynamoDB est proprietaire. Pas de portabilite.

  3. CE QUE TU PERDS EN QUITTANT POSTGRESQL
  {'-'*50}
  a) Transactions ACID multi-tables : DynamoDB transactions = max 25 items,
     pas de transaction cross-table complexe.
  b) SQL ad hoc : plus de "SELECT ... WHERE ... GROUP BY ... HAVING ..."
     pour le debugging ou le reporting. Il faut pre-definir tous les access patterns.
  c) Ecosysteme d'outils : pgAdmin, pg_dump, EXPLAIN ANALYZE, extensions
     (PostGIS, pg_trgm, pgvector). DynamoDB a des outils limites.
  d) Schema enforcement : DynamoDB n'a pas de schema rigide.
     Risque de donnees malformees en production.

  4. ALTERNATIVE SANS QUITTER POSTGRESQL
  {'-'*50}
  Les 3 problemes identifies :

  Probleme 1 : Writes a 95% capacite
  Solution : PostgreSQL partitioning par date (monthly) + archivage des vieilles partitions.
  Reduit la taille de la table active. Alternative : Citus pour le sharding horizontal.

  Probleme 2 : Replication lag > 10s
  Solution : Passer en replication synchrone pour 1 follower (synchronous_commit).
  Ou ajouter des followers (plus de replicas = moins de charge par replica).
  Ou investiguer la cause du lag (vacuum, long-running queries sur les followers).

  Probleme 3 : Jointures > 5 secondes
  Solution : Materialized views pre-calculees pour les queries frequentes.
  Ou CQRS : un read model denormalise (table pre-jointe) mis a jour par triggers/CDC.
  Ou indexes manquants (EXPLAIN ANALYZE pour identifier les full scans).

  5. DECISION FRAMEWORK
  {'-'*50}
  Migrer vers DynamoDB SI ET SEULEMENT SI :
  - Les alternatives PostgreSQL ont ete essayees et echouent
  - Le workload est principalement key-value (> 80% des queries sont par PK)
  - L'equipe a au moins 1 personne experimentee avec DynamoDB
  - Le budget DynamoDB ($) est < 2x le cout PostgreSQL actuel
  - Les jointures complexes peuvent etre eliminees par denormalisation
  - Le planning permet 2+ mois de migration

  Si UNE de ces conditions n'est pas remplie : rester sur PostgreSQL
  et optimiser (Citus, partitioning, CQRS, materialized views).""")


# =============================================================================
# MEDIUM -- Exercice 3 : Index strategy pour dashboard analytics
# =============================================================================

def medium_3_index_strategy():
    """Solution pour la strategie d'indexation."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Index Strategy — Dashboard Analytics")
    print(SEPARATOR)

    print(f"""
  1. INDEX POUR CHAQUE QUERY
  {'-'*50}

  Query 1 (90% du trafic) :
  SELECT count(*), sum(amount) FROM events
  WHERE tenant_id = ? AND event_type = ? AND created_at BETWEEN ? AND ?

  Index : CREATE INDEX idx_events_q1
          ON events (tenant_id, event_type, created_at)
          INCLUDE (amount);

  Ordre : tenant_id (egalite) -> event_type (egalite) -> created_at (range)
  Regle : les colonnes d'egalite en premier, la colonne de range en dernier.
  Le INCLUDE (amount) fait un covering index (voir Q2).

  Query 2 (8% du trafic) :
  SELECT * FROM events
  WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT 20

  Index : CREATE INDEX idx_events_q2
          ON events (tenant_id, user_id, created_at DESC);

  Pas de covering index possible ici (SELECT *).

  Query 3 (2% du trafic) :
  SELECT DISTINCT event_type FROM events
  WHERE tenant_id = ? AND created_at > NOW() - INTERVAL '24h'

  Index : pas d'index dedie necessaire. L'index idx_events_q1 couvre
  partiellement (tenant_id, event_type). Ou bien un index :
  CREATE INDEX idx_events_q3
  ON events (tenant_id, created_at) INCLUDE (event_type);

  Mais a 2% du trafic, le cout d'un index dedie est discutable.
  Le partitioning par date (voir Q5) resoudra ce probleme plus elegamment.

  2. COVERING INDEX
  {'-'*50}
  Oui, pour Query 1 :
  Index (tenant_id, event_type, created_at) INCLUDE (amount)

  La query a besoin de : tenant_id, event_type, created_at (filtres) + amount (aggregation).
  Toutes ces colonnes sont dans l'index -> index-only scan.
  Le moteur SQL ne lit JAMAIS la table, uniquement l'index.
  Gain : evite les random I/O vers les pages de la table (heap).

  3. OVERHEAD EN ESPACE DISQUE
  {'-'*50}""")

    # Estimation de la taille de la table
    row_size_estimate = 16 + 16 + 50 + 16 + 12 + 200 + 8  # UUID + UUID + varchar + UUID + decimal + JSONB + timestamp
    # ~318 bytes par ligne, mais avec overhead PostgreSQL (~24 bytes header) -> ~342 bytes

    msgs_per_sec = 50_000
    rows_per_day = msgs_per_sec * 86_400
    rows_per_month = rows_per_day * 30

    table_size_per_month_gb = rows_per_month * 342 / (1024 ** 3)

    print(f"  Taille estimee par ligne : ~342 bytes (colonnes + header PostgreSQL)")
    print(f"  Lignes/jour = {msgs_per_sec:,} * 86400 = {rows_per_day:,.0f}")
    print(f"  Lignes/mois = {rows_per_month:,.0f}")
    print(f"  Taille table/mois = {table_size_per_month_gb:,.0f} Go")

    # Chaque index ~ 30% de la taille des colonnes indexees
    # idx_q1 : (UUID 16 + varchar 50 + timestamp 8 + decimal 12) = 86 bytes/entry
    idx_q1_size = rows_per_month * 86 / (1024 ** 3)
    # idx_q2 : (UUID 16 + UUID 16 + timestamp 8) = 40 bytes/entry
    idx_q2_size = rows_per_month * 40 / (1024 ** 3)

    print(f"\n  Index idx_events_q1 : ~{idx_q1_size:,.0f} Go/mois")
    print(f"  Index idx_events_q2 : ~{idx_q2_size:,.0f} Go/mois")
    print(f"  Total indexes       : ~{idx_q1_size + idx_q2_size:,.0f} Go/mois")
    print(f"  Ratio indexes/table : ~{(idx_q1_size + idx_q2_size) / table_size_per_month_gb * 100:.0f}%")

    print(f"""
  4. IMPACT SUR LES 50K INSERTS/SEC
  {'-'*50}
  Chaque INSERT doit mettre a jour :
  - La table (1 write)
  - idx_events_q1 (1 write B-Tree)
  - idx_events_q2 (1 write B-Tree)
  = 3 writes par INSERT au lieu de 1

  Avec 2 index : throughput d'insert reduit de ~30-50%
  (depend du cache, de la taille des index, et du hardware).

  Si les 50K inserts/sec sont critiques :
  - Option 1 : batch inserts (COPY au lieu de INSERT)
  - Option 2 : creer les index seulement sur les partitions froides (voir Q5)
  - Option 3 : accepter la degradation (30-50% de 50K = 25-35K inserts/sec
    avec index, ce qui reste souvent suffisant)

  5. PARTITIONING STRATEGY
  {'-'*50}
  Type : Range partitioning par created_at
  Granularite : mensuelle (chaque mois = une partition)

  CREATE TABLE events (...) PARTITION BY RANGE (created_at);
  CREATE TABLE events_2026_01 PARTITION OF events
      FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
  -- Repeter pour chaque mois

  Avantages :
  - Les queries avec WHERE created_at BETWEEN ... scannent uniquement
    les partitions pertinentes (partition pruning)
  - Les vieilles partitions peuvent etre archivees (detach + move to cold storage)
  - Les index sont par partition (plus petits, plus rapides a maintenir)
  - VACUUM est plus rapide par partition

  6. QUAND PASSER A CLICKHOUSE
  {'-'*50}
  PostgreSQL atteint ses limites pour l'analytics quand :
  - La table depasse ~1B lignes (les aggregations prennent > 10s)
  - Les queries analytiques (GROUP BY, SUM, COUNT) sont > 50% du trafic
  - Le p99 des queries analytiques depasse le SLO (ex: > 5s)
  - Le workload analytique degrade les performances des inserts

  ClickHouse :
  - Stockage colonnaire : les aggregations sur 1B lignes prennent 100ms-1s
  - Compression 10-20x (colonnaire compresse mieux que row-based)
  - Pas de transactions, pas de updates -> insert-only = parfait pour events

  Strategie de transition :
  - Garder PostgreSQL pour les inserts (source of truth)
  - CDC (Change Data Capture) vers ClickHouse pour les analytics
  - Le dashboard lit depuis ClickHouse, les writes vont dans PostgreSQL""")


# =============================================================================
# HARD -- Exercice 1 : Stockage pour systeme de paiement (esquisse)
# =============================================================================

def hard_1_payment_storage():
    """Solution esquissee pour le systeme de paiement multi-region."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Systeme de paiement multi-region")
    print(SEPARATOR)

    # Estimation de stockage
    tps = 10_000
    secs_per_year = 365.25 * 24 * 3600
    txns_per_year = tps * secs_per_year  # Moyenne, pas pic
    # En realite, pic != moyenne. Utilisons un facteur de 0.3 (moyenne = 30% du pic)
    avg_tps = tps * 0.3  # 3K TPS moyen
    txns_per_year_avg = avg_tps * secs_per_year

    # Taille d'une transaction
    # payment_id (16) + amount (12) + currency (3) + merchant_id (16) + customer_id (16)
    # + status (10) + created_at (8) + updated_at (8) + metadata (200) + indexes overhead
    txn_size = 16 + 12 + 3 + 16 + 16 + 10 + 8 + 8 + 200  # ~289 bytes
    txn_with_overhead = txn_size * 2  # Overhead PostgreSQL + indexes ~ 2x

    # Audit log : chaque changement d'etat = 1 entry
    # En moyenne 3 changements par transaction (pending -> confirmed -> settled)
    audit_entries_per_txn = 3
    audit_entry_size = 200  # payment_id + old_state + new_state + timestamp + actor

    print(f"\n  ESTIMATION DE STOCKAGE")
    print(f"  {'-'*50}")
    print(f"  TPS pic : {tps:,} | TPS moyen (30% du pic) : {avg_tps:,.0f}")
    print(f"  Transactions/an : {txns_per_year_avg:,.0f}")
    print(f"  Taille/txn (avec overhead) : {txn_with_overhead} bytes")

    storage_txn_7y_tb = txns_per_year_avg * 7 * txn_with_overhead / (1024 ** 4)
    storage_audit_7y_tb = txns_per_year_avg * 7 * audit_entries_per_txn * audit_entry_size / (1024 ** 4)

    print(f"  Stockage transactions 7 ans : {storage_txn_7y_tb:,.0f} To")
    print(f"  Stockage audit log 7 ans    : {storage_audit_7y_tb:,.0f} To")
    print(f"  Total                       : {storage_txn_7y_tb + storage_audit_7y_tb:,.0f} To")

    print(f"""
  1. CHOIX DE DB
  {'-'*50}

  a) Transactions de paiement : PostgreSQL (avec Citus pour le sharding)
     - ACID obligatoire : un paiement ne peut pas etre "partiellement cree"
     - State machine avec contraintes CHECK sur les transitions
     - Strong consistency obligatoire
     - Citus pour le sharding horizontal quand on depasse 1 noeud

  b) Dashboard analytics : ClickHouse
     - Aggregations rapides (volume/jour, montant total, taux de succes)
     - Les queries analytics ne doivent PAS impacter les transactions OLTP
     - CDC depuis PostgreSQL -> ClickHouse (Debezium)
     - Eventual consistency acceptable (dashboard rafraichi toutes les 10s)

  c) Audit log immutable : Apache Kafka + S3/GCS
     - Kafka : log append-only, immutable, ordonne, replique
     - Chaque changement d'etat publie un event dans Kafka
     - Kafka -> S3 (archivage long terme 7 ans, Parquet format)
     - Kafka retention : 30 jours (pour le replay si necessaire)
     - S3 : compliance, immutabilite (Object Lock), cout faible

  2. SCHEMA DE DONNEES
  {'-'*50}""")

    print("""  -- Table principale des paiements
  CREATE TABLE payments (
      payment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      merchant_id   UUID NOT NULL,
      customer_id   UUID NOT NULL,
      amount        DECIMAL(12,2) NOT NULL CHECK (amount > 0),
      currency      VARCHAR(3) NOT NULL,
      status        VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','confirmed','settled','failed','refunded')),
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      region        VARCHAR(2) NOT NULL  -- 'EU' ou 'US' pour GDPR
  );

  -- Table des refunds
  CREATE TABLE refunds (
      refund_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      payment_id    UUID NOT NULL REFERENCES payments(payment_id),
      amount        DECIMAL(12,2) NOT NULL CHECK (amount > 0),
      reason        TEXT,
      status        VARCHAR(20) NOT NULL DEFAULT 'pending',
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  -- State machine : transitions valides
  -- pending -> confirmed (payment provider confirms)
  -- pending -> failed (payment provider rejects)
  -- confirmed -> settled (T+1 settlement)
  -- confirmed -> refunded (merchant initiates refund)
  -- settled -> refunded (post-settlement refund)
  -- Enforce via trigger ou application-level check""")

    print(f"""
  3. SHARDING STRATEGY
  {'-'*50}
  Shard key : payment_id (UUID, distribution uniforme par hash)

  Pourquoi payment_id :
  - Chaque payment est un agregat independant (pas de jointure cross-payment)
  - Les writes sont distribues uniformement (UUID = random)
  - Le lookup par payment_id est le pattern le plus frequent

  Probleme : "lister les paiements d'un merchant"
  - payment_id comme shard key = scatter-gather sur TOUTES les shards
  - Solution : Global Secondary Index (GSI) par merchant_id
    Ou : table denormalisee merchant_payments (merchant_id, payment_id, created_at)
    avec shard key = merchant_id

  Nombre de shards :
  - PostgreSQL gere ~5K TPS par noeud (avec les transactions et index)
  - 10K TPS pic / 5K = 2 shards minimum
  - Avec marge (3x) : 6 shards
  - Start with 8 shards (puissance de 2, marge pour la croissance)

  4. REPLICATION MULTI-REGION
  {'-'*50}
  Contrainte GDPR : les donnees EU ne quittent pas l'EU.

  Architecture :
  - Region EU : shards pour les payments ou region='EU'
  - Region US : shards pour les payments ou region='US'
  - Pas de replication cross-region des donnees de paiement
  - Le routage est fait par l'API Gateway selon le merchant/customer region

  Consistance :
  - Intra-region : replication synchrone (1 leader + 1 sync follower)
  - Cross-region : AUCUNE replication des transactions
  - Le dashboard analytics peut agreger des deux regions
    (ClickHouse replicas en read-only dans chaque region)

  Latence : pas de compromis cross-region pour les transactions.
  Chaque transaction est locale a sa region.

  5. DISPONIBILITE 99.99%
  {'-'*50}
  SPOFs identifies :
  a) Leader PostgreSQL -> Failover automatique (Patroni + etcd)
  b) API Gateway -> Multi-AZ, auto-scaling
  c) Kafka cluster -> 3+ brokers, replication factor 3
  d) DNS -> Route53 health checks, multi-region failover

  Mecanismes :
  - Patroni : failover PostgreSQL automatique en < 30s
  - Health checks sur chaque composant (5s interval)
  - Circuit breaker sur les appels inter-services
  - Chaos engineering : Chaos Monkey, Litmus, random pod kills weekly

  6. ARCHIVAGE
  {'-'*50}
  - Partitioning par mois sur la table payments
  - Donnees > 1 an : detach partition, export vers S3 (Parquet)
  - Donnees > 3 ans : tier cold (S3 Glacier)
  - Index uniquement sur les partitions recentes (< 1 an)
  - Les queries sur les anciennes donnees passent par Athena/Presto sur S3""")


# =============================================================================
# HARD -- Exercice 2 : Post-mortem migration (analyse textuelle)
# =============================================================================

def hard_2_postmortem():
    """Solution pour le post-mortem de la migration qui a casse la production."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem — Migration sessions PG -> Redis")
    print(SEPARATOR)

    print(f"""
  1. ROOT CAUSE ANALYSIS
  {'-'*50}

  Cause racine : dimensionnement memoire Redis insuffisant pour 800M sessions,
  combine avec la suppression prematuree du fallback PostgreSQL.

  Facteurs contributifs :

  a) Pas de capacity planning :
     800M sessions * taille_moyenne = ? Go.
     Si une session fait 500 bytes en moyenne : 800M * 500 = 400 Go.
     Le Redis provisionne avait probablement 64-128 Go.
     Detection : un simple calcul avant la migration aurait montre le gap.

  b) Pas de load test realiste :
     Tester avec 100K sessions ne revele pas un probleme a 800M.
     Detection : load test avec le volume reel (ou un pourcentage significatif)
     sur un Redis de meme taille que la prod.

  c) Suppression prematuree du double-write (15:00, 1h apres le debut) :
     "Tout semble OK" apres 1h ne prouve rien. Les sessions s'accumulent
     progressivement. Le probleme OOM arrive quand la RAM est pleine, pas avant.
     Detection : attendre au moins 24-48h avant de supprimer le fallback.
     Definir un critere objectif ("quand 95%+ du trafic sessions est stable
     sur Redis pendant 48h, supprimer le double-write").

  d) Pas de monitoring sur l'usage memoire Redis :
     OOM a 17:00 = la memoire montait depuis 14:00 mais personne ne regardait.
     Detection : alerte sur used_memory > 80% de maxmemory.
     Alerte sur evicted_keys > 0.

  e) Pas de rollback plan teste :
     A 17:10, le rollback echoue car les nouvelles sessions ne sont pas dans PG.
     Detection : inclure "rollback plan" dans la checklist de migration.
     Tester le rollback AVANT de commencer la migration.

  2. ANALYSE DES DECISIONS
  {'-'*50}

  A quel moment s'arreter :
  - A 15:00 ("tout semble OK"). 1 heure n'est PAS suffisant pour valider
    une migration de 800M sessions. Le critere "semble OK" est subjectif.
  - Regle : pas de suppression du fallback avant 48h + metriques objectives OK.

  Pourquoi "tout semble OK" etait dangereux :
  - Les sessions se remplissent progressivement (TTL = 30 min).
  - En 1h, seules les sessions actives sont dans Redis (~10-20% du total).
  - Les 800M sessions seront dans Redis en ~30 min * facteur de renouvellement.
  - Le vrai test est quand Redis atteint le steady state (24-48h).

  Pourquoi le rollback a echoue :
  - Le double-write vers PostgreSQL a ete supprime a 15:00.
  - Toute session creee entre 15:00 et 17:10 n'existe QUE dans Redis.
  - Rollback vers PG = perdre 2h10 de sessions.
  - C'est exactement pourquoi le fallback ne doit jamais etre supprime
    avant la validation complete.

  3. CORRECTIONS
  {'-'*50}

  a) Capacity planning : calculer AVANT la migration.
     800M sessions * 500 bytes = 400 Go minimum.
     Avec overhead Redis (~2x pour les structures internes) = 800 Go.
     Provisionner 1 To de RAM (Redis Cluster de 10+ noeuds a 128 Go chacun).

  b) Load test : reproduire le volume reel.
     Script qui cree 800M sessions dans un Redis de test.
     Mesurer : temps, memoire, latence, throughput.

  c) Migration progressive :
     Phase 1 : double-write PG + Redis (100% des sessions)
     Phase 2 : canary reads (5% du trafic lit depuis Redis, 95% depuis PG)
     Phase 3 : augmenter progressivement (25%, 50%, 75%, 100%)
     Phase 4 : attendre 48h a 100% reads Redis + double-write PG
     Phase 5 : supprimer le double-write UNIQUEMENT si metriques OK 48h

  d) Monitoring :
     - Alerte : used_memory > 70% de maxmemory (warning)
     - Alerte : used_memory > 85% de maxmemory (critical)
     - Alerte : evicted_keys > 0 (immediate, car eviction = perte de sessions)
     - Dashboard : nombre de sessions dans Redis vs PostgreSQL

  4. ARCHITECTURE REDIS CIBLE
  {'-'*50}

  Topology :
  - Redis Cluster (pas Sentinel) : 10+ noeuds master, chacun 128 Go RAM
  - 1 replica par master (20 noeuds total)
  - Multi-AZ pour la resilience

  Persistence :
  - AOF (Append Only File) : chaque ecriture est logguee
  - fsync = everysec (compromis performance/durabilite : perte max 1 seconde)
  - RDB snapshot toutes les heures (pour le recovery rapide)

  Eviction policy :
  - maxmemory-policy = noeviction
  - POURQUOI : les sessions ne doivent JAMAIS etre evictees silencieusement.
  - Si Redis est plein, les nouvelles ecritures echouent avec une erreur.
  - L'application detecte l'erreur et fallback sur PostgreSQL.
  - Alternative : allkeys-lru si la perte de sessions anciennes est acceptable.
    MAIS : un utilisateur deconnecte sans raison = mauvaise UX.

  Dimensionnement RAM :""")

    sessions = 800_000_000
    session_size_bytes = 500  # bytes moyen par session
    redis_overhead_factor = 2.0  # overhead structures internes Redis
    total_ram_gb = sessions * session_size_bytes * redis_overhead_factor / (1024 ** 3)
    nodes_128gb = math.ceil(total_ram_gb / 128)

    print(f"  800M sessions * 500 bytes * 2x overhead = {total_ram_gb:,.0f} Go")
    print(f"  Noeuds necessaires (128 Go/noeud) = {nodes_128gb}")
    print(f"  Avec replicas : {nodes_128gb * 2} noeuds total")
    print(f"  Marge 30% : {math.ceil(nodes_128gb * 1.3)} masters + {math.ceil(nodes_128gb * 1.3)} replicas")

    print(f"""
  Garder PostgreSQL comme fallback :
  OUI, pendant au moins 3-6 mois apres la migration complete.
  - Double-write permanent (le cout est faible : PG gere facilement les writes)
  - Si Redis tombe : fallback transparent vers PG (latence 5-15ms au lieu de <1ms)
  - Monitoring : comparer le nombre de sessions PG vs Redis en continu

  5. CHECKLIST DE MIGRATION GENERIQUE
  {'-'*50}

  AVANT la migration :
  [ ] Capacity planning : calcul du volume reel sur la cible
  [ ] Load test sur la cible avec le volume reel (ou 50%+)
  [ ] Rollback plan documente et TESTE (dry-run du rollback)
  [ ] Monitoring en place AVANT le debut (alertes sur la cible)
  [ ] Runbook d'incident : qui appeler, quoi faire si ca casse
  [ ] Communication equipe : tous informes du plan et du timing

  PENDANT la migration :
  [ ] Double-write active (source + cible)
  [ ] Canary reads progressifs (5% -> 25% -> 50% -> 100%)
  [ ] Validation de coherence : comparer les reads source vs cible
  [ ] Monitoring continu des metriques cles (latence, erreurs, ressources)
  [ ] Checkpoint a chaque etape : "continuer ou rollback ?"

  APRES la migration :
  [ ] Dual-run minimum 48h a 100% reads sur la cible
  [ ] Ne PAS supprimer le fallback avant validation complete
  [ ] Criteres objectifs de succes definis et mesures
  [ ] Post-mortem planifie (meme si tout se passe bien)
  [ ] Documentation mise a jour (architecture, runbooks, monitoring)""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute toutes les solutions."""
    print("\n" + "#" * 60)
    print("#  SOLUTIONS -- JOUR 2 : STOCKAGE & DATABASES")
    print("#" * 60)

    # Easy
    easy_1_sql_or_nosql()
    easy_2_index_or_not()
    easy_3_replication()

    # Medium
    medium_1_messaging_sharding()
    medium_2_migration_analysis()
    medium_3_index_strategy()

    # Hard
    hard_1_payment_storage()
    hard_2_postmortem()

    print("\n" + "#" * 60)
    print("#  FIN DES SOLUTIONS")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

"""
Solutions -- Exercices Jour 1 : Principes fondamentaux

Ce fichier contient les solutions calculees pour les exercices Easy, Medium, et Hard.
Chaque solution montre le raisonnement etape par etape.

Usage:
    python 01-principes-fondamentaux.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Estimation QPS d'un service de notifications
# =============================================================================

def easy_1_notifications():
    """Solution pour l'estimation du service de notifications."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Service de notifications push")
    print(SEPARATOR)

    dau = 5_000_000
    notifs_per_user = 8
    payload_bytes = 500
    peak_factor = 4
    retention_days = 30

    # 1. QPS moyen
    total_notifs_per_day = dau * notifs_per_user  # 40M notifs/jour
    qps_avg = total_notifs_per_day / 86_400
    print(f"\n  1. QPS moyen")
    print(f"     Total notifs/jour = {dau:,} * {notifs_per_user} = {total_notifs_per_day:,}")
    print(f"     QPS moyen = {total_notifs_per_day:,} / 86400 = {qps_avg:,.0f} req/s")

    # 2. QPS pic
    qps_peak = qps_avg * peak_factor
    print(f"\n  2. QPS pic")
    print(f"     QPS pic = {qps_avg:,.0f} * {peak_factor} = {qps_peak:,.0f} req/s")

    # 3. Bande passante sortante en pic
    # Attention : Mbps = megabits per second (pas megabytes)
    bandwidth_bytes_per_sec = qps_peak * payload_bytes
    bandwidth_mbps = bandwidth_bytes_per_sec * 8 / 1_000_000  # bytes -> bits -> megabits
    print(f"\n  3. Bande passante sortante (pic)")
    print(f"     = {qps_peak:,.0f} req/s * {payload_bytes} bytes * 8 bits/byte")
    print(f"     = {bandwidth_mbps:,.1f} Mbps")
    print(f"     = {bandwidth_mbps / 1000:.2f} Gbps")

    # 4. Stockage pour 30 jours
    storage_per_day_bytes = total_notifs_per_day * payload_bytes
    storage_per_day_gb = storage_per_day_bytes / (1024**3)
    storage_total_gb = storage_per_day_gb * retention_days
    print(f"\n  4. Stockage (30 jours)")
    print(f"     Par jour = {total_notifs_per_day:,} * {payload_bytes} = {storage_per_day_bytes / 1e9:.1f} Go")
    print(f"     30 jours = {storage_total_gb:.1f} Go")


# =============================================================================
# EASY -- Exercice 2 : CP ou AP
# =============================================================================

def easy_2_cp_ap():
    """Solution pour le choix CP vs AP."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : CP ou AP -- Choisis ton camp")
    print(SEPARATOR)

    choices = [
        (
            "Vote en ligne (election nationale)",
            "CP",
            "Un double vote ou un vote perdu est inacceptable. "
            "Mieux vaut refuser temporairement un votant que compter un vote en double."
        ),
        (
            "Compteur de vues YouTube",
            "AP",
            "Un compteur inexact de quelques unites est invisible pour l'utilisateur. "
            "L'indisponibilite du compteur bloquerait l'affichage de la video."
        ),
        (
            "Gestion d'inventaire e-commerce",
            "CP",
            "Vendre un produit en rupture de stock = commande annulee = client furieux. "
            "Mieux vaut afficher 'indisponible' temporairement."
        ),
        (
            "Feed d'actualites (reseau social)",
            "AP",
            "Un post qui apparait 2 secondes en retard est invisible. "
            "Un feed qui ne charge pas = utilisateur qui part."
        ),
        (
            "Transfert d'argent bancaire",
            "CP",
            "Un solde incorrect = perte financiere. Les transactions doivent etre ACID. "
            "Une indisponibilite temporaire est preferee a un debit en double."
        ),
        (
            "Cache DNS",
            "AP",
            "Un enregistrement DNS legerement obsolete (TTL) est tolerable. "
            "Un DNS indisponible = internet inaccessible pour les utilisateurs."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Choix : {choice}")
        print(f"     Raison : {justification}")

    print(f"\n  Note : L'inventaire e-commerce est un cas nuance. Certains systemes")
    print(f"  tolerent un overselling de 0.1% et reconciled apres (AP avec compensation).")
    print(f"  La reponse depend du contexte metier exact.")


# =============================================================================
# EASY -- Exercice 3 : Les nines en pratique
# =============================================================================

def easy_3_nines():
    """Solution pour les calculs de SLA."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Les nines en pratique")
    print(SEPARATOR)

    uptime_pct = 99.95
    downtime_fraction = 1 - uptime_pct / 100  # 0.0005

    # 1. Downtime autorise
    seconds_per_month = 30 * 24 * 3600  # ~2.592M secondes
    seconds_per_day = 24 * 3600  # 86400 secondes
    dt_month = downtime_fraction * seconds_per_month
    dt_day = downtime_fraction * seconds_per_day

    print(f"\n  1. Downtime autorise pour SLA {uptime_pct}%")
    print(f"     Par mois = {downtime_fraction} * {seconds_per_month:,} = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     Par jour = {downtime_fraction} * {seconds_per_day:,} = {dt_day:.1f}s")

    # 2. Maintenance hebdo de 15 min
    maintenance_per_month = 15 * 4  # 4 dimanches par mois, en minutes
    maintenance_seconds = maintenance_per_month * 60
    print(f"\n  2. Maintenance 15 min/dimanche = {maintenance_per_month} min/mois = {maintenance_seconds}s/mois")
    print(f"     Budget downtime/mois = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     Maintenance seule    = {maintenance_seconds}s = {maintenance_per_month} min")
    if maintenance_seconds > dt_month:
        print(f"     INCOMPATIBLE : maintenance ({maintenance_per_month} min) > budget ({dt_month / 60:.1f} min)")
    else:
        remaining = dt_month - maintenance_seconds
        print(f"     Compatible : il reste {remaining:.0f}s = {remaining / 60:.1f} min de budget")

    # 3. Incident de 2h ce mois
    incident_seconds = 2 * 3600  # 7200s
    remaining_after_incident = dt_month - incident_seconds
    print(f"\n  3. Incident de 2h = {incident_seconds}s")
    print(f"     Budget total  = {dt_month:.0f}s = {dt_month / 60:.1f} min")
    print(f"     Apres incident = {remaining_after_incident:.0f}s = {remaining_after_incident / 60:.1f} min")
    if remaining_after_incident < 0:
        print(f"     SLA BREACH : budget depasse de {abs(remaining_after_incident):.0f}s")
    else:
        print(f"     Il reste {remaining_after_incident:.0f}s de budget pour le mois")

    # 4. SLA combine
    sla_interne = 99.95 / 100
    sla_ext_1 = 99.99 / 100
    sla_ext_2 = 99.99 / 100
    # Le SLA combine = produit des SLAs de tous les composants sur le chemin critique
    sla_combined = sla_interne * sla_ext_1 * sla_ext_2
    print(f"\n  4. SLA combine")
    print(f"     = {sla_interne} * {sla_ext_1} * {sla_ext_2}")
    print(f"     = {sla_combined * 100:.4f}%")
    print(f"     Le SLA du systeme est plafonne par le maillon le plus faible.")
    print(f"     Ici, c'est le service interne a 99.95% qui domine.")


# =============================================================================
# MEDIUM -- Exercice 1 : Service de stockage de photos
# =============================================================================

def medium_1_photos():
    """Solution pour l'estimation du service de photos."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Service de stockage de photos")
    print(SEPARATOR)

    dau = 100_000_000
    uploaders_pct = 0.20
    photos_per_uploader = 2
    readers_pct = 0.80
    photos_per_feed_load = 30
    feed_loads_per_day = 5
    photo_size_bytes = 2 * 1024 * 1024  # 2 Mo
    metadata_bytes = 500
    peak_factor = 3

    # 1. QPS ecriture (upload)
    uploaders = dau * uploaders_pct  # 20M
    total_uploads = uploaders * photos_per_uploader  # 40M/jour
    qps_write_avg = total_uploads / 86_400
    qps_write_peak = qps_write_avg * peak_factor

    print(f"\n  1. QPS ecriture (upload)")
    print(f"     Uploaders = {dau:,} * {uploaders_pct} = {uploaders:,.0f}")
    print(f"     Uploads/jour = {uploaders:,.0f} * {photos_per_uploader} = {total_uploads:,.0f}")
    print(f"     QPS moyen = {total_uploads:,.0f} / 86400 = {qps_write_avg:,.0f} req/s")
    print(f"     QPS pic   = {qps_write_avg:,.0f} * {peak_factor} = {qps_write_peak:,.0f} req/s")

    # 2. QPS lecture (feed)
    readers = dau * readers_pct  # 80M
    total_reads = readers * photos_per_feed_load * feed_loads_per_day  # 12B/jour
    qps_read_avg = total_reads / 86_400
    qps_read_peak = qps_read_avg * peak_factor

    print(f"\n  2. QPS lecture (feed)")
    print(f"     Readers = {dau:,} * {readers_pct} = {readers:,.0f}")
    print(f"     Photos lues/jour = {readers:,.0f} * {photos_per_feed_load} * {feed_loads_per_day} = {total_reads:,.0f}")
    print(f"     QPS moyen = {total_reads:,.0f} / 86400 = {qps_read_avg:,.0f} req/s")
    print(f"     QPS pic   = {qps_read_avg:,.0f} * {peak_factor} = {qps_read_peak:,.0f} req/s")

    # 3. Ratio lecture/ecriture
    ratio = qps_read_avg / qps_write_avg
    print(f"\n  3. Ratio lecture/ecriture = {ratio:.0f}:1")
    print(f"     Systeme fortement oriente lecture -> CDN + cache essentiels")

    # 4. Bande passante
    bw_write_peak_gbps = qps_write_peak * photo_size_bytes * 8 / 1e9
    bw_read_peak_gbps = qps_read_peak * photo_size_bytes * 8 / 1e9
    # En realite, les lectures sont souvent servies par CDN, pas les serveurs d'origine
    print(f"\n  4. Bande passante (pic)")
    print(f"     Ecriture = {qps_write_peak:,.0f} * {photo_size_bytes / 1e6:.0f} Mo = {bw_write_peak_gbps:,.0f} Gbps")
    print(f"     Lecture  = {qps_read_peak:,.0f} * {photo_size_bytes / 1e6:.0f} Mo = {bw_read_peak_gbps:,.0f} Gbps")
    print(f"     (En pratique, 90%+ des lectures sont servies par CDN)")

    # 5. Stockage
    storage_per_day_tb = total_uploads * photo_size_bytes / (1024**4)
    storage_1y_pb = storage_per_day_tb * 365 / 1024
    storage_5y_pb = storage_1y_pb * 5

    print(f"\n  5. Stockage")
    print(f"     Par jour  = {total_uploads:,.0f} * {photo_size_bytes / 1e6:.0f} Mo = {storage_per_day_tb:.1f} To/jour")
    print(f"     1 an      = {storage_per_day_tb:.1f} * 365 = {storage_per_day_tb * 365:.0f} To = {storage_1y_pb:.1f} Po")
    print(f"     5 ans     = {storage_5y_pb:.1f} Po")

    # 6. Nombre de serveurs pour la lecture
    server_capacity_rps = 10_000
    servers_needed = math.ceil(qps_read_peak / server_capacity_rps)
    print(f"\n  6. Serveurs pour le pic de lecture")
    print(f"     = {qps_read_peak:,.0f} / {server_capacity_rps:,} = {servers_needed} serveurs")
    print(f"     (+ marge 30% pour redondance = {math.ceil(servers_needed * 1.3)} serveurs)")

    # Bonus : bottleneck
    print(f"\n  BONUS : Bottleneck principal")
    print(f"     Bande passante sortante ({bw_read_peak_gbps:,.0f} Gbps) est astronomique.")
    print(f"     Sans CDN, aucun datacenter ne peut servir ce trafic.")
    print(f"     Solution : CDN (CloudFront, Fastly) pour 95%+ des lectures.")
    print(f"     Le stockage ({storage_5y_pb:.0f} Po sur 5 ans) necessite un object store (S3).")


# =============================================================================
# MEDIUM -- Exercice 2 : Choix de DB
# =============================================================================

def medium_2_db_choice():
    """Solution pour le choix de base de donnees."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Tradeoff Analysis -- Choix de DB")
    print(SEPARATOR)

    choices = [
        {
            "donnee": "Catalogue produits",
            "db": "MongoDB (ou Elasticsearch pour la recherche)",
            "consistency": "Eventual consistency",
            "justification": (
                "Schema flexible : les produits ont des attributs variables par categorie "
                "(vetements = taille/couleur, electronique = specs techniques). "
                "MongoDB gere nativement les documents avec schemas differents. "
                "Ratio 1000:1 lecture/ecriture -> lecture optimisee avec read replicas."
            ),
            "alternatives_ecartees": (
                "PostgreSQL (JSONB) : possible mais moins performant pour les queries "
                "sur des champs imbriques a fort volume. "
                "Cassandra : overkill pour 10M de produits, et les queries ad hoc sont limitees."
            ),
            "risque": (
                "MongoDB peut avoir des problemes de performance sur les aggregations complexes. "
                "Mitigation : Elasticsearch en read replica pour la recherche full-text et les facettes."
            ),
        },
        {
            "donnee": "Commandes",
            "db": "PostgreSQL",
            "consistency": "Strong consistency",
            "justification": (
                "Transactions ACID obligatoires : une commande implique un debit stock, "
                "une creation de ligne de commande, une reservation de paiement. "
                "Integrite referentielle entre commande/lignes/client/produit."
            ),
            "alternatives_ecartees": (
                "MongoDB : pas de transactions multi-documents natives avant 4.0, et meme "
                "apres, moins mature que PostgreSQL pour les transactions complexes. "
                "DynamoDB : pas de jointures, transactions limitees a 25 items."
            ),
            "risque": (
                "50K commandes/jour = ~0.6 req/s, ce n'est pas un probleme de scale. "
                "Risque : si le volume monte a 5M/jour, le sharding PostgreSQL est complexe. "
                "Mitigation : Citus (PostgreSQL distribue) ou migration vers un event-sourcing pattern."
            ),
        },
        {
            "donnee": "Sessions utilisateur",
            "db": "Redis",
            "consistency": "Eventual consistency (acceptable ici : perte tolerable)",
            "justification": (
                "Acces sub-milliseconde obligatoire (chaque requete HTTP verifie la session). "
                "TTL natif de 30 min avec expiration automatique. "
                "Perte tolerable = pas besoin de persistence durable."
            ),
            "alternatives_ecartees": (
                "PostgreSQL : trop lent pour un acces par requete HTTP (meme avec index). "
                "Memcached : possible mais pas de persistence meme partielle, "
                "pas de structures de donnees avancees. "
                "DynamoDB : latence > Redis, cout plus eleve pour ce pattern."
            ),
            "risque": (
                "Redis est single-threaded (io-threads aide mais limite) : un pic peut saturer. "
                "En cas de panne Redis, TOUTES les sessions sont perdues. "
                "Mitigation : Redis Sentinel ou Redis Cluster pour la HA. "
                "Fallback : re-authentification transparente si session perdue."
            ),
        },
    ]

    for choice in choices:
        print(f"\n  {'-'*56}")
        print(f"  {choice['donnee']}")
        print(f"  {'-'*56}")
        print(f"  DB choisie     : {choice['db']}")
        print(f"  Consistency    : {choice['consistency']}")
        print(f"  Justification  : {choice['justification']}")
        print(f"  Ecarte         : {choice['alternatives_ecartees']}")
        print(f"  Risque + mitig : {choice['risque']}")


# =============================================================================
# MEDIUM -- Exercice 3 : Latency Budget
# =============================================================================

def medium_3_latency_budget():
    """Solution pour la decomposition de latence."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Latency Budget")
    print(SEPARATOR)

    gateway = 5
    auth = 15
    postgres = 20
    redis_cache = 2
    cache_hit_rate = 0.80
    reco = 100
    price = 30

    # 1. Sans cache, sequentiel
    seq_no_cache = gateway + auth + postgres + reco + price
    print(f"\n  1. Sequentiel, sans cache")
    print(f"     Gateway({gateway}) + Auth({auth}) + PG({postgres}) + Reco({reco}) + Price({price})")
    print(f"     = {seq_no_cache} ms")
    print(f"     SLO 200ms -> DEPASSE ({seq_no_cache} > 200)")

    # 2. Avec cache (hit rate 80%), sequentiel
    # p99 = worst case = cache miss (car au p99, on tombe dans les 20% de miss)
    # En fait, au p99, la probabilite de cache miss est ce qui importe
    # Mais pour un SEUL appel, p99 signifie : dans le pire 1% des cas
    # Le cache miss arrive 20% du temps -> au p99, c'est un cache miss
    db_latency_p99 = postgres  # Cache miss = on tape la DB
    seq_with_cache_p99 = gateway + auth + db_latency_p99 + reco + price
    seq_with_cache_avg = gateway + auth + (cache_hit_rate * redis_cache + (1 - cache_hit_rate) * postgres) + reco + price

    print(f"\n  2. Sequentiel, avec cache (80% hit rate)")
    print(f"     Latence moyenne DB = 0.8*{redis_cache} + 0.2*{postgres} = {cache_hit_rate * redis_cache + (1 - cache_hit_rate) * postgres:.1f} ms")
    print(f"     p50 (cache hit)  = {gateway} + {auth} + {redis_cache} + {reco} + {price} = {gateway + auth + redis_cache + reco + price} ms")
    print(f"     p99 (cache miss) = {gateway} + {auth} + {postgres} + {reco} + {price} = {seq_with_cache_p99} ms")
    print(f"     SLO 200ms -> p99 DEPASSE ({seq_with_cache_p99} > 200)")

    # 3. Optimisations proposees
    print(f"\n  3. Optimisations")
    print(f"     a) Paralleliser Product lookup, Reco, et Price (independants)")
    print(f"     b) Pre-fetch les recommandations (cache les resultats)")
    print(f"     c) Fusionner Auth dans le Gateway (middleware, pas un appel reseau)")

    # 4. Redesign
    print(f"\n  4. Redesign optimise")
    print(f"     Client -> Gateway+Auth (5ms) -> [Parallel: DB/Cache + Reco + Price]")
    print(f"")
    print(f"     Phase 1 (sequentiel) : Gateway + Auth middleware = {gateway}ms")
    print(f"     Phase 2 (parallele)  : max(DB({postgres}), Reco({reco}), Price({price})) = {max(postgres, reco, price)}ms")

    # Auth est maintenant un middleware dans le Gateway (pas un appel reseau separe)
    # On economise 15ms de latence reseau
    optimized_p99 = gateway + max(postgres, reco, price)
    optimized_p50 = gateway + max(redis_cache, reco, price)

    print(f"\n  5. Latence apres optimisation")
    print(f"     p50 (cache hit)  = {gateway} + max({redis_cache}, {reco}, {price}) = {gateway} + {max(redis_cache, reco, price)} = {optimized_p50} ms")
    print(f"     p99 (cache miss) = {gateway} + max({postgres}, {reco}, {price}) = {gateway} + {max(postgres, reco, price)} = {optimized_p99} ms")
    print(f"     SLO 200ms -> p99 = {optimized_p99}ms -> RESPECTE")

    print(f"\n  Resume des gains :")
    print(f"     Avant : {seq_no_cache}ms (sequentiel, sans cache)")
    print(f"     Apres : {optimized_p99}ms (parallele, p99 cache miss)")
    print(f"     Gain  : {seq_no_cache - optimized_p99}ms ({(1 - optimized_p99/seq_no_cache)*100:.0f}% de reduction)")


# =============================================================================
# HARD -- Exercice 1 : Compteur de vues (solution esquissee)
# =============================================================================

def hard_1_view_counter():
    """Solution esquissee pour le compteur de vues en temps reel."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Compteur de vues -- Estimation & Architecture")
    print(SEPARATOR)

    # --- Estimation ---
    views_per_day = 1_000_000_000
    qps_avg = views_per_day / 86_400
    qps_peak = qps_avg * 5  # Video virale = spike
    # On suppose 500M de videos au total, chaque compteur = 8 bytes (int64) + 16 bytes video_id
    num_videos = 500_000_000
    counter_size = 8 + 16  # video_id (UUID) + count (int64)
    # Analytics : 1 row par video par heure = 500M * 24 * 365 * (8+16+8) = enorme
    # On pre-aggregate par heure
    analytics_row_size = 16 + 8 + 8  # video_id + timestamp + count

    print(f"\n  ESTIMATION")
    print(f"  {'-'*50}")
    print(f"  Vues/jour               : {views_per_day:,}")
    print(f"  QPS moyen               : {qps_avg:,.0f}")
    print(f"  QPS pic (x5, virale)    : {qps_peak:,.0f}")
    print(f"  Compteurs (RAM)         : {num_videos:,} * {counter_size} B = {num_videos * counter_size / 1e9:.1f} Go")
    print(f"  Analytics/an (hourly)   : {num_videos:,} * 8760h * {analytics_row_size}B = {num_videos * 8760 * analytics_row_size / 1e12:.1f} To")
    print(f"  Bande passante (ecriture): {qps_peak * 100 * 8 / 1e6:,.0f} Mbps (100 bytes/req)")

    # --- Architecture ---
    print(f"\n  ARCHITECTURE")
    print(f"  {'-'*50}")
    print(f"""
  Write path (incrementer une vue) :
    Client -> API Gateway -> Kafka (topic: view-events)
    Kafka -> View Counter Service -> Redis (INCR atomique)
    Kafka -> Analytics Aggregator -> ClickHouse (batch insert hourly)

  Read path (afficher le compteur) :
    Client -> API Gateway -> Redis (GET counter)
    Fallback si miss -> ClickHouse (SUM des aggregations)

  Analytics path (dashboard createur) :
    Client -> API Gateway -> ClickHouse (pre-aggregated tables)
""")

    print(f"  CHOIX DE DB")
    print(f"  {'-'*50}")
    print(f"  Compteurs temps reel : Redis")
    print(f"    - INCR atomique, sub-ms, parfait pour ~{num_videos * counter_size / 1e9:.0f} Go de donnees")
    print(f"    - Consistency : eventual (acceptable, tolerance 1-2%)")
    print(f"  Analytics : ClickHouse")
    print(f"    - Optimise pour les aggregations sur colonnes (SUM, COUNT par heure/jour)")
    print(f"    - Consistency : eventual (batch processing, OK pour analytics)")

    print(f"\n  TRADEOFFS")
    print(f"  {'-'*50}")
    print(f"  1. Kafka buffer au lieu d'ecriture directe Redis")
    print(f"     + Absorbe les pics de trafic (video virale)")
    print(f"     + Decouple le write path du counter service")
    print(f"     - Ajoute 10-100ms de latence entre la vue et le compteur")
    print(f"     Accepte car : tolerance de 1-2% sur le compteur affiche")
    print(f"")
    print(f"  2. Redis (AP) au lieu de PostgreSQL (CP) pour les compteurs")
    print(f"     + Sub-ms latence de lecture")
    print(f"     + INCR atomique sans lock global")
    print(f"     - Pas de durabilite garantie (perte possible au crash)")
    print(f"     Accepte car : analytics dans ClickHouse = source de verite")
    print(f"")
    print(f"  3. Pre-aggregation hourly au lieu de raw events")
    print(f"     + Stockage 8760x plus petit que les events bruts")
    print(f"     + Queries analytics rapides")
    print(f"     - Perte de granularite (pas de vue par seconde)")
    print(f"     Accepte car : les createurs n'ont pas besoin de granularite < 1h")

    print(f"\n  SLIs / SLOs")
    print(f"  {'-'*50}")
    print(f"  1. Write latency p99 < 10ms (SLI: latence d'ecriture dans Kafka)")
    print(f"  2. Counter staleness < 30s (SLI: age du compteur Redis vs Kafka offset)")
    print(f"  3. Availability > 99.9% (SLI: taux de requetes en succes)")


# =============================================================================
# HARD -- Exercice 2 : Cascading Failures (analyse textuelle)
# =============================================================================

def hard_2_cascading_failures():
    """Solution pour l'analyse de cascading failures."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Cascading Failures -- Analyse")
    print(SEPARATOR)

    print(f"""
  1. PROPAGATION DE LA PANNE
  {'-'*50}
  Etape 1 : Payment Service repond en 30s au lieu de 200ms.
  Etape 2 : Les threads de Order Service qui appellent Payment sont bloques 30s.
            Chaque commande consomme un thread pendant 30s au lieu de 200ms (150x plus long).
  Etape 3 : Le thread pool de Order Service se remplit. Nouvelles requetes -> timeout ou rejet.
  Etape 4 : Les API servers (API-1/2/3) qui attendent Order Service sont bloques aussi.
            Leurs threads se remplissent progressivement.
  Etape 5 : Les API servers ne peuvent plus traiter AUCUNE requete, meme celles
            qui n'utilisent PAS le Payment Service (User, Product).
  Etape 6 : Le Load Balancer detecte les API servers comme "unhealthy" -> downtime total.

  Ordre d'impact : Payment -> Order Svc -> API servers -> TOUT le systeme

  2. CALCUL D'IMPACT
  {'-'*50}
  Thread pool par API server : 200 threads
  3 API servers = 600 threads total
  Avec Payment a 200ms : 1 thread traite 5 commandes/seconde
  Avec Payment a 30s   : 1 thread traite 1 commande/30s = 0.033/s

  Threads consommes par les commandes :
  - Avant : a 100 commandes/s, il faut 100/5 = 20 threads (10% du pool)
  - Apres : a 100 commandes/s, il faut 100 * 30s = 3000 threads -> IMPOSSIBLE

  Saturation : Les 600 threads sont consommes en 600 * (1/100 commandes/s) = 6 secondes
  En ~6 secondes, tous les threads sont bloques. Le systeme est down.

  3. MECANISMES DE PROTECTION
  {'-'*50}
  a) Circuit Breaker :
     COMMENT : Apres N echecs/timeouts consecutifs, le circuit "s'ouvre" et court-circuite
     les appels vers Payment (renvoie une erreur immediatement sans appeler le service).
     POURQUOI : Libere les threads instantanement. Empeche l'accumulation.

  b) Timeout + Retry avec backoff exponentiel :
     COMMENT : Timeout de 2s sur l'appel Payment. Si timeout, retry apres 1s, 2s, 4s.
     POURQUOI : Limite le temps de blocage d'un thread a 2s (pas 30s).
     Le backoff evite de surcharger un service deja en difficulte.

  c) Bulkhead (isolation des thread pools) :
     COMMENT : Thread pool dedie pour les appels Payment (ex: 30 threads sur 200).
     Les 170 autres threads restent disponibles pour User/Product/autres.
     POURQUOI : Un service lent ne peut consommer que SES threads, pas les autres.

  d) Queue de decouplage (Kafka) :
     COMMENT : Order Service publie un event "order.created" dans Kafka.
     Un consumer separe traite le paiement de maniere asynchrone.
     POURQUOI : Le thread de l'API est libere immediatement. Le paiement est traite en background.

  e) Fallback / Degraded mode :
     COMMENT : Si Payment est indisponible, la commande est creee en etat "pending_payment".
     L'utilisateur est informe que le paiement sera traite sous 5 min.
     POURQUOI : Le service reste "disponible" meme si degrade. Meilleur que rien.

  4. REDESIGN RESILIENT
  {'-'*50}
  Flux de commande redesigne :
    Client -> API -> Order Service : cree la commande (status=pending) -> repond au client
    Order Service -> Kafka : publie "order.created"
    Payment Consumer <- Kafka : traite le paiement de maniere asynchrone
    Payment Consumer -> Kafka : publie "payment.confirmed" ou "payment.failed"
    Order Service <- Kafka : met a jour la commande -> notifie le client (websocket/push)

  Consistency model : eventual consistency entre creation et confirmation.
  La commande passe par les etats : pending -> confirmed / failed.
  C'est le Saga pattern.

  Informer l'utilisateur : "Votre commande est enregistree. Confirmation de paiement
  sous quelques secondes." + notification push quand le paiement est traite.

  5. SLA COMPOSE
  {'-'*50}""")

    services = [0.9995, 0.9995, 0.9995, 0.995]  # 3 internes + Payment
    sla_sync = 1.0
    for s in services:
        sla_sync *= s
    print(f"  SLA synchrone = 99.95% * 99.95% * 99.95% * 99.5% = {sla_sync * 100:.3f}%")
    print(f"  Downtime/an = {(1 - sla_sync) * 365.25 * 24:.1f} heures")

    print(f"""
  Pour atteindre > 99.9% malgre Payment a 99.5% :
  - Decoupler Payment via Kafka (async)
  - Le flux synchrone ne depend plus de Payment :
    SLA sync = 99.95% * 99.95% * 99.95% = {0.9995**3 * 100:.3f}%
  - Payment failures sont retries automatiquement par le consumer Kafka
  - Avec retry + dead letter queue, le taux de succes effectif de Payment monte a ~99.99%
  - SLA effectif du flux complet > 99.9%""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute toutes les solutions."""
    print("\n" + "#" * 60)
    print("#  SOLUTIONS -- JOUR 1 : PRINCIPES FONDAMENTAUX")
    print("#" * 60)

    # Easy
    easy_1_notifications()
    easy_2_cp_ap()
    easy_3_nines()

    # Medium
    medium_1_photos()
    medium_2_db_choice()
    medium_3_latency_budget()

    # Hard
    hard_1_view_counter()
    hard_2_cascading_failures()

    print("\n" + "#" * 60)
    print("#  FIN DES SOLUTIONS")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

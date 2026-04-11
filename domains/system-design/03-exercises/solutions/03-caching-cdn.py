"""
Solutions -- Exercices Jour 3 : Caching & CDN

Ce fichier contient les solutions calculees pour les exercices Easy, Medium, et Hard.
Chaque solution montre le raisonnement etape par etape.

Usage:
    python 03-caching-cdn.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Quelle strategie de cache ?
# =============================================================================

def easy_1_cache_strategy():
    """Solution pour le choix de strategie de cache par use case."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Quelle strategie de cache ?")
    print(SEPARATOR)

    choices = [
        (
            "Profils utilisateur (50K reads/sec, 10 writes/sec)",
            "Cache-Aside",
            "Read-heavy (ratio 5000:1). Cache-aside est le choix par defaut : "
            "on ne met en cache que les profils effectivement demandes. "
            "10 writes/sec = invalidation negligeable. TTL de 5-15 min "
            "pour tolerer le stale eventuel."
        ),
        (
            "Compteur de vues videos",
            "Write-Behind (Write-Back)",
            "Write-heavy : chaque vue = 1 increment. Write-behind accumule "
            "les increments en memoire (Redis INCR) et flush en batch vers "
            "la DB toutes les 5-10 secondes. On tolere la perte de quelques "
            "vues en cas de crash (pas critique). Tradeoff : consistance "
            "faible mais performance maximale."
        ),
        (
            "Stock e-commerce (survente couteuse)",
            "Write-Through",
            "La consistance est critique : afficher un stock > 0 alors que "
            "le stock reel est 0 = survente = perte financiere. Write-through "
            "garantit que le cache et la DB sont toujours synchrones. "
            "Le cout en latence d'ecriture est acceptable car les mises a jour "
            "de stock sont moins frequentes que les lectures."
        ),
        (
            "Config globale pour un cluster K8s",
            "Read-Through + Cache L1 in-process",
            "Donnee identique pour tous les pods, change rarement. "
            "Cache L1 in-process (dict local) avec TTL 30s evite "
            "meme le round-trip reseau vers Redis. Read-through car "
            "l'app n'a pas besoin de connaitre la source (DB vs Redis)."
        ),
        (
            "Dashboard analytics (aggregation 24h)",
            "Cache-Aside avec TTL long",
            "Les aggregations sont couteuses (scan de millions de lignes). "
            "Le resultat est identique pour tous les users. Cache-aside "
            "avec TTL de 5-15 min. Ou materialized view en DB + cache-aside. "
            "Le stale de quelques minutes est acceptable sur un dashboard."
        ),
        (
            "Sessions utilisateur (login/logout)",
            "Write-Through ou Cache-Aside",
            "La session doit etre a jour (logout = suppression immediate). "
            "Write-through pour les ecritures (login cree la session dans "
            "cache + DB atomiquement). Cache-aside pour les lectures. "
            "TTL = duree de la session (30 min). Redis est le choix naturel "
            "car les sessions sont des key-value simples avec TTL natif."
        ),
    ]

    for i, (system, choice, justification) in enumerate(choices, 1):
        print(f"\n  {i}. {system}")
        print(f"     Strategie : {choice}")
        print(f"     Raison : {justification}")


# =============================================================================
# EASY -- Exercice 2 : Cache-Control headers
# =============================================================================

def easy_2_cache_control():
    """Solution pour les headers Cache-Control par type de contenu."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Cache-Control headers")
    print(SEPARATOR)

    headers = [
        (
            "app.a3f2b1c.js (JS avec hash dans le nom)",
            "Cache-Control: public, max-age=31536000, immutable",
            "public : tout cache (CDN, proxy, browser) peut stocker. "
            "max-age=31536000 : 1 an (maximum pratique). "
            "immutable : le browser ne revalidera jamais. "
            "Le hash dans le nom garantit que tout changement = nouvelle URL. "
            "C'est le pattern standard pour les assets bundles (webpack, vite)."
        ),
        (
            "/api/me (profil utilisateur connecte)",
            "Cache-Control: private, no-cache",
            "private : seul le browser peut cacher (pas le CDN, car le contenu "
            "est personnalise). no-cache : le browser doit revalider a chaque "
            "fois (If-None-Match avec ETag). Cela economise la bande passante "
            "(304 si pas de changement) tout en garantissant la fraicheur."
        ),
        (
            "/api/products (catalogue mis a jour toutes les heures)",
            "Cache-Control: public, s-maxage=3600, max-age=300, stale-while-revalidate=60",
            "public : identique pour tous les users. "
            "s-maxage=3600 : le CDN cache pendant 1h (synchrone avec les mises a jour). "
            "max-age=300 : le browser cache pendant 5 min (revalidation plus frequente cote client). "
            "stale-while-revalidate=60 : servir le stale pendant 60s en arriere-plan pendant la revalidation."
        ),
        (
            "/login (page HTML de connexion)",
            "Cache-Control: no-cache",
            "no-cache : la page peut etre cachee mais doit etre revalidee. "
            "On veut que le browser revalide car le HTML peut changer "
            "(nouveau build, CSRF token). Le CDN peut aussi cacher "
            "avec revalidation (ETag). Pas no-store car la page n'est "
            "pas sensible en elle-meme."
        ),
        (
            "PDF de facture (utilisateur authentifie)",
            "Cache-Control: private, no-store",
            "private : ne pas cacher sur le CDN (document confidentiel). "
            "no-store : ne pas stocker du tout (meme sur le disque local). "
            "Les factures contiennent des donnees financieres sensibles. "
            "Alternative acceptable : private, no-cache si on veut permettre "
            "le cache browser avec revalidation."
        ),
    ]

    for i, (resource, header, justification) in enumerate(headers, 1):
        print(f"\n  {i}. {resource}")
        print(f"     Header : {header}")
        print(f"     Raison : {justification}")


# =============================================================================
# EASY -- Exercice 3 : Dimensionnement memoire Redis
# =============================================================================

def easy_3_redis_sizing():
    """Solution pour le dimensionnement memoire Redis."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Dimensionnement memoire Redis")
    print(SEPARATOR)

    # Donnees
    uuid_bytes = 36
    role_bytes = 10
    token_bytes = 64
    timestamp_bytes = 8
    preferences_bytes = 200
    redis_overhead = 2.5  # Facteur d'overhead Redis
    concurrent_sessions = 2_000_000
    masters = 3
    replicas_per_master = 1

    # 1. Taille brute d'une session
    raw_size = uuid_bytes + role_bytes + token_bytes + timestamp_bytes + preferences_bytes
    print(f"\n  1. Taille brute d'une session :")
    print(f"     user_id (UUID)    : {uuid_bytes} bytes")
    print(f"     role              : {role_bytes} bytes")
    print(f"     token             : {token_bytes} bytes")
    print(f"     last_seen         : {timestamp_bytes} bytes")
    print(f"     preferences (JSON): {preferences_bytes} bytes")
    print(f"     Total brut        : {raw_size} bytes")

    # 2. Memoire totale avec overhead Redis
    raw_total = concurrent_sessions * raw_size
    total_with_overhead = raw_total * redis_overhead
    raw_total_gb = raw_total / (1024 ** 3)
    total_gb = total_with_overhead / (1024 ** 3)
    print(f"\n  2. Memoire totale pour {concurrent_sessions:,} sessions :")
    print(f"     Brut              : {concurrent_sessions:,} * {raw_size} = {raw_total:,} bytes = {raw_total_gb:.2f} Go")
    print(f"     Avec overhead {redis_overhead}x : {total_with_overhead:,.0f} bytes = {total_gb:.2f} Go")

    # 3. RAM par noeud master
    ram_per_master = total_gb / masters
    print(f"\n  3. RAM par noeud master ({masters} masters) :")
    print(f"     {total_gb:.2f} Go / {masters} = {ram_per_master:.2f} Go par master")
    print(f"     Recommandation : prevoir 2x pour la marge -> {ram_per_master * 2:.1f} Go par master")

    # 4. Total avec replicas
    total_nodes = masters * (1 + replicas_per_master)
    total_ram = total_gb * (1 + replicas_per_master)
    print(f"\n  4. Avec replicas ({replicas_per_master} replica par master) :")
    print(f"     Noeuds totaux     : {masters} masters + {masters * replicas_per_master} replicas = {total_nodes}")
    print(f"     RAM totale cluster: {total_gb:.2f} * {1 + replicas_per_master} = {total_ram:.2f} Go")
    print(f"     Les replicas sont essentiels pour la HA (si un master tombe,")
    print(f"     son replica est promu automatiquement)")

    # 5. Eviction policy
    print(f"\n  5. maxmemory-policy recommandee :")
    print(f"     volatile-ttl")
    print(f"     Raison : les sessions ont toutes un TTL (30 min). volatile-ttl")
    print(f"     evince les cles avec le TTL le plus court en premier, ce qui est")
    print(f"     logique pour les sessions (les plus anciennes expirent d'abord).")
    print(f"     Alternative : volatile-lru si les sessions ont des TTL identiques")
    print(f"     (evince les moins recemment accedees).")
    print(f"     NE PAS utiliser allkeys-* car les sessions DOIVENT avoir un TTL.")


# =============================================================================
# MEDIUM -- Exercice 1 : Cache pour un feed social
# =============================================================================

def medium_1_social_feed():
    """Solution pour la couche de cache d'un feed social."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Cache pour un feed social")
    print(SEPARATOR)

    users = 200_000_000
    avg_following = 200
    posts_per_min = 500_000
    feed_size = 50  # Posts par feed

    print(f"\n  1. Type de cache :")
    print(f"     Redis (distributed cache) — PAS de CDN.")
    print(f"     Pourquoi pas CDN : le feed est personnalise (chaque user voit")
    print(f"     un contenu different). Le CDN cache du contenu identique pour")
    print(f"     tous les users d'une meme region. Ici, le cache doit etre")
    print(f"     par user -> seul Redis permet ca efficacement.")

    print(f"\n  2. Structure Redis : Sorted Set")
    print(f"     Cle : feed:{{user_id}}")
    print(f"     Score : timestamp du post")
    print(f"     Member : post_id (reference, pas le contenu entier)")
    print(f"     Commande : ZREVRANGE feed:{{user_id}} 0 49 -> les 50 derniers posts")
    print(f"     Les details du post sont dans un Hash separe : post:{{post_id}}")

    print(f"\n  3. Gestion de la taille :")
    print(f"     ZREMRANGEBYRANK feed:{{user_id}} 0 -501 apres chaque ZADD")
    print(f"     -> garde seulement les 500 derniers posts en cache")
    print(f"     (10 pages de 50). Au-dela, la pagination va en DB.")

    print(f"\n  4. Fanout-on-write vs Fanout-on-read :")
    print(f"     Fanout-on-write : quand un user poste, on ecrit dans le feed")
    print(f"     cache de CHAQUE follower. Pour un user avec 200 followers,")
    print(f"     ca fait 200 ZADD Redis. C'est rapide et le read est en O(1).")
    print(f"     Fanout-on-read : quand un user ouvre son feed, on aggrege les")
    print(f"     posts de tous les comptes qu'il suit. Moins de writes, mais")
    print(f"     le read est lent (200 queries pour merger 200 timelines).")

    print(f"\n     Solution hybride (approche Twitter) :")
    print(f"     - Users normaux (< 10K followers) : fanout-on-write")
    print(f"     - Celebrities (> 10K followers) : fanout-on-read")
    print(f"     Quand un user ouvre son feed : lire le feed pre-calcule + merger")
    print(f"     les posts des celebrities qu'il suit.")

    print(f"\n  5. Celebrities :")
    print(f"     Un celebrity a 50M followers. Fanout-on-write = 50M ZADD par post.")
    print(f"     A {posts_per_min:,} posts/min, c'est insoutenable.")
    print(f"     Solution : ne PAS pre-calculer le feed pour les followers de celebrities.")
    print(f"     Au read, merger le feed pre-calcule avec les derniers posts des celebrities.")

    # 6. Estimation memoire
    post_id_size = 20  # bytes (UUID compact ou snowflake ID)
    score_size = 8     # bytes (timestamp double)
    redis_entry_overhead = 40  # bytes overhead par entry dans un sorted set
    entries_per_feed = 500
    active_users_pct = 0.3  # 30% ont un feed en cache (les actifs recents)
    active_users = int(users * active_users_pct)

    feed_size_bytes = entries_per_feed * (post_id_size + score_size + redis_entry_overhead)
    total_bytes = active_users * feed_size_bytes
    total_tb = total_bytes / (1024 ** 4)

    print(f"\n  6. Estimation memoire :")
    print(f"     Taille par entry : {post_id_size} + {score_size} + {redis_entry_overhead} = {post_id_size + score_size + redis_entry_overhead} bytes")
    print(f"     Taille par feed ({entries_per_feed} entries) : {feed_size_bytes:,} bytes = {feed_size_bytes/1024:.1f} Ko")
    print(f"     Users actifs avec feed en cache (30%) : {active_users:,}")
    print(f"     Total : {active_users:,} * {feed_size_bytes:,} = {total_bytes:,} bytes = {total_tb:.1f} To")
    print(f"     + les Hash des posts eux-memes (~2-5 To supplementaires)")

    print(f"\n  7. Invalidation :")
    print(f"     Event-driven : quand un post est publie, un worker Kafka")
    print(f"     fait le fanout (ZADD dans les feeds des followers).")
    print(f"     TTL de 24h sur les feeds : filet de securite, evite les")
    print(f"     feeds fantomes d'users inactifs. Pas besoin de TTL court")
    print(f"     car l'event-driven garantit la fraicheur.")


# =============================================================================
# MEDIUM -- Exercice 2 : Diagnostic cache miss rate
# =============================================================================

def medium_2_cache_diagnosis():
    """Solution pour le diagnostic d'un cache miss rate eleve."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Diagnostic cache miss rate eleve")
    print(SEPARATOR)

    num_keys = 15_000_000
    avg_size = 500  # bytes
    redis_overhead = 2.5
    current_ram_gb = 8

    print(f"\n  1. Cause racine :")
    print(f"     Le cache est SOUS-DIMENSIONNE. 15M cles * 500 bytes * 2.5 overhead")
    print(f"     = ~17.5 Go necessaires, mais seulement 8 Go disponibles.")
    print(f"     Redis evince en permanence (12K evictions/sec) pour faire de la place.")
    print(f"     En plus, les cles search: (25% du cache = 3.75M cles) ont un taux")
    print(f"     de re-acces < 5%, ce qui signifie qu'elles gaspillent ~4.4 Go de cache")
    print(f"     pour des donnees rarement relues.")

    # 2. Calcul memoire ideale
    raw_total = num_keys * avg_size
    ideal_total = raw_total * redis_overhead
    ideal_gb = ideal_total / (1024 ** 3)
    print(f"\n  2. Memoire ideale :")
    print(f"     {num_keys:,} * {avg_size} bytes = {raw_total:,} bytes = {raw_total/(1024**3):.1f} Go brut")
    print(f"     Avec overhead Redis ({redis_overhead}x) : {ideal_gb:.1f} Go")
    print(f"     Deficit : {ideal_gb:.1f} - {current_ram_gb} = {ideal_gb - current_ram_gb:.1f} Go manquants")

    # 3. Actions par impact
    print(f"\n  3. Actions par ordre d'impact :")

    search_keys = int(num_keys * 0.25)
    search_memory = search_keys * avg_size * redis_overhead / (1024 ** 3)

    print(f"\n     Action 1 (impact le plus fort, cout zero) : Reduire les cles search:")
    print(f"     Les {search_keys:,} cles search: occupent ~{search_memory:.1f} Go")
    print(f"     avec un taux de re-acces de 5%. Options :")
    print(f"     - Reduire le TTL de 1h a 5 min")
    print(f"     - Ou ne cacher que les recherches les plus frequentes (top 10%)")
    print(f"     Impact estime : libere ~{search_memory * 0.8:.1f} Go -> hit rate +20-25%")

    print(f"\n     Action 2 (impact moyen, cout modere) : Augmenter la RAM Redis")
    print(f"     Passer de 8 Go a 20 Go (ou 2 * 10 Go en cluster)")
    print(f"     Cout : ~$200-400/mois supplementaires")
    print(f"     Impact estime : hit rate +15-20% (plus d'evictions)")

    print(f"\n     Action 3 (impact complementaire) : Ajouter un L1 in-process cache")
    print(f"     Pour les cles les plus hot (session: et product: les plus accedees)")
    print(f"     Cache local de 100 Mo par instance avec TTL 30s")
    print(f"     Impact estime : hit rate +5-10% et reduction de la charge Redis de 30%")

    # 4. Amelioration estimee
    print(f"\n  4. Amelioration estimee du hit rate :")
    print(f"     Actuel : 45%")
    print(f"     Apres action 1 : ~70% (+25%)")
    print(f"     Apres action 1+2 : ~88% (+18%)")
    print(f"     Apres action 1+2+3 : ~92% (+4%)")

    # 5. Decision
    print(f"\n  5. Augmenter la RAM OU optimiser ?")
    print(f"     D'ABORD optimiser (action 1) : gratuit, impact immediat.")
    print(f"     ENSUITE augmenter la RAM si le hit rate reste < 85%.")
    print(f"     Justification : les cles search: gaspillent {search_memory:.1f} Go pour")
    print(f"     des donnees accedees a 5%. Les supprimer libere de la place pour les")
    print(f"     cles product: et session: qui ont un re-acces bien plus eleve.")

    print(f"\n  6. Dashboard monitoring (5 metriques) :")
    metrics = [
        ("Hit rate (%)", "> 85%", "< 70%", "L'indicateur #1 de sante du cache"),
        ("Eviction rate (/sec)", "< 100/s", "> 1K/s", "Signal que le cache est trop petit"),
        ("Memory usage (%)", "< 80%", "> 95%", "Marge avant les evictions massives"),
        ("Latency p99 (ms)", "< 2ms", "> 10ms", "Degradation = surcharge ou reseau"),
        ("Key count (total)", "stable +/-10%", "chute brutale", "Un drop = TTL massif ou FLUSHALL"),
    ]
    for name, ok, alert, why in metrics:
        print(f"     - {name} : OK={ok}, ALERTE={alert} ({why})")


# =============================================================================
# MEDIUM -- Exercice 3 : CDN strategy multi-region
# =============================================================================

def medium_3_cdn_strategy():
    """Solution pour la strategie CDN multi-region."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : CDN strategy multi-region")
    print(SEPARATOR)

    print(f"\n  1. Architecture CDN :")
    print(f"     Provider : CloudFront (integration native S3) ou Cloudflare")
    print(f"     Niveaux : 2 niveaux (edge + origin shield)")
    print(f"     - Edge : POPs dans chaque region (Paris, Francfort, NYC, Tokyo...)")
    print(f"     - Origin Shield : 1 cache intermediaire par region (reduit la charge origin)")
    print(f"     - Origin : les 3 regions API (US-East, EU-West, APAC)")
    print(f"     Route les requetes vers l'origin la plus proche (latency-based routing)")

    print(f"\n  2. Securisation des documents :")
    print(f"     Signed URLs (CloudFront) ou Signed Cookies (pour un domaine entier)")
    print(f"     Flow : l'API genere une signed URL avec expiration (15 min)")
    print(f"     Le CDN verifie la signature avant de servir le document.")
    print(f"     Si la signature est invalide ou expiree -> 403 Forbidden.")
    print(f"     Avantage : le document reste en cache CDN, mais seuls les users")
    print(f"     avec une URL signee valide y accedent.")

    print(f"\n  3. Invalidation documents (< 5 min) :")
    print(f"     Option A : TTL court (s-maxage=300) -> le CDN revalide toutes les 5 min")
    print(f"     Option B : Purge API (CloudFront CreateInvalidation) declenche par l'update")
    print(f"     Recommandation : TTL 300s + purge API pour les updates urgents.")
    print(f"     Le purge API prend 30-60s sur CloudFront (propagation globale).")

    print(f"\n  4. Cache headers par type :")
    headers = [
        ("app.hash.js (React)", "Cache-Control: public, max-age=31536000, immutable",
         "Hash dans le nom = versionne. Cache 1 an."),
        ("/api/documents/list", "Cache-Control: private, no-cache",
         "Personnalise (liste des documents du user). Revalidation obligatoire."),
        ("/documents/{id}/download", "Cache-Control: private, no-cache + Signed URL",
         "Document confidentiel. Le CDN cache le fichier mais exige une signed URL."),
        ("index.html", "Cache-Control: no-cache",
         "Toujours revalider pour pointer vers le dernier build (hash dans les <script>)."),
    ]
    for resource, header, reason in headers:
        print(f"     {resource}")
        print(f"       {header}")
        print(f"       -> {reason}")

    print(f"\n  5. Mesure du hit rate :")
    print(f"     CloudFront fournit des metriques natives :")
    print(f"     - Hit rate par distribution (global)")
    print(f"     - Hit rate par path pattern (/static/*, /api/*, /documents/*)")
    print(f"     - Bandwidth savings = bytes served from edge / total bytes")
    print(f"     Objectifs :")
    print(f"     - Assets statiques : > 95% hit rate")
    print(f"     - Documents : > 60% hit rate (re-telechargements)")
    print(f"     - API : < 10% hit rate (personnalise, normal)")
    print(f"     Si le hit rate assets < 90%, verifier les headers.")

    # 6. Estimation cout
    print(f"\n  6. Estimation cout CDN (CloudFront) :")
    static_gb = 500       # Go/mois d'assets statiques
    docs_gb = 2000        # Go/mois de documents PDF
    api_gb = 100          # Go/mois de reponses API
    total_gb = static_gb + docs_gb + api_gb
    cost_per_gb = 0.085   # $/Go pour les premiers 10 To
    requests_millions = 50  # Millions de requetes/mois
    request_cost_per_10k = 0.01  # $/10K requetes HTTPS

    bandwidth_cost = total_gb * cost_per_gb
    request_cost = requests_millions * 1000 * request_cost_per_10k
    total_cost = bandwidth_cost + request_cost

    print(f"     Bandwidth :")
    print(f"       Assets statiques : {static_gb} Go/mois")
    print(f"       Documents PDF    : {docs_gb} Go/mois")
    print(f"       API responses    : {api_gb} Go/mois")
    print(f"       Total            : {total_gb} Go/mois")
    print(f"       Cout bandwidth   : {total_gb} * ${cost_per_gb} = ${bandwidth_cost:.0f}/mois")
    print(f"     Requetes :")
    print(f"       {requests_millions}M requetes * ${request_cost_per_10k}/10K = ${request_cost:.0f}/mois")
    print(f"     Total CDN          : ~${total_cost:.0f}/mois")
    print(f"     (Pour un SaaS moyen, $200-500/mois est typique)")


# =============================================================================
# HARD -- Exercice 1 : Cache e-commerce Black Friday
# =============================================================================

def hard_1_black_friday():
    """Solution pour le caching layer Black Friday."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Cache e-commerce Black Friday")
    print(SEPARATOR)

    products = 5_000_000
    product_size_kb = 2
    normal_reads = 50_000
    peak_reads = 1_000_000
    peak_writes = 40_000
    flash_deals = 100

    print(f"\n  1. Architecture multi-tier :")
    print(f"     L1 : In-process cache (Caffeine/dict)")
    print(f"       - Contenu : prix des flash deals, config promo, feature flags")
    print(f"       - TTL : 5-10 secondes (prix ne doivent pas etre stale > 10s)")
    print(f"       - Taille : ~50 Mo par instance")
    print(f"       - Avantage : 0 latence reseau, absorbe les cles ultra-hot")
    print(f"     L2 : Redis Cluster")
    print(f"       - Contenu : catalogue complet, sessions, paniers")
    print(f"       - TTL : 5 min (catalogue), 30 min (sessions)")
    print(f"       - Le prix a un TTL de 10s OU event-driven invalidation")
    print(f"     L3 : CDN (CloudFront)")
    print(f"       - Contenu : assets statiques, images produits")
    print(f"       - TTL : 1 an (versionne par hash)")

    # 2. Dimensionnement Redis
    catalog_gb = (products * product_size_kb * 1024) / (1024 ** 3)
    catalog_with_overhead = catalog_gb * 2.5
    # Redis peut faire ~100K ops/sec par noeud
    ops_per_node = 100_000
    nodes_for_throughput = math.ceil(peak_reads / ops_per_node)

    print(f"\n  2. Dimensionnement Redis :")
    print(f"     Memoire catalogue : {products:,} * {product_size_kb} Ko = {catalog_gb:.1f} Go brut")
    print(f"     Avec overhead Redis (2.5x) : {catalog_with_overhead:.1f} Go")
    print(f"     + sessions + paniers : ~5-10 Go supplementaires")
    print(f"     Total memoire : ~{catalog_with_overhead + 10:.0f} Go")
    print(f"     Noeuds pour {peak_reads:,} reads/sec : {peak_reads:,} / {ops_per_node:,} = {nodes_for_throughput} noeuds")
    print(f"     Avec replicas (1 par master) : {nodes_for_throughput * 2} noeuds")
    print(f"     Shard key : hash(product_id) % 16384 (Redis Cluster standard)")

    # 3. Flash deals
    print(f"\n  3. Flash deals :")
    print(f"     Probleme : a 14:00 exactement, 100 produits passent en promo.")
    print(f"     Des millions d'users chargent la page en meme temps.")
    print(f"     -> Cache stampede sur 100 cles simultanement.")
    print(f"     Solution :")
    print(f"     a) Cache warming 5 min AVANT le deal (charger les 100 produits dans")
    print(f"        L1 + L2 avec le nouveau prix)")
    print(f"     b) stale-while-revalidate : servir le cache existant pendant le rebuild")
    print(f"     c) Mutex par cle pour les misses residuels")
    print(f"\n     Reservation de stock avec Redis :")
    print(f"     SET stock:product_123 500   # Initialiser le stock")
    print(f"     DECR stock:product_123      # Atomique ! Retourne le stock restant")
    print(f"     Si DECR retourne < 0 : survente -> INCR pour annuler + refuser la commande")
    print(f"     DECR est atomique en Redis -> pas de race condition meme avec 100K requetes")

    # 4. Cache warming
    print(f"\n  4. Cache warming :")
    print(f"     Donnees a pre-charger :")
    print(f"     - 100 flash deals (prix, description, stock)")
    print(f"     - Top 10K produits les plus vus (historique)")
    print(f"     - Config globale (seuils promo, feature flags)")
    print(f"     Comment eviter de surcharger la DB :")
    print(f"     - Rate limiter : 1000 queries/sec max pendant le warm")
    print(f"     - Commencer 30 min avant le pic")
    print(f"     - Utiliser un read replica pour le warm (pas le master)")
    print(f"     Timing : T-30min (bulk warm) -> T-5min (refresh flash deals) -> T-0 (go)")

    # 5. Resilience
    print(f"\n  5. Resilience :")
    print(f"     Si Redis tombe pendant le Black Friday :")
    print(f"     a) Circuit breaker : detecte les timeouts Redis (> 5ms)")
    print(f"        Apres 10 echecs consecutifs -> ouvrir le circuit")
    print(f"     b) Fallback : servir les donnees du L1 cache (stale mais disponible)")
    print(f"     c) Rate limiter sur la DB : max 5K queries/sec (vs 50K sans cache)")
    print(f"        -> degrade mais ne tue pas la DB")
    print(f"     d) Page de file d'attente virtuelle si la charge depasse la capacite")
    print(f"     Detection : alerte si hit rate < 60% pendant 30 secondes")

    # 6. Monitoring
    print(f"\n  6. Monitoring (8 metriques) :")
    metrics = [
        ("Redis hit rate", "< 75%", "Cache sous-performant"),
        ("Redis eviction rate", "> 5K/s", "Cache trop petit pour le workload"),
        ("Redis latency p99", "> 5ms", "Redis surcharge ou reseau"),
        ("Redis memory usage %", "> 85%", "Risque d'eviction imminente"),
        ("DB connections active", "> 160/200", "Pool quasi-sature"),
        ("DB CPU %", "> 70%", "DB surchargee (cache inefficace)"),
        ("API error rate (5xx)", "> 0.1%", "Degradation visible par les users"),
        ("Stock DECR rate/sec", "> stock initial", "Survente potentielle"),
    ]
    for name, threshold, meaning in metrics:
        print(f"     - {name} : alerte si {threshold} ({meaning})")

    # Budget
    print(f"\n  Budget estimation :")
    redis_nodes = nodes_for_throughput * 2  # Avec replicas
    redis_cost_per_node = 800  # $/mois pour un r6g.xlarge (32 Go RAM)
    redis_cost = redis_nodes * redis_cost_per_node
    cdn_cost = 8000  # $/mois pour le pic Black Friday
    total = redis_cost + cdn_cost
    print(f"     Redis : {redis_nodes} noeuds * ${redis_cost_per_node}/mois = ${redis_cost:,}/mois")
    print(f"     CDN   : ~${cdn_cost:,}/mois (pic Black Friday)")
    print(f"     Total : ~${total:,}/mois (dans le budget de $50K)")


# =============================================================================
# HARD -- Exercice 2 : Post-mortem cache incident
# =============================================================================

def hard_2_postmortem():
    """Solution pour le post-mortem de l'incident cache."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Post-mortem — Le cache qui a casse le paiement")
    print(SEPARATOR)

    print(f"\n  1. Chaine causale complete :")
    chain = [
        ("PROCESSUS", "Script marketing ecrit directement en DB",
         "Guardrail manquant : toute ecriture doit passer par le service applicatif (API)"),
        ("ARCHITECTURE", "Le cache n'est pas invalide car le script contourne le service",
         "Guardrail manquant : CDC (Change Data Capture) pour capturer les writes DB directs"),
        ("ARCHITECTURE", "Stale cache pendant 5 min (TTL) avec des prix incorrects",
         "Guardrail manquant : TTL de 5 min trop long pour les prix (devrait etre 10-30s)"),
        ("PROCESSUS", "FLUSHALL comme reaction d'urgence",
         "Guardrail manquant : runbook d'incident qui interdit FLUSHALL en production"),
        ("ARCHITECTURE", "Cache stampede massif (15M cache miss simultanes)",
         "Guardrail manquant : mecanisme anti-stampede (mutex, stale-while-revalidate)"),
        ("ARCHITECTURE", "DB saturee (100% CPU, connexions epuisees)",
         "Guardrail manquant : circuit breaker + rate limiter entre app et DB"),
        ("MONITORING", "Erreurs 500 en cascade, site down 35 min",
         "Guardrail manquant : graceful degradation (servir le stale plutot que des 500)"),
    ]
    for category, cause, guardrail in chain:
        print(f"     [{category}] {cause}")
        print(f"       -> {guardrail}")

    print(f"\n  2. Le FLUSHALL etait-il la bonne decision ?")
    print(f"     NON. C'est l'erreur qui a transforme un probleme mineur")
    print(f"     (prix stale pendant < 5 min) en incident majeur (35 min de downtime).")
    print(f"\n     Alternatives au FLUSHALL :")
    print(f"     a) Invalider SEULEMENT les 500 cles concernees :")
    print(f"        for product_id in promo_products:")
    print(f"            redis.delete(f'product:price:{{product_id}}')")
    print(f"        Etaler sur 5 secondes (100 DEL/sec) pour eviter le stampede")
    print(f"     b) Attendre que le TTL expire naturellement (< 5 min)")
    print(f"        Le stale price est un probleme, mais 35 min de downtime est pire")
    print(f"     c) Reduire le TTL a 10s pour les cles de prix :")
    print(f"        redis.expire(f'product:price:{{id}}', 10)")
    print(f"        Les cles expirent naturellement en 10s sans stampede")

    print(f"\n     Si FLUSHALL etait absolument necessaire :")
    print(f"     1. Activer le circuit breaker sur la DB AVANT le flush")
    print(f"     2. Limiter le refill a 1000 queries/sec (rate limiter)")
    print(f"     3. Utiliser stale-while-revalidate pour servir le vieux cache")
    print(f"     4. Flusher par chunks (SCAN + DELETE) au lieu de FLUSHALL")

    print(f"\n  3. Architecture corrigee :")
    print(f"     a) Empecher les ecritures directes en DB :")
    print(f"        - Principe : tout changement de prix passe par l'API (pas de SQL direct)")
    print(f"        - Enforcement : le user DB du script a seulement SELECT, pas UPDATE")
    print(f"        - Backup : CDC (Debezium) surveille la table prices en temps reel")
    print(f"     b) 3 mecanismes contre l'inconsistance de prix :")
    print(f"        1. TTL court (10s) sur les cles de prix")
    print(f"        2. CDC via Debezium : capture les writes DB -> invalide le cache")
    print(f"        3. Double-check au checkout : relire le prix en DB avant de creer la commande")
    print(f"     c) Safe cache invalidation pattern :")
    print(f"        1. Invalider les cles par batch (100/sec)")
    print(f"        2. Utiliser le lock/mutex pour le rebuild")
    print(f"        3. stale-while-revalidate : servir l'ancien pendant le rebuild")

    print(f"\n  4. Architecture event-driven pour les prix :")
    print(f"     Option 1 : CDC avec Debezium")
    print(f"       PostgreSQL WAL -> Debezium -> Kafka topic 'price-changes'")
    print(f"       -> Consumer invalide les cles Redis correspondantes")
    print(f"       Avantage : capture TOUT, meme les scripts SQL directs")
    print(f"     Option 2 : Outbox pattern")
    print(f"       L'API ecrit dans la table prices ET dans une table outbox")
    print(f"       Un worker poll la table outbox et invalide le cache")
    print(f"       Avantage : transactionnel (write + event dans la meme TX)")
    print(f"     Recommandation : CDC (Debezium) car il capture aussi les scripts directs")

    print(f"\n  5. Resilience patterns :")
    print(f"     Circuit breaker (seuils concrets) :")
    print(f"       - CLOSED (normal) : < 160 connexions DB actives")
    print(f"       - OPEN (fallback) : > 160 connexions pendant 5 secondes")
    print(f"       - Fallback : servir le cache stale + header X-Cache-Stale: true")
    print(f"       - HALF-OPEN : apres 30s, laisser passer 10% du trafic pour tester")
    print(f"     Graceful degradation :")
    print(f"       - Servir les prix stale avec un bandeau 'prix indicatif'")
    print(f"       - Verifier le prix reel au checkout (DB directement)")
    print(f"       - Mieux vaut un prix stale que une erreur 500")

    print(f"\n     Runbook (10 etapes) :")
    steps = [
        "NE PAS FLUSHALL (jamais en production sans protection)",
        "Identifier les cles impactees (quel prefixe, combien)",
        "Activer le circuit breaker si pas deja actif",
        "Invalider les cles specifiques par batch (100/sec max)",
        "Monitorer le hit rate et le DB CPU en temps reel",
        "Si le hit rate chute < 50% : activer le rate limiter DB (1K QPS max)",
        "Si la DB est saturee : activer le fallback stale-cache (servir l'ancien)",
        "Communiquer l'incident en interne (Slack #incidents)",
        "Quand le cache est reconstruit (hit rate > 80%) : desactiver les protections",
        "Post-mortem dans les 24h : timeline, root cause, actions correctives",
    ]
    for i, step in enumerate(steps, 1):
        print(f"       {i:2d}. {step}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Execute toutes les solutions."""
    print("\n" + "=" * 60)
    print("  SOLUTIONS — JOUR 3 : CACHING & CDN")
    print("=" * 60)

    # Easy
    easy_1_cache_strategy()
    easy_2_cache_control()
    easy_3_redis_sizing()

    # Medium
    medium_1_social_feed()
    medium_2_cache_diagnosis()
    medium_3_cdn_strategy()

    # Hard
    hard_1_black_friday()
    hard_2_postmortem()

    print(f"\n{'=' * 60}")
    print("  FIN DES SOLUTIONS")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
Solutions -- Exercices Jour 7 : Design classiques

Ce fichier contient les walkthroughs pedagogiques des 3 exercices Easy.
Chaque solution suit le framework en 6 etapes : clarify, estimate,
high-level, deep dive, bottlenecks, extensions.

Usage:
    python 07-design-classiques.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Design Pastebin
# =============================================================================


def easy_1_pastebin():
    """Solution : design Pastebin."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Design Pastebin")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  Questions a poser :
  1. Taille max par paste ? (1-10 MB selon enonce)
  2. Expiration min/max ? (1h a jamais)
  3. Password protection attendue ?
  4. Syntax highlighting server-side ou client-side ?
  5. Paste editable apres creation ? Si oui, versioning ?
  6. Search sur le contenu ? (impacte Elasticsearch)
  7. Users authentifies ou anonymes ? Account necessaire ?
  8. API publique en plus de l UI ?
  9. Quel marche ? France seulement, global, GDPR ?

  ----- CAPACITY ESTIMATION -----
  1M DAU, reads:writes = 20:1, so users regardent ~20 pastes par jour.
  Writes : 1M * 1 paste/day = 1M pastes/jour = 11.6/s avg, ~35/s peak
  Reads  : 1M * 20 reads/day = 20M/jour = 230/s avg, ~700/s peak

  Taille moyenne : disons 50 KB (entre note court et dump de 10 MB).
  Storage/jour : 1M * 50 KB = 50 GB/jour = ~18 TB/an (avant compression).
  Apres gzip (compression 3-5x sur texte) : ~4-6 TB/an.

  Bandwidth CDN : 20M reads * 50 KB = 1 TB/jour out. CDN indispensable.

  ----- ARCHITECTURE HIGH-LEVEL -----
  Client --> CDN --> API Gateway --> App Service --> Metadata KV (Cassandra)
                                            |              |
                                            v              v
                                   Object Storage (S3)   Cache Redis
                                   pour le body
  (Workers separes : expiration cleanup, thumbnail/preview, abuse detection)

  ----- DEEP DIVE : STOCKAGE -----
  Deux niveaux :
  1. Metadata (Cassandra ou DynamoDB) :
     pastes (
       paste_id      TEXT PRIMARY KEY,    -- base62, ~8 chars
       user_id       TEXT,                -- nullable (anonymous)
       title         TEXT,
       language      TEXT,                -- 'python', 'js', ...
       s3_key        TEXT,                -- pointeur vers le body en S3
       created_at    TIMESTAMPTZ,
       expires_at    TIMESTAMPTZ,         -- NULL = never
       password_hash TEXT,
       size_bytes    INT,
       views         INT
     )
     Partition key = paste_id : O(1) lookup, uniforme.

  2. Body en Object Storage (S3 / GCS) :
     Les 50 KB moyens sont stockes hors du KV. Pourquoi ?
     - Moins cher par GB ($0.023/GB/mois S3 vs $0.25+ DynamoDB)
     - Pas de limite de taille par row
     - Acces direct depuis le CDN via presigned URL

  ----- CACHE STRATEGY -----
  - Cache Redis pour les pastes hot (viennent d'etre crees + trending)
  - Key : paste:{id} -> JSON metadata + body (si < 100 KB)
  - TTL : 5-15 min initial, extend sur chaque read (TTL-based LRU)
  - Hot paste viral : potentiellement des dizaines de k reads/sec.
    CDN devant le cache absorbe ca car on peut cacher la reponse
    complete HTTP (headers Cache-Control: public, max-age=60).

  Hit rate vise : 90%+ grace au CDN et au Redis.

  ----- BOTTLENECKS -----
  1. Hot paste viral :
     Probleme : un paste link poste sur HackerNews = 50K reads/s sur 1 paste.
     Solution : CDN HTTP cache (max-age=60). Le CDN absorbe 99% du trafic.
     Redis en backup. Le storage n'est pas touche.

  2. Expiration a grande echelle :
     Probleme : purge des pastes expires. On ne peut pas juste 'DELETE
     WHERE expires_at < NOW()' sur une table de 500M rows.
     Solution : TTL natif Cassandra (DELETE automatique). OU un worker
     qui scan par bucket de jour et delete en batch.

  3. Abuse (stockage de malware, dead drops) :
     Solution : scanner les pastes (regex / ML), rate limit par IP,
     captcha pour les gros pastes, signalement utilisateur.

  ----- EXTENSIONS -----
  - Versioning des pastes
  - Syntax highlighting via Prism.js cote client
  - Search sur les pastes publics (Elasticsearch)
  - API keys pour les integrations developpeurs
  - Embedding iframe sur d'autres sites
    """)


# =============================================================================
# EASY -- Exercice 2 : Design Instagram feed
# =============================================================================


def easy_2_instagram_feed():
    """Solution : design Instagram feed."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Design Instagram feed")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  1. Feed chronologique ou algorithmique ? (enonce = chrono)
  2. Stories aussi ou juste les posts ?
  3. Videos supportees ou photos seulement ?
  4. Multi-region (global) ?
  5. Likes / comments inclus dans le feed ?
  6. Max taille d'un follow graph (combien de suivis max par user) ?
  7. Expiration des photos ?

  ----- CAPACITY ESTIMATION -----
  500M DAU, 1 post/day, 20 feed views/day.

  Writes : 500M / 86400 = ~5800 posts/sec avg, ~17K peak
  Feed reads : 500M * 20 = 10B/day = 116K reads/sec avg, ~350K peak

  Photos : 2 MB * 500M = 1 PB/day de media -> 365 PB/an
  -> S3 + CDN obligatoire. Le stockage est le COST driver principal.

  Bandwidth out : 10B reads * 2 MB/photo * 0.1 photos affichees par feed = 2 PB/day
  -> CDN avec cache hit rate tres eleve (95%+) obligatoire.

  ----- ARCHITECTURE HIGH-LEVEL -----
  Upload path :
    Client -> CDN -> API Gateway -> Upload service -> S3
                                           |
                                           v
                                     Kafka (PhotoUploaded)
                                           |
                              +------------+-------------+
                              v            v             v
                        Image processor  Fanout worker   Analytics
                        (thumbnails)    (pre-compute     (ingest)
                                         feeds)

  Read path :
    Client -> CDN -> API Gateway -> Feed service -> Redis (precomputed feed)
                                                          |
                                                          v (miss ou tail)
                                                     Cassandra (posts)

  ----- DEEP DIVE : FANOUT ON WRITE vs FANOUT ON READ -----
  Fanout on write (push) :
    Quand Alice post, on pre-ecrit post_id dans la feed Redis de chacun
    de ses followers.
    - Read : O(1), RANGE sur un sorted set Redis.
    - Write : O(N) ou N = nombre de followers.
    - Mauvais pour les celebrites (Kylie Jenner 400M followers = 400M
      writes par post, serveur KO).

  Fanout on read (pull) :
    On stocke juste le post. A chaque feed open, on va chercher les posts
    de chaque suivi et on merge.
    - Read : O(M * log P) ou M = suivis et P = posts/suivi.
    - Write : O(1).
    - Mauvais pour les users avec beaucoup de suivis (scan de 500 timelines).

  SOLUTION HYBRIDE (Instagram reel) :
    - Users normaux (< 100K followers) : fanout on write.
    - Celebrites (> 100K followers) : fanout on read.
    - Quand un user ouvre son feed :
      1. Recupere sa feed precalculee (users normaux)
      2. Pull les posts recents des celebrites qu'il suit (< 10 typiquement)
      3. Merge et trie chronologiquement.
    - Le compromis est ajustable : le seuil 100K peut etre adapte.

  Calcul :
    Sans hybride : 500M users * 500 followers moyen = 250B fanout writes/jour.
    Avec hybride (exclure les 1000 celebrites) : ~10% de reduction sur
    le total mais elimine les pics catastrophiques de 400M/post.
    Les celebrites (quelques milliers) n'impactent pas le systeme.

  ----- STORAGE PHOTOS -----
  - Upload direct vers S3 via presigned URLs (pas de proxy par l'API).
  - Worker async : genere thumbnails (150x150, 320x320, 1080x1080),
    formats (jpg, webp, avif), et watermark eventuel.
  - CDN devant S3 avec cache tres long (immutable, max-age=1y)
    car les photos sont immutables une fois uploadees.
  - Hot regions : multi-region replication S3, CDN PoP globalement.

  ----- BOTTLENECKS -----
  1. Celebrity problem : resolu par le hybride decrit ci-dessus.

  2. Cost du storage (365 PB/an) :
     - Compression aggressive (AVIF remplace JPEG, -40% taille)
     - Tiering : photos vieilles > 90 jours -> S3 Infrequent Access
     - Deduplication via content hash (pas critique pour Instagram
       car les photos sont uniques, mais utile contre les reposts)

  3. Cold start pour les nouveaux users :
     - Pas de feed precalcule : on fallback sur fanout-on-read des
       users qu'il suit. Premier feed charge plus lentement (~500 ms).

  ----- EXTENSIONS -----
  - Feed algorithmique (ML ranking : engagement, relation, timing)
  - Stories (retention 24h, flux separe)
  - Explore tab (recommendation engine)
  - Live video (streaming RTMP -> HLS)
  - Direct messages
  - Insights / analytics pour les createurs
    """)


# =============================================================================
# EASY -- Exercice 3 : Notification System
# =============================================================================


def easy_3_notification_system():
    """Solution : design Notification System."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : Notification System")
    print(SEPARATOR)

    print("""
  ----- CLARIFICATION -----
  1. Qui declenche les events ? (autres services internes)
  2. Quels canaux ? (push, email, SMS, in-app, webhook)
  3. Templates : maintenus par qui, en quelle langue ?
  4. I18n requis ? (EN, FR, ES...)
  5. Preferences granularite ? (par type de notif ? par canal ?)
  6. SLA latence : temps reel (< 1s) ou batch (< 5 min) ?
  7. Quels tiers en prod ? (SendGrid, Twilio, FCM, APNs deja negocies ?)
  8. Compliance : GDPR (opt-out, unsubscribe), CAN-SPAM ?

  ----- CAPACITY ESTIMATION -----
  100M users * 10 notifs/day = 1B notifs/day = 11.6K/sec avg, 35K peak
  Events entrants : souvent moins que notifs car 1 event -> N canaux
    -> Disons 500M events/day = 5.8K events/sec avg, 17K peak
  Storage log : 500 bytes/notif * 1B = 500 GB/day pour l'audit (retention 30j)

  ----- ARCHITECTURE HIGH-LEVEL -----
  Event Producers --> Kafka (topic: notifications-raw)
                           |
                           v
                  [Notification Router]
                           |
                  +--------+---------+--------+
                  v        v         v        v
             Pref Svc  Template  Dedup   Rate Limit
                  |        |         |        |
                  +--------+---------+--------+
                           |
                           v
                     Kafka (topic: notifications-ready)
                           |
              +------------+------------+------------+
              v            v            v            v
         Push Worker  Email Worker  SMS Worker   In-App Worker
              |            |            |            |
              v            v            v            v
            FCM/APNs   SendGrid      Twilio      WebSocket
              |            |            |            |
              v            v            v            v
                    Delivery DB + DLQ (on failure)

  ----- FLOW D UN EVENT 'new_follower' -----
  1. UserService detecte qu'Alice a commence a suivre Bob.
  2. UserService publie dans Kafka 'notifications-raw' :
     {
       event_id: 'evt_xyz789',
       event_type: 'new_follower',
       actor_id: 'alice',
       recipient_id: 'bob',
       created_at: ...
     }
  3. Notification Router consomme l'event :
     3a. Dedup check : has evt_xyz789 already been processed ? (Redis SET)
     3b. Fetch Bob preferences : Bob a bien opt-in aux notifs follower
         par push et email, pas par SMS.
     3c. Rate limit check : Bob a deja recu 5 push dans l'heure ? (token bucket)
     3d. Render template : charge le template 'new_follower' en FR
         (langue de Bob), injecte 'Alice te suit maintenant'.
     3e. Publish dans 'notifications-ready' avec 1 msg par canal.
  4. Push Worker consomme :
     4a. Lookup Bob device tokens (user_devices table).
     4b. Send to FCM / APNs in parallel.
     4c. Sur success : log en DB (delivery_log).
     4d. Sur failure : retry 3x avec backoff, puis DLQ.

  ----- DEDUPLICATION -----
  Chaque event a un 'event_id' unique genere par le producer.
  Au debut du Notification Router :
    if redis.set(f'notif:dedup:{event_id}', '1', nx=True, ex=3600):
        process()
    else:
        skip()  # deja vu

  Ainsi meme si Kafka livre 2x le meme message (at-least-once), on
  ne process qu'une fois. TTL 1h = protection contre doublons courts.
  Pour une protection plus longue, stocker en DB avec contrainte UNIQUE
  sur event_id (table 'processed_events').

  ----- RATE LIMITING -----
  Granularite : par (user_id, channel, notification_type).
  Exemple : Bob peut recevoir max 20 push/heure pour les 'new_follower'.
  Algo : token bucket en Redis, key = f'rl:{user}:{channel}:{type}'.
  Si depasse : on skip + log (metric 'rate_limited_count').
  Alerte si > X% des notifs sont skipees (signal : un producer en loop).

  ----- RETRY STRATEGY -----
  Cas : SendGrid renvoie 503 pour 1 email specifique.
  Strategy :
  - Exponential backoff : 1s, 2s, 4s, 8s, 16s jusqu a 5 tentatives.
  - Jitter pour eviter le thundering herd.
  - Apres 5 echecs : envoi en DLQ Kafka.
  - Un worker 'dlq-handler' inspecte la DLQ :
    - Si l email est definitif (format invalide) : log et drop.
    - Si c'est transient (503, 502) : republish dans 'notifications-ready'
      pour une nouvelle tentative 1h plus tard.
  - Alerte ops si DLQ > 1000 items (quelque chose de systemique).

  Circuit breaker sur l'appel a SendGrid :
  - Apres 10 echecs consecutifs -> OPEN, on arrete de taper pendant 30s.
  - Pendant ce temps, les messages sont mis en queue locale.
  - HALF_OPEN apres 30s : on teste avec 1 requete.
  - Si OK -> CLOSED, on flush la queue.

  ----- TRADEOFFS DISCUTES -----
  1. Push vs Poll :
     Push (Kafka) = temps reel, latence faible, complexite plus elevee.
     Poll = simple, batching possible, latence 1-5 min.
     Pour 35K peak/s, push est obligatoire.

  2. Sync vs Async :
     Les events sont traites async via Kafka. L'event producer ne doit
     JAMAIS bloquer sur l'envoi de la notif. Il publish et oublie.
     Latence utilisateur finale : 1-5 secondes pour les non-critiques.

  3. Template service centralise vs embarque :
     Centralise (microservice dedie) = gestion produit du contenu,
     i18n, A/B test. Embarque dans le router = plus simple mais pas
     scalable en equipe.
     Choix : centralise pour une app a 100M users.

  ----- EXTENSIONS -----
  - A/B testing sur les templates
  - Digest quotidien (batch des notifs non-critiques dans un email)
  - Quiet hours (ne pas envoyer entre 22h et 8h locale)
  - Analytics : delivery rate, open rate, click rate, unsubscribe rate
  - Webhook callback vers le producer pour confirmation
    """)


def main():
    easy_1_pastebin()
    easy_2_instagram_feed()
    easy_3_notification_system()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 7.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

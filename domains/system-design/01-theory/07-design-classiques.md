# Jour 7 — Design classiques (entretiens systemes)

## Pourquoi ce jour change tout

Les 6 jours precedents t'ont donne les briques : scalabilite, DB, cache, queues, LB, API design. Aujourd'hui, on assemble. Un **entretien system design** dure 45 a 60 minutes. L'intervieweur te donne un enonce vague ("Design Twitter") et te laisse piloter. C'est l'exercice le plus stressant du parcours tech senior, et celui qui decide le plus souvent des offres L5/L6 (senior/staff) chez FAANG.

La bonne nouvelle : **ce n'est pas un test de connaissance, c'est un test de methode**. Tu n'as pas besoin de connaitre 200 systemes. Tu as besoin d'un framework clair que tu appliques a n'importe quel enonce. Ce jour te donne ce framework et te fait pratiquer sur 3 designs de reference (URL shortener, Twitter, Chat).

**Key takeaway** : Un bon candidat pose 5 minutes de questions avant de dessiner quoi que ce soit. Un mauvais candidat saute direct sur "je vais mettre Redis et Kafka". L'ordre importe plus que le contenu.

---

## Le framework en 6 etapes

```
1. Clarify        (5 min)   — questions, scope, SLA
2. Estimate       (5 min)   — capacity, QPS, storage
3. High-level     (10 min)  — boites et fleches, data flow
4. Deep dive      (15 min)  — 1-2 composants en detail
5. Bottlenecks    (5 min)   — ou ca casse, comment scaler
6. Extensions     (5 min)   — ce qu'on ferait avec plus de temps
```

### Etape 1 — Clarify (5 minutes, obligatoire)

Ne jamais commencer a dessiner sans cette etape. L'intervieweur te teste sur ta capacite a **reduire l'ambiguite**.

**Questions obligatoires :**
- Quel est le **core feature** ? ("Design Twitter" = tweets + timeline + follow, ou ca inclut les DMs, trending, hashtags, ads ?)
- Qui sont les **utilisateurs** ? (B2C mondial ? B2B ? combien ?)
- Quel est le **volume** ? (100K users ou 500M ?)
- Read-heavy ou write-heavy ? (Twitter = read-heavy, chat = write-heavy)
- SLA attendu ? (99.9% ? 99.99% ? latence p99 ?)
- Multi-region requis ?
- Questions specifiques au probleme : "Les tweets peuvent-ils etre edites ? Quelle longueur ? Media ?"

Ecris les reponses au tableau. Tu y reviendras.

### Etape 2 — Estimate (5 minutes)

Chiffre vite pour dimensionner. Approximations grossieres, pas de calculs precis.

**Metriques a estimer :**
- **Daily Active Users (DAU)**
- **Actions par user par jour** (ex : 5 tweets posts, 50 tweets lus)
- **QPS (queries per second)** moyen et peak (peak = 3x moyen typique)
- **Stockage** par action * retention
- **Bandwidth** (si media)

**Exemple Twitter :**
- 200M DAU * 2 tweets/jour = 400M tweets/jour = **4600 tweets/sec moyen, ~15K peak**
- 200M DAU * 100 timeline reads/jour = 20B reads/jour = **230K reads/sec moyen, ~700K peak**
- 300 bytes par tweet * 400M = 120 GB/jour = **44 TB/an**
- Media : 10% des tweets avec 1 MB moyen = 40 TB/jour = **15 PB/an** (CDN obligatoire)

Ces chiffres guident toutes les decisions suivantes. Ils montrent a l'intervieweur que tu ne design pas a l'aveugle.

### Etape 3 — High-level (10 minutes)

Dessine les **composants majeurs** et les **flux**. Rien de detaille. L'intervieweur doit comprendre le systeme en 30 secondes en regardant ton dessin.

**Composants typiques :**
- Clients (web, mobile)
- CDN
- API Gateway / Load Balancer
- Application services (stateless)
- Caches (Redis)
- DBs (SQL et/ou NoSQL)
- Message queues (Kafka)
- Object storage (S3)
- Search (Elasticsearch)

**Flux a dessiner** (les 2-3 plus importants) :
- Create : comment un tweet est ecrit et persiste
- Read : comment une timeline est servie
- (optionnel) : notifications, search

### Etape 4 — Deep dive (15 minutes)

L'intervieweur va pointer 1 ou 2 composants et te dire "detaille ca". Tu dois plonger :
- Schema de la DB (tables, indexes, cles)
- Algorithme utilise (ex : fanout on write vs fanout on read)
- Gestion du sharding / partitioning
- Cache strategy
- Gestion de la consistance

C'est la que tes jours 1 a 6 paient. Tu ressors les briques de maniere pertinente.

### Etape 5 — Bottlenecks (5 minutes)

Anticipe ou ca va casser a l'echelle :
- **Hot partitions** (ex : Justin Bieber a 100M followers)
- **Read amplification** (1 write -> 100M reads)
- **Single point of failure**
- **Ecritures concurrentes** sur la meme ressource

Propose des **mitigations** : sharding, cache, precomputation, queue...

### Etape 6 — Extensions (5 minutes)

Ce que tu ferais si tu avais plus de temps. Ca montre que tu as une vision product en plus de technique :
- Analytics / monitoring
- ML / recommandations
- Spam / moderation
- Multi-region / disaster recovery
- Feature evolutions (dark mode, export, API publique)

---

## Design #1 — URL Shortener (TinyURL)

### 1. Clarify

- Core : generer une URL courte a partir d'une longue, rediriger vers l'originale.
- Users : 10M DAU global.
- Read vs Write : tres read-heavy (100:1 typique).
- SLA : redirection < 100 ms p99, 99.99% uptime.
- Custom short URLs (vanity URLs) ? Oui, optionnel.
- TTL (expiration) ? Oui, optionnel.
- Analytics (clics par URL) ? Oui.

### 2. Estimate

- 10M DAU * 100 clicks/day = **1B redirects/day = 12K QPS moyen, 35K peak**
- 10M DAU * 5 shortens/day = **50M shortens/day = 580 QPS moyen, 2K peak**
- 50M * 365 = 18B URLs/an, avec 500 bytes par entree = **9 TB/an**
- Cache des URLs chaudes : 20% du trafic sur 5% des URLs -> ~500 MB de cache

### 3. High-level

```
[Client] -> [CDN] -> [LB L7] -> [API shortener] -> [KV Store (Cassandra)]
                                       |                    |
                                       v                    v
                                     [Redis cache]       [Kafka -> analytics]
```

### 4. Deep dive : encoding et KV

**Encoding :** comment generer un code court a partir d'une longue URL ?

**Option A : hash de l'URL (MD5 -> base62)**
- `md5(url)[:7]` donne un code court, mais 2 URLs identiques = meme code (deduplication gratuite).
- Probleme : collisions possibles (peu probables avec 7 chars en base62 = 3.5 trillions).
- Il faut gerer le cas "meme code pour URLs differentes" = verif puis retry avec salt.

**Option B : compteur global + base62**
- Un counter global auto-incremente. Ex : 1, 2, 3, ... 58, 59, 60...
- Encoder en base62 (a-z A-Z 0-9) : 1 -> "1", 62 -> "10", ...
- 7 chars = 62^7 = 3.5 trillions d'URLs possibles, largement.
- Probleme : besoin d'un compteur global, single point of failure.
- Solution : range allocation. Chaque serveur reserve un batch (ex : 100K ids) et les consomme localement.

**Schema de table** (Cassandra) :
```
urls (short_code PRIMARY KEY, long_url, created_by, created_at, expires_at)
```
`short_code` est la partition key -> O(1) lookup.

**Flow de redirection :**
```
GET /abc123
-> check Redis cache -> HIT : 301 redirect (2 ms)
                     -> MISS : query Cassandra -> populate cache -> 301 redirect (15 ms)
```

### 5. Bottlenecks

- **Compteur global** : resolu par range allocation.
- **URLs chaudes** : cache Redis + TTL, hit rate > 95% facile.
- **Cassandra hotspot** : pas de hotspot car `short_code` est uniformement distribue.
- **Analytics** : on ne veut pas bloquer la redirection pour logger. -> async via Kafka.

### 6. Extensions

- Vanity URLs : `POST /shorten` avec `custom_code`. Check unicite.
- Expiration : TTL Cassandra + Redis.
- Analytics : dashboards par URL (top countries, referrers).
- Rate limiting anti-spam.
- Custom domains (branded short URLs).

---

## Design #2 — Twitter timeline (la reference absolue)

### 1. Clarify

- Core : post a tweet, follow users, voir ma home timeline (tweets des gens que je suis).
- Users : 200M DAU.
- Read vs Write : tres read-heavy. Ratio ~200:1.
- Tweet = 280 chars + optional media.
- Timeline ordonne chronologiquement (pas algorithmique dans cette version).

### 2. Estimate

- 200M * 2 tweets/day = **4.6K tweets/sec moyen, 15K peak**
- 200M * 50 timeline reads/day = **115K reads/sec moyen, 350K peak**
- 300 B/tweet * 400M = 120 GB/day = **44 TB/an** (sans media)
- Followers distribution : long tail, avec mega-users (Justin Bieber = 100M followers)

### 3. High-level

```
                    +----------------+
[Client]--> [CDN] ->| API Gateway    |
                    +----------------+
                       |         |
           +-----------+         +------------+
           v                                  v
   [Write path]                        [Read path]
   Post tweet                          Get timeline
           |                                  |
           v                                  v
   [Tweet Service]                    [Timeline Service]
           |                                  |
           v                                  v
    [Cassandra tweets]                 [Redis timeline cache]
           |                                  ^
           v                                  |
    [Kafka event]                      [Fanout worker]
           |                                  ^
           +----------------------------------+
```

### 4. Deep dive : Fanout on Write vs Fanout on Read

C'est LA question. Cette discussion est attendue.

**Fanout on Write (push)** :
Quand Alice poste un tweet, on ecrit ce tweet dans les timelines caches de **tous ses followers** (fanout).
- Read : tres rapide, la timeline est pre-calculee. `LRANGE timeline:{user_id}`.
- Write : lent, ecriture dans N boites aux lettres. Pour Justin Bieber = 100M ecritures par tweet.

**Fanout on Read (pull)** :
On ne stocke que le tweet. Quand un user ouvre sa home, on pull les tweets de tous les gens qu'il suit et on merge.
- Read : lent, il faut scanner N timelines.
- Write : tres rapide, juste l'insert du tweet.

**Le compromis Twitter (hybride)** :
- **Fanout on write** pour les users normaux (< 10K followers).
- **Fanout on read** pour les mega-users (Bieber, Obama, ...). Quand un user ouvre sa home, on merge (fanout-write cache) + (pull des mega-users qu'il suit).
- Best of both worlds : gere le long tail sans ecrouler le systeme pour les posts de Bieber.

**Schema Redis timeline cache** :
```
Key : timeline:{user_id}
Type : sorted set
Score : tweet_timestamp
Member : tweet_id
Max size : 1000 (on ne cache que les 1000 tweets les plus recents)
```

**Schema Cassandra tweets** :
```
tweets (
  tweet_id UUID,
  user_id TEXT,
  content TEXT,
  media_url TEXT,
  created_at TIMESTAMP,
  PRIMARY KEY ((user_id), created_at DESC)
)
```
Partition par user_id + tri chronologique inverse = les tweets d'un user sont faciles a lister.

### 5. Bottlenecks

- **Celebrity problem** : fanout-on-write impossible pour Bieber. -> Hybride.
- **Read amplification** : 1 tweet = 100M reads potentiels. -> Cache Redis. Aggressive caching.
- **Consistency** : OK si un tweet met 1-2 secondes a apparaitre pour les followers (eventually consistent).
- **Media** : CDN obligatoire, stockage sur S3.

### 6. Extensions

- Timeline algorithmique (ML ranking)
- Search (Elasticsearch sur les tweets)
- Notifications push
- Trending topics (streaming, sliding window counter)
- Replies / threads
- Spam / moderation

---

## Design #3 — Real-time Chat (WhatsApp-like)

### 1. Clarify

- 1-to-1 messaging et group chat (< 256 membres).
- Users : 100M DAU, 1B messages/jour.
- Delivery : "sent", "delivered", "read".
- Message history : recuperable sur plusieurs appareils.
- Online status ? Oui.
- Media (images, videos) ? Oui.

### 2. Estimate

- 1B messages/day = **11K QPS moyen, 35K peak**
- 1 message = ~300 bytes * 1B = 300 GB/day = **110 TB/an**
- Media separe : ~2 MB moyen * 200M media/day = **70 PB/an** sur S3 + CDN
- 100M connexions WebSocket permanentes simultanees (peak) -> ~500-1000 serveurs WebSocket (chaque serveur tient 100-200K conns)

### 3. High-level

```
[Mobile client] <-- WebSocket --> [Chat Service (stateful)]
                                         |
                                         v
                            [Message broker (Kafka / Redis)]
                                         |
                            +------------+-----------+
                            v            v           v
                      [Persist DB]  [Push notif]  [Media storage/CDN]
                       Cassandra     FCM/APNs        S3
```

Les clients tiennent une connexion **WebSocket permanente** avec un serveur chat. Quand Alice envoie un message a Bob :
1. Alice -> WS -> Chat Service A.
2. Chat Service A persiste dans Cassandra.
3. Chat Service A cherche quel serveur WS tient la connexion de Bob (via un registry Redis).
4. Chat Service A publie le message sur le canal du server de Bob (via pub/sub Redis ou Kafka).
5. Chat Service B recoit et pousse vers Bob via sa WebSocket.
6. Si Bob est offline : FCM/APNs push notification + stockage pour rattrapage.

### 4. Deep dive : stockage des messages et ordering

**Schema Cassandra :**
```
messages (
  conversation_id TEXT,
  message_id TIMEUUID,
  sender_id TEXT,
  content TEXT,
  media_url TEXT,
  status TEXT,         -- sent / delivered / read
  PRIMARY KEY ((conversation_id), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC)
```

Partition par `conversation_id` (1 partition par chat 1-to-1 ou par groupe). Tri par TIMEUUID = ordre chronologique et deduplication gratuite.

**Pourquoi TIMEUUID et pas timestamp ?** TIMEUUID = timestamp + random. Deux messages envoyes a la meme milliseconde ont des IDs differents. Timestamp nu causerait des collisions sur les gros groupes.

**Ordering** : la partition cassandra garantit l'ordre. Mais les messages arrivent potentiellement "out of order" a cause du reseau. Chaque message est timestampe cote serveur (moment de reception), c'est ce timestamp qui determine l'ordre final.

### 5. Bottlenecks

- **Hot conversation** : un groupe de 256 membres avec trafic intense. Partition chaude. -> Sub-partitioning par jour : `conversation_id + day_bucket`.
- **Fanout des groupes** : envoyer un message de groupe = N ecritures WS. Pour 256 membres -> 256 pushes. Gerable.
- **Registry des connexions** : ou est Bob ? Redis hash `user_to_server`. Update a chaque connect/disconnect.
- **Push notifications** : si Bob est offline, rediriger vers FCM/APNs sans bloquer.
- **Reconnexion** : en WS, les clients doivent resync les messages manques depuis leur dernier `last_message_id` connu.

### 6. Extensions

- End-to-end encryption (Signal protocol)
- Status messages (typing, online, last seen)
- Message reactions et edit
- Voice/video calls (WebRTC)
- Multi-device sync
- Archive / search

---

## Flash cards

**Q1** : Quelles sont les 6 etapes du framework d'entretien system design ?
**R** : Clarify, Estimate, High-level, Deep dive, Bottlenecks, Extensions.

**Q2** : Qu'est-ce que le "celebrity problem" dans Twitter et comment le resoudre ?
**R** : Un user avec 100M followers rend fanout-on-write impossible (100M ecritures par tweet). Solution : hybride. Fanout-on-write pour les users normaux, fanout-on-read pour les mega-users.

**Q3** : Pourquoi utiliser TIMEUUID plutot qu'un timestamp pour les message_id d'un chat ?
**R** : TIMEUUID = timestamp + aleatoire. Evite les collisions quand plusieurs messages arrivent a la meme milliseconde. Garde l'ordre chronologique tout en ayant unicite.

**Q4** : Dans un URL shortener, quelle est la difference entre hash et compteur pour generer les short codes ?
**R** : Hash (md5 -> base62) : deduplication gratuite, mais gestion des collisions. Compteur : pas de collision, mais besoin d'un counter global distribue (range allocation).

**Q5** : Que faut-il toujours faire AVANT de dessiner en entretien system design ?
**R** : Clarifier le scope et estimer les ordres de grandeur. Sans ca, tu designs a l'aveugle et tu rates les tradeoffs critiques.

---

## Key takeaways

1. **Framework en 6 etapes** : clarify -> estimate -> high-level -> deep dive -> bottlenecks -> extensions. Ne jamais sauter clarify et estimate.
2. **Capacity estimation** est essentielle. Fais des ordres de grandeur, pas des calculs precis.
3. **URL shortener** : encoding par compteur + base62, cache Redis agressif, KV store partitionne.
4. **Twitter** : hybride fanout-on-write/read pour resoudre le celebrity problem. Timeline precalculee en sorted set Redis.
5. **Chat** : WebSocket + registry de connexions, persist en Cassandra avec TIMEUUID, push notifs pour offline.
6. **Sharding par entite logique** : user_id pour Twitter, conversation_id pour chat, short_code pour URL.
7. **Cache est toujours present** dans les designs read-heavy. 90%+ du trafic doit etre absorbe avant la DB.
8. **En entretien, parler des tradeoffs est plus important que d'avoir la "bonne" reponse**. Il n'y a jamais UNE reponse en system design, seulement des choix justifies.

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **System Design Interview Vol 1** (Alex Xu, ByteByteGo 2020) — 16 designs reference avec 188 diagrammes : URL shortener, Twitter timeline, chat system, web crawler. Le livre etalon des entretiens FAANG. https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF
- **System Design Interview Vol 2** (Alex Xu & Sahn Lam, ByteByteGo 2022) — sequel : payment system, Google Drive, YouTube, search autocomplete, distributed message queue. Plus avance sur les tradeoffs. https://www.amazon.com/System-Design-Interview-Insiders-Guide/dp/1736049119
- **The System Design Primer** (Donne Martin, GitHub, 200K+ stars) — repo communautaire couvrant scalabilite, CAP, replication, sharding, plus questions d'entretien resolues. Inclut Anki decks. https://github.com/donnemartin/system-design-primer
- **Designing Data-Intensive Applications** (Martin Kleppmann, O'Reilly 2017) — Part III (Derived Data) explique les architectures lambda/kappa qui sous-tendent Twitter, Netflix, search engines. https://dataintensive.net/

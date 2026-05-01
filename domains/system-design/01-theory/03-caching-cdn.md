# Jour 3 — Caching & CDN

## Pourquoi le cache est le levier #1 de performance

**Exemple d'abord** : Tu as un endpoint API qui retourne le profil d'un utilisateur. Sans cache, chaque requete fait : DNS lookup -> load balancer -> app server -> query PostgreSQL -> serialisation JSON -> reponse. Latence typique : 15-50ms. Avec un cache Redis devant la DB, la meme requete prend < 1ms. Et si tu mets un CDN devant ton API, les clients proches d'un edge server obtiennent la reponse en < 5ms sans meme toucher ton backend.

Le cache exploite un principe fondamental : **la localite temporelle**. Si une donnee a ete lue recemment, elle sera probablement relue bientot. Les 20% des donnees les plus populaires representent souvent 80% des acces (distribution Zipf). Un cache de quelques Go peut absorber la majorite du trafic et diviser la charge DB par 10.

**Key takeaway** : Le cache n'est pas une optimisation optionnelle. C'est un composant structurant de l'architecture. Un systeme sans cache, c'est comme un processeur sans L1/L2 — techniquement fonctionnel, mais inutilisable a l'echelle.

---

## Les niveaux de cache

Le cache est present a **chaque couche** d'un systeme, du transistor au continent. Comprendre ces niveaux, c'est comprendre ou optimiser.

| Niveau | Technologie | Taille typique | Latence | Gere par |
|---|---|---|---|---|
| **L1 CPU cache** | SRAM | 64 Ko | ~1 ns | Hardware |
| **L2 CPU cache** | SRAM | 256 Ko - 1 Mo | ~3-5 ns | Hardware |
| **L3 CPU cache** | SRAM | 8-64 Mo | ~10-20 ns | Hardware |
| **RAM (page cache OS)** | DRAM | 16-512 Go | ~100 ns | Kernel |
| **Application cache** | In-process (dict, guava, caffeine) | Mo - quelques Go | ~100 ns - 1 us | Code applicatif |
| **Distributed cache** | Redis, Memcached | Go - To | 0.1 - 1 ms (reseau) | Infrastructure |
| **CDN edge cache** | Nginx, Varnish, CloudFront | To | 5 - 50 ms | Provider CDN |
| **Browser cache** | Disque local client | Mo - Go | 0 ms (pas de requete) | HTTP headers |

### Pourquoi autant de niveaux ?

Chaque niveau est un compromis **taille vs latence**. Plus le cache est rapide, plus il est petit et cher. L'idee est d'avoir des "filtres" successifs : si le L1 miss, on essaie le L2, puis le L3, puis la RAM, puis Redis, puis la DB. A chaque niveau, on absorbe une portion du trafic.

**Analogie** : Pense a une bibliotheque. Ton bureau (L1) a 5 livres que tu utilises constamment. L'etagere dans ta chambre (L2) en a 100. Le salon (L3) en a 500. Le garage (RAM) contient des cartons. La bibliotheque municipale (Redis), c'est des milliers de livres. Et Amazon (DB), c'est tous les livres du monde. Tu ne vas a Amazon que si aucun niveau plus proche n'a le livre.

### Chiffres cles pour les entretiens

| Acces | Latence | Rapport vs DB |
|---|---|---|
| Redis GET | 0.1 - 0.5 ms | ~100x plus rapide que PostgreSQL |
| Memcached GET | 0.1 - 0.5 ms | ~100x plus rapide que PostgreSQL |
| PostgreSQL SELECT (indexe) | 5 - 15 ms | Reference |
| CDN hit (meme continent) | 5 - 30 ms | Mais pas de charge backend |
| CDN hit (autre continent sans CDN) | 100 - 300 ms | La latence reseau domine |

---

## Strategies de cache

### 1. Cache-Aside (Lazy Loading)

C'est la strategie la plus repandue. L'application gere explicitement le cache.

**Flow :**
```
1. L'app recoit une requete pour la cle K
2. L'app verifie le cache
   - HIT  -> retourner la valeur du cache
   - MISS -> lire depuis la DB, ecrire dans le cache, retourner la valeur
3. Pour les ecritures : ecrire dans la DB, puis invalider le cache (DELETE, pas SET)
```

**Code pattern :**
```python
def get_user(user_id: str) -> dict:
    # 1. Verifier le cache
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    # 2. Cache miss -> lire depuis la DB
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)

    # 3. Ecrire dans le cache avec TTL
    redis.setex(f"user:{user_id}", 300, json.dumps(user))  # TTL = 5 min

    return user

def update_user(user_id: str, data: dict):
    # 1. Ecrire dans la DB
    db.execute("UPDATE users SET ... WHERE id = %s", user_id)

    # 2. INVALIDER le cache (pas le mettre a jour !)
    redis.delete(f"user:{user_id}")
```

**Pourquoi DELETE et pas SET lors de l'ecriture ?** Si tu fais un SET, il y a une race condition : entre le write DB et le SET cache, un autre thread peut lire la vieille valeur de la DB et la mettre dans le cache. Le DELETE force un cache miss qui lira la valeur fraiche.

**Avantages :**
- Simple a implementer et a raisonner
- Le cache ne contient que les donnees effectivement demandees (pas de gaspillage)
- Resilient : si le cache tombe, l'app continue via la DB (degrade mais fonctionnel)

**Inconvenients :**
- Premiere requete toujours lente (cache miss)
- Risque de donnees stale entre le write DB et l'invalidation du cache
- Cache stampede possible (voir section dediee)

### 2. Write-Through

Chaque ecriture passe par le cache, qui ecrit ensuite dans la DB.

**Flow :**
```
1. L'app ecrit dans le cache
2. Le cache ecrit de maniere synchrone dans la DB
3. Confirmation a l'app seulement quand les deux sont OK
```

**Quand l'utiliser :**
- Quand la consistance entre cache et DB est critique
- Quand les lectures sont frequentes ET les ecritures moderees
- Quand tu ne peux pas tolerer de stale data

**Garantie de consistance :** Le cache est toujours a jour car chaque write le traverse. Pas de fenetre de stale data.

**Inconvenients :**
- Latence d'ecriture plus elevee (cache + DB synchrone)
- Le cache contient potentiellement des donnees jamais lues (gaspillage)
- Plus complexe a implementer (le cache doit connaitre la DB)

### 3. Write-Behind (Write-Back)

L'ecriture va dans le cache, et le cache flush vers la DB de maniere asynchrone en batch.

**Flow :**
```
1. L'app ecrit dans le cache
2. Confirmation immediate a l'app
3. Le cache accumule les ecritures et les flush vers la DB en batch (toutes les N secondes ou toutes les N ecritures)
```

**Quand ca vaut le coup :**
- Write-heavy workloads (compteurs, analytics, metrics)
- Quand la latence d'ecriture est critique
- Quand on peut tolerer une perte de donnees en cas de crash du cache

**Risques :**
- **Perte de donnees** : si le cache crash avant le flush, les ecritures non persistees sont perdues
- **Complexite** : gestion du buffer, retry, idempotence des writes
- **Inconsistance** : la DB est en retard par rapport au cache

**Mitigation** : Redis avec AOF (Append-Only File) toutes les secondes reduit la fenetre de perte a 1 seconde.

### 4. Read-Through

Similaire au cache-aside, mais c'est le **cache lui-meme** qui va chercher la donnee en cas de miss.

**Flow :**
```
1. L'app demande la cle K au cache
2. Si HIT -> retourner la valeur
3. Si MISS -> le cache lui-meme interroge la DB, stocke la valeur, et la retourne
```

**Difference avec cache-aside :** L'application ne connait pas la DB. Elle ne parle qu'au cache. Le cache est responsable de la logique de chargement. C'est plus propre architecturalement mais le cache devient un point critique.

**Exemple de produit :** Amazon DynamoDB Accelerator (DAX) est un cache read-through devant DynamoDB. L'application utilise le meme SDK DynamoDB, et DAX intercepte les reads de maniere transparente.

### Tableau comparatif des 4 strategies

| Strategie | Latence read | Latence write | Consistance | Risque de perte | Complexite | Use case typique |
|---|---|---|---|---|---|---|
| **Cache-Aside** | Miss = lent, Hit = rapide | Write DB + invalidate | Stale possible (court) | Aucun | Faible | General purpose, le defaut |
| **Write-Through** | Toujours rapide | Lent (cache + DB sync) | Forte | Aucun | Moyenne | Donnees critiques lues souvent |
| **Write-Behind** | Toujours rapide | Tres rapide | Faible (DB en retard) | Oui (crash cache) | Elevee | Compteurs, analytics, metrics |
| **Read-Through** | Miss = lent, Hit = rapide | N/A (combine avec write-*) | Stale possible (court) | Aucun | Moyenne | Abstraction propre (DAX) |

**Regle d'or en entretien** : commence par cache-aside. C'est le defaut. Ne propose write-through/write-behind que si le cas l'exige (consistance critique ou write-heavy).

---

## Invalidation — Le probleme le plus dur en computer science

> "There are only two hard things in Computer Science: cache invalidation and naming things." — Phil Karlton

L'invalidation est difficile parce qu'elle doit repondre a une question impossible : **quand est-ce que la donnee dans le cache ne reflete plus la realite ?** Il n'y a pas de solution parfaite, seulement des tradeoffs.

### 1. TTL (Time-To-Live)

La methode la plus simple. Chaque entree a une duree de vie fixe.

```python
redis.setex("user:123", 300, data)  # Expire dans 5 minutes
```

**Avantages :** Simple, previsible, auto-nettoyant.

**Inconvenients :** La donnee peut etre stale pendant tout le TTL. Un TTL trop court = taux de miss eleve. Un TTL trop long = donnees obsoletes.

**Comment choisir le TTL :**

| Type de donnee | TTL recommande | Justification |
|---|---|---|
| Config / feature flags | 30s - 5 min | Change rarement, mais l'impact d'un stale est modere |
| Profil utilisateur | 5 - 15 min | Change peu, tolerance au stale |
| Feed / timeline | 1 - 5 min | Change souvent, mais stale acceptable |
| Stock / inventaire | 10 - 30s | Change souvent, stale = sur-vente possible |
| Resultat de recherche | 1 - 60 min | Depend de la frequence de mise a jour de l'index |
| Session utilisateur | 30 min - 24h | Expire quand l'utilisateur est inactif |

### 2. Event-driven invalidation

L'ecriture dans la DB declenche un evenement qui invalide le cache.

```
Write DB -> Emit event (Kafka, Redis Pub/Sub, CDC) -> Cache listener -> DELETE key
```

**Avantages :** Invalidation quasi temps reel. Pas de stale prolonge.

**Inconvenients :** Complexite. Besoin d'un bus d'evenements. Que faire si l'evenement est perdu ? Combinaison habituelle : event-driven + TTL comme filet de securite.

**CDC (Change Data Capture)** : Debezium lit le WAL (Write-Ahead Log) de PostgreSQL et emet un evenement pour chaque changement. Le cache subscriber invalide les cles correspondantes. C'est la solution la plus robuste car elle capture TOUT, meme les ecritures directes en DB.

### 3. Versioning

Au lieu d'invalider, on change la cle du cache.

```python
# Version 1
cache_key = f"user:{user_id}:v1"

# Apres un update, incrementer la version
cache_key = f"user:{user_id}:v2"
# L'ancienne cle expire naturellement via TTL
```

**Avantages :** Pas de race condition. Pas besoin de DELETE explicite. Utile pour les assets statiques (CSS, JS : `app.v3.js`).

**Inconvenients :** Il faut stocker et distribuer le numero de version. Double storage pendant la transition.

### 4. Cache Stampede (Thundering Herd)

**Le probleme** : Une cle populaire expire. 1000 requetes arrivent simultanement, toutes font un cache miss, toutes requetent la DB en parallele. La DB est surchargee.

```
TTL expire pour la cle "homepage_feed"
                      |
    Thread 1 -> cache MISS -> query DB  \
    Thread 2 -> cache MISS -> query DB   |
    Thread 3 -> cache MISS -> query DB   | -> DB surchargee
    ...                                  |
    Thread 1000 -> cache MISS -> query DB/
```

**Solutions :**

#### a) Locking (Mutex)

Un seul thread reconstruit le cache. Les autres attendent.

```python
def get_with_lock(key: str) -> dict:
    value = redis.get(key)
    if value:
        return json.loads(value)

    # Tenter d'acquerir le lock
    lock_acquired = redis.set(f"lock:{key}", "1", nx=True, ex=5)  # NX = seulement si n'existe pas

    if lock_acquired:
        # Ce thread reconstruit le cache
        value = db.query(...)
        redis.setex(key, 300, json.dumps(value))
        redis.delete(f"lock:{key}")
        return value
    else:
        # Un autre thread reconstruit deja -> attendre et retenter
        time.sleep(0.05)
        return get_with_lock(key)  # Retry
```

#### b) Probabilistic Early Expiration (PER)

Chaque thread a une probabilite croissante de recomputer le cache AVANT l'expiration.

```python
def get_with_early_recompute(key: str, ttl: int = 300, beta: float = 1.0):
    value, expiry = redis.get_with_ttl(key)

    if value is None:
        # Cache miss -> recompute
        value = db.query(...)
        redis.setex(key, ttl, value)
        return value

    remaining_ttl = expiry - time.time()

    # Plus le TTL restant est faible, plus la probabilite de recompute augmente
    # Formule : remaining_ttl < beta * log(random())  -> recompute
    if remaining_ttl < beta * (-math.log(random.random())):
        value = db.query(...)
        redis.setex(key, ttl, value)

    return value
```

**Avantage :** Pas de lock, pas d'attente. Un seul thread (statistiquement) recompute avant l'expiration.

#### c) Background refresh

Un worker periodique rafraichit les cles critiques avant leur expiration. Simple mais ne scale pas pour des millions de cles.

---

## Eviction Policies

Quand le cache est plein, il faut choisir quelle entree supprimer pour faire de la place.

| Policy | Algorithme | Quand l'utiliser | Inconvenient |
|---|---|---|---|
| **LRU** (Least Recently Used) | Supprime l'entree la plus anciennement accedee | General purpose, le defaut | Peut evincer une entree frequente mais non accedee recemment |
| **LFU** (Least Frequently Used) | Supprime l'entree la moins souvent accedee | Quand certaines cles sont "toujours chaudes" | Lent a s'adapter aux changements de popularite |
| **FIFO** (First In, First Out) | Supprime la plus ancienne entree inseree | Donnees avec duree de vie naturelle (logs, events) | Ignore la frequence et la recence |
| **TTL** | Supprime les entrees expirees en premier | Quand chaque entree a une duree de vie definie | Ne resout pas le probleme si toutes les TTL sont identiques |
| **Random** | Supprime une entree au hasard | Quand les access patterns sont uniformes | Imprevisible, mais etonnamment efficace en pratique |

### LRU vs LFU — Le choix classique en entretien

**LRU** : Meilleur quand les access patterns changent dans le temps. Un pic de popularite d'un article remplace naturellement les anciennes cles. Implementable avec un doubly-linked list + hashmap en O(1).

**LFU** : Meilleur quand certaines cles sont toujours populaires (page d'accueil, config globale). Mais un item historiquement populaire qui n'est plus utile reste "colle" dans le cache. Solution : LFU avec decroissance temporelle (time-decay LFU).

**Redis utilise un approximated LRU/LFU** : Au lieu de tracker tous les acces, Redis echantillonne N cles aleatoires et evince la pire parmi l'echantillon. `maxmemory-samples 10` donne un bon compromis precision/performance.

---

## Redis en profondeur

Redis n'est pas "juste un cache". C'est un serveur de structures de donnees en memoire. Comprendre ses types, c'est savoir quand l'utiliser au-dela du simple key-value.

### Data structures

| Structure | Commandes cles | Use case | Complexite |
|---|---|---|---|
| **String** | GET, SET, INCR, SETEX | Cache simple, compteurs, sessions | O(1) |
| **Hash** | HGET, HSET, HGETALL | Objets avec champs (profil user) | O(1) par champ, O(n) pour HGETALL |
| **List** | LPUSH, RPUSH, LPOP, LRANGE | Files d'attente, historique recent | O(1) push/pop, O(n) range |
| **Set** | SADD, SREM, SISMEMBER, SINTER | Tags, relations, dedup | O(1) add/check, O(n*m) intersection |
| **Sorted Set** | ZADD, ZRANGE, ZRANGEBYSCORE | Leaderboards, feeds tries par score/date | O(log n) add, O(log n + k) range |
| **Stream** | XADD, XREAD, XREADGROUP | Event streaming, message queue | O(1) add, O(n) read |
| **HyperLogLog** | PFADD, PFCOUNT | Comptage de valeurs uniques approche | O(1), ~12 Ko fixe |
| **Bitmap** | SETBIT, GETBIT, BITCOUNT | Flags utilisateur, daily active users | O(1) par bit |

### Exemples concrets

**Leaderboard temps reel (Sorted Set) :**
```redis
ZADD leaderboard 1500 "player_1"
ZADD leaderboard 2300 "player_2"
ZADD leaderboard 1800 "player_3"

ZREVRANGE leaderboard 0 9 WITHSCORES   # Top 10
ZRANK leaderboard "player_1"            # Position du joueur
```

**Rate limiter (String + INCR) :**
```redis
INCR rate:user:123:minute:202604111430
EXPIRE rate:user:123:minute:202604111430 60
# Si INCR retourne > 100, bloquer la requete
```

**Session store (Hash) :**
```redis
HSET session:abc123 user_id "42" role "admin" last_seen "2026-04-11T14:30:00"
EXPIRE session:abc123 1800  # 30 min TTL
HGET session:abc123 user_id
```

### Persistence

Redis est in-memory mais propose deux mecanismes de persistence :

| Mecanisme | Fonctionnement | Durabilite | Impact performance |
|---|---|---|---|
| **RDB** (snapshot) | Snapshot periodique de la DB entiere (fork + write) | Perte possible depuis le dernier snapshot | Minimal (fork en background) |
| **AOF** (Append-Only File) | Log de chaque operation d'ecriture | Perte max = 1 seconde (avec `appendfsync everysec`) | Leger overhead en ecriture |
| **RDB + AOF** | Les deux combines | Meilleure durabilite, AOF pour recovery, RDB pour les backups | Combine les deux overheads |
| **Aucun** | Purement in-memory | Perte totale au restart | Performance maximale |

**Recommandation production :** AOF avec `appendfsync everysec` + RDB toutes les heures pour les backups. Le restart se fait depuis l'AOF (plus complet).

### Cluster mode

**Redis Sentinel** : Haute disponibilite. Un groupe de sentinels surveille le master et promoit un replica en cas de failure. Automatique. Pas de sharding.

**Redis Cluster** : Sharding + haute disponibilite. Les donnees sont reparties sur 16384 slots hash. Chaque noeud gere un sous-ensemble de slots. Si un master tombe, son replica est promu.

```
                     [Client]
                        |
            +-----------+-----------+
            |           |           |
    [Node A]       [Node B]       [Node C]
    slots 0-5460   slots 5461-10922  slots 10923-16383
       |               |               |
    [Replica A]   [Replica B]     [Replica C]
```

**Limitation cluster :** Les operations multi-cles (MGET, transactions) ne fonctionnent que si toutes les cles sont sur le meme slot. Solution : **hash tags** — `{user:123}:profile` et `{user:123}:sessions` iront sur le meme slot grace au `{}`.

---

## CDN — Content Delivery Network

### Comment ca marche

Un CDN est un reseau de serveurs repartis geographiquement (edge servers) qui cachent le contenu proche des utilisateurs finaux.

```
Utilisateur Paris                         Utilisateur Tokyo
      |                                        |
  [Edge Paris]                           [Edge Tokyo]
      |                                        |
      +---------> [Origin Server NYC] <--------+
```

**Terminologie :**
- **Edge server** : Serveur au plus proche de l'utilisateur, dans un POP (Point of Presence)
- **Origin** : Le serveur d'origine qui detient le contenu "source de verite"
- **POP** : Datacenter du CDN. CloudFront a ~450 POPs dans le monde. Cloudflare ~310.
- **Cache HIT** : Le contenu est dans l'edge, pas besoin de contacter l'origin
- **Cache MISS** : L'edge n'a pas le contenu, il le fetche depuis l'origin

### Pull vs Push CDN

| | Pull CDN | Push CDN |
|---|---|---|
| **Fonctionnement** | L'edge fetche le contenu de l'origin au premier miss | Tu uploades le contenu sur le CDN en amont |
| **Contenu** | Ideal pour le contenu dynamique ou genere a la demande | Ideal pour les assets statiques connus a l'avance |
| **Premiere requete** | Lente (cache miss -> fetch origin) | Rapide (deja en cache) |
| **Invalidation** | TTL ou purge API | Re-push + purge |
| **Exemples** | CloudFront, Cloudflare, Fastly | S3 + CloudFront, Akamai NetStorage |
| **Utilisation** | 90% des cas | Assets lourds (videos, binaires) |

### Cache headers HTTP

Le CDN (et le browser) obeit aux headers HTTP pour savoir quoi cacher et combien de temps.

#### Cache-Control

Le header le plus important. Il controle le comportement du cache a chaque niveau.

| Directive | Effet |
|---|---|
| `public` | N'importe quel cache (CDN, proxy, browser) peut stocker |
| `private` | Seul le browser peut stocker (pas le CDN) |
| `no-cache` | Le cache peut stocker, mais doit revalider avant chaque utilisation |
| `no-store` | Ne rien cacher du tout |
| `max-age=3600` | Valide pendant 3600 secondes |
| `s-maxage=3600` | Comme max-age mais uniquement pour les caches partages (CDN) |
| `stale-while-revalidate=60` | Servir le stale pendant 60s en arriere-plan pendant la revalidation |
| `immutable` | Ne jamais revalider (le contenu ne changera pas) |

**Exemples courants :**

```http
# Asset statique avec hash (app.a3f2b1.js) -> cache agressif
Cache-Control: public, max-age=31536000, immutable

# Page HTML dynamique -> toujours revalider
Cache-Control: no-cache

# API response -> cacheable 5 min par le CDN, 1 min par le browser
Cache-Control: public, s-maxage=300, max-age=60, stale-while-revalidate=30

# Donnees sensibles -> jamais cacher
Cache-Control: private, no-store
```

#### ETag et Last-Modified

Mecanismes de **revalidation conditionnelle**. Le client demande au serveur si le contenu a change.

```
Client                          Server
  |                               |
  | GET /api/user/123             |
  | If-None-Match: "abc123"       |   <- ETag de la derniere reponse
  |------------------------------>|
  |                               |
  | 304 Not Modified              |   <- Le contenu n'a pas change, pas de body
  |<------------------------------|
```

**ETag** : Hash du contenu. Si le contenu change, l'ETag change. Le serveur compare l'ETag et repond 304 si identique.

**Last-Modified** : Date de derniere modification. Moins precis que l'ETag (resolution a la seconde).

**Avantage** : Economise la bande passante. Le serveur repond 304 sans renvoyer le body (quelques octets vs potentiellement des Mo).

### CDN pour API vs Static Assets

| | Static Assets | API Responses |
|---|---|---|
| **Cacheable ?** | Toujours | Depend (GET oui, POST non) |
| **TTL typique** | Long (1 an avec hash) | Court (secondes a minutes) |
| **Invalidation** | Versionning dans le filename | TTL + purge API |
| **Headers** | `immutable, max-age=31536000` | `s-maxage=60, stale-while-revalidate=30` |
| **Hit rate** | > 95% | 30-80% selon le pattern |
| **Exemples** | CSS, JS, images, fonts | Feed social, prix, config |

**CDN pour les API — quand ca vaut le coup :**
- Contenu identique pour tous les users (feed public, catalogue, meteo)
- Forte localite geographique (users concentres dans une region)
- Latence origin > 100ms (origin US, users EU)

**CDN pour les API — quand eviter :**
- Contenu personnalise (profil, panier, recommendations)
- Donnees temps reel (chat, notifications)
- Requetes POST/PUT/DELETE (non cacheable par defaut)

---

## Patterns avances

### 1. Cache Warming

Preremplir le cache avec les donnees les plus demandees avant que le trafic n'arrive.

**Quand :**
- Apres un deploy (le nouveau serveur a un cache vide)
- Avant un pic prevu (Black Friday, lancement produit)
- Au restart d'un cluster Redis

**Comment :**
```python
def warm_cache():
    # Charger les top 10K produits les plus vus
    top_products = db.query("SELECT * FROM products ORDER BY view_count DESC LIMIT 10000")
    for product in top_products:
        redis.setex(f"product:{product.id}", 3600, json.dumps(product))
```

**Risque :** Si le warm charge trop de donnees d'un coup, ca peut saturer la DB ou le reseau. Etaler le warm sur quelques minutes avec un rate limiter.

### 2. Multi-tier Caching

Plusieurs niveaux de cache en cascade.

```
Request -> L1 (in-process, dict local) -> L2 (Redis) -> L3 (DB)
```

**L1 (local, in-process)** : Ultra rapide (~ns), mais limite a une instance. Pas partage entre les pods/serveurs. Taille : quelques Mo.

**L2 (Redis, distribue)** : Plus lent (~ms), mais partage entre toutes les instances. Taille : Go-To.

**Avantage :** Le L1 absorbe les cles "ultra hot" sans meme toucher le reseau. Ideal pour les configs, feature flags, et donnees de reference.

**Inconvenient :** Invalidation complexe. Quand une donnee change, il faut invalider le L1 de CHAQUE instance. Solutions : TTL court sur L1 (10-30s), ou pub/sub pour broadcast l'invalidation.

### 3. Negative Caching

Cacher aussi les resultats "vide" (donnee non trouvee).

```python
def get_user(user_id: str):
    cached = redis.get(f"user:{user_id}")
    if cached == "NULL":     # Negative cache hit
        return None
    if cached:
        return json.loads(cached)

    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if user is None:
        # Cacher l'absence avec un TTL court
        redis.setex(f"user:{user_id}", 60, "NULL")  # 1 min
        return None

    redis.setex(f"user:{user_id}", 300, json.dumps(user))
    return user
```

**Pourquoi c'est important :** Sans negative caching, un attaquant peut envoyer des requetes pour des IDs inexistants. Chaque requete fait un cache miss + une query DB inutile. Avec le negative caching, la 2e requete est absorbe par le cache.

**Use case critique :** Protection contre les attaques de cache penetration (requetes massives pour des cles inexistantes).

### 4. Cache Stampede Prevention — Recap

| Technique | Principe | Complexite | Use case |
|---|---|---|---|
| **Mutex/Lock** | Un seul thread reconstruit | Faible | Cles moderement populaires |
| **PER (Early Expiration)** | Recompute probabiliste avant expiration | Moyenne | Cles tres populaires |
| **Background refresh** | Worker periodique | Faible | Cles critiques connues a l'avance |
| **Stale-while-revalidate** | Servir le vieux pendant le rebuild | Faible | Quand le stale est acceptable |

---

## Chiffres a connaitre pour les entretiens

### Hit rate typiques

| Systeme | Hit rate attendu | Impact |
|---|---|---|
| CDN (assets statiques) | 95-99% | Seul 1-5% du trafic touche l'origin |
| CDN (API cacheable) | 40-80% | Depend des access patterns |
| Redis (application cache) | 80-95% | La DB ne voit que 5-20% des reads |
| Browser cache | 60-90% | Reduction massive du trafic reseau |
| L1 in-process cache | 30-60% | Depend de la taille et de la diversite des cles |

### Latences avec/sans cache

| Scenario | Sans cache | Avec cache | Gain |
|---|---|---|---|
| Profil utilisateur | 15 ms (DB) | 0.5 ms (Redis) | 30x |
| Image produit | 200 ms (origin S3) | 10 ms (CDN edge) | 20x |
| Feed social (50 posts) | 80 ms (DB + joins) | 2 ms (Redis sorted set) | 40x |
| Config globale | 5 ms (DB) | 0.001 ms (L1 in-process) | 5000x |
| Page HTML complete | 300 ms (SSR) | 15 ms (CDN) | 20x |

### Cout memoire Redis

| Donnee | Taille typique par entree | 1M entrees |
|---|---|---|
| Session (string, 200 bytes) | ~300 bytes (overhead Redis) | ~300 Mo |
| Profil user (hash, 10 champs) | ~500 bytes | ~500 Mo |
| Leaderboard entry (sorted set) | ~100 bytes par member | ~100 Mo |
| Compteur (string) | ~80 bytes | ~80 Mo |

**Regle de pouce :** Redis consomme 2-3x la taille brute des donnees a cause de l'overhead des structures internes (pointeurs, metadata). Pour 1 Go de donnees utiles, prevois 2-3 Go de RAM Redis.

---

## Flash Cards — Q&A

### Q1
**Q** : Tu as un endpoint API lu 10K fois/sec qui retourne des donnees identiques pour tous les utilisateurs. Les donnees changent toutes les 5 minutes. Quelle strategie de cache proposes-tu ?

**R** : **Multi-tier : CDN + Redis.** Le CDN (s-maxage=300, stale-while-revalidate=30) absorbe 90%+ du trafic sans toucher le backend. Redis (TTL=300s) sert de fallback pour les cache miss CDN. L'origin ne recoit que quelques requetes/sec au lieu de 10K. Strategie d'invalidation : TTL simple suffit car le contenu change de facon previsible (toutes les 5 min). Pas besoin d'event-driven.

---

### Q2
**Q** : Explique la difference entre `no-cache` et `no-store` dans Cache-Control.

**R** : `no-cache` autorise le stockage mais **oblige la revalidation** avant chaque utilisation (le cache envoie un If-None-Match/If-Modified-Since au serveur). Si le contenu n'a pas change, le serveur repond 304 (pas de body, economie de bande passante). `no-store` **interdit completement le stockage** — ni en memoire, ni sur disque. Chaque requete fait un round-trip complet. Utiliser `no-store` pour les donnees sensibles (tokens, donnees financieres personnelles). Utiliser `no-cache` pour les donnees qui doivent etre fraiches mais ou la revalidation economise de la bande passante.

---

### Q3
**Q** : 1000 requetes arrivent simultanement pour une cle Redis qui vient d'expirer. Quel est le probleme et comment le resoudre ?

**R** : **Cache stampede (thundering herd).** Les 1000 requetes font un cache miss et requetent la DB en parallele, risquant de la surcharger. Solutions par ordre de preference : (1) **Mutex/lock** : un seul thread reconstruit, les autres attendent. Simple, efficace pour la plupart des cas. (2) **PER (Probabilistic Early Recomputation)** : un thread recompute AVANT l'expiration, les autres ne voient jamais le miss. Ideal pour les cles ultra-populaires. (3) **stale-while-revalidate** : servir la vieille valeur pendant le rebuild. Acceptable si le stale est tolere.

---

### Q4
**Q** : Tu utilises cache-aside. Un thread A met a jour un user en DB puis invalide le cache. Entre le write DB et l'invalidation, un thread B lit la vieille valeur de la DB et la met dans le cache. Resultat ?

**R** : **Donnee stale dans le cache indefiniment** (jusqu'au prochain TTL ou invalidation). C'est la race condition classique du cache-aside. Le thread B a lu une valeur ancienne de la DB et l'a ecrite dans le cache APRES l'invalidation du thread A. Le cache contient maintenant la vieille valeur, et personne ne la ré-invalidera. **Mitigation** : (1) TTL obligatoire sur chaque entree (filet de securite). (2) Utiliser write-through si la consistance est critique. (3) Delayed invalidation : invalider le cache une deuxieme fois apres un court delai (double-delete strategy).

---

### Q5
**Q** : Quand utiliserais-tu LFU au lieu de LRU comme eviction policy ?

**R** : **LFU quand certaines cles ont une popularite stable dans le temps.** Exemple : un e-commerce ou la page d'accueil, les top 100 produits, et la config globale sont accedees en permanence. LRU pourrait evincer ces cles si un pic temporaire de trafic sur d'autres cles remplit le cache. LFU protege les cles a haute frequence. **Inconvenient** : LFU est "lent a oublier" — une cle historiquement populaire mais devenue inutile reste en cache. Solution : LFU avec decay (Redis `volatile-lfu` ou `allkeys-lfu` utilise un compteur logarithmique avec decroissance temporelle).

# Jour 5 — Load Balancing & Networking

## Pourquoi le load balancer est le point d'entree de tout systeme scalable

**Exemple d'abord** : Tu as 4 serveurs web derriere `api.monapp.com`. Sans load balancer, le DNS resout vers une seule IP. Ce serveur prend toute la charge, les 3 autres sont inactifs. Pire, si ce serveur tombe, tout le site est down.

Avec un load balancer (LB) devant, les clients tapent `api.monapp.com` qui resout vers l'IP du LB. Le LB repartit les requetes sur les 4 serveurs. Si l'un tombe, le LB detecte via healthcheck et arrete de lui envoyer du trafic. Les utilisateurs ne voient rien. Tu viens de transformer une architecture single-point-of-failure en architecture horizontalement scalable.

**Key takeaway** : Un load balancer n'est pas qu'un "distributeur de requetes". C'est la **brique de fiabilite** numero 1 : il decouple les clients des serveurs, permet le rolling deploy, l'auto-scaling, le blue-green, la gestion de pannes. Tout systeme de production serieux a un LB.

---

## L4 vs L7 — la difference qui change tout

Les LB operent a deux niveaux du modele OSI :

### L4 — Layer 4 (Transport : TCP/UDP)

Le LB regarde uniquement les infos TCP/UDP : IP source, IP destination, ports. Il **ne lit pas le contenu** des paquets. Il forwarde le TCP stream sans le comprendre.

**Avantages :**
- **Tres rapide** : pas de parsing applicatif. Un LB L4 traite des millions de packets/sec.
- **Protocole-agnostique** : marche pour HTTP, SMTP, SSH, base de donnees, n'importe quoi.
- **Moins cher en CPU** : pas de TLS termination, pas de parsing.

**Inconvenients :**
- Pas de routing intelligent (on ne peut pas router `/api/*` vers un pool et `/static/*` vers un autre).
- Pas de manipulation de headers HTTP.
- Pas d'observabilite par URL.

**Exemples** : AWS NLB (Network Load Balancer), HAProxy en mode TCP, LVS Linux.

### L7 — Layer 7 (Application : HTTP, gRPC)

Le LB termine la connexion TCP, parse le HTTP, et fait des decisions basees sur le contenu : URL, headers, cookies, method, body.

**Avantages :**
- **Routing par URL** : `/api/*` -> pool API, `/static/*` -> pool CDN origin.
- **Sticky sessions** : router le meme user vers le meme serveur via cookie.
- **TLS termination** : dechiffre HTTPS, re-chiffre vers le backend ou pas.
- **Retry automatique** sur erreurs 5xx.
- **Rate limiting, auth, manipulation de headers.**
- **Observabilite riche** : logs par URL, par status code.

**Inconvenients :**
- **Plus lent** : parsing HTTP coute du CPU.
- **Stateful** : connexions TCP terminees, donc le LB a besoin de memoire par connexion.
- **HTTP-only** : pas pour SSH ou protocoles binaires custom.

**Exemples** : nginx, Envoy, Traefik, AWS ALB, HAProxy en mode HTTP, Caddy.

### Table recapitulative

| Critere | L4 (TCP) | L7 (HTTP) |
|---|---|---|
| Parsing applicatif | Non | Oui (HTTP) |
| Routing par URL | Non | Oui |
| TLS termination | Optionnelle (SNI) | Oui (standard) |
| Retry automatique | Non | Oui |
| Rate limiting | Limite | Riche |
| Throughput | Tres haut | Haut mais moindre |
| Sticky sessions | Par IP/port | Par cookie / header |
| Exemples | NLB, HAProxy TCP | nginx, Envoy, ALB |

**Regle pratique** : en 2026, la plupart des architectures utilisent **L7** (nginx/Envoy/ALB). L4 est reserve au trafic non-HTTP, aux cas de throughput extreme, ou en amont d'un L7 pour multiplier les couches.

---

## Les algorithmes de load balancing

### 1. Round Robin

Requete i va au serveur `i % N`. Simple et stateless.

```
Backends : [A, B, C]
Req1 -> A, Req2 -> B, Req3 -> C, Req4 -> A, Req5 -> B, ...
```

**Bon** : simple, equitable si les requetes ont toutes la meme duree.
**Mauvais** : si une requete est 100x plus longue qu'une autre, un serveur se retrouve surcharge sans que le LB le sache.

### 2. Weighted Round Robin

Chaque serveur a un poids. Un serveur plus puissant recoit plus de requetes.

```
Backends : [A(poids=3), B(poids=1), C(poids=1)]
Ordre : A, A, A, B, C, A, A, A, B, C, ...
```

**Use case** : canary deploy (10% du trafic sur la nouvelle version), ou serveurs heterogenes (mix de 16-core et 64-core).

### 3. Least Connections

Chaque requete est envoyee au serveur qui a le **moins de connexions actives**.

```
Backends : [A(conn=3), B(conn=7), C(conn=2)]  -> Req -> C (le moins charge)
```

**Bon** : s'adapte aux requetes de durees variables. Si un serveur est lent, il accumule des connexions et le LB l'evite naturellement.
**Mauvais** : necessite de tracker les connexions actives (stateful). Dans un cluster LB, il faut partager l'etat.

### 4. Least Response Time

Variante de "least connections" qui prend en compte la latence moyenne observee. Le LB envoie la requete au serveur le plus rapide.

### 5. IP Hash (Source Hashing)

`hash(client_ip) % N` -> le meme client va toujours au meme serveur.

**Use case** : sticky sessions sans cookie. Le cache applicatif local a un hit rate eleve car chaque user va toujours au meme serveur.
**Probleme** : si un serveur tombe ou est ajoute, `N` change et 100% des clients sont reshuffles. Pas ideal.

### 6. Consistent Hashing (l'algorithme a connaitre absolument)

Le nom du jeu. Au lieu de `hash(client) % N`, on place les serveurs et les clients sur un **anneau** (0 a 2^32). Chaque client va au serveur le plus proche dans le sens horaire.

```
Anneau :
   0 --- A --- client1 --- B --- client2 --- C --- client3 --- 2^32 -> 0
         client3 va a A (prochain serveur horaire)
         client1 va a B
         client2 va a C
```

**Pourquoi c'est genial** : quand un serveur tombe, seuls les clients assignes a ce serveur sont redistribues (environ `1/N`). Les autres continuent a aller au meme serveur. C'est la base de :
- Partitionnement DynamoDB, Cassandra, Riak
- Distribution CDN (Akamai, Cloudflare)
- Sharding Redis Cluster
- Discord's message routing

**Amelioration : virtual nodes**. Chaque serveur est represente par K points sur l'anneau (typiquement K=100-200). Cela ameliore la distribution uniforme et evite qu'un serveur chope une "zone morte" de l'anneau.

### Tableau comparatif

| Algo | Stateful | Equite | Resilience au deploy | Use case |
|---|---|---|---|---|
| Round Robin | Non | Bonne si req homogenes | Bonne | Defaut simple |
| Weighted RR | Non | Adaptable | Bonne | Canary, heterogene |
| Least Conn | Oui | Excellente | Bonne | Req duree variable |
| IP Hash | Non | Moyenne | Mauvaise (reshuffle) | Sticky simple |
| Consistent Hash | Oui (anneau) | Bonne (avec vnodes) | Excellente | Cache, DB sharding |

---

## Reverse proxy vs Forward proxy

Les deux sont des "proxies", mais ils sont a des endroits differents.

### Forward proxy (client-side)

Place entre le **client** et Internet. Le client sait qu'il passe par le proxy.

```
Client --> Forward Proxy --> Internet --> Serveurs
```

**Use cases** : filtrage corporate, VPN, anonymisation (Tor), cache client (squid).

### Reverse proxy (server-side)

Place entre Internet et les **serveurs**. Le client ne sait pas qu'il y a un proxy — il pense parler au serveur.

```
Client --> Internet --> Reverse Proxy --> Serveurs internes
```

**Use cases** : load balancing, TLS termination, caching (Varnish), WAF (Web Application Firewall), API gateway. 99% des "proxies" dont tu entends parler en backend sont des reverse proxies.

---

## DNS-based Load Balancing

Au lieu (ou en plus) d'un LB TCP/HTTP, on peut faire du LB au niveau DNS : le DNS repond avec plusieurs IPs (round robin) ou avec l'IP la plus proche du client (GeoDNS).

### Round Robin DNS

```
dig api.monapp.com
  -> 54.12.34.1  (serveur US)
  -> 54.12.34.2  (serveur US)
  -> 54.12.34.3  (serveur US)
```

Le client prend la premiere IP. Differents clients ont differents ordres donc la charge est repartie.

**Limites :**
- TTL DNS : les caches intermediaires conservent la reponse 30s a 24h. Impossible de retirer un serveur en panne rapidement.
- Pas de healthcheck : le DNS peut retourner une IP morte.
- Pas d'intelligence : pas de "least connections".

### GeoDNS (global load balancing)

Le DNS retourne l'IP du serveur **le plus proche geographiquement** du client. C'est comme ca que fonctionne le multi-region : Cloudflare, Route53, Google Cloud DNS.

```
Client FR --> dig api.app.com --> 54.x.x.x (serveur Paris)
Client US --> dig api.app.com --> 12.y.y.y (serveur Virginia)
```

**Utilise pour** : CDN, multi-region failover, low-latency global apps.

**Combine avec LB regional** : GeoDNS route vers la region, puis un LB L4/L7 route vers les serveurs de cette region.

---

## Rate limiting — proteger les services de l'abus

Le rate limiting controle le debit de requetes qu'un client peut envoyer. C'est une protection contre : DDoS, scrapers, bugs qui bouclent, API clients mal codes.

### 1. Token Bucket

**Image mentale** : un seau contient des "tokens". Chaque requete consomme 1 token. Le seau se remplit a un taux fixe (ex : 10 tokens/sec). Si le seau est vide, la requete est rejetee (429 Too Many Requests).

**Proprietes** :
- Permet les **burst** : si le seau est plein (capacity=100), tu peux envoyer 100 requetes d'un coup.
- Lisse sur la duree : tu ne peux pas depasser le taux moyen.

**Utilise par** : AWS API Gateway, Stripe, GitHub API.

### 2. Leaky Bucket

**Image mentale** : un seau qui fuit a debit constant. Les requetes entrent en haut, sortent en bas a un rythme fixe. Si le seau deborde, les requetes sont rejetees.

**Difference avec token bucket** : le leaky bucket **lisse le trafic** sortant (debit sortant constant). Le token bucket **autorise les bursts**.

**Utilise par** : traffic shaping reseau, Nginx `limit_req`.

### 3. Fixed Window Counter

On compte les requetes dans une fenetre fixe (ex : "60 req/minute de 14:00 a 14:01"). Simple, mais souffre de l'effet "double" : si tu fais 60 req a 14:00:59 et 60 req a 14:01:00, tu as fait 120 req en 2 secondes alors que la limite etait 60/minute.

### 4. Sliding Window Log

On stocke les **timestamps** des N derniers requetes. Pour chaque nouvelle requete, on compte combien sont dans la derniere minute. Precis mais couteux en memoire.

### 5. Sliding Window Counter (le meilleur compromis)

Variante approximative du sliding window : on combine le compteur de la fenetre courante et celui de la precedente avec un poids. Tres precis, peu de memoire. C'est ce qu'utilise Cloudflare.

### Table recapitulative

| Algo | Precision | Memoire | Bursts | Implementation |
|---|---|---|---|---|
| Token Bucket | Bonne | Faible | Oui | Simple |
| Leaky Bucket | Bonne | Faible | Non | Simple |
| Fixed Window | Moyenne (effet double) | Tres faible | Double autorise | Tres simple |
| Sliding Window Log | Parfaite | O(N) | Oui | Complexe |
| Sliding Window Counter | Bonne | Faible | Oui | Moyenne |

---

## Circuit Breaker — arreter de frapper un service mort

**Probleme** : ton service A appelle le service B. B tombe. A continue a envoyer des requetes vers B, elles timeout a 30s chacune. Les threads de A s'accumulent, les connexions saturent, A tombe aussi. **Cascade de pannes.**

**Solution** : un circuit breaker entre A et B. Il a 3 etats.

```
     [CLOSED]  --success-->  [CLOSED]
        |
     (N echecs consecutifs)
        v
     [OPEN]    --timeout-->  [HALF_OPEN]
        ^                         |
        |                    (test request)
     (test fail)                   |
                              (success) --> [CLOSED]
```

### Les 3 etats

- **CLOSED** : tout va bien, les requetes passent normalement.
- **OPEN** : le service a trop echoue. Toutes les requetes sont rejetees **immediatement** (sans meme tenter l'appel) avec une erreur ou une valeur de fallback. Empeche la saturation de A.
- **HALF_OPEN** : apres un timeout (ex : 30s), le breaker autorise **une seule requete test**. Si elle reussit, on repasse CLOSED. Sinon, on retourne OPEN.

**Fallback** : quand OPEN, on peut retourner une valeur degradee (ex : cache stale, valeur par defaut) plutot qu'une erreur brute. C'est le **graceful degradation**.

**Utilise par** : Hystrix (Netflix, le projet qui a popularise le pattern), Resilience4j, Istio, Envoy, Linkerd.

---

## Retry strategies — la ou on se fait mal

**Mauvais retry** : "En cas d'erreur, retry immediatement et indefiniment". Consequence : tu amplifies la panne (retry storm), tu provoques des tsunamis de trafic sur un service deja faible.

### Bon retry : exponential backoff + jitter

```
Attempt 1 : immediat
Attempt 2 : attendre 1s
Attempt 3 : attendre 2s
Attempt 4 : attendre 4s
Attempt 5 : attendre 8s
...
```

**Pourquoi exponentiel** : donne au service en panne le temps de recuperer. Chaque retry est plus espace.

**Pourquoi jitter (randomisation)** : sans jitter, si 1000 clients ont demarre en meme temps, ils retryeront tous en meme temps (1s apres, 2s apres...). -> **thundering herd**. Avec jitter (randomiser de 0% a 100% du delay), les retries sont etales.

**Formule classique** : `delay = min(cap, base * 2^attempt) * random(0.5, 1.5)`.

### Ce qu'il ne faut PAS retry

- **Requetes non-idempotentes** (POST de creation, payment) : risque de doublon. Utiliser des idempotency keys.
- **Erreurs 4xx** (mauvaise requete) : c'est toi qui est cassé, pas le serveur. Retry = meme erreur.
- **Timeouts trop courts** : retry sur timeout 1s alors que le serveur repond en 1.1s = charge x2 gratuitement.

### Budget de retry

Limite le **pourcentage de retries** par rapport au trafic total (ex : max 10% de retries). Empeche qu'en cas de panne, 100% du trafic devienne des retries et achever le service.

---

## Real-world : comment les grands systemes font ca

### Netflix — Hystrix, OSS du circuit breaker
Netflix a popularise le circuit breaker via Hystrix (2012). Ils avaient des cascades de pannes tuant l'app entiere des qu'un micro-service secondaire tombait. Avec Hystrix, ils isolent chaque dependance dans un breaker + thread pool. Projet retiré en 2018 au profit de Resilience4j, mais l'idee est devenue standard.

### Cloudflare — rate limiting sliding window counter
Cloudflare annonce publiquement qu'ils utilisent le sliding window counter algorithm pour leur rate limiting a l'echelle du web (millions de sites). Precision proche du log sans le cout memoire.

### Discord — consistent hashing + vnodes
Discord route les messages de 150M+ users via consistent hashing pour trouver le serveur responsable d'un channel. Chaque noeud a 256 vnodes, ce qui equilibre parfaitement la charge meme avec des channels tres actifs.

### AWS — NLB + ALB en couches
Une architecture AWS typique : Route53 (DNS) -> NLB (L4, TLS passthrough, throughput max) -> ALB (L7, routing par URL) -> ECS/EKS pods. Plusieurs couches, chacune son role.

---

## Flash cards

**Q1** : Quelle est la difference fondamentale entre L4 et L7 LB ?
**R** : L4 regarde seulement les IPs/ports TCP sans lire le contenu -> tres rapide, protocole-agnostique. L7 parse le HTTP (URL, headers) -> routing intelligent mais plus cher en CPU.

**Q2** : Pourquoi le consistent hashing est-il meilleur que `hash(key) % N` pour du sharding ?
**R** : Avec `hash % N`, changer N (ajout/suppression d'un noeud) reshuffle quasiment tous les keys. Avec consistent hashing, seuls ~1/N des keys sont reassignés. C'est critique pour les caches et les DB distribuees.

**Q3** : Pourquoi ajouter du "jitter" dans les retries ?
**R** : Pour eviter le thundering herd. Sans jitter, tous les clients retryent en meme temps et creent des vagues de trafic synchronisees. Avec jitter, les retries sont etales.

**Q4** : A quoi sert l'etat HALF_OPEN du circuit breaker ?
**R** : A tester si le service distant s'est retabli, avec une seule requete, avant de readmettre le trafic normal. Evite de rouvrir brutalement le robinet et de retomber en panne.

**Q5** : Token bucket vs leaky bucket : quelle difference ?
**R** : Token bucket autorise les bursts (tu peux consommer tous les tokens d'un coup). Leaky bucket lisse le trafic sortant a debit constant (pas de bursts). Le choix depend de si tu veux proteger le service aval (leaky) ou juste limiter la moyenne (token).

---

## Key takeaways

1. **L7 par defaut**, L4 pour throughput extreme ou non-HTTP. La plupart des systemes modernes utilisent nginx/Envoy/ALB en L7.
2. **Consistent hashing** est un must-know pour les interviews senior. Tout systeme qui shard (cache, DB, messaging) l'utilise.
3. **Least connections** s'adapte aux requetes de duree variable mieux que round robin. Preferer en presence d'heterogeneite.
4. **GeoDNS + LB regional** est le pattern standard pour le multi-region.
5. **Rate limiting obligatoire sur toute API publique**. Token bucket ou sliding window counter.
6. **Circuit breaker** empeche les cascades de pannes. Chaque appel a un service externe doit etre protege par un breaker.
7. **Exponential backoff + jitter** pour les retries. Sans jitter, tu crees un thundering herd.
8. **En entretien** : parler de healthchecks, de drain mode (retrait gracieux d'un backend), de TLS termination, et de session affinity si c'est pertinent.

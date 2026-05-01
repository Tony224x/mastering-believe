# Exercices Easy — Load Balancing & Networking

---

## Exercice 1 : Choisir l'algorithme de load balancing

### Objectif
Savoir mapper un cas d'usage a l'algorithme de LB optimal.

### Consigne
Pour chaque scenario, indique quel algorithme de LB tu choisis (**round robin**, **weighted RR**, **least connections**, **consistent hashing**, **IP hash**). Justifie en une phrase.

1. Un cluster de 10 serveurs web stateless identiques qui servent une API REST avec des requetes de duree homogene (~50 ms).
2. Un deploy canary : tu veux envoyer 5% du trafic sur ta V2, 95% sur la V1.
3. Un cluster Memcached de 20 noeuds qui doit eviter de reshuffler le cache quand un noeud tombe.
4. Un service d'analytics qui genere des rapports (certains prennent 100 ms, d'autres 30 secondes).
5. Un systeme WebSocket chat ou chaque user doit rester connecte au meme serveur (pour conserver son etat en memoire).
6. Un cluster heterogene : 4 serveurs avec 16 CPU et 2 serveurs avec 64 CPU.

### Criteres de reussite
- [ ] 6/6 choix corrects avec justification
- [ ] Consistent hashing propose pour Memcached (point cle)
- [ ] Least connections propose pour le service analytics (duree variable)
- [ ] Weighted RR propose pour le canary et le cluster heterogene
- [ ] IP hash (ou sticky session) propose pour le WebSocket

---

## Exercice 2 : Conception d'un rate limiter pour une API publique

### Objectif
Concevoir un rate limiter complet pour une API publique, avec les bonnes limites et les bons algorithmes.

### Consigne
Tu lances une API publique de type `api.monapp.com`. Les clients sont :
- **Free tier** : 100 req/min, 1000 req/jour
- **Pro tier** : 1000 req/min, 50K req/jour
- **Enterprise** : illimite, mais protection anti-abus

**Questions :**
1. Quel algorithme de rate limiting choisis-tu pour la limite par minute ? Justifie.
2. Quel(s) algorithme(s) pour la limite par jour ?
3. Quelle est la **cle** sur laquelle tu rate-limites ? API key ? IP ? User ID ?
4. Ou stocker les compteurs ? Dans le LB ? En Redis ? En memoire par pod ?
5. Quels headers HTTP renvoyer au client pour lui communiquer son quota restant ? (standards)
6. Quel status HTTP renvoyer quand la limite est atteinte ?
7. Comment gerer un pic autorise (ex : un client Pro qui fait 2000 req en 1 sec mais 900 les 59 sec suivantes) ?

### Criteres de reussite
- [ ] Token bucket ou sliding window counter choisi pour le per-minute (avec bursts)
- [ ] Fixed window ou sliding window pour le per-day (OK, moins precis acceptable)
- [ ] Rate limit par API key (principal), fallback par IP pour les anonymes
- [ ] Redis (Lua script) propose pour partager l'etat entre N instances de LB/API
- [ ] Headers : `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (ou `RateLimit-*` RFC 9239)
- [ ] 429 Too Many Requests avec `Retry-After`
- [ ] La gestion du burst explique token bucket vs fixed window

---

## Exercice 3 : Debugger une cascade de pannes

### Objectif
Diagnostiquer et corriger une cascade de pannes classique en production.

### Consigne
**Symptomes en prod :**
- A 14:05, le service `recommendations-api` commence a avoir de la latence (p99 passe de 50 ms a 2 s).
- A 14:07, le service `home-page` commence aussi a ralentir, puis les timeouts explosent.
- A 14:10, tout le site est down. Les logs de `home-page` montrent : "connection pool exhausted", "thread pool full".
- A 14:15, les ops restart `recommendations-api` qui reprend, mais `home-page` reste bloque.

**Contexte :** `home-page` appelle `recommendations-api` pour afficher des produits personnalises. L'appel est synchrone, pas de circuit breaker, timeout de 30s. Retry immediate 3 fois en cas d'echec.

**A rendre :**
1. Explique pourquoi `home-page` est tombe alors que seul `recommendations-api` etait degradé.
2. Explique pourquoi `home-page` reste bloque meme apres la reprise de `recommendations-api`.
3. Liste 4 ameliorations concretes a implementer pour eviter que ca se reproduise (chacune cible un aspect different).
4. Quelle valeur de timeout recommandes-tu pour cet appel ? Pourquoi pas 30s ?
5. Que faudrait-il afficher a la place des reco si le circuit breaker est ouvert ?

### Criteres de reussite
- [ ] La cascade est expliquee : thread pool / connection pool epuise par des appels lents
- [ ] Le retry-storm est mentionne (les retries empirent)
- [ ] Au moins 4 ameliorations : circuit breaker, timeout court (< 1s), fallback, limiter les retries, bulkhead, etc.
- [ ] Le timeout recommande est court (200-500 ms) avec justification "mieux vaut echouer vite"
- [ ] Le fallback est propose : cache, liste statique de produits populaires, ou section masquee

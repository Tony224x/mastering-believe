# Exercices Medium — Load Balancing & Networking

---

## Exercice 1 : Concevoir un rate limiter distribue

### Objectif
Passer de "je connais les 5 algos de rate limiting" a "je sais en choisir un, le dimensionner et le rendre coherent sur un cluster".

### Consigne
Tu protèges une API publique. Regle metier : **chaque client (API key) a droit a 1000 requetes/minute**, avec tolerance de petits bursts. Ton API tourne sur **8 instances** derriere un load balancer L7, sans sticky sessions (un client peut taper n'importe quelle instance).

**Questions :**
1. Choisis l'algorithme (token bucket, leaky bucket, fixed window, sliding window log/counter). Justifie au regard de "1000/min + petits bursts".
2. Probleme du compteur local : si chaque instance compte de son cote, un client peut faire 8 x 1000 = 8000 req/min. Comment rendre le compteur coherent sur les 8 instances ?
3. Donne la structure Redis exacte (cle, type, commandes) pour ton algorithme. Quelle est l'operation atomique cle ?
4. Le fixed window souffre de "l'effet double bordure". Explique-le avec un exemple chiffre (limite 1000/min) et dis comment le sliding window counter le corrige.
5. Que renvoie l'API quand la limite est depassee ? Quels headers ajouter pour que le client se comporte bien ?
6. Redis ajoute un round-trip par requete (~1ms). A 50K req/s global, est-ce un probleme ? Propose une optimisation hybride.

### Criteres de reussite
- [ ] Token bucket OU sliding window counter choisi et justifie (bursts toleres + lissage)
- [ ] Le compteur est centralise dans Redis (etat partage) pour eviter le 8x
- [ ] Structure Redis correcte + operation atomique (ex : INCR + EXPIRE, ou script Lua / token bucket atomique)
- [ ] L'effet double bordure est explique avec chiffres (1000 a t=59s + 1000 a t=61s = 2000 en 2s)
- [ ] 429 Too Many Requests + headers (Retry-After, X-RateLimit-Remaining/Reset)
- [ ] Optimisation hybride proposee (compteur local approximatif + sync Redis, ou pre-allocation de tokens par instance)

---

## Exercice 2 : Consistent hashing pour un cache distribue

### Objectif
Maitriser le consistent hashing : pourquoi il bat `hash % N`, le role des virtual nodes, le comportement a l'ajout/retrait d'un noeud.

### Consigne
Tu shards un cache Redis sur **10 noeuds**. La cle est l'`user_id`. Le cache stocke 200M sessions, hit rate cible > 95%.

**Questions :**
1. Avec `hash(user_id) % 10`, que se passe-t-il quand tu ajoutes un 11e noeud ? Calcule (ordre de grandeur) la fraction de cles remappees et l'impact sur le hit rate juste apres.
2. Avec consistent hashing (anneau), quelle fraction de cles est remappee quand tu passes de 10 a 11 noeuds ? Compare a la question 1.
3. Sans virtual nodes, pourquoi la repartition de charge peut-elle etre desequilibree avec seulement 10 noeuds ? Combien de vnodes par noeud recommandes-tu et pourquoi ?
4. Un noeud tombe (panne). Ou vont ses cles ? Le hit rate global chute-t-il et de combien ? Y a-t-il un risque de surcharge sur le noeud voisin ?
5. Quel est le lien entre ce mecanisme et le sharding de DynamoDB/Cassandra ?

### Criteres de reussite
- [ ] `hash % N` : ~N/(N+1) des cles remappees (≈ 90%+) → effondrement temporaire du hit rate
- [ ] Consistent hashing : seulement ~1/(N+1) des cles remappees (≈ 9%)
- [ ] Les virtual nodes (typiquement 100-200/noeud) sont justifies pour lisser la distribution
- [ ] Panne d'un noeud : ses cles (~1/N) vont au prochain noeud horaire ; risque de hot spot mitige par les vnodes
- [ ] Le lien avec le partitionnement DynamoDB/Cassandra est fait

---

## Exercice 3 : Choix d'algorithme de load balancing + healthchecks

### Objectif
Choisir le bon algorithme de LB selon le profil de charge et concevoir des healthchecks/drain corrects.

### Consigne
Pour chacun de ces 3 services derriere un LB, choisis l'algorithme de load balancing (round robin, weighted RR, least connections, IP hash, consistent hash) et explique pourquoi les autres seraient moins bons :

1. **Service de transcoding video** : requetes de durees tres variables (10s a 10min selon la video), serveurs homogenes.
2. **Service stateless de calcul de prix** : requetes courtes (~20ms), homogenes, trafic eleve.
3. **Migration canary** : tu veux router 5% du trafic vers la v2 (nouvelle version) et 95% vers la v1.

**Puis :**
4. Decris un healthcheck correct (endpoint, frequence, seuils de healthy/unhealthy). Pourquoi un simple "TCP connect OK" ne suffit pas ?
5. Tu veux retirer un backend pour maintenance SANS couper les requetes en cours. Decris le "drain mode" / connection draining.
6. Un healthcheck trop agressif (toutes les 1s, 1 echec = down) peut causer un "flapping". Explique le risque et le bon reglage.

### Criteres de reussite
- [ ] Transcoding → least connections (s'adapte aux durees variables), round robin ecarte
- [ ] Calcul de prix → round robin (simple, requetes homogenes), least connections inutile
- [ ] Canary → weighted round robin (poids 95/5)
- [ ] Healthcheck applicatif (ex : GET /healthz qui verifie DB/deps), pas juste TCP connect
- [ ] Drain mode : LB arrete d'envoyer du NOUVEAU trafic mais laisse finir les requetes en cours
- [ ] Le flapping est explique + bon reglage (ex : 3 echecs consecutifs avant unhealthy, hysteresis)

# Exercices Medium — Load Balancing & Networking

---

## Exercice 1 : Consistent hashing — dimensionner et rebalancer

### Objectif
Comprendre quantitativement pourquoi le consistent hashing minimise les deplacements de cles lors d'un changement de topologie.

### Consigne
Tu geres un cluster de cache distribue de **8 noeuds** stockant **80 millions de cles** (reparties uniformement).

1. Avec un sharding **modulo** (`hash(key) % 8`), quelle fraction des cles change de noeud si tu ajoutes un 9e noeud (`% 9`) ? Donne le raisonnement : quelle est la probabilite qu'une cle tombe sur le meme noeud avant et apres ?
2. Avec du **consistent hashing** (sans virtual nodes), combien de cles bougent en moyenne lors de l'ajout du 9e noeud ?
3. Sans virtual nodes, quel probleme de distribution apparait avec seulement 8 noeuds sur l'anneau ? Chiffre l'ecart possible entre le noeud le plus charge et le moins charge.
4. Tu ajoutes **150 virtual nodes par noeud physique**. Explique ce que ca change pour : la variance de charge, le cout de lookup, le rebalancing.
5. Un noeud tombe en panne. Decris ou va son trafic dans les deux schemas (modulo vs consistent hashing + vnodes), et la consequence sur le cache hit rate global.

### Criteres de reussite
- [ ] Le modulo est identifie comme deplacant ~89% des cles (8/9) lors du passage 8 -> 9 noeuds
- [ ] Le consistent hashing deplace ~1/9 des cles (~8.9M) : seules les cles du nouveau segment bougent
- [ ] Le probleme sans vnodes est la distribution non uniforme (un noeud peut recevoir 2-3x la charge moyenne)
- [ ] Les vnodes reduisent la variance et repartissent le failover sur TOUS les noeuds restants
- [ ] L'impact cache : modulo = hit rate s'effondre (~quasi 0 juste apres), consistent hashing = baisse limitee a la part du noeud perdu (~12.5%)

---

## Exercice 2 : Rate limiter distribue multi-instances

### Objectif
Passer d'un rate limiter local a un rate limiter correct sur une flotte de N gateways.

### Consigne
Ton API publique a un quota de **100 req/min par cle API**. Le trafic entre par **10 instances** d'API gateway derriere un load balancer round robin.

1. Chaque gateway applique aujourd'hui un token bucket LOCAL de 100 req/min. Quel est le quota effectif reel d'un client ? Pourquoi ?
2. Premiere correction naive : 10 req/min par gateway (100/10). Quels sont les deux problemes de cette approche (pense au routing round robin et au scaling de la flotte) ?
3. Concois la solution centralisee avec Redis :
   - Quelle structure / commandes Redis pour un sliding window counter ? (INCR + EXPIRE, ou sorted set)
   - Pourquoi le tout doit etre atomique (script Lua) ?
4. Le call Redis ajoute ~1 ms par requete et Redis devient un SPOF. Propose une architecture hybride (cache local + sync Redis) et decris le tradeoff de precision.
5. Quels headers HTTP renvoyer au client (limite, restant, reset) et quel status code en cas de depassement ?

### Criteres de reussite
- [ ] Le quota effectif local est identifie : jusqu'a 1 000 req/min (10 x 100)
- [ ] Les problemes du quota divise : repartition non uniforme et recalcul a chaque scaling
- [ ] La solution Redis est atomique (script Lua ou commande unique), avec fenetre glissante
- [ ] L'approche hybride accepte un depassement borne (ex : +5-10%) en echange de la latence et de la resilience
- [ ] Reponse client : 429 Too Many Requests + headers X-RateLimit-Limit / Remaining / Reset (ou Retry-After)

---

## Exercice 3 : Plan de deploiement multi-region

### Objectif
Concevoir le routage global d'une application servie depuis 3 regions.

### Consigne
Ton SaaS sert **5M d'utilisateurs** : 50% en Europe, 35% en Amerique du Nord, 15% en Asie. Tu deploies dans 3 regions (eu-west, us-east, ap-southeast). SLO : latence API < 150 ms au p95 pour 95% des utilisateurs ; disponibilite 99.95%.

1. Concois la chaine de routage complete : GeoDNS -> ? -> ? Justifie chaque etage (GeoDNS, anycast ou pas, LB regional L7).
2. Le TTL DNS est un parametre cle pour le failover. Quel TTL choisis-tu et quel est le tradeoff (failover rapide vs charge DNS / cache busting) ?
3. Une region entiere (eu-west) tombe. Decris le failover : qui detecte, combien de temps, ou va le trafic europeen, et quelle surcharge ca represente pour us-east (en %) ?
4. Chaque region doit-elle etre dimensionnee pour absorber le failover ? Calcule la capacite cible de us-east si elle doit absorber tout le trafic EU en plus du sien.
5. Les donnees utilisateur sont en PostgreSQL. Propose une strategie de replication compatible avec ce routage (primary par region ? global primary + read replicas ?) et identifie le probleme des writes cross-region.

### Criteres de reussite
- [ ] La chaine contient GeoDNS (latency-based ou geo-based) + healthchecks + LB L7 regional
- [ ] Le TTL propose est court (30-60s) avec le tradeoff explicite
- [ ] Le failover EU -> US est chiffre : us-east passe de 35% a 85% du trafic global (~x2.4)
- [ ] Le dimensionnement N+1 regional est aborde (capacite de reserve ou autoscaling rapide)
- [ ] Le probleme des writes cross-region est identifie (latence 80-150 ms, conflits) avec une reponse (pinning des writes au primary, ou partitionnement des users par region)

# Exercices Hard — Design classiques (entretiens)

---

## Exercice 1 : Design complet d'un Twitter-like a 200M DAU (entretien 45 min)

### Objectif
Mener un design d'entretien senior de bout en bout : framework 6 etapes, chiffres, hybride fanout, sharding, et failure modes du celebrity problem. C'est le design de reference absolu.

### Consigne
On te demande : **"Design Twitter"** — post, follow, home timeline chronologique, avec media.

**Contexte :**
- 200M DAU, 2 tweets/jour/user, 50 ouvertures de home/jour/user
- Tweet = 280 chars (~300 bytes) + 10% avec media (1 Mo moyen)
- Distribution des followers : long tail + 3 mega-comptes a 100M followers
- SLA : home load < 200 ms p99, eventual consistency OK (1-2s)

**Livre :**

1. **Clarify + Estimate** :
   - Liste 5 questions de clarification.
   - Chiffre : tweets/s (moyen + peak), reads/s (moyen + peak), storage tweets/an, storage media/an.

2. **High-level** :
   - Dessine (ASCII) le write path et le read path, avec les composants (LB, API, services, Cassandra, Redis, Kafka, CDN, S3).

3. **Deep dive — fanout hybride** :
   - Explique fanout-on-write vs fanout-on-read avec leurs couts.
   - Decris precisement l'hybride (seuil, qui est en write, qui est en read).
   - Donne le schema Redis du timeline cache (type, key, score, member, max size) et le schema Cassandra des tweets (partition key, clustering).

4. **Sharding & hot partitions** :
   - Comment shardes-tu les tweets ? Les timelines ?
   - Le celebrity problem : pourquoi fanout-on-write casse pour un compte a 100M followers ? Donne le chiffre d'ecritures par tweet.

5. **Bottlenecks** :
   - Read amplification (1 write -> N reads) : comment l'absorbes-tu ?
   - Media a 15 PB/an : ou et comment ?

6. **Extensions** :
   - 3 extensions credibles (search, ranking ML, trending) avec la techno associee.

### Criteres de reussite
- [ ] 5 questions de clarification + estimations chiffrees (~4.6K tweets/s moyen ~15K peak ; ~115K reads/s moyen ~350K peak ; ~44 TB/an tweets ; ~15 PB/an media)
- [ ] Write path et read path dessines avec les bons composants (Cassandra tweets, Redis timeline cache, Kafka fanout, CDN+S3 media)
- [ ] Fanout-on-write (read rapide / write lourd) vs fanout-on-read (write rapide / read lourd) expliques avec leurs couts
- [ ] Hybride decrit : write pour comptes < ~10K followers, read pour mega-comptes, merge a la lecture
- [ ] Schemas donnes : Redis sorted set (key timeline:{user}, score=ts, member=tweet_id, max 800-1000) ; Cassandra PRIMARY KEY ((user_id), created_at DESC)
- [ ] Celebrity problem chiffre : 100M ecritures/tweet -> fanout-on-write impossible -> read pour ces comptes
- [ ] Read amplification absorbee par cache Redis agressif (90%+) ; media sur S3 + CDN ; 3 extensions credibles

---

## Exercice 2 : Post-mortem — la home timeline qui a explose un dimanche soir

### Objectif
Analyser un incident de hot partition / fanout sur un design de timeline, reconstituer la cascade et concevoir les garde-fous.

### Consigne
Voici le rapport d'incident (resume) d'un Twitter-like en prod.

**Contexte** : fanout-on-write PUR (pas d'hybride). Pas de seuil pour les gros comptes. Le timeline cache est un seul cluster Redis sans sharding par defaut. Pas de rate limiting sur le fanout. Les workers de fanout consomment une queue Kafka unique partagee par tous les comptes.

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 20:00 | Un compte a 80M followers (un artiste) poste un tweet pendant un evenement live. |
| 20:00 | Le fanout-on-write declenche 80M ecritures dans les timelines caches Redis pour CE seul tweet. |
| 20:01 | La queue Kafka de fanout se remplit : les 80M ecritures monopolisent les workers, les tweets des users NORMAUX attendent derriere. |
| 20:03 | Le cluster Redis (non shardé sur la hot key) sature en write : latence p99 passe de 5 ms a 800 ms. |
| 20:05 | Les home loads des users normaux ralentissent (lecture du timeline cache lente) -> p99 home > 3 s. |
| 20:08 | L'artiste poste 2 autres tweets (live). 240M ecritures de fanout supplementaires en queue. |
| 20:10 | La queue Kafka a un lag de plusieurs minutes : les nouveaux tweets de TOUT LE MONDE apparaissent avec 5-10 min de retard. |
| 20:15 | Les users se plaignent : "ma timeline est gelée". Les metriques API (5xx, latence API) restent vertes (les requetes repondent, juste lentement et avec des donnees vieilles). |
| 20:30 | Mitigation manuelle : on coupe le fanout-on-write pour ce compte, on bascule ses followers en fanout-on-read. La queue se vide en 20 min. |
| 21:00 | Retour a la normale. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la cascade complete.
   - Pour chaque maillon, le garde-fou manquant. Classe : architecture, process, monitoring.
   - Pourquoi un seul gros compte a degrade l'experience de TOUS les users ?

2. **Le chiffre** :
   - Calcule le nombre total d'ecritures de fanout declenchees par les 3 tweets de l'artiste.
   - Compare au fanout normal d'un user median (200 followers). Combien de tweets medians "equivalent" en charge a 1 tweet de l'artiste ?

3. **Pourquoi les metriques etaient vertes** :
   - Pourquoi l'API renvoie 200 alors que l'experience est cassee ?
   - Quelle metrique AURAIT du alerter ? (lag de queue, p99 Redis, fraicheur de timeline)

4. **L'hybride aurait-il evite ca ?** :
   - Explique precisement comment l'hybride (read pour les mega-comptes) elimine cette cascade.
   - Quel seuil de followers choisirais-tu pour basculer un compte en read ?

5. **Isolation** :
   - Pourquoi un gros compte ne devrait jamais partager la meme queue/les memes workers que les comptes normaux ? (rappelle le bulkhead du J5)

6. **Runbook** :
   - Un runbook de 7 etapes pour "un mega-compte sature le fanout en prod".

### Criteres de reussite
- [ ] Cascade reconstituee : tweet mega-compte -> 80M ecritures fanout -> queue Kafka monopolisee -> Redis hot key saturee -> home loads lents pour TOUS -> lag de fraicheur -> tweets supplementaires aggravent
- [ ] Garde-fous manquants classes : archi (pas d'hybride, pas de sharding hot key, pas de bulkhead/queue dediee), process (pas de seuil gros compte), monitoring (pas d'alerte sur lag de queue / p99 Redis)
- [ ] Le blast radius global est explique : queue + Redis + workers PARTAGES -> le gros compte affame tout le monde
- [ ] Chiffre : 3 tweets * 80M = 240M ecritures ; vs median 200 -> 1 tweet artiste ≈ 400K tweets medians en charge
- [ ] Metriques vertes expliquees : l'API repond 200 (lent + donnees vieilles), l'erreur n'est pas un 5xx ; alerte attendue = lag Kafka + p99 Redis + freshness de timeline
- [ ] L'hybride elimine la cascade (mega-compte en read = 0 ecriture fanout) ; seuil credible (~10K-1M followers selon le cout)
- [ ] Isolation via bulkhead : queue/workers dedies aux gros comptes ; runbook actionnable (couper le fanout du compte, bascule en read, vider la queue, etc.)

# Exercices Hard — Message Queues & Event-Driven

---

## Exercice 1 : Design d'un pipeline d'ingestion temps reel (Uber-like dispatch)

### Objectif
Concevoir un pipeline event-driven complet a tres haut debit, avec ordering, multi-consommateurs et resilience, chiffres a l'appui.

### Consigne
Tu conçois le pipeline de localisation d'une plateforme de VTC (type Uber) :

**Chiffres :**
- 500K chauffeurs actifs en pic, chacun emet sa position toutes les 3 secondes
- Payload : ~500 bytes (lat, lng, heading, speed, driver_id, ts)
- 5 consommateurs independants du flux : matching, ETA, pricing dynamique, fraud detection, data lake
- Le matching exige l'ordre des positions PAR chauffeur (sinon ETA incoherente)
- Retention : 24h pour le matching/ETA, 7 jours pour le data lake
- SLA : une position doit etre disponible pour le matching en < 1s (p99)

**Livre :**

1. **Dimensionnement** :
   - Throughput moyen (events/sec et MB/sec) et pic.
   - Nombre de partitions Kafka (pose ta regle). Cle de partition ?
   - Stockage 24h et 7j, avec replication factor 3.

2. **Ordering & hot partitions** :
   - Comment garantir l'ordre par chauffeur ?
   - Une grande ville (ex : finale de coupe du monde) concentre 80K chauffeurs sur une zone. Risque de hot partition ? Comment l'eviter sans casser l'ordre par chauffeur ?

3. **Multi-retention** :
   - Le data lake veut 7j mais le matching seulement 24h. Comment servir les deux sans payer 7j de retention sur tout ?

4. **Resilience** :
   - Un consumer group (fraud) prend du retard (lag de 2M messages). Impacte-t-il les autres consommateurs ? Pourquoi ?
   - Que se passe-t-il pendant un rebalance quand un broker tombe ? Comment limiter l'impact sur le SLA < 1s ?

5. **Garanties de livraison** :
   - Quelle garantie pour le matching (perte d'une position tolerable ?) vs le data lake (audit) ?
   - Le pricing fait un INSERT en DB par position. Comment le rendre idempotent face a un at-least-once ?

6. **Monitoring** :
   - Les 6 metriques critiques du pipeline et leurs seuils d'alerte.

### Criteres de reussite
- [ ] Throughput coherent (~167K events/s moyen, ~80 MB/s ; pic dimensionne)
- [ ] Partitions justifiees par une regle debit (≥ 16-32, multiple du parallelisme cible)
- [ ] Cle de partition = `driver_id` (ordre par chauffeur garanti)
- [ ] La hot partition geographique est evitee car la cle est `driver_id` (pas la zone) — point cle
- [ ] Multi-retention geree par topics separes OU tiered storage (24h hot + offload S3 pour le data lake)
- [ ] Le lag d'un consumer group n'impacte PAS les autres (offsets independants par groupe)
- [ ] Garanties differenciees : at-most/at-least-once pour le matching, at-least-once + idempotence pour le data lake
- [ ] Idempotence pricing via UPSERT/clé `(driver_id, ts)`
- [ ] 6 metriques incluant consumer lag, partition skew, throughput, p99 end-to-end, under-replicated partitions, DLQ size

---

## Exercice 2 : Post-mortem — La tempete de retries qui a noye le systeme

### Objectif
Analyser un incident event-driven en cascade (retry storm + poison message + DLQ saturee), identifier la chaine causale et concevoir les protections.

### Consigne
Voici le rapport d'incident (resume) :

**Contexte** : Une marketplace utilise Kafka. Un consumer group `order-processor` (12 consommateurs, 12 partitions) consomme le topic `orders` et, pour chaque commande, appelle 3 services aval : `inventory`, `payment`, `email`. La logique de retry est : "en cas d'echec, retry immediatement, sans limite". Pas de DLQ configuree au depart.

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 09:00 | Deploiement d'une nouvelle version de `inventory`. Un bug fait planter le service sur les commandes contenant un produit avec stock negatif (cas non gere). |
| 09:02 | Les premieres commandes "stock negatif" arrivent. `inventory` renvoie 500. Le consumer retry **immediatement** et **en boucle** sur ces messages. |
| 09:05 | Comme il n'y a pas de DLQ, les "poison messages" bloquent l'avancement des offsets sur leurs partitions. Le consumer ne commit plus. |
| 09:08 | Les retries en boucle generent 200K req/s vers `inventory` (vs 3K normal). `inventory` est completement sature, repond 500 a TOUT, meme aux commandes valides. |
| 09:10 | `payment` et `email`, appeles apres `inventory`, ne recoivent plus rien : le lag du consumer group explose (montee a 4M messages). |
| 09:15 | L'equipe ajoute une DLQ en urgence et redeploie le consumer. Mais le consumer redemarre depuis le dernier offset commite (09:02) et **rejoue 4M messages**, re-tapant `inventory`. |
| 09:18 | Rebalance massif (redeploiement) → consommation pausee 40s → le lag monte encore. |
| 09:25 | L'equipe rollback `inventory` a la version stable. Les commandes valides repassent, mais les poison messages tournent toujours en boucle. |
| 09:40 | Mise en place d'un retry limite (5 tentatives) + DLQ. Les poison messages partent en DLQ. Le systeme se draine. |
| 10:30 | Lag resorbe. Bilan : 1h30 de retard sur le traitement des commandes, ~12K emails de confirmation envoyes 2-3 fois (clients confus), SLA casse. |

**Questions :**

1. **Root cause analysis** :
   - Identifie la chaine causale complete (pas un seul point).
   - Pour chaque maillon, le guardrail manquant.
   - Classe les causes : processus, architecture, monitoring.

2. **Le retry infini sans DLQ** :
   - Pourquoi un retry immediat et illimite transforme un bug local en panne globale ?
   - Pourquoi l'absence de DLQ bloque-t-elle l'avancement des offsets (head-of-line blocking) ?
   - Propose la strategie de retry correcte (chiffres : tentatives, backoff, jitter, budget).

3. **Le replay de 4M messages** :
   - Pourquoi redeployer le consumer a aggrave la situation ?
   - Comment aurait-on du drainer/skipper les poison messages sans tout rejouer ?

4. **Idempotence des emails** :
   - Pourquoi 12K emails ont ete envoyes en double ?
   - Concois le mecanisme qui garantit "1 email par commande" malgre un at-least-once et des rejeux.

5. **Architecture corrigee** :
   - Concois un retry tiered (retry rapide → retry lent → DLQ) avec des chiffres.
   - Ou placer un circuit breaker entre le consumer et `inventory` ?
   - Comment isoler un poison message pour qu'il ne bloque pas sa partition entiere ?

6. **Runbook** :
   - Un runbook de 8 etapes pour un "retry storm + consumer lag" en production.

### Criteres de reussite
- [ ] Chaine causale complete : bug inventory → retry infini → saturation inventory → head-of-line blocking (pas de DLQ) → lag explose → redeploy rejoue 4M → rebalance → doublons email
- [ ] Le retry immediat illimite est identifie comme l'amplificateur (retry storm)
- [ ] Le head-of-line blocking par poison message (offsets non commites) est explique
- [ ] Strategie de retry chiffree : max 3-5 tentatives, exponential backoff + jitter, retry budget (ex : max 10% du trafic)
- [ ] Le replay de 4M est evite par seek/skip des offsets poison ou consommation depuis un offset cible, pas un redeploy "from last commit"
- [ ] Idempotence email via cle unique (`order_id`) stockee avant envoi (dedup), pas un simple try/catch
- [ ] Retry tiered concret (ex : 3 retries immediats → topic retry 30s → topic retry 5min → DLQ)
- [ ] Circuit breaker entre consumer et inventory avec seuils (ex : > 50% erreurs sur 10s → open)
- [ ] Le runbook commence par "stopper l'amplification" (pause consumer / open breaker), pas par "redeployer"

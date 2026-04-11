# Jour 4 — Message Queues & Event-Driven Architecture

## Pourquoi decoupler avec des messages ?

**Exemple d'abord** : Tu construis un site e-commerce. Quand un utilisateur passe commande, il faut : (1) debiter la carte, (2) decrementer le stock, (3) envoyer un email de confirmation, (4) envoyer l'ordre a l'entrepot, (5) mettre a jour l'analytics, (6) notifier le programme de fidelite. Si tu fais tout cela dans la requete HTTP `POST /orders`, l'utilisateur attend 8 secondes, et si un des 6 services est en panne, la commande echoue entierement.

Avec une **queue de messages**, tu fais l'inverse : la requete HTTP ne fait qu'une chose — ecrire un evenement `OrderPlaced` dans la queue. Les 6 services consomment cet evenement **de maniere asynchrone**, chacun a son rythme. Si l'entrepot est en panne, le message reste dans la queue et sera traite quand il revient. L'utilisateur recoit sa confirmation HTTP en 50 ms au lieu de 8 secondes.

**Key takeaway** : Une queue transforme un systeme synchrone fragile (tout ou rien, temps de reponse = somme des composants les plus lents) en un systeme asynchrone resilient (chaque composant avance a son rythme, les pannes sont absorbees). C'est l'un des leviers architecturaux les plus puissants du backend moderne.

---

## Pub/Sub vs Point-to-Point : les deux modeles de base

### Point-to-Point (Queue classique)

Un message est consomme par **un seul** consommateur. Plusieurs consommateurs peuvent partager la charge (load balancing), mais chaque message n'est traite qu'une fois au total.

```
Producer --> [M1, M2, M3, M4] --> Consumer A (M1, M3)
                                  Consumer B (M2, M4)
```

**Use case typique** : traitement de taches (envoyer un email, generer un PDF, redimensionner une image). On ne veut pas envoyer 3 fois le meme email.

**Exemples** : RabbitMQ work queue, AWS SQS standard queue, Celery.

### Pub/Sub (Publish/Subscribe)

Un message est distribue a **tous** les abonnes interesses. Chaque abonne recoit sa propre copie.

```
Producer --> Topic "OrderPlaced" --> Subscriber Email Service
                                     Subscriber Analytics Service
                                     Subscriber Warehouse Service
```

**Use case typique** : propagation d'evenements metier. Quand une commande est creee, plusieurs services doivent reagir independamment.

**Exemples** : Kafka topics, Google Pub/Sub, Redis Pub/Sub, AWS SNS.

### Le modele hybride : Kafka

Kafka combine les deux avec la notion de **consumer group** :
- Les messages sont publies dans un **topic** (pub/sub)
- Un **consumer group** se partage les messages d'un topic (point-to-point)
- Plusieurs consumer groups lisent le meme topic independamment (pub/sub entre groupes)

```
Producer --> Topic "Orders" --> Group "email-service" [Consumer1, Consumer2]
                              --> Group "analytics"    [Consumer3]
                              --> Group "warehouse"   [Consumer4, Consumer5]
```

Chaque groupe recoit tous les messages une fois. A l'interieur d'un groupe, les consommateurs se partagent les partitions.

---

## Kafka vs RabbitMQ vs SQS — le comparatif qu'il faut connaitre

| Critere | Kafka | RabbitMQ | AWS SQS |
|---|---|---|---|
| **Modele** | Log distribue append-only | Broker smart (exchanges, routing) | Queue managee simple |
| **Throughput** | Tres eleve (millions msg/s) | Moyen (10K-100K msg/s) | Eleve (scalable infini) |
| **Latence** | 2-10 ms | 1-5 ms | 10-100 ms |
| **Retention** | Longue (jours, semaines) | Courte (jusqu'a consommation) | 14 jours max |
| **Ordering** | Par partition | Par queue | FIFO queue uniquement |
| **Replay** | Oui (lecture depuis un offset) | Non | Non |
| **Routing** | Simple (topic + partition) | Riche (direct, topic, fanout, headers) | Limite (SNS en amont) |
| **Delivery** | At-least-once (exactly-once optionnel) | At-most / at-least / exactly selon config | At-least-once |
| **Ops** | Complexe (zookeeper, brokers) | Moyen (cluster, plugins) | Aucune (managee) |
| **Use case** | Event sourcing, streaming, logs | Taches metier complexes, routing | Queue simple sans ops |

### Quand choisir quoi ?

**Kafka** si :
- Tu veux une source de verite des evenements (event sourcing)
- Tu as besoin de rejouer l'historique
- Tu traites des millions de messages par seconde
- Tu veux que plusieurs systemes consomment le meme flux independamment

**RabbitMQ** si :
- Tu as besoin de routing complexe (exchanges, topic patterns)
- Tu veux des priorites, des TTLs par message, des dead letter queues sophistiquees
- Ton throughput reste modere (< 100K msg/s)

**SQS** si :
- Tu es sur AWS et tu veux zero ops
- Tes besoins sont simples (file de taches)
- Tu acceptes une latence un peu plus elevee

---

## Architecture Kafka — ce qu'il faut comprendre

### Topics, partitions, offsets

```
Topic "orders" (3 partitions)

Partition 0 : [M0, M1, M2, M3, M4]            <- offsets 0 a 4
Partition 1 : [M0, M1, M2, M3, M4, M5, M6]    <- offsets 0 a 6
Partition 2 : [M0, M1, M2]                    <- offsets 0 a 2
```

- Un **topic** est divise en **partitions** (parallelisme et scalabilite).
- Chaque partition est un **log append-only** ordonne.
- Chaque message a un **offset** (position dans la partition).
- L'**ordre est garanti seulement au sein d'une partition**, pas entre partitions.

### Comment choisir la partition ?

- **Par cle** : `hash(key) % num_partitions`. Les messages avec la meme cle vont toujours dans la meme partition. Utile pour garantir l'ordre par entite (ex: tous les evenements d'un user dans la meme partition).
- **Round-robin** : si pas de cle, distribution uniforme.
- **Custom** : partitionner par region, par tenant, etc.

### Consumer groups et rebalancing

Un consumer group est un ensemble de consommateurs qui se partagent les partitions.

```
Topic "orders" (3 partitions)
Group "email-service" (2 consommateurs)

Avant rebalance :
  Consumer A -> Partitions 0, 1
  Consumer B -> Partition 2

Apres ajout d'un consumer C :
  Consumer A -> Partition 0
  Consumer B -> Partition 1
  Consumer C -> Partition 2
```

**Regle** : on ne peut pas avoir plus de consommateurs actifs que de partitions dans un groupe. Si tu as 3 partitions et 5 consommateurs, 2 sont inactifs. Donc on dimensionne toujours le nombre de partitions au max de parallelisme souhaite.

Le **rebalance** se declenche quand un consommateur rejoint ou quitte le groupe. Pendant le rebalance, la consommation est pausee (quelques secondes). C'est un point douloureux a eviter dans les gros clusters.

---

## Garanties de livraison : at-most-once, at-least-once, exactly-once

### At-most-once ("fire and forget")

Le message est livre **zero ou une fois**. Perte possible, duplication impossible.

**Implementation** : le consommateur ack le message AVANT de le traiter.

**Use case** : telemetrie, logs non critiques ou la perte de 0.01% est acceptable.

### At-least-once (le defaut en pratique)

Le message est livre **une ou plusieurs fois**. Perte impossible, duplication possible.

**Implementation** : le consommateur ack le message APRES l'avoir traite. Si le consommateur crash entre traitement et ack, le message sera retraite.

**Consequence** : le consommateur DOIT etre **idempotent** (rejouer le meme message N fois = meme effet qu'une fois). C'est la realite de 95% des systemes en production.

### Exactly-once (le Saint Graal)

Le message est livre **exactement une fois**. C'est techniquement impossible sur un reseau non-fiable dans le cas general (theoreme). En pratique, on l'obtient via :
- **Transactions** cote broker (Kafka transactional producer + isolation_level=read_committed)
- **Deduplication** cote consommateur (stocker les ids deja traites)
- **Idempotence** de l'effet (UPSERT plutot que INSERT)

La verite : "exactly-once processing" n'existe pas vraiment. Ce qui existe, c'est "at-least-once + idempotence = effectively exactly-once". Retiens ca.

---

## Dead Letter Queue (DLQ)

Quand un message ne peut pas etre traite apres N retries (3-5 typiquement), il est envoye dans une **DLQ** pour inspection manuelle ou traitement differe.

```
Queue principale : [M1, M2, M3-POISON, M4, M5]

Apres 3 tentatives sur M3 qui plantent :
Queue principale : [M4, M5]
DLQ              : [M3-POISON]
```

**Pourquoi une DLQ ?** Sans DLQ, un "poison message" (corrompu, bug) bloquerait indefiniment la queue. La DLQ isole les messages defectueux et permet de continuer a traiter le reste.

**Pattern** : monitoring sur la taille de la DLQ. Si > 10, alerte PagerDuty. On analyse les messages, on corrige le bug, on replay la DLQ vers la queue principale.

---

## Event Sourcing — l'architecture qui stocke tout

**Principe** : au lieu de stocker l'etat actuel (CRUD classique), on stocke la **sequence d'evenements** qui ont mene a cet etat. L'etat se reconstruit en rejouant les evenements.

### Exemple : compte bancaire

**CRUD classique :**
```
Table accounts : [id=42, balance=150]
```
Tu as perdu l'historique. Tu ne sais pas comment on est arrive a 150.

**Event sourcing :**
```
Events : [
  AccountOpened(id=42, initial=0),
  Deposited(id=42, amount=100),
  Deposited(id=42, amount=80),
  Withdrawn(id=42, amount=30),
]
```
Balance actuelle = somme des events = 150. Mais tu as TOUT l'historique, auditable, rejouable.

### Avantages
- **Audit trail complet** (obligatoire en finance, sante)
- **Time travel** : etat a n'importe quel moment passe
- **Debug facile** : rejoue les events pour reproduire un bug
- **Projections multiples** : plusieurs vues de lecture (CQRS) construites a partir du meme flux

### Inconvenients
- **Complexite** : beaucoup plus de code a ecrire
- **Schema evolution** : versionner les events est delicat
- **Requetes ad-hoc difficiles** : pas facile de faire "donne-moi tous les comptes avec balance > 1000"
- **Stockage** : l'historique grandit indefiniment (compaction, snapshotting)

**Quand l'utiliser ?** Dans les domaines ou l'historique est valeur metier (finance, sante, legal). Pas pour un simple CRUD de profils utilisateur.

---

## CQRS — Command Query Responsibility Segregation

**Principe** : separer le modele d'ecriture (commands) du modele de lecture (queries). Les deux sont optimises differemment.

```
Write side                    Read side
---------                     ---------
Commands --> Domain --> DB --> Events --> Projections (denormalized views)
(POST)                                    (GET : rapides, pretes a afficher)
```

### Pourquoi ?

- Les ecritures font 5% du trafic mais portent la complexite metier.
- Les lectures font 95% du trafic mais doivent etre rapides.
- Les optimiser separement permet d'avoir : une DB normalisee pour les writes, des vues denormalisees pour les reads.

### Combine avec event sourcing

Les events produits par le write side alimentent les projections du read side via un bus (Kafka). Chaque projection est une vue specialisee (timeline, dashboard, search index).

**Exemple Twitter** : quand tu tweetes, c'est une command. L'event `TweetPosted` alimente plusieurs projections : ton profil, les timelines de tes followers, l'index de recherche Elasticsearch, les analytics.

---

## Saga Pattern — transactions distribuees sans 2PC

**Probleme** : dans une architecture microservices, une "transaction" peut toucher 4 services (order, payment, inventory, shipping). Impossible d'avoir une transaction ACID classique entre eux. Que faire si le paiement reussit mais le stock manque ?

**Solution** : le **saga pattern**. Une saga est une sequence de transactions locales, chacune dans un service, avec des **compensations** en cas d'echec.

### Deux styles

**Orchestrated saga (coordinateur central)**
```
Orchestrator --> Order Service   : CreateOrder   [OK]
             --> Payment Service : ChargeCard    [OK]
             --> Inventory       : ReserveStock  [FAIL!]
             --> Payment Service : RefundCard    [compensation]
             --> Order Service   : CancelOrder   [compensation]
```

**Choreographed saga (via events)**
```
OrderCreated --> PaymentService reagit --> PaymentCharged
PaymentCharged --> InventoryService reagit --> StockReserveFailed
StockReserveFailed --> PaymentService reagit --> PaymentRefunded
PaymentRefunded --> OrderService reagit --> OrderCancelled
```

### Tradeoffs

- **Orchestrated** : logique centrale claire, debug facile, mais couplage fort au coordinateur.
- **Choreographed** : decouplage maximal, mais logique metier eparpillee dans plusieurs services (difficile a suivre).

En pratique, les equipes commencent en choreographed et basculent en orchestrated des que la saga depasse 4-5 etapes.

---

## Alternatives modernes a Kafka (2024-2026)

Kafka reste le standard, mais deux challengers ont gagne du terrain en 2024-2026 et valent le detour pour un **nouveau** deploiement.

### RedPanda

- **Compatible Kafka API** : memes clients, meme protocole, switch possible sans reecrire le code.
- **C++ natif, thread-per-core, pas de JVM** : 10x moins de latence tail (p99 a quelques ms), memoire footprint 6x plus faible.
- **Operationellement plus simple** : un seul binaire, pas de Zookeeper ni KRaft externe a gerer, cluster plus rapide a provisionner.
- **Quand** : nouveau deploiement avec une equipe qui ne veut pas payer la dette operationnelle Kafka. RedPanda Cloud est aussi une option.

### Warpstream

- **Stockage directement sur S3** (ou equivalent blob storage), **zero disk local** sur les brokers.
- Compatible Kafka API.
- **Tres cost-efficient** pour des workloads asymetriques (beaucoup de retention, lectures rares, replay frequent). Cout de stockage divise par 5-10 vs Kafka sur EBS.
- Latence plus elevee (S3 domine) : 100-500 ms, pas pour du realtime sub-100ms.
- **Quand** : pipelines data, event archiving, audit logs, ou la retention longue domine le cout et ou on peut tolerer quelques centaines de ms de latence.

### Le verdict

Kafka reste valide pour les ecosystemes existants et les equipes qui maitrisent deja l'operationnel. Pour un **nouveau** projet 2026, **RedPanda** est souvent le meilleur defaut (latence + ops simple), et **Warpstream** gagne si la retention et le cout de stockage dominent.

---

## Redis : les alternatives qui montent

Redis n'est pas une queue mais est massivement utilise pour du pub/sub leger, des streams, et comme broker Celery / BullMQ. Apres le changement de licence Redis (2024), deux alternatives drop-in ont emerge :

- **Dragonfly** : compatible protocole Redis, **multi-threaded** (vs Redis single-thread), revendique jusqu'a 25x de throughput sur une meme machine, memoire optimisee. Choix serieux pour remplacer Redis sur une instance unique ou la perf domine.
- **Valkey** : **fork open source de Redis** lance apres le changement de licence, soutenu par AWS, Google Cloud, Oracle et la Linux Foundation. Compatibilite 100% avec Redis OSS, evolution de la codebase continue sous licence BSD. Le choix par defaut si tu veux rester sur l'ecosysteme Redis sans les contraintes de la nouvelle licence.

**Regle** : pour un nouveau deploiement 2026, Valkey est le drop-in par defaut (compatibilite totale, licence propre). Dragonfly si tu as besoin de scale vertical serieux sur une seule machine. Redis reste valide si tu es deja en prod et que la licence ne te pose pas de probleme.

---

## Real-world : comment les grands systemes utilisent les queues

### Uber — realtime dispatch
Uber utilise Kafka pour propager les events de localisation des chauffeurs (millions par seconde). Chaque event `LocationUpdate` alimente le matching algorithm, les ETAs, le pricing, l'analytics, la fraud detection — tous consomment le meme topic independamment.

### LinkedIn — origine de Kafka
LinkedIn a invente Kafka en 2011 pour unifier leur pipeline de donnees. Avant, ils avaient N-squared integrations entre services. Apres, tous les services publient dans Kafka et les consumers s'abonnent. Kafka est ne de ce besoin concret.

### Netflix — event sourcing pour les viewings
Netflix stocke chaque event de visionnage dans Kafka : `play`, `pause`, `seek`, `stop`. Cette source de verite alimente : l'historique, les recommandations, l'analytics, le billing (pour les partenaires), le restart-where-you-left.

### Shopify — saga pour checkout
Shopify utilise des sagas pour orchestrer un checkout : valider le panier, calculer les taxes, charger le paiement, reserver le stock, creer l'ordre. Chaque etape peut echouer et compenser les precedentes.

---

## Flash cards

**Q1** : Quelle est la difference entre at-least-once et exactly-once, et laquelle est realiste ?
**R** : At-least-once garantit qu'un message est traite au moins une fois (duplications possibles). Exactly-once est techniquement impossible sur un reseau non-fiable. En pratique on fait "at-least-once + consommateur idempotent" = "effectively exactly-once".

**Q2** : Dans Kafka, combien de consommateurs peuvent traiter un topic de 5 partitions dans un consumer group ?
**R** : Au maximum 5. Un consommateur traite 1+ partitions, une partition est traitee par exactement un consommateur du groupe. Au-dela de 5, les consommateurs supplementaires sont inactifs.

**Q3** : Pourquoi a-t-on besoin d'une Dead Letter Queue ?
**R** : Pour isoler les "poison messages" qui echouent systematiquement et bloqueraient la queue principale. La DLQ permet de continuer a traiter les messages sains pendant qu'on enquete sur les defectueux.

**Q4** : Quand choisir event sourcing plutot que CRUD ?
**R** : Quand l'historique a de la valeur metier (audit, compliance, debug, time travel). Pas pour un CRUD simple — le cout de complexite ne serait pas justifie.

**Q5** : Qu'est-ce qu'une saga et pourquoi l'utiliser ?
**R** : Une sequence de transactions locales dans plusieurs services avec compensations en cas d'echec. On l'utilise parce qu'on ne peut pas avoir de transaction ACID distribuee entre microservices.

---

## Key takeaways

1. **Queue = decouplage temporel** : producer et consumer avancent a leur rythme. Tolere les pannes transitoires.
2. **Kafka = log partitionne**, RabbitMQ = broker smart, SQS = queue managee. Pas le meme outil pour le meme usage.
3. **Ordering garanti seulement dans une partition Kafka.** Pour avoir l'ordre par entite, cle de partition = id de l'entite.
4. **At-least-once + idempotence** est la realite. Exactly-once est un marketing term.
5. **DLQ obligatoire** en production. Sans DLQ, un seul poison message fige tout.
6. **Event sourcing** donne un audit trail complet mais au prix d'une complexite forte. A reserver aux domaines ou l'historique est valeur metier.
7. **Saga** = transactions distribuees sans 2PC. Commencer en choreographed, passer en orchestrated des que ca depasse 4-5 etapes.
8. **En entretien** : propose une queue des que tu vois un traitement long, multiple destinataires, ou un besoin de resilience.

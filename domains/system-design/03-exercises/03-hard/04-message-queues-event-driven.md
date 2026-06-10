# Exercices Hard — Message Queues & Event-Driven

---

## Exercice 1 : Event backbone d'une plateforme de paiement

### Objectif
Concevoir l'epine dorsale evenementielle d'un systeme de paiement avec des garanties fortes, sous contraintes chiffrees.

### Consigne
Tu concois le bus d'evenements central d'une plateforme de paiement (type Stripe simplifie).

**Contraintes chiffrees :**
- 30 000 transactions/sec en pic (Black Friday), 4 000 en moyenne
- AUCUNE transaction perdue, AUCUN double traitement visible (debit en double = incident reglementaire)
- Latence de bout en bout (transaction acceptee -> ledger mis a jour) : < 2 s au p99
- 6 consommateurs : ledger, fraud, notifications, analytics, webhooks clients, archivage
- Retention : replay possible sur 30 jours ; archivage 7 ans (conformite)
- Budget infra bus + stockage : 25 000 $/mois max

**Livre :**
1. **Topologie Kafka** : topics, nombre de partitions (justifie par le throughput : 1 partition ~ 10 Mo/s, event ~ 1 Ko), cle de partitionnement, replication, `acks` et `min.insync.replicas`.
2. **Garanties** : decris precisement la chaine producer -> broker -> consumer qui garantit "pas de perte, pas de double effet" (producer idempotent, transactions Kafka ou outbox, consumers idempotents). Ou places-tu la frontiere exactly-once vs at-least-once + idempotence ?
3. **Ordering** : les events d'un meme compte doivent etre traites dans l'ordre (authorization -> capture -> refund). Comment le garantir avec 30K tx/s ? Quel est l'impact d'un hot account (un marchand = 5% du trafic) ?
4. **Estimation stockage et cout** : volume/jour en pic et en moyenne, stockage 30 jours avec RF=3, strategie d'archivage 7 ans (tiered storage / S3). Montre que tu tiens les 25K$/mois (hypotheses de prix a poser explicitement).
5. **Failure drill** : un broker meurt en plein Black Friday ; puis le consumer ledger accumule 20 min de lag. Decris les comportements attendus, les alertes, et les actions (automatiques et humaines).
6. **3 tradeoffs explicites** ("j'ai choisi X plutot que Y parce que... et j'accepte la consequence Z").

### Criteres de reussite
- [ ] Partitions calculees : 30K events/s x 1 Ko = ~30 Mo/s -> minimum 3-4 partitions par le throughput, mais 50-100 retenues pour le parallelisme des consumers, cle = account_id
- [ ] Config durabilite explicite : RF=3, min.insync.replicas=2, acks=all, producer idempotent (enable.idempotence)
- [ ] La garantie globale est at-least-once + idempotence by design (ledger avec cle unique transaction_id), exactly-once Kafka limite au perimetre Kafka-to-Kafka
- [ ] Le hot account est identifie comme limite du partitionnement par cle (une partition saturee) avec une mitigation (sous-cle, traitement dedie)
- [ ] Stockage chiffre : ~345 Go/jour en moyenne (4K x 1 Ko x 86400), ~2.6 To/jour en pic theorique soutenu, x3 replication, 30 j en ligne + tiered storage S3 pour les 7 ans, avec un calcul de cout pose
- [ ] Le failure drill couvre : election de nouveau leader (pas de perte si ISR>=2), alerte lag consumer avec seuil relie au SLO de 2 s, scaling du consumer ledger
- [ ] 3 tradeoffs explicites avec consequence acceptee

---

## Exercice 2 : Migration d'un monolithe CRUD vers l'event sourcing — ou pas

### Objectif
Decider sous contraintes si l'event sourcing se justifie, et concevoir la migration progressive sans big bang.

### Consigne
Une plateforme de gestion de flotte logistique (10 ans d'age, monolithe + PostgreSQL 4 To) souffre : pas d'audit trail fiable (litiges clients), impossibilite de reconstruire l'etat passe ("quel etait le statut de la livraison X mardi a 14h ?"), et couplage fort (8 modules ecrivent dans les memes tables).

**Contraintes chiffrees :**
- 5 000 writes/sec en pic sur les entites chaudes (livraisons, vehicules)
- L'equipe : 12 devs backend, AUCUNE experience event sourcing
- Migration sans interruption de service (SLA 99.9% maintenu pendant la migration)
- Les requetes de reporting actuelles (200+ requetes SQL complexes) doivent continuer a fonctionner
- Deadline business : audit trail fiable exige par un client majeur dans 6 mois

**Livre :**
1. **Decision argumentee** : event sourcing complet, event sourcing sur un sous-domaine, ou simple audit log (CDC/outbox) ? Croise chaque option avec les 5 contraintes et choisis.
2. **Architecture cible** du perimetre retenu : event store (techno ?), projections/read models, snapshots (a partir de combien d'events par aggregate ?), schema d'un event.
3. **Plan de migration** en phases (strangler pattern) : comment co-existent l'ancien CRUD et le nouveau systeme ? Dans quel sens coule la verite a chaque phase ? Quel est le point de non-retour ?
4. **Le probleme des 200 requetes SQL** : comment les preserver (projections vers les memes tables ? read model SQL dedie ?) sans reecrire le reporting ?
5. **Schema evolution** : dans 2 ans, le format de l'event `DeliveryStatusChanged` doit changer. Quelles regles poses-tu DES MAINTENANT (upcasting, versioning d'events) ?
6. **Chiffrage du risque** : estime l'effort (en dev-mois) de chaque option de la question 1 et mets-le en face de la deadline de 6 mois.

### Criteres de reussite
- [ ] La decision est un compromis defendable : audit trail par CDC/outbox d'abord (repond a la deadline 6 mois), event sourcing eventuellement limite au sous-domaine livraisons — le full event sourcing est rejete (equipe novice + 6 mois + 200 requetes)
- [ ] L'architecture du perimetre retenu est complete : append-only store, projections asynchrones, snapshots (ex : tous les 100-500 events), event avec event_id, aggregate_id, version, timestamp, payload, schema_version
- [ ] Le plan strangler a >= 3 phases avec direction du flux de verite explicite a chaque phase (CRUD source -> double-write/CDC -> event store source)
- [ ] Les requetes SQL sont preservees via des projections qui materialisent les MEMES tables/vues qu'avant
- [ ] Les regles d'evolution : events immuables, jamais de mutation retroactive, schema_version + upcasters a la lecture
- [ ] Le chiffrage distingue clairement les options (ex : CDC = 2-4 dev-mois, ES sous-domaine = 8-12, ES complet = 24+) et la conclusion en decoule

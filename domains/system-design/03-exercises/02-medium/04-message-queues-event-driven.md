# Exercices Medium — Message Queues & Event-Driven

---

## Exercice 1 : Choisir le broker et dimensionner les partitions

### Objectif
Passer du "je connais Kafka/RabbitMQ/SQS" au "je sais lequel choisir et le dimensionner avec des chiffres".

### Consigne
Tu conçois le bus d'evenements d'une plateforme de paiement. Trois flux coexistent :

| Flux | Volume | Contraintes |
|---|---|---|
| **A — Transactions** | 30K events/sec (pic 90K) | Ordre strict par compte, rejouable sur 30 jours pour l'audit, consomme par 4 equipes (ledger, fraude, notif, analytics) |
| **B — Emails transactionnels** | 2K events/sec | Une seule equipe consomme, chaque email envoye 1 fois, throughput modere |
| **C — Webhooks sortants vers les marchands** | 8K events/sec | Retries avec backoff, isolation des marchands lents, DLQ obligatoire |

**Questions :**
1. Pour chacun des 3 flux, choisis le broker (Kafka, RabbitMQ, SQS) et le modele (point-to-point vs pub/sub). Justifie.
2. Pour le flux A : combien de partitions pour absorber le pic ? Pose ta regle de calcul (debit par consommateur). Justifie le choix de la cle de partition.
3. Pour le flux A : si une equipe veut traiter en parallele avec 50 consommateurs dans son consumer group, est-ce possible avec ton nombre de partitions ? Sinon que faire ?
4. Calcule le stockage Kafka du flux A pour 30 jours de retention (payload ~600 bytes), replication factor 3.
5. Pour le flux C : explique pourquoi un marchand lent ("noisy neighbor") peut bloquer les webhooks des autres marchands, et comment l'isoler.

### Criteres de reussite
- [ ] Flux A → Kafka (pub/sub via consumer groups, retention longue, replay audit)
- [ ] Flux B → SQS ou RabbitMQ (point-to-point, throughput modere, zero ops pour SQS)
- [ ] Flux C → RabbitMQ ou SQS avec DLQ + retries (routing/isolation par marchand)
- [ ] Nombre de partitions du flux A justifie par une regle debit (ex : ~10 MB/s ou ~10K msg/s par consommateur → 9-12 partitions au pic)
- [ ] Cle de partition = `account_id` (garantit l'ordre par compte)
- [ ] Le cas "50 consommateurs > N partitions" est identifie (consommateurs inactifs) → augmenter les partitions
- [ ] Stockage 30j coherent (ordre de grandeur quelques dizaines de TB avec RF=3)
- [ ] Le noisy neighbor est isole par queue/partition dediee ou concurrency limit par marchand

---

## Exercice 2 : Saga de checkout — orchestration et compensations

### Objectif
Concevoir une transaction distribuee resiliente avec le saga pattern, et raisonner sur les compensations.

### Consigne
Un checkout e-commerce touche 4 services, chacun avec sa propre DB (pas de transaction ACID globale possible) :

```
1. OrderService    : creer la commande (statut PENDING)
2. PaymentService  : debiter la carte (appel API externe, peut timeout)
3. InventoryService: reserver le stock
4. ShippingService : creer le bon de livraison
```

**Questions :**
1. Ecris la saga (sequence d'etapes) et, pour CHAQUE etape, sa compensation. Que se passe-t-il si l'etape 3 (stock) echoue ?
2. Compare orchestrated saga vs choreographed saga pour ce cas precis : laquelle choisis-tu et pourquoi ?
3. L'etape 2 (paiement) a timeout : la reponse n'est jamais revenue. As-tu debite ou pas ? Comment ton design rend cette etape sure a rejouer (idempotence) ?
4. Une compensation peut elle-meme echouer (ex : le refund API est down). Comment garantis-tu qu'elle finira par s'executer ?
5. Pendant la saga (entre la creation et la confirmation), dans quel etat est la commande pour l'utilisateur ? Quel consistency model assumes-tu ?

### Criteres de reussite
- [ ] Les 4 etapes ont chacune une compensation explicite (CancelOrder, RefundPayment, ReleaseStock, CancelShipping)
- [ ] Echec etape 3 → compensations en ordre inverse (release n/a, refund payment, cancel order)
- [ ] Le choix orchestrated est argumente (logique centrale, 4-5 etapes, debug facile)
- [ ] L'idempotence du paiement repose sur une idempotency key (pas de double debit au retry)
- [ ] Les compensations sont rejouees jusqu'au succes (retry + backoff, DLQ si echec persistant, alerte humaine)
- [ ] La commande est en etat intermediaire (PENDING) → consistency eventuelle assumee cote user

---

## Exercice 3 : Event sourcing + CQRS pour un wallet

### Objectif
Decider quand event sourcing est justifie et concevoir les projections de lecture (CQRS).

### Consigne
Tu construis un wallet (porte-monnaie) pour une fintech. Operations : `Credited`, `Debited`, `Frozen`, `Unfrozen`. Contraintes metier : audit complet exige par le regulateur, capacite a reconstruire le solde a n'importe quelle date passee, et un dashboard de lecture rapide ("solde courant" et "historique des 50 dernieres operations").

**Questions :**
1. Justifie pourquoi event sourcing est pertinent ici (vs un simple CRUD `balance` mis a jour).
2. Donne le flux d'ecriture : que stocke-t-on exactement ? Comment calcule-t-on le solde courant ?
3. Le solde se calcule en rejouant TOUS les events. Pour un compte avec 2M operations, c'est lent. Quelle technique evite de tout rejouer ?
4. Decris 2 projections CQRS distinctes alimentees par le meme flux d'events, et la DB adaptee a chacune.
5. Un bug a produit des events `Debited` errones la semaine derniere. Tu ne peux pas modifier l'historique (immutable). Comment corriges-tu le solde proprement ?

### Criteres de reussite
- [ ] Event sourcing justifie par l'audit reglementaire + time travel (pas pour le confort)
- [ ] Le flux d'ecriture append des events immutables ; le solde = fold (somme) des events
- [ ] Le snapshotting est propose (solde a T + rejouer seulement les events depuis T)
- [ ] 2 projections distinctes (ex : solde courant en Redis/KV ; historique en table SQL indexee par date)
- [ ] La correction se fait par events compensatoires (ex : `Credited` correctif ou `Reversed`), jamais par edition de l'historique

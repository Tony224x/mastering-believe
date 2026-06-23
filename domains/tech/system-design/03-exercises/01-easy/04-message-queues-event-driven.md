# Exercices Easy — Message Queues & Event-Driven

---

## Exercice 1 : Queue ou pas queue ?

### Objectif
Identifier rapidement quand un systeme a besoin d'une queue de messages et quand c'est inutile (voire nuisible).

### Consigne
Pour chaque scenario, indique si une queue est appropriee. Si oui, propose **point-to-point** ou **pub/sub** et une technologie (Kafka, RabbitMQ, SQS, Redis). Justifie en une ou deux phrases.

1. Un endpoint `POST /signup` qui doit envoyer un email de bienvenue (qui peut prendre 2-5s via SMTP).
2. Un endpoint `GET /users/:id` qui retourne les infos d'un utilisateur depuis PostgreSQL.
3. Un systeme IoT qui ingere 500K events/sec de capteurs et doit les exposer a 5 equipes differentes (ML, monitoring, billing, alerting, data lake).
4. Un ordre de bourse (trading) qui doit etre execute en < 10 microsecondes.
5. Un worker qui redimensionne des images uploadees par les utilisateurs (5-30s par image).
6. Un endpoint `GET /search` qui doit retourner des resultats en < 100 ms.

### Criteres de reussite
- [ ] 6/6 decisions justifiees
- [ ] Au moins un cas identifie comme "pas de queue" (queries synchrones)
- [ ] Kafka propose pour le cas IoT multi-consommateurs (pub/sub + throughput eleve)
- [ ] SQS ou RabbitMQ propose pour les taches background simples
- [ ] Le cas trading rejete (latence queue > latence requise)

---

## Exercice 2 : Idempotence d'un consommateur

### Objectif
Concevoir un consommateur at-least-once qui reste correct meme si un message est traite plusieurs fois.

### Consigne
Tu developpes un worker qui consomme les events `PaymentRequested` d'un topic Kafka et doit debiter le compte de l'utilisateur. Le payload est :

```json
{
  "payment_id": "pay_abc123",
  "user_id": "user_42",
  "amount_cents": 2500,
  "timestamp": "2026-04-11T10:00:00Z"
}
```

Kafka est at-least-once : ton worker PEUT recevoir le meme message 2 fois (ex : crash avant commit de l'offset, consumer replace, etc.).

**A rendre :**
1. Le pseudo-code (Python/SQL) du worker, avec la strategie d'idempotence claire.
2. Le schema de la table `payments` qui supporte l'idempotence.
3. Que se passe-t-il si le message est recu 3 fois ? Decris chaque etape.
4. Pourquoi une simple verification "SELECT ... WHERE payment_id" avant INSERT ne suffit pas (race condition) ?
5. Quelle primitive SQL / Postgres resout le probleme proprement ?

### Criteres de reussite
- [ ] L'idempotence est basee sur une cle unique (`payment_id`)
- [ ] La table a une contrainte UNIQUE sur `payment_id`
- [ ] Le code utilise `INSERT ... ON CONFLICT DO NOTHING` (ou equivalent)
- [ ] La race condition de "check then insert" est expliquee
- [ ] La consequence des 3 receptions est decrite (1 insert, 2 no-op)

---

## Exercice 3 : Dimensionnement Kafka pour une appli

### Objectif
Estimer le nombre de partitions, le throughput, et le stockage pour un deploiement Kafka realiste.

### Consigne
Tu deploies Kafka pour une application de ride-sharing (type Uber) qui ingere :

**Donnees :**
- 200K chauffeurs actifs simultanement
- Chaque chauffeur envoie sa position toutes les 4 secondes
- Payload par event : ~400 bytes (JSON : lat, lng, heading, speed, driver_id, timestamp)
- 3 consumer groups : matching algorithm, fraud detection, analytics data lake
- Retention Kafka : 7 jours
- Peak = 3x la moyenne
- Replication factor = 3 (standard Kafka)

**Questions :**
1. Calcule le throughput moyen en events/sec et en MB/sec.
2. Calcule le throughput peak (3x).
3. Combien de partitions recommandes-tu pour ce topic ? Justifie avec une regle (ex : un consumer peut traiter X MB/sec).
4. Calcule le stockage brut necessaire sur 7 jours (avant replication).
5. Calcule le stockage total sur le cluster avec replication factor 3.
6. Combien de brokers Kafka suggerer si chaque broker a 2 TB de disque ?

### Criteres de reussite
- [ ] Throughput moyen correct (~50K events/sec, ~20 MB/sec)
- [ ] Throughput peak correct (~150K events/sec, ~60 MB/sec)
- [ ] Nombre de partitions justifie (typiquement 30-60, basee sur 10 MB/s par consumer)
- [ ] Stockage brut 7j correct (~12 TB)
- [ ] Stockage replique correct (~36 TB)
- [ ] Nombre de brokers suggere coherent (minimum 6-8 avec marge)

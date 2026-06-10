# Exercices Medium — Message Queues & Event-Driven

---

## Exercice 1 : Saga pour une commande e-commerce

### Objectif
Concevoir une saga (choreographed puis orchestrated) pour une transaction distribuee multi-services, avec les compensations.

### Consigne
Le flux de commande d'un e-commerce traverse 4 services, chacun avec sa propre base de donnees :

1. **Order Service** : cree la commande
2. **Inventory Service** : reserve le stock
3. **Payment Service** : debite la carte
4. **Shipping Service** : cree l'etiquette d'expedition

Pas de transaction distribuee (pas de 2PC). Chaque etape peut echouer.

**A rendre :**
1. Dessine la saga **choreographed** (chaque service ecoute les events des autres) : liste les events publies/consommes par chaque service, dans l'ordre nominal.
2. Pour chaque etape, definis la **transaction de compensation** (que faire si l'etape suivante echoue ?).
3. Deroule le scenario d'echec : le paiement est refuse APRES la reservation du stock. Decris event par event ce qui se passe.
4. Refais le meme design en **orchestrated** (un orchestrateur central pilote). Dessine le diagramme d'etats de l'orchestrateur (PENDING, STOCK_RESERVED, PAID, SHIPPED, COMPENSATING, FAILED, COMPLETED).
5. Compare les deux approches : a partir de quel critere bascules-tu en orchestrated ?

### Criteres de reussite
- [ ] La version choreographed liste au moins 6 events nommes (order.created, stock.reserved, payment.failed, etc.)
- [ ] Chaque etape a une compensation explicite (liberer le stock, rembourser, annuler la commande)
- [ ] Le scenario d'echec montre la compensation en cascade (payment.failed -> stock.released -> order.cancelled)
- [ ] Le diagramme d'etats orchestrated contient au moins 6 etats dont un etat COMPENSATING
- [ ] Le critere de bascule mentionne le nombre d'etapes (4-5+) et/ou la difficulte de debug du choreographed

---

## Exercice 2 : Backpressure et consumer lag

### Objectif
Diagnostiquer et resoudre un probleme de consumer lag croissant sur Kafka.

### Consigne
Ton topic Kafka `orders` recoit **20 000 events/sec** en moyenne (40 000 en pic, 2h par jour). Il a **24 partitions**. Ton consumer group `enrichment` a 8 consumers ; chaque consumer traite un event en **2 ms** (appel DB inclus), sequentiellement.

**Questions :**
1. Calcule le throughput maximum du consumer group actuel (events/sec). Tient-il la charge moyenne ? Le pic ?
2. Pendant le pic de 2h, calcule le lag accumule (en events) a la fin du pic.
3. Combien de temps faut-il pour resorber ce lag une fois le trafic revenu a la moyenne ?
4. Propose 3 solutions pour tenir le pic, et pour chacune indique sa limite :
   - scaling horizontal des consumers (jusqu'a combien ? pourquoi ?)
   - batching des appels DB
   - traitement asynchrone/parallele dans chaque consumer (quel risque pour l'ordering ?)
5. La latence end-to-end max acceptable est de 30 secondes. Quelle alerte mets-tu en place (metrique + seuil) ?

### Criteres de reussite
- [ ] Throughput max actuel calcule : 8 consumers * 500 events/sec = 4 000 events/sec (insuffisant meme en moyenne)
- [ ] Le plafond de scaling identifie : 24 consumers max (1 par partition)
- [ ] Le lag du pic est calcule avec la difference production - consommation
- [ ] Le risque d'ordering du traitement parallele intra-consumer est mentionne (ordre par cle perdu)
- [ ] L'alerte est basee sur le consumer lag (events ou secondes) avec un seuil justifie par le SLA de 30s

---

## Exercice 3 : Outbox pattern — publier sans perdre d'events

### Objectif
Resoudre le probleme du dual-write (ecrire en DB ET publier dans Kafka de maniere atomique).

### Consigne
Ton Order Service fait aujourd'hui ceci :

```python
def create_order(order):
    db.insert("orders", order)          # write 1
    kafka.publish("order.created", order)  # write 2
```

1. Decris 2 scenarios de panne ou ce code laisse le systeme incoherent (DB sans event, ou event sans DB).
2. Pourquoi inverser l'ordre des deux writes ne resout rien ?
3. Concois la solution avec le **transactional outbox pattern** :
   - Schema SQL de la table `outbox`
   - Pseudo-code du write (commande + outbox dans la meme transaction)
   - Pseudo-code du relay (polling ou CDC) qui publie vers Kafka
4. Le relay peut publier deux fois le meme event (crash entre publish et marquage). Quelle garantie de livraison obtient-on ? Comment le consommateur doit-il se proteger ?
5. Compare polling relay vs CDC (Debezium) : latence, charge DB, complexite operationnelle.

### Criteres de reussite
- [ ] Les 2 scenarios d'incoherence sont decrits (crash entre write 1 et write 2, et l'inverse apres inversion)
- [ ] La table outbox contient au minimum : id, aggregate_id, event_type, payload, created_at, published_at (ou status)
- [ ] L'insert metier et l'insert outbox sont dans la MEME transaction SQL
- [ ] La garantie identifiee est at-least-once, avec consommateur idempotent en face
- [ ] La comparaison polling vs CDC mentionne la latence (secondes vs quasi temps reel) et la charge du polling

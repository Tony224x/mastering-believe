# Exercices Hard — Principes fondamentaux

---

## Exercice 1 : Design from Scratch — Compteur de vues en temps reel

### Objectif
Appliquer tous les principes fondamentaux (estimation, CAP, consistency, SLA) dans un mini-design complet.

### Consigne
Tu dois concevoir un **systeme de comptage de vues en temps reel** pour une plateforme video (type YouTube).

**Requirements fonctionnels :**
- Chaque lecture de video incremente un compteur
- Le compteur est affiche sur la page de la video
- Un dashboard analytics montre les vues par heure/jour/mois
- Les createurs voient leurs stats en "quasi temps reel" (< 30 secondes de delai)

**Requirements non-fonctionnels :**
- 1 milliard de vues par jour
- Le compteur affiche n'a pas besoin d'etre exact a l'unite pres (tolerance de 1-2%)
- Le dashboard analytics doit etre exact a terme (eventual consistency OK)
- Disponibilite > 99.9%
- Latence d'ecriture (incrementer) : < 10ms au p99

**Livre :**

1. **Estimation back-of-the-envelope** :
   - QPS moyen et pic
   - Stockage pour 5 ans (combien de videos, combien de compteurs)
   - Bande passante

2. **Choix d'architecture** :
   - Quelle DB pour les compteurs temps reel ? Pourquoi ?
   - Quelle DB pour les analytics (aggregations) ? Pourquoi ?
   - Quel consistency model pour chaque composant ?
   - Comment gerer le pic de trafic (video virale = 10x le trafic normal) ?

3. **Schema de donnees** :
   - Structure des compteurs
   - Structure des aggregations analytics

4. **Diagramme d'architecture** (ASCII ou description textuelle) :
   - Flux d'ecriture (vue -> compteur)
   - Flux de lecture (page video -> compteur affiche)
   - Flux analytics (compteurs -> aggregation -> dashboard)

5. **Tradeoffs explicites** :
   - Au moins 3 decisions avec "j'ai choisi X au lieu de Y parce que..."
   - Pour chaque tradeoff, la consequence negative acceptee

6. **SLO et monitoring** :
   - Definir 3 SLIs et leurs SLOs
   - Que monitorer en priorite ?

### Criteres de reussite
- [ ] Estimation complete et coherente (QPS, stockage, bande passante)
- [ ] Architecture en couches avec separation read/write path
- [ ] Consistency model justifie pour chaque composant
- [ ] Au moins une technique de scaling identifiee (sharding, buffering, batching)
- [ ] Gestion du cas "video virale" (burst de trafic)
- [ ] 3+ tradeoffs explicites et argumentes
- [ ] SLIs/SLOs definis et realistes

---

## Exercice 2 : Failure Analysis — Cascading Failures

### Objectif
Comprendre comment les principes fondamentaux interagissent lors d'une panne et concevoir la resilience.

### Consigne
Voici l'architecture d'un systeme e-commerce :

```
                    Load Balancer
                    /     |     \
               API-1   API-2   API-3
                |         |       |
         +------+----+----+------+-------+
         |           |           |       |
    User Service  Product Svc  Order Svc  Payment Svc
    (PostgreSQL)  (MongoDB)   (PostgreSQL)  (External API)
         |           |           |
       Redis       Redis      Kafka
      (cache)     (cache)    (events)
```

**Scenario** : Le service Payment (API externe) commence a repondre en 30 secondes au lieu de 200ms (degradation, pas une panne complete).

**Analyse :**

1. **Propagation de la panne** :
   - Decris etape par etape comment cette latence se propage dans le systeme
   - Quels services sont impactes et dans quel ordre ?
   - A quel moment le systeme entier tombe ?

2. **Calcul d'impact** :
   - Si chaque API server a un pool de 200 threads, et que le Payment Service repond en 30s, combien de commandes/minute peuvent etre traitees avant thread exhaustion ?
   - Combien de temps avant que les API servers soient satures ?

3. **Mecanismes de protection** (pour chacun, explique COMMENT il fonctionne et POURQUOI il aide) :
   - Circuit breaker
   - Timeout + retry avec backoff exponentiel
   - Bulkhead pattern (isolation des thread pools)
   - Queue de decouplage (Kafka)
   - Fallback / degraded mode

4. **Redesign resilient** :
   - Redessine le flux de commande pour qu'une panne du Payment Service n'impacte PAS les autres services
   - Quel consistency model choisis-tu pour la commande entre la creation et la confirmation de paiement ?
   - Comment informer l'utilisateur ?

5. **SLA compose** :
   - Si chaque service interne a un SLA de 99.95% et le Payment Service externe a un SLA de 99.5%, quel est le SLA maximum du flux de commande ?
   - Propose une architecture qui permet d'avoir un SLA de commande > 99.9% malgre le Payment Service a 99.5%

### Criteres de reussite
- [ ] La cascade est decrite etape par etape avec les mecanismes precis (thread exhaustion, connection pool, timeout)
- [ ] Le calcul de thread exhaustion est correct et montre le temps avant saturation
- [ ] Les 5 mecanismes de protection sont expliques avec le "comment" et le "pourquoi"
- [ ] Le redesign decouple le paiement du reste du flux (event-driven ou saga)
- [ ] Le SLA compose est calcule correctement
- [ ] La solution pour depasser 99.9% est realiste (queue + async processing + retry)

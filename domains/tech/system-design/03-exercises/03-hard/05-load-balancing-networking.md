# Exercices Hard — Load Balancing & Networking

---

## Exercice 1 : Architecture multi-region resiliente (global load balancing)

### Objectif
Concevoir le routage et la resilience d'un service mondial : GeoDNS, LB regionaux, failover, et les pieges du multi-region.

### Consigne
Tu conçois la couche reseau d'une application mondiale (type service SaaS B2B) :

**Chiffres :**
- 3 regions : US-East, EU-West, APAC
- 2M requetes/sec global en pic
- Repartition : 45% EU, 35% US, 20% APAC
- SLA : p99 < 150ms intra-region, disponibilite 99.99% (52 min de downtime/an max)
- Une region peut tomber entierement (panne datacenter, catastrophe)

**Livre :**

1. **Routage global** :
   - Comment router un user vers la region la plus proche ? (GeoDNS, anycast, autre ?)
   - Quel est le probleme du TTL DNS pour retirer rapidement une region morte ?
   - Combien de couches de LB (global + regional) et le role de chacune ?

2. **Dimensionnement par region** :
   - Si un serveur tient 20K req/s, combien de serveurs par region au pic ?
   - Tu dois pouvoir absorber la perte d'une region : chaque region doit pouvoir prendre le trafic d'une autre. Recalcule le sur-provisionnement necessaire.

3. **Failover** :
   - US-East tombe a 14h00. Decris la sequence de bascule. Combien de temps avant que les users US soient servis par une autre region ?
   - Quel est l'impact latence pour un user US re-route vers EU-West (~80ms transatlantique) ?
   - Comment eviter le "thundering herd" quand 700K req/s d'un coup basculent sur les regions survivantes ?

4. **Etat et consistance** :
   - Si tes sessions sont en cache regional, que se passe-t-il pour un user re-route vers une autre region ? Propose une mitigation.
   - Les requetes d'ecriture multi-region : strong vs eventual consistency ? Quel tradeoff ?

5. **Calcul de SLA** :
   - Si chaque region a un SLA de 99.95%, quel est le SLA d'un deploiement 3-region avec failover (au moins 1 region up) ? Compare au SLA d'une seule region.
   - Justifie pourquoi le multi-region ameliore la dispo mais complique tout le reste.

6. **Monitoring** :
   - Les 6 metriques a surveiller au niveau global + le seuil qui declenche un failover automatique.

### Criteres de reussite
- [ ] GeoDNS (ou anycast) pour router vers la region proche + 2 couches (global routing + LB regional L7)
- [ ] Le TTL DNS est identifie comme le frein au retrait rapide (caches intermediaires) → TTL court + healthcheck failover, ou anycast pour s'en affranchir
- [ ] Dimensionnement de base correct (~100 serveurs EU au pic) ET sur-provisionnement pour absorber la perte d'1 region (~+50% de capacite par region survivante)
- [ ] Le thundering herd au failover est gere (rate limiting, autoscaling pre-warm, admission control)
- [ ] La perte de session regionale est traitee (session globale repliquee, re-auth, ou sessions stateless/JWT)
- [ ] Le SLA 3-region avec failover est calcule (1 - (1-0.9995)^3 ≈ 99.99999%) et compare a 99.95% mono-region
- [ ] 6 metriques + seuil de failover (ex : healthcheck regional en echec > 30s OU p99 > seuil)

---

## Exercice 2 : Post-mortem — La cascade de pannes par retry storm

### Objectif
Analyser un incident reseau ou un retry mal concu + l'absence de circuit breaker ont provoque une cascade, calculer la saturation des thread pools, et concevoir les protections.

### Consigne
Voici le rapport d'incident (resume) :

**Contexte** : Une plateforme de reservation. Architecture : LB L7 → 30 API servers (pool de 200 threads chacun) → `booking-service` → `payment-gateway` (API externe). Le client retry automatiquement 3x sur erreur, sans backoff. Pas de circuit breaker. Timeout des appels payment = 30s.

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 20:00 | Heure de pointe (concert populaire en vente). Trafic nominal : 6K req/s, latence payment 200ms. |
| 20:05 | Le `payment-gateway` externe commence a se degrader : latence monte a 30s (au timeout), il ne tombe pas completement (40% des appels reussissent encore). |
| 20:06 | Les appels payment occupent les threads des API servers pendant 30s chacun (au lieu de 200ms). |
| 20:07 | Les threads s'accumulent. Les pools de 200 threads se remplissent de requetes bloquees sur payment. |
| 20:08 | Pools satures → les API servers ne peuvent plus accepter de nouvelles requetes, MEME celles qui ne touchent pas payment (ex : consulter une reservation). |
| 20:09 | Le client recoit des timeouts → retry 3x sans backoff → le trafic effectif triple (18K req/s) → aggravation. |
| 20:10 | Le LB voit les API servers comme unhealthy (healthcheck timeout) et les sort de la rotation un par un → les survivants recoivent encore plus de charge → ils tombent aussi. |
| 20:12 | Effondrement total. 100% d'erreurs. |
| 20:20 | L'equipe identifie le payment-gateway comme cause initiale. |
| 20:25 | Mise en place d'un timeout agressif (2s) sur payment + desactivation du retry client. |
| 20:35 | Ajout d'un circuit breaker manuel (feature flag) qui coupe les appels payment et passe en mode degrade. |
| 20:50 | Systeme retabli en mode degrade (reservations sans paiement immediat, paiement async). 50 min de downtime. |

**Questions :**

1. **Root cause analysis** :
   - La cause initiale (payment lent) n'a pas tue le systeme seule. Identifie la chaine d'amplification complete.
   - Pour chaque maillon, le guardrail manquant. Classe : processus, architecture, monitoring.

2. **Calcul de saturation** :
   - Avec 200 threads par server et un appel payment de 30s, combien de req/s touchant payment un seul server peut-il traiter avant saturation des threads ?
   - Avec 30 servers, combien de req/s payment au total avant saturation complete ?
   - Compare au trafic payment reel pour montrer a quel moment ca casse.

3. **Le retry sans backoff** :
   - Pourquoi 3 retries sans backoff ont aggrave au lieu d'aider ?
   - Propose la strategie correcte (chiffres : tentatives, backoff, jitter, retry budget, timeout).

4. **Pourquoi un service non-payment est tombe aussi** :
   - Explique le mecanisme (thread pool partage). Quel pattern aurait isole payment ?

5. **Architecture corrigee** :
   - Place un circuit breaker (etats + seuils concrets).
   - Applique le bulkhead pattern : combien de threads dedier a payment max ?
   - Concois le mode degrade : que sert-on quand payment est down, sans 500 ?
   - Le LB a sorti les servers sains de la rotation : comment regler le healthcheck pour eviter ce comportement destructeur ?

6. **Runbook** :
   - Un runbook de 8 etapes pour une cascade "dependance externe lente" en production.

### Criteres de reussite
- [ ] Chaine complete : payment lent → threads bloques 30s → pool sature → service entier bloque (thread pool partage) → retry sans backoff triple le trafic → LB sort les servers sains → effondrement
- [ ] Calcul de saturation correct : 1 server = 200 threads / 30s ≈ 6.7 req/s payment ; 30 servers ≈ 200 req/s avant saturation → bien en-dessous du trafic
- [ ] Le retry sans backoff est identifie comme amplificateur (retry storm) + strategie chiffree (max 2-3, backoff exp + jitter, retry budget ≤ 10%, timeout court 2s)
- [ ] Le thread pool PARTAGE explique pourquoi un service non-payment tombe ; le bulkhead pattern l'aurait isole
- [ ] Circuit breaker avec seuils concrets (ex : > 50% erreurs ou > 20 timeouts/10s → OPEN, HALF-OPEN apres 30s)
- [ ] Bulkhead : pool dedie borne pour payment (ex : 40 threads max) → payment ne peut pas affamer les 160 autres
- [ ] Mode degrade : reservation acceptee + paiement async (queue), pas de 500
- [ ] Healthcheck regle pour ne pas sortir tout le monde en cascade (seuil global, slow-start a la reintegration)
- [ ] Runbook commence par "stopper l'amplification" (couper retry / open breaker / timeout court), pas par "ajouter des serveurs"

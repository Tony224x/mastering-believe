# Exercices Hard — Load Balancing & Networking

---

## Exercice 1 : Edge layer global d'une plateforme video live

### Objectif
Concevoir la couche d'entree complete (DNS, LB, protection) d'un service mondial a tres fort trafic et bursts extremes.

### Consigne
Tu concois l'edge layer d'une plateforme de streaming live (type Twitch simplifie) pour la partie **API et chat** (pas la video elle-meme).

**Contraintes chiffrees :**
- 20M d'utilisateurs simultanes en pic mondial, repartis : 40% Ameriques, 35% Europe, 25% Asie
- Chat : 3M de messages/sec en pic global, connexions WebSocket persistantes (1 par viewer)
- API : 800K req/s en pic (auth, metadata, follows...)
- Un evenement majeur (finale e-sport) peut faire passer un stream de 10K a 3M de viewers en 10 minutes
- SLO : connexion chat etablie < 1 s au p95 ; messages delivres < 500 ms au p95 ; disponibilite 99.95%
- Attaques DDoS L7 regulieres (jusqu'a 5M req/s malveillantes observees)

**Livre :**
1. **Chaine de routage** : du resolveur DNS du client jusqu'au pod applicatif. Choisis et justifie chaque etage : GeoDNS vs anycast, L4 vs L7 a chaque niveau, TLS termination ou.
2. **Le probleme WebSocket** : 20M de connexions persistantes ne se load-balancent pas comme du HTTP stateless. Combien de connexions par LB/gateway (hypothese : 200K conn/instance) ? Que se passe-t-il au deploiement d'une nouvelle version (drain) ? Comment eviter la reconnection storm ?
3. **Le burst x300** : 10K -> 3M viewers en 10 min sur UN stream. Quel composant sature en premier ? Concois le mecanisme d'absorption (pre-scaling sur signal, shuffle sharding des streams sur les gateways, admission control). Calcule le rythme de scale-up necessaire (instances/minute).
4. **DDoS** : separe le traitement du trafic legitime du malveillant : a quel etage filtrer quoi (anycast scrubbing, rate limiting par IP/token, challenge) ? Pourquoi le rate limiting par IP seul est insuffisant ?
5. **Degradation controlee** : a 110% de capacite, que degrades-tu en premier dans le chat (et que proteges-tu absolument) ? Definis 3 niveaux de load shedding.
6. **3 tradeoffs explicites** avec consequences acceptees.

### Criteres de reussite
- [ ] La chaine est complete et justifiee : GeoDNS (TTL court) -> PoPs anycast -> L4 (connexions WS, throughput) -> L7 regional (routing, TLS) -> services ; TLS termine a l'edge
- [ ] Calcul WebSocket : 20M / 200K = 100 instances gateway minimum (+ marge N+20%), drain progressif avec reconnexion etalee (jitter) pour eviter le thundering herd
- [ ] Le burst est traite : ~3M connexions nouvelles / 600 s = 5K conn/s sur un stream ; saturation identifiee (gateways WS ou pub/sub du chat) ; scale-up chiffre (ex : 15 instances/min) + pre-provisioning sur les events planifies
- [ ] Le DDoS est filtre en couches : volumetrique a l'edge anycast, L7 par scoring (IP + token + comportement), challenge/JS pour les non-authentifies ; IP seule insuffisante (botnets distribues, NAT partage qui penalise des users legitimes)
- [ ] Le load shedding a 3 niveaux concrets (ex : 1. couper l'historique du chat, 2. sampler l'affichage des messages des gros streams, 3. refuser les nouvelles connexions non-abonnes) en protegeant l'envoi/connexion de base
- [ ] 3 tradeoffs explicites (ex : anycast vs GeoDNS, L4 vs L7 pour les WS, cout du headroom permanent vs risque)

---

## Exercice 2 : Resilience inter-services — budget de retry et anti-cascade

### Objectif
Concevoir la politique complete timeout/retry/circuit breaking d'une architecture microservices, avec demonstration chiffree des effets d'amplification.

### Consigne
Une plateforme e-commerce a cette chaine d'appels synchrone pour `GET /checkout/summary` :

```
Client -> Edge (timeout 10s)
  -> BFF (appelle 3 services)
      -> Cart Service      -> Pricing Service -> Promo Service
      -> Shipping Service  -> Carrier API (externe, flaky)
      -> User Service
```

**Contraintes chiffrees :**
- SLO endpoint : 800 ms au p99, disponibilite 99.9%
- Latences p50/p99 mesurees : Cart 20/80 ms, Pricing 30/120 ms, Promo 15/60 ms, Shipping 40/150 ms, Carrier API 200/2000 ms (et 2% d'erreurs), User 10/40 ms
- Trafic : 5 000 req/s en pic sur cet endpoint
- Chaque service fait aujourd'hui : timeout 5 s, 3 retries immediats — partout

**Livre :**
1. **Demonstration de l'amplification** : avec la config actuelle (3 retries a chaque etage), combien d'appels peut recevoir le Promo Service pour UNE requete client si Pricing ET BFF retryent ? Generalise la formule pour N etages a R retries. Que se passe-t-il a 5 000 req/s quand Promo degrade ?
2. **Budget de latence et timeouts** : distribue le budget de 800 ms le long de la chaine (les timeouts doivent DECROITRE en descendant). Donne le timeout de chaque appel et verifie la coherence avec les p99 mesures. Quel probleme poses le Carrier API a 2000 ms de p99 ?
3. **Politique de retry** : qui a le droit de retry (un seul etage !), combien, avec quel backoff/jitter, et sur quelles erreurs uniquement ? Introduis le retry budget (ex : max 10% de requetes additionnelles).
4. **Carrier API** : 2% d'erreurs et p99 a 2 s incompatibles avec le SLO. Concois la solution complete : circuit breaker (seuils), cache des tarifs (TTL ?), fallback (tarif estime), et l'impact business du fallback.
5. **Hedging** : pour Shipping (interne, idempotent), evalue le hedged request (envoyer une 2e requete a p95). Gain sur le p99 ? Surcout en QPS ?
6. **Validation** : decris le test de chaos (faute injectee, metriques observees, critere de succes) qui prouve que la cascade de l'etape 1 ne peut plus arriver.

### Criteres de reussite
- [ ] L'amplification est calculee : (1+3) x (1+3) = jusqu'a 16 appels a Promo pour 1 requete client ; formule (R+1)^N ; a 5K req/s -> 80K req/s potentielles sur Promo = meltdown
- [ ] Les timeouts decroissent de l'edge vers les feuilles (ex : Edge 800 -> BFF 700 -> branches 500 -> feuilles 150-300 ms) et chaque timeout est > p99 du service appele (sauf Carrier, traite a part)
- [ ] Un seul etage retrie (le BFF ou le caller direct de la feuille), 1 retry max, backoff + jitter, seulement sur erreurs idempotentes/transitoires (timeout, 503), avec retry budget global
- [ ] Carrier API : circuit breaker chiffre (ex : open si error rate > 10% sur 30 s), cache tarifs avec TTL justifie (heures — les tarifs bougent peu), fallback tarif estime avec mention du risque (ecart de prix absorbe ou affiche)
- [ ] Hedging : 2e requete a ~150 ms (p95), p99 effectif ~ max reduit, surcout ~5% de QPS — gain/cout explicites
- [ ] Le chaos test est concret : injecter +1900 ms sur Promo, verifier que le p99 endpoint reste < 800 ms (fallback/timeout) et que le QPS sur Promo n'explose pas (amplification < 1.1x)

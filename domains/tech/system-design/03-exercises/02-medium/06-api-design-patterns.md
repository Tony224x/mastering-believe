# Exercices Medium — API Design & Patterns

---

## Exercice 1 : Concevoir une API de paiement idempotente

### Objectif
Concevoir une API critique (paiement) avec idempotency keys, status codes corrects et gestion des retries — le standard Stripe.

### Consigne
Tu conçois `POST /v1/charges` pour facturer un client. Le client (mobile, reseau instable) peut retry une requete dont la reponse ne lui est jamais parvenue.

**Questions :**
1. Decris le mecanisme complet d'idempotency key cote serveur : reception, verification, stockage, reponse. Quel TTL pour la cle et pourquoi ?
2. Donne le schema de la table/structure qui stocke les cles d'idempotence (champs + contraintes).
3. Le client envoie 2 fois la MEME requete (meme idempotency key) en parallele (race condition) avant que la premiere soit terminee. Que doit faire le serveur pour ne PAS debiter 2 fois ? Quelle primitive resout ca ?
4. Le client renvoie la meme idempotency key mais avec un body DIFFERENT (montant change). Que repond le serveur ?
5. Mappe les status codes HTTP pour chaque cas : succes, carte refusee, cle rejouee (idempotent hit), key reutilisee avec body different, rate limit.
6. Pourquoi un POST qui retourne toujours `200 {"success": false}` en cas d'erreur casse les clients ?

### Criteres de reussite
- [ ] Le flux idempotency key est complet (check → si existe : renvoyer la reponse stockee, sinon : executer + stocker) avec TTL 24h+
- [ ] La table a une contrainte UNIQUE sur la cle + stocke la reponse (status + body) + un etat (in_progress/done)
- [ ] La race concurrente est geree (lock / INSERT ... ON CONFLICT / etat in_progress) → un seul debit
- [ ] Cle reutilisee avec body different → 422 (ou 409) avec erreur explicite (mismatch)
- [ ] Status codes corrects : 201/200 succes, 402 carte refusee, 200 idempotent hit, 422 mismatch, 429 rate limit
- [ ] Le 200 + success:false est rejete (casse retries/breakers/monitoring qui se basent sur le status)

---

## Exercice 2 : REST vs gRPC vs GraphQL — choix argumente

### Objectif
Choisir le bon paradigme d'API selon le contexte et raisonner sur les tradeoffs.

### Consigne
Une entreprise construit une plateforme. Plusieurs surfaces d'API coexistent :

1. **API publique pour les developpeurs tiers** (integrations, webhooks, doc, SDKs).
2. **Communication interne entre microservices** (order-service ↔ inventory-service ↔ pricing-service), haut debit, latence critique.
3. **API pour l'app mobile** : ecrans heterogenes (feed, profil, parametres), besoin de minimiser les requetes et la data sur reseau mobile.

**Questions :**
1. Pour chacune des 3 surfaces, choisis REST / gRPC / GraphQL et justifie avec 2-3 arguments techniques.
2. Pour le cas mobile (GraphQL probable) : quel est le risque du N+1 query et comment DataLoader le resout ?
3. Pour le cas mobile : comment empeches-tu une "greedy query" malveillante (profondeur/complexite illimitee) ?
4. Pour le cas interne (gRPC probable) : pourquoi le contrat protobuf et HTTP/2 sont des avantages ? Quel est le piege cote cacheabilite ?
5. L'API publique REST doit evoluer sans casser. Donne 2 changements "additifs" (pas de nouvelle version) et 2 changements "breaking" (nouvelle version requise).

### Criteres de reussite
- [ ] API publique → REST (universel, cacheable, lisible, SDKs/OpenAPI)
- [ ] Interne → gRPC (typage protobuf, binaire compact, HTTP/2 multiplexing, streaming)
- [ ] Mobile → GraphQL (pas d'over/under-fetching, 1 requete pour ecrans heterogenes)
- [ ] Le N+1 et DataLoader (batching des resolvers) sont expliques
- [ ] Les greedy queries sont limitees (depth limit, complexity scoring, timeouts)
- [ ] gRPC non cacheable par les proxies HTTP classiques (piege identifie)
- [ ] Additifs (ajouter un field/endpoint optionnel) vs breaking (renommer/supprimer/changer un type) correctement classes

---

## Exercice 3 : Strategie de pagination + versioning pour une API publique

### Objectif
Choisir la bonne strategie de pagination selon le dataset, et concevoir une politique de versioning/deprecation propre.

### Consigne
Tu conçois une API publique avec deux endpoints :
- `GET /v1/admin/users` : dashboard interne, dataset de ~50K users, besoin de "aller a la page 42".
- `GET /v1/events` : flux d'evenements quasi infini (milliards), insertions constantes, consomme par des integrations.

**Questions :**
1. Pour chacun des 2 endpoints, choisis la pagination (offset, cursor, keyset) et justifie.
2. Pour `/v1/events` : explique pourquoi l'offset pagination degrade (`OFFSET 1000000`) et pourquoi le cursor est O(1). Donne la requete SQL cote serveur.
3. Pour `/v1/events` : que contient le cursor opaque ? Que se passe-t-il si un evenement est insere/supprime entre deux pages, en offset vs en cursor ?
4. Tu dois introduire un breaking change sur `/v1/events` (renommer un field). Decris ta strategie : nouvelle version, periode de deprecation, headers, communication.
5. Quel header indique aux clients la date de coupure d'une version ? Combien de preavis donnes-tu ?

### Criteres de reussite
- [ ] Admin users → offset (petit dataset, "jump to page" requis)
- [ ] Events → cursor (gros dataset, stable face aux inserts, O(1))
- [ ] La degradation O(offset) vs O(1) est expliquee avec la requete SQL (WHERE id > cursor ORDER BY id LIMIT n)
- [ ] Le cursor opaque encode la position ; offset = risque de doublon/saut, cursor = stable
- [ ] Versioning : /v2/ + deprecation 6-12 mois + logging des acces v1 + communication clients
- [ ] Le header Sunset est cite avec un preavis realiste (6-12 mois)

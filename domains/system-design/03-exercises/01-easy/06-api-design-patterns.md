# Exercices Easy — API Design & Patterns

---

## Exercice 1 : Critiquer une API mal designee

### Objectif
Savoir reperer les erreurs classiques de REST design et proposer des corrections.

### Consigne
Voici une API existante. Pour chaque endpoint, identifie **au moins 2 problemes** et propose une version corrigee.

```
GET  /getUser?userId=42
POST /createUser  (body: {name, email})  -> 200 {success: true, id: 42}
POST /deleteUser  (body: {id: 42})        -> 200 {success: true}
GET  /listUsers?page=1&per_page=20        -> 200 [... users ...]
POST /updateUserEmail (body: {id, email}) -> 200 {success: true}

Quand une erreur survient :
200 OK
{
  "success": false,
  "errorMessage": "User already exists"
}
```

### Questions
1. Liste les problemes de chaque endpoint.
2. Propose une version corrigee (URL + verbe HTTP + status codes).
3. Propose un format d'erreur coherent.
4. Explique pourquoi "200 avec success: false" est un anti-pattern.

### Criteres de reussite
- [ ] Verbes dans les URLs repere (getUser, createUser, etc.)
- [ ] 200 pour les erreurs identifie comme anti-pattern
- [ ] Les URLs corrigees sont RESTful : `/users`, `/users/{id}`
- [ ] Les codes HTTP appropries sont utilises : 201 pour POST, 204 pour DELETE, 400/404/409 pour les erreurs
- [ ] Le format d'erreur propose contient un code stable (`code` vs juste `errorMessage`)
- [ ] La pagination offset est remise en question (preferer cursor)

---

## Exercice 2 : Design d'une API de paiement idempotente

### Objectif
Concevoir un endpoint `POST /payments` robuste aux retries reseau.

### Consigne
Tu construis une API de paiement. Le client (mobile app) peut subir des timeouts reseau. Tu ne dois jamais facturer un client 2 fois.

**A rendre :**
1. Le contrat de l'endpoint :
   - URL + verbe HTTP
   - Headers requis (auth, idempotency, content-type)
   - Schema de la requete (JSON)
   - Schemas des reponses (success et erreurs)
2. Le flow d'idempotency cote serveur (pseudo-code). Qu'est-ce qui est stocke, ou, avec quel TTL ?
3. Scenario : le client fait une requete, recoit un timeout, et retry 5 minutes plus tard. Decris ce qui se passe cote serveur pour les 2 requetes.
4. Scenario : deux requetes arrivent **en parallele** avec la meme idempotency key. Comment eviter la race condition ?
5. Que se passe-t-il si le client retry avec la meme idempotency key mais un **body different** (ex : amount=200 au lieu de 100) ? Quelle reponse renvoyer ?

### Criteres de reussite
- [ ] L'endpoint est `POST /payments` (ou `/v1/payments`)
- [ ] Header `Idempotency-Key` est documente
- [ ] Le schema inclut `amount`, `currency`, `source` (carte)
- [ ] Le status 201 Created est utilise pour la premiere reussite
- [ ] TTL recommande : 24h (justifier)
- [ ] La race condition est resolue par un verrou (UNIQUE constraint, Redis SETNX, ou SELECT FOR UPDATE)
- [ ] Un body different avec meme cle retourne 422 ou 409 avec un code clair (`idempotency_key_reused`)

---

## Exercice 3 : Choisir REST, gRPC ou GraphQL

### Objectif
Savoir argumenter un choix de protocole API pour differents contextes.

### Consigne
Pour chacun des systemes suivants, choisis **REST**, **gRPC**, ou **GraphQL** (ou une combinaison) et justifie. Aucune reponse unique : c'est la justification qui compte.

1. **API publique** pour des developpeurs tiers qui integrent ton SaaS (ex : une API de meteo). Les clients sont varies : scripts Python, apps mobiles, integrations no-code (Zapier).
2. **Backend interne** d'un monolith Python qui se decoupe en microservices. Tous les services sont ecrits en Python et Go. 200 RPS entre eux.
3. **API mobile** pour une app de reseau social avec feed personnalise, profils, DMs, notifications. Chaque ecran affiche des donnees composees (post + author + comments + likes).
4. **Service de ML** qui recoit un input structure (image, texte) et retourne une prediction. Latence critique (< 50 ms p99). Appele depuis un backend Python.
5. **Admin dashboard** interne utilise par 20 collegues (CRUD sur users, produits, orders). Dev avec React.
6. **API de streaming** qui pousse des evenements en temps reel vers un client (cours de bourse, chat, live updates).

### Criteres de reussite
- [ ] 6/6 choix justifies par des criteres techniques (typage, perf, browser, streaming, cache)
- [ ] REST choisi pour l'API publique #1 (universalite)
- [ ] gRPC propose pour les cas #2 et #4 (perf + typage interne)
- [ ] GraphQL (ou une variation) pour le cas #3 (ecrans composes)
- [ ] Streaming : mention de gRPC streaming, WebSocket, ou SSE pour le #6
- [ ] Au moins une reponse mentionne qu'on peut combiner (ex : REST public + gRPC interne)

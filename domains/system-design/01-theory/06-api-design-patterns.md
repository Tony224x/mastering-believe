# Jour 6 — API Design & Patterns

## Pourquoi l'API design est un sujet strategique

**Exemple d'abord** : Tu integres l'API Stripe pour la premiere fois. En 20 minutes, tu as cree une charge. Tu lis la doc, tu comprends immediatement les concepts (`Customer`, `PaymentIntent`, `Charge`), les erreurs sont explicites (`card_declined`), tu recupees le status code avant meme de lire le body. A l'inverse, certaines APIs de banques francaises necessitent 2 semaines d'integration, retournent des codes SOAP cryptiques, et echouent silencieusement.

La **difference de productivite** entre une bonne et une mauvaise API n'est pas de 20%. C'est de **10x**. Une mauvaise API cree une dette de maintenance que tu paieras pour toujours : versioning impossible, clients casses, support de 100 cas particuliers. Une bonne API est un actif qui rayonne sur toute l'organisation.

**Key takeaway** : Une API est un **contrat**. Une fois que tu as des clients, tu ne peux plus casser. Chaque decision de design coute cher si elle est mauvaise. L'API design est donc a traiter comme du produit, pas comme du code.

---

## Les trois grandes familles : REST, gRPC, GraphQL

### REST — l'universel

REST est un **style architectural** base sur HTTP. Les ressources sont exposees via des URLs, manipulees via des verbes HTTP (GET, POST, PUT, DELETE), et representees en JSON (ou XML avant).

```
GET    /users              -- liste les users
GET    /users/42           -- recupere user 42
POST   /users              -- cree un user
PUT    /users/42           -- remplace user 42
PATCH  /users/42           -- modifie partiellement user 42
DELETE /users/42           -- supprime user 42
```

**Forces :**
- **Universel** : tout client HTTP (browser, curl, mobile) parle REST.
- **Cacheable** : GET peut etre cache par le CDN, les proxies, le browser.
- **Lisible** : les URLs et status codes sont comprehensibles sans doc.
- **Stateless** : chaque requete contient tout, facile a scaler horizontalement.
- **Simple** : pas de schemas complexes, pas de codegen obligatoire.

**Faiblesses :**
- **Over-fetching** : GET /users/42 retourne tout, meme si tu veux juste le nom.
- **Under-fetching** : GET /users/42 + GET /users/42/orders + GET /users/42/profile = 3 requetes pour afficher une page.
- **Pas typé** : le client doit parser le JSON a l'aveugle, pas de schema enforce.
- **Pas bidirectionnel** : pas de streaming natif (il existe SSE et WebSockets mais c'est hors-REST).

### gRPC — le contrat strict

gRPC est un framework RPC (Remote Procedure Call) de Google base sur **Protocol Buffers** (protobuf) et **HTTP/2**. Tu definis un `.proto` qui est le contrat. Le codegen produit le client et le serveur dans 10+ langages.

```protobuf
service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc CreateUser (CreateUserRequest) returns (User);
  rpc StreamUsers (StreamRequest) returns (stream User);
}

message User {
  string id = 1;
  string email = 2;
  int64 created_at = 3;
}
```

**Forces :**
- **Fortement typé** : le codegen garantit qu'un field existe et a le bon type.
- **Binaire (protobuf)** : 3-10x plus petit que JSON, parsing 5-10x plus rapide.
- **HTTP/2 multiplexing** : plusieurs requetes sur une seule connexion TCP.
- **Streaming bidirectionnel** : client -> serveur, serveur -> client, ou les deux en parallele.
- **Contrat versionable** : protobuf gere les champs optionnels, deprecation, compat ascendante native.

**Faiblesses :**
- **Browser-unfriendly** : gRPC ne marche pas en JS dans le navigateur sans un proxy (gRPC-Web).
- **Tooling** : plus lourd (codegen, .proto a gerer, CI plus complexe).
- **Pas cacheable** : le HTTP/2 binaire n'est pas cache par les proxies HTTP classiques.
- **Debugging** : tu ne peux pas juste curl, il faut un client gRPC.

**Use case typique** : communication interne entre microservices (backend <-> backend). Souvent accompagne de REST pour l'exterieur.

### GraphQL — la flexibilite cote client

GraphQL est un **langage de query** invente par Facebook. Le client decrit exactement les champs qu'il veut, le serveur repond avec cette structure.

```graphql
query {
  user(id: "42") {
    name
    email
    orders(last: 5) {
      id
      total
      items {
        name
        price
      }
    }
  }
}
```

**Forces :**
- **Pas d'over-fetching** : le client demande exactement ce qu'il veut.
- **Pas d'under-fetching** : une seule requete chope tout (user + orders + items).
- **Schema typé et introspection** : les IDEs genrent l'autocomplete automatiquement.
- **Versioning evolutif** : ajouter un field ne casse rien, on deprecie les anciens.

**Faiblesses :**
- **Complexite backend** : resolver par field, risque de N+1 queries (besoin de DataLoader).
- **Cache HTTP inutilisable** : tout passe par POST sur `/graphql`, pas de GET cacheable.
- **Auth et rate limiting difficiles** : il faut parser la query pour savoir quoi limiter.
- **Risque de "greedy queries"** : un client mal intentionne peut demander une query exponentielle. Il faut des limites (depth, complexity).

**Use case typique** : APIs clientes (mobile, SPA) ou les ecrans affichent des vues heterogenes avec des besoins de donnees variables. Facebook, GitHub, Shopify.

### Table de decision

| Critere | REST | gRPC | GraphQL |
|---|---|---|---|
| Client browser | Parfait | Necessite proxy | Parfait |
| Typage strict | Non (JSON lib) | Oui (protobuf) | Oui (schema) |
| Cache HTTP | Oui | Non | Non (POST) |
| Over-fetching | Oui | Moyen | Non |
| Streaming | Hacky (SSE/WS) | Natif bidirectionnel | Subscriptions |
| Performance | Bonne | Excellente | Bonne |
| Developer experience client | Bonne | Excellent (codegen) | Excellent |
| Debug (curl) | Oui | Non | JSON |
| Ecosystem 2026 | Dominant | Backend-to-backend | Facebook, GitHub, Shopify |

**Regle en entretien** : si c'est une API publique facing, REST. Si c'est internal service-to-service, gRPC. Si le frontend a des ecrans heterogenes avec des vues riches, GraphQL.

---

## Principes REST qui comptent vraiment

### 1. Les ressources, pas les actions

**Mauvais** : `POST /createUser`, `POST /deleteUser?id=42`, `POST /getUserOrders?id=42`
**Bon** : `POST /users`, `DELETE /users/42`, `GET /users/42/orders`

La ressource est le "nom". Le verbe HTTP est l'action. Jamais de verbes dans les URLs.

### 2. Les status codes HTTP correctement

| Code | Quand | Exemple |
|---|---|---|
| 200 OK | Success avec body | GET /users/42 |
| 201 Created | Ressource creee | POST /users |
| 204 No Content | Success sans body | DELETE /users/42 |
| 400 Bad Request | Mauvais input | email invalide |
| 401 Unauthorized | Pas authentifié | Token manquant |
| 403 Forbidden | Authentifié mais pas autorise | User lambda sur /admin |
| 404 Not Found | Ressource inexistante | GET /users/999 |
| 409 Conflict | Etat incoherent | Email deja pris |
| 422 Unprocessable Entity | Validation semantique | Cohorte vide dans un job |
| 429 Too Many Requests | Rate limit | API free tier |
| 500 Internal Server Error | Bug serveur | Exception non rattrapee |
| 503 Service Unavailable | Maintenance/overload | Shedding |

**Regle** : si ton API retourne toujours 200 avec un field `success: false`, ce n'est pas REST. Tu casses les clients (retries, monitoring, breakers) qui se basent sur les status codes.

### 3. Consistance dans les reponses

Tous les endpoints de la meme API doivent retourner les erreurs dans le meme format.

```json
{
  "error": {
    "code": "user_not_found",
    "message": "No user with id '999'",
    "details": { "id": "999" },
    "request_id": "req_abc123"
  }
}
```

Pourquoi `code` et `message` ? `code` est stable (machine-readable), `message` est human (peut changer). Ne code jamais ta logique sur le contenu du message, mais sur le code.

### 4. HATEOAS — la navigabilite (optionnel, utile)

**H**ypermedia **A**s **T**he **E**ngine **O**f **A**pplication **S**tate. L'idee : chaque reponse contient des liens vers les actions possibles suivantes, comme une page web.

```json
{
  "id": "42",
  "status": "draft",
  "_links": {
    "self": { "href": "/orders/42" },
    "submit": { "href": "/orders/42/submit", "method": "POST" },
    "cancel": { "href": "/orders/42", "method": "DELETE" }
  }
}
```

Peu d'APIs le font vraiment. Mais le principe ("le serveur decide ce que le client peut faire ensuite") est utile pour des workflows complexes.

---

## Versioning — la question sans reponse parfaite

Des que tu as des clients, tu ne peux pas casser ton API sans preavis. Tu dois donc **versionner**.

### Option 1 : URL versioning (le plus courant)

```
https://api.monapp.com/v1/users
https://api.monapp.com/v2/users
```

**Bon** : explicite, lisible dans les logs, facile a cacher/router.
**Mauvais** : les URLs changent entre versions, clients qui switchent doivent tout remplacer.

### Option 2 : Header versioning

```http
GET /users HTTP/1.1
Accept: application/vnd.monapp.v2+json
```

**Bon** : les URLs restent stables (good for HATEOAS, SEO, bookmarks).
**Mauvais** : invisible dans les logs, plus dur a curl, plus complexe a cacher.

### Option 3 : Content negotiation / Media types

```
Accept: application/json;version=2
```

Similaire au header versioning. Utilise par GitHub.

### Strategie pratique

- **Commence en /v1/**. Tout le monde fait ca, c'est une norme de facto.
- **Additive changes** (ajouter un field) : PAS besoin d'une nouvelle version. Les anciens clients ignorent les fields inconnus.
- **Breaking changes** (renommer, supprimer, changer le type d'un field) : NECESSITE une nouvelle version.
- **Deprecation policy** : annoncer 6 a 12 mois a l'avance, logger les acces a v1, prevenir les clients, forcer le move.
- **Sunset header** : `Sunset: Sat, 31 Dec 2026 23:59:59 GMT` indique au client quand la version sera coupee.

---

## Idempotence — la cle de la fiabilite

**Definition** : une requete est idempotente si **l'appeler N fois a le meme effet que l'appeler 1 fois**.

### Les verbes natifs

- **GET, HEAD, PUT, DELETE** : idempotents par definition HTTP.
- **POST** : NON idempotent par defaut. Deux POST /orders creent deux orders.

### Le probleme concret

Tu fais `POST /payments` pour facturer 100 EUR. La reponse ne revient pas (timeout reseau). Le paiement a-t-il abouti ? Tu ne sais pas. Si tu retry, risque de double facturation. Si tu ne retry pas, risque de ne jamais debiter.

### La solution : Idempotency Keys

Le client genere une cle unique par operation metier et l'envoie avec chaque requete :

```http
POST /payments HTTP/1.1
Idempotency-Key: pay_a1b2c3d4e5f6

{"amount": 10000, "currency": "EUR"}
```

**Cote serveur :**
1. Verifier si la cle existe deja en DB (Redis ou table dediee).
2. Si elle existe : retourner la **reponse enregistree precedemment**. NE PAS re-executer.
3. Si elle n'existe pas : executer l'operation ET stocker la cle + reponse avec un TTL (24h typique).

**Implemente par** : Stripe, Square, PayPal, Shopify, Twilio. C'est LE standard pour les APIs de paiement.

**Retention** : 24h minimum (permet retry apres reconnexion), souvent 1 semaine.

---

## Pagination — 3 strategies a connaitre

### 1. Offset Pagination (le plus simple, souvent mauvais)

```
GET /users?limit=20&offset=0
GET /users?limit=20&offset=20
GET /users?limit=20&offset=40
```

**Cote DB** : `SELECT ... ORDER BY id LIMIT 20 OFFSET 40`.

**Probleme 1** : `OFFSET 10000` force la DB a scanner 10000 rows avant de les jeter. Performance degrade lineairement.
**Probleme 2** : si un item est insere ou supprime entre deux pages, tu peux voir un item 2 fois ou le manquer.

**Quand l'utiliser** : petit dataset, admin dashboards, jamais en production grand public.

### 2. Cursor Pagination (le meilleur pour les feeds)

```
GET /users?limit=20
-> { "data": [...], "next_cursor": "eyJpZCI6MTIzfQ==" }

GET /users?limit=20&cursor=eyJpZCI6MTIzfQ==
-> { "data": [...], "next_cursor": "eyJpZCI6MTQzfQ==" }
```

Le `cursor` est un **blob opaque** (base64 d'un JSON) qui encode la position. Le serveur sait decoder et reprendre exactement la ou on s'etait arrete.

**Cote DB** : `SELECT ... WHERE id > :cursor ORDER BY id LIMIT 20`. Tres rapide avec un index.

**Forces** : performance constante O(1 page), stable face aux inserts/deletes, permet de reprendre apres interruption.

**Faiblesses** : pas de "jump to page 50", pas de total count.

**Utilise par** : Twitter, Instagram, Stripe, GitHub, Facebook.

### 3. Keyset Pagination (cursor simple)

Variante simple du cursor : on utilise directement un field comme cursor (pas encode).

```
GET /users?after_id=123&limit=20
```

**Cote DB** : `SELECT ... WHERE id > 123 ORDER BY id LIMIT 20`.

Identique au cursor mais sans encoding. Moins flexible (un seul field, pas de tri complexe) mais plus simple.

### Table comparative

| Strategie | Performance | Stabilite | UX (jump to page) | Utilise par |
|---|---|---|---|---|
| Offset | O(offset) degrade | Instable | Oui | Admin dashboards |
| Cursor | O(1) constant | Stable | Non | Feeds sociaux |
| Keyset | O(1) constant | Stable | Non | APIs simples |

---

## API Gateway — le concept qui unifie tout

Un **API Gateway** est une couche entre les clients et tes microservices backend. Il centralise les concerns transverses.

```
Client --> API Gateway --> [Service A, Service B, Service C, ...]
              |
              |-- Auth (JWT validation)
              |-- Rate limiting
              |-- Routing (path -> service)
              |-- Aggregation (1 request -> N service calls)
              |-- Observability (logs, traces, metrics)
              |-- Transformations (version, protocol)
```

**Responsabilites classiques :**
1. **Authentification** : valider le JWT, injecter `user_id` dans les headers pour le backend.
2. **Rate limiting** : par user, par IP, par endpoint.
3. **Routing** : `/users/*` -> user-service, `/orders/*` -> order-service.
4. **Protocol translation** : REST externe -> gRPC interne.
5. **Response aggregation** : combine plusieurs services en une seule reponse (BFF — Backend for Frontend).
6. **Caching** : cache les GET frequents.
7. **Observability** : logs centralises, tracing distribue (X-Request-ID).

**Exemples** : Kong, AWS API Gateway, Apigee, Tyk, Envoy avec filtres, Istio.

**Attention** : le gateway peut devenir un monolith distribue si tu y mets trop de logique metier. Il doit rester **generique** (infra/transverse), pas faire de business logic.

---

## OpenAPI — la specification as code

**OpenAPI** (anciennement Swagger) est un standard pour decrire une API REST en YAML ou JSON.

```yaml
openapi: 3.0.0
info:
  title: My API
  version: 1.0.0
paths:
  /users/{id}:
    get:
      summary: Get a user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
components:
  schemas:
    User:
      type: object
      properties:
        id: { type: string }
        email: { type: string }
```

**Pourquoi c'est indispensable :**
1. **Doc interactive** (Swagger UI, Redoc) generee automatiquement.
2. **Client codegen** dans 30+ langages.
3. **Validation** des requetes en entree et des reponses en sortie.
4. **Contract testing** entre services (verifier qu'un service respecte son spec).
5. **Mocking** : tu peux generer un faux serveur qui repond selon le spec avant que le backend soit ecrit.

**Regle** : tout API publique doit avoir un OpenAPI a jour. C'est non-negociable en 2026.

---

## Real-world : comment les grands font

### Stripe — la reference du REST
Stripe est cite dans toutes les conferences comme la reference du REST design. Idempotency keys, error codes stables, versioning par header (`Stripe-Version: 2023-10-16`), pagination cursor, doc OpenAPI impeccable. Ils font evoluer leur API depuis 2011 sans casser personne grace a une discipline extreme.

### GitHub — REST + GraphQL cote a cote
GitHub expose deux APIs : REST v3 (historique) et GraphQL v4 (moderne). Les clients choisissent. REST est utilise pour les integrations simples, GraphQL pour les apps riches qui veulent minimiser les requetes. Les deux coexistent sans friction.

### Google — gRPC en interne, REST en externe
Google utilise gRPC pour quasi tout son trafic internal. Ses APIs publiques (Cloud, YouTube Data API) sont REST/JSON pour l'ecosystem, avec des clients officiels generes qui traduisent vers gRPC en interne.

### Netflix — BFF pattern
Netflix a plusieurs clients (iOS, Android, TV, web). Chaque client a un BFF (Backend for Frontend) qui est un API Gateway specialise. Le BFF iOS aggregre les services pour retourner exactement ce que l'app iOS a besoin, le BFF TV retourne autre chose. Aucune logique metier dans les BFFs, juste de l'aggregation et de la transformation.

---

## Flash cards

**Q1** : Quand choisir gRPC plutot que REST ?
**R** : Pour la communication backend-to-backend (microservices) ou le typage strict et la performance comptent. Pas pour une API publique facing browser.

**Q2** : Pourquoi les idempotency keys sont critiques pour les APIs de paiement ?
**R** : Parce qu'un timeout reseau laisse le client dans l'incertitude (paye ou pas ?). L'idempotency key permet de retry en toute securite : si la cle existe deja, le serveur retourne la meme reponse sans re-executer.

**Q3** : Pourquoi eviter la pagination par offset sur les grands datasets ?
**R** : Parce que `OFFSET 10000` force la DB a scanner 10000 lignes pour les jeter. Performance degrade lineairement. Cursor pagination est O(1) constant avec un index.

**Q4** : Quelle est la difference entre 401 et 403 ?
**R** : 401 = pas authentifie (pas de token, token invalide). 403 = authentifie mais pas autorise pour cette ressource specifique.

**Q5** : Qu'est-ce qu'un API Gateway ne doit PAS faire ?
**R** : Il ne doit pas contenir de business logic. Son role est infra/transverse : auth, rate limit, routing, observability. Si tu mets la logique metier dedans, tu crees un monolith cache.

---

## Key takeaways

1. **REST pour le public, gRPC pour l'internal, GraphQL pour les clients heterogenes.** Pas de guerre, chacun son cas.
2. **Ressources > actions.** Pas de verbes dans les URLs (`/createUser` est wrong).
3. **Status codes HTTP corrects.** Pas de 200 OK avec `success: false`.
4. **Idempotency keys** pour toute operation non-idempotente critique (paiement, creation de ressource).
5. **Cursor pagination** est le defaut pour les APIs publiques avec de gros datasets. Offset seulement pour l'admin.
6. **Versioning par URL (/v1/)** est le plus simple et le plus courant. Deprecate avec 6-12 mois de preavis.
7. **API Gateway centralise l'infra** (auth, rate limit, routing), pas la logique metier.
8. **OpenAPI obligatoire** : doc interactive, codegen, validation, contract testing.
9. **En entretien** : parler d'idempotency, de pagination cursor, d'API versioning strategy, et de backward compatibility.

---

## Pour aller plus loin

Ressources canoniques sur le sujet :

- **Google API Design Guide** (Google Cloud, officiel) — guide utilise en interne chez Google depuis 2014 : resource-oriented design, naming, standard methods, errors, versioning. La reference REST/gRPC. https://cloud.google.com/apis/design
- **gRPC Documentation** (gRPC.io, officiel) — concepts (services, RPC types, streaming), generation de stubs, deadlines, interceptors. Couvre 12+ langages. https://grpc.io/docs/
- **System Design Interview Vol 2** (Alex Xu, ByteByteGo 2022) — chapitres dedies a "Design a payment system" et "Distributed message queue" detaillent idempotency keys et retries cote API. https://www.amazon.com/System-Design-Interview-Volume-Insiders/dp/1736049119
- **Designing Data-Intensive Applications** (Martin Kleppmann, O'Reilly 2017) — Ch 4 (Encoding and Evolution) traite schema evolution, backward/forward compatibility, Avro/Protobuf — fondations du versioning d'API. https://dataintensive.net/

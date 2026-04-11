"""
Solutions -- Exercices Jour 6 : API Design & Patterns

Solutions detaillees des 3 exercices Easy.

Usage:
    python 06-api-design-patterns.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercice 1 : Critiquer une API mal designee
# =============================================================================


def easy_1_critique_bad_api():
    """Solution : identifier les anti-patterns et proposer un fix."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Critique d'une API mal designee")
    print(SEPARATOR)

    print("""
  Problemes identifies :

  1. GET /getUser?userId=42
     PROBLEMES :
     - Verbe ('get') dans l URL alors que GET est deja le verbe HTTP.
     - userId en query param alors que c'est une ressource identifiee.
     - Pas de 404 si user inexistant : renvoie quoi ? 200 avec body vide ?
     FIX :
     GET /users/42
     -> 200 {"id": "42", "email": "..."}
     -> 404 {"error": {"code": "user_not_found", ...}}

  2. POST /createUser -> 200 {success: true, id: 42}
     PROBLEMES :
     - 'create' dans l URL.
     - 200 OK alors que c'est une creation : doit etre 201 Created.
     - 'success: true' est inutile : si status < 400, c'est bon.
     - La reponse ne contient pas le header Location: /users/42 (RFC).
     FIX :
     POST /users
     Body : {"email": "...", "name": "..."}
     -> 201 Created
     Headers : Location: /users/42
     Body : {"id": "42", "email": "...", "created_at": "..."}
     -> 409 Conflict si email deja pris.

  3. POST /deleteUser (body: {id: 42}) -> 200
     PROBLEMES :
     - 'delete' dans l URL alors qu il existe DELETE.
     - POST au lieu de DELETE empeche le cache HTTP et trompe les proxies.
     - L id dans le body au lieu de l URL.
     FIX :
     DELETE /users/42
     -> 204 No Content (aucun body a retourner apres delete)
     -> 404 si user inexistant

  4. GET /listUsers?page=1&per_page=20
     PROBLEMES :
     - 'list' dans l URL. GET /users suffit.
     - Offset pagination (page/per_page) : O(offset) qui degrade, instable
       face aux inserts/deletes.
     - Pas de wrapper de pagination : renvoie juste [...].
     FIX :
     GET /users?limit=20&cursor=<opaque>
     -> 200 {
          "data": [...],
          "next_cursor": "eyJpZCI6MTIzfQ==",
          "has_more": true
        }

  5. POST /updateUserEmail (body: {id, email})
     PROBLEMES :
     - Verbe + champ specifique dans l URL = explosion combinatoire
       (updateUserEmail, updateUserName, updateUserPhone, ...).
     - Devrait etre PATCH /users/42 pour une mise a jour partielle.
     FIX :
     PATCH /users/42
     Body : {"email": "new@example.com"}
     -> 200 {"id": "42", "email": "new@example.com", ...}
     -> 422 Unprocessable Entity si email invalide
     -> 409 Conflict si email deja pris

  6. Format d erreur : 200 OK avec success:false
     PROBLEMES :
     - Casse tous les clients HTTP qui se basent sur le status code :
       * Les retries (sur 5xx) ne se declenchent pas.
       * Les monitorings (4xx, 5xx rates) affichent 0% d erreurs.
       * Les load balancers pensent que le backend va bien.
       * Les circuit breakers ne se declenchent jamais.
     - 'errorMessage' est un free text qui peut changer : impossible
       pour le client de coder de la logique dessus.

     FIX : format d erreur coherent, standard :
     HTTP 4xx ou 5xx + body :
     {
       "error": {
         "code": "email_already_exists",        <- STABLE, machine-readable
         "message": "Email is already used",    <- human, peut changer
         "details": {"email": "bob@example.com"},
         "request_id": "req_abc123"
       }
     }

  Resume :
     - Verbe HTTP dans l URL = interdit
     - 200 OK pour tout = casse les clients
     - Offset pagination sur les gros datasets = a eviter
     - Format d erreur standard = indispensable
    """)


# =============================================================================
# EASY -- Exercice 2 : API de paiement idempotente
# =============================================================================


def easy_2_idempotent_payment_api():
    """Solution : design complet d un endpoint POST /payments."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : POST /payments idempotent")
    print(SEPARATOR)

    print("""
  1. Contrat de l endpoint :

     POST /v1/payments
     Headers :
       Authorization: Bearer <token>
       Content-Type: application/json
       Idempotency-Key: <uuid-v4>         (OBLIGATOIRE pour POST)
       X-Request-Id: <uuid>                (optionnel, pour le tracing)

     Request body :
     {
       "amount_cents": 10000,
       "currency": "EUR",
       "source": "card_xyz",               // token de carte Stripe/Ayden
       "description": "Premium plan 2026",
       "metadata": {"user_id": "42"}
     }

     Success response (premier succes) :
     HTTP/1.1 201 Created
     {
       "id": "pay_abc123",
       "status": "succeeded",
       "amount_cents": 10000,
       "currency": "EUR",
       "created_at": "2026-04-11T10:00:00Z"
     }

     Idempotent replay (meme cle, meme body) :
     HTTP/1.1 201 Created       <- meme status que la premiere fois
     X-Idempotent-Replay: true
     (meme body)

     Erreurs :
     400 : body invalide (amount negatif, currency inconnue...)
     401 : auth manquante ou invalide
     402 : card_declined / insufficient_funds (code stable)
     409 : idempotency_key_reused_with_different_body
     422 : validation semantique (ex : user banni)
     429 : rate limit
     500/503 : erreur serveur / downstream (retry autorise)

  2. Flow d idempotency cote serveur :

     Table 'idempotency_records' (Postgres) :
       key           TEXT PRIMARY KEY,
       request_hash  TEXT NOT NULL,          -- hash sha256 du body
       response_status INT,
       response_body   JSONB,
       created_at    TIMESTAMPTZ DEFAULT NOW(),
       ttl_expires_at TIMESTAMPTZ

     TTL recommande : 24 heures.
     - Assez long pour couvrir les retries apres reconnexion mobile
     - Assez court pour libererer de la place (on ne garde pas eternellement)
     - Stripe utilise 24h aussi

  3. Scenario : timeout + retry 5 min plus tard (meme body)

     T0 : client envoie POST avec Idempotency-Key=K
          serveur insert la cle (verrou), appelle Stripe, charge OK
          serveur enregistre (K, body_hash, 201, response_body) en DB
          serveur renvoie 201 -> reseau timeout -> client recoit rien

     T0+5min : client retry avec le meme K et le meme body
          serveur verifie : SELECT * FROM idempotency_records WHERE key=K
          Resultat : trouve (K, body_hash, 201, response_body)
          serveur verifie body_hash == sha256(new_body) -> match
          serveur retourne 201 + response_body stocke
          -> NE RAPPELLE PAS STRIPE.
          -> Une seule charge a ete effectuee.

  4. Race condition : 2 requetes en parallele avec meme cle

     Sans protection : les deux processus verifient 'key not found'
     en meme temps, puis les deux appellent Stripe en meme temps.
     Double charge !

     Solution robuste : UNIQUE constraint + INSERT ON CONFLICT.

     Pseudo-code :
     def process_payment(key, body):
         body_hash = sha256(body)

         # Step 1 : tente d'inserer la cle (marqueur 'in_progress')
         try:
             db.execute('''
                 INSERT INTO idempotency_records (key, request_hash, state)
                 VALUES (%s, %s, 'in_progress')
             ''', (key, body_hash))
         except UniqueViolation:
             # Un autre processus a deja pris la cle
             existing = db.fetch('SELECT * FROM idempotency_records WHERE key=%s', (key,))
             if existing['state'] == 'in_progress':
                 # Attendre un peu puis re-verifier, ou retourner 409
                 raise APIError('idempotency_key_in_progress', status=409)
             if existing['request_hash'] != body_hash:
                 raise APIError('idempotency_key_reused', status=422)
             return existing['response_status'], existing['response_body']

         # Step 2 : on a le verrou. Appeler Stripe.
         response = stripe.charge(body)
         # Step 3 : update le record avec la reponse finale
         db.execute('''
             UPDATE idempotency_records
             SET state='done', response_status=%s, response_body=%s
             WHERE key=%s
         ''', (201, response, key))
         return 201, response

     Alternative : Redis SETNX avec TTL.
        ok = redis.set(f'idem:{key}', 'in_progress', nx=True, ex=300)
        if not ok:
            return cached_response_or_409()

  5. Retry avec meme cle mais body different :

     Cote client : c'est une ERREUR d'usage. La meme cle doit designer
     une meme operation. Changer le amount avec la meme cle = bug client.

     Reponse serveur :
     HTTP 422 Unprocessable Entity
     {
       "error": {
         "code": "idempotency_key_reused",
         "message": "This Idempotency-Key was already used with a different request body.",
         "details": {
           "expected_hash": "a1b2...",
           "got_hash": "c3d4..."
         }
       }
     }

     OU HTTP 409 Conflict (Stripe utilise 400 avec code specifique).
     Ne PAS traiter la requete : preferer echouer que risquer l'ambiguite.
    """)


# =============================================================================
# EASY -- Exercice 3 : REST vs gRPC vs GraphQL
# =============================================================================


def easy_3_protocol_choice():
    """Solution : matrice de decision protocol pour 6 scenarios."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : REST, gRPC, ou GraphQL ?")
    print(SEPARATOR)

    choices = [
        (
            "API publique SaaS meteo, clients varies (Python, mobile, Zapier)",
            "REST",
            "REST est universel : curl, Postman, Zapier, n importe quel langage. "
            "Cache HTTP = CDN friendly (les requetes meteo sont cacheables). "
            "Pas besoin de codegen force pour les clients. Doc Swagger/OpenAPI "
            "fournit une experience riche. gRPC serait hostile aux integrations "
            "no-code. GraphQL serait overkill pour des queries simples."
        ),
        (
            "Backend interne Python+Go, 200 RPS, microservices",
            "gRPC",
            "Communication backend-to-backend = pas de contraintes browser. "
            "gRPC donne : typage strict (protobuf enforce les schemas), "
            "performance (binaire, HTTP/2 multiplexing), codegen polyglotte "
            "(Python + Go natif), et streaming natif si besoin. 200 RPS = "
            "la serialisation JSON couterait 20-30% de CPU qu on economise."
        ),
        (
            "App mobile reseau social : feed, profils, DMs, comments",
            "GraphQL (+ REST pour quelques endpoints)",
            "Ecrans composes avec donnees heterogenes = cas ideal GraphQL. "
            "Un ecran profil a besoin de user + posts + followers_count + "
            "mutual_friends. En REST ce serait 4-5 requetes (waterfall + "
            "latence mobile). En GraphQL, 1 query qui retourne exactement "
            "ce qui est affiche. Facebook/Instagram l ont invente pour ca."
        ),
        (
            "Service ML : input structure -> prediction, < 50 ms p99",
            "gRPC",
            "Latence critique : binaire + HTTP/2 sont 2-5x plus rapides "
            "que REST/JSON pour le meme payload. Typage strict evite les "
            "bugs de parsing. Appele depuis un backend Python = pas de "
            "contrainte browser. Bonus : streaming gRPC pour du realtime ML."
        ),
        (
            "Admin dashboard interne React, 20 collegues, CRUD",
            "REST",
            "20 collegues = pas de perf critique. CRUD simple (users, "
            "products, orders) = pas besoin de la flexibilite GraphQL. "
            "REST est le plus simple a dev pour le back et a consommer "
            "pour le front. OpenAPI + openapi-generator = typage TypeScript "
            "cote front gratuit. Pas besoin d en faire plus."
        ),
        (
            "API streaming realtime : bourse, chat, live updates",
            "WebSocket ou gRPC streaming (pas REST)",
            "REST est request/response unidirectionnel : chaque update "
            "necessite une nouvelle requete = gaspillage. Les options : "
            "(1) WebSocket pour le web (standard, lu par tous les browsers), "
            "(2) gRPC streaming bidirectionnel si client non-browser, "
            "(3) Server-Sent Events (SSE) pour du push unidirectionnel simple. "
            "Pour un chat moderne : WebSocket est le choix par defaut. "
            "Bonus : GraphQL subscriptions fonctionnent sur WebSocket."
        ),
    ]

    for i, (scenario, choice, reason) in enumerate(choices, 1):
        print(f"\n  {i}. {scenario}")
        print(f"     Choix : {choice}")
        print(f"     Raison : {reason}")

    print("\n  Insight final :")
    print("     - Tu peux COMBINER : REST public + gRPC interne est le pattern")
    print("       dominant dans les grosses boites (Google, Netflix, Shopify).")
    print("     - Le choix depend du CLIENT autant que du serveur. Si c'est un")
    print("       browser, REST/GraphQL. Si c'est un autre backend, gRPC.")
    print("     - Le streaming necessite un protocole bidirectionnel : WebSocket,")
    print("       gRPC streaming, ou SSE selon le contexte.")


def main():
    easy_1_critique_bad_api()
    easy_2_idempotent_payment_api()
    easy_3_protocol_choice()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 6.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

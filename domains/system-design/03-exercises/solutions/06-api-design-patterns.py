"""
Solutions -- Exercices Jour 6 : API Design & Patterns

Solutions detaillees des exercices Easy, Medium et Hard.

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


# =============================================================================
# MEDIUM -- Exercice 1 : API de reservation de salles
# =============================================================================


def medium_1_reservation_api():
    """Solution : endpoints, idempotence, erreurs, pagination, versioning."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : API de reservation de salles")
    print(SEPARATOR)

    print("""
  1. ENDPOINTS
     GET    /v1/rooms                          200
     GET    /v1/rooms/{id}                     200, 404
     GET    /v1/rooms/{id}/availability?from=&to=   200, 400, 404
     POST   /v1/reservations                   201, 400, 401, 409, 422
     GET    /v1/reservations/{id}              200, 404
     PATCH  /v1/reservations/{id}              200, 404, 409
     DELETE /v1/reservations/{id}              204, 404, 410
     GET    /v1/reservations?user_id=&cursor=  200
     Ressources au pluriel, zero verbe, la dispo est une sous-ressource.

  2. IDEMPOTENCE DU POST
     Header obligatoire : Idempotency-Key: <uuid client>
     - 1er appel : on stocke (key, hash(body), status, response) PUIS on
       traite. Reponse 201 persistee.
     - Rejeu (meme key, meme body) : on REJOUE la reponse stockee (201),
       on ne recree rien.
     - Meme key, body DIFFERENT : 422 (erreur d'integration cote client).
     - Requete encore en cours avec la meme key : 409 'processing'
       (ou attente courte puis replay).
     Retention des cles : 24-48 h (TTL), suffisant pour couvrir les
     retries reseau et les doubles clics.

  3. CONFLIT DE RESERVATION
     Status : 409 Conflict
     {
       "error": {
         "type": "conflict",
         "code": "room_already_booked",
         "message": "Room R-204 is already booked from 14:00 to 15:00.",
         "details": {"room_id": "R-204", "conflicting_reservation": "..."},
         "request_id": "req_8fz2..."
       }
     }
     Code machine-readable stable (les clients codent dessus), message
     humain, request_id pour le support.

  4. PAGINATION CURSOR
     GET /v1/reservations?limit=50&cursor=eyJpZCI6IDk4NzZ9
     Reponse : { "data": [...], "next_cursor": "...", "has_more": true }
     Cursor = base64({last_id, last_created_at}) : OPAQUE (le client ne
     le construit pas) et STABLE : une insertion pendant la pagination ne
     decale pas les pages. L'offset, lui, re-scanne N lignes (lent) et
     produit des doublons/trous quand des lignes s'inserent.

  5. RENOMMAGE start -> start_time
     Etape 1 : additif -- v1 renvoie LES DEUX champs (start + start_time),
       accepte les deux en ecriture. Doc + header Deprecation sur start.
     Etape 2 : annonce -- changelog, emails, Sunset: <date> (6-12 mois).
     Etape 3 : mesure -- on ne coupe que quand le trafic utilisant
       'start' tombe sous un seuil (~quelques %), relances ciblees.
     Etape 4 : /v2/ sans le champ legacy ; v1 maintenu jusqu'a la date
       Sunset annoncee.
     Un breaking change sans phase additive = clients casses = tickets.""")


# =============================================================================
# MEDIUM -- Exercice 2 : Migration REST -> gRPC interne
# =============================================================================


def medium_2_grpc_migration():
    """Solution : perimetre, gain de latence, .proto, compat browser."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Migration vers gRPC interne")
    print(SEPARATOR)

    print("""
  1. PERIMETRE
     gRPC : TOUS les appels service-a-service internes (les 12 micro-
     services entre eux) -- fort volume, contrats stricts, pas de browser.
     REST : l'API publique (partenaires, webapps) et tout ce qu'un
     browser consomme directement.
     Regle : gRPC pour l'interne, REST (ou GraphQL) pour le bord.""")

    network, serial, processing = 2, 4, 6
    new_serial = 0.5
    print(f"  2. GAIN DE LATENCE")
    print(f"     Avant : {network} (reseau) + {serial} (serde JSON) + {processing} (traitement) = {network+serial+processing} ms")
    print(f"     Apres : {network} + ~{new_serial} (protobuf binaire) + {processing} = {network+new_serial+processing} ms")
    print(f"     ~30% de gain par appel ; sur une chaine de 4 appels internes,")
    print(f"     c'est ~14 ms au p50 -- et les payloads passent de 8 Ko a ~2 Ko")
    print(f"     (moins de bande passante, moins de GC).")

    print('''
  3. FICHIER .proto
     syntax = "proto3";
     package rooms.v1;

     service RoomAvailability {
       // Unary : une question, une reponse.
       rpc CheckAvailability(CheckAvailabilityRequest)
           returns (CheckAvailabilityResponse);
       // Server streaming : le serveur pousse chaque changement.
       rpc WatchAvailability(WatchAvailabilityRequest)
           returns (stream AvailabilityEvent);
     }

     message CheckAvailabilityRequest {
       string room_id = 1;
       int64 from_unix = 2;
       int64 to_unix = 3;
     }
     message CheckAvailabilityResponse {
       bool available = 1;
       repeated TimeSlot conflicts = 2;
     }
     message WatchAvailabilityRequest { string room_id = 1; }
     message AvailabilityEvent {
       string room_id = 1;
       bool available = 2;
       int64 changed_at_unix = 3;
     }
     message TimeSlot { int64 from_unix = 1; int64 to_unix = 2; }

  4. CLIENTS BROWSER
     Option A : gRPC-Web + proxy (Envoy) -- garde un seul contrat proto,
       mais outillage browser plus lourd, debugging moins familier.
     Option B : gateway REST devant les services gRPC (transcoding
       grpc-gateway / endpoints REST dedies) -- les clients web restent
       en REST/JSON standard.
     Recommandation : B pour des clients heterogenes/publics (zero
     friction), A si on controle 100% des frontends et qu'on veut le
     streaming proto de bout en bout.

  5. REGLES DE COMPATIBILITE PROTOBUF
     - Ne JAMAIS reutiliser ni renumeroter un numero de champ
       (reserver les numeros supprimes : reserved 4;).
     - N'ajouter que des champs OPTIONNELS (proto3 : tout est optional) ;
       les vieux clients ignorent les champs inconnus.
     - Ne jamais changer le TYPE d'un champ existant ; pour un changement
       structurel, nouveau champ (ou nouveau message/version de service).''')


# =============================================================================
# MEDIUM -- Exercice 3 : API Gateway
# =============================================================================


def medium_3_api_gateway():
    """Solution : perimetre gateway, dimensionnement, trust, rollout."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : API Gateway sans monolithe")
    print(SEPARATOR)

    import math
    peak = 25_000
    per_pod = 4_000
    pods = math.ceil(peak / per_pod) + 2

    print("""
  1. PERIMETRE
     Dans le gateway (infra transverse, identique pour tous) :
       auth JWT, API keys, rate limiting, CORS, TLS, logging des acces,
       routing, compression, request-id.
     Dans les services (logique metier) :
       validation metier, autorisations fines (ownership d'une ressource),
       toute transformation de donnees metier.
     Critere : si la regle peut s'exprimer SANS connaitre le domaine
     metier, elle va au gateway. Sinon, service.

  2. ENRICHISSEMENT 'PREMIUM' AU GATEWAY : REFUSE.
     C'est de la logique metier (segmentation client + composition de
     reponse). L'accepter transforme le gateway en monolithe cache :
     deploys couples, equipe gateway goulot, tests metier dans l'infra.
     A faire dans un service (ou un BFF dedie au client concerne).""")

    print(f"  3. DIMENSIONNEMENT")
    print(f"     ceil({peak:,} / {per_pod:,}) = {math.ceil(peak/per_pod)} pods + 2 de marge = {pods} pods")
    print(f"     Metriques : latence AJOUTEE par le gateway (p99 < 5 ms),")
    print(f"     error rate par route, CPU/memoire, saturation des connexions")
    print(f"     amont/aval. Le gateway doit etre STATELESS -> scale lineaire.")

    print("""
  4. CONFIANCE DOWNSTREAM
     Le gateway valide le JWT puis transmet l'identite via des headers
     internes (X-User-Id, X-Scopes) SIGNES (JWT interne court ou mTLS
     entre gateway et services).
     Sans signature ni mTLS : n'importe quel pod compromis (ou un dev
     avec kubectl port-forward) peut forger X-User-Id = admin.
     Regle zero-trust : le reseau interne n'est pas une frontiere de
     securite ; chaque service verifie la signature (verification legere,
     pas re-validation complete aupres de l'IdP).

  5. ROLLOUT SANS BIG BANG
     1) Deployer le gateway en pass-through devant UN endpoint non
        critique (ex : /health ou un GET interne) -- valider la latence.
     2) Migrer endpoint par endpoint (le DNS/LB route progressivement),
        en doublonnant temporairement l'auth (gateway + service) jusqu'a
        la bascule.
     3) Shadow traffic / canary 5% par endpoint avant 100%.
     4) Une fois tous les endpoints migres : retirer le code d'auth
        duplique des services (le gain final).
     Duree typique : quelques semaines, reversible a chaque etape.""")


# =============================================================================
# HARD -- Exercice 1 : API open banking
# =============================================================================


def hard_1_open_banking():
    """Solution : virements idempotents, webhooks fiables, machine a etats."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : API bancaire (open banking)")
    print(SEPARATOR)

    print("""
  1. ENDPOINTS VIREMENT + SUIVI ASYNCHRONE
     POST /v1/payments
       Idempotency-Key obligatoire (400 si absente).
       201 { id, status: "initiated", _links: { self, events } }
     GET /v1/payments/{id}                 -> statut courant (polling)
     GET /v1/payments/{id}/events          -> historique des transitions
     Webhooks : payment.pending_authorization, payment.executed,
       payment.rejected, pousses vers l'URL enregistree de la fintech.
     Les DEUX coexistent : webhooks pour la latence, polling comme
     source de verite rejouable (exigence regulateur + rattrapage).

  2. WEBHOOKS FIABLES
     - Signature : HMAC-SHA256 du body avec un secret PAR fintech,
       header X-Signature + timestamp signe (anti-replay, fenetre 5 min).
       Rotation du secret supportee (2 secrets actifs pendant la rotation).
     - Retries : backoff exponentiel 1min -> 1h, pendant 72 h, puis DLQ
       + alerte a la fintech (email/dashboard).
     - Ordering : NON garanti (retries) -> chaque event porte un numero
       de sequence par payment_id ; le consommateur reordonne ou ignore
       les events plus vieux que son etat courant.
     - Reconciliation : GET /v1/events?after_cursor=... permet de
       rattraper tout ce qui a ete manque (les webhooks ne sont qu'une
       optimisation de latence, jamais la source de verite).

  3. MACHINE A ETATS
     initiated -> pending_authorization -> executed
          |                |          \\-> rejected
          v                v
       cancelled       cancelled
     Transitions interdites : executed -> cancelled, rejected -> *.
     Expose : champ status + status_history [{status, at}].
     PATCH /v1/payments/{id} {action: "cancel"} :
       - initiated / pending_authorization : 200, status=cancelled
       - executed : 409 { code: "payment_already_executed" }
         (la reponse correcte est alors un virement RETOUR, pas un cancel)
       - rejected/cancelled : 409 idempotent-friendly (deja terminal)

  4. IDEMPOTENCE SOUS CONCURRENCE (2 pods, 3 ms d'ecart)
     Primitive atomique en DB :
       INSERT INTO idempotency_keys (key, body_hash, status)
       VALUES (:k, :h, 'processing')
       ON CONFLICT (key) DO NOTHING;
     - Pod A gagne l'insert -> il traite le virement.
     - Pod B : insert no-op -> il lit la ligne :
         status='processing' -> 409 Conflict + Retry-After: 1
         (ou long-poll quelques centaines de ms puis replay du resultat)
         status='completed'  -> rejoue la reponse stockee.
     Jamais deux virements : la contrainte UNIQUE est la garantie, pas
     le code applicatif.

  5. RATE LIMITING CONTRACTUEL
     Token bucket par fintech : rate = 100 req/s, burst = 300-500.
     Le bucket absorbe les rafales legitimes de debut de mois SANS
     augmenter le debit soutenu. Granularite : quota global + sous-quota
     sur les POST (les ecritures coutent plus cher que les GET).
     429 + Retry-After + X-RateLimit-* ; les depassements sont visibles
     dans le dashboard fintech AVANT d'etre bloquants (alertes a 80%).

  6. ENGAGEMENT 24 MOIS
     Interdits en v1 : retirer/renommer un champ, changer un type ou la
     semantique d'un status code, ajouter un champ OBLIGATOIRE en input,
     changer les regles de signature webhook.
     Autorises : champs additifs, nouveaux endpoints, nouveaux events.
     Sortie de v2 : annonce + doc de migration -> period parallele
     24 mois -> headers Deprecation/Sunset sur v1 -> suivi du trafic v1
     par fintech -> relances ciblees -> extinction quand < 1% du trafic
     (ou date contractuelle atteinte).

  TRADEOFFS EXPLICITES
     1) Webhooks at-least-once + reconciliation plutot que 'exactly-once'
        promis : honnete et robuste ; cout = les fintechs doivent
        dedupliquer (doc + SDKs fournis).
     2) 409 'processing' plutot qu'attente bloquante sur l'idempotence
        concurrente : API previsible, pas de threads bloques ; cout = un
        retry cote client.
     3) Token bucket avec burst genereux plutot que fenetre stricte :
        absorbe la paie du 1er du mois ; cout = pics instantanes a
        provisionner cote infra.""")


# =============================================================================
# HARD -- Exercice 2 : Facade GraphQL/BFF sur 3 backends
# =============================================================================


def hard_2_federation_facade():
    """Solution : choix de facade, absorption Reviews, N+1, protections."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Facade au-dessus de 3 backends heterogenes")
    print(SEPARATOR)

    print("""
  1. DECISION
     - GraphQL federation partout : flexible, MAIS 4 devs, 40 partenaires
       B2B en REST stable, et 3 backends qu'on ne peut pas modifier ->
       cout d'apprentissage et d'operation eleve pour un besoin
       principalement MOBILE.
     - BFF mobile dedie : repond exactement au besoin (1 appel/ecran),
       simple, MAIS risque de proliferation de BFFs si chaque client en
       veut un.
     - Gateway + endpoints composites REST : proche du BFF, mutualise.
     CHOIX pragmatique : un BFF mobile (REST composite : GET /screens/
     product/{id}) qui agrege les 3 backends ; les partenaires B2B
     continuent sur l'API Catalog REST inchangee. GraphQL n'est PAS
     necessaire tant qu'il n'y a qu'un ou deux ecrans consommateurs --
     a reconsiderer si les clients se multiplient.""")

    hit_rate_needed = 1 - 50 / 8000
    print(f"  2. ABSORPTION DE REVIEWS (50 req/s pour 8 000 req/s d'ecrans)")
    print(f"     Hit rate cache minimal = 1 - 50/8000 = {hit_rate_needed:.1%}")
    print("""     Dispositif :
     - cache Redis par product_id : resume des reviews (note moyenne,
       compte, top-3) -- TTL 1-6 h : des reviews fraiches a la minute
       n'apportent rien sur un ecran produit.
     - stale-while-revalidate : on sert le perime et on rafraichit en
       arriere-plan (le refresh est borne a 50 req/s par un rate limiter
       DEVANT Reviews -- protection absolue du legacy).
     - request coalescing : 1 000 demandes simultanees du meme produit
       = 1 seul appel a Reviews (single-flight).
     - degradation : si Reviews est down et cache vide -> l'ecran
       s'affiche SANS le bloc reviews (partial response documentee).

  3. LATENCE (< 250 ms p95)
     Fan-out PARALLELE depuis le BFF :
       latence = max(Catalog 80, Orders 60, Reviews-cache ~5) + ~20 ms
       de composition ~= 100-120 ms p95. Budget tenu.
     Timeout par source : Catalog 150 ms, Orders 120 ms, Reviews 100 ms.
     Si Reviews timeout : reponse partielle avec "reviews": null +
     header/champ degraded=true -- l'app affiche l'ecran sans le bloc.
     Catalog en echec = la SEULE erreur bloquante (pas d'ecran sans
     produit).

  4. PROBLEME N+1
     Naif : 1 requete liste (20 produits) + 20 appels Reviews.
     Dataloader : pendant le traitement de la requete, les 20 demandes
     get_reviews(product_id) sont COLLECTEES puis emises en UN SEUL
     appel batch (GET /reviews?ids=1,2,...,20) a la fin du tick.
     Resultat : 21 appels -> 2 appels (90% de reduction), et le batch
     est lui-meme cacheable.

  5. PROTECTION DE LA FACADE (si GraphQL retenu un jour, ou pour le BFF)
     Scoring de complexite : cout = somme(cout_champ x multiplicateur
     des listes imbriquees), profondeur max 6, cout max 500 points par
     requete, budget par token partenaire (ex : 10K points/min).
     Exemple : products(first:100) { reviews(first:100) } = 100 x 100 =
     10 000 points -> REJETEE avant execution.
     + persisted queries pour les clients connus (la requete arbitraire
     devient l'exception, pas la regle).

  6. CE QUE LA FACADE AGGRAVE (honnetete)
     1) Debugging : un ecran = 3 backends -> tracing distribue OBLIGATOIRE
        (span par source, request-id propage).
     2) Caching HTTP : les reponses composites se cachent mal ->
        on cache par SOURCE dans le BFF (Redis), pas la reponse entiere.
     3) Couplage de deploiement : le BFF doit suivre les evolutions des
        3 backends -> contrats versionnes + tests de contrat en CI.
     + un composant de plus en prod (astreinte, capacite) : assume.""")


def main():
    easy_1_critique_bad_api()
    easy_2_idempotent_payment_api()
    easy_3_protocol_choice()
    medium_1_reservation_api()
    medium_2_grpc_migration()
    medium_3_api_gateway()
    hard_1_open_banking()
    hard_2_federation_facade()
    print(f"\n{SEPARATOR}")
    print("  Fin des solutions Jour 6.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

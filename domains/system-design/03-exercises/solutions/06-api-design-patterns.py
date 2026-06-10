"""
Solutions -- Day 6 Exercises: API Design & Patterns

Detailed solutions of the 3 Easy exercises.

Usage:
    python 06-api-design-patterns.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# EASY -- Exercise 1 : Critiquing a badly designed API
# =============================================================================


def easy_1_critique_bad_api():
    """Solution: identify the anti-patterns and propose a fix."""
    print(f"\n{SEPARATOR}")
    print("  EASY 1 : Critique of a badly designed API")
    print(SEPARATOR)

    print("""
  Identified problems :

  1. GET /getUser?userId=42
     PROBLEMS :
     - Verb ('get') in the URL when GET is already the HTTP verb.
     - userId as a query param when it is an identified resource.
     - No 404 if the user does not exist : what does it return? 200 with an empty body?
     FIX :
     GET /users/42
     -> 200 {"id": "42", "email": "..."}
     -> 404 {"error": {"code": "user_not_found", ...}}

  2. POST /createUser -> 200 {success: true, id: 42}
     PROBLEMS :
     - 'create' in the URL.
     - 200 OK when it is a creation : must be 201 Created.
     - 'success: true' is useless : if status < 400, it's fine.
     - The response lacks the Location: /users/42 header (RFC).
     FIX :
     POST /users
     Body : {"email": "...", "name": "..."}
     -> 201 Created
     Headers : Location: /users/42
     Body : {"id": "42", "email": "...", "created_at": "..."}
     -> 409 Conflict if the email is already taken.

  3. POST /deleteUser (body: {id: 42}) -> 200
     PROBLEMS :
     - 'delete' in the URL when DELETE exists.
     - POST instead of DELETE prevents HTTP caching and confuses proxies.
     - The id in the body instead of the URL.
     FIX :
     DELETE /users/42
     -> 204 No Content (no body to return after a delete)
     -> 404 if the user does not exist

  4. GET /listUsers?page=1&per_page=20
     PROBLEMS :
     - 'list' in the URL. GET /users is enough.
     - Offset pagination (page/per_page) : O(offset) which degrades, unstable
       in the face of inserts/deletes.
     - No pagination wrapper : just returns [...].
     FIX :
     GET /users?limit=20&cursor=<opaque>
     -> 200 {
          "data": [...],
          "next_cursor": "eyJpZCI6MTIzfQ==",
          "has_more": true
        }

  5. POST /updateUserEmail (body: {id, email})
     PROBLEMS :
     - Verb + specific field in the URL = combinatorial explosion
       (updateUserEmail, updateUserName, updateUserPhone, ...).
     - Should be PATCH /users/42 for a partial update.
     FIX :
     PATCH /users/42
     Body : {"email": "new@example.com"}
     -> 200 {"id": "42", "email": "new@example.com", ...}
     -> 422 Unprocessable Entity if the email is invalid
     -> 409 Conflict if the email is already taken

  6. Error format : 200 OK with success:false
     PROBLEMS :
     - Breaks every HTTP client that relies on the status code :
       * Retries (on 5xx) never trigger.
       * Monitoring (4xx, 5xx rates) shows 0% errors.
       * Load balancers think the backend is healthy.
       * Circuit breakers never trip.
     - 'errorMessage' is free text that can change : impossible
       for the client to build logic on top of it.

     FIX : consistent, standard error format :
     HTTP 4xx or 5xx + body :
     {
       "error": {
         "code": "email_already_exists",        <- STABLE, machine-readable
         "message": "Email is already used",    <- human, can change
         "details": {"email": "bob@example.com"},
         "request_id": "req_abc123"
       }
     }

  Summary :
     - HTTP verb in the URL = forbidden
     - 200 OK for everything = breaks the clients
     - Offset pagination on big datasets = to avoid
     - Standard error format = essential
    """)


# =============================================================================
# EASY -- Exercise 2 : Idempotent payment API
# =============================================================================


def easy_2_idempotent_payment_api():
    """Solution: full design of a POST /payments endpoint."""
    print(f"\n{SEPARATOR}")
    print("  EASY 2 : Idempotent POST /payments")
    print(SEPARATOR)

    print("""
  1. Endpoint contract :

     POST /v1/payments
     Headers :
       Authorization: Bearer <token>
       Content-Type: application/json
       Idempotency-Key: <uuid-v4>         (MANDATORY for POST)
       X-Request-Id: <uuid>                (optional, for tracing)

     Request body :
     {
       "amount_cents": 10000,
       "currency": "EUR",
       "source": "card_xyz",               // Stripe/Ayden card token
       "description": "Premium plan 2026",
       "metadata": {"user_id": "42"}
     }

     Success response (first success) :
     HTTP/1.1 201 Created
     {
       "id": "pay_abc123",
       "status": "succeeded",
       "amount_cents": 10000,
       "currency": "EUR",
       "created_at": "2026-04-11T10:00:00Z"
     }

     Idempotent replay (same key, same body) :
     HTTP/1.1 201 Created       <- same status as the first time
     X-Idempotent-Replay: true
     (same body)

     Errors :
     400 : invalid body (negative amount, unknown currency...)
     401 : missing or invalid auth
     402 : card_declined / insufficient_funds (stable code)
     409 : idempotency_key_reused_with_different_body
     422 : semantic validation (e.g. banned user)
     429 : rate limit
     500/503 : server / downstream error (retry allowed)

  2. Server-side idempotency flow :

     Table 'idempotency_records' (Postgres) :
       key           TEXT PRIMARY KEY,
       request_hash  TEXT NOT NULL,          -- sha256 hash of the body
       response_status INT,
       response_body   JSONB,
       created_at    TIMESTAMPTZ DEFAULT NOW(),
       ttl_expires_at TIMESTAMPTZ

     Recommended TTL : 24 hours.
     - Long enough to cover retries after mobile reconnection
     - Short enough to free space (we don't keep them forever)
     - Stripe also uses 24h

  3. Scenario : timeout + retry 5 min later (same body)

     T0 : client sends POST with Idempotency-Key=K
          server inserts the key (lock), calls Stripe, charge OK
          server records (K, body_hash, 201, response_body) in the DB
          server returns 201 -> network timeout -> client receives nothing

     T0+5min : client retries with the same K and the same body
          server checks : SELECT * FROM idempotency_records WHERE key=K
          Result : finds (K, body_hash, 201, response_body)
          server checks body_hash == sha256(new_body) -> match
          server returns 201 + the stored response_body
          -> Does NOT call Stripe again.
          -> Only one charge was made.

  4. Race condition : 2 parallel requests with the same key

     Without protection : both processes check 'key not found'
     at the same time, then both call Stripe at the same time.
     Double charge !

     Robust solution : UNIQUE constraint + INSERT ON CONFLICT.

     Pseudo-code :
     def process_payment(key, body):
         body_hash = sha256(body)

         # Step 1 : try to insert the key ('in_progress' marker)
         try:
             db.execute('''
                 INSERT INTO idempotency_records (key, request_hash, state)
                 VALUES (%s, %s, 'in_progress')
             ''', (key, body_hash))
         except UniqueViolation:
             # Another process already took the key
             existing = db.fetch('SELECT * FROM idempotency_records WHERE key=%s', (key,))
             if existing['state'] == 'in_progress':
                 # Wait a bit then re-check, or return 409
                 raise APIError('idempotency_key_in_progress', status=409)
             if existing['request_hash'] != body_hash:
                 raise APIError('idempotency_key_reused', status=422)
             return existing['response_status'], existing['response_body']

         # Step 2 : we hold the lock. Call Stripe.
         response = stripe.charge(body)
         # Step 3 : update the record with the final response
         db.execute('''
             UPDATE idempotency_records
             SET state='done', response_status=%s, response_body=%s
             WHERE key=%s
         ''', (201, response, key))
         return 201, response

     Alternative : Redis SETNX with a TTL.
        ok = redis.set(f'idem:{key}', 'in_progress', nx=True, ex=300)
        if not ok:
            return cached_response_or_409()

  5. Retry with the same key but a different body :

     Client side : it's a usage ERROR. The same key must designate
     the same operation. Changing the amount with the same key = client bug.

     Server response :
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

     OR HTTP 409 Conflict (Stripe uses 400 with a specific code).
     Do NOT process the request : prefer failing over risking ambiguity.
    """)


# =============================================================================
# EASY -- Exercise 3 : REST vs gRPC vs GraphQL
# =============================================================================


def easy_3_protocol_choice():
    """Solution: protocol decision matrix for 6 scenarios."""
    print(f"\n{SEPARATOR}")
    print("  EASY 3 : REST, gRPC, or GraphQL ?")
    print(SEPARATOR)

    choices = [
        (
            "Public weather SaaS API, varied clients (Python, mobile, Zapier)",
            "REST",
            "REST is universal : curl, Postman, Zapier, any language. "
            "HTTP caching = CDN friendly (weather requests are cacheable). "
            "No forced codegen for the clients. Swagger/OpenAPI docs "
            "provide a rich experience. gRPC would be hostile to no-code "
            "integrations. GraphQL would be overkill for simple queries."
        ),
        (
            "Internal Python+Go backend, 200 RPS, microservices",
            "gRPC",
            "Backend-to-backend communication = no browser constraints. "
            "gRPC offers : strict typing (protobuf enforces the schemas), "
            "performance (binary, HTTP/2 multiplexing), polyglot codegen "
            "(native Python + Go), and native streaming if needed. At 200 RPS = "
            "JSON serialization would cost 20-30% CPU that we save."
        ),
        (
            "Social network mobile app : feed, profiles, DMs, comments",
            "GraphQL (+ REST for a few endpoints)",
            "Composite screens with heterogeneous data = ideal GraphQL case. "
            "A profile screen needs user + posts + followers_count + "
            "mutual_friends. In REST that would be 4-5 requests (waterfall + "
            "mobile latency). In GraphQL, 1 query that returns exactly "
            "what is displayed. Facebook/Instagram invented it for that."
        ),
        (
            "ML service : structured input -> prediction, < 50 ms p99",
            "gRPC",
            "Critical latency : binary + HTTP/2 are 2-5x faster "
            "than REST/JSON for the same payload. Strict typing avoids "
            "parsing bugs. Called from a Python backend = no browser "
            "constraint. Bonus : gRPC streaming for realtime ML."
        ),
        (
            "Internal React admin dashboard, 20 colleagues, CRUD",
            "REST",
            "20 colleagues = no critical perf. Simple CRUD (users, "
            "products, orders) = no need for GraphQL's flexibility. "
            "REST is the simplest to develop on the backend and consume "
            "on the frontend. OpenAPI + openapi-generator = free TypeScript "
            "typing on the frontend. No need to do more."
        ),
        (
            "Realtime streaming API : stock market, chat, live updates",
            "WebSocket or gRPC streaming (not REST)",
            "REST is unidirectional request/response : each update "
            "requires a new request = waste. The options : "
            "(1) WebSocket for the web (standard, supported by all browsers), "
            "(2) bidirectional gRPC streaming if the client is not a browser, "
            "(3) Server-Sent Events (SSE) for simple unidirectional push. "
            "For a modern chat : WebSocket is the default choice. "
            "Bonus : GraphQL subscriptions run over WebSocket."
        ),
    ]

    for i, (scenario, choice, reason) in enumerate(choices, 1):
        print(f"\n  {i}. {scenario}")
        print(f"     Choice : {choice}")
        print(f"     Reason : {reason}")

    print("\n  Final insight :")
    print("     - You can COMBINE : public REST + internal gRPC is the dominant")
    print("       pattern at big companies (Google, Netflix, Shopify).")
    print("     - The choice depends on the CLIENT as much as the server. If it's a")
    print("       browser, REST/GraphQL. If it's another backend, gRPC.")
    print("     - Streaming requires a bidirectional protocol : WebSocket,")
    print("       gRPC streaming, or SSE depending on the context.")


def main():
    easy_1_critique_bad_api()
    easy_2_idempotent_payment_api()
    easy_3_protocol_choice()
    print(f"\n{SEPARATOR}")
    print("  End of Day 6 solutions.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

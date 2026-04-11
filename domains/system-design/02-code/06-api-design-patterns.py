"""
Jour 6 -- API Design & Patterns
Demonstrations interactives en Python.

Usage:
    python 06-api-design-patterns.py

Simulations :
- Endpoint REST avec idempotency keys (table en memoire)
- Cursor pagination avec encoding base64
- Mini API Gateway qui route/authentifie/rate-limite en amont des handlers
- Error response factory coherente

Le but : MONTRER les patterns sans dependre de FastAPI/Flask ; chaque
pattern est implemente "a la main" pour exposer la logique.
"""

import base64
import hmac
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Error response standard + exception helpers
# =============================================================================


class APIError(Exception):
    """Exception de base qui se traduit en reponse d'erreur structuree.

    WHY une classe dediee ? Pour centraliser la forme de la reponse d'erreur
    (code stable, message human, status HTTP). Les clients codent sur 'code'
    (stable), pas sur 'message' (peut changer).
    """

    def __init__(self, code: str, message: str, status: int = 400, details: dict = None):
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}
        super().__init__(message)

    def to_dict(self, request_id: str) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }


# =============================================================================
# SECTION 2 : Mock database + idempotency store
# =============================================================================


class UserDB:
    """Faux store d'utilisateurs pour la demo.

    WHY ne pas utiliser SQLite ? Pour garder le script runnable sans
    fichier externe et focaliser sur les patterns API, pas sur la DB.
    """

    def __init__(self):
        self.users: dict[str, dict] = {}
        # Ordered list of ids to simulate a keyset/cursor
        self.ids: list[str] = []

    def create(self, user: dict) -> dict:
        uid = user.get("id") or str(uuid.uuid4())
        user["id"] = uid
        user["created_at"] = time.time()
        self.users[uid] = user
        self.ids.append(uid)
        return user

    def get(self, uid: str) -> Optional[dict]:
        return self.users.get(uid)

    def list_after(self, after_id: Optional[str], limit: int) -> list[dict]:
        """Retourne les 'limit' users apres 'after_id' (keyset pagination).

        WHY keyset plutot qu offset ? OFFSET 10000 scannerait 10000 rows
        en DB. WHERE id > :after_id avec un index = O(log N), independant
        de la profondeur de la page.
        """
        if after_id is None:
            start = 0
        else:
            try:
                start = self.ids.index(after_id) + 1
            except ValueError:
                start = 0
        return [self.users[uid] for uid in self.ids[start:start + limit]]


class IdempotencyStore:
    """Store des idempotency keys -> reponse mise en cache.

    WHY ? Les operations critiques (payments, create-order) peuvent etre
    retryees par le client apres un timeout. Sans cette store, on risque
    une double execution. Avec : la 2e (3e, ...) requete avec la meme cle
    renvoie la reponse memorisee SANS re-executer la logique.

    En prod : Redis avec TTL 24h, ou table Postgres avec index unique.
    """

    def __init__(self, ttl_sec: int = 24 * 3600):
        self.store: dict[str, tuple[float, int, dict]] = {}  # key -> (timestamp, status, body)
        self.ttl = ttl_sec

    def get(self, key: str) -> Optional[tuple[int, dict]]:
        entry = self.store.get(key)
        if entry is None:
            return None
        ts, status, body = entry
        if time.time() - ts > self.ttl:
            del self.store[key]
            return None
        return status, body

    def put(self, key: str, status: int, body: dict):
        self.store[key] = (time.time(), status, body)


# =============================================================================
# SECTION 3 : Cursor pagination helpers
# =============================================================================


def encode_cursor(payload: dict) -> str:
    """Encode un cursor opaque en base64.

    WHY opaque ? Le client ne doit pas deviner ni forger des cursors.
    En encodant en base64 (voire signe HMAC pour prod), on garde une
    API OU le serveur est libre de changer sa strategie interne sans
    casser les clients. Le cursor est un token 'give-me-more'.
    """
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode())
        return json.loads(raw)
    except Exception:
        raise APIError("invalid_cursor", "The cursor is malformed", status=400)


# =============================================================================
# SECTION 4 : Handlers REST (logique metier)
# =============================================================================


def handler_create_user(db: UserDB, body: dict) -> tuple[int, dict]:
    """POST /users

    Validation minimaliste pour la demo. En prod, utiliser pydantic /
    OpenAPI validator pour enforcer le schema.
    """
    email = body.get("email")
    if not email or "@" not in email:
        raise APIError("invalid_email", "Email is missing or invalid", status=400,
                       details={"field": "email"})
    user = db.create({"email": email, "name": body.get("name", "")})
    return 201, user


def handler_get_user(db: UserDB, uid: str) -> tuple[int, dict]:
    """GET /users/{id}"""
    user = db.get(uid)
    if not user:
        raise APIError("user_not_found", f"No user with id {uid}", status=404,
                       details={"id": uid})
    return 200, user


def handler_list_users(db: UserDB, query: dict) -> tuple[int, dict]:
    """GET /users?cursor=X&limit=N

    Cursor-based pagination. Retourne data + next_cursor.
    """
    try:
        limit = int(query.get("limit", 5))
    except ValueError:
        raise APIError("invalid_limit", "limit must be an integer", status=400)
    if limit < 1 or limit > 100:
        raise APIError("invalid_limit", "limit must be in [1, 100]", status=400)

    cursor_str = query.get("cursor")
    after_id = None
    if cursor_str:
        cursor = decode_cursor(cursor_str)
        after_id = cursor.get("after_id")

    users = db.list_after(after_id, limit)
    next_cursor = None
    if len(users) == limit:
        # Il y a (peut-etre) encore des donnees : on construit un next cursor
        next_cursor = encode_cursor({"after_id": users[-1]["id"]})

    return 200, {
        "data": users,
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None,
    }


# =============================================================================
# SECTION 5 : API Gateway -- middleware chain
# =============================================================================


@dataclass
class Request:
    """Representation simplifiee d'une requete HTTP."""

    method: str
    path: str
    headers: dict = field(default_factory=dict)
    query: dict = field(default_factory=dict)
    body: dict = field(default_factory=dict)


@dataclass
class Response:
    status: int
    body: dict
    headers: dict = field(default_factory=dict)


# Token bucket simplifie (voir jour 5 pour la version complete)
class SimpleRateLimiter:
    def __init__(self, per_sec: float, capacity: int):
        self.per_sec = per_sec
        self.capacity = capacity
        self.buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def allow(self, key: str) -> bool:
        now = time.time()
        tokens, last = self.buckets.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - last) * self.per_sec)
        if tokens >= 1:
            self.buckets[key] = (tokens - 1, now)
            return True
        self.buckets[key] = (tokens, now)
        return False


class APIGateway:
    """Mini API Gateway qui illustre les responsabilites d'un gateway :
    - Authentification (Bearer token)
    - Rate limiting
    - Routing (method + path -> handler)
    - Idempotency enforcement pour les POST
    - Error handling centralise
    - Generation de request_id

    WHY centraliser ces concerns ? Parce qu'ils sont generiques et
    doivent etre appliques uniformement. Si chaque service les
    reimplemente, on a des divergences et des bugs.
    """

    def __init__(self, db: UserDB, idempo: IdempotencyStore):
        self.db = db
        self.idempo = idempo
        self.rate_limiter = SimpleRateLimiter(per_sec=5, capacity=10)
        # Token de demo en dur. En prod : JWT ou token introspection.
        self.valid_tokens = {"secret-token-abc": "user_42"}
        # Route table : (method, path_pattern) -> handler
        self.routes: list[tuple[str, str, Callable]] = [
            ("POST", "/users", self._route_create_user),
            ("GET", "/users/{id}", self._route_get_user),
            ("GET", "/users", self._route_list_users),
        ]

    def handle(self, req: Request) -> Response:
        """Entry point : orchestre tout le pipeline gateway."""
        request_id = str(uuid.uuid4())[:8]
        try:
            # 1. Auth
            user_id = self._authenticate(req)
            # 2. Rate limiting (par user)
            if not self.rate_limiter.allow(user_id):
                raise APIError("rate_limited", "Too many requests", status=429,
                               details={"retry_after_seconds": 1})
            # 3. Idempotency check (pour les writes)
            if req.method == "POST":
                idem_key = req.headers.get("Idempotency-Key")
                if idem_key:
                    cached = self.idempo.get(idem_key)
                    if cached:
                        status, body = cached
                        return Response(status, body, {"X-Request-Id": request_id,
                                                        "X-Idempotent-Replay": "true"})
            # 4. Route
            handler = self._match_route(req)
            if handler is None:
                raise APIError("not_found", f"No route {req.method} {req.path}", status=404)
            status, body = handler(req)
            # 5. Save idempotent response
            if req.method == "POST":
                idem_key = req.headers.get("Idempotency-Key")
                if idem_key:
                    self.idempo.put(idem_key, status, body)
            return Response(status, body, {"X-Request-Id": request_id})
        except APIError as e:
            return Response(e.status, e.to_dict(request_id), {"X-Request-Id": request_id})
        except Exception as e:
            # Fallback : erreur 500 generique, JAMAIS le stack trace en prod
            err = APIError("internal_error", "An unexpected error occurred", status=500)
            return Response(500, err.to_dict(request_id), {"X-Request-Id": request_id})

    def _authenticate(self, req: Request) -> str:
        """Extrait et valide le Bearer token. Retourne user_id."""
        auth = req.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise APIError("unauthorized", "Missing Bearer token", status=401)
        token = auth[len("Bearer "):]
        user_id = self.valid_tokens.get(token)
        if user_id is None:
            raise APIError("unauthorized", "Invalid token", status=401)
        return user_id

    def _match_route(self, req: Request) -> Optional[Callable]:
        """Matcher simplifie pour des routes avec un seul {param}."""
        for method, pattern, handler in self.routes:
            if method != req.method:
                continue
            if "{" not in pattern:
                if req.path == pattern:
                    return handler
                continue
            # Pattern avec 1 segment variable, type /users/{id}
            p_parts = pattern.split("/")
            r_parts = req.path.split("/")
            if len(p_parts) != len(r_parts):
                continue
            params = {}
            match = True
            for p, r in zip(p_parts, r_parts):
                if p.startswith("{") and p.endswith("}"):
                    params[p[1:-1]] = r
                elif p != r:
                    match = False
                    break
            if match:
                # On bind les params dans req.body pour simplifier
                req.body["_path_params"] = params
                return handler
        return None

    def _route_create_user(self, req: Request) -> tuple[int, dict]:
        return handler_create_user(self.db, req.body)

    def _route_get_user(self, req: Request) -> tuple[int, dict]:
        uid = req.body.get("_path_params", {}).get("id", "")
        return handler_get_user(self.db, uid)

    def _route_list_users(self, req: Request) -> tuple[int, dict]:
        return handler_list_users(self.db, req.query)


# =============================================================================
# SECTION 6 : Demos
# =============================================================================


def demo_happy_path():
    print(f"\n{SEPARATOR}\n  DEMO 1 : Happy path via le gateway\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())

    req = Request(
        method="POST",
        path="/users",
        headers={"Authorization": "Bearer secret-token-abc"},
        body={"email": "alice@example.com", "name": "Alice"},
    )
    res = gw.handle(req)
    print(f"  POST /users : status={res.status}, body={res.body}, headers={res.headers}")

    req = Request(
        method="GET",
        path=f"/users/{res.body['id']}",
        headers={"Authorization": "Bearer secret-token-abc"},
    )
    res = gw.handle(req)
    print(f"  GET /users/{{id}} : status={res.status}, body={res.body}")


def demo_idempotency():
    print(f"\n{SEPARATOR}\n  DEMO 2 : Idempotency key replay\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())

    headers = {"Authorization": "Bearer secret-token-abc", "Idempotency-Key": "req-xyz-001"}
    body = {"email": "bob@example.com"}

    r1 = gw.handle(Request("POST", "/users", headers, body=dict(body)))
    print(f"  1re requete : status={r1.status}, id={r1.body.get('id')}, replay={r1.headers.get('X-Idempotent-Replay')}")

    r2 = gw.handle(Request("POST", "/users", headers, body=dict(body)))
    print(f"  2e requete  : status={r2.status}, id={r2.body.get('id')}, replay={r2.headers.get('X-Idempotent-Replay')}")
    print(f"  Meme id retourne : {r1.body.get('id') == r2.body.get('id')}")
    print(f"  DB contient : {len(db.users)} user(s) -- pas de doublon.")


def demo_error_responses():
    print(f"\n{SEPARATOR}\n  DEMO 3 : Format d'erreur coherent\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())

    # Token manquant -> 401
    r = gw.handle(Request("GET", "/users/123", {}))
    print(f"  Sans token  : status={r.status}, body={r.body}")

    # Token invalide -> 401
    r = gw.handle(Request("GET", "/users/123", {"Authorization": "Bearer wrong"}))
    print(f"  Bad token   : status={r.status}, body={r.body}")

    # User inexistant -> 404
    r = gw.handle(Request("GET", "/users/999", {"Authorization": "Bearer secret-token-abc"}))
    print(f"  404         : status={r.status}, body={r.body}")

    # Email invalide -> 400
    r = gw.handle(Request("POST", "/users", {"Authorization": "Bearer secret-token-abc"},
                          body={"email": "not-an-email"}))
    print(f"  400         : status={r.status}, body={r.body}")


def demo_cursor_pagination():
    print(f"\n{SEPARATOR}\n  DEMO 4 : Cursor pagination\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())
    # Pour cette demo, on relache le rate limiter pour pouvoir creer
    # 12 users sans se faire bloquer (le bucket default est 10).
    gw.rate_limiter = SimpleRateLimiter(per_sec=100, capacity=100)
    headers = {"Authorization": "Bearer secret-token-abc"}
    # Creer 12 users
    for i in range(12):
        gw.handle(Request("POST", "/users", headers, body={"email": f"u{i}@ex.com"}))

    # Premiere page
    r = gw.handle(Request("GET", "/users", headers, query={"limit": "5"}))
    print(f"  Page 1 : {len(r.body['data'])} users, has_more={r.body['has_more']}")
    print(f"  next_cursor = {r.body['next_cursor']}")

    # Page suivante
    r = gw.handle(Request("GET", "/users", headers,
                          query={"limit": "5", "cursor": r.body["next_cursor"]}))
    print(f"  Page 2 : {len(r.body['data'])} users, has_more={r.body['has_more']}")

    r = gw.handle(Request("GET", "/users", headers,
                          query={"limit": "5", "cursor": r.body["next_cursor"]}))
    print(f"  Page 3 : {len(r.body['data'])} users, has_more={r.body['has_more']}")


def demo_rate_limiting():
    print(f"\n{SEPARATOR}\n  DEMO 5 : Rate limiting sur /users\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())
    headers = {"Authorization": "Bearer secret-token-abc"}

    results = []
    for i in range(15):
        r = gw.handle(Request("GET", "/users", headers, query={"limit": "5"}))
        results.append(r.status)
    print(f"  15 requetes rapides : statuses = {results}")
    allowed = sum(1 for s in results if s == 200)
    rejected = sum(1 for s in results if s == 429)
    print(f"  Allowed: {allowed}, 429 Rate Limited: {rejected}")
    print(f"  (capacity 10, refill 5/s -> 10 allowed in a burst then 429)")


def main():
    demo_happy_path()
    demo_idempotency()
    demo_error_responses()
    demo_cursor_pagination()
    demo_rate_limiting()
    print(f"\n{SEPARATOR}\n  Fin des demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()

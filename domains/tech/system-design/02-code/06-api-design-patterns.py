"""
Day 6 -- API Design & Patterns
Interactive demonstrations in Python.

Usage:
    python 06-api-design-patterns.py

Simulations:
- REST endpoint with idempotency keys (in-memory table)
- Cursor pagination with base64 encoding
- Mini API Gateway that routes/authenticates/rate-limits upstream of the handlers
- Consistent error response factory

The goal: SHOW the patterns without depending on FastAPI/Flask; each
pattern is implemented "by hand" to expose the logic.
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
# SECTION 1 : Standard error response + exception helpers
# =============================================================================


class APIError(Exception):
    """Base exception that translates into a structured error response.

    WHY a dedicated class? To centralize the shape of the error response
    (stable code, human message, HTTP status). Clients code against 'code'
    (stable), not 'message' (can change).
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
    """Fake user store for the demo.

    WHY not use SQLite? To keep the script runnable without an
    external file and focus on the API patterns, not on the DB.
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
        """Returns the 'limit' users after 'after_id' (keyset pagination).

        WHY keyset rather than offset? OFFSET 10000 would scan 10000 rows
        in the DB. WHERE id > :after_id with an index = O(log N), independent
        of the page depth.
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
    """Store of idempotency keys -> cached response.

    WHY? Critical operations (payments, create-order) can be
    retried by the client after a timeout. Without this store, we risk
    a double execution. With it: the 2nd (3rd, ...) request with the same key
    returns the memorized response WITHOUT re-executing the logic.

    In prod: Redis with a 24h TTL, or a Postgres table with a unique index.
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
    """Encodes an opaque cursor in base64.

    WHY opaque? The client must neither guess nor forge cursors.
    By encoding in base64 (or even HMAC-signed for prod), we keep an
    API WHERE the server is free to change its internal strategy without
    breaking the clients. The cursor is a 'give-me-more' token.
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
# SECTION 4 : REST handlers (business logic)
# =============================================================================


def handler_create_user(db: UserDB, body: dict) -> tuple[int, dict]:
    """POST /users

    Minimalist validation for the demo. In prod, use pydantic /
    an OpenAPI validator to enforce the schema.
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

    Cursor-based pagination. Returns data + next_cursor.
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
        # There is (maybe) more data: we build a next cursor
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
    """Simplified representation of an HTTP request."""

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


# Simplified token bucket (see day 5 for the full version)
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
    """Mini API Gateway illustrating a gateway's responsibilities:
    - Authentication (Bearer token)
    - Rate limiting
    - Routing (method + path -> handler)
    - Idempotency enforcement for POSTs
    - Centralized error handling
    - request_id generation

    WHY centralize these concerns? Because they are generic and
    must be applied uniformly. If each service reimplements
    them, we get divergences and bugs.
    """

    def __init__(self, db: UserDB, idempo: IdempotencyStore):
        self.db = db
        self.idempo = idempo
        self.rate_limiter = SimpleRateLimiter(per_sec=5, capacity=10)
        # Hardcoded demo token. In prod: JWT or token introspection.
        self.valid_tokens = {"secret-token-abc": "user_42"}
        # Route table : (method, path_pattern) -> handler
        self.routes: list[tuple[str, str, Callable]] = [
            ("POST", "/users", self._route_create_user),
            ("GET", "/users/{id}", self._route_get_user),
            ("GET", "/users", self._route_list_users),
        ]

    def handle(self, req: Request) -> Response:
        """Entry point: orchestrates the whole gateway pipeline."""
        request_id = str(uuid.uuid4())[:8]
        try:
            # 1. Auth
            user_id = self._authenticate(req)
            # 2. Rate limiting (per user)
            if not self.rate_limiter.allow(user_id):
                raise APIError("rate_limited", "Too many requests", status=429,
                               details={"retry_after_seconds": 1})
            # 3. Idempotency check (for writes)
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
            # Fallback: generic 500 error, NEVER the stack trace in prod
            err = APIError("internal_error", "An unexpected error occurred", status=500)
            return Response(500, err.to_dict(request_id), {"X-Request-Id": request_id})

    def _authenticate(self, req: Request) -> str:
        """Extracts and validates the Bearer token. Returns user_id."""
        auth = req.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise APIError("unauthorized", "Missing Bearer token", status=401)
        token = auth[len("Bearer "):]
        user_id = self.valid_tokens.get(token)
        if user_id is None:
            raise APIError("unauthorized", "Invalid token", status=401)
        return user_id

    def _match_route(self, req: Request) -> Optional[Callable]:
        """Simplified matcher for routes with a single {param}."""
        for method, pattern, handler in self.routes:
            if method != req.method:
                continue
            if "{" not in pattern:
                if req.path == pattern:
                    return handler
                continue
            # Pattern with 1 variable segment, like /users/{id}
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
                # We bind the params into req.body to keep things simple
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
    print(f"\n{SEPARATOR}\n  DEMO 1 : Happy path through the gateway\n{SEPARATOR}")
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
    print(f"  1st request : status={r1.status}, id={r1.body.get('id')}, replay={r1.headers.get('X-Idempotent-Replay')}")

    r2 = gw.handle(Request("POST", "/users", headers, body=dict(body)))
    print(f"  2nd request : status={r2.status}, id={r2.body.get('id')}, replay={r2.headers.get('X-Idempotent-Replay')}")
    print(f"  Same id returned : {r1.body.get('id') == r2.body.get('id')}")
    print(f"  DB contains : {len(db.users)} user(s) -- no duplicate.")


def demo_error_responses():
    print(f"\n{SEPARATOR}\n  DEMO 3 : Consistent error format\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())

    # Missing token -> 401
    r = gw.handle(Request("GET", "/users/123", {}))
    print(f"  No token    : status={r.status}, body={r.body}")

    # Invalid token -> 401
    r = gw.handle(Request("GET", "/users/123", {"Authorization": "Bearer wrong"}))
    print(f"  Bad token   : status={r.status}, body={r.body}")

    # Nonexistent user -> 404
    r = gw.handle(Request("GET", "/users/999", {"Authorization": "Bearer secret-token-abc"}))
    print(f"  404         : status={r.status}, body={r.body}")

    # Invalid email -> 400
    r = gw.handle(Request("POST", "/users", {"Authorization": "Bearer secret-token-abc"},
                          body={"email": "not-an-email"}))
    print(f"  400         : status={r.status}, body={r.body}")


def demo_cursor_pagination():
    print(f"\n{SEPARATOR}\n  DEMO 4 : Cursor pagination\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())
    # For this demo, we loosen the rate limiter so we can create
    # 12 users without getting blocked (the default bucket is 10).
    gw.rate_limiter = SimpleRateLimiter(per_sec=100, capacity=100)
    headers = {"Authorization": "Bearer secret-token-abc"}
    # Create 12 users
    for i in range(12):
        gw.handle(Request("POST", "/users", headers, body={"email": f"u{i}@ex.com"}))

    # First page
    r = gw.handle(Request("GET", "/users", headers, query={"limit": "5"}))
    print(f"  Page 1 : {len(r.body['data'])} users, has_more={r.body['has_more']}")
    print(f"  next_cursor = {r.body['next_cursor']}")

    # Next page
    r = gw.handle(Request("GET", "/users", headers,
                          query={"limit": "5", "cursor": r.body["next_cursor"]}))
    print(f"  Page 2 : {len(r.body['data'])} users, has_more={r.body['has_more']}")

    r = gw.handle(Request("GET", "/users", headers,
                          query={"limit": "5", "cursor": r.body["next_cursor"]}))
    print(f"  Page 3 : {len(r.body['data'])} users, has_more={r.body['has_more']}")


def demo_rate_limiting():
    print(f"\n{SEPARATOR}\n  DEMO 5 : Rate limiting on /users\n{SEPARATOR}")
    db = UserDB()
    gw = APIGateway(db, IdempotencyStore())
    headers = {"Authorization": "Bearer secret-token-abc"}

    results = []
    for i in range(15):
        r = gw.handle(Request("GET", "/users", headers, query={"limit": "5"}))
        results.append(r.status)
    print(f"  15 rapid requests : statuses = {results}")
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
    print(f"\n{SEPARATOR}\n  End of demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()

"""
Solutions -- Day 6 MEDIUM Exercises: API Design & Patterns

Worked solutions. Exercise 1 implements a tiny idempotency-key store to
DEMONSTRATE the dedup behaviour (with assertions); exercises 2-3 are structured
reasoning with small checkable facts.

Usage:
    python3 06-api-design-patterns-medium.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# MEDIUM -- Exercise 1 : Idempotent payment API
# =============================================================================

class IdempotencyStore:
    """Minimal server-side idempotency store (the Stripe pattern)."""

    def __init__(self):
        # key -> {"state", "request_hash", "response"}
        self._store = {}
        self.charges_executed = 0          # counts real (non-replayed) charges

    def charge(self, idem_key, amount):
        """Return (status_code, body). Re-execution is prevented by the store."""
        request_hash = hash(("charge", amount))
        existing = self._store.get(idem_key)

        if existing is not None:
            # Same key, different body -> reject (mismatch).
            if existing["request_hash"] != request_hash:
                return 422, {"error": "idempotency_key_reuse_with_different_body"}
            if existing["state"] == "in_progress":
                # Concurrent duplicate arriving before the first finished.
                return 409, {"error": "request_in_progress"}
            # Replay of a completed request -> return the stored response.
            return existing["response"]

        # First time we see this key: mark in_progress (the atomic insert),
        # then execute exactly once.
        self._store[idem_key] = {
            "state": "in_progress",
            "request_hash": request_hash,
            "response": None,
        }
        self.charges_executed += 1                     # the ONE real charge
        response = (201, {"id": "ch_1", "amount": amount, "status": "succeeded"})
        self._store[idem_key].update(state="done", response=response)
        return response


def medium_1_idempotent_payment():
    """Idempotent POST /v1/charges."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : Idempotent payment API")
    print(SEPARATOR)

    print("\n  1. Idempotency-key flow :")
    print("     receive key -> look it up -> if found & same body : return the STORED")
    print("     response (no re-charge) ; if not found : execute once, store key +")
    print("     response. TTL >= 24h so a client can safely retry after reconnecting.")

    print("\n  2. Storage schema :")
    print("     idempotency_keys(key PK/UNIQUE, request_hash, state, response_status,")
    print("                      response_body, created_at) ; UNIQUE on key.")

    # 3. Demonstrate: same key twice (replay) charges only once.
    store = IdempotencyStore()
    r1 = store.charge("idem_abc", 10000)
    r2 = store.charge("idem_abc", 10000)               # replay
    print("\n  3. Same key twice (replay / concurrent) :")
    print(f"     first  -> {r1}")
    print(f"     replay -> {r2}  (same response, NO second charge)")
    print(f"     charges actually executed : {store.charges_executed}")
    print("     Concurrency : the in_progress state (atomic insert / row lock /")
    print("     INSERT ... ON CONFLICT) blocks a parallel duplicate from charging.")
    assert store.charges_executed == 1                 # billed exactly once
    assert r1 == r2

    # 4. Same key, different body -> 422.
    r3 = store.charge("idem_abc", 99999)
    print("\n  4. Same key, DIFFERENT body :")
    print(f"     -> {r3}  (reject : the key is bound to the original request)")
    assert r3[0] == 422
    assert store.charges_executed == 1                 # still no extra charge

    print("\n  5. Status codes :")
    codes = [
        ("Success (new charge)", "201 Created (or 200)"),
        ("Card declined", "402 Payment Required"),
        ("Idempotent replay (same key+body)", "200/201 with stored response"),
        ("Key reused with different body", "422 Unprocessable Entity (or 409)"),
        ("Rate limit", "429 Too Many Requests"),
    ]
    for case, code in codes:
        print(f"     {case:<38} -> {code}")

    print("\n  6. Why 200 + success:false breaks clients :")
    print("     Clients, retries, circuit breakers and monitoring all key off the")
    print("     HTTP status. A 200 that actually failed -> retries never trigger,")
    print("     breakers never open, dashboards stay green while users are failing.")


# =============================================================================
# MEDIUM -- Exercise 2 : REST vs gRPC vs GraphQL
# =============================================================================

def medium_2_paradigm_choice():
    """Choose the API paradigm per surface."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : REST vs gRPC vs GraphQL")
    print(SEPARATOR)

    print("\n  1. Paradigm per surface :")
    choices = [
        ("Public third-party API", "REST",
         "Universal HTTP, cacheable (GET), readable, SDKs + OpenAPI codegen."),
        ("Internal microservice-to-microservice", "gRPC",
         "Protobuf strong typing, compact binary, HTTP/2 multiplexing, streaming."),
        ("Mobile app (heterogeneous screens)", "GraphQL",
         "No over/under-fetching, one round-trip for varied screens, schema typed."),
    ]
    for surface, choice, why in choices:
        print(f"     {surface:<42} -> {choice}")
        print(f"        {why}")

    print("\n  2. N+1 query + DataLoader (mobile/GraphQL) :")
    print("     Resolving 'users { orders }' naively runs 1 query for users + 1 per")
    print("     user for their orders = N+1. DataLoader BATCHES the per-user fetches")
    print("     into a single 'WHERE user_id IN (...)' and caches within the request.")

    print("\n  3. Greedy queries :")
    print("     A malicious deeply-nested query can explode cost. Defend with a depth")
    print("     limit, a complexity score budget per query, and per-query timeouts.")

    print("\n  4. gRPC internal : pros + caching trap :")
    print("     Pros : protobuf contract (compat-aware), compact, HTTP/2 multiplexing.")
    print("     Trap : binary HTTP/2 is NOT cacheable by classic HTTP proxies/CDN.")

    print("\n  5. Additive vs breaking on the public REST API :")
    print("     Additive (no new version) : add a new optional field ; add a new endpoint.")
    print("     Breaking (new version)   : rename a field ; remove a field ; change a")
    print("                                 field's type ; tighten validation.")


# =============================================================================
# MEDIUM -- Exercise 3 : Pagination + versioning
# =============================================================================

def medium_3_pagination_versioning():
    """Pagination strategy + versioning policy."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Pagination + versioning")
    print(SEPARATOR)

    print("\n  1. Pagination per endpoint :")
    print("     /v1/admin/users (~50K, 'jump to page 42') -> OFFSET pagination.")
    print("       Small dataset, internal, page jumps needed -> offset is fine.")
    print("     /v1/events (billions, constant inserts) -> CURSOR pagination.")
    print("       Huge, insert-heavy -> cursor is O(1) and stable.")

    print("\n  2. Offset degradation vs cursor O(1) :")
    print("     OFFSET 1000000 forces the DB to scan and discard 1M rows -> O(offset).")
    print("     Cursor : SELECT ... WHERE id > :cursor ORDER BY id LIMIT n -> uses the")
    print("     index, reads only n rows -> O(1) per page regardless of depth.")

    print("\n  3. Opaque cursor + insert/delete behaviour :")
    print("     The cursor is an opaque blob (base64 of the last position, e.g. id/ts).")
    print("     Offset : an insert/delete between pages shifts rows -> duplicates or")
    print("     skipped items. Cursor : anchored on a stable key -> no dup/skip.")

    print("\n  4. Breaking change on /v1/events (rename a field) :")
    print("     Ship /v2/events with the new name ; keep /v1 running ; announce a")
    print("     6-12 month deprecation ; log v1 usage to find lagging clients ;")
    print("     email/changelog the migration ; only then sunset v1.")

    print("\n  5. Sunset header :")
    print("     Sunset: <date> tells clients when a version is cut. Give 6-12 months")
    print("     of notice for a public API (clients you don't control move slowly).")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 6 MEDIUM : API DESIGN & PATTERNS")
    print("=" * 60)
    medium_1_idempotent_payment()
    medium_2_paradigm_choice()
    medium_3_pagination_versioning()
    print(f"\n{'=' * 60}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
Solutions -- Day 6 HARD Exercises: API Design & Patterns

Worked solutions. The platform sizing and the rate-limit / idempotency facts are
made concrete with small runnable demos (a tiered rate limiter and a multi-region
idempotency store) plus assertions on the key numbers; the design and post-mortem
reasoning is printed and pinned by checkable facts.

Usage:
    python3 06-api-design-patterns-hard.py
"""

SEPARATOR = "=" * 60


# =============================================================================
# HARD -- Exercise 1 : API + Gateway of a multi-tenant platform
# =============================================================================

# Tier -> requests/second quota. A shared store (Redis) holds this config so the
# gateway looks up the tenant's tier and enforces a DIFFERENT limit per tenant.
TENANT_TIERS = {
    "free": 10,
    "pro": 1_000,
    "enterprise": 50_000,
}


class TieredRateLimiter:
    """
    Token-bucket-ish per-tenant limiter keyed by (tenant, tier).

    WHY a shared store in real life: with 100K req/s across many gateway nodes,
    the counter must be GLOBAL (Redis), otherwise each node enforces its own
    fraction of the quota. Here we model the per-tenant counter in-memory to
    demonstrate the tier differentiation, not the distributed counting.
    """

    def __init__(self):
        self._used = {}                    # tenant_id -> requests consumed this window

    def allow(self, tenant_id, tier):
        limit = TENANT_TIERS[tier]         # tier -> quota (config-driven)
        used = self._used.get(tenant_id, 0)
        if used >= limit:
            return False                   # 429 Too Many Requests
        self._used[tenant_id] = used + 1
        return True


def hard_1_multi_tenant_api():
    """Design + size the API/gateway of a B2B SaaS (Stripe/Twilio class)."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Multi-tenant API + Gateway")
    print(SEPARATOR)

    print("\n  1. Layers (request path) :")
    print("     client -> CDN/LB -> API Gateway -> BFF -> internal gRPC services")
    print("     The gateway does INFRA ONLY : authN, rate limit, routing, TLS,")
    print("     observability. It must NOT hold business logic (the 'gateway")
    print("     monolith' anti-pattern : every change re-touches the gateway and")
    print("     it becomes a deploy bottleneck + single point of coupling).")

    print("\n  2. Auth & multi-tenant :")
    print("     Tenant authenticates with an API key (server-to-server) or OAuth2")
    print("     client-credentials -> the gateway validates it ONCE and injects")
    print("     tenant_id + scopes into internal headers (or a short-lived signed")
    print("     JWT). Internal services TRUST these headers (zero re-auth).")
    print("     Per-tenant rate limit = config (tier -> quota) in a shared Redis :")
    limiter = TieredRateLimiter()
    # free tenant: 10 allowed, 11th blocked.
    allowed_free = sum(limiter.allow("t_free", "free") for _ in range(15))
    # enterprise tenant: far higher ceiling.
    allowed_ent = sum(limiter.allow("t_ent", "enterprise") for _ in range(100))
    print(f"       free tenant   : {allowed_free}/15 allowed (quota {TENANT_TIERS['free']})")
    print(f"       enterprise    : {allowed_ent}/100 allowed (quota {TENANT_TIERS['enterprise']})")
    assert allowed_free == TENANT_TIERS["free"]       # free capped at its quota
    assert allowed_ent == 100                          # enterprise nowhere near its cap
    assert TENANT_TIERS["enterprise"] > TENANT_TIERS["free"] * 100  # tiers truly differ

    print("\n  3. Idempotency at scale (multi-region) :")
    store = MultiRegionIdemStore()
    # Same idempotency key retried in a DIFFERENT region must not double-charge.
    r1 = store.charge("eu-west", "idem-42", amount=100)
    r2 = store.charge("us-east", "idem-42", amount=100)   # retry lands elsewhere
    print(f"       1st call (eu-west) : {r1}")
    print(f"       retry  (us-east)   : {r2}  <- replayed, NOT a 2nd charge")
    assert store.charges_executed == 1                 # executed exactly once
    assert r1 == r2                                     # identical response replayed
    print("     The defi : the key must be globally visible/consistent. Options :")
    print("     (a) globally-replicated idem store (strong/quorum -> cross-region")
    print("         round-trip cost), or (b) route a given idem key STICKILY to one")
    print("         region (consistent hashing) so the retry finds the original.")

    print("\n  4. Versioning & evolution :")
    print("     URI versioning (/v1/, /v2/) routed by the gateway. Thousands of")
    print("     UNCONTROLLED clients -> never break in place. Migration is forced")
    print("     gradually : announce -> Deprecation + Sunset headers -> 6-12 month")
    print("     notice -> finally 410 Gone. A tenant pinned to v1 keeps hitting v1.")
    sunset_months = 9
    assert 6 <= sunset_months <= 12

    print("\n  5. BFF (Backend For Frontend) :")
    print("     One BFF per client because their needs DIFFER : mobile is bandwidth/")
    print("     latency constrained, so the mobile BFF aggregates (e.g. home screen =")
    print("     profile + last orders + recommendations in ONE call) and trims fields;")
    print("     the web BFF can make several finer calls. The BFF aggregates/shapes,")
    print("     it holds NO business logic.")

    print("\n  6. Internal protocol :")
    print("     REST (external, web-friendly) -> gRPC (internal, typed, fast) is")
    print("     translated at the gateway/BFF edge. Benefit : strong typing + perf")
    print("     between services. Cost : a REST<->gRPC mapping layer to maintain.")

    print("\n  7. Resilience & contracts :")
    print("     Contract testing (consumer-driven + protobuf schema compat checks)")
    print("     ensures an internal service can't silently break its consumers.")
    print("     OpenAPI, 3 concrete uses : (1) human/SDK documentation, (2) client")
    print("     codegen, (3) request/response validation + contract tests in CI.")


# Reused: a tiny multi-region idempotency store (replicated logical view).
class MultiRegionIdemStore:
    """
    Models a GLOBALLY-shared idempotency table : whichever region sees the key
    first executes once; any region that sees it again replays the stored
    response. (Real systems back this by a replicated/quorum store.)
    """

    def __init__(self):
        self._global = {}                  # idem_key -> stored response
        self.charges_executed = 0

    def charge(self, region, idem_key, amount):
        if idem_key in self._global:
            return self._global[idem_key]  # replay -> no second side effect
        self.charges_executed += 1         # the ONE real charge
        resp = (201, {"id": "ch_1", "amount": amount, "status": "succeeded"})
        self._global[idem_key] = resp
        return resp


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- the breaking change that broke 4000 clients
# =============================================================================

def is_breaking_change(old_value, new_value):
    """
    A change is BREAKING when an existing consumer that worked before can stop
    working. Heuristic used here : same field, CHANGED TYPE -> breaking.
    (Adding a brand-new field is additive/safe because clients ignore unknowns.)
    """
    return type(old_value) is not type(new_value)


def hard_2_breaking_change_postmortem():
    """Post-mortem: type change of an existing field shipped straight to /v1/."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : Breaking-change post-mortem (4000 integrations)")
    print(SEPARATOR)

    print("\n  1. Root cause analysis :")
    print("     The team changed order.status from int 1 to string \"paid\" IN PLACE")
    print("     in /v1/. Internal tests passed because they don't run third-party")
    print("     client code -> they never see the broken `status === 1` checks.")
    # The crux fact: changing the TYPE of an existing field is breaking.
    assert is_breaking_change(1, "paid") is True
    assert is_breaking_change({"a": 1}, {"a": 1, "b": 2}) is False  # adding a key = additive
    chain = [
        ("PROCESS", "no impact review / no consumer sign-off before shipping",
         "Missing guardrail : breaking-change review + change classification"),
        ("ARCHITECTURE", "shipped in place in /v1/ (no new version, not additive)",
         "Missing guardrail : versioning + additive-only within a version"),
        ("MONITORING", "no contract testing / schema diff in CI",
         "Missing guardrail : consumer-driven contracts + schema diff gate"),
    ]
    for cat, cause, guardrail in chain:
        print(f"     [{cat}] {cause}")
        print(f"       -> {guardrail}")
    print("     INVISIBLE provider-side : the response is still 200 OK with a valid")
    print("     body; the failure is in the CLIENT's parsing/comparison. So 5xx and")
    print("     latency stay green -- nothing in the provider dashboards moves.")

    print("\n  2. Additive vs breaking :")
    print("     Rule : adding a NEW field is safe (clients ignore unknown fields);")
    print("     renaming / removing / changing the TYPE of an existing field is")
    print("     breaking. The non-breaking fix here : keep status:int AND add a new")
    print("     status_label:\"paid\" field -> purely additive, nobody breaks.")

    print("\n  3. Why the rollback created NEW victims :")
    print("     Between 11:00 and 12:30 some clients adapted to the string format.")
    print("     The change wasn't backward-compatible in BOTH directions, so")
    print("     reverting int->string broke the freshly-adapted clients (double-bind).")
    print("     Lesson : additive-only changes are reversible without new victims.")

    print("\n  4. Detection (what would have caught it) :")
    detectors = [
        "Contract testing / consumer-driven contracts (Pact)",
        "Schema diff in CI (flag a type change as breaking -> block merge)",
        "Canary on a small subset of clients before full rollout",
        "OpenAPI validation comparing response shape to the published spec",
    ]
    for d in detectors:
        print(f"     - {d}")

    print("\n  5. Corrected evolution policy :")
    policy = [
        "Versioning (/v1/, /v2/) + route by version",
        "Additive-only WITHIN a version (never change/remove an existing field)",
        "Schema validation + breaking-change diff gate in CI",
        "Deprecation with Deprecation/Sunset headers, 6-12 month notice",
        "Communication : changelog, email, dashboards of who still uses the old field",
    ]
    for p in policy:
        print(f"     - {p}")
    print("     Real breaking change later, step by step over 6-12 months :")
    steps_v2 = [
        "Ship the new shape as /v2/ (v1 untouched)",
        "Mark v1 deprecated (Deprecation + Sunset headers, announce the date)",
        "Provide a migration guide + dual-write/dual-read window",
        "Track per-tenant v1 usage; nudge laggards individually",
        "At sunset, return 410 Gone on v1",
    ]
    for i, s in enumerate(steps_v2, 1):
        print(f"       {i}. {s}")

    print("\n  6. Runbook (7 steps) -- breaking change shipped by mistake :")
    runbook = [
        "Assess the double-bind : who broke on the new format vs who already adapted",
        "DON'T blind-rollback : a naive revert makes new victims",
        "If feasible, serve BOTH formats temporarily (additive bridge field)",
        "Communicate immediately to all integrators (status page + targeted email)",
        "Add the missing CI gate (schema diff) so it can't recur",
        "Plan the real change as a proper /v2/ with deprecation window",
        "Post-mortem in 24h : timeline, missing guardrails, action items",
    ]
    for i, s in enumerate(runbook, 1):
        print(f"     {i}. {s}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 6 HARD : API DESIGN & PATTERNS")
    print("=" * 60)
    hard_1_multi_tenant_api()
    hard_2_breaking_change_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

"""
Day 24 -- Inference engineering for agents: structured outputs, routing, caching.

Demonstrates:
  1. ConstrainedDecoder -- token-masking toy + validate/re-prompt fallback that
                           guarantees schema-valid JSON tool calls
  2. ModelRouter        -- route each request to a weak (cheap) or strong
                           (expensive) model by complexity, with a cost counter
  3. PromptCache        -- prefix cache (hash of the static prefix) measuring hit
                           rate and tokens/cost saved

All stdlib, no API key required. The "models" are deterministic mocks so the
focus stays on the inference-engineering machinery, not on any LLM.

Run:
    python domains/tech/agentic-ai/02-code/24-inference-engineering.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ===========================================================================
# 1. CONSTRAINED DECODING / STRUCTURED OUTPUTS
# ===========================================================================
#
# Two complementary techniques are shown:
#   (a) token masking -- the *server-side* ideal: at each step we only allow
#       tokens that keep the output syntactically valid. We model this on a
#       tiny token vocabulary to make the mechanism visible.
#   (b) validate + re-prompt -- the *API-side* fallback used when you cannot
#       touch the decoder (remote model). We validate against a JSON schema
#       and feed the error back until it parses.

class TokenMaskedEnumDecoder:
    """Toy decoder that can ONLY emit one of a fixed set of allowed values.

    This mirrors constrained decoding for an `enum` field: invalid tokens are
    masked (their probability is forced to zero) so the model physically
    cannot produce an out-of-grammar value.
    """

    def __init__(self, allowed: list[str]) -> None:
        self.allowed = allowed

    def decode(self, raw_logits: dict[str, float]) -> str:
        # Mask everything not in the allowed set by dropping it entirely;
        # then pick the highest-scoring *allowed* token. Because masking
        # happens BEFORE the argmax, an invalid token can never win.
        masked = {tok: score for tok, score in raw_logits.items()
                  if tok in self.allowed}
        if not masked:
            # If the model assigned no mass to any valid token, we still
            # return a valid one (the grammar is the source of truth).
            return self.allowed[0]
        return max(masked, key=masked.get)


def validate_json(raw: str, schema: dict) -> tuple[bool, dict | None, str]:
    """Minimal JSON-schema validator (subset: type/required/enum/additionalProperties).

    Returns (ok, parsed, error_message).
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, None, f"invalid JSON: {exc}"

    if schema.get("type") == "object" and not isinstance(parsed, dict):
        return False, None, "expected a JSON object"

    props: dict = schema.get("properties", {})
    required: list = schema.get("required", [])

    for key in required:
        if key not in parsed:
            return False, None, f"missing required field '{key}'"

    if schema.get("additionalProperties") is False:
        for key in parsed:
            if key not in props:
                return False, None, f"unexpected field '{key}'"

    for key, spec in props.items():
        if key not in parsed:
            continue
        value = parsed[key]
        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(value, str):
            return False, None, f"field '{key}' must be a string"
        # bool is a subclass of int in Python: reject True/False for integer fields
        # so a "schema-valid" tool call cannot smuggle a boolean into an int slot.
        if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return False, None, f"field '{key}' must be an integer"
        if "enum" in spec and value not in spec["enum"]:
            return False, None, f"field '{key}'={value!r} not in enum {spec['enum']}"
        if expected_type == "integer" and isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                return False, None, f"field '{key}' below minimum"
            if "maximum" in spec and value > spec["maximum"]:
                return False, None, f"field '{key}' above maximum"

    return True, parsed, ""


class ConstrainedDecoder:
    """Validate + re-prompt loop guaranteeing schema-valid output.

    `flaky_generator(prompt, attempt)` simulates a remote LLM: it may produce
    malformed JSON on early attempts and fix itself once the error is fed back
    (exactly what a good re-prompt achieves in practice).
    """

    def __init__(self, schema: dict, max_retries: int = 3) -> None:
        self.schema = schema
        self.max_retries = max_retries
        self.attempts_used = 0

    def generate(self, generator: Callable[[str, int], str], prompt: str) -> dict:
        self.attempts_used = 0
        current = prompt
        last_err = ""
        for attempt in range(self.max_retries + 1):
            self.attempts_used += 1
            raw = generator(current, attempt)
            ok, parsed, err = validate_json(raw, self.schema)
            if ok:
                return parsed
            last_err = err
            # Re-prompt: append the precise error so the model can self-correct.
            current = (f"{prompt}\n\nERROR: {err}\n"
                       f"Return ONLY valid JSON matching the schema.")
        raise ValueError(f"max retries exceeded; last error: {last_err}")


# ===========================================================================
# 2. MODEL ROUTING
# ===========================================================================

@dataclass
class ModelSpec:
    """A model the router can dispatch to."""
    name: str
    tier: str                 # "weak" or "strong"
    cost_in: float            # $ per 1M input tokens
    cost_out: float           # $ per 1M output tokens


COMPLEX_KEYWORDS = (
    "analyse", "analyze", "compare", "explique", "explain", "design",
    "architecture", "debug", "prove", "demontre", "reasoning", "plan",
)


@dataclass
class RouteDecision:
    model: ModelSpec
    complexity: float
    reason: str


class ModelRouter:
    """Heuristic router (a transparent stand-in for a trained RouteLLM router).

    A complexity score is computed from length + complexity keywords; requests
    above `threshold` go to the strong model, the rest to the weak one.
    """

    def __init__(self, weak: ModelSpec, strong: ModelSpec, threshold: float = 1.0) -> None:
        self.weak = weak
        self.strong = strong
        self.threshold = threshold
        self.total_cost = 0.0
        self.routed: dict[str, int] = {"weak": 0, "strong": 0}

    def complexity(self, query: str) -> float:
        length_signal = len(query.split()) / 100.0
        keyword_signal = sum(1 for kw in COMPLEX_KEYWORDS if kw in query.lower())
        return length_signal + keyword_signal

    def route(self, query: str) -> RouteDecision:
        c = self.complexity(query)
        if c >= self.threshold:
            return RouteDecision(self.strong, c, "complex -> strong")
        return RouteDecision(self.weak, c, "simple -> weak")

    def call(self, query: str, est_in_tokens: int, est_out_tokens: int) -> RouteDecision:
        decision = self.route(query)
        cost = (est_in_tokens / 1e6) * decision.model.cost_in + \
               (est_out_tokens / 1e6) * decision.model.cost_out
        self.total_cost += cost
        self.routed[decision.model.tier] += 1
        return decision


def all_strong_cost(queries: list[tuple[str, int, int]], strong: ModelSpec) -> float:
    """Baseline cost if every request used the strong model."""
    total = 0.0
    for _q, tin, tout in queries:
        total += (tin / 1e6) * strong.cost_in + (tout / 1e6) * strong.cost_out
    return total


# ===========================================================================
# 3. PROMPT CACHING (prefix cache)
# ===========================================================================

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    tokens_billed_full: int = 0     # what we would pay without cache
    tokens_billed_cached: int = 0   # what we actually pay with cache

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_billed_full - self.tokens_billed_cached


class PromptCache:
    """Prefix cache modeled on Anthropic-style cache_control.

    The long static prefix (system prompt + tool schemas) is hashed once; on a
    cache hit, cached input tokens are billed at `read_discount` of full price.
    """

    def __init__(self, read_discount: float = 0.10) -> None:
        self.read_discount = read_discount      # -90% => pay 10%
        self._seen: set[str] = set()
        self.stats = CacheStats()

    @staticmethod
    def _key(prefix: str) -> str:
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def call(self, prefix: str, prefix_tokens: int, suffix_tokens: int) -> bool:
        """Account one call. Returns True on cache hit (prefix already seen)."""
        key = self._key(prefix)
        # Full price: prefix + suffix always billed at 100% without cache.
        self.stats.tokens_billed_full += prefix_tokens + suffix_tokens
        if key in self._seen:
            # Hit: prefix billed at read_discount, suffix always full.
            self.stats.hits += 1
            self.stats.tokens_billed_cached += int(prefix_tokens * self.read_discount) + suffix_tokens
            return True
        # Miss: cache write, prefix billed at full price (here we ignore the
        # +25% write surcharge for clarity), suffix full.
        self.stats.misses += 1
        self._seen.add(key)
        self.stats.tokens_billed_cached += prefix_tokens + suffix_tokens
        return False


# ===========================================================================
# DEMO
# ===========================================================================

def _demo_constrained() -> None:
    print("=" * 60)
    print("1. CONSTRAINED DECODING")
    print("=" * 60)

    # (a) token masking on an enum field
    decoder = TokenMaskedEnumDecoder(allowed=["active", "idle", "error"])
    # The model "wants" to emit "running" (highest raw score) -- but it is masked.
    raw_logits = {"running": 9.5, "active": 4.0, "idle": 2.0, "banana": 8.0}
    chosen = decoder.decode(raw_logits)
    print(f"raw argmax would be 'running' (9.5); masked decode -> {chosen!r}")
    assert chosen in decoder.allowed

    # (b) validate + re-prompt fallback
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "cancel", "stop"]},
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["action", "query"],
        "additionalProperties": False,
    }

    def flaky_generator(prompt: str, attempt: int) -> str:
        # Attempt 0: malformed (extra field). Attempt 1+: corrected.
        if attempt == 0:
            return '{"action": "search", "query": "trucks", "oops": 1}'
        return '{"action": "search", "query": "trucks", "limit": 10}'

    cd = ConstrainedDecoder(schema, max_retries=3)
    result = cd.generate(flaky_generator, "List available trucks")
    print(f"re-prompt loop produced valid JSON in {cd.attempts_used} attempts: {result}")
    assert result["action"] == "search"


def _demo_routing() -> None:
    print("\n" + "=" * 60)
    print("2. MODEL ROUTING")
    print("=" * 60)

    weak = ModelSpec("gpt-weak-mini", "weak", cost_in=0.40, cost_out=1.60)
    strong = ModelSpec("gpt-strong", "strong", cost_in=2.00, cost_out=8.00)
    router = ModelRouter(weak, strong, threshold=1.0)

    rng = random.Random(7)
    simple = ["Format this date", "Classify sentiment: great!", "Extract the order id",
              "Translate hello", "Uppercase this string", "Is this spam? yes/no"]
    complex_ = ["Analyse the trade-offs and design an architecture for ...",
                "Explain step by step and prove the invariant holds in ...",
                "Compare these three approaches and recommend with reasoning ..."]
    queries: list[tuple[str, int, int]] = []
    for _ in range(40):
        q = rng.choice(simple)
        queries.append((q, 200, 80))
    for _ in range(10):
        q = rng.choice(complex_)
        queries.append((q, 1200, 600))
    rng.shuffle(queries)

    for q, tin, tout in queries:
        router.call(q, tin, tout)

    baseline = all_strong_cost(queries, strong)
    saved_pct = (1 - router.total_cost / baseline) * 100
    print(f"requests: {len(queries)} (routed weak={router.routed['weak']}, "
          f"strong={router.routed['strong']})")
    print(f"all-strong cost : ${baseline:.4f}")
    print(f"routed cost     : ${router.total_cost:.4f}")
    print(f"savings         : {saved_pct:.1f}%")
    assert router.total_cost < baseline


def _demo_caching() -> None:
    print("\n" + "=" * 60)
    print("3. PROMPT CACHING")
    print("=" * 60)

    cache = PromptCache(read_discount=0.10)   # Anthropic-style -90%
    system_prefix = "SYSTEM PROMPT + TOOL SCHEMAS " * 400   # ~the static context
    prefix_tokens = 10_000
    # 1 cold call (miss) + 99 warm calls (hits) on the same prefix.
    for i in range(100):
        cache.call(system_prefix, prefix_tokens, suffix_tokens=120)

    s = cache.stats
    print(f"calls: {s.hits + s.misses} (hits={s.hits}, misses={s.misses}, "
          f"hit_rate={s.hit_rate:.0%})")
    print(f"tokens billed without cache : {s.tokens_billed_full:,}")
    print(f"tokens billed with cache    : {s.tokens_billed_cached:,}")
    print(f"tokens saved                : {s.tokens_saved:,} "
          f"({s.tokens_saved / s.tokens_billed_full:.0%})")
    assert s.tokens_saved > 0


if __name__ == "__main__":
    _demo_constrained()
    _demo_routing()
    _demo_caching()
    print("\nDemo complete: structured outputs + routing + caching combined.")

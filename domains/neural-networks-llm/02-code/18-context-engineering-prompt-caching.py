"""
Jour 18 — Context engineering & prompt caching : simulateur
=============================================================
Pure Python. Simule un cache de prefixe type "Anthropic cache_control"
pour comprendre comment le hit rate evolue et calculer l'economie reelle
sur un trafic realiste.

Contenu :
  1. Un cache de prefixe (trie) qui stocke les "K/V" hashes
  2. Facturation differenciee cache hit / cache write / miss
  3. Simulation de trafic avec 3 strategies de construction du contexte
  4. Comparaison des couts, latences et hit rates

Run : python 02-code/18-context-engineering-prompt-caching.py
"""

from __future__ import annotations
import sys, io, random, hashlib, time
from dataclasses import dataclass, field
from collections import defaultdict

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(2026)

# ============================================================================
# 1) Cache de prefixe avec TTL
# ============================================================================


@dataclass
class CacheEntry:
    token_count: int
    created_at: float
    last_hit_at: float


class PrefixCache:
    """
    Modelise le prompt caching d'Anthropic/OpenAI : un prefixe identique
    (hashe) est conserve jusqu'a TTL secondes apres son dernier hit.
    """
    def __init__(self, ttl_s: float = 300.0):
        self.ttl_s = ttl_s
        self.entries: dict[str, CacheEntry] = {}

    def lookup(self, prefix_hash: str, token_count: int, now: float) -> int:
        """
        Retourne le nombre de tokens en cache hit (0 si miss). Expire les
        vieilles entrees au passage.
        """
        # eviction TTL
        dead = [k for k, v in self.entries.items()
                if now - v.last_hit_at > self.ttl_s]
        for k in dead:
            del self.entries[k]

        entry = self.entries.get(prefix_hash)
        if entry is None:
            # Miss : on creera une entree au write.
            return 0
        entry.last_hit_at = now
        return entry.token_count

    def write(self, prefix_hash: str, token_count: int, now: float):
        self.entries[prefix_hash] = CacheEntry(token_count, now, now)


def hash_prefix(parts: list[str]) -> str:
    # Le hash depend uniquement du contenu et de l'ordre.
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ============================================================================
# 2) Pricing model (approche Anthropic 2025-2026 TTL 5 min)
# ============================================================================
PRICE_INPUT_PER_M = 3.0         # USD / 1M tokens, frais
PRICE_CACHE_READ_PER_M = 0.3     # 10% du prix
PRICE_CACHE_WRITE_PER_M = 3.75   # 125% pour TTL ~5 min
PRICE_OUTPUT_PER_M = 15.0


@dataclass
class CallBill:
    input_fresh: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0

    def cost_usd(self) -> float:
        return (
            self.input_fresh * PRICE_INPUT_PER_M
            + self.cache_read * PRICE_CACHE_READ_PER_M
            + self.cache_write * PRICE_CACHE_WRITE_PER_M
            + self.output * PRICE_OUTPUT_PER_M
        ) / 1_000_000


# ============================================================================
# 3) Simulateur d'appel LLM avec prompt caching
# ============================================================================

def call_llm(context_parts: list[tuple[str, int, bool]],
             output_tokens: int,
             cache: PrefixCache,
             now: float) -> tuple[CallBill, float]:
    """
    context_parts : liste ordonnee de (content, token_count, cacheable).
    Seul le prefixe *continu* de parts cacheables = True en tete est cache.
    Strategie : on cache le plus grand prefixe cacheable possible, on hash
    les parts cumulees.
    """
    bill = CallBill(output=output_tokens)
    # Reperer la frontiere du prefixe cacheable
    prefix_cacheable_end = 0
    for i, (_, _, cacheable) in enumerate(context_parts):
        if cacheable:
            prefix_cacheable_end = i + 1
        else:
            break

    prefix_parts = [c for c, _, _ in context_parts[:prefix_cacheable_end]]
    prefix_tokens = sum(t for _, t, _ in context_parts[:prefix_cacheable_end])
    suffix_tokens = sum(t for _, t, _ in context_parts[prefix_cacheable_end:])

    latency_ms = 50.0  # TTFT de base
    if prefix_parts:
        ph = hash_prefix(prefix_parts)
        hit_tokens = cache.lookup(ph, prefix_tokens, now)
        if hit_tokens > 0:
            bill.cache_read += hit_tokens
            latency_ms += 0.05 * hit_tokens / 1000 * 1000   # cache = rapide (0.05 ms/tok)
        else:
            bill.cache_write += prefix_tokens
            cache.write(ph, prefix_tokens, now)
            latency_ms += 0.25 * prefix_tokens / 1000 * 1000  # traitement plein

    bill.input_fresh += suffix_tokens
    latency_ms += 0.25 * suffix_tokens / 1000 * 1000
    latency_ms += 0.5 * output_tokens   # generation (2k tok/s ≈ 0.5ms/tok)
    return bill, latency_ms


# ============================================================================
# 4) Trois strategies de context engineering
# ============================================================================

SYSTEM_PROMPT = "You are a support agent for ACME Corp. Follow the style guide..."  # ~8k tok
TOOLS_DEFS = "tool: search_kb, tool: create_ticket, tool: send_email"  # ~2k tok
HOT_DOCS = "FAQ + top-100 articles..."  # ~20k tok


def build_context_bad(user_query: str, now_str: str,
                       retrieved: str) -> list[tuple[str, int, bool]]:
    """Anti-pattern : timestamp dans le systeme, ordre instable."""
    return [
        (f"Current time: {now_str}. {SYSTEM_PROMPT}", 8200, True),
        (TOOLS_DEFS, 2000, True),
        (retrieved, 3000, False),
        (HOT_DOCS, 20000, True),
        (user_query, 100, False),
    ]


def build_context_medium(user_query: str, now_str: str,
                          retrieved: str) -> list[tuple[str, int, bool]]:
    """OK : prefixe stable pour le system mais retrieval insere AVANT hot_docs,
    ce qui casse la longueur du prefixe cacheable."""
    return [
        (SYSTEM_PROMPT, 8000, True),
        (TOOLS_DEFS, 2000, True),
        (retrieved, 3000, False),           # <-- casse le prefixe ici
        (HOT_DOCS, 20000, False),           # force non-cacheable car apres un break
        (f"[now={now_str}] {user_query}", 120, False),
    ]


def build_context_good(user_query: str, now_str: str,
                        retrieved: str) -> list[tuple[str, int, bool]]:
    """Optimise : prefixe maximal stable + retrieval en suffixe + timestamp en query."""
    return [
        (SYSTEM_PROMPT, 8000, True),
        (TOOLS_DEFS, 2000, True),
        (HOT_DOCS, 20000, True),
        # Suffixe : tout ce qui varie par query
        (f"Retrieved context:\n{retrieved}", 3000, False),
        (f"[now={now_str}] User: {user_query}", 120, False),
    ]


# ============================================================================
# 5) Simulation : 500 queries sur 1h
# ============================================================================


def simulate(build_fn, label: str, n_queries: int = 500):
    cache = PrefixCache(ttl_s=300)
    now = 0.0
    total_bill = CallBill()
    latencies = []
    for i in range(n_queries):
        now += random.expovariate(n_queries / 3600)  # trafic Poisson sur 1h
        now_str = f"2026-04-18T10:{int(now / 60):02d}:00"
        retrieved_chunk = f"chunk_{i % 50}"  # 50 chunks possibles
        ctx = build_fn(f"query {i}", now_str, retrieved_chunk)
        bill, lat = call_llm(ctx, output_tokens=400, cache=cache, now=now)
        total_bill.input_fresh += bill.input_fresh
        total_bill.cache_read += bill.cache_read
        total_bill.cache_write += bill.cache_write
        total_bill.output += bill.output
        latencies.append(lat)
    total_tokens = (total_bill.input_fresh + total_bill.cache_read
                    + total_bill.cache_write)
    hit_rate = total_bill.cache_read / max(total_tokens, 1)
    avg_lat = sum(latencies) / len(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]
    print(f"\n  Strategy: {label}")
    print(f"    input_fresh : {total_bill.input_fresh:>10,} tokens")
    print(f"    cache_read  : {total_bill.cache_read:>10,} tokens")
    print(f"    cache_write : {total_bill.cache_write:>10,} tokens")
    print(f"    output      : {total_bill.output:>10,} tokens")
    print(f"    hit_rate    : {hit_rate:>10.1%}")
    print(f"    cost        : ${total_bill.cost_usd():>9.3f}")
    print(f"    avg latency : {avg_lat:>10.0f} ms")
    print(f"    p95 latency : {p95_lat:>10.0f} ms")


print("=" * 70)
print("Simulation : 500 queries, meme charge, 3 strategies de contexte")
print("=" * 70)

simulate(build_context_bad, "BAD : timestamp dans system prompt, ordre instable")
simulate(build_context_medium, "MEDIUM : prefixe stable, timestamp dans query")
simulate(build_context_good, "GOOD : prefixe maximal, retrieval en suffixe")

print("""
Lecons :
  - BAD : timestamp en prefixe = cache mort, facture x2-3.
  - MEDIUM : bon hit rate sur les docs chauds, mais prefix coupe par
    l'ordre (retrieval milieu du prompt).
  - GOOD : hit rate maximal, factures divisees par 2-3 sur volume. Meme
    qualite modele, seul le placement change.

Pour ton propre produit :
  1. Logger cache_read_tokens vs input_tokens par endpoint.
  2. Viser un hit rate > 50%.
  3. Evaluer le TTL long (1h Anthropic) si le trafic est soutenu mais
     irregulier.
  4. Re-tester apres chaque changement de system prompt (peut casser
     silencieusement le cache).
""")

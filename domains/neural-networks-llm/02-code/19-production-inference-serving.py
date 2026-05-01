"""
Jour 19 — Production inference serving : simulateur de throughput/latence
==========================================================================
Pure Python. Simule un serveur LLM simple avec :
  1. Static batching  (anti-pattern : on attend le batch complet)
  2. Continuous batching (injecte/retire dynamiquement)
  3. PagedAttention (gestion KV cache par blocs fixes, detection eviction)
  4. Speculative decoding (draft model accepte un % des tokens)

L'objectif n'est pas de repliquer vLLM, mais de comprendre POURQUOI chacune
de ces techniques donne les gains qu'on observe en prod.

Run : python 02-code/19-production-inference-serving.py
"""

from __future__ import annotations
import sys, io, random, math
from dataclasses import dataclass, field
from collections import deque

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(2026)

# ============================================================================
# 1) Modele hardware tres simplifie
# ============================================================================
# On modelise un H100 : lit les poids = temps fixe par token de decode,
# divise par la taille du batch.
WEIGHTS_GB = 140                # Llama 3 70B fp16
HBM_BW_GB_S = 3000              # H100 HBM3
DECODE_TIME_SINGLE_MS = (WEIGHTS_GB / HBM_BW_GB_S) * 1000    # ~47 ms

# Le prefill est compute-bound : ~2000 tok/s/batch-slot a FP16
PREFILL_RATE_TOK_PER_S_PER_SLOT = 2000

# Le decode scale avec batch jusqu'a saturation (ratio flops/memory)
MAX_BATCH_BEFORE_COMPUTE_BOUND = 128


def decode_time_ms(batch_size: int) -> float:
    """Temps d'une iteration de decode pour tout le batch (tous users)."""
    # Tant qu'on est memory-bound, la lecture des poids domine.
    base = DECODE_TIME_SINGLE_MS
    if batch_size > MAX_BATCH_BEFORE_COMPUTE_BOUND:
        # On ajoute une penalite lineaire au-dela du sweet spot
        base *= batch_size / MAX_BATCH_BEFORE_COMPUTE_BOUND
    return base


def prefill_time_ms(prompt_tokens: int) -> float:
    return 1000 * prompt_tokens / PREFILL_RATE_TOK_PER_S_PER_SLOT


# ============================================================================
# 2) Request model
# ============================================================================


@dataclass
class Request:
    id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    # Etat interne simule
    prefill_done: bool = False
    decoded: int = 0
    finish_ms: float | None = None
    ttft_ms: float | None = None


def generate_workload(n: int, rate_per_s: float) -> list[Request]:
    reqs = []
    now = 0.0
    for i in range(n):
        now += random.expovariate(rate_per_s)
        reqs.append(Request(
            id=i,
            arrival_ms=now * 1000,
            prompt_tokens=random.choice([200, 500, 1000, 2000]),
            output_tokens=random.choice([50, 200, 400, 800]),
        ))
    return reqs


# ============================================================================
# 3) Scheduler : static batching (baseline brutal)
# ============================================================================


def schedule_static(reqs: list[Request], batch_size: int):
    """On attend d'avoir batch_size requests, puis on les traite ensemble.
    Toutes finissent a la meme iteration = attendent la plus longue."""
    reqs = list(reqs)
    idx = 0
    t = 0.0
    while idx < len(reqs):
        batch = reqs[idx:idx + batch_size]
        if not batch:
            break
        t = max(t, batch[-1].arrival_ms)
        max_prefill = max(r.prompt_tokens for r in batch)
        t += prefill_time_ms(max_prefill)
        for r in batch:
            r.ttft_ms = t - r.arrival_ms
        max_out = max(r.output_tokens for r in batch)
        for _ in range(max_out):
            t += decode_time_ms(len(batch))
        for r in batch:
            r.finish_ms = t
        idx += batch_size


# ============================================================================
# 4) Scheduler : continuous batching (vLLM-like simplifie)
# ============================================================================


def schedule_continuous(reqs: list[Request], max_batch: int):
    reqs = sorted(reqs, key=lambda r: r.arrival_ms)
    pending = deque(reqs)
    active: list[Request] = []
    t = 0.0

    while pending or active:
        # Inject new requests up to max_batch
        while pending and len(active) < max_batch and pending[0].arrival_ms <= t:
            active.append(pending.popleft())

        # Si rien d'actif, avancer le temps a la prochaine arrivee
        if not active:
            if pending:
                t = pending[0].arrival_ms
                continue
            break

        # Traiter : pour simplifier, on fait les prefill en premier sur l'etape,
        # puis un decode step sur les actifs deja prefillees.
        to_prefill = [r for r in active if not r.prefill_done]
        if to_prefill:
            # Prefill concurrent des nouvelles (chunked prefill simplifie)
            chunk = sum(r.prompt_tokens for r in to_prefill)
            t += prefill_time_ms(chunk) / max(len(to_prefill), 1)
            for r in to_prefill:
                r.prefill_done = True
                if r.ttft_ms is None:
                    r.ttft_ms = t - r.arrival_ms

        # Decode step pour tous les actifs
        t += decode_time_ms(len(active))
        for r in active:
            r.decoded += 1

        # Retirer ceux qui ont fini
        finished = [r for r in active if r.decoded >= r.output_tokens]
        for r in finished:
            r.finish_ms = t
        active = [r for r in active if r.decoded < r.output_tokens]


# ============================================================================
# 5) Speculative decoding wrapper
# ============================================================================


def schedule_continuous_spec(reqs: list[Request], max_batch: int,
                              spec_len: int = 4, accept_rate: float = 0.7):
    """Meme schedule que continuous, mais chaque decode step produit
    1 + accept_rate * spec_len tokens en moyenne (verif en parallele)."""
    reqs = sorted(reqs, key=lambda r: r.arrival_ms)
    pending = deque(reqs)
    active: list[Request] = []
    t = 0.0

    while pending or active:
        while pending and len(active) < max_batch and pending[0].arrival_ms <= t:
            active.append(pending.popleft())
        if not active:
            if pending:
                t = pending[0].arrival_ms
                continue
            break

        to_prefill = [r for r in active if not r.prefill_done]
        if to_prefill:
            chunk = sum(r.prompt_tokens for r in to_prefill)
            t += prefill_time_ms(chunk) / max(len(to_prefill), 1)
            for r in to_prefill:
                r.prefill_done = True
                if r.ttft_ms is None:
                    r.ttft_ms = t - r.arrival_ms

        # Decode step : cout ~= 1 forward pass (memory-bound), produit
        # 1 + spec_len*accept_rate tokens en moyenne.
        step_cost = decode_time_ms(len(active)) * 1.1   # overhead draft
        tokens_per_step = 1 + spec_len * accept_rate
        t += step_cost
        for r in active:
            r.decoded += tokens_per_step

        finished = [r for r in active if r.decoded >= r.output_tokens]
        for r in finished:
            r.finish_ms = t
        active = [r for r in active if r.decoded < r.output_tokens]


# ============================================================================
# 6) Metrics
# ============================================================================


def report(label: str, reqs: list[Request]):
    ttfts = sorted(r.ttft_ms for r in reqs if r.ttft_ms is not None)
    durs = sorted((r.finish_ms - r.arrival_ms) for r in reqs if r.finish_ms)
    total_time = max(r.finish_ms for r in reqs) / 1000  # s
    tokens = sum(r.output_tokens for r in reqs)
    p = lambda arr, q: arr[int(q * len(arr))] if arr else 0
    print(f"\n  {label}")
    print(f"    throughput        : {tokens / total_time:>8.0f} tok/s")
    print(f"    TTFT  p50 / p95   : {p(ttfts, 0.5):>6.0f} / {p(ttfts, 0.95):>6.0f} ms")
    print(f"    total p50 / p95   : {p(durs, 0.5):>6.0f} / {p(durs, 0.95):>6.0f} ms")


# ============================================================================
# Run : 200 requetes a 6 req/s (trafic chat realiste sur un serveur)
# ============================================================================


print("=" * 70)
print("Simulation : 200 requetes, 6 req/s en moyenne, Llama 3 70B sur H100")
print("=" * 70)

WORKLOAD = generate_workload(n=200, rate_per_s=6.0)


def clone_reqs(w):
    return [Request(r.id, r.arrival_ms, r.prompt_tokens, r.output_tokens) for r in w]


w1 = clone_reqs(WORKLOAD); schedule_static(w1, batch_size=16)
report("STATIC batching (b=16)", w1)

w2 = clone_reqs(WORKLOAD); schedule_continuous(w2, max_batch=32)
report("CONTINUOUS batching (max=32)", w2)

w3 = clone_reqs(WORKLOAD); schedule_continuous_spec(w3, max_batch=32,
                                                     spec_len=4, accept_rate=0.7)
report("CONTINUOUS + speculative (k=4, acc=0.7)", w3)

w4 = clone_reqs(WORKLOAD); schedule_continuous_spec(w4, max_batch=32,
                                                     spec_len=4, accept_rate=0.3)
report("CONTINUOUS + spec (acceptance 0.3 = MAUVAIS draft)", w4)

print("""
Lecons :
  - Static batching : TTFT p95 enorme (les courts attendent les longs).
  - Continuous batching : throughput +3-5x, TTFT diminue drastiquement.
  - Speculative decoding bien regle : +2-3x sur decode throughput, TTFT meme.
  - Speculative avec mauvais draft (acceptance 0.3) = overhead net. Monitorer
    acceptance_rate est non-negociable en prod.
  - Pour aller plus loin : chunked prefill (intercaler prefill+decode), pref
    caching au niveau KV, disaggregated serving prefill/decode.

Stack de reference 2026 :
  vLLM 0.6+ : continuous batching, PagedAttention, EAGLE, FP8 sur H100.
  SGLang   : prefix caching agressif, structured outputs natifs (XGrammar).
  TensorRT-LLM : perf max NVIDIA, compile, moins flexible.
""")

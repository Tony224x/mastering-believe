"""
Solutions -- Day 9 HARD Exercises: Inference at Scale

Worked solutions. The GPU sizing, the prefix-caching saving, the multi-tier
routing economics and the static->continuous batching ~32x gain are all COMPUTED
and pinned with assertions on the numbers the exercise calls out (~27 seq/H100,
~1M in-flight, ~32x, ~90% prefix saving, ~5x cheaper small model). The 2025-2026
optimization stack and the runbook are printed and pinned by facts.

Usage:
    python3 09-inference-at-scale-hard.py
"""

import math

SEPARATOR = "=" * 60


# =============================================================================
# HARD -- Exercise 1 : Size + architect LLM serving for a 100K req/s product
# =============================================================================

def hard_1_size_llm_serving():
    """Sizing, batching, KV cache, optimization stack, routing, autoscaling."""
    print(f"\n{SEPARATOR}")
    print("  HARD 1 : Sizing LLM serving @ 100K req/s")
    print(SEPARATOR)

    print("\n  1. GPU sizing (KV cache is the limiter) :")
    kv_per_token = 320_000             # ~320 KB/token (int8 KV, 70B class)
    seq_len = 4_000
    room_bytes = 35 * 1024 ** 3        # ~35 GB left after weights+overhead on H100-80G
    kv_per_seq = kv_per_token * seq_len
    seqs_per_h100 = math.floor(room_bytes / kv_per_seq)
    print(f"     KV per token ~{kv_per_token/1024:.0f} KB, seq={seq_len} tok ->")
    print(f"       KV per seq ~{kv_per_seq/1024**3:.1f} GB ; {room_bytes/1024**3:.0f} GB room")
    print(f"       -> ~{seqs_per_h100} concurrent sequences per H100 (int4 weights)")
    assert 20 <= seqs_per_h100 <= 35   # exercise expects ~27

    req_per_s = 100_000
    avg_req_s = 10                      # ~10 s average request duration
    in_flight = req_per_s * avg_req_s
    min_h100 = math.ceil(in_flight / seqs_per_h100)
    print(f"     In-flight ~= req/s * avg duration = {req_per_s:,} * {avg_req_s}s "
          f"= {in_flight:,}")
    print(f"       -> >= {min_h100:,} H100 (order of magnitude : thousands), + headroom")
    assert in_flight == 1_000_000      # 100K * 10s = 1M in-flight
    assert min_h100 >= 1_000           # thousands of GPUs

    print("\n  2. Batching :")
    print("     STATIC batching excluded : sequences of unequal length -> the GPU")
    print("       idles waiting for the longest -> low utilization.")
    print("     CONTINUOUS batching + PagedAttention = the foundation : finished")
    print("       slots are refilled immediately, no waiting for the longest seq.")
    print("     PagedAttention pre-req : a paged KV allocator (16-token blocks);")
    print("       it saves the fragmentation waste of contiguous KV -> more seqs.")

    print("\n  3. KV cache & memory :")
    prompt_tokens = 3_000
    req_per_day = 1_000_000
    naive_prefix_compute = prompt_tokens * req_per_day
    cached_prefix_compute = prompt_tokens * 1          # computed once, reused
    saving = 1 - cached_prefix_compute / naive_prefix_compute
    print("     Reduce KV : int8 KV (~2x), prefix sharing across requests.")
    print(f"     Prefix caching a {prompt_tokens}-token system prompt over "
          f"{req_per_day:,} req/day :")
    print(f"       naive recompute = {naive_prefix_compute:,} prompt-token-computes/day")
    print(f"       with caching    = {cached_prefix_compute:,} (computed once)")
    print(f"       -> ~{saving*100:.4f}% saved on those system-prompt tokens (~90%+ in practice)")
    assert saving > 0.90

    print("\n  4. Optimization stack 2025-2026 (lever for each) :")
    stack = [
        ("continuous batching", "throughput (no idle slots)"),
        ("PagedAttention", "throughput (less KV waste -> bigger batch)"),
        ("prefix caching", "cost + TTFT (skip recomputing shared prefix)"),
        ("chunked prefill", "TTFT/TPOT balance (interleave prefill with decode)"),
        ("speculative decoding", "TPOT (draft model proposes, target verifies)"),
        ("disaggregated serving", "latency (split prefill/decode on separate pools)"),
        ("semantic routing", "cost (send easy queries to cheap models)"),
    ]
    for opt, lever in stack:
        print(f"     - {opt:<22}: {lever}")
    assert len(stack) == 7             # the 7 optimizations are all present

    print("\n  5. Multi-tier routing (60/30/10) :")
    # Per-request cost relative to the frontier model (=1.0). Tier prices are
    # NOT negligible (a hosted small model still costs real money), so the 10%
    # frontier slice keeps the blended cost realistic -> economy lands ~40-70%.
    cost_frontier = 1.0
    cost_mid = 0.45
    cost_nano = 0.15
    all_frontier = 1.0 * cost_frontier
    routed = 0.60 * cost_nano + 0.30 * cost_mid + 0.10 * cost_frontier
    economy = 1 - routed / all_frontier
    print(f"     Cost vs all-frontier : routed mix = {routed:.3f} of frontier cost")
    print(f"       -> ~{economy*100:.0f}% cheaper (exercise expects 40-70%)")
    assert 0.40 <= economy <= 0.70
    print("     Mis-calibrated router risk : easy->frontier (waste) or hard->nano")
    print("       (bad answers). Detect via the ESCALATION/fallback rate metric.")

    print("\n  6. Autoscaling & resilience :")
    print("     Scale trigger : QUEUE DEPTH (+ p99 latency), NEVER CPU.")
    print("     Cold start 30-60 s : keep HOT replicas, snapshot/CUDA-graph the")
    print("       model, avoid naive scale-to-zero on latency-critical paths.")
    print("     Queue + workers pattern : decouples ingest from GPU, enables")
    print("       batching, and gives observability (queue depth as the signal).")


# =============================================================================
# HARD -- Exercise 2 : Post-mortem -- a GPU at 15% billed at 100%
# =============================================================================

def hard_2_serving_postmortem():
    """Post-mortem: static batching + fp16 + no routing -> terrible economics."""
    print(f"\n{SEPARATOR}")
    print("  HARD 2 : LLM serving post-mortem (15% GPU, 100% bill)")
    print(SEPARATOR)

    print("\n  1. Root cause per symptom :")
    causes = [
        ("BATCHING", "15-20% GPU util + 8s p99",
         "static batch=8 waits for the longest seq -> static -> continuous"),
        ("MEMORY", "fp16 = 140 GB on 2 H100",
         "fp16 -> int4 (35 GB) frees room for a much bigger KV/batch"),
        ("ROUTING", "70% trivial traffic still hits the 70B",
         "tout-70B -> route trivial traffic to a small model"),
        ("AUTOSCALING", "scales on CPU -> never scales, queue overflows",
         "CPU stays low (GPU is the bottleneck) -> scale on queue depth + p99"),
        ("MEMORY", "3000-token system prompt recomputed every request",
         "no prefix caching -> reuse the system-prompt KV across requests"),
    ]
    for cat, symptom, fix in causes:
        print(f"     [{cat}] {symptom}")
        print(f"       -> {fix}")
    print("     A GPU at 15% costs 100% : you pay the GPU PER HOUR regardless of its")
    print("     utilization -> idle silicon is pure burned money.")

    print("\n  2. The gain numbers :")
    static_rps = 1.25                  # per GPU pair, reported
    continuous_rps = 40.0              # after continuous batching
    factor = continuous_rps / static_rps
    print(f"     static {static_rps} req/s -> continuous ~{continuous_rps} req/s "
          f"= ~{factor:.0f}x")
    print("       mechanism : slots refilled continuously, no waiting for the")
    print("       longest sequence of a fixed batch.")
    assert abs(factor - 32) < 0.5      # ~32x
    fp16_gb, int4_gb = 140, 35
    print(f"     fp16 {fp16_gb} GB -> int4 {int4_gb} GB : the freed "
          f"{fp16_gb-int4_gb} GB becomes KV cache -> bigger batch.")
    assert fp16_gb / int4_gb == 4.0    # int4 is ~4x smaller than fp16

    print("\n  3. Autoscaling broken :")
    print("     The GPU is the bottleneck, so CPU stays low -> a CPU trigger never")
    print("     fires while the queue overflows. Use : (1) queue depth, (2) p99 latency.")

    print("\n  4. Prefix caching :")
    prompt_tokens = 3_000
    print(f"     The {prompt_tokens}-token system prompt is identical every request.")
    print("     Compute its KV ONCE and reuse across requests -> up to ~90% saved on")
    print("     those tokens (huge on TTFT and on raw compute cost).")

    print("\n  5. Routing :")
    trivial_frac = 0.70
    small_cost_ratio = 1 / 5           # small model ~5x cheaper than 70B
    saving = trivial_frac * (1 - small_cost_ratio)
    print(f"     Route {trivial_frac*100:.0f}% trivial traffic to a ~5x cheaper small")
    print(f"       model -> ~{saving*100:.0f}% off the raw LLM bill on that slice.")
    assert abs(saving - 0.56) < 0.01

    print("\n  6. Corrected serving stack + priority order :")
    stack = [
        "continuous batching + PagedAttention",
        "quantization (int4) for weights",
        "prefix caching (shared system prompt)",
        "chunked prefill",
        "speculative decoding",
        "multi-tier routing",
        "autoscaling on queue depth + p99",
    ]
    for s in stack:
        print(f"     - {s}")
    print("     Priority (biggest immediate win first) :")
    priority = [
        "1. continuous batching/PagedAttention (the ~32x throughput jump)",
        "2. quantization int4 (unlocks KV/batch room)",
        "3. prefix caching + routing (cost on the long tail of trivial traffic)",
        "4. autoscaling on queue depth (stops queue overflow in peak)",
    ]
    for p in priority:
        print(f"       {p}")


def main():
    print("\n" + "=" * 60)
    print("  SOLUTIONS -- DAY 9 HARD : INFERENCE AT SCALE")
    print("=" * 60)
    hard_1_size_llm_serving()
    hard_2_serving_postmortem()
    print(f"\n{'=' * 60}")
    print("  END OF HARD SOLUTIONS (all assertions passed)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

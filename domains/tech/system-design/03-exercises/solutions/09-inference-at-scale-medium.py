"""
Solutions -- Day 9 MEDIUM Exercises: Inference at Scale

Worked solutions. GPU memory budget, continuous-batching utilization, and the
routing + prefix-caching cost math are computed with assertions.

Usage:
    python3 09-inference-at-scale-medium.py
"""

SEPARATOR = "=" * 70


# =============================================================================
# MEDIUM -- Exercise 1 : GPU memory budget -- KV cache vs weights
# =============================================================================

def medium_1_gpu_memory():
    """Compute what fits on a GPU and derive the max batch."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 1 : GPU memory budget -- KV cache vs weights")
    print(SEPARATOR)

    gpu_gb = 80
    overhead_gb = 10
    w_int8 = 70
    w_int4 = 35
    seq_tokens = 4000
    kv_per_token_bytes = 320_000          # ~320 KB/token (decimal)

    # KV per sequence in GB (decimal)
    kv_per_seq_gb = seq_tokens * kv_per_token_bytes / 1e9     # ~1.28 GB

    # 1. int8
    kv_room_int8 = gpu_gb - w_int8 - overhead_gb
    print(f"\n  1. int8 (70 GB weights) :")
    print(f"     KV room = 80 - 70 - 10 = {kv_room_int8} GB -> essentially NOTHING left.")
    print(f"     int8 70B barely fits one H100; batching is ~impossible on a single GPU.")

    # 2. int4
    kv_room_int4 = gpu_gb - w_int4 - overhead_gb
    seqs_int4 = int(kv_room_int4 / kv_per_seq_gb)
    print(f"\n  2. int4 (35 GB weights) :")
    print(f"     KV room = 80 - 35 - 10 = {kv_room_int4} GB")
    print(f"     KV/seq  = {seq_tokens} * 320 KB = {kv_per_seq_gb:.2f} GB")
    print(f"     -> ~{seqs_int4} sequences of {seq_tokens} tokens in parallel")

    print(f"\n  3. Why the batch is dictated by the KV cache :")
    print(f"     weights are FIXED; the KV cache grows with batch * seq length, so it is")
    print(f"     the KV cache that caps how many sequences run at once.")

    print(f"\n  4. Why PagedAttention helps :")
    print(f"     contiguous allocation reserves {seq_tokens} tokens up front even for short")
    print(f"     sequences (wasted). Paging (16-token non-contiguous blocks) only uses")
    print(f"     what each sequence needs -> more sequences fit.")

    # 5. KV int8
    seqs_int4_kv8 = int(kv_room_int4 / (kv_per_seq_gb / 2))
    print(f"\n  5. KV cache in int8 (half the size) :")
    print(f"     KV/seq -> {kv_per_seq_gb/2:.2f} GB -> ~{seqs_int4_kv8} sequences (~2x more).")

    # ---- assertions ----
    assert kv_room_int8 == 0, kv_room_int8
    assert kv_room_int4 == 35, kv_room_int4
    assert abs(kv_per_seq_gb - 1.28) < 0.01, kv_per_seq_gb
    assert seqs_int4 == 27, seqs_int4
    assert seqs_int4_kv8 == 54, seqs_int4_kv8
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 2 : Continuous batching vs static
# =============================================================================

def medium_2_continuous_batching():
    """Quantify why continuous batching multiplies throughput."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 2 : Continuous batching vs static")
    print(SEPARATOR)

    # seq lengths in a static batch of 4
    lengths = [1000, 50, 50, 50]
    slots = len(lengths)

    # 1. static : wait for the longest
    total_steps = max(lengths)                  # 1000
    useful_slot_steps = sum(lengths)            # 1150
    available_slot_steps = slots * total_steps  # 4000
    print(f"\n  1. Static batching :")
    print(f"     total steps = max(lengths) = {total_steps}")
    print(f"     useful slot-steps = {lengths[0]}+{lengths[1]}+{lengths[2]}+{lengths[3]} = {useful_slot_steps}")
    print(f"     available slot-steps = {slots} * {total_steps} = {available_slot_steps}")

    # 2. utilization
    util = useful_slot_steps / available_slot_steps
    print(f"\n  2. Average slot utilization :")
    print(f"     {useful_slot_steps} / {available_slot_steps} = {util:.1%}")
    print(f"     -> 3 of 4 slots sit idle ~95% of the time.")

    print(f"\n  3. Continuous batching :")
    print(f"     as soon as B/C/D finish at 50, a new sequence fills the free slot.")
    print(f"     Under sustained load, slot utilization tends toward ~100%.")

    print(f"\n  4. Absorbing the queue :")
    print(f"     yes -- while A generates its 1000 tokens, the slots freed by B/C/D take")
    print(f"     new 50-token requests from the queue -> throughput jumps ~3x+.")

    print(f"\n  5. Technical pre-requisite :")
    print(f"     PagedAttention (non-contiguous paged KV cache) so sequences can be")
    print(f"     inserted/removed mid-batch without reallocation.")

    # ---- assertions ----
    assert total_steps == 1000
    assert useful_slot_steps == 1150
    assert available_slot_steps == 4000
    assert abs(util - 0.2875) < 1e-6, util       # ~29%
    print("\n  [assertions OK]")


# =============================================================================
# MEDIUM -- Exercise 3 : Semantic routing + prefix caching -- cost saving
# =============================================================================

def medium_3_routing_caching():
    """Quantify routing + prefix caching on the bill."""
    print(f"\n{SEPARATOR}")
    print("  MEDIUM 3 : Semantic routing + prefix caching -- cost saving")
    print(SEPARATOR)

    reqs = 1_000_000
    price_big = 5 / 1e6          # $/token
    price_nano = 0.3 / 1e6
    tokens_in = 3000
    tokens_out = 500
    tokens_total = tokens_in + tokens_out      # 3500
    system_tokens = 2500
    non_cached_tokens = tokens_total - system_tokens   # 1000

    # 1. baseline
    baseline = reqs * tokens_total * price_big
    print(f"\n  1. Baseline (all big model, no cache) :")
    print(f"     {reqs:,} * {tokens_total} * $5/1M = ${baseline:,.0f}/day")

    # 2. routing only : 60% nano, 40% big
    cost_nano = 0.60 * reqs * tokens_total * price_nano
    cost_big = 0.40 * reqs * tokens_total * price_big
    routing = cost_nano + cost_big
    print(f"\n  2. Routing only (60% nano + 40% big) :")
    print(f"     nano : ${cost_nano:,.0f} + big : ${cost_big:,.0f} = ${routing:,.0f}/day")

    # 3. prefix caching only (all big) : system tokens billed at 10%
    eff_tokens = system_tokens * 0.10 + non_cached_tokens     # 1250
    caching = reqs * eff_tokens * price_big
    print(f"\n  3. Prefix caching only (all big) :")
    print(f"     per req : {system_tokens}*0.1 + {non_cached_tokens} = {eff_tokens:.0f} token-equiv")
    print(f"     {reqs:,} * {eff_tokens:.0f} * $5/1M = ${caching:,.0f}/day")

    # 4. routing + caching (caching applied to the big-model 40%)
    cost_big_cached = 0.40 * reqs * eff_tokens * price_big
    combined = cost_nano + cost_big_cached
    reduction = 1 - combined / baseline
    print(f"\n  4. Routing + prefix caching (cache on the big-model part) :")
    print(f"     nano ${cost_nano:,.0f} + big-cached ${cost_big_cached:,.0f} = ${combined:,.0f}/day")
    print(f"     reduction vs baseline : {reduction:.0%}")

    print(f"\n  5. Mis-calibrated router :")
    print(f"     sending COMPLEX queries to nano degrades quality. Watch the escalation /")
    print(f"     fallback rate and per-tier satisfaction to detect a bad router.")

    # ---- assertions ----
    assert abs(baseline - 17_500) < 1, baseline
    assert abs(routing - 7_630) < 1, routing
    assert abs(caching - 6_250) < 1, caching
    assert abs(combined - 3_130) < 1, combined
    assert 0.80 < reduction < 0.84, reduction        # ~82%
    print("\n  [assertions OK]")


def main():
    print("\n" + SEPARATOR)
    print("  SOLUTIONS -- DAY 9 MEDIUM : INFERENCE AT SCALE")
    print(SEPARATOR)
    medium_1_gpu_memory()
    medium_2_continuous_batching()
    medium_3_routing_caching()
    print(f"\n{SEPARATOR}")
    print("  END OF MEDIUM SOLUTIONS (all assertions passed)")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()

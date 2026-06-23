"""
Day 18 - Long context: Flash Attention, RoPE scaling, sliding window, attention sinks

Pure NumPy implementation. Fixed seed. Runs in < 30s on CPU.

Goal: build intuition on the memory wall, the tiling trick (Flash Attention),
RoPE scaling variants (PI / NTK-aware / YaRN), sliding window, and attention sinks.

We do NOT implement a real GPU Flash Attention (no CUDA here). We model the
memory footprint and verify that tiled attention produces identical numerical
output to the naive computation, which is the load-bearing claim of the paper.

Run: python 18-long-context-attention-scaling.py
"""

import numpy as np
import math
import time

np.random.seed(42)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# PART 1 - The memory wall: naive attention vs tiled attention
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V):
    """
    Vanilla attention. Materializes the full N x N score matrix.
    This is what runs out of memory on long sequences.

    Q, K, V : shape (N, d)
    Returns : output (N, d), peak_score_matrix_bytes (theoretical)
    """
    N, d = Q.shape
    # The matrix S is the bottleneck: O(N^2) memory.
    S = Q @ K.T / math.sqrt(d)
    P = softmax(S, axis=-1)
    O = P @ V
    # Peak memory for the score matrix in fp16 (2 bytes).
    peak_bytes = N * N * 2
    return O, peak_bytes


def tiled_attention(Q, K, V, block_size=128):
    """
    Flash Attention v1 idea, simulated in NumPy.

    We never materialize the N x N matrix. Instead we process Q in blocks,
    iterate K, V in blocks, and maintain an online softmax (running max + sum).

    This computes EXACTLY the same output as naive_attention (up to fp errors)
    but with O(N) memory instead of O(N^2).
    """
    N, d = Q.shape
    O = np.zeros_like(Q)
    # Per-Q-block running statistics for online softmax.
    # m_i = running max of scores; l_i = running sum of exp(scores - m_i).
    m = np.full((N,), -np.inf)
    l = np.zeros((N,))

    # Block over Q (outer), then over K (inner). Online softmax handles streaming.
    for i in range(0, N, block_size):
        Qi = Q[i:i + block_size]                 # (Bq, d)
        Oi = np.zeros_like(Qi)                   # accumulator for this Q block
        mi = np.full((Qi.shape[0],), -np.inf)    # running max for this Q block
        li = np.zeros((Qi.shape[0],))            # running denom for this Q block

        for j in range(0, N, block_size):
            Kj = K[j:j + block_size]             # (Bk, d)
            Vj = V[j:j + block_size]             # (Bk, d)
            Sij = Qi @ Kj.T / math.sqrt(d)       # (Bq, Bk) - tile only

            # Online softmax update (Tri Dao, 2022).
            mij = np.max(Sij, axis=-1)           # (Bq,)
            mi_new = np.maximum(mi, mij)         # new running max
            # Rescale previous accumulators with the max delta.
            alpha = np.exp(mi - mi_new)
            beta = np.exp(mij - mi_new)
            Pij = np.exp(Sij - mi_new[:, None])  # (Bq, Bk)

            li = alpha * li + np.sum(Pij, axis=-1)
            Oi = alpha[:, None] * Oi + Pij @ Vj
            mi = mi_new

        # Final normalization for this Q block.
        Oi = Oi / li[:, None]
        O[i:i + block_size] = Oi

    # Peak memory: only one tile of S in SRAM at a time -> O(B^2).
    peak_bytes = block_size * block_size * 2
    return O, peak_bytes


def part1_memory_wall():
    banner("PART 1 - The memory wall: naive vs tiled (Flash) attention")

    print("Theoretical peak attention-matrix memory in fp16, single head:\n")
    print(f"{'N':>10} | {'Naive O(N^2)':>15} | {'Tiled O(B^2)':>15}")
    print("-" * 50)
    for N in [1024, 4096, 16384, 100_000, 1_000_000]:
        naive_mb = N * N * 2 / (1024 ** 2)
        tiled_mb = 128 * 128 * 2 / (1024 ** 2)
        # Format gracefully for absurd values.
        naive_str = f"{naive_mb:.1f} MB" if naive_mb < 1024 else f"{naive_mb / 1024:.2f} GB"
        print(f"{N:>10} | {naive_str:>15} | {tiled_mb:>10.4f} MB")

    print("\nNumerical equivalence check on N=512, d=64:")
    N, d = 512, 64
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    t0 = time.time()
    O_naive, mem_naive = naive_attention(Q, K, V)
    t_naive = time.time() - t0

    t0 = time.time()
    O_tiled, mem_tiled = tiled_attention(Q, K, V, block_size=64)
    t_tiled = time.time() - t0

    max_diff = np.max(np.abs(O_naive - O_tiled))
    print(f"  naive  : output shape {O_naive.shape}, time {t_naive*1000:.1f} ms, S matrix ~ {mem_naive/1024:.1f} KB")
    print(f"  tiled  : output shape {O_tiled.shape}, time {t_tiled*1000:.1f} ms, tile     ~ {mem_tiled/1024:.1f} KB")
    print(f"  max abs diff : {max_diff:.2e}  (should be ~ 1e-6)")
    assert max_diff < 1e-4, "Tiled attention should match naive numerically"
    print("  -> numerical equivalence: OK")


# ---------------------------------------------------------------------------
# PART 2 - RoPE scaling: PI vs NTK-aware vs YaRN
# ---------------------------------------------------------------------------

def rope_frequencies(d, base=10000.0):
    """Standard RoPE inverse frequencies for d/2 pairs."""
    i = np.arange(0, d, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d))


def rope_pi_frequencies(d, base, scale):
    """
    Position Interpolation (PI, Meta 2023).
    Effective position m_eff = m / scale -> equivalent to dividing freqs by scale.
    """
    inv_freq = rope_frequencies(d, base)
    return inv_freq / scale


def rope_ntk_frequencies(d, base, scale):
    """
    NTK-aware scaling (bloc97, 2023).
    Increase the base instead of compressing positions, preserving high freqs.
    """
    new_base = base * (scale ** (d / (d - 2)))
    return rope_frequencies(d, new_base)


def rope_yarn_frequencies(d, base, scale, alpha=1.0, beta=32.0, L_train=4096):
    """
    YaRN (Peng et al., 2023). Per-band scaling: high freqs untouched,
    low freqs interpolated (PI-like), mid band smooth NTK transition.

    alpha, beta : transition window in number of original-train wavelengths.
    """
    inv_freq = rope_frequencies(d, base)
    inv_freq_pi = inv_freq / scale
    # Wavelength of each freq dimension (in tokens).
    wavelen = 2 * math.pi / inv_freq
    # Ratio of wavelength to L_train tells us if a freq is high (<<1) or low (>>1).
    ratio = L_train / wavelen
    # Linear ramp between alpha and beta (in band ratio).
    ramp = np.clip((ratio - alpha) / (beta - alpha), 0.0, 1.0)
    # ramp = 0 -> low freq -> use PI ; ramp = 1 -> high freq -> use original.
    inv_freq_yarn = (1 - ramp) * inv_freq_pi + ramp * inv_freq
    return inv_freq_yarn


def part2_rope_scaling():
    banner("PART 2 - RoPE scaling: extend a 4K model to 32K")

    d = 64                # head dim
    base = 10000.0
    L_train = 4096
    L_target = 32768
    scale = L_target / L_train   # = 8.0

    f_orig = rope_frequencies(d, base)
    f_pi = rope_pi_frequencies(d, base, scale)
    f_ntk = rope_ntk_frequencies(d, base, scale)
    f_yarn = rope_yarn_frequencies(d, base, scale, L_train=L_train)

    # Show how each method affects representative frequencies.
    print(f"Head dim d={d}, base={base}, L_train={L_train}, L_target={L_target}, scale={scale}\n")
    print(f"{'dim_pair':>9} | {'orig freq':>11} | {'PI freq':>11} | {'NTK freq':>11} | {'YaRN freq':>11}")
    print("-" * 65)
    show_idx = [0, 4, 8, 16, 24, 30]   # spread across the spectrum
    for k in show_idx:
        print(f"{k:>9} | {f_orig[k]:>11.4e} | {f_pi[k]:>11.4e} | {f_ntk[k]:>11.4e} | {f_yarn[k]:>11.4e}")

    # Quantify high-freq preservation: ratio of new / orig at high freqs (k=0).
    print("\nHigh-freq preservation (ratio new/orig at k=0, higher = better local resolution):")
    print(f"  PI   : {f_pi[0]/f_orig[0]:.3f}  (squashed by 8x -> bad local precision)")
    print(f"  NTK  : {f_ntk[0]/f_orig[0]:.3f}  (almost untouched)")
    print(f"  YaRN : {f_yarn[0]/f_orig[0]:.3f}  (untouched at high freq)")

    # Quantify low-freq spread: at the lowest freq, PI compresses fully.
    print("\nLow-freq behavior (k=last pair, low freqs handle long-range positions):")
    print(f"  PI   freq / orig freq at last pair : {f_pi[-1]/f_orig[-1]:.3f}  (= 1/scale = {1/scale:.3f})")
    print(f"  NTK  freq / orig freq at last pair : {f_ntk[-1]/f_orig[-1]:.3f}")
    print(f"  YaRN freq / orig freq at last pair : {f_yarn[-1]/f_orig[-1]:.3f}  (PI-like at low freq)")
    print("\nTakeaway: YaRN keeps high freqs intact (local) while compressing low freqs (range).")


# ---------------------------------------------------------------------------
# PART 3 - Sliding window attention
# ---------------------------------------------------------------------------

def make_sliding_mask(N, W):
    """
    Causal sliding-window mask.
    Position i can attend to positions [max(0, i-W+1), i].
    1 = visible, 0 = masked.
    """
    mask = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        lo = max(0, i - W + 1)
        mask[i, lo:i + 1] = 1.0
    return mask


def attention_with_mask(Q, K, V, mask):
    """Naive attention with an additive mask (-inf where masked)."""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    S = np.where(mask > 0, S, -1e9)
    P = softmax(S, axis=-1)
    return P, P @ V


def part3_sliding_window():
    banner("PART 3 - Sliding window attention (Mistral-style)")

    N, d, W = 64, 16, 8
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    full_mask = np.tril(np.ones((N, N), dtype=np.float32))   # causal full
    sw_mask = make_sliding_mask(N, W)

    P_full, _ = attention_with_mask(Q, K, V, full_mask)
    P_sw, _ = attention_with_mask(Q, K, V, sw_mask)

    print(f"N={N}, window W={W}")
    print(f"  Full causal : avg # tokens attended per row = {full_mask.sum(axis=1).mean():.1f}")
    print(f"  Sliding W={W} : avg # tokens attended per row = {sw_mask.sum(axis=1).mean():.1f}")

    # Show that attention beyond the window is exactly 0.
    far_attn_full = P_full[N - 1, :N - W].sum()
    far_attn_sw = P_sw[N - 1, :N - W].sum()
    print(f"\nAt last token (i={N-1}), attention to tokens > W away from i:")
    print(f"  Full causal : {far_attn_full:.4f}  (non-zero, attends to early tokens)")
    print(f"  Sliding W   : {far_attn_sw:.6f}  (zero, masked out)")

    # Receptive field via stacked layers.
    L_layers = 32
    receptive = L_layers * W
    print(f"\nReceptive field after L={L_layers} stacked sliding layers : {receptive} tokens")
    print(f"-> a token at position {receptive} can theoretically influence the output,")
    print("   but information dilutes at every layer hop in practice.")


# ---------------------------------------------------------------------------
# PART 4 - StreamingLLM: attention sinks vs softmax collapse
# ---------------------------------------------------------------------------

def attention_perplexity_proxy(Q, K, V, mask):
    """
    Proxy for 'how confident is the attention'. We measure the negative log
    of the mean diagonal-ish probability mass. If the model is forced to
    spread uniformly because it cannot drain attention, this rises sharply.

    We also return the entropy of the attention rows (high entropy = model
    cannot pick anything specific = collapse-like behavior).
    """
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    S = np.where(mask > 0, S, -1e9)
    P = softmax(S, axis=-1)
    # Entropy of attention rows (averaged over tokens).
    eps = 1e-12
    entropy = -np.sum(P * np.log(P + eps), axis=-1).mean()
    return P, entropy


def part4_attention_sinks():
    banner("PART 4 - StreamingLLM: attention sinks prevent softmax collapse")

    # We simulate a long sequence and 3 strategies for the last token's KV cache:
    #   A) Full causal (oracle)
    #   B) Sliding window only (drop the early tokens)
    #   C) Sliding window + 4 attention sinks (StreamingLLM)
    #
    # We synthesize Q and K such that early tokens act as "boring" tokens
    # that the model wants to drain attention to (low information).

    N, d = 256, 32
    Q = np.random.randn(N, d).astype(np.float32) * 0.5
    K = np.random.randn(N, d).astype(np.float32) * 0.5
    V = np.random.randn(N, d).astype(np.float32)

    # Make the first 4 tokens look like natural sinks: their K vectors are
    # softly aligned with most Q vectors -> they receive attention from many.
    sink_dir = np.random.randn(d).astype(np.float32)
    for i in range(4):
        K[i] = sink_dir + 0.05 * np.random.randn(d)
    # Make all Q vectors lean toward this direction (boring tokens drain here).
    for i in range(N):
        Q[i] = Q[i] + 0.4 * sink_dir

    W = 32  # sliding window

    # Strategy A : full causal (oracle baseline)
    mask_full = np.tril(np.ones((N, N), dtype=np.float32))
    P_full, ent_full = attention_perplexity_proxy(Q, K, V, mask_full)

    # Strategy B : sliding window only, first tokens dropped
    mask_sw = make_sliding_mask(N, W)
    P_sw, ent_sw = attention_perplexity_proxy(Q, K, V, mask_sw)

    # Strategy C : sliding window + 4 attention sinks (always visible)
    mask_sink = mask_sw.copy()
    n_sinks = 4
    mask_sink[:, :n_sinks] = 1.0
    # Maintain causal: a token at position i still cannot see positions > i.
    causal = np.tril(np.ones((N, N), dtype=np.float32))
    mask_sink = mask_sink * causal
    P_sink, ent_sink = attention_perplexity_proxy(Q, K, V, mask_sink)

    # Look at the LAST token's attention distribution.
    last = N - 1
    print(f"N={N}, sliding window W={W}, n_sinks={n_sinks}")
    print(f"\nAttention entropy (avg across all tokens):")
    print(f"  Full causal        : {ent_full:.4f}")
    print(f"  Sliding only       : {ent_sw:.4f}")
    print(f"  Sliding + 4 sinks  : {ent_sink:.4f}")
    print("\nNote: in this toy single-layer setup, adding sinks LOWERS entropy")
    print("(attention concentrates on the sink tokens) rather than recovering the")
    print("full-causal entropy. This is expected: the StreamingLLM phenomenon")
    print("(Xiao 2023) manifests on long autoregressive generation across many")
    print("layers in real LLMs, where the softmax-must-sum-to-1 collapse compounds")
    print("layer over layer. A pure NumPy single-pass attention does not reproduce")
    print("this dynamic. See Xiao et al. 2023 figure 4 for the real perplexity")
    print("explosion when early tokens are dropped on actual decoder LLMs.")

    # Mass redirected onto the (formerly) sink positions when we keep them.
    sink_mass_full = P_full[last, :n_sinks].sum()
    sink_mass_sink = P_sink[last, :n_sinks].sum()
    print(f"\nAttention mass at last token landing on first {n_sinks} tokens (the sinks):")
    print(f"  Full causal       : {sink_mass_full:.4f}")
    print(f"  Sliding + sinks   : {sink_mass_sink:.4f}  (sinks absorb a large share of attention)")

    # Drift of the output vector: how far is sliding-only output from the oracle?
    O_full = P_full @ V
    O_sw = P_sw @ V
    O_sink = P_sink @ V
    drift_sw = np.linalg.norm(O_sw[last] - O_full[last])
    drift_sink = np.linalg.norm(O_sink[last] - O_full[last])
    print(f"\nL2 drift of the LAST token output vs full-causal oracle:")
    print(f"  Sliding only       : {drift_sw:.4f}")
    print(f"  Sliding + 4 sinks  : {drift_sink:.4f}")
    print("\n-> Honest takeaway: in this toy setup, neither sliding nor sliding+sinks")
    print("   reproduces full-causal output exactly (both drift). The point of")
    print("   StreamingLLM is operational stability over millions of generated tokens,")
    print("   not single-pass numerical fidelity. To observe the real effect, run")
    print("   a multi-layer decoder LM and stream past its training context length.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    part1_memory_wall()
    part2_rope_scaling()
    part3_sliding_window()
    part4_attention_sinks()
    print(f"\nTotal runtime : {time.time() - t_start:.2f} s")

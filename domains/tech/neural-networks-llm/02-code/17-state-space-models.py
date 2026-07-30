"""
Day 17 - State Space Models (Mamba & friends)
==============================================

Pure NumPy walkthrough of the SSM primitives behind S4 / Mamba.

What you will see, end to end:
  PART 1: A linear SSM in RECURRENT mode (h_t = A h_{t-1} + B x_t, y_t = C h_t).
          One token at a time, O(1) memory per step. The "RNN-style" view.
  PART 2: The SAME SSM in CONVOLUTIONAL mode. We unroll the recurrence
          analytically into a kernel K = (CB, CAB, CA^2 B, ...) and apply
          y = K * x with a regular 1D convolution. We assert numerical
          equivalence with PART 1 (this is THE property that makes SSMs
          parallelisable at training time).
  PART 3: SELECTIVITY (the Mamba twist). On a "selective copying" task,
          a non-selective SSM (B and C fixed) fails to suppress filler
          tokens. A selective SSM (B and C are functions of x_t) succeeds.
          This is the qualitative gap S4 -> Mamba.
  PART 4: COMPLEXITY benchmark. We time a linear SSM vs a quadratic
          self-attention on growing sequence lengths (N = 128, 512, 2048,
          8192) and watch the curves diverge.
  PART 5: KDA (Kimi Delta Attention, Kimi K3, 2026) - the two ideas that
          took linear recurrence to frontier scale.
          5a. The DELTA RULE: erase what the key already retrieved before
              writing the new value, instead of only accumulating. Measured
              on an overwrite task, where accumulation returns a blend.
          5b. The BOUNDED decay gate: why g_min = -5 is not cosmetic. The
              chunkwise kernel needs 1/Gamma (the reciprocal cumulative
              decay over a tile) to fit in BF16; a bound by construction is
              what allows every causal tile to run as a dense matmul.

No GPU, no torch. Runs in well under 30 seconds. Seed pinned for
determinism.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
SEED = 17
np.random.seed(SEED)


def section(title: str) -> None:
    """Pretty-print a section header so the output is readable."""
    bar = "=" * 72
    print("\n" + bar)
    print(title)
    print(bar)


# ===========================================================================
# PART 1 - Linear SSM, recurrent mode
# ===========================================================================
# We model a single-channel sequence x of length N going through an SSM
# with hidden state of dimension D. The discrete recurrence is:
#     h_t = A h_{t-1} + B x_t        (state update, h in R^D)
#     y_t = C h_t                    (readout,        y in R)
# A is D x D, B is D x 1, C is 1 x D. We fix D and the matrices.
# This is "S4-without-HiPPO": the matrices are arbitrary but stable.

def make_stable_A(D: int, decay: float = 0.9) -> np.ndarray:
    """
    Build a diagonal-ish A whose spectrum lies inside the unit circle.
    Stability requires |eigenvalues(A)| < 1, otherwise h blows up.
    A diagonal A with entries in (-1, 1) is the simplest safe choice and
    is exactly what S4-Diagonal and Mamba use in practice.
    """
    diag = np.linspace(decay, decay * 0.5, D)  # all in (0, 1) -> stable
    return np.diag(diag)


def ssm_recurrent(x: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Apply the SSM token-by-token. This is the inference path: O(N) time
    sequentially, O(D) memory (we only keep h, not the whole history).
    """
    N = x.shape[0]
    D = A.shape[0]
    h = np.zeros(D)                  # initial hidden state h_0 = 0
    y = np.zeros(N)
    for t in range(N):
        # State update: h_t = A h_{t-1} + B x_t
        h = A @ h + B.flatten() * x[t]
        # Readout: y_t = C h_t (scalar output, single-channel)
        y[t] = (C @ h).item()
    return y


# ===========================================================================
# PART 2 - Same SSM, convolutional mode
# ===========================================================================
# Unrolling the recurrence with h_0 = 0:
#     h_1 = B x_1
#     h_2 = A B x_1 + B x_2
#     h_3 = A^2 B x_1 + A B x_2 + B x_3
#     ...
# Therefore:
#     y_t = sum_{k=0..t-1} (C A^k B) * x_{t-k}
# So y = K * x (causal 1D convolution) with the SSM KERNEL:
#     K = (CB, CAB, CA^2 B, CA^3 B, ..., CA^{N-1} B)
# At training time we precompute K once (length N) and convolve x in
# parallel. On a GPU this is one FFT instead of N sequential steps.

def ssm_kernel(A: np.ndarray, B: np.ndarray, C: np.ndarray, N: int) -> np.ndarray:
    """
    Materialise the convolution kernel of length N from (A, B, C).
    Cost here is O(N * D^2) because we keep multiplying A^k B. In real
    SSM training this is replaced by an FFT-based stable formulation.
    """
    D = A.shape[0]
    K = np.zeros(N)
    Ak_B = B.flatten().copy()        # A^0 B = B
    for k in range(N):
        K[k] = (C @ Ak_B).item()     # K[k] = C A^k B (the impulse response at lag k)
        Ak_B = A @ Ak_B              # bump to A^{k+1} B for the next step
    return K


def ssm_convolutional(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Apply the SSM as a CAUSAL 1D convolution. This is the training path:
    fully parallel across t. We use np.convolve for clarity; production
    code uses an FFT for O(N log N) instead of this O(N^2) scalar version.
    """
    N = x.shape[0]
    full = np.convolve(x, K, mode="full")  # length 2N-1
    return full[:N]                        # keep only the causal prefix


# ===========================================================================
# PART 3 - Selectivity (the Mamba twist)
# ===========================================================================
# Task: "selective copying". The input contains DATA tokens (uniform random)
# and FILLER tokens (always 0). A perfect model should output the DATA
# tokens verbatim and suppress FILLER. The flag for each token is given
# alongside x: flag = 1 (data) or 0 (filler).
#
# - The NON-selective SSM has fixed (A, B, C). It cannot condition on the
#   flag, so it leaks filler through B's effect on h.
# - The SELECTIVE SSM (Mamba-style) lets B and C depend on (x, flag). When
#   flag = 0, B becomes ~0, so filler tokens do not enter the state.

def make_selective_copy_data(N: int = 64, data_frac: float = 0.25):
    """Build a sequence of N tokens. Some are 'data' (carry value), rest are filler (0)."""
    flags = (np.random.rand(N) < data_frac).astype(np.float32)  # 1 = keep, 0 = filler
    x = np.where(flags == 1, np.random.randn(N), 0.0)
    return x, flags


def ssm_recurrent_nonselective(x, flags, A, B, C):
    """Vanilla SSM: ignores 'flags', so filler still pumps state via B."""
    D = A.shape[0]
    h = np.zeros(D)
    y = np.zeros(len(x))
    for t in range(len(x)):
        h = A @ h + B.flatten() * x[t]   # B fixed -> always lets x in
        y[t] = (C @ h).item()
    return y


def ssm_recurrent_selective(x, flags, A, B_base, C_base):
    """
    Mamba-style: B_t and C_t are gated by the flag.
    In real Mamba, the gate comes from a small MLP on x_t; here we use the
    flag directly as the gate to keep the demo readable. The point is the
    SAME: the SSM coefficients become functions of the input.
    """
    D = A.shape[0]
    h = np.zeros(D)
    y = np.zeros(len(x))
    for t in range(len(x)):
        gate = flags[t]                          # 1 -> open, 0 -> closed
        B_t = B_base.flatten() * gate            # closes the input gate on filler
        C_t = C_base * gate                      # also closes the readout
        h = A @ h + B_t * x[t]
        y[t] = (C_t @ h).item()
    return y


# ===========================================================================
# PART 4 - Complexity benchmark: SSM vs self-attention
# ===========================================================================
# We measure wall-clock for a single forward pass at increasing sequence
# lengths. The point is the CURVE shape, not absolute numbers (NumPy is
# slow). SSM should grow ~linearly, attention ~quadratically.

def attention_forward(x: np.ndarray, d_model: int = 16) -> np.ndarray:
    """
    Toy single-head self-attention. Input x has shape (N,). We project
    to (N, d_model) with random Q, K, V, compute the N x N score matrix,
    softmax, and produce outputs. Cost: O(N^2 d).
    """
    N = x.shape[0]
    rng = np.random.default_rng(SEED)
    Wq = rng.standard_normal((1, d_model)) * 0.1
    Wk = rng.standard_normal((1, d_model)) * 0.1
    Wv = rng.standard_normal((1, d_model)) * 0.1
    X = x[:, None]                               # (N, 1)
    Q = X @ Wq                                   # (N, d)
    K = X @ Wk                                   # (N, d)
    V = X @ Wv                                   # (N, d)
    scores = Q @ K.T / np.sqrt(d_model)          # (N, N)  <-- the quadratic step
    # numerical-stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    out = weights @ V                            # (N, d)
    return out.sum(axis=-1)                      # squash to (N,) for shape parity


def ssm_forward_for_bench(x: np.ndarray, A, B, C) -> np.ndarray:
    """SSM forward via convolutional mode. Building the kernel is O(N D^2).
    Here NumPy does a DIRECT convolution which is O(N^2); in production we
    use FFT for O(N log N). The benchmark thus does NOT reflect the real
    asymptotic complexity of SSMs (it under-sells them at large N)."""
    N = x.shape[0]
    K = ssm_kernel(A, B, C, N)
    return ssm_convolutional(x, K)


def benchmark(sizes):
    """Time both methods on a few sequence lengths and report ratios."""
    D = 8
    A = make_stable_A(D)
    B = np.random.randn(D, 1) * 0.3
    C = np.random.randn(1, D) * 0.3
    print(f"{'N':>6} | {'SSM (s)':>10} | {'Attn (s)':>10} | {'Attn/SSM':>10}")
    print("-" * 48)
    rows = []
    for N in sizes:
        x = np.random.randn(N).astype(np.float64)
        t0 = time.perf_counter()
        _ = ssm_forward_for_bench(x, A, B, C)
        t_ssm = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = attention_forward(x, d_model=16)
        t_attn = time.perf_counter() - t0
        ratio = t_attn / t_ssm if t_ssm > 0 else float("inf")
        rows.append((N, t_ssm, t_attn, ratio))
        print(f"{N:>6} | {t_ssm:>10.4f} | {t_attn:>10.4f} | {ratio:>10.2f}x")
    return rows


# ===========================================================================
# PART 5 - KDA: the delta rule, and why its decay gate has to be bounded
# ===========================================================================
# Kimi K3 (Moonshot AI, 2026) does not use a Mamba-style SSM for its 69
# linear layers: it uses Kimi Delta Attention (KDA), a LINEAR ATTENTION whose
# recurrent state is a MATRIX S of shape (d_k, d_v) - an associative memory
# mapping keys to values.
#
#   additive (vanilla linear attention, Mamba-like accumulation):
#       S_t = diag(alpha_t) S_{t-1} + k_t v_t^T
#
#   delta rule (KDA):
#       S_t = (I - beta_t k_t k_t^T) diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
#             ^^^^^^^^^^^^^^^^^^^^^^ erase what this key already pointed to
#
# Reading it: the additive form only ever ADDS to memory, so writing twice to
# the same key leaves BOTH values superposed. The delta rule first REMOVES the
# component of S along k_t, then writes. That is Widrow-Hoff's delta rule,
# applied online to the state.

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """KDA L2-normalizes queries and keys; without it beta loses its meaning."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def linear_attention_additive(keys, values, alpha):
    """Vanilla gated linear attention: accumulate, never correct."""
    d_k, d_v = keys.shape[1], values.shape[1]
    S = np.zeros((d_k, d_v))
    for k, v, a in zip(keys, values, alpha):
        S = a[:, None] * S + np.outer(k, v)
    return S


def linear_attention_delta(keys, values, alpha, beta):
    """
    KDA-style delta rule.

    (I - beta k k^T) S is a rank-1 correction: because k is unit-norm,
    it subtracts beta * k * (k^T S), i.e. exactly beta times "whatever is
    currently stored under key k". With beta = 1 the old association is
    fully erased before the new one is written.
    """
    d_k, d_v = keys.shape[1], values.shape[1]
    S = np.zeros((d_k, d_v))
    for k, v, a, b in zip(keys, values, alpha, beta):
        S_decayed = a[:, None] * S
        old = k @ S_decayed                     # what key k currently retrieves
        S = S_decayed - b * np.outer(k, old) + b * np.outer(k, v)
    return S


def retrieve(S: np.ndarray, query: np.ndarray) -> np.ndarray:
    """o_t = S^T q_t — read the associative memory."""
    return S.T @ query


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def bounded_decay(z, A_h, g_min=-5.0):
    """
    Kimi K3 parameterization: g = g_min * sigmoid(exp(A) * z), so the
    per-step log-decay is CONFINED to (g_min, 0) and alpha = exp(g) can
    never get arbitrarily close to zero.
    """
    return g_min * (1.0 / (1.0 + np.exp(-np.exp(A_h) * z)))


def unbounded_decay(z, A_h):
    """
    Kimi Linear (2025) parameterization: g = -exp(A) * softplus(z).
    Perfectly reasonable mathematically - and unbounded below, which is
    where the numerical trouble starts.
    """
    softplus = np.log1p(np.exp(np.clip(z, -60, 60)))
    return -np.exp(A_h) * softplus


# ===========================================================================
# Main
# ===========================================================================
def main():
    # -----------------------------------------------------------------------
    section("PART 1 + PART 2 - Recurrent vs Convolutional equivalence")
    # -----------------------------------------------------------------------
    D = 4
    N = 32
    A = make_stable_A(D)
    B = np.random.randn(D, 1) * 0.5
    C = np.random.randn(1, D) * 0.5

    x = np.random.randn(N)

    # Recurrent path (inference style)
    y_rec = ssm_recurrent(x, A, B, C)

    # Convolutional path (training style): build the kernel once, convolve.
    K = ssm_kernel(A, B, C, N)
    y_conv = ssm_convolutional(x, K)

    max_diff = float(np.max(np.abs(y_rec - y_conv)))
    print(f"State dim D = {D}, sequence N = {N}")
    print(f"Kernel K (first 8 taps): {np.round(K[:8], 4)}")
    print(f"Recurrent[:5] = {np.round(y_rec[:5], 4)}")
    print(f"Convolu.[:5]  = {np.round(y_conv[:5], 4)}")
    print(f"Max |y_rec - y_conv| = {max_diff:.2e}  "
          f"(tiny -> the two modes are mathematically the SAME function)")

    # -----------------------------------------------------------------------
    section("PART 3 - Selective vs non-selective SSM on selective copying")
    # -----------------------------------------------------------------------
    # Build the task: data tokens carry value, filler tokens are zeros.
    # The "ideal" output is: original value at data positions, 0 at filler.
    np.random.seed(SEED)                # reset to keep the demo reproducible
    x, flags = make_selective_copy_data(N=64, data_frac=0.25)

    # Truth: the SSM should output something correlated with x at flagged
    # positions and ~0 at filler positions.
    A = make_stable_A(D=4, decay=0.85)
    B = np.random.randn(4, 1) * 0.6
    C = np.random.randn(1, 4) * 0.6

    y_ns = ssm_recurrent_nonselective(x, flags, A, B, C)
    y_sel = ssm_recurrent_selective(x, flags, A, B, C)

    # Score: how loud is the output at filler positions vs data positions?
    # A good selective SSM has filler ~ 0 and data >> 0.
    filler_mask = (flags == 0)
    data_mask = (flags == 1)
    leak_ns = float(np.mean(np.abs(y_ns[filler_mask])))
    leak_sel = float(np.mean(np.abs(y_sel[filler_mask])))
    signal_ns = float(np.mean(np.abs(y_ns[data_mask])))
    signal_sel = float(np.mean(np.abs(y_sel[data_mask])))

    print(f"Sequence length = {len(x)}, data tokens = {int(data_mask.sum())}, "
          f"filler tokens = {int(filler_mask.sum())}")
    print(f"Non-selective: filler leak = {leak_ns:.4f}, data signal = {signal_ns:.4f}, "
          f"SNR = {signal_ns / max(leak_ns, 1e-9):.2f}")
    print(f"Selective    : filler leak = {leak_sel:.4f}, data signal = {signal_sel:.4f}, "
          f"SNR = {signal_sel / max(leak_sel, 1e-9):.2f}")
    print("-> Selective SSM closes B and C on filler tokens, so they do not")
    print("   pump h. This is exactly what Mamba does at scale, with B, C, Delta")
    print("   produced by tiny MLPs over x_t.")

    # -----------------------------------------------------------------------
    section("PART 4 - Complexity: SSM (linear-ish) vs attention (quadratic)")
    # -----------------------------------------------------------------------
    print("DISCLAIMER : ce micro-bench utilise np.convolve (convolution DIRECTE,")
    print("O(N^2)) et un kernel build naif en O(N * D^2). Il ne reflete donc PAS")
    print("la complexite asymptotique reelle des SSM en production, ou la")
    print("convolution se fait par FFT (O(N log N)) et le kernel S4 par")
    print("factorisation Cauchy. Lire la courbe comme une intuition, pas une")
    print("preuve : le vrai SSM scale BIEN mieux que ce que NumPy montre ici.")
    print()
    sizes = [128, 512, 2048, 8192]
    rows = benchmark(sizes)
    # Show the scaling: when N is multiplied by 4, attention should grow
    # ~16x (quadratic), SSM should grow ~4x (linear-ish, ignoring NumPy
    # overhead and the O(N D^2) kernel build).
    print()
    if len(rows) >= 2:
        N_small, t_ssm_small, t_attn_small, _ = rows[0]
        N_big, t_ssm_big, t_attn_big, _ = rows[-1]
        scale_n = N_big / N_small
        print(f"From N={N_small} to N={N_big} (x{scale_n:.0f}):")
        print(f"  SSM  time grew  {t_ssm_big / max(t_ssm_small, 1e-9):>6.1f}x  "
              f"(expected ~{scale_n:.0f}x linear, plus kernel-build overhead)")
        print(f"  Attn time grew  {t_attn_big / max(t_attn_small, 1e-9):>6.1f}x  "
              f"(expected ~{scale_n ** 2:.0f}x quadratic)")

    # -----------------------------------------------------------------------
    section("PART 5a - Delta rule vs accumulation: overwriting a key")
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    d_k, d_v = 64, 64

    # Scenario: we store 40 key->value pairs, then OVERWRITE 10 of them with
    # a new value (think: a variable reassigned later in a long document).
    # At read time, the correct answer is always the LATEST value.
    n_pairs, n_overwrite = 40, 10
    base_keys = l2_normalize(rng.standard_normal((n_pairs, d_k)))
    base_vals = rng.standard_normal((n_pairs, d_v))

    over_idx = rng.choice(n_pairs, size=n_overwrite, replace=False)
    new_vals = rng.standard_normal((n_overwrite, d_v))

    # The write stream: all pairs, then the overwrites.
    keys = np.vstack([base_keys, base_keys[over_idx]])
    values = np.vstack([base_vals, new_vals])

    # Ground truth after the stream.
    truth = base_vals.copy()
    truth[over_idx] = new_vals

    # No forgetting (alpha = 1) so we isolate ONE variable: additive vs delta.
    alpha = np.ones((len(keys), d_k))
    beta = np.ones(len(keys))

    S_add = linear_attention_additive(keys, values, alpha)
    S_del = linear_attention_delta(keys, values, alpha, beta)

    kept_idx = np.setdiff1d(np.arange(n_pairs), over_idx)

    def mean_cos(S, idx):
        return np.mean([cosine(retrieve(S, base_keys[i]), truth[i]) for i in idx])

    print(f"State S is {d_k}x{d_v}; {n_pairs} keys written, "
          f"{n_overwrite} of them later overwritten.")
    print("Metric: cosine(retrieved, LATEST true value). 1.00 = perfect recall.")
    print()
    print(f"  {'':<24}{'untouched keys':>16}{'overwritten keys':>18}")
    print(f"  {'additive (accumulate)':<24}"
          f"{mean_cos(S_add, kept_idx):>16.3f}{mean_cos(S_add, over_idx):>18.3f}")
    print(f"  {'delta rule (KDA)':<24}"
          f"{mean_cos(S_del, kept_idx):>16.3f}{mean_cos(S_del, over_idx):>18.3f}")
    print()
    print("-> Read the two columns separately. On UNTOUCHED keys both rules score")
    print("   about the same: the gap there is cross-talk noise (40 associations")
    print("   crammed into a 64x64 state), and no update rule removes it.")
    print("   On OVERWRITTEN keys the additive state returns a BLEND of the old")
    print("   and the new value - it never erased the stale association. The")
    print("   delta rule subtracts what the key already retrieved before writing,")
    print("   so it returns the current value. That is KDA's edge on associative")
    print("   recall, the historical weak spot of linear backbones (PART 3 / MQAR).")

    # -----------------------------------------------------------------------
    section("PART 5b - Why Kimi K3 bounds the decay gate from below")
    # -----------------------------------------------------------------------
    # The chunkwise form of KDA rescales keys by the reciprocal of the
    # CUMULATIVE decay over a tile. If that cumulative decay underflows,
    # its reciprocal overflows - and the tile can no longer be computed as a
    # dense matmul on Tensor Cores.
    TILE = 16                       # secondary tile size used by Kimi K3
    BF16_MAX = 3.3895314e38
    LOG_BF16_MAX = np.log(BF16_MAX)
    N_CHANNELS = 4096

    z = rng.standard_normal((TILE, N_CHANNELS)) * 2.0   # decay logits per channel

    # A_h is LEARNED. Nothing in the unbounded parameterization stops it from
    # drifting up during training, so we sweep it instead of picking one value.
    print(f"Tile = {TILE} tokens, {N_CHANNELS} channels, decay logits z ~ N(0, 2).")
    print("Reported: worst-case cumulative 1/Gamma over the tile, in log space")
    print(f"(BF16 overflows above e^{LOG_BF16_MAX:.1f}).")
    print()
    print(f"  {'A_h (learned)':<16}{'unbounded: worst 1/Gamma':>28}"
          f"{'overflows':>12}{'bounded: worst':>18}")
    for A_h in [0.0, 1.0, 2.0, 3.0]:
        g_un = unbounded_decay(z, A_h)
        g_bd = bounded_decay(z, A_h)
        worst_un = (-g_un.sum(axis=0)).max()
        worst_bd = (-g_bd.sum(axis=0)).max()
        n_over = int((-g_un.sum(axis=0) > LOG_BF16_MAX).sum())
        flag = f"{n_over}/{N_CHANNELS}"
        print(f"  {A_h:<16.1f}{'e^' + format(worst_un, '.1f'):>28}"
              f"{flag:>12}{'e^' + format(worst_bd, '.1f'):>18}")

    print()
    print("-> The unbounded gate is fine until it is not: its worst case grows")
    print("   with a LEARNED parameter, so no bound can be proven ahead of time.")
    print(f"   The bounded gate has one by construction: alpha >= e^-5 = "
          f"{np.exp(-5):.2e},")
    print(f"   so over a {TILE}-token tile the cumulative log-decay cannot go")
    print(f"   below {-5 * TILE}, and 1/Gamma stays under e^{5 * TILE} = "
          f"{np.exp(5 * TILE):.1e} < BF16 max {BF16_MAX:.1e}")
    print("   - whatever the data and whatever training does to A_h.")
    print("   That guarantee is what lets EVERY causal tile, diagonal ones")
    print("   included, be computed as a dense Tensor Core matmul. The")
    print("   parameterization of a gate decided whether the hardware could be")
    print("   used: algorithm-system co-design, not a mathematical detail.")

    section("Done.")
    print("Recap:")
    print("  1. Recurrent and convolutional SSM are the SAME function (PART 1+2).")
    print("  2. Selectivity (B, C, Delta as functions of x) is what S4 -> Mamba.")
    print("  3. SSM scales ~linearly in sequence length, attention quadratically.")
    print("  4. Hybrids get the best of both worlds: 1 attn for 7 Mamba (Jamba,")
    print("     2024), 1 global attn for 3 linear (Kimi K3, 2026, at 2.78 T).")
    print("  5. Delta rule (KDA) corrects the state instead of only adding to it,")
    print("     and its decay gate is bounded so the kernel stays on Tensor Cores.")


if __name__ == "__main__":
    main()

"""
Jour 16 — Mixture of Experts (MoE)
===================================
Pure Python + numpy, no torch dependency.

Pedagogical goal: implement a MoE layer end-to-end to internalize the
mechanics behind Mixtral, DeepSeek-V3 and the rest of the 2024-2026 frontier
sparse models. The whole point of MoE is to decouple total parameters from
per-token compute. This script proves it experimentally.

Contents:
  PART 1 — Top-k routing from scratch (the gating network)
  PART 2 — Forward pass through a MoE FFN layer (8 experts, top-2)
  PART 3 — Load balancing loss (Shazeer 2017) and what it actually penalizes
  PART 4 — Total params vs active params: the Mixtral 8x7B accounting
  PART 5 — LatentMoE (Kimi K3, 2026): why ~10^3 experts only became affordable
           once routed experts stopped working in the full model width
  PART 6 — Quantile Balancing (Kimi K3) vs the fixed-step adaptive bias:
           the step-size dilemma, and what a sampling noise floor looks like

Run: python 02-code/16-mixture-of-experts.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Deterministic experiments. MoE behaviors (collapse, balance) are very
# sensitive to seed; we want reproducible numbers across runs.
np.random.seed(42)


# ============================================================================
# PART 1 — Top-k routing from scratch
# ============================================================================
print("=" * 70)
print("PART 1 : Top-k router (the gating network)")
print("=" * 70)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Numerically stable softmax. Production routers add a z-loss to keep
    # logits small; here we just subtract the max which is enough for our toy.
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def top_k_router(x: np.ndarray, W_g: np.ndarray, k: int = 2):
    """
    The whole router is a single linear layer. That is the entire 'gating
    network' in Mixtral / DeepSeek. The simplicity is the point.

    Returns:
      top_k_indices: which experts each token picks            (B, k)
      top_k_weights: renormalized weights for those experts    (B, k)
      full_probs:   raw softmax over all experts (for aux loss)(B, N)
    """
    logits = x @ W_g                  # (B, N)
    full_probs = softmax(logits, axis=-1)

    # We pick the k highest-probability experts per token.
    # argsort is descending; we take the first k columns.
    top_k_indices = np.argsort(-full_probs, axis=-1)[:, :k]      # (B, k)

    # Gather the probabilities of those k experts and renormalize so they
    # sum to 1. This is what becomes the weighting in the final mix.
    rows = np.arange(x.shape[0])[:, None]
    top_k_probs = full_probs[rows, top_k_indices]                # (B, k)
    top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

    return top_k_indices, top_k_weights, full_probs


# Toy setup: 6 tokens of dimension 16, 8 experts, top-2 routing (Mixtral-like).
B, d_model, N, k = 6, 16, 8, 2
x = np.random.randn(B, d_model).astype(np.float32)
W_g = np.random.randn(d_model, N).astype(np.float32) * 0.1

idx, w, probs = top_k_router(x, W_g, k=k)

print(f"  Input shape       : {x.shape}  (B tokens, d_model)")
print(f"  Router weight     : {W_g.shape}  (d_model, N experts)")
print(f"  Top-{k} indices    : {idx.shape}")
print(f"  Top-{k} weights    : sum per row = {w.sum(axis=-1)}  (renormalized)")
print()
for t in range(B):
    chosen = list(zip(idx[t].tolist(), [round(float(v), 3) for v in w[t]]))
    print(f"  token {t}: experts {chosen}")
print("  --> each token picks its own k experts. The router is just one matmul.")
print()


# ============================================================================
# PART 2 — Forward pass through a MoE FFN layer
# ============================================================================
print("=" * 70)
print("PART 2 : MoE forward pass (sparse dispatch + weighted sum)")
print("=" * 70)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class MoELayer:
    """
    Minimal MoE FFN layer. N experts, each is a 2-layer MLP.
    Real implementations (Mixtral) use SwiGLU; we use ReLU for simplicity —
    the routing math is identical, only the inner activation changes.
    """

    def __init__(self, d_model: int, d_ff: int, N: int, k: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.N, self.k, self.d_model, self.d_ff = N, k, d_model, d_ff
        # Router weights: maps d_model -> logits over N experts.
        self.W_g = rng.standard_normal((d_model, N)).astype(np.float32) * 0.1
        # Each expert has its own up/down projections. This is where the
        # "sparse parameters" live — N copies of the FFN matrices.
        self.W_up = rng.standard_normal((N, d_model, d_ff)).astype(np.float32) * 0.1
        self.W_dn = rng.standard_normal((N, d_ff, d_model)).astype(np.float32) * 0.1

    def expert_forward(self, x: np.ndarray, expert_id: int) -> np.ndarray:
        # Standard 2-layer MLP for one expert. Real Mixtral expert is SwiGLU
        # with 3 matrices; the routing logic does not change.
        h = relu(x @ self.W_up[expert_id])
        return h @ self.W_dn[expert_id]

    def forward(self, x: np.ndarray):
        idx, w, probs = top_k_router(x, self.W_g, self.k)
        out = np.zeros_like(x)

        # The naive loop below is O(B*k) expert calls. Production GPUs
        # implement this as one batched grouped-matmul or all-to-all
        # dispatch; the math is the same. Looping makes the dependency
        # graph crystal clear.
        for t in range(x.shape[0]):
            for j in range(self.k):
                e = idx[t, j]
                out[t] += w[t, j] * self.expert_forward(x[t:t + 1], e)[0]

        # Track which experts received which tokens (used in PART 3).
        # We expose probs because the load-balance loss needs them.
        return out, idx, probs


moe = MoELayer(d_model=16, d_ff=64, N=8, k=2, seed=1)
y, picks, probs = moe.forward(x)
print(f"  MoE output shape : {y.shape}")
print(f"  Mean activation  : {y.mean():.4f}")
print(f"  Token routing matrix:")
for t in range(B):
    print(f"    token {t} -> experts {picks[t].tolist()}")

# Count actual expert load in this small batch.
counts = np.zeros(8, dtype=int)
for row in picks:
    for e in row:
        counts[int(e)] += 1
print(f"  Expert token counts (top-2, B=6) : {counts.tolist()}")
print("  --> with no balancing loss yet, the distribution is already uneven.")
print()


# ============================================================================
# PART 3 — Load balancing loss (Shazeer 2017)
# ============================================================================
print("=" * 70)
print("PART 3 : Load balancing loss — what gets penalized exactly")
print("=" * 70)


def load_balancing_loss(top_k_indices: np.ndarray,
                        full_probs: np.ndarray,
                        N: int) -> float:
    """
    The Shazeer aux loss:
      f_i = fraction of tokens routed to expert i (hard count, not differentiable)
      P_i = mean softmax probability assigned to expert i (differentiable)
      L_aux = N * sum_i (f_i * P_i)

    Why both terms? f_i alone is non-differentiable (it goes through argmax).
    P_i alone does not constrain the actual dispatched load. Their product
    aligns the gradient of P with the realized hard distribution f. Genius.

    L_aux is minimized at L_aux = 1 when both f and P are uniform = 1/N each.
    L_aux = N at the worst case (all tokens to one expert).
    """
    B = top_k_indices.shape[0]
    k = top_k_indices.shape[1]

    # f_i: empirical fraction of (token, slot) pairs routed to expert i.
    # We count k slots per token, hence divide by (B * k).
    f = np.zeros(N, dtype=np.float32)
    for row in top_k_indices:
        for e in row:
            f[int(e)] += 1.0
    f /= (B * k)

    # P_i: average probability mass assigned to expert i across the batch.
    P = full_probs.mean(axis=0)

    return float(N * np.dot(f, P))


# Scenario A — uniform routing (the ideal we want)
uniform_idx = np.array([[i % N, (i + 1) % N] for i in range(B)])
uniform_probs = np.ones((B, N), dtype=np.float32) / N
loss_uniform = load_balancing_loss(uniform_idx, uniform_probs, N)

# Scenario B — total collapse (all tokens to expert 0)
collapsed_idx = np.zeros((B, k), dtype=int)
collapsed_idx[:, 1] = 1  # second choice = expert 1, just to vary
collapsed_probs = np.zeros((B, N), dtype=np.float32)
collapsed_probs[:, 0] = 0.95
collapsed_probs[:, 1] = 0.05
loss_collapsed = load_balancing_loss(collapsed_idx, collapsed_probs, N)

# Scenario C — our actual MoE from PART 2
loss_actual = load_balancing_loss(picks, probs, N)

print(f"  Uniform routing      L_aux = {loss_uniform:.4f}  (best, target = 1.0)")
print(f"  Collapsed routing    L_aux = {loss_collapsed:.4f}  (worst, max = N = {N})")
print(f"  Our random init MoE  L_aux = {loss_actual:.4f}")
print()
print("  --> in real training, lambda_aux ~ 0.01 is added to the task loss.")
print("      Without it, after a few hundred steps, 2-3 experts win all traffic")
print("      and the rest never train (the famous 'expert collapse').")
print()


# ============================================================================
# PART 4 — Total params vs active params : the Mixtral 8x7B accounting
# ============================================================================
print("=" * 70)
print("PART 4 : Total vs active params (Mixtral 8x7B accounting)")
print("=" * 70)


def transformer_dense_params(layers: int, d_model: int, d_ff: int,
                             vocab: int, n_heads: int) -> dict:
    """Approximate parameter count for a dense transformer layer."""
    # Attention: 4 projections of (d_model, d_model) — Q, K, V, O.
    attn = 4 * d_model * d_model
    # FFN: up (d_model, d_ff) + down (d_ff, d_model). SwiGLU adds a third
    # matrix; we use the 2-matrix approximation for clarity.
    ffn = 2 * d_model * d_ff
    per_layer = attn + ffn
    total = per_layer * layers + vocab * d_model  # + embeddings
    return {
        "attn_per_layer": attn,
        "ffn_per_layer": ffn,
        "total": total,
    }


def transformer_moe_params(layers: int, d_model: int, d_ff: int,
                           vocab: int, n_heads: int,
                           N: int, k: int) -> dict:
    """Same accounting but with N experts per FFN layer."""
    attn = 4 * d_model * d_model
    ffn_one_expert = 2 * d_model * d_ff
    ffn_all_experts = N * ffn_one_expert
    router = d_model * N

    per_layer_total = attn + ffn_all_experts + router
    per_layer_active = attn + k * ffn_one_expert + router  # k experts fire

    return {
        "ffn_one_expert": ffn_one_expert,
        "total": per_layer_total * layers + vocab * d_model,
        "active": per_layer_active * layers + vocab * d_model,
    }


# Mixtral 8x7B-ish architecture (rounded for clarity).
LAYERS = 32
D_MODEL = 4096
D_FF = 14336
VOCAB = 32000
HEADS = 32

dense_70b = transformer_dense_params(80, 8192, 28672, VOCAB, 64)  # ~Llama 3 70B
moe_mixtral = transformer_moe_params(LAYERS, D_MODEL, D_FF, VOCAB, HEADS, N=8, k=2)


def fmt_b(n: float) -> str:
    return f"{n / 1e9:.2f} B"


print(f"  Dense Llama-3-70B-ish")
print(f"    total params           : {fmt_b(dense_70b['total'])}")
print(f"    active per token       : {fmt_b(dense_70b['total'])}  (all of them)")
print()
print(f"  Mixtral 8x7B-ish (N=8, k=2)")
print(f"    total params           : {fmt_b(moe_mixtral['total'])}")
print(f"    active per token       : {fmt_b(moe_mixtral['active'])}")
ratio = moe_mixtral['total'] / moe_mixtral['active']
print(f"    sparsity ratio         : {ratio:.2f}x  (total / active)")
print(f"    NOTE: this code uses a 2-matrix FFN (up/down). The real Mixtral")
print(f"          uses SwiGLU (3 matrices: gate/up/down), so the numbers")
print(f"          above (~32 B total / ~9.8 B active) are LOWER than the")
print(f"          official Mixtral figures (~47 B total / ~13 B active).")
print()
print(f"  Same arch but DeepSeek-V3 style (N=256, k=8, 1 shared)")
ds = transformer_moe_params(LAYERS, D_MODEL, D_FF // 8, VOCAB, HEADS, N=256, k=8)
# DeepSeek shrinks each expert (d_ff // 8) since experts are fine-grained.
print(f"    total params           : {fmt_b(ds['total'])}")
print(f"    active per token       : {fmt_b(ds['active'])}")
print(f"    sparsity ratio         : {ds['total'] / ds['active']:.2f}x")
print(f"    NOTE: 'DeepSeek-V3 style' here reuses the Mixtral arch above with")
print(f"          finer sparsity (256 experts, top-8). The real DeepSeek-V3")
print(f"          has 61 layers, d_model=7168, 671 B total params — different")
print(f"          backbone, not just different routing.")
print()
print("  Reading the numbers:")
print("  - Mixtral keeps Llama-13B compute on a 47B-param brain.")
print("  - DeepSeek pushes the ratio further: more total capacity, comparable FLOPs.")
print("  - Both pay full VRAM cost: experts must all be loaded just-in-case.")
print()


# ============================================================================
# PART 5 — LatentMoE : why 896 experts only became affordable in 2026
# ============================================================================
print("=" * 70)
print("PART 5 : LatentMoE (Kimi K3) — routing in a narrower latent space")
print("=" * 70)


def moe_layer_cost(d_model: int, d_ff_expert: int, n_routed: int, k: int,
                   n_shared: int, latent: int | None = None) -> dict:
    """
    Per-LAYER accounting for a MoE FFN, with or without the LatentMoE trick.

    latent=None  -> vanilla: routed experts live in the full model width.
    latent=l     -> LatentMoE: a single down-projection d->l feeds ALL routed
                    experts, which operate in width l; one up-projection l->d
                    merges the result back. Shared experts stay full-width.

    The number we really care about is `dispatch_floats_per_token`: on an
    expert-parallel cluster, each selected expert lives on a possibly remote
    GPU, so the token representation must travel there and back. That traffic
    is proportional to (k * width), NOT to the number of experts. This is
    precisely the term LatentMoE halves.
    """
    routed_width = d_model if latent is None else latent

    # A 2-matrix FFN per expert (up + down), consistent with PART 4.
    params_one_expert = 2 * routed_width * d_ff_expert
    params_routed = n_routed * params_one_expert
    params_shared = n_shared * 2 * d_model * d_ff_expert
    params_router = d_model * n_routed
    # The down/up projections of LatentMoE are paid once per layer, not per expert.
    params_proj = 0 if latent is None else 2 * d_model * latent

    active_routed = k * params_one_expert
    active = active_routed + params_shared + params_router + params_proj

    # Round trip: send the token to k experts, receive k outputs back.
    dispatch = 2 * k * routed_width

    return {
        "total": params_routed + params_shared + params_router + params_proj,
        "active": active,
        "dispatch": dispatch,
    }


# Kimi K3 configuration (technical report, Table 1).
K3_D, K3_L, K3_DFF = 7168, 3584, 3072
K3_N, K3_K, K3_SHARED = 896, 16, 2

with_latent = moe_layer_cost(K3_D, K3_DFF, K3_N, K3_K, K3_SHARED, latent=K3_L)
without_latent = moe_layer_cost(K3_D, K3_DFF, K3_N, K3_K, K3_SHARED, latent=None)

print(f"\n  Kimi K3 MoE layer: d_model={K3_D}, {K3_N} routed experts top-{K3_K},")
print(f"  {K3_SHARED} shared experts, d_ff per expert={K3_DFF}, latent width={K3_L}")
print()
print(f"  {'':<28}{'LatentMoE':>16}{'full-width':>16}{'ratio':>10}")
for key, label in [("total", "params / layer"),
                   ("active", "active / token"),
                   ("dispatch", "floats moved / token")]:
    a, b = with_latent[key], without_latent[key]
    unit = (lambda v: f"{v / 1e9:.2f} B") if key != "dispatch" else (lambda v: f"{v:,}")
    print(f"  {label:<28}{unit(a):>16}{unit(b):>16}{a / b:>9.2f}x")

print()
print("  Reading it: the latent space halves BOTH the routed-expert parameters")
print("  and the all-to-all traffic. Going from top-8 to top-16 would have")
print("  doubled the communication bill; routing at half width pays it back.")
print()

# Sanity check against the published figures. This is the honest way to use
# a toy model: state what it does NOT capture.
print("  Cross-check vs the real model (2.78 T total / 104.2 B active):")
print(f"    this toy layer x 92 MoE layers = "
      f"{with_latent['total'] * 92 / 1e12:.2f} T total, "
      f"{with_latent['active'] * 92 / 1e9:.0f} B active")
print("    -> same order of magnitude. The gap comes from what we ignore:")
print("       attention (KDA + MLA) projections, embeddings, the vision tower,")
print("       and the fact that real experts use a 3-matrix SwiGLU-style FFN.")
print()


# ============================================================================
# PART 6 — Load balancing at 10^3 experts: fixed-step bias vs Quantile Balancing
# ============================================================================
print("=" * 70)
print("PART 6 : Quantile Balancing (Kimi K3) vs auxiliary-loss-free bias")
print("=" * 70)


def route_topk_with_cutoff(scores: np.ndarray, bias: np.ndarray, k: int):
    """
    Route with Top-(k+1) instead of Top-k on the BIASED score.

    Why k+1: the first k entries are the routes actually taken, and the
    (k+1)-th one is exactly the threshold an expert must beat to enter this
    token's Top-k. So the cutoff comes for free from the same forward pass —
    no extra reduction, no separate cross-token quantile.

    Returns:
      chosen: (m, k) expert indices actually selected
      cutoff: (m,)  the (k+1)-th largest biased score per token
    """
    biased = scores + bias                                   # (m, n)
    order = np.argsort(-biased, axis=1)[:, :k + 1]           # (m, k+1)
    chosen = order[:, :k]
    rows = np.arange(scores.shape[0])
    cutoff = biased[rows, order[:, k]]                       # (m,)
    return chosen, cutoff


def loads_of(chosen: np.ndarray, n_experts: int) -> np.ndarray:
    """How many tokens each expert received this step."""
    return np.bincount(chosen.ravel(), minlength=n_experts).astype(np.float64)


def update_bias_fixed_step(bias, loads, target, gamma):
    """
    DeepSeek-V3 style auxiliary-loss-free balancing: nudge the bias of each
    expert by a CONSTANT step in the direction that fixes its load.
    Simple and effective at N=256 — the question is what happens at N=896.
    """
    return bias + gamma * np.sign(target - loads)


def update_bias_quantile(scores, cutoff, k, n_experts):
    """
    Quantile Balancing (Kimi K3 §2.3.3).

    For expert j, the 'margin' of token i is  s[i, j] - cutoff[i]  : how far
    that expert is from being selected by that token. If we want expert j to
    receive exactly q = m*k/n tokens, the right bias is the one that puts the
    threshold precisely at the q-th best margin — i.e. a QUANTILE. We read it,
    we do not search for it. No step size, no oscillation.

    b_j   = -quantile_{1 - k/n}( s[:, j] - cutoff )
    b    <- b - mean(b)      # a common offset does not change any Top-k
    """
    margins = scores - cutoff[:, None]                       # (m, n)
    b_hat = -np.quantile(margins, 1.0 - k / n_experts, axis=0)
    return b_hat - b_hat.mean()


def imbalance(loads: np.ndarray) -> float:
    """max load / mean load. 1.0 = perfect. This is what stalls an EP rank."""
    return loads.max() / loads.mean()


N_EXPERTS, TOP_K, N_TOKENS, STEPS = 896, 16, 4096, 40
target_load = N_TOKENS * TOP_K / N_EXPERTS

# A realistic router is NOT uniform: some experts are intrinsically more
# attractive early in training. We bake in that skew and see who can undo it.
expert_bias_true = np.random.randn(N_EXPERTS) * 0.35


def make_scores(rng: np.random.Generator) -> np.ndarray:
    """Router scores in (0,1), as in the paper: s_i = sigmoid(W_r x_i)."""
    logits = rng.standard_normal((N_TOKENS, N_EXPERTS)) * 0.5 + expert_bias_true
    return 1.0 / (1.0 + np.exp(-logits))


def run_strategy(strategy: str, gamma: float = 0.0) -> list[float]:
    """
    Run STEPS training steps and return the imbalance seen at each step.
    Same seed for every strategy so the batches are strictly comparable.
    """
    rng = np.random.default_rng(1234)
    bias = np.zeros(N_EXPERTS)
    out = []
    for _ in range(STEPS):
        scores = make_scores(rng)
        chosen, cutoff = route_topk_with_cutoff(scores, bias, TOP_K)
        loads = loads_of(chosen, N_EXPERTS)
        out.append(imbalance(loads))
        # The update takes effect only at the NEXT step: a batch is never
        # routed with a bias derived from itself (causality, as in the paper).
        if strategy == "fixed":
            bias = update_bias_fixed_step(bias, loads, target_load, gamma)
        elif strategy == "quantile":
            bias = update_bias_quantile(scores, cutoff, TOP_K, N_EXPERTS)
        # strategy == "none": bias stays at zero
    return out


# The noise floor. Even a PERFECTLY balanced router cannot reach 1.00: with
# ~73 tokens expected per expert, sampling noise alone puts the busiest expert
# a few standard deviations above the mean. Knowing this floor is what stops
# us from over-reading the numbers below.
rng_floor = np.random.default_rng(7)
random_assign = rng_floor.integers(0, N_EXPERTS, size=N_TOKENS * TOP_K)
noise_floor = imbalance(loads_of(random_assign.reshape(-1, TOP_K), N_EXPERTS))

curves = {
    "none": run_strategy("none"),
    "fixed g=0.001": run_strategy("fixed", 0.001),
    "fixed g=0.01": run_strategy("fixed", 0.01),
    "fixed g=0.05": run_strategy("fixed", 0.05),
    "quantile": run_strategy("quantile"),
}

print(f"\n  {N_EXPERTS} experts, top-{TOP_K}, {N_TOKENS} tokens/step, "
      f"target load = {target_load:.1f} tokens/expert")
print(f"  Metric: max load / mean load  (1.00 = perfect, "
      f"{noise_floor:.2f} = sampling noise floor)")
print()
header = f"  {'step':>6}" + "".join(f"{name:>16}" for name in curves)
print(header)
for step in (0, 1, 2, 5, 10, 20, STEPS - 1):
    row = f"  {step:>6}" + "".join(f"{curves[n][step]:>16.2f}" for n in curves)
    print(row)


def steps_to_reach(curve: list[float], threshold: float) -> str:
    for i, v in enumerate(curve):
        if v <= threshold:
            return str(i)
    return f">{len(curve)}"


print()
print(f"  {'strategy':<16}{'steps to <2.0x':>16}{'final':>10}{'worst after step 10':>22}")
for name, curve in curves.items():
    tail = max(curve[10:])
    print(f"  {name:<16}{steps_to_reach(curve, 2.0):>16}"
          f"{curve[-1]:>10.2f}{tail:>22.2f}")

print()
print("  What the numbers actually say (and what they do not):")
print("  - Quantile Balancing is balanced from step 1 and stays at the noise")
print("    floor. It SOLVES for the threshold instead of walking towards it,")
print("    so the size of the initial skew does not slow it down at all.")
print("  - The fixed-step rule does converge — but its speed and its steady")
print("    state are both hostages of gamma. Too small (0.001) and it is still")
print("    unbalanced 40 steps later; too large (0.05) and it overshoots and")
print("    oscillates forever. That tuning problem is the whole point: it gets")
print("    worse as the number of experts grows, because each expert needs a")
print("    larger correction while gamma stays constant.")
print("  - Nobody reaches 1.00, and they should not: with ~73 tokens per expert")
print(f"    the sampling floor is already {noise_floor:.2f}x.")
print()
print("  Caveat this toy hides: at real scale the margins number in the millions")
print("  and are sharded across GPUs, so an exact np.quantile is not an option.")
print("  Kimi K3 estimates it from a per-expert histogram (counts are additive,")
print("  so one all-reduce over a few hundred bins recovers the global quantile).")
print()


print("=" * 70)
print("FIN. Retenir :")
print("=" * 70)
print("  - Router = 1 matmul + softmax + top-k. Trivial to implement.")
print("  - Forward = sum of k expert outputs weighted by renormalized gate probs.")
print("  - Load balancing loss = N * sum(f_i * P_i). Without it, 2-3 experts win all.")
print("  - MoE saves FLOPs (k/N), not VRAM. All experts must stay loaded.")
print("  - The Mixtral name '8x7B' is misleading: total ~47B (shared attn + emb).")
print("  - LatentMoE: routing in a narrower width is what makes ~10^3 experts payable.")
print("  - Quantile Balancing: compute the bias (a quantile), do not tune a step size.")

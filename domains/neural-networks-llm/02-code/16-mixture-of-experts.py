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


print("=" * 70)
print("FIN. Retenir :")
print("=" * 70)
print("  - Router = 1 matmul + softmax + top-k. Trivial to implement.")
print("  - Forward = sum of k expert outputs weighted by renormalized gate probs.")
print("  - Load balancing loss = N * sum(f_i * P_i). Without it, 2-3 experts win all.")
print("  - MoE saves FLOPs (k/N), not VRAM. All experts must stay loaded.")
print("  - The Mixtral name '8x7B' is misleading: total ~47B (shared attn + emb).")

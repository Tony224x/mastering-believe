"""
Jour 5 — Attention Mechanism from scratch
===========================================
Pure NumPy first (so it runs anywhere). Then PyTorch for the real thing.
PyTorch import is wrapped in try/except so this script runs even without torch.

Covers:
  1. Scaled dot-product attention in NumPy (forward only)
  2. Self-attention with Q, K, V projections
  3. Causal masking
  4. Multi-head attention
  5. Visualization of an attention matrix on a tiny sentence
  6. Same thing in PyTorch (skipped if torch missing)

Run: python 02-code/05-attention-mechanism.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# PART 1: Scaled Dot-Product Attention in NumPy
# ============================================================================

def softmax(x, axis=-1):
    """Numerically stable softmax along a given axis."""
    # Subtract max for numerical stability — softmax is shift-invariant
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Shapes:
      Q    : (n_q, d_k)
      K    : (n_k, d_k)
      V    : (n_k, d_v)
      mask : optional (n_q, n_k) with 0 = allowed, 1 = blocked
             (we add -inf on blocked positions BEFORE softmax)

    Returns:
      output          : (n_q, d_v)
      attention_weights : (n_q, n_k)  — useful for inspection
    """
    d_k = Q.shape[-1]

    # Step 1: similarity scores between each query and each key
    # scores[i, j] = Q_i . K_j
    scores = Q @ K.T  # (n_q, n_k)

    # Step 2: scale by sqrt(d_k)
    # WHY: without scaling, variance of dot products grows with d_k.
    # Large scores make softmax output a near-one-hot vector — gradient vanishes.
    # Dividing by sqrt(d_k) keeps the std ~1 regardless of dimension.
    scores = scores / np.sqrt(d_k)

    # Step 3: apply mask if provided (set blocked positions to -inf)
    if mask is not None:
        scores = np.where(mask == 1, -np.inf, scores)

    # Step 4: softmax over the keys axis
    # Each row of `weights` is a probability distribution that sums to 1
    weights = softmax(scores, axis=-1)

    # Step 5: weighted sum of values
    output = weights @ V  # (n_q, d_v)

    return output, weights


# ============================================================================
# PART 2: Self-Attention layer (Q, K, V projected from the same input)
# ============================================================================

class SelfAttentionNumpy:
    """
    Single-head self-attention layer (NumPy).
    Input:  X of shape (seq_len, d_model)
    Output: (seq_len, d_model)
    """

    def __init__(self, d_model, d_k=None, d_v=None, seed=0):
        rng = np.random.RandomState(seed)
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        # Three separate projection matrices — the model learns distinct roles
        # for queries, keys, and values.
        # WHY small init: keeps the dot products in a "sane" range at step 0.
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.randn(d_model, self.d_k) * scale
        self.W_K = rng.randn(d_model, self.d_k) * scale
        self.W_V = rng.randn(d_model, self.d_v) * scale

    def forward(self, X, mask=None):
        # Project input into Q, K, V subspaces
        Q = X @ self.W_Q  # (seq_len, d_k)
        K = X @ self.W_K  # (seq_len, d_k)
        V = X @ self.W_V  # (seq_len, d_v)
        return scaled_dot_product_attention(Q, K, V, mask=mask)


# ============================================================================
# PART 3: Causal mask (for autoregressive generation)
# ============================================================================

def causal_mask(seq_len):
    """
    Returns a (seq_len, seq_len) mask with 1s in the upper triangle (excluding
    diagonal). Position i is allowed to attend to positions 0..i (inclusive).

    Example (seq_len=4):
      0 1 1 1      <- pos 0 can only see itself
      0 0 1 1      <- pos 1 can see 0, 1
      0 0 0 1      <- pos 2 can see 0, 1, 2
      0 0 0 0      <- pos 3 can see 0, 1, 2, 3
    """
    # np.triu(..., k=1) = upper triangle, k=1 means ABOVE the diagonal
    return np.triu(np.ones((seq_len, seq_len), dtype=np.int32), k=1)


# ============================================================================
# PART 4: Multi-Head Attention in NumPy
# ============================================================================

class MultiHeadAttentionNumpy:
    """
    Multi-head self-attention.
    d_model = n_heads * d_head
    """

    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(d_model)

        # One big projection matrix for Q, K, V each — we'll reshape into heads
        self.W_Q = rng.randn(d_model, d_model) * scale
        self.W_K = rng.randn(d_model, d_model) * scale
        self.W_V = rng.randn(d_model, d_model) * scale

        # Output projection — mixes information across heads
        self.W_O = rng.randn(d_model, d_model) * scale

    def forward(self, X, mask=None):
        seq_len, _ = X.shape

        # Project and reshape into heads
        Q = X @ self.W_Q  # (seq, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Reshape: (seq, d_model) -> (seq, n_heads, d_head) -> (n_heads, seq, d_head)
        # WHY transpose: we want to do one independent attention per head
        Q = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        head_outputs = []
        all_weights = []
        for h in range(self.n_heads):
            out, w = scaled_dot_product_attention(Q[h], K[h], V[h], mask=mask)
            head_outputs.append(out)   # (seq, d_head)
            all_weights.append(w)      # (seq, seq)

        # Concatenate along last dim: (seq, n_heads * d_head) = (seq, d_model)
        concat = np.concatenate(head_outputs, axis=-1)

        # Final output projection
        output = concat @ self.W_O  # (seq, d_model)

        return output, np.stack(all_weights, axis=0)  # weights: (n_heads, seq, seq)


# ============================================================================
# PART 5: Visualize an attention matrix as ASCII
# ============================================================================

def print_attention_matrix(weights, tokens, title=""):
    """
    Print a (seq, seq) attention matrix as ASCII with token labels.
    Shows which token attends to which.
    """
    print(f"\n  {title}")
    n = len(tokens)
    # Header: destination tokens (what each query attends to)
    header = "         " + " ".join(f"{t:>6s}" for t in tokens)
    print(header)
    for i in range(n):
        row_vals = " ".join(f"{weights[i, j]:>6.3f}" for j in range(n))
        print(f"  {tokens[i]:>6s} | {row_vals}")


# ============================================================================
# MAIN DEMO (NumPy)
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PART 1: Scaled Dot-Product Attention on a toy example")
    print("=" * 70)

    # 3 queries, 4 keys/values, each of dim 4
    Q = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    K = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    V = np.array([
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
        [40.0, 0.0],
    ])

    out, w = scaled_dot_product_attention(Q, K, V)
    print(f"\nQ shape : {Q.shape}")
    print(f"K shape : {K.shape}")
    print(f"V shape : {V.shape}")
    print(f"Output  : {out}")
    print(f"Weights (rows sum to 1):")
    for i, row in enumerate(w):
        print(f"  query {i}: {np.round(row, 3).tolist()}  (sum={row.sum():.3f})")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: Self-attention on a tiny sentence")
    print("=" * 70)

    # A fake 5-token sentence — each token is a 8-dim embedding (random)
    tokens = ["le", "chat", "noir", "dort", "."]
    seq_len = len(tokens)
    d_model = 8
    X = np.random.randn(seq_len, d_model) * 0.5

    sa = SelfAttentionNumpy(d_model=d_model, seed=1)
    out, w = sa.forward(X)

    print(f"\nInput X shape  : {X.shape}")
    print(f"Output shape   : {out.shape}")
    print_attention_matrix(w, tokens, title="Self-attention weights (random init):")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: Causal mask for autoregressive generation")
    print("=" * 70)

    cmask = causal_mask(seq_len)
    print(f"\nCausal mask (1 = blocked, 0 = allowed):")
    for row in cmask:
        print(f"  {row.tolist()}")

    out_causal, w_causal = sa.forward(X, mask=cmask)
    print_attention_matrix(w_causal, tokens, title="Self-attention WITH causal mask:")
    print("\n  Observation: upper triangle is 0 — each token only attends to itself and the past.")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 4: Multi-head attention")
    print("=" * 70)

    n_heads = 4
    mha = MultiHeadAttentionNumpy(d_model=d_model, n_heads=n_heads, seed=2)
    out_mh, w_mh = mha.forward(X)

    print(f"\nInput shape        : {X.shape}")
    print(f"Output shape       : {out_mh.shape}")
    print(f"Weights shape      : {w_mh.shape}   (n_heads, seq, seq)")
    print(f"Each head has d_head = {mha.d_head}")

    for h in range(n_heads):
        print_attention_matrix(w_mh[h], tokens, title=f"Head {h} attention:")

    print("\n  Each head sees a different pattern (random init → random patterns).")
    print("  After training, heads specialize (coref, syntax, adjacency, etc.).")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 5: Effect of scaling on softmax sharpness")
    print("=" * 70)

    # Demonstrate why we divide by sqrt(d_k)
    for d_k in [4, 64, 512]:
        Q_demo = np.random.randn(1, d_k)
        K_demo = np.random.randn(5, d_k)
        scores_unscaled = (Q_demo @ K_demo.T)[0]
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        w_unscaled = softmax(scores_unscaled)
        w_scaled = softmax(scores_scaled)
        print(f"\n  d_k = {d_k}")
        print(f"    scores (unscaled)   : std = {scores_unscaled.std():6.2f}")
        print(f"    scores (scaled)     : std = {scores_scaled.std():6.2f}")
        print(f"    softmax (unscaled)  : {np.round(w_unscaled, 3).tolist()}")
        print(f"    softmax (scaled)    : {np.round(w_scaled, 3).tolist()}")
        print(f"    entropy (unscaled)  : {-np.sum(w_unscaled * np.log(w_unscaled + 1e-9)):.3f}")
        print(f"    entropy (scaled)    : {-np.sum(w_scaled * np.log(w_scaled + 1e-9)):.3f}")

    print("\n  Observation: as d_k grows, unscaled softmax becomes one-hot (low entropy),")
    print("  killing the gradient. Scaled version stays soft.")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 6: Same thing in PyTorch (if available)")
    print("=" * 70)

    try:
        import torch
        import torch.nn.functional as F

        print(f"\n  PyTorch version: {torch.__version__}")

        # Same sentence, but as a torch tensor
        X_t = torch.tensor(X, dtype=torch.float32)

        # Use the built-in F.scaled_dot_product_attention
        # We need (batch, heads, seq, dim) shape for it
        d_k = d_model
        Q_t = X_t.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d_model)
        K_t = X_t.unsqueeze(0).unsqueeze(0)
        V_t = X_t.unsqueeze(0).unsqueeze(0)

        # Without mask
        out_t = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        print(f"  Output shape (no mask)     : {tuple(out_t.shape)}")

        # With causal mask
        out_t_causal = F.scaled_dot_product_attention(Q_t, K_t, V_t, is_causal=True)
        print(f"  Output shape (causal mask) : {tuple(out_t_causal.shape)}")

        # Manual implementation for pedagogy
        def attention_torch(Q, K, V, mask=None):
            """Q, K, V: (seq, d_k). Returns (seq, d_v), (seq, seq)."""
            d_k = Q.shape[-1]
            scores = Q @ K.T / (d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 1, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            return weights @ V, weights

        mask_t = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        out_manual, w_manual = attention_torch(X_t, X_t, X_t, mask=mask_t)
        print(f"  Manual torch attention output shape: {tuple(out_manual.shape)}")
        print(f"  Manual weights row 0 (token 'le'): {np.round(w_manual[0].numpy(), 3).tolist()}")
        print(f"  Expected: [1.0, 0, 0, 0, 0] — first token only sees itself.")

    except ImportError:
        print("\n  PyTorch is not installed. Skipping PyTorch section.")
        print("  To enable: pip install torch")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
  Attention = softmax(Q K^T / sqrt(d_k)) V
    - Q, K, V are learned projections of the input
    - Masking adds -inf to blocked positions before softmax
    - Multi-head: h parallel attentions in d_model/h subspaces, then concat
    - Scaling by sqrt(d_k) is NOT optional — it's what keeps the gradient alive

  Tomorrow (J6): the full Transformer block = MHA + FFN + LayerNorm + residuals.
""")

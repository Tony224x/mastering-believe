"""
Solutions — Jour 5 : Attention Mechanism
=========================================
Solutions for the 8 exercises (easy, medium, hard).

Run: python 03-exercises/solutions/05-attention-mechanism.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.set_printoptions(precision=4, suppress=True)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# EXERCISE 1: Single attention head by hand
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Compute one attention head by hand")
print("=" * 70)

Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])
K = np.array([[1.0, 0.0],
              [0.0, 1.0]])
V = np.array([[10.0, 0.0],
              [0.0, 10.0]])

# Step 1: Q @ K^T
scores = Q @ K.T
print(f"\n--- Step 1: Q @ K^T ---")
print(f"  scores = {scores.tolist()}")
print(f"  -> identity matrix because Q and K are identity")

# Step 2: scale
d_k = 2
scores_scaled = scores / np.sqrt(d_k)
print(f"\n--- Step 2: divide by sqrt(d_k) = sqrt(2) ~ 1.414 ---")
print(f"  scores_scaled = {scores_scaled}")
print(f"  max value = {1.0/np.sqrt(2):.4f} (= 1/sqrt(2))")

# Step 3: softmax
weights = softmax(scores_scaled, axis=-1)
print(f"\n--- Step 3: softmax over each row ---")
print(f"  weights = {weights}")
print(f"  row 0 sum = {weights[0].sum():.4f}")
print(f"  row 1 sum = {weights[1].sum():.4f}")

# Manual softmax for row 0: [exp(1/sqrt(2)), exp(0)] / sum
e1 = np.exp(1.0 / np.sqrt(2))
e2 = np.exp(0.0)
print(f"  Manual row 0: exp(0.707)={e1:.4f}, exp(0)={e2:.4f}")
print(f"  -> [{e1/(e1+e2):.4f}, {e2/(e1+e2):.4f}]")

# Step 4: output
output = weights @ V
print(f"\n--- Step 4: output = weights @ V ---")
print(f"  output = {output}")
print(f"  -> query 0 gets mostly V_0 (~10, 0), query 1 gets mostly V_1 (~0, 10)")

print("\n--- Q5: Interpretation ---")
print("  Query 0 attends mostly to V_0 because Q_0 aligns with K_0 (dot product = 1).")
print("  Query 1 attends mostly to V_1 because Q_1 aligns with K_1.")
print("  This is exactly what attention is meant to do: route values based on Q-K similarity.")

# Bonus: without scaling
print("\n--- Bonus: without the sqrt(d_k) scaling ---")
weights_unscaled = softmax(scores, axis=-1)
print(f"  weights unscaled = {weights_unscaled}")
print(f"  Row 0 peak value : {weights_unscaled[0, 0]:.4f} (vs {weights[0, 0]:.4f} with scaling)")
print("  -> without scaling, scores are larger -> softmax is SHARPER (more peaked).")
print("  With higher d_k, the difference becomes dramatic (softmax -> one-hot).")


# ============================================================================
# EXERCISE 2: Compute Q, K, V from input and projection matrices
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Compute Q, K, V from X and W_Q, W_K, W_V")
print("=" * 70)

X = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]], dtype=float)

W_Q = np.array([[1, 0],
                [0, 1],
                [0, 0],
                [0, 0]], dtype=float)

W_K = np.array([[0, 0],
                [0, 0],
                [1, 0],
                [0, 1]], dtype=float)

W_V = np.array([[1, 0],
                [1, 0],
                [0, 1],
                [0, 1]], dtype=float)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print(f"\n  X = \n{X}")
print(f"\n  Q = X @ W_Q =\n{Q}")
print(f"\n  K = X @ W_K =\n{K}")
print(f"\n  V = X @ W_V =\n{V}")

# Step 4: scores
scores = Q @ K.T / np.sqrt(2)
print(f"\n  scores (Q @ K^T / sqrt(2)) =\n{scores}")

weights = softmax(scores, axis=-1)
print(f"\n  attention weights =\n{weights}")

# Token "le" has Q = [1, 0]
# K[0] = [1, 0] (from "le")       -> dot = 1
# K[1] = [0, 1] (from "chat")     -> dot = 0
# K[2] = [0, 0] (from "dort")     -> dot = 0
print("\n--- Q5: For 'le' (row 0), which key has the highest score? ---")
print(f"  Row 0 of Q @ K^T: {(Q @ K.T)[0].tolist()}")
print("  -> K[0] (from 'le' itself) has the highest score.")
print("  Reason: W_Q and W_K project the first 2 dims of X for queries,")
print("  and the last 2 dims of X for keys. The token 'le' has [1,0,1,0]")
print("  so its Q = [1, 0] matches K[0] = [1, 0] by chance.")

print("""
--- Q6: Why 3 separate W_Q, W_K, W_V matrices? ---

  The three roles are fundamentally different:
    Q: "what am I looking for?" (represents the current token's query)
    K: "how do I advertise myself?" (makes me findable)
    V: "what do I return if you pick me?" (the content I provide)

  Having 3 separate projections lets the model learn distinct behaviors for
  each role. A single projection would force Q = K = V, which wastes capacity:
  e.g., in a dictionary lookup, the key "chat" is different from its value
  "cat" — they serve different purposes.

  Empirically, the Transformer with distinct W_Q/W_K/W_V is much more expressive.
""")


# ============================================================================
# EXERCISE 3: Sinusoidal positional encoding by hand
# ============================================================================

print("=" * 70)
print("EXERCISE 3: Sinusoidal positional encoding")
print("=" * 70)

d_model = 4
max_pos = 4

# Build PE matrix
PE = np.zeros((max_pos, d_model))
for pos in range(max_pos):
    for dim in range(d_model):
        # i is the index of the sin/cos pair
        i = dim // 2
        denom = 10000 ** (2 * i / d_model)
        if dim % 2 == 0:
            PE[pos, dim] = np.sin(pos / denom)
        else:
            PE[pos, dim] = np.cos(pos / denom)

print(f"\nPE matrix shape: {PE.shape}")
print(f"\nPE =\n{PE}")

# Q1: for each dimension, describe parity, i, denom, function
print("\n--- Q1: Dimension-by-dimension breakdown ---")
print("  dim  | parity | i | denominator         | function")
print("  -----+--------+---+---------------------+---------")
for dim in range(d_model):
    parity = "even" if dim % 2 == 0 else "odd"
    i = dim // 2
    denom = 10000 ** (2 * i / d_model)
    func = "sin" if dim % 2 == 0 else "cos"
    print(f"  {dim:3d}  | {parity:>6s} | {i} | {denom:>19.4f} | {func}")

print("\n--- Q2: PE(pos=0) ---")
print(f"  PE(0) = {PE[0].tolist()}")
print("  sin(0) = 0, cos(0) = 1")
print("  -> [0, 1, 0, 1] (alternating)")

print("\n--- Q3: Low dims vary fast, high dims vary slowly ---")
print(f"  PE(0, dim=0) = {PE[0, 0]:.4f}, PE(3, dim=0) = {PE[3, 0]:.4f}")
print(f"     -> variation : {abs(PE[3, 0] - PE[0, 0]):.4f}")
print(f"  PE(0, dim=2) = {PE[0, 2]:.4f}, PE(3, dim=2) = {PE[3, 2]:.4f}")
print(f"     -> variation : {abs(PE[3, 2] - PE[0, 2]):.4f}")
print("  Low dims (0, 1) use small denominators -> high frequency -> fast variation")
print("  High dims (2, 3) use large denominators -> low frequency -> slow variation")

print("""
--- Q4: Why multi-frequency sinusoids instead of scalar positions? ---

  Using position as a scalar (0, 1, 2, ..., 1000) has two problems:

  1. Unbounded magnitude: position 1000 would dominate the embedding vector
     in amplitude, overshadowing the content.

  2. No exploitable structure: the model can't easily compute "distance between
     positions" or "relative offset" from a single scalar.

  Sinusoidal PE encodes position as a multi-scale clock: each dim is a sin/cos
  at a different frequency. The model can combine these to extract:
    - Absolute position (from the combination of all frequencies)
    - Relative position (differences are easy to compute in sin/cos space)
  And the values stay bounded in [-1, 1] regardless of position.

--- Q5 (Bonus): Linearity trick for relative positions ---

  sin(pos + k) = sin(pos)*cos(k) + cos(pos)*sin(k)
  cos(pos + k) = cos(pos)*cos(k) - sin(pos)*sin(k)

  So PE(pos + k) can be expressed as a LINEAR transformation of PE(pos),
  where the transformation depends only on k. The model can learn "shift by k"
  as a linear operation — this is thought to help extrapolation to sequence
  lengths longer than those seen during training.

  Note: in practice, learned positional embeddings (used in GPT-2, BERT)
  perform similarly to sinusoidal ones. Both are still used.
  Modern models often use rotary position embeddings (RoPE, LLaMA).
""")

# ============================================================================
# EXERCISE 4 (MEDIUM): Causal scaled dot-product attention from scratch
# ============================================================================

print("=" * 70)
print("EXERCISE 4 (MEDIUM): Batched causal attention + properties")
print("=" * 70)


def causal_attention(Q, K, V):
    """Q, K, V: (batch, n_heads, T, d_k). Fully vectorized.
    Mask is applied BEFORE softmax with -inf (multiplying weights by 0
    AFTER softmax would break row normalization)."""
    d_k = Q.shape[-1]
    T = Q.shape[-2]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # (B, H, T, T)
    causal = np.tril(np.ones((T, T), dtype=bool))         # True = allowed
    scores = np.where(causal, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


rng4 = np.random.default_rng(0)
B4, H4, T4, dk4 = 2, 2, 6, 8
Q4 = rng4.standard_normal((B4, H4, T4, dk4))
K4 = rng4.standard_normal((B4, H4, T4, dk4))
V4 = rng4.standard_normal((B4, H4, T4, dk4))

out4, w4 = causal_attention(Q4, K4, V4)
assert out4.shape == (B4, H4, T4, dk4) and w4.shape == (B4, H4, T4, T4)

# Property 1: rows sum to 1
row_sums = w4.sum(axis=-1)
print(f"\n  normalization: max |row_sum - 1| = {np.abs(row_sums - 1).max():.2e}")
assert np.abs(row_sums - 1).max() < 1e-9

# Property 2: strict causality of the weights (upper triangle exactly 0)
upper = w4[..., ~np.tril(np.ones((T4, T4), dtype=bool))]
print(f"  causality: max weight above diagonal = {np.abs(upper).max():.2e}")
assert np.abs(upper).max() == 0.0

# Property 3: no leakage — perturbing the LAST token's V must not change
# the outputs of earlier positions
V4b = V4.copy()
V4b[..., -1, :] += 100.0
out4b, _ = causal_attention(Q4, K4, V4b)
leak = np.abs(out4b[..., :-1, :] - out4[..., :-1, :]).max()
print(f"  no-leak: max change on positions 0..T-2 = {leak:.2e}")
assert leak < 1e-12
print("  [PASS] normalization, causality, no-leak")

# Scaling effect: entropy of softmax rows with / without 1/sqrt(d_k), d_k=64
# Score variance is ~d_k for N(0,1) Q, K -> without scaling the softmax
# saturates (low entropy), gradients die.
rng4b = np.random.default_rng(1)
dk_big, T_big = 64, 16
Qb = rng4b.standard_normal((T_big, dk_big))
Kb = rng4b.standard_normal((T_big, dk_big))
scores_raw = Qb @ Kb.T


def mean_row_entropy(scores):
    p = softmax(scores, axis=-1)
    return float(np.mean(-np.sum(p * np.log(p + 1e-12), axis=-1)))


H_scaled = mean_row_entropy(scores_raw / np.sqrt(dk_big))
H_unscaled = mean_row_entropy(scores_raw)
print(f"\n  d_k=64: mean row entropy scaled = {H_scaled:.3f} nats, "
      f"unscaled = {H_unscaled:.3f} nats")
assert H_scaled - H_unscaled > 1.0
print("  [PASS] scaling keeps the softmax soft (entropy gap > 1 nat)")


# ============================================================================
# EXERCISE 5 (MEDIUM): Multi-head split/merge shape gymnastics
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5 (MEDIUM): split_heads / merge_heads")
print("=" * 70)

# Paper predictions for batch=2, T=10, d_model=512, n_heads=8:
#   after X @ W_Q     : (2, 10, 512)
#   after split_heads : (2, 8, 10, 64)
#   attention scores  : (2, 8, 10, 10)
#   after merge_heads : (2, 10, 512)


def split_heads(x, n_heads):
    """(B, T, d_model) -> (B, n_heads, T, d_head).
    reshape FIRST (split last dim into heads), THEN transpose. Doing a
    direct reshape to (B, n_heads, T, d_head) would slice along T and mix
    timesteps across heads."""
    B, T, d_model = x.shape
    d_head = d_model // n_heads
    return x.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, n_heads, T, d_head) -> (B, T, d_model). Exact inverse."""
    B, n_heads, T, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, n_heads * d_head)


rng5 = np.random.default_rng(2)
B5, T5, dm5, nh5 = 2, 10, 512, 8
X5 = rng5.standard_normal((B5, T5, dm5))

split5 = split_heads(X5, nh5)
assert split5.shape == (2, 8, 10, 64)
merged5 = merge_heads(split5)
assert merged5.shape == (2, 10, 512)

# Round-trip must be EXACT (pure data movement, no arithmetic)
assert np.array_equal(merged5, X5)
print("\n  merge(split(x)) == x exactly  [PASS]")

# The WRONG version: reshape without transpose mixes timesteps
wrong5 = X5.reshape(B5, nh5, T5, dm5 // nh5)
print(f"  naive reshape == correct split? {np.array_equal(wrong5, split5)} (expected False)")
assert not np.array_equal(wrong5, split5)

# Each head must see contiguous columns of x
for h in range(nh5):
    d_head = dm5 // nh5
    assert np.array_equal(split5[:, h], X5[:, :, h * d_head:(h + 1) * d_head])
print("  per-head contiguity: head h == x[..., h*d_head:(h+1)*d_head]  [PASS]")

# Full multi-head forward with random projections
Wq5 = rng5.standard_normal((dm5, dm5)) * 0.02
Wk5 = rng5.standard_normal((dm5, dm5)) * 0.02
Wv5 = rng5.standard_normal((dm5, dm5)) * 0.02
Wo5 = rng5.standard_normal((dm5, dm5)) * 0.02

q5 = split_heads(X5 @ Wq5, nh5); assert q5.shape == (2, 8, 10, 64)
k5 = split_heads(X5 @ Wk5, nh5)
v5 = split_heads(X5 @ Wv5, nh5)
att5, w5 = causal_attention(q5, k5, v5)
assert w5.shape == (2, 8, 10, 10)
out5 = merge_heads(att5) @ Wo5
assert out5.shape == (2, 10, 512)
print(f"  full MHA forward: output shape {out5.shape}  [PASS]")


# ============================================================================
# EXERCISE 6 (MEDIUM): Debugging a broken attention
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6 (MEDIUM): Three silent attention bugs")
print("=" * 70)

# BUG 1: scores / d_k        -> should be / sqrt(d_k). Over-shrinks scores:
#         attention becomes too uniform, the model can't focus.
# BUG 2: softmax(axis=0)     -> normalizes COLUMNS instead of rows: each
#         query no longer holds a probability distribution over keys.
# BUG 3: mask applied AFTER softmax by multiplication -> rows no longer sum
#         to 1: probability mass leaks, outputs are under-scaled, and future
#         information already influenced the softmax denominator.


def attention_buggy(Q, K, V, causal=True):
    T, d_k = Q.shape
    scores = Q @ K.T / d_k
    weights = softmax(scores, axis=0)
    if causal:
        mask = np.tril(np.ones((T, T)))
        weights = weights * mask
    return weights @ V


def attention_fixed(Q, K, V, causal=True):
    T, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)          # FIX 1
    if causal:
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)   # FIX 3: mask BEFORE softmax
    return softmax(scores, axis=-1) @ V       # FIX 2: rows


def attention_reference(Q, K, V, causal=True):
    """Independent reference: explicit per-row loop."""
    T, d_k = Q.shape
    out = np.zeros_like(V[:T])
    for i in range(T):
        limit = i + 1 if causal else T
        s = np.array([Q[i] @ K[j] / np.sqrt(d_k) for j in range(limit)])
        p = np.exp(s - s.max()); p = p / p.sum()
        out[i] = sum(p[j] * V[j] for j in range(limit))
    return out


rng6 = np.random.default_rng(5)
T6, dk6 = 5, 64
Q6 = rng6.standard_normal((T6, dk6))
K6 = rng6.standard_normal((T6, dk6))
V6 = rng6.standard_normal((T6, dk6))

diff_fix = np.abs(attention_fixed(Q6, K6, V6) - attention_reference(Q6, K6, V6)).max()
print(f"\n  fixed vs independent reference: max diff = {diff_fix:.2e}")
assert diff_fix < 1e-8

# Targeted test 1 (scaling): with Q == K, the diagonal must dominate
# reasonably; the /d_k version flattens everything. Compare to reference.
diff_bug = np.abs(attention_buggy(Q6, K6, V6) - attention_reference(Q6, K6, V6)).max()
print(f"  buggy  vs reference: max diff = {diff_bug:.2e} (clearly wrong)")
assert diff_bug > 1e-3

# Targeted test 2 (softmax axis): rows must sum to 1, not columns


def get_weights(fn_style, Q, K, V):
    T, d_k = Q.shape
    if fn_style == 'buggy':
        w = softmax(Q @ K.T / d_k, axis=0)
        w = w * np.tril(np.ones((T, T)))
    else:
        s = np.where(np.tril(np.ones((T, T), dtype=bool)),
                     Q @ K.T / np.sqrt(d_k), -np.inf)
        w = softmax(s, axis=-1)
    return w


w_b = get_weights('buggy', Q6, K6, V6)
w_f = get_weights('fixed', Q6, K6, V6)
print(f"  buggy row sums: {np.round(w_b.sum(axis=-1), 3)}  (should all be 1!)")
print(f"  fixed row sums: {np.round(w_f.sum(axis=-1), 3)}")
assert np.abs(w_f.sum(axis=-1) - 1).max() < 1e-9
assert np.abs(w_b.sum(axis=-1) - 1).max() > 1e-3

# Targeted test 3 (mask): fixed version stays normalized WITH the mask —
# impossible when masking after softmax without renormalizing.
# Causality of the fixed version: token i insensitive to tokens > i
V6b = V6.copy(); V6b[-1] += 50
leak6 = np.abs(attention_fixed(Q6, K6, V6b)[:-1] - attention_fixed(Q6, K6, V6)[:-1]).max()
assert leak6 < 1e-12
print(f"  fixed causality leak = {leak6:.1e}  [PASS]")
print("  [PASS] each targeted test fails on buggy, passes on fixed")


# ============================================================================
# EXERCISE 7 (HARD): MHA forward + backward with gradient check
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7 (HARD): Multi-Head Attention backward, gradient-checked")
print("=" * 70)


class MultiHeadAttention:
    """NumPy MHA with full manual backward. No biases. Causal."""

    def __init__(self, d_model, n_heads, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        s = 0.3
        self.Wq = rng.standard_normal((d_model, d_model)) * s
        self.Wk = rng.standard_normal((d_model, d_model)) * s
        self.Wv = rng.standard_normal((d_model, d_model)) * s
        self.Wo = rng.standard_normal((d_model, d_model)) * s

    def _split(self, x):
        T = x.shape[0]
        return x.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, T, dh)

    def _merge(self, x):
        H, T, dh = x.shape
        return x.transpose(1, 0, 2).reshape(T, H * dh)

    def forward(self, X):
        self.X = X
        T = X.shape[0]
        self.Q = self._split(X @ self.Wq)   # (H, T, dh)
        self.K = self._split(X @ self.Wk)
        self.V = self._split(X @ self.Wv)
        self.mask = np.tril(np.ones((T, T), dtype=bool))
        S = self.Q @ np.swapaxes(self.K, -1, -2) / np.sqrt(self.d_head)
        S = np.where(self.mask, S, -np.inf)
        self.P = softmax(S, axis=-1)        # (H, T, T), masked entries -> 0
        self.O = self.P @ self.V            # (H, T, dh)
        self.merged = self._merge(self.O)   # (T, d_model)
        return self.merged @ self.Wo

    def backward(self, dout):
        """dout: (T, d_model) = dL/d(output). Returns grads + dX."""
        dWo = self.merged.T @ dout
        dmerged = dout @ self.Wo.T
        dO = self._split(dmerged)                       # (H, T, dh)

        dP = dO @ np.swapaxes(self.V, -1, -2)           # (H, T, T)
        dV = np.swapaxes(self.P, -1, -2) @ dO           # (H, T, dh)

        # Softmax backward (full Jacobian, row-wise):
        # dS = P * (dP - sum(dP * P, axis=-1, keepdims=True))
        dS = self.P * (dP - np.sum(dP * self.P, axis=-1, keepdims=True))
        # Masked positions have P == 0 -> dS == 0 automatically (checked below)
        self.dS = dS

        dS_scaled = dS / np.sqrt(self.d_head)
        dQ = dS_scaled @ self.K                         # (H, T, dh)
        dK = np.swapaxes(dS_scaled, -1, -2) @ self.Q

        dQm, dKm, dVm = self._merge(dQ), self._merge(dK), self._merge(dV)
        dWq = self.X.T @ dQm
        dWk = self.X.T @ dKm
        dWv = self.X.T @ dVm
        dX = dQm @ self.Wq.T + dKm @ self.Wk.T + dVm @ self.Wv.T
        return {'Wq': dWq, 'Wk': dWk, 'Wv': dWv, 'Wo': dWo}, dX


rng7 = np.random.default_rng(7)
T7, dm7, nh7 = 4, 8, 2
X7 = rng7.standard_normal((T7, dm7)) * 0.5
G7 = rng7.standard_normal((T7, dm7))    # fixed matrix for scalar loss
mha7 = MultiHeadAttention(dm7, nh7, seed=1)

# Forward equivalence vs a simple per-head reference
out7 = mha7.forward(X7)
ref_heads = []
for h in range(nh7):
    dh = dm7 // nh7
    q = (X7 @ mha7.Wq)[:, h*dh:(h+1)*dh]
    k = (X7 @ mha7.Wk)[:, h*dh:(h+1)*dh]
    v = (X7 @ mha7.Wv)[:, h*dh:(h+1)*dh]
    ref_heads.append(attention_fixed(q, k, v, causal=True))
ref7 = np.concatenate(ref_heads, axis=-1) @ mha7.Wo
print(f"\n  forward vs simple reference: max diff = {np.abs(out7 - ref7).max():.2e}")
assert np.abs(out7 - ref7).max() < 1e-12

# Backward + gradient check on L = sum(forward(X) * G)
grads7, dX7 = mha7.backward(G7)

# Masked score gradients must be exactly zero
masked_grads = mha7.dS[:, ~mha7.mask]
assert np.abs(masked_grads).max() == 0.0
print("  gradients of masked scores are exactly 0  [PASS]")


def loss7():
    return float(np.sum(mha7.forward(X7) * G7))


print(f"  {'param':>5} | {'max rel err':>12}")
worst7 = 0.0
for name in ['Wq', 'Wk', 'Wv', 'Wo']:
    gn = numerical_grad7 = np.zeros_like(getattr(mha7, name))
    arr = getattr(mha7, name)
    eps = 1e-5
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        arr[idx] = old + eps; lp = loss7()
        arr[idx] = old - eps; lm = loss7()
        arr[idx] = old
        gn[idx] = (lp - lm) / (2 * eps)
    e = (np.abs(grads7[name] - gn) /
         (np.abs(grads7[name]) + np.abs(gn) + 1e-8)).max()
    worst7 = max(worst7, e)
    print(f"  {name:>5} | {e:>12.2e}")

# dX check
gnX = np.zeros_like(X7)
eps = 1e-5
for idx in np.ndindex(X7.shape):
    old = X7[idx]
    X7[idx] = old + eps; lp = loss7()
    X7[idx] = old - eps; lm = loss7()
    X7[idx] = old
    gnX[idx] = (lp - lm) / (2 * eps)
mha7.forward(X7)   # restore caches
eX = (np.abs(dX7 - gnX) / (np.abs(dX7) + np.abs(gnX) + 1e-8)).max()
worst7 = max(worst7, eX)
print(f"  {'dX':>5} | {eX:>12.2e}")
assert worst7 < 1e-5
print(f"  [PASS] full MHA gradient check (worst rel err {worst7:.2e})")
# Bonus note: dX[0] is non-zero even though position 0 only attends to
# itself — its K and V contribute to the outputs of positions 1..T-1, so
# the gradient flows BACK UP the causal mask through K and V.


# ============================================================================
# EXERCISE 8 (HARD): Cross-attention + padding masks
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 (HARD): Cross-attention & padding")
print("=" * 70)


def cross_attention(Q_dec, K_enc, V_enc, src_pad_mask):
    """Q_dec: (T_dec, d), K_enc/V_enc: (T_enc, d), src_pad_mask: (T_enc,) bool.
    No causal mask: the decoder may look at the WHOLE encoder output.
    Padded positions get weight exactly 0 (masked with -inf before softmax)."""
    d = Q_dec.shape[-1]
    scores = Q_dec @ K_enc.T / np.sqrt(d)              # (T_dec, T_enc)
    scores = np.where(src_pad_mask[None, :], scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ V_enc, weights


# --- Constructed retrieval test ---
d8 = 4
K_enc8 = np.eye(d8) * 10.0                  # 4 near-orthogonal keys
V_enc8 = np.arange(16, dtype=float).reshape(4, 4)
rng8 = np.random.default_rng(8)
target_idx = 2
Q_dec8 = (K_enc8[target_idx] + rng8.standard_normal(d8) * 0.1)[None, :]

out8, w8 = cross_attention(Q_dec8, K_enc8, V_enc8, np.ones(4, dtype=bool))
print(f"\n  retrieval: attention weights = {np.round(w8[0], 4)}")
assert w8[0, target_idx] > 0.95
cos8 = (out8[0] @ V_enc8[target_idx]) / (
    np.linalg.norm(out8[0]) * np.linalg.norm(V_enc8[target_idx]))
print(f"  cosine(output, V[target]) = {cos8:.4f}")
assert cos8 > 0.99
print("  [PASS] query retrieves the matching key/value")

# --- Padding test: batch of 2 encoder sequences, real lengths {3, 5} ---
T_enc8, T_dec8 = 5, 3
lengths8 = [3, 5]
rngp = np.random.default_rng(9)
for b, L in enumerate(lengths8):
    pad_mask = np.arange(T_enc8) < L                    # True = real token
    K_b = rngp.standard_normal((T_enc8, d8))
    V_b = rngp.standard_normal((T_enc8, d8))
    Q_b = rngp.standard_normal((T_dec8, d8))

    out_b, w_b = cross_attention(Q_b, K_b, V_b, pad_mask)
    assert np.abs(w_b[:, ~pad_mask]).max() == 0.0 if (~pad_mask).any() else True
    assert np.abs(w_b.sum(axis=-1) - 1).max() < 1e-9

    # Invariance: changing the CONTENT of padded K/V must not change output
    K_b2, V_b2 = K_b.copy(), V_b.copy()
    K_b2[~pad_mask] += 100.0
    V_b2[~pad_mask] -= 50.0
    out_b2, _ = cross_attention(Q_b, K_b2, V_b2, pad_mask)
    inv = np.abs(out_b2 - out_b).max()
    print(f"  seq {b} (len {L}): pad weights = 0, rows sum to 1, "
          f"pad-content invariance = {inv:.1e}")
    assert inv < 1e-12
print("  [PASS] padding is fully inert")


# --- Combined causal + padding mask in decoder self-attention ---
def self_attention_causal_padded(X, pad_mask):
    """Combined mask = causal AND not-padding. Fully masked rows (padded
    queries) are handled by forcing attention to self: their output is
    garbage anyway (they are padding) but must not produce NaN. We document
    this choice; an alternative is zeroing their output."""
    T, d = X.shape
    scores = X @ X.T / np.sqrt(d)
    allowed = np.tril(np.ones((T, T), dtype=bool)) & pad_mask[None, :]
    # Strategy for all-masked rows: allow self-attention so softmax is defined
    for i in range(T):
        if not allowed[i].any():
            allowed[i, i] = True
    scores = np.where(allowed, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ X, w


X8 = rngp.standard_normal((5, d8))
pad8 = np.array([True, True, True, False, False])   # 2 padded decoder slots
out_sa, w_sa = self_attention_causal_padded(X8, pad8)
assert np.isfinite(out_sa).all() and np.isfinite(w_sa).all()
# Real tokens: never attend to padding nor future
for i in range(3):
    assert np.abs(w_sa[i, 3:]).max() == 0.0           # no padding
    assert np.abs(w_sa[i, i + 1:3]).max() == 0.0      # no future
assert np.abs(w_sa.sum(axis=-1) - 1).max() < 1e-9
print("\n  combined causal+padding: no NaN, no attention to future or padding")
print("  (fully-masked padded rows fall back to self-attention, documented)")
print("  [PASS]")

print("\n" + "=" * 70)
print("All 8 exercises completed.")
print("=" * 70)

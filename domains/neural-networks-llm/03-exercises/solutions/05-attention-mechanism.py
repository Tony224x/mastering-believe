"""
Solutions — Jour 5 : Attention Mechanism
=========================================
Solutions for the 3 easy exercises.

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

print("=" * 70)
print("All 3 exercises completed.")
print("=" * 70)

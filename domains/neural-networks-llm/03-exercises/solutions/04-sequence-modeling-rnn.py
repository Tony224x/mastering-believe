"""
Solutions — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)
========================================================
Solutions for the 3 easy exercises.

Run: python 03-exercises/solutions/04-sequence-modeling-rnn.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================================
# EXERCISE 1: Forward pass of a vanilla RNN by hand
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Forward pass of a vanilla RNN by hand")
print("=" * 70)

# Parameters given
W_xh = np.array([[0.5, -0.3],
                 [0.1, 0.4]])
W_hh = np.array([[0.2, 0.1],
                 [-0.1, 0.3]])
b_h = np.array([0.0, 0.0])

# Input sequence
x1 = np.array([1.0, 0.0])
x2 = np.array([0.0, 1.0])
x3 = np.array([1.0, 1.0])

# Initial state
h0 = np.array([0.0, 0.0])

# Step 1
print("\n--- Step 1 ---")
z1_xh = W_xh @ x1
z1_hh = W_hh @ h0
z1 = z1_xh + z1_hh + b_h
h1 = np.tanh(z1)
print(f"  W_xh @ x_1           = {z1_xh}")
print(f"  W_hh @ h_0           = {z1_hh}")
print(f"  pre-activation z_1   = {z1}")
print(f"  h_1 = tanh(z_1)      = {h1}")

# Step 2
print("\n--- Step 2 ---")
z2_xh = W_xh @ x2
z2_hh = W_hh @ h1
z2 = z2_xh + z2_hh + b_h
h2 = np.tanh(z2)
print(f"  W_xh @ x_2           = {z2_xh}")
print(f"  W_hh @ h_1           = {z2_hh}")
print(f"  pre-activation z_2   = {z2}")
print(f"  h_2 = tanh(z_2)      = {h2}")

# Step 3
print("\n--- Step 3 ---")
z3_xh = W_xh @ x3
z3_hh = W_hh @ h2
z3 = z3_xh + z3_hh + b_h
h3 = np.tanh(z3)
print(f"  W_xh @ x_3           = {z3_xh}")
print(f"  W_hh @ h_2           = {z3_hh}")
print(f"  pre-activation z_3   = {z3}")
print(f"  h_3 = tanh(z_3)      = {h3}")

# Question 4: all-zero inputs
print("\n--- Question 4: all-zero inputs ---")
h_zero = h0.copy()
for _ in range(3):
    z = W_xh @ np.zeros(2) + W_hh @ h_zero
    h_zero = np.tanh(z)
print(f"  h_3 (zero inputs)    = {h_zero}")
print("  Explanation: with all x_t = 0 and h_0 = 0, we get W_hh @ 0 = 0,")
print("  tanh(0) = 0, so h stays zero forever. Information must come from inputs.")

print("\n--- Question 5: does h_3 still carry info about x_1? ---")
print(f"  h_3 = {h3}")
print("  Yes — x_1 influenced h_1 which influenced h_2 which influenced h_3.")
print("  But the influence is compressed non-linearly through tanh at each step.")
print("  After many steps, the contribution of x_1 can become negligible")
print("  (the vanishing problem). Here with T=3 the effect is still visible.")


# ============================================================================
# EXERCISE 2: Vanishing gradient — product of Jacobians
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Vanishing gradient — product of Jacobians")
print("=" * 70)


def matrix_power_norm(W, T):
    """Compute ||W^T|| (Frobenius norm of W raised to the T-th power)."""
    M = np.eye(W.shape[0])
    for _ in range(T):
        M = W @ M
    return np.linalg.norm(M, ord='fro')


Ts = [5, 10, 20, 50]

# Case 1: spectral radius 0.5 (vanishing)
print("\n--- Case 1: W_hh = 0.5 * I (spectral radius = 0.5) ---")
W1 = 0.5 * np.eye(2)
print(f"  {'T':>5s} | {'||W^T||':>18s}")
print(f"  {'-' * 30}")
for T in Ts:
    n = matrix_power_norm(W1, T)
    print(f"  {T:>5d} | {n:>18.6e}")
print("  -> Vanishing gradient: norm decays toward 0")

# Case 2: spectral radius 1.5 (exploding)
print("\n--- Case 2: W_hh = 1.5 * I (spectral radius = 1.5) ---")
W2 = 1.5 * np.eye(2)
print(f"  {'T':>5s} | {'||W^T||':>18s}")
print(f"  {'-' * 30}")
for T in Ts:
    n = matrix_power_norm(W2, T)
    print(f"  {T:>5d} | {n:>18.6e}")
print("  -> Exploding gradient: norm grows unboundedly")

# Case 3: spectral radius 1.0 (identity)
print("\n--- Case 3: W_hh = I (spectral radius = 1.0) ---")
W3 = np.eye(2)
print(f"  {'T':>5s} | {'||W^T||':>18s}")
print(f"  {'-' * 30}")
for T in Ts:
    n = matrix_power_norm(W3, T)
    print(f"  {T:>5d} | {n:>18.6e}")
print("  -> Stable: norm stays constant")
print("  This is ideal but NOT realistic: if W_hh = I, the hidden state")
print("  ignores x_t completely (it just copies h_{t-1} through tanh).")
print("  In practice we want W_hh close to orthogonal (all singular values = 1),")
print("  which is why orthogonal initialization helps a lot.")

print("""
--- Q4: relationship between eigenvalue lambda and gradient behavior ---

  lambda < 1  -> vanishing gradient  (gradient shrinks exponentially)
  lambda = 1  -> stable              (gradient norm stays constant)
  lambda > 1  -> exploding gradient  (gradient grows exponentially)

--- Q5: two mitigation techniques (without changing architecture) ---

  1. Gradient clipping: if ||grad|| > threshold, rescale it.
     Prevents explosion but does nothing for vanishing.

  2. Orthogonal initialization of W_hh: set W_hh so that its singular values
     are all exactly 1. At init time, the gradient product is well-behaved.
     It drifts during training but helps a lot in practice.

  (Bonus: identity initialization for ReLU RNNs, Sutskever's IRNN, 2015)
""")


# ============================================================================
# EXERCISE 3: LSTM gates
# ============================================================================

print("=" * 70)
print("EXERCISE 3: LSTM gates — role and scenarios")
print("=" * 70)

print("""
--- Q1: Gate activations and roles ---

  | Gate | Activation | Output   | Role                                    |
  |------|------------|----------|-----------------------------------------|
  | f_t  | sigmoid    | [0, 1]   | How much of c_{t-1} to KEEP (forget)    |
  | i_t  | sigmoid    | [0, 1]   | How much of c~_t to ADD (input)         |
  | o_t  | sigmoid    | [0, 1]   | How much of c_t to EXPOSE (output)      |
  | c~_t | tanh       | [-1, 1]  | The candidate content to write          |

  f, i, o are switches (0 = closed, 1 = open).
  c~ is the actual information to store.
""")

# Scenario 1: full forget=1, full input=0 -> keep everything
print("--- Q2: Scenario f=1, i=0 ---")
f = np.array([1.0, 1.0, 1.0])
i = np.array([0.0, 0.0, 0.0])
c_prev = np.array([2.0, 3.0, 4.0])
c_tilde = np.array([5.0, 6.0, 7.0])
c_new = f * c_prev + i * c_tilde
print(f"  c_prev   = {c_prev}")
print(f"  c_tilde  = {c_tilde}")
print(f"  c_new    = {c_new}")
print("  -> c_t = c_{t-1}: the cell state is frozen (perfect memory).")
print("  Metaphor: 'I don't erase anything, I don't add anything'.")
print("  This is how the LSTM preserves information across many timesteps.")

# Scenario 2: full forget=0, full input=1 -> overwrite
print("\n--- Q3: Scenario f=0, i=1 ---")
f = np.array([0.0, 0.0, 0.0])
i = np.array([1.0, 1.0, 1.0])
c_new = f * c_prev + i * c_tilde
print(f"  c_new    = {c_new}")
print("  -> c_t = c~_t: old memory erased, fully overwritten.")
print("  Metaphor: 'blank page, write new notes'.")

# Scenario 3: mixed
print("\n--- Q4: Mixed scenario ---")
f = np.array([1.0, 0.0, 1.0])
i = np.array([0.0, 1.0, 0.0])
c_prev = np.array([2.0, 3.0, 4.0])
c_tilde = np.array([5.0, 6.0, 7.0])
c_new = f * c_prev + i * c_tilde
print(f"  f_t      = {f}")
print(f"  i_t      = {i}")
print(f"  c_prev   = {c_prev}")
print(f"  c_tilde  = {c_tilde}")
print(f"  c_new    = {c_new}   (expected: [2, 6, 4])")
print("  Dimension-wise interpretation:")
print("    dim 0: f=1, i=0 -> keep old value (2.0)")
print("    dim 1: f=0, i=1 -> replace with new (6.0)")
print("    dim 2: f=1, i=0 -> keep old value (4.0)")
print("  Each dimension can independently decide to remember or forget.")

# Q5: why LSTM doesn't vanish
print("""
--- Q5: why LSTM avoids vanishing gradient ---

  c_t = f_t * c_{t-1} + i_t * c~_t

  The connection c_{t-1} -> c_t is ADDITIVE (no W_hh matrix in between).
  When f_t ~ 1, the derivative dc_t/dc_{t-1} ~ 1, so the gradient passes
  through UNCHANGED. Over many timesteps, gradients don't shrink or explode
  along the c-state pathway.

  In contrast, a vanilla RNN has h_t = tanh(W_hh @ h_{t-1} + ...), which
  requires multiplying by W_hh at every step, leading to lambda^T decay/growth.

  The LSTM creates a "gradient highway" through the cell state.
""")

print("\n" + "=" * 70)
print("All 3 exercises completed.")
print("=" * 70)

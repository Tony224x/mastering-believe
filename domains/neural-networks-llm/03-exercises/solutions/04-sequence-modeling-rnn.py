"""
Solutions — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)
========================================================
Solutions for the 8 exercises (easy, medium, hard).

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

# ============================================================================
# EXERCISE 4 (MEDIUM): Batched RNN forward — predict then verify shapes
# ============================================================================

print("=" * 70)
print("EXERCISE 4 (MEDIUM): Batched RNN forward + shape discipline")
print("=" * 70)

# Config: batch=3, T=5, input_dim=4, hidden_dim=6, output_dim=2
# Predicted shapes (done on paper first):
#   X     : (3, 5, 4)      h_t : (3, 6)       W_xh : (4, 6)
#   W_hh  : (6, 6)         W_hy: (6, 2)
#   H     : (3, 5, 6)      Y   : (3, 5, 2)
rng4 = np.random.default_rng(42)
B4, T4, D_in4, D_h4, D_out4 = 3, 5, 4, 6, 2

X4 = rng4.standard_normal((B4, T4, D_in4))
h0_4 = rng4.standard_normal((B4, D_h4)) * 0.5
W_xh4 = rng4.standard_normal((D_in4, D_h4)) * 0.5
W_hh4 = rng4.standard_normal((D_h4, D_h4)) * 0.5
W_hy4 = rng4.standard_normal((D_h4, D_out4)) * 0.5
b_h4 = np.zeros(D_h4)
b_y4 = np.zeros(D_out4)

# Shape predictions verified by assert
assert X4.shape == (3, 5, 4) and W_xh4.shape == (4, 6) and W_hh4.shape == (6, 6)
assert W_hy4.shape == (6, 2) and h0_4.shape == (3, 6)


def rnn_forward_batched(X, h0, W_xh, W_hh, W_hy, b_h, b_y):
    """Batched RNN: only loop over time, batch handled by matmul."""
    B, T, _ = X.shape
    H = np.zeros((B, T, W_hh.shape[0]))
    Y = np.zeros((B, T, W_hy.shape[1]))
    h = h0
    for t in range(T):
        # (B, D_in) @ (D_in, D_h) + (B, D_h) @ (D_h, D_h) -> (B, D_h)
        h = np.tanh(X[:, t] @ W_xh + h @ W_hh + b_h)
        H[:, t] = h
        Y[:, t] = h @ W_hy + b_y
    return H, Y


def rnn_forward_naive(X, h0, W_xh, W_hh, W_hy, b_h, b_y):
    """Reference: one sample, one timestep at a time (slow, but obviously right)."""
    B, T, _ = X.shape
    H = np.zeros((B, T, W_hh.shape[0]))
    Y = np.zeros((B, T, W_hy.shape[1]))
    for b in range(B):
        h = h0[b]
        for t in range(T):
            h = np.tanh(X[b, t] @ W_xh + h @ W_hh + b_h)
            H[b, t] = h
            Y[b, t] = h @ W_hy + b_y
    return H, Y


H4, Y4 = rnn_forward_batched(X4, h0_4, W_xh4, W_hh4, W_hy4, b_h4, b_y4)
H4n, Y4n = rnn_forward_naive(X4, h0_4, W_xh4, W_hh4, W_hy4, b_h4, b_y4)

assert H4.shape == (3, 5, 6) and Y4.shape == (3, 5, 2)
diff_H = np.abs(H4 - H4n).max()
diff_Y = np.abs(Y4 - Y4n).max()
print(f"\n  H shape: {H4.shape}  Y shape: {Y4.shape}  (as predicted)")
print(f"  Batched vs naive: max diff H = {diff_H:.2e}, Y = {diff_Y:.2e}")
assert diff_H < 1e-12 and diff_Y < 1e-12

# h0 dependency: changing h0 must change H[:, 0]
H4b, _ = rnn_forward_batched(X4, h0_4 + 1.0, W_xh4, W_hh4, W_hy4, b_h4, b_y4)
delta_h0 = np.abs(H4b[:, 0] - H4[:, 0]).max()
print(f"  h0 dependency: max |H[:,0] change| after h0 shift = {delta_h0:.4f} (> 0)")
assert delta_h0 > 1e-3
print("  [PASS] all shape and equivalence checks")


# ============================================================================
# EXERCISE 5 (MEDIUM): Gradient check of a single RNN cell
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5 (MEDIUM): RNN cell backward + gradient check")
print("=" * 70)

rng5 = np.random.default_rng(7)
D_in5, D_h5 = 3, 4
x5 = rng5.standard_normal(D_in5) * 0.5
hprev5 = rng5.standard_normal(D_h5) * 0.5
Wxh5 = rng5.standard_normal((D_in5, D_h5)) * 0.5
Whh5 = rng5.standard_normal((D_h5, D_h5)) * 0.5
bh5 = rng5.standard_normal(D_h5) * 0.5


def cell_loss(x, hprev, Wxh, Whh, bh):
    """L = 0.5 * sum(h^2) with h = tanh(x@Wxh + hprev@Whh + bh)."""
    h = np.tanh(x @ Wxh + hprev @ Whh + bh)
    return 0.5 * np.sum(h ** 2)


def rnn_cell_backward(x, hprev, Wxh, Whh, bh):
    """Analytical gradients of L = 0.5*sum(h^2) w.r.t. all 5 inputs.

    Chain rule, step by step:
      dL/dh = h                       (derivative of 0.5*h^2)
      dh/dz = 1 - h^2                 (tanh')
      dz    = dL/dh * dh/dz           (elementwise)
      dWxh  = outer(x, dz)            (z = x @ Wxh + ...)
      dWhh  = outer(hprev, dz)
      dbh   = dz
      dx    = Wxh @ dz                (z is linear in x)
      dhprev= Whh @ dz
    """
    z = x @ Wxh + hprev @ Whh + bh
    h = np.tanh(z)
    dh = h                       # dL/dh for L = 0.5*sum(h^2)
    dz = dh * (1.0 - h ** 2)     # through tanh
    dWxh = np.outer(x, dz)
    dWhh = np.outer(hprev, dz)
    dbh = dz
    dx = Wxh @ dz
    dhprev = Whh @ dz
    return dWxh, dWhh, dbh, dx, dhprev


def numerical_grad(f, arr, eps=1e-6):
    """Central finite differences, element by element."""
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + eps
        lp = f()
        arr[idx] = old - eps
        lm = f()
        arr[idx] = old
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def rel_err(a, n):
    return np.abs(a - n) / (np.abs(a) + np.abs(n) + 1e-8)


grads_analytic = rnn_cell_backward(x5, hprev5, Wxh5, Whh5, bh5)
loss_fn5 = lambda: cell_loss(x5, hprev5, Wxh5, Whh5, bh5)
names5 = ['dW_xh', 'dW_hh', 'db_h', 'dx', 'dh_prev']
arrays5 = [Wxh5, Whh5, bh5, x5, hprev5]

print(f"\n  {'grad':>8} | {'max rel err':>12} | status")
worst5 = 0.0
for name, arr, ga in zip(names5, arrays5, grads_analytic):
    gn = numerical_grad(loss_fn5, arr)
    e = rel_err(ga, gn).max()
    worst5 = max(worst5, e)
    print(f"  {name:>8} | {e:>12.2e} | {'PASS' if e < 1e-6 else 'FAIL'}")
assert worst5 < 1e-6
print(f"  [PASS] all 5 gradients verified element-wise (worst: {worst5:.2e})")


# ============================================================================
# EXERCISE 6 (MEDIUM): Debug a broken LSTM cell
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6 (MEDIUM): Debugging a broken LSTM cell")
print("=" * 70)

# The 3 bugs and their training symptoms:
#  BUG 1: f = tanh(...) -> forget gate in [-1, 1]. A NEGATIVE forget gate
#         FLIPS the sign of the memory instead of attenuating it.
#         Symptom: cell state oscillates, long-term memory is unusable.
#  BUG 2: c = f * g     -> c_prev never enters the new cell state.
#         There is NO memory pathway at all: the LSTM degenerates into a
#         feedforward gate on the current candidate.
#         Symptom: trains fine on short-range patterns, total failure on
#         any task requiring memory of more than 1 step.
#  BUG 3: h = o * c     -> h is unbounded (c can grow linearly over time
#         since c = f*c_prev + i*g accumulates).
#         Symptom: hidden state magnitude drifts upward, downstream
#         activations saturate, training becomes unstable on long sequences.

rng6 = np.random.default_rng(42)
D_in6, D_h6 = 3, 4
Wf6 = rng6.standard_normal((D_h6, D_h6 + D_in6)) * 0.5
Wi6 = rng6.standard_normal((D_h6, D_h6 + D_in6)) * 0.5
Wg6 = rng6.standard_normal((D_h6, D_h6 + D_in6)) * 0.5
Wo6 = rng6.standard_normal((D_h6, D_h6 + D_in6)) * 0.5
bf6, bi6, bg6, bo6 = (np.zeros(D_h6) for _ in range(4))
params6 = (Wf6, Wi6, Wg6, Wo6, bf6, bi6, bg6, bo6)


def lstm_cell_buggy(x, h_prev, c_prev, params):
    Wf, Wi, Wg, Wo, bf, bi, bg, bo = params
    z = np.concatenate([h_prev, x])
    f = np.tanh(Wf @ z + bf)        # BUG 1: should be sigmoid
    i = sigmoid(Wi @ z + bi)
    g = np.tanh(Wg @ z + bg)
    o = sigmoid(Wo @ z + bo)
    c = f * g                        # BUG 2: should be f*c_prev + i*g
    h = o * c                        # BUG 3: should be o * tanh(c)
    return h, c


def lstm_cell_fixed(x, h_prev, c_prev, params, forced_gates=None):
    """Standard LSTM equations. forced_gates lets tests pin f and i."""
    Wf, Wi, Wg, Wo, bf, bi, bg, bo = params
    z = np.concatenate([h_prev, x])
    f = sigmoid(Wf @ z + bf)         # FIX 1: sigmoid keeps f in [0, 1]
    i = sigmoid(Wi @ z + bi)
    g = np.tanh(Wg @ z + bg)
    o = sigmoid(Wo @ z + bo)
    if forced_gates is not None:     # test hook: pin f and i to constants
        f, i = forced_gates
    c = f * c_prev + i * g           # FIX 2: additive memory pathway
    h = o * np.tanh(c)               # FIX 3: bound h through tanh(c)
    return h, c, (f, i, o)


x6 = rng6.standard_normal(D_in6)
h6 = rng6.standard_normal(D_h6) * 0.5
c6 = rng6.standard_normal(D_h6) * 2.0  # large c_prev to make memory visible

h_fix, c_fix, (f6, i6, o6) = lstm_cell_fixed(x6, h6, c6, params6)
print(f"\n  fixed gates f = {np.round(f6, 4)}")
print(f"  fixed gates i = {np.round(i6, 4)}")
print(f"  fixed gates o = {np.round(o6, 4)}")
assert np.all((f6 >= 0) & (f6 <= 1)) and np.all((i6 >= 0) & (i6 <= 1)) \
    and np.all((o6 >= 0) & (o6 <= 1))
print("  [PASS] all gates in [0, 1]")

# Forced-gate tests: f=1, i=0 -> perfect memory ; f=0, i=1 -> overwrite
ones6, zeros6 = np.ones(D_h6), np.zeros(D_h6)
_, c_keep, _ = lstm_cell_fixed(x6, h6, c6, params6, forced_gates=(ones6, zeros6))
assert np.abs(c_keep - c6).max() < 1e-12
print(f"  [PASS] f=1, i=0  ->  c == c_prev exactly (max diff {np.abs(c_keep - c6).max():.1e})")

_, c_over, _ = lstm_cell_fixed(x6, h6, c6, params6, forced_gates=(zeros6, ones6))
g_expected = np.tanh(Wg6 @ np.concatenate([h6, x6]) + bg6)
assert np.abs(c_over - g_expected).max() < 1e-12
print("  [PASS] f=0, i=1  ->  c == candidate g exactly")

h_bug, c_bug = lstm_cell_buggy(x6, h6, c6, params6)
print(f"  buggy vs fixed: |h diff| = {np.abs(h_bug - h_fix).max():.4f}, "
      f"|c diff| = {np.abs(c_bug - c_fix).max():.4f}")
print("  -> the buggy cell silently computes something completely different")


# ============================================================================
# EXERCISE 7 (HARD): LSTM from scratch + long-range gradient measurement
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7 (HARD): LSTM vs RNN — long-range gradient survival")
print("=" * 70)

rng7 = np.random.default_rng(0)
D_in7, D_h7 = 4, 8

# --- RNN init: spectral radius forced to 0.5 to expose vanishing ---
W_raw = rng7.standard_normal((D_h7, D_h7))
rho = np.abs(np.linalg.eigvals(W_raw)).max()
Whh7 = 0.5 * W_raw / rho          # spectral radius exactly 0.5
Wxh7 = rng7.standard_normal((D_in7, D_h7)) * 0.5
bh7 = np.zeros(D_h7)

# --- LSTM init: small weights, forget bias = +1.0 ---
# The +1 forget bias means f ~ sigmoid(1) ~ 0.73 at init: the LSTM starts
# in "mostly remember" mode, which keeps the c-pathway gradient alive.
Wl7 = {k: rng7.standard_normal((D_h7, D_h7 + D_in7)) * 0.3 for k in 'figo'}
bl7 = {k: np.zeros(D_h7) for k in 'figo'}
bl7['f'] += 1.0                    # forget bias trick


def rnn_loss_from_seq(X):
    h = np.zeros(D_h7)
    for t in range(X.shape[0]):
        h = np.tanh(X[t] @ Wxh7 + h @ Whh7 + bh7)
    return np.sum(h ** 2)


def lstm_loss_from_seq(X):
    h, c = np.zeros(D_h7), np.zeros(D_h7)
    for t in range(X.shape[0]):
        z = np.concatenate([h, X[t]])
        f = sigmoid(Wl7['f'] @ z + bl7['f'])
        i = sigmoid(Wl7['i'] @ z + bl7['i'])
        g = np.tanh(Wl7['g'] @ z + bl7['g'])
        o = sigmoid(Wl7['o'] @ z + bl7['o'])
        c = f * c + i * g
        h = o * np.tanh(c)
    return np.sum(h ** 2)


def grad_wrt_x0(loss_fn, X, eps=1e-4):
    """Finite differences on x_0 only: 2*input_dim forwards. Cheap and
    framework-free. Note: values below ~1e-12 are at the float64 noise
    floor of the finite-difference estimate."""
    g = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp, Xm = X.copy(), X.copy()
        Xp[0, j] += eps
        Xm[0, j] -= eps
        g[j] = (loss_fn(Xp) - loss_fn(Xm)) / (2 * eps)
    return g


print(f"\n  Loss = sum(h_T^2). Measuring ||dL/dx_0|| by finite differences.")
print(f"  {'T':>4} | {'||dL/dx0|| RNN':>16} | {'||dL/dx0|| LSTM':>16} | {'ratio LSTM/RNN':>15}")
print(f"  {'-'*4} | {'-'*16} | {'-'*16} | {'-'*15}")

results7 = {}
for T in [5, 20, 40]:
    X = rng7.standard_normal((T, D_in7)) * 0.5
    g_rnn = np.linalg.norm(grad_wrt_x0(rnn_loss_from_seq, X))
    g_lstm = np.linalg.norm(grad_wrt_x0(lstm_loss_from_seq, X))
    ratio = g_lstm / max(g_rnn, 1e-300)
    results7[T] = (g_rnn, g_lstm, ratio)
    print(f"  {T:>4} | {g_rnn:>16.3e} | {g_lstm:>16.3e} | {ratio:>15.3e}")

assert results7[40][0] < 1e-6, "RNN gradient should have vanished at T=40"
assert results7[40][2] > 1e3, "LSTM should retain >1000x more gradient at T=40"
print("\n  [PASS] at T=40: RNN gradient < 1e-6 and LSTM/RNN ratio > 1e3")
print("""
  WHY the LSTM wins: the cell pathway is c_t = f_t * c_{t-1} + i_t * g_t.
  Going back one step multiplies the gradient by f_t (elementwise, ~0.73
  here) — no W_hh matrix product, no tanh saturation stacked T times.
  The RNN pathway multiplies by diag(1-h^2) @ W_hh^T at EVERY step:
  with spectral radius 0.5 that is at best (0.5)^T -> 9e-13 at T=40.""")


# ============================================================================
# EXERCISE 8 (HARD): Full BPTT + exploding gradients + clipping
# ============================================================================

print("=" * 70)
print("EXERCISE 8 (HARD): BPTT from scratch, explosion, clipping")
print("=" * 70)

rng8 = np.random.default_rng(3)
D_in8, D_h8 = 2, 8


def init_params8(spectral_radius):
    W = rng8.standard_normal((D_h8, D_h8))
    W = spectral_radius * W / np.abs(np.linalg.eigvals(W)).max()
    return {
        'Wxh': rng8.standard_normal((D_in8, D_h8)) * 0.3,
        'Whh': W,
        'bh': np.zeros(D_h8),
        'Why': rng8.standard_normal((D_h8, 1)) * 0.3,
        'by': np.zeros(1),
    }


def rnn_bptt(X, target, p):
    """Forward + full BPTT for loss L = (y - target)^2, y = h_T @ Why + by.
    Returns (loss, grads). Weights are SHARED across time: gradients are
    ACCUMULATED at every timestep."""
    T = X.shape[0]
    hs = [np.zeros(D_h8)]
    for t in range(T):
        hs.append(np.tanh(X[t] @ p['Wxh'] + hs[-1] @ p['Whh'] + p['bh']))
    y = hs[-1] @ p['Why'] + p['by']
    loss = float(((y - target) ** 2).sum())

    g = {k: np.zeros_like(v) for k, v in p.items()}
    dy = 2.0 * (y - target)                    # scalar-ish (1,)
    g['Why'] = np.outer(hs[-1], dy)
    g['by'] = dy.copy()
    dh = p['Why'] @ dy                          # dL/dh_T
    for t in range(T - 1, -1, -1):
        dz = dh * (1.0 - hs[t + 1] ** 2)        # through tanh
        g['Wxh'] += np.outer(X[t], dz)          # accumulate (shared weights!)
        g['Whh'] += np.outer(hs[t], dz)
        g['bh'] += dz
        dh = p['Whh'] @ dz                      # recurrent term to h_{t-1}
    return loss, g


# --- 1+2. Gradient check on T=6 ---
p8 = init_params8(spectral_radius=0.9)
X8 = rng8.standard_normal((6, D_in8)) * 0.5
target8 = 1.0
_, grads8 = rnn_bptt(X8, target8, p8)

worst8 = 0.0
for name in ['Wxh', 'Whh', 'bh']:
    f8 = lambda: rnn_bptt(X8, target8, p8)[0]
    gn = numerical_grad(f8, p8[name], eps=1e-5)
    e = rel_err(grads8[name], gn).max()
    worst8 = max(worst8, e)
    print(f"  gradient check {name:>4}: max rel err = {e:.2e}")
assert worst8 < 1e-5
print(f"  [PASS] BPTT gradient check (worst {worst8:.2e})")

# --- 3. Provoke explosion: spectral radius 1.5, tiny inputs (linear regime) ---
# With tiny inputs the hidden state stays in tanh's quasi-linear zone, so the
# backward Jacobian is ~W_hh^T at every step -> gradient grows like 1.5^T.
print("\n  Exploding gradients with spectral radius 1.5 (inputs * 0.01):")
p_exp = init_params8(spectral_radius=1.5)
print(f"  {'T':>4} | {'||dWhh||':>14}")
norms8 = {}
for T in [3, 6, 9, 12, 40]:
    Xe = rng8.standard_normal((T, D_in8)) * 0.01   # tiny inputs: tanh linear zone
    _, ge = rnn_bptt(Xe, target8, p_exp)
    norms8[T] = np.linalg.norm(ge['Whh'])
    print(f"  {T:>4} | {norms8[T]:>14.4e}")
print(f"  growth ||dWhh||(12) / ||dWhh||(3) = {norms8[12] / norms8[3]:.1f}x")
assert norms8[12] > 20 * norms8[3], "expected exponential-ish growth in linear regime"
# At T=40 the state has saturated tanh (|h| ~ 1): the derivative (1-h^2) ~ 0
# CAPS the explosion. In real training, explosions show up as intermittent
# spikes precisely when trajectories cross the linear zone.
print(f"  note: at T=40 the norm is {norms8[40]:.3e} — tanh saturation caps the")
print("  explosion; in practice explosions appear as intermittent spikes.")


# --- 4. Global-norm clipping ---
def clip_gradients(grads, max_norm):
    """Rescale ALL gradients by a single factor if global norm > max_norm.
    Preserves the gradient DIRECTION (unlike per-element clipping)."""
    total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / total
        return {k: g * scale for k, g in grads.items()}, total
    return grads, total


_, gbig = rnn_bptt(rng8.standard_normal((30, D_in8)) * 0.05, target8, p_exp)
clipped, norm_before = clip_gradients(gbig, max_norm=1.0)
norm_after = np.sqrt(sum(float(np.sum(g ** 2)) for g in clipped.values()))
print(f"\n  clipping: norm {norm_before:.3e} -> {norm_after:.6f} (target 1.0)")
assert abs(norm_after - 1.0) < 1e-9
# Direction preserved: clipped gradient is colinear with original
cos = np.sum(gbig['Whh'] * clipped['Whh']) / (
    np.linalg.norm(gbig['Whh']) * np.linalg.norm(clipped['Whh']))
assert abs(cos - 1.0) < 1e-12
print(f"  direction preserved: cos(original, clipped) = {cos:.12f}")
# No-op below threshold
small = {k: v * 1e-6 for k, v in gbig.items()}
same, _ = clip_gradients(small, max_norm=1.0)
assert all(np.array_equal(same[k], small[k]) for k in small)
print("  below threshold: gradients unchanged  [PASS]")

# --- 5. Training with vs without clipping on a toy task: y = sum(X) ---
print("\n  Toy task: predict sum(X) over T=10 (exploding init, lr=0.01)")


def train8(use_clip, steps=200, lr=0.01):
    rng_t = np.random.default_rng(11)
    p = init_params8(spectral_radius=1.5)
    first = last = None
    for s in range(steps):
        X = rng_t.standard_normal((10, D_in8)) * 0.3
        tgt = X.sum()
        loss, g = rnn_bptt(X, tgt, p)
        if first is None:
            first = loss
        if not np.isfinite(loss) or loss > 1e3:
            return first, float('inf')           # diverged
        if use_clip:
            g, _ = clip_gradients(g, max_norm=1.0)
        for k in p:
            p[k] -= lr * g[k]
        last = loss
    return first, last


loss0_nc, lossF_nc = train8(use_clip=False)
loss0_c, lossF_c = train8(use_clip=True)
print(f"  without clipping: first loss {loss0_nc:.3f} -> {'DIVERGED' if not np.isfinite(lossF_nc) else f'{lossF_nc:.4f}'}")
print(f"  with    clipping: first loss {loss0_c:.3f} -> {lossF_c:.4f}")
assert not np.isfinite(lossF_nc) or lossF_nc > 1e3, "unclipped run should diverge"
assert lossF_c < loss0_c / 10, "clipped run should converge (loss / 10)"
print("  [PASS] clipping turns a divergent run into a convergent one")

print("\n" + "=" * 70)
print("All 8 exercises completed.")
print("=" * 70)

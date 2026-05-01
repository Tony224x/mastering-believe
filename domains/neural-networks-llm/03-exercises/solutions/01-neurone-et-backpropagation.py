"""
Solutions — Jour 1 : Le neurone & backpropagation
==================================================
Solutions pour les 8 exercices (easy, medium, hard).

Run: python 03-exercises/solutions/01-neurone-et-backpropagation.py
"""

import numpy as np
import time

# ============================================================================
# EXERCISE 1: Forward pass a la main
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Forward pass a la main")
print("=" * 70)

# Parameters
x1, x2, x3 = 1.0, -0.5, 0.3
w1, w2, w3 = 0.2, 0.8, -0.5
b = 0.15
y_target = 0.0

# Step 1: Weighted sum
z = w1*x1 + w2*x2 + w3*x3 + b
print(f"\nStep 1 -- Weighted sum:")
print(f"  z = {w1}*{x1} + {w2}*{x2} + {w3}*{x3} + {b}")
print(f"  z = {w1*x1} + {w2*x2} + {w3*x3} + {b}")
print(f"  z = {z:.4f}")

# Step 2: Sigmoid activation
a = 1.0 / (1.0 + np.exp(-z))
print(f"\nStep 2 -- Sigmoid:")
print(f"  a = 1 / (1 + e^(-{z:.4f}))")
print(f"  a = 1 / (1 + {np.exp(-z):.4f})")
print(f"  a = {a:.4f}")

# Step 3: MSE loss
loss = (a - y_target) ** 2
print(f"\nStep 3 -- MSE Loss:")
print(f"  L = ({a:.4f} - {y_target})^2")
print(f"  L = {a - y_target:.4f}^2")
print(f"  L = {loss:.4f}")


# ============================================================================
# EXERCISE 2: 3 activation functions from scratch
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Activation Functions & Derivatives")
print("=" * 70)


def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    """σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_fn(z):
    """tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))"""
    pos = np.exp(z)
    neg = np.exp(-z)
    return (pos - neg) / (pos + neg)


def tanh_deriv(z):
    """tanh'(z) = 1 - tanh(z)^2"""
    t = tanh_fn(z)
    return 1.0 - t ** 2


def relu(z):
    """ReLU(z) = max(0, z)"""
    return np.maximum(0.0, z)


def relu_deriv(z):
    """ReLU'(z) = 0 if z < 0, 1 if z > 0 (0 at z=0 by convention)"""
    return np.where(np.asarray(z) > 0, 1.0, 0.0)


# Display table
z_values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"\n  {'z':>5} | {'sig':>7} {'sig_d':>7} | {'tanh':>7} {'tanh_d':>7} | {'relu':>7} {'relu_d':>7}")
print(f"  {'-'*5} | {'-'*7} {'-'*7} | {'-'*7} {'-'*7} | {'-'*7} {'-'*7}")

for z_val in z_values:
    print(f"  {z_val:>5.1f} | {sigmoid(z_val):>7.4f} {sigmoid_deriv(z_val):>7.4f} "
          f"| {tanh_fn(z_val):>7.4f} {tanh_deriv(z_val):>7.4f} "
          f"| {relu(z_val):>7.4f} {relu_deriv(z_val):>7.4f}")

# Numerical gradient verification
print(f"\n  Gradient check (numerical vs analytical, epsilon=1e-7):")
epsilon = 1e-7

functions = [('sigmoid', sigmoid, sigmoid_deriv),
             ('tanh', tanh_fn, tanh_deriv),
             ('relu', relu, relu_deriv)]

for name, fn, fn_deriv in functions:
    max_error = 0.0
    for z_val in z_values:
        # Skip z=0 for ReLU (not differentiable there — known edge case)
        if name == 'relu' and z_val == 0.0:
            continue
        numerical = (fn(z_val + epsilon) - fn(z_val - epsilon)) / (2 * epsilon)
        analytical = fn_deriv(z_val)
        error = abs(numerical - analytical)
        max_error = max(max_error, error)
    status = "PASS" if max_error < 1e-5 else "FAIL"
    note = " (z=0 skipped: ReLU not differentiable there)" if name == 'relu' else ""
    print(f"  {name:>8}: max error = {max_error:.2e} [{status}]{note}")


# ============================================================================
# EXERCISE 3: MSE vs Cross-Entropy
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: MSE vs Cross-Entropy Comparison")
print("=" * 70)

cases = [
    (0.9, 1.0, "Correct, high confidence"),
    (0.5, 1.0, "Uncertain"),
    (0.1, 1.0, "Very wrong"),
    (0.9, 0.0, "Very wrong (inverted)"),
    (0.01, 0.0, "Correct, high confidence"),
]

print(f"\n  {'Case':>30} | {'pred':>5} {'tgt':>4} | {'MSE':>8} {'BCE':>8} | {'dMSE':>8} {'dBCE':>8}")
print(f"  {'-'*30} | {'-'*5} {'-'*4} | {'-'*8} {'-'*8} | {'-'*8} {'-'*8}")

for pred, target, description in cases:
    # MSE and its gradient
    mse = (pred - target) ** 2
    grad_mse = 2 * (pred - target)

    # Binary Cross-Entropy and its gradient
    # Clip to avoid log(0)
    p = np.clip(pred, 1e-7, 1 - 1e-7)
    bce = -(target * np.log(p) + (1 - target) * np.log(1 - p))
    grad_bce = -target / p + (1 - target) / (1 - p)

    print(f"  {description:>30} | {pred:>5.2f} {target:>4.1f} | {mse:>8.4f} {bce:>8.4f} | {grad_mse:>8.4f} {grad_bce:>8.4f}")

print("""
  Analysis:
  - Case "Very wrong (inverted)": pred=0.9, target=0.0
    MSE = 0.81, but BCE = 2.30 -> BCE penalizes 2.8x more!
  - Case "Very wrong": pred=0.1, target=1.0
    MSE = 0.81, but BCE = 2.30 -> same strong penalty

  BCE penalizes confident-and-wrong predictions MUCH more than MSE.
  This is desirable for classification: being confidently wrong should
  be punished severely to push the model toward calibrated probabilities.

  Additionally, BCE + sigmoid gives a clean gradient (a - y) that doesn't
  saturate, while MSE + sigmoid has a gradient that vanishes when sigmoid
  saturates -- exactly when the prediction is most wrong.
""")


# ============================================================================
# EXERCISE 4: Backpropagation a la main + verification
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Manual Backpropagation -- Verified by Code")
print("=" * 70)

# Network parameters
x = np.array([[1.0, 0.0]])  # shape (1, 2)
y_true_ex4 = np.array([[1.0]])

# Weights
W_h = np.array([[0.5, 0.1],    # row 0: x1→h1, x1→h2
                 [-0.3, 0.7]])  # row 1: x2→h1, x2→h2
b_h = np.array([[0.0, 0.0]])

W_o = np.array([[0.4],          # h1→o
                 [-0.6]])        # h2→o
b_o = np.array([[0.1]])

# --- Forward pass ---
print("\n--- Forward Pass ---")
z_h = x @ W_h + b_h
print(f"  z_h1 = {x[0,0]}*{W_h[0,0]} + {x[0,1]}*{W_h[1,0]} + {b_h[0,0]} = {z_h[0,0]:.4f}")
print(f"  z_h2 = {x[0,0]}*{W_h[0,1]} + {x[0,1]}*{W_h[1,1]} + {b_h[0,1]} = {z_h[0,1]:.4f}")

a_h = sigmoid(z_h)
print(f"  a_h1 = sigmoid({z_h[0,0]:.4f}) = {a_h[0,0]:.4f}")
print(f"  a_h2 = sigmoid({z_h[0,1]:.4f}) = {a_h[0,1]:.4f}")

z_o = a_h @ W_o + b_o
print(f"  z_o = {a_h[0,0]:.4f}*{W_o[0,0]} + {a_h[0,1]:.4f}*{W_o[1,0]} + {b_o[0,0]} = {z_o[0,0]:.4f}")

a_o = sigmoid(z_o)
print(f"  a_o = sigmoid({z_o[0,0]:.4f}) = {a_o[0,0]:.4f}")

loss_ex4 = np.mean((a_o - y_true_ex4) ** 2)
print(f"  Loss MSE = ({a_o[0,0]:.4f} - {y_true_ex4[0,0]})^2 = {loss_ex4:.6f}")

# --- Backward pass ---
print("\n--- Backward Pass ---")
n = 1

# Output layer
dL_da_o = (2.0/n) * (a_o - y_true_ex4)
da_o_dz_o = a_o * (1 - a_o)
delta_o = dL_da_o * da_o_dz_o
print(f"  dL/da_o = 2*({a_o[0,0]:.4f} - {y_true_ex4[0,0]}) = {dL_da_o[0,0]:.6f}")
print(f"  da_o/dz_o = {a_o[0,0]:.4f} * (1 - {a_o[0,0]:.4f}) = {da_o_dz_o[0,0]:.6f}")
print(f"  delta_o = {dL_da_o[0,0]:.6f} * {da_o_dz_o[0,0]:.6f} = {delta_o[0,0]:.6f}")

dW_o = a_h.T @ delta_o
db_o = np.sum(delta_o, axis=0, keepdims=True)
print(f"\n  dL/dw5 = a_h1 * delta_o = {a_h[0,0]:.4f} * {delta_o[0,0]:.6f} = {dW_o[0,0]:.6f}")
print(f"  dL/dw6 = a_h2 * delta_o = {a_h[0,1]:.4f} * {delta_o[0,0]:.6f} = {dW_o[1,0]:.6f}")
print(f"  dL/db_o = delta_o = {db_o[0,0]:.6f}")

# Hidden layer
dL_da_h = delta_o @ W_o.T
da_h_dz_h = a_h * (1 - a_h)
delta_h = dL_da_h * da_h_dz_h
print(f"\n  dL/da_h1 = delta_o * w5 = {delta_o[0,0]:.6f} * {W_o[0,0]} = {dL_da_h[0,0]:.6f}")
print(f"  dL/da_h2 = delta_o * w6 = {delta_o[0,0]:.6f} * {W_o[1,0]} = {dL_da_h[0,1]:.6f}")

dW_h = x.T @ delta_h
db_h = np.sum(delta_h, axis=0, keepdims=True)
print(f"\n  delta_h1 = {dL_da_h[0,0]:.6f} * {da_h_dz_h[0,0]:.6f} = {delta_h[0,0]:.6f}")
print(f"  delta_h2 = {dL_da_h[0,1]:.6f} * {da_h_dz_h[0,1]:.6f} = {delta_h[0,1]:.6f}")

print(f"\n  dL/dw1 = x1 * delta_h1 = {x[0,0]} * {delta_h[0,0]:.6f} = {dW_h[0,0]:.6f}")
print(f"  dL/dw2 = x2 * delta_h1 = {x[0,1]} * {delta_h[0,0]:.6f} = {dW_h[1,0]:.6f}")
print(f"  dL/dw3 = x1 * delta_h2 = {x[0,0]} * {delta_h[0,1]:.6f} = {dW_h[0,1]:.6f}")
print(f"  dL/dw4 = x2 * delta_h2 = {x[0,1]} * {delta_h[0,1]:.6f} = {dW_h[1,1]:.6f}")

# --- Weight update ---
lr_ex4 = 0.1
print(f"\n--- Weight Update (lr={lr_ex4}) ---")
W_h_new = W_h - lr_ex4 * dW_h
W_o_new = W_o - lr_ex4 * dW_o
b_h_new = b_h - lr_ex4 * db_h
b_o_new = b_o - lr_ex4 * db_o

for name, old, new in [("w1", W_h[0,0], W_h_new[0,0]), ("w2", W_h[1,0], W_h_new[1,0]),
                         ("w3", W_h[0,1], W_h_new[0,1]), ("w4", W_h[1,1], W_h_new[1,1]),
                         ("w5", W_o[0,0], W_o_new[0,0]), ("w6", W_o[1,0], W_o_new[1,0])]:
    print(f"  {name}: {old:.4f} -> {new:.6f}")


# ============================================================================
# EXERCISE 5: Learning rate study
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Learning Rate Empirical Study")
print("=" * 70)


class SimpleNN:
    """Reusable 2-layer NN for exercises."""

    def __init__(self, sizes, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(sizes[0], sizes[1]) * np.sqrt(1.0 / sizes[0])
        self.b1 = np.zeros((1, sizes[1]))
        self.W2 = np.random.randn(sizes[1], sizes[2]) * np.sqrt(1.0 / sizes[1])
        self.b2 = np.zeros((1, sizes[2]))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, X, y, lr):
        n = X.shape[0]
        pred = self.forward(X)
        loss = np.mean((pred - y) ** 2)

        # Backward
        dL_da2 = (2.0/n) * (pred - y)
        delta2 = dL_da2 * pred * (1 - pred)
        dW2 = self.a1.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2 @ self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Update
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss


# XOR data
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

learning_rates_test = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
epochs_test = 5000

print(f"\n  {'LR':>8} | {'Final Loss':>12} | {'Epochs to <0.01':>18} | {'Status':>15}")
print(f"  {'-'*8} | {'-'*12} | {'-'*18} | {'-'*15}")

for test_lr in learning_rates_test:
    net = SimpleNN([2, 4, 1], seed=42)
    epoch_threshold = "never"

    losses = []
    diverged = False
    for ep in range(epochs_test):
        loss = net.train_step(X_xor, y_xor, test_lr)
        losses.append(loss)

        if epoch_threshold == "never" and loss < 0.01:
            epoch_threshold = str(ep)

        if np.isnan(loss) or loss > 1e6:
            diverged = True
            break

    if diverged:
        status = "DIVERGED"
        final = float('inf')
    elif losses[-1] < 0.01:
        status = "CONVERGED"
        final = losses[-1]
    elif losses[-1] < 0.1:
        status = "SLOW"
        final = losses[-1]
    else:
        status = "NOT CONVERGED"
        final = losses[-1]

    final_str = f"{final:.6f}" if not diverged else "inf"
    print(f"  {test_lr:>8.3f} | {final_str:>12} | {epoch_threshold:>18} | {status:>15}")

# LR scheduler bonus
print(f"\n  --- Bonus: LR Scheduler (halve every 1000 epochs) ---")
for init_lr in [2.0, 5.0]:
    # Fixed LR
    net_fixed = SimpleNN([2, 4, 1], seed=42)
    for ep in range(5000):
        loss_fixed = net_fixed.train_step(X_xor, y_xor, init_lr)

    # Scheduled LR
    net_sched = SimpleNN([2, 4, 1], seed=42)
    current_lr = init_lr
    for ep in range(5000):
        if ep > 0 and ep % 1000 == 0:
            current_lr *= 0.5  # halve LR
        loss_sched = net_sched.train_step(X_xor, y_xor, current_lr)

    print(f"  init_lr={init_lr}: Fixed -> {loss_fixed:.6f}, Scheduled -> {loss_sched:.6f}")


# ============================================================================
# EXERCISE 6: Mini-batch from scratch
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: SGD vs Mini-batch vs Batch")
print("=" * 70)

# Generate 2-gaussian dataset
np.random.seed(42)
n_per_class = 100
X_c0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([1, 1])
X_c1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1, -1])
X_data = np.vstack([X_c0, X_c1])
y_data = np.array([0]*n_per_class + [1]*n_per_class, dtype=float).reshape(-1, 1)

# Shuffle
perm = np.random.permutation(len(X_data))
X_data = X_data[perm]
y_data = y_data[perm]

epochs_gd = 500
lr_gd = 1.0
batch_size = 16

results_gd = {}

for variant in ['batch', 'sgd', 'minibatch']:
    net = SimpleNN([2, 8, 1], seed=42)
    loss_history = []
    start_time = time.time()

    for ep in range(epochs_gd):
        if variant == 'batch':
            loss = net.train_step(X_data, y_data, lr_gd)
            loss_history.append(loss)

        elif variant == 'sgd':
            # Shuffle each epoch
            perm_ep = np.random.permutation(len(X_data))
            for idx in perm_ep:
                net.train_step(X_data[idx:idx+1], y_data[idx:idx+1], lr_gd)
            # Record full-dataset loss
            pred = net.forward(X_data)
            loss_history.append(np.mean((pred - y_data)**2))

        elif variant == 'minibatch':
            perm_ep = np.random.permutation(len(X_data))
            for start in range(0, len(X_data), batch_size):
                end = min(start + batch_size, len(X_data))
                batch_idx = perm_ep[start:end]
                net.train_step(X_data[batch_idx], y_data[batch_idx], lr_gd)
            pred = net.forward(X_data)
            loss_history.append(np.mean((pred - y_data)**2))

    elapsed = time.time() - start_time

    # Accuracy
    preds = net.forward(X_data)
    accuracy = np.mean((preds > 0.5).astype(float) == y_data) * 100

    results_gd[variant] = {
        'losses': loss_history,
        'time': elapsed,
        'accuracy': accuracy,
        'final_loss': loss_history[-1]
    }

print(f"\n  {'Variant':>12} | {'Final Loss':>12} | {'Accuracy':>10} | {'Time (s)':>10}")
print(f"  {'-'*12} | {'-'*12} | {'-'*10} | {'-'*10}")
for v in ['batch', 'sgd', 'minibatch']:
    r = results_gd[v]
    print(f"  {v:>12} | {r['final_loss']:>12.6f} | {r['accuracy']:>9.1f}% | {r['time']:>10.4f}")

print("""
  Analysis:
  - Batch: smoothest loss curve, but sees the full dataset before each update
  - SGD: most updates per epoch (N updates), noisier, but can escape local minima
  - Mini-batch: compromise -- stable enough, fast enough, GPU-friendly (parallelism)
  - All three converge to similar accuracy (the destination is the same, the path differs)
  - Mini-batch is the standard because it exploits GPU parallelism on batched matrix ops
""")


# ============================================================================
# EXERCISE 7: Deep Network — Generalized N-layer backprop
# ============================================================================

print("=" * 70)
print("EXERCISE 7: DeepNetwork -- Arbitrary Depth")
print("=" * 70)


class DeepNetwork:
    """N-layer neural network with generalized backpropagation.

    Supports: sigmoid, tanh, relu activations
    Supports: MSE, BCE losses
    Uses Xavier init for sigmoid/tanh, He init for relu.
    """

    def __init__(self, layer_sizes, activations, loss_fn='mse', seed=42):
        """
        layer_sizes: list of ints, e.g. [2, 8, 4, 1]
        activations: list of strings, len = len(layer_sizes) - 1
                     e.g. ['relu', 'relu', 'sigmoid']
        loss_fn: 'mse' or 'bce'
        """
        assert len(activations) == len(layer_sizes) - 1
        np.random.seed(seed)

        self.n_layers = len(layer_sizes) - 1  # number of weight matrices
        self.activations = activations
        self.loss_fn = loss_fn

        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            # He init for ReLU, Xavier for sigmoid/tanh
            if activations[i] == 'relu':
                scale = np.sqrt(2.0 / n_in)  # He initialization
            else:
                scale = np.sqrt(1.0 / n_in)  # Xavier initialization

            W = np.random.randn(n_in, n_out) * scale
            b_param = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b_param)

    def _activate(self, z, activation):
        """Apply activation function."""
        if activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _activate_deriv(self, a, z, activation):
        """Derivative of activation w.r.t. z."""
        if activation == 'sigmoid':
            return a * (1.0 - a)
        elif activation == 'tanh':
            return 1.0 - a ** 2
        elif activation == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, X):
        """Forward pass through all layers.
        Stores z and a for each layer (needed for backprop).
        """
        self.z_cache = []  # pre-activation values
        self.a_cache = [X]  # post-activation values (input is a_0)

        current = X
        for i in range(self.n_layers):
            z = current @ self.weights[i] + self.biases[i]
            a = self._activate(z, self.activations[i])
            self.z_cache.append(z)
            self.a_cache.append(a)
            current = a

        return current  # final output

    def compute_loss(self, y_true):
        """Compute loss and return scalar value."""
        pred = self.a_cache[-1]
        n = y_true.shape[0]

        if self.loss_fn == 'mse':
            self.loss = np.mean((pred - y_true) ** 2)
        elif self.loss_fn == 'bce':
            p = np.clip(pred, 1e-7, 1 - 1e-7)
            self.loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        return self.loss

    def backward(self, y_true):
        """Backpropagation through all layers using the chain rule."""
        n = y_true.shape[0]
        pred = self.a_cache[-1]

        # Gradient of loss w.r.t. final activation
        if self.loss_fn == 'mse':
            dL_da = (2.0 / n) * (pred - y_true)
        elif self.loss_fn == 'bce':
            p = np.clip(pred, 1e-7, 1 - 1e-7)
            dL_da = (-y_true / p + (1 - y_true) / (1 - p)) / n

        self.grad_W = []
        self.grad_b = []

        # Backward through each layer (from last to first)
        delta = dL_da  # this will become ∂L/∂z_l at each step
        for i in range(self.n_layers - 1, -1, -1):
            # delta is currently ∂L/∂a_l — need to multiply by ∂a_l/∂z_l
            da_dz = self._activate_deriv(self.a_cache[i + 1], self.z_cache[i],
                                          self.activations[i])
            delta_z = delta * da_dz  # ∂L/∂z_l

            # Gradients for this layer's weights and biases
            dW = self.a_cache[i].T @ delta_z  # ∂L/∂W_l = a_{l-1}^T @ δ_l
            db = np.sum(delta_z, axis=0, keepdims=True)  # ∂L/∂b_l

            # Store gradients (prepend so index matches)
            self.grad_W.insert(0, dW)
            self.grad_b.insert(0, db)

            # Propagate error to previous layer: ∂L/∂a_{l-1} = δ_l @ W_l^T
            delta = delta_z @ self.weights[i].T

    def update(self, lr):
        """Gradient descent weight update."""
        for i in range(self.n_layers):
            self.weights[i] -= lr * self.grad_W[i]
            self.biases[i] -= lr * self.grad_b[i]

    def train_step(self, X, y, lr):
        self.forward(X)
        loss = self.compute_loss(y)
        self.backward(y)
        self.update(lr)
        return loss


# --- Test 1: XOR ---
print("\n  Test 1: XOR (2->4->1)")
net_deep = DeepNetwork([2, 4, 1], ['relu', 'sigmoid'], loss_fn='bce', seed=42)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

for ep in range(5000):
    loss = net_deep.train_step(X_xor, y_xor, lr=0.5)

preds_xor = net_deep.forward(X_xor)
acc_xor = np.mean((preds_xor > 0.5).astype(float) == y_xor) * 100
print(f"    Final loss: {loss:.6f}, Accuracy: {acc_xor:.0f}%")
for i in range(4):
    print(f"    {X_xor[i]} -> {preds_xor[i][0]:.4f} (target: {y_xor[i][0]})")

# --- Test 2: Circles ---
print("\n  Test 2: Concentric circles (2->8->4->1)")
try:
    from sklearn.datasets import make_circles, make_moons
    X_circ, y_circ_flat = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    y_circ = y_circ_flat.reshape(-1, 1).astype(float)

    net_circ = DeepNetwork([2, 8, 4, 1], ['relu', 'relu', 'sigmoid'], loss_fn='bce', seed=42)
    for ep in range(3000):
        loss = net_circ.train_step(X_circ, y_circ, lr=0.1)

    preds_circ = net_circ.forward(X_circ)
    acc_circ = np.mean((preds_circ > 0.5).astype(float) == y_circ) * 100
    print(f"    Final loss: {loss:.6f}, Accuracy: {acc_circ:.0f}%")

    # --- Test 3: Moons ---
    print("\n  Test 3: Moons (2->8->4->1)")
    X_moon, y_moon_flat = make_moons(n_samples=200, noise=0.1, random_state=42)
    y_moon = y_moon_flat.reshape(-1, 1).astype(float)

    net_moon = DeepNetwork([2, 8, 4, 1], ['relu', 'relu', 'sigmoid'], loss_fn='bce', seed=42)
    for ep in range(3000):
        loss = net_moon.train_step(X_moon, y_moon, lr=0.1)

    preds_moon = net_moon.forward(X_moon)
    acc_moon = np.mean((preds_moon > 0.5).astype(float) == y_moon) * 100
    print(f"    Final loss: {loss:.6f}, Accuracy: {acc_moon:.0f}%")

except ImportError:
    print("    [sklearn not available -- skipping circles/moons tests]")

# --- Gradient Check ---
print("\n  Gradient Check (numerical vs analytical):")
net_check = DeepNetwork([2, 4, 2, 1], ['relu', 'relu', 'sigmoid'], loss_fn='mse', seed=42)
X_check = np.array([[0.5, -0.3], [0.2, 0.8]])
y_check = np.array([[1.0], [0.0]])

# Run forward + backward to get analytical gradients
net_check.forward(X_check)
net_check.compute_loss(y_check)
net_check.backward(y_check)

epsilon = 1e-5
max_errors = []
total_checked = 0
total_passed = 0

for l in range(len(net_check.weights)):
    for i in range(net_check.weights[l].shape[0]):
        for j in range(net_check.weights[l].shape[1]):
            original = net_check.weights[l][i, j]

            net_check.weights[l][i, j] = original + epsilon
            net_check.forward(X_check)
            loss_plus = net_check.compute_loss(y_check)

            net_check.weights[l][i, j] = original - epsilon
            net_check.forward(X_check)
            loss_minus = net_check.compute_loss(y_check)

            net_check.weights[l][i, j] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = net_check.grad_W[l][i, j]

            denom = abs(analytical) + abs(numerical) + 1e-8
            error = abs(analytical - numerical) / denom

            total_checked += 1
            if error < 1e-4:
                total_passed += 1
            else:
                print(f"    MISMATCH L{l}[{i},{j}]: anal={analytical:.8f}, "
                      f"num={numerical:.8f}, err={error:.6f}")

print(f"    Checked {total_checked} weights: {total_passed}/{total_checked} passed (threshold: 1e-4)")


# ============================================================================
# EXERCISE 8: Loss Surface Visualization
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Loss Surface & Gradient Trajectory")
print("=" * 70)

# Simple 2-point problem with single neuron (2 weights, no bias)
X_surf = np.array([[1.0, 0.0], [0.0, 1.0]])
y_surf = np.array([[1.0], [0.0]])


def compute_loss_surface(w1_range, w2_range):
    """Compute MSE loss for a zone of (w1, w2) values.
    Neuron: a = sigmoid(w1*x1 + w2*x2), Loss = MSE.
    """
    W1, W2 = np.meshgrid(w1_range, w2_range)
    losses = np.zeros_like(W1)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = np.array([W1[i, j], W2[i, j]])
            # Forward: prediction for each sample
            z = X_surf @ w.reshape(-1, 1)  # (2, 1)
            a = sigmoid(z)
            loss = np.mean((a - y_surf) ** 2)
            losses[i, j] = loss

    return W1, W2, losses


def gradient_trajectory(w1_init, w2_init, lr, epochs):
    """Run gradient descent and record the trajectory of (w1, w2)."""
    w = np.array([w1_init, w2_init]).reshape(2, 1)
    trajectory = [(w[0, 0], w[1, 0])]

    for _ in range(epochs):
        # Forward
        z = X_surf @ w  # (2, 1)
        a = sigmoid(z)

        # Loss gradient
        n = X_surf.shape[0]
        dL_da = (2.0 / n) * (a - y_surf)
        da_dz = a * (1 - a)
        delta = dL_da * da_dz
        dL_dw = X_surf.T @ delta  # (2, 1)

        # Update
        w = w - lr * dL_dw
        trajectory.append((w[0, 0], w[1, 0]))

    return trajectory


# Compute loss surface
w_range = np.linspace(-5, 5, 100)
W1, W2, L = compute_loss_surface(w_range, w_range)

# Compute trajectories for different learning rates
trajectories = {}
for test_lr in [0.1, 1.0, 5.0]:
    trajectories[test_lr] = gradient_trajectory(4.0, -4.0, test_lr, 100)

# Print summary (text-based since matplotlib may not be available)
print("\n  Loss surface computed: 100x100 zone, w1/w2 in [-5, 5]")
print(f"  Min loss on zone: {L.min():.6f}")
min_idx = np.unravel_index(L.argmin(), L.shape)
print(f"  Minimum at: w1={W1[min_idx]:.2f}, w2={W2[min_idx]:.2f}")

print("\n  Gradient descent trajectories (start: w1=4, w2=-4):")
for test_lr, traj in trajectories.items():
    start = traj[0]
    end = traj[-1]
    # Compute final loss
    w_final = np.array([end[0], end[1]]).reshape(2, 1)
    z_final = X_surf @ w_final
    a_final = sigmoid(z_final)
    loss_final = np.mean((a_final - y_surf) ** 2)
    print(f"    lr={test_lr}: ({start[0]:.1f}, {start[1]:.1f}) -> ({end[0]:.2f}, {end[1]:.2f}), loss={loss_final:.6f}")

# Matplotlib visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(18, 5))

    # Plot 1: 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(W1, W2, L, cmap=cm.viridis, alpha=0.8, edgecolor='none')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Surface (3D)')
    ax1.view_init(elev=30, azim=45)

    # Plot 2: Contour plot with all trajectories
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(W1, W2, L, levels=30, cmap=cm.viridis)
    plt.colorbar(contour, ax=ax2)

    colors_lr = {0.1: 'red', 1.0: 'white', 5.0: 'cyan'}
    for test_lr, traj in trajectories.items():
        xs = [t[0] for t in traj]
        ys = [t[1] for t in traj]
        ax2.plot(xs, ys, '-o', color=colors_lr[test_lr], markersize=2,
                linewidth=1.5, label=f'lr={test_lr}')
        ax2.plot(xs[0], ys[0], 'o', color=colors_lr[test_lr], markersize=8)  # start
        ax2.plot(xs[-1], ys[-1], '*', color=colors_lr[test_lr], markersize=12)  # end

    ax2.set_xlabel('w1')
    ax2.set_ylabel('w2')
    ax2.set_title('Contours + GD Trajectories')
    ax2.legend()

    # Plot 3: Loss over epochs for each LR
    ax3 = fig.add_subplot(133)
    for test_lr, traj in trajectories.items():
        losses_traj = []
        for w1_t, w2_t in traj:
            w_t = np.array([w1_t, w2_t]).reshape(2, 1)
            z_t = X_surf @ w_t
            a_t = sigmoid(z_t)
            l_t = np.mean((a_t - y_surf) ** 2)
            losses_traj.append(l_t)
        ax3.plot(losses_traj, label=f'lr={test_lr}', linewidth=1.5)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE Loss')
    ax3.set_title('Loss per Epoch')
    ax3.legend()
    ax3.zone(True, alpha=0.3)

    plt.suptitle('Exercise 8: Loss Surface & Gradient Descent Visualization', fontsize=13)
    plt.tight_layout()
    plt.savefig('loss_surface_visualization.png', dpi=150)
    print("\n    [Matplotlib] Saved to loss_surface_visualization.png")
    plt.close()

except ImportError:
    print("\n    [Matplotlib not installed -- text output only]")

print("""
  Answer to the convexity question:
  ----------------------------------
  For a single neuron with sigmoid activation, the loss surface is NOT convex
  (it's sigmoid-shaped in each weight dimension, creating a wavy landscape).

  For a network with hidden layers, the loss surface is HIGHLY non-convex:
  - Multiple local minima (many "valleys" of different depths)
  - Saddle points (flat in some directions, curved in others -- very common
    in high dimensions, more common than local minima actually)
  - Plateaus (flat regions where the gradient is near-zero)

  This means:
  1. GD can converge to different solutions depending on initialization
  2. There's no guarantee of finding the global minimum
  3. In practice, modern deep nets have so many parameters that most local
     minima are "good enough" -- the bigger problem is saddle points
  4. Techniques like momentum, Adam, and learning rate scheduling help
     navigate this landscape more effectively
""")

print("=" * 70)
print("ALL EXERCISES COMPLETED")
print("=" * 70)

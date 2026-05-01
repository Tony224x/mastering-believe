"""
Solutions — Jour 2 : Reseaux denses (MLP)
==========================================
Solutions pour les 8 exercices (easy, medium, hard).

Run: python 03-exercises/solutions/02-reseaux-denses-mlp.py
"""

import sys
import io
import numpy as np
import time

# Force UTF-8 output to handle special characters in comments/docstrings
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# HELPER: Activation functions (reused across exercises)
# ============================================================================

def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z))"""
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


def relu(z):
    """ReLU(z) = max(0, z)"""
    return np.maximum(0, z)


def relu_deriv(z):
    """d/dz ReLU = 1 if z > 0, 0 otherwise"""
    return (z > 0).astype(float)


def softmax(z):
    """Numerically stable softmax with log-sum-exp trick."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ============================================================================
# EXERCISE 1: Forward pass matriciel a la main
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Forward pass matriciel a la main")
print("=" * 70)

# Network [3, 2, 1] — 3 inputs, 2 hidden (ReLU), 1 output (sigmoid)
X = np.array([[1.0, 0.5, -0.3]])    # (1, 3) — one sample

W1 = np.array([[ 0.2,  0.5],        # (3, 2)
               [-0.1,  0.3],
               [ 0.4, -0.2]])

b1 = np.array([[0.1, -0.1]])        # (1, 2)

W2 = np.array([[ 0.6],              # (2, 1)
               [-0.4]])

b2 = np.array([[0.05]])             # (1, 1)

y_true = 1.0

# Step 1: Z1 = X @ W1 + b1
# (1,3) @ (3,2) = (1,2), then add (1,2) broadcast
print("\nStep 1 — Z1 = X @ W1 + b1")
print(f"  Dimensions: ({X.shape}) @ ({W1.shape}) + ({b1.shape}) = (1, 2)")
XW1 = X @ W1
print(f"  X @ W1 = {XW1}")
Z1 = XW1 + b1
print(f"  Z1 = X@W1 + b1 = {Z1}")

# Detailed: z1_1 = 1.0*0.2 + 0.5*(-0.1) + (-0.3)*0.4 + 0.1
#          = 0.2 - 0.05 - 0.12 + 0.1 = 0.13
# z1_2 = 1.0*0.5 + 0.5*0.3 + (-0.3)*(-0.2) + (-0.1)
#       = 0.5 + 0.15 + 0.06 - 0.1 = 0.61
print(f"  z1_1 = 1.0*0.2 + 0.5*(-0.1) + (-0.3)*0.4 + 0.1 = {Z1[0,0]:.4f}")
print(f"  z1_2 = 1.0*0.5 + 0.5*0.3 + (-0.3)*(-0.2) + (-0.1) = {Z1[0,1]:.4f}")

# Step 2: A1 = ReLU(Z1)
A1 = relu(Z1)
print(f"\nStep 2 — A1 = ReLU(Z1)")
print(f"  A1 = ReLU({Z1}) = {A1}")
print(f"  (Both values > 0, so ReLU doesn't change them)")

# Step 3: Z2 = A1 @ W2 + b2
# (1,2) @ (2,1) = (1,1), then add (1,1)
print(f"\nStep 3 — Z2 = A1 @ W2 + b2")
print(f"  Dimensions: ({A1.shape}) @ ({W2.shape}) + ({b2.shape}) = (1, 1)")
Z2 = A1 @ W2 + b2
print(f"  Z2 = {A1[0,0]:.4f}*0.6 + {A1[0,1]:.4f}*(-0.4) + 0.05")
print(f"  Z2 = {A1[0,0]*0.6:.4f} + {A1[0,1]*(-0.4):.4f} + 0.05 = {Z2[0,0]:.4f}")

# Step 4: A2 = sigmoid(Z2)
A2 = sigmoid(Z2)
print(f"\nStep 4 — A2 = sigmoid(Z2)")
print(f"  A2 = sigmoid({Z2[0,0]:.4f}) = 1 / (1 + e^(-{Z2[0,0]:.4f})) = {A2[0,0]:.4f}")

# Step 5: BCE loss
eps = 1e-12
p = np.clip(A2[0, 0], eps, 1 - eps)
bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
print(f"\nStep 5 — BCE Loss (y=1)")
print(f"  L = -[1*log({p:.4f}) + 0*log({1-p:.4f})]")
print(f"  L = -log({p:.4f}) = {bce:.4f}")


# ============================================================================
# EXERCISE 2: Compare 3 loss functions
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Comparing MSE, BCE, and CCE")
print("=" * 70)

# Binary classification: MSE vs BCE
predictions = [0.99, 0.9, 0.7, 0.5, 0.1, 0.01]
y = 1.0

print(f"\n  Target y = {y}")
print(f"  {'p':>5s} | {'MSE':>8s} | {'BCE':>8s} | {'MSE grad':>9s} | {'BCE grad':>9s}")
print(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9}")

for p in predictions:
    mse = (p - y) ** 2
    bce = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    mse_grad = 2 * (p - y)                              # ∂MSE/∂p = 2(p - y)
    bce_grad = -y / (p + 1e-12) + (1 - y) / (1 - p + 1e-12)  # ∂BCE/∂p
    print(f"  {p:>5.2f} | {mse:>8.4f} | {bce:>8.4f} | {mse_grad:>+9.4f} | {bce_grad:>+9.4f}")

print("""
  Observations:
  - When p=0.01 (very wrong, target=1): MSE=0.98, BCE=4.61 — BCE penalizes 4.7x more
  - BCE gradient at p=0.01: -100 vs MSE gradient: -1.98 — BCE gradient 50x stronger
  - BCE forces faster learning from confident mistakes — no gradient saturation
""")

# Multi-class: Softmax + CCE
print("  Multi-class (3 classes, true class = 0):")
print(f"  {'Case':<15s} | {'Logits':>20s} | {'Softmax':>25s} | {'CCE':>8s} | {'Gradient':>25s}")
print(f"  {'-'*15} | {'-'*20} | {'-'*25} | {'-'*8} | {'-'*25}")

cases = [
    ("Confident+right", [3.0, 1.0, 0.5]),
    ("Confident+wrong", [0.5, 3.0, 1.0]),
    ("Uncertain",       [1.0, 1.0, 1.0]),
]
y_oh = np.array([1, 0, 0])

for label, logits in cases:
    z = np.array([logits])
    p = softmax(z)[0]
    cce = -np.sum(y_oh * np.log(p + 1e-12))
    grad = p - y_oh                                    # elegant: softmax - onehot
    logit_str = f"[{', '.join(f'{l:.1f}' for l in logits)}]"
    prob_str = f"[{', '.join(f'{pp:.4f}' for pp in p)}]"
    grad_str = f"[{', '.join(f'{g:+.4f}' for g in grad)}]"
    print(f"  {label:<15s} | {logit_str:>20s} | {prob_str:>25s} | {cce:>8.4f} | {grad_str:>25s}")

print("""
  Key: gradient = softmax - onehot.
  For the confident wrong case, gradient pushes class 1 down (-) and class 0 up (+).
""")


# ============================================================================
# EXERCISE 3: Adam step by step
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Adam Step-by-Step for a Single Weight")
print("=" * 70)

w = 5.0
gradients = [2.0, 1.8, 2.1, -0.5, -0.3, 1.5, 1.0, 0.8, 0.5, 0.2]
lr = 0.1
beta1, beta2 = 0.9, 0.999
eps = 1e-8

m, v = 0.0, 0.0
w_sgd = 5.0  # for comparison

print(f"\n  Initial w = {w}, lr = {lr}, β1 = {beta1}, β2 = {beta2}")
print(f"\n  {'t':>3s} | {'g':>6s} | {'m':>8s} | {'v':>10s} | {'m_hat':>8s} | {'v_hat':>10s} | {'Δw':>8s} | {'w_adam':>8s} | {'w_sgd':>8s}")
print(f"  {'-'*3} | {'-'*6} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8}")

for t, g in enumerate(gradients, 1):
    # Adam update
    m = beta1 * m + (1 - beta1) * g             # 1st moment: running mean of gradient
    v = beta2 * v + (1 - beta2) * g ** 2         # 2nd moment: running mean of gradient^2
    m_hat = m / (1 - beta1 ** t)                  # bias correction (crucial early)
    v_hat = v / (1 - beta2 ** t)                  # bias correction
    delta_w = lr * m_hat / (np.sqrt(v_hat) + eps) # actual step size
    w = w - delta_w

    # SGD update (for comparison)
    w_sgd = w_sgd - lr * g

    print(f"  {t:>3d} | {g:>+6.2f} | {m:>8.4f} | {v:>10.6f} | {m_hat:>8.4f} | {v_hat:>10.6f} | {delta_w:>8.4f} | {w:>8.4f} | {w_sgd:>8.4f}")

print(f"""
  Observations:
  - m (1st moment) smooths the gradient direction — when gradient flips sign (t=4,5),
    m doesn't flip immediately, it gradually adjusts.
  - v (2nd moment) tracks gradient magnitude — stays high because of earlier large gradients.
  - Bias correction: at t=1, m_hat = 10x m because 1/(1-0.9^1) = 10. By t=10, factor is ~1.5.
  - Adam moves more smoothly than SGD. SGD overshoots on large gradients (t=1-3)
    and barely moves on small ones (t=10).
  - Final: Adam w = {w:.4f}, SGD w = {w_sgd:.4f}
""")


# ============================================================================
# EXERCISE 4: Multi-class MLP (softmax + CCE)
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Multi-class MLP with Softmax + CCE")
print("=" * 70)

# Generate 3-class dataset
np.random.seed(42)
n = 100
X0 = np.random.randn(n, 2) * 0.4 + np.array([0, 2])
X1 = np.random.randn(n, 2) * 0.4 + np.array([-1.5, -1])
X2 = np.random.randn(n, 2) * 0.4 + np.array([1.5, -1])
X_mc = np.vstack([X0, X1, X2])
y_labels = np.array([0]*n + [1]*n + [2]*n)
y_onehot = np.zeros((3*n, 3))
y_onehot[np.arange(3*n), y_labels] = 1

# Shuffle
perm = np.random.permutation(3*n)
X_mc = X_mc[perm]
y_onehot = y_onehot[perm]
y_labels = y_labels[perm]

print(f"\n  Dataset: 3 Gaussian clusters in 2D")
print(f"  {3*n} samples, {3} classes")


class MultiClassMLP:
    """MLP for multi-class classification with softmax + CCE."""

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.L = len(layer_sizes) - 1
        self.W = []
        self.b = []
        for l in range(self.L):
            n_in, n_out = layer_sizes[l], layer_sizes[l+1]
            self.W.append(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            self.b.append(np.zeros((1, n_out)))

    def forward(self, X):
        self.A = [X]
        self.Z = [None]
        A_prev = X
        for l in range(self.L):
            Z_l = A_prev @ self.W[l] + self.b[l]
            self.Z.append(Z_l)
            if l < self.L - 1:
                A_l = relu(Z_l)                          # hidden: ReLU
            else:
                A_l = softmax(Z_l)                       # output: softmax
            self.A.append(A_l)
            A_prev = A_l
        return self.A[-1]

    def compute_loss(self, y_true_oh):
        m = y_true_oh.shape[0]
        p = np.clip(self.A[-1], 1e-12, 1.0)
        self.loss = -np.mean(np.sum(y_true_oh * np.log(p), axis=1))
        return self.loss

    def backward(self, X, y_true_oh):
        m = X.shape[0]
        self.dW = [None] * self.L
        self.db = [None] * self.L

        # Output: gradient = (softmax - y) / m  (combined softmax + CCE)
        delta = (1.0 / m) * (self.A[-1] - y_true_oh)

        for l in range(self.L - 1, -1, -1):
            self.dW[l] = self.A[l].T @ delta
            self.db[l] = np.sum(delta, axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_deriv(self.Z[l])


# Train
net_mc = MultiClassMLP([2, 32, 16, 3], seed=42)

# Adam optimizer state
m_W = [np.zeros_like(w) for w in net_mc.W]
m_b = [np.zeros_like(b) for b in net_mc.b]
v_W = [np.zeros_like(w) for w in net_mc.W]
v_b = [np.zeros_like(b) for b in net_mc.b]
adam_lr, adam_b1, adam_b2, adam_eps = 0.005, 0.9, 0.999, 1e-8

print(f"\n  Architecture: [2, 32, 16, 3] | Optimizer: Adam(lr={adam_lr})")
print(f"  {'Epoch':>6s} | {'Loss':>8s} | {'Accuracy':>10s}")
print(f"  {'-'*6} | {'-'*8} | {'-'*10}")

for epoch in range(501):
    # Forward + loss
    probs = net_mc.forward(X_mc)
    loss = net_mc.compute_loss(y_onehot)

    # Accuracy
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_labels)

    if epoch % 50 == 0:
        print(f"  {epoch:>6d} | {loss:>8.4f} | {acc:>9.2%}")

    # Backward
    net_mc.backward(X_mc, y_onehot)

    # Adam step
    t = epoch + 1
    for l in range(net_mc.L):
        m_W[l] = adam_b1 * m_W[l] + (1 - adam_b1) * net_mc.dW[l]
        m_b[l] = adam_b1 * m_b[l] + (1 - adam_b1) * net_mc.db[l]
        v_W[l] = adam_b2 * v_W[l] + (1 - adam_b2) * net_mc.dW[l]**2
        v_b[l] = adam_b2 * v_b[l] + (1 - adam_b2) * net_mc.db[l]**2
        mhat_W = m_W[l] / (1 - adam_b1**t)
        mhat_b = m_b[l] / (1 - adam_b1**t)
        vhat_W = v_W[l] / (1 - adam_b2**t)
        vhat_b = v_b[l] / (1 - adam_b2**t)
        net_mc.W[l] -= adam_lr * mhat_W / (np.sqrt(vhat_W) + adam_eps)
        net_mc.b[l] -= adam_lr * mhat_b / (np.sqrt(vhat_b) + adam_eps)

# Confusion matrix
final_preds = np.argmax(net_mc.forward(X_mc), axis=1)
confusion = np.zeros((3, 3), dtype=int)
for true, pred in zip(y_labels, final_preds):
    confusion[true, pred] += 1

print(f"\n  Confusion Matrix:")
print(f"  {'':>12s}  Pred 0  Pred 1  Pred 2")
for i in range(3):
    print(f"  {'Actual ' + str(i):>12s}  {confusion[i, 0]:>6d}  {confusion[i, 1]:>6d}  {confusion[i, 2]:>6d}")
print(f"\n  Final accuracy: {np.mean(final_preds == y_labels):.2%}")

# Gradient check (5 random weights)
print(f"\n  Gradient Check (5 random weights):")
net_mc.forward(X_mc)
net_mc.compute_loss(y_onehot)
net_mc.backward(X_mc, y_onehot)

np.random.seed(123)
epsilon = 1e-7
all_ok = True
for _ in range(5):
    l = np.random.randint(0, net_mc.L)
    i = np.random.randint(0, net_mc.W[l].shape[0])
    j = np.random.randint(0, net_mc.W[l].shape[1])

    original = net_mc.W[l][i, j]

    net_mc.W[l][i, j] = original + epsilon
    net_mc.forward(X_mc)
    loss_plus = net_mc.compute_loss(y_onehot)

    net_mc.W[l][i, j] = original - epsilon
    net_mc.forward(X_mc)
    loss_minus = net_mc.compute_loss(y_onehot)

    net_mc.W[l][i, j] = original

    numerical = (loss_plus - loss_minus) / (2 * epsilon)
    analytical = net_mc.dW[l][i, j]
    error = abs(analytical - numerical) / (abs(analytical) + abs(numerical) + 1e-8)

    status = "PASS" if error < 1e-5 else "FAIL"
    if error >= 1e-5:
        all_ok = False
    print(f"    Layer {l}, W[{i},{j}]: analytical={analytical:.8f}, numerical={numerical:.8f}, error={error:.2e} [{status}]")

print(f"  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")


# ============================================================================
# EXERCISE 5: Systematic optimizer comparison
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Systematic Optimizer Comparison")
print("=" * 70)


def make_moons(n=300, noise=0.15, seed=42):
    """Generate two interleaved half-circles (moons dataset)."""
    np.random.seed(seed)
    t = np.linspace(0, np.pi, n)
    x1 = np.c_[np.cos(t), np.sin(t)]
    x2 = np.c_[np.cos(t) + 0.5, -np.sin(t) + 0.5]
    X = np.vstack([x1, x2]) + np.random.randn(2*n, 2) * noise
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    perm = np.random.permutation(2*n)
    return X[perm], y[perm]


class SimpleMLP:
    """Minimal MLP for optimizer testing: [n_in, h1, h2, 1], ReLU, sigmoid, BCE."""

    def __init__(self, sizes, seed=42):
        np.random.seed(seed)
        self.L = len(sizes) - 1
        self.W = [np.random.randn(sizes[l], sizes[l+1]) * np.sqrt(2.0 / sizes[l])
                  for l in range(self.L)]
        self.b = [np.zeros((1, sizes[l+1])) for l in range(self.L)]

    def forward(self, X):
        self.A = [X]
        self.Z = [None]
        A = X
        for l in range(self.L):
            Z = A @ self.W[l] + self.b[l]
            self.Z.append(Z)
            A = relu(Z) if l < self.L - 1 else sigmoid(Z)
            self.A.append(A)
        return self.A[-1]

    def loss_and_backward(self, X, y):
        m = X.shape[0]
        p = np.clip(self.A[-1], 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        delta = (1.0 / m) * (self.A[-1] - y)
        self.dW = [None] * self.L
        self.db = [None] * self.L
        for l in range(self.L - 1, -1, -1):
            self.dW[l] = self.A[l].T @ delta
            self.db[l] = np.sum(delta, axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ self.W[l].T) * relu_deriv(self.Z[l])
        return loss


X_moons, y_moons = make_moons(n=300, noise=0.15, seed=42)
n_tr = int(0.7 * len(X_moons))
X_tr, y_tr = X_moons[:n_tr], y_moons[:n_tr]
X_vl, y_vl = X_moons[n_tr:], y_moons[n_tr:]

optimizer_configs = [
    ("SGD lr=0.1", 'sgd', {'lr': 0.1}),
    ("Momentum",   'momentum', {'lr': 0.05, 'beta': 0.9}),
    ("RMSProp",    'rmsprop', {'lr': 0.005, 'beta': 0.999}),
    ("Adam",       'adam', {'lr': 0.005}),
]

print(f"\n  Dataset: moons, {n_tr} train, {len(X_moons)-n_tr} val")
print(f"  Architecture: [2, 32, 16, 1]")

opt_results = {}

for name, opt_type, params in optimizer_configs:
    net = SimpleMLP([2, 32, 16, 1], seed=42)
    lr = params['lr']

    # Optimizer state
    if opt_type == 'momentum':
        vw = [np.zeros_like(w) for w in net.W]
        vb = [np.zeros_like(b) for b in net.b]
        beta = params['beta']
    elif opt_type == 'rmsprop':
        sw = [np.zeros_like(w) for w in net.W]
        sb = [np.zeros_like(b) for b in net.b]
        beta = params['beta']
    elif opt_type == 'adam':
        mw = [np.zeros_like(w) for w in net.W]
        mb = [np.zeros_like(b) for b in net.b]
        vw2 = [np.zeros_like(w) for w in net.W]
        vb2 = [np.zeros_like(b) for b in net.b]

    losses = []
    t_start = time.time()
    epochs_to_02 = None

    for epoch in range(1000):
        net.forward(X_tr)
        loss = net.loss_and_backward(X_tr, y_tr)
        losses.append(loss)

        if loss < 0.2 and epochs_to_02 is None:
            epochs_to_02 = epoch

        # Optimizer step
        for l in range(net.L):
            if opt_type == 'sgd':
                net.W[l] -= lr * net.dW[l]
                net.b[l] -= lr * net.db[l]
            elif opt_type == 'momentum':
                vw[l] = beta * vw[l] + net.dW[l]
                vb[l] = beta * vb[l] + net.db[l]
                net.W[l] -= lr * vw[l]
                net.b[l] -= lr * vb[l]
            elif opt_type == 'rmsprop':
                sw[l] = beta * sw[l] + (1-beta) * net.dW[l]**2
                sb[l] = beta * sb[l] + (1-beta) * net.db[l]**2
                net.W[l] -= lr * net.dW[l] / (np.sqrt(sw[l]) + 1e-8)
                net.b[l] -= lr * net.db[l] / (np.sqrt(sb[l]) + 1e-8)
            elif opt_type == 'adam':
                t = epoch + 1
                mw[l] = 0.9 * mw[l] + 0.1 * net.dW[l]
                mb[l] = 0.9 * mb[l] + 0.1 * net.db[l]
                vw2[l] = 0.999 * vw2[l] + 0.001 * net.dW[l]**2
                vb2[l] = 0.999 * vb2[l] + 0.001 * net.db[l]**2
                mwh = mw[l] / (1 - 0.9**t)
                mbh = mb[l] / (1 - 0.9**t)
                vwh = vw2[l] / (1 - 0.999**t)
                vbh = vb2[l] / (1 - 0.999**t)
                net.W[l] -= lr * mwh / (np.sqrt(vwh) + 1e-8)
                net.b[l] -= lr * mbh / (np.sqrt(vbh) + 1e-8)

    elapsed = time.time() - t_start

    # Validation accuracy
    val_pred = net.forward(X_vl)
    val_acc = np.mean((val_pred > 0.5).astype(float) == y_vl)

    opt_results[name] = {
        'losses': losses,
        'final_loss': losses[-1],
        'epochs_to_02': epochs_to_02 if epochs_to_02 else '>1000',
        'val_acc': val_acc,
        'time': elapsed,
    }

print(f"\n  {'Optimizer':<15s} | {'Final Loss':>10s} | {'Epochs<0.2':>10s} | {'Val Acc':>8s} | {'Time':>7s}")
print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*7}")
for name, res in opt_results.items():
    print(f"  {name:<15s} | {res['final_loss']:>10.4f} | {str(res['epochs_to_02']):>10s} | {res['val_acc']:>7.2%} | {res['time']:>6.3f}s")

print("""
  Analysis:
  - SGD is slowest to converge — it oscillates without momentum to stabilize
  - Momentum helps significantly — it accumulates velocity in consistent directions
  - RMSProp adapts LR per parameter — converges faster than vanilla SGD
  - Adam typically converges fastest — combines momentum + adaptive LR
  - All reach similar final accuracy — the optimizer affects SPEED, not final quality (mostly)
""")


# ============================================================================
# EXERCISE 6: Overfitting study
# ============================================================================

print("=" * 70)
print("EXERCISE 6: Overfitting & Regularization Study")
print("=" * 70)

# Small dataset for easy overfitting
np.random.seed(42)
n_small = 50
X_s0 = np.random.randn(n_small, 2) * 0.5 + np.array([1, 1])
X_s1 = np.random.randn(n_small, 2) * 0.5 + np.array([-1, -1])
X_small = np.vstack([X_s0, X_s1])
y_small = np.array([0]*n_small + [1]*n_small).reshape(-1, 1).astype(float)
perm = np.random.permutation(2*n_small)
X_small, y_small = X_small[perm], y_small[perm]

# Split 60/40
n_split = int(0.6 * len(X_small))
X_st, y_st = X_small[:n_split], y_small[:n_split]
X_sv, y_sv = X_small[n_split:], y_small[n_split:]

print(f"\n  Small dataset: {n_split} train, {len(X_small)-n_split} val")
print(f"  Big network: [2, 256, 128, 64, 1] — way too big for {n_split} samples")


def train_with_reg(X_t, y_t, X_v, y_v, l2=0.0, dropout=0.0, epochs=1500, early_stop_patience=0):
    """Train a big MLP with optional L2, dropout, and early stopping.
    Returns train/val loss histories and best val accuracy."""
    np.random.seed(42)
    sizes = [2, 256, 128, 64, 1]
    L = len(sizes) - 1
    W = [np.random.randn(sizes[l], sizes[l+1]) * np.sqrt(2.0/sizes[l]) for l in range(L)]
    b = [np.zeros((1, sizes[l+1])) for l in range(L)]

    # Adam state
    mw = [np.zeros_like(w) for w in W]
    mb_ = [np.zeros_like(bb) for bb in b]
    vw = [np.zeros_like(w) for w in W]
    vb_ = [np.zeros_like(bb) for bb in b]

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_W = [w.copy() for w in W]
    best_b = [bb.copy() for bb in b]

    for epoch in range(epochs):
        # Forward (training mode)
        A = [X_t]
        Z_cache = [None]
        masks = [None]
        a = X_t
        for l in range(L):
            z = a @ W[l] + b[l]
            Z_cache.append(z)
            if l < L - 1:
                a = relu(z)
                if dropout > 0:
                    mask = (np.random.rand(*a.shape) > dropout).astype(float)
                    a = a * mask / (1.0 - dropout)
                    masks.append(mask)
                else:
                    masks.append(None)
            else:
                a = sigmoid(z)
                masks.append(None)
            A.append(a)

        # Train loss (BCE + L2)
        m = X_t.shape[0]
        p = np.clip(A[-1], 1e-12, 1-1e-12)
        data_loss = -np.mean(y_t * np.log(p) + (1-y_t) * np.log(1-p))
        reg_loss = l2 * sum(np.sum(w**2) for w in W) if l2 > 0 else 0
        train_losses.append(data_loss + reg_loss)

        # Backward
        delta = (1.0/m) * (A[-1] - y_t)
        dW_list = [None]*L
        db_list = [None]*L
        for l in range(L-1, -1, -1):
            dW_list[l] = A[l].T @ delta + (2*l2*W[l] if l2 > 0 else 0)
            db_list[l] = np.sum(delta, axis=0, keepdims=True)
            if l > 0:
                delta = (delta @ W[l].T) * relu_deriv(Z_cache[l])
                if masks[l] is not None:
                    delta = delta * masks[l] / (1.0 - dropout)

        # Adam step
        t = epoch + 1
        for l in range(L):
            mw[l] = 0.9*mw[l] + 0.1*dW_list[l]
            mb_[l] = 0.9*mb_[l] + 0.1*db_list[l]
            vw[l] = 0.999*vw[l] + 0.001*dW_list[l]**2
            vb_[l] = 0.999*vb_[l] + 0.001*db_list[l]**2
            W[l] -= 0.005 * (mw[l]/(1-0.9**t)) / (np.sqrt(vw[l]/(1-0.999**t)) + 1e-8)
            b[l] -= 0.005 * (mb_[l]/(1-0.9**t)) / (np.sqrt(vb_[l]/(1-0.999**t)) + 1e-8)

        # Validation (no dropout)
        a_v = X_v
        for l in range(L):
            z_v = a_v @ W[l] + b[l]
            a_v = relu(z_v) if l < L-1 else sigmoid(z_v)
        pv = np.clip(a_v, 1e-12, 1-1e-12)
        vl = -np.mean(y_v * np.log(pv) + (1-y_v) * np.log(1-pv))
        va = np.mean((a_v > 0.5).astype(float) == y_v)
        val_losses.append(vl)
        val_accs.append(va)

        # Early stopping
        if early_stop_patience > 0:
            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                best_W = [w.copy() for w in W]
                best_b = [bb.copy() for bb in b]
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'best_val_acc': max(val_accs),
        'final_epoch': len(train_losses),
    }


# Experiment A: Baseline (no regularization)
print("\n  --- Experiment A: Baseline (no reg) ---")
res_a = train_with_reg(X_st, y_st, X_sv, y_sv)
overfit_epoch = np.argmin(res_a['val_loss'])
print(f"    Best val loss at epoch {overfit_epoch}: {res_a['val_loss'][overfit_epoch]:.4f}")
print(f"    Final train loss: {res_a['train_loss'][-1]:.4f}")
print(f"    Final val loss: {res_a['val_loss'][-1]:.4f}")
print(f"    Best val acc: {res_a['best_val_acc']:.2%}")
gap = res_a['train_loss'][-1] - res_a['val_loss'][-1]
print(f"    Train-Val gap: {abs(gap):.4f} <- overfitting!")

# Experiment B: L2 sweep
print("\n  --- Experiment B: L2 Regularization Sweep ---")
l2_values = [0, 0.0001, 0.001, 0.01, 0.1]
print(f"  {'L2 lambda':<12s} | {'Best Val Acc':>12s} | {'Final Train Loss':>16s} | {'Final Val Loss':>14s}")
print(f"  {'-'*12} | {'-'*12} | {'-'*16} | {'-'*14}")
for l2_val in l2_values:
    res = train_with_reg(X_st, y_st, X_sv, y_sv, l2=l2_val)
    print(f"  {l2_val:<12.4f} | {res['best_val_acc']:>11.2%} | {res['train_loss'][-1]:>16.4f} | {res['val_loss'][-1]:>14.4f}")

# Experiment C: Dropout sweep
print("\n  --- Experiment C: Dropout Sweep ---")
dr_values = [0, 0.1, 0.3, 0.5, 0.7]
print(f"  {'Dropout':>8s} | {'Best Val Acc':>12s} | {'Final Train Loss':>16s} | {'Final Val Loss':>14s}")
print(f"  {'-'*8} | {'-'*12} | {'-'*16} | {'-'*14}")
for dr in dr_values:
    res = train_with_reg(X_st, y_st, X_sv, y_sv, dropout=dr)
    print(f"  {dr:>8.1f} | {res['best_val_acc']:>11.2%} | {res['train_loss'][-1]:>16.4f} | {res['val_loss'][-1]:>14.4f}")

# Experiment D: Early stopping
print("\n  --- Experiment D: Early Stopping (patience=50) ---")
res_es = train_with_reg(X_st, y_st, X_sv, y_sv, early_stop_patience=50)
print(f"    Stopped at epoch: {res_es['final_epoch']}")
print(f"    Best val acc: {res_es['best_val_acc']:.2%}")
print(f"    Final val loss: {res_es['val_loss'][-1]:.4f}")

# Experiment E: Combined
print("\n  --- Experiment E: Combined (L2=0.001 + Dropout=0.3 + Early Stop) ---")
res_comb = train_with_reg(X_st, y_st, X_sv, y_sv, l2=0.001, dropout=0.3, early_stop_patience=50)
print(f"    Stopped at epoch: {res_comb['final_epoch']}")
print(f"    Best val acc: {res_comb['best_val_acc']:.2%}")
print(f"    Final val loss: {res_comb['val_loss'][-1]:.4f}")

print(f"""
  Summary:
  - Baseline overfits heavily: train loss → 0, val loss increases
  - L2 regularization: moderate λ (0.001) reduces overfitting without killing capacity
  - Dropout: 0.3-0.5 is optimal — too high (0.7) underfits
  - Early stopping: simple and effective — stops before overfitting gets bad
  - Combined: best generalization — each technique addresses a different aspect:
    * L2 prevents large weights (simpler function)
    * Dropout prevents co-adaptation (ensemble effect)
    * Early stopping prevents memorization (time limit)
""")


# ============================================================================
# EXERCISE 7 (HARD): Mini Deep Learning Framework (simplified demo)
# ============================================================================

print("=" * 70)
print("EXERCISE 7: Mini Deep Learning Framework (Architecture Demo)")
print("=" * 70)


class Layer:
    """Base class for all layers."""
    def forward(self, x, training=True):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def params(self):
        return []  # (param, grad) tuples


class LinearLayer(Layer):
    """Fully connected layer: Z = X @ W + b"""
    def __init__(self, n_in, n_out, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((1, n_out))
        self.dW = None
        self.db = None

    def forward(self, X, training=True):
        self.X = X                          # cache for backward
        return X @ self.W + self.b          # (m, n_in) @ (n_in, n_out) + (1, n_out)

    def backward(self, grad):
        # grad: (m, n_out) — gradient flowing from the layer above
        self.dW = self.X.T @ grad           # (n_in, m) @ (m, n_out) = (n_in, n_out)
        self.db = np.sum(grad, axis=0, keepdims=True)  # (1, n_out)
        return grad @ self.W.T              # (m, n_out) @ (n_out, n_in) = (m, n_in)

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLULayer(Layer):
    """ReLU activation: A = max(0, Z)"""
    def forward(self, Z, training=True):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, grad):
        return grad * (self.Z > 0).astype(float)


class SigmoidLayer(Layer):
    """Sigmoid activation: A = 1 / (1 + e^(-Z))"""
    def forward(self, Z, training=True):
        self.A = sigmoid(Z)
        return self.A

    def backward(self, grad):
        return grad * self.A * (1.0 - self.A)


class DropoutLayer(Layer):
    """Inverted dropout: randomly zero neurons, scale up survivors."""
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, A, training=True):
        if training and self.rate > 0:
            self.mask = (np.random.rand(*A.shape) > self.rate).astype(float)
            return A * self.mask / (1.0 - self.rate)
        self.mask = None
        return A

    def backward(self, grad):
        if self.mask is not None:
            return grad * self.mask / (1.0 - self.rate)
        return grad


class Sequential:
    """Stack layers and propagate forward/backward."""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params


class BCEWithLogitsLoss:
    """Numerically stable BCE: combines sigmoid + BCE."""
    def forward(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true
        self.m = y_true.shape[0]
        # Stable: max(0, z) - z*y + log(1 + e^(-|z|))
        stable_loss = np.maximum(0, logits) - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
        self.loss = np.mean(stable_loss)
        return self.loss

    def backward(self):
        # Gradient = sigmoid(logits) - y
        return (1.0 / self.m) * (sigmoid(self.logits) - self.y_true)


# Build and test the framework
print("\n  Building model with the mini-framework...")
model = Sequential([
    LinearLayer(2, 64, seed=42),
    ReLULayer(),
    DropoutLayer(0.3),
    LinearLayer(64, 32, seed=43),
    ReLULayer(),
    LinearLayer(32, 1, seed=44),
])
criterion = BCEWithLogitsLoss()

# Adam for the framework
adam_params = []
for p, _ in model.parameters():
    adam_params.append({
        'm': np.zeros_like(p),
        'v': np.zeros_like(p),
    })

fw_lr = 0.005

# Generate spiral dataset for testing
np.random.seed(42)
n_sp = 200
t0 = np.linspace(0, 4*np.pi, n_sp) + 0
t1 = np.linspace(0, 4*np.pi, n_sp) + np.pi
r = np.linspace(0.1, 1, n_sp)
X_sp = np.vstack([
    np.c_[r * np.cos(t0), r * np.sin(t0)] + np.random.randn(n_sp, 2)*0.015,
    np.c_[r * np.cos(t1), r * np.sin(t1)] + np.random.randn(n_sp, 2)*0.015,
])
y_sp = np.vstack([np.zeros((n_sp, 1)), np.ones((n_sp, 1))])
perm = np.random.permutation(2*n_sp)
X_sp, y_sp = X_sp[perm], y_sp[perm]

print(f"  Architecture: 2 -> 64 (ReLU, Drop=0.3) -> 32 (ReLU) -> 1")
print(f"  Dataset: spirals, {2*n_sp} points")
print(f"  Optimizer: Adam(lr={fw_lr})")

print(f"\n  {'Epoch':>6s} | {'Loss':>8s} | {'Accuracy':>10s}")
print(f"  {'-'*6} | {'-'*8} | {'-'*10}")

for epoch in range(501):
    # Forward
    logits = model.forward(X_sp, training=True)
    loss = criterion.forward(logits, y_sp)

    # Backward
    grad = criterion.backward()
    model.backward(grad)

    # Adam step
    t = epoch + 1
    for idx, (p, dp) in enumerate(model.parameters()):
        if dp is None:
            continue
        adam_params[idx]['m'] = 0.9 * adam_params[idx]['m'] + 0.1 * dp
        adam_params[idx]['v'] = 0.999 * adam_params[idx]['v'] + 0.001 * dp**2
        mh = adam_params[idx]['m'] / (1 - 0.9**t)
        vh = adam_params[idx]['v'] / (1 - 0.999**t)
        p -= fw_lr * mh / (np.sqrt(vh) + 1e-8)

    if epoch % 100 == 0:
        # Accuracy (inference mode)
        pred = sigmoid(model.forward(X_sp, training=False))
        acc = np.mean((pred > 0.5).astype(float) == y_sp)
        print(f"  {epoch:>6d} | {loss:>8.4f} | {acc:>9.2%}")

pred_final = sigmoid(model.forward(X_sp, training=False))
acc_final = np.mean((pred_final > 0.5).astype(float) == y_sp)
print(f"\n  Final accuracy: {acc_final:.2%}")
print(f"  Framework working correctly: {'YES' if acc_final > 0.85 else 'needs tuning'}")


# ============================================================================
# EXERCISE 8 (HARD): LR Range Test (simplified)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Learning Rate Range Test")
print("=" * 70)

print("\n  Running LR Range Test on MLP [2, 32, 16, 1] with spirals...")

lr_min, lr_max = 1e-7, 10.0
n_steps = 200

net_lr = SimpleMLP([2, 32, 16, 1], seed=42)
lr_log = []
loss_log = []

for step in range(n_steps):
    # Exponentially increase LR
    lr_current = lr_min * (lr_max / lr_min) ** (step / n_steps)
    lr_log.append(lr_current)

    # One training step
    net_lr.forward(X_sp)
    loss = net_lr.loss_and_backward(X_sp, y_sp)
    loss_log.append(loss)

    # SGD step with current LR
    for l in range(net_lr.L):
        net_lr.W[l] -= lr_current * net_lr.dW[l]
        net_lr.b[l] -= lr_current * net_lr.db[l]

    # Stop if loss explodes
    if np.isnan(loss) or loss > 10:
        break

# Find optimal LR: steepest negative slope
best_idx = 0
best_slope = 0
for i in range(5, len(loss_log) - 5):
    slope = (loss_log[i+5] - loss_log[i-5]) / 10
    if slope < best_slope:
        best_slope = slope
        best_idx = i

optimal_lr = lr_log[best_idx] if best_idx > 0 else 0.001

print(f"\n  LR Range: {lr_min:.0e} to {lr_max:.0e} ({n_steps} steps)")
print(f"  Loss at start: {loss_log[0]:.4f}")
print(f"  Min loss: {min(loss_log):.4f} at lr={lr_log[np.argmin(loss_log)]:.4e}")
print(f"  Optimal LR (steepest descent): {optimal_lr:.4e}")
print(f"  Recommended LR: ~{optimal_lr/3:.4e} (1/3 of optimal for safety margin)")

# Show LR vs loss in a table
print(f"\n  {'LR':>12s} | {'Loss':>10s} | {'Comment':>20s}")
print(f"  {'-'*12} | {'-'*10} | {'-'*20}")
sample_indices = np.linspace(0, min(len(lr_log)-1, n_steps-1), 10, dtype=int)
for idx in sample_indices:
    lr_val = lr_log[idx]
    l_val = loss_log[idx]
    comment = ""
    if idx == best_idx:
        comment = "<-- OPTIMAL"
    elif l_val == min(loss_log):
        comment = "<-- MIN LOSS"
    elif idx > 0 and loss_log[idx] > loss_log[idx-1] * 1.5:
        comment = "<-- DIVERGING"
    print(f"  {lr_val:>12.4e} | {l_val:>10.4f} | {comment:>20s}")

# Cosine annealing demo
print(f"\n  --- Cosine Annealing LR Schedule ---")

def cosine_annealing(t, T, lr_max, lr_min=1e-6):
    """lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T))

# Compare: Adam fixed vs Adam + cosine
net_fixed = SimpleMLP([2, 32, 16, 1], seed=42)
net_cosine = SimpleMLP([2, 32, 16, 1], seed=42)

# Adam states for both
states = {}
for name, net in [('fixed', net_fixed), ('cosine', net_cosine)]:
    states[name] = {
        'mw': [np.zeros_like(w) for w in net.W],
        'mb': [np.zeros_like(b) for b in net.b],
        'vw': [np.zeros_like(w) for w in net.W],
        'vb': [np.zeros_like(b) for b in net.b],
    }

fixed_losses = []
cosine_losses = []
base_lr = 0.005
T_max = 1000

for epoch in range(T_max):
    lr_fixed = base_lr
    lr_cos = cosine_annealing(epoch, T_max, base_lr)

    for name, net, lr_val, loss_list in [
        ('fixed', net_fixed, lr_fixed, fixed_losses),
        ('cosine', net_cosine, lr_cos, cosine_losses),
    ]:
        net.forward(X_sp)
        loss = net.loss_and_backward(X_sp, y_sp)
        loss_list.append(loss)

        t = epoch + 1
        s = states[name]
        for l in range(net.L):
            s['mw'][l] = 0.9*s['mw'][l] + 0.1*net.dW[l]
            s['mb'][l] = 0.9*s['mb'][l] + 0.1*net.db[l]
            s['vw'][l] = 0.999*s['vw'][l] + 0.001*net.dW[l]**2
            s['vb'][l] = 0.999*s['vb'][l] + 0.001*net.db[l]**2
            mh = s['mw'][l]/(1-0.9**t)
            mbh = s['mb'][l]/(1-0.9**t)
            vh = s['vw'][l]/(1-0.999**t)
            vbh = s['vb'][l]/(1-0.999**t)
            net.W[l] -= lr_val * mh / (np.sqrt(vh) + 1e-8)
            net.b[l] -= lr_val * mbh / (np.sqrt(vbh) + 1e-8)

# Results
fixed_acc = np.mean((sigmoid(net_fixed.forward(X_sp)) > 0.5).astype(float) == y_sp)
cosine_acc = np.mean((sigmoid(net_cosine.forward(X_sp)) > 0.5).astype(float) == y_sp)

print(f"  {'Schedule':<20s} | {'Final Loss':>10s} | {'Accuracy':>10s}")
print(f"  {'-'*20} | {'-'*10} | {'-'*10}")
print(f"  {'Adam fixed lr'::<20s} | {fixed_losses[-1]:>10.4f} | {fixed_acc:>9.2%}")
print(f"  {'Adam + cosine'::<20s} | {cosine_losses[-1]:>10.4f} | {cosine_acc:>9.2%}")

print("""
  Cosine annealing gradually reduces LR from lr_max to ~0, allowing:
  1. Fast exploration early (high LR)
  2. Fine convergence late (low LR)
  This often reaches slightly better minima than a fixed LR.
""")


# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("ALL EXERCISES COMPLETED")
print("=" * 70)
print("""
  Exercise 1: Forward pass with matrix dimensions annotated at every step
  Exercise 2: MSE vs BCE vs CCE — BCE penalizes confident errors exponentially
  Exercise 3: Adam internals — momentum + adaptive LR + bias correction
  Exercise 4: Multi-class MLP with softmax+CCE, gradient check validates backprop
  Exercise 5: 4 optimizers compared — Adam fastest, all reach similar final quality
  Exercise 6: Overfitting diagnosed and fixed with L2, dropout, early stopping
  Exercise 7: Mini framework with modular layers, sequential model, BCE loss
  Exercise 8: LR range test finds optimal LR, cosine annealing improves convergence
""")

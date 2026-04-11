"""
Jour 2 — Reseaux denses (MLP) FROM SCRATCH
============================================
Pure Python + NumPy. No PyTorch.
Every line is commented with the math it implements.

Covers:
  1. Configurable MLP class (variable layers/neurons)
  2. Forward pass with dimension annotations
  3. Three loss functions (MSE, Binary CE, Categorical CE + Softmax)
  4. Four optimizers (SGD, Momentum, RMSProp, Adam) side-by-side
  5. Training on a spiral dataset
  6. Overfitting demo, then fix with L2 reg + dropout

Run: python 02-code/02-reseaux-denses-mlp.py
"""

import sys
import io
import numpy as np
import time

# Force UTF-8 output to handle special characters in comments/docstrings
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# PART 0: Helper functions
# ============================================================================

def ascii_plot(values: list, width: int = 60, height: int = 12, title: str = ""):
    """Plot a list of values as an ASCII chart in the terminal."""
    if len(values) > width:
        step = len(values) // width
        sampled = [values[i] for i in range(0, len(values), step)][:width]
    else:
        sampled = values

    # Filter out inf/nan for display
    clean = [v for v in sampled if np.isfinite(v)]
    if not clean:
        print(f"  {title}: all values are inf/nan")
        return

    max_val = max(clean)
    min_val = min(clean)
    val_range = max_val - min_val if max_val != min_val else 1.0

    print(f"\n  {title}")
    print(f"  {'-' * (width + 10)}")

    for row in range(height):
        threshold = max_val - (row / (height - 1)) * val_range
        line = ""
        for val in sampled:
            if not np.isfinite(val):
                line += "!"
            elif val >= threshold:
                line += "#"
            else:
                line += " "
        if row == 0:
            label = f"{max_val:.4f}"
        elif row == height - 1:
            label = f"{min_val:.4f}"
        else:
            label = ""
        print(f"  {label:>8s} |{line}|")

    print(f"  {'':>8s} +{'-' * width}+")
    print(f"  {'':>8s}  0{' ' * (width - 8)}Epoch {len(values)}")


# ============================================================================
# PART 1: Activation Functions
# ============================================================================

def relu(Z):
    """ReLU(z) = max(0, z). Derivative = 1 if z > 0, else 0.
    Used in hidden layers — no vanishing gradient for positive inputs."""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """d/dz ReLU = 1 if z > 0, 0 otherwise.
    At z=0, we use 0 (subgradient convention)."""
    return (Z > 0).astype(float)


def sigmoid(Z):
    """σ(z) = 1 / (1 + e^(-z)). Maps to (0, 1).
    Used for binary classification output layer."""
    # Clip to prevent overflow in exp
    Z_clip = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z_clip))


def sigmoid_derivative(A):
    """σ'(z) = σ(z) * (1 - σ(z)). Takes A = σ(z) as input (already computed)."""
    return A * (1.0 - A)


def softmax(Z):
    """softmax(z_i) = e^{z_i} / Σ_j e^{z_j}.
    Numerically stable version: subtract max (log-sum-exp trick).
    Used for multi-class classification output layer."""
    # Subtract max per sample for numerical stability — prevents overflow
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)  # (m, K) - (m, 1) = (m, K)
    exp_Z = np.exp(Z_shifted)                           # safe: max exponent is 0
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True) # normalize to sum=1


# ============================================================================
# PART 2: Loss Functions
# ============================================================================

def mse_loss(y_pred, y_true):
    """MSE = (1/m) * Σ (y_pred - y_true)^2
    For regression. Gradient = (2/m) * (y_pred - y_true)."""
    m = y_true.shape[0]
    loss = np.mean((y_pred - y_true) ** 2)                          # scalar
    grad = (2.0 / m) * (y_pred - y_true)                            # (m, n_out)
    return loss, grad


def binary_cross_entropy_loss(y_pred, y_true):
    """BCE = -(1/m) Σ [y*log(p) + (1-y)*log(1-p)]
    For binary classification with sigmoid output.
    Gradient w.r.t. pre-sigmoid z simplifies to (p - y), but here we return
    gradient w.r.t. y_pred = p for generality."""
    m = y_true.shape[0]
    # Clip predictions to avoid log(0) = -inf
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    # ∂BCE/∂p = -(y/p) + (1-y)/(1-p), averaged over batch
    grad = (1.0 / m) * (-y_true / p + (1 - y_true) / (1 - p))      # (m, 1)
    return loss, grad


def categorical_cross_entropy_loss(y_pred_softmax, y_true_onehot):
    """CCE = -(1/m) Σ Σ_k y_k * log(p_k)
    For multi-class classification with softmax output.
    y_true_onehot: (m, K) one-hot encoded.
    y_pred_softmax: (m, K) softmax probabilities.
    Gradient w.r.t. pre-softmax logits z = p - y (elegant!)."""
    m = y_true_onehot.shape[0]
    eps = 1e-12
    p = np.clip(y_pred_softmax, eps, 1.0)
    # Only the probability of the true class matters (one-hot selects it)
    loss = -np.mean(np.sum(y_true_onehot * np.log(p), axis=1))      # scalar
    # Gradient w.r.t. logits z (combined softmax + CCE): p - y
    grad = (1.0 / m) * (y_pred_softmax - y_true_onehot)             # (m, K)
    return loss, grad


# ============================================================================
# PART 3: Loss Functions Demo
# ============================================================================

print("=" * 70)
print("PART 1: Loss Functions Comparison")
print("=" * 70)

# Binary classification examples
predictions = [0.9, 0.5, 0.1, 0.9, 0.01]
targets     = [1.0, 1.0, 1.0, 0.0, 0.0]
labels      = ["correct+confident", "uncertain", "very wrong",
               "very wrong (inv)", "correct+confident"]

print(f"\n  {'Case':<22s} | {'Pred':>5s} | {'True':>5s} | {'MSE':>8s} | {'BCE':>8s} | {'MSE grad':>9s} | {'BCE grad':>9s}")
print(f"  {'-'*22} | {'-'*5} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9}")

for pred, tgt, lbl in zip(predictions, targets, labels):
    p = np.array([[pred]])
    y = np.array([[tgt]])
    mse_l, mse_g = mse_loss(p, y)
    bce_l, bce_g = binary_cross_entropy_loss(p, y)
    print(f"  {lbl:<22s} | {pred:>5.2f} | {tgt:>5.1f} | {mse_l:>8.4f} | {bce_l:>8.4f} | {mse_g[0,0]:>+9.4f} | {bce_g[0,0]:>+9.4f}")

print("""
  Key insight: BCE penalizes confident wrong predictions MUCH more than MSE.
  pred=0.9, true=0.0 -> MSE=0.81, BCE=2.30. BCE gradient is also much stronger.
  This is why BCE is preferred for classification -- it learns faster from mistakes.
""")

# Multi-class example
print("  Multi-class example (3 classes):")
logits = np.array([[2.0, 1.0, 0.5]])
y_onehot = np.array([[1, 0, 0]])        # true class = 0
probs = softmax(logits)
cce_loss_val, cce_grad = categorical_cross_entropy_loss(probs, y_onehot)
print(f"  Logits:       {logits[0]}")
print(f"  Softmax:      [{', '.join(f'{p:.4f}' for p in probs[0])}] (sum={probs[0].sum():.4f})")
print(f"  True class:   0 (one-hot: {y_onehot[0]})")
print(f"  CCE loss:     {cce_loss_val:.4f}")
print(f"  Gradient:     [{', '.join(f'{g:+.4f}' for g in cce_grad[0])}]  (= softmax - onehot)")


# ============================================================================
# PART 4: Configurable MLP Class
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Configurable MLP Class from Scratch")
print("=" * 70)


class MLP:
    """Multi-Layer Perceptron with configurable architecture.

    Supports:
    - Arbitrary number of layers and neurons per layer
    - ReLU hidden activations, sigmoid or softmax output
    - MSE, BCE, or CCE loss
    - L2 regularization
    - Dropout (inverted)

    Example:
        net = MLP([2, 64, 32, 3], output='softmax', loss='cce')
        net.forward(X)                   # returns predictions
        loss = net.compute_loss(y)       # returns scalar loss
        net.backward(X, y)               # computes all gradients
    """

    def __init__(self, layer_sizes, output='sigmoid', loss='bce',
                 l2_lambda=0.0, dropout_rate=0.0, seed=42):
        """Initialize MLP with He initialization for ReLU hidden layers.

        Args:
            layer_sizes: list of ints, e.g. [2, 64, 32, 1]
            output: 'sigmoid' for binary, 'softmax' for multi-class, 'linear' for regression
            loss: 'mse', 'bce', or 'cce'
            l2_lambda: L2 regularization strength (0 = no reg)
            dropout_rate: probability of dropping a neuron (0 = no dropout)
            seed: random seed for reproducibility
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1            # number of weight layers
        self.output_type = output
        self.loss_type = loss
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.training = True                      # toggle for dropout behavior

        # Initialize weights and biases for each layer
        self.W = []   # W[l] has shape (n_{l}, n_{l+1})
        self.b = []   # b[l] has shape (1, n_{l+1})

        for l in range(self.L):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l + 1]

            # He initialization: W ~ N(0, sqrt(2/n_in))
            # Factor of 2 compensates for ReLU killing half the neurons
            W_l = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b_l = np.zeros((1, n_out))

            self.W.append(W_l)
            self.b.append(b_l)

        # Print architecture summary
        total_params = sum(W.size + b.size for W, b in zip(self.W, self.b))
        print(f"\n  MLP Architecture: {' -> '.join(str(s) for s in layer_sizes)}")
        print(f"  Output: {output} | Loss: {loss} | L2: {l2_lambda} | Dropout: {dropout_rate}")
        print(f"  Total parameters: {total_params}")
        for l in range(self.L):
            print(f"    Layer {l+1}: W{self.W[l].shape} + b{self.b[l].shape}"
                  f" = {self.W[l].size + self.b[l].size} params")

    def forward(self, X):
        """Forward pass through all layers.

        Hidden layers: Z = A_prev @ W + b, then A = ReLU(Z)
        Output layer: Z = A_prev @ W + b, then A = sigmoid/softmax/linear(Z)

        All intermediate values are cached for backprop.

        Args:
            X: input data, shape (m, n_features)
        Returns:
            predictions, shape (m, n_output)
        """
        self.A = [X]                 # A[0] = input, A[l] = activation of layer l
        self.Z = [None]              # Z[0] unused, Z[l] = pre-activation of layer l
        self.masks = [None]          # dropout masks

        A_prev = X                   # (m, n_0)

        for l in range(self.L):
            # Linear transformation: Z = A_prev @ W + b
            # Dimensions: (m, n_{l}) @ (n_{l}, n_{l+1}) + (1, n_{l+1}) = (m, n_{l+1})
            Z_l = A_prev @ self.W[l] + self.b[l]
            self.Z.append(Z_l)

            if l < self.L - 1:
                # Hidden layer: ReLU activation
                A_l = relu(Z_l)      # (m, n_{l+1}) element-wise

                # Apply dropout during training (inverted dropout)
                if self.training and self.dropout_rate > 0:
                    # Generate binary mask: each neuron survives with prob (1-p)
                    mask = (np.random.rand(*A_l.shape) > self.dropout_rate).astype(float)
                    A_l = A_l * mask                  # zero out dropped neurons
                    A_l = A_l / (1.0 - self.dropout_rate)  # scale up to maintain expected value
                    self.masks.append(mask)
                else:
                    self.masks.append(None)
            else:
                # Output layer: apply output activation
                if self.output_type == 'sigmoid':
                    A_l = sigmoid(Z_l)
                elif self.output_type == 'softmax':
                    A_l = softmax(Z_l)
                else:  # linear
                    A_l = Z_l
                self.masks.append(None)

            self.A.append(A_l)
            A_prev = A_l

        return self.A[-1]           # final output

    def compute_loss(self, y_true):
        """Compute loss + L2 penalty.

        Returns scalar loss value.
        """
        y_pred = self.A[-1]

        # Data loss
        if self.loss_type == 'mse':
            self.data_loss, self.grad_output = mse_loss(y_pred, y_true)
        elif self.loss_type == 'bce':
            self.data_loss, self.grad_output = binary_cross_entropy_loss(y_pred, y_true)
        elif self.loss_type == 'cce':
            self.data_loss, self.grad_output = categorical_cross_entropy_loss(y_pred, y_true)

        # L2 regularization penalty: λ * Σ ||W||^2
        self.reg_loss = 0.0
        if self.l2_lambda > 0:
            for l in range(self.L):
                self.reg_loss += np.sum(self.W[l] ** 2)  # Frobenius norm squared
            self.reg_loss *= self.l2_lambda

        self.total_loss = self.data_loss + self.reg_loss
        return self.total_loss

    def backward(self, X, y_true):
        """Backpropagation through all layers.

        Computes dW[l] and db[l] for each layer l.

        For output layer with sigmoid+BCE or softmax+CCE:
            delta_L = A_L - y  (elegant simplified gradient)

        For hidden layers (ReLU):
            delta_l = (delta_{l+1} @ W_{l+1}^T) * ReLU'(Z_l)

        Weight gradients:
            dW_l = A_{l-1}^T @ delta_l
            db_l = sum(delta_l, axis=0)
        """
        m = X.shape[0]

        # Initialize gradient storage
        self.dW = [None] * self.L
        self.db = [None] * self.L

        # --- Output layer gradient ---
        if (self.loss_type == 'bce' and self.output_type == 'sigmoid') or \
           (self.loss_type == 'cce' and self.output_type == 'softmax'):
            # Combined gradient: ∂L/∂Z_L = (A_L - y) / m
            # This is the elegant simplification when loss and activation are matched
            delta = (1.0 / m) * (self.A[-1] - y_true)
        elif self.loss_type == 'mse' and self.output_type == 'sigmoid':
            # ∂L/∂Z = ∂L/∂A * ∂A/∂Z = grad_output * sigmoid'(A)
            delta = self.grad_output * sigmoid_derivative(self.A[-1])
        elif self.loss_type == 'mse' and self.output_type == 'linear':
            # ∂L/∂Z = ∂L/∂A (no activation derivative for linear)
            delta = self.grad_output
        else:
            # General case: grad_output already computed in compute_loss
            delta = self.grad_output

        # --- Backpropagate through all layers ---
        for l in range(self.L - 1, -1, -1):  # from L-1 down to 0
            # dW_l = A_{l}^T @ delta        — (n_l, m) @ (m, n_{l+1}) = (n_l, n_{l+1})
            self.dW[l] = self.A[l].T @ delta

            # Add L2 regularization gradient: ∂reg/∂W = 2λW
            if self.l2_lambda > 0:
                self.dW[l] += 2 * self.l2_lambda * self.W[l]

            # db_l = sum(delta, axis=0)      — (1, n_{l+1})
            self.db[l] = np.sum(delta, axis=0, keepdims=True)

            if l > 0:
                # Propagate delta to previous layer
                # delta_prev = delta @ W_l^T * ReLU'(Z_l)
                delta = delta @ self.W[l].T              # (m, n_l)
                delta = delta * relu_derivative(self.Z[l])  # element-wise

                # Apply dropout mask (same neurons that were dropped in forward)
                if self.training and self.masks[l] is not None:
                    delta = delta * self.masks[l]
                    delta = delta / (1.0 - self.dropout_rate)


# ============================================================================
# PART 5: Optimizers
# ============================================================================

class SGDOptimizer:
    """Vanilla SGD: w = w - lr * dw
    Simple but oscillates and gets stuck at saddle points."""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.name = f"SGD(lr={lr})"

    def step(self, mlp):
        for l in range(mlp.L):
            mlp.W[l] -= self.lr * mlp.dW[l]
            mlp.b[l] -= self.lr * mlp.db[l]


class MomentumOptimizer:
    """SGD with Momentum: accumulates velocity in consistent gradient directions.
    v = β*v + dw,  w = w - lr * v
    Like a ball rolling downhill — accelerates in ravines, dampens oscillations."""

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v_W = None   # velocity for weights (initialized on first call)
        self.v_b = None   # velocity for biases
        self.name = f"Momentum(lr={lr}, beta={beta})"

    def step(self, mlp):
        # Initialize velocity arrays on first call (match shapes)
        if self.v_W is None:
            self.v_W = [np.zeros_like(mlp.W[l]) for l in range(mlp.L)]
            self.v_b = [np.zeros_like(mlp.b[l]) for l in range(mlp.L)]

        for l in range(mlp.L):
            # v = β * v_prev + gradient  (accumulate velocity)
            self.v_W[l] = self.beta * self.v_W[l] + mlp.dW[l]
            self.v_b[l] = self.beta * self.v_b[l] + mlp.db[l]
            # w = w - lr * v             (update with velocity, not raw gradient)
            mlp.W[l] -= self.lr * self.v_W[l]
            mlp.b[l] -= self.lr * self.v_b[l]


class RMSPropOptimizer:
    """RMSProp: adaptive LR per parameter using running mean of squared gradients.
    s = β*s + (1-β)*dw^2,  w = w - lr * dw / (√s + ε)
    Parameters with large gradients get smaller steps (and vice versa)."""

    def __init__(self, lr=0.001, beta=0.999, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s_W = None   # running mean of squared gradients for W
        self.s_b = None
        self.name = f"RMSProp(lr={lr})"

    def step(self, mlp):
        if self.s_W is None:
            self.s_W = [np.zeros_like(mlp.W[l]) for l in range(mlp.L)]
            self.s_b = [np.zeros_like(mlp.b[l]) for l in range(mlp.L)]

        for l in range(mlp.L):
            # s = β * s + (1-β) * grad^2   (exponential moving average of squared grad)
            self.s_W[l] = self.beta * self.s_W[l] + (1 - self.beta) * mlp.dW[l] ** 2
            self.s_b[l] = self.beta * self.s_b[l] + (1 - self.beta) * mlp.db[l] ** 2
            # w -= lr * grad / (√s + ε)    (normalize by gradient magnitude)
            mlp.W[l] -= self.lr * mlp.dW[l] / (np.sqrt(self.s_W[l]) + self.eps)
            mlp.b[l] -= self.lr * mlp.db[l] / (np.sqrt(self.s_b[l]) + self.eps)


class AdamOptimizer:
    """Adam: Momentum (1st moment) + RMSProp (2nd moment) + bias correction.
    m = β1*m + (1-β1)*g       (mean of gradients = direction)
    v = β2*v + (1-β2)*g^2     (mean of squared gradients = scale)
    m_hat = m / (1-β1^t)      (bias correction — crucial early on)
    v_hat = v / (1-β2^t)
    w -= lr * m_hat / (√v_hat + ε)

    Default hyperparameters (β1=0.9, β2=0.999, ε=1e-8) work for almost everything."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = None   # 1st moment (mean) for weights
        self.m_b = None
        self.v_W = None   # 2nd moment (variance) for weights
        self.v_b = None
        self.t = 0        # timestep (for bias correction)
        self.name = f"Adam(lr={lr})"

    def step(self, mlp):
        self.t += 1

        if self.m_W is None:
            self.m_W = [np.zeros_like(mlp.W[l]) for l in range(mlp.L)]
            self.m_b = [np.zeros_like(mlp.b[l]) for l in range(mlp.L)]
            self.v_W = [np.zeros_like(mlp.W[l]) for l in range(mlp.L)]
            self.v_b = [np.zeros_like(mlp.b[l]) for l in range(mlp.L)]

        for l in range(mlp.L):
            # 1st moment: m = β1 * m + (1-β1) * grad  (momentum / direction)
            self.m_W[l] = self.beta1 * self.m_W[l] + (1 - self.beta1) * mlp.dW[l]
            self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * mlp.db[l]

            # 2nd moment: v = β2 * v + (1-β2) * grad^2  (scale / adaptive LR)
            self.v_W[l] = self.beta2 * self.v_W[l] + (1 - self.beta2) * mlp.dW[l] ** 2
            self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * mlp.db[l] ** 2

            # Bias correction: compensate for initialization at 0
            # Without this, early estimates are biased toward 0
            m_hat_W = self.m_W[l] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[l] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[l] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[l] / (1 - self.beta2 ** self.t)

            # Update: w -= lr * m_hat / (√v_hat + ε)
            mlp.W[l] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            mlp.b[l] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# ============================================================================
# PART 6: Generate Spiral Dataset
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Generate Spiral Classification Dataset")
print("=" * 70)


def make_spirals(n_points=200, n_classes=2, noise=0.2, seed=42):
    """Generate spiral dataset — 2D points from 2 interleaved spirals.
    This is NOT linearly separable — needs a nonlinear model to classify.

    Returns X (n_points*n_classes, 2) and y (n_points*n_classes, 1)."""
    np.random.seed(seed)
    X = np.zeros((n_points * n_classes, 2))  # 2D coordinates
    y = np.zeros((n_points * n_classes, 1))  # class labels

    for cls in range(n_classes):
        start = cls * n_points
        # Angle goes from 0 to ~4π (2 full turns)
        t = np.linspace(0, 4 * np.pi, n_points) + cls * np.pi  # offset by π per class
        # Radius increases linearly with angle
        r = np.linspace(0.1, 1.0, n_points)
        X[start:start + n_points, 0] = r * np.cos(t) + np.random.randn(n_points) * noise * 0.1
        X[start:start + n_points, 1] = r * np.sin(t) + np.random.randn(n_points) * noise * 0.1
        y[start:start + n_points, 0] = cls

    # Shuffle
    perm = np.random.permutation(n_points * n_classes)
    return X[perm], y[perm]


X_all, y_all = make_spirals(n_points=200, n_classes=2, noise=0.15, seed=42)

# Split into train (70%) and validation (30%) — crucial for detecting overfitting
n_total = X_all.shape[0]
n_train = int(0.7 * n_total)
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val, y_val = X_all[n_train:], y_all[n_train:]

print(f"\n  Dataset: 2 interleaved spirals")
print(f"  Total: {n_total} points | Train: {n_train} | Validation: {n_total - n_train}")
print(f"  Features: 2 (x, y coordinates)")
print(f"  Classes: 2 (spiral 0 and spiral 1)")
print(f"  Class balance - Train: {np.mean(y_train):.2f} | Val: {np.mean(y_val):.2f}")


# ============================================================================
# PART 7: Training Loop — Compare 4 Optimizers
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Comparing 4 Optimizers on Spiral Classification")
print("=" * 70)
print("""
  Same architecture [2, 32, 16, 1], same data, different optimizers.
  Watch convergence speed and final accuracy.
""")


def train_model(layer_sizes, optimizer, X_tr, y_tr, X_v, y_v,
                epochs=500, loss_type='bce', output_type='sigmoid',
                l2_lambda=0.0, dropout_rate=0.0, seed=42, verbose=True):
    """Train an MLP and return history of train/val losses and accuracy.

    Returns dict with keys: train_loss, val_loss, train_acc, val_acc."""
    mlp = MLP(layer_sizes, output=output_type, loss=loss_type,
              l2_lambda=l2_lambda, dropout_rate=dropout_rate, seed=seed)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Training step ---
        mlp.training = True
        mlp.forward(X_tr)
        train_loss = mlp.compute_loss(y_tr)
        mlp.backward(X_tr, y_tr)
        optimizer.step(mlp)

        # --- Validation (no dropout) ---
        mlp.training = False
        val_pred = mlp.forward(X_v)
        val_loss = mlp.compute_loss(y_v)

        # --- Accuracy ---
        train_pred = mlp.forward(X_tr)
        mlp.training = False
        train_acc = np.mean((train_pred > 0.5).astype(float) == y_tr)
        val_acc = np.mean((val_pred > 0.5).astype(float) == y_v)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch:>4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                  f" | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    return mlp, history


# --- Compare 4 optimizers ---
arch = [2, 32, 16, 1]
results = {}

optimizers = [
    SGDOptimizer(lr=0.1),
    MomentumOptimizer(lr=0.05, beta=0.9),
    RMSPropOptimizer(lr=0.005),
    AdamOptimizer(lr=0.005),
]

for opt in optimizers:
    print(f"\n  --- {opt.name} ---")
    mlp, hist = train_model(
        arch, opt, X_train, y_train, X_val, y_val,
        epochs=500, seed=42
    )
    results[opt.name] = hist

# Summary table
print("\n" + "-" * 70)
print(f"  {'Optimizer':<30s} | {'Final Train Loss':>16s} | {'Final Val Acc':>13s} | {'Best Val Acc':>12s}")
print(f"  {'-'*30} | {'-'*16} | {'-'*13} | {'-'*12}")

for name, hist in results.items():
    final_tl = hist['train_loss'][-1]
    final_va = hist['val_acc'][-1]
    best_va = max(hist['val_acc'])
    print(f"  {name:<30s} | {final_tl:>16.4f} | {final_va:>12.2%} | {best_va:>11.2%}")

# ASCII plot for the best optimizer (Adam typically)
best_opt = min(results.keys(), key=lambda k: results[k]['train_loss'][-1])
ascii_plot(results[best_opt]['train_loss'], title=f"Train Loss - {best_opt}")


# ============================================================================
# PART 8: Overfitting Demo
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Overfitting Demo — Too many parameters, too little data")
print("=" * 70)
print("""
  We use a HUGE network [2, 128, 128, 64, 1] on our small dataset.
  Watch train loss go to ~0 while val loss increases — classic overfitting.
""")

# Use only 60 training points to make overfitting obvious
X_train_small = X_train[:60]
y_train_small = y_train[:60]

print(f"\n  Training set: {X_train_small.shape[0]} points (intentionally small)")
print(f"  Network: [2, 128, 128, 64, 1] = way too many params for 60 points")

overfit_opt = AdamOptimizer(lr=0.005)
print("\n  --- Training (NO regularization) ---")
_, overfit_hist = train_model(
    [2, 128, 128, 64, 1], overfit_opt,
    X_train_small, y_train_small, X_val, y_val,
    epochs=1000, seed=42, verbose=True
)

# Detect overfitting: val loss increasing while train loss decreasing
min_val_loss_epoch = np.argmin(overfit_hist['val_loss'])
print(f"\n  Overfitting detected!")
print(f"  Best val loss at epoch {min_val_loss_epoch}: {overfit_hist['val_loss'][min_val_loss_epoch]:.4f}")
print(f"  Final val loss at epoch 999:  {overfit_hist['val_loss'][-1]:.4f} (increased!)")
print(f"  Final train loss:             {overfit_hist['train_loss'][-1]:.4f} (near zero)")
print(f"  Train Acc: {overfit_hist['train_acc'][-1]:.2%} vs Val Acc: {overfit_hist['val_acc'][-1]:.2%}")
print(f"  Gap = {overfit_hist['train_acc'][-1] - overfit_hist['val_acc'][-1]:.2%} <- overfitting signal")


# ============================================================================
# PART 9: Fix with L2 Regularization
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Fix Overfitting with L2 Regularization")
print("=" * 70)
print("""
  Same huge network, same small dataset, but now with L2 penalty (weight decay).
  L2 adds λ * Σw² to the loss — penalizes large weights, forces simpler solutions.
""")

l2_values = [0.0, 0.0001, 0.001, 0.01]
l2_results = {}

for l2 in l2_values:
    label = f"L2={l2}"
    print(f"\n  --- {label} ---")
    opt = AdamOptimizer(lr=0.005)
    _, hist = train_model(
        [2, 128, 128, 64, 1], opt,
        X_train_small, y_train_small, X_val, y_val,
        epochs=800, l2_lambda=l2, seed=42, verbose=False
    )
    l2_results[label] = hist
    gap = hist['train_acc'][-1] - hist['val_acc'][-1]
    print(f"    Final: Train Acc={hist['train_acc'][-1]:.2%} | Val Acc={hist['val_acc'][-1]:.2%} | Gap={gap:.2%}")

print("\n  Summary — L2 Regularization Effect:")
print(f"  {'L2 Lambda':<12s} | {'Train Acc':>10s} | {'Val Acc':>10s} | {'Gap':>8s} | {'Overfitting?':>12s}")
print(f"  {'-'*12} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*12}")
for label, hist in l2_results.items():
    ta = hist['train_acc'][-1]
    va = hist['val_acc'][-1]
    gap = ta - va
    status = "YES" if gap > 0.10 else "mild" if gap > 0.05 else "NO"
    print(f"  {label:<12s} | {ta:>9.2%} | {va:>9.2%} | {gap:>7.2%} | {status:>12s}")


# ============================================================================
# PART 10: Fix with Dropout
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Fix Overfitting with Dropout")
print("=" * 70)
print("""
  Same network, same data, but now randomly dropping neurons during training.
  Each batch trains a different "sub-network" → ensemble effect → better generalization.
""")

dropout_rates = [0.0, 0.2, 0.4, 0.6]
dropout_results = {}

for dr in dropout_rates:
    label = f"Dropout={dr}"
    print(f"\n  --- {label} ---")
    opt = AdamOptimizer(lr=0.005)
    _, hist = train_model(
        [2, 128, 128, 64, 1], opt,
        X_train_small, y_train_small, X_val, y_val,
        epochs=800, dropout_rate=dr, seed=42, verbose=False
    )
    dropout_results[label] = hist
    gap = hist['train_acc'][-1] - hist['val_acc'][-1]
    print(f"    Final: Train Acc={hist['train_acc'][-1]:.2%} | Val Acc={hist['val_acc'][-1]:.2%} | Gap={gap:.2%}")

print("\n  Summary — Dropout Effect:")
print(f"  {'Dropout Rate':<14s} | {'Train Acc':>10s} | {'Val Acc':>10s} | {'Gap':>8s} | {'Overfitting?':>12s}")
print(f"  {'-'*14} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*12}")
for label, hist in dropout_results.items():
    ta = hist['train_acc'][-1]
    va = hist['val_acc'][-1]
    gap = ta - va
    status = "YES" if gap > 0.10 else "mild" if gap > 0.05 else "NO"
    print(f"  {label:<14s} | {ta:>9.2%} | {va:>9.2%} | {gap:>7.2%} | {status:>12s}")


# ============================================================================
# PART 11: Combined — L2 + Dropout (the real-world approach)
# ============================================================================

print("\n" + "=" * 70)
print("PART 8: Combined Regularization — L2 + Dropout")
print("=" * 70)
print("""
  In practice, you combine multiple regularization techniques.
  Best combo: moderate L2 (0.001) + moderate dropout (0.3).
""")

print("\n  --- No Regularization ---")
opt_noreg = AdamOptimizer(lr=0.005)
_, hist_noreg = train_model(
    [2, 128, 128, 64, 1], opt_noreg,
    X_train_small, y_train_small, X_val, y_val,
    epochs=800, l2_lambda=0.0, dropout_rate=0.0, seed=42, verbose=False
)

print("\n  --- L2=0.001 + Dropout=0.3 ---")
opt_reg = AdamOptimizer(lr=0.005)
_, hist_reg = train_model(
    [2, 128, 128, 64, 1], opt_reg,
    X_train_small, y_train_small, X_val, y_val,
    epochs=800, l2_lambda=0.001, dropout_rate=0.3, seed=42, verbose=False
)

print(f"\n  Comparison:")
print(f"  {'Config':<25s} | {'Train Acc':>10s} | {'Val Acc':>10s} | {'Gap':>8s}")
print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*8}")
for label, h in [("No regularization", hist_noreg), ("L2=0.001 + Drop=0.3", hist_reg)]:
    ta, va = h['train_acc'][-1], h['val_acc'][-1]
    print(f"  {label:<25s} | {ta:>9.2%} | {va:>9.2%} | {ta-va:>7.2%}")

ascii_plot(hist_noreg['val_loss'], title="Val Loss — NO regularization")
ascii_plot(hist_reg['val_loss'], title="Val Loss — L2 + Dropout")


# ============================================================================
# PART 12: Matplotlib Plots (if available)
# ============================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Optimizer comparison (train loss)
    ax = axes[0, 0]
    for name, hist in results.items():
        ax.plot(hist['train_loss'], label=name, linewidth=1.2)
    ax.set_title('Optimizer Comparison — Train Loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Optimizer comparison (val accuracy)
    ax = axes[0, 1]
    for name, hist in results.items():
        ax.plot(hist['val_acc'], label=name, linewidth=1.2)
    ax.set_title('Optimizer Comparison — Val Accuracy', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Overfitting (train vs val loss)
    ax = axes[1, 0]
    ax.plot(overfit_hist['train_loss'], label='Train Loss', color='#2563eb')
    ax.plot(overfit_hist['val_loss'], label='Val Loss', color='#dc2626')
    ax.axvline(x=min_val_loss_epoch, color='gray', linestyle='--', alpha=0.5,
               label=f'Best val epoch ({min_val_loss_epoch})')
    ax.set_title('Overfitting — Train vs Val Loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Regularized vs not
    ax = axes[1, 1]
    ax.plot(hist_noreg['val_loss'], label='No Reg (val)', color='#dc2626', linewidth=1.2)
    ax.plot(hist_reg['val_loss'], label='L2+Dropout (val)', color='#16a34a', linewidth=1.2)
    ax.plot(hist_noreg['train_loss'], label='No Reg (train)', color='#dc2626', linestyle='--', alpha=0.5)
    ax.plot(hist_reg['train_loss'], label='L2+Dropout (train)', color='#16a34a', linestyle='--', alpha=0.5)
    ax.set_title('Effect of Regularization', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Day 2 — MLP: Optimizers, Overfitting & Regularization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('day2_mlp_results.png', dpi=150)
    print("\n  [Matplotlib] All plots saved to day2_mlp_results.png")
    plt.close()
except ImportError:
    print("\n  [Matplotlib not installed — ASCII plots above are the fallback]")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY — What We Built Today")
print("=" * 70)
print("""
  1. Loss functions: MSE (regression), BCE (binary), CCE+Softmax (multi-class)
     - BCE penalizes confident wrong predictions exponentially
     - Softmax+CCE gradient = p - y (elegant, no saturation)

  2. Configurable MLP from scratch: arbitrary layers, He init, dropout, L2

  3. Four optimizers:
     - SGD: simple but oscillates
     - Momentum: accumulates velocity, dampens oscillations
     - RMSProp: adaptive LR per parameter
     - Adam: Momentum + RMSProp + bias correction = best default

  4. Overfitting: train loss ↓ while val loss ↑ = memorization
     - Fix with L2 regularization (penalize large weights)
     - Fix with Dropout (random neuron deactivation = ensemble)
     - Best: combine both

  Next: Day 3 — CNNs, convolutions, pooling, batch normalization.
""")

"""
Jour 1 — Le neurone & backpropagation FROM SCRATCH
===================================================
Pure Python + NumPy. No PyTorch.
Every line is commented with the math it implements.

Run: python 02-code/01-neurone-et-backpropagation.py
"""

import numpy as np

# ============================================================================
# PART 1: A SINGLE NEURON — Forward + Backward
# ============================================================================

print("=" * 70)
print("PART 1: Single Neuron -- Forward & Backward Pass")
print("=" * 70)


def sigmoid(z: float) -> float:
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))
    Maps any real number to (0, 1). Used for binary classification output.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a: float) -> float:
    """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))
    Note: we pass 'a' which is already sigmoid(z), so we don't recompute.
    Maximum value is 0.25 (at z=0). This causes vanishing gradient in deep nets.
    """
    return a * (1.0 - a)


# --- Single neuron parameters ---
x = np.array([0.5, 0.8])       # 2 inputs
w = np.array([0.4, -0.3])      # 2 weights (what the neuron learns)
b = 0.1                         # bias (shifts the decision boundary)
y_true = 1.0                    # target value

# --- Forward pass ---
# Step 1: Weighted sum — z = w^T * x + b = Σ(w_i * x_i) + b
z = np.dot(w, x) + b           # dot product: 0.4*0.5 + (-0.3)*0.8 + 0.1 = 0.06
print(f"\nWeighted sum: z = {z:.4f}")

# Step 2: Activation — a = σ(z)
a = sigmoid(z)                  # sigmoid(0.06) ≈ 0.5150
print(f"Activation:   a = sigmoid({z:.4f}) = {a:.4f}")

# Step 3: Loss (MSE for one sample) — L = (a - y)^2
loss = (a - y_true) ** 2       # (0.5150 - 1.0)^2 = 0.2352
print(f"Loss (MSE):   L = ({a:.4f} - {y_true})^2 = {loss:.4f}")

# --- Backward pass (computing gradients using chain rule) ---
# Chain rule: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w
#
# ∂L/∂a = 2*(a - y)           — derivative of MSE
# ∂a/∂z = a*(1-a)             — derivative of sigmoid
# ∂z/∂w = x                   — derivative of linear combination w.r.t weights
# ∂z/∂b = 1                   — derivative of linear combination w.r.t bias

dL_da = 2.0 * (a - y_true)     # gradient of loss w.r.t. activation
print(f"\n--- Backward Pass ---")
print(f"dL/da = 2*({a:.4f} - {y_true}) = {dL_da:.4f}")

da_dz = sigmoid_derivative(a)  # gradient of activation w.r.t. pre-activation
print(f"da/dz = {a:.4f} * (1 - {a:.4f}) = {da_dz:.4f}")

# Delta = combined gradient flowing back through this neuron
delta = dL_da * da_dz           # this is dL/dz -- the error signal
print(f"delta = dL/dz = {dL_da:.4f} * {da_dz:.4f} = {delta:.4f}")

# Gradient of loss w.r.t. each weight: dL/dw_i = delta * x_i
dL_dw = delta * x               # element-wise: delta * [x1, x2]
dL_db = delta * 1.0             # bias gradient is just delta

print(f"dL/dw1 = {delta:.4f} * {x[0]} = {dL_dw[0]:.4f}")
print(f"dL/dw2 = {delta:.4f} * {x[1]} = {dL_dw[1]:.4f}")
print(f"dL/db  = {delta:.4f}")

# --- Weight update (gradient descent) ---
# Rule: w_new = w_old - learning_rate * ∂L/∂w
# The minus sign means we go OPPOSITE to the gradient (downhill on the loss surface)
lr = 0.5  # learning rate
w_new = w - lr * dL_dw
b_new = b - lr * dL_db

print(f"\n--- Weight Update (lr={lr}) ---")
print(f"w1: {w[0]:.4f} -> {w_new[0]:.4f}")
print(f"w2: {w[1]:.4f} -> {w_new[1]:.4f}")
print(f"b:  {b:.4f} -> {b_new:.4f}")

# Verify: new forward pass should give a prediction closer to 1.0
z_new = np.dot(w_new, x) + b_new
a_new = sigmoid(z_new)
loss_new = (a_new - y_true) ** 2
print(f"\nAfter update: prediction = {a_new:.4f} (was {a:.4f}), loss = {loss_new:.4f} (was {loss:.4f})")


# ============================================================================
# PART 2: 2-Layer Neural Network from Scratch
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: 2-Layer Neural Network -- XOR Problem")
print("=" * 70)
print("""
Architecture: 2 inputs -> 4 hidden neurons -> 1 output
Activation: sigmoid everywhere
Loss: MSE
XOR is the classic problem that a single neuron CANNOT solve (not linearly separable).
A 2-layer network CAN solve it -- this is why hidden layers matter.
""")


class NeuralNetwork:
    """A simple 2-layer neural network built from scratch.

    Architecture:
        Input (2) → Hidden (n_hidden) → Output (1)

    Math:
        Hidden layer: z_h = X @ W_h + b_h,  a_h = sigmoid(z_h)
        Output layer: z_o = a_h @ W_o + b_o, a_o = sigmoid(z_o)
        Loss (MSE):   L = mean((a_o - y)^2)
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int, seed: int = 42):
        """Initialize weights randomly (small values centered on 0).
        Random init breaks symmetry — if all weights are equal, all neurons
        compute the same thing and the network can never learn different features.
        """
        np.random.seed(seed)

        # Xavier initialization: scale by 1/sqrt(n_in) to keep variance stable
        # across layers. Without this, activations either explode or vanish.
        self.W_h = np.random.randn(n_input, n_hidden) * np.sqrt(1.0 / n_input)
        self.b_h = np.zeros((1, n_hidden))  # biases init to 0 is fine

        self.W_o = np.random.randn(n_hidden, n_output) * np.sqrt(1.0 / n_hidden)
        self.b_o = np.zeros((1, n_output))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: compute prediction for input X.

        X shape: (batch_size, n_input)

        Step 1: z_h = X @ W_h + b_h       — linear combination for hidden layer
        Step 2: a_h = sigmoid(z_h)          — non-linear activation
        Step 3: z_o = a_h @ W_o + b_o      — linear combination for output
        Step 4: a_o = sigmoid(z_o)          — final prediction in (0, 1)
        """
        # Hidden layer
        self.z_h = X @ self.W_h + self.b_h  # (batch, n_hidden) — matrix multiply
        self.a_h = sigmoid(self.z_h)          # element-wise sigmoid

        # Output layer
        self.z_o = self.a_h @ self.W_o + self.b_o  # (batch, n_output)
        self.a_o = sigmoid(self.z_o)                 # final prediction

        return self.a_o

    def compute_loss(self, y_true: np.ndarray) -> float:
        """MSE loss: L = (1/n) * Σ(a_o - y)^2
        We average over all samples in the batch.
        """
        self.loss = np.mean((self.a_o - y_true) ** 2)
        return self.loss

    def backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """Backpropagation: compute gradients of loss w.r.t. all parameters.

        This is the CHAIN RULE applied layer by layer, from output back to input.

        For output layer:
            ∂L/∂a_o = 2/n * (a_o - y)          — MSE derivative
            ∂a_o/∂z_o = a_o * (1 - a_o)        — sigmoid derivative
            δ_o = ∂L/∂z_o = ∂L/∂a_o * ∂a_o/∂z_o

            ∂L/∂W_o = a_h^T @ δ_o              — gradient for output weights
            ∂L/∂b_o = sum(δ_o)                  — gradient for output bias

        For hidden layer (chain rule continues):
            ∂L/∂a_h = δ_o @ W_o^T              — error propagated backward
            ∂a_h/∂z_h = a_h * (1 - a_h)        — sigmoid derivative
            δ_h = ∂L/∂a_h * ∂a_h/∂z_h

            ∂L/∂W_h = X^T @ δ_h                — gradient for hidden weights
            ∂L/∂b_h = sum(δ_h)                  — gradient for hidden bias
        """
        n = X.shape[0]  # batch size, used to average gradients

        # --- Output layer gradients ---
        # ∂L/∂z_o = ∂L/∂a_o * ∂a_o/∂z_o
        dL_da_o = (2.0 / n) * (self.a_o - y_true)       # (batch, 1)
        da_o_dz_o = sigmoid_derivative(self.a_o)          # (batch, 1)
        delta_o = dL_da_o * da_o_dz_o                     # (batch, 1) element-wise

        # ∂L/∂W_o = a_h^T @ δ_o — each hidden neuron's activation × error signal
        self.dW_o = self.a_h.T @ delta_o                  # (n_hidden, 1)
        # ∂L/∂b_o = sum of δ_o over batch — bias affects all samples equally
        self.db_o = np.sum(delta_o, axis=0, keepdims=True)  # (1, 1)

        # --- Hidden layer gradients ---
        # Propagate error backward: ∂L/∂a_h = δ_o @ W_o^T
        dL_da_h = delta_o @ self.W_o.T                    # (batch, n_hidden)
        da_h_dz_h = sigmoid_derivative(self.a_h)          # (batch, n_hidden)
        delta_h = dL_da_h * da_h_dz_h                     # (batch, n_hidden)

        # ∂L/∂W_h = X^T @ δ_h — each input × error signal for hidden layer
        self.dW_h = X.T @ delta_h                         # (n_input, n_hidden)
        self.db_h = np.sum(delta_h, axis=0, keepdims=True)  # (1, n_hidden)

    def update_weights(self, lr: float) -> None:
        """Gradient descent: w = w - lr * ∂L/∂w
        The minus sign moves weights in the OPPOSITE direction of the gradient,
        which is downhill on the loss surface — toward lower error.
        """
        self.W_h -= lr * self.dW_h
        self.b_h -= lr * self.db_h
        self.W_o -= lr * self.dW_o
        self.b_o -= lr * self.db_o

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One complete training iteration: forward -> loss -> backward -> update."""
        self.forward(X)
        loss = self.compute_loss(y)
        self.backward(X, y)
        self.update_weights(lr)
        return loss


# --- XOR Dataset ---
# XOR truth table: the output is 1 when inputs differ, 0 when they're the same.
# This is NOT linearly separable — no single line can separate the 1s from the 0s.
# That's why you need at least one hidden layer.
X_xor = np.array([
    [0, 0],  # → 0
    [0, 1],  # → 1
    [1, 0],  # → 1
    [1, 1],  # → 0
])
y_xor = np.array([
    [0],
    [1],
    [1],
    [0],
])

# --- Train the network ---
nn = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, seed=42)
epochs = 10000
lr = 2.0  # higher LR works well for this tiny problem

print("Training on XOR...")
losses = []
for epoch in range(epochs):
    loss = nn.train_step(X_xor, y_xor, lr)
    losses.append(loss)

    # Print progress at key milestones
    if epoch < 10 or epoch % 1000 == 0 or epoch == epochs - 1:
        preds = nn.forward(X_xor)
        pred_str = ", ".join([f"{p[0]:.4f}" for p in preds])
        print(f"  Epoch {epoch:>5d} | Loss: {loss:.6f} | Predictions: [{pred_str}]")

# --- Final results ---
print(f"\n--- Final Results after {epochs} epochs ---")
predictions = nn.forward(X_xor)
for i in range(len(X_xor)):
    pred = predictions[i][0]
    target = y_xor[i][0]
    rounded = 1 if pred > 0.5 else 0
    status = "OK" if rounded == target else "WRONG"
    print(f"  Input: {X_xor[i]} -> Predicted: {pred:.4f} (rounded: {rounded}) | Target: {target} | {status}")


# ============================================================================
# PART 3: Visualize Training Loss
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Loss Curve Visualization")
print("=" * 70)

# --- ASCII loss curve (works everywhere, no dependencies) ---
def ascii_plot(values: list, width: int = 60, height: int = 15, title: str = "Loss"):
    """Plot a list of values as an ASCII chart in the terminal."""
    # Sample values to fit the width
    if len(values) > width:
        step = len(values) // width
        sampled = [values[i] for i in range(0, len(values), step)][:width]
    else:
        sampled = values

    max_val = max(sampled)
    min_val = min(sampled)
    val_range = max_val - min_val if max_val != min_val else 1.0

    print(f"\n  {title}")
    print(f"  {'-' * (width + 6)}")

    for row in range(height):
        # Map row to value (top = max, bottom = min)
        threshold = max_val - (row / (height - 1)) * val_range
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "#"
            else:
                line += " "
        # Y-axis label
        if row == 0:
            label = f"{max_val:.4f}"
        elif row == height - 1:
            label = f"{min_val:.4f}"
        else:
            label = "      "
        print(f"  {label:>7s} |{line}|")

    print(f"  {'':>7s} +{'-' * width}+")
    print(f"  {'':>7s}  Epoch 0{' ' * (width - 14)}Epoch {len(values)}")


ascii_plot(losses, title="MSE Loss over Training (XOR)")

# --- Matplotlib plot (if available) ---
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(losses, linewidth=1.5, color='#2563eb')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Loss -- 2-Layer NN on XOR', fontsize=14)
    ax.set_yscale('log')  # log scale shows convergence better
    ax.zone(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve_xor.png', dpi=150)
    print("\n  [Matplotlib] Loss curve saved to loss_curve_xor.png")
    plt.close()
except ImportError:
    print("\n  [Matplotlib not installed — ASCII plot above is the fallback]")


# ============================================================================
# PART 4: Effect of Different Learning Rates
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Effect of Different Learning Rates")
print("=" * 70)
print("""
Same network, same XOR problem, different learning rates.
Watch how lr affects convergence:
  - Too small -> loss barely moves
  - Good range -> smooth convergence
  - Too large -> loss oscillates or diverges
""")

learning_rates = [0.01, 0.1, 0.5, 2.0, 5.0, 20.0]
lr_results = {}

for test_lr in learning_rates:
    net = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, seed=42)
    lr_losses = []

    for epoch in range(5000):
        loss = net.train_step(X_xor, y_xor, test_lr)
        lr_losses.append(loss)

        # Detect divergence early (loss going to infinity)
        if np.isnan(loss) or loss > 1e6:
            lr_losses.append(float('inf'))
            break

    lr_results[test_lr] = lr_losses

    # Report final state
    final_loss = lr_losses[-1]
    if np.isinf(final_loss) or np.isnan(final_loss):
        status = "DIVERGED"
    elif final_loss < 0.01:
        status = "CONVERGED"
    elif final_loss < 0.1:
        status = "SLOW CONVERGENCE"
    else:
        status = "NOT CONVERGED"

    final_preds = net.forward(X_xor)
    pred_str = ", ".join([f"{p[0]:.2f}" for p in final_preds])
    print(f"  lr={test_lr:<6} | Final loss: {final_loss:>10.6f} | Status: {status:<18} | Preds: [{pred_str}]")


# --- Matplotlib comparison (if available) ---
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, test_lr in enumerate(learning_rates):
        ax = axes[idx]
        lr_losses = lr_results[test_lr]

        # Clip for plotting (don't plot infinity)
        plot_losses = [min(l, 1.0) for l in lr_losses if not np.isnan(l) and not np.isinf(l)]

        ax.plot(plot_losses, linewidth=1.2, color='#2563eb')
        ax.set_title(f'lr = {test_lr}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(0, 0.35)
        ax.zone(True, alpha=0.3)

    plt.suptitle('Effect of Learning Rate on XOR Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=150)
    print("\n  [Matplotlib] Comparison saved to learning_rate_comparison.png")
    plt.close()
except ImportError:
    pass


# ============================================================================
# PART 5: Activation Functions Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Activation Functions -- Values & Derivatives")
print("=" * 70)
print("""
Comparing sigmoid, tanh, and ReLU at different input values.
Notice how sigmoid/tanh derivatives vanish for large |z|, but ReLU stays at 1.
""")


def tanh(z):
    """tanh(z) = (e^z - e^-z) / (e^z + e^-z). Range: (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(z):
    """d/dz tanh(z) = 1 - tanh(z)^2. Max = 1 at z=0."""
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    """ReLU(z) = max(0, z). Range: [0, +inf)."""
    return np.maximum(0, z)


def relu_derivative(z):
    """d/dz ReLU(z) = 0 if z<0, 1 if z>0 (undefined at z=0, we use 0)."""
    return np.where(np.asarray(z) > 0, 1.0, 0.0)


# Show values at key points
test_points = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]

print(f"\n  {'z':>6s} | {'sigmoid':>8s} {'s_deriv':>8s} | {'tanh':>8s} {'t_deriv':>8s} | {'ReLU':>8s} {'r_deriv':>8s}")
print(f"  {'-'*6:s} | {'-'*8:s} {'-'*8:s} | {'-'*8:s} {'-'*8:s} | {'-'*8:s} {'-'*8:s}")

for z_val in test_points:
    sig_val = sigmoid(z_val)
    sig_der = sigmoid_derivative(sig_val)
    tan_val = tanh(z_val)
    tan_der = tanh_derivative(z_val)
    rel_val = relu(z_val)
    rel_der = relu_derivative(z_val)

    print(f"  {z_val:>6.1f} | {sig_val:>8.4f} {sig_der:>8.4f} | {tan_val:>8.4f} {tan_der:>8.4f} | {rel_val:>8.4f} {rel_der:>8.4f}")

print("""
Key observations:
  - Sigmoid derivative maxes at 0.25 -> after 10 layers: 0.25^10 ~ 0.000001 (vanishing!)
  - Tanh derivative maxes at 1.0 but drops quickly -> still vanishes in deep nets
  - ReLU derivative is exactly 1.0 for all z > 0 -> gradient passes through unchanged
  - ReLU derivative is 0 for z < 0 -> "dead neurons" (dying ReLU problem)
""")

# ============================================================================
# PART 6: SGD vs Mini-batch vs Batch — Demo
# ============================================================================

print("=" * 70)
print("PART 6: SGD vs Mini-batch vs Batch Gradient Descent")
print("=" * 70)
print("""
We create a slightly larger dataset (16 samples of XOR with noise)
and compare the three gradient descent variants.
""")

# Create noisy XOR dataset (16 samples)
np.random.seed(123)
n_samples = 16
X_noisy = np.random.rand(n_samples, 2)                        # random inputs in [0, 1]
y_noisy = ((X_noisy[:, 0] > 0.5) ^ (X_noisy[:, 1] > 0.5))   # XOR logic
y_noisy = y_noisy.astype(float).reshape(-1, 1)                # shape (16, 1)
# Add noise to make it more realistic
y_noisy = np.clip(y_noisy + np.random.randn(n_samples, 1) * 0.05, 0, 1)


def train_variant(X, y, variant, epochs=2000, lr=1.0, batch_size=4):
    """Train with different GD variants and return loss history.

    variant: 'batch' (all data), 'sgd' (1 sample), 'minibatch' (batch_size samples)
    """
    net = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, seed=42)
    loss_history = []

    for epoch in range(epochs):
        if variant == 'batch':
            # Batch GD: use ALL samples for each update
            loss = net.train_step(X, y, lr)
            loss_history.append(loss)

        elif variant == 'sgd':
            # SGD: use ONE random sample per update
            idx = np.random.randint(0, len(X))
            X_sample = X[idx:idx+1]  # shape (1, 2) — keep dimensions
            y_sample = y[idx:idx+1]
            net.train_step(X_sample, y_sample, lr)
            # Record loss on full dataset for fair comparison
            net.forward(X)
            loss = net.compute_loss(y)
            loss_history.append(loss)

        elif variant == 'minibatch':
            # Mini-batch GD: use batch_size random samples per update
            indices = np.random.choice(len(X), size=batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            net.train_step(X_batch, y_batch, lr)
            # Record loss on full dataset for fair comparison
            net.forward(X)
            loss = net.compute_loss(y)
            loss_history.append(loss)

    return loss_history


# Run all three variants
batch_losses = train_variant(X_noisy, y_noisy, 'batch', epochs=2000, lr=1.0)
sgd_losses = train_variant(X_noisy, y_noisy, 'sgd', epochs=2000, lr=1.0)
mini_losses = train_variant(X_noisy, y_noisy, 'minibatch', epochs=2000, lr=1.0, batch_size=4)

print(f"\n  After 2000 epochs:")
print(f"  {'Variant':<12} | {'Final Loss':>10} | {'Min Loss':>10} | {'Stability':>10}")
print(f"  {'-'*12:s} | {'-'*10:s} | {'-'*10:s} | {'-'*10:s}")

for name, hist in [('Batch', batch_losses), ('SGD', sgd_losses), ('Mini-batch', mini_losses)]:
    final = hist[-1]
    minimum = min(hist)
    # Stability = std dev of last 200 losses (lower = smoother)
    stability = np.std(hist[-200:])
    print(f"  {name:<12} | {final:>10.6f} | {minimum:>10.6f} | {stability:>10.6f}")

print("""
Observations:
  - Batch: smoothest convergence, but slowest per-epoch (uses all data)
  - SGD: fastest updates but very noisy -- loss oscillates heavily
  - Mini-batch: best compromise -- reasonably smooth AND fast
""")

# --- Matplotlib comparison (if available) ---
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Batch GD', 'Stochastic GD (1 sample)', 'Mini-batch GD (batch=4)']
    all_losses = [batch_losses, sgd_losses, mini_losses]
    colors = ['#2563eb', '#dc2626', '#16a34a']

    for ax, title, losses_list, color in zip(axes, titles, all_losses, colors):
        ax.plot(losses_list, linewidth=0.8, color=color, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_ylim(0, 0.35)
        ax.zone(True, alpha=0.3)

    plt.suptitle('Gradient Descent Variants Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gd_variants_comparison.png', dpi=150)
    print("  [Matplotlib] Comparison saved to gd_variants_comparison.png")
    plt.close()
except ImportError:
    pass


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY -- What We Built Today")
print("=" * 70)
print("""
1. Single neuron: forward pass (z = w.x + b, a = sig(z)) and backward pass (chain rule)
2. 2-layer neural network: solves XOR (non-linearly separable problem)
3. Backpropagation: computed gradients layer by layer using the chain rule
4. Gradient descent: updated weights to minimize loss
5. Compared learning rates: too small -> stuck, too large -> diverge
6. Compared GD variants: batch (stable), SGD (noisy), mini-batch (best compromise)
7. Compared activations: ReLU > sigmoid/tanh for hidden layers (no vanishing gradient)

Next: Day 2 -- Dense networks (MLP), more optimizers (Adam), regularization.
""")

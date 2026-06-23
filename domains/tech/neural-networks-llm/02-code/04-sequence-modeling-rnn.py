"""
Jour 4 — Sequence Modeling: RNN from scratch in NumPy
======================================================
Pure Python + NumPy. No torch. No framework.
Every line commented with WHY.

Covers:
  1. Vanilla RNN cell from scratch (forward + backward)
  2. BPTT (Backpropagation Through Time)
  3. Tiny character-level RNN training loop on a small corpus
  4. Gradient norm monitoring — observe vanishing / exploding

Run: python 02-code/04-sequence-modeling-rnn.py
"""

import sys
import io
import numpy as np

# Force UTF-8 output so special characters in strings never break the console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# PART 1: A Vanilla RNN Cell — Forward Pass
# ============================================================================

def rnn_cell_forward(x_t, h_prev, W_xh, W_hh, W_hy, b_h, b_y):
    """
    One step of a vanilla RNN.

    h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    y_t = W_hy @ h_t + b_y

    Shapes:
      x_t     : (input_dim,)
      h_prev  : (hidden_dim,)
      W_xh    : (hidden_dim, input_dim)
      W_hh    : (hidden_dim, hidden_dim)
      W_hy    : (output_dim, hidden_dim)
      b_h     : (hidden_dim,)
      b_y     : (output_dim,)

    Returns:
      h_t     : new hidden state
      y_t     : output logits
      cache   : values needed for backward pass
    """
    # Pre-activation: linear combination of input and previous hidden state
    # WHY the same W_hh at every time step? Parameter sharing — like a conv
    # filter applied at every spatial location, we apply the same RNN "filter"
    # at every time step. Fewer parameters, generalizes to any sequence length.
    z_h = W_xh @ x_t + W_hh @ h_prev + b_h

    # tanh activation — bounded in (-1, 1) so the hidden state cannot blow up
    # across time steps. ReLU would be disastrous here (no upper bound).
    h_t = np.tanh(z_h)

    # Output projection (linear, no activation — softmax applied later at loss)
    y_t = W_hy @ h_t + b_y

    # Cache for backward pass
    cache = (x_t, h_prev, h_t, z_h)
    return h_t, y_t, cache


# ============================================================================
# PART 2: A Vanilla RNN Cell — Backward Pass
# ============================================================================

def rnn_cell_backward(dh_t, dy_t, cache, W_hh, W_hy):
    """
    Backward through one RNN step.

    Given:
      dh_t : gradient of loss w.r.t. h_t FROM THE FUTURE (coming from h_{t+1})
      dy_t : gradient of loss w.r.t. y_t (from the output layer at time t)

    We combine them: the total gradient on h_t is dh_t (from future) plus
    the contribution routed through y_t.
    """
    x_t, h_prev, h_t, z_h = cache

    # Gradient on h_t from the output at time t
    # dL/dh_t (from y_t) = W_hy.T @ dy_t
    dh_from_y = W_hy.T @ dy_t

    # Total gradient on h_t: from future h_{t+1} + from current y_t
    dh_total = dh_t + dh_from_y

    # Backprop through tanh: d/dz tanh(z) = 1 - tanh^2(z) = 1 - h^2
    # WHY: we need gradient on the pre-activation z_h to propagate further
    dz_h = dh_total * (1.0 - h_t ** 2)

    # Gradients on parameters
    dW_xh = np.outer(dz_h, x_t)       # d/dW_xh of (W_xh @ x_t) = outer(dz, x)
    dW_hh = np.outer(dz_h, h_prev)    # d/dW_hh of (W_hh @ h_prev) = outer(dz, h_prev)
    db_h = dz_h                       # bias gradient is just dz_h

    dW_hy = np.outer(dy_t, h_t)       # d/dW_hy of (W_hy @ h_t) = outer(dy, h)
    db_y = dy_t

    # Gradient flowing BACK to h_{t-1} — this is the recurrent gradient flow
    # This is the term that multiplies W_hh again and again → vanishing/exploding
    dh_prev = W_hh.T @ dz_h

    grads = {
        'dW_xh': dW_xh,
        'dW_hh': dW_hh,
        'dW_hy': dW_hy,
        'db_h': db_h,
        'db_y': db_y,
        'dh_prev': dh_prev,
    }
    return grads


# ============================================================================
# PART 3: Softmax + Cross-Entropy (for char-level classification)
# ============================================================================

def softmax(z):
    """Numerically stable softmax."""
    z = z - np.max(z)  # subtract max for stability — same output, no overflow
    e = np.exp(z)
    return e / np.sum(e)


def cross_entropy_loss(probs, target_idx):
    """
    Cross-entropy for classification with a single target index.
    loss = -log(probs[target_idx])
    """
    # Clip to avoid log(0) = -inf
    p = max(probs[target_idx], 1e-12)
    return -np.log(p)


# ============================================================================
# PART 4: Tiny Character-Level RNN — Training Loop
# ============================================================================

def train_char_rnn(corpus, hidden_dim=32, seq_len=15, lr=0.1, n_iters=1500):
    """
    Train a tiny character-level RNN to predict the next character.

    Architecture:
      input  : one-hot character (vocab_size)
      hidden : RNN with tanh (hidden_dim)
      output : linear → softmax → predict next character (vocab_size)

    We train on small overlapping windows of `seq_len` characters.
    """
    # Build vocabulary from all unique chars in the corpus
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"Corpus length : {len(corpus)} chars")
    print(f"Vocab size    : {vocab_size}")
    print(f"Sample chars  : {chars[:20]}")

    # Initialize weights — small random values to break symmetry
    # WHY small: tanh saturates at |z| > 3, we want to start in the linear regime
    W_xh = np.random.randn(hidden_dim, vocab_size) * 0.01
    W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
    W_hy = np.random.randn(vocab_size, hidden_dim) * 0.01
    b_h = np.zeros(hidden_dim)
    b_y = np.zeros(vocab_size)

    # For monitoring gradient norms — we will watch vanishing/exploding
    grad_norms_history = []
    loss_history = []

    for it in range(n_iters):
        # Pick a random starting position in the corpus
        start = np.random.randint(0, len(corpus) - seq_len - 1)
        inputs = [char_to_idx[c] for c in corpus[start:start + seq_len]]
        targets = [char_to_idx[c] for c in corpus[start + 1:start + seq_len + 1]]

        # --- FORWARD PASS: unroll the RNN over seq_len time steps ---
        h_prev = np.zeros(hidden_dim)
        caches = []
        probs_list = []
        loss = 0.0

        for t in range(seq_len):
            # One-hot encode the input character
            x_t = np.zeros(vocab_size)
            x_t[inputs[t]] = 1.0

            # Forward through one RNN cell
            h_prev, y_t, cache = rnn_cell_forward(x_t, h_prev, W_xh, W_hh, W_hy, b_h, b_y)
            caches.append(cache)

            # Softmax + cross-entropy
            probs = softmax(y_t)
            probs_list.append(probs)
            loss += cross_entropy_loss(probs, targets[t])

        loss /= seq_len

        # --- BACKWARD PASS: BPTT ---
        # Initialize accumulated gradients to zero
        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        dW_hy = np.zeros_like(W_hy)
        db_h = np.zeros_like(b_h)
        db_y = np.zeros_like(b_y)

        # Gradient flowing from "the future" — at the last step, no future
        dh_next = np.zeros(hidden_dim)

        # Iterate BACKWARDS through time (this is the "T" in BPTT)
        for t in reversed(range(seq_len)):
            # Gradient on output y_t: softmax + cross-entropy has a clean form
            # dL/dy = probs - one_hot(target)
            dy = probs_list[t].copy()
            dy[targets[t]] -= 1.0
            dy /= seq_len  # because loss was averaged

            # Backprop through the cell — combines dh_next (from future) + dy
            grads_t = rnn_cell_backward(dh_next, dy, caches[t], W_hh, W_hy)

            # Accumulate gradients over time (same parameters used at every step)
            dW_xh += grads_t['dW_xh']
            dW_hh += grads_t['dW_hh']
            dW_hy += grads_t['dW_hy']
            db_h += grads_t['db_h']
            db_y += grads_t['db_y']

            # Propagate the gradient to the previous time step
            dh_next = grads_t['dh_prev']

        # --- GRADIENT CLIPPING: protect against exploding gradients ---
        # WHY: without clipping, a single bad batch can send grad to infinity
        # and destroy all weights. Clip by global norm to a max of 5.0.
        grad_list = [dW_xh, dW_hh, dW_hy, db_h, db_y]
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grad_list))
        clip = 5.0
        if total_norm > clip:
            scale = clip / (total_norm + 1e-6)
            dW_xh *= scale
            dW_hh *= scale
            dW_hy *= scale
            db_h *= scale
            db_y *= scale

        # Record stats BEFORE clipping for monitoring
        grad_norms_history.append(total_norm)
        loss_history.append(loss)

        # --- SGD UPDATE ---
        W_xh -= lr * dW_xh
        W_hh -= lr * dW_hh
        W_hy -= lr * dW_hy
        b_h -= lr * db_h
        b_y -= lr * db_y

        if (it + 1) % 150 == 0:
            print(f"  Iter {it+1:5d} | loss = {loss:.4f} | grad_norm = {total_norm:8.4f}")

    return {
        'W_xh': W_xh, 'W_hh': W_hh, 'W_hy': W_hy, 'b_h': b_h, 'b_y': b_y,
        'vocab_size': vocab_size,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'loss_history': loss_history,
        'grad_norms_history': grad_norms_history,
    }


# ============================================================================
# PART 5: Sampling from a trained RNN
# ============================================================================

def sample_from_rnn(model, seed_char, length=80):
    """
    Generate characters autoregressively from the trained RNN.
    At each step: feed the last char, get a distribution, sample, repeat.
    """
    W_xh, W_hh, W_hy = model['W_xh'], model['W_hh'], model['W_hy']
    b_h, b_y = model['b_h'], model['b_y']
    vocab_size = model['vocab_size']
    char_to_idx = model['char_to_idx']
    idx_to_char = model['idx_to_char']

    h = np.zeros(W_hh.shape[0])
    current_idx = char_to_idx.get(seed_char, 0)
    output = [seed_char]

    for _ in range(length):
        # One-hot the current character
        x = np.zeros(vocab_size)
        x[current_idx] = 1.0
        # Forward step
        h, y, _ = rnn_cell_forward(x, h, W_xh, W_hh, W_hy, b_h, b_y)
        probs = softmax(y)
        # Sample from the distribution (randomness adds diversity)
        current_idx = int(np.random.choice(vocab_size, p=probs))
        output.append(idx_to_char[current_idx])

    return ''.join(output)


# ============================================================================
# PART 6: Experiment — Observe Vanishing / Exploding Gradients
# ============================================================================

def demo_vanishing_exploding(seq_lengths=(5, 20, 50, 100)):
    """
    Show how the gradient norm through W_hh behaves for different
    sequence lengths and different initializations of W_hh.

    We compute the norm of the product:
       prod_{t=1..T} (diag(1 - h_t^2) @ W_hh)

    For simplicity we start from a random h_0 and propagate the gradient
    as if dh_T = ones.
    """
    print("\n" + "=" * 70)
    print("PART 6: Vanishing vs Exploding Gradients — empirical demonstration")
    print("=" * 70)

    hidden_dim = 16
    # Three scenarios: W_hh scaled so spectral radius < 1, = 1, > 1
    scales = {
        'vanishing (scale=0.5)': 0.5,
        'stable (scale=1.0)': 1.0,
        'exploding (scale=1.5)': 1.5,
    }

    for label, scale in scales.items():
        print(f"\n  Scenario: {label}")
        print(f"  {'T':>5s} | {'||grad||':>15s}")
        print(f"  {'-' * 25}")

        # Build a W_hh with approximately the target spectral radius
        W_hh = np.random.randn(hidden_dim, hidden_dim)
        # Normalize by its spectral norm so the largest singular value is ~1
        s = np.linalg.svd(W_hh, compute_uv=False)[0]
        W_hh = W_hh / s * scale

        for T in seq_lengths:
            # Fake a sequence of hidden states
            h = np.random.randn(hidden_dim) * 0.1
            # Gradient from the end: ones
            dh = np.ones(hidden_dim)

            # Backprop T times through the same cell (no input contribution)
            for _ in range(T):
                # diag(1 - h^2) factor (we use a fresh h to simulate a trajectory)
                h = np.tanh(W_hh @ h)
                dh = (1.0 - h ** 2) * (W_hh.T @ dh)

            norm = np.linalg.norm(dh)
            print(f"  {T:5d} | {norm:15.6e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PART 1-4: Train a tiny char-level RNN")
    print("=" * 70)

    # Small repetitive corpus — easy for a tiny RNN to memorize
    CORPUS = (
        "le chat mange la souris. "
        "le chien court dans le jardin. "
        "le chat dort sur le tapis. "
        "le chien aime le chat. "
    ) * 8

    model = train_char_rnn(CORPUS, hidden_dim=48, seq_len=20, lr=0.1, n_iters=1200)

    print("\n--- Sample from trained model ---")
    sample = sample_from_rnn(model, seed_char='l', length=120)
    print(f"  Generated: {sample}")

    # Summary of training curves — first vs last 50 iters
    losses = model['loss_history']
    print(f"\n  Loss first 50 avg  : {np.mean(losses[:50]):.4f}")
    print(f"  Loss last 50 avg   : {np.mean(losses[-50:]):.4f}")
    print(f"  Loss improvement   : {np.mean(losses[:50]) - np.mean(losses[-50:]):.4f}")

    grads = model['grad_norms_history']
    print(f"  Grad norm first 50 : {np.mean(grads[:50]):.4f}")
    print(f"  Grad norm last 50  : {np.mean(grads[-50:]):.4f}")

    # Part 6: explicit demonstration of vanishing/exploding
    demo_vanishing_exploding(seq_lengths=(5, 20, 50, 100))

    print("\n" + "=" * 70)
    print("Key takeaways from the experiments")
    print("=" * 70)
    print("""
  1. A vanilla RNN CAN learn — but only short dependencies (~10-20 chars).
  2. Without gradient clipping, training explodes quickly.
  3. As T grows, gradient norm either vanishes (scale<1) or explodes (scale>1).
  4. LSTM/GRU fix this with gated cell states (tomorrow we see Transformers
     which fix it differently — by removing recurrence altogether).
""")

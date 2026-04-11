"""
Solutions — Jour 7 : Mini-Transformer
======================================
Solutions for the 3 easy exercises.

Run: python 03-exercises/solutions/07-mini-transformer.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.set_printoptions(precision=4, suppress=True)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# EXERCISE 1: Attention by hand on 2 tokens
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Manual attention forward pass on 2 tokens")
print("=" * 70)

# Embedding table for 4 tokens in dim 2
E = np.array([
    [1.0, 0.0],    # A
    [0.0, 1.0],    # B
    [1.0, 1.0],    # C
    [-1.0, 0.0],   # D
])
# Positional embeddings for 2 positions
PE = np.array([
    [0.1, 0.0],   # pos 0
    [0.0, 0.1],   # pos 1
])

# Identity projections
I = np.eye(2)
W_Q, W_K, W_V, W_O = I, I, I, I

# Input: tokens [A, B]
token_ids = [0, 1]

# Step 1: embeddings
print("\n--- Step 1: embeddings ---")
X = np.stack([E[tid] + PE[pos] for pos, tid in enumerate(token_ids)])
print(f"  X (tokens embedded + position) =\n{X}")

# Step 2: Q, K, V
print("\n--- Step 2: Q, K, V with identity projections ---")
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
print(f"  Q = {Q}")
print(f"  K = {K}")
print(f"  V = {V}")

# Step 3: scores
print("\n--- Step 3: scores = Q @ K^T / sqrt(2) ---")
scores = Q @ K.T / np.sqrt(2)
print(f"  scores =\n{scores}")

# Step 4: causal mask
print("\n--- Step 4: causal mask ---")
mask = np.array([[1, 0],    # pos 0 sees only itself
                 [1, 1]])    # pos 1 sees pos 0 and itself
print(f"  mask (1=allowed, 0=blocked) =\n{mask}")
scores_masked = np.where(mask == 1, scores, -np.inf)
print(f"  scores after mask =\n{scores_masked}")

# Step 5: softmax
print("\n--- Step 5: softmax per row ---")
weights = softmax(scores_masked, axis=-1)
print(f"  weights =\n{weights}")
print("\n  Row 0 (token A): full attention on itself (A can't see B - future).")
print("  Row 1 (token B): distributes attention across A and itself.")

# Step 6: output
print("\n--- Step 6: output = weights @ V ---")
output = weights @ V
print(f"  output =\n{output}")

# Interpretation
print("""
--- Q7: Interpretation ---

  For token A (row 0): output = V[A] only. A's representation after
  attention is unaffected by B, because B is in the future.

  For token B (row 1): output is a weighted sum of V[A] and V[B]. B can
  look back at A and incorporate information from it.

  This is exactly what the causal mask enforces: information flows forward
  in time only. It's crucial for autoregressive generation — during training
  the model sees the full sequence, but each position can only "see" its
  own past, mimicking the generation scenario where the future doesn't exist.
""")


# ============================================================================
# EXERCISE 2: Changing n_head
# ============================================================================

print("=" * 70)
print("EXERCISE 2: Changing n_head from 4 to 8")
print("=" * 70)

n_embed = 48

# Q1: head_dim for different configs
for n_head in [2, 4, 8, 16]:
    divisible = (n_embed % n_head == 0)
    head_dim = n_embed // n_head if divisible else n_embed / n_head
    status = "OK" if divisible else "INVALID"
    print(f"  n_head = {n_head:2d}  ->  head_dim = {head_dim}  [{status}]")

print("\n--- Q1 answer ---")
print("  n_embed must be divisible by n_head. For n_embed=48, valid values")
print("  are: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48. n_head=5 would fail.")

# Q2: parameter counts
print("\n--- Q2: Parameter counts ---")
for n_head in [4, 8]:
    # Attention: W_Q, W_K, W_V, W_O, all (n_embed, n_embed)
    # (Bias=False as in the mini-GPT code)
    attn_params = 4 * n_embed * n_embed
    head_dim = n_embed // n_head
    print(f"\n  n_head = {n_head}")
    print(f"    head_dim      = {head_dim}")
    print(f"    attn params   = 4 * {n_embed}^2 = {attn_params}")

print("\n  -> EXACTLY THE SAME number of parameters in both configs!")

# Q3: Explanation
print("""
--- Q3: Why the same count? ---

  The projections W_Q, W_K, W_V, W_O are all (n_embed, n_embed) regardless
  of n_head. The multi-head split happens by RESHAPING the output of these
  projections into (n_head, head_dim), but no extra weights are introduced.

  So the total parameter count is independent of n_head. The only thing
  that changes is HOW the computation is organized internally.
""")

# Q4: Code modification
print("--- Q4: Code modification ---")
print("""
  Change this line:
      config = dict(n_embed=48, n_head=4, ...)
  To:
      config = dict(n_embed=48, n_head=8, ...)

  That's the only change needed. (Make sure n_embed remains divisible by n_head.)
""")

# Q5: More heads tradeoffs
print("""
--- Q5: 4 heads vs 8 heads — which is better? ---

  For the same parameter count:
  - More heads = more independent "attention patterns" in parallel
    (each head can specialize in a different relation type: coref, syntax, etc.)
  - Fewer heads = each head has more dimensions (richer per-head representation)

  In practice, moderate n_head values (8-16 for medium models) work best.
  Too few heads -> limited diversity. Too many heads -> each head becomes
  too low-dimensional to be expressive.

  Empirically, the original Transformer used n_head=8 for d_model=512
  (head_dim=64), and this has become a de-facto standard.

--- Q6: Why GPT-3 uses 96 heads? ---

  GPT-3 has d_model=12288 and n_head=96, so head_dim=128. At this scale,
  having more heads means more parallel specialization AND the head_dim
  stays large enough (128) to be expressive. It's a win-win at huge scale.

  In general: scale heads AND head_dim together as the model grows.
""")


# ============================================================================
# EXERCISE 3: Temperature sampling
# ============================================================================

print("=" * 70)
print("EXERCISE 3: Temperature sampling")
print("=" * 70)

logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
print(f"\nLogits: {logits}")

# Q1: T = 1.0
print("\n--- Q1: T = 1.0 (standard softmax) ---")
probs_1 = softmax(logits)
for i, p in enumerate(probs_1):
    print(f"  token {i} (logit {logits[i]:+.1f}) : p = {p:.4f}")
print(f"  Sum: {probs_1.sum():.4f}")
print(f"  Most likely: token {np.argmax(probs_1)} with p = {probs_1.max():.4f}")

# Q2: T = 0.5
print("\n--- Q2: T = 0.5 (sharper) ---")
logits_sharp = logits / 0.5
probs_05 = softmax(logits_sharp)
for i, p in enumerate(probs_05):
    print(f"  token {i} : p = {p:.4f}")
print(f"  Most likely now : {probs_05.max():.4f}  (vs {probs_1.max():.4f} before)")
print(f"  -> Distribution is PEAKIER, concentrates mass on the top token.")

# Q3: T = 2.0
print("\n--- Q3: T = 2.0 (flatter) ---")
logits_flat = logits / 2.0
probs_2 = softmax(logits_flat)
for i, p in enumerate(probs_2):
    print(f"  token {i} : p = {p:.4f}")
print(f"  Most likely now : {probs_2.max():.4f}  (vs {probs_1.max():.4f} at T=1)")
print(f"  -> Distribution is FLATTER, less concentrated.")

# Q4: Greedy
print("\n--- Q4: Greedy decoding ---")
print(f"  Greedy always picks argmax(logits) = token {np.argmax(logits)}.")
print("  Problem for long text generation:")
print("    - Deterministic: always the same output for the same prompt.")
print("    - Tends to fall into repetitive loops (e.g., 'the the the the...')")
print("    - No diversity — can't explore alternative continuations.")

# Q5: Limits
print("\n--- Q5: Temperature limits ---")
for T in [0.01, 0.1, 1.0, 5.0, 100.0]:
    p = softmax(logits / T)
    print(f"  T = {T:6.2f}  ->  probs = {p}")
print("\n  As T -> 0: distribution becomes one-hot (= greedy).")
print("  As T -> inf: distribution becomes uniform (every token equally likely).")

# Q6: Practical temperatures
print("""
--- Q6: Practical temperature choices ---

  | Task                    | Suggested T | Why                              |
  |-------------------------|-------------|----------------------------------|
  | Code generation         | 0.0 - 0.3   | Precision matters, syntax strict |
  | Factual Q&A             | 0.0 - 0.4   | Avoid hallucinations             |
  | Summarization           | 0.5 - 0.7   | Some flexibility, stay faithful  |
  | Conversational chatbot  | 0.7 - 1.0   | Natural, varied responses        |
  | Creative writing, poem  | 0.9 - 1.3   | Encourage diversity and surprise |
  | Brainstorming           | 1.0 - 1.5   | Maximum exploration              |

  Modern LLMs (GPT-4, Claude) often combine temperature with top-p (nucleus)
  sampling to avoid rare tokens completely derailing the output.
""")

print("=" * 70)
print("All 3 exercises completed.")
print("End of Week 1 - capstone done!")
print("=" * 70)

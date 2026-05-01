"""
Jour 10 — Fine-tuning & Alignment: DPO and LoRA from scratch
=============================================================
PyTorch if available, NumPy fallback otherwise.

Covers:
  1. DPO loss function with full math exposed
  2. LoRA module wrapping a Linear layer
  3. Parameter count comparison
  4. Mini training loop to show DPO converges

Run: python 02-code/10-fine-tuning-alignment.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.manual_seed(42)
except ImportError:
    HAS_TORCH = False
    print("[info] PyTorch not available — using NumPy-only fallback.")


# ============================================================================
# PART 1: DPO loss from scratch (NumPy)
# ============================================================================

print("=" * 70)
print("PART 1: DPO loss from scratch")
print("=" * 70)


def log_prob_sequence_np(probs_per_token, target_ids):
    """
    Compute sum of log probabilities of the target token ids.

    Args:
      probs_per_token: (seq_len, vocab) — probability distributions per step
      target_ids: (seq_len,) — the actual tokens generated

    Returns: scalar = sum of log p(token_i)

    WHY sum and not mean: DPO uses the total log-likelihood of the sequence,
    not the per-token average. This matches the formulation in the paper.
    """
    eps = 1e-10  # avoid log(0)
    # Select only the probs corresponding to the target tokens
    token_probs = probs_per_token[np.arange(len(target_ids)), target_ids]
    return np.sum(np.log(token_probs + eps))


def dpo_loss_np(policy_logp_chosen, policy_logp_rejected,
                ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """
    DPO loss computation.

    Formula:
      L = -log sigmoid(
          beta * ((logp_policy(y_w) - logp_ref(y_w))
                - (logp_policy(y_l) - logp_ref(y_l)))
      )

    Args:
      policy_logp_chosen: log P(y_w | x) under the model being trained
      policy_logp_rejected: log P(y_l | x) under the model being trained
      ref_logp_chosen: log P(y_w | x) under the frozen reference model
      ref_logp_rejected: log P(y_l | x) under the frozen reference model
      beta: KL-penalty coefficient (typical 0.1 to 0.5)

    Returns:
      loss: scalar loss (lower = better)
      reward_margin: scalar = how much more the model prefers w vs l
    """
    # Log-ratio for chosen and rejected
    # r_theta(y_w) = beta * log(π_θ(y_w) / π_ref(y_w))
    #             = beta * (logp_policy - logp_ref)
    chosen_rewards = beta * (policy_logp_chosen - ref_logp_chosen)
    rejected_rewards = beta * (policy_logp_rejected - ref_logp_rejected)

    # The margin we want to maximize
    reward_margin = chosen_rewards - rejected_rewards

    # Binary classification via sigmoid, then log
    # L = -log(sigmoid(margin)) = -log(1 / (1 + exp(-margin))) = log(1 + exp(-margin))
    # Use a numerically stable version:
    # For positive x: log(1 + exp(-x))
    # For negative x: -x + log(1 + exp(x))
    if reward_margin >= 0:
        loss = math.log(1 + math.exp(-reward_margin))
    else:
        loss = -reward_margin + math.log(1 + math.exp(reward_margin))
    return loss, reward_margin


# Demo: run DPO loss on fake log-probs
# Scenario 1: policy already prefers chosen -> low loss
policy_w = -2.0   # log P(y_w) under policy (close to 1 in prob-space)
policy_l = -5.0   # log P(y_l) under policy
ref_w = -2.5      # log P(y_w) under reference (slightly worse)
ref_l = -3.0      # log P(y_l) under reference (about the same)

loss1, margin1 = dpo_loss_np(policy_w, policy_l, ref_w, ref_l, beta=0.1)
print(f"\nScenario 1 — policy prefere chosen plus que ref:")
print(f"  policy logp chosen:   {policy_w}")
print(f"  policy logp rejected: {policy_l}")
print(f"  ref    logp chosen:   {ref_w}")
print(f"  ref    logp rejected: {ref_l}")
print(f"  reward margin: {margin1:+.4f}")
print(f"  loss: {loss1:.4f}  (low = good)")

# Scenario 2: policy prefers rejected -> high loss, pushes to correct
policy_w = -5.0
policy_l = -2.0
loss2, margin2 = dpo_loss_np(policy_w, policy_l, ref_w, ref_l, beta=0.1)
print(f"\nScenario 2 — policy prefere rejected (erreur):")
print(f"  reward margin: {margin2:+.4f}")
print(f"  loss: {loss2:.4f}  (high = gradient va pousser dans la bonne direction)")


# ============================================================================
# PART 2: LoRA module from scratch
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: LoRA Linear wrapper")
print("=" * 70)


class LoRALinearNumpy:
    """
    Pure NumPy LoRA wrapper around a Linear layer.

    Structure:
      y = x @ W.T + bias         (frozen base)
        + (x @ A.T) @ B.T * scale  (trainable LoRA adapter)

    Where:
      W: (out, in) - frozen
      A: (r, in) - trainable, random init
      B: (out, r) - trainable, zero init (!)

    WHY zero init for B: at step 0, B @ A = 0, so the LoRA layer output
    equals the frozen output. The adapter only kicks in after gradient
    updates. This guarantees no regression at the start of training.
    """

    def __init__(self, in_features, out_features, r=8, alpha=16):
        # Frozen base weights — random init for the demo
        self.W = np.random.randn(out_features, in_features) * 0.02
        self.bias = np.zeros(out_features)

        # Trainable LoRA matrices
        # A: random small values (like a Linear layer init)
        self.A = np.random.randn(r, in_features) * 0.01
        # B: ZERO init so B @ A = 0 at the start
        self.B = np.zeros((out_features, r))

        # scaling factor — controls how much the adapter contributes
        self.scale = alpha / r

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

    def forward(self, x):
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        # Frozen base output
        base = x @ self.W.T + self.bias
        # LoRA adapter output: (x @ A^T) then @ B^T, scaled
        # Shape: (batch, in) @ (in, r) -> (batch, r) @ (r, out) -> (batch, out)
        lora_out = (x @ self.A.T) @ self.B.T * self.scale
        return base + lora_out

    def trainable_params(self):
        """Return the count of trainable parameters (A + B only)."""
        return self.A.size + self.B.size

    def total_params(self):
        """Total params including frozen base."""
        return self.W.size + self.bias.size + self.A.size + self.B.size


# Demo: parameter count for different r values
d_in = d_out = 4096
print(f"\nDimension: d_in = d_out = {d_in}")
print(f"\n{'r':>4s} {'trainable':>14s} {'full_FT':>14s} {'compression':>14s}")
print("-" * 50)
full_ft = d_in * d_out  # full fine-tuning would train all of W
for r in [1, 4, 8, 16, 64]:
    layer = LoRALinearNumpy(d_in, d_out, r=r)
    trainable = layer.trainable_params()
    ratio = full_ft / trainable
    print(f"{r:>4d} {trainable:>14,d} {full_ft:>14,d} {ratio:>13.1f}x")


# Sanity check: at init, LoRA output == base output (because B = 0)
layer = LoRALinearNumpy(8, 4, r=2)
x_sample = np.random.randn(3, 8)
out_with_lora = layer.forward(x_sample)
# Manually compute "pure base" output by temporarily setting A to zero
base_only = x_sample @ layer.W.T + layer.bias
print(f"\nInitial equivalence check (B = 0, so LoRA adds 0):")
print(f"  max diff between LoRA and base: {np.max(np.abs(out_with_lora - base_only)):.6e}")
print("  -> LoRA output equals the base output at init. Good.")


# ============================================================================
# PART 3: PyTorch version + mini DPO training loop
# ============================================================================

if HAS_TORCH:
    print("\n" + "=" * 70)
    print("PART 3: PyTorch DPO + LoRA")
    print("=" * 70)

    class LoRALinear(nn.Module):
        """LoRA wrapping a Linear layer, PyTorch version."""

        def __init__(self, in_features, out_features, r=8, alpha=16):
            super().__init__()
            # The frozen Linear — note: we do NOT call nn.Linear.register_parameter
            # on it, but we mark requires_grad=False so backprop skips it.
            self.linear = nn.Linear(in_features, out_features)
            for p in self.linear.parameters():
                p.requires_grad = False

            self.r = r
            self.alpha = alpha
            self.scale = alpha / r

            # Trainable adapters
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        def forward(self, x):
            base = self.linear(x)
            lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
            return base + lora

        def trainable_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def total_params(self):
            return sum(p.numel() for p in self.parameters())

    # Build a tiny LoRA layer and a matching plain Linear for comparison
    lora = LoRALinear(64, 64, r=4, alpha=8)
    plain = nn.Linear(64, 64)

    print(f"\nLoRA r=4 trainable: {lora.trainable_params()} / {lora.total_params()}")
    print(f"Plain Linear total: {sum(p.numel() for p in plain.parameters())}")

    # ------------------------------------------------------------------
    # Mini DPO training loop on synthetic "preferences"
    # ------------------------------------------------------------------
    # Toy setup: a LLM on a 5-token vocab. We build a reference "policy"
    # with random weights and train a policy to prefer token 0 over token 4.
    # "chosen" = token 0, "rejected" = token 4.

    vocab = 5

    class TinyLM(nn.Module):
        """A tiny language model: just a learnable 5-dim vector of logits."""
        def __init__(self):
            super().__init__()
            # Start from uniform logits
            self.logits = nn.Parameter(torch.zeros(vocab))

        def log_prob(self, token_id):
            return F.log_softmax(self.logits, dim=-1)[token_id]

    policy = TinyLM()
    ref = TinyLM()
    # Freeze the reference
    for p in ref.parameters():
        p.requires_grad = False

    # Small optimizer
    optim = torch.optim.SGD(policy.parameters(), lr=0.5)
    beta = 0.1

    chosen_id = 0
    rejected_id = 4

    print("\nMini DPO training (prefere token 0 sur token 4):")
    print(f"{'step':>5s} {'p(chosen)':>12s} {'p(rejected)':>12s} {'loss':>8s}")

    for step in range(30):
        # Forward: compute log-probs of chosen and rejected under both models
        policy_logp_w = policy.log_prob(chosen_id)
        policy_logp_l = policy.log_prob(rejected_id)
        ref_logp_w = ref.log_prob(chosen_id).detach()
        ref_logp_l = ref.log_prob(rejected_id).detach()

        # DPO loss
        chosen_rewards = beta * (policy_logp_w - ref_logp_w)
        rejected_rewards = beta * (policy_logp_l - ref_logp_l)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % 5 == 0 or step == 29:
            probs = F.softmax(policy.logits, dim=-1)
            print(f"{step:>5d} {probs[chosen_id].item():>12.4f} "
                  f"{probs[rejected_id].item():>12.4f} {loss.item():>8.4f}")

    print("""
Observation: la probabilite de 'chosen' augmente, celle de 'rejected' baisse.
C'est exactement ce que DPO est cense faire: pousser la policy vers les
preferences chosen sans reward model, sans RL, juste avec du backprop.
""")
else:
    print("\n[info] Skipping PART 3 (no PyTorch).")

print("=" * 70)
print("Fin — tu as implemente DPO et LoRA.")
print("=" * 70)

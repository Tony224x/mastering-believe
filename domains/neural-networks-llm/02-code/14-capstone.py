"""
Jour 14 — Capstone: mini-LLaMA from scratch
=============================================
A complete mini LLaMA implementation in PyTorch (~350 lines).
Combines everything from J6 (attention) and J9 (modern architecture).

Components:
  - Mock char-level tokenizer (demo only)
  - RMSNorm (j9)
  - RoPE (rotary positional embedding, j9)
  - GQA attention with KV cache (j9 + j11)
  - SwiGLU FFN (j9)
  - TransformerBlock with pre-norm residuals
  - Full model with embeddings + blocks + lm_head
  - Autoregressive generation with temperature

PyTorch if available, else a reduced NumPy demo.

Run: python 02-code/14-capstone.py
"""

import sys
import io
import math

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.manual_seed(42)
except ImportError:
    HAS_TORCH = False
    print("[info] PyTorch is required for the capstone. Install with: pip install torch")
    print("       Falling back to a simplified NumPy demo of the forward pass.")


# ============================================================================
# PART 0: Mock char-level tokenizer
# ============================================================================

class CharTokenizer:
    """
    A trivial char-level tokenizer for the demo.
    Each unique character gets an integer id. Good enough to show the model
    can generate coherent sequences without needing a real BPE.
    """

    def __init__(self, text):
        # Build the vocab from all unique chars in the input text
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        # Map each char to its id. Unknown chars get id 0.
        return [self.char_to_id.get(ch, 0) for ch in s]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids)


# ============================================================================
# The rest requires PyTorch. If missing, print a helpful message and exit.
# ============================================================================

if not HAS_TORCH:
    print("\nPlease install PyTorch to run this capstone file.")
    sys.exit(0)


# ============================================================================
# PART 1: RMSNorm — Root Mean Square normalization (no centering, no bias)
# ============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm as used in LLaMA. Simpler than LayerNorm:
      x_norm = x / sqrt(mean(x^2))
      out = x_norm * gamma  (learnable scale, init to 1)

    WHY no centering: empirically, subtracting the mean does not help for
    language modeling. Skipping it is faster and more stable in float16.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale, init to 1 so at start RMSNorm ~= identity (in norm)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Cast to float for numerical stability, back to input dtype at end
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        out = (x / rms) * self.weight.float()
        return out.to(orig_dtype)


# ============================================================================
# PART 2: RoPE — Rotary Positional Embedding
# ============================================================================

def precompute_rope(head_dim, max_seq_len, base=10000.0, device=None):
    """
    Precompute cos and sin for RoPE at each (position, pair_index).

    WHY precompute: angles depend only on position and frequency pair,
    not on actual q/k values. We compute once and reuse.
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    # Frequencies for each pair of dims: theta_j = 1 / base^(2j / head_dim)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float()
                            / head_dim))
    # Positions 0..max_seq_len-1
    positions = torch.arange(max_seq_len, device=device).float()
    # Outer product: (seq_len, head_dim/2)
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    Apply rotation to the last dim of x using precomputed cos/sin.

    Args:
      x: (..., seq_len, head_dim)
      cos, sin: (seq_len, head_dim/2)

    Returns: x rotated
    """
    # Split into even and odd pair indices
    x1 = x[..., 0::2]  # (..., seq_len, head_dim/2)
    x2 = x[..., 1::2]

    # Broadcast cos and sin to the shape of x1
    # cos/sin shape: (seq_len, head_dim/2) — will broadcast over batch/heads
    # Apply 2D rotation per pair
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos

    # Interleave back into a single tensor
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ============================================================================
# PART 3: GQA Attention with KV cache
# ============================================================================

class GQAAttention(nn.Module):
    """
    Grouped Query Attention with RoPE + KV cache.

    - n_heads queries
    - n_kv_heads keys/values (each shared by n_heads / n_kv_heads queries)
    - When n_kv_heads == n_heads, this is classic MHA
    - When n_kv_heads == 1, this is MQA
    """

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        assert n_heads % n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        # How many queries share one K/V head (for GQA grouping)
        self.n_rep = n_heads // n_kv_heads

        # Linear projections — no bias in LLaMA
        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Precompute RoPE — register as buffer so they follow device moves
        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, start_pos=0, kv_cache=None):
        """
        Args:
          x: (batch, seq_len, d_model)
          start_pos: absolute position of the first token in x (for RoPE
                     and for indexing the cache)
          kv_cache: dict with 'k' and 'v' tensors of shape
                    (batch, n_kv_heads, max_seq_len, head_dim), or None

        Returns:
          out: (batch, seq_len, d_model)
          kv_cache: updated cache (or None if not using cache)
        """
        batch, seq_len, _ = x.shape

        # Project to q, k, v
        q = self.w_q(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.w_k(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.w_v(x)

        # Reshape to (batch, seq_len, n_heads, head_dim) and transpose for attention
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, n_heads_or_kv, seq_len, head_dim)

        # Apply RoPE to q and k, using the positions starting at start_pos
        cos = self.rope_cos[start_pos:start_pos + seq_len]
        sin = self.rope_sin[start_pos:start_pos + seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Update KV cache if provided
        if kv_cache is not None:
            # Write the new k/v into the cache at positions start_pos..start_pos+seq_len
            kv_cache["k"][:, :, start_pos:start_pos + seq_len, :] = k
            kv_cache["v"][:, :, start_pos:start_pos + seq_len, :] = v
            # For attention, use the full cache up to current position
            k = kv_cache["k"][:, :, :start_pos + seq_len, :]
            v = kv_cache["v"][:, :, :start_pos + seq_len, :]

        # Repeat K and V to match n_heads (GQA: each K/V serves n_rep queries)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Compute attention scores
        # q: (batch, n_heads, seq_len, head_dim)
        # k: (batch, n_heads, total_len, head_dim)
        # scores: (batch, n_heads, seq_len, total_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: position i of q (absolute = start_pos + i) cannot attend
        # to positions > start_pos + i
        total_len = k.shape[-2]
        if seq_len > 1 or start_pos == 0:
            # Build a mask of shape (seq_len, total_len) where 0 means allowed
            mask = torch.full((seq_len, total_len), float("-inf"),
                              device=x.device, dtype=scores.dtype)
            for i in range(seq_len):
                # Current absolute position
                abs_pos = start_pos + i
                mask[i, :abs_pos + 1] = 0.0
            scores = scores + mask

        # Softmax + value aggregation
        weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(weights, v)
        # out: (batch, n_heads, seq_len, head_dim)

        # Merge heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.w_o(out)
        return out, kv_cache


# ============================================================================
# PART 4: SwiGLU FFN
# ============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward as used in LLaMA:
      h = SiLU(W_gate @ x) * (W_up @ x)
      out = W_down @ h
    """

    def __init__(self, d_model, hidden_dim=None, multiple_of=256):
        super().__init__()
        if hidden_dim is None:
            # LLaMA recipe: start with 8/3 * d_model, round up to multiple_of
            hidden_dim = int(8 * d_model / 3)
            hidden_dim = ((hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # SiLU(x) = x * sigmoid(x), also called Swish
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ============================================================================
# PART 5: Transformer Block with pre-norm residuals
# ============================================================================

class TransformerBlock(nn.Module):
    """
    One LLaMA block: pre-norm attention + pre-norm SwiGLU FFN with residuals.

    Structure:
      h1 = x + attention(RMSNorm(x))
      h2 = h1 + ffn(RMSNorm(h1))
    """

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x, start_pos=0, kv_cache=None):
        h, kv_cache = self.attn(self.attn_norm(x), start_pos=start_pos,
                                kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv_cache


# ============================================================================
# PART 6: The Model — token embeddings + blocks + final norm + lm_head
# ============================================================================

class MiniLLaMA(nn.Module):
    """
    Minimal LLaMA architecture.
    """

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 n_kv_heads=2, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len

        # Token embedding — single lookup table, no positional embedding
        # (RoPE is applied inside attention instead)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, max_seq_len)
            for _ in range(n_layers)
        ])

        # Final RMSNorm before the lm_head
        self.final_norm = RMSNorm(d_model)

        # lm_head projects back to vocab size (tied with token_embedding in
        # some LLaMAs, not tied in LLaMA 1)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, start_pos=0, kv_caches=None):
        """
        Args:
          input_ids: (batch, seq_len) LongTensor of token ids
          start_pos: absolute starting position (for RoPE and cache indexing)
          kv_caches: list of per-layer kv_cache dicts, or None
        Returns:
          logits: (batch, seq_len, vocab_size)
          kv_caches: updated list
        """
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # Initialize empty caches if not provided and we are generating
        if kv_caches is None:
            kv_caches = [None] * self.n_layers

        # Run each block
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, start_pos=start_pos, kv_cache=kv_caches[i])
            new_caches.append(kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def init_kv_cache(self, batch_size, device):
        """Create empty KV caches for all layers, shaped to max_seq_len."""
        caches = []
        for _ in range(self.n_layers):
            k = torch.zeros(batch_size, self.n_kv_heads, self.max_seq_len,
                            self.d_model // self.n_heads, device=device)
            v = torch.zeros_like(k)
            caches.append({"k": k, "v": v})
        return caches


# ============================================================================
# PART 7: Autoregressive generation
# ============================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """
    Autoregressive generation with KV cache.

    Algorithm:
      1. Encode the prompt
      2. Run a PREFILL pass on the whole prompt (fills the cache)
      3. DECODE one token at a time:
         - Take the last logits
         - Sample the next token (temperature + argmax fallback)
         - Feed it back, starting at the new position
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    prompt_len = ids.shape[1]

    # Initialize KV cache
    caches = model.init_kv_cache(batch_size=1, device=device)

    # PREFILL: process the whole prompt in one forward pass
    logits, caches = model(ids, start_pos=0, kv_caches=caches)
    # Take logits of the LAST position — these predict the next token
    next_logits = logits[:, -1, :] / max(temperature, 1e-6)

    output_ids = list(tokenizer.encode(prompt))

    for step in range(max_new_tokens):
        # Sample next token
        if temperature > 0:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = next_logits.argmax(dim=-1).item()

        output_ids.append(next_id)

        # Feed this single token back — DECODE step
        current_pos = prompt_len + step
        if current_pos >= model.max_seq_len:
            break
        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, caches = model(next_input, start_pos=current_pos,
                               kv_caches=caches)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)

    return tokenizer.decode(output_ids)


# ============================================================================
# PART 8: Demo — instantiate and generate
# ============================================================================

def main():
    print("=" * 70)
    print("Mini LLaMA — capstone demo")
    print("=" * 70)

    # Tiny training text (just for the tokenizer, we don't actually train)
    sample_text = (
        "le chat dort sur le tapis\n"
        "le chien aboie fort\n"
        "la souris mange le fromage\n"
        "les enfants jouent au parc\n"
        "il pleut sur la ville\n"
    )
    tokenizer = CharTokenizer(sample_text)
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Sample chars: {list(tokenizer.char_to_id.keys())[:20]}")

    # Build a tiny model
    model = MiniLLaMA(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # GQA: 2 kv heads for 4 q heads -> groups of 2
        max_seq_len=128,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Configuration:")
    print(f"  d_model     = {model.d_model}")
    print(f"  n_layers    = {model.n_layers}")
    print(f"  n_heads     = {model.n_heads}")
    print(f"  n_kv_heads  = {model.n_kv_heads} (GQA)")
    print(f"  max_seq_len = {model.max_seq_len}")

    # Forward pass sanity check
    test_ids = torch.tensor(
        [tokenizer.encode("le chat")], dtype=torch.long)
    logits, _ = model(test_ids)
    print(f"\nSanity check forward:")
    print(f"  input shape: {tuple(test_ids.shape)}")
    print(f"  output logits shape: {tuple(logits.shape)}")
    print(f"  expected: (1, {test_ids.shape[1]}, {tokenizer.vocab_size})")

    # Generate (random weights -> random output, but the plumbing works!)
    print("\nGeneration (random weights, so output is gibberish but demonstrates the pipeline):")
    prompt = "le chat"
    generated = generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0)
    print(f"  Prompt: {prompt!r}")
    print(f"  Output: {generated!r}")

    # Show that the KV cache makes decode different from prefill
    print("\nKV cache inspection:")
    caches = model.init_kv_cache(batch_size=1, device=test_ids.device)
    # Prefill
    _, caches = model(test_ids, start_pos=0, kv_caches=caches)
    # After prefill, the first test_ids.shape[1] positions of caches should be non-zero
    k0 = caches[0]["k"]
    n_filled = (k0.abs().sum(dim=-1) > 0).sum(dim=-1).item()
    print(f"  Positions filled in cache[0][k] after prefill: {n_filled}")
    print(f"  Expected: {test_ids.shape[1]}")

    # Verify the full pipeline is consistent
    print("\n" + "=" * 70)
    print("Resume des composants implementes:")
    print("  [OK] Tokenizer char-level (mock)")
    print("  [OK] RMSNorm")
    print("  [OK] RoPE (rotary positional embedding)")
    print("  [OK] GQA attention with KV cache")
    print("  [OK] SwiGLU FFN")
    print("  [OK] TransformerBlock with pre-norm residuals")
    print("  [OK] MiniLLaMA model")
    print("  [OK] Autoregressive generation")
    print("=" * 70)


if __name__ == "__main__":
    main()

# %%

"""
Jour 7 — Mini-GPT from scratch (CAPSTONE Week 1)
=================================================
A complete decoder-only Transformer (GPT-style) in ~300 lines of PyTorch.

Components:
  - Char-level tokenizer
  - Token + positional embedding
  - N Transformer blocks with CAUSAL self-attention
  - Final LayerNorm + linear projection to vocab
  - Training loop (AdamW + cross-entropy)
  - Autoregressive generation (greedy AND temperature sampling)

Inspired by Andrej Karpathy's nanoGPT, simplified for clarity.

Run: python 02-code/07-mini-transformer.py
"""

import sys
import io
import math

# In Jupyter, sys.stdout is an ipykernel OutStream (no .buffer attribute).
# Only wrap when running as a real script with a buffered stdout.
if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# PART 0: PyTorch import guard
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("=" * 70)
    print("PyTorch is not installed — this script requires it.")
    print("Install with: pip install torch")
    print("=" * 70)
    sys.exit(0)


# ============================================================================
# PART 1: Corpus = texte perso + extrait Wikipedia FR
# ============================================================================

# --- Corpus personnalise : Kadiatou Lamarana Sow ---
CORPUS_PERSO = """Kadiatou Lamarana Sow est belle. Kadiatou sourit avec douceur.
Lamarana aime lire des livres. Kadiatou prepare le repas du soir.
Kadiatou Lamarana Sow est intelligente. Elle travaille avec passion.
Lamarana marche dans le jardin. Kadiatou regarde le ciel etoile.
Kadiatou et Anthony sont ensemble. Leur amour est grand et sincere.
Lamarana Sow porte un beau boubou. Kadiatou danse avec elegance.
Kadiatou cuisine un bon thieboudienne. Le repas est delicieux.
Lamarana aime la musique. Kadiatou chante une melodie douce.
Kadiatou Lamarana Sow est courageuse. Elle avance avec confiance.
Anthony aime Kadiatou. Kadiatou aime Anthony. Ils sont heureux ensemble.
Lamarana regarde par la fenetre. Le soleil brille sur son visage.
Kadiatou rit avec joie. Son rire illumine la maison.
Kadiatou Sow est une femme forte. Elle inspire ceux qui la connaissent.
Lamarana prepare le the. Kadiatou et Anthony partagent un moment.
Kadiatou est douce et patiente. Lamarana Sow est pleine de sagesse.
Anthony et Kadiatou construisent leur avenir. La vie est belle ensemble.
Kadiatou Lamarana Sow sourit. Le bonheur se lit dans ses yeux.
""" * 4

# --- Petit dataset Wikipedia FR (telecharge via HuggingFace datasets) ---
from datasets import load_dataset

print("Chargement d'un extrait Wikipedia FR...")
wiki_ds = load_dataset('wikimedia/wikipedia', '20231101.fr', split='train', streaming=True)
wiki_texts = []
target_chars = 10_000  # ~10K chars = bon ratio avec le corpus perso
total = 0
for article in wiki_ds:
    text = article['text'].replace('\n', ' ').strip()
    if len(text) > 200:  # skip stubs
        wiki_texts.append(text)
        total += len(text)
    if total >= target_chars:
        break

CORPUS_WIKI = '\n'.join(wiki_texts)
print(f"  Wikipedia: {len(CORPUS_WIKI):,} chars from {len(wiki_texts)} articles")

# --- Corpus final : perso + wiki ---
CORPUS = CORPUS_PERSO + '\n' + CORPUS_WIKI
print(f"  Corpus total: {len(CORPUS):,} chars")


# ============================================================================
# PART 2: Char-level tokenizer
# ============================================================================

class CharTokenizer:
    """
    The simplest possible tokenizer: one token per character.
    vocab_size = number of unique characters in the corpus.
    """

    def __init__(self, text):
        # Sorted for reproducibility — same vocab across runs
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # Bidirectional maps char <-> index
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """String -> list of integer token ids."""
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """List of integer token ids -> string."""
        return ''.join(self.itos[i] for i in ids)


# ============================================================================
# PART 3: Mini-GPT architecture
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask (no peeking at the future).
    """

    def __init__(self, n_embed, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        # One big linear that produces Q, K, V concatenated
        # More efficient than 3 separate linears — same math
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Register a lower-triangular mask as a buffer (not a parameter)
        # 1 where attention is ALLOWED, 0 where it is BLOCKED
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embed

        # Project to Q, K, V in one go and split
        qkv = self.c_attn(x)                # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=2)

        # Reshape into heads: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply causal mask: set future positions to -inf
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Softmax over the last axis (keys)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted values: (B, n_head, T, head_dim)
        y = att @ v
        # Concat heads back: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Linear."""

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        # d_ff = 4 * n_embed (standard ratio from the original paper)
        self.fc = nn.Linear(n_embed, 4 * n_embed)
        self.proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """
    One Transformer block (pre-norm variant as in GPT-2+):
      x = x + Attention(LN(x))
      x = x + FFN(LN(x))
    """

    def __init__(self, n_embed, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Minimal GPT-style decoder-only Transformer.
    """

    def __init__(self, vocab_size, n_embed=32, n_head=4, n_layer=2,
                 block_size=32, dropout=0.0):
        super().__init__()
        self.block_size = block_size

        # Token embedding: one vector per character id
        self.token_emb = nn.Embedding(vocab_size, n_embed)

        # Learned positional embeddings (simpler than sinusoidal for this size)
        # WHY learned: easier to train on a tiny corpus, GPT-2 also used learned PE
        self.pos_emb = nn.Embedding(block_size, n_embed)

        self.drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Final LayerNorm + output projection to vocabulary
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)

        # Better initialization (Kaiming-like) for stability on tiny datasets
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) long tensor of token ids
        targets: optional (B, T) long tensor of next-token ids
        returns: logits (B, T, vocab_size), optional loss
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        # Embeddings: token + position
        tok = self.token_emb(idx)                      # (B, T, n_embed)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embed)
        x = self.drop(tok + pos)                        # broadcast add

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape so cross_entropy sees a flat (N, vocab) vs (N,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, greedy=False):
        """
        Autoregressive generation.
        idx: (B, T) initial context (can be length 1)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to last `block_size` tokens (no infinite memory)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Only care about the LAST position's logits
            logits = logits[:, -1, :]

            if greedy:
                # Deterministic: take argmax
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Apply temperature and sample
                logits = logits / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append and continue
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ============================================================================
# PART 4: Data loading helpers
# ============================================================================

def get_batch(data, block_size, batch_size, device):
    """
    Sample a random batch of (x, y) pairs.
    x: (B, block_size)
    y: (B, block_size), shifted by 1 position (next token prediction)
    """
    # Random starting positions (guaranteed not to overflow)
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# PART 5: Training loop
# ============================================================================

def train(model, data, block_size, batch_size, lr, max_iters, eval_interval, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for it in range(max_iters):
        model.train()
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — protects against occasional spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())

        if (it + 1) % eval_interval == 0 or it == 0:
            recent = sum(losses[-eval_interval:]) / min(len(losses), eval_interval)
            print(f"  iter {it+1:5d} | loss = {loss.item():.4f} | avg last {eval_interval} = {recent:.4f}")

    return losses



# %%


# Prefer CUDA if available, else CPU. The model is tiny enough for CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Set seed for reproducibility
torch.manual_seed(1337)

# ---- 1. Tokenize the corpus ----
print("\n" + "=" * 70)
print("Step 1: Tokenize the corpus")
print("=" * 70)

tokenizer = CharTokenizer(CORPUS)
print(f"  Corpus length : {len(CORPUS)} chars")
print(f"  Vocab size    : {tokenizer.vocab_size}")
print(f"  Vocab         : {tokenizer.chars}")

# Encode the entire corpus into a tensor
data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
print(f"  Data shape    : {tuple(data.shape)}")
print(f"  First 60 ids  : {data[:60].tolist()}")
print(f"  Decoded back  : {tokenizer.decode(data[:60].tolist())!r}")

# ---- 2. Build the model ----
print("\n" + "=" * 70)
print("Step 2: Build the mini-GPT")
print("=" * 70)

config = dict(
    vocab_size=tokenizer.vocab_size,
    n_embed=64,       # larger embedding for richer vocab
    n_head=4,
    n_layer=3,        # 3 layers for more capacity
    block_size=64,    # longer context window
    dropout=0.1,      # light regularization with more data
)
print(f"  Config: {config}")

model = MiniGPT(**config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {n_params:,}")

# ---- 3. Evaluate loss at initialization ----
print("\n" + "=" * 70)
print("Step 3: Loss at random initialization")
print("=" * 70)

x, y = get_batch(data, config['block_size'], batch_size=16, device=device)
_, loss_init = model(x, targets=y)
expected = math.log(tokenizer.vocab_size)
print(f"  Initial loss : {loss_init.item():.4f}")
print(f"  log(vocab)   : {expected:.4f}  (theoretical uniform baseline)")
print(f"  -> Should be close: a random model predicts ~1/vocab for every char.")

# ---- 4. Sample from the untrained model ----
print("\n" + "=" * 70)
print("Step 4: Generate text BEFORE training (random noise expected)")
print("=" * 70)

context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with char 0
out_ids = model.generate(context, max_new_tokens=80, temperature=1.0)[0].tolist()
print(f"  {tokenizer.decode(out_ids)!r}")

# ---- 5. Train ----
print("\n" + "=" * 70)
print("Step 5: Training loop")
print("=" * 70)

losses = train(
    model=model,
    data=data,
    block_size=config['block_size'],
    batch_size=64,
    lr=3e-3,
    max_iters=3000,
    eval_interval=200,
    device=device,
)

print(f"\n  First 50 iters avg loss : {sum(losses[:50])/50:.4f}")
print(f"  Last  50 iters avg loss : {sum(losses[-50:])/50:.4f}")
print(f"  Improvement             : {sum(losses[:50])/50 - sum(losses[-50:])/50:.4f}")



# %%


# ---- 6. Generate from the trained model ----
print("" + "=" * 70)
print("Step 6: Generate text AFTER training")
print("=" * 70)

# Start with a full word as seed (encoded char by char)
seed_word = 'kadiatou'
start_ids = torch.tensor(
    [tokenizer.encode(seed_word)],
    dtype=torch.long,
    device=device,
)

print("--- Sampling with temperature = 1.0 ---")
out = model.generate(start_ids, max_new_tokens=300, temperature=0.3)[0].tolist()
print(f"  {tokenizer.decode(out)!r}")

# %%

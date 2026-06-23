"""
Day 22 - Vision-Language Models : ViT, CLIP, SigLIP, LLaVA-style projector.

Pure NumPy. No torch, no PIL. Synthetic images = numpy arrays.
Runs in < 30s without GPU.

Sections:
  PART 1 - ViT patch embedding (manual, on a 32x32 image, patch 8x8)
  PART 2 - 1D vs 2D positional encoding (toy grid-matching task)
  PART 3 - CLIP InfoNCE loss vs SigLIP sigmoid loss on the same batch
  PART 4 - LLaVA-style projector : ViT tokens -> LLM space, concat with text
  PART 5 - Token budget analysis : 256/512/1024 with patch 14 vs 16
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# PART 1 - ViT patch embedding (manual)
# ---------------------------------------------------------------------------
# Goal: take a synthetic 32x32 RGB image, cut it into 8x8 patches,
# project each patch linearly into d=64 vectors. Result : a sequence of tokens.
print("=" * 70)
print("PART 1 - ViT patch embedding (32x32 image, 8x8 patches, d=64)")
print("=" * 70)

H, W, C = 32, 32, 3                   # image dims
P = 8                                  # patch size
D = 64                                 # token embedding dim

# Synthetic image: random RGB pixels in [0, 1].
image = np.random.rand(H, W, C).astype(np.float32)
print(f"image.shape = {image.shape}                    (H, W, C)")

# Step 1: cut into patches by reshaping. We want a (num_patches, P*P*C) matrix.
n_h, n_w = H // P, W // P             # 4 x 4 = 16 patches
# View image as (n_h, P, n_w, P, C) then transpose -> (n_h, n_w, P, P, C).
patches = image.reshape(n_h, P, n_w, P, C).transpose(0, 2, 1, 3, 4)
patches = patches.reshape(n_h * n_w, P * P * C)
print(f"patches.shape = {patches.shape}                (num_patches, P*P*C)")

# Step 2: linear projection patch_dim -> D.
W_proj = np.random.randn(P * P * C, D).astype(np.float32) * 0.02
b_proj = np.zeros(D, dtype=np.float32)
tokens = patches @ W_proj + b_proj
print(f"tokens.shape = {tokens.shape}                  (num_patches, D)")

# Step 3: add a [CLS] token in front (a learned vector, here random).
cls_token = np.random.randn(1, D).astype(np.float32) * 0.02
seq = np.concatenate([cls_token, tokens], axis=0)
print(f"seq.shape (with CLS) = {seq.shape}             ({n_h*n_w}+1, D)")
print()

# ---------------------------------------------------------------------------
# PART 2 - 1D vs 2D positional encoding on a grid-matching toy task
# ---------------------------------------------------------------------------
# Toy task: given a sequence of patches from a 4x4 grid, can a linear probe
# predict the (row, col) of each patch from the patch_embed + PE ?
# We compare 1D PE (one vector per index 0..15) vs 2D PE (row vec + col vec).
print("=" * 70)
print("PART 2 - 1D vs 2D positional encoding (grid-matching task)")
print("=" * 70)

GRID = 4                                # 4x4 grid
NPATCH = GRID * GRID                    # 16 patches
PE_DIM = 16                             # tiny dim for clarity

# Same patch content for all positions (we want the model to use ONLY the PE).
# That isolates the contribution of positional encoding.
content = np.random.randn(PE_DIM).astype(np.float32)

# 1D PE: one learned vector per index 0..15.
pe_1d = np.random.randn(NPATCH, PE_DIM).astype(np.float32) * 0.5
seq_1d = content[None, :] + pe_1d        # (16, PE_DIM), each pos = same content + its PE

# 2D PE: PE[r, c] = row_emb[r] + col_emb[c]. Naturally encodes 2D structure.
row_emb = np.random.randn(GRID, PE_DIM).astype(np.float32) * 0.5
col_emb = np.random.randn(GRID, PE_DIM).astype(np.float32) * 0.5
pe_2d = np.zeros((NPATCH, PE_DIM), dtype=np.float32)
for r in range(GRID):
    for c in range(GRID):
        pe_2d[r * GRID + c] = row_emb[r] + col_emb[c]
seq_2d = content[None, :] + pe_2d

# Linear probe: predict (row, col) from each token.
# We solve W via least squares: targets are (row, col) normalized to [0, 1].
targets = np.array([[r / (GRID - 1), c / (GRID - 1)]
                    for r in range(GRID) for c in range(GRID)],
                   dtype=np.float32)        # (16, 2)

def linear_probe_mse(X, y):
    # Closed form least squares: W = (X^T X)^-1 X^T y
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    W, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ W
    return float(np.mean((pred - y) ** 2))

mse_1d = linear_probe_mse(seq_1d, targets)
mse_2d = linear_probe_mse(seq_2d, targets)
print(f"Probe MSE on (row, col) — 1D PE : {mse_1d:.6f}")
print(f"Probe MSE on (row, col) — 2D PE : {mse_2d:.6f}")
# NB: with 16 points in dim 16, the linear probe is *underconstrained*
# (16 unknowns per output dim, 16 equations) — both PEs can hit MSE ~0.
# This printout is a CONCEPTUAL illustration, not a rigorous benchmark
# demonstrating the superiority of 2D PE. The real argument lives at
# scale (variable resolutions, longer sequences, generalization).
print("Both can fit this toy task (underconstrained: 16 points in dim 16),")
print("so MSE ~0 here does NOT prove 2D PE is better. The actual win of 2D PE")
print("appears at scale: variable resolutions, longer sequences, generalization,")
print("because (row, col) is encoded *additively* and disentangled.")
print()

# ---------------------------------------------------------------------------
# PART 3 - CLIP InfoNCE loss vs SigLIP sigmoid loss
# ---------------------------------------------------------------------------
# Setup : N image embeddings and N caption embeddings, paired on the diagonal.
# We compute both losses on the same batch and compare gradients qualitatively.
print("=" * 70)
print("PART 3 - CLIP (InfoNCE / softmax) vs SigLIP (sigmoid) on same batch")
print("=" * 70)

N = 8                                    # batch size
EMB = 32                                 # joint embedding dim

# Random "image" and "caption" features (already L2-normalized).
def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

img = l2norm(np.random.randn(N, EMB).astype(np.float32))
txt = l2norm(np.random.randn(N, EMB).astype(np.float32))

# Cheat a bit: nudge each text to be a noisy copy of its paired image
# so the diagonal has higher similarity (realistic well-trained batch).
txt = l2norm(txt * 0.3 + img * 0.7)

# Similarity matrix S[i, j] = sim(img_i, txt_j). Already cosine since L2-normed.
S = img @ txt.T
print(f"Similarity matrix S.shape = {S.shape}, diag mean = {np.diag(S).mean():.3f}, "
      f"off-diag mean = {(S.sum() - np.trace(S)) / (N * N - N):.3f}")

# --- CLIP InfoNCE loss ---
# Temperature is a learned scalar; we set T = 0.07 (CLIP default).
T = 0.07
logits = S / T

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def nll_along_diagonal(logits):
    # Cross-entropy where target is index i (the diagonal) for each row i.
    probs = softmax(logits, axis=-1)
    diag_probs = np.diag(probs)
    return -np.mean(np.log(diag_probs + 1e-12))

clip_loss = 0.5 * (nll_along_diagonal(logits) + nll_along_diagonal(logits.T))
print(f"CLIP InfoNCE loss (T=0.07)        : {clip_loss:.4f}")

# --- SigLIP sigmoid loss ---
# Each pair (i, j) is a binary classification: is it a true match (1) or not (0)?
# Add a learned bias term to compensate the imbalance (1 positive vs N-1 negatives).
T_sig = 10.0                              # temperature for SigLIP (different convention: scaled up)
b_sig = -10.0                             # bias initialized very negative (imbalance)

logits_sig = S * T_sig + b_sig            # (N, N)
labels = 2 * np.eye(N, dtype=np.float32) - 1  # +1 on diag, -1 off-diag (SigLIP paper)

def log_sigmoid(x):
    # numerically stable log(1 / (1 + exp(-x)))
    return -np.logaddexp(0.0, -x)

# SigLIP loss (per the paper) : -mean log sigmoid(label * logit) over all N*N pairs
siglip_loss = -np.mean(log_sigmoid(labels * logits_sig))
print(f"SigLIP sigmoid loss (T=10, b=-10)  : {siglip_loss:.4f}")

print("Note: CLIP loss requires the *full batch* in memory to softmax-normalize.")
print("      SigLIP loss is a sum of independent binary tasks -> trivially shardable.")
print("      That is why SigLIP scales to batches of 1M+ in practice.")
print()

# ---------------------------------------------------------------------------
# PART 4 - LLaVA-style projector: ViT tokens -> LLM space, concat with text
# ---------------------------------------------------------------------------
# We take the (16, D=64) tokens from PART 1 (without CLS, like LLaVA),
# project them with a 2-layer MLP into the LLM hidden dim (here d_llm=128),
# and concatenate with text tokens to get the final LLM context.
print("=" * 70)
print("PART 4 - LLaVA-style projector : ViT tokens -> LLM space + concat")
print("=" * 70)

D_VIT = D                                 # 64 (from PART 1)
D_LLM = 128                               # toy LLM hidden dim
N_TEXT = 10                               # 10 text tokens (e.g. "Describe this image:")

# Use the patch tokens from PART 1 (drop the CLS, like LLaVA does).
vit_tokens = tokens                        # (16, 64)
print(f"vit_tokens.shape (from ViT)        = {vit_tokens.shape}")

# 2-layer MLP projector: D_VIT -> D_LLM -> D_LLM (GELU in between).
W1 = np.random.randn(D_VIT, D_LLM).astype(np.float32) * 0.02
b1 = np.zeros(D_LLM, dtype=np.float32)
W2 = np.random.randn(D_LLM, D_LLM).astype(np.float32) * 0.02
b2 = np.zeros(D_LLM, dtype=np.float32)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

projected_visual = gelu(vit_tokens @ W1 + b1) @ W2 + b2
print(f"projected_visual.shape (in LLM space) = {projected_visual.shape}")

# Synthetic text token embeddings (would normally come from LLM token_embed table).
text_tokens = np.random.randn(N_TEXT, D_LLM).astype(np.float32) * 0.02
print(f"text_tokens.shape                  = {text_tokens.shape}")

# Final LLM context: [text_prefix, image_tokens, text_suffix]
# Here we stick the image tokens between two halves of the text.
prefix = text_tokens[:5]
suffix = text_tokens[5:]
llm_context = np.concatenate([prefix, projected_visual, suffix], axis=0)
print(f"llm_context.shape (final input to LLM) = {llm_context.shape}")
print(f"  -> {prefix.shape[0]} text + {projected_visual.shape[0]} image + "
      f"{suffix.shape[0]} text tokens")
print()

# ---------------------------------------------------------------------------
# PART 5 - Token budget analysis : 256/512/1024 with patch 14 vs 16
# ---------------------------------------------------------------------------
# How many visual tokens does an image consume at different resolutions?
# That answers : why is 1024x1024 expensive in API calls?
print("=" * 70)
print("PART 5 - Image token budget (patch 14 vs 16)")
print("=" * 70)
resolutions = [224, 256, 336, 512, 768, 1024]
patches = [14, 16, 32]

header = "Resolution".ljust(12) + " | " + " | ".join(f"P={p:<3}".ljust(10) for p in patches)
print(header)
print("-" * len(header))
for res in resolutions:
    cells = []
    for p in patches:
        # We compute floor(res/p) per side to mimic the grid (real impls may pad).
        n = (res // p) ** 2
        cells.append(f"{n:>5} tok".ljust(10))
    print(f"{res}x{res}".ljust(12) + " | " + " | ".join(cells))

print()
# Sanity check : 1024 / 14 = 73, 73*73 = 5329. Show the cost in compute terms.
N_IMG = (1024 // 14) ** 2
print(f"For a 1024x1024 image with patch 14 -> {N_IMG} tokens.")
print(f"Self-attention cost is O(N^2) = {N_IMG ** 2:,} pairwise operations per layer.")
print("That is the heaviest part of a VLM forward pass on a single high-res image.")
print()
print("Cost-optimization tools VLM designers reach for:")
print("  * Pooling 2x2 after ViT  -> divide tokens by 4")
print("  * Perceiver Resampler    -> fixed K tokens regardless of input")
print("  * Q-Former (BLIP-2)      -> learned compression N -> K")
print("  * AnyRes / dynamic res   -> grid of low-res tiles + global thumbnail")
print()
print("Done.")

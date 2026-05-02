"""
J19 — VLA introduction: a tiny pedagogical Vision-Language-Action policy.

What this file demonstrates (in ~300 lines, no heavy deps):
  - A toy "grid-world image + instruction" dataset: 8x8 grayscale images with
    an arrow on a random cell, and a textual command "move arrow up/down/left/right".
  - A tiny vision encoder (Conv -> flatten -> linear).
  - A tiny text encoder (token id embedding + mean pooling) — no HF transformers.
  - A causal transformer policy that mixes vision + text tokens and predicts
    a sequence of DISCRETIZED action tokens (RT-1 / RT-2 / OpenVLA style).
  - Action discretization via uniform binning (N_BINS bins per dim).
  - Training loop (cross-entropy over action bins) and a quick eval that
    reports per-token accuracy.

Why this is a *VLA* in miniature:
  Inputs:  image (vision) + instruction (language)
  Outputs: action token sequence (discretized continuous deltas)
  Training: cross-entropy on discrete action tokens, exactly like RT-1 / OpenVLA
            treat 256-bin per-dim discretization (REFERENCES.md #13).

This is intentionally tiny — designed to run on CPU in < 1 minute.

References:
  - REFERENCES.md #17 (Octo, RSS 2024) — transformer policy on multi-embodiment data
  - REFERENCES.md #13 (OpenVLA, 2024) — action discretization details, RT-2 lineage
"""
# requires: torch>=2.0

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 1. Reproducibility & device
# ---------------------------------------------------------------------------

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 2. Synthetic dataset
# ---------------------------------------------------------------------------
# A scene = 8x8 grayscale image with a single "arrow" pixel block. The
# instruction is one of {"up", "down", "left", "right"}. The ground-truth
# action is the 2D delta (dx, dy) the arrow should move by.
#
# Rationale: this mirrors a VLA in miniature — instruction + visual context
# fully determine the action. The model must read both modalities to succeed.

GRID = 8
DIRECTIONS = ["up", "down", "left", "right"]
DIR_TO_DELTA = {
    "up":    (0, -1),
    "down":  (0, +1),
    "left":  (-1, 0),
    "right": (+1, 0),
}

# Tiny vocab — pad/cls + the 4 direction words. We do NOT use a real tokenizer.
VOCAB = {"<pad>": 0, "<cls>": 1, "up": 2, "down": 3, "left": 4, "right": 5}
PAD_ID = VOCAB["<pad>"]
CLS_ID = VOCAB["<cls>"]
MAX_TEXT_LEN = 4  # ample for our toy command


def encode_text(direction: str) -> torch.Tensor:
    """Encode a direction word into a fixed-length token id tensor (cls + word + pads)."""
    ids = [CLS_ID, VOCAB[direction]] + [PAD_ID] * (MAX_TEXT_LEN - 2)
    return torch.tensor(ids, dtype=torch.long)


def render_scene(arrow_x: int, arrow_y: int) -> torch.Tensor:
    """Draw an 8x8 image with a single "arrow" at (arrow_x, arrow_y)."""
    img = torch.zeros(1, GRID, GRID)  # 1 channel
    img[0, arrow_y, arrow_x] = 1.0
    # Add a small visual hint of orientation (a 1-pixel "shaft" upward)
    if arrow_y - 1 >= 0:
        img[0, arrow_y - 1, arrow_x] = 0.5
    return img


# ---- Action discretization (RT-1 / OpenVLA style) -----------------------
# Each action dim is binned into N_BINS = 16 bins covering [-1, +1].
# In our toy task only deltas {-1, 0, +1} occur, but we keep a coarse
# binning to *show* the discretization machinery.

N_BINS = 16
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


def action_to_tokens(delta: Tuple[int, int]) -> torch.Tensor:
    """Continuous (dx, dy) -> two discrete bin tokens, like RT-1/OpenVLA."""
    dx, dy = delta
    out = []
    for v in (dx, dy):
        # Clip then map to [0, N_BINS-1] uniformly
        v = max(min(v, ACTION_HIGH), ACTION_LOW)
        scaled = (v - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW)  # in [0, 1]
        bin_id = int(round(scaled * (N_BINS - 1)))
        out.append(bin_id)
    return torch.tensor(out, dtype=torch.long)


def tokens_to_action(tokens: torch.Tensor) -> Tuple[float, float]:
    """Inverse of action_to_tokens — used at inference to decode a continuous action."""
    out = []
    for bin_id in tokens.tolist():
        scaled = bin_id / (N_BINS - 1)
        v = ACTION_LOW + scaled * (ACTION_HIGH - ACTION_LOW)
        out.append(v)
    return out[0], out[1]


@dataclass
class Sample:
    image: torch.Tensor      # (1, 8, 8)
    text_ids: torch.Tensor   # (MAX_TEXT_LEN,)
    action_tokens: torch.Tensor  # (2,) -> bin id per dim (dx, dy)


class ArrowDataset(Dataset):
    """Random arrows + instructions. Action = direction delta (in cells)."""

    def __init__(self, n: int = 2000, seed: int = SEED):
        rng = random.Random(seed)
        self.samples: List[Sample] = []
        for _ in range(n):
            x = rng.randrange(GRID)
            y = rng.randrange(GRID)
            direction = rng.choice(DIRECTIONS)
            dx, dy = DIR_TO_DELTA[direction]
            self.samples.append(Sample(
                image=render_scene(x, y),
                text_ids=encode_text(direction),
                action_tokens=action_to_tokens((dx, dy)),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.image, s.text_ids, s.action_tokens


# ---------------------------------------------------------------------------
# 3. Mini VLA model
# ---------------------------------------------------------------------------
# Vision encoder: small Conv -> tokens of dim D (analogous to ViT patches).
# Text encoder:   embedding + small linear projection (like a tiny T5 stub).
# Backbone:       a few self-attention layers over (vision_tokens + text_tokens).
# Action head:    one classification head per action dim (over N_BINS).

D_MODEL = 64       # transformer hidden size
N_HEADS = 4        # attention heads
N_LAYERS = 2       # transformer layers
N_VISION_TOKENS = 4  # number of "patch" tokens we emit from the conv encoder
N_ACTION_DIMS = 2    # (dx, dy)


class VisionEncoder(nn.Module):
    """Tiny conv encoder that emits N_VISION_TOKENS tokens of dim D_MODEL."""

    def __init__(self, d_model: int = D_MODEL, n_tokens: int = N_VISION_TOKENS):
        super().__init__()
        self.n_tokens = n_tokens
        # 1 -> 16 -> 32 channels, then pool to 2x2 -> 4 spatial positions = 4 tokens.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # -> (B, 32, 2, 2)
        )
        self.proj = nn.Linear(32, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 8, 8)
        feat = self.conv(x)               # (B, 32, 2, 2)
        feat = feat.flatten(2).transpose(1, 2)  # (B, 4, 32)
        return self.proj(feat)            # (B, 4, D_MODEL)


class TextEncoder(nn.Module):
    """Token-id embedding + linear projection (no positional info needed at this scale)."""

    def __init__(self, vocab_size: int = len(VOCAB), d_model: int = D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, MAX_TEXT_LEN) -> (B, MAX_TEXT_LEN, D_MODEL)
        return self.emb(ids)


class TinyVLA(nn.Module):
    """A tiny Vision-Language-Action policy with discrete action heads."""

    def __init__(self):
        super().__init__()
        self.vision_enc = VisionEncoder()
        self.text_enc = TextEncoder()

        # Learnable type embeddings to distinguish modalities (vision vs text).
        self.type_vision = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.type_text = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # One classification head per action dim, each predicting over N_BINS.
        self.action_heads = nn.ModuleList(
            [nn.Linear(D_MODEL, N_BINS) for _ in range(N_ACTION_DIMS)]
        )

        # A learnable "action readout" token, prepended to the sequence.
        # We will use its final hidden state to predict each action dim.
        self.readout = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        # Init
        for p in [self.type_vision, self.type_text, self.readout]:
            nn.init.normal_(p, std=0.02)

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            image:    (B, 1, 8, 8)
            text_ids: (B, MAX_TEXT_LEN)
        Returns:
            list of length N_ACTION_DIMS, each a (B, N_BINS) tensor of logits.
        """
        B = image.size(0)
        v = self.vision_enc(image) + self.type_vision   # (B, 4, D)
        t = self.text_enc(text_ids) + self.type_text    # (B, T, D)
        readout = self.readout.expand(B, -1, -1)        # (B, 1, D)

        # Concat: [readout, vision tokens, text tokens]
        x = torch.cat([readout, v, t], dim=1)
        h = self.backbone(x)                            # (B, 1+4+T, D)

        h_readout = h[:, 0, :]                          # (B, D)
        # Each action head independently produces logits over N_BINS bins.
        return [head(h_readout) for head in self.action_heads]


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: TinyVLA, loader: DataLoader, opt: torch.optim.Optimizer) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for image, text_ids, action_tokens in loader:
        image = image.to(DEVICE)
        text_ids = text_ids.to(DEVICE)
        action_tokens = action_tokens.to(DEVICE)  # (B, 2)

        logits_per_dim = model(image, text_ids)  # list of (B, N_BINS)
        loss = 0.0
        for d_idx, logits in enumerate(logits_per_dim):
            loss = loss + F.cross_entropy(logits, action_tokens[:, d_idx])
        loss = loss / N_ACTION_DIMS

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = image.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: TinyVLA, loader: DataLoader) -> Tuple[float, float]:
    """Returns (mean_loss, per-dim accuracy averaged over both action dims)."""
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    for image, text_ids, action_tokens in loader:
        image = image.to(DEVICE)
        text_ids = text_ids.to(DEVICE)
        action_tokens = action_tokens.to(DEVICE)

        logits_per_dim = model(image, text_ids)
        loss = 0.0
        for d_idx, logits in enumerate(logits_per_dim):
            loss = loss + F.cross_entropy(logits, action_tokens[:, d_idx])
            pred = logits.argmax(dim=-1)
            correct += (pred == action_tokens[:, d_idx]).sum().item()
        loss = loss / N_ACTION_DIMS

        bs = image.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1), correct / (n * N_ACTION_DIMS)


# ---------------------------------------------------------------------------
# 5. Quick demo: train ~5 epochs on CPU and decode one sample
# ---------------------------------------------------------------------------

def main() -> None:
    train_ds = ArrowDataset(n=2000, seed=SEED)
    val_ds = ArrowDataset(n=400, seed=SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = TinyVLA().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[J19] TinyVLA params: {n_params:,} | device: {DEVICE}")
    print(f"[J19] Action discretization: {N_BINS} bins per dim "
          f"(range [{ACTION_LOW}, {ACTION_HIGH}])")

    for epoch in range(1, 6):
        train_loss = train_one_epoch(model, train_loader, opt)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Decode one sample end-to-end (image + "left" -> tokens -> continuous action).
    image = render_scene(arrow_x=4, arrow_y=4).unsqueeze(0).to(DEVICE)
    text_ids = encode_text("left").unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_per_dim = model(image, text_ids)
        bin_ids = torch.tensor([logits.argmax(dim=-1).item() for logits in logits_per_dim])
    dx, dy = tokens_to_action(bin_ids)
    print(f"[J19] sample inference: instruction='left' -> bins={bin_ids.tolist()} "
          f"-> continuous action (dx={dx:+.3f}, dy={dy:+.3f})")
    print("[J19] Expected sign: dx < 0, dy ~= 0 for 'left'.")


if __name__ == "__main__":
    main()

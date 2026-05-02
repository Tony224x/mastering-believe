"""
J19 VLA introduction — consolidated solutions for easy / medium / hard.

Run all solutions:    python 19-vla-introduction.py
Run only one part:    python 19-vla-introduction.py --part easy|medium|hard

References:
  - REFERENCES.md #17 (Octo, RSS 2024) — multi-step chunking + diffusion head rationale
  - REFERENCES.md #13 (OpenVLA, 2024) — 256-bin per-dim discretization, RT-2 lineage
"""
# requires: torch>=2.0

from __future__ import annotations

import argparse
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# EASY — Action discretization helpers
# ===========================================================================

def discretize(value: float, n_bins: int, lo: float, hi: float) -> int:
    """Map a continuous value clipped to [lo, hi] onto an integer bin in [0, n_bins-1]."""
    v = max(min(value, hi), lo)
    scaled = (v - lo) / (hi - lo)              # in [0, 1]
    return int(round(scaled * (n_bins - 1)))


def undiscretize(bin_id: int, n_bins: int, lo: float, hi: float) -> float:
    """Inverse mapping — returns the bin center value."""
    bin_id = max(0, min(bin_id, n_bins - 1))
    scaled = bin_id / (n_bins - 1)
    return lo + scaled * (hi - lo)


def solve_easy() -> None:
    print("\n=== EASY: action discretization round-trip ===")
    lo, hi = -0.05, 0.05

    # Theoretical max quantization error for uniform binning.
    for n_bins in (16, 256):
        max_err = (hi - lo) / (2 * (n_bins - 1))
        resolution_mm = (hi - lo) * 1000.0 / (n_bins - 1)
        print(f"  n_bins={n_bins:3d} | max quantization error = {max_err*1000:.4f} mm "
              f"| resolution = {resolution_mm:.4f} mm/bin")

    test_values = [-0.05, -0.0123, 0.0, 0.0349, 0.05]
    print(f"\n  Round-trip check with n_bins=256, range=[{lo}, {hi}]:")
    worst = 0.0
    for v in test_values:
        b = discretize(v, 256, lo, hi)
        v_back = undiscretize(b, 256, lo, hi)
        err = abs(v - v_back)
        worst = max(worst, err)
        print(f"    v={v:+.4f}  ->  bin={b:3d}  ->  v_back={v_back:+.4f}  | err={err*1000:.4f} mm")
    print(f"  worst observed error = {worst*1000:.4f} mm "
          f"(theoretical bound: {(hi-lo)*1000/(2*255):.4f} mm)")
    print("  Takeaway: 256 bins on +/- 5 cm gives sub-mm precision; for surgery we'd need more.")


# ===========================================================================
# MEDIUM — Multi-step chunking VLA (Octo-flavored, but with discrete action tokens)
# ===========================================================================

GRID = 8
DIRECTIONS = ["up", "down", "left", "right"]
DIR_TO_DELTA = {"up": (0, -1), "down": (0, +1), "left": (-1, 0), "right": (+1, 0)}
VOCAB = {"<pad>": 0, "<cls>": 1, "up": 2, "down": 3, "left": 4, "right": 5, "go": 6}
PAD_ID = VOCAB["<pad>"]
CLS_ID = VOCAB["<cls>"]
MAX_TEXT_LEN = 4

N_BINS = 16
ACTION_LOW, ACTION_HIGH = -1.0, 1.0
N_ACTION_DIMS = 2


def encode_text(word: str) -> torch.Tensor:
    ids = [CLS_ID, VOCAB[word]] + [PAD_ID] * (MAX_TEXT_LEN - 2)
    return torch.tensor(ids, dtype=torch.long)


def render_scene(arrow_x: int, arrow_y: int) -> torch.Tensor:
    img = torch.zeros(1, GRID, GRID)
    img[0, arrow_y, arrow_x] = 1.0
    if arrow_y - 1 >= 0:
        img[0, arrow_y - 1, arrow_x] = 0.5
    return img


def discretize_action(v: float) -> int:
    return discretize(v, N_BINS, ACTION_LOW, ACTION_HIGH)


def undiscretize_action(bin_id: int) -> float:
    return undiscretize(bin_id, N_BINS, ACTION_LOW, ACTION_HIGH)


class ChunkedArrowDataset(Dataset):
    """K-step trajectory dataset. At each sample: image + word -> K (dx, dy) pairs."""

    def __init__(self, n: int = 2000, k: int = 4, seed: int = SEED, ambiguous: bool = False):
        rng = random.Random(seed)
        self.k = k
        self.samples = []
        for _ in range(n):
            x = rng.randrange(GRID)
            y = rng.randrange(GRID)
            if ambiguous:
                word = "go"
                # Each sample randomly picks one of the 4 directions (multimodal target).
                direction = rng.choice(DIRECTIONS)
            else:
                direction = rng.choice(DIRECTIONS)
                word = direction
            dx_unit, dy_unit = DIR_TO_DELTA[direction]

            steps: List[Tuple[int, int]] = []
            cx, cy = x, y
            for _ in range(k):
                nx = max(0, min(GRID - 1, cx + dx_unit))
                ny = max(0, min(GRID - 1, cy + dy_unit))
                steps.append((nx - cx, ny - cy))  # actual delta after clipping
                cx, cy = nx, ny

            # Discretize each step's (dx, dy)
            action_tokens = torch.tensor(
                [[discretize_action(dx), discretize_action(dy)] for dx, dy in steps],
                dtype=torch.long,
            )  # (k, 2)
            action_continuous = torch.tensor(steps, dtype=torch.float32)  # (k, 2)

            self.samples.append({
                "image": render_scene(x, y),
                "text_ids": encode_text(word),
                "action_tokens": action_tokens,         # (K, 2)
                "action_continuous": action_continuous, # (K, 2)
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["image"], s["text_ids"], s["action_tokens"], s["action_continuous"]


# --- Model -----------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2


class VisionEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(32, d_model)

    def forward(self, x):
        f = self.conv(x).flatten(2).transpose(1, 2)  # (B, 4, 32)
        return self.proj(f)


class TextEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d_model, padding_idx=PAD_ID)

    def forward(self, ids):
        return self.emb(ids)


class ChunkedTinyVLA(nn.Module):
    """Multi-step VLA. Two heads:
       - tokenization head: classification logits of shape (B, K, N_ACTION_DIMS, N_BINS)
       - regression head:   continuous output of shape (B, K, N_ACTION_DIMS)
       Use whichever the loss expects.
    """

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.vision_enc = VisionEncoder()
        self.text_enc = TextEncoder()
        self.type_vision = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.type_text = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.readout = nn.Parameter(torch.zeros(1, 1, D_MODEL))

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=N_LAYERS)

        # Tokenization head -> logits for (K, N_ACTION_DIMS, N_BINS)
        self.head_tok = nn.Linear(D_MODEL, k * N_ACTION_DIMS * N_BINS)
        # Regression head -> (K, N_ACTION_DIMS)
        self.head_reg = nn.Linear(D_MODEL, k * N_ACTION_DIMS)

        for p in [self.type_vision, self.type_text, self.readout]:
            nn.init.normal_(p, std=0.02)

    def encode(self, image, text_ids):
        B = image.size(0)
        v = self.vision_enc(image) + self.type_vision
        t = self.text_enc(text_ids) + self.type_text
        readout = self.readout.expand(B, -1, -1)
        h = self.backbone(torch.cat([readout, v, t], dim=1))
        return h[:, 0, :]  # (B, D)

    def forward_tok(self, image, text_ids):
        h = self.encode(image, text_ids)
        logits = self.head_tok(h).view(-1, self.k, N_ACTION_DIMS, N_BINS)
        return logits

    def forward_reg(self, image, text_ids):
        h = self.encode(image, text_ids)
        return self.head_reg(h).view(-1, self.k, N_ACTION_DIMS)


def train_chunked_tok(model, loader, opt, epochs=5, log=True):
    model.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for image, text_ids, act_tok, _ in loader:
            image = image.to(DEVICE); text_ids = text_ids.to(DEVICE); act_tok = act_tok.to(DEVICE)
            logits = model.forward_tok(image, text_ids)  # (B, K, D, N_BINS)
            B, K, D, _ = logits.shape
            loss = F.cross_entropy(logits.view(-1, N_BINS), act_tok.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * B; n += B
        if log:
            print(f"  [tok] epoch {ep} loss={total/max(n,1):.4f}")


def train_chunked_reg(model, loader, opt, epochs=5, log=True):
    model.train()
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for image, text_ids, _, act_cont in loader:
            image = image.to(DEVICE); text_ids = text_ids.to(DEVICE); act_cont = act_cont.to(DEVICE)
            pred = model.forward_reg(image, text_ids)  # (B, K, 2)
            loss = F.mse_loss(pred, act_cont)
            opt.zero_grad(); loss.backward(); opt.step()
            B = image.size(0); total += loss.item() * B; n += B
        if log:
            print(f"  [reg] epoch {ep} loss={total/max(n,1):.4f}")


@torch.no_grad()
def eval_tok_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for image, text_ids, act_tok, _ in loader:
        image = image.to(DEVICE); text_ids = text_ids.to(DEVICE); act_tok = act_tok.to(DEVICE)
        logits = model.forward_tok(image, text_ids)
        pred = logits.argmax(dim=-1)  # (B, K, D)
        correct += (pred == act_tok).sum().item()
        total += act_tok.numel()
    return correct / max(total, 1)


def solve_medium() -> None:
    print("\n=== MEDIUM: K=4 chunked tokenization VLA ===")
    K = 4
    train_ds = ChunkedArrowDataset(n=2000, k=K, seed=SEED, ambiguous=False)
    val_ds = ChunkedArrowDataset(n=400, k=K, seed=SEED + 1, ambiguous=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = ChunkedTinyVLA(k=K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_chunked_tok(model, train_loader, opt, epochs=5)
    acc = eval_tok_accuracy(model, val_loader)
    print(f"  per-token val accuracy = {acc:.3f}  (expected > 0.85)")

    # Decode one sample fully
    image = render_scene(4, 4).unsqueeze(0).to(DEVICE)
    text_ids = encode_text("up").unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model.forward_tok(image, text_ids)  # (1, K, 2, N_BINS)
        bins = logits.argmax(dim=-1)[0]              # (K, 2)
    decoded = [(undiscretize_action(int(b[0])), undiscretize_action(int(b[1]))) for b in bins]
    print(f"  predicted trajectory for 'up' starting at (4,4):")
    for i, (dx, dy) in enumerate(decoded):
        print(f"    step {i}: dx={dx:+.3f} dy={dy:+.3f}  (expected ~ ( 0.0, -1.0 ))")


# ===========================================================================
# HARD — Tokenization vs regression on an ambiguous "go" instruction
# ===========================================================================

def direction_from_delta(dx: float, dy: float) -> str:
    """Snap a continuous (dx, dy) onto the closest of the 4 cardinal directions."""
    candidates = {name: (cx, cy) for name, (cx, cy) in DIR_TO_DELTA.items()}
    best, best_dist = None, float("inf")
    for name, (cx, cy) in candidates.items():
        d = (dx - cx) ** 2 + (dy - cy) ** 2
        if d < best_dist:
            best_dist = d; best = name
    return best


def solve_hard() -> None:
    print("\n=== HARD: tokenization vs regression on ambiguous 'go' ===")
    K = 1  # single-step is enough to expose mode-collapse cleanly

    # --- Variant A: tokenization on non-ambiguous data
    nonamb_train = ChunkedArrowDataset(n=2000, k=K, seed=SEED, ambiguous=False)
    nonamb_val = ChunkedArrowDataset(n=400, k=K, seed=SEED + 1, ambiguous=False)
    loader_train = DataLoader(nonamb_train, batch_size=64, shuffle=True)
    loader_val = DataLoader(nonamb_val, batch_size=128, shuffle=False)

    print("  -- Variant A: tokenization, non-ambiguous --")
    model_A = ChunkedTinyVLA(k=K).to(DEVICE)
    opt = torch.optim.AdamW(model_A.parameters(), lr=3e-3)
    train_chunked_tok(model_A, loader_train, opt, epochs=5, log=False)
    acc_A = eval_tok_accuracy(model_A, loader_val)
    print(f"    per-token accuracy = {acc_A:.3f}")

    # --- Variant B: regression on non-ambiguous data
    print("  -- Variant B: regression (MSE), non-ambiguous --")
    model_B = ChunkedTinyVLA(k=K).to(DEVICE)
    opt = torch.optim.AdamW(model_B.parameters(), lr=3e-3)
    train_chunked_reg(model_B, loader_train, opt, epochs=5, log=False)
    # Direction-recovery accuracy from continuous prediction
    correct, total = 0, 0
    with torch.no_grad():
        model_B.eval()
        for image, text_ids, _, act_cont in loader_val:
            image = image.to(DEVICE); text_ids = text_ids.to(DEVICE); act_cont = act_cont.to(DEVICE)
            pred = model_B.forward_reg(image, text_ids)[:, 0, :]  # (B, 2)
            true = act_cont[:, 0, :]
            for p, t in zip(pred.cpu().tolist(), true.cpu().tolist()):
                if direction_from_delta(*p) == direction_from_delta(*t):
                    correct += 1
                total += 1
    print(f"    direction-recovery accuracy = {correct/max(total,1):.3f}")

    # --- Variant C: ambiguous "go" target (multimodal)
    print("  -- Variant C: ambiguous 'go' (multimodal target) --")
    amb_train = ChunkedArrowDataset(n=4000, k=K, seed=SEED + 7, ambiguous=True)
    amb_val = ChunkedArrowDataset(n=200, k=K, seed=SEED + 8, ambiguous=True)
    loader_amb_train = DataLoader(amb_train, batch_size=64, shuffle=True)
    loader_amb_val = DataLoader(amb_val, batch_size=128, shuffle=False)

    # Tokenization on ambiguous data
    model_C_tok = ChunkedTinyVLA(k=K).to(DEVICE)
    opt = torch.optim.AdamW(model_C_tok.parameters(), lr=3e-3)
    train_chunked_tok(model_C_tok, loader_amb_train, opt, epochs=8, log=False)

    # Regression on ambiguous data
    model_C_reg = ChunkedTinyVLA(k=K).to(DEVICE)
    opt = torch.optim.AdamW(model_C_reg.parameters(), lr=3e-3)
    train_chunked_reg(model_C_reg, loader_amb_train, opt, epochs=8, log=False)

    # Inspect 100 inferences with instruction "go" — what directions do they cover?
    image = render_scene(4, 4).unsqueeze(0).to(DEVICE)
    text_ids = encode_text("go").unsqueeze(0).to(DEVICE)

    # Regression: deterministic, so 1 forward suffices — but we average over varied images.
    reg_dirs = []
    with torch.no_grad():
        model_C_reg.eval()
        for x in range(GRID):
            for y in range(GRID):
                im = render_scene(x, y).unsqueeze(0).to(DEVICE)
                pred = model_C_reg.forward_reg(im, text_ids)[0, 0].cpu().tolist()
                reg_dirs.append(direction_from_delta(*pred))
    reg_unique = sorted(set(reg_dirs))
    # Mean predicted delta over all 64 grid positions
    with torch.no_grad():
        sum_dx, sum_dy, count = 0.0, 0.0, 0
        for x in range(GRID):
            for y in range(GRID):
                im = render_scene(x, y).unsqueeze(0).to(DEVICE)
                p = model_C_reg.forward_reg(im, text_ids)[0, 0].cpu().tolist()
                sum_dx += p[0]; sum_dy += p[1]; count += 1
        mean_pred = (sum_dx / count, sum_dy / count)
    print(f"    [regression on ambiguous] mean predicted delta = "
          f"({mean_pred[0]:+.3f}, {mean_pred[1]:+.3f}) "
          f"(expect ~ (0, 0): MSE collapses to the mean of the 4 modes)")
    print(f"    [regression on ambiguous] unique directions covered across grid = {reg_unique}")

    # Tokenization: sample with temperature to expose multimodality
    tok_dirs = []
    with torch.no_grad():
        model_C_tok.eval()
        for _ in range(100):
            logits = model_C_tok.forward_tok(image, text_ids)[0, 0]  # (2, N_BINS)
            probs = F.softmax(logits / 1.0, dim=-1)  # temperature 1
            sampled_bins = [torch.multinomial(probs[d], num_samples=1).item() for d in range(N_ACTION_DIMS)]
            dxdy = (undiscretize_action(sampled_bins[0]), undiscretize_action(sampled_bins[1]))
            tok_dirs.append(direction_from_delta(*dxdy))
    counts = {d: tok_dirs.count(d) for d in DIRECTIONS}
    print(f"    [tokenization on ambiguous, sampled] direction counts over 100 draws = {counts}")
    print(f"    [tokenization on ambiguous] unique directions covered = "
          f"{sorted(set(tok_dirs))}")
    print("    Takeaway: tokenization preserves multimodality (softmax holds 4 modes), "
          "regression collapses to the mean.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["easy", "medium", "hard", "all"], default="all")
    args = parser.parse_args()

    if args.part in ("easy", "all"):
        solve_easy()
    if args.part in ("medium", "all"):
        solve_medium()
    if args.part in ("hard", "all"):
        solve_hard()


if __name__ == "__main__":
    main()

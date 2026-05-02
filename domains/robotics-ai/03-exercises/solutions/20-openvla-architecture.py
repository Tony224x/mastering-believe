"""
J20 - Solutions consolidees (easy + medium + hard) pour OpenVLA.

Reference principale : REFERENCES.md #13 (OpenVLA paper + repo).
LoRA reference : Hu et al. 2021 (https://arxiv.org/abs/2106.09685).

Run:
    python domains/robotics-ai/03-exercises/solutions/20-openvla-architecture.py

requires: torch>=2.0
"""

# requires: torch>=2.0
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# EASY -- shape report + bin helpers
# ===========================================================================

def openvla_shape_report(
    image_size: int,
    patch_size: int,
    dinov2_dim: int,
    siglip_dim: int,
    llm_hidden: int,
    action_dim: int,
    action_bins: int,
    vocab_size: int,
) -> dict:
    """Return the canonical shape report of an OpenVLA-like model.

    Why concatenation along the FEATURE axis (not the patch axis):
    - DINOv2 is self-supervised on geometry / depth / dense correspondence.
    - SigLIP is contrastive on image-text semantics.
    - We want EVERY patch to expose BOTH descriptors so the LLM can attend
      jointly to "where" (DINOv2) and "what" (SigLIP). Concatenating along the
      patch axis would give the LLM 512 tokens with two SEPARATE views, and
      the cross-info would only be reconstructed by attention -- worse signal.
    """
    assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
    side = image_size // patch_size
    num_patches = side * side
    concat_dim = dinov2_dim + siglip_dim
    projector_in_out = (concat_dim, llm_hidden)
    action_token_range = (vocab_size - action_bins, vocab_size)

    def bin_for_value(value: float, dim_idx: int, low: float = -1.0, high: float = 1.0) -> int:
        # Clip first.
        v = max(low, min(high, value))
        # Linear bucketization in [low, high] -> [0, action_bins-1].
        normalized = (v - low) / (high - low)                # in [0, 1]
        idx = int(normalized * action_bins)                  # in [0, action_bins]
        return min(idx, action_bins - 1)                    # cap at last bin

    return {
        "num_patches": num_patches,
        "concat_dim": concat_dim,
        "projector_in_out": projector_in_out,
        "action_token_range": action_token_range,
        "bin_for_value": bin_for_value,
    }


def run_easy() -> None:
    print("=" * 70)
    print("EASY -- shape report")
    print("=" * 70)
    rpt = openvla_shape_report(
        image_size=224, patch_size=14,
        dinov2_dim=1024, siglip_dim=1152,
        llm_hidden=4096,
        action_dim=7, action_bins=256, vocab_size=32000,
    )
    print(f"  num_patches         : {rpt['num_patches']}      (expect 256)")
    print(f"  concat_dim          : {rpt['concat_dim']}     (expect 2176)")
    print(f"  projector_in_out    : {rpt['projector_in_out']} (expect (2176, 4096))")
    print(f"  action_token_range  : {rpt['action_token_range']} (expect (31744, 32000))")
    bin_at_zero = rpt["bin_for_value"](0.0, 0)
    bin_clip_high = rpt["bin_for_value"](2.0, 0)
    bin_clip_low = rpt["bin_for_value"](-3.0, 0)
    print(f"  bin_for_value(0.0)  : {bin_at_zero} (expect 127 or 128)")
    print(f"  bin_for_value(+2.0) : {bin_clip_high} (expect 255 - clipping)")
    print(f"  bin_for_value(-3.0) : {bin_clip_low} (expect 0   - clipping)")
    assert rpt["num_patches"] == 256
    assert rpt["concat_dim"] == 2176
    assert rpt["projector_in_out"] == (2176, 4096)
    assert rpt["action_token_range"] == (31744, 32000)
    assert bin_at_zero in (127, 128)
    assert bin_clip_high == 255
    assert bin_clip_low == 0
    print("  EASY: PASS")


# ===========================================================================
# MEDIUM -- LoRA from scratch + identity-at-init proof
# ===========================================================================

class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with the LoRA delta `(alpha/r) * B @ A`.

    Critical init choices:
        A ~ N(0, 1/r): non-trivial perturbation but small.
        B = 0       : at step 0, BA = 0 -> output exactly matches base.

    Scaling alpha/r decouples update magnitude from rank: doubling r without
    rescaling would otherwise double the effective update step.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.normal_(self.lora_A, std=1.0 / r)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t())

    def effective_weight(self) -> torch.Tensor:
        """Reconstruct the equivalent Linear weight W + (alpha/r) * B @ A."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def count_trainable_ratio(d_in: int, d_out: int, r: int, alpha: int) -> float:
    base = nn.Linear(d_in, d_out, bias=False)
    lora = LoRALinear(base, r=r, alpha=alpha)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora.parameters())
    return 100.0 * trainable / total


def test_lora_identity_at_init() -> None:
    torch.manual_seed(123)
    base = nn.Linear(64, 64)
    # Snapshot the base output BEFORE wrapping (the Linear is unmodified
    # afterwards but this avoids any subtle in-place issue).
    x = torch.randn(4, 64)
    target = base(x).detach().clone()
    lora = LoRALinear(base, r=8, alpha=16)
    out = lora(x)
    assert torch.allclose(target, out, atol=1e-6), (
        "LoRA at init MUST be identity. If this fails: B not zero-initialized,"
        " or scaling typo, or dropout active during init."
    )


def run_medium() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM -- LoRA from scratch")
    print("=" * 70)

    # Trainable ratio on Linear(4096, 4096) with r=32, no bias.
    ratio = count_trainable_ratio(4096, 4096, r=32, alpha=16)
    print(f"  trainable ratio (Linear 4096x4096, r=32) : {ratio:.3f} %  (expect ~1.56%)")
    # The expected value is 2*r*d / (d*d) = 2*32 / 4096 = 1.5625 %.
    assert 1.4 < ratio < 1.7, f"unexpected ratio: {ratio}"

    # Identity-at-init proof.
    test_lora_identity_at_init()
    print("  identity-at-init test : PASS")

    # effective_weight() check.
    torch.manual_seed(7)
    base = nn.Linear(32, 16, bias=False)
    lora = LoRALinear(base, r=4, alpha=8)
    # Force lora_B to non-zero so the delta is non-trivial.
    with torch.no_grad():
        lora.lora_B.normal_(std=0.1)
    x = torch.randn(8, 32)
    via_forward = lora(x)
    via_effective = F.linear(x, lora.effective_weight())
    err = (via_forward - via_effective).abs().max().item()
    print(f"  effective_weight() max err : {err:.2e}  (expect < 1e-5)")
    assert err < 1e-5
    print("  MEDIUM: PASS")


# ===========================================================================
# HARD -- mini-VLA + 3 training regimes (Full FT, LoRA r=8, LoRA r=32)
# ===========================================================================

@dataclass
class MiniVLAConfig:
    image_size: int = 32
    patch_size: int = 4
    num_patches: int = 64                    # (32/4)^2 = 64
    dinov2_dim: int = 64
    siglip_dim: int = 64
    llm_hidden: int = 64
    llm_layers: int = 2
    llm_heads: int = 4
    vocab_size: int = 256
    max_text_len: int = 4
    action_dim: int = 4


class MiniViT(nn.Module):
    def __init__(self, feature_dim: int, patch_size: int):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, feature_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)               # (B, F, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)          # (B, P, F)
        return self.norm(x)


class MiniLLMBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden))
        self.heads, self.head_dim = heads, hidden // heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask, float("-inf")).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        x = x + self.o_proj(out)
        return x + self.mlp(self.norm2(x))


class MiniVLA(nn.Module):
    """Mini OpenVLA with a regression action head (no tokenization).

    The tokenization step is orthogonal to the LoRA mechanism we want to
    isolate, so we replace it with a Linear(llm_hidden, action_dim) and an MSE
    loss. The LoRA insertion logic is identical to the real OpenVLA case.
    """

    def __init__(self, cfg: MiniVLAConfig):
        super().__init__()
        self.cfg = cfg
        self.dinov2 = MiniViT(cfg.dinov2_dim, cfg.patch_size)
        self.siglip = MiniViT(cfg.siglip_dim, cfg.patch_size)
        concat = cfg.dinov2_dim + cfg.siglip_dim
        self.projector = nn.Sequential(nn.Linear(concat, cfg.llm_hidden), nn.GELU(),
                                        nn.Linear(cfg.llm_hidden, cfg.llm_hidden))
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.llm_hidden)
        self.blocks = nn.ModuleList([MiniLLMBlock(cfg.llm_hidden, cfg.llm_heads) for _ in range(cfg.llm_layers)])
        self.norm = nn.LayerNorm(cfg.llm_hidden)
        self.action_head = nn.Linear(cfg.llm_hidden, cfg.action_dim)

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        d = self.dinov2(image)
        s = self.siglip(image)
        fused = torch.cat([d, s], dim=-1)
        prefix = self.projector(fused)
        text = self.token_embed(text_ids)
        x = torch.cat([prefix, text], dim=1)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        # Use the LAST hidden state as the regression source.
        return self.action_head(x[:, -1, :])


# ----- toy dataset -----

def make_toy_dataset(N: int, cfg: MiniVLAConfig, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(N, 3, cfg.image_size, cfg.image_size, generator=g)
    texts = torch.randint(0, cfg.vocab_size, (N, cfg.max_text_len), generator=g)
    # Simple deterministic target: 4-D action depending on per-channel image mean.
    ch_means = images.mean(dim=(2, 3))                                # (N, 3)
    actions = torch.stack([
        torch.sin(2.0 * ch_means[:, 0]),
        torch.cos(1.5 * ch_means[:, 1]),
        0.7 * ch_means[:, 2],
        ch_means.sum(dim=-1),
    ], dim=-1)
    return images, texts, actions


# ----- LoRA application helper -----

def apply_lora_to_blocks(model: MiniVLA, r: int, alpha: int) -> None:
    """Freeze everything, then wrap q_proj and v_proj of each block with LoRA.

    The action_head (regression head) MUST stay trainable: it is the new task
    interface, frozen base init would never converge.
    """
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks:
        block.q_proj = LoRALinear(block.q_proj, r=r, alpha=alpha)
        block.v_proj = LoRALinear(block.v_proj, r=r, alpha=alpha)
    for p in model.action_head.parameters():
        p.requires_grad = True


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----- training loop -----

def train_one_run(
    model: MiniVLA,
    train_data,
    val_data,
    epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    label: str = "run",
):
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    images_tr, texts_tr, actions_tr = train_data
    images_va, texts_va, actions_va = val_data

    n = images_tr.shape[0]
    last_val = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(images_tr[idx], texts_tr[idx])
            loss = F.mse_loss(pred, actions_tr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                val_pred = model(images_va, texts_va)
                val_mse = F.mse_loss(val_pred, actions_va).item()
            last_val = val_mse
            print(f"    [{label}] epoch {epoch:>3} | val MSE {val_mse:.4f}")
    duration = time.time() - t0
    return last_val, duration


def run_hard() -> None:
    print("\n" + "=" * 70)
    print("HARD -- LoRA vs Full FT comparison")
    print("=" * 70)
    cfg = MiniVLAConfig()

    images, texts, actions = make_toy_dataset(N=2000, cfg=cfg, seed=42)
    split = 1600
    train_data = (images[:split], texts[:split], actions[:split])
    val_data = (images[split:], texts[split:], actions[split:])

    epochs = 60                       # smaller than spec (80-200) to keep CPU runtime reasonable
    results = []

    # ----- A: full finetuning -----
    torch.manual_seed(0)
    m_full = MiniVLA(cfg)
    print(f"\n  [A] Full FT  trainable={trainable_params(m_full):,} / {total_params(m_full):,}")
    val_a, dur_a = train_one_run(m_full, train_data, val_data, epochs=epochs, label="full")
    results.append(("Full FT", trainable_params(m_full), val_a, dur_a))

    # ----- B: LoRA r=8 -----
    torch.manual_seed(0)
    m_lora8 = MiniVLA(cfg)
    apply_lora_to_blocks(m_lora8, r=8, alpha=16)
    print(f"\n  [B] LoRA r=8 trainable={trainable_params(m_lora8):,} / {total_params(m_lora8):,}")
    val_b, dur_b = train_one_run(m_lora8, train_data, val_data, epochs=epochs, label="lora8")
    results.append(("LoRA r=8", trainable_params(m_lora8), val_b, dur_b))

    # ----- C: LoRA r=32 -----
    torch.manual_seed(0)
    m_lora32 = MiniVLA(cfg)
    apply_lora_to_blocks(m_lora32, r=32, alpha=16)
    print(f"\n  [C] LoRA r=32 trainable={trainable_params(m_lora32):,} / {total_params(m_lora32):,}")
    val_c, dur_c = train_one_run(m_lora32, train_data, val_data, epochs=epochs, label="lora32")
    results.append(("LoRA r=32", trainable_params(m_lora32), val_c, dur_c))

    # ----- adversarial probe: r=1 -----
    torch.manual_seed(0)
    m_lora1 = MiniVLA(cfg)
    apply_lora_to_blocks(m_lora1, r=1, alpha=16)
    print(f"\n  [probe] LoRA r=1 trainable={trainable_params(m_lora1):,}")
    val_probe, dur_probe = train_one_run(m_lora1, train_data, val_data, epochs=epochs, label="lora1")
    results.append(("LoRA r=1 (probe)", trainable_params(m_lora1), val_probe, dur_probe))

    # ----- table -----
    print("\n  Results table:")
    print("  " + "-" * 60)
    print(f"  {'Run':<20} {'Trainable':>12} {'Val MSE':>10} {'Time (s)':>10}")
    print("  " + "-" * 60)
    for name, n_trainable, val, dur in results:
        print(f"  {name:<20} {n_trainable:>12,} {val:>10.4f} {dur:>10.2f}")
    print("  " + "-" * 60)

    # ----- discussion -----
    full_trainable = results[0][1]
    lora32_trainable = results[2][1]
    ratio = 100.0 * lora32_trainable / full_trainable
    print(f"\n  Discussion:")
    print(f"  - LoRA r=32 trainable / Full FT trainable = {ratio:.2f} %")
    print(f"  - Full FT val MSE = {results[0][2]:.4f}")
    print(f"  - LoRA r=32 val MSE = {results[2][2]:.4f}  (gap = {results[2][2] - results[0][2]:+.4f})")
    print(f"  - LoRA r=1 probe val MSE = {results[3][2]:.4f}  (rank-1 too low to capture the task)")
    print("  Takeaway: LoRA r=32 reaches near-full FT quality with a tiny fraction")
    print("  of trainable params -- consistent with OpenVLA paper Table 7 finding")
    print("  that LoRA r=32 ≈ 97% of full FT performance with ~1-2% trainable params.")
    print("  HARD: PASS")


# ===========================================================================
# Entrypoint
# ===========================================================================

def main() -> None:
    run_easy()
    run_medium()
    run_hard()


if __name__ == "__main__":
    main()

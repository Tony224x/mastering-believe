"""
J20 - OpenVLA architecture skeleton + LoRA from scratch (pedagogical).

Goal: build a *simplified* OpenVLA forward pass to internalize the shapes,
then implement LoRA from scratch on a Linear layer. We do NOT download Llama2
7B (way too heavy for a teaching module); we replace each heavy block with a
small mock that has the *same input/output contract* as the real one.

References (REFERENCES.md):
    #13 OpenVLA paper + repo (Kim, Pertsch et al., 2024)
        https://arxiv.org/abs/2406.09246
        https://github.com/openvla/openvla
    LoRA original: Hu et al. 2021 (https://arxiv.org/abs/2106.09685)

Run:
    python domains/robotics-ai/02-code/20-openvla-architecture.py

requires: torch>=2.0
"""

# requires: torch>=2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# We keep the real OpenVLA dimensions where they teach something, and shrink
# the LLM hidden size so the demo runs on CPU in seconds.

@dataclass
class OpenVLAMiniConfig:
    image_size: int = 224          # OpenVLA uses 224x224
    patch_size: int = 14           # ViT-L/14 patches
    num_patches: int = 256         # (224/14)^2 = 256

    # Real OpenVLA: DINOv2-L (1024) + SigLIP-SO400M (1152) -> 2176 concat.
    # We mimic the SAME concatenation logic but with smaller numbers.
    dinov2_dim: int = 256          # mock DINOv2 feature dim
    siglip_dim: int = 256          # mock SigLIP feature dim

    # Real OpenVLA: Llama2-7B hidden = 4096. Pedagogical mini: 128.
    llm_hidden: int = 128
    llm_layers: int = 2
    llm_heads: int = 4

    vocab_size: int = 32000        # Llama2 tokenizer vocab size
    action_bins: int = 256         # last 256 tokens of vocab reused for actions
    action_dim: int = 7            # 3 translation + 3 rotation + 1 gripper

    max_text_len: int = 16         # short instruction


# ---------------------------------------------------------------------------
# 2. Mock vision encoders
# ---------------------------------------------------------------------------
# Real DINOv2 / SigLIP are huge ViTs. We only care that they:
#   - take a (B, 3, H, W) image
#   - output (B, num_patches, feature_dim) per-patch features
# So we mock them with tiny conv-based "patch embedders".

class MockViT(nn.Module):
    """A 2-layer fake ViT: convolutional patchifier + 1 transformer block.

    The contract matches the real DINOv2/SigLIP encoders: input (B, 3, 224, 224)
    becomes (B, num_patches, feature_dim).
    """

    def __init__(self, feature_dim: int, patch_size: int, num_patches: int):
        super().__init__()
        # Patchify with a single Conv2d (a-la ViT). 14x14 patches over 224x224.
        self.patch_embed = nn.Conv2d(3, feature_dim, kernel_size=patch_size, stride=patch_size)
        # Single attention block keeps the demo light but realistic in spirit.
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.expected_patches = num_patches

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, 224, 224)
        x = self.patch_embed(image)              # (B, F, 16, 16)
        x = x.flatten(2).transpose(1, 2)         # (B, 256, F)
        assert x.shape[1] == self.expected_patches, "patch count mismatch"
        attn_out, _ = self.attn(x, x, x)         # self-attention across patches
        return self.norm(x + attn_out)           # residual + norm


# ---------------------------------------------------------------------------
# 3. Mock language model (decoder-only Transformer, Llama2-style)
# ---------------------------------------------------------------------------
# We rebuild a tiny decoder-only Transformer. The point is to expose where the
# autoregressive action prediction happens. We keep nn.MultiheadAttention for
# brevity (Llama2 uses RoPE+GQA, but the block-level shape contract is the same).

class MockLLMBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=heads, batch_first=True)
        # We expose the q_proj/v_proj as separate Linear modules that LoRA can
        # later wrap. Real OpenVLA fine-tuning targets exactly these.
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.SiLU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.heads = heads
        self.head_dim = hidden // heads

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H). Manual attention so we can see q_proj/v_proj being used.
        B, T, H = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        # scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        out = self.o_proj(out)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class MockLlama(nn.Module):
    def __init__(self, cfg: OpenVLAMiniConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.llm_hidden)
        self.blocks = nn.ModuleList([
            MockLLMBlock(cfg.llm_hidden, cfg.llm_heads) for _ in range(cfg.llm_layers)
        ])
        self.norm = nn.LayerNorm(cfg.llm_hidden)
        # Tied lm_head: map back to the full vocab. The action bins are the
        # last `action_bins` indices of this vocab.
        self.lm_head = nn.Linear(cfg.llm_hidden, cfg.vocab_size, bias=False)

    def forward(self, prefix_embeds: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        # prefix_embeds: (B, P, H) already projected vision features (P=256 patches)
        # text_ids: (B, T_text) language instruction token ids
        text_embeds = self.token_embed(text_ids)             # (B, T_text, H)
        x = torch.cat([prefix_embeds, text_embeds], dim=1)   # (B, P + T_text, H)

        # causal mask over the full sequence
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)                             # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# 4. The OpenVLA mini policy
# ---------------------------------------------------------------------------

class OpenVLAMini(nn.Module):
    """Pedagogical clone of OpenVLA: dual vision encoder + projector + LLM +
    action detokenization."""

    def __init__(self, cfg: OpenVLAMiniConfig):
        super().__init__()
        self.cfg = cfg

        # Two parallel vision encoders, just like the real OpenVLA.
        self.dinov2 = MockViT(cfg.dinov2_dim, cfg.patch_size, cfg.num_patches)
        self.siglip = MockViT(cfg.siglip_dim, cfg.patch_size, cfg.num_patches)

        # Concatenate per-patch then project to LLM hidden size.
        # This mirrors `prismatic/models/backbones/vision/` in the OpenVLA repo.
        concat_dim = cfg.dinov2_dim + cfg.siglip_dim
        self.projector = nn.Sequential(
            nn.Linear(concat_dim, cfg.llm_hidden),
            nn.GELU(),
            nn.Linear(cfg.llm_hidden, cfg.llm_hidden),
        )

        self.llm = MockLlama(cfg)

        # Quantile bin edges per action dimension (would be computed from the
        # OXE dataset 1%-99% quantiles in the real pipeline). We mock them.
        # Shape: (action_dim, action_bins + 1).
        edges = torch.linspace(-1.0, 1.0, cfg.action_bins + 1)
        self.register_buffer(
            "bin_edges",
            edges.unsqueeze(0).expand(cfg.action_dim, -1).contiguous(),
        )

    # ------- forward pass -------

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """Replicates OpenVLA's dual-encoder concatenation."""
        d_feat = self.dinov2(image)                          # (B, 256, F_d)
        s_feat = self.siglip(image)                          # (B, 256, F_s)
        # Concatenation along the feature axis, NOT the patch axis. This is the
        # key design choice: each patch gets BOTH a DINOv2 and a SigLIP descriptor.
        fused = torch.cat([d_feat, s_feat], dim=-1)          # (B, 256, F_d+F_s)
        return self.projector(fused)                         # (B, 256, llm_hidden)

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        prefix = self.encode_vision(image)                   # (B, 256, H)
        return self.llm(prefix, text_ids)                    # (B, T, vocab)

    # ------- action token <-> action value -------

    def action_token_range(self) -> tuple[int, int]:
        """The last `action_bins` indices of the vocab map to action bins."""
        end = self.cfg.vocab_size
        start = end - self.cfg.action_bins
        return start, end

    def detokenize(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """(B, action_dim) token ids -> (B, action_dim) continuous values via
        the per-dimension quantile bin centers.
        """
        start, _ = self.action_token_range()
        bin_idx = action_token_ids - start                   # 0..action_bins-1
        # Pick the bin centers per dim.
        centers = 0.5 * (self.bin_edges[:, :-1] + self.bin_edges[:, 1:])  # (D, B)
        # Gather: for each (batch, dim), pick centers[dim, bin_idx[batch, dim]]
        gathered = centers.unsqueeze(0).expand(bin_idx.shape[0], -1, -1)  # (B,D,Bins)
        return gathered.gather(2, bin_idx.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def predict_action(self, image: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """Greedy autoregressive decode of `action_dim` tokens."""
        prefix = self.encode_vision(image)
        text_embeds = self.llm.token_embed(text_ids)
        x = torch.cat([prefix, text_embeds], dim=1)

        start, end = self.action_token_range()
        produced: list[torch.Tensor] = []

        for _ in range(self.cfg.action_dim):
            T = x.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            h = x
            for block in self.llm.blocks:
                h = block(h, mask)
            h = self.llm.norm(h)
            logits = self.llm.lm_head(h[:, -1, :])           # (B, vocab)
            # Force the next token to land in the action vocab range.
            mask_vocab = torch.full_like(logits, float("-inf"))
            mask_vocab[:, start:end] = 0.0
            next_token = (logits + mask_vocab).argmax(dim=-1)  # (B,)
            produced.append(next_token)
            # Append the new embedding to the running sequence.
            x = torch.cat([x, self.llm.token_embed(next_token).unsqueeze(1)], dim=1)

        action_tokens = torch.stack(produced, dim=1)         # (B, action_dim)
        return self.detokenize(action_tokens)


# ---------------------------------------------------------------------------
# 5. LoRA from scratch
# ---------------------------------------------------------------------------
# Goal: implement LoRA without depending on `peft`. We build a wrapper that
# replaces a Linear layer with `Linear + alpha/r * B @ A`, freezes the base,
# and ONLY trains A and B.

class LoRALinear(nn.Module):
    """Replaces a frozen `nn.Linear` with W + (alpha/r) * B @ A.

    A is initialized Gaussian, B is initialized to zero so that at step 0 the
    output equals the original frozen Linear (essential to preserve pretraining).
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRA only wraps nn.Linear"
        self.base = base
        # Freeze the base weights: this is what makes LoRA cheap.
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        d_in = base.in_features
        d_out = base.out_features
        # A: (r, d_in), Gaussian init.
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        nn.init.normal_(self.lora_A, std=1.0 / r)
        # B: (d_out, r), zero init -> at step 0, BA = 0 -> output unchanged.
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        base_out = self.base(x)
        # LoRA delta: x @ A.T @ B.T scaled by alpha/r
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def merge_into_base(self) -> None:
        """Bake LoRA into the frozen base weights, then drop A and B.

        Used once fine-tuning is over and you want to deploy a single Linear.
        """
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)  # (d_out, d_in)
            self.base.weight.add_(delta)
        # zero out so that the module becomes equivalent to base only
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()


def apply_lora_to_llm(model: OpenVLAMini, r: int = 8, alpha: int = 16) -> int:
    """Walk the LLM, wrap every q_proj/v_proj with LoRA, freeze the rest.

    Returns the number of trainable parameters after wrapping.
    """
    # 1. Freeze EVERYTHING first.
    for p in model.parameters():
        p.requires_grad = False

    # 2. Wrap q_proj and v_proj of every transformer block.
    for block in model.llm.blocks:
        block.q_proj = LoRALinear(block.q_proj, r=r, alpha=alpha)
        block.v_proj = LoRALinear(block.v_proj, r=r, alpha=alpha)

    # 3. Count trainable params (only the new LoRA A/B should be trainable).
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(0)
    cfg = OpenVLAMiniConfig()
    model = OpenVLAMini(cfg)

    # Fake batch: 2 images, 2 short instructions.
    B = 2
    image = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text_ids = torch.randint(0, cfg.vocab_size - cfg.action_bins, (B, cfg.max_text_len))

    print("=== Forward pass shape check ===")
    vision_out = model.encode_vision(image)
    print(f"  vision tokens : {tuple(vision_out.shape)}  (expected (B, 256, llm_hidden))")
    logits = model(image, text_ids)
    print(f"  llm logits    : {tuple(logits.shape)}  (expected (B, 256+max_text_len, vocab))")

    print("\n=== Predict 7-D action ===")
    action = model.predict_action(image, text_ids)
    print(f"  action shape  : {tuple(action.shape)}  (expected (B, 7))")
    print(f"  sample action : {action[0].tolist()}")

    print("\n=== LoRA from scratch ===")
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  before LoRA: {trainable_before:,} / {total_before:,} trainable")

    trainable_after = apply_lora_to_llm(model, r=8, alpha=16)
    total_after = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable_after / total_after
    print(f"  after  LoRA: {trainable_after:,} / {total_after:,} trainable ({pct:.3f} %)")

    # Sanity check: at step 0 the model with LoRA should produce IDENTICAL
    # logits to before, because B was zero-initialized.
    print("\n=== LoRA initial-equivalence check ===")
    # Re-run the same input on the LoRA-equipped model.
    logits_lora = model(image, text_ids)
    diff = (logits - logits_lora).abs().max().item()
    print(f"  max |logits - logits_lora| = {diff:.6e}  (should be ~0)")
    assert diff < 1e-4, "LoRA at init must be identity!"

    # Take ONE training step to show that LoRA params get gradients while the
    # base remains frozen.
    print("\n=== One LoRA training step ===")
    target = torch.randint(0, cfg.vocab_size, (B, logits_lora.shape[1]))
    loss = F.cross_entropy(logits_lora.flatten(0, 1), target.flatten(0, 1))
    loss.backward()

    # Confirm: a base param has no grad, a LoRA param does.
    base_grad = model.llm.blocks[0].q_proj.base.weight.grad
    lora_a_grad = model.llm.blocks[0].q_proj.lora_A.grad
    print(f"  base q_proj.weight.grad is None : {base_grad is None}")
    print(f"  lora_A.grad norm                 : {lora_a_grad.norm().item():.4f}")

    # Merge LoRA back into the base (deployment-time op).
    print("\n=== Merge LoRA into base ===")
    for block in model.llm.blocks:
        block.q_proj.merge_into_base()
        block.v_proj.merge_into_base()
    # After merge, A and B are zero so the module behaves exactly like the
    # merged base Linear. We could replace LoRALinear with its `.base` for a
    # smaller deployable model.
    logits_merged = model(image, text_ids)
    print(f"  logits_merged shape : {tuple(logits_merged.shape)}")
    print("  done.")


if __name__ == "__main__":
    main()

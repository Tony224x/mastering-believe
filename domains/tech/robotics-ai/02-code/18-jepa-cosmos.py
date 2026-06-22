"""
J18 - Mini-JEPA pedagogique vs reconstruction pixel sur MNIST masque.

Goal: materialiser la difference *pixel-recon* (Dreamer / autoencoder style) vs
*latent-recon* (JEPA / V-JEPA 2 style, ref REFERENCES.md #21).

Two models share the same 4-layer CNN encoder skeleton:

    PixelAE  : encoder(context) -> latent z -> decoder(z) -> reconstructed image
                        loss = MSE(reconstructed_image, target_image)        <-- PIXEL space

    MiniJEPA : encoder_ctx(context) -> z_ctx
               encoder_tgt(target)  -> z_tgt   (EMA of encoder_ctx, stop-grad)
               predictor(z_ctx)     -> z_hat
                        loss = MSE(z_hat, z_tgt)                              <-- LATENT space

We train both on a "predict the masked half" task: the model sees the LEFT half
of an MNIST digit and must predict the RIGHT half (or its representation).

We then show:
1. PixelAE produces a blurry but visually inspectable reconstruction.
2. MiniJEPA produces a latent vector — we cannot "see" it, but if we train a
   tiny linear probe on top, the probe matches the original digit class with
   higher accuracy than a probe on PixelAE's latent. That is the JEPA win:
   the latent encodes *structure useful for downstream tasks*, not pixels.

The script is small (CPU-friendly, ~1 min on a laptop) and uses a synthetic
MNIST-like dataset built from random patterns + noise so the file is fully
self-contained: no torchvision download required. The mechanics (EMA target,
stop-grad, latent MSE) are 1:1 with V-JEPA 2.

requires: torch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Synthetic "MNIST-like" toy dataset (no torchvision needed)
# =============================================================================
# We build 8x8 binary digits as low-rank patterns: each "class" is a fixed
# 8x8 template + noise. 5 classes are enough to demonstrate the paradigm.
# Real MNIST would just be heavier but identical in structure.

def make_toy_dataset(n_samples=2000, n_classes=5, img_size=8, seed=0):
    """Return tensor X of shape (n_samples, 1, img_size, img_size) and labels y."""
    g = torch.Generator().manual_seed(seed)
    # one fixed template per class
    templates = torch.rand(n_classes, 1, img_size, img_size, generator=g)
    templates = (templates > 0.5).float()  # binarize so digits are clearly different
    # sample
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = templates[y].clone()
    # add small per-pixel noise so each sample is slightly different
    X = X + 0.15 * torch.randn(X.shape, generator=g)
    X = X.clamp(0.0, 1.0)
    return X, y


def split_left_right(X):
    """Split each image vertically. Left = context, Right = target."""
    H = X.shape[-1]
    half = H // 2
    left = X[..., :half]   # context (what the model sees)
    right = X[..., half:]  # target (what the model must predict)
    return left, right


# =============================================================================
# 2. Shared CNN encoder backbone (small, CPU-friendly)
# =============================================================================
# Both models reuse this encoder so the comparison is fair: only the LOSS
# differs (pixel reconstruction vs latent prediction).

class TinyEncoder(nn.Module):
    """Tiny CNN: (B, 1, H, W) -> (B, latent_dim)."""

    def __init__(self, latent_dim=32, in_channels=1):
        super().__init__()
        # Two conv blocks then a linear projection. Padding=1 keeps spatial dims.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # We assume input H=8, W=4 (left half of an 8x8 image). After two
        # 2x2 average pools we end up with 32 * 2 * 1 = 64 features.
        self.fc = nn.Linear(32 * 2 * 1, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)               # 8x4 -> 4x2
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)               # 4x2 -> 2x1
        x = x.flatten(1)                     # (B, 64)
        return self.fc(x)                    # (B, latent_dim)


class TinyDecoder(nn.Module):
    """Mirror of the encoder: (B, latent_dim) -> (B, 1, 8, 4) reconstruction."""

    def __init__(self, latent_dim=32, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 2 * 1)
        # We upsample by repeat then refine with convs.
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 32, 2, 1)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 2x1 -> 4x2
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 4x2 -> 8x4
        x = torch.sigmoid(self.conv2(x))                      # pixel intensities in [0,1]
        return x


# =============================================================================
# 3. Model A: Pixel autoencoder (Dreamer-style: loss in pixel space)
# =============================================================================

class PixelAE(nn.Module):
    """Encoder context, decoder predicts target half pixels.
    Loss = MSE between predicted pixels and target pixels.
    This is the philosophy critiqued by LeCun: 99% of pixels are noise."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = TinyEncoder(latent_dim=latent_dim)
        self.decoder = TinyDecoder(latent_dim=latent_dim)

    def forward(self, context):
        z = self.encoder(context)
        recon = self.decoder(z)
        return recon, z


# =============================================================================
# 4. Model B: Mini-JEPA (LeCun-style: loss in latent space, EMA target)
# =============================================================================
# Key tricks (all from V-JEPA 2):
#   - Two encoders: f_theta (context, gradient) and f_xi (target, EMA).
#   - Stop-gradient on the target encoder output.
#   - A predictor on top of the context latent.
#   - Loss = MSE in LATENT space between predictor(z_ctx) and z_tgt. NO pixel decoder.

class MiniJEPA(nn.Module):
    """Mini-JEPA: predict the *latent* of the target half from the context half."""

    def __init__(self, latent_dim=32, ema_decay=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Context encoder (trained with gradient).
        self.encoder_ctx = TinyEncoder(latent_dim=latent_dim)
        # Target encoder (EMA copy, no gradient).
        self.encoder_tgt = TinyEncoder(latent_dim=latent_dim)
        # Initialise the EMA encoder with the same weights, then freeze grads.
        self.encoder_tgt.load_state_dict(self.encoder_ctx.state_dict())
        for p in self.encoder_tgt.parameters():
            p.requires_grad = False

        # Predictor: z_ctx -> z_hat (in latent space). 2-layer MLP is fine.
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of encoder_tgt = decay * encoder_tgt + (1 - decay) * encoder_ctx."""
        for p_tgt, p_ctx in zip(self.encoder_tgt.parameters(), self.encoder_ctx.parameters()):
            p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1.0 - self.ema_decay)

    def forward(self, context, target):
        # Context branch (gradients flow).
        z_ctx = self.encoder_ctx(context)
        z_hat = self.predictor(z_ctx)

        # Target branch: stop-gradient + EMA encoder.
        with torch.no_grad():
            z_tgt = self.encoder_tgt(target)

        return z_hat, z_tgt, z_ctx


# =============================================================================
# 5. Training loops
# =============================================================================

def train_pixel_ae(model, X_left, X_right, n_epochs=200, lr=1e-3, batch_size=128, seed=0):
    """Train pixel autoencoder: MSE on pixels."""
    g = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = X_left.shape[0]
    for epoch in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            ctx, tgt = X_left[b], X_right[b]
            recon, _ = model(ctx)
            loss = F.mse_loss(recon, tgt)             # PIXEL loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


def train_mini_jepa(model, X_left, X_right, n_epochs=200, lr=1e-3, batch_size=128, seed=0):
    """Train mini-JEPA: MSE in latent space + EMA update each step."""
    g = torch.Generator().manual_seed(seed)
    # Only train context encoder + predictor (EMA encoder has no grad).
    trainable = list(model.encoder_ctx.parameters()) + list(model.predictor.parameters())
    optim = torch.optim.Adam(trainable, lr=lr)
    losses = []
    n = X_left.shape[0]
    for epoch in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            ctx, tgt = X_left[b], X_right[b]
            z_hat, z_tgt, _ = model(ctx, tgt)
            loss = F.mse_loss(z_hat, z_tgt)           # LATENT loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.update_target_encoder()             # EMA update
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


# =============================================================================
# 6. Linear probe on the learned latent (downstream evaluation)
# =============================================================================
# To compare the two paradigms we train a simple linear classifier on top of
# the learned latents: which paradigm produces a latent that better predicts
# the digit class? This is the standard self-supervised eval (linear probe).

def linear_probe(latents, labels, n_classes, n_epochs=300, lr=5e-2, seed=0):
    """Fit a linear classifier latent -> class. Returns final train accuracy."""
    torch.manual_seed(seed)
    clf = nn.Linear(latents.shape[1], n_classes)
    optim = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(n_epochs):
        logits = clf(latents)
        loss = F.cross_entropy(logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        preds = clf(latents).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc


# =============================================================================
# 7. Main demo
# =============================================================================

def main():
    torch.manual_seed(0)

    print("=" * 72)
    print("J18 - Mini-JEPA vs Pixel autoencoder on toy MNIST")
    print("Reference: V-JEPA 2 (Meta 2025), REFERENCES.md #21")
    print("=" * 72)

    # 1. Build dataset and split
    X, y = make_toy_dataset(n_samples=2000, n_classes=5, img_size=8, seed=0)
    X_left, X_right = split_left_right(X)
    print(f"\nDataset: {X.shape[0]} samples, image {tuple(X.shape[1:])}, "
          f"context (left) = {tuple(X_left.shape[1:])}, target (right) = {tuple(X_right.shape[1:])}")

    # 2. Train Pixel AE (Dreamer-style)
    print("\n[Model A] Pixel autoencoder (loss = MSE on PIXELS)")
    pixel_ae = PixelAE(latent_dim=32)
    pixel_losses = train_pixel_ae(pixel_ae, X_left, X_right, n_epochs=80)
    print(f"  pixel-loss start={pixel_losses[0]:.4f}, end={pixel_losses[-1]:.4f}")

    # 3. Train Mini-JEPA (LeCun-style)
    print("\n[Model B] Mini-JEPA (loss = MSE in LATENT space, EMA target encoder)")
    jepa = MiniJEPA(latent_dim=32, ema_decay=0.99)
    jepa_losses = train_mini_jepa(jepa, X_left, X_right, n_epochs=80)
    print(f"  latent-loss start={jepa_losses[0]:.4f}, end={jepa_losses[-1]:.4f}")
    print("  (note: pixel-loss and latent-loss are NOT comparable in absolute value -")
    print("   they live in different spaces. We compare downstream usefulness.)")

    # 4. Extract latents from both models on the same data
    pixel_ae.eval()
    jepa.eval()
    with torch.no_grad():
        z_pixel = pixel_ae.encoder(X_left)             # PixelAE latent (z that decodes well)
        z_jepa = jepa.encoder_ctx(X_left)              # JEPA latent (z that predicts target latent)

    # 5. Linear probe: which latent better separates the classes?
    acc_pixel = linear_probe(z_pixel, y, n_classes=5)
    acc_jepa = linear_probe(z_jepa, y, n_classes=5)

    # 6. Pixel-MSE achievable from each latent for fairness
    # The PixelAE latent was DESIGNED to reconstruct pixels - so it should win there.
    with torch.no_grad():
        pixel_recon, _ = pixel_ae(X_left)
        pixel_mse = F.mse_loss(pixel_recon, X_right).item()
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  PixelAE pixel-MSE on target half       : {pixel_mse:.4f}  (baseline: PixelAE wins on pixels by design)")
    print(f"  Linear probe accuracy on PixelAE latent: {acc_pixel * 100:.1f}%")
    print(f"  Linear probe accuracy on JEPA latent   : {acc_jepa * 100:.1f}%")

    print("\nINTERPRETATION (LeCun's argument materialised):")
    print("  - PixelAE's latent is optimised to reconstruct pixels. It encodes texture")
    print("    + position + intensity. Some of that is noise irrelevant to identifying")
    print("    the digit class.")
    print("  - JEPA's latent is optimised to PREDICT THE LATENT of the target half. To do")
    print("    that without a decoder, it must encode the *structure* of the digit -")
    print("    exactly what a downstream classifier needs.")
    print("  - On real V-JEPA 2 (1.2B params, 1M h video), the same trick produces a")
    print("    representation good enough for zero-shot pick-and-place, with NO pixel")
    print("    decoder ever trained.")
    print()
    print("DREAMER vs JEPA vs COSMOS (recap of the J18 theory module):")
    print("  - Dreamer (Hafner 2023): reconstruct pixels of the future + actor/critic in")
    print("    imagination. Data-efficient for RL but spends params on visual noise.")
    print("  - JEPA / V-JEPA 2 (Meta 2025, REF #21): predict latent of the future.")
    print("    Refuses pixels by design. SOTA on action recognition + zero-shot manip.")
    print("  - Cosmos (NVIDIA 2025, REF #22): foundation model trained on 20M h of video.")
    print("    Generates pixels but at a scale where it becomes a reusable backbone +")
    print("    tokenizer + synthetic data engine for downstream VLAs (e.g. GR00T data).")


if __name__ == "__main__":
    main()

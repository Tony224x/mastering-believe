"""
J18 - Solutions consolidees easy / medium / hard.

EASY  : reponses textuelles (10 scenarios classifies Dreamer / JEPA / Cosmos).
MEDIUM: detection de collapse sans EMA + linear probe sous bruit + scan latent_dim.
HARD  : JEPA action-conditionnee sur env toy 8x8 + planning latent goal-image
        + comparaison vs pixel planner + stress test stochastique.

Usage:
    python 18-jepa-cosmos.py easy
    python 18-jepa-cosmos.py medium
    python 18-jepa-cosmos.py hard
    python 18-jepa-cosmos.py all      (defaut)

Source principale: V-JEPA 2 (Meta 2025, REFERENCES.md #21).
Source complementaire: NVIDIA Cosmos (REFERENCES.md #22).

requires: torch
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers partages (repris de 02-code/18-jepa-cosmos.py, condense)
# =============================================================================

def make_toy_dataset(n_samples=2000, n_classes=5, img_size=8, seed=0, noise_amp=0.15):
    """Reutilise le toy dataset MNIST-like."""
    g = torch.Generator().manual_seed(seed)
    templates = (torch.rand(n_classes, 1, img_size, img_size, generator=g) > 0.5).float()
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    X = templates[y].clone()
    X = X + noise_amp * torch.randn(X.shape, generator=g)
    return X.clamp(0.0, 1.0), y


def split_left_right(X):
    H = X.shape[-1]
    return X[..., :H // 2], X[..., H // 2:]


class TinyEncoder(nn.Module):
    """CNN encoder ; param `in_h, in_w` permet de l'utiliser sur image 8x4 (half) ou 8x8 (full)."""

    def __init__(self, latent_dim=32, in_h=8, in_w=4, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # apres deux pools 2x2: H/4, W/4
        h_out = max(in_h // 4, 1)
        w_out = max(in_w // 4, 1)
        self.fc = nn.Linear(32 * h_out * w_out, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        return self.fc(x.flatten(1))


class TinyDecoder(nn.Module):
    """Decoder symetrique. `out_h, out_w` configurable."""

    def __init__(self, latent_dim=32, out_h=8, out_w=4, out_channels=1):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.h_in = max(out_h // 4, 1)
        self.w_in = max(out_w // 4, 1)
        self.fc = nn.Linear(latent_dim, 32 * self.h_in * self.w_in)
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 32, self.h_in, self.w_in)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="nearest")
        x = torch.sigmoid(self.conv2(x))
        return x


class PixelAE(nn.Module):
    def __init__(self, latent_dim=32, in_h=8, in_w=4):
        super().__init__()
        self.encoder = TinyEncoder(latent_dim, in_h, in_w)
        self.decoder = TinyDecoder(latent_dim, in_h, in_w)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class MiniJEPA(nn.Module):
    def __init__(self, latent_dim=32, ema_decay=0.99, in_h=8, in_w=4):
        super().__init__()
        self.ema_decay = ema_decay
        self.encoder_ctx = TinyEncoder(latent_dim, in_h, in_w)
        self.encoder_tgt = TinyEncoder(latent_dim, in_h, in_w)
        self.encoder_tgt.load_state_dict(self.encoder_ctx.state_dict())
        for p in self.encoder_tgt.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, latent_dim))

    @torch.no_grad()
    def update_target(self, hard_sync=False):
        if hard_sync:
            self.encoder_tgt.load_state_dict(self.encoder_ctx.state_dict())
            return
        for p_t, p_c in zip(self.encoder_tgt.parameters(), self.encoder_ctx.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1.0 - self.ema_decay)

    def forward(self, ctx, tgt):
        z_ctx = self.encoder_ctx(ctx)
        z_hat = self.predictor(z_ctx)
        with torch.no_grad():
            z_tgt = self.encoder_tgt(tgt)
        return z_hat, z_tgt, z_ctx


def linear_probe(latents, labels, n_classes, n_epochs=300, lr=5e-2, seed=0):
    torch.manual_seed(seed)
    clf = nn.Linear(latents.shape[1], n_classes)
    optim = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(n_epochs):
        loss = F.cross_entropy(clf(latents), labels)
        optim.zero_grad(); loss.backward(); optim.step()
    with torch.no_grad():
        return (clf(latents).argmax(1) == labels).float().mean().item()


# =============================================================================
# EASY - reponses textuelles
# =============================================================================

EASY_ANSWERS = """\
EASY - 10 scenarios classifies (Dreamer / JEPA / Cosmos / autre)

 1. Quadrupede MuJoCo, 30 min GPU, RL data-efficient
    -> Dreamer. L'imagination latent permet d'apprendre actor/critic dans le world model
       sans collecter 100M steps reels (DreamerV3 Hafner 2023, REF #20).

 2. Pretrain encodeur sur 1M h video YouTube + branche en backbone VLA
    -> JEPA / V-JEPA 2. Self-supervised pretrain sur video sans label, latent transferable
       vers downstream (REF #21, V-JEPA 2 Meta 2025).

 3. Augmenter dataset 200 demos -> generer 50k trajectoires plausibles
    -> Cosmos (synthetic data). Foundation video au scale 20M h, exactement le pattern
       utilise pour generer 780k trajectoires synthetiques GR00T (REF #22).

 4. Robot zero-shot pick-and-place a partir d'une goal image
    -> JEPA / V-JEPA 2. Use case demontre par Meta 2025 : encoder o_t et o_goal en latent,
       planifier l'action qui rapproche z_t+1 de z_goal (REF #21, blog Meta).

 5. Decoder produit images floues, on suggere de l'enlever
    -> JEPA. C'est exactement l'argument LeCun : la MSE pixel sur futur multimodal moyenne
       les modes -> blur. Solution = abandonner les pixels, predire le latent (REF #21).
       Detail: 99% des pixels sont du bruit pour la decision robotique.

 6. Tokenizer videos d'usine en tokens compacts, pas de GPU
    -> Cosmos-Tokenizer. Tokenizer pre-entraine, reutilisable plug-and-play (REF #22).

 7. Atari 100k interactions, apprendre actor-critic en imagination
    -> Dreamer. Imagination = collecter rollouts dans le world model au lieu de l'env
       reel, c'est le paradigme central de DreamerV3 (REF #20).

 8. Blog AI Meta juin 2025 : pick-and-place objets jamais vus depuis 1M h video sans label
    -> JEPA / V-JEPA 2. Description directe de V-JEPA 2 (REF #21).

 9. Capstone Diffusion Policy : "ajouter un world model pour generer le futur du PushT" ?
    -> AUCUN des trois. Diffusion Policy (Chi 2023, REF #19) est un *action diffusion*
       (predit la sequence d'actions), pas un world model. Ajouter un world model est
       hors du scope du capstone et ne fait pas du tout la meme chose. Confusion classique
       a eviter : world model = futur du monde ; diffusion policy = action multimodale.

10. Planning dans l'espace latent (encoder o_t, o_goal, search action z_t -> z_goal)
    -> JEPA. C'est exactement l'usage demontre dans V-JEPA 2 (REF #21).
"""


def easy():
    print(EASY_ANSWERS)


# =============================================================================
# MEDIUM - detection collapse + probe sous bruit + scan latent_dim
# =============================================================================

def train_jepa_medium(model, X_left, X_right, n_epochs=60, batch_size=128, lr=1e-3,
                      seed=0, hard_sync=False):
    g = torch.Generator().manual_seed(seed)
    trainable = list(model.encoder_ctx.parameters()) + list(model.predictor.parameters())
    optim = torch.optim.Adam(trainable, lr=lr)
    losses = []
    n = X_left.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            z_hat, z_tgt, _ = model(X_left[b], X_right[b])
            loss = F.mse_loss(z_hat, z_tgt)
            optim.zero_grad(); loss.backward(); optim.step()
            model.update_target(hard_sync=hard_sync)
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / max(nb, 1))
    return losses


def train_pixelae_medium(model, X_left, X_right, n_epochs=60, batch_size=128, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = X_left.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            recon, _ = model(X_left[b])
            loss = F.mse_loss(recon, X_right[b])
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / max(nb, 1))
    return losses


def medium():
    print("=" * 72)
    print("MEDIUM - Partie 1: detection du collapse sans EMA")
    print("=" * 72)
    torch.manual_seed(0)
    X, y = make_toy_dataset(n_samples=1000, n_classes=5, seed=0)
    X_left, X_right = split_left_right(X)

    # Sans EMA (hard sync apres chaque step) -> collapse possible
    jepa_collapse = MiniJEPA(latent_dim=32, ema_decay=0.0)
    losses_c = train_jepa_medium(jepa_collapse, X_left, X_right, n_epochs=40, hard_sync=True)
    with torch.no_grad():
        z_c = jepa_collapse.encoder_ctx(X_left)
        norm_c = z_c.norm(dim=1).mean().item()
        var_c = z_c.std(dim=0).mean().item()
        acc_c = linear_probe(z_c, y, n_classes=5)

    # Avec EMA actif -> stable
    jepa_ok = MiniJEPA(latent_dim=32, ema_decay=0.99)
    losses_ok = train_jepa_medium(jepa_ok, X_left, X_right, n_epochs=40, hard_sync=False)
    with torch.no_grad():
        z_ok = jepa_ok.encoder_ctx(X_left)
        norm_ok = z_ok.norm(dim=1).mean().item()
        var_ok = z_ok.std(dim=0).mean().item()
        acc_ok = linear_probe(z_ok, y, n_classes=5)

    print(f"\nSans EMA (hard sync, ema=0.0): final loss={losses_c[-1]:.5f}")
    print(f"  ||z||_2 mean = {norm_c:.4f}  | mean per-dim std = {var_c:.4f}  | probe acc = {acc_c*100:.1f}%")
    print(f"\nAvec EMA (decay=0.99):         final loss={losses_ok[-1]:.5f}")
    print(f"  ||z||_2 mean = {norm_ok:.4f}  | mean per-dim std = {var_ok:.4f}  | probe acc = {acc_ok*100:.1f}%")
    print()
    print("INTERPRETATION:")
    print("  Sans EMA, le predictor + l'encoder optimisent ensemble une cible qui")
    print("  bouge en meme temps qu'eux : la solution triviale (z = const) minimise")
    print("  la loss latente sans encoder rien d'utile -> per-dim std s'effondre, probe")
    print("  acc tombe vers 1/n_classes = 20%.")
    print("  Avec EMA, la cible bouge LENTEMENT. Le predictor doit predire une cible")
    print("  qui n'est pas immediatement collapsable -> la representation reste informative.")
    print("  Meme strategie chez BYOL et DINO (stop-grad + EMA target).")

    # Partie 2 - probe sous bruit
    print("\n" + "=" * 72)
    print("MEDIUM - Partie 2: linear probe sous bruit pixel target")
    print("=" * 72)
    print(f"{'noise_amp':>12} {'PixelAE pixel-MSE':>20} {'PixelAE probe acc':>22} {'JEPA probe acc':>20}")
    for noise_amp in [0.0, 0.5, 1.0, 2.0]:
        Xn, yn = make_toy_dataset(n_samples=1000, n_classes=5, seed=1, noise_amp=0.15)
        # injecter du bruit additionnel uniquement sur la moitie target
        Xn_left, Xn_right = split_left_right(Xn)
        Xn_right = (Xn_right + noise_amp * torch.randn(Xn_right.shape)).clamp(0, 1)
        # train PixelAE
        pae = PixelAE(latent_dim=32)
        train_pixelae_medium(pae, Xn_left, Xn_right, n_epochs=30)
        # train JEPA
        jepa = MiniJEPA(latent_dim=32, ema_decay=0.99)
        train_jepa_medium(jepa, Xn_left, Xn_right, n_epochs=30, hard_sync=False)
        with torch.no_grad():
            recon, z_p = pae(Xn_left)
            pixel_mse = F.mse_loss(recon, Xn_right).item()
            z_j = jepa.encoder_ctx(Xn_left)
        acc_p = linear_probe(z_p, yn, n_classes=5)
        acc_j = linear_probe(z_j, yn, n_classes=5)
        print(f"{noise_amp:>12.2f} {pixel_mse:>20.4f} {acc_p*100:>20.1f}% {acc_j*100:>18.1f}%")
    print("\nObservation attendue: PixelAE pixel-MSE explose (le decoder essaie de")
    print("reconstruire le bruit), tandis que la probe acc JEPA reste plus stable")
    print("car la JEPA a appris la STRUCTURE (la classe), pas les pixels bruits.")
    print("C'est la materialisation directe de l'argument LeCun (REF #21).")

    # Partie 3 - scan latent_dim
    print("\n" + "=" * 72)
    print("MEDIUM - Partie 3: scan latent_dim sur JEPA")
    print("=" * 72)
    Xs, ys = make_toy_dataset(n_samples=1000, n_classes=5, seed=2)
    Xs_left, Xs_right = split_left_right(Xs)
    print(f"{'latent_dim':>12} {'probe acc':>12}")
    for d in [2, 8, 32, 128]:
        m = MiniJEPA(latent_dim=d, ema_decay=0.99)
        train_jepa_medium(m, Xs_left, Xs_right, n_epochs=30, hard_sync=False)
        with torch.no_grad():
            zd = m.encoder_ctx(Xs_left)
        acc = linear_probe(zd, ys, n_classes=5)
        print(f"{d:>12d} {acc*100:>10.1f}%")
    print("\nSweet spot typique sur cette tache toy: 16-64. Trop petit (d=2) -> info")
    print("perdue. Trop grand (d=128) -> capacite gaspillee, gradient dilue.")


# =============================================================================
# HARD - JEPA action-conditionnee + planning latent goal-image
# =============================================================================
# Toy env: point sur grille 8x8, actions {up, down, left, right, stay}.

GRID_SIZE = 8
ACTION_DELTAS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (0, 0),    # stay
}
N_ACTIONS = len(ACTION_DELTAS)


def grid_step(pos, action):
    """pos = (r, c) ; action int. Retourne nouvelle position bornee."""
    r, c = pos
    dr, dc = ACTION_DELTAS[action]
    return max(0, min(GRID_SIZE - 1, r + dr)), max(0, min(GRID_SIZE - 1, c + dc))


def pos_to_obs(pos):
    """Position (r, c) -> image 8x8 binaire (1 pixel actif)."""
    img = torch.zeros(1, GRID_SIZE, GRID_SIZE)
    img[0, pos[0], pos[1]] = 1.0
    return img


def make_video_dataset(n_samples=5000, seed=0, action_up_fail_p=0.0):
    """Genere transitions (o_t, a_t, o_t+1) sur la grille."""
    g = torch.Generator().manual_seed(seed)
    obs_t, acts, obs_tp1 = [], [], []
    rand_pos = torch.randint(0, GRID_SIZE, (n_samples, 2), generator=g)
    rand_acts = torch.randint(0, N_ACTIONS, (n_samples,), generator=g)
    rand_fails = torch.rand(n_samples, generator=g)
    for i in range(n_samples):
        pos = (int(rand_pos[i, 0]), int(rand_pos[i, 1]))
        a = int(rand_acts[i])
        # stochasticity: action `up` (id=0) fails with probability action_up_fail_p
        if a == 0 and rand_fails[i] < action_up_fail_p:
            new_pos = pos  # stay
        else:
            new_pos = grid_step(pos, a)
        obs_t.append(pos_to_obs(pos))
        acts.append(a)
        obs_tp1.append(pos_to_obs(new_pos))
    return torch.stack(obs_t), torch.tensor(acts), torch.stack(obs_tp1)


class ActionCondJEPA(nn.Module):
    """JEPA conditionnee sur action discrete (env grille 8x8)."""

    def __init__(self, latent_dim=32, action_emb_dim=8, n_actions=N_ACTIONS, ema_decay=0.99):
        super().__init__()
        self.ema_decay = ema_decay
        self.encoder_ctx = TinyEncoder(latent_dim, in_h=GRID_SIZE, in_w=GRID_SIZE)
        self.encoder_tgt = TinyEncoder(latent_dim, in_h=GRID_SIZE, in_w=GRID_SIZE)
        self.encoder_tgt.load_state_dict(self.encoder_ctx.state_dict())
        for p in self.encoder_tgt.parameters():
            p.requires_grad = False
        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim))

    @torch.no_grad()
    def update_target(self):
        for p_t, p_c in zip(self.encoder_tgt.parameters(), self.encoder_ctx.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1.0 - self.ema_decay)

    def predict_z(self, o_t, a):
        """o_t: (B, 1, 8, 8) ; a: (B,) long. Retourne z_hat (B, latent_dim)."""
        z = self.encoder_ctx(o_t)
        ae = self.action_emb(a)
        return self.predictor(torch.cat([z, ae], dim=-1))

    def forward(self, o_t, a, o_tp1):
        z_hat = self.predict_z(o_t, a)
        with torch.no_grad():
            z_tgt = self.encoder_tgt(o_tp1)
        return z_hat, z_tgt


class ActionCondPixelAE(nn.Module):
    """PixelAE action-conditionne (baseline pour planning pixel)."""

    def __init__(self, latent_dim=32, action_emb_dim=8, n_actions=N_ACTIONS):
        super().__init__()
        self.encoder = TinyEncoder(latent_dim, in_h=GRID_SIZE, in_w=GRID_SIZE)
        self.action_emb = nn.Embedding(n_actions, action_emb_dim)
        self.fuse = nn.Linear(latent_dim + action_emb_dim, latent_dim)
        self.decoder = TinyDecoder(latent_dim, out_h=GRID_SIZE, out_w=GRID_SIZE)

    def forward(self, o_t, a):
        z = self.encoder(o_t)
        ae = self.action_emb(a)
        z2 = self.fuse(torch.cat([z, ae], dim=-1))
        return self.decoder(z2)


def train_action_jepa(model, ot, a, otp1, n_epochs=60, batch_size=256, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    trainable = (list(model.encoder_ctx.parameters()) + list(model.predictor.parameters())
                 + list(model.action_emb.parameters()))
    optim = torch.optim.Adam(trainable, lr=lr)
    losses = []
    n = ot.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            z_hat, z_tgt = model(ot[b], a[b], otp1[b])
            loss = F.mse_loss(z_hat, z_tgt)
            optim.zero_grad(); loss.backward(); optim.step()
            model.update_target()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / max(nb, 1))
    return losses


def train_action_pixelae(model, ot, a, otp1, n_epochs=60, batch_size=256, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    n = ot.shape[0]
    for ep in range(n_epochs):
        idx = torch.randperm(n, generator=g)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            recon = model(ot[b], a[b])
            loss = F.mse_loss(recon, otp1[b])
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item(); nb += 1
        losses.append(ep_loss / max(nb, 1))
    return losses


def plan_action_jepa(model, o_t, o_goal):
    """Pick discrete action that minimises ||predictor(z_t, a) - z_goal||^2."""
    with torch.no_grad():
        z_goal = model.encoder_tgt(o_goal)            # encoder cible figee
        best_a, best_score = 0, float("inf")
        for a in range(N_ACTIONS):
            a_t = torch.tensor([a], dtype=torch.long)
            z_hat = model.predict_z(o_t, a_t)
            score = (z_hat - z_goal).pow(2).sum().item()
            if score < best_score:
                best_score = score; best_a = a
        return best_a


def plan_action_pixel(pae, o_t, o_goal):
    """Baseline: pick action whose decoded image is closest to goal image."""
    with torch.no_grad():
        best_a, best_score = 0, float("inf")
        for a in range(N_ACTIONS):
            a_t = torch.tensor([a], dtype=torch.long)
            recon = pae(o_t, a_t)
            score = (recon - o_goal).pow(2).sum().item()
            if score < best_score:
                best_score = score; best_a = a
        return best_a


def evaluate_planner(planner_fn, n_episodes=200, max_steps=10, seed=42):
    """planner_fn(o_t, o_goal) -> action int. Renvoie taux de succes."""
    g = torch.Generator().manual_seed(seed)
    rand_pos = torch.randint(0, GRID_SIZE, (n_episodes, 4), generator=g)
    successes = 0
    for ep in range(n_episodes):
        start = (int(rand_pos[ep, 0]), int(rand_pos[ep, 1]))
        goal = (int(rand_pos[ep, 2]), int(rand_pos[ep, 3]))
        if start == goal:
            successes += 1; continue
        cur = start
        for _ in range(max_steps):
            o_t = pos_to_obs(cur).unsqueeze(0)
            o_g = pos_to_obs(goal).unsqueeze(0)
            a = planner_fn(o_t, o_g)
            cur = grid_step(cur, a)
            if cur == goal:
                successes += 1; break
    return successes / n_episodes


def random_planner(o_t, o_goal):
    return int(torch.randint(0, N_ACTIONS, (1,)).item())


def hard():
    print("=" * 72)
    print("HARD - JEPA action-conditionnee + planning latent goal-image")
    print("=" * 72)
    torch.manual_seed(0)

    # 1. Dataset deterministe
    print("\n[1] Dataset deterministe 8x8 grid, 5 actions, 5000 transitions")
    ot, a, otp1 = make_video_dataset(n_samples=5000, seed=0, action_up_fail_p=0.0)

    # 2. Train JEPA
    print("[2] Train ActionCondJEPA (40 epochs)")
    jepa = ActionCondJEPA(latent_dim=32, ema_decay=0.99)
    j_losses = train_action_jepa(jepa, ot, a, otp1, n_epochs=40)
    print(f"   loss: start={j_losses[0]:.5f} -> end={j_losses[-1]:.5f}")

    # 3. Train PixelAE baseline
    print("[3] Train ActionCondPixelAE (40 epochs) - baseline pixel")
    pae = ActionCondPixelAE(latent_dim=32)
    p_losses = train_action_pixelae(pae, ot, a, otp1, n_epochs=40)
    print(f"   loss: start={p_losses[0]:.5f} -> end={p_losses[-1]:.5f}")

    # 4. Evaluate planners (deterministic env)
    print("\n[4] Planner evaluation (200 episodes, max 10 steps, deterministe)")
    rate_random = evaluate_planner(random_planner, n_episodes=200)
    rate_jepa = evaluate_planner(lambda o, g: plan_action_jepa(jepa, o, g), n_episodes=200)
    rate_pixel = evaluate_planner(lambda o, g: plan_action_pixel(pae, o, g), n_episodes=200)
    print(f"   random      success rate = {rate_random*100:.1f}%")
    print(f"   pixel plan  success rate = {rate_pixel*100:.1f}%")
    print(f"   JEPA  plan  success rate = {rate_jepa*100:.1f}%")

    # 5. Stress test: action up fails 30% of the time
    print("\n[5] Stress test stochastique (action up echoue 30% du temps)")
    ot_s, a_s, otp1_s = make_video_dataset(n_samples=5000, seed=1, action_up_fail_p=0.3)
    jepa_s = ActionCondJEPA(latent_dim=32, ema_decay=0.99)
    train_action_jepa(jepa_s, ot_s, a_s, otp1_s, n_epochs=40)
    pae_s = ActionCondPixelAE(latent_dim=32)
    train_action_pixelae(pae_s, ot_s, a_s, otp1_s, n_epochs=40)

    # mesurer variance pixel-wise des reconstructions
    with torch.no_grad():
        same_input = ot_s[:64]
        recon_var = []
        for ai in range(N_ACTIONS):
            a_t = torch.full((same_input.shape[0],), ai, dtype=torch.long)
            r = pae_s(same_input, a_t)
            recon_var.append(r.std().item())
    print(f"   PixelAE recon mean per-action std: {sum(recon_var)/len(recon_var):.4f}")
    print("   (sous stochasticite, le PixelAE moyenne les modes -> reconstructions floues")
    print("    sur l'action up; les autres actions restent nettes.)")

    rate_jepa_s = evaluate_planner(lambda o, g: plan_action_jepa(jepa_s, o, g), n_episodes=200)
    rate_pixel_s = evaluate_planner(lambda o, g: plan_action_pixel(pae_s, o, g), n_episodes=200)
    print(f"   JEPA  plan (stochastic): {rate_jepa_s*100:.1f}%")
    print(f"   Pixel plan (stochastic): {rate_pixel_s*100:.1f}%")
    print("   La JEPA reste plus stable car elle encode la structure spatiale, pas les")
    print("   modes pixel. Cela illustre miniature-style le pattern V-JEPA 2 (REF #21).")

    print("\n[Bonus] Pour brancher Cosmos (REF #22) en remplacement: utiliser")
    print("Cosmos-Tokenizer comme encoder pre-entraine, ne reapprendre que predictor + EMA.")


# =============================================================================
# Entry point
# =============================================================================

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "easy":
        easy()
    elif arg == "medium":
        medium()
    elif arg == "hard":
        hard()
    elif arg == "all":
        easy(); print(); medium(); print(); hard()
    else:
        print(f"unknown arg: {arg}. usage: easy | medium | hard | all")


if __name__ == "__main__":
    main()

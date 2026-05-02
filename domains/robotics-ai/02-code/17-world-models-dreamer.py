"""
J17 — World models / Dreamer-style RSSM toy implementation.

# requires: torch>=2.0 numpy

Goal: a *pedagogical* mini-Dreamer on a tiny gridworld so the RSSM is readable
and testable on CPU in <30s. We do NOT aim at SOTA — we want each piece of the
DreamerV3 architecture (Hafner 2023, REFERENCES.md #20) to be visible:

    encoder       -> compresses obs into embedding
    RSSM cell     -> deterministic h_t (GRU) + stochastic z_t (categorical)
    decoder       -> reconstructs obs (training signal)
    reward head   -> predicts r_t from latent
    continue head -> predicts whether episode goes on

We also show:
    * KL(prior || posterior) loss as it's used in Dreamer (with KL balancing)
    * symlog transform from DreamerV3
    * a 5-step *latent imagination* rollout
    * a tiny actor-critic update on imagined trajectories

The env is a 5x5 gridworld where the agent must reach the goal. Obs = one-hot
of the position (25 dims). It's intentionally trivial so the world model has
something to fit, not so hard it requires real Dreamer scale.

Run:
    python 17-world-models-dreamer.py
Expected:
    - reconstruction loss decreases
    - reward prediction loss decreases
    - imagination rollout produces coherent latent trajectory
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Tiny gridworld env (no gymnasium dep — keeps the code self-contained)
# -----------------------------------------------------------------------------
class GridWorld:
    """5x5 gridworld. Actions: 0=up 1=right 2=down 3=left. Reward 1 at goal."""

    def __init__(self, size: int = 5, max_steps: int = 25) -> None:
        self.size = size
        self.max_steps = max_steps
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self) -> np.ndarray:
        self.x, self.y = 0, 0  # always start top-left for determinism
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        # one-hot encoding over the 25 cells; this is what the encoder eats
        obs = np.zeros(self.size * self.size, dtype=np.float32)
        obs[self.x * self.size + self.y] = 1.0
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        # standard 4-direction move with wall-clipping
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        self.x = max(0, min(self.size - 1, self.x + dx))
        self.y = max(0, min(self.size - 1, self.y + dy))
        self.t += 1
        done = (self.x, self.y) == self.goal or self.t >= self.max_steps
        # sparse reward: 1.0 only at goal — RSSM's reward head will learn this
        reward = 1.0 if (self.x, self.y) == self.goal else 0.0
        return self._obs(), reward, done


# -----------------------------------------------------------------------------
# 2. DreamerV3 helpers
# -----------------------------------------------------------------------------
def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog transform — handles arbitrary reward magnitudes."""
    # sign(x) * log(|x| + 1) — smooth, monotonic, identity around 0
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog — used to decode predicted reward back to env scale."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# -----------------------------------------------------------------------------
# 3. RSSM cell — the heart of Dreamer
# -----------------------------------------------------------------------------
class RSSMCell(nn.Module):
    """
    Recurrent State-Space Model cell (Hafner 2019, kept in V2/V3).

    State is dual:
        h_t : deterministic, carries long-term memory (GRU hidden state)
        z_t : stochastic, captures uncertainty
              -> in DreamerV3, z_t is *categorical*: n_categories x n_classes
                 with straight-through gradient. We mimic that here in a small
                 way (8 categories x 8 classes = 64-dim flattened latent).

    Two distributions on z_t:
        prior     p(z_t | h_t)         — used at imagination time
        posterior q(z_t | h_t, e_t)    — used at training time (sees obs)
    KL(posterior || prior) regularizes them to agree.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int = 64,
        n_cat: int = 8,
        n_cls: int = 8,
    ) -> None:
        super().__init__()
        self.n_cat = n_cat
        self.n_cls = n_cls
        self.z_dim = n_cat * n_cls  # flat representation of categorical z
        self.hidden_dim = hidden_dim

        # GRU drives h_t from previous (z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(self.z_dim + action_dim, hidden_dim)

        # prior: predicts categorical logits over z given only h_t
        # (no observation — this is the imagination-time path)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

        # posterior: predicts categorical logits over z given h_t + obs embedding
        # (sees reality — this is the training-time encoding path)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

    def _sample_categorical(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample z from categorical(logits) with straight-through estimator.
        Returns (sample_one_hot_flat, logits_reshaped_for_kl).
        """
        b = logits.size(0)
        # reshape to (batch, n_cat, n_cls) so each n_cat slot is its own dist
        logits_r = logits.view(b, self.n_cat, self.n_cls)
        probs = F.softmax(logits_r, dim=-1)
        # gumbel-style sampling, but simpler: take categorical sample
        dist = torch.distributions.Categorical(probs=probs)
        sample = dist.sample()  # (b, n_cat) integer indices
        one_hot = F.one_hot(sample, num_classes=self.n_cls).float()  # (b, n_cat, n_cls)
        # straight-through: forward = one_hot, backward = probs
        # this lets gradient flow through the discrete sample
        st = one_hot + probs - probs.detach()
        return st.view(b, -1), logits_r

    def imagine_step(
        self, h_prev: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagination step (no obs): h <- GRU; z <- prior."""
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        prior_logits = self.prior_net(h)
        z, prior_logits_r = self._sample_categorical(prior_logits)
        return h, z, prior_logits_r

    def observe_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One training step (sees obs): emits both prior and posterior for KL."""
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        prior_logits = self.prior_net(h)
        post_logits = self.posterior_net(torch.cat([h, embed], dim=-1))
        # we sample z from the posterior at training (richer info)
        z, post_logits_r = self._sample_categorical(post_logits)
        prior_logits_r = prior_logits.view(-1, self.n_cat, self.n_cls)
        return h, z, prior_logits_r, post_logits_r


# -----------------------------------------------------------------------------
# 4. Encoder, decoder, heads
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    """obs (one-hot 25) -> embedding (32)."""

    def __init__(self, obs_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """latent (h+z) -> reconstructed obs logits."""

    def __init__(self, latent_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ELU(), nn.Linear(64, obs_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    """Generic MLP head used for reward, continue, value."""

    def __init__(self, latent_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ELU(), nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# 5. Full WorldModel module
# -----------------------------------------------------------------------------
@dataclass
class Config:
    obs_dim: int = 25
    action_dim: int = 4
    embed_dim: int = 32
    hidden_dim: int = 64
    n_cat: int = 8
    n_cls: int = 8
    imagination_horizon: int = 5
    kl_weight: float = 1.0
    free_bits: float = 1.0  # DreamerV3 free-bits to prevent posterior collapse


class WorldModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.obs_dim, cfg.embed_dim)
        self.rssm = RSSMCell(cfg.action_dim, cfg.embed_dim, cfg.hidden_dim, cfg.n_cat, cfg.n_cls)
        latent_dim = cfg.hidden_dim + cfg.n_cat * cfg.n_cls  # h ++ z
        self.decoder = Decoder(latent_dim, cfg.obs_dim)
        self.reward_head = Head(latent_dim, 1)
        self.continue_head = Head(latent_dim, 1)
        self.actor = Head(latent_dim, cfg.action_dim)
        self.critic = Head(latent_dim, 1)

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.cfg.hidden_dim, device=device)
        z = torch.zeros(batch, self.cfg.n_cat * self.cfg.n_cls, device=device)
        return h, z

    def world_loss(
        self,
        obs_seq: torch.Tensor,    # (T, B, obs_dim)
        action_seq: torch.Tensor, # (T, B, action_dim) one-hot
        reward_seq: torch.Tensor, # (T, B)
        done_seq: torch.Tensor,   # (T, B)
    ) -> dict[str, torch.Tensor]:
        """Compute Dreamer-style world model loss on a trajectory batch."""
        T, B = obs_seq.shape[:2]
        device = obs_seq.device

        h, z = self.initial_state(B, device)
        recon_loss = 0.0
        reward_loss = 0.0
        cont_loss = 0.0
        kl_loss = 0.0

        for t in range(T):
            embed = self.encoder(obs_seq[t])
            h, z, prior_logits, post_logits = self.rssm.observe_step(
                h, z, action_seq[t], embed
            )
            latent = torch.cat([h, z], dim=-1)

            # reconstruction loss — softmax cross-entropy (obs is one-hot)
            recon_logits = self.decoder(latent)
            recon_loss = recon_loss + F.cross_entropy(
                recon_logits, obs_seq[t].argmax(dim=-1), reduction="mean"
            )

            # reward — symlog target, MSE on symlog space (DreamerV3)
            r_pred = self.reward_head(latent).squeeze(-1)
            r_tgt = symlog(reward_seq[t])
            reward_loss = reward_loss + F.mse_loss(r_pred, r_tgt)

            # continue (1 - done) — BCE
            c_pred = self.continue_head(latent).squeeze(-1)
            c_tgt = 1.0 - done_seq[t]
            cont_loss = cont_loss + F.binary_cross_entropy_with_logits(c_pred, c_tgt)

            # KL between posterior and prior over categorical distributions
            # KL balancing (DreamerV3): we mostly pull prior towards posterior,
            # not the other way around. Implemented as average of two terms with
            # different stop-gradients.
            post_dist = torch.distributions.Categorical(logits=post_logits)
            prior_dist = torch.distributions.Categorical(logits=prior_logits)
            # train prior to match posterior (sg on posterior)
            kl_a = torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(logits=post_logits.detach()),
                prior_dist,
            ).sum(-1).mean()
            # train posterior to be close to prior (sg on prior)
            kl_b = torch.distributions.kl.kl_divergence(
                post_dist,
                torch.distributions.Categorical(logits=prior_logits.detach()),
            ).sum(-1).mean()
            # free-bits floor: don't penalize KL below the threshold
            kl_t = 0.8 * torch.clamp(kl_a, min=self.cfg.free_bits) + 0.2 * torch.clamp(
                kl_b, min=self.cfg.free_bits
            )
            kl_loss = kl_loss + kl_t

        # average over time so loss magnitude is independent of T
        recon_loss = recon_loss / T
        reward_loss = reward_loss / T
        cont_loss = cont_loss / T
        kl_loss = kl_loss / T

        total = recon_loss + reward_loss + cont_loss + self.cfg.kl_weight * kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "continue": cont_loss,
            "kl": kl_loss,
        }

    def imagine(
        self, h: torch.Tensor, z: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Roll out `horizon` steps in the latent imagination using the *prior*.
        At each step, sample action from the actor head and step the prior.
        Returns (latents, rewards_pred, values_pred) for actor-critic training.
        """
        latents = []
        rewards = []
        values = []
        for _ in range(horizon):
            latent = torch.cat([h, z], dim=-1)
            latents.append(latent)
            # actor outputs action logits — categorical over discrete actions
            logits = self.actor(latent)
            probs = F.softmax(logits, dim=-1)
            a_idx = torch.distributions.Categorical(probs=probs).sample()
            a_onehot = F.one_hot(a_idx, num_classes=self.cfg.action_dim).float()
            # predict reward from current latent (in symlog space, decode back)
            r_pred = symexp(self.reward_head(latent).squeeze(-1))
            rewards.append(r_pred)
            values.append(self.critic(latent).squeeze(-1))
            # advance imagination one step
            h, z, _ = self.rssm.imagine_step(h, z, a_onehot)
        return torch.stack(latents), torch.stack(rewards), torch.stack(values)


# -----------------------------------------------------------------------------
# 6. Training harness — collect, train world model, train AC in imagination
# -----------------------------------------------------------------------------
def collect_episode(env: GridWorld, cfg: Config) -> dict[str, np.ndarray]:
    """Collect one episode with a uniformly random policy.

    For a toy env with sparse reward, random exploration is enough to get a
    handful of goal hits per 100 episodes — that's all the world model needs."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs = env.reset()
    done = False
    while not done:
        a = random.randint(0, cfg.action_dim - 1)
        obs_list.append(obs)
        a_oh = np.zeros(cfg.action_dim, dtype=np.float32)
        a_oh[a] = 1.0
        act_list.append(a_oh)
        obs, r, done = env.step(a)
        rew_list.append(r)
        done_list.append(float(done))
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "act": np.array(act_list, dtype=np.float32),
        "rew": np.array(rew_list, dtype=np.float32),
        "done": np.array(done_list, dtype=np.float32),
    }


def stack_batch(eps: list[dict[str, np.ndarray]], T: int) -> tuple[torch.Tensor, ...]:
    """Pad/truncate episodes to length T and stack as (T, B, ...)."""
    B = len(eps)
    obs = np.zeros((T, B, eps[0]["obs"].shape[1]), dtype=np.float32)
    act = np.zeros((T, B, eps[0]["act"].shape[1]), dtype=np.float32)
    rew = np.zeros((T, B), dtype=np.float32)
    done = np.zeros((T, B), dtype=np.float32)
    for j, ep in enumerate(eps):
        L = min(T, len(ep["obs"]))
        obs[:L, j] = ep["obs"][:L]
        act[:L, j] = ep["act"][:L]
        rew[:L, j] = ep["rew"][:L]
        done[:L, j] = ep["done"][:L]
        if L < T:
            done[L - 1 :, j] = 1.0
    return (
        torch.from_numpy(obs),
        torch.from_numpy(act),
        torch.from_numpy(rew),
        torch.from_numpy(done),
    )


def actor_critic_imagination_loss(
    wm: WorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """
    Train actor-critic in imagination: starting from real observed states,
    roll out `horizon` steps under the world model and use the imagined
    rewards + bootstrapped values as a TD target.
    """
    T, B = obs.shape[:2]
    device = obs.device
    h, z = wm.initial_state(B, device)
    # encode the first observation to get a realistic starting latent
    embed = wm.encoder(obs[0])
    h, z, _, _ = wm.rssm.observe_step(h, z, actions[0], embed)

    latents, rewards, values = wm.imagine(h.detach(), z.detach(), horizon)
    # naive lambda=0 TD target: G_t = r_t + gamma * V_{t+1}
    gamma = 0.99
    with torch.no_grad():
        bootstrap = values[-1]
        targets = []
        running = bootstrap
        for t in reversed(range(horizon)):
            running = rewards[t] + gamma * running
            targets.insert(0, running)
        target_v = torch.stack(targets)

    critic_loss = F.mse_loss(values, target_v)
    # actor loss: maximize predicted return = minimize -V
    # (in the real Dreamer this uses reparam through the actor entropy bonus,
    # we keep it simple here — the toy env doesn't need more)
    actor_loss = -values.mean()
    return critic_loss + actor_loss


def train(cfg: Config = Config(), n_iters: int = 80, batch: int = 8, seq_len: int = 12) -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    env = GridWorld()
    wm = WorldModel(cfg).to(device)
    opt_world = torch.optim.Adam(
        list(wm.encoder.parameters())
        + list(wm.rssm.parameters())
        + list(wm.decoder.parameters())
        + list(wm.reward_head.parameters())
        + list(wm.continue_head.parameters()),
        lr=3e-4,
    )
    opt_ac = torch.optim.Adam(
        list(wm.actor.parameters()) + list(wm.critic.parameters()), lr=3e-4
    )

    history = {"recon": [], "reward": [], "kl": [], "ac": []}
    for it in range(n_iters):
        # 1. collect a fresh batch of episodes (random policy — pedagogical)
        eps = [collect_episode(env, cfg) for _ in range(batch)]
        obs, act, rew, done = stack_batch(eps, seq_len)
        obs, act, rew, done = obs.to(device), act.to(device), rew.to(device), done.to(device)

        # 2. world model update
        losses = wm.world_loss(obs, act, rew, done)
        opt_world.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt_world.step()

        # 3. actor-critic imagination update (every step here for simplicity)
        ac_loss = actor_critic_imagination_loss(wm, obs, act, cfg.imagination_horizon)
        opt_ac.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt_ac.step()

        history["recon"].append(losses["recon"].item())
        history["reward"].append(losses["reward"].item())
        history["kl"].append(losses["kl"].item())
        history["ac"].append(ac_loss.item())

        if it % 10 == 0 or it == n_iters - 1:
            print(
                f"iter {it:3d} | recon {losses['recon'].item():.3f}"
                f" | reward {losses['reward'].item():.4f}"
                f" | kl {losses['kl'].item():.3f}"
                f" | ac {ac_loss.item():.3f}"
            )

    # sanity check: reconstruction loss should have decreased meaningfully
    early = sum(history["recon"][:5]) / 5
    late = sum(history["recon"][-5:]) / 5
    print(
        f"\nSanity: avg recon early={early:.3f} -> late={late:.3f}"
        f" ({'OK' if late < early else 'STAGNANT'})"
    )

    # demo: roll out 5 steps of latent imagination from a fresh start
    print("\n--- imagination rollout (5 steps from initial state) ---")
    with torch.no_grad():
        h, z = wm.initial_state(1, device)
        latents, rewards, values = wm.imagine(h, z, horizon=5)
        for t, (r, v) in enumerate(zip(rewards.squeeze(1), values.squeeze(1))):
            print(f"  step {t}: imagined reward={r.item():+.3f} value={v.item():+.3f}")


if __name__ == "__main__":
    train()

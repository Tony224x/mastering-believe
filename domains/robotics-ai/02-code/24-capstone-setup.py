# requires: numpy>=1.24 matplotlib>=3.7
"""
J24 - Capstone Day 1 - PushT environment + scripted expert + dataset generator.

This file is the *data producer* for the 5-day capstone (J24-J28). It builds a
toy PushT environment in pure numpy (no MuJoCo, no pymunk, no GPU), runs a
geometric scripted expert to generate ~200 successful demos, and writes them
to disk in a format that J25-J28 will consume directly.

What this file produces (the contract for downstream days):
    artifacts/pusht_demos/
        meta.json          # high-level config (fps, dims, n_episodes, ...)
        episodes.jsonl     # one JSON dict per episode with metadata
        data.npz           # the actual tensors:
            obs       (T_total, 5)   float32  [agent_x, agent_y, block_x, block_y, block_theta]
            action    (T_total, 2)   float32  [delta_x, delta_y]  in pixels per step
            ep_id     (T_total,)     int32    episode index for each transition
            ep_start  (N_ep,)        int32    starting offset of each episode
            ep_length (N_ep,)        int32    length of each episode

Run:
    python domains/robotics-ai/02-code/24-capstone-setup.py
    python domains/robotics-ai/02-code/24-capstone-setup.py --plot   # also save 6 trajectories

Source references:
    REFERENCES.md #19 -- Chi et al., Diffusion Policy, RSS 2023 (pusht_env.py).
    REFERENCES.md #27 -- LeRobotDataset v3.0 (schema-first dataset philosophy).
    REFERENCES.md #24 -- MuJoCo docs (context: skipped here, 2D collision is enough).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================================
# 1. World configuration
# ============================================================================
# PushT lives on a 512x512 px table (same as Chi 2023). The T-block is built
# from two axis-aligned rectangles: a horizontal bar (top of the T) and a
# vertical bar (stem of the T). The agent is a circle that pushes the block.
#
# We keep everything in pixels so the magnitudes match the original repo,
# which makes hyperparameters (learning rate, action scale) carry over to
# J25-J26 without ad-hoc rescaling.


@dataclass
class WorldConfig:
    table_size: int = 512                  # square table, px
    agent_radius: float = 15.0             # pusher radius, px
    block_bar_w: float = 120.0             # T horizontal bar width
    block_bar_h: float = 30.0              # T horizontal bar height
    block_stem_w: float = 30.0             # T vertical stem width
    block_stem_h: float = 90.0             # T vertical stem height (below bar)
    target_pos: tuple = (256.0, 256.0)     # T target center (top of bar)
    target_theta: float = 0.0              # target orientation (radians)
    a_max: float = 12.0                    # max |action| in pixels per step
    iou_success: float = 0.50              # success threshold (lowered from
                                           # the original 0.95 of Chi 2023:
                                           # our 2D kinematic dynamics is much
                                           # less precise than the pymunk-based
                                           # PushT, so we relax the bar to
                                           # something the heuristic expert
                                           # can hit reliably. The downstream
                                           # policy is evaluated against the
                                           # *same* threshold so the
                                           # comparison stays internally
                                           # consistent.)
    max_steps: int = 200
    fps: int = 10                          # nominal control frequency


# ============================================================================
# 2. Geometry helpers (pure numpy)
# ============================================================================
# We do NOT use shapely or pymunk: we hand-roll the few primitives we need.
# IoU is computed on a 64x64 rasterization (cheap, deterministic, easy to
# reason about). It is plenty accurate for a 0.85 threshold.

def _rasterize_t(cx: float, cy: float, theta: float, cfg: WorldConfig,
                 res: int = 64) -> np.ndarray:
    """Return a (res, res) bool mask of the T-block centered at (cx, cy).

    The T is drawn in object-local frame as the union of two rectangles, then
    rotated by `theta` and translated by (cx, cy). We sample the centers of
    each pixel and test point-in-rectangle in object frame.
    """
    # World-to-image scale: image pixel index -> world (x, y).
    cell = cfg.table_size / res
    ix, iy = np.meshgrid(np.arange(res), np.arange(res), indexing="xy")
    wx = (ix + 0.5) * cell - cx
    wy = (iy + 0.5) * cell - cy
    # Rotate world->object frame by -theta.
    cos_t, sin_t = math.cos(-theta), math.sin(-theta)
    ox = cos_t * wx - sin_t * wy
    oy = sin_t * wx + cos_t * wy

    # Bar: centered at object origin (0, 0), height block_bar_h.
    in_bar = (
        (np.abs(ox) <= cfg.block_bar_w / 2.0)
        & (np.abs(oy) <= cfg.block_bar_h / 2.0)
    )
    # Stem: centered below the bar, top at oy = block_bar_h/2.
    stem_cy = cfg.block_bar_h / 2.0 + cfg.block_stem_h / 2.0
    in_stem = (
        (np.abs(ox) <= cfg.block_stem_w / 2.0)
        & (np.abs(oy - stem_cy) <= cfg.block_stem_h / 2.0)
    )
    return in_bar | in_stem


def iou_t(block_pos: np.ndarray, block_theta: float, cfg: WorldConfig,
          res: int = 64) -> float:
    """IoU between current block pose and target pose."""
    a = _rasterize_t(block_pos[0], block_pos[1], block_theta, cfg, res)
    b = _rasterize_t(cfg.target_pos[0], cfg.target_pos[1], cfg.target_theta,
                     cfg, res)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# ============================================================================
# 3. Toy 2D PushT environment
# ============================================================================
# Dynamics summary:
#   - At each step, the agent receives a desired displacement action (dx, dy),
#     clipped to ||a|| <= a_max.
#   - We move the agent by that delta. If the agent ends up overlapping the
#     T-block, we apply a *kinematic push*: the block translates by the same
#     delta projected onto the agent->block axis, with a small rotational
#     coupling proportional to the lever arm.
#   - This is much simpler than rigid body, but reproduces the *qualitative*
#     behavior PushT exposes (push direction matters, the block can rotate
#     when pushed off-center). For a pedagogical pipeline, that is all we
#     need: the multimodality of expert demos comes from the *strategy*
#     (which side to approach), not from delicate physics.


class PushTEnv:
    """Minimal pure-numpy PushT environment."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.block_pos = np.zeros(2, dtype=np.float32)
        self.block_theta = 0.0
        self._steps = 0
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Seed the env's internal RNG (used by reset)."""
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        S = self.cfg.table_size
        # Random block pose, with safety margin from walls.
        margin = 100.0
        self.block_pos = self._rng.uniform(margin, S - margin, size=2).astype(np.float32)
        self.block_theta = float(self._rng.uniform(-math.pi / 6, math.pi / 6))
        # Agent placed away from the block (so the expert has to plan a contact).
        for _ in range(50):
            cand = self._rng.uniform(margin, S - margin, size=2).astype(np.float32)
            if np.linalg.norm(cand - self.block_pos) > 80.0:
                self.agent_pos = cand
                break
        self._steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.block_pos[0], self.block_pos[1],
            self.block_theta,
        ], dtype=np.float32)

    def _clip_action(self, a: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(a))
        if n > self.cfg.a_max:
            a = a * (self.cfg.a_max / n)
        return a.astype(np.float32)

    def step(self, action: np.ndarray):
        """Advance one step. Returns (obs, reward, done, info)."""
        a = self._clip_action(np.asarray(action, dtype=np.float32))
        # Move agent.
        self.agent_pos = self.agent_pos + a
        # Clip agent to table.
        self.agent_pos = np.clip(self.agent_pos, self.cfg.agent_radius,
                                 self.cfg.table_size - self.cfg.agent_radius)
        # Detect contact with block (cheap circle vs AABB-of-T).
        # We approximate the T-block by its tight bounding box in world
        # coordinates for the contact test only. This is intentionally
        # crude -- enough to make the block move when the agent is "behind".
        b_extent = max(self.cfg.block_bar_w,
                       self.cfg.block_stem_h + self.cfg.block_bar_h) / 2.0
        d_vec = self.agent_pos - self.block_pos
        dist = float(np.linalg.norm(d_vec))
        if dist < (self.cfg.agent_radius + b_extent):
            # Push: translate block by the projection of action onto (block - agent).
            push_dir = -d_vec  # from agent toward block center
            push_norm = float(np.linalg.norm(push_dir))
            if push_norm > 1e-6:
                push_unit = push_dir / push_norm
                # Translation: only the component of action aligned with push_unit.
                a_along = float(np.dot(a, push_unit))
                if a_along > 0.0:
                    # Effective transfer ratio: 0.95 of the aligned action
                    # propagates to the block (very plastic kinematic contact).
                    self.block_pos = self.block_pos + push_unit * a_along * 0.95
                    # Rotation coupling: small torque proportional to lever arm
                    # (cross product magnitude between contact offset and action).
                    lever = float(d_vec[0] * a[1] - d_vec[1] * a[0]) / max(b_extent, 1.0)
                    self.block_theta += 0.0015 * lever
        # Clip block to table.
        self.block_pos = np.clip(self.block_pos, 50.0, self.cfg.table_size - 50.0)
        self._steps += 1
        iou = iou_t(self.block_pos, self.block_theta, self.cfg)
        done = (iou >= self.cfg.iou_success) or (self._steps >= self.cfg.max_steps)
        return self.observation(), float(iou), bool(done), {"iou": iou}


# ============================================================================
# 4. Scripted expert
# ============================================================================
# Two-phase heuristic: (1) circle around the block to a contact point on the
# *chosen side*, (2) push toward the target. The chosen side is randomized
# per-episode to inject the multimodality that makes Diffusion Policy
# necessary in the first place (see theory key takeaway #2).


def expert_step(obs: np.ndarray, cfg: WorldConfig, side: int,
                phase: list, rng: np.random.Generator) -> np.ndarray:
    """Return one expert action given the current observation.

    Three-phase heuristic:
        1. "stage"  : navigate to a *staging point* placed FAR behind the
                      block (offset to the chosen side) so the straight-line
                      path never goes through the block.
        2. "approach": close in on the contact point from the staging side.
        3. "push"   : keep pushing along the (target - block) direction.

    Args:
        obs: (5,) [agent_x, agent_y, block_x, block_y, block_theta].
        cfg: WorldConfig.
        side: -1 for left contour, +1 for right contour.
        phase: mutable single-element list holding the current phase
            ("stage", "approach", or "push"). Passed as a list so the
            caller can keep track across env steps without a class.
        rng: numpy RNG for action noise.
    """
    agent = obs[:2]
    block = obs[2:4]
    target = np.asarray(cfg.target_pos, dtype=np.float32)

    push_vec = target - block
    push_norm = float(np.linalg.norm(push_vec))
    if push_norm < 1e-3:
        return np.zeros(2, dtype=np.float32)
    push_unit = push_vec / push_norm
    perp = np.array([-push_unit[1], push_unit[0]], dtype=np.float32)  # +90 deg

    # Staging: 110 px behind block + 60 px lateral on chosen side. This is
    # well outside the block bounding box so a straight approach never
    # collides en route.
    staging = block - push_unit * 110.0 + perp * (side * 60.0)
    # Final contact point: 55 px behind block, smaller lateral offset.
    contact = block - push_unit * 55.0 + perp * (side * 18.0)

    if phase[0] == "stage":
        delta = staging - agent
        if float(np.linalg.norm(delta)) < 25.0:
            phase[0] = "approach"
        a = delta
    elif phase[0] == "approach":
        delta = contact - agent
        if float(np.linalg.norm(delta)) < 14.0:
            phase[0] = "push"
        a = delta
    else:  # push phase
        # Push along push_unit at full speed, with a lateral correction
        # term that keeps the agent aligned with the contact point if the
        # block drifts. We do NOT scale down near the goal: the env's
        # `done` condition triggers automatically once IoU >= threshold,
        # so overshoot is naturally prevented at the episode level.
        a = push_unit * cfg.a_max + 0.4 * (contact - agent)

    # Inject mild gaussian noise BEFORE the final clip -- diversifies the
    # demos while keeping the saved action strictly inside the action bound.
    a = a + rng.normal(0.0, 0.5, size=2).astype(np.float32)
    n = float(np.linalg.norm(a))
    if n > cfg.a_max:
        a = a * (cfg.a_max / n)
    return a.astype(np.float32)


def collect_episode(env: PushTEnv, cfg: WorldConfig, side: int,
                    rng: np.random.Generator, seed: int):
    """Collect one episode under the scripted expert.

    Returns (obs_arr, action_arr, success, length).
    """
    obs = env.reset(seed=seed)
    obs_buf, act_buf = [obs.copy()], []
    phase = ["stage"]
    success = False
    for _ in range(cfg.max_steps):
        a = expert_step(obs, cfg, side, phase, rng)
        obs, _r, done, info = env.step(a)
        act_buf.append(a)
        obs_buf.append(obs.copy())
        if done and info["iou"] >= cfg.iou_success:
            success = True
            break
        if done:
            break
    # `obs_buf` has T+1 entries (including final), `act_buf` has T.
    # We align to T transitions (obs_t, action_t) by dropping the last obs.
    obs_arr = np.stack(obs_buf[:-1], axis=0).astype(np.float32)
    act_arr = np.stack(act_buf, axis=0).astype(np.float32)
    return obs_arr, act_arr, success, len(act_buf)


# ============================================================================
# 5. Dataset writer
# ============================================================================


@dataclass
class DatasetMeta:
    n_episodes: int
    n_transitions: int
    obs_dim: int
    action_dim: int
    fps: int
    max_episode_len: int
    iou_success: float
    a_max: float
    table_size: int
    seed: int
    notes: str = field(default="J24 capstone -- 2D toy PushT, scripted expert")


def save_dataset(out_dir: Path, episodes: list, cfg: WorldConfig, seed: int) -> DatasetMeta:
    """Persist a list of (obs_arr, action_arr, success, side) episodes to disk.

    Layout matches the contract documented at the top of this file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all episodes into flat arrays + offset/length tables.
    obs_chunks, act_chunks, ep_id_chunks = [], [], []
    ep_start, ep_length = [], []
    cursor = 0
    episodes_jsonl = []
    for i, (obs_arr, act_arr, success, side) in enumerate(episodes):
        n = act_arr.shape[0]
        obs_chunks.append(obs_arr)
        act_chunks.append(act_arr)
        ep_id_chunks.append(np.full((n,), i, dtype=np.int32))
        ep_start.append(cursor)
        ep_length.append(n)
        cursor += n
        episodes_jsonl.append({
            "episode_index": i,
            "length": int(n),
            "success": bool(success),
            "side": "left" if side == -1 else "right",
        })

    obs_all = np.concatenate(obs_chunks, axis=0)
    act_all = np.concatenate(act_chunks, axis=0)
    ep_id = np.concatenate(ep_id_chunks, axis=0)
    ep_start_arr = np.asarray(ep_start, dtype=np.int32)
    ep_length_arr = np.asarray(ep_length, dtype=np.int32)

    np.savez(
        out_dir / "data.npz",
        obs=obs_all,
        action=act_all,
        ep_id=ep_id,
        ep_start=ep_start_arr,
        ep_length=ep_length_arr,
    )

    meta = DatasetMeta(
        n_episodes=len(episodes),
        n_transitions=int(obs_all.shape[0]),
        obs_dim=int(obs_all.shape[1]),
        action_dim=int(act_all.shape[1]),
        fps=cfg.fps,
        max_episode_len=cfg.max_steps,
        iou_success=cfg.iou_success,
        a_max=cfg.a_max,
        table_size=cfg.table_size,
        seed=seed,
    )
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep) + "\n")

    return meta


# ============================================================================
# 6. Main pipeline
# ============================================================================


def generate_dataset(target_n: int = 100, seed: int = 0,
                     out_dir: Optional[Path] = None) -> tuple:
    """Generate `target_n` successful demos with rejection sampling.

    Returns (meta, episodes) so the caller can also plot the trajectories.
    """
    if out_dir is None:
        out_dir = Path("artifacts/pusht_demos")
    cfg = WorldConfig()
    env = PushTEnv(cfg)
    rng = np.random.default_rng(seed)

    episodes = []
    attempts = 0
    # The toy kinematic dynamics has a ~30-40% success rate per attempt,
    # so we budget 6x rejection sampling to safely hit `target_n`.
    max_attempts = target_n * 6
    while len(episodes) < target_n and attempts < max_attempts:
        attempts += 1
        side = -1 if rng.random() < 0.5 else +1   # randomize the multimodal side
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs_arr, act_arr, success, length = collect_episode(env, cfg, side, rng, ep_seed)
        if success and length >= 5:
            episodes.append((obs_arr, act_arr, success, side))
    success_rate = len(episodes) / max(attempts, 1)
    print(f"[generate] kept {len(episodes)}/{attempts} candidates "
          f"(success rate = {success_rate:.1%})")

    meta = save_dataset(out_dir, episodes, cfg, seed)
    print(f"[generate] wrote {meta.n_episodes} episodes "
          f"({meta.n_transitions} transitions) to {out_dir}")
    return meta, episodes


def maybe_plot(episodes: list, out_dir: Path, n_to_plot: int = 6) -> None:
    """If matplotlib is available, save a quick visual sanity check.

    Plots `n_to_plot` agent + block trajectories on a single canvas. We skip
    silently if matplotlib import fails (so the file remains usable on
    minimal environments)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("[plot] matplotlib unavailable -- skipping figure")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, 512); ax.set_ylim(0, 512); ax.set_aspect("equal")
    ax.set_title(f"PushT scripted demos (first {n_to_plot} episodes)")
    colors = plt.cm.tab10.colors
    for i, (obs_arr, _act_arr, success, side) in enumerate(episodes[:n_to_plot]):
        c = colors[i % len(colors)]
        ax.plot(obs_arr[:, 0], obs_arr[:, 1], color=c, alpha=0.7,
                label=f"ep{i} side={'L' if side == -1 else 'R'} succ={success}")
        ax.plot(obs_arr[:, 2], obs_arr[:, 3], color=c, ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")
    out_path = out_dir / "trajectories.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="target number of successful demos (default 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="artifacts/pusht_demos")
    parser.add_argument("--plot", action="store_true",
                        help="also save trajectories.png in out_dir")
    args = parser.parse_args()

    out_dir = Path(args.out)
    meta, episodes = generate_dataset(target_n=args.n, seed=args.seed, out_dir=out_dir)

    # Quick sanity report on what we just wrote -- the kind of diagnostics
    # the J25 dataloader will expect.
    print("\n=== dataset summary ===")
    print(json.dumps(asdict(meta), indent=2))
    lengths = [act.shape[0] for (_o, act, _s, _side) in episodes]
    sides = [side for (_o, _a, _s, side) in episodes]
    print(f"episode length  mean={np.mean(lengths):.1f} "
          f"min={min(lengths)} max={max(lengths)}")
    print(f"side balance    left={sides.count(-1)}  right={sides.count(+1)}")

    if args.plot:
        maybe_plot(episodes, out_dir)


if __name__ == "__main__":
    main()

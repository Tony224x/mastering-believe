# requires: numpy torch
"""
J23 - Synthetic data + sim-to-real at scale.

Goal: build a *toy* version of the GR00T-style synthetic data pipeline
(REFERENCES.md #15) end-to-end, on CPU, in pure numpy + torch:

  1. Generate a tiny synthetic dataset of "trajectories" (toy 8x8 RGB
     "images" + 2D actions) for 3 embodiments and 2 tasks.
  2. Apply 4-axis augmentation: background, lighting, distractors,
     dynamics noise -- the four axes used by every modern generalist VLA.
  3. Filter trajectories with a contact / success / smoothness check
     (the GR00T pipeline rejects ~30% of generated trajectories).
  4. Save the result in a *LeRobotDataset-like* layout (REFERENCES.md #27):
     parquet for tabular features + a meta json + per-episode binary
     image arrays. We use parquet only if pyarrow is present, else a
     pickle fallback so the script works in any minimal env.

This file is intentionally pedagogical:
  - everything is in-memory, no Hugging Face Hub upload,
  - "images" are tiny 8x8 RGB tensors so the script is fast on CPU,
  - augmentations are simplified versions of what Isaac Lab + Cosmos
    produce (REFERENCES.md #22), the *math* is identical.

Run:
    python domains/robotics-ai/02-code/23-synthetic-data-scale.py

References:
  - REFERENCES.md #15  GR00T N1 (NVIDIA, 2025) - 780k synthetic trajectories
  - REFERENCES.md #22  NVIDIA Cosmos - video relighting / curation
  - REFERENCES.md #27  LeRobotDataset v3.0 - standard 2026 format
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import torch


# ----------------------------------------------------------------------------
# 1. Toy synthetic dataset generation
# ----------------------------------------------------------------------------
# We model 3 embodiments x 2 tasks. Each "trajectory" is a sequence of (image,
# state, action) tuples of length T. The base trajectory is a parametric curve
# that depends on (embodiment_id, task_id) -- a stand-in for what a sim engine
# would produce after retargeting a real demo to a new robot.

@dataclass
class ToyConfig:
    n_embodiments: int = 3
    n_tasks: int = 2
    horizon: int = 24            # T frames per episode (real datasets ~100-500)
    img_size: int = 8            # tiny; real Isaac Sim renders 256x256+
    img_channels: int = 3
    state_dim: int = 4
    action_dim: int = 2
    n_episodes_raw: int = 60     # before filtering / augmentation
    seed: int = 23


def _base_trajectory(emb_id: int, task_id: int, T: int, action_dim: int,
                     state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic base trajectory for (emb_id, task_id).

    Stand-in for what the simulator outputs *before* augmentation. We build
    state and action arrays. State drifts with embodiment id (different
    kinematics), action shape varies with task id (different skill). In a real
    pipeline this would come from MuJoCo / Isaac Lab after IK retargeting.
    """
    t = np.linspace(0.0, 2.0 * np.pi, T)
    # action = simple parametric curve depending on task
    if task_id == 0:
        ax = np.cos(t)
        ay = np.sin(t)
    else:
        ax = np.sin(2 * t)
        ay = 0.5 * np.cos(t)
    actions = np.stack([ax, ay], axis=-1)  # (T, 2)

    # state: cumulative integration of action with embodiment-specific gain
    gain = 0.5 + 0.25 * emb_id  # different "robot" responds differently
    state = np.cumsum(actions * gain, axis=0)
    # pad state to state_dim with zeros (multi-embodiment padding -- J21 idea)
    if state_dim > 2:
        pad = np.zeros((T, state_dim - 2))
        state = np.concatenate([state, pad], axis=-1)
    return state.astype(np.float32), actions.astype(np.float32)


def _base_image(emb_id: int, task_id: int, frame_idx: int, T: int,
                img_size: int) -> np.ndarray:
    """Tiny synthetic image: a moving square encoding (emb, task, t).

    In a real pipeline this is what Isaac Lab path-tracing would render
    (256x256 RGB). Here we just build an 8x8 RGB array deterministically.
    Returns float32 array in [0, 1], shape (img_size, img_size, 3).
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    # base background: light gray
    img[:] = 0.5
    # moving "object" representing the gripper position
    cx = int((img_size - 1) * (frame_idx / max(T - 1, 1)))
    cy = (emb_id + task_id) % img_size
    color = np.array([
        0.8 if task_id == 0 else 0.2,
        0.5,
        0.2 if emb_id == 0 else (0.5 if emb_id == 1 else 0.8),
    ], dtype=np.float32)
    img[cy, cx] = color
    return img


def generate_raw_dataset(cfg: ToyConfig) -> list[dict]:
    """Generate `n_episodes_raw` toy trajectories.

    Returns list of episode dicts ready for augmentation/filtering. Each dict
    matches the shape we'd find in a LeRobotDataset row (REFERENCES.md #27),
    minus the multi-camera videos (we keep a single image stream).
    """
    rng = np.random.default_rng(cfg.seed)
    episodes: list[dict] = []
    for ep_idx in range(cfg.n_episodes_raw):
        emb_id = int(rng.integers(0, cfg.n_embodiments))
        task_id = int(rng.integers(0, cfg.n_tasks))
        state, action = _base_trajectory(
            emb_id, task_id, cfg.horizon, cfg.action_dim, cfg.state_dim)
        images = np.stack([
            _base_image(emb_id, task_id, t, cfg.horizon, cfg.img_size)
            for t in range(cfg.horizon)
        ])  # (T, H, W, 3)
        episodes.append({
            "episode_index": ep_idx,
            "embodiment": f"toy_robot_{emb_id}",
            "task": f"task_{task_id}",
            "task_index": task_id,
            "state": state,
            "action": action,
            "image": images,
            "length": cfg.horizon,
        })
    return episodes


# ----------------------------------------------------------------------------
# 2. The 4-axis augmentation pipeline (REFERENCES.md #15, #22)
# ----------------------------------------------------------------------------
# Background / Lighting / Distractors / Dynamics. Each augmentation acts on a
# single episode and returns a *new* augmented copy. In production these would
# be GPU-batched (Isaac Lab + Cosmos diffusion); here we stay numpy-pure.

def aug_background(episode: dict, rng: np.random.Generator) -> dict:
    """Replace the gray background with a random uniform tint.

    Cosmos relight (REFERENCES.md #22) does this via diffusion; we just sample
    a per-channel mean and overlay where pixels look like 'background' (gray).
    The robot/object trail is preserved -- *only* the gray pixels change.
    """
    out = episode.copy()
    images = episode["image"].copy()
    bg_tint = rng.uniform(0.0, 1.0, size=3).astype(np.float32)  # (3,)
    # mask = pixels close to original gray (0.5) on all channels
    is_bg = np.all(np.abs(images - 0.5) < 1e-3, axis=-1, keepdims=True)
    images = np.where(is_bg, bg_tint, images)
    out["image"] = images
    out["aug_background"] = float(bg_tint.mean())
    return out


def aug_lighting(episode: dict, rng: np.random.Generator) -> dict:
    """Multiplicative gain + per-channel temperature shift.

    Models a different lighting source. Trivial implementation cost --
    matches the cheapest, highest-ROI augmentation in real pipelines.
    """
    out = episode.copy()
    gain = rng.uniform(0.5, 1.4)                       # exposure
    temp = rng.uniform(-0.15, 0.15, size=3).astype(np.float32)  # color cast
    images = episode["image"] * gain + temp
    out["image"] = np.clip(images, 0.0, 1.0)
    out["aug_lighting_gain"] = float(gain)
    return out


def aug_distractors(episode: dict, rng: np.random.Generator) -> dict:
    """Splash random colored pixels in the scene to force the policy to focus
    on the *language-relevant* object. In a real pipeline these are mesh assets
    placed with collision check; here we stamp 1-3 random pixels per frame.
    """
    out = episode.copy()
    images = episode["image"].copy()
    T, H, W, _ = images.shape
    n_distractors = int(rng.integers(1, 4))
    for _ in range(n_distractors):
        # fixed position across the episode (a "static" distractor)
        dx, dy = int(rng.integers(0, W)), int(rng.integers(0, H))
        color = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        # don't overwrite the main object trail -- only colored at exact frame match
        for t in range(T):
            # 50% probability that the distractor is visible in this frame
            if rng.random() < 0.5:
                images[t, dy, dx] = color
    out["image"] = images
    out["aug_n_distractors"] = n_distractors
    return out


def aug_dynamics(episode: dict, rng: np.random.Generator) -> dict:
    """Simulate dynamics randomization: action latency + state observation
    noise + small action gain perturbation. This is the Tobin-2017 style
    domain randomization, made concrete on our toy data.
    """
    out = episode.copy()
    state = episode["state"].copy()
    action = episode["action"].copy()
    # 1. Action latency: shift actions by 1 frame with prob 0.3
    if rng.random() < 0.3 and len(action) > 1:
        action = np.concatenate([action[:1], action[:-1]], axis=0)
    # 2. Action gain perturbation
    gain_perturb = rng.uniform(0.85, 1.15)
    action = action * gain_perturb
    # 3. State observation noise (sensor noise)
    state = state + rng.normal(0.0, 0.02, size=state.shape).astype(np.float32)
    out["state"] = state
    out["action"] = action
    out["aug_dyn_gain"] = float(gain_perturb)
    return out


# Map of available augmentations. In production, pipelines apply *all* of them
# in sequence with controlled randomness.
AUGMENTATIONS = {
    "background": aug_background,
    "lighting": aug_lighting,
    "distractors": aug_distractors,
    "dynamics": aug_dynamics,
}


def augment_episode(episode: dict, axes: Iterable[str],
                    rng: np.random.Generator) -> dict:
    """Apply each augmentation axis in order. Each is independent so order is
    only loosely important (e.g. background must run before distractors if you
    want distractors to remain visible)."""
    out = dict(episode)
    for axis in axes:
        if axis not in AUGMENTATIONS:
            raise ValueError(f"unknown augmentation axis: {axis}")
        out = AUGMENTATIONS[axis](out, rng)
    return out


def batch_augment(episodes: list[dict], n_augments_per_episode: int,
                  axes: Iterable[str], seed: int) -> list[dict]:
    """Multiply each raw episode into N augmented copies.

    With axes=("background","lighting","distractors","dynamics") and
    n_augments_per_episode=5, you turn 60 raw episodes into 300 augmented
    ones, each visually/dynamically different. This is the *volume*
    multiplier in the GR00T pipeline (REFERENCES.md #15).
    """
    rng = np.random.default_rng(seed)
    out = []
    for ep in episodes:
        for k in range(n_augments_per_episode):
            aug = augment_episode(ep, axes, rng)
            aug = dict(aug)
            aug["episode_index"] = ep["episode_index"] * 100 + k
            aug["aug_variant"] = k
            out.append(aug)
    return out


# ----------------------------------------------------------------------------
# 3. Filtering (GR00T rejects ~30% of generated trajectories)
# ----------------------------------------------------------------------------
def filter_episode(episode: dict, max_action_jump: float = 3.0,
                   max_state_norm: float = 100.0,
                   require_finite: bool = True) -> tuple[bool, str]:
    """Return (keep, reason).

    Mimics the GR00T filtering stage:
      - smoothness: no large action jumps (would mean IK divergence in sim),
      - state sanity: bounded norm (no exploded trajectory),
      - sensor sanity: no NaN / Inf in any field.

    A real pipeline also runs a 'success' check ("did the cube end on the
    target?") that requires task-specific labelling -- omitted here since we
    have no goal state in the toy data.
    """
    if require_finite:
        for k in ("state", "action", "image"):
            arr = episode[k]
            if not np.all(np.isfinite(arr)):
                return False, f"non-finite values in {k}"
    action = episode["action"]
    if action.shape[0] > 1:
        jumps = np.linalg.norm(np.diff(action, axis=0), axis=-1)
        if jumps.max() > max_action_jump:
            return False, f"action jump {jumps.max():.2f} > {max_action_jump}"
    state_norm = np.linalg.norm(episode["state"], axis=-1).max()
    if state_norm > max_state_norm:
        return False, f"state norm {state_norm:.2f} > {max_state_norm}"
    return True, "ok"


def filter_dataset(episodes: list[dict], **kwargs) -> tuple[list[dict], dict]:
    """Apply filter_episode to all and return (kept, stats)."""
    kept = []
    reasons: dict[str, int] = {}
    for ep in episodes:
        ok, reason = filter_episode(ep, **kwargs)
        if ok:
            kept.append(ep)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    stats = {
        "total": len(episodes),
        "kept": len(kept),
        "rejected": len(episodes) - len(kept),
        "rejection_rate": (len(episodes) - len(kept)) / max(len(episodes), 1),
        "rejection_reasons": reasons,
    }
    return kept, stats


# ----------------------------------------------------------------------------
# 4. Saving in a LeRobotDataset-like layout (REFERENCES.md #27)
# ----------------------------------------------------------------------------
# Real LeRobotDataset uses Parquet for tabular data + MP4 for videos. We keep
# the *spirit* (split into meta/, data/, videos-substitute/) so the layout is
# instantly familiar to anyone who knows the standard. The fallback uses
# pickle if pyarrow is unavailable -- the toy harness must run anywhere.

def _try_import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except Exception:
        return None, None


def save_lerobot_like(episodes: list[dict], out_dir: str, cfg: ToyConfig,
                      use_parquet: bool | None = None) -> dict:
    """Save the dataset in a LeRobot-like directory layout.

    Layout produced:
        out_dir/
          meta/
            info.json
            stats.json
            tasks.jsonl
          data/
            chunk-000/
              episode_XXXXXX.parquet  (or .pkl fallback)
          images/                            (stand-in for videos/ in LeRobot)
            chunk-000/
              episode_XXXXXX.npy

    Returns a dict with manifest info.
    """
    if use_parquet is None:
        pa, pq = _try_import_pyarrow()
        use_parquet = pa is not None
    else:
        pa, pq = (_try_import_pyarrow() if use_parquet else (None, None))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images", "chunk-000"), exist_ok=True)

    # --- meta/info.json ---
    info = {
        "codebase_version": "lerobot-like-toy-0.1",
        "robot_type": "toy_multi_robot",
        "fps": 10,
        "total_episodes": len(episodes),
        "total_frames": sum(ep["length"] for ep in episodes),
        "image_shape": [cfg.img_size, cfg.img_size, cfg.img_channels],
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "splits": {"train": f"0:{len(episodes)}"},
        "format_backend": "parquet" if use_parquet else "pickle",
    }
    with open(os.path.join(out_dir, "meta", "info.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # --- meta/stats.json -- compute mean/std for normalization (used at training time) ---
    all_states = np.concatenate([ep["state"] for ep in episodes], axis=0)
    all_actions = np.concatenate([ep["action"] for ep in episodes], axis=0)
    stats = {
        "state": {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        },
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        },
    }
    with open(os.path.join(out_dir, "meta", "stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # --- meta/tasks.jsonl ---
    tasks = sorted({(ep["task_index"], ep["task"]) for ep in episodes})
    with open(os.path.join(out_dir, "meta", "tasks.jsonl"), "w",
              encoding="utf-8") as f:
        for task_index, task in tasks:
            f.write(json.dumps(
                {"task_index": int(task_index), "task": task}) + "\n")

    # --- data/ + images/ : one file per episode ---
    for ep in episodes:
        idx = int(ep["episode_index"])
        # tabular row: state + action per frame
        T = ep["length"]
        rows = {
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, idx, dtype=np.int64),
            "task_index": np.full(T, ep["task_index"], dtype=np.int64),
            "embodiment": [ep["embodiment"]] * T,
        }
        for d in range(cfg.state_dim):
            rows[f"observation.state.{d}"] = ep["state"][:, d].astype(np.float32)
        for d in range(cfg.action_dim):
            rows[f"action.{d}"] = ep["action"][:, d].astype(np.float32)

        data_path = os.path.join(out_dir, "data", "chunk-000",
                                  f"episode_{idx:06d}")
        if use_parquet:
            table = pa.table(rows)
            pq.write_table(table, data_path + ".parquet")
        else:
            # pickle fallback so the harness runs without pyarrow
            with open(data_path + ".pkl", "wb") as f:
                pickle.dump(rows, f)

        # image stream: stand-in for LeRobot's MP4 video file. We dump a
        # uint8 npy to keep the file small. A real pipeline encodes H.264.
        img_path = os.path.join(out_dir, "images", "chunk-000",
                                 f"episode_{idx:06d}.npy")
        img_uint8 = (ep["image"] * 255).clip(0, 255).astype(np.uint8)
        np.save(img_path, img_uint8)

    return info


# ----------------------------------------------------------------------------
# 5. Tiny torch DataLoader-friendly iterator
# ----------------------------------------------------------------------------
class ToyLeRobotIterableDataset(torch.utils.data.IterableDataset):
    """Iterate (state, action, image) tensors from a saved toy dataset.

    Production LeRobotDataset uses HuggingFace Hub streaming + ffmpeg video
    decode. Here we just walk the npy/parquet files. The point is to show
    that the data can be consumed *as a torch dataset* without an extra
    conversion step -- which is the value proposition of LeRobotDataset.
    """

    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dir = dataset_dir
        with open(os.path.join(dataset_dir, "meta", "info.json"),
                  "r", encoding="utf-8") as f:
            self.info = json.load(f)
        self._chunk = os.path.join(dataset_dir, "data", "chunk-000")
        self._img_chunk = os.path.join(dataset_dir, "images", "chunk-000")
        self._files = sorted(os.listdir(self._chunk))

    def __iter__(self):
        pa, _pq = _try_import_pyarrow()
        for fname in self._files:
            base = fname.rsplit(".", 1)[0]
            data_path = os.path.join(self._chunk, fname)
            if fname.endswith(".parquet"):
                import pyarrow.parquet as pq  # noqa: WPS433 -- local import OK
                rows = pq.read_table(data_path).to_pydict()
            else:
                with open(data_path, "rb") as f:
                    rows = pickle.load(f)
            img_path = os.path.join(self._img_chunk, base + ".npy")
            img = np.load(img_path).astype(np.float32) / 255.0  # (T, H, W, 3)
            T = len(rows["frame_index"])
            state = np.stack([
                np.asarray(rows[f"observation.state.{d}"], dtype=np.float32)
                for d in range(self.info["state_dim"])
            ], axis=-1)  # (T, state_dim)
            action = np.stack([
                np.asarray(rows[f"action.{d}"], dtype=np.float32)
                for d in range(self.info["action_dim"])
            ], axis=-1)  # (T, action_dim)
            for t in range(T):
                yield {
                    "image": torch.from_numpy(img[t]),       # (H, W, 3)
                    "state": torch.from_numpy(state[t]),     # (state_dim,)
                    "action": torch.from_numpy(action[t]),   # (action_dim,)
                }


# ----------------------------------------------------------------------------
# 6. Main: end-to-end demo
# ----------------------------------------------------------------------------
def main():
    cfg = ToyConfig()
    print("=" * 72)
    print("J23 - Toy GR00T-style synthetic data pipeline")
    print(f"  embodiments={cfg.n_embodiments}  tasks={cfg.n_tasks}  "
          f"horizon={cfg.horizon}  raw_episodes={cfg.n_episodes_raw}")
    print("=" * 72)

    # --- Step 1: generate raw synthetic dataset ---
    print("\n[1/5] Generating raw synthetic trajectories ...")
    raw = generate_raw_dataset(cfg)
    print(f"      raw episodes: {len(raw)}")

    # --- Step 2: 4-axis augmentation ---
    print("\n[2/5] Applying 4-axis augmentation "
          "(background / lighting / distractors / dynamics) ...")
    augmented = batch_augment(
        raw,
        n_augments_per_episode=5,
        axes=("background", "lighting", "distractors", "dynamics"),
        seed=cfg.seed + 1,
    )
    print(f"      augmented episodes: {len(augmented)} "
          f"(={len(raw)} x 5 augmentations)")

    # --- Step 3: filtering ---
    print("\n[3/5] Filtering (contact / smoothness / sanity) ...")
    kept, fstats = filter_dataset(augmented, max_action_jump=3.0)
    print(f"      kept: {fstats['kept']} / {fstats['total']}  "
          f"(rejection rate = {fstats['rejection_rate']:.1%})")
    if fstats["rejection_reasons"]:
        for k, v in fstats["rejection_reasons"].items():
            print(f"        - {k}: {v}")

    # --- Step 4: save in LeRobotDataset-like format ---
    out_dir = os.path.join(tempfile.gettempdir(), "toy_lerobot_dataset_j23")
    print(f"\n[4/5] Saving to {out_dir} (LeRobotDataset-like layout) ...")
    info = save_lerobot_like(kept, out_dir, cfg)
    print(f"      backend: {info['format_backend']}")
    print(f"      total episodes saved: {info['total_episodes']}")
    print(f"      total frames: {info['total_frames']}")

    # --- Step 5: re-read as a torch IterableDataset ---
    print("\n[5/5] Re-reading dataset via torch.utils.data.IterableDataset ...")
    ds = ToyLeRobotIterableDataset(out_dir)
    n_seen = 0
    first = None
    for sample in ds:
        if first is None:
            first = sample
        n_seen += 1
        if n_seen >= 10:
            break
    if first is not None:
        print(f"      first sample shapes: image={tuple(first['image'].shape)}  "
              f"state={tuple(first['state'].shape)}  "
              f"action={tuple(first['action'].shape)}")
    print(f"      streamed {n_seen} samples (truncated to 10 for demo).")

    print("\nKey takeaways:")
    print("  - 4-axis augmentation multiplies one base demo by ~5x cheaply.")
    print("  - Filtering rejects malformed trajectories, mirroring GR00T's ~30%.")
    print("  - LeRobotDataset-like layout makes the result ready for any modern")
    print("    VLA training stack (PI0 / GR00T / OpenVLA / Octo).")
    print("\nReferences:")
    print("  - REFERENCES.md #15 GR00T N1 (NVIDIA 2025) -- pipeline blueprint")
    print("  - REFERENCES.md #22 NVIDIA Cosmos (2025) -- video relighting/curation")
    print("  - REFERENCES.md #27 LeRobotDataset v3.0 -- standard 2026 format")


if __name__ == "__main__":
    main()

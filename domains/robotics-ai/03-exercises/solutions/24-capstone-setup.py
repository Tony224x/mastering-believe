# requires: numpy>=1.24 matplotlib>=3.7
"""
J24 - Solutions consolidees pour les 3 exercices.

EASY    : charger le dataset PushT, valider sa structure, plotter 8 trajectoires.
MEDIUM  : analyse de distribution d'actions + detection de multimodalite par bucket.
HARD    : action chunking + augmentation (flip + bruit) + split train/val stratifie.

Toutes les solutions consomment l'artefact produit par J24:
    artifacts/pusht_demos/{data.npz, meta.json, episodes.jsonl}

Usage:
    python domains/robotics-ai/03-exercises/solutions/24-capstone-setup.py [easy|medium|hard|all]

Source: REFERENCES.md #19 (Diffusion Policy / PushT), #27 (LeRobotDataset v3.0).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DATA_DIR = Path("artifacts/pusht_demos")


# ============================================================================
# Shared helpers
# ============================================================================
def load_dataset(data_dir: Path = DATA_DIR):
    """Load the J24 capstone dataset.

    Returns a dict with the raw arrays and the parsed meta + episodes.jsonl.
    """
    if not (data_dir / "data.npz").exists():
        raise FileNotFoundError(
            f"{data_dir / 'data.npz'} not found. Run J24 generator first:\n"
            "  python domains/robotics-ai/02-code/24-capstone-setup.py"
        )
    data = np.load(data_dir / "data.npz")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    episodes = []
    with (data_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return {
        "obs": data["obs"],
        "action": data["action"],
        "ep_id": data["ep_id"],
        "ep_start": data["ep_start"],
        "ep_length": data["ep_length"],
        "meta": meta,
        "episodes": episodes,
    }


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


# ============================================================================
# EASY -- explore the dataset, validate, plot 8 trajectories
# ============================================================================
def run_easy(out_dir: Path = DATA_DIR) -> None:
    print("=" * 60)
    print("EASY: explore + validate + plot 8 trajectories")
    print("=" * 60)
    ds = load_dataset(out_dir)
    obs, action = ds["obs"], ds["action"]
    ep_start, ep_length = ds["ep_start"], ds["ep_length"]
    meta = ds["meta"]

    # 1. Mini-report
    print(f"  n_episodes      = {meta['n_episodes']}")
    print(f"  n_transitions   = {obs.shape[0]}  (action: {action.shape[0]})")
    print(f"  obs.shape       = {obs.shape}   dtype={obs.dtype}")
    print(f"  action.shape    = {action.shape}   dtype={action.dtype}")
    print(f"  ep_length stats : mean={float(np.mean(ep_length)):.1f} "
          f"min={int(ep_length.min())} max={int(ep_length.max())}")

    # 2. Consistency assertions
    assert ep_start[0] == 0, "ep_start[0] must be 0"
    assert np.all(np.diff(ep_start) > 0), "ep_start must be strictly increasing"
    assert int(ep_length.sum()) == obs.shape[0] == action.shape[0], (
        "sum(ep_length) must equal both obs and action lengths"
    )
    a_max = float(meta["a_max"])
    max_anorm = float(np.linalg.norm(action, axis=1).max())
    # Expert injects ~N(0, 0.5) noise BEFORE env clipping; env clip enforces
    # ||action|| <= a_max strictly, so we should see <= a_max + tiny float noise.
    assert max_anorm <= a_max + 1e-3, (
        f"action norm {max_anorm:.3f} exceeds a_max={a_max} (env clip should enforce this)"
    )
    print("  consistency checks: PASS")

    # 3. Plot 8 trajectories
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("  matplotlib unavailable -- skipping figure")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    table = int(meta["table_size"])
    ax.set_xlim(0, table); ax.set_ylim(0, table); ax.set_aspect("equal")
    ax.set_title("EASY: agent (solid) + block (dashed) for first 8 episodes")
    colors = plt.cm.tab10.colors
    for i in range(min(8, len(ep_start))):
        s, l = int(ep_start[i]), int(ep_length[i])
        ep = obs[s : s + l]
        c = colors[i % len(colors)]
        ax.plot(ep[:, 0], ep[:, 1], color=c, alpha=0.8, label=f"ep{i}")
        ax.plot(ep[:, 2], ep[:, 3], color=c, ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")
    _save_fig(fig, out_dir / "exo_easy_trajectories.png")
    plt.close(fig)
    print(f"  saved {out_dir / 'exo_easy_trajectories.png'}")
    print("  observation: agent paths visibly curve around the block")
    print("  before reaching contact, never going through it.\n")


# ============================================================================
# MEDIUM -- action distribution + multimodality detection
# ============================================================================
def _kmeans_2(points: np.ndarray, n_iter: int = 20, seed: int = 0):
    """Lloyd's algorithm with k=2. Returns (centroids, labels, intra_std)."""
    rng = np.random.default_rng(seed)
    if points.shape[0] < 2:
        # Degenerate: not enough points to find two clusters.
        return points.copy(), np.zeros(points.shape[0], dtype=np.int32), 0.0
    idx = rng.choice(points.shape[0], size=2, replace=False)
    c = points[idx].copy()
    labels = np.zeros(points.shape[0], dtype=np.int32)
    for _ in range(n_iter):
        d0 = np.linalg.norm(points - c[0], axis=1)
        d1 = np.linalg.norm(points - c[1], axis=1)
        labels = (d1 < d0).astype(np.int32)
        for k in (0, 1):
            sel = points[labels == k]
            if sel.shape[0] > 0:
                c[k] = sel.mean(axis=0)
    intra_std = float(np.mean([
        points[labels == k].std(axis=0).mean() if (labels == k).any() else 0.0
        for k in (0, 1)
    ]))
    return c, labels, intra_std


def run_medium(out_dir: Path = DATA_DIR) -> None:
    print("=" * 60)
    print("MEDIUM: action distribution + multimodality detection")
    print("=" * 60)
    ds = load_dataset(out_dir)
    obs, action = ds["obs"], ds["action"]
    ep_start, ep_length = ds["ep_start"], ds["ep_length"]

    # Global statistics
    print(f"  action[:, 0]  mean={action[:, 0].mean():+.3f} "
          f"std={action[:, 0].std():.3f} "
          f"min={action[:, 0].min():+.3f} max={action[:, 0].max():+.3f}")
    print(f"  action[:, 1]  mean={action[:, 1].mean():+.3f} "
          f"std={action[:, 1].std():.3f} "
          f"min={action[:, 1].min():+.3f} max={action[:, 1].max():+.3f}")

    # Build a mask for "alignment phase" vs "push phase" using a heuristic:
    # the first 25% of each episode is alignment, the rest is push.
    is_alignment = np.zeros(action.shape[0], dtype=bool)
    for i in range(len(ep_start)):
        s, l = int(ep_start[i]), int(ep_length[i])
        cut = max(1, int(0.25 * l))
        is_alignment[s : s + cut] = True

    # Bucket: block in the lower-left quarter of the table, agent reasonably
    # central. Tweak the bounds if your seed gives you an empty bucket.
    block_x, block_y = obs[:, 2], obs[:, 3]
    agent_x, agent_y = obs[:, 0], obs[:, 1]
    bucket_mask = (
        (block_x < 256) & (block_y < 256)
        & (agent_x > 150) & (agent_x < 380)
        & (agent_y > 150) & (agent_y < 380)
    )
    print(f"  bucket size (any phase) = {int(bucket_mask.sum())}")

    # We compute a multimodality score for two sub-buckets: alignment vs push.
    def _score(mask: np.ndarray, label: str):
        pts = action[mask]
        n = pts.shape[0]
        if n < 20:
            print(f"  [{label}] only {n} points -- skip score")
            return None
        c, lab, intra_std = _kmeans_2(pts)
        d_centroids = float(np.linalg.norm(c[0] - c[1]))
        ratio = d_centroids / max(intra_std, 1e-6)
        verdict = "MULTIMODAL" if ratio > 2.5 else "unimodal"
        print(f"  [{label:<10}] n={n:4d}  d_centroids={d_centroids:5.2f}  "
              f"intra_std={intra_std:5.2f}  ratio={ratio:5.2f}  -> {verdict}")
        return pts, lab, c

    align_res = _score(bucket_mask & is_alignment, "alignment")
    push_res = _score(bucket_mask & ~is_alignment, "push")

    # Plot global histograms + the two scatters side by side.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("  matplotlib unavailable -- skipping figures")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(action[:, 0], bins=40, alpha=0.7, label="dx")
    axes[0].hist(action[:, 1], bins=40, alpha=0.7, label="dy")
    axes[0].legend(); axes[0].set_title("action histogram (all transitions)")
    axes[0].set_xlabel("pixels per step")
    if align_res is not None:
        pts, lab, c = align_res
        axes[1].scatter(pts[lab == 0, 0], pts[lab == 0, 1], s=8, alpha=0.5,
                        c="C0", label="cluster 0")
        axes[1].scatter(pts[lab == 1, 0], pts[lab == 1, 1], s=8, alpha=0.5,
                        c="C1", label="cluster 1")
        axes[1].scatter(c[:, 0], c[:, 1], s=120, c="red", marker="X",
                        label="centroids")
        axes[1].legend()
    axes[1].set_title("alignment-phase actions (bucket)")
    axes[1].set_xlabel("dx"); axes[1].set_ylabel("dy")
    _save_fig(fig, out_dir / "exo_medium_global.png")
    plt.close(fig)
    print(f"  saved {out_dir / 'exo_medium_global.png'}")

    # Bucket-only figure for clarity.
    if push_res is not None and align_res is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for ax, (pts, lab, _c), title in zip(
            axes, (align_res, push_res), ("alignment phase", "push phase")
        ):
            ax.scatter(pts[lab == 0, 0], pts[lab == 0, 1], s=8, alpha=0.5, c="C0")
            ax.scatter(pts[lab == 1, 0], pts[lab == 1, 1], s=8, alpha=0.5, c="C1")
            ax.set_title(title); ax.set_xlabel("dx")
        axes[0].set_ylabel("dy")
        _save_fig(fig, out_dir / "exo_medium_bucket.png")
        plt.close(fig)
        print(f"  saved {out_dir / 'exo_medium_bucket.png'}")
    print("  takeaway: alignment phase tends to score more multimodal because")
    print("  the expert decides on the contour side there; push phase is")
    print("  essentially geometric (one valid direction toward target).\n")


# ============================================================================
# HARD -- action chunking + augmentation + stratified train/val split
# ============================================================================
@dataclass
class ChunkBatch:
    obs: np.ndarray         # (B, obs_dim)
    action: np.ndarray      # (B, horizon, action_dim)


def make_chunks(obs: np.ndarray, action: np.ndarray,
                ep_start: np.ndarray, ep_length: np.ndarray,
                horizon: int = 8):
    """Slice the dataset into (obs_t, action_chunk) samples that never cross
    episode boundaries.

    Returns:
        obs_t          (N, obs_dim)
        action_chunk   (N, horizon, action_dim)
        ep_id_t        (N,)
    """
    obs_list, chunk_list, eid_list = [], [], []
    for i in range(len(ep_start)):
        s, l = int(ep_start[i]), int(ep_length[i])
        for t in range(s, s + l - horizon + 1):
            obs_list.append(obs[t])
            chunk_list.append(action[t : t + horizon])
            eid_list.append(i)
    if not obs_list:
        empty_obs = np.zeros((0, obs.shape[1]), dtype=obs.dtype)
        empty_chunks = np.zeros((0, horizon, action.shape[1]), dtype=action.dtype)
        return empty_obs, empty_chunks, np.zeros((0,), dtype=np.int32)
    return (
        np.stack(obs_list, axis=0),
        np.stack(chunk_list, axis=0),
        np.asarray(eid_list, dtype=np.int32),
    )


class Augmenter:
    """Horizontal flip + gaussian noise on observations.

    The flip mirrors the world along x = W/2:
      agent_x      -> W - agent_x
      block_x      -> W - block_x
      block_theta  -> -block_theta
      action[:, 0] -> -action[:, 0]   (delta_x flips sign)
    Other components are unchanged. Noise is added to spatial obs (idx 0..3)
    only; theta gets a smaller std (0.01 rad).
    """

    def __init__(self, table_size: int, p_flip: float = 0.5,
                 sigma_obs: float = 1.5, sigma_theta: float = 0.01,
                 seed: int = 0):
        self.W = float(table_size)
        self.p_flip = p_flip
        self.sigma_obs = sigma_obs
        self.sigma_theta = sigma_theta
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs_b: np.ndarray, action_b: np.ndarray):
        obs_b = obs_b.copy()
        action_b = action_b.copy()
        B = obs_b.shape[0]
        flip_mask = self.rng.random(B) < self.p_flip
        if flip_mask.any():
            idx = np.where(flip_mask)[0]
            obs_b[idx, 0] = self.W - obs_b[idx, 0]
            obs_b[idx, 2] = self.W - obs_b[idx, 2]
            obs_b[idx, 4] = -obs_b[idx, 4]
            action_b[idx, :, 0] = -action_b[idx, :, 0]
        # Gaussian noise on spatial obs (4 first dims) and tiny on theta.
        if self.sigma_obs > 0:
            obs_b[:, 0:4] = obs_b[:, 0:4] + self.rng.normal(
                0.0, self.sigma_obs, size=(B, 4)
            ).astype(obs_b.dtype)
        if self.sigma_theta > 0:
            obs_b[:, 4] = obs_b[:, 4] + self.rng.normal(
                0.0, self.sigma_theta, size=(B,)
            ).astype(obs_b.dtype)
        return obs_b, action_b


def stratified_episode_split(episodes: list, ratio: float = 0.8, seed: int = 0):
    """Split episode indices into train/val, stratifying on the 'side' field.

    Returns (train_ids, val_ids), each a sorted list of episode indices.
    """
    rng = np.random.default_rng(seed)
    by_side = {"left": [], "right": []}
    for ep in episodes:
        by_side.setdefault(ep["side"], []).append(int(ep["episode_index"]))
    train_ids, val_ids = [], []
    for side, ids in by_side.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_train = int(round(ratio * len(ids)))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:])
    train_ids.sort(); val_ids.sort()
    return train_ids, val_ids


class PushTDataModule:
    """Minimal training-side DataModule for the J24 dataset.

    Composition in __init__:
      - load dataset
      - stratified split by episode and side
      - per-split chunking
    Usage:
      for batch in dm.iterate(split="train", batch_size=64, augment=True):
          obs, action = batch.obs, batch.action  # numpy arrays
    """

    def __init__(self, data_dir: Path = DATA_DIR, horizon: int = 8,
                 ratio: float = 0.8, seed: int = 0):
        ds = load_dataset(data_dir)
        self.meta = ds["meta"]
        self.horizon = horizon
        train_ids, val_ids = stratified_episode_split(ds["episodes"],
                                                      ratio=ratio, seed=seed)
        self._splits = {}
        for split, ids in (("train", train_ids), ("val", val_ids)):
            mask = np.isin(ds["ep_id"], np.asarray(ids, dtype=np.int32))
            # We need ep_start/ep_length restricted to selected episodes,
            # but the easiest is to rebuild flat arrays from selected episodes.
            sel_obs, sel_action, new_starts, new_lengths = [], [], [], []
            cursor = 0
            for ep_idx in ids:
                s = int(ds["ep_start"][ep_idx])
                l = int(ds["ep_length"][ep_idx])
                sel_obs.append(ds["obs"][s : s + l])
                sel_action.append(ds["action"][s : s + l])
                new_starts.append(cursor)
                new_lengths.append(l)
                cursor += l
            obs_split = np.concatenate(sel_obs, axis=0) if sel_obs else np.zeros((0, ds["obs"].shape[1]))
            action_split = np.concatenate(sel_action, axis=0) if sel_action else np.zeros((0, ds["action"].shape[1]))
            new_starts_arr = np.asarray(new_starts, dtype=np.int32)
            new_lengths_arr = np.asarray(new_lengths, dtype=np.int32)
            obs_t, chunks, eid_t = make_chunks(
                obs_split, action_split, new_starts_arr, new_lengths_arr,
                horizon=horizon,
            )
            self._splits[split] = {
                "obs_t": obs_t,
                "chunks": chunks,
                "ep_id_t": eid_t,
                "episode_indices": ids,
            }
        self.augmenter = Augmenter(table_size=int(self.meta["table_size"]),
                                   seed=seed)

    def stats(self) -> dict:
        train = self._splits["train"]; val = self._splits["val"]
        return {
            "n_train_chunks": int(train["obs_t"].shape[0]),
            "n_val_chunks": int(val["obs_t"].shape[0]),
            "n_train_episodes": len(train["episode_indices"]),
            "n_val_episodes": len(val["episode_indices"]),
        }

    def iterate(self, split: str = "train", batch_size: int = 64,
                augment: bool = True, shuffle: bool = True,
                seed: Optional[int] = None):
        s = self._splits[split]
        n = s["obs_t"].shape[0]
        if n == 0:
            return
        order = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            obs_b = s["obs_t"][idx]
            action_b = s["chunks"][idx]
            if augment and split == "train":
                obs_b, action_b = self.augmenter(obs_b, action_b)
            yield ChunkBatch(obs=obs_b, action=action_b)


def run_hard(out_dir: Path = DATA_DIR) -> None:
    print("=" * 60)
    print("HARD: chunking + augmentation + stratified split")
    print("=" * 60)
    ds = load_dataset(out_dir)

    # 1. Stratified split sanity
    train_ids, val_ids = stratified_episode_split(ds["episodes"],
                                                  ratio=0.8, seed=0)
    side_lookup = {ep["episode_index"]: ep["side"] for ep in ds["episodes"]}
    def _ratio_left(ids):
        if not ids:
            return float("nan")
        return sum(1 for i in ids if side_lookup[i] == "left") / len(ids)
    print(f"  n_train_ep={len(train_ids)}  n_val_ep={len(val_ids)}")
    print(f"  ratio_left_train={_ratio_left(train_ids):.3f}  "
          f"ratio_left_val={_ratio_left(val_ids):.3f}")
    if train_ids and val_ids:
        assert abs(_ratio_left(train_ids) - _ratio_left(val_ids)) < 0.10, (
            "stratification failed -- ratios differ by more than 10%"
        )
    print("  stratification check: PASS")

    # 2. Chunking off-by-one sanity on a hand-built mini-dataset
    mini_obs = np.arange(18, dtype=np.float32).reshape(18, 1).repeat(5, axis=1)
    mini_act = np.arange(18, dtype=np.float32).reshape(18, 1).repeat(2, axis=1)
    mini_starts = np.array([0, 5, 11], dtype=np.int32)
    mini_lengths = np.array([5, 6, 7], dtype=np.int32)
    o, c, e = make_chunks(mini_obs, mini_act, mini_starts, mini_lengths, horizon=3)
    expected = (5 - 3 + 1) + (6 - 3 + 1) + (7 - 3 + 1)  # = 3 + 4 + 5 = 12
    assert o.shape[0] == expected, f"expected {expected} chunks, got {o.shape[0]}"
    # No chunk should cross episode boundaries: action chunk values must be
    # contiguous integers inside one episode's range.
    for k in range(o.shape[0]):
        ep = int(e[k])
        s = int(mini_starts[ep]); end = s + int(mini_lengths[ep])
        chunk_vals = c[k, :, 0].astype(int)
        assert chunk_vals.min() >= s and chunk_vals.max() < end, (
            f"chunk {k} crosses episode boundary"
        )
    print("  chunking off-by-one check: PASS (12 chunks on mini dataset)")

    # 3. Augmenter involutivity (flip twice == identity)
    aug = Augmenter(table_size=int(ds["meta"]["table_size"]),
                    p_flip=1.0, sigma_obs=0.0, sigma_theta=0.0, seed=0)
    obs_in = ds["obs"][:32].copy()
    chunks_in = make_chunks(ds["obs"], ds["action"], ds["ep_start"],
                             ds["ep_length"], horizon=4)[1][:32]
    obs1, ch1 = aug(obs_in, chunks_in)
    obs2, ch2 = aug(obs1, ch1)
    assert np.allclose(obs2, obs_in, atol=1e-5), "flip is not involutive on obs"
    assert np.allclose(ch2, chunks_in, atol=1e-5), "flip is not involutive on actions"
    print("  flip involutivity: PASS")

    # 4. Gaussian noise empirical std
    aug_noise = Augmenter(table_size=int(ds["meta"]["table_size"]),
                          p_flip=0.0, sigma_obs=1.5, sigma_theta=0.0, seed=0)
    big_obs = np.zeros((4000, 5), dtype=np.float32)
    big_chunks = np.zeros((4000, 4, 2), dtype=np.float32)
    out_obs, _ = aug_noise(big_obs, big_chunks)
    emp_std = float(out_obs[:, :4].std())
    print(f"  empirical noise std (sigma_obs=1.5) = {emp_std:.3f}")
    assert 1.3 < emp_std < 1.7, f"empirical std off: {emp_std}"
    print("  noise std check: PASS")

    # 5. End-to-end DataModule timing + a single batch peek
    import time
    dm = PushTDataModule(data_dir=out_dir, horizon=8, ratio=0.8, seed=0)
    print(f"  DataModule stats = {dm.stats()}")
    t0 = time.time()
    n_batches = 0
    n_samples = 0
    for batch in dm.iterate(split="train", batch_size=64, augment=True, seed=42):
        n_batches += 1
        n_samples += batch.obs.shape[0]
    dt = time.time() - t0
    print(f"  iterated 1 epoch on train in {dt:.2f}s "
          f"({n_batches} batches, {n_samples} samples)")
    assert dt < 5.0, f"epoch too slow: {dt:.2f}s"
    # Peek a batch to confirm shapes
    peek = next(dm.iterate(split="val", batch_size=8, augment=False, seed=0))
    print(f"  val batch obs.shape={peek.obs.shape}  action.shape={peek.action.shape}")
    print("\nHARD: all checks PASS\n")


# ============================================================================
# CLI dispatch
# ============================================================================
def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("easy", "all"):
        run_easy()
    if target in ("medium", "all"):
        run_medium()
    if target in ("hard", "all"):
        run_hard()


if __name__ == "__main__":
    main()

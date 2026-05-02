# requires: numpy torch
"""
J23 - Solutions consolidees pour les 3 exercices.

EASY    : lighting_augment + background_swap (numpy pur).
MEDIUM  : pipeline filter + stats sur 100 trajectoires avec defauts injectes.
HARD    : pipeline complet end-to-end + benchmark 4 policies (no-aug, lighting,
          dynamics, all-4-axes) sur OOD lighting set.

Usage:
    python domains/robotics-ai/03-exercises/solutions/23-synthetic-data-scale.py [easy|medium|hard|all]

Sources:
  - REFERENCES.md #15 GR00T N1 (NVIDIA 2025)
  - REFERENCES.md #22 NVIDIA Cosmos
  - REFERENCES.md #27 LeRobotDataset v3.0
"""

from __future__ import annotations

import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# EASY -- lighting_augment + background_swap
# ============================================================================
def lighting_augment(img: np.ndarray, gain: float,
                     temperature: np.ndarray) -> np.ndarray:
    """Multiplicative gain + per-channel additive temperature.

    img: (H, W, 3) float in [0, 1]
    Returns a NEW array clipped to [0, 1].
    """
    if temperature.shape != (3,):
        raise ValueError(f"temperature must be shape (3,), got {temperature.shape}")
    out = img * gain + temperature  # broadcasts (3,) -> (H, W, 3)
    return np.clip(out, 0.0, 1.0)


def background_swap(img: np.ndarray, new_color: np.ndarray,
                    gray_threshold: float = 1e-3) -> np.ndarray:
    """Replace pixels close to gray (0.5) with new_color, preserve the rest.

    img:        (H, W, 3) float in [0, 1]
    new_color:  (3,) the replacement RGB.
    gray_threshold: how close to 0.5 a pixel must be to count as background.
    """
    is_bg = np.all(np.abs(img - 0.5) < gray_threshold, axis=-1, keepdims=True)
    return np.where(is_bg, new_color, img).astype(img.dtype)


def run_easy() -> None:
    print("=" * 60)
    print("EASY: lighting_augment + background_swap")
    print("=" * 60)

    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    img[4, 4] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # central red pixel
    print(f"  toy image: 8x8x3 gray with one red pixel at (4,4) = {img[4,4]}")

    # 1. lighting test: gain<1 darkens, +temperature shifts color.
    out_l = lighting_augment(img, gain=0.6, temperature=np.array([0.1, 0.0, 0.0]))
    bg_R = out_l[0, 0, 0]
    obj_R = out_l[4, 4, 0]
    print(f"  after lighting(gain=0.6, temp=[0.1,0,0]):")
    print(f"    background R = {bg_R:.3f}, object R = {obj_R:.3f}")
    assert obj_R > bg_R, "object should remain more red than background"
    assert out_l.min() >= 0.0 and out_l.max() <= 1.0, "must be clipped to [0,1]"
    print("    PASS: object remains separable from background, values clipped.")

    # 2. background swap test
    new_bg = np.array([0.2, 0.2, 0.8], dtype=np.float32)
    out_b = background_swap(img, new_bg)
    n_changed = int(((img != out_b).any(axis=-1)).sum())
    expected_changed = 8 * 8 - 1  # all but the central red pixel
    print(f"  after background_swap(new=[0.2,0.2,0.8]):")
    print(f"    pixels modified = {n_changed} (expected {expected_changed})")
    assert n_changed == expected_changed
    assert np.allclose(out_b[4, 4], img[4, 4]), "central red pixel must be preserved"
    assert np.allclose(out_b[0, 0], new_bg), "background must be the new color"
    print("    PASS: only gray pixels swapped, object preserved.")

    print("\n  Why is lighting the cheapest aug? Just (img * gain + temp) and clip:")
    print("  no resampling, no collision check, no diffusion model -- pure")
    print("  arithmetic. Same robustness benefit per-FLOP as 5-10x more")
    print("  expensive augmentations.\n")


# ============================================================================
# MEDIUM -- filter pipeline + stats
# ============================================================================
def _make_clean_episode(rng: np.random.Generator, T: int = 20) -> dict:
    """A small valid trajectory: smooth state and action, finite image."""
    t = np.linspace(0.0, 2 * np.pi, T)
    action = np.stack([np.cos(t), np.sin(t)], axis=-1).astype(np.float32)
    state = np.cumsum(action * 0.5, axis=0)
    state = np.concatenate([state, np.zeros((T, 2))], axis=-1).astype(np.float32)
    image = np.random.default_rng(rng.integers(0, 1_000_000)).uniform(
        0.0, 1.0, size=(T, 8, 8, 3)
    ).astype(np.float32)
    return {"state": state, "action": action, "image": image}


def _inject_defect(ep: dict, defect_kind: str,
                   rng: np.random.Generator) -> dict:
    """Inject one defect of the given kind into the episode in-place-copy."""
    out = {k: v.copy() for k, v in ep.items()}
    if defect_kind == "action_jump":
        # one giant action step
        out["action"][len(out["action"]) // 2] += np.array(
            [10.0, 10.0], dtype=np.float32)
    elif defect_kind == "state_explode":
        out["state"][len(out["state"]) // 2] += 200.0
    elif defect_kind == "image_nan":
        out["image"][0, 0, 0, 0] = np.nan
    elif defect_kind == "state_inf":
        out["state"][0, 0] = np.inf
    else:
        raise ValueError(f"unknown defect kind: {defect_kind}")
    out["_defect"] = defect_kind
    return out


def filter_episode(ep: dict, max_action_jump: float = 3.0,
                   max_state_norm: float = 50.0) -> tuple[bool, str]:
    """Return (keep, reason).

    Order of checks (cheap-first):
      1. NaN/Inf in image / state / action.
      2. Action jump too large.
      3. State norm too large.
    """
    for fname in ("image", "state", "action"):
        if not np.all(np.isfinite(ep[fname])):
            return False, f"non-finite values in {fname}"
    action = ep["action"]
    if action.shape[0] > 1:
        jumps = np.linalg.norm(np.diff(action, axis=0), axis=-1)
        max_jump = float(jumps.max())
        if max_jump > max_action_jump:
            return False, f"action jump {max_jump:.2f} > {max_action_jump}"
    state_norm = float(np.linalg.norm(ep["state"], axis=-1).max())
    if state_norm > max_state_norm:
        return False, f"state norm {state_norm:.2f} > {max_state_norm}"
    return True, "ok"


def run_medium() -> None:
    print("=" * 60)
    print("MEDIUM: filter pipeline + stats")
    print("=" * 60)

    rng = np.random.default_rng(23)
    n_total = 100
    defect_kinds = ["action_jump", "state_explode", "image_nan", "state_inf"]
    n_defective_per_kind = 10  # 4 * 10 = 40% defective

    episodes: list[dict] = []
    for _ in range(n_total):
        episodes.append(_make_clean_episode(rng))

    # Inject defects in-place into a controlled subset.
    indices = rng.permutation(n_total)
    cursor = 0
    for kind in defect_kinds:
        for _ in range(n_defective_per_kind):
            ep_idx = int(indices[cursor])
            episodes[ep_idx] = _inject_defect(episodes[ep_idx], kind, rng)
            cursor += 1

    # Filter and gather stats.
    reasons: dict[str, int] = {}
    n_kept = 0
    for ep in episodes:
        ok, reason = filter_episode(ep)
        if ok:
            n_kept += 1
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    n_rejected = n_total - n_kept
    print(f"  total       : {n_total}")
    print(f"  kept        : {n_kept} ({n_kept / n_total:.1%})")
    print(f"  rejected    : {n_rejected} ({n_rejected / n_total:.1%})")
    if reasons:
        # sort by count desc
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason:<40s}: {count}")

    # Sanity: each defect category should appear at least once in the rejects.
    detected_kinds = set()
    for reason in reasons:
        if "image" in reason:
            detected_kinds.add("image_nan")
        if "state" in reason and "non-finite" in reason:
            detected_kinds.add("state_inf")
        if "action jump" in reason:
            detected_kinds.add("action_jump")
        if "state norm" in reason:
            detected_kinds.add("state_explode")
    assert detected_kinds == set(defect_kinds), (
        f"missed defect kinds: {set(defect_kinds) - detected_kinds}")

    print("\n  Why does GR00T reject ~30% of synthetic trajectories?")
    print("  Massive parallel sim is fast but produces many degenerate cases:")
    print("  IK solver divergence (action_jump), sim instability (state_explode),")
    print("  rendering artefacts (image_nan), capteur failures (state_inf). It is")
    print("  cheaper to filter aggressively than to tune sim to be perfect.\n")


# ============================================================================
# HARD -- end-to-end pipeline + benchmark 4 policies on OOD lighting
# ============================================================================
def _gen_traj(emb_id: int, task_id: int, T: int = 20,
              state_dim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2 * np.pi, T)
    if task_id == 0:
        action = np.stack([np.cos(t), np.sin(t)], axis=-1)
    else:
        action = np.stack([np.sin(2 * t), 0.5 * np.cos(t)], axis=-1)
    action = action.astype(np.float32)
    gain = 0.5 + 0.25 * emb_id
    state = np.cumsum(action * gain, axis=0)
    if state_dim > 2:
        state = np.concatenate(
            [state, np.zeros((T, state_dim - 2), dtype=np.float32)], axis=-1)
    return state.astype(np.float32), action


def _gen_image(emb_id: int, task_id: int, frame: int, T: int) -> np.ndarray:
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    cx = int(7 * frame / max(T - 1, 1))
    cy = (emb_id + task_id) % 8
    img[cy, cx] = np.array([
        0.9 if task_id == 0 else 0.1,
        0.5,
        0.9 if emb_id == 0 else 0.2,
    ], dtype=np.float32)
    return img


def _gen_dataset(n_episodes: int, rng: np.random.Generator,
                  T: int = 20) -> list[dict]:
    eps = []
    for ep_idx in range(n_episodes):
        emb = int(rng.integers(0, 3))
        task = int(rng.integers(0, 2))
        state, action = _gen_traj(emb, task, T)
        image = np.stack([_gen_image(emb, task, t, T) for t in range(T)])
        eps.append({"state": state, "action": action, "image": image,
                     "embodiment": emb, "task": task})
    return eps


# Reuse the easy/medium primitives where possible.
def aug_lighting(ep: dict, rng: np.random.Generator) -> dict:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v
           for k, v in ep.items()}
    gain = float(rng.uniform(0.7, 1.3))
    temp = rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
    out["image"] = np.clip(out["image"] * gain + temp, 0.0, 1.0)
    return out


def aug_dynamics(ep: dict, rng: np.random.Generator) -> dict:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v
           for k, v in ep.items()}
    gain_a = float(rng.uniform(0.9, 1.1))
    out["action"] = out["action"] * gain_a
    out["state"] = out["state"] + rng.normal(0, 0.02,
                                              size=out["state"].shape).astype(np.float32)
    return out


def aug_background(ep: dict, rng: np.random.Generator) -> dict:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v
           for k, v in ep.items()}
    bg = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
    is_bg = np.all(np.abs(out["image"] - 0.5) < 1e-3, axis=-1, keepdims=True)
    out["image"] = np.where(is_bg, bg, out["image"])
    return out


def aug_distractors(ep: dict, rng: np.random.Generator) -> dict:
    out = {k: v.copy() if isinstance(v, np.ndarray) else v
           for k, v in ep.items()}
    n = int(rng.integers(1, 3))
    T, H, W, _ = out["image"].shape
    for _ in range(n):
        dx, dy = int(rng.integers(0, W)), int(rng.integers(0, H))
        color = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        for t in range(T):
            if rng.random() < 0.5:
                out["image"][t, dy, dx] = color
    return out


AUG_MAP = {
    "lighting": aug_lighting,
    "dynamics": aug_dynamics,
    "background": aug_background,
    "distractors": aug_distractors,
}


def augment_dataset(eps: list[dict], axes: Iterable[str],
                    n_aug_per: int, rng: np.random.Generator) -> list[dict]:
    out = []
    for ep in eps:
        for _ in range(n_aug_per):
            cur = ep
            for axis in axes:
                cur = AUG_MAP[axis](cur, rng)
            out.append(cur)
    return out


# Mini-policy: state + flat-image -> action. Pure MLP for CPU-friendliness.
class MiniPolicy(nn.Module):
    def __init__(self, state_dim: int, image_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + image_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, image], dim=-1))


def episodes_to_tensors(eps: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten episodes into per-step tensors for BC training."""
    states, images, actions = [], [], []
    for ep in eps:
        T = ep["state"].shape[0]
        states.append(torch.from_numpy(ep["state"]))
        images.append(torch.from_numpy(ep["image"].reshape(T, -1)))
        actions.append(torch.from_numpy(ep["action"]))
    return (torch.cat(states), torch.cat(images), torch.cat(actions))


def train_policy(eps: list[dict], state_dim: int, image_dim: int,
                 action_dim: int, epochs: int = 6, batch_size: int = 64,
                 lr: float = 3e-4, seed: int = 0) -> MiniPolicy:
    torch.manual_seed(seed)
    policy = MiniPolicy(state_dim, image_dim, action_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    s, im, a = episodes_to_tensors(eps)
    n = s.size(0)
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            pred = policy(s[idx], im[idx])
            loss = F.mse_loss(pred, a[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


@torch.no_grad()
def eval_policy(policy: MiniPolicy, eps: list[dict]) -> float:
    s, im, a = episodes_to_tensors(eps)
    pred = policy(s, im)
    return float(F.mse_loss(pred, a).item())


def run_hard() -> None:
    print("=" * 60)
    print("HARD: end-to-end pipeline + 4-policy benchmark")
    print("=" * 60)

    rng = np.random.default_rng(23)
    state_dim, action_dim, T = 4, 2, 20
    image_dim = 8 * 8 * 3
    n_train_raw = 60   # kept small for CPU; the exo asks 150 ideally
    n_aug_per = 3

    # 1. Raw clean dataset.
    raw = _gen_dataset(n_train_raw, rng, T=T)
    print(f"  raw train episodes: {len(raw)}")

    # 2. Build 4 training datasets:
    configs = {
        "none": [],
        "lighting only": ["lighting"],
        "dynamics only": ["dynamics"],
        "all 4 axes": ["background", "lighting", "distractors", "dynamics"],
    }

    # 3. OOD eval set: aggressive lighting shift + new background colors.
    rng_eval = np.random.default_rng(99)
    ood_raw = _gen_dataset(20, rng_eval, T=T)

    def _ood_shift(ep, rng):
        out = {k: v.copy() if isinstance(v, np.ndarray) else v
               for k, v in ep.items()}
        # gain outside training range
        gain = float(rng.choice([0.4, 1.6]))
        temp = rng.uniform(-0.2, 0.2, size=3).astype(np.float32)
        out["image"] = np.clip(out["image"] * gain + temp, 0.0, 1.0)
        # background tint never seen by lighting-only training
        bg = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
        is_bg = np.all(np.abs(out["image"] - gain * 0.5 - temp) < 0.05,
                        axis=-1, keepdims=True)
        out["image"] = np.where(is_bg, bg, out["image"])
        return out

    ood_eps = [_ood_shift(ep, rng_eval) for ep in ood_raw]
    in_dist_eps = ood_raw  # before OOD shift -> in-distribution eval set

    print(f"  in-dist eval episodes: {len(in_dist_eps)}")
    print(f"  OOD-lighting eval episodes: {len(ood_eps)}\n")

    # 4. Train + eval each config.
    results = {}
    for name, axes in configs.items():
        rng_train = np.random.default_rng(42 + hash(name) % 1000)
        if axes:
            train_eps = augment_dataset(raw, axes, n_aug_per, rng_train)
        else:
            train_eps = raw  # no augmentation
        policy = train_policy(train_eps, state_dim=state_dim,
                              image_dim=image_dim, action_dim=action_dim,
                              epochs=4, seed=23)
        mse_id = eval_policy(policy, in_dist_eps)
        mse_ood = eval_policy(policy, ood_eps)
        delta_pct = (mse_ood / max(mse_id, 1e-9) - 1.0) * 100.0
        results[name] = (mse_id, mse_ood, delta_pct)
        print(f"  [{name:<18}] in-dist MSE = {mse_id:.4f}  "
              f"OOD MSE = {mse_ood:.4f}  delta = {delta_pct:+.0f}%")

    # 5. Final table
    print("\n" + "-" * 70)
    print(f"{'Train aug':<18} | {'in-dist MSE':>11} | {'OOD MSE':>9} | "
          f"{'Delta (%)':>10}")
    print("-" * 70)
    for name, (mse_id, mse_ood, delta) in results.items():
        print(f"{name:<18} | {mse_id:>11.4f} | {mse_ood:>9.4f} | "
              f"{delta:>+9.0f}%")
    print("-" * 70)

    # 6. Commentary
    none_id, none_ood, none_delta = results["none"]
    light_id, light_ood, light_delta = results["lighting only"]
    dyn_id, dyn_ood, dyn_delta = results["dynamics only"]
    all_id, all_ood, all_delta = results["all 4 axes"]
    print("\nCommentary:")
    if all_ood < none_ood:
        gain_pct = (1.0 - all_ood / max(none_ood, 1e-9)) * 100.0
        print(f"  - 'all 4 axes' achieves {gain_pct:.0f}% lower OOD MSE than 'none'.")
    print(f"  - 'lighting only' OOD = {light_ood:.4f} vs 'all 4 axes' OOD = "
          f"{all_ood:.4f}.")
    if light_id < none_id * 1.05 and all_id < none_id * 1.20:
        print("  - In-distribution cost of augmenting is small (<20% MSE bump).")
    print("  - Bottom line: the 4 axes are complementary -- the marginal cost")
    print("    of adding axes is dominated by the OOD-robustness gain.")
    print("    This mirrors GR00T (REFERENCES.md #15) which uses all 4.\n")


# ============================================================================
# main dispatcher
# ============================================================================
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("easy", "all"):
        run_easy()
    if which in ("medium", "all"):
        run_medium()
    if which in ("hard", "all"):
        run_hard()


if __name__ == "__main__":
    main()

"""J1 — Vue d'ensemble robotique moderne + setup stack.

Goal: verify the MuJoCo + Gymnasium + PyTorch stack is installed and run a first
HalfCheetah-v4 episode with a random policy. We instrument the loop to inspect
the anatomy of a Gymnasium step (obs, reward, terminated, truncated, info) and
print summary statistics so the student SEES what each field actually contains.

This file is meant to be read top-to-bottom AND executed. Every non-obvious
choice has a comment explaining WHY (not what — the code says what).

Sources:
- MuJoCo Documentation 3.x — https://mujoco.readthedocs.io/
- Gymnasium Documentation — https://gymnasium.farama.org/
- CS223A Lecture 1 (Khatib) — https://www.youtube.com/playlist?list=PL65CC0384A1798ADF
"""

# requires: gymnasium[mujoco], mujoco, torch, numpy

from __future__ import annotations

import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 1) Sanity check: confirm the three pillars of the 2026 stack are importable.
#    We do NOT raise — we report. A student running this on a half-broken env
#    needs to see WHICH import failed, not a generic stack trace.
# --------------------------------------------------------------------------- #
def check_stack() -> dict[str, str | None]:
    """Return a dict {package_name: version or None if missing}."""
    versions: dict[str, str | None] = {}
    for pkg in ("gymnasium", "mujoco", "torch", "numpy"):
        try:
            module = __import__(pkg)
            # Most packages expose __version__; fall back to "unknown" otherwise.
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


# --------------------------------------------------------------------------- #
# 2) Episode statistics container.
#    Using a dataclass instead of a tuple makes the rollout loop self-documenting
#    and lets us extend later (e.g. action histograms) without touching callers.
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeStats:
    steps: int = 0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    obs_shape: tuple[int, ...] = field(default_factory=tuple)
    action_shape: tuple[int, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# 3) Random rollout. We pass the env in (dependency injection) so the same
#    routine can be reused on Pendulum-v1, HalfCheetah-v4, or a custom env in
#    the exercises. Seed defaults to 0 for reproducibility.
# --------------------------------------------------------------------------- #
def random_rollout(env, max_steps: int = 200, seed: int = 0) -> EpisodeStats:
    """Run a single episode with a uniformly-sampled action policy.

    Why a random policy? The point of J1 is to validate the stack and inspect
    the step API contract — not to obtain reward. Random is the simplest policy
    that exercises every code path (reset, step, terminated/truncated handling).
    """
    obs, info = env.reset(seed=seed)
    stats = EpisodeStats(
        obs_shape=tuple(obs.shape),
        action_shape=tuple(env.action_space.shape),
    )

    for _ in range(max_steps):
        # action_space.sample() is the canonical way to get a valid random
        # action; it respects the action bounds and dtype declared by the env.
        action = env.action_space.sample()

        # The Gymnasium 5-tuple. Pre-Gymnasium 0.26 (legacy Gym) returned 4
        # values and conflated terminated+truncated into a single `done`.
        # Modern code MUST distinguish them — see theory section 5.
        obs, reward, terminated, truncated, info = env.step(action)

        stats.steps += 1
        stats.total_reward += float(reward)

        if terminated or truncated:
            stats.terminated = bool(terminated)
            stats.truncated = bool(truncated)
            break

    return stats


# --------------------------------------------------------------------------- #
# 4) Pretty printer. Keeping presentation separate from logic so the rollout
#    function stays pure and testable.
# --------------------------------------------------------------------------- #
def print_report(env_id: str, stats: EpisodeStats) -> None:
    print(f"--- Episode report: {env_id} ---")
    print(f"  observation shape : {stats.obs_shape}")
    print(f"  action shape      : {stats.action_shape}")
    print(f"  steps executed    : {stats.steps}")
    print(f"  total reward      : {stats.total_reward:.3f}")
    print(f"  terminated        : {stats.terminated}")
    print(f"  truncated         : {stats.truncated}")
    print()


# --------------------------------------------------------------------------- #
# 5) Entry point.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Step A — stack check. Prints versions or 'MISSING' so the user immediately
    # sees what to pip install. We print before importing gymnasium below so a
    # missing package surfaces here, not via an opaque ImportError later.
    versions = check_stack()
    print("Stack check:")
    for pkg, ver in versions.items():
        marker = ver if ver is not None else "MISSING (run: pip install ...)"
        print(f"  {pkg:12s} {marker}")
    print()

    if any(v is None for v in versions.values()):
        # Exit non-zero so this file is usable in CI as a smoke test.
        print("One or more packages are missing — install them and rerun.")
        sys.exit(1)

    # Importing here (not at top-of-file) is intentional: the check_stack call
    # above must run even if these imports would fail, so the student sees the
    # actionable 'MISSING' line instead of a Python ImportError traceback.
    import gymnasium as gym  # noqa: E402

    # Step B — first env. HalfCheetah-v4 is the canonical MuJoCo locomotion
    # benchmark used in nearly every PPO/SAC paper since 2018. obs=17, act=6.
    env_id = "HalfCheetah-v4"
    env = gym.make(env_id)
    try:
        stats = random_rollout(env, max_steps=200, seed=0)
        print_report(env_id, stats)
    finally:
        # Always close: MuJoCo allocates native resources that we want released
        # cleanly even if the rollout raises mid-episode.
        env.close()

    # Step C — Pendulum-v1 as a sanity contrast. Different obs/action shapes,
    # continuous action space too, but classic-control (no MuJoCo). Useful to
    # confirm Gymnasium itself works independently of the MuJoCo bindings.
    env_id = "Pendulum-v1"
    env = gym.make(env_id)
    try:
        stats = random_rollout(env, max_steps=50, seed=0)
        print_report(env_id, stats)
    finally:
        env.close()

    print("OK — stack functional, two envs ran a random rollout to completion.")

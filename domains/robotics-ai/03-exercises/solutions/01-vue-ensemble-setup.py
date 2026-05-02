"""J1 — Solutions for the three exercises (easy / medium / hard).

Each section is independently runnable from the smoke test in __main__.

Sources:
- Gymnasium docs (env API + custom env): https://gymnasium.farama.org/
- MuJoCo docs (HalfCheetah-v4): https://mujoco.readthedocs.io/
"""

# requires: gymnasium[mujoco], mujoco, numpy

from __future__ import annotations

import sys


# === EASY === #
# 10 random steps on Pendulum-v1, log every step, print obs/action shapes.
def run_easy() -> None:
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=42)
        for t in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"t={t} reward={float(reward):.3f} "
                f"terminated={terminated} truncated={truncated}"
            )
        # Shapes are inspected AFTER the loop on purpose: it forces the student
        # to realize obs is the post-step observation, not the initial one.
        print(f"obs shape    : {obs.shape}")
        print(f"action shape : {env.action_space.sample().shape}")
    finally:
        env.close()


# === MEDIUM === #
# 5 episodes on HalfCheetah-v4, per-episode stats + aggregate mean/std.
def run_medium() -> None:
    # Guarded imports: explain to the student exactly what to install if either
    # gymnasium or mujoco is missing. This pattern is reused in real codebases
    # for optional heavy deps.
    try:
        import gymnasium as gym
        import mujoco  # noqa: F401  (imported for side-effect of validating install)
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print('Install with: pip install "gymnasium[mujoco]" mujoco numpy')
        sys.exit(1)

    env = gym.make("HalfCheetah-v4")
    rewards: list[float] = []
    lengths: list[int] = []

    print(f"{'episode':>8} {'steps':>6} {'reward':>10} {'end_reason':>12}")
    try:
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            # No max_steps cap here — we let the env's TimeLimit wrapper handle
            # truncation (default 1000 for HalfCheetah-v4). That's the correct
            # idiom: trust the env contract instead of imposing our own.
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
            end_reason = "terminated" if terminated else "truncated"
            rewards.append(total_reward)
            lengths.append(steps)
            print(f"{ep:>8d} {steps:>6d} {total_reward:>10.2f} {end_reason:>12}")
    finally:
        env.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    print(f"mean_reward = {mean_r:.2f}, std_reward = {std_r:.2f}, mean_length = {mean_len:.1f}")
    # Sanity comment for the student: random policy on HalfCheetah typically
    # scores ~-300..-50 over 1000 steps. Anything around +1000 means you wired
    # the wrong env (e.g. Hopper or a reward-shaped variant).


# === HARD === #
# Custom Gymnasium env: 1D point with quadratic reward, registered + checked.
def _build_one_d_point_env_class():
    """Factory so the class is built only when gymnasium is importable."""
    import gymnasium as gym
    import numpy as np

    class OneDPointEnv(gym.Env):
        """A point on [-10, 10] with action = velocity in [-1, 1]."""

        metadata: dict = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            # float32 is intentional: matches torch default on most ops and
            # avoids silent dtype promotion when feeding into a policy net.
            self.observation_space = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(1,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self._x: float = 0.0
            self._t: int = 0

        def reset(self, seed: int | None = None, options=None):
            super().reset(seed=seed)
            # Use the seeded np_random provided by gym.Env for reproducibility.
            self._x = float(self.np_random.uniform(-1.0, 1.0))
            self._t = 0
            obs = np.array([self._x], dtype=np.float32)
            return obs, {}

        def step(self, action):
            # Clip to advertised bounds so downstream code (policy nets) cannot
            # accidentally violate the action_space contract.
            a = float(np.clip(action, -1.0, 1.0))
            self._x = self._x + a * 0.1
            self._t += 1
            reward = -(self._x ** 2)  # quadratic, dense, signed.
            terminated = bool(abs(self._x) > 10.0)
            truncated = bool(self._t >= 200)
            obs = np.array([self._x], dtype=np.float32)
            return obs, reward, terminated, truncated, {}

    return OneDPointEnv


def run_hard() -> None:
    import gymnasium as gym
    import numpy as np
    from gymnasium.utils.env_checker import check_env

    OneDPointEnv = _build_one_d_point_env_class()

    # Run the official conformance check. It catches dtype mismatches, missing
    # info dicts, mutable observation_space defaults, and other classic bugs.
    check_env(OneDPointEnv(), skip_render_check=True)

    # Register so we can instantiate via gym.make like any standard env.
    # The 'OneDPoint-v0' id is namespaced in the default registry — fine for a
    # course exercise, would be a custom namespace in a real package.
    gym.register(
        id="OneDPoint-v0",
        entry_point=lambda: OneDPointEnv(),
    )

    def random_rollout(env, seed: int) -> tuple[int, float, bool, bool]:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            total += float(r)
            steps += 1
        return steps, total, terminated, truncated

    def aggregate(env_id: str, n: int = 10) -> tuple[float, float, float]:
        rewards = []
        lengths = []
        env = gym.make(env_id)
        try:
            for s in range(n):
                steps, total, _, _ = random_rollout(env, seed=s)
                rewards.append(total)
                lengths.append(steps)
        finally:
            env.close()
        return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(lengths))

    own_mean, own_std, own_len = aggregate("OneDPoint-v0", n=10)
    pen_mean, pen_std, pen_len = aggregate("Pendulum-v1", n=10)

    print(f"{'env':<14} {'mean_reward':>14} {'std_reward':>14} {'mean_len':>10}")
    print(f"{'OneDPoint-v0':<14} {own_mean:>14.3f} {own_std:>14.3f} {own_len:>10.1f}")
    print(f"{'Pendulum-v1':<14} {pen_mean:>14.3f} {pen_std:>14.3f} {pen_len:>10.1f}")


# --------------------------------------------------------------------------- #
# Smoke test — runs all three sections, but each in a try/except so that a
# missing optional dep (e.g. mujoco) doesn't block the others.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== EASY ===")
    try:
        run_easy()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional in smoke test
        print(f"easy section failed: {exc!r}")

    print("\n=== MEDIUM ===")
    try:
        run_medium()
    except SystemExit:
        # run_medium calls sys.exit(1) if mujoco is missing — surface but don't
        # propagate so the hard section still runs.
        print("medium section skipped (missing dependency).")
    except Exception as exc:  # noqa: BLE001
        print(f"medium section failed: {exc!r}")

    print("\n=== HARD ===")
    try:
        run_hard()
    except Exception as exc:  # noqa: BLE001
        print(f"hard section failed: {exc!r}")

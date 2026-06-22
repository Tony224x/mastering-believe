"""
Solutions J12 - SAC, MPC, Model-based RL.

Trois solutions pedagogiques, executables independamment :
  - solve_easy()   : open-loop vs closed-loop sur 20 steps de pendule.
  - solve_medium() : MPC avec modele biaise (model bias illustre).
  - solve_hard()   : mini-Dyna sketch (modele appris numpy + critic tabulaire-discretise).

On reutilise les fonctions du fichier 02-code/12-sac-mpc-model-based.py via copy
locale (le fichier de solutions doit rester self-contained et tourner py_compile
sans dependre d'un import relatif que pytest/CI ne saurait pas resoudre).

# requires: numpy, torch (optional), stable-baselines3 (optional)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Re-implementation locale du pendule (evite import relatif fragile).
# -----------------------------------------------------------------------------


@dataclass
class PendulumParams:
    g: float = 10.0
    L: float = 1.0
    m: float = 1.0
    b: float = 0.1
    dt: float = 0.05
    u_max: float = 2.0


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def pendulum_step(state: np.ndarray, u: np.ndarray, p: PendulumParams) -> np.ndarray:
    theta, theta_dot = state[..., 0], state[..., 1]
    u_clipped = np.clip(u, -p.u_max, p.u_max)
    theta_ddot = (
        (p.g / p.L) * np.sin(theta) - p.b * theta_dot + u_clipped / (p.m * p.L**2)
    )
    new_theta_dot = theta_dot + theta_ddot * p.dt
    new_theta = angle_normalize(theta + new_theta_dot * p.dt)
    return np.stack([new_theta, new_theta_dot], axis=-1)


def pendulum_reward(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    theta, theta_dot = state[..., 0], state[..., 1]
    return -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)


def rollout_reward(
    init_state: np.ndarray, actions: np.ndarray, p: PendulumParams
) -> np.ndarray:
    N, H = actions.shape
    states = np.broadcast_to(init_state, (N, 2)).copy()
    total = np.zeros(N)
    for t in range(H):
        u_t = actions[:, t]
        total += pendulum_reward(states, u_t)
        states = pendulum_step(states, u_t, p)
    return total


def cem_plan(
    state: np.ndarray,
    p: PendulumParams,
    horizon: int = 20,
    n_samples: int = 200,
    n_elites: int = 20,
    n_iters: int = 5,
    init_mean: np.ndarray | None = None,
) -> np.ndarray:
    if init_mean is None:
        mean = np.zeros(horizon)
    else:
        mean = init_mean.copy()
    std = np.full(horizon, p.u_max / 2)
    for _ in range(n_iters):
        noise = np.random.randn(n_samples, horizon) * std[None, :]
        actions = np.clip(mean[None, :] + noise, -p.u_max, p.u_max)
        rewards = rollout_reward(state, actions, p)
        elite_idx = np.argsort(rewards)[-n_elites:]
        elites = actions[elite_idx]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
    return mean


# =============================================================================
# Easy : open-loop vs closed-loop
# =============================================================================
def solve_easy(seed: int = 0) -> dict:
    """
    Compare 20 steps open-loop (1 plan applique tel quel) vs closed-loop (re-plan a chaque step).
    Sans bruit puis avec bruit observable.
    """
    np.random.seed(seed)
    p = PendulumParams()
    init_state = np.array([np.pi + 0.01 * np.random.randn(), 0.0])

    # --- Open-loop : 1 seule planification au step 0
    plan_once = cem_plan(init_state, p, horizon=20)
    state = init_state.copy()
    open_reward = 0.0
    for t in range(20):
        u = plan_once[t]
        open_reward += float(pendulum_reward(state, u))
        state = pendulum_step(state, np.array(u), p)

    # --- Closed-loop : re-plan a chaque step
    state = init_state.copy()
    closed_reward = 0.0
    plan = np.zeros(20)
    for t in range(20):
        plan = cem_plan(state, p, horizon=20, init_mean=plan)
        u = plan[0]
        closed_reward += float(pendulum_reward(state, u))
        state = pendulum_step(state, np.array(u), p)
        plan = np.concatenate([plan[1:], plan[-1:]])

    # --- Open-loop avec bruit observation injecte (simule erreur de modele/capteur)
    state = init_state.copy()
    open_noisy_reward = 0.0
    for t in range(20):
        u = plan_once[t]
        open_noisy_reward += float(pendulum_reward(state, u))
        state = pendulum_step(state, np.array(u), p)
        state = state + 0.05 * np.random.randn(2)  # perturbation par step

    return {
        "open_loop_clean": open_reward,
        "closed_loop_clean": closed_reward,
        "open_loop_noisy": open_noisy_reward,
        "verdict": (
            "Closed-loop > open-loop des qu'il y a bruit. Sans bruit la difference est marginale."
        ),
    }


# =============================================================================
# Medium : MPC avec modele biaise
# =============================================================================
def cem_plan_with_model(
    state: np.ndarray,
    p_model: PendulumParams,  # le planner croit en p_model
    horizon: int = 20,
    n_samples: int = 200,
    n_elites: int = 20,
    n_iters: int = 5,
    init_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Variante de cem_plan qui utilise p_model pour les rollouts internes."""
    return cem_plan(
        state,
        p_model,
        horizon=horizon,
        n_samples=n_samples,
        n_elites=n_elites,
        n_iters=n_iters,
        init_mean=init_mean,
    )


def run_mpc_with_bias(
    p_real: PendulumParams,
    p_model: PendulumParams,
    n_steps: int = 100,
    horizon: int = 20,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """MPC ou le planner croit en p_model alors que l'env tourne sous p_real."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    state = np.array([np.pi + 0.01 * rng.standard_normal(), 0.0])
    plan = np.zeros(horizon)
    cumulative = 0.0
    thetas = np.zeros(n_steps + 1)
    thetas[0] = state[0]
    for t in range(n_steps):
        plan = cem_plan_with_model(state, p_model, horizon=horizon, init_mean=plan)
        u = plan[0]
        cumulative += float(pendulum_reward(state, u))
        state = pendulum_step(state, np.array(u), p_real)
        thetas[t + 1] = state[0]
        plan = np.concatenate([plan[1:], plan[-1:]])
    return cumulative, thetas


def solve_medium() -> dict:
    """Lance MPC avec 3 niveaux de bias modele et reporte les rewards."""
    p_real = PendulumParams()
    configs = {
        "perfect": PendulumParams(),  # = p_real
        "biased_30pct": PendulumParams(L=1.3, m=1.5),
        "biased_severe": PendulumParams(L=2.0, m=2.5, g=8.0),
    }
    results = {}
    for name, p_model in configs.items():
        reward, thetas = run_mpc_with_bias(p_real, p_model, n_steps=100, seed=0)
        results[name] = {
            "reward": reward,
            "final_theta": float(thetas[-1]),
            "stabilized": abs(thetas[-1]) < 0.5,  # < ~30 degres = on est haut
        }
    return results


# =============================================================================
# Hard : mini-Dyna numpy
# =============================================================================
# On evite torch pour garder ce fichier executable sans dependance optionnelle.
# Strategie : critic tabulaire sur etat discretise (theta, theta_dot) + policy
# epsilon-greedy sur actions discretisees. Modele dynamique = MLP 2-couches en
# numpy avec backprop manuel (delta-state regression).


class TinyMLP:
    """MLP 2 couches en numpy (in -> hidden -> out). Backprop manuel."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, lr: float = 1e-3):
        rng = np.random.default_rng(42)
        # He init
        self.W1 = rng.standard_normal((dim_in, dim_hidden)) * np.sqrt(2.0 / dim_in)
        self.b1 = np.zeros(dim_hidden)
        self.W2 = rng.standard_normal((dim_hidden, dim_out)) * np.sqrt(2.0 / dim_hidden)
        self.b2 = np.zeros(dim_out)
        self.lr = lr

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)  # ReLU
        y = h1 @ self.W2 + self.b2
        cache = {"x": x, "h1": h1, "z1": z1}
        return y, cache

    def step(self, x: np.ndarray, target: np.ndarray) -> float:
        y, cache = self.forward(x)
        diff = y - target  # (B, dim_out)
        loss = float((diff**2).mean())
        # Backprop
        dy = 2 * diff / x.shape[0]
        dW2 = cache["h1"].T @ dy
        db2 = dy.sum(axis=0)
        dh1 = dy @ self.W2.T
        dz1 = dh1 * (cache["z1"] > 0)
        dW1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)
        # SGD
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss


def solve_hard(n_steps: int = 1500, seed: int = 0) -> dict:
    """
    Mini-Dyna pedagogique : critic tabulaire + modele appris MLP. On compare :
      - Baseline : Q-learning sur 1500 steps reels.
      - Dyna   : Q-learning + modele appris + K=5 updates imagines par step.
    On mesure le reward moyen sur les 100 derniers steps de chaque mode.
    """
    np.random.seed(seed)
    p_real = PendulumParams()

    # Discretisation : grille (theta, theta_dot) en 21x21, actions {-2, -1, 0, 1, 2}.
    n_theta, n_thetad = 21, 21
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    thetad_bins = np.linspace(-8, 8, n_thetad + 1)
    actions = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    n_act = len(actions)

    def discretize(state: np.ndarray) -> tuple[int, int]:
        i = int(np.clip(np.digitize(state[0], theta_bins) - 1, 0, n_theta - 1))
        j = int(np.clip(np.digitize(state[1], thetad_bins) - 1, 0, n_thetad - 1))
        return i, j

    def run_one(use_dyna: bool) -> tuple[float, list[float]]:
        Q = np.zeros((n_theta, n_thetad, n_act))
        model = TinyMLP(dim_in=3, dim_hidden=32, dim_out=2, lr=1e-3) if use_dyna else None
        buffer: list[tuple[np.ndarray, int, float, np.ndarray]] = []
        gamma = 0.95
        eps = 0.2
        alpha = 0.3

        state = np.array([np.pi + 0.01 * np.random.randn(), 0.0])
        rewards_log: list[float] = []
        for step in range(n_steps):
            i, j = discretize(state)
            # eps-greedy
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_act)
            else:
                a_idx = int(np.argmax(Q[i, j]))
            u = actions[a_idx]

            r = float(pendulum_reward(state, u))
            next_state = pendulum_step(state, np.array(u), p_real)
            ni, nj = discretize(next_state)

            # TD update (real step)
            target = r + gamma * float(np.max(Q[ni, nj]))
            Q[i, j, a_idx] += alpha * (target - Q[i, j, a_idx])
            buffer.append((state.copy(), a_idx, r, next_state.copy()))

            # Train modele appris + K Dyna updates (sur transitions imaginees)
            if use_dyna and len(buffer) > 32:
                # Batch entrainement modele sur 32 transitions reelles
                idx = np.random.randint(0, len(buffer), size=32)
                xs = np.array(
                    [
                        np.concatenate([buffer[k][0], [actions[buffer[k][1]]]])
                        for k in idx
                    ]
                )
                deltas = np.array([buffer[k][3] - buffer[k][0] for k in idx])
                model.step(xs, deltas)

                # K rollouts imagines de longueur 1 (court = securite cf. MBPO)
                K = 5
                idx_k = np.random.randint(0, len(buffer), size=K)
                for k in idx_k:
                    s_im = buffer[k][0]
                    a_idx_im = np.random.randint(n_act)
                    u_im = actions[a_idx_im]
                    inp = np.concatenate([s_im, [u_im]])[None, :]
                    delta_pred, _ = model.forward(inp)
                    s_next_im = s_im + delta_pred[0]
                    s_next_im[0] = angle_normalize(s_next_im[0])
                    r_im = float(pendulum_reward(s_im, u_im))
                    ii, jj = discretize(s_im)
                    nii, njj = discretize(s_next_im)
                    target_im = r_im + gamma * float(np.max(Q[nii, njj]))
                    Q[ii, jj, a_idx_im] += alpha * (target_im - Q[ii, jj, a_idx_im])

            rewards_log.append(r)
            state = next_state

        return float(np.mean(rewards_log[-100:])), rewards_log

    baseline_avg, _ = run_one(use_dyna=False)
    dyna_avg, _ = run_one(use_dyna=True)
    return {
        "baseline_last100_avg_reward": baseline_avg,
        "dyna_last100_avg_reward": dyna_avg,
        "note": (
            "Dyna devrait converger un peu plus vite (reward moyen plus haut) car chaque "
            "step reel produit 1 + K=5 updates Q. Le gain depend de la qualite du modele "
            "appris. Cf. MBPO (Janner 2019) pour la version SOTA avec ensembles + SAC."
        ),
    }


# =============================================================================
# Entrypoint
# =============================================================================
def main() -> None:
    print("=== Easy ===")
    res_easy = solve_easy()
    for k, v in res_easy.items():
        print(f"  {k}: {v}")

    print("\n=== Medium ===")
    res_med = solve_medium()
    for k, v in res_med.items():
        print(f"  {k}: {v}")

    print("\n=== Hard ===")
    res_hard = solve_hard(n_steps=1500)
    for k, v in res_hard.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

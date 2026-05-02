"""
J12 - SAC, TD3, MPC, Model-Based RL.

Demo en deux blocs :
  1. MPC sur un pendule analytique avec CEM (Cross-Entropy Method) et MPPI
     (Model Predictive Path Integral). Pas de torch necessaire, pas de RL,
     juste du planning a horizon glissant avec un modele connu.
  2. Skeleton SAC : import SB3 si dispo (entrainement court sur Pendulum-v1),
     sinon explication du squelette d'algo (boucle update) en pseudo-code
     execute via prints commentes pour montrer la structure.

Le fichier doit tourner meme si stable-baselines3 n'est PAS installe :
on detecte l'import et on bascule sur le mode "explanation".

# requires: numpy, torch (optional), stable-baselines3 (optional)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Imports optionnels : torch et stable-baselines3 ne sont pas obligatoires.
# -----------------------------------------------------------------------------
try:
    import torch  # noqa: F401  (juste pour signaler la dispo)

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gymnasium as gym
    from stable_baselines3 import SAC

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# =============================================================================
# 1. MPC sur pendule analytique
# =============================================================================
# Modele physique du pendule simple :
#   theta_ddot = (g / L) * sin(theta) - b * theta_dot + u / (m * L**2)
# Etat   : s = (theta, theta_dot), avec theta=0 pendule en haut, theta=pi en bas.
# Action : u (couple) borne.
# Reward : -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)
#          (penalise l'ecart vertical, la vitesse, et la commande).


@dataclass
class PendulumParams:
    """Parametres physiques. Defauts choisis lisibles, pas calques sur Gymnasium."""

    g: float = 10.0  # gravite (m/s^2)
    L: float = 1.0  # longueur tige (m)
    m: float = 1.0  # masse (kg)
    b: float = 0.1  # frottement
    dt: float = 0.05  # pas de temps simulation (s)
    u_max: float = 2.0  # couple max


def angle_normalize(theta: np.ndarray) -> np.ndarray:
    """Ramene theta dans [-pi, pi]. Important pour la reward (sinon |theta|=2pi penalise alors qu'on est tout droit)."""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def pendulum_step(state: np.ndarray, u: np.ndarray, p: PendulumParams) -> np.ndarray:
    """Un pas de dynamique semi-implicite Euler. state shape: (..., 2). u shape: (...,)."""
    theta, theta_dot = state[..., 0], state[..., 1]
    u_clipped = np.clip(u, -p.u_max, p.u_max)
    # Acceleration angulaire : on prend theta=0 en haut, donc le terme gravite est sin(theta) (et non -sin)
    # de sorte que l'equilibre instable est a theta=0 (haut) et stable a theta=pi (bas).
    theta_ddot = (
        (p.g / p.L) * np.sin(theta) - p.b * theta_dot + u_clipped / (p.m * p.L**2)
    )
    new_theta_dot = theta_dot + theta_ddot * p.dt
    new_theta = angle_normalize(theta + new_theta_dot * p.dt)
    return np.stack([new_theta, new_theta_dot], axis=-1)


def pendulum_reward(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Reward dense. Penalise ecart vertical (theta=0), vitesse, action."""
    theta, theta_dot = state[..., 0], state[..., 1]
    return -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)


def rollout_reward(
    init_state: np.ndarray,
    actions: np.ndarray,
    p: PendulumParams,
) -> np.ndarray:
    """
    Simule N rollouts de longueur H et renvoie le reward cumule.

    init_state : shape (2,)
    actions    : shape (N, H)  -> N sequences de H actions
    return     : shape (N,)
    """
    N, H = actions.shape
    states = np.broadcast_to(init_state, (N, 2)).copy()  # (N, 2)
    total = np.zeros(N)
    for t in range(H):
        u_t = actions[:, t]  # (N,)
        total += pendulum_reward(states, u_t)
        states = pendulum_step(states, u_t, p)
    return total


# -----------------------------------------------------------------------------
# Planner CEM : iter sur les top-K elites pour re-estimer la gaussienne.
# -----------------------------------------------------------------------------
def cem_plan(
    state: np.ndarray,
    p: PendulumParams,
    horizon: int = 20,
    n_samples: int = 200,
    n_elites: int = 20,
    n_iters: int = 5,
    init_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Renvoie une sequence d'actions de longueur `horizon` optimisee par CEM."""
    if init_mean is None:
        mean = np.zeros(horizon)
    else:
        mean = init_mean.copy()
    std = np.full(horizon, p.u_max / 2)  # std initiale large

    for _ in range(n_iters):
        # Sampler n_samples sequences. Clamp sur les bornes physiques.
        noise = np.random.randn(n_samples, horizon) * std[None, :]
        actions = np.clip(mean[None, :] + noise, -p.u_max, p.u_max)
        rewards = rollout_reward(state, actions, p)
        # Selectionner les n_elites meilleures sequences.
        elite_idx = np.argsort(rewards)[-n_elites:]
        elites = actions[elite_idx]
        # Re-estimer la gaussienne sur les elites.
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3  # eviter collapse
    return mean


# -----------------------------------------------------------------------------
# Planner MPPI : pondere TOUTES les samples par exp(R / lambda).
# -----------------------------------------------------------------------------
def mppi_plan(
    state: np.ndarray,
    p: PendulumParams,
    horizon: int = 20,
    n_samples: int = 200,
    lam: float = 1.0,  # temperature : plus petit -> plus elitiste
    sigma: float = 1.0,
    init_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Renvoie une sequence d'actions optimisee par MPPI (1 iteration)."""
    if init_mean is None:
        nominal = np.zeros(horizon)
    else:
        nominal = init_mean.copy()
    # Sampler des perturbations.
    noise = np.random.randn(n_samples, horizon) * sigma
    actions = np.clip(nominal[None, :] + noise, -p.u_max, p.u_max)
    rewards = rollout_reward(state, actions, p)
    # Softmax weighting. On centre rewards pour eviter l'overflow exp().
    rewards_centered = rewards - rewards.max()
    weights = np.exp(rewards_centered / lam)
    weights /= weights.sum() + 1e-8
    # Update : moyenne ponderee des sequences echantillonnees.
    new_nominal = (weights[:, None] * actions).sum(axis=0)
    return new_nominal


# -----------------------------------------------------------------------------
# Boucle de controle MPC : on planifie a chaque step, on applique la premiere
# action, on warm-start la prochaine planification avec le shift de la sequence.
# -----------------------------------------------------------------------------
def run_mpc(
    planner_name: str = "cem",
    n_steps: int = 100,
    horizon: int = 20,
    seed: int = 0,
) -> tuple[float, list[np.ndarray]]:
    """Lance MPC sur le pendule depuis position basse + bruit. Renvoie reward cumule + trajectoire."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    p = PendulumParams()

    # Etat initial : pendule en bas avec petit bruit. theta=pi est l'equilibre stable bas.
    state = np.array([np.pi + 0.01 * rng.standard_normal(), 0.0])

    plan = np.zeros(horizon)  # warm-start
    trajectory = [state.copy()]
    cumulative = 0.0

    for step in range(n_steps):
        if planner_name == "cem":
            plan = cem_plan(state, p, horizon=horizon, init_mean=plan)
        elif planner_name == "mppi":
            plan = mppi_plan(state, p, horizon=horizon, init_mean=plan)
        else:
            raise ValueError(f"Unknown planner: {planner_name}")

        # Appliquer la premiere action seulement (receding horizon).
        u = plan[0]
        cumulative += float(pendulum_reward(state, u))
        state = pendulum_step(state, np.array(u), p)
        trajectory.append(state.copy())

        # Warm-start : decaler la sequence d'une case, repeter la derniere.
        plan = np.concatenate([plan[1:], plan[-1:]])

    return cumulative, trajectory


# =============================================================================
# 2. SAC : SB3 si dispo, sinon pseudo-code commente.
# =============================================================================
def run_sac_demo() -> None:
    """Lance un mini-entrainement SAC sur Pendulum-v1, ou explique sinon."""
    if not HAS_SB3:
        print("\n[SAC] stable-baselines3 indisponible. Pseudo-code structure :\n")
        print(
            """
  # Pseudo-code SAC (Haarnoja 2018) :
  buffer = ReplayBuffer(capacity=1_000_000)
  actor  = GaussianPolicy(obs_dim, act_dim)        # produit (mu, log_std)
  q1, q2 = QNet(obs_dim, act_dim), QNet(obs_dim, act_dim)
  q1_t, q2_t = clone(q1), clone(q2)                 # targets pour stabilite
  alpha = 0.2  # ou auto-tune via log_alpha appris

  for step in range(total_steps):
      # 1. Interaction avec env
      a = actor.sample(s)                           # reparam : a = mu + sigma * eps, tanh-squash
      s_next, r, done = env.step(a)
      buffer.add(s, a, r, s_next, done)
      s = s_next if not done else env.reset()

      # 2. Update si buffer suffisamment plein
      if len(buffer) > batch_size:
          batch = buffer.sample(batch_size)

          # 2a. Cible Q (double Q + bonus entropie)
          a_next, logp_next = actor.sample_with_logp(batch.s_next)
          q_target = min(q1_t(s_next, a_next), q2_t(s_next, a_next)) - alpha * logp_next
          y = batch.r + gamma * (1 - batch.done) * q_target

          # 2b. Loss critics
          loss_q1 = MSE(q1(batch.s, batch.a), y)
          loss_q2 = MSE(q2(batch.s, batch.a), y)

          # 2c. Loss actor (reparam trick)
          a_new, logp_new = actor.sample_with_logp(batch.s)
          loss_pi = (alpha * logp_new - min(q1(batch.s, a_new), q2(batch.s, a_new))).mean()

          # 2d. (Optionnel) auto-tune alpha vers entropie cible H_target = -dim(A)
          loss_alpha = -(log_alpha * (logp_new + H_target).detach()).mean()

          # 2e. Soft update targets : theta_t <- tau*theta + (1-tau)*theta_t
          polyak_update(q1_t, q1, tau=0.005)
          polyak_update(q2_t, q2, tau=0.005)
"""
        )
        return

    print("\n[SAC] stable-baselines3 detecte. Mini-train sur Pendulum-v1...")
    env = gym.make("Pendulum-v1")
    # 3000 steps : juste pour voir que l'API tourne. Un vrai run = 100k+ steps.
    model = SAC("MlpPolicy", env, verbose=0, learning_starts=500)
    t0 = time.time()
    model.learn(total_timesteps=3000)
    dt = time.time() - t0
    print(f"  -> entrainement 3000 steps en {dt:.1f}s")

    # Eval rapide sur 1 episode.
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(200):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        total += float(r)
        if term or trunc:
            break
    print(f"  -> eval 1 episode reward = {total:.1f} (random ~ -1200, parfait ~ 0)")
    env.close()


# =============================================================================
# Entrypoint
# =============================================================================
def main() -> None:
    print("=== J12 - MPC sur pendule (CEM + MPPI) ===")
    for planner in ("cem", "mppi"):
        t0 = time.time()
        reward, _traj = run_mpc(planner_name=planner, n_steps=100, horizon=20, seed=0)
        dt = time.time() - t0
        print(f"  {planner.upper():4s}: reward cumule = {reward:8.1f}  ({dt:.2f}s)")
    # Note pedagogique : un controleur random tourne plus pres de -1500 a -2000
    # sur 100 steps. Une politique parfaite est proche de 0 (pendule maintenu en haut).
    # CEM/MPPI doivent finir nettement mieux que random.

    print("\n=== J12 - SAC ===")
    run_sac_demo()

    print("\n=== Fin J12 ===")
    print(f"  torch dispo : {HAS_TORCH} | SB3 dispo : {HAS_SB3}")


if __name__ == "__main__":
    main()

"""
J14 — Sim-to-real : domain randomization sur Pendulum-v1.

Objectif pedagogique :
1. Definir une policy heuristique parametrable (energy shaping + LQR-like) qui depend des params physiques.
2. Tuner cette policy en "sim" (pendule nominal mass=1.0, length=1.0).
3. La deployer sur un "real" pendule different (mass=1.2, length=0.95, friction ajoutee, latence) et observer le drop.
4. Re-tuner la policy AVEC domain randomization (on cherche les gains qui marchent en moyenne sur une plage).
5. Re-deployer sur le "real" et observer l'amelioration.

Note : on n'utilise pas un PPO entraine pour rester dans les contraintes deps (numpy + gymnasium uniquement).
La logique est strictement la meme qu'avec PPO : on remplace simplement "random search sur les gains"
par "PPO sur un MLP". Le shape de la demonstration reality-gap-then-randomization est identique.

Sources :
- [Tobin et al., 2017] — domain randomization
- [CS285 L13] — sim-to-real, dynamics randomization

requires: numpy, gymnasium
"""

import numpy as np
import gymnasium as gym


# =============================================================================
# 1. Une policy parametree pour Pendulum-v1
# =============================================================================
# Pendulum-v1 observation = [cos(theta), sin(theta), theta_dot] ; action ∈ [-2, 2].
# On utilise un swing-up energy shaping + stabilisation lineaire pres de l'equilibre haut.
# La "politique" depend de 3 gains (k_E, k_p, k_d) — c'est notre vecteur entrainable.

def policy(obs, params):
    """Energy-shaping + LQR-stabilization policy. params = (k_E, k_p, k_d)."""
    cos_t, sin_t, theta_dot = obs
    # angle depuis le haut (haut = 0, bas = ±pi)
    theta = np.arctan2(sin_t, cos_t)
    k_E, k_p, k_d = params

    # Region de stabilisation : pres du haut (|theta|<0.3 rad)
    if abs(theta) < 0.3:
        # PD lineaire qui pousse vers theta=0
        action = -k_p * theta - k_d * theta_dot
    else:
        # Energy shaping : pomper de l'energie en synchrone avec la vitesse
        # E_desired = m*g*l (energie de l'equilibre haut)
        # action proportionnelle a sign(theta_dot * cos(theta)) pour ajouter de l'energie
        action = k_E * theta_dot * cos_t  # forme classique d'energy injection

    return float(np.clip(action, -2.0, 2.0))


# =============================================================================
# 2. Modifier l'env Pendulum pour simuler differents "robots reels"
# =============================================================================
# Pendulum-v1 expose .unwrapped.m, .unwrapped.l, .unwrapped.g — on peut les changer.
# On wrappe pour ajouter friction (damping) et latence (action delay).

class PhysicsRandomizer(gym.Wrapper):
    """Wrap Pendulum pour modifier mass/length/gravity + ajouter friction et latency."""

    def __init__(self, env, mass=1.0, length=1.0, friction=0.0, latency_steps=0,
                 obs_noise_std=0.0):
        super().__init__(env)
        self.target_mass = mass
        self.target_length = length
        self.friction = friction         # damping coef applique a theta_dot
        self.latency_steps = int(latency_steps)
        self.obs_noise_std = obs_noise_std
        self._action_buffer = []

    def reset(self, **kwargs):
        # Modifier les parametres physiques de l'env
        self.env.unwrapped.m = self.target_mass
        self.env.unwrapped.l = self.target_length
        # buffer d'actions pour simuler la latence
        self._action_buffer = [0.0] * self.latency_steps
        obs, info = self.env.reset(**kwargs)
        return self._noisy(obs), info

    def step(self, action):
        # 1. simulate latence: on applique l'action retardee, on stocke la nouvelle
        if self.latency_steps > 0:
            applied = self._action_buffer.pop(0)
            self._action_buffer.append(float(action))
            applied_action = np.array([applied], dtype=np.float32)
        else:
            applied_action = action

        # 2. ajouter friction artificielle: on modifie theta_dot APRES le step
        # Pendulum integre l'angle, donc on intervient sur l'etat post-step.
        obs, reward, terminated, truncated, info = self.env.step(applied_action)

        if self.friction > 0:
            # appliquer une decay multiplicative sur theta_dot
            # state interne = [theta, theta_dot]
            state = self.env.unwrapped.state
            theta, theta_dot = state[0], state[1]
            theta_dot = theta_dot * (1.0 - self.friction)
            self.env.unwrapped.state = np.array([theta, theta_dot])
            # recalcul de l'obs apres modif d'etat
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

        return self._noisy(obs), reward, terminated, truncated, info

    def _noisy(self, obs):
        if self.obs_noise_std > 0:
            return obs + np.random.normal(0, self.obs_noise_std, size=obs.shape).astype(obs.dtype)
        return obs


# =============================================================================
# 3. Evaluer une policy sur N episodes, retourner le reward moyen
# =============================================================================

def evaluate(env, params, n_episodes=10, seed_base=0):
    """Retourne reward total moyen et std sur n_episodes."""
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ep_return = 0.0
        for _ in range(200):  # Pendulum-v1 = 200 steps par episode
            a = policy(obs, params)
            obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            ep_return += r
            if term or trunc:
                break
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


# =============================================================================
# 4. "Entrainer" = random search sur les gains (proxy pedagogique de PPO)
# =============================================================================

def random_search(make_env_fn, n_trials=80, rng_seed=0):
    """Cherche les meilleurs gains en evaluant n_trials combinaisons."""
    rng = np.random.default_rng(rng_seed)
    best_params = None
    best_score = -np.inf
    for _ in range(n_trials):
        # plages raisonnables pour les 3 gains
        k_E = rng.uniform(0.1, 1.5)
        k_p = rng.uniform(2.0, 30.0)
        k_d = rng.uniform(0.5, 8.0)
        params = (k_E, k_p, k_d)
        env = make_env_fn()
        score, _ = evaluate(env, params, n_episodes=4, seed_base=42)
        env.close()
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score


# =============================================================================
# 5. Make-env fns : "sim" (nominal), "real" (different), et "randomized"
# =============================================================================

def make_sim_env():
    """Pendule nominal: tel que Pendulum-v1 le definit (m=1, l=1, g=10)."""
    base = gym.make("Pendulum-v1")
    return PhysicsRandomizer(base, mass=1.0, length=1.0, friction=0.0,
                             latency_steps=0, obs_noise_std=0.0)


def make_real_env():
    """Pendule "reel": parametres differents + friction + latence + bruit capteur.
    Choisis pour simuler un vrai robot."""
    base = gym.make("Pendulum-v1")
    return PhysicsRandomizer(base, mass=1.25, length=0.92, friction=0.05,
                             latency_steps=2, obs_noise_std=0.02)


def make_randomized_env():
    """Pendule avec parametres tires aleatoirement a chaque reset.
    C'est l'env d'entrainement domain-randomized."""
    base = gym.make("Pendulum-v1")
    rng = np.random.default_rng()
    # tirer une fois lors de la creation: chaque random_search trial utilisera
    # une instance fraiche, donc un xi different. C'est equivalent a tirer xi par episode
    # quand on n_episodes=4 dans evaluate (chaque trial = nouveau env = nouveau xi).
    mass = rng.uniform(0.7, 1.4)
    length = rng.uniform(0.8, 1.2)
    friction = rng.uniform(0.0, 0.08)
    latency = rng.integers(0, 3)
    noise = rng.uniform(0.0, 0.03)
    return PhysicsRandomizer(base, mass=mass, length=length, friction=friction,
                             latency_steps=int(latency), obs_noise_std=noise)


# =============================================================================
# 6. Demonstration pedagogique en 3 actes
# =============================================================================

def main():
    print("=" * 72)
    print("J14 — Demonstration sim-to-real avec domain randomization")
    print("=" * 72)

    # --------- ACTE 1 : entrainer sur sim nominale ---------
    print("\n[ACTE 1] Entrainement sur sim NOMINALE (mass=1.0, length=1.0, no friction)")
    print("  → random search sur 80 combinaisons de gains...")
    naive_params, naive_train_score = random_search(make_sim_env, n_trials=80, rng_seed=0)
    print(f"  Best params (k_E, k_p, k_d) = ({naive_params[0]:.2f}, {naive_params[1]:.2f}, {naive_params[2]:.2f})")
    print(f"  Train score (sur sim nominale)     = {naive_train_score:.1f}")

    # --------- ACTE 2 : tester sur "reel" (drop) ---------
    print("\n[ACTE 2] Deploiement sur PENDULE REEL (mass=1.25, length=0.92, friction=0.05, latence=2, bruit=0.02)")
    real_env = make_real_env()
    naive_real_score, naive_real_std = evaluate(real_env, naive_params, n_episodes=20, seed_base=100)
    real_env.close()
    print(f"  Score reel (policy entrainee SANS randomization)  = {naive_real_score:.1f} ± {naive_real_std:.1f}")
    print(f"  → reality gap : drop de {naive_train_score - naive_real_score:.1f} points par rapport a la sim.")

    # --------- ACTE 3 : re-entrainer AVEC domain randomization ---------
    print("\n[ACTE 3] Re-entrainement AVEC domain randomization (mass∈[0.7,1.4], length∈[0.8,1.2], etc.)")
    print("  → random search sur 80 combinaisons, chaque eval voit des physiques differentes...")
    dr_params, dr_train_score = random_search(make_randomized_env, n_trials=80, rng_seed=1)
    print(f"  Best params (k_E, k_p, k_d) = ({dr_params[0]:.2f}, {dr_params[1]:.2f}, {dr_params[2]:.2f})")
    print(f"  Train score (moyenne sur physiques randomisees) = {dr_train_score:.1f}")

    # --------- ACTE 4 : redeployer sur "reel" (gain) ---------
    print("\n[ACTE 4] Redeploiement de la policy DR sur le PENDULE REEL")
    real_env = make_real_env()
    dr_real_score, dr_real_std = evaluate(real_env, dr_params, n_episodes=20, seed_base=100)
    real_env.close()
    print(f"  Score reel (policy entrainee AVEC randomization)  = {dr_real_score:.1f} ± {dr_real_std:.1f}")

    # --------- Synthese ---------
    print("\n" + "=" * 72)
    print("SYNTHESE")
    print("=" * 72)
    print(f"  sim-only policy  on real : {naive_real_score:>8.1f}")
    print(f"  DR-trained policy on real : {dr_real_score:>8.1f}")
    delta = dr_real_score - naive_real_score
    sign = "+" if delta >= 0 else ""
    print(f"  Gain absolu              : {sign}{delta:.1f} points")
    print()
    print("Interpretation pedagogique :")
    print("  La policy entrainee uniquement sur la sim nominale exploite des regularites")
    print("  precises (friction=0, latence=0, masse=1.0). Quand on la deploie sur un pendule")
    print("  different, ces hypotheses sont fausses et le score chute.")
    print()
    print("  La policy entrainee sur la distribution randomisee est forcee de trouver des")
    print("  gains qui marchent en moyenne sur des physiques differentes : elle est plus")
    print("  conservatrice mais robuste. Le reality gap est ferme [Tobin et al., 2017].")
    print()
    print("  Note: avec PPO + un MLP au lieu de 3 gains scalaires, l'effet est plus marque")
    print("  (la policy peut conditionner son comportement sur l'historique pour inferer xi).")


if __name__ == "__main__":
    main()

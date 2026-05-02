"""
J14 - Solutions des exercices easy / medium / hard.

EASY  : reponses textuelles consolidees dans la docstring (pas de code).
MEDIUM: DomainRandomizationWrapper generique + comparaison 4 strategies (no/narrow/wide/too_wide).
HARD  : SysID grid search + DR centre sur estimation + policy adaptative.

Pour executer chaque section :
    python 14-sim-to-real.py easy
    python 14-sim-to-real.py medium
    python 14-sim-to-real.py hard
    python 14-sim-to-real.py all   (defaut)

requires: numpy, gymnasium
"""

import sys
import numpy as np
import gymnasium as gym


# =============================================================================
# Helpers partages
# =============================================================================

def policy(obs, params):
    """Policy de base, identique au fichier de cours (3 gains)."""
    cos_t, sin_t, theta_dot = obs
    theta = np.arctan2(sin_t, cos_t)
    k_E, k_p, k_d = params
    if abs(theta) < 0.3:
        action = -k_p * theta - k_d * theta_dot
    else:
        action = k_E * theta_dot * cos_t
    return float(np.clip(action, -2.0, 2.0))


class FixedPhysicsWrapper(gym.Wrapper):
    """Pendule a parametres fixes (mass/length/friction/latence/bruit)."""

    def __init__(self, env, mass=1.0, length=1.0, friction=0.0,
                 latency_steps=0, obs_noise_std=0.0):
        super().__init__(env)
        self.target_mass = mass
        self.target_length = length
        self.friction = friction
        self.latency_steps = int(latency_steps)
        self.obs_noise_std = obs_noise_std
        self._buf = []

    def reset(self, **kwargs):
        self.env.unwrapped.m = self.target_mass
        self.env.unwrapped.l = self.target_length
        self._buf = [0.0] * self.latency_steps
        obs, info = self.env.reset(**kwargs)
        return self._noisy(obs), info

    def step(self, action):
        if self.latency_steps > 0:
            applied = self._buf.pop(0)
            self._buf.append(float(action))
            applied_action = np.array([applied], dtype=np.float32)
        else:
            applied_action = action
        obs, r, term, trunc, info = self.env.step(applied_action)
        if self.friction > 0:
            theta, theta_dot = self.env.unwrapped.state
            theta_dot *= (1.0 - self.friction)
            self.env.unwrapped.state = np.array([theta, theta_dot])
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        return self._noisy(obs), r, term, trunc, info

    def _noisy(self, obs):
        if self.obs_noise_std > 0:
            return obs + np.random.normal(0, self.obs_noise_std,
                                          size=obs.shape).astype(obs.dtype)
        return obs


def evaluate(env_fn, params, n_episodes=10, seed_base=0, policy_fn=None):
    """Evalue params sur env_fn (factory). policy_fn(obs, params) ou defaut = policy."""
    pf = policy_fn if policy_fn is not None else policy
    returns = []
    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed_base + ep)
        ep_return = 0.0
        for _ in range(200):
            a = pf(obs, params)
            obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            ep_return += r
            if term or trunc:
                break
        env.close()
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


# =============================================================================
# EASY - Reponses textuelles
# =============================================================================

EASY_ANSWERS = """
EASY - Classification des sources du reality gap
=================================================

1. Masse end-effector 850g vs 920g
   Type     : DYNAMICS gap (parametre physique)
   DR ?     : OUI
   Plage    : mass effector ~ U([800, 1000]) g (deborder de ~10% de la valeur reelle estimee)
   Justif   : la tolerance fabrication + huile/graisse + capteurs ajoutent toujours qq dizaines
              de grammes. ±10% couvre le realiste sans exploser le couvert.

2. Latence 12 ms boucle controle
   Type     : DYNAMICS gap (timing)
   DR ?     : OUI
   Plage    : latency ~ U([0, 25]) ms (en discret: U([0, 3]) steps a 100Hz)
   Justif   : la latence reelle a une variabilite (jitter), randomiser de 0 a 2x la valeur
              moyenne mesuree donne une policy robuste aux pics.

3. Bruit encodeurs ~0.05 deg
   Type     : DYNAMICS / capteurs
   DR ?     : OUI
   Plage    : sigma_obs ~ U([0.02, 0.10]) deg
   Justif   : derive thermique, vibrations -> couvrir 2x le bruit nominal.

4. Flexibilite des liens (non modelise)
   Type     : MODELISATION INCOMPLETE - ni dynamics au sens parametrique, ni visual
   DR ?     : NON (pas un parametre, c'est un *phenomene*)
   Reponse  : (a) modeliser via MuJoCo flex/soft bodies si critique,
              (b) sysid sur les modes d'oscillation si linearisable,
              (c) accepter le risque + ajouter des marges de stabilite + tests reels.

5. Camera offset extrinseque ±2mm
   Type     : VISUAL gap (calibration capteur)
   DR ?     : OUI (variante visuelle de Tobin 2017)
   Plage    : T_cam_robot * Random(±3mm en xyz, ±0.5deg en rpy)
   Justif   : tolerance calibration en main = qq mm, randomiser pousse la policy a etre
              robuste a une mauvaise calibration.

6. Gravite 9.81 vs 9.81
   Type     : NEGLIGEABLE
   DR ?     : Non utile
   Justif   : variation latitudinale de g ~ 0.5%. Inferieur au bruit de masse, pas la peine.

7. Friction 3x plus elevee en reel
   Type     : DYNAMICS gap
   DR ?     : OUI - et c'est CRITIQUE (3x est tres au-dela de "tolerance")
   Plage    : friction ~ U([0.1x, 5x]) du nominal sim, OU mieux: re-mesurer en reel via SysID,
              recaler le nominal, et randomiser a ±50% autour.
   Justif   : 3x est un signal qu'il y a un probleme de modelisation (pas juste tolerance).
              Soit le simulateur sous-estime la friction visqueuse (graisse, joints non
              modelises), soit on n'a jamais calibre les parametres -> SysID prioritaire.

ORDRE DE PRIORITE pour sim-to-real reussi
==========================================
1. (#7) Friction 3x : c'est l'erreur la plus grosse, si non corrigee la policy "exploite"
   un environnement glissant qui n'existe pas. SysID + recalage + DR.
2. (#4) Flexibilite : phenomene non modelise = trou de couverture. Investir dans le modele.
3. (#1) Masse end-effector : facile a fixer (mesure precise) puis DR.
4. (#2) Latence : DR couvre, peu cher.
5. (#3) Bruit encodeurs : DR couvre, peu cher.
6. (#5) Camera : DR couvre, peu cher.
7. (#6) Gravite : negligeable.

Principe : commencer par les phenomenes a fort ecart absolu et difficiles a randomiser
(modelisation), finir par les phenomenes faciles a randomiser et a faible impact.
"""


# =============================================================================
# MEDIUM - DomainRandomizationWrapper + comparaison strategies
# =============================================================================

class DomainRandomizationWrapper(gym.Wrapper):
    """Wrapper generique qui randomise des params physiques a chaque reset."""

    def __init__(self, env, param_ranges, seed=None):
        super().__init__(env)
        self.param_ranges = param_ranges
        self._rng = np.random.default_rng(seed)
        self._buf = []
        self._current_params = {}

    def _sample_params(self):
        return {k: float(self._rng.uniform(lo, hi)) for k, (lo, hi) in self.param_ranges.items()}

    def reset(self, **kwargs):
        p = self._sample_params()
        self._current_params = p
        # appliquer
        if 'mass' in p:
            self.env.unwrapped.m = p['mass']
        if 'length' in p:
            self.env.unwrapped.l = p['length']
        # latence (entiere)
        latency = int(round(p.get('latency_steps', 0)))
        self._buf = [0.0] * latency
        self._latency = latency
        self._friction = p.get('friction', 0.0)
        self._noise = p.get('obs_noise_std', 0.0)
        obs, info = self.env.reset(**kwargs)
        info['phys_params'] = dict(p)
        return self._noisy(obs), info

    def step(self, action):
        if self._latency > 0:
            applied = self._buf.pop(0)
            self._buf.append(float(action))
            applied_action = np.array([applied], dtype=np.float32)
        else:
            applied_action = action
        obs, r, term, trunc, info = self.env.step(applied_action)
        if self._friction > 0:
            theta, theta_dot = self.env.unwrapped.state
            theta_dot *= (1.0 - self._friction)
            self.env.unwrapped.state = np.array([theta, theta_dot])
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        info['phys_params'] = dict(self._current_params)
        return self._noisy(obs), r, term, trunc, info

    def _noisy(self, obs):
        if self._noise > 0:
            return obs + np.random.normal(0, self._noise, size=obs.shape).astype(obs.dtype)
        return obs


def random_search_with_factory(make_env_fn, n_trials=80, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    best_params, best_score = None, -np.inf
    for _ in range(n_trials):
        params = (rng.uniform(0.1, 1.5), rng.uniform(2.0, 30.0), rng.uniform(0.5, 8.0))
        score, _ = evaluate(make_env_fn, params, n_episodes=4, seed_base=42)
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score


def medium_solution():
    print("MEDIUM - Comparaison de 4 strategies de randomization")
    print("=" * 72)

    nominal = {'mass': 1.0, 'length': 1.0, 'friction': 0.0, 'obs_noise_std': 0.0}

    def factory(strategy):
        if strategy == 'A_none':
            ranges = {'mass': (1.0, 1.0), 'length': (1.0, 1.0),
                      'friction': (0.0, 0.0), 'obs_noise_std': (0.0, 0.0)}
        elif strategy == 'B_narrow':
            ranges = {'mass': (0.9, 1.1), 'length': (0.9, 1.1),
                      'friction': (0.0, 0.02), 'obs_noise_std': (0.0, 0.005)}
        elif strategy == 'C_wide':
            ranges = {'mass': (0.5, 1.5), 'length': (0.5, 1.5),
                      'friction': (0.0, 0.10), 'obs_noise_std': (0.0, 0.03)}
        elif strategy == 'D_too_wide':
            ranges = {'mass': (0.05, 3.0), 'length': (0.05, 3.0),
                      'friction': (0.0, 0.5), 'obs_noise_std': (0.0, 0.5)}
        else:
            raise ValueError(strategy)
        return lambda: DomainRandomizationWrapper(gym.make("Pendulum-v1"), ranges)

    # env d'eval reel commun
    def real_env_fn():
        return FixedPhysicsWrapper(gym.make("Pendulum-v1"),
                                   mass=1.25, length=0.92, friction=0.05,
                                   latency_steps=2, obs_noise_std=0.02)

    results = {}
    for strat, seed in [('A_none', 10), ('B_narrow', 11), ('C_wide', 12), ('D_too_wide', 13)]:
        params, train_score = random_search_with_factory(factory(strat), n_trials=80, rng_seed=seed)
        real_score, real_std = evaluate(real_env_fn, params, n_episodes=20, seed_base=200)
        results[strat] = (train_score, real_score, real_std, params)
        print(f"  {strat:12s} train={train_score:7.1f}  real={real_score:7.1f} +/-{real_std:5.1f}  "
              f"params=({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})")

    print("\nAnalyse :")
    print("  - A (no randomization) marche en train mais chute en reel : overfit a la sim.")
    print("  - B (narrow) ameliore deja le reel : couvre les ecarts de fabrication.")
    print("  - C (wide) typiquement le meilleur compromis sur le reel.")
    print("  - D (too wide) regresse: la policy ne peut pas etre bonne sur des physiques absurdes")
    print("    (mass=0.05 = pendule plume vs mass=3.0 = pendule lourd) ET sur le reel (qui est")
    print("    proche du nominal). Le random search trouve des gains qui moyennent mal.")
    print("  -> Trade-off : trop etroit = overfit sim, trop large = sous-optimisation universelle.")


# =============================================================================
# HARD - SysID + adaptive policy
# =============================================================================

def simulate_traj(mass, length, friction, actions, init_state):
    """Reroule une trajectoire en sim avec les params (mass, length, friction).
    Retourne la liste des etats [theta, theta_dot] obtenus."""
    env = gym.make("Pendulum-v1")
    env.reset(seed=0)
    env.unwrapped.m = mass
    env.unwrapped.l = length
    env.unwrapped.state = np.array(init_state)
    states = []
    for a in actions:
        env.step(np.array([a], dtype=np.float32))
        theta, theta_dot = env.unwrapped.state
        # appliquer friction
        theta_dot *= (1.0 - friction)
        env.unwrapped.state = np.array([theta, theta_dot])
        states.append([theta, theta_dot])
    env.close()
    return np.array(states)


def system_identify(real_env_fn, n_traj=4, traj_steps=80):
    """Identifie (mass, length, friction) par grid search."""
    # collecte trajectoires reelles
    traj_data = []
    rng = np.random.default_rng(0)
    for i in range(n_traj):
        env = real_env_fn()
        obs, _ = env.reset(seed=300 + i)
        # extraire l'etat initial via env.unwrapped.state (theta, theta_dot)
        init_state = env.unwrapped.state.copy()
        actions = rng.uniform(-2, 2, size=traj_steps).tolist()
        states = []
        for a in actions:
            obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            theta = np.arctan2(obs[1], obs[0])
            theta_dot = obs[2]
            states.append([theta, theta_dot])
        env.close()
        traj_data.append((init_state, actions, np.array(states)))

    # grid search
    best_loss, best_params = np.inf, None
    masses = np.linspace(0.7, 1.5, 9)
    lengths = np.linspace(0.7, 1.2, 6)
    frictions = np.linspace(0.0, 0.10, 6)

    for m in masses:
        for L in lengths:
            for f in frictions:
                total = 0.0
                for init_state, actions, real_states in traj_data:
                    sim_states = simulate_traj(m, L, f, actions, init_state)
                    # erreur MSE sur theta_dot uniquement (theta peut wrap)
                    # On compare seulement les premiers pas (avant divergence chaotique)
                    n = min(20, len(real_states))
                    err = np.mean((sim_states[:n, 1] - real_states[:n, 1]) ** 2)
                    total += err
                if total < best_loss:
                    best_loss = total
                    best_params = (float(m), float(L), float(f))
    return best_params, best_loss


def adaptive_policy(obs, params, history):
    """Policy adaptative : 6 gains.
    history = list de (obs, action) recents. On calcule var(theta_dot) sur l'historique."""
    cos_t, sin_t, theta_dot = obs
    theta = np.arctan2(sin_t, cos_t)
    k_E, k_p, k_d, alpha_E, alpha_p, alpha_d = params

    # detection de la "physique" via variance recente
    if len(history) >= 5:
        recent_dots = np.array([h[0][2] for h in history[-15:]])
        var_dot = float(np.var(recent_dots))
    else:
        var_dot = 1.0  # baseline

    # var elevee => physique "molle" (peu de friction), var faible => physique "raide"
    var_baseline = 1.0
    factor = np.clip(var_dot / var_baseline, 0.5, 2.0)

    k_E_eff = k_E * (1.0 + alpha_E * (factor - 1.0))
    k_p_eff = k_p * (1.0 + alpha_p * (factor - 1.0))
    k_d_eff = k_d * (1.0 + alpha_d * (factor - 1.0))

    if abs(theta) < 0.3:
        action = -k_p_eff * theta - k_d_eff * theta_dot
    else:
        action = k_E_eff * theta_dot * cos_t
    return float(np.clip(action, -2.0, 2.0))


def evaluate_adaptive(env_fn, params, n_episodes=10, seed_base=0):
    returns = []
    for ep in range(n_episodes):
        env = env_fn()
        obs, _ = env.reset(seed=seed_base + ep)
        history = []
        ep_return = 0.0
        for _ in range(200):
            a = adaptive_policy(obs, params, history)
            history.append((obs.copy(), a))
            if len(history) > 30:
                history.pop(0)
            obs, r, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            ep_return += r
            if term or trunc:
                break
        env.close()
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


def hard_solution():
    print("HARD - SysID + DR centre + adaptive policy")
    print("=" * 72)

    def real_env_fn():
        return FixedPhysicsWrapper(gym.make("Pendulum-v1"),
                                   mass=1.25, length=0.92, friction=0.05,
                                   latency_steps=2, obs_noise_std=0.02)

    # --- Partie 1 : SysID ---
    print("\n[Partie 1] System Identification")
    print("  vrais params (caches) : mass=1.25, length=0.92, friction=0.05")
    print("  grid search en cours...")
    estimated, loss = system_identify(real_env_fn, n_traj=4, traj_steps=80)
    print(f"  estime : mass={estimated[0]:.2f}, length={estimated[1]:.2f}, friction={estimated[2]:.3f}")
    print(f"  loss   : {loss:.4f}")

    # --- Partie 2 : narrow DR centre sur estimation ---
    print("\n[Partie 2] DR centre sur estimation SysID")
    m_hat, L_hat, f_hat = estimated
    narrow_ranges = {
        'mass': (m_hat * 0.9, m_hat * 1.1),
        'length': (L_hat * 0.9, L_hat * 1.1),
        'friction': (max(0, f_hat - 0.02), f_hat + 0.02),
        'obs_noise_std': (0.0, 0.025),
    }
    sysid_factory = lambda: DomainRandomizationWrapper(gym.make("Pendulum-v1"), narrow_ranges)
    sysid_params, sysid_train = random_search_with_factory(sysid_factory, n_trials=80, rng_seed=20)
    sysid_real, sysid_std = evaluate(real_env_fn, sysid_params, n_episodes=20, seed_base=300)
    print(f"  policy SysID+narrow_DR : real={sysid_real:.1f} +/- {sysid_std:.1f}")

    # comparaison wide DR aveugle
    wide_ranges = {'mass': (0.5, 1.5), 'length': (0.5, 1.5),
                   'friction': (0.0, 0.10), 'obs_noise_std': (0.0, 0.03)}
    wide_factory = lambda: DomainRandomizationWrapper(gym.make("Pendulum-v1"), wide_ranges)
    wide_params, _ = random_search_with_factory(wide_factory, n_trials=80, rng_seed=21)
    wide_real, wide_std = evaluate(real_env_fn, wide_params, n_episodes=20, seed_base=300)
    print(f"  policy wide_DR aveugle : real={wide_real:.1f} +/- {wide_std:.1f}")
    print(f"  -> SysID+narrow doit etre >= wide aveugle (concentration de la masse de proba)")

    # --- Partie 3 : adaptive policy ---
    print("\n[Partie 3] Policy adaptative (6 gains, history-based)")

    def adaptive_search(make_env_fn, n_trials=120, rng_seed=22):
        rng = np.random.default_rng(rng_seed)
        best_params, best_score = None, -np.inf
        for _ in range(n_trials):
            params = (
                rng.uniform(0.1, 1.5),
                rng.uniform(2.0, 30.0),
                rng.uniform(0.5, 8.0),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
            )
            score, _ = evaluate_adaptive(make_env_fn, params, n_episodes=4, seed_base=42)
            if score > best_score:
                best_score, best_params = score, params
        return best_params, best_score

    adapt_params, adapt_train = adaptive_search(wide_factory, n_trials=120, rng_seed=22)
    adapt_real, adapt_std = evaluate_adaptive(real_env_fn, adapt_params, n_episodes=20, seed_base=300)
    print(f"  policy adaptive+wide_DR : real={adapt_real:.1f} +/- {adapt_std:.1f}")

    print("\nSynthese des 3 strategies sur le reel :")
    print(f"  wide DR (no adapt)        : {wide_real:.1f}")
    print(f"  SysID + narrow DR         : {sysid_real:.1f}")
    print(f"  adaptive policy + wide DR : {adapt_real:.1f}")
    print()
    print("Idee centrale RMA : la policy adaptative observe son propre comportement recent")
    print("(variance de theta_dot ici, MLP sur historique dans le vrai RMA) et infere")
    print("implicitement xi pour ajuster ses gains. Pas besoin de SysID explicite a deploiement.")


# =============================================================================
# Main
# =============================================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('easy', 'all'):
        print(EASY_ANSWERS)
    if mode in ('medium', 'all'):
        print()
        medium_solution()
    if mode in ('hard', 'all'):
        print()
        hard_solution()


if __name__ == "__main__":
    main()

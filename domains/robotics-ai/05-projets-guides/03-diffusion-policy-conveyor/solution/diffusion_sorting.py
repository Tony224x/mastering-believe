"""
Correction commentee — Mini Diffusion Policy sur le poste de tri FleetSim.

Probleme : les demonstrations expertes contournent le pilier central par la
gauche OU la droite (50/50). Un BC entraine en MSE moyenne les deux modes et
fonce droit dans le pilier. Une diffusion policy echantillonne un mode entier
coherent — c'est exactement l'argument de Chi 2023 (theorie J16), demontre ici
en numpy pur avec le debruiteur optimal en forme fermee (pas de reseau : la
version apprise UNet 1D, c'est le capstone J24-J28).

Cle de lecture : chaque commentaire explique le POURQUOI. Numpy seul,
deterministe, < 60 s CPU.

Run: python diffusion_sorting.py
"""
from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------------------
# Geometrie du poste de tri et hyperparametres.
# ---------------------------------------------------------------------------
PILLAR_C = np.array([0.0, 0.5])   # pilier central du poste
PILLAR_R = 0.15
GOAL_Y = 0.95                     # zone d'expedition : y >= GOAL_Y et |x| <= 0.25
N_DEMOS = 300
DEMO_STEPS = 30                   # pas par demonstration experte
H = 8                             # horizon du chunk d'actions (Diffusion Policy)
T_A = 4                           # actions executees avant replanification
K_DIFF = 50                       # pas de diffusion DDPM
H_OBS = 0.08                      # bande passante du noyau d'observation
MAX_STEPS = 60                    # budget de pas par rollout


# ---------------------------------------------------------------------------
# 1. Demonstrations expertes multimodales
# ---------------------------------------------------------------------------
def expert_trajectory(side: float, rng: np.random.Generator) -> np.ndarray:
    """Une demo experte (DEMO_STEPS+1, 2) contournant le pilier du cote `side`.

    Bezier quadratique : le point de controle lateral ecarte la courbe du
    pilier. Les VRAIS experts ne repassent jamais par le milieu — c'est ce
    qui rend la moyenne des demos physiquement invalide.
    """
    p0 = np.array([rng.uniform(-0.05, 0.05), 0.0])
    ctrl = np.array([side * rng.uniform(0.50, 0.60), 0.5])
    p2 = np.array([rng.uniform(-0.03, 0.03), 1.0])
    t = np.linspace(0.0, 1.0, DEMO_STEPS + 1)[:, None]
    traj = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p2
    # Bruit leger : un expert humain ne rejoue jamais deux fois le meme geste.
    traj[1:-1] += rng.normal(0.0, 0.004, traj[1:-1].shape)
    return traj


def hits_pillar(p: np.ndarray) -> bool:
    return float(np.linalg.norm(p - PILLAR_C)) < PILLAR_R


def make_demos(n_demos: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (obs (N, 2), chunks (N, H*2), ratio_gauche).

    Fenetre glissante : a chaque pas t d'une demo, obs = position courante et
    chunk = les H deltas suivants aplatis. Le chunking (predire une SEQUENCE,
    pas une action) vient de Diffusion Policy : il rend chaque echantillon
    temporellement coherent — un chunk entier va a gauche OU a droite.
    """
    all_obs, all_chunks = [], []
    n_left = 0
    for i in range(n_demos):
        side = -1.0 if rng.random() < 0.5 else 1.0   # 50/50 : multimodalite maximale
        n_left += side < 0
        traj = expert_trajectory(side, rng)
        assert not any(hits_pillar(p) for p in traj), "demo experte en collision"
        deltas = np.diff(traj, axis=0)               # (DEMO_STEPS, 2)
        # Padding par repetition du dernier delta : les chunks de fin de demo
        # restent de taille H sans inventer de nouveau mouvement.
        padded = np.vstack([deltas, np.repeat(deltas[-1:], H - 1, axis=0)])
        for t in range(DEMO_STEPS):
            all_obs.append(traj[t])
            all_chunks.append(padded[t:t + H].ravel())
    return np.array(all_obs), np.array(all_chunks), n_left / n_demos


# ---------------------------------------------------------------------------
# 2. Baseline BC = moyenne conditionnelle (regression a noyau)
# ---------------------------------------------------------------------------
def obs_log_kernel(obs: np.ndarray, dataset_obs: np.ndarray) -> np.ndarray:
    """log K(obs, obs_i) gaussien. En log-espace des le depart : ces poids se
    COMBINENT avec les vraisemblances de diffusion dans le debruiteur."""
    d2 = np.sum((dataset_obs - obs) ** 2, axis=1)
    return -d2 / (2 * H_OBS ** 2)


def bc_policy(obs: np.ndarray, dataset_obs: np.ndarray, dataset_chunks: np.ndarray) -> np.ndarray:
    """E[chunk | obs] — exactement ce qu'apprend un MLP entraine en MSE.

    Pourquoi un estimateur a noyau plutot qu'un MLP : meme limite theorique
    (la MSE converge vers la moyenne conditionnelle), zero entrainement, et la
    comparaison avec la diffusion utilise le MEME conditionnement — la seule
    difference restante est moyenne vs echantillonnage. C'est l'ablation propre.
    """
    log_w = obs_log_kernel(obs, dataset_obs)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return w @ dataset_chunks


# ---------------------------------------------------------------------------
# 3. DDPM : schedule cosine + debruiteur optimal en forme fermee
# ---------------------------------------------------------------------------
def cosine_alpha_bars(k_steps: int) -> np.ndarray:
    """Schedule cosine (Nichol & Dhariwal 2021) — detruit la structure plus
    progressivement que la schedule lineaire (cf. exercice easy J16)."""
    s = 0.008
    steps = np.arange(k_steps + 1)
    f = np.cos((steps / k_steps + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return np.clip(ab[1:], 1e-5, 0.9999)            # ab[k] pour k = 0..K-1


ALPHA_BARS = cosine_alpha_bars(K_DIFF)
# alphas_k = ab_k / ab_{k-1} ; betas en decoulent — relations DDPM standard.
ALPHAS = ALPHA_BARS / np.concatenate([[1.0], ALPHA_BARS[:-1]])
BETAS = 1.0 - ALPHAS


def denoise_batch(
    a_k: np.ndarray, k: int, log_w_obs: np.ndarray, chunks_norm: np.ndarray
) -> np.ndarray:
    """E[a0 | ak, obs] EXACT sous la distribution empirique. a_k : (B, D).

    Sur un dataset fini, la posterior du debruiteur est un softmax sur les
    chunks experts : log w_i = log N(ak; sqrt(ab_k) a0_i, (1-ab_k) I)
                             + log K(obs, obs_i).
    C'est la cible vers laquelle un UNet entraine en epsilon-prediction
    converge (theorie J15 : le score optimal est celui du mixture-of-Gaussians
    empirique). Logsumexp obligatoire : a k petit, (1-ab_k) ~ 1e-4 et les
    vraisemblances font underflow en espace direct.
    """
    ab = ALPHA_BARS[k]
    # (B, N) : distances entre chaque sample bruite et chaque chunk expert bruite.
    d2 = ((a_k[:, None, :] - np.sqrt(ab) * chunks_norm[None, :, :]) ** 2).sum(axis=-1)
    log_w = -d2 / (2 * (1 - ab)) + log_w_obs[None, :]
    log_w -= log_w.max(axis=1, keepdims=True)       # stabilite numerique (logsumexp)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    return w @ chunks_norm                           # (B, D) : moyenne posterior


def ddpm_sample_batch(
    n_samples: int, obs: np.ndarray, dataset_obs: np.ndarray, chunks_norm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sampling ancestral DDPM, B chunks en parallele. -> (B, D) normalises.

    Le noyau d'observation est calcule UNE fois : l'obs ne change pas pendant
    les K pas de debruitage (c'est la conditionnement FiLM du vrai modele,
    version pauvre). Le batch vectorise les 100 samples de l'analyse de
    multimodalite — boucler ferait x100 sur le runtime.
    """
    log_w_obs = obs_log_kernel(obs, dataset_obs)
    a_k = rng.standard_normal((n_samples, chunks_norm.shape[1]))
    for k in range(K_DIFF - 1, -1, -1):
        a0_hat = denoise_batch(a_k, k, log_w_obs, chunks_norm)
        ab, alpha, beta = ALPHA_BARS[k], ALPHAS[k], BETAS[k]
        # eps_hat se deduit de a0_hat : les deux parametrisations sont equivalentes.
        eps_hat = (a_k - np.sqrt(ab) * a0_hat) / np.sqrt(1 - ab)
        mean = (a_k - beta / np.sqrt(1 - ab) * eps_hat) / np.sqrt(alpha)
        if k > 0:
            ab_prev = ALPHA_BARS[k - 1]
            sigma = np.sqrt(beta * (1 - ab_prev) / (1 - ab))
            a_k = mean + sigma * rng.standard_normal(a_k.shape)
        else:
            a_k = mean                               # dernier pas : pas de bruit injecte
    return a_k


# ---------------------------------------------------------------------------
# 4. Receding horizon + evaluation
# ---------------------------------------------------------------------------
def rollout(policy: str, dataset, norm, rng: np.random.Generator) -> dict:
    """Un episode complet. policy : 'bc' ou 'diffusion'.

    Receding horizon (Chi 2023) : on echantillonne un chunk de H actions, on
    n'en execute que T_A, puis on replanifie depuis la nouvelle obs — reactif
    aux derives sans perdre la coherence intra-chunk.
    """
    dataset_obs, dataset_chunks = dataset
    mu, sigma = norm
    chunks_norm = (dataset_chunks - mu) / sigma
    pos = np.array([0.0, 0.0])
    trace = [pos.copy()]
    for _ in range(MAX_STEPS // T_A):
        if policy == "bc":
            chunk = bc_policy(pos, dataset_obs, dataset_chunks)
        else:
            z = ddpm_sample_batch(1, pos, dataset_obs, chunks_norm, rng)[0]
            chunk = z * sigma + mu                  # denormalisation (cf. J24)
        actions = chunk.reshape(H, 2)[:T_A]
        for a in actions:
            pos = pos + a
            trace.append(pos.copy())
            if hits_pillar(pos):
                return {"success": False, "reason": "collision", "trace": np.array(trace)}
            if pos[1] >= GOAL_Y and abs(pos[0]) <= 0.25:
                side = "left" if min(p[0] for p in trace) < -0.1 else "right"
                return {"success": True, "reason": "goal", "side": side, "trace": np.array(trace)}
    return {"success": False, "reason": "timeout", "trace": np.array(trace)}


def evaluate(policy: str, dataset, norm, seed: int, n_episodes: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    results = [rollout(policy, dataset, norm, rng) for _ in range(n_episodes)]
    sides = [r["side"] for r in results if r["success"]]
    return {
        "success_rate": sum(r["success"] for r in results) / n_episodes,
        "n_left": sides.count("left"),
        "n_right": sides.count("right"),
        "reasons": [r["reason"] for r in results],
    }


# ---------------------------------------------------------------------------
# Main : verifie chaque critere de reussite par une assertion.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)                  # seed fixe : determinisme LogiSim

    # --- Critere 1 : dataset multimodal, demos sans collision ---
    dataset_obs, dataset_chunks, left_ratio = make_demos(N_DEMOS, rng)
    assert 0.40 <= left_ratio <= 0.60, left_ratio
    print(f"[1] {N_DEMOS} demos, {len(dataset_obs)} paires (obs, chunk), "
          f"ratio gauche {left_ratio:.2f} (dans [0.40, 0.60]), zero collision experte  OK")

    # Normalisation des chunks : la diffusion suppose des donnees ~ N(0, I) a k=K.
    mu, sigma = dataset_chunks.mean(axis=0), dataset_chunks.std(axis=0) + 1e-8
    chunks_norm = (dataset_chunks - mu) / sigma
    dataset = (dataset_obs, dataset_chunks)
    norm = (mu, sigma)
    start_obs = np.array([0.0, 0.0])

    # --- Critere 2 : le mode averaging de BC, mesure ---
    # dx total du chunk = somme des deltas lateraux. Les experts partent
    # franchement d'un cote ; BC moyenne et part tout droit.
    near_start = np.linalg.norm(dataset_obs - start_obs, axis=1) < 0.05
    expert_dx = np.abs(dataset_chunks[near_start].reshape(-1, H, 2)[:, :, 0].sum(axis=1))
    bc_chunk = bc_policy(start_obs, dataset_obs, dataset_chunks)
    bc_dx = abs(float(bc_chunk.reshape(H, 2)[:, 0].sum()))
    assert bc_dx < 0.05, bc_dx
    assert expert_dx.mean() > 0.15, expert_dx.mean()
    print(f"[2] Mode averaging : |dx| BC = {bc_dx:.3f} (< 0.05) vs |dx| expert moyen = "
          f"{expert_dx.mean():.3f} (> 0.15) — BC produit un geste inexistant  OK")

    # --- Critere 3 : multimodalite des chunks diffuses au depart ---
    samples = ddpm_sample_batch(100, start_obs, dataset_obs, chunks_norm,
                                np.random.default_rng(1))
    samples = samples * sigma + mu
    sample_dx = samples.reshape(-1, H, 2)[:, :, 0].sum(axis=1)
    frac_left = float(np.mean(sample_dx < 0))
    assert 0.25 <= frac_left <= 0.75, frac_left
    assert np.all(np.abs(sample_dx) > 0.05), np.abs(sample_dx).min()
    print(f"[3] Diffusion au depart : 100 chunks, {frac_left:.0%} a gauche "
          f"(dans [25%, 75%]), aucun chunk 'moyen' (|dx| min = "
          f"{np.abs(sample_dx).min():.3f} > 0.05)  OK")

    # --- Critere 4 : success rates BC vs diffusion ---
    bc_eval = evaluate("bc", dataset, norm, seed=2)
    diff_eval = evaluate("diffusion", dataset, norm, seed=3)
    assert bc_eval["success_rate"] <= 0.20, bc_eval
    assert diff_eval["success_rate"] >= 0.90, diff_eval
    print(f"[4] Success rate sur 20 rollouts : diffusion {diff_eval['success_rate']:.2f} "
          f"(>= 0.90), BC {bc_eval['success_rate']:.2f} (<= 0.20, "
          f"{bc_eval['reasons'].count('collision')} collisions pilier)  OK")

    # --- Critere 5 : les deux modes sont utilises en closed-loop ---
    assert diff_eval["n_left"] >= 3 and diff_eval["n_right"] >= 3, diff_eval
    print(f"[5] Rollouts diffusion reussis : {diff_eval['n_left']} a gauche, "
          f"{diff_eval['n_right']} a droite (>= 3 chacun)  OK")

    # --- Critere 6 : determinisme + budget temps ---
    diff_eval2 = evaluate("diffusion", dataset, norm, seed=3)
    assert diff_eval2 == diff_eval, "non deterministe a seed fixe"
    elapsed = time.time() - t0
    assert elapsed < 60.0, f"{elapsed:.1f}s"
    print(f"[6] Determinisme OK, runtime total {elapsed:.1f}s (< 60 s)  OK")

    print("\nTous les criteres de reussite passent.")

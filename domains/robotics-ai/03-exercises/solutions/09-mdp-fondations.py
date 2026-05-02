"""
Solutions des exercices J9 (easy / medium / hard).

# requires: numpy

Source : Sutton & Barto, RL: An Introduction, 2018, ch. 3-4.
"""

import time

import numpy as np


# ===========================================================================
# EASY — Mini-MDP 3 etats, deterministe, calcul a la main puis verification
# ===========================================================================

def solution_easy() -> None:
    """MDP 3 etats deterministe ; on resout V_pi a la main et on verifie."""
    print("=" * 60)
    print("EASY — Mini-MDP 3 etats, Bellman a la main")
    print("=" * 60)

    # --- Partie A : definition formelle ---
    # S = {A=0, B=1, C=2} ; A = {a1=0, a2=1}
    # P[s, a, s'] = 1 si transition deterministe (s, a) -> s'
    # R[s, a, s'] = recompense
    # gamma = 0.5 ; C terminal (self-loop, R=0)

    n_states = 3
    n_actions = 2
    gamma = 0.5
    A_, B_, C_ = 0, 1, 2

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    # Depuis A : a1 -> A, a2 -> B
    P[A_, 0, A_] = 1.0
    P[A_, 1, B_] = 1.0
    # Depuis B : a1 -> A, a2 -> C avec R=+10
    P[B_, 0, A_] = 1.0
    P[B_, 1, C_] = 1.0
    R[B_, 1, C_] = 10.0
    # C terminal : self-loop pour les deux actions, R=0
    P[C_, 0, C_] = 1.0
    P[C_, 1, C_] = 1.0

    # --- Partie B : evaluation de pi = (a2, a2, a1) ---
    # Bellman expected (deterministe ici) :
    #   V(A) = R(A, a2, B) + gamma * V(B) = 0 + 0.5 * V(B)
    #   V(B) = R(B, a2, C) + gamma * V(C) = 10 + 0.5 * V(C)
    #   V(C) = 0
    # => V(B) = 10, V(A) = 5
    # Note : intuition initiale "V(A) = 2.5" (cas si V(B)=5) -- verifions par calcul.

    pi = np.array([1, 1, 0])  # a2 depuis A et B, a1 depuis C

    def policy_evaluation_exact(pi, P, R, gamma):
        """Resout (I - gamma P_pi) V = R_pi avec numpy.linalg.solve (formule fermee)."""
        n = P.shape[0]
        P_pi = P[np.arange(n), pi, :]
        R_pi = R[np.arange(n), pi, :]
        expected_r = np.sum(P_pi * R_pi, axis=1)
        return np.linalg.solve(np.eye(n) - gamma * P_pi, expected_r)

    V_pi = policy_evaluation_exact(pi, P, R, gamma)
    print(f"\nV_pi(A) = {V_pi[A_]:.4f}  (attendu : 5.0 -- 0 + 0.5 * 10)")
    print(f"V_pi(B) = {V_pi[B_]:.4f}  (attendu : 10.0)")
    print(f"V_pi(C) = {V_pi[C_]:.4f}  (attendu : 0.0)")

    # --- Partie C : politique alternative pi' = (a1, a2, a1) ---
    # pi'(A) = a1 -> boucle infinie sur A sans recompense.
    # V_{pi'}(A) = 0 + gamma * V_{pi'}(A) => V_{pi'}(A) (1 - gamma) = 0 => V = 0.
    # Sans gamma < 1, l'equation V = V est indeterminee : la somme infinie de
    # zeros est nulle mais le point fixe n'est plus unique. C'est l'argument
    # mathematique justifiant gamma < 1.
    pi_prime = np.array([0, 1, 0])
    V_pi_prime = policy_evaluation_exact(pi_prime, P, R, gamma)
    print(f"\nV_pi'(A) = {V_pi_prime[A_]:.4f}  (attendu : 0.0 -- boucle sans recompense)")
    print(f"V_pi'(B) = {V_pi_prime[B_]:.4f}  (attendu : 10.0)")

    print("\nConclusion easy : pi >> pi' a l'etat A. Sans gamma < 1, le point fixe")
    print("de Bellman n'est plus unique pour les politiques avec boucles -- d'ou")
    print("la necessite mathematique du discount.")


# ===========================================================================
# MEDIUM — Value Iteration sur GridWorld 5x5 avec 3 lavas
# ===========================================================================

GRID_SIZE_5 = 5
N_STATES_5 = GRID_SIZE_5 * GRID_SIZE_5
GOAL_5 = (4, 4)
LAVAS_5 = {(1, 1), (1, 3), (3, 1)}
TERMINALS_5 = LAVAS_5 | {GOAL_5}

ACTION_DELTAS_5 = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
LEFT_OF_5 = {0: 2, 1: 3, 2: 1, 3: 0}
RIGHT_OF_5 = {0: 3, 1: 2, 2: 0, 3: 1}


def _step_5(r, c, action):
    dr, dc = ACTION_DELTAS_5[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE_5 and 0 <= nc < GRID_SIZE_5:
        return nr, nc
    return r, c


def build_mdp_5x5(living_penalty: float):
    """Construit P et R pour le GridWorld 5x5 (3 lavas)."""
    P = np.zeros((N_STATES_5, 4, N_STATES_5))
    R = np.zeros((N_STATES_5, 4, N_STATES_5))

    def s_of(r, c):
        return r * GRID_SIZE_5 + c

    for s in range(N_STATES_5):
        r, c = s // GRID_SIZE_5, s % GRID_SIZE_5
        if (r, c) in TERMINALS_5:
            for a in range(4):
                P[s, a, s] = 1.0
            continue
        for a in range(4):
            outcomes = [(a, 0.8), (LEFT_OF_5[a], 0.1), (RIGHT_OF_5[a], 0.1)]
            for actual_a, prob in outcomes:
                nr, nc = _step_5(r, c, actual_a)
                ns = s_of(nr, nc)
                P[s, a, ns] += prob
                if (nr, nc) == GOAL_5:
                    rew = 1.0
                elif (nr, nc) in LAVAS_5:
                    rew = -1.0
                else:
                    rew = living_penalty
                R[s, a, ns] += prob * rew
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(P > 0, R / np.maximum(P, 1e-12), 0.0)
    return P, R


def value_iteration(P, R, gamma, eps=1e-8, max_iter=10000):
    n_states = P.shape[0]
    V = np.zeros(n_states)
    for k in range(max_iter):
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            V = V_new
            return V, np.argmax(Q, axis=1), k + 1
        V = V_new
    Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
    return V, np.argmax(Q, axis=1), max_iter


def render_policy_5(pi):
    syms = {0: "^", 1: "v", 2: "<", 3: ">"}
    lines = []
    for r in range(GRID_SIZE_5):
        row = []
        for c in range(GRID_SIZE_5):
            if (r, c) == GOAL_5:
                row.append("G")
            elif (r, c) in LAVAS_5:
                row.append("X")
            else:
                row.append(syms[int(pi[r * GRID_SIZE_5 + c])])
        lines.append(" ".join(row))
    return "\n".join(lines)


def solution_medium() -> None:
    print("\n" + "=" * 60)
    print("MEDIUM — VI sur GridWorld 5x5 avec 3 lavas")
    print("=" * 60)

    configs = [
        ("(a) gamma=0.9, living=0.0", 0.9, 0.0),
        ("(b) gamma=0.9, living=-0.04", 0.9, -0.04),
        ("(c) gamma=0.5, living=-0.04", 0.5, -0.04),
    ]
    results = []
    for name, gamma, lp in configs:
        P, R = build_mdp_5x5(living_penalty=lp)
        V, pi, n_iter = value_iteration(P, R, gamma=gamma, eps=1e-6)
        results.append((name, gamma, V, pi, n_iter))
        print(f"\n--- {name} | converge en {n_iter} iterations ---")
        print(render_policy_5(pi))

    print("\n--- Reponses aux questions ---")
    print("Q1 : avec living_penalty=0, le robot n'a aucun incitatif a finir vite ;")
    print("     plusieurs politiques sont co-optimales (ties), donc argmax peut")
    print("     pointer vers une direction qui ne va pas directement au goal.")
    print("Q2 : avec gamma=0.5, les recompenses lointaines sont fortement actualisees")
    print("     (0.5^10 ~ 0.001). Le robot devient myope : pres des lavas il prefere")
    print("     parfois s'eloigner que tenter de contourner pour atteindre le goal.")
    print("Q3 : le nombre d'iterations grandit en O(1 / (1 - gamma)).")
    print("     gamma=0.9 -> taux de contraction 0.9 ; gamma=0.5 -> 0.5 (converge")
    print("     beaucoup plus vite). On observe en pratique :")
    for name, gamma, _, _, n_iter in results:
        print(f"     {name}: {n_iter} iterations")


# ===========================================================================
# HARD — Modified Policy Iteration + benchmark VI/PI/MPI
# ===========================================================================

def policy_evaluation_iter(pi, P, R, gamma, m=None, eps=1e-10, max_iter=100000):
    """Iter Bellman expected pour pi.
    Si m est fourni : exactement m iterations.
    Sinon : iterer jusqu'a eps."""
    n = P.shape[0]
    V = np.zeros(n)
    P_pi = P[np.arange(n), pi, :]
    R_pi = R[np.arange(n), pi, :]
    expected_r = np.sum(P_pi * R_pi, axis=1)
    if m is not None:
        for _ in range(m):
            V = expected_r + gamma * (P_pi @ V)
        return V, m
    for k in range(max_iter):
        V_new = expected_r + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < eps:
            return V_new, k + 1
        V = V_new
    return V, max_iter


def policy_iteration(P, R, gamma, eps_inner=1e-10, max_iter=200):
    n = P.shape[0]
    pi = np.zeros(n, dtype=np.int64)
    total_updates = 0
    for k in range(max_iter):
        V, n_inner = policy_evaluation_iter(pi, P, R, gamma, eps=eps_inner)
        total_updates += n_inner
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
        pi_new = np.argmax(Q, axis=1)
        total_updates += 1  # une amelioration = une update
        if np.array_equal(pi_new, pi):
            return V, pi, k + 1, total_updates
        pi = pi_new
    V, _ = policy_evaluation_iter(pi, P, R, gamma, eps=eps_inner)
    return V, pi, max_iter, total_updates


def modified_policy_iteration(P, R, gamma, m, eps=1e-8, max_iter=10000):
    """MPI (Sutton & Barto, ch. 4.6) : m etapes d'evaluation partielle de pi
    a partir du V courant (warm-started), puis amelioration de pi.

    Critere d'arret : la politique est stable ET la valeur du Bellman optimal
    appliquee a V est proche de V (residu de Bellman optimality < eps).
    """
    n = P.shape[0]
    pi = np.zeros(n, dtype=np.int64)
    V = np.zeros(n)
    total_updates = 0
    prev_pi = None

    for k in range(max_iter):
        # m etapes de Bellman expected pour pi, en partant du V courant (warm-start)
        P_pi = P[np.arange(n), pi, :]
        R_pi = R[np.arange(n), pi, :]
        expected_r = np.sum(P_pi * R_pi, axis=1)
        for _ in range(m):
            V = expected_r + gamma * (P_pi @ V)
        total_updates += m

        # Amelioration : politique gloutonne par rapport a V
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
        TV = np.max(Q, axis=1)  # T* V (operateur de Bellman optimal applique a V)
        pi_new = np.argmax(Q, axis=1)
        total_updates += 1

        # Convergence : pi stable + residu de Bellman optimal sur V < eps
        bellman_residual = float(np.max(np.abs(TV - V)))
        if prev_pi is not None and np.array_equal(pi_new, prev_pi) and bellman_residual < eps:
            return V, pi_new, k + 1, total_updates
        prev_pi = pi
        pi = pi_new

    return V, pi, max_iter, total_updates


def value_iteration_full(P, R, gamma, eps=1e-8, max_iter=10000):
    """Variante de VI qui retourne aussi total_updates pour le benchmark."""
    n = P.shape[0]
    V = np.zeros(n)
    for k in range(max_iter):
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            V = V_new
            pi = np.argmax(Q, axis=1)
            return V, pi, k + 1, k + 1
        V = V_new
    Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
    return V, np.argmax(Q, axis=1), max_iter, max_iter


def random_mdp(n_states, n_actions, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.standard_normal((n_states, n_actions, n_states))
    assert np.allclose(P.sum(axis=2), 1.0)
    return P, R


def solution_hard() -> None:
    print("\n" + "=" * 60)
    print("HARD — MPI + benchmark VI / PI / MPI sur MDPs aleatoires")
    print("=" * 60)

    gamma = 0.95
    eps = 1e-8
    sizes = [10, 50, 200]

    print(f"\nConfig : gamma={gamma}, eps={eps}, n_actions=4, seed=0")
    print(f"\n{'n_states':>8} | {'algo':<12} | {'iter_ext':>8} | {'updates':>8} | {'time_ms':>8} | V*_diff")
    print("-" * 75)

    for n in sizes:
        P, R = random_mdp(n, 4, seed=0)

        # VI
        t0 = time.perf_counter()
        V_vi, _, n_iter_vi, upd_vi = value_iteration_full(P, R, gamma, eps=eps)
        t_vi = (time.perf_counter() - t0) * 1000

        # PI
        t0 = time.perf_counter()
        V_pi, _, n_iter_pi, upd_pi = policy_iteration(P, R, gamma)
        t_pi = (time.perf_counter() - t0) * 1000
        diff_pi = float(np.max(np.abs(V_pi - V_vi)))

        print(f"{n:>8} | {'VI':<12} | {n_iter_vi:>8} | {upd_vi:>8} | {t_vi:>8.2f} | --      ")
        print(f"{n:>8} | {'PI':<12} | {n_iter_pi:>8} | {upd_pi:>8} | {t_pi:>8.2f} | {diff_pi:.2e}")

        for m in [1, 5, 10, 50]:
            t0 = time.perf_counter()
            V_mpi, _, n_iter_mpi, upd_mpi = modified_policy_iteration(P, R, gamma, m=m, eps=eps)
            t_mpi = (time.perf_counter() - t0) * 1000
            diff_mpi = float(np.max(np.abs(V_mpi - V_vi)))
            print(f"{n:>8} | {'MPI(m=' + str(m) + ')':<12} | {n_iter_mpi:>8} | {upd_mpi:>8} | {t_mpi:>8.2f} | {diff_mpi:.2e}")
        print()

    print("--- Reponses aux questions ---")
    print("Q1 : oui, PI converge en O(10) iterations externes vs O(100-500) pour VI a gamma=0.95.")
    print("Q2 : pour n=200, MPI avec m=5..10 est typiquement le meilleur compromis :")
    print("     suffisamment d'evaluation partielle pour stabiliser V, mais pas trop")
    print("     pour ne pas gaspiller des updates sur une politique sous-optimale.")
    print("Q3 : pour |S|=10^6 (bras 6-DOF discretise), aucun de ces algos tabulaires")
    print("     ne tient en memoire (matrice P de 10^12 * |A| entrees). On passe au")
    print("     RL approxime (DQN, J10) ou aux methodes de policy gradient (J11).")


# ===========================================================================
# Main
# ===========================================================================

def main():
    solution_easy()
    solution_medium()
    solution_hard()


if __name__ == "__main__":
    main()

"""
J9 — MDPs, Bellman, value iteration, policy iteration.

GridWorld 4x4 from scratch :
- 16 etats (case (i, j))
- 4 actions (haut, bas, gauche, droite)
- transitions stochastiques (80% action voulue, 10% derape gauche, 10% derape droite)
- recompenses : +1 a (3, 3) (goal), -1 a (1, 1) (lava), -0.04 living penalty
- gamma = 0.9

On code value iteration et policy iteration a la main, en numpy pur.
On compare la convergence des deux algorithmes.

Source : Sutton & Barto, RL: An Introduction, 2018, ch. 3-4.

# requires: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Definition du MDP : GridWorld 4x4
# ---------------------------------------------------------------------------

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE  # 16 etats indexes 0..15
N_ACTIONS = 4  # 0=haut, 1=bas, 2=gauche, 3=droite

GOAL = (3, 3)  # case but, recompense +1
LAVA = (1, 1)  # case lava, recompense -1
TERMINAL_STATES = {GOAL, LAVA}

LIVING_PENALTY = -0.04  # cout de chaque pas (pousse a finir vite)
GOAL_REWARD = 1.0
LAVA_REWARD = -1.0

GAMMA = 0.9  # facteur d'actualisation

# Probabilite : 80% action voulue, 10% derape a 90deg gauche, 10% a droite.
# Si on tape un mur ou qu'on essaie de sortir de la grille, on reste sur place.
P_INTENDED = 0.8
P_LEFT_SLIP = 0.1
P_RIGHT_SLIP = 0.1

# Vecteurs deplacement par action : (delta_row, delta_col)
ACTION_DELTAS = {
    0: (-1, 0),  # haut : ligne diminue
    1: (1, 0),   # bas
    2: (0, -1),  # gauche
    3: (0, 1),   # droite
}

# Action perpendiculaire gauche / droite (utile pour le derapage stochastique).
# Si on veut aller "haut" et qu'on derape "a gauche", on va effectivement vers la gauche.
LEFT_OF = {0: 2, 1: 3, 2: 1, 3: 0}   # haut->gauche, bas->droite, gauche->bas, droite->haut
RIGHT_OF = {0: 3, 1: 2, 2: 0, 3: 1}  # haut->droite, etc.


def state_to_rc(s: int) -> tuple[int, int]:
    """Convertit un index d'etat 0..15 en (row, col)."""
    return s // GRID_SIZE, s % GRID_SIZE


def rc_to_state(r: int, c: int) -> int:
    """Convertit (row, col) en index d'etat."""
    return r * GRID_SIZE + c


def step_grid(r: int, c: int, action: int) -> tuple[int, int]:
    """Applique deterministiquement une action depuis (r, c).
    Retourne la nouvelle case ; reste sur place si on tape un mur."""
    dr, dc = ACTION_DELTAS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return nr, nc
    return r, c  # mur : on ne bouge pas


def build_mdp() -> tuple[np.ndarray, np.ndarray]:
    """Construit P et R sous forme tabulaire.

    P[s, a, s'] = probabilite de transition.
    R[s, a, s'] = recompense recue lors de la transition (s, a) -> s'.

    Convention : pour un etat terminal, on boucle sur soi-meme avec recompense 0
    (equivalent a "fin d'episode" pour les algos de DP).
    """
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    for s in range(N_STATES):
        r, c = state_to_rc(s)

        # Etats terminaux : self-loop, recompense 0 (l'episode est fini)
        if (r, c) in TERMINAL_STATES:
            for a in range(N_ACTIONS):
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0
            continue

        for a in range(N_ACTIONS):
            # Trois successeurs possibles avec leurs probabilites
            outcomes = [
                (a, P_INTENDED),
                (LEFT_OF[a], P_LEFT_SLIP),
                (RIGHT_OF[a], P_RIGHT_SLIP),
            ]
            for actual_action, prob in outcomes:
                nr, nc = step_grid(r, c, actual_action)
                ns = rc_to_state(nr, nc)
                P[s, a, ns] += prob

                # Recompense determinee par l'etat d'arrivee
                if (nr, nc) == GOAL:
                    rew = GOAL_REWARD
                elif (nr, nc) == LAVA:
                    rew = LAVA_REWARD
                else:
                    rew = LIVING_PENALTY
                # On accumule la recompense ponderee (esperance), puis on
                # divisera par P[s, a, ns] pour obtenir la R[s, a, ns] effective.
                R[s, a, ns] += prob * rew

    # Normalisation : R[s, a, s'] doit etre la recompense conditionnelle a la
    # transition. On a stocke prob*rew, donc on divise par P quand P > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(P > 0, R / np.maximum(P, 1e-12), 0.0)

    # Verification : sum sur s' = 1 pour chaque (s, a)
    assert np.allclose(P.sum(axis=2), 1.0), "Probabilites de transition invalides"
    return P, R


# ---------------------------------------------------------------------------
# 2. Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = GAMMA,
    eps: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Resout l'equation de Bellman optimality par iteration de point fixe.

    Retourne :
        V : valeur optimale par etat, shape (N_STATES,)
        pi : politique optimale (action par etat), shape (N_STATES,)
        deltas : historique des ecarts ||V_{k+1} - V_k||_inf (pour analyse de convergence)
    """
    V = np.zeros(N_STATES)
    deltas = []

    for k in range(max_iter):
        # Q[s, a] = somme_{s'} P(s'|s,a) * (R(s, a, s') + gamma * V(s'))
        # Vectorise : (N_STATES, N_ACTIONS, N_STATES) * (N_STATES,) broadcast
        Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
        # V_{k+1}(s) = max_a Q(s, a)
        V_new = np.max(Q, axis=1)

        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new

        if delta < eps:
            break

    # Extraction de la politique gloutonne par rapport a V*
    Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
    pi = np.argmax(Q, axis=1)
    return V, pi, deltas


# ---------------------------------------------------------------------------
# 3. Policy Iteration
# ---------------------------------------------------------------------------

def policy_evaluation(
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = GAMMA,
    eps: float = 1e-10,
    max_iter: int = 10000,
) -> np.ndarray:
    """Evalue exactement la politique deterministe pi en iterant Bellman expected.

    On pourrait resoudre le systeme lineaire directement avec np.linalg.solve sur
    (I - gamma * P_pi) V = R_pi ; on prefere l'iteration pour rester pedagogique
    et coller au pseudo-code de Sutton & Barto.
    """
    V = np.zeros(N_STATES)
    # P_pi[s, s'] = P(s' | s, pi(s)) ; R_pi[s, s'] = R(s, pi(s), s')
    P_pi = P[np.arange(N_STATES), pi, :]
    R_pi = R[np.arange(N_STATES), pi, :]
    # E_pi[r + gamma V(s')] = sum_{s'} P_pi[s, s'] * (R_pi[s, s'] + gamma V[s'])
    expected_r = np.sum(P_pi * R_pi, axis=1)  # (N_STATES,)

    for _ in range(max_iter):
        V_new = expected_r + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < eps:
            V = V_new
            break
        V = V_new
    return V


def policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Rend la politique gloutonne par rapport a V."""
    Q = np.sum(P * (R + gamma * V[None, None, :]), axis=2)
    return np.argmax(Q, axis=1)


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = GAMMA,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Alterne evaluation et amelioration jusqu'au point fixe.

    Retourne :
        V : valeur sous pi*
        pi : politique optimale
        n_iter : nombre d'iterations externes (evaluation + amelioration)
    """
    # Politique initiale arbitraire : tout le monde fait "haut"
    pi = np.zeros(N_STATES, dtype=np.int64)

    for k in range(max_iter):
        V = policy_evaluation(pi, P, R, gamma)
        pi_new = policy_improvement(V, P, R, gamma)
        if np.array_equal(pi_new, pi):
            return V, pi, k + 1  # converge
        pi = pi_new

    # Si on sort de la boucle sans converger, on rend le dernier etat
    V = policy_evaluation(pi, P, R, gamma)
    return V, pi, max_iter


# ---------------------------------------------------------------------------
# 4. Affichage
# ---------------------------------------------------------------------------

ACTION_SYMBOL = {0: "^", 1: "v", 2: "<", 3: ">"}


def render_value(V: np.ndarray) -> str:
    """Affiche la grille avec les valeurs."""
    lines = []
    for r in range(GRID_SIZE):
        row_strs = []
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if (r, c) == GOAL:
                row_strs.append(" GOAL ")
            elif (r, c) == LAVA:
                row_strs.append(" LAVA ")
            else:
                row_strs.append(f"{V[s]:+.3f}")
        lines.append(" | ".join(row_strs))
    return "\n".join(lines)


def render_policy(pi: np.ndarray) -> str:
    """Affiche la grille avec les actions optimales."""
    lines = []
    for r in range(GRID_SIZE):
        row_strs = []
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            if (r, c) == GOAL:
                row_strs.append("G")
            elif (r, c) == LAVA:
                row_strs.append("X")
            else:
                row_strs.append(ACTION_SYMBOL[int(pi[s])])
        lines.append(" ".join(row_strs))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("J9 — GridWorld 4x4 : Value Iteration vs Policy Iteration")
    print("=" * 60)

    P, R = build_mdp()
    print(f"\nMDP : {N_STATES} etats, {N_ACTIONS} actions, gamma = {GAMMA}")
    print(f"Goal = {GOAL} (R=+1), Lava = {LAVA} (R=-1), living penalty = {LIVING_PENALTY}\n")

    # ----- Value Iteration -----
    print("--- Value Iteration ---")
    V_vi, pi_vi, deltas = value_iteration(P, R)
    print(f"Convergence en {len(deltas)} iterations (eps=1e-8)")
    # Quelques deltas pour montrer la decroissance geometrique
    show_idx = [0, 5, 10, 20, 50, len(deltas) - 1]
    show_idx = [i for i in show_idx if i < len(deltas)]
    print("Decroissance ||V_{k+1} - V_k||_inf (preuve de contraction) :")
    for i in show_idx:
        print(f"  iter {i:3d}: delta = {deltas[i]:.3e}")

    print("\nV* (value iteration) :")
    print(render_value(V_vi))
    print("\npi* (value iteration) :")
    print(render_policy(pi_vi))

    # ----- Policy Iteration -----
    print("\n--- Policy Iteration ---")
    V_pi, pi_pi, n_iter = policy_iteration(P, R)
    print(f"Convergence en {n_iter} iterations externes (eval + improvement)")

    print("\nV* (policy iteration) :")
    print(render_value(V_pi))
    print("\npi* (policy iteration) :")
    print(render_policy(pi_pi))

    # ----- Comparaison -----
    print("\n--- Comparaison VI vs PI ---")
    diff_v = np.max(np.abs(V_vi - V_pi))
    # Les deux algos doivent retourner la meme V* (a eps pres). Pour la politique,
    # il peut y avoir des "ties" (deux actions de meme Q-value) et argmax pick la
    # premiere. On verifie donc que pour les etats ou pi differe, les Q-valeurs
    # des deux actions choisies sont egales -- ce qui prouve que les deux politiques
    # sont co-optimales.
    Q_vi = np.sum(P * (R + GAMMA * V_vi[None, None, :]), axis=2)
    diff_states = np.where(pi_vi != pi_pi)[0]
    print(f"Ecart max sur V* : {diff_v:.3e}  (doit etre ~0, modulo eps)")
    print(f"Etats ou pi differe : {len(diff_states)}  (ties tolereeds si Q-values egales)")
    for s in diff_states:
        q_vi = Q_vi[s, pi_vi[s]]
        q_pi = Q_vi[s, pi_pi[s]]
        r, c = state_to_rc(s)
        print(f"  etat ({r},{c}): VI={ACTION_SYMBOL[int(pi_vi[s])]} (Q={q_vi:.6f})  "
              f"PI={ACTION_SYMBOL[int(pi_pi[s])]} (Q={q_pi:.6f})  diff={abs(q_vi - q_pi):.2e}")
    print("\nNote : VI fait beaucoup d'iterations bon marche, PI peu d'iterations cheres.")
    print("Sur ce GridWorld 16 etats, PI converge typiquement en < 6 iterations,")
    print("alors que VI prend des dizaines d'iterations pour eps=1e-8.")


if __name__ == "__main__":
    main()

"""
J6 - Controle classique : PID brut vs LQR sur un pendule inverse.

# requires: numpy, scipy, mujoco

Ce script :
  1. Modelise un pendule inverse non-lineaire (theta_ddot = (g/l) sin(theta) + u / (m l^2)).
  2. Linearise autour de l'equilibre instable theta = 0.
  3. Calcule un gain LQR via scipy.linalg.solve_continuous_are.
  4. Simule en parallele : LQR vs PID brut, depuis la meme condition initiale.
  5. Affiche metriques (settling time, overshoot, RMSE) et trajectoires.

Reference : [Tedrake, ch. 7], [Siciliano et al., 2009, ch. 8].
"""

import numpy as np
from scipy.linalg import solve_continuous_are

# -----------------------------------------------------------------------------
# 1. Parametres physiques du pendule
# -----------------------------------------------------------------------------
# On choisit des valeurs realistes : pendule de 1 m, masse 1 kg, gravite Terre.
# Ces valeurs determinent la frequence propre = sqrt(g/l) ~ 3.13 rad/s,
# autrement dit le pendule double son ecart en ~0.32 s sans controle.
g = 9.81   # gravite (m/s^2)
m = 1.0    # masse au bout du pendule (kg)
l = 1.0    # longueur du pendule (m)
b = 0.1    # frottement visqueux articulaire (N.m.s/rad), petit mais non nul

# Limites actionneur (anti-windup PID + saturation realiste)
U_MAX = 20.0  # N.m

# -----------------------------------------------------------------------------
# 2. Dynamique non-lineaire (utilisee pour la VRAIE simulation)
# -----------------------------------------------------------------------------
def pendulum_dynamics(state, u):
    """
    Dynamique non-lineaire du pendule inverse, articulation au pivot.
    state = [theta, theta_dot], avec theta = 0 a la verticale (equilibre instable).
    u = couple applique a l'articulation.

    Equation : (m l^2) theta_ddot = m g l sin(theta) - b theta_dot + u
    """
    theta, theta_dot = state
    # On utilise sin(theta) et non theta : on garde la non-linearite vraie pour la simu.
    theta_ddot = (g / l) * np.sin(theta) - (b / (m * l ** 2)) * theta_dot + u / (m * l ** 2)
    return np.array([theta_dot, theta_ddot])


def rk4_step(state, u, dt):
    """Integrateur Runge-Kutta 4 : plus precis que Euler pour des dt > 1ms."""
    k1 = pendulum_dynamics(state, u)
    k2 = pendulum_dynamics(state + 0.5 * dt * k1, u)
    k3 = pendulum_dynamics(state + 0.5 * dt * k2, u)
    k4 = pendulum_dynamics(state + dt * k3, u)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -----------------------------------------------------------------------------
# 3. Linearisation autour de l'equilibre instable theta = 0
# -----------------------------------------------------------------------------
# En lineaire : sin(theta) ~ theta, donc
#   theta_ddot = (g/l) theta - (b / m l^2) theta_dot + u / (m l^2)
# Forme x_dot = A x + B u avec x = [theta, theta_dot]^T.
A = np.array([
    [0.0, 1.0],
    [g / l, -b / (m * l ** 2)],
])
B = np.array([
    [0.0],
    [1.0 / (m * l ** 2)],
])

# Verification rapide : valeurs propres de A.
# On attend une racine reelle positive (mode instable).
eigvals_A = np.linalg.eigvals(A)
print(f"Valeurs propres de A (linearise) : {eigvals_A}")
print(f"  -> mode instable confirme (Re > 0) : {(eigvals_A.real > 0).any()}\n")

# -----------------------------------------------------------------------------
# 4. Calcul du gain LQR via Riccati algebrique continue
# -----------------------------------------------------------------------------
# Choix de Q, R par heuristique de Bryson (cf. theorie) :
#   - on tolere ~0.3 rad d'erreur sur theta -> Q[0,0] = 1/0.3^2 ~ 11
#   - on tolere ~1 rad/s sur theta_dot     -> Q[1,1] = 1/1^2 = 1
#   - on tolere couple jusqu'a 20 N.m      -> R = 1/20^2 = 0.0025
# Le ratio Q/R determine l'agressivite. Plus c'est eleve, plus K est grand.
Q = np.diag([10.0, 1.0])
R = np.array([[1.0]])

P = solve_continuous_are(A, B, Q, R)         # solveur scipy : ARE
K_lqr = np.linalg.solve(R, B.T @ P)          # K = R^-1 B^T P
print(f"Gain LQR K = {K_lqr.flatten()}")
# On peut verifier la stabilite en boucle fermee : valeurs propres de (A - B K).
A_cl = A - B @ K_lqr
print(f"Valeurs propres boucle fermee LQR : {np.linalg.eigvals(A_cl)}")
print(f"  -> toutes a Re < 0 (stable) : {(np.linalg.eigvals(A_cl).real < 0).all()}\n")


# -----------------------------------------------------------------------------
# 5. Controleurs : LQR + PID brut (a but de comparaison)
# -----------------------------------------------------------------------------
class PIDController:
    """PID articulaire SISO avec anti-windup par clamping et derivee de la mesure.

    On cible theta = 0 (verticale). Sur le pendule inverse, le PID brut a deux
    handicaps :
      - il ne sait pas que A est instable (pas de modele) ;
      - il reagit a l'erreur passee, jamais en feedforward.
    """

    def __init__(self, kp, ki, kd, u_max=U_MAX):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.u_max = u_max
        self.integral = 0.0
        self.prev_meas = None  # on derive la MESURE (anti derivative kick)

    def step(self, state, dt):
        theta, theta_dot = state
        target = 0.0
        error = target - theta

        # Integrateur avec anti-windup : on n'incremente que si la commande
        # ne sature pas dans le mauvais sens (clamping classique).
        candidate = self.integral + error * dt
        # On accepte d'abord, on clamp apres calcul.
        self.integral = candidate

        # Derivee de la mesure (theta_dot directement disponible dans l'etat ;
        # en pratique on filtrerait passe-bas).
        d_meas = theta_dot

        # Sortie PID. Note : derive de la MESURE => signe oppose a -de/dt.
        u = self.kp * error + self.ki * self.integral - self.kd * d_meas

        # Saturation + back-calculation simple.
        if u > self.u_max:
            self.integral -= error * dt   # rollback de l'integrale (clamping)
            u = self.u_max
        elif u < -self.u_max:
            self.integral -= error * dt
            u = -self.u_max

        return u


def lqr_controller(state, K=K_lqr, u_max=U_MAX):
    """u = -K x avec saturation. Aucune memoire interne (pas d'integrale)."""
    u = float(-(K @ state).item())
    return float(np.clip(u, -u_max, u_max))


# -----------------------------------------------------------------------------
# 6. Simulation comparative
# -----------------------------------------------------------------------------
def simulate(controller_fn, x0, dt=0.005, T=3.0, is_pid=False):
    """Simule le pendule non-lineaire avec le controleur passe en argument.

    controller_fn : appelable (state, dt?) -> u.
    Retourne (times, states, controls).
    """
    n_steps = int(T / dt)
    times = np.linspace(0.0, T, n_steps + 1)
    states = np.zeros((n_steps + 1, 2))
    controls = np.zeros(n_steps + 1)
    states[0] = x0

    for k in range(n_steps):
        if is_pid:
            u = controller_fn.step(states[k], dt)
        else:
            u = controller_fn(states[k])
        controls[k] = u
        states[k + 1] = rk4_step(states[k], u, dt)

    controls[-1] = controls[-2]  # padding pour avoir meme longueur
    return times, states, controls


# Condition initiale : pendule a 0.2 rad (~11.5 deg) de la verticale, vitesse nulle.
# Suffisamment proche pour que la linearisation soit valable, suffisamment loin
# pour que le controle ait du travail.
x0 = np.array([0.2, 0.0])
dt = 0.005
T = 3.0

# PID regle "raisonnablement" (Ziegler-Nichols approche, puis ajuste a la main
# pour eviter l'instabilite totale). Volontairement non-optimal : on veut
# illustrer ses limites face a un mode instable.
pid = PIDController(kp=50.0, ki=5.0, kd=10.0)

t_lqr, x_lqr, u_lqr = simulate(lqr_controller, x0, dt=dt, T=T, is_pid=False)
t_pid, x_pid, u_pid = simulate(pid, x0, dt=dt, T=T, is_pid=True)


# -----------------------------------------------------------------------------
# 7. Metriques
# -----------------------------------------------------------------------------
def settling_time(times, theta, threshold=0.02):
    """Temps a partir duquel |theta| reste sous threshold rad jusqu'a la fin."""
    inside = np.abs(theta) < threshold
    if not inside.any():
        return float("inf")
    # Cherche le dernier instant ou on sort, puis renvoie le suivant.
    outside_idx = np.where(~inside)[0]
    if len(outside_idx) == 0:
        return times[0]
    last_out = outside_idx[-1]
    if last_out + 1 >= len(times):
        return float("inf")
    return float(times[last_out + 1])


def overshoot(theta):
    """Overshoot = max(theta) - 0 si theta initial > 0, sinon |min(theta)|."""
    return float(np.max(np.abs(theta)))


def rmse(theta):
    return float(np.sqrt(np.mean(theta ** 2)))


print("=" * 60)
print("Comparaison LQR vs PID brut")
print("=" * 60)
print(f"{'Metrique':<25}{'LQR':>15}{'PID brut':>15}")
print(f"{'-' * 55}")
print(f"{'Settling time (|theta|<0.02 rad)':<25}"
      f"{settling_time(t_lqr, x_lqr[:, 0]):>14.3f}s"
      f"{settling_time(t_pid, x_pid[:, 0]):>14.3f}s")
print(f"{'Overshoot max |theta|':<25}"
      f"{overshoot(x_lqr[:, 0]):>14.3f} rad"
      f"{overshoot(x_pid[:, 0]):>14.3f} rad")
print(f"{'RMSE theta':<25}"
      f"{rmse(x_lqr[:, 0]):>14.3f} rad"
      f"{rmse(x_pid[:, 0]):>14.3f} rad")
print(f"{'Couple max':<25}"
      f"{np.max(np.abs(u_lqr)):>14.2f} N.m"
      f"{np.max(np.abs(u_pid)):>14.2f} N.m")
print()


# -----------------------------------------------------------------------------
# 8. Visualisation (optionnelle, ne plante pas si matplotlib absent)
# -----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_lqr, x_lqr[:, 0], label="LQR", linewidth=2)
    axes[0].plot(t_pid, x_pid[:, 0], label="PID brut", linewidth=2, linestyle="--")
    axes[0].axhline(0.0, color="k", linewidth=0.5)
    axes[0].set_ylabel("theta (rad)")
    axes[0].legend()
    axes[0].set_title("Pendule inverse - stabilisation")

    axes[1].plot(t_lqr, u_lqr, label="LQR", linewidth=2)
    axes[1].plot(t_pid, u_pid, label="PID brut", linewidth=2, linestyle="--")
    axes[1].set_ylabel("Couple u (N.m)")
    axes[1].set_xlabel("temps (s)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("pendule_lqr_vs_pid.png", dpi=120)
    print("Graphique sauvegarde dans pendule_lqr_vs_pid.png")
except ImportError:
    print("matplotlib absent, skipping plot.")


# -----------------------------------------------------------------------------
# 9. Note sur MuJoCo
# -----------------------------------------------------------------------------
# La meme demarche se transpose telle quelle a un pendule MuJoCo :
#   1. Charger un MJCF de pendule (ex : mujoco_menagerie ou modele jouet).
#   2. Construire (A, B) par perturbation numerique autour de qpos = 0 :
#        for i in range(nq):
#            qpos += eps ; mj_forward ; mesurer qacc ; (A, B) par differences.
#   3. Calculer K via solve_continuous_are.
#   4. A chaque step : data.ctrl[:] = -K @ np.concatenate([data.qpos, data.qvel]).
# La forme analytique 2D ci-dessus suffit pour ce module ; la version MuJoCo
# fait l'objet de l'exercice 03-hard du jour.

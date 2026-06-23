"""
J6 - Solutions des exercices easy / medium / hard.

# requires: numpy, scipy

Chaque solution est independante et lance ses propres simulations + tableaux.
La solution hard fait du computed torque sur un bras 2-DOF planaire et illustre
sa robustesse a une erreur de modele de 20%.

Reference : [Siciliano et al., 2009, ch. 7-8], [Tedrake, ch. 7].
"""

import numpy as np
from scipy.linalg import solve_continuous_are


# =============================================================================
# Exercice EASY : PID sur double integrateur
# =============================================================================
def solve_easy():
    """Reponse ecrite : voir docstring en bas."""

    def simulate_pid(kp, ki, kd, x0=(1.0, 0.0), T=5.0, dt=0.01):
        n = int(T / dt)
        x = np.array(x0, dtype=float)
        integrale = 0.0
        traj = np.zeros(n + 1)
        traj[0] = x[0]
        u_hist = np.zeros(n + 1)
        for k in range(n):
            error = 0.0 - x[0]
            integrale += error * dt
            u = kp * error + ki * integrale - kd * x[1]  # derive de la mesure
            # Double integrateur : x_ddot = u, integration semi-implicit Euler.
            x[1] += u * dt
            x[0] += x[1] * dt
            traj[k + 1] = x[0]
            u_hist[k + 1] = u
        return np.linspace(0, T, n + 1), traj, u_hist

    configs = [
        ("(a) P seul    Kp=10",        dict(kp=10, ki=0, kd=0)),
        ("(b) PD       Kp=10,Kd=2",   dict(kp=10, ki=0, kd=2)),
        ("(c) PID      Kp=10,Ki=5,Kd=2", dict(kp=10, ki=5, kd=2)),
        ("(d) Kp eleve Kp=100,Kd=2",  dict(kp=100, ki=0, kd=2)),
    ]
    print("=" * 70)
    print("EXERCICE EASY - PID sur double integrateur")
    print("=" * 70)
    print(f"{'Config':<32}{'Settling':>12}{'Overshoot':>12}{'RMSE':>12}")
    for name, gains in configs:
        t, x, u = simulate_pid(**gains)
        # Settling : dernier instant ou |x| > 0.02, on prend le suivant.
        outside = np.where(np.abs(x) > 0.02)[0]
        settling = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else float("inf")
        overshoot = float(np.max(np.abs(x)))
        rmse = float(np.sqrt(np.mean(x ** 2)))
        print(f"{name:<32}{settling:>11.3f}s{overshoot:>11.3f} {rmse:>11.3f}")
    print()
    print("Reponse : un controleur P pur sur double integrateur sans frottement")
    print("place les poles boucle fermee a +-j*sqrt(Kp), sur l'axe imaginaire.")
    print("Le systeme est conservatif : l'energie injectee oscille indefiniment.")
    print("Il faut un terme dissipatif (Kd ou frottement) pour amortir.\n")


# =============================================================================
# Exercice MEDIUM : LQR sur masse-ressort instable
# =============================================================================
def solve_medium():
    print("=" * 70)
    print("EXERCICE MEDIUM - LQR sur masse-ressort instable")
    print("=" * 70)

    m, k_spring, c = 1.0, -2.0, 0.1
    A = np.array([[0.0, 1.0], [-k_spring / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])
    print(f"Valeurs propres de A : {np.linalg.eigvals(A)}  (au moins une Re>0)\n")

    cases = [
        ("L1 effort cher",       np.diag([1.0, 1.0]),   np.array([[1.0]])),
        ("L2 etat cher",         np.diag([100.0, 1.0]), np.array([[1.0]])),
        ("L3 action quasi grat.", np.diag([10.0, 1.0]), np.array([[0.01]])),
    ]

    def simulate(controller, x0=(0.5, 0.0), T=5.0, dt=0.005, is_pid=False):
        n = int(T / dt)
        x = np.array(x0, dtype=float)
        traj = np.zeros((n + 1, 2))
        u_hist = np.zeros(n + 1)
        traj[0] = x
        integrale = 0.0
        for i in range(n):
            if is_pid:
                error = 0.0 - x[0]
                integrale += error * dt
                u = controller["kp"] * error + controller["ki"] * integrale - controller["kd"] * x[1]
            else:
                u = float(-(controller @ x).item())
            # Dynamique : m x_ddot = -k x - c x_dot + u  (k negatif ici).
            x_ddot = (-k_spring * x[0] - c * x[1] + u) / m
            x[1] += x_ddot * dt
            x[0] += x[1] * dt
            traj[i + 1] = x
            u_hist[i + 1] = u
        return np.linspace(0, T, n + 1), traj, u_hist

    def metrics(t, x, u):
        outside = np.where(np.abs(x[:, 0]) > 0.02)[0]
        sett = t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else float("inf")
        return sett, float(np.max(np.abs(x[:, 0]))), float(np.sqrt(np.mean(x[:, 0] ** 2))), float(np.max(np.abs(u)))

    print(f"{'Controleur':<25}{'Settling':>12}{'Overshoot':>12}{'RMSE':>10}{'|u|max':>10}")
    for name, Q, R in cases:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        eig_cl = np.linalg.eigvals(A - B @ K)
        assert (eig_cl.real < 0).all(), f"{name} : LQR n'a pas stabilise !"
        t, x, u = simulate(K)
        s, o, r, um = metrics(t, x, u)
        print(f"{name:<25}{s:>11.3f}s{o:>11.3f} {r:>9.3f} {um:>9.2f}")

    pid_gains = {"kp": 20.0, "ki": 0.0, "kd": 5.0}
    t, x, u = simulate(pid_gains, is_pid=True)
    s, o, r, um = metrics(t, x, u)
    print(f"{'PID Z-N':<25}{s:>11.3f}s{o:>11.3f} {r:>9.3f} {um:>9.2f}")

    print()
    print("Reponse :")
    print(" - L2 (Q etat cher) est le plus rapide : le LQR penalise tres durement")
    print("   l'erreur, donc K est plus grand et boucle fermee plus rapide.")
    print(" - L1 (R grand) utilise le moins d'effort : on paye cher chaque N.m,")
    print("   le solveur trouve une commande douce mais lente.")
    print(" - PID Z-N est lisible/transparent et ne demande PAS de modele (A, B).")
    print("   Sur ce systeme instable connu, le LQR domine ; en industrie, le PID")
    print("   gagne sur l'absence de modele et la robustesse aux variations parametriques.\n")


# =============================================================================
# Exercice HARD : Computed torque sur bras 2-DOF planaire
# =============================================================================
def solve_hard():
    print("=" * 70)
    print("EXERCICE HARD - Computed torque vs PID sur bras 2-DOF")
    print("=" * 70)

    # Parametres reels (utilises par la simulation)
    g_grav = 9.81
    real = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, lc1=0.5, lc2=0.5)
    real["I1"] = real["m1"] * real["l1"] ** 2 / 12.0
    real["I2"] = real["m2"] * real["l2"] ** 2 / 12.0

    def dynamics_matrices(q, qd, p):
        """Retourne M, C @ qd, g pour le bras 2-DOF planaire vertical."""
        m1, m2 = p["m1"], p["m2"]
        l1, lc1, lc2 = p["l1"], p["lc1"], p["lc2"]
        I1, I2 = p["I1"], p["I2"]
        q1, q2 = q
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        # Matrice d'inertie M(q)
        M11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I1 + I2
        M12 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2
        M22 = m2 * lc2 ** 2 + I2
        M = np.array([[M11, M12], [M12, M22]])
        # Coriolis (h = -m2 l1 lc2 sin q2)
        h = -m2 * l1 * lc2 * s2
        # C @ qd avec convention sign(h) -> [h*qd2*(2*qd1+qd2), -h*qd1^2]
        Cqd = np.array([
            h * qd[1] * (2 * qd[0] + qd[1]),
            -h * qd[0] ** 2,
        ])
        # Gravite (axes : q1 mesure depuis horizontale)
        gvec = np.array([
            (m1 * lc1 + m2 * l1) * g_grav * np.cos(q1) + m2 * lc2 * g_grav * np.cos(q1 + q2),
            m2 * lc2 * g_grav * np.cos(q1 + q2),
        ])
        return M, Cqd, gvec

    def step_dynamics(q, qd, tau, p, dt):
        """Avance la dynamique : qdd = M^-1 (tau - C qd - g). RK4."""
        def f(state, _t):
            qq = state[:2]
            qqd = state[2:]
            Mm, Cc, gg = dynamics_matrices(qq, qqd, p)
            qdd = np.linalg.solve(Mm, tau - Cc - gg)
            return np.concatenate([qqd, qdd])
        s0 = np.concatenate([q, qd])
        k1 = f(s0, 0)
        k2 = f(s0 + 0.5 * dt * k1, 0)
        k3 = f(s0 + 0.5 * dt * k2, 0)
        k4 = f(s0 + dt * k3, 0)
        s1 = s0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s1[:2], s1[2:]

    # Trajectoire de reference
    T_total = 4.0
    dt = 0.001
    n_steps = int(T_total / dt)
    times = np.linspace(0, T_total, n_steps + 1)
    omega = 2 * np.pi / 4.0
    q_d  = np.stack([0.5 * np.sin(omega * times),
                     0.3 * np.cos(omega * times)], axis=1)
    qd_d = np.stack([0.5 * omega * np.cos(omega * times),
                     -0.3 * omega * np.sin(omega * times)], axis=1)
    qdd_d = np.stack([-0.5 * omega ** 2 * np.sin(omega * times),
                      -0.3 * omega ** 2 * np.cos(omega * times)], axis=1)

    # Gains
    Kp_pid = np.array([100.0, 100.0])
    Ki_pid = np.array([10.0, 10.0])
    Kd_pid = np.array([20.0, 20.0])
    Kp_ctc = 100.0
    Kd_ctc = 2.0 * np.sqrt(Kp_ctc)

    def run(controller_kind, model_params):
        """controller_kind in {'pid', 'ctc'}; model_params : params utilises PAR LE CONTROLEUR."""
        q = np.zeros(2)
        qd = np.zeros(2)
        traj = np.zeros((n_steps + 1, 2))
        traj[0] = q
        integrale = np.zeros(2)
        for k in range(n_steps):
            e = q_d[k] - q
            ed = qd_d[k] - qd
            if controller_kind == "pid":
                integrale += e * dt
                tau = Kp_pid * e + Ki_pid * integrale + Kd_pid * ed
            else:
                # Computed torque utilisant model_params (peut etre faux)
                M_hat, C_hat, g_hat = dynamics_matrices(q, qd, model_params)
                tau = M_hat @ (qdd_d[k] + Kp_ctc * e + Kd_ctc * ed) + C_hat + g_hat
            # Simulation : utilise TOUJOURS les vrais parametres reels
            q, qd = step_dynamics(q, qd, tau, real, dt)
            traj[k + 1] = q
        rmse = np.sqrt(np.mean((traj - q_d) ** 2, axis=0))
        return rmse

    # Controleur avec modele exact
    rmse_pid_exact = run("pid", real)
    rmse_ctc_exact = run("ctc", real)

    # Controleur avec erreur de modele : masses surestimees de 20%
    bad = dict(real)
    bad["m1"] = real["m1"] * 1.2
    bad["m2"] = real["m2"] * 1.2
    bad["I1"] = bad["m1"] * bad["l1"] ** 2 / 12.0
    bad["I2"] = bad["m2"] * bad["l2"] ** 2 / 12.0
    rmse_pid_bad = run("pid", bad)  # le PID n'utilise pas le modele -> identique
    rmse_ctc_bad = run("ctc", bad)

    print(f"{'Cas':<25}{'RMSE q1 (rad)':>18}{'RMSE q2 (rad)':>18}")
    print(f"{'PID (modele exact)':<25}{rmse_pid_exact[0]:>17.4f} {rmse_pid_exact[1]:>17.4f}")
    print(f"{'CTC (modele exact)':<25}{rmse_ctc_exact[0]:>17.4f} {rmse_ctc_exact[1]:>17.4f}")
    print(f"{'PID (erreur modele)':<25}{rmse_pid_bad[0]:>17.4f} {rmse_pid_bad[1]:>17.4f}")
    print(f"{'CTC (erreur modele)':<25}{rmse_ctc_bad[0]:>17.4f} {rmse_ctc_bad[1]:>17.4f}")

    print()
    print("Reponse :")
    print(" - Sans erreur de modele, le CTC tracke quasiment parfaitement : la dynamique")
    print("   d'erreur devient e_ddot + Kd e_dot + Kp e = 0, donc convergence exponentielle.")
    print(" - Avec 20% d'erreur sur les masses, le decouplage n'est plus exact :")
    print("   les forces de Coriolis et la gravite ne sont compensees qu'approximativement.")
    print("   La RMSE remonte. Le PID, qui n'utilise pas de modele, reste insensible aux")
    print("   parametres mais traque mal initialement.")
    print(" - Pour restaurer la robustesse : adaptive control (Slotine-Li) qui ajuste les")
    print("   parametres en ligne, ou robust control H-inf qui tolere une enveloppe d'incertitude.")


if __name__ == "__main__":
    solve_easy()
    solve_medium()
    solve_hard()

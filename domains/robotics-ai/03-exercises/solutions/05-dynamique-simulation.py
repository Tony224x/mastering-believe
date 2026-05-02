"""
Solutions J5 — easy + medium + hard.

Run :
  python 05-dynamique-simulation.py

# requires: mujoco, numpy
"""

from __future__ import annotations

import math
import sys

import numpy as np

try:
    import mujoco  # type: ignore
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print(
        "[INFO] Module `mujoco` non installe. La partie EASY (numpy seul) tourne ; "
        "MEDIUM et HARD seront skippees. `pip install mujoco` pour les debloquer."
    )


# ===========================================================================
# Easy : pendule numerique (Euler semi-implicite) vs analytique petit-angle.
# ===========================================================================
def solve_easy() -> None:
    print("=" * 70)
    print("EASY — pendule simple : numerique vs analytique petit-angle")
    print("=" * 70)

    g, L, m = 9.81, 1.0, 1.0
    omega = math.sqrt(g / L)  # pulsation petit-angle
    dt = 1e-3
    T = 5.0
    n_steps = int(T / dt)

    for theta0 in (0.1, 1.5):
        # Etat numerique : on integre l'equation EXACTE θ̈ = -(g/L) sin θ.
        theta = theta0
        theta_dot = 0.0
        # Energie totale a t=0 : a θ̇=0 donc T=0, V = mgL(1 - cos θ).
        E0 = m * g * L * (1.0 - math.cos(theta))

        # On stocke pour l'affichage horaire.
        log_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        log = {}
        log[0.0] = (theta, theta0 * math.cos(omega * 0.0))

        for k in range(1, n_steps + 1):
            # Euler semi-implicite : on met d'abord a jour la vitesse, PUIS la position
            # avec la nouvelle vitesse. C'est l'ordre qui rend le schema symplectique.
            theta_dot += dt * (-(g / L) * math.sin(theta))
            theta += dt * theta_dot
            t = k * dt
            # On capture aux instants entiers (au plus proche).
            for tl in log_times:
                if abs(t - tl) < dt / 2 and tl not in log:
                    log[tl] = (theta, theta0 * math.cos(omega * tl))

        # Energie a la fin
        E_end = 0.5 * m * L * L * theta_dot * theta_dot + m * g * L * (1.0 - math.cos(theta))
        rel_drift = abs(E_end - E0) / (abs(E0) + 1e-12)

        print(f"\n--- theta_0 = {theta0:.2f} rad ---")
        print(f"{'t (s)':>6}  {'theta_num':>10}  {'theta_ana':>10}  {'|ecart|':>10}")
        for tl in log_times:
            tn, ta = log[tl]
            print(f"{tl:6.1f}  {tn:10.4f}  {ta:10.4f}  {abs(tn - ta):10.4f}")
        print(f"Derive d'energie totale : {rel_drift * 100:.4f} %")


# ===========================================================================
# Medium : matrice d'inertie d'un bras 2-DOF planaire en MuJoCo.
# ===========================================================================
ARM_2DOF_MJCF = """
<mujoco model="bras_2dof_planaire">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -9.81">
    <flag energy="enable"/>
  </option>
  <worldbody>
    <!-- Lien 1 : 0.5 m de long, 1 kg, hinge autour de y a la base -->
    <body name="link1" pos="0 0 1.0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.03" mass="1.0"/>
      <!-- Lien 2 : 0.4 m de long, 0.5 kg, hinge en bout du lien 1 -->
      <body name="link2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.025" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def inertia_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Retourne M(q) en forme dense via mj_fullM."""
    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M


def solve_medium() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM — matrice d'inertie M(q) d'un bras 2-DOF")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_string(ARM_2DOF_MJCF)
    data = mujoco.MjData(model)

    test_configs = [
        np.array([0.0, 0.0]),
        np.array([math.pi / 4, math.pi / 4]),
        np.array([math.pi / 2, math.pi / 2]),
        np.array([0.0, math.pi / 2]),
    ]

    cond_numbers = []
    for q in test_configs:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        # mj_forward propage les positions Cartesiennes — indispensable avant mj_fullM
        # car M(q) depend de la geometrie courante.
        mujoco.mj_forward(model, data)

        M = inertia_matrix(model, data)
        eigvals = np.linalg.eigvalsh(M)
        cond = eigvals.max() / eigvals.min()
        cond_numbers.append(cond)

        print(f"\nq = [{q[0]:+.3f}, {q[1]:+.3f}] rad")
        print(f"M(q) =\n{M}")
        print(f"  symetrique ? {np.allclose(M, M.T)}")
        print(f"  valeurs propres : {eigvals}  (toutes > 0 : {(eigvals > 0).all()})")
        print(f"  condition number : {cond:.3f}")

    worst_idx = int(np.argmax(cond_numbers))
    worst_q = test_configs[worst_idx]
    print(
        f"\nConfiguration la plus mal conditionnee : q = {worst_q.tolist()} "
        f"(cond = {cond_numbers[worst_idx]:.3f})."
    )
    print(
        "Interpretation : quand le bras est etendu (j2 ~ 0), l'inertie reflechie au "
        "joint 1 est elevee tandis que celle de j2 reste petite — fort couplage et "
        "fort ratio d'inerties, donc gains de controleur fixes mal repartis."
    )


# ===========================================================================
# Hard : compensation gravite via mj_rne, avec test adversarial parametres.
# ===========================================================================
def gravity_torques(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Retourne g(q) = couple statique necessaire pour maintenir l'equilibre.

    Astuce : avec qvel=0 et qacc=0 force, mj_rne calcule M·0 + C·0 + g = g.
    On sauvegarde-restaure qvel/qacc pour ne pas casser la simu en cours.
    """
    qvel_save = data.qvel.copy()
    qacc_save = data.qacc.copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    out = np.zeros(model.nv)
    # Le 2eme arg de mj_rne (flg_acc=0) signifie : on prend qacc tel quel.
    mujoco.mj_rne(model, data, 0, out)
    data.qvel[:] = qvel_save
    data.qacc[:] = qacc_save
    return out


def simulate_2dof(
    model: mujoco.MjModel,
    q0: np.ndarray,
    duration: float,
    apply_gravity_comp: bool,
    perturbation_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simule le bras 2-DOF.

    Si apply_gravity_comp=True, on applique a chaque pas le couple g(q) calcule
    sur un *clone* du modele dont les masses sont multipliees par perturbation_factor
    (1.0 = compensation parfaite).
    """
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Modele utilise pour CALCULER g(q) (peut differer du modele de simu).
    if perturbation_factor != 1.0:
        model_for_comp = mujoco.MjModel.from_xml_string(ARM_2DOF_MJCF)
        model_for_comp.body_mass[:] = model.body_mass * perturbation_factor
        # Il faut aussi rappeler mj_setConst si on touche les inerties — ici masses seules.
        data_for_comp = mujoco.MjData(model_for_comp)
    else:
        model_for_comp = model
        data_for_comp = data  # pas besoin de duplicata si non perturbe

    n_steps = int(duration / model.opt.timestep)
    log_times = [0.0, 1.0, 2.0, 5.0]
    log_q: dict[float, np.ndarray] = {0.0: q0.copy()}

    for k in range(1, n_steps + 1):
        if apply_gravity_comp:
            if perturbation_factor != 1.0:
                # Synchronise data_for_comp sur l'etat courant
                data_for_comp.qpos[:] = data.qpos
                data_for_comp.qvel[:] = 0.0
                mujoco.mj_forward(model_for_comp, data_for_comp)
                tau_g = gravity_torques(model_for_comp, data_for_comp)
            else:
                tau_g = gravity_torques(model, data)
            data.qfrc_applied[:] = tau_g
        else:
            data.qfrc_applied[:] = 0.0

        mujoco.mj_step(model, data)
        t = k * model.opt.timestep
        for tl in log_times:
            if abs(t - tl) < model.opt.timestep / 2 and tl not in log_q:
                log_q[tl] = data.qpos.copy()

    # Energie totale finale
    energy_final = float(data.energy[0] + data.energy[1])
    q_log = np.array([log_q[t] for t in log_times])
    return np.array(log_times), q_log, energy_final


def solve_hard() -> None:
    print("\n" + "=" * 70)
    print("HARD — compensation de gravite open-loop avec mj_rne")
    print("=" * 70)

    model = mujoco.MjModel.from_xml_string(ARM_2DOF_MJCF)
    # Conservation propre : pas de damping ni frictionloss.
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    q0 = np.array([math.pi / 3, -math.pi / 4])

    # Phase A : chute libre
    print("\n--- Phase A : chute libre (aucun couple applique) ---")
    times, q_log_A, E_A = simulate_2dof(model, q0, duration=2.0, apply_gravity_comp=False)
    for t, q in zip(times, q_log_A, strict=False):
        if t <= 2.0:
            print(f"t = {t:.1f} s  ->  q = [{q[0]:+.4f}, {q[1]:+.4f}]")

    # Phase B : compensation gravite parfaite
    print("\n--- Phase B : compensation gravite via mj_rne (modele exact) ---")
    times, q_log_B, E_B = simulate_2dof(model, q0, duration=5.0, apply_gravity_comp=True)
    for t, q in zip(times, q_log_B, strict=False):
        drift = float(np.linalg.norm(q - q0))
        print(f"t = {t:.1f} s  ->  q = [{q[0]:+.4f}, {q[1]:+.4f}]   ||q-q0|| = {drift:.6f}")

    # Phase B' : compensation imparfaite (masses sur-estimees a +10 %)
    print("\n--- Phase B' : compensation avec masses sur-estimees a +10 % ---")
    times, q_log_C, E_C = simulate_2dof(
        model, q0, duration=5.0, apply_gravity_comp=True, perturbation_factor=1.1
    )
    for t, q in zip(times, q_log_C, strict=False):
        drift = float(np.linalg.norm(q - q0))
        print(f"t = {t:.1f} s  ->  q = [{q[0]:+.4f}, {q[1]:+.4f}]   ||q-q0|| = {drift:.6f}")

    print(
        "\nLecon : compensation parfaite -> ||q-q0|| reste ~0 (open-loop tient). "
        "Compensation imparfaite (+10 % masse) -> derive lente : g(q) calcule est "
        "trop fort, le bras se redresse legerement. Sans feedback, l'erreur "
        "parametrique s'integre. -> J6 : on ajoute un PD pour fermer la boucle."
    )


if __name__ == "__main__":
    solve_easy()
    if HAS_MUJOCO:
        solve_medium()
        solve_hard()
    else:
        print("\n[SKIP] medium + hard : mujoco non installe.")

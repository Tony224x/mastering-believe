"""
J5 — Dynamique + simulation MuJoCo hands-on.

Charge le Franka Panda depuis MuJoCo Menagerie, le fait tomber sous gravite (sans
controle, sans table), et verifie la conservation de l'energie totale T + V.

Si Menagerie n'est pas trouve, fallback sur un MJCF inline (pendule simple) pour
que le script tourne dans tous les environnements.

Sources :
  [Lynch & Park, 2017, ch. 8] — formulation Lagrange et Newton-Euler
  [Siciliano et al., 2009, ch. 7] — proprietes structurelles
  [MuJoCo docs, computation/integrator] — Euler semi-implicite, energie

Run :
  python 05-dynamique-simulation.py

# requires: mujoco, numpy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco  # type: ignore
except ImportError:
    print(
        "[ERR] Le module `mujoco` n'est pas installe. Installe-le via :\n"
        "        pip install mujoco\n"
        "      Le script peut alors faire tomber le Franka (si Menagerie clone)\n"
        "      ou le pendule MJCF inline en fallback. Sortie graceful."
    )
    sys.exit(0)


# ---------------------------------------------------------------------------
# 1. MJCF de fallback : pendule simple sans dissipation, pour valider que le
#    script tourne meme si Menagerie n'est pas installe. On active
#    explicitement le calcul d'energie via <flag energy="enable"/>.
# ---------------------------------------------------------------------------
PENDULUM_MJCF = """
<mujoco model="pendule_conservatif">
  <option timestep="0.001" integrator="Euler" gravity="0 0 -9.81">
    <!-- enable=energy fait que MuJoCo calcule data.energy = [T, V] gratuitement -->
    <flag energy="enable"/>
  </option>
  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <!-- hinge sans damping ni frictionloss : systeme conservatif -->
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <!-- masse 1 kg, tige de 1 m, inertie ponctuelle au bout -->
      <geom type="capsule" fromto="0 0 0  0 0 -1" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _try_load_franka() -> mujoco.MjModel | None:
    """
    Tente de charger le Franka Panda depuis Menagerie.

    Cherche dans les emplacements habituels. Si rien n'est trouve, retourne None
    et on tombera sur le pendule. La logique est volontairement large car
    l'utilisateur peut avoir clone Menagerie n'importe ou.
    """
    candidates = [
        # Convention : ~/mujoco_menagerie/...
        Path.home() / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml",
        # Convention : repo voisin du projet courant
        Path.cwd() / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml",
        # Variable d'env explicite : MUJOCO_MENAGERIE=/chemin/vers/mujoco_menagerie
        Path(os.environ.get("MUJOCO_MENAGERIE", "")) / "franka_emika_panda" / "panda.xml",
    ]

    for path in candidates:
        if path.is_file():
            try:
                model = mujoco.MjModel.from_xml_path(str(path))
                # Active le calcul d'energie a la volee si pas deja active dans le MJCF.
                # Bit ENERGY dans mjOption.enableflags = 1 << mjENBL_ENERGY (= 1 << 1 = 2).
                model.opt.enableflags |= (1 << mujoco.mjtEnableBit.mjENBL_ENERGY)
                # Coupe la dissipation pour un test de conservation d'energie propre :
                # on remet a zero le damping et frictionloss articulaire.
                model.dof_damping[:] = 0.0
                model.dof_frictionloss[:] = 0.0
                print(f"[OK] Modele Franka Panda charge depuis : {path}")
                return model
            except Exception as exc:  # noqa: BLE001 — on tolere et fallback
                print(f"[WARN] Echec chargement {path} : {exc}")
                continue

    print("[INFO] Menagerie introuvable — fallback sur pendule MJCF inline.")
    return None


def load_model() -> tuple[mujoco.MjModel, str]:
    """Renvoie un modele utilisable et un label pour l'affichage."""
    franka = _try_load_franka()
    if franka is not None:
        return franka, "Franka Panda (Menagerie)"
    # Le pendule a deja energy=enable dans son MJCF, pas besoin de toggler.
    return mujoco.MjModel.from_xml_string(PENDULUM_MJCF), "Pendule simple (fallback)"


def kinetic_energy(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """
    Energie cinetique calculee a la main : T = 1/2 q_dotᵀ M q_dot.

    Sert a recouper le data.energy[0] que MuJoCo calcule en interne — le pas-a-pas
    main est pedagogique et permet de comprendre que M(q) est l'inertie generalisee.
    """
    nv = model.nv
    full_M = np.zeros((nv, nv))
    # mj_fullM convertit la matrice d'inertie compacte de MuJoCo en sa forme dense.
    mujoco.mj_fullM(model, full_M, data.qM)
    return float(0.5 * data.qvel @ full_M @ data.qvel)


def run_freefall(steps: int = 2000) -> None:
    """
    Simule le robot tombant sous sa seule gravite, sans actuateur ni table,
    et logge l'energie totale toutes les `log_every` iterations.
    """
    model, label = load_model()
    data = mujoco.MjData(model)

    print(f"\n=== Simulation : {label} ===")
    print(f"nq = {model.nq}, nv = {model.nv}, dt = {model.opt.timestep}")

    # Pose initiale "raisonnable" :
    # - Sur Franka, on prend la pose neutre par defaut (qpos = 0) et on lui donne
    #   un petit coup de vitesse sur l'epaule pour que la chute ne soit pas triviale.
    # - Sur le pendule, on l'incline a 0.5 rad pour qu'il oscille.
    if "Pendule" in label:
        data.qpos[0] = 0.5  # angle initial du pendule (rad)
    else:
        # Pose Franka : configuration "ready" approximative, doigts ouverts si presents.
        # On evite de toucher qpos pour ne pas se planter sur les versions de Menagerie
        # qui ajoutent une base flottante : on prend juste qpos par defaut (mj_resetData
        # le met a la qpos0 du modele).
        mujoco.mj_resetData(model, data)
        # Petit kick sur la premiere articulation pour briser la symetrie statique.
        if model.nv > 0:
            data.qvel[0] = 0.2

    # Synchronise xpos/xmat/qfrc_bias avec qpos/qvel + calcule data.energy.
    mujoco.mj_forward(model, data)

    energies: list[tuple[float, float, float, float]] = []
    log_every = max(1, steps // 10)

    for step in range(steps):
        mujoco.mj_step(model, data)
        if step % log_every == 0 or step == steps - 1:
            t_mj = float(data.energy[0])  # cinetique calculee par MuJoCo
            v_mj = float(data.energy[1])  # potentielle calculee par MuJoCo
            t_manual = kinetic_energy(model, data)
            energies.append((data.time, t_mj, v_mj, t_manual))

    # ---------------- Affichage ----------------
    print("\nt (s)    T_mujoco (J)   V_mujoco (J)   T_manual (J)   T+V (J)")
    print("-" * 70)
    e0 = None
    for t, t_mj, v_mj, t_manual in energies:
        e_total = t_mj + v_mj
        if e0 is None:
            e0 = e_total
        drift = e_total - e0
        print(f"{t:6.3f}   {t_mj:11.4f}    {v_mj:11.4f}    {t_manual:11.4f}    "
              f"{e_total:8.4f}  (derive {drift:+.4f})")

    e_first = energies[0][1] + energies[0][2]
    e_last = energies[-1][1] + energies[-1][2]
    rel_drift = abs(e_last - e_first) / (abs(e_first) + 1e-9)
    print(f"\nDerive relative energie totale : {rel_drift * 100:.3f} %")
    if rel_drift < 0.05:
        print("==> Conservation d'energie respectee a < 5 % (Euler semi-implicite OK).")
    else:
        print("==> Derive importante : reduire dt ou verifier les parametres dissipation.")


if __name__ == "__main__":
    run_freefall(steps=2000)

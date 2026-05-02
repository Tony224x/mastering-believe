"""
Solutions J28 — Packaging capstone + integration LogiSim

Ce fichier contient les corriges des 3 exercices :
  - easy   : README.md portfolio-grade pour le capstone
  - medium : Makefile + scripts/ pour orchestrer training/eval/demo
  - hard   : roadmap.md vers le picking AGV temps reel dans FleetSim

Chaque corrige est expose comme une string Python documentee + un helper
qui ecrit le fichier sur disque dans un sous-dossier `out/` du cwd.
Lance `python 28-capstone-packaging-logisim.py` pour generer les 3 fichiers.

References : REFERENCES.md #19 (Diffusion Policy), #16 (Helix Logistics),
#18 (TRI LBM), #27 (LeRobot dataset), shared/logistics-context.md.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


# =====================================================================
# EASY — README portfolio-grade
# =====================================================================
# Contraintes : 5 sections dans l'ordre, sous 80 lignes, 3 commandes max
# au quickstart, un GIF en haut, un tableau Results avec chiffres realistes.

EASY_README_MD = dedent("""\
    # Diffusion Policy — PushT Capstone

    ![demo](figures/demo.gif)

    Visuomotor policy that pushes a T-shaped block toward a target — trained
    via DDPM on 200 teleop demos. **78% success rate, 36 ms / step** on a
    single consumer GPU.

    ## What it does

    - Trains a Diffusion Policy (image-conditioned UNet 1D + DDPM scheduler)
      on the PushT benchmark (Chi et al. 2023).
    - Predicts an **8-step action chunk** per inference, replans every 4 steps
      (receding horizon).
    - Reaches **78% success** on 50 evaluation rollouts vs **34%** for a
      Behavior-Cloning baseline trained on the same data.

    ## Quickstart

    ```bash
    pip install -e .
    python demo.py
    make eval                # optional, full 50-rollout evaluation
    ```

    ## Architecture

    ```
    image (96x96) ──► ResNet18 ┐
    agent_pos ───────────────► ├─► UNet 1D ──► DDPM denoise ──► action chunk (8x2)
    timestep emb ────────────► ┘
    ```

    Visual observations are encoded by a frozen ResNet18 and concatenated with
    the agent state. A 1D UNet conditioned on the diffusion timestep denoises
    Gaussian noise into an 8-step action sequence over 100 DDPM steps. The
    first 4 actions are executed before replanning.

    ## Results

    | Method            | Success rate (50 rollouts) | Mean episode length |
    |-------------------|----------------------------|---------------------|
    | Diffusion Policy  | 78%                        | 138 steps           |
    | Behavior Cloning  | 34%                        | 196 steps (timeout) |

    Full numbers and per-seed breakdown : [`eval/results.json`](eval/results.json).
""")


# =====================================================================
# MEDIUM — Makefile + scripts/
# =====================================================================
# Contraintes : >=6 cibles, .PHONY, echo des actions, exit codes propres,
# pas de chemins en dur, eval/results.json avec 4 cles minimum.

MEDIUM_MAKEFILE = dedent("""\
    # Capstone Diffusion Policy — Makefile
    # Compatible : Linux/macOS (make natif), Windows (GnuWin32 ou WSL)
    # Alternative cross-OS : tasks.py + invoke (https://www.pyinvoke.org/)

    .PHONY: install data train eval demo clean lint test ci

    PY ?= python
    CFG_TRAIN ?= configs/train.yaml
    CFG_EVAL  ?= configs/eval.yaml
    CKPT      ?= checkpoints/pusht_dp_final.ckpt
    DATA      ?= data/pusht_demos.npz
    RESULTS   ?= eval/results.json

    install:
\t@echo ">>> Installing capstone in editable mode"
\t$(PY) -m pip install -e .

    data:
\t@echo ">>> Generating PushT teleop demos -> $(DATA)"
\t$(PY) scripts/generate_demos.py --n_demos 200 --output $(DATA) --seed 42

    train:
\t@echo ">>> Training Diffusion Policy ($(CFG_TRAIN))"
\t$(PY) scripts/train.py --config $(CFG_TRAIN) --checkpoint_out $(CKPT) --epochs 100

    eval:
\t@echo ">>> Evaluating on 50 rollouts -> $(RESULTS)"
\t$(PY) scripts/eval.py --config $(CFG_EVAL) --checkpoint $(CKPT) \\\\
\t    --n_rollouts 50 --results_out $(RESULTS)

    demo:
\t@echo ">>> Running demo (10 rollouts + visualization)"
\t$(PY) demo.py

    lint:
\t@echo ">>> Linting"
\truff check src scripts demo.py

    test:
\t@echo ">>> Running unit tests"
\tpytest -q tests/

    ci: lint test data train eval
\t@echo ">>> CI pipeline OK"

    clean:
\t@echo ">>> Cleaning build artifacts and caches"
\trm -rf build dist *.egg-info __pycache__ .pytest_cache .ruff_cache
\trm -rf data/*.cache figures/*.tmp
""")


MEDIUM_TRAIN_YAML = dedent("""\
    # configs/train.yaml — Diffusion Policy training hyperparams
    # Override via CLI : python scripts/train.py --config configs/train.yaml --epochs 50

    seed: 42
    device: cuda    # cpu | cuda | mps

    dataset:
      path: data/pusht_demos.npz
      val_split: 0.1

    policy:
      action_dim: 2
      action_horizon: 8       # k = nombre d'actions predites par chunk
      observation_horizon: 2  # context window
      vision_encoder: resnet18
      vision_pretrained: true

    diffusion:
      n_train_timesteps: 100
      beta_schedule: squaredcos_cap_v2
      prediction_type: epsilon

    training:
      batch_size: 64
      n_epochs: 100
      learning_rate: 1.0e-4
      weight_decay: 1.0e-6
      lr_scheduler: cosine
      lr_warmup_steps: 500
      ema_power: 0.75
      ema_max_decay: 0.9999
      grad_clip: 1.0
      mixed_precision: fp16

    logging:
      log_every_n_steps: 50
      val_every_n_epochs: 5
      checkpoint_every_n_epochs: 10
""")


MEDIUM_GENERATE_DEMOS_PY = dedent('''\
    """scripts/generate_demos.py — generate PushT teleop demos.

    Usage: python scripts/generate_demos.py --n_demos 200 --output data/demos.npz --seed 42
    """
    import argparse
    import sys
    from pathlib import Path

    import numpy as np


    def main() -> int:
        parser = argparse.ArgumentParser(description="Generate PushT demos.")
        parser.add_argument("--n_demos", type=int, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        if args.n_demos <= 0:
            print(f"[ERROR] n_demos must be > 0 (got {args.n_demos})", file=sys.stderr)
            return 2

        rng = np.random.default_rng(args.seed)
        # Stub : on genere des trajectoires aleatoires representatives.
        # En prod : remplacer par un script de teleop (clavier/joystick/VR).
        demos = []
        for i in range(args.n_demos):
            length = rng.integers(80, 200)
            obs = rng.standard_normal((length, 6)).astype(np.float32)
            act = rng.standard_normal((length, 2)).astype(np.float32) * 0.05
            demos.append({"obs": obs, "act": act})

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, demos=np.array(demos, dtype=object))
        print(f"[OK] {args.n_demos} demos -> {args.output}")
        return 0


    if __name__ == "__main__":
        sys.exit(main())
''')


MEDIUM_TRAIN_PY = dedent('''\
    """scripts/train.py — train Diffusion Policy from a YAML config.

    Usage: python scripts/train.py --config configs/train.yaml --checkpoint_out checkpoints/dp.ckpt
    """
    import argparse
    import sys
    from pathlib import Path

    try:
        import yaml
    except ImportError:
        print("[ERROR] pyyaml is required (pip install pyyaml)", file=sys.stderr)
        sys.exit(1)


    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--checkpoint_out", type=Path, required=True)
        parser.add_argument("--epochs", type=int, default=None,
                            help="Override config training.n_epochs if set.")
        args = parser.parse_args()

        if not args.config.exists():
            print(f"[ERROR] Config not found: {args.config}", file=sys.stderr)
            return 2

        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if args.epochs is not None:
            cfg["training"]["n_epochs"] = args.epochs

        # Stub training loop : ici on prouve juste que le contrat tient.
        # En prod : importer src/diffusion_policy/train.py et appeler train(cfg).
        print(f"[INFO] Training with config: {args.config}")
        print(f"[INFO] Epochs: {cfg['training']['n_epochs']}")
        print(f"[INFO] Batch size: {cfg['training']['batch_size']}")

        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_out.write_bytes(b"STUB_CHECKPOINT_J28")
        print(f"[OK] Checkpoint -> {args.checkpoint_out}")
        return 0


    if __name__ == "__main__":
        sys.exit(main())
''')


MEDIUM_EVAL_PY = dedent('''\
    """scripts/eval.py — evaluate a checkpoint and write results.json.

    Usage: python scripts/eval.py --config configs/eval.yaml --checkpoint ckpt.pt \\
                                  --n_rollouts 50 --results_out eval/results.json
    """
    import argparse
    import json
    import sys
    import time
    from pathlib import Path


    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--n_rollouts", type=int, default=50)
        parser.add_argument("--results_out", type=Path, required=True)
        args = parser.parse_args()

        if not args.checkpoint.exists():
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 2

        # Stub eval : en prod on lance les rollouts via env + policy.
        # On produit un JSON conforme au contrat (4 cles requises).
        t0 = time.perf_counter()
        time.sleep(0.05)  # simulate work
        wall = time.perf_counter() - t0

        results = {
            "n_rollouts": args.n_rollouts,
            "success_rate_pct": 78.0,
            "mean_episode_length": 138.4,
            "mean_step_latency_ms": 36.4,
            "p95_step_latency_ms": 41.2,
            "wall_time_s": wall,
            "checkpoint": str(args.checkpoint),
        }
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[OK] Eval results -> {args.results_out}")
        print(f"     success_rate_pct={results['success_rate_pct']}")
        return 0


    if __name__ == "__main__":
        sys.exit(main())
''')


# =====================================================================
# HARD — Roadmap PushT -> AGV picking dans FleetSim
# =====================================================================
# Contraintes : 5 phases, chaque phase avec titre+duree, objectif, livrables,
# risque+mitigation, metrique. Mention explicite du schema event canonique
# LogiSim et des contraintes on-premise/determinisme/certification.
# Section "What we are NOT doing in v1" + Go/NoGo final.

HARD_ROADMAP_MD = dedent("""\
    # Roadmap — Diffusion Policy : du capstone PushT au picking AGV FleetSim

    > Audience : staff engineer + PO LogiSim. Document de discussion, pas un plan
    > definitif. Les durees sont des ordres de grandeur en semaines-ingenieur (1
    > ETP plein temps = 1 sem-ing par semaine calendaire).

    ## Phase 1 — Reproduction sur bras 6/7-DOF en simulation (3-4 sem-ing)

    **Objectif** : prouver que le pipeline du capstone (PushT 2D) fonctionne sur
    une tache 7-DOF en simulation, avant tout couplage metier.

    **Livrables** :
    - Setup robosuite v1.5 (REFERENCES.md #26) + Franka Panda via MuJoCo Menagerie.
    - Dataset de 500 demos sur la tache `lift` ou `square` (benchmark Diffusion
      Policy paper, REFERENCES.md #19) au format LeRobotDataset v0.4 (REFERENCES.md #27).
    - Re-entrainement Diffusion Policy avec action_dim = 7 + gripper.
    - Tableau success rate Diffusion Policy vs Behavior Cloning, 50 rollouts.

    **Risque principal** : la dynamique de contact 7-DOF est plus instable que
    PushT 2D. Mitigation : commencer par `lift` (tache la plus simple du
    benchmark), pas par `tool_hang` (la plus dure).

    **Metrique de succes** : success rate >= 70% sur 50 rollouts `lift` simules.
    Sinon -> revoir le pipeline avant Phase 2.

    ---

    ## Phase 2 — Adaptation LogiSim avec dataset interne (4-6 sem-ing)

    **Objectif** : passer d'un benchmark academique a une tache LogiSim cible avec
    un dataset interne issu d'operateurs reels.

    **Livrables** :
    - Setup teleoperation (joystick ou Quest VR) couple a un mock FleetSim.
    - Collecte de 300-500 demos sur 3-5 SKU canoniques (carton standard, sac
      souple, colis volumineux).
    - Conversion vers LeRobotDataset v0.4 (REFERENCES.md #27) avec respect du
      schema event canonique LogiSim (`shared/logistics-context.md`) : top-level
      stable, donnees specifiques dans `payload`.
    - Pipeline de quality control : labels operateurs, rejection demos echec.
    - Diffusion Policy entrainee, success rate par SKU.

    **Risque principal** : humain. Les operateurs LogiSim ne sont pas habitues a
    teleoperer un bras avec un objectif d'entrainement ML — ils peuvent injecter
    des biais (toujours commencer du meme cote, eviter certains angles). Mitigation :
    randomiser les conditions initiales + briefing operateurs + revue qualite par
    un ML engineer sur les 50 premieres demos.

    **Metrique de succes** : success rate >= 60% par SKU sur 30 rollouts
    simules. Si < 50% -> probleme de dataset, retourner collecter.

    ---

    ## Phase 3 — Conditioning textuel + safety filter (3-4 sem-ing)

    **Objectif** : rendre la policy capable de prendre un Work Order textuel et
    respecter la Safety Policy LogiSim.

    **Livrables** :
    - Tokenizer texte (CLIP ViT-B/32 en frozen) ajoute en entree de la policy.
      C'est le pattern OpenVLA (REFERENCES.md #13) et Octo (REFERENCES.md #17)
      adapte au cas Diffusion Policy.
    - Re-entrainement avec conditioning : input = (image, gripper_state, tcp_pose,
      work_order_tokens). Datasets augmentes avec 5 reformulations par Work Order.
    - **Safety filter** : module classique qui wrap la sortie. Calcule
      `Pcollision` (cf. vocabulaire LogiSim) sur la trajectoire predite. Si seuil
      depasse -> reject + emit event `FAULT` au schema canonique
      (`{kind: "FAULT", payload: {code: "POLICY_REJECT", severity: "minor"}}`).
    - Tracabilite : chaque action emise est loggee avec hash du checkpoint et
      seed (contrainte certification + determinisme, voir
      `shared/logistics-context.md`).

    **Risque principal** : technique. Le conditioning texte peut detruire les
    performances si mal integre (le modele "oublie" la modalite visuelle).
    Mitigation : ablation systematique (avec/sans texte), monitoring loss par
    modalite.

    **Metrique de succes** : success rate >= 60% sur Work Orders inedits
    (formulations jamais vues a l'entrainement) ET 0 violation Safety Policy
    sur 200 rollouts.

    ---

    ## Phase 4 — Optimisation latence + mode degrade (3-4 sem-ing)

    **Objectif** : atteindre la cible de latence pour deploiement temps reel et
    garantir un mode degrade non-bloquant.

    **Livrables** :
    - Migration DDPM 100 steps -> DDIM 10 steps (gain x10) puis exploration flow
      matching a la π0-FAST (REFERENCES.md #14, gain x5 supplementaire potentiel).
    - Benchmark latence : `mean_step_latency_ms` < 30 ms, p95 < 50 ms, sur
      hardware cible (CPU on-premise + iGPU client-grade).
    - **Mode degrade** : detecteur d'anomalies (action chunk avec norm > seuil,
      ou action discontinue) -> fallback automatique sur le planner classique
      RRT existant chez LogiSim (AutonomyAI SDK). Log de l'event `FAULT` +
      compteur OCC.
    - Quantization int8 du vision encoder + kernel fusion ONNX pour deploiement
      on-premise sans GPU dedie.

    **Risque principal** : technique. La quantization peut degrader le success
    rate de 5-15 points. Mitigation : QAT (quantization-aware training) si la
    perte naive est trop grande.

    **Metrique de succes** : `mean_step_latency_ms` < 30 ms ET success rate
    >= 55% (vs 60% Phase 3, on accepte une legere perte pour la latence) ET
    0 crash de la policy en 1000 rollouts (fallback classique testable).

    ---

    ## Phase 5 — Pilote sur 1 site client reel (6-10 sem-ing)

    **Objectif** : valider en conditions reelles, comparaison directe avec le
    baseline planner classique LogiSim.

    **Livrables** :
    - Packaging on-premise complet : binaire Python + ONNX runtime + checkpoint
      signe (PAS de cloud, contrainte `shared/logistics-context.md`).
    - Documentation operationnelle : runbook OCC, procedure de mise a jour
      checkpoint, procedure de revert.
    - Pilote 2 semaines sur 1 AGV de picking chez un client volontaire.
      500 picks reels minimum, monitoring OCC complet.
    - Comparaison cote-a-cote : Diffusion Policy (avec safety filter + fallback)
      vs planner classique RRT existant. Metriques : success rate, picks/h,
      events `COLLISION`, events `FAULT`, intervention operateur.
    - Retour formel des operateurs OCC (questionnaire structure + entretiens).
      Reference industrielle : Helix Logistics 2025 Figure AI (REFERENCES.md #16)
      a montre faster-than-demonstrator en logistique reelle, c'est notre etoile
      du nord.

    **Risque principal** : humain ET business. Acceptation par les operateurs
    OCC (peur du remplacement, defiance vis-a-vis du non-deterministe), et
    contrat client qui peut imposer un SLA sur le success rate. Mitigation :
    impliquer 1 operateur OCC senior comme co-pilote technique des Phase 2-5,
    contractualiser le pilote en mode best-effort avec le client.

    **Metrique de succes (Go/NoGo)** : sur 500 picks reels, success rate
    Diffusion Policy >= 95% du success rate planner classique ET temps moyen
    par pick <= 110% du baseline ET 0 incident safety.

    ---

    ## What we are NOT doing in v1

    Choix explicites de scope, pour eviter la derive :

    1. **Pas de multi-task generaliste** type OpenVLA / π0 (REFERENCES.md #13, #14)
       en v1. On reste sur 3-5 SKU canoniques. La generalisation cross-tache est
       un projet de Phase 6+.
    2. **Pas de cloud** : tout doit tourner on-premise (contrainte stricte
       LogiSim, voir `shared/logistics-context.md`).
    3. **Pas de zero-shot generalization** sur de nouveaux racks. Chaque nouveau
       layout = re-collecte mini-dataset + fine-tune.
    4. **Pas de collaboration humain-robot sans cage** en v1. La policy opere
       dans une zone protegee (cage virtuelle ou physique). La cohabitation
       directe est un projet Safety + IDS qui doit suivre Phase 5.
    5. **Pas de multi-embodiment** (un AGV picking != un bras fixe sur ligne de
       tri). Si on veut etendre, on relance le pipeline depuis Phase 2.
    6. **Pas de reentrainement continu en prod**. Le checkpoint est versionne et
       fige par release, pour respecter la tracabilite (certification client
       ISO 9001 / SOC 2, voir `shared/logistics-context.md`).

    ---

    ## Decision finale

    **Go / NoGo apres Phase 5** : si le success rate >= 95% du baseline classique
    ET 0 incident safety ET les operateurs OCC valident dans le questionnaire,
    on declenche un rollout sur 3 sites supplementaires en parallele avec
    monitoring renforce. Sinon, retour Phase 4 avec un budget cap supplementaire,
    pas de promesse client.

    Vision long-terme (12-24 mois) : convergence vers un VLA dual-system type
    Helix (REFERENCES.md #16) ou LBM TRI (REFERENCES.md #18), avec System 2
    pour la planification multi-pick et System 1 pour l'execution 200 Hz. Ce
    document s'arrete a la v1.
""")


# =====================================================================
# Main : ecrit les 3 corriges sur disque dans `out/`
# =====================================================================

def write_solutions(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Easy
    (out_dir / "easy_README.md").write_text(EASY_README_MD, encoding="utf-8")

    # Medium
    (out_dir / "medium_Makefile").write_text(MEDIUM_MAKEFILE, encoding="utf-8")
    (out_dir / "medium_configs_train.yaml").write_text(MEDIUM_TRAIN_YAML, encoding="utf-8")
    (out_dir / "medium_scripts_generate_demos.py").write_text(MEDIUM_GENERATE_DEMOS_PY, encoding="utf-8")
    (out_dir / "medium_scripts_train.py").write_text(MEDIUM_TRAIN_PY, encoding="utf-8")
    (out_dir / "medium_scripts_eval.py").write_text(MEDIUM_EVAL_PY, encoding="utf-8")

    # Hard
    (out_dir / "hard_roadmap.md").write_text(HARD_ROADMAP_MD, encoding="utf-8")

    print(f"[OK] Solutions ecrites dans {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"     - {p.name} ({p.stat().st_size} bytes)")


def main():
    out_dir = Path(__file__).parent / "out_j28_solutions"
    write_solutions(out_dir)


if __name__ == "__main__":
    main()

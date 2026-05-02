# Plan figé domaine `robotics-ai` (28 jours)

> Contrat Phase 3 → Phase 4. Source de vérité pour les subagents qui produiront chaque module-jour.
> **Convention** : 1 subagent = 1 jour. Chaque subagent ne lit QUE ce fichier + REFERENCES.md, jamais les autres jours (qui n'existent pas encore).

---

## J1 — Vue d'ensemble robotique moderne + setup stack

- **Concepts clés** :
  - Pile complète robotique : capteurs → perception → planning → contrôle → actuation
  - Deux ères : classique (Khatib/Tedrake/Abbeel) vs IA générative (Finn/Levine/Chi)
  - Pourquoi MuJoCo+Gymnasium+PyTorch est le stack 2026 dominant
  - Premier env MuJoCo qui tourne (HalfCheetah-v4 ou Pendulum)
  - Anatomie d'un step Gymnasium (`reset`, `step`, observation, reward, terminated, truncated)
- **Acquis fin de jour** : MuJoCo+Gymnasium installés, premier env qui tourne avec policy random, vision globale du domaine.
- **Sources autorisées** :
  - REFERENCES.md #24 (MuJoCo Documentation)
  - REFERENCES.md #25 (Gymnasium)
  - REFERENCES.md #5 (CS223A Khatib L1)
- **Stack** : Python, MuJoCo 3.x, Gymnasium, PyTorch
- **Slug** : `01-vue-ensemble-setup`

## J2 — Transformations 3D : SE(3), rotations, twists

- **Concepts clés** :
  - Rotations : matrices, axe-angle, quaternions
  - Translations + rotations = SE(3), matrices homogènes 4x4
  - Composition de transformations (chaîne cinématique)
  - Twists et screws (representation dual)
  - Pourquoi PoE (Lynch) est plus propre que DH
- **Acquis fin de jour** : manipuler SE(3) en numpy, convertir entre représentations, composer des poses.
- **Sources autorisées** :
  - REFERENCES.md #1 (Lynch & Park ch. 3)
  - REFERENCES.md #5 (CS223A L2-3)
- **Stack** : numpy, scipy.spatial.transform
- **Slug** : `02-transformations-3d`

## J3 — Cinématique directe (Forward Kinematics)

- **Concepts clés** :
  - Définition FK : configuration articulaire → pose end-effector
  - Représentation Denavit-Hartenberg (classique et modifiée)
  - Product of Exponentials (Lynch) — formulation moderne
  - Implémenter FK pour bras 2-DOF puis 6-DOF (Franka Panda via Menagerie)
- **Acquis fin de jour** : coder FK from scratch en numpy, vérifier vs MuJoCo `mj_forward`.
- **Sources autorisées** :
  - REFERENCES.md #1 (Lynch & Park ch. 4)
  - REFERENCES.md #5 (CS223A L4)
  - REFERENCES.md #28 (MuJoCo Menagerie pour Panda MJCF)
- **Stack** : numpy, mujoco
- **Slug** : `03-cinematique-directe`

## J4 — Cinématique inverse + Jacobiens

- **Concepts clés** :
  - IK analytique (closed-form) vs numérique (Newton-Raphson, damped least squares)
  - Singularités, redondance (n-DOF > 6)
  - Jacobien géométrique vs analytique
  - Vélocités end-effector ↔ vélocités articulaires
  - Forces statiques (transposée du Jacobien)
- **Acquis fin de jour** : résoudre IK numériquement pour Franka, calculer Jacobien, gérer singularité.
- **Sources autorisées** :
  - REFERENCES.md #1 (Lynch & Park ch. 5-6)
  - REFERENCES.md #5 (CS223A L6-8)
- **Stack** : numpy, scipy.optimize
- **Slug** : `04-cinematique-inverse-jacobiens`

## J5 — Dynamique + simulation MuJoCo hands-on

- **Concepts clés** :
  - Newton-Euler récursif (forward sweep + backward sweep)
  - Formulation Lagrange : M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
  - Inertie, masse, frottements, contacts
  - Charger un robot MuJoCo Menagerie, simuler avec contacts, lire qpos/qvel
  - Pas de temps, intégration semi-implicit Euler
- **Acquis fin de jour** : robot Franka qui tombe sous gravité dans MuJoCo, calculer son énergie.
- **Sources autorisées** :
  - REFERENCES.md #1 (Lynch & Park ch. 8)
  - REFERENCES.md #2 (Siciliano ch. 7)
  - REFERENCES.md #24 (MuJoCo docs : XML, contacts)
- **Stack** : mujoco, numpy
- **Slug** : `05-dynamique-simulation`

## J6 — Contrôle classique : PID, computed torque, LQR

- **Concepts clés** :
  - PID articulaire (gains, anti-windup, dérivée du measurement)
  - Computed torque : feedforward dynamique + feedback
  - Operational space control (Khatib)
  - LQR : formulation Q/R, équation de Riccati, finite-horizon vs infinite-horizon
  - LQR linéarisé autour d'un équilibre (pendule inversé)
- **Acquis fin de jour** : stabiliser pendule inversé avec LQR, tracker trajectoire avec computed torque.
- **Sources autorisées** :
  - REFERENCES.md #2 (Siciliano ch. 8 — control)
  - REFERENCES.md #3 (Tedrake ch. 7-8 — LQR, Lyapunov)
- **Stack** : numpy, scipy, mujoco
- **Slug** : `06-controle-classique`

## J7 — Perception 3D pour robotique

- **Concepts clés** :
  - Caméras pinhole, intrinsèques/extrinsèques, calibration
  - Depth sensors (stereo, ToF, structured light)
  - Point clouds : représentation, voxelisation, ICP
  - NeRF / 3D Gaussian Splatting — état de l'art reconstruction 3D
  - Du point cloud à la pose d'objet (PoseCNN, FoundationPose)
- **Acquis fin de jour** : générer point cloud depuis MuJoCo render, faire ICP basique entre 2 nuages.
- **Sources autorisées** :
  - REFERENCES.md #24 (MuJoCo render)
  - Open3D docs (open3d.org) — outil canonique perception 3D Python
  - V-JEPA 2 paper (REFERENCES.md #21) pour vision moderne
- **Stack** : open3d, numpy, mujoco
- **Slug** : `07-perception-3d-robotique`

## J8 — Motion planning : RRT, PRM, trajectory optimization

- **Concepts clés** :
  - Configuration space, obstacles dans C-space
  - RRT (Rapidly-exploring Random Tree), RRT*, RRT-Connect
  - PRM (Probabilistic Roadmap)
  - Trajectory optimization : direct collocation, shooting
  - Combinaison planning + control (RRT pour trajectoire, MPC pour suivi)
- **Acquis fin de jour** : implémenter RRT 2D, planifier dans C-space d'un bras 2-DOF avec obstacles.
- **Sources autorisées** :
  - REFERENCES.md #4 (LaValle ch. 5)
  - REFERENCES.md #3 (Tedrake ch. 10 — trajopt)
- **Stack** : numpy, matplotlib
- **Slug** : `08-motion-planning`

## J9 — MDPs, Bellman, value/policy iteration

- **Concepts clés** :
  - Markov Decision Process : (S, A, P, R, γ)
  - Value function V(s), Q-function Q(s,a)
  - Équation de Bellman (expected/optimality)
  - Value iteration, policy iteration, convergence
  - GridWorld pédagogique
- **Acquis fin de jour** : résoudre GridWorld par VI/PI from scratch.
- **Sources autorisées** :
  - REFERENCES.md #6 (Sutton & Barto ch. 3-4)
  - REFERENCES.md #11 (CS285 L4)
- **Stack** : numpy
- **Slug** : `09-mdp-fondations`

## J10 — Q-learning, DQN

- **Concepts clés** :
  - TD-learning : TD(0), n-step TD
  - Q-learning : off-policy, target update
  - DQN : replay buffer, target network, ε-greedy (Mnih 2015)
  - Double DQN, Dueling DQN
  - Discret vs continu : pourquoi DQN ne marche pas sur MuJoCo nativement
- **Acquis fin de jour** : implémenter Q-learning tabulaire sur GridWorld, DQN sur CartPole.
- **Sources autorisées** :
  - REFERENCES.md #6 (S&B ch. 6)
  - REFERENCES.md #11 (CS285 L7-8)
  - REFERENCES.md #9 (CleanRL DQN single-file)
- **Stack** : torch, gymnasium
- **Slug** : `10-q-learning-dqn`

## J11 — Policy gradients, PPO sur MuJoCo

- **Concepts clés** :
  - REINFORCE (vanilla policy gradient)
  - Advantage estimation (GAE), baseline, A2C
  - PPO clip objective (Schulman 2017)
  - PPO sur MuJoCo HalfCheetah/Ant : config standard
  - Stable-Baselines3 vs CleanRL — quand utiliser quoi
- **Acquis fin de jour** : entraîner PPO sur CartPole avec CleanRL en moins d'1 minute, lancer PPO sur HalfCheetah.
- **Sources autorisées** :
  - REFERENCES.md #8 (PPO Schulman 2017)
  - REFERENCES.md #7 (Spinning Up VPG/PPO)
  - REFERENCES.md #9 (CleanRL ppo_continuous_action.py)
- **Stack** : torch, gymnasium[mujoco]
- **Slug** : `11-policy-gradients-ppo`

## J12 — SAC, TD3, MPC, model-based RL

- **Concepts clés** :
  - SAC (max-entropy RL, Haarnoja 2018) vs PPO : sample efficiency
  - TD3 : twin Q, delayed updates, target smoothing
  - MPC : optimisation horizon glissant, CEM, MPPI
  - Model-based RL : Dyna, MBPO, world model + planner
  - Quand utiliser model-based vs model-free
- **Acquis fin de jour** : SAC sur HalfCheetah, MPC simple sur pendule.
- **Sources autorisées** :
  - REFERENCES.md #11 (CS285 L11-12)
  - REFERENCES.md #7 (Spinning Up SAC)
  - REFERENCES.md #3 (Tedrake MPC)
- **Stack** : torch, stable-baselines3, gymnasium
- **Slug** : `12-sac-mpc-model-based`

## J13 — Imitation Learning : BC, DAgger, IRL/GAIL

- **Concepts clés** :
  - Behavior Cloning : supervised learning sur (s, a)
  - Distribution shift (le problème central de BC)
  - DAgger : on-policy correction de BC
  - IRL : inverse RL, MaxEnt IRL
  - GAIL : adversarial IL
  - Quand IL > RL : quand on a des démos mais pas de reward
- **Acquis fin de jour** : BC + DAgger sur CartPole avec démos d'expert PPO entraîné.
- **Sources autorisées** :
  - REFERENCES.md #10 (Survey IL Zare 2024)
  - REFERENCES.md #11 (CS285 L2)
  - REFERENCES.md #12 (CS224R L2)
- **Stack** : torch, gymnasium
- **Slug** : `13-imitation-learning`

## J14 — Sim-to-real : domain randomization

- **Concepts clés** :
  - Reality gap : pourquoi une policy entraînée en sim échoue en réel
  - Domain randomization (Tobin 2017) : visuel
  - Dynamics randomization : mass, friction, latence, capteur noise
  - System identification + adaptation
  - Why simu seul ne suffit pas (mais on est full-simu pour ce cours)
- **Acquis fin de jour** : entraîner PPO avec randomization MuJoCo et tester robustesse.
- **Sources autorisées** :
  - REFERENCES.md #11 (CS285 L13)
  - REFERENCES.md #8 (PPO comme baseline)
  - Tobin et al. 2017 (arxiv 1703.06907)
- **Stack** : torch, gymnasium, mujoco
- **Slug** : `14-sim-to-real`

## J15 — Diffusion + flow matching unifié

- **Concepts clés** :
  - Score matching : ∇log p(x), denoising score matching
  - DDPM (Ho 2020) : forward noising + reverse denoising
  - Flow matching (Lipman 2022) : ODE déterministe vs SDE stochastique
  - Vue unifiée : DDPM = cas particulier de flow matching avec schedule cosinus
  - Classifier-free guidance (Ho & Salimans 2022)
- **Acquis fin de jour** : entraîner DDPM jouet sur MNIST 2D, échantillonner.
- **Sources autorisées** :
  - REFERENCES.md #23 (MIT 6.S184 lectures + notes)
  - DDPM Ho 2020 (arxiv 2006.11239)
- **Stack** : torch
- **Slug** : `15-diffusion-flow-matching`

## J16 — Diffusion Policy (Chi 2023) deep dive

- **Concepts clés** :
  - Action sequences (chunking) vs action unique
  - Receding horizon : exécuter k actions, replanifier
  - Conditioning visuel : encoder ResNet18 ou ViT
  - UNet 1D vs Transformer pour denoiser
  - Pourquoi diffusion > BC sur multimodal action distributions
- **Acquis fin de jour** : lire et comprendre le repo `real-stanford/diffusion_policy`, identifier chaque module.
- **Sources autorisées** :
  - REFERENCES.md #19 (Diffusion Policy paper + repo)
  - REFERENCES.md #23 (MIT 6.S184 background)
- **Stack** : torch, repo diffusion_policy (cloner pour lecture)
- **Slug** : `16-diffusion-policy`

## J17 — World models : Dreamer V1→V3

- **Concepts clés** :
  - World model = représentation latente + transition + reward
  - RSSM (Recurrent State-Space Model)
  - Latent imagination : entraîner actor-critic dans imagination du modèle
  - Dreamer V3 (Hafner 2023) : single config qui marche sur 150+ tâches
  - Pros/cons vs model-free RL
- **Acquis fin de jour** : comprendre l'architecture RSSM, faire tourner DreamerV3 sur Atari simple.
- **Sources autorisées** :
  - REFERENCES.md #20 (DreamerV3 Hafner 2023)
  - REFERENCES.md #11 (CS285 L11 model-based)
- **Stack** : torch (notebook conceptuel + repo officiel pour exécution si GPU)
- **Slug** : `17-world-models-dreamer`

## J18 — JEPA + NVIDIA Cosmos

- **Concepts clés** :
  - JEPA (LeCun) : prédiction dans l'espace latent vs pixels
  - I-JEPA, V-JEPA, V-JEPA 2 — applications robotique
  - Pourquoi LeCun pense que generative pixel-level est une impasse
  - NVIDIA Cosmos : foundation models physical AI, tokenizers vidéo, 20M h training
  - Comparaison Dreamer vs JEPA vs Cosmos
- **Acquis fin de jour** : comprendre les 3 paradigmes et savoir quand chaque approche fait sens.
- **Sources autorisées** :
  - REFERENCES.md #21 (V-JEPA 2 Meta 2025)
  - REFERENCES.md #22 (NVIDIA Cosmos 2025)
- **Stack** : conceptuel + V-JEPA 2 inference si GPU dispo
- **Slug** : `18-jepa-cosmos`

## J19 — VLA introduction : RT-1/RT-2, Open X, Octo

- **Concepts clés** :
  - VLA = Vision-Language-Action
  - RT-1 (2022) : transformer policy + tokens action discrets
  - RT-2 (2023) : co-fine-tuning VLM + actions, generalization
  - Open X-Embodiment dataset (2024) : standardisation
  - Octo (RSS 2024) : transformer policy open-source sur 800k trajectoires
  - Action tokenization vs action regression
- **Acquis fin de jour** : comprendre le tokenizer d'action, savoir lire un dataset Open X.
- **Sources autorisées** :
  - REFERENCES.md #17 (Octo RSS 2024)
  - REFERENCES.md #13 (OpenVLA — pour le contexte RT-2)
- **Stack** : torch, transformers, datasets
- **Slug** : `19-vla-introduction`

## J20 — OpenVLA architecture + fine-tuning LoRA

- **Concepts clés** :
  - OpenVLA architecture : Llama2 7B + DINOv2 + SigLIP + action head
  - Pourquoi 7B bat RT-2-X 55B
  - Fine-tuning LoRA sur tâche custom
  - Quantization 4-bit pour déploiement consumer GPU
  - Limitations connues (single-arm, peu de dynamique)
- **Acquis fin de jour** : architecture OpenVLA claire, savoir où on insérerait LoRA.
- **Sources autorisées** :
  - REFERENCES.md #13 (OpenVLA paper + repo)
- **Stack** : torch, transformers, peft (LoRA)
- **Slug** : `20-openvla-architecture`

## J21 — π0 / π0.5 (Physical Intelligence)

- **Concepts clés** :
  - π0 (Black 2024) : VLM pré-entraîné + flow matching action head
  - Multi-embodiment : un seul modèle, plusieurs robots
  - π0.5 : open-world generalization (kitchens/bedrooms inédits)
  - π0-FAST tokenizer : 5x plus rapide
  - Comparaison π0 vs OpenVLA vs Octo
- **Acquis fin de jour** : comprendre flow matching action head, contraste avec diffusion DDPM (J15).
- **Sources autorisées** :
  - REFERENCES.md #14 (π0 + π0.5 + Pi blog)
- **Stack** : conceptuel + lerobot inference si GPU
- **Slug** : `21-pi0-physical-intelligence`

## J22 — Frontier humanoid : GR00T N1, Helix, LBM TRI

- **Concepts clés** :
  - GR00T N1 (NVIDIA 2025) : dual-system System1/System2, synthetic data
  - Helix (Figure 2025) : 35 DoF @ 200Hz, multi-robot, déploiement logistique réel
  - LBM TRI : 1700h données, 80% moins de data sur nouvelles tâches
  - System1 (fast/reactive) + System2 (slow/reasoning) — Kahneman appliqué robots
  - Convergence industry 2025 sur ce pattern
- **Acquis fin de jour** : carte mentale du paysage VLA frontier 2025-2026.
- **Sources autorisées** :
  - REFERENCES.md #15 (GR00T N1 NVIDIA)
  - REFERENCES.md #16 (Helix Figure)
  - REFERENCES.md #18 (TRI LBM)
- **Stack** : conceptuel
- **Slug** : `22-frontier-humanoid`

## J23 — Synthetic data + sim-to-real à scale

- **Concepts clés** :
  - Synthetic data : pourquoi 780k trajectoires GR00T générées en sim
  - Pipeline NVIDIA Cosmos pour curation video data
  - Isaac Lab pour génération massive
  - Augmentation : background, lighting, distractors
  - Open X-Embodiment : standard datasets multi-robots
  - Limitations sim-to-real à scale (encore le reality gap résiduel)
- **Acquis fin de jour** : comprendre comment on assemble un dataset pour entraîner un VLA généraliste.
- **Sources autorisées** :
  - REFERENCES.md #22 (NVIDIA Cosmos)
  - REFERENCES.md #15 (GR00T data pipeline)
  - REFERENCES.md #27 (LeRobot dataset format)
- **Stack** : conceptuel + datasets HuggingFace
- **Slug** : `23-synthetic-data-scale`

## J24 — Capstone setup : MuJoCo PushT + démos + LeRobotDataset

- **Concepts clés** :
  - PushT environment : tâche canonique benchmark Diffusion Policy
  - Génération démos : keyboard téléopération + script automatique
  - LeRobotDataset format : Parquet + MP4, streamable
  - Augmentation données (image crop, color jitter)
  - Train/val split stratégie
- **Acquis fin de jour** : 100-200 démos PushT générées, dataset LeRobot prêt à charger.
- **Sources autorisées** :
  - REFERENCES.md #19 (Diffusion Policy repo — environment)
  - REFERENCES.md #27 (LeRobot dataset)
  - REFERENCES.md #24 (MuJoCo)
- **Stack** : mujoco, gymnasium, lerobot, numpy
- **Slug** : `24-capstone-setup`

## J25 — Capstone architecture : ResNet18 + UNet 1D + DDPM

- **Concepts clés** :
  - Vision encoder : ResNet18 from torchvision (frozen ou fine-tuned)
  - Goal/state conditioning : embed pose actuelle
  - UNet 1D pour denoising sur séquences d'actions
  - DDPM scheduler : forward/reverse process, beta schedule
  - Action chunking (k=8 ou 16 typique)
- **Acquis fin de jour** : architecture complète Diffusion Policy implémentée (sans entraînement).
- **Sources autorisées** :
  - REFERENCES.md #19 (Diffusion Policy repo — model code)
  - REFERENCES.md #23 (MIT 6.S184 pour DDPM)
- **Stack** : torch, torchvision, einops
- **Slug** : `25-capstone-architecture`

## J26 — Capstone training loop + hyperparams

- **Concepts clés** :
  - Training loop : `loss = MSE(predicted_noise, true_noise)`
  - EMA (Exponential Moving Average) sur poids
  - Optimizer AdamW, learning rate schedule
  - Batch size, gradient accumulation
  - Wandb logging, sample reconstruction visuels
  - Mixed precision (fp16) pour GPU consumer
- **Acquis fin de jour** : training tourne, loss descend, premières actions générées sample.
- **Sources autorisées** :
  - REFERENCES.md #19 (Diffusion Policy — training config)
- **Stack** : torch, wandb (optionnel), accelerate
- **Slug** : `26-capstone-training`

## J27 — Capstone eval + ablations + baseline BC

- **Concepts clés** :
  - Évaluation : success rate sur N rollouts (typiquement 50-100)
  - Métriques : success rate, episode length, action smoothness
  - Baseline : Behavior Cloning simple — comparaison directe
  - Ablations : sans action chunking, sans EMA, schedule alternative
  - Receding horizon en eval : exécuter Tα steps puis replanifier
  - Latence : combien de ms par step (compatible 10Hz contrôle ?)
- **Acquis fin de jour** : tableau résultats avec success rate Diffusion Policy vs BC.
- **Sources autorisées** :
  - REFERENCES.md #19 (Diffusion Policy paper §6)
- **Stack** : torch, mujoco, matplotlib (plots)
- **Slug** : `27-capstone-eval-ablations`

## J28 — Capstone packaging + intégration LogiSim + retrospective

- **Concepts clés** :
  - Packaging : checkpoint final, README démo, Makefile/scripts
  - Démo finale : script `python demo.py` lance eval visuelle
  - Analogie LogiSim : un robot picking dans un entrepôt FleetSim — comment Diffusion Policy s'y plugge
  - Roadmap : étapes pour passer de PushT → tâches manipulation réelles → VLA
  - Retrospective : ce qu'on a appris, où sont les frontières, suite logique (CMU 16-831, UPenn MEAM 5200, lecture de papers récents)
- **Acquis fin de jour** : projet shippable avec README portfolio-grade, retro écrite.
- **Sources autorisées** :
  - REFERENCES.md #16 (Helix Logistics — analogie LogiSim)
  - REFERENCES.md #18 (TRI LBM — vision long-terme)
  - `shared/logistics-context.md`
- **Stack** : tous packagés (mujoco, torch, lerobot)
- **Slug** : `28-capstone-packaging-logisim`

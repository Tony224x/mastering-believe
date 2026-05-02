# Robotics & AI — robotique moderne à l'ère de l'IA générative

## Scope

Maîtriser la robotique moderne au croisement des fondations classiques (cinématique, dynamique, contrôle, planning) et des approches IA générative qui dominent la recherche et l'industrie en 2025-2026 :

- **Fondations classiques** : SE(3), forward/inverse kinematics, jacobiens, dynamique (Newton-Euler / Lagrange), contrôle (PID, computed torque, LQR), motion planning (RRT, trajectory optimization), perception 3D.
- **Deep RL & Imitation Learning** : MDPs, policy gradients, PPO/SAC, model-based RL, behavior cloning, DAgger, sim-to-real.
- **Foundation Models pour robots** : VLA (RT-2, OpenVLA, π0/π0.5, GR00T, Helix), world models (Dreamer V3, V-JEPA 2, NVIDIA Cosmos), Large Behavior Models (TRI).
- **Diffusion Policies** : score matching, flow matching, Diffusion Policy (Chi 2023) — capstone implémenté from scratch.

**Hors scope** : ROS2 (focus simulation pure), hardware réel (full-simu), SLAM en profondeur (mentionné J7 perception, pas un cours dédié).

## Prerequisites

- Python solide (`numpy`, classes, type hints)
- PyTorch — savoir entraîner un réseau dense / convolutif (équivalent `domains/neural-networks-llm/` J1-J10)
- Algèbre linéaire (matrices, valeurs propres) et probabilités de base
- Pas de background robotique requis — le domaine bootstrap les fondations classiques en J2-J7

## Planning (4 semaines, 28 jours)

| Sem | Jour | Module | Temps estimé |
|-----|------|--------|--------------|
| 1 | J1 | Vue d'ensemble + setup MuJoCo/Gymnasium/PyTorch | 2-3h |
| 1 | J2 | Transformations 3D : SE(3), rotations, twists | 2-3h |
| 1 | J3 | Cinématique directe (PoE) | 2-3h |
| 1 | J4 | Cinématique inverse + Jacobiens | 3h |
| 1 | J5 | Dynamique + simulation MuJoCo hands-on | 3h |
| 1 | J6 | Contrôle classique : PID, computed torque, LQR | 2-3h |
| 1 | J7 | Perception 3D pour robotique : RGB-D, point clouds, calibration | 2-3h |
| 2 | J8 | Motion planning : RRT/RRT*, PRM, trajectory optimization | 3h |
| 2 | J9 | MDP, Bellman, value/policy iteration | 2h |
| 2 | J10 | Q-learning, DQN | 2-3h |
| 2 | J11 | Policy gradients : REINFORCE, A2C, PPO sur MuJoCo | 3h |
| 2 | J12 | SAC, TD3, MPC, model-based RL (Dyna, MBPO) | 3h |
| 2 | J13 | Imitation Learning : BC, DAgger, IRL/GAIL | 2-3h |
| 2 | J14 | Sim-to-real : domain randomization | 2-3h |
| 3 | J15 | Diffusion + flow matching unifié | 3h |
| 3 | J16 | Diffusion Policy (Chi 2023) deep dive | 3h |
| 3 | J17 | World models : Dreamer V1→V3, RSSM | 2-3h |
| 3 | J18 | JEPA + NVIDIA Cosmos — prédiction latente vs pixels | 2-3h |
| 3 | J19 | VLA intro : RT-1/RT-2, Open X-Embodiment, Octo | 2-3h |
| 3 | J20 | OpenVLA architecture + fine-tuning LoRA | 3h |
| 3 | J21 | π0 / π0.5 (Physical Intelligence) | 2-3h |
| 4 | J22 | Frontier humanoid : GR00T N1 + Helix + LBM TRI | 2-3h |
| 4 | J23 | Synthetic data + sim-to-real à scale | 2-3h |
| 4 | J24 | Capstone setup : MuJoCo PushT + démos + LeRobotDataset | 3h |
| 4 | J25 | Capstone architecture : ResNet18 + UNet 1D + DDPM | 3-4h |
| 4 | J26 | Capstone training loop + hyperparams | 3-4h |
| 4 | J27 | Capstone eval + ablations + baseline BC | 3h |
| 4 | J28 | Capstone packaging + intégration LogiSim + retrospective | 2-3h |

Total : ~75-85h sur 28 jours, soit ~3h/jour en moyenne.

## Criteres de reussite

À la fin du domaine, l'apprenant doit pouvoir :

- [ ] **Lire un paper VLA récent** (π0.5, GR00T N1, Helix, OpenVLA) et identifier l'architecture, les choix de loss, le protocole d'entraînement, les benchmarks.
- [ ] **Implémenter Diffusion Policy from scratch** sur MuJoCo PushT : générer démos, entraîner UNet 1D + DDPM, évaluer success rate, comparer à baseline BC.
- [ ] **Setuper un environnement MuJoCo** avec un robot de Menagerie, le piloter en PID/computed torque, le pousser via PPO sur Gymnasium.
- [ ] **Expliquer la différence** entre policy direct (BC, PPO), diffusion policy, world model + planner, VLA dual-system — et savoir laquelle choisir selon le problème.
- [ ] **Comprendre les compromis sim-to-real** : domain randomization, dynamics randomization, observation noise, et savoir quel niveau utiliser selon la cible.
- [ ] **Discuter avec un roboticien senior** sans paniquer sur les fondations (FK/IK/Jacobiens/LQR) ni sur les approches modernes (diffusion, flow matching, JEPA).

## Ressources externes

Liste exhaustive : voir [`REFERENCES.md`](./REFERENCES.md) (29 sources tier-1 réparties sur 5 axes).

Top 5 si tu ne devais en garder que 5 :
- Lynch & Park, *Modern Robotics* (2017) — fondations
- Sutton & Barto, *Reinforcement Learning* 2e (2018) — RL
- Tedrake, *Underactuated Robotics* (live MIT 6.832) — contrôle moderne
- Chi et al., *Diffusion Policy* (RSS 2023) — capstone
- π0/π0.5 (Physical Intelligence 2024-2025) — frontier VLA

## Projets guides

3 projets en contexte LogiSim/FleetSim (entrepôts robotisés, AGV, picking automatisé) dans [`05-projets-guides/`](./05-projets-guides/) — voir aussi [`shared/logistics-context.md`](../../shared/logistics-context.md).

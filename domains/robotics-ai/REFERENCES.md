# REFERENCES — `robotics-ai` (28 jours)

> Sources de tier-1 retenues à l'issue de la Phase 1 du skill `mastering-domain-creator`.
> Source de vérité pour les modules J1..J28. Toutes les URLs sont vérifiées WebFetch/WebSearch (mai 2026).
> Wiki interne (`references/wiki/`) : `cs223a-intro-robotics`, `cs287-advanced-robotics`, `cs285-deep-rl`, `cs224r-deep-rl`, `6.4210-robotic-manipulation`, `6.832-underactuated-robotics`, topic `robotics`.

---

## Axe 1 — Fondations robotique classique

1. **Modern Robotics: Mechanics, Planning, and Control** — Lynch & Park, 2017 (Cambridge UP). http://hades.mech.northwestern.edu/index.php/Modern_Robotics
   Pourquoi : manuel canonique post-2015, formulation produit-d'exponentielles (PoE) plus propre que DH classique, livre + cours edX + code Python officiel.
   Couvre : J1 (transformations SE(3), twists), J2 (FK via PoE), J3 (jacobiens, IK), J4 (dynamique Lagrange/Newton-Euler).

2. **Robotics: Modelling, Planning and Control** — Siciliano, Sciavicco, Villani, Oriolo, 2009 (Springer). https://link.springer.com/book/10.1007/978-1-84628-642-1
   Pourquoi : standard de fait européen, traitement rigoureux de la dynamique et du contrôle articulaire (computed torque, impedance), complémentaire à Lynch sur le contrôle.
   Couvre : J4 (Newton-Euler récursif), J5 (PID articulaire, computed torque, impedance/force control).

3. **Underactuated Robotics** — Russ Tedrake, MIT 6.832, édition vivante 2024. https://underactuated.csail.mit.edu/
   Pourquoi : référence ouverte sur LQR, trajectory optimization (DIRCOL, iLQR) et MPC pour systèmes non-linéaires, avec notebooks Drake exécutables.
   Couvre : J5 (LQR, MPC), J6 (trajectory optimization, direct collocation).

4. **Planning Algorithms** — Steven LaValle, 2006 (Cambridge UP), édition libre en ligne. http://lavalle.pl/planning/
   Pourquoi : référence fondatrice et toujours canonique pour RRT/PRM, configuration space, sampling-based planning. L'auteur a inventé RRT.
   Couvre : J6 (motion planning, RRT/RRT*, PRM, C-space).

5. **Stanford CS223A — Introduction to Robotics** — Oussama Khatib, lectures Stanford 2008. https://www.youtube.com/playlist?list=PL65CC0384A1798ADF
   Pourquoi : cours de référence sur les fondations cinématique/dynamique par l'auteur du operational space control ; complément vidéo aux livres.
   Couvre : J1-J4 (rotations, FK/IK, jacobiens, dynamique).

---

## Axe 2 — Deep RL + Imitation Learning

6. **Reinforcement Learning: An Introduction (2nd ed.)** — Sutton & Barto, 2018. http://incompleteideas.net/book/the-book-2nd.html
   Pourquoi : la bible canonique du RL, gratuite, indispensable pour MDPs / Bellman / TD / policy gradients.
   Couvre : J7 (MDPs, Bellman), J8 (TD, Q-learning), J9 (policy gradients fondamentaux).

7. **OpenAI Spinning Up in Deep RL** — Achiam (OpenAI), 2018+. https://spinningup.openai.com/en/latest/
   Pourquoi : pont pédagogique entre Sutton & Barto et les papiers (VPG/TRPO/PPO/DDPG/TD3/SAC) avec implémentations de référence.
   Couvre : J9 (PPO, TRPO), J10 (SAC, TD3, DDPG), J11 (actor-critic, exploration).

8. **Proximal Policy Optimization Algorithms** — Schulman et al., 2017. https://arxiv.org/abs/1707.06347
   Pourquoi : papier séminal du workhorse du Deep RL appliqué à la robotique (PPO clip), prérequis direct des labs sim-to-real.
   Couvre : J9 (policy gradients on-policy), J13 (sim-to-real), J14 (RLHF/RL fine-tuning des VLAs).

9. **CleanRL** — Huang et al., 2022+. https://github.com/vwxyzjn/cleanrl
   Pourquoi : implémentations single-file lisibles ligne-à-ligne, benchmarkées (PPO/SAC/DQN/TD3/DDPG/RND).
   Couvre : J9-J11 (lecture/modif d'algos), J12 (offline RL), capstone.

10. **A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges** — Zare, Kebria, Khosravi, Nahavandi, 2024. https://arxiv.org/abs/2309.02473
    Pourquoi : survey récent et exhaustif couvrant Behavior Cloning, DAgger, IRL, GAIL, et leur combinaison avec RL.
    Couvre : J11 (BC, DAgger), J12 (offline + IL), J13 (sim-to-real combiné IL+RL).

11. **Berkeley CS285 — Deep Reinforcement Learning** — Sergey Levine, Fall 2023. https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps
    Pourquoi : référence algorithmique deep RL la plus rigoureuse côté granularité (PG, AC, Q, MBRL, offline RL, IRL, RL & LLMs).
    Couvre : J7-J14 (toute la séquence RL).

12. **Stanford CS224R — Deep RL** — Chelsea Finn, Spring 2025. https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL
    Pourquoi : pendant Stanford robotique-first, lectures sur diffusion policy, sim-to-real, VLA.
    Couvre : J11-J16 (IL moderne, sim2real, VLA preview).

---

## Axe 3 — Vision-Language-Action & Foundation Models pour robots

13. **OpenVLA: An Open-Source Vision-Language-Action Model** — Kim, Pertsch et al. (Stanford / Berkeley / TRI), 2024. https://arxiv.org/abs/2406.09246 — Repo : https://github.com/openvla/openvla
    Pourquoi : VLA 7B open-source de référence (Llama2 + DINOv2/SigLIP), bat RT-2-X 55B, fine-tunable sur GPU consumer — pierre angulaire pédagogique.
    Couvre : J18 (RT-1/RT-2 lineage), J19 (architecture VLA), J20 (fine-tuning LoRA + déploiement quantizé).

14. **π0: A Vision-Language-Action Flow Model for General Robot Control** — Black, Brown, Driess, Finn et al. (Physical Intelligence), 2024. https://arxiv.org/abs/2410.24164 — Blog : https://www.pi.website/blog/pi0 — π0.5 : https://www.pi.website/download/pi05.pdf
    Pourquoi : SOTA generalist policy avec flow matching sur VLM pré-entraîné, multi-embodiment dexterous ; π0.5 ajoute open-world generalization ; π0-FAST tokenizer 5×.
    Couvre : J20 (flow matching/diffusion action heads), J21 (π0/π0.5 deep dive), J24 (open-world generalization).

15. **GR00T N1: An Open Foundation Model for Generalist Humanoid Robots** — NVIDIA Research, mars 2025. https://arxiv.org/abs/2503.14734 — Repo : https://github.com/NVIDIA/Isaac-GR00T
    Pourquoi : VLA dual-system (System2 VLM + System1 diffusion transformer) sur mix réel + 780k trajectoires synthétiques ; modèle ouvert, écosystème Isaac/Newton, adopté par Figure/1X/Apptronik.
    Couvre : J22 (architectures dual-system System1/System2), J23 (synthetic data pipelines + sim-to-real), J24 (humanoid deployment).

16. **Helix: A Vision-Language-Action Model for Generalist Humanoid Control** — Figure AI, février 2025 + Helix Logistics 2025. https://www.figure.ai/news/helix — https://www.figure.ai/news/helix-logistics
    Pourquoi : premier VLA driving humanoid upper-body complet (35 DoF @ 200Hz) avec System2 7B + System1 80M ; faster-than-demonstrator en logistique réelle — résonance directe contexte LogiSim.
    Couvre : J22 (high-rate continuous control), J23 (multi-robot coordination), J24 (capstone logistique).

17. **Octo: An Open-Source Generalist Robot Policy** — Octo Model Team (Berkeley/Stanford/CMU/Google), RSS 2024. https://arxiv.org/abs/2405.12213 — Site : https://octo-models.github.io/
    Pourquoi : transformer policy entraîné sur 800k trajectoires Open X-Embodiment, language ou goal-image conditioned, fine-tuning rapide sur 9 plateformes — baseline open avant l'ère VLA.
    Couvre : J18 (Open X-Embodiment & generalist policies), J19 (transformer action heads vs diffusion).

18. **Toyota Research Institute LBM — A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation** — TRI, 2024-2025. https://toyotaresearchinstitute.github.io/lbm1/
    Pourquoi : 1700h de données, ViT multimodal + transformer denoiser, 80% moins de data pour nouvelles tâches — diffusion policies industrialisées.
    Couvre : J21-J22 (LBM + scaling diffusion policies).

---

## Axe 4 — World Models & Diffusion Policies

19. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** — Chi et al. (Columbia/TRI), RSS 2023 (best paper) — IJRR 2024. https://diffusion-policy.cs.columbia.edu/ — Code : https://github.com/real-stanford/diffusion_policy
    Pourquoi : papier fondateur + repo open-source MIT-license complet (training, eval, configs, checkpoints) — base directe du capstone.
    Couvre : J16-J17 (théorie diffusion policy, score matching), J25-J28 (capstone implémentation).

20. **Mastering Diverse Domains through World Models (DreamerV3)** — Hafner, Pasukonis, Ba, Lillicrap, 2023. https://arxiv.org/abs/2301.04104
    Pourquoi : référence canonique des world models RL ; single config qui généralise sur 150+ tâches, premier à collecter des diamants dans Minecraft sans curriculum.
    Couvre : J15 (world models RL, latent imagination, RSSM).

21. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** — Meta FAIR / LeCun, 2025. https://arxiv.org/abs/2506.09985 — Blog : https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/
    Pourquoi : dernière itération JEPA appliquée robotique (zero-shot pick-and-place via goal images, 62h Droid data) — incarne la vision LeCun "world models > generative pixels". Open-weights.
    Couvre : J15-J16 (JEPA / I-JEPA / V-JEPA, prédiction dans l'espace latent vs pixel-space).

22. **Cosmos World Foundation Model Platform for Physical AI** — NVIDIA (Balaji et al.), Janvier 2025. https://arxiv.org/abs/2501.03575 — Code : https://github.com/nvidia-cosmos
    Pourquoi : foundation models open-weight (diffusion + autoregressive) entraînés sur 20M h de vidéo physique ; tokenizers vidéo et pipeline de curation réutilisables.
    Couvre : J15-J17 (world models foundation-scale, video tokenizers, post-training pour Physical AI).

23. **MIT 6.S184 — Generative AI with Stochastic Differential Equations (Flow Matching & Diffusion Models)** — Holderrieth & Erives, IAP 2025. https://diffusion.csail.mit.edu/2025/index.html — Notes PDF : https://diffusion.csail.mit.edu/docs/lecture-notes.pdf — Vidéos : https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH
    Pourquoi : fondations mathématiques rigoureuses (SDEs → score matching → flow matching → CFG) avec exercices code — comble le gap théorique avant Diffusion Policy.
    Couvre : J16 (théorie score matching / flow matching), J17 (mathématiques sous-jacentes du capstone).

---

## Axe 5 — Simulation & Tooling

24. **MuJoCo Documentation (3.x)** — Google DeepMind, 2026. https://mujoco.readthedocs.io/
    Pourquoi : moteur physique de référence pour robot learning (20× plus rapide qu'Isaac Sim sur quadrupède), maintenu activement par DeepMind, standard 2026 confirmé.
    Couvre : setup, MJCF, contacts/contraintes, viewer Python, intégration RL — toute la pile code.

25. **Gymnasium Documentation** — Farama Foundation, 2026. https://gymnasium.farama.org/
    Pourquoi : fork officiel maintenu d'OpenAI Gym (abandonné), API standard `env.reset()/step()` consommée par tous les frameworks RL.
    Couvre : intégration env, wrappers, vector envs, custom environments.

26. **robosuite v1.5 (docs + GitHub)** — ARISE Initiative (Stanford/UT Austin), 2024-2026. https://robosuite.ai/docs/ et https://github.com/ARISE-Initiative/robosuite
    Pourquoi : framework de manipulation modulaire sur MuJoCo, embodiments humanoïdes, whole-body controllers, photoreal rendering — benchmark de référence pour la manipulation.
    Couvre : tâches manipulation, controllers, teleop, génération de démos.

27. **LeRobot v0.4 + LeRobotDataset v3.0** — Hugging Face, 2025-2026. https://huggingface.co/docs/lerobot/index — https://huggingface.co/blog/lerobot-release-v040
    Pourquoi : standard 2026 pour datasets de démonstrations (Parquet+MP4 streamables) et policies VLA (PI0.5, GR00T N1.5) prêtes à l'emploi.
    Couvre : datasets imitation learning, fine-tuning policies, dataset format, streaming Hub.

28. **MuJoCo Menagerie** — Google DeepMind, 2026. https://github.com/google-deepmind/mujoco_menagerie
    Pourquoi : modèles MJCF curatés et qualité-contrôlés (UR5e, Franka, Unitree H1/Go2, Spot, ALOHA) — évite de bricoler des URDF.
    Couvre : tous les jours qui chargent un robot.

29. **Stable-Baselines3** + **CleanRL** — DLR-RM / Costa Huang, 2026. https://stable-baselines3.readthedocs.io/ — https://github.com/vwxyzjn/cleanrl
    Pourquoi : SB3 = `agent.learn()` haut-niveau fiable, intégration Gymnasium native ; CleanRL = single-file pédagogique pour comprendre PPO/SAC ligne par ligne. Complémentaires.
    Couvre : jours RL baselines (PPO, SAC sur MuJoCo).

---

## Notes

- **Stack confirmé** : MuJoCo + Gymnasium + PyTorch + LeRobot pour datasets/policies + robosuite pour manipulation. Isaac Lab non retenu (lourd, GPU NVIDIA only — surdimensionné).
- **Capstone stack final** (J25-J28) : repo `real-stanford/diffusion_policy` comme référence + reimplémentation pédagogique simplifiée sur MuJoCo PushT.
- **Wikis internes utilisés** : 6 cours university (CS223A, CS287, CS285, CS224R, 6.4210, 6.832) déjà indexés dans `references/wiki/courses/`.
- **Bonus surveys consultables si lecture transverse** : "VLA: A Survey" (https://vla-survey.github.io/), Decision Transformer (Chen 2021).

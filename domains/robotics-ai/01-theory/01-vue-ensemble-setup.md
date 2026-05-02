# J1 — Vue d'ensemble robotique moderne + setup stack

## Pourquoi ce module
La robotique 2026 vit deux mondes simultanés : la pile classique (modèles physiques + contrôleurs) et la vague IA générative (policies apprises bout-en-bout). Avant de coder quoi que ce soit, il faut une carte mentale du pipeline et un environnement de simulation qui tourne. Ce module pose les deux : on installe la stack `MuJoCo + Gymnasium + PyTorch` puis on fait tourner un premier env `HalfCheetah-v4` avec une policy aléatoire.

---

## 1. Exemple concret : un robot picking dans un entrepôt

Imagine un bras robot Franka Panda dans un entrepôt logistique. Un colis arrive sur un convoyeur. Le robot doit le saisir et le poser dans le bon bac. Pour réussir, il enchaîne :

1. **Capteurs** : caméra RGBD + encoders articulaires lisent l'état du monde (image + 7 angles).
2. **Perception** : un modèle estime la pose 6D du colis (position + orientation).
3. **Planning** : un planificateur calcule une trajectoire articulaire qui évite les obstacles.
4. **Contrôle** : à 1 kHz, un contrôleur (PID, computed torque, ou policy neural) traduit la trajectoire désirée en couples articulaires.
5. **Actuation** : les moteurs reçoivent les couples et bougent les articulations.

Chaque étape est un sous-problème étudié depuis 50 ans (Khatib, Lynch, Tedrake) et chaque étape a aujourd'hui un pendant "appris" (Finn, Levine, Chi). Le but du domaine est de comprendre les deux ères et de pouvoir naviguer entre elles.

> **Key takeaway** — La robotique = pipeline `capteurs → perception → planning → contrôle → actuation`. Chaque maillon a une version classique (modèle explicite) et une version IA (réseau appris).

---

## 2. La pile complète, formalisée

| Étage | Entrée | Sortie | Fréquence typique | Approche classique | Approche IA |
|---|---|---|---|---|---|
| Capteurs | physique | mesures brutes | 30 Hz à 1 kHz | calibration, filtres | self-supervised features |
| Perception | mesures | état (pose, scène) | 10-30 Hz | ICP, SLAM, PnP | DINOv2, FoundationPose, V-JEPA |
| Planning | état + but | trajectoire | 1-10 Hz | RRT*, trajopt | Diffusion Policy, world models |
| Contrôle | trajectoire + état | couples / vitesses | 100 Hz à 1 kHz | PID, LQR, MPC | RL (PPO, SAC), VLA |
| Actuation | couples | mouvement | 1 kHz+ | drivers moteurs | identique |

Les fréquences sont importantes : un VLA tournant à 5 Hz ne peut **pas** remplacer un PID à 1 kHz. C'est pourquoi les architectures modernes (GR00T N1, Helix de Figure) sont dual-system : System2 lent (raisonnement, ~10 Hz) + System1 rapide (contrôle, 200 Hz).

> **Key takeaway** — Toutes les couches ne tournent pas à la même fréquence. Une policy "intelligente" lente doit être appariée à un contrôleur rapide en aval [Lynch & Park, 2017, ch. 1].

---

## 3. Deux ères, une seule discipline

### Ère classique (Khatib, Tedrake, Siciliano, LaValle)
- Modèles physiques explicites : `M(q) q̈ + C(q,q̇) q̇ + g(q) = τ`
- Garanties théoriques (stabilité Lyapunov, optimalité LQR)
- Très bon pour : manipulation industrielle structurée, locomotion classique, drones
- Faiblesse : généralisation à des scènes inédites coûte une re-modélisation manuelle

### Ère IA générative (Finn, Levine, Chi, Black, Hafner)
- Policies apprises : `π_θ(a | o)` avec o = pixels + langage
- Pas de modèle dynamique explicite — appris implicitement
- Très bon pour : manipulation dexterous, multi-tâches, généralisation visuelle
- Faiblesse : besoin de masses de données, garanties de stabilité plus faibles

Aucune des deux ne remplace l'autre. Un VLA comme π0 (Physical Intelligence, 2024) repose sur un **flow matching head** (IA) mais s'exécute par-dessus des contrôleurs articulaires bas-niveau (classique). Le maillon faible reste classique parce qu'il est temps-réel.

> **Key takeaway** — Maîtriser la robotique 2026 = lire Lynch & Park ET le papier Diffusion Policy. Les deux sont des prérequis, pas des concurrents [CS223A, Khatib L1].

---

## 4. Stack 2026 : pourquoi MuJoCo + Gymnasium + PyTorch

### MuJoCo 3.x
- Moteur physique racheté par DeepMind (open-source en 2022).
- Contacts avec contraintes douces (soft constraints) → numériquement stable même pour la manipulation.
- Maintenance active, port WebAssembly, MJX (XLA backend) pour entraînement RL massivement parallèle.
- ~20× plus rapide qu'Isaac Sim sur des tâches quadrupèdes selon les benchmarks 2024.

### Gymnasium
- Fork officiel maintenu d'OpenAI Gym (qui n'est plus maintenu depuis 2022).
- API standard `obs, info = env.reset()` et `obs, reward, terminated, truncated, info = env.step(action)`.
- Tous les frameworks RL modernes (CleanRL, Stable-Baselines3, RLlib) consomment cette API.

### PyTorch
- Standard de fait pour la recherche depuis 2019.
- Tous les VLA récents (OpenVLA, π0, GR00T N1) sont en PyTorch.
- Bon support `torch.compile`, FSDP, et inférence quantifiée.

> **Key takeaway** — Ce trio est le lowest-common-denominator de la recherche 2026. Apprends-le, et tu peux lire 95 % des repos publiés [MuJoCo docs §1].

---

## 5. Anatomie d'un step Gymnasium

Le contrat d'un environnement Gymnasium tient en deux méthodes :

```python
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

- `obs` : observation à l'instant t (numpy array, shape dépend de l'env)
- `reward` : scalar récompense reçue pour avoir exécuté `action` dans l'état précédent
- `terminated` : `True` si l'épisode finit pour raison "naturelle" (échec, but atteint)
- `truncated` : `True` si l'épisode finit par limite de temps (TimeLimit wrapper)
- `info` : dict d'infos auxiliaires (debug, métriques cachées)

La distinction `terminated` vs `truncated` est cruciale pour le RL : on bootstrap la value function quand `truncated=True` mais pas quand `terminated=True`. Un bug courant pré-Gymnasium 1.0 était de confondre les deux.

Pour `HalfCheetah-v4` : observation = 17 dims (pos + vel articulaires), action = 6 dims (couples normalisés [-1, 1]), reward = vitesse vers l'avant - coût énergétique.

> **Key takeaway** — Le step renvoie 5 valeurs, pas 4. `terminated` et `truncated` sont distincts et le distinguo affecte les algos de RL [Gymnasium docs, Env API].

---

## 6. Setup pratique

```bash
pip install "gymnasium[mujoco]" mujoco torch
python -c "import gymnasium, mujoco, torch; print(gymnasium.__version__, mujoco.__version__, torch.__version__)"
```

Sur Windows, MuJoCo 3.x s'installe via wheel `pip` sans configuration manuelle (différent de MuJoCo 2.x qui demandait une licence et un binaire). Premier env qui tourne en 5 lignes :

```python
import gymnasium as gym
env = gym.make("HalfCheetah-v4")
obs, _ = env.reset(seed=0)
for _ in range(100):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
env.close()
```

Si ces 5 lignes tournent, tu as la stack. Le code applique du jour fait exactement ça avec un peu plus d'instrumentation.

> **Key takeaway** — Une policy aléatoire qui tourne sans crash = victoire du jour 1. Le RL "intelligent" arrive à partir du J9 [MuJoCo docs §Gymnasium].

---

## Spaced repetition

**Q1.** Cite les 5 maillons du pipeline robotique du capteur à l'actuation.
> Capteurs → perception → planning → contrôle → actuation.

**Q2.** Quelle est la différence entre `terminated` et `truncated` dans Gymnasium ?
> `terminated` = fin "naturelle" (succès, échec, état terminal MDP). `truncated` = fin par limite de temps externe. Pour le RL on bootstrap la value function quand `truncated=True`, jamais quand `terminated=True`.

**Q3.** Pourquoi MuJoCo plutôt qu'un autre moteur en 2026 ?
> Open-source DeepMind, contacts numériquement stables, MJX/XLA pour parallélisation GPU, ~20× plus rapide qu'Isaac Sim sur quadrupèdes, wheels pip sans licence.

**Q4.** Pourquoi les architectures VLA modernes (Helix, GR00T) sont-elles dual-system ?
> Parce qu'un VLA "intelligent" tourne typiquement à 5-10 Hz, insuffisant pour le contrôle articulaire qui exige 100 Hz à 1 kHz. System2 (lent, raisonnement) + System1 (rapide, contrôle) résout l'écart de fréquences.

**Q5.** Citez un représentant de l'ère classique et un de l'ère IA générative en robotique.
> Classique : Oussama Khatib (operational space control, CS223A) ou Russ Tedrake (LQR, trajopt, MIT 6.832). IA : Cheng Chi (Diffusion Policy 2023) ou Sergey Levine (CS285, GR00T, π0).

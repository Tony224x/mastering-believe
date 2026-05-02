# J28 — Capstone packaging + integration LogiSim + retrospective

> Dernier jour. Tu as un modele Diffusion Policy entraine et evalue sur PushT. Aujourd'hui : tu en fais un livrable shippable, tu projettes l'usage dans un contexte logistique reel (LogiSim/FleetSim), et tu retraces 28 jours pour cartographier la suite logique.

---

## 1. Concret avant abstrait : la demo qu'un recruteur peut lancer en 30 secondes

Un capstone qui n'est pas demontrable n'existe pas. Imagine la scene : tu envoies le repo a un staff engineer Kalira ou a un recruteur senior chez TRI. Ce qu'il fait :

```bash
git clone <ton-repo>
cd capstone-diffusion-policy
make demo            # ou : python demo.py
```

En 30 secondes il doit voir :

1. Un checkpoint Diffusion Policy se charge (tu l'as commit en LFS ou tu l'auto-genere si absent).
2. Une fenetre matplotlib affiche un rollout : le bras 2D pousse le T jusqu'a la cible.
3. Stdout imprime un rapport : `Success rate: 78% over 50 rollouts | Mean episode length: 142 steps | Latency per step: 38 ms`.
4. Une section LogiSim simulee montre comment la meme policy s'integrerait dans FleetSim sur un AGV de picking.

Pas de README de 12 pages a lire. Pas de 14 commandes a chainer. **Une commande, un resultat visuel, un nombre.** C'est la difference entre un projet d'ecole et un livrable ingenieur.

A partir de ce concret, on extrait trois principes : *packaging*, *integration domaine*, *retrospective tracee*. Chacun merite sa section.

---

## 2. Packaging d'un capstone : checkpoint, README, scripts

### 2.1 Anatomie d'un repo de capstone shippable

```
capstone-diffusion-policy/
├── README.md                    # 1 page, GIF demo en haut, install + run en bas
├── Makefile                     # make install / make data / make train / make eval / make demo
├── pyproject.toml               # dependances pinnees (lock le stack)
├── checkpoints/
│   └── pusht_dp_final.ckpt      # poids EMA finaux (Git LFS si > 50 MB)
├── data/
│   └── pusht_demos.zarr         # dataset LeRobot ou pickle des demos
├── configs/
│   ├── train.yaml               # hyperparams figes pour reproduire
│   └── eval.yaml                # config eval (n_rollouts, seed, env)
├── scripts/
│   ├── train.py                 # python scripts/train.py --config configs/train.yaml
│   ├── eval.py                  # produit eval/results.json + figures/
│   └── generate_demos.py        # genere les demos PushT
├── src/
│   └── diffusion_policy/        # ton package importable
├── demo.py                      # POINT D'ENTREE : python demo.py
├── eval/
│   └── results.json             # success rate, latency, ablations
└── figures/
    ├── rollouts.png
    └── training_loss.png
```

**Regles d'or** :

- **Un seul point d'entree visible** (`demo.py` ou `make demo`). Tout le reste est invocable mais optionnel.
- **Pas de chemins en dur** : config via YAML, override via CLI flags.
- **Checkpoint commit ou re-generable** : si > 100 MB GitHub bloque, alors prevoir un script de download (HF Hub, S3, GCS) ou Git LFS.
- **Resultats traces dans le repo** (`eval/results.json`, `figures/*.png`). Le lecteur doit voir les chiffres sans relancer.

### 2.2 Le README portfolio-grade (1 page, 5 sections)

Un bon README de capstone tient sur un ecran. Structure :

1. **Title + 1-line pitch** + GIF de la demo (≤ 5 MB).
2. **What it does** : 3 bullets, dont un avec un chiffre concret (success rate, latence).
3. **Quickstart** : 3 commandes max (`pip install -e .`, `python demo.py`, optionnel `make eval`).
4. **Architecture** : 1 schema (ResNet18 → UNet 1D → DDPM scheduler → action chunking) + 4-6 lignes de prose.
5. **Results** : tableau success rate Diffusion Policy vs BC + lien vers `eval/results.json`.

Ce qu'on **n'inclut pas** dans le README : justification academique, theorie, derivation des SDEs. Ces choses vivent dans `01-theory/` (le repo apprentissage). Le README du capstone parle au recruteur, pas a l'etudiant.

### 2.3 Makefile : la convention universelle pour les ML repos

```makefile
.PHONY: install data train eval demo clean

install:
	pip install -e .

data:
	python scripts/generate_demos.py --n_demos 200 --output data/pusht_demos.zarr

train:
	python scripts/train.py --config configs/train.yaml

eval:
	python scripts/eval.py --config configs/eval.yaml --checkpoint checkpoints/pusht_dp_final.ckpt

demo:
	python demo.py

clean:
	rm -rf checkpoints/*.tmp data/*.cache figures/*.png
```

Pourquoi un Makefile alors qu'on est en Python ? Parce que tout ML engineer 2026 sait lire `make train`, et que c'est la norme implicite des repos open-source robotique (cf. `real-stanford/diffusion_policy`, `openvla/openvla`, `huggingface/lerobot` — REFERENCES.md #19, #13, #27). Alternative : `scripts/` + un `tasks.py` (invoke / nox). Aussi acceptable.

---

## 3. Demo finale : `python demo.py` en moins d'une minute

### 3.1 Ce que `demo.py` doit faire

Ordre des operations :

```python
def main():
    # 1. Charger ou generer un mini-checkpoint (auto-fallback)
    policy = load_or_init_policy(CHECKPOINT_PATH)

    # 2. Lancer N rollouts (typiquement 5-10 pour la demo, vs 50 pour l'eval)
    results = run_rollouts(policy, env, n=10)

    # 3. Visualiser : trajectoires plottees + GIF/MP4 si possible
    plot_rollouts(results, save_to="figures/demo_rollouts.png")

    # 4. Imprimer le rapport
    print_report(results)
    # Success rate: 78% (8/10) | Mean length: 138 steps | Latency: 36.4 ms/step

    # 5. Mock LogiSim : montre l'integration AGV picking
    demo_logisim_integration(policy)
```

**Astuce auto-fallback** : si le checkpoint n'est pas la (cas typique sur une fresh clone), `load_or_init_policy` initialise une policy random ou minimale en interne. La demo tourne quand meme — mais le rapport indique `[WARN] Using untrained policy, success rate is illustrative`. Mieux vaut une demo qui marche avec disclaimer qu'une demo qui crashe.

### 3.2 Sortie standard du rapport

Format que tout le monde reconnait (S&B, CleanRL, Spinning Up — REFERENCES.md #6, #9, #7) :

```
======================================================================
DIFFUSION POLICY — PushT eval (n=10)
======================================================================
Success rate     : 78.0% (8/10)
Mean ep length   : 138.4 (+/- 22.1)
Mean step latency: 36.4 ms (95p: 41.2 ms)
Total wall time  : 4.7 s
Checkpoint       : checkpoints/pusht_dp_final.ckpt (12.4 MB)
======================================================================
```

**Deux nombres comptent vraiment** :
- **Success rate** : la metrique end-to-end. Si elle bouge, tu as quelque chose a dire.
- **Step latency** : le critere de deployabilite. 36 ms = compatible 25 Hz, donc deployable sur un robot reel (200 Hz Helix est plus exigeant — REFERENCES.md #16).

---

## 4. Analogie LogiSim : du PushT au picking d'AGV dans FleetSim

> **Reference metier** : `shared/logistics-context.md` (FleetSim, AutonomyAI SDK, Work Orders, OCC).
> **Reference frontiere** : Helix Logistics 2025, Figure AI, premier deploiement industriel d'un VLA en logistique reelle (REFERENCES.md #16).

### 4.1 Le scenario : un AGV de picking dans un entrepot FleetSim

Dans le contexte LogiSim, un **AGV de picking** est un AGV equipe d'un bras manipulateur 7-DOF (typique : Franka Panda ou UR5e). Il navigue dans un rack, identifie un colis, le saisit, le depose dans un panier embarque, repart vers le prochain pick. Aujourd'hui chez LogiSim, la trajectoire bras est **scriptee** (planner classique : RRT + IK + suivi de trajectoire). Ca marche tant que le rack est calibre et que les colis sont dans des slots normalises.

Probleme reel : quand un colis est mal pose, qu'un autre colis bloque l'acces, ou qu'un nouveau type d'emballage arrive (cellophane glissant, sac souple), le planner classique echoue silencieusement (`PICKUP` event avec `ok=False` selon le schema canonique — voir `shared/logistics-context.md`).

### 4.2 Comment Diffusion Policy s'y plugge

Le contrat exact :

| Element capstone (PushT)               | Equivalent LogiSim (AGV picking)                                 |
|----------------------------------------|------------------------------------------------------------------|
| Observation : image RGB 96x96 + agent_pos | Image RGB 224x224 (camera bras) + gripper_state + tcp_pose       |
| Action : (dx, dy) du pusher 2D            | Trajectoire articulaire 7-DOF + commande gripper (open/close)    |
| Action chunking : k=8 actions a la fois   | Idem, k=8 ou 16 (compatible 10-25 Hz du loop bras)               |
| Dataset : 200 demos teleoperation joystick | Demos teleoperees par operateurs LogiSim (joystick ou VR)        |
| Eval : success rate sur PushT             | Success rate sur 50 picks reels en simulation FleetSim           |

**Le pipeline d'integration** :

```
FleetSim tick (10 Hz)                       Diffusion Policy
       |                                          |
       |  observation = {camera_rgb,              |
       |                 gripper_state,           |
       |                 tcp_pose,                |
       |                 work_order_text}         |
       |  -------------------------------------> |
       |                                          | (denoising 8 actions)
       |  action_chunk[8] = [(q1..q7, grip)]     |
       |  <------------------------------------- |
       |                                          |
       |  execute action_chunk via                |
       |  AutonomyAI SDK low-level controller     |
       |                                          |
       |  emit events MOVE / PICKUP / DROPOFF    |
       |  selon schema canonique LogiSim          |
```

Trois adaptations critiques pour passer du capstone PushT au prod LogiSim :

1. **Augmenter le conditioning** : ajouter le texte du Work Order (`order_type`, `details` du schema LogiSim) en entree du modele. C'est exactement ce que font les VLA (OpenVLA, π0 — REFERENCES.md #13, #14). Diffusion Policy seule + texte = un mini-VLA.
2. **Garantir la latence** : DDPM standard fait 100 steps de denoising. Sur un loop bras 25 Hz, c'est too slow. Solutions : DDIM (10 steps), consistency models, ou flow matching (1-5 steps comme π0-FAST — REFERENCES.md #14).
3. **Safety policy** : dans LogiSim, une policy ne peut pas commander librement le bras pres d'un humain (cf. *Safety Policy* dans `shared/logistics-context.md`). On wrappe la sortie Diffusion Policy avec un *safety filter* qui rejette les actions dont la `Pcollision` (voir vocabulaire LogiSim) depasse un seuil. Le filter est code classique, pas appris.

### 4.3 Ce que le capstone ne couvre PAS (et qu'il faudrait pour FleetSim)

- **Multi-task** : le capstone fait *une* tache (pusht). Un AGV picking doit savoir picker des dizaines de SKU.
- **Generalisation cross-environnement** : un nouveau rack = re-collecter des demos. Pas viable. Solution = VLA pre-entraines (OpenVLA, π0, GR00T N1 — REFERENCES.md #13, #14, #15) qui font du zero-shot ou few-shot.
- **Long-horizon** : picker un colis = 5-10 secondes. Decharger un camion = 20 minutes. Diffusion Policy seule cale au-dela de 30 secondes. Solution = system 2 / planner haut-niveau (LBM TRI, Helix System 2 — REFERENCES.md #18, #16).

---

## 5. Roadmap : de PushT vers la manipulation reelle puis vers les VLA

Trois paliers bien identifies dans la litterature 2024-2026.

### 5.1 Palier 1 — De PushT a une vraie tache de manipulation (1-2 mois)

- Passer du **2D** au **6-DOF** : prendre un environnement robosuite (REFERENCES.md #26) ou MuJoCo Menagerie avec bras Franka.
- Tache cible : **block-stacking** ou **peg-insertion** (benchmark Diffusion Policy paper, REFERENCES.md #19).
- Generer 200-500 demos via teleop (clavier ou Leap Motion) ou expert PPO entraine.
- Re-entrainer Diffusion Policy avec le meme code que le capstone, juste action_dim = 7 + grip.

**Sources** : repo `real-stanford/diffusion_policy` configs `lift`, `square`, `tool_hang`. Lire `train_diffusion_unet_image_workspace.yaml`.

### 5.2 Palier 2 — Generalisation multi-task et multi-embodiment (3-6 mois)

- Passer a **Octo** (REFERENCES.md #17) ou **OpenVLA** (REFERENCES.md #13). Action tokenization + transformer policy. Fine-tune LoRA sur ta tache custom.
- Dataset : Open X-Embodiment (800k+ trajectoires multi-robots) + tes propres demos.
- Cible : **un seul checkpoint** qui marche sur 5-10 taches distinctes.

**Sources** : OpenVLA repo + paper, LeRobot v0.4 datasets format (REFERENCES.md #27).

### 5.3 Palier 3 — Vers l'industrialisation type Helix / LBM TRI (6-12+ mois)

- Architecture **dual-system** : System 2 (VLM 7B) raisonne, System 1 (diffusion ou flow head 80M-300M) execute a 200 Hz. Cf. Helix (REFERENCES.md #16) et GR00T N1 (REFERENCES.md #15).
- Synthetic data a scale : 100k+ trajectoires generees en sim (Cosmos pipeline, NVIDIA — REFERENCES.md #22).
- Sim-to-real : domain randomization + adaptation (deja vu en J14 du parcours).
- Deploiement reel : c'est la frontiere. Helix Logistics 2025 (REFERENCES.md #16) montre que c'est faisable mais demande une equipe entiere.

**Vision long-terme** : LBM TRI (REFERENCES.md #18) montre que 1700h de donnees + diffusion policies = 80% de reduction de data sur de nouvelles taches. La direction est tracee.

---

## 6. Retrospective : 28 jours, 5-10 acquis tangibles

A la fin de ce parcours, tu peux honnetement claim ce qui suit. Pas plus, pas moins.

### 6.1 Acquis techniques solides

1. **Pile robotique classique maitrisee** : SE(3), FK/IK, dynamique Lagrange, PID/LQR, RRT — tu peux lire un papier de robotique sans buter sur la notation.
2. **MuJoCo + Gymnasium en mains** : tu sais charger un robot Menagerie, ecrire un env custom, lancer un rollout, lire qpos/qvel. C'est la baseline de tout robot learning 2026.
3. **Deep RL fonctionnel** : tu as entraine PPO et SAC sur MuJoCo en moins d'une heure. Tu sais lire CleanRL ligne par ligne (REFERENCES.md #9).
4. **Imitation learning + sim-to-real** : tu sais quand IL bat RL (demos abondantes, reward absente) et tu connais le reality gap.
5. **Diffusion / flow matching demystifies** : tu peux deriver le DDPM training objective et expliquer pourquoi flow matching est une generalisation propre (REFERENCES.md #23).
6. **Capstone Diffusion Policy de bout en bout** : dataset → architecture (ResNet18 + UNet 1D + DDPM) → training loop avec EMA → eval avec success rate. Le repo est shippable.
7. **Lecture du paysage VLA frontier** : OpenVLA, π0, GR00T N1, Helix, LBM TRI. Tu sais qui fait quoi, quelles sont les limites de chacun.
8. **Vision LogiSim claire** : tu sais ce qu'il faudrait pour porter Diffusion Policy dans FleetSim (texte conditioning, latence, safety filter, multi-task).

### 6.2 Frontieres honnetement identifiees

- **Pas teste sur du materiel reel** : tout est en simulation. Le sim-to-real residuel reste la grande inconnue.
- **VLAs de bout en bout pas encore entraines** : tu as lu OpenVLA / π0, mais le fine-tuning LoRA reel demande un GPU 24+ GB que tu n'as peut-etre pas.
- **Long-horizon faible** : 30+ secondes de tache reste hors de portee sans un System 2 explicite.
- **Synthetic data a scale** : Cosmos / Isaac Lab restent a explorer.

### 6.3 Suite logique : 3 ressources pour passer au niveau suivant

1. **CMU 16-831 / Stanford CS336 / UPenn MEAM 5200** : cours universitaires academiques pour solidifier la theorie classique non couverte ici (mecanique avancee, control adaptatif).
2. **Lecture continue de papers** : suivre l'arxiv-sanity feed sur cs.RO, particulierement les outputs de Physical Intelligence, Toyota Research Institute, NVIDIA Robotics, Figure AI, Berkeley AI Research.
3. **Build something real** : decrocher un robot reel — un Franka Panda, un Aloha, ou meme un SO-100 LeRobot a 250$ — et reproduire le capstone dessus. C'est la seule maniere de toucher le sim-to-real.

---

## 7. Key takeaway

> **Ce qui rend un capstone "pro" n'est pas l'astuce du modele, c'est la combinaison : un livrable demonstrable en 30s + un repo lisible + un alignement clair avec un domaine metier (LogiSim/FleetSim) + une retrospective honnete sur ce qui marche, ce qui ne marche pas, et la suite.** Diffusion Policy sur PushT est ton point d'entree. La trajectoire OpenVLA → π0 → Helix est la suite logique. Tu n'es pas a la frontiere ; tu es maintenant en mesure de la lire et de t'en approcher.

---

## 8. Spaced-repetition Q&A

**Q1 — Quelle est la regle "une commande, un resultat" pour un capstone ML ?**
R — `python demo.py` (ou `make demo`) doit afficher en moins d'une minute : un visuel de rollout + un rapport texte avec success rate et latence. Pas plus.

**Q2 — Pourquoi inclure un Makefile dans un repo Python ?**
R — Convention de fait des repos ML/robotique 2026. `make train`, `make eval`, `make demo` sont parses instantanement par tout ingenieur. Reduit le cout cognitif de relecture du repo.

**Q3 — Quels sont les 3 changements minimaux pour passer Diffusion Policy de PushT a un AGV picking FleetSim ?**
R — (1) Conditioning sur texte du Work Order, (2) reduction de la latence de denoising (DDIM ou flow matching), (3) safety filter wrappant les actions sortantes pour respecter la Safety Policy LogiSim.

**Q4 — Pourquoi Helix (Figure 2025) est cite comme reference d'integration logistique ?**
R — Premier VLA deploye reellement en logistique (35 DoF @ 200 Hz, multi-robot, faster-than-demonstrator) avec System 2 7B + System 1 80M. Materialise le pattern dual-system applique au monde reel (REFERENCES.md #16).

**Q5 — Quelle est la prochaine etape concrete apres ce parcours pour quelqu'un sans budget GPU pro ?**
R — Reproduire le capstone sur un bras 6-DOF en simulation robosuite (peg-insertion ou block-stacking). Si budget : SO-100 LeRobot ~250$ pour toucher le sim-to-real reel. Sinon : LoRA fine-tuning d'OpenVLA en 4-bit sur GPU 16 GB cloud (Colab Pro / Vast.ai).

---

## Sources citees

- REFERENCES.md #16 — Helix Figure AI 2025 + Helix Logistics (analogie LogiSim, dual-system applique a la logistique reelle)
- REFERENCES.md #18 — TRI Large Behavior Models (vision long-terme : 1700h donnees, 80% moins data)
- REFERENCES.md #19 — Diffusion Policy paper + repo Columbia/Stanford (base directe du capstone)
- REFERENCES.md #13 — OpenVLA (palier de generalisation multi-task)
- REFERENCES.md #14 — π0 / π0.5 Physical Intelligence (palier flow matching + open-world)
- REFERENCES.md #15 — GR00T N1 NVIDIA (synthetic data + dual-system)
- REFERENCES.md #17 — Octo (transformer policy multi-embodiment)
- REFERENCES.md #22 — NVIDIA Cosmos (synthetic data foundation-scale)
- REFERENCES.md #23 — MIT 6.S184 (fondations theoriques)
- REFERENCES.md #6, #7, #9 — Sutton & Barto, Spinning Up, CleanRL (formats de reporting)
- REFERENCES.md #26 — robosuite v1.5 (env manipulation 6-DOF)
- REFERENCES.md #27 — LeRobot v0.4 (dataset format)
- `shared/logistics-context.md` — contexte LogiSim/FleetSim, schema event canonique, vocabulaire metier (AGV, Work Order, OCC, Safety Policy, Pcollision)

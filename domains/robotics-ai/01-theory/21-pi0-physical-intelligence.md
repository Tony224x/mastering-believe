# J21 — π0 / π0.5 (Physical Intelligence)

> Durée d'étude : 45-60 min
> Prérequis : J15 (diffusion + flow matching), J16 (Diffusion Policy), J17 (theorie SDE/ODE), J18-J20 (RT-2 → OpenVLA → architectures VLA).
> Source principale : REFERENCES.md #14 — Black et al. (Physical Intelligence), "π0: A Vision-Language-Action Flow Model for General Robot Control" (2024) + π0.5 (2025) + blog https://www.pi.website/blog/pi0.

---

## 1. Hook — un robot qui plie ton linge dans une chambre qu'il n'a jamais vue

En avril 2025, Physical Intelligence publie une vidéo : un robot mobile à deux bras entre dans une **chambre d'inconnu** (jamais vue à l'entraînement), ouvre les tiroirs, replie un t-shirt, range les chaussettes. Pas de re-tuning, pas de démos sur cette pièce. Le modèle s'appelle **π0.5** et il est l'évolution directe de π0 — première vraie démonstration de "open-world generalization" sur des tâches mobiles dexterous.

Le truc à retenir : **π0 n'est pas un modèle "en plus" comme RT-2 ou OpenVLA. C'est le premier qui combine :**
1. un **VLM pré-entraîné** (PaliGemma 3B, gelé puis fine-tuné) — backbone vision-langage classique vu en J19/J20,
2. une **flow matching action head** (au lieu de DDPM ou de tokens discrets) — c'est le saut conceptuel du jour,
3. un **entraînement multi-embodiment** sur 7 robots différents (single-arm UR5e, ALOHA, Trossen Mobile, etc.) avec **10 000 heures** de démos.

Le résultat : un seul checkpoint qui contrôle un single-arm, un bi-bras et un mobile manipulator avec la même policy. Plus précis que diffusion policy et plus rapide que RT-2 grâce au tokenizer **π0-FAST** (5× plus rapide).

> **Key takeaway #1**
> π0 = recette VLA moderne 2024-2025 = (VLM gelé + LoRA partiel) + (flow matching action head qui génère 50 actions futures à 50 Hz) + (mix multi-embodiment massif). Tout le reste du module explique pourquoi chaque ingrédient compte.

---

## 2. Architecture π0 — VLM + flow matching action expert

### 2.1 Vue d'ensemble

```
                ┌─────────────────────────────────────┐
                │          PaliGemma VLM 3B           │
  images RGB ──►│  (vision encoder SigLIP + LM)       │
  instruction ──►│                                     │
                │  → tokens latents h ∈ R^{T × d}     │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │     Action Expert (300M params)     │
  noised actions  │  conditioned on h via cross-attn   │
  A_t = A + σε ──►│  flow matching head                │
                │  prédit le champ de vélocité v_θ    │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
              ODE solve (Euler, 5-10 steps)
                                 │
                                 ▼
              chunk d'actions A* ∈ R^{H × d_action}
              (H=50 frames @ 50 Hz = 1 seconde)
```

Décomposition :

- **VLM backbone** : PaliGemma 3B (Google, 2024). C'est un classique vision-langage — pas spécifique robotique. Il fournit une représentation jointe images+text qui n'est PAS dégradée pendant l'entraînement robotique grâce à un trick : **mixed gradient flow** (les gradients de la action head ne traversent que partiellement le VLM).
- **Action expert** : un *petit* transformer (300M) attaché en parallèle. C'est lui qui apprend la flow matching. Reçoit le bruit + un timestep + les tokens du VLM, prédit une vélocité.
- **Action chunk** : on ne prédit pas 1 action mais **un chunk de H=50 actions** (concept ré-emprunté à ACT/Diffusion Policy J16). Permet le receding-horizon control à haute fréquence sans replanifier à chaque pas.

### 2.2 Pourquoi un "expert" séparé et pas un seul gros modèle ?

Trade-off pratique : le VLM PaliGemma est lent à inférer (3B params). Si on le fait re-rouler à chaque step de l'ODE flow matching (5-10 steps), c'est ingérable @ 50 Hz. **Solution π0** : faire passer l'image + instruction UNE FOIS dans le VLM (calcul cher), puis l'ODE solve ne tourne QUE l'action expert (300M, rapide). On amortit le VLM sur les 50 actions générées.

> **Key takeaway #2**
> Pattern "frozen-ish VLM + lightweight action expert" = clé de l'inférence haute fréquence. Le VLM contribue la sémantique long-terme (1 forward pass), l'action expert contribue la dynamique court-terme (5-10 forward passes mais cheap).

---

## 3. Flow matching action head — le saut conceptuel vs DDPM

### 3.1 Rappel J15-J16 : DDPM dans Diffusion Policy

En J15 (diffusion + flow matching), on a vu que DDPM apprend à **prédire le bruit** ε_θ(x_t, t) à partir d'un schéma de bruitage discret en T=1000 steps :

```
forward  : x_t = √α_bar_t · x_0 + √(1 - α_bar_t) · ε        (closed form)
reverse  : x_{t-1} = f(x_t, ε_θ(x_t, t)) + σ_t · z          (z ~ N(0,I))   ← SDE stochastique
```

Diffusion Policy (J16) utilise DDPM avec T=100 steps de denoising par inférence — coût rédhibitoire @ 50 Hz.

### 3.2 Flow matching π0 : ODE plutôt que SDE

**Flow matching** (Lipman, Chen et al. 2023) reformule le problème comme un **transport optimal** entre une distribution simple (gaussienne) et la distribution cible (actions expertes). On n'apprend plus du bruit — on apprend un **champ de vélocité** :

```
v_θ(A_τ, τ) ≈ A* - A_0       où A_τ = (1-τ)·A_0 + τ·A*    (interpolation linéaire)
                              τ ∈ [0, 1] continu, pas de schedule discret
```

À l'inférence, on résout une ODE (déterministe, pas de bruit) :

```
dA/dτ = v_θ(A_τ, τ)
A_0 ~ N(0, I)         (point de départ : bruit pur)
A_1 = A*              (point d'arrivée : action propre, attendue)
```

Avec un solveur Euler simple, **5 à 10 steps suffisent** au lieu de 100 pour DDPM.

### 3.3 Tableau de contraste — DDPM vs Flow matching π0

| Aspect | DDPM (Diffusion Policy J16) | Flow matching (π0) |
|---|---|---|
| Cible apprise | bruit ε | vélocité v = (target - source) |
| Schedule | discret, T=100..1000 steps | continu, τ ∈ [0,1] |
| Inférence | SDE stochastique (z ajouté à chaque step) | ODE déterministe |
| Nb steps inférence | 50-100 (DDIM accélère à 10-20) | **5-10 Euler steps** |
| Qualité asymptotique | excellente, peut "regenerer" finement | équivalente sur actions |
| Trade-off | qualité > vitesse | vitesse ≈ qualité (action chunks petits) |
| Math sous-jacente | score matching / Langevin | optimal transport / continuous normalizing flows |

> **Key takeaway #3**
> Pour les actions robotiques (espace petit, 7-14 dim, courtes séquences), **flow matching domine DDPM en pratique** : même qualité, 5-10× plus rapide à l'inférence. C'est précisément pourquoi π0 a abandonné DDPM. Pour de la génération d'images haute-résolution, DDPM/score matching reste compétitif (espace immense, on peut payer le coût).

### 3.4 Loss training

Trivialement simple côté implémentation :

```
L_FM(θ) = E_{A*, A_0~N(0,I), τ~U(0,1)} [ ‖ v_θ(A_τ, τ, h) - (A* - A_0) ‖² ]
                                                                ▲
                                                                │
                                                  vélocité-cible (constante)
```

Comparer avec la loss DDPM :

```
L_DDPM(θ) = E_{x_0, ε, t} [ ‖ ε_θ(x_t, t) - ε ‖² ]    (bruit-cible)
```

Même squelette MSE — la différence est dans **ce qu'on régresse** et **comment on échantillonne à l'inférence**.

---

## 4. Multi-embodiment — un modèle, plusieurs robots

### 4.1 Le problème

Robot single-arm UR5e : 7 DoF. Bi-bras ALOHA : 14 DoF. Mobile manipulator Trossen : 14 DoF + 2 DoF base. Chacun a son espace d'action propre, ses cinématiques propres, ses capteurs propres.

**Recette naïve** : un modèle par robot. Coûteux, pas de transfer learning entre embodiments.

**Recette π0** : un seul modèle, espace d'action **paddé à dim_max = 18** (le max parmi les 7 robots du dataset). Pour un robot avec moins de DoF, les dimensions inutiles sont masquées (mask de loss + zéro-injection). Le modèle apprend **implicitement** quel sous-espace est valide pour chaque embodiment via le contexte image (forme du robot dans la caméra) et un token d'embodiment optionnel.

### 4.2 Pourquoi ça marche

Hypothèse de Physical Intelligence (validée empiriquement) : **les compétences manipulatives partagent une structure géométrique commune** au-delà de l'embodiment. "Saisir une tasse" implique des trajectoires similaires en termes de pose end-effector, peu importe que l'arm soit un UR5e ou un bras ALOHA. Le modèle généralise sur la skill, pas sur le moteur.

Conséquence dataset : 10 000 heures de démos hétérogènes >> 1 000 heures par robot dédiées. Power scaling = power generalization.

> **Key takeaway #4**
> Multi-embodiment training n'est PAS du multi-task naïf. C'est un pari sur la **transférabilité géométrique** des skills. Compatible avec Open X-Embodiment (J18) et la philosophie "RT-X".

---

## 5. π0.5 (2025) — open-world generalization

π0 fait du multi-embodiment dans **des environnements vus à l'entraînement** (même cuisines, mêmes labos). π0.5 (Intrator et al., Physical Intelligence, avril 2025, REFERENCES.md #14, https://www.pi.website/download/pi05.pdf) ajoute :

1. **Co-training avec données web** : on mixe les démos robotiques avec des **descriptions sémantiques** ("ouvre le tiroir du haut") issues du web → capacité high-level reasoning préservée.
2. **Hierarchical inference** : le VLM produit d'abord un **plan de sous-tâches** en langage naturel (ex : "1. ouvrir tiroir, 2. trouver chaussette, 3. la poser"), puis chaque sous-tâche conditionne l'action expert.
3. **Mobile manipulation extensive** : 400+ heures dans des **maisons inconnues** (Airbnb, locations).

Résultat : transferts vers chambres/cuisines complètement inédites en zero-shot. C'est l'amorce de ce que Physical Intelligence appelle **"foundation policy"** — la même promesse que GPT-3 a faite pour le langage, appliquée au contrôle moteur.

### 5.1 Limites encore présentes

- Domaine **manipulation seule** (pas de locomotion bipède — voir J22 pour Helix/GR00T).
- **Recovery / failure modes** encore fragiles : si le robot rate la première saisie, il peut rester bloqué.
- Inférence **cloud-side** souvent (le 3B est lourd pour de l'embarqué).

---

## 6. π0-FAST tokenizer — l'optim qui fait passer le temps réel

Problème identifié 2024 : même avec flow matching, on tokenize la **sortie** action en VLM-style tokens (RT-2 héritage) → encoding lent.

π0-FAST (Pertsch et al., janvier 2025, blog https://www.pi.website/blog/pi0-fast) propose un **tokenizer fréquence-aware** basé sur DCT (Discrete Cosine Transform) :

- L'action chunk H=50 est exprimée dans le **domaine fréquentiel** (basses fréquences = trajectoire grossière, hautes = détails).
- Tokenization vector-quantized des coefficients DCT → moins de tokens, mieux structurés.
- **5× plus rapide** à l'inférence sur le même hardware, qualité préservée.

C'est le genre d'astuce d'ingénieur appliquée qui fait passer un modèle "intéressant en lab" à "déployable @ 50 Hz". À retenir comme illustration : **les optimisations de représentation valent souvent plus qu'un changement d'architecture.**

---

## 7. Comparaison π0 vs OpenVLA (J20) vs Octo (J19)

| Critère | Octo (J19, RSS 2024) | OpenVLA (J20, 2024) | π0 (J21, 2024-2025) |
|---|---|---|---|
| Backbone | Transformer from scratch (93M / 27M) | Llama2 7B + DINOv2/SigLIP | PaliGemma 3B |
| Action head | Diffusion (DDPM) ou MSE | Token-discretized (RT-2 style) | **Flow matching** |
| Pré-entraînement | Open X-Embedded (800k traj.) | Internet text + vision | Internet (PaliGemma) + 10 000h robot |
| Multi-embodiment | Oui (9 robots) | Limité (zoom Bridge/Franka) | **Oui, premier choix** (7 robots, padding) |
| Open / closed | Open weights, MIT | Open weights, code, dataset | **Code partiel, weights π0 disponibles** |
| Fine-tune-friendly | Très (100 démos / 4h GPU) | LoRA bien documenté | LoRA + recipes officielles |
| Spécificité | "Modulaire" — head échangeable | "Le BERT du VLA" | **"Le foundation policy"** |
| Faiblesse | Backbone faible vs 7B+ | Inférence lente (token decode) | Code/dataset partiellement clos |

**Lecture rapide** : les trois sont des ancêtres communs (RT-1/RT-2 lineage, J18) mais incarnent trois philosophies :
- **Octo** = "garder léger, modulaire, open" — pour itérer en lab.
- **OpenVLA** = "réutiliser un LLM massif open, simple, fine-tunable" — pour déployer.
- **π0** = "scale data + flow matching + multi-embodiment" — pour atteindre la generalist policy.

> **Key takeaway #5**
> π0 n'est pas "meilleur" intrinsèquement — il fait des choix d'ingénierie différents. Ce qui le rend marquant en 2024-2025 : combinaison flow matching + multi-embodiment + scale data, et la trajectoire vers π0.5 (open-world) puis vers les systèmes dual System1/System2 (J22).

---

## 8. Architecture comparée DDPM vs flow matching — vue d'ingénieur

Pour bien matérialiser le contraste, retenir cette comparaison côté inférence (cf. code J21 où on l'implémente) :

```python
# DDPM (Diffusion Policy J16) — pseudo
A = randn(H, d_action)
for t in range(T_train, 0, -1):           # T=100 typically
    eps_pred = eps_theta(A, t, context)
    A = scheduler.step(A, eps_pred, t)    # adds noise z each step (SDE)
return A

# Flow matching (π0 J21) — pseudo
A = randn(H, d_action)
dt = 1.0 / N_steps                        # N=5..10
for k in range(N_steps):
    tau = k * dt
    v_pred = v_theta(A, tau, context)
    A = A + dt * v_pred                   # pure Euler step (ODE)
return A
```

Différences observables :

1. **Boucle 10× plus courte** côté flow matching.
2. **Pas de scheduler** complexe (`betas`, `alpha_bar`, `posterior_variance`) — juste `dt`.
3. **Pas de bruit injecté** à l'inférence — déterministe (ODE).
4. **Initialisation identique** (bruit gaussien) → même prior, intuition transférable.

C'est exactement ce qu'on va coder en J21 (`02-code/21-pi0-physical-intelligence.py`).

---

## 9. Synthèse / Mind map du jour

```
π0 (Physical Intelligence 2024)
├─ Architecture
│  ├─ VLM PaliGemma 3B (gelé/partiel)
│  └─ Action expert 300M (flow matching)
├─ Training
│  ├─ 10 000h démos multi-embodiment
│  └─ 7 robots, action space paddé à dim 18
├─ Inférence
│  ├─ ODE Euler 5-10 steps
│  ├─ Action chunk H=50 @ 50 Hz
│  └─ π0-FAST tokenizer (5× plus rapide)
├─ Évolutions
│  ├─ π0.5 → open-world (chambres inconnues, plan hiérarchique)
│  └─ π0-FAST → tokenizer DCT
└─ Position dans l'écosystème
   ├─ vs Octo (J19)     → moins modulaire, plus fort
   ├─ vs OpenVLA (J20)  → flow matching > token decode
   └─ vs DDPM (J16)     → ODE > SDE pour les actions

Trajectoire 2025+ → System1/System2 (J22), foundation policies, deployment
```

---

## 10. Q&A spaced repetition

**Q1.** En une phrase, quelle est la différence fondamentale entre flow matching et DDPM côté inférence ?
**R.** Flow matching résout une **ODE déterministe** en 5-10 steps Euler ; DDPM résout une **SDE stochastique** en 50-100 steps avec scheduler complexe.

**Q2.** Pourquoi π0 utilise un "action expert" séparé du VLM plutôt qu'un seul gros modèle de bout-en-bout ?
**R.** Parce que le VLM (3B) est trop lent à re-rouler à chaque step de l'ODE. On l'amortit : 1 forward pass VLM + 5-10 forward passes du petit action expert (300M).

**Q3.** Comment π0 gère-t-il des robots aux espaces d'action différents (UR5e 7 DoF vs ALOHA 14 DoF) ?
**R.** L'espace d'action est **paddé à dim_max=18**, les dimensions inutiles sont masquées dans la loss. Le modèle apprend implicitement quel sous-espace activer via les indices visuels et un token embodiment optionnel.

**Q4.** Quelle est l'apport principal de π0.5 par rapport à π0 ?
**R.** **Open-world generalization** : transfert zero-shot vers maisons jamais vues, grâce à co-training web + plan hiérarchique en langage naturel + 400h démos en environnements inédits.

**Q5.** Pourquoi π0-FAST est-il important pour le déploiement réel ?
**R.** Il remplace le tokenizer naïf par une représentation **DCT (fréquence)** vector-quantized → 5× plus rapide à l'inférence sans perte de qualité. C'est ce qui fait passer le modèle de "lent en démo" à "déployable @ 50 Hz".

**Q6 (bonus).** En une phrase, place π0 parmi Octo, OpenVLA et Diffusion Policy.
**R.** Diffusion Policy (J16) = recette diffusion mono-tâche locale ; Octo (J19) = generalist transformer modulaire open ; OpenVLA (J20) = LLM 7B + tokens d'action ; π0 = VLM 3B + flow matching + multi-embodiment scale → première foundation policy crédible.

---

## 11. Sources citées (REFERENCES.md)

- **#14** — Black, Brown, Driess, Finn et al. (Physical Intelligence), *"π0: A Vision-Language-Action Flow Model for General Robot Control"*, 2024, https://arxiv.org/abs/2410.24164 ; π0.5 paper https://www.pi.website/download/pi05.pdf ; blog https://www.pi.website/blog/pi0 et π0-FAST https://www.pi.website/blog/pi0-fast. **Source principale du module.**
- **#13** — Kim, Pertsch et al., *"OpenVLA"*, 2024 (comparaison J20).
- **#17** — Octo Model Team, *"Octo"*, RSS 2024 (comparaison J19).
- **#19** — Chi et al., *"Diffusion Policy"*, RSS 2023 (rappel J16, baseline DDPM).
- **#23** — Holderrieth & Erives, MIT 6.S184 (théorie flow matching, J17).

---

## 12. Pour aller plus loin (optionnel)

- Lire le papier π0 (REFERENCES.md #14) en focus sur **§3 (Architecture)** et **§5 (Multi-embodiment evaluation)**.
- Survoler le blog π0-FAST pour l'astuce DCT.
- Code π0 (HuggingFace LeRobot, REFERENCES.md #27) → on le manipulera en J24-J25 capstone.

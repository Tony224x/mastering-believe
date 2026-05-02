# J18 — JEPA + NVIDIA Cosmos : trois paradigmes pour predire le futur d'un robot

> **Objectif du jour** : comparer trois paradigmes de world models — Dreamer (generer les pixels du futur), JEPA (predire dans un espace latent), Cosmos (foundation model entraine sur 20M h de video) — et savoir quand chacun est le bon choix pour la robotique.

---

## 1. Le probleme, en concret

> Un robot Franka tient une eponge dans sa pince. Il est devant un evier. **Question** : que va-t-il se passer si je commande l'action "tourner le robinet" ?
>
> Trois reponses possibles, chacune incarnant un paradigme :
>
> 1. **Dreamer (Hafner 2023)** : "je vais generer les 16 prochaines images RGB 64x64 du futur, pixel par pixel. Tu y verras l'eau qui coule, des reflets, peut-etre du bruit dans le coin de l'image."
> 2. **JEPA (LeCun, V-JEPA 2 2025)** : "je vais predire un **vecteur latent** representant l'etat du monde dans 1 seconde. Ce vecteur n'est pas une image — c'est juste 'evier rempli, eponge encore seche, robinet ouvert'. Pas de pixels."
> 3. **Cosmos (NVIDIA 2025)** : "j'ai vu 20 millions d'heures de videos de gens qui font la vaisselle, je peux generer une video haute-resolution du futur, ou bien fournir mes representations a une policy en aval."
>
> Le robot, lui, n'a pas besoin de voir les pixels exacts pour saisir le savon ensuite. Il a besoin de **savoir que l'evier est rempli**. C'est la cle de l'argument LeCun : **predire des pixels qui ne servent a rien, c'est gaspiller des parametres pour modeliser du bruit**.

Ce module n'a pas pour but d'entrainer un world model de zero (ce serait un cours entier). Il etablit la carte conceptuelle des trois familles, leurs hypotheses, et **quand chacune est le bon outil**.

---

## 2. Paradigme 1 — Dreamer : reconstruire les pixels du futur

**Idee centrale.** Apprendre un modele du monde qui, etant donne `(o_t, a_t)`, predit `o_{t+1}`. Pour entrainer ce modele, on minimise l'**erreur de reconstruction des pixels** : `loss = MSE(decoder(z_{t+1}), o_{t+1})`.

**Architecture (DreamerV3, Hafner 2023)**.

| Bloc | Role |
|---|---|
| **Encoder** `q(z_t \| o_t)` | Image RGB → latent stochastique `z_t` |
| **RSSM** transition | `(z_t, a_t) → z_{t+1}` (Recurrent State-Space Model) |
| **Decoder** | `z_t → o_hat_t` (image reconstruite) |
| **Reward head** | `z_t → r_hat_t` |
| **Actor / Critic** | entraines **dans l'imagination** (rollouts dans le modele, pas dans l'env) |

**Force.** DreamerV3 fait tourner **une seule config** sur 150+ taches (Atari, DMC, Crafter, Minecraft) et resout Minecraft sans curriculum [DreamerV3 Hafner 2023]. L'**imagination** permet d'apprendre l'actor/critic sans collecter des millions d'interactions reelles : le modele genere ses propres rollouts.

**Faiblesse cle.** La loss de reconstruction force le modele a allouer de la capacite pour les **details pixel** : ombres, texture du bois, herbe qui bouge dans le vent. Or, **ces details ne servent presque jamais a la decision**. Si tu veux savoir "ou est la balle", tu n'as pas besoin de savoir "quelle nuance d'orange exactement". Le decoder consomme des parametres pour predire de l'**entropie** (bruit visuel) au lieu de structure.

> **Mnemo** — *Dreamer = generer des pixels parfaits, et apprendre a y survivre.*

---

## 3. Paradigme 2 — JEPA : predire dans l'espace latent (LeCun)

**L'argument LeCun.** Generer des pixels du futur est **mathematiquement impossible** au sens fort : la distribution `p(o_{t+1} | o_t, a_t)` est multimodale (plein de futurs possibles : la branche peut se plier a gauche ou a droite), et la MSE pixel **moyenne** ces modes — produisant des images floues. Pire, **on gaspille** : 99% des pixels sont du bruit pour la decision robotique (texture du carrelage, reflets, etc.).

Solution : **abandonner les pixels**. Predire seulement une **representation abstraite** du futur. C'est la philosophie **Joint Embedding Predictive Architecture** (JEPA).

**Architecture I-JEPA / V-JEPA / V-JEPA 2** [V-JEPA 2 Meta 2025, ref #21].

```
             contexte x_c                    cible x_t (futur ou masque)
                  |                                    |
              encoder f_theta                   encoder f_xi  (EMA de f_theta)
                  |                                    |
                z_c                                  z_t   (figee, stop-gradient)
                  |                                    |
                  +---> predictor g_phi(z_c) -> z_hat_t
                                |
                          loss = ||z_hat_t - z_t||^2     <-- MSE DANS L'ESPACE LATENT
```

Trois ingredients critiques :

1. **Deux encoders**. `f_theta` (context encoder, entraine par gradient) et `f_xi` (target encoder, mis a jour par **EMA** des poids de `f_theta`). L'EMA evite le collapse trivial `z = 0`.
2. **Stop-gradient sur la cible**. Le gradient ne remonte pas dans `f_xi`. C'est la regle d'or de tous les self-supervised "siamese" (BYOL, DINO, MoCo).
3. **Loss dans l'espace latent, jamais sur les pixels**. `loss = MSE(predictor(z_context), z_target)`. **Aucun decoder image**.

**Pourquoi ca marche sans collapse trivial ?** Le target encoder bouge lentement (EMA), il n'est pas optimisable directement, donc on ne peut pas tricher en collapsant `z = const`. Et le predictor doit predire la cible **avant** de la voir — il apprend la structure utile du monde.

**V-JEPA 2 (Meta 2025)** est la specialisation video : 1.2B parametres, entraine sur 1M h de video internet + 62h de data Droid (datasets robotique). Le modele permet **zero-shot pick-and-place** via goal images : on encode `(o_t, o_goal)` dans l'espace latent, on planifie en cherchant l'action qui rapproche `z_{t+1}` de `z_goal`. **Pas de generation pixel.**

> **Key takeaway**
>
> JEPA = **predire une description abstraite du futur, pas le futur lui-meme**. C'est ce que ton cerveau fait quand tu attrapes une balle au vol : tu ne te visualises pas une image photoreale de la trajectoire, tu predis seulement *"elle va arriver dans 0.3s a hauteur d'epaule"*.

---

## 4. Paradigme 3 — Cosmos : foundation model video pour Physical AI (NVIDIA)

**Idee centrale.** Si on a 20 millions d'heures de video physique (humains qui manipulent, robots qui bougent, scenes naturelles), on peut entrainer un **world foundation model** qui sert de base pre-entrainee pour tout pipeline robotique en aval — exactement comme GPT sert de base pour les LLMs.

**Cosmos (Balaji et al., NVIDIA 2025)** [ref #22] propose **deux familles** de world models en open weights :

| Famille | Type | Usage |
|---|---|---|
| **Cosmos-Diffusion** | Diffusion latent | Generation video conditionnee (text/video → video) |
| **Cosmos-Autoregressive** | Token-AR (sur tokens video) | Generation streaming (frame par frame) |

Plus deux composants reutilisables :

- **Cosmos-Tokenizer** : compresseur video discrete et continue (jusqu'a 1024:1) — sert d'**input layer** pour entrainer un world model ou un VLA.
- **Pipeline de curation** : 100M shots filtres, deduplique, scenes physiques pertinentes — meme pipeline utilise pour data GR00T (J22).

**Quel rapport avec la robotique ?** Trois usages typiques :

1. **Synthetic data**. Generer des heures de demonstrations video (pre/post-training data pour VLA). Utilise par GR00T pour 780k trajectoires synthetiques.
2. **Tokenizer reutilisable**. Encoder ses propres videos en tokens compactes, puis entrainer un policy transformer dessus (au lieu de re-entrainer un encodeur ResNet/ViT from scratch).
3. **Foundation backbone**. Fine-tuner Cosmos sur ses propres demos pour predire le futur de son robot — couteux mais possible avec des LoRA.

**Difference cle vs Dreamer**. Dreamer s'entraine **online** sur l'experience du robot (peu de data, beaucoup de gradient). Cosmos est entraine **offline** une fois pour toutes sur internet-scale data (20M h), puis re-utilise. C'est la difference **task-specific RL world model** vs **foundation pretrain**.

**Difference cle vs JEPA**. Cosmos genere **des pixels** (videos). JEPA refuse les pixels. Cosmos parie sur l'echelle ("avec assez de data, generer un peu de bruit visuel n'est pas grave"). JEPA parie sur la structure ("meme avec infinite data, gaspiller des parametres a modeliser le bruit, c'est sub-optimal").

---

## 5. Comparaison synthetique des trois paradigmes

| Critere | Dreamer V1-V3 | V-JEPA 2 | NVIDIA Cosmos |
|---|---|---|---|
| **Loss** | Reconstruction pixel + reward | MSE espace latent | Diffusion / token-AR sur tokens video |
| **Echelle d'entrainement** | Online (rollouts du robot) | 1M h video + 62h Droid | 20M h video internet curee |
| **Output** | Image reconstruite + valeur | Vecteur latent abstrait | Video haute-res |
| **Generation pixel** | Oui (decoder explicit) | Non (philosophie LeCun) | Oui (foundation video) |
| **Use case principal** | RL avec imagination, peu de data | Self-supervised pretraining + planning latent | Synthetic data + tokenizer + backbone |
| **Compute** | Modere (1 GPU jours) | Lourd (multi-GPU semaines) | Tres lourd (cluster mois) |
| **Rapport `param/decision-utile`** | Faible (gaspille sur pixels) | Eleve (focus latent) | Moyen (pixels mais transferable) |
| **Source** | Hafner 2023, ref #20 | Meta FAIR 2025, ref #21 | NVIDIA 2025, ref #22 |

---

## 6. Pourquoi LeCun pense que pixel-level generative est une impasse pour la robotique

L'argument est articule dans plusieurs talks et papiers (LeCun 2022 "A Path Towards Autonomous Machine Intelligence", reaffirme dans le blog V-JEPA 2 [ref #21]) :

1. **Le futur est multimodal**. Une feuille au vent peut se plier a gauche **ou** a droite. Predire la moyenne pixel = image floue. Predire seulement le mode dominant = mode collapse. Un encoder latent peut au contraire encoder une distribution sur les abstractions ("la feuille bouge un peu") sans devoir trancher pixel par pixel.
2. **99% des pixels sont du bruit pour la decision**. Pour saisir une tasse, peu importe la nuance exacte de bois sur la table. Pour stabiliser un humanoide, peu importe la trame du tapis. Allouer des parametres a generer ce bruit, c'est dilapider la capacite du modele.
3. **Les representations utiles sont structurelles, pas perceptuelles**. Un humain qui regarde une scene n'encode pas une image en RAM ; il encode `{personne assise, tasse, table, etc.}`. C'est exactement ce que fait l'encoder JEPA.
4. **Empiriquement**. V-JEPA 2 atteint le SOTA sur Something-Something v2 (action recognition) et fait du zero-shot manipulation, **sans jamais avoir genere une image**. Si pixels etaient indispensables, ce ne serait pas possible.

**Contre-argument** (NVIDIA, OpenAI Sora-style). Avec **assez** de data et de compute, generer des pixels n'est plus un cout, c'est un signal d'apprentissage gratuit. Cosmos parie sur cet argument scaling. Le verdict 2026 n'est pas tranche.

---

## 7. Quand utiliser quoi en pratique ?

| Si ton besoin est... | Choix recommande |
|---|---|
| Apprendre une policy RL avec peu d'interactions reelles | **Dreamer** (imagination = data efficiency) |
| Pre-trainer un encodeur visuel pour ton VLA, sans contraindre la decision | **V-JEPA 2** (latent representation, transfert downstream) |
| Generer 100k trajectoires video pour data augmentation | **Cosmos-Diffusion** (synthetic data scale) |
| Tokenizer ta propre video pour un transformer policy | **Cosmos-Tokenizer** |
| Planifier en latent (goal-image conditioned) | **V-JEPA 2** (zero-shot demontre) |
| Faire un robot picker dans un entrepot logistique | Pas un world model unique : un **VLA** (J19-J22) eventuellement avec encoder JEPA ou Cosmos en backbone |

**Cle pour le capstone robotique du domaine.** Le capstone de ce cours (J24-J28) utilise une **Diffusion Policy** [ref #19], qui n'est pas un world model — c'est une policy generative directement sur les actions, pas sur le futur du monde. Le world model et la diffusion policy sont **deux familles distinctes** : world model = "predire le futur" ; diffusion policy = "predire l'action multimodale".

---

## 8. Conclusion en 4 phrases

1. **Dreamer** apprend un world model en reconstruisant les pixels du futur, et entraine un actor/critic dans son imagination — efficace en data, mais gaspille des parametres a modeliser le bruit visuel [Hafner 2023].
2. **JEPA** (LeCun) refuse les pixels et predit seulement dans l'espace latent : V-JEPA 2 montre que le pretraining self-supervised sur video peut atteindre du zero-shot manipulation **sans jamais decoder une image** [V-JEPA 2 2025, ref #21].
3. **Cosmos** (NVIDIA) parie sur la scale : 20M h de video physique, foundation models open-weight (diffusion + AR), tokenizers, pipeline de curation — utilises pour la synthetic data de GR00T et comme backbone reutilisable [Cosmos 2025, ref #22].
4. Les trois sont complementaires : **Dreamer pour le RL data-efficient**, **JEPA pour le pretraining structure**, **Cosmos pour la synthetic data scale et tokenizers**. Le pari pixels-vs-latent est encore ouvert pour 2026.

---

## Flash cards (spaced repetition)

**Q1.** Quelle est la difference centrale, en une phrase, entre Dreamer et JEPA ?
**R1.** Dreamer minimise une **erreur de reconstruction pixel** (MSE entre image generee et image reelle, via un decoder explicite). JEPA minimise une **MSE dans l'espace latent** entre la prediction du predictor et un target encoder (pas de decoder, jamais de pixels).

**Q2.** Pourquoi JEPA n'a-t-il pas un collapse trivial `z = 0` ?
**R2.** Le **target encoder est mis a jour par EMA** des poids du context encoder (et un stop-gradient l'empeche d'etre optimise directement). Le predictor doit predire la cible avant de la voir. Cette asymetrie evite la solution triviale et force l'encoder a apprendre une structure utile.

**Q3.** Citer 3 utilisations concretes de NVIDIA Cosmos pour la robotique.
**R3.** (i) Generation de **synthetic data** (videos demonstrations) — utilise par GR00T pour 780k trajectoires ; (ii) **Cosmos-Tokenizer** comme input layer reutilisable pour un policy transformer ; (iii) **Foundation backbone** pre-entraine, fine-tunable via LoRA pour predire le futur d'un robot specifique.

**Q4.** Quel est l'argument principal de LeCun contre la generation pixel-level pour la robotique ?
**R4.** (i) Le futur est multimodal et la MSE pixel **moyenne les modes** → images floues ; (ii) **99% des pixels sont du bruit** pour la decision robotique (texture du carrelage, ombres) — gaspiller des parametres pour les modeliser dilapide la capacite du modele. JEPA encode au contraire seulement la **structure utile** du monde dans un espace latent abstrait.

**Q5.** Sur quelle echelle de data Cosmos est-il entraine, et qu'est-ce que ca change vs Dreamer ?
**R5.** **20M heures de video physique cure**. Cosmos est un **foundation model offline** entraine une fois pour toutes (echelle internet), reutilisable comme backbone — analogue de GPT pour la robotique. Dreamer est entraine **online** sur l'experience du robot lui-meme (peu de data, beaucoup de gradient task-specific).

---

## Sources

- [V-JEPA 2 Meta 2025] — *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* — REFERENCES.md #21 — arxiv 2506.09985, blog AI Meta. Source principale pour la philosophie JEPA / LeCun et l'application robotique zero-shot.
- [NVIDIA Cosmos 2025] — *Cosmos World Foundation Model Platform for Physical AI* — REFERENCES.md #22 — arxiv 2501.03575, github nvidia-cosmos. Source principale pour le foundation model video, tokenizers, pipeline data, scale 20M h.
- [DreamerV3 Hafner 2023] — *Mastering Diverse Domains through World Models* — REFERENCES.md #20 — arxiv 2301.04104. Reference pour le paradigme generation pixel + imagination latente.

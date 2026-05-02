# J22 — Frontier humanoid : GR00T N1, Helix, LBM TRI

> **Objectif du jour** : comprendre le pattern dual-system System1/System2 qui structure les VLAs humanoides 2025-2026 (GR00T N1, Helix, LBM), pourquoi l'industrie converge dessus, et savoir le situer face a OpenVLA / pi0 / Octo (J19-J21).

---

## Mise en bouche : un robot Figure 02 plie un colis

Imagine un humanoide Figure 02 deploye dans un entrepot logistique. Un superviseur lui dit en francais : *"prends la boite rouge sur l'etagere 3, scanne le code-barres, depose-la dans le bac rotatif"*. Le robot a 35 articulations a piloter (bras + mains + torse + tete) a 200 Hz, soit une commande toutes les 5 ms.

Comment tient-il les deux ordres de grandeur :
- **Comprendre** une instruction langagiere ambigue, raisonner sur la scene, decider la sequence d'actions
- **Executer** un controle continu fin a 200 Hz sur 35 DoF, reagir aux contacts, rattraper une glissade

Reponse industrie 2025 : **on ne le fait pas avec un seul reseau**. On utilise deux modeles couples :
- **System2** : un VLM (~7B parametres) qui tourne lentement (7-9 Hz), regarde la scene + l'instruction, produit une representation latente "what to do"
- **System1** : un reseau leger (~80M-300M parametres) qui tourne vite (200 Hz), prend la latente System2 + les observations courantes, sort les couples articulaires

C'est exactement ce qu'a fait Daniel Kahneman pour decrire la cognition humaine dans *Thinking, Fast and Slow* (2011) : System 1 = rapide/intuitif/reflexe, System 2 = lent/deliberatif/raisonne. Les humanoides 2025 font la meme chose avec des reseaux differents, et c'est devenu **le pattern dominant** chez NVIDIA (GR00T N1), Figure (Helix) et Toyota (LBM).

> **Cle** : la convergence n'est pas un hasard pedagogique. Elle vient d'une contrainte physique : aucun GPU 2026 ne fait tourner un VLM 7B a 200 Hz. Il faut decoupler.

---

## 1. Le probleme : pourquoi un seul reseau ne suffit pas

### 1.1 Contrainte de latence

| Tache | Frequence requise | Latence budget |
|-------|---|---|
| Raisonnement haut-niveau (langage, scene) | 5-10 Hz | 100-200 ms |
| Controle articulaire continu | 100-1000 Hz | 1-10 ms |
| Reaction tactile fine (slip, contact) | 500-1000 Hz | 1-2 ms |

Un VLM type Llama 7B fait au mieux 30-50 tokens/s sur un GPU H100, soit ~7 Hz pour produire une action chunk de quelques tokens. Impossible de l'utiliser pour fermer la boucle de controle moteur.

### 1.2 Contrainte de generalisation

Les VLAs monolithiques (RT-2, OpenVLA, pi0 — J19-J21) traitent tout au meme rythme : ils tokenizent les actions et les decodent par le modele de langage. Resultat : on plafonne a quelques Hz de controle effectif, ce qui interdit les mains dexterees rapides et les humanoides whole-body.

Constat industrie 2024-2025 : **le controle continu et le raisonnement langagier vivent sur deux echelles de temps incompatibles**. La solution est architecturale.

---

## 2. Le pattern dual-system

### 2.1 Anatomie generique

```
        +---------------------------+
        |  Instruction langage      |
        |  Vue camera (low-rate)    |
        +-----------+---------------+
                    v
        +---------------------------+
        |  System2 : VLM (~7B)      |   7-9 Hz
        |  - Scene + langage        |   raisonne
        |  - Plan / sous-but        |
        +-----------+---------------+
              latent embedding (z_t)
                    v
        +---------------------------+
        |  Etat proprioceptif       |
        |  Contacts, depth          |
        +-----------+---------------+
                    v
        +---------------------------+
        |  System1 : action head    |   200 Hz
        |  (~80M-300M, diffusion    |   reactif
        |   ou flow matching)       |
        +-----------+---------------+
                    v
              actions tau (joints)
```

System2 conditionne System1 via une representation continue (embedding, tokens, vecteur de but). System1 n'a JAMAIS a comprendre le langage : il consomme des features deja distillees.

### 2.2 Ce que ca change vs VLA monolithique

| Dimension | VLA mono (OpenVLA, pi0) | Dual-system (GR00T, Helix) |
|---|---|---|
| Frequence controle effective | 5-30 Hz | 100-200 Hz |
| DoF realistes | 7-14 (single arm) | 35+ (humanoide complet) |
| Reasoning multi-etapes | bon | tres bon (System2 specialise) |
| Cout entrainement | un modele a re-finetuner | deux modeles, possible swap independant |
| Latence par action | ~30-200 ms | 5-10 ms (System1 seul tourne en boucle serree) |

---

## 3. GR00T N1 — NVIDIA, mars 2025

**Reference** : REFERENCES.md #15 — Bjorck et al., *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, arXiv 2503.14734. Repo : https://github.com/NVIDIA/Isaac-GR00T.

### 3.1 Architecture

- **System2** : Eagle-2 VLM (~2B parametres). Image + instruction → tokens latents.
- **System1** : Diffusion Transformer (DiT) + cross-attention sur les tokens System2. Sort des chunks d'actions de 16 pas a ~120 Hz.
- **Embodiment-agnostic** : memes poids fonctionnent sur Fourier GR-1, 1X Neo, Apptronik Apollo apres fine-tune leger.

### 3.2 Innovation principale : la donnee synthetique massive

GR00T N1 est entraine sur un mix tres deliberatif :
- **88h de teleoperation reelle** sur GR-1
- **780 000 trajectoires synthetiques** generees dans NVIDIA Isaac Sim + Cosmos (J18, J23)
- **Videos humaines** pour pre-training perception (egocentric data)

Le ratio sim/reel ~9000:1 montre la these NVIDIA : **on ne resoudra pas la robotique humanoide sans pipeline de donnees synthetiques industriel**. Le bottleneck n'est plus l'algorithme mais le data engine.

### 3.3 Pourquoi c'est ouvert

NVIDIA publie poids + code (Apache 2.0) parce que leur business est l'infra (GPU + Isaac Lab + Cosmos), pas le modele. Strategie : devenir le standard d'environnement, comme CUDA pour le deep learning.

---

## 4. Helix — Figure AI, fevrier 2025

**Reference** : REFERENCES.md #16 — Figure AI blog, *Helix: A Vision-Language-Action Model for Generalist Humanoid Control* (https://www.figure.ai/news/helix), suivi de *Helix Logistics* (juin 2025, https://www.figure.ai/news/helix-logistics).

### 4.1 Architecture

- **System2** : VLM ~7B, tourne a 7-9 Hz sur GPU embarque
- **System1** : ~80M parametres, **200 Hz**, sort directement les couples (pas de tokenization d'action)
- **35 DoF whole-body** : bras gauche, bras droit, mains 5-doigts, torse, tete, base mobile

Specificite : **Helix produit un signal continu** (pas des chunks de tokens). System2 envoie un latent vector, System1 le traite via un MLP/transformer leger en regression directe.

### 4.2 Multi-robot collaboration

Demo phare fevrier 2025 : **deux robots Helix differents** se coordonnent pour ranger des courses, en suivant une instruction unique : *"put away the groceries"*. Ils n'ont jamais vu ces objets avant. Important parce que cela demontre :
- **Generalisation zero-shot** (le scenario a ete tourne sur des courses fraiches)
- **Coordination implicite** sans reward sharing ni planificateur central — chaque robot a son propre Helix, ils s'observent mutuellement via la camera

### 4.3 Helix Logistics : faster than human demonstrators

Mai-juin 2025 : Figure deploie Helix dans un vrai entrepot. Resultat publie : **les robots executent les taches plus vite que les teleoperateurs qui les ont demontrees**. C'est la bascule "generalisation > imitation pure". Resonance directe avec **shared/logistics-context.md** : c'est exactement le scenario LogiSim/FleetSim que les capstones du domaine ciblent (J28).

---

## 5. LBM — Toyota Research Institute

**Reference** : REFERENCES.md #18 — TRI, *A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*, https://toyotaresearchinstitute.github.io/lbm1/.

### 5.1 Positionnement

LBM (Large Behavior Models) est la lignee TRI inspiree directement de Diffusion Policy (J16 — Chi et al., RSS 2023, papier issu du meme labo). Approche moins "humanoide whole-body" que Helix, plus "manipulation bi-manuelle dexteree industrielle".

- **Donnees** : 1700 heures de teleoperation cumulees (un des plus gros datasets industriels documentes)
- **Architecture** : ViT multi-vues + transformer denoiser (descendance directe de Diffusion Policy avec scaling)
- **Deux echelles** : un modele "small" (specialiste) vs "LBM" (generaliste pre-entraine)

### 5.2 Resultat cle : 80% moins de donnees sur tache nouvelle

Etude TRI 2024-2025 : pour atteindre un meme niveau de performance sur une tache jamais vue, le modele "LBM pre-entraine puis fine-tune" a besoin de **~80% moins de demonstrations** que le modele "from-scratch". C'est l'argument scaling : **plus on accumule de donnees multi-taches, mieux on transfere**, exactement comme pour les LLMs.

### 5.3 Le dual-system chez TRI

LBM n'utilise pas explicitement la nomenclature System1/System2, mais l'architecture est equivalente :
- Un encodeur perception/langage lent
- Un denoiser diffusion rapide qui sort des chunks d'actions a haute frequence
- Le decoupage temporel est equivalent a System2 → System1

---

## 6. Convergence industry 2025 : carte mentale

```
  CONTROLE CLASSIQUE          IL SIMPLE          DIFFUSION POLICY (J16)
  PID, LQR, MPC                BC/DAgger          Chi 2023 — un seul reseau
  (J6)                         (J13)              denoiser
                                                       |
                                                       v
                          VLA monolithique  J19-J21
                          RT-1 / RT-2 / Octo / OpenVLA / pi0
                          un transformer, action tokens
                                  |
                                  | bottleneck latence + DoF
                                  v
                       DUAL-SYSTEM 2025-2026  (J22)
                       System2 (VLM lent) + System1 (head rapide)
                              /        |          \
                       GR00T N1     Helix         LBM (TRI)
                       NVIDIA       Figure        Toyota
                       2B+DiT       7B+80M        ViT+denoiser
                       sim-data     35DoF/200Hz   1700h reel
```

### 6.1 Ce qui caracterise la frontiere 2025-2026

1. **Decouplage temporel obligatoire** : impossible de faire tenir raisonnement + controle 200 Hz dans un seul reseau.
2. **Embodiment-agnostic** : un modele, plusieurs robots (multi-platform GR00T, multi-robot Helix).
3. **Donnees synthetiques de masse** : Cosmos + Isaac Lab cote NVIDIA, pipelines proprietaires cote Figure/TRI.
4. **Open-weights selectif** : NVIDIA ouvre (infra business), Figure/TRI gardent (produit business).
5. **Deploiement industriel reel** : Helix Logistics, Apptronik chez Mercedes, 1X chez 1X. On sort du lab.

### 6.2 Ce qui reste ouvert

- **Reaction tactile haute-frequence** (>500 Hz) reste mal traitee — System1 n'est pas encore assez rapide pour les boucles de slip detection fines.
- **Long-horizon reasoning** (taches multi-minutes avec sous-buts emergents) : System2 hallucine encore.
- **Failure modes** : un humanoide qui tombe, c'est 100k$ de robot. Aucun papier 2025 ne publie de protocole de safety formel.
- **Cout energetique** : 7B + 80M tournant en boucle 200 Hz sur batterie reste un probleme matthematique. La Figure 02 a une autonomie de 5h.

---

## 7. Quand utiliser quoi (cheat sheet praticien)

| Situation | Choix | Pourquoi |
|---|---|---|
| Bras unique, frequence ~10 Hz, GPU consumer | OpenVLA (J20) | Mono, fine-tunable, cheap |
| Bras unique, multi-task generalist, dexterity moyenne | pi0 / pi0.5 (J21) | Flow matching efficace |
| Manipulation bi-manuelle, beaucoup de demos | LBM-style (TRI) | Diffusion policy scaled |
| Humanoide whole-body, 30+ DoF, 100+ Hz controle | Helix-style ou GR00T N1 | Dual-system obligatoire |
| Recherche, data sim massive, GPU NVIDIA | GR00T N1 (open) | Ecosysteme Isaac/Cosmos |
| Capstone perso, GPU consumer | Diffusion Policy (J16) sur PushT | Le seul reproduisible easy |

---

## 8. Lien avec le capstone fil-rouge (J24-J28)

Le capstone du domaine implemente Diffusion Policy (Chi 2023) — c'est-a-dire l'**ancetre direct** de System1 chez tout le monde. Comprendre J22 te donne :
- La perspective : tu n'implementes pas System1+System2, tu implementes le brique fondamentale qui sert de System1 dans GR00T/Helix/LBM.
- L'angle d'attaque pour la suite : si tu veux pousser au-dela du capstone, tu colles un VLM (Llama 3 / Qwen-VL) en amont qui produit un embedding de but, et tu conditionnes le denoiser dessus. C'est exactement la trajectoire industrielle.

> **Key takeaway**
>
> > Le pattern dual-system (System2 VLM lent + System1 action head rapide) est la **convergence industrielle 2025-2026** pour les humanoides. GR00T N1 (NVIDIA), Helix (Figure) et LBM (TRI) l'adoptent independamment parce que la contrainte de latence (200 Hz controle vs 7 Hz raisonnement) impose de decoupler. Le verrou de la prochaine vague n'est plus l'algorithme mais le **pipeline de donnees synthetiques** (Cosmos + Isaac).

---

## Spaced repetition — Q&A

**Q1.** Pourquoi un VLM monolithique 7B ne peut pas piloter un humanoide a 200 Hz ?

> Parce que meme sur un H100, un VLM 7B sort au mieux 30-50 tokens/s soit ~7 Hz, et le controle articulaire d'un humanoide whole-body (35 DoF) demande 100-200 Hz pour la stabilite et la reactivite contact. La difference est de deux ordres de grandeur — il faut decoupler reasoning lent et controle rapide.

**Q2.** Quelle est la difference d'objectif entre System1 et System2 dans le pattern dual-system humanoide ?

> System2 (VLM ~7B, lent) consomme l'instruction langagiere et la scene visuelle pour produire une representation latente du sous-but. System1 (~80M-300M, rapide) consomme cette latente + les observations proprioceptives haute frequence et sort les couples articulaires a 200 Hz. System1 n'a jamais a comprendre le langage.

**Q3.** Pourquoi NVIDIA ouvre GR00T N1 (poids + code) alors que Figure ne publie pas Helix ?

> Le business de NVIDIA est l'infrastructure (GPU, Isaac Lab, Cosmos) : faire de GR00T un standard ouvert encourage l'adoption de l'ecosysteme NVIDIA. Le business de Figure est le produit fini (humanoide deploye en logistique) : Helix est un avantage competitif a proteger. Open-source vs closed depend du business model, pas du merite technique.

**Q4.** Quelle est l'innovation principale de GR00T N1 cote donnees ?

> Le ratio synthetique/reel : 88h de teleoperation reelle vs 780 000 trajectoires generees en simulation (NVIDIA Isaac Sim + Cosmos). C'est ~9000:1, et cela demontre que le scaling humanoide passe par un pipeline de donnees synthetiques industrialise, pas par la teleoperation pure. Le bottleneck devient le data engine.

**Q5.** Comment LBM (TRI) demontre l'effet "scaling foundation model" pour la robotique ?

> En montrant qu'un modele LBM pre-entraine sur 1700h multi-taches puis fine-tune sur une tache nouvelle a besoin de **~80% moins de demonstrations** qu'un modele entraine from-scratch. C'est l'analogue robotique du transfer learning des LLMs : plus on accumule de donnees multi-taches diverses, mieux on transfere — y compris en manipulation dexteree.

---

## Sources citees

- REFERENCES.md #15 — GR00T N1 (NVIDIA Research, mars 2025) — https://arxiv.org/abs/2503.14734
- REFERENCES.md #16 — Helix + Helix Logistics (Figure AI, fevrier-juin 2025) — https://www.figure.ai/news/helix
- REFERENCES.md #18 — Toyota Research Institute LBM — https://toyotaresearchinstitute.github.io/lbm1/
- REFERENCES.md #14 (rappel pi0/pi0.5 — pour contraste, vu en J21)
- REFERENCES.md #13 (rappel OpenVLA — pour contraste, vu en J20)

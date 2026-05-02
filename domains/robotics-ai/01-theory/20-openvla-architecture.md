# J20 — OpenVLA : architecture detaillee + fine-tuning LoRA

> Module-jour 20 (45-60 min de lecture). Source principale : REFERENCES.md #13 (OpenVLA, Kim, Pertsch et al., 2024 — https://arxiv.org/abs/2406.09246 + repo https://github.com/openvla/openvla).
> Prerequis : J18 (RT-1/RT-2/Octo, action tokenization), J15 (diffusion/flow matching), notions Transformer.

---

## 1. Pourquoi ce module — l'exemple concret avant la theorie

Tu veux qu'un bras Franka prenne un mug rouge sur ton bureau quand tu lui dis *"pick up the red mug"*. En 2023, la seule policy qui sait faire ca proprement est **RT-2-X** (Google, 55 milliards de parametres, fermee, GPU TPU obligatoires). En juin 2024, l'equipe Stanford/Berkeley/TRI sort **OpenVLA** :

- **7B parametres** (~8x plus petit que RT-2-X)
- **+16,5 points** de success rate absolu sur 29 taches WidowX/Google Robot
- **Open-weights, code MIT-license**, fine-tunable sur **1 seule RTX 4090** via LoRA
- Inference quantize 4-bit qui tient en **~7 GB VRAM**

Le punchline : **architecture > scale brute**. OpenVLA combine trois ingredients deja matures (Llama2 7B, DINOv2, SigLIP) avec une recette de tokenization d'actions sobre. C'est devenu la baseline open de reference pour 2024-2026 et le point d'entree obligatoire avant de plonger dans pi0 (J21) ou GR00T (J22).

Ton objectif aujourd'hui : savoir dessiner l'architecture au tableau, comprendre **ou** on insere LoRA pour un fine-tuning task-specific, et savoir quand cette approche est insuffisante.

---

## 2. Anatomie d'OpenVLA — les 4 blocs

### 2.1 Vue d'ensemble (un diagramme dans la tete)

```
   image (224x224)
         |
   +----[DINOv2 ViT-L]----+         "pick up the red mug"
   |                      |                |
   +----[SigLIP ViT-L]----+         [Llama2 tokenizer]
         |                                 |
   concat patches (256 tokens, 2048-d)     |
         |                                 |
         +-----> [projector MLP] -----> Llama2 7B decoder ---> 7 action tokens
                                                                   |
                                                  detokenize (256 bins per dim)
                                                                   |
                                              action 7-D : delta_x,y,z, delta_R(rpy), gripper
```

OpenVLA est un **decoder-only Transformer** (Llama2) qui voit du texte ET de l'image, et qui sort 7 tokens d'action, exactement comme il sortirait du texte. C'est la grande astuce : **les actions sont du texte** dans le vocabulaire du LLM.

### 2.2 Le bloc vision : DINOv2 + SigLIP en parallele (PAS en serie)

Beaucoup de VLMs n'ont qu'un encodeur vision (CLIP, SigLIP, ou DINOv2). OpenVLA en met **deux en parallele** :

- **DINOv2 ViT-L/14** (Meta, self-supervised, ~300M params) : excellent sur la **geometrie spatiale** (correspondance pixel, profondeur implicite).
- **SigLIP ViT-SO400M/14** (Google, contrastif image-texte, ~400M params) : excellent sur la **semantique** (reconnaitre "mug rouge" vs "tasse bleue").

Pour chaque image 224x224, on extrait 256 patches (16x16 patches de 14x14 pixels). On obtient :

- DINOv2 : `(256, 1024)` features
- SigLIP : `(256, 1152)` features

On les **concatene par patch** (axe feature) : `(256, 2176)`. Un MLP projector reduit a `(256, 4096)` (la dimension cachee de Llama2 7B).

> **Pourquoi deux encodeurs ?** Ablation du papier (Table 5) : DINOv2-only baisse de 4 points, SigLIP-only baisse de 3 points sur les benchmarks BridgeData/Google Robot. Les deux sont **complementaires**, pas redondants.

### 2.3 Le LLM : Llama2 7B base (pas Instruct)

Apres le bloc vision, on prepend les 256 tokens visuels devant les tokens de prompt texte. Llama2 traite tout comme une seule sequence. **Choix critique** : Llama2 **base** (pas chat/instruct) parce qu'on ne veut pas de comportement RLHF qui pourrait refuser des instructions ("I'm sorry, I can't help with that"). On veut un predicteur de tokens brut.

Le contexte typique : ~280 tokens (vision) + ~10-20 tokens (instruction langage) + 7 tokens (action). Tres court — Llama2 est largement sous-utilise en termes de contexte, ce qui laisse de la marge pour conditionner sur l'historique d'observations dans des variantes futures.

### 2.4 L'action head : tokenization discrete a 256 bins

C'est l'heritage de RT-2. Une action 7-D (translation 3-D, rotation rpy 3-D, gripper 1-D) est **discretisee** :

- Chaque dimension est binnee en **256 valeurs** (8 bits) selon les quantiles 1%-99% du dataset Open X-Embodiment.
- Les 256 bins reutilisent les 256 derniers tokens du vocabulaire Llama2 (jamais utilises dans le langage naturel anglais).
- Predire une action = predire 7 tokens autoregressivement.

> **Concretement** : si Llama2 sort le token `bin_42` pour la dimension `delta_x`, on regarde la table de bins et on retrouve `delta_x ≈ +0.012 m`.

**Pourquoi ce design ?** Pas de tete specialisee, pas de loss MSE, **pas de modification d'architecture** vs Llama2 standard. La cross-entropy classique suffit. Le LLM apprend ses actions comme il a appris le mot "the".

### Key takeaway encadre

```
+---------------------------------------------------------------+
| OpenVLA = Llama2 7B + (DINOv2 // SigLIP) + tokens d'action.   |
| Pas de tete custom : les actions sont DU TEXTE.               |
| 7B bat 55B parce que les encodeurs vision sont mieux choisis  |
| (DINOv2 spatial + SigLIP semantique en parallele).            |
+---------------------------------------------------------------+
```

---

## 3. Pourquoi 7B bat RT-2-X 55B — l'analyse fine

C'est la question centrale du papier. Quatre hypotheses (toutes documentees Table 4-5 du papier OpenVLA, REFERENCES.md #13).

### 3.1 Donnees : Open X-Embodiment, mais bien curees

OpenVLA s'entraine sur **970k trajectoires** d'Open X-Embodiment, RT-2-X sur ~1.4M. **Moins de donnees, plus propres** : OpenVLA filtre agressivement les datasets bruites, garde 23 sous-ensembles (vs ~50 chez RT-2-X). Lecon : **la qualite > quantite** des que tu as franchi le seuil critique (~500k traj).

### 3.2 Encodeurs vision modernes vs RT-2-X

RT-2-X utilise PaLI-X / PaLM-E avec un encodeur ViT-22B issu de pretraining classification. OpenVLA utilise **DINOv2 (self-supervised)** + **SigLIP (contrastif moderne)**. SigLIP est **strictement superieur a CLIP** sur les benchmarks (Zhai 2023), et DINOv2 capte la geometrie 3D que CLIP rate completement.

Plus precisement : un VLA a besoin de localiser des objets dans l'espace ("le mug **rouge** a **gauche**"), pas seulement de les classifier. DINOv2 a ete pretraine specifiquement sur des taches dense (depth, segmentation) — c'est exactement ce dont un robot a besoin.

### 3.3 LLM choisi pour la genericite, pas la taille

PaLI-X 55B a beaucoup de capacite linguistique inutilisee (poemes, code, math). Pour un VLA, on veut un LLM qui :
- Sait suivre des instructions courtes ("pick up X")
- Predit bien la sequence suivante de tokens
- N'a pas ete RLHF-isolated

Llama2 7B coche ces 3 cases avec **8x moins de params**. Le surplus de PaLI-X 55B n'aide pas la tache action — il ralentit l'inference (15 Hz pour OpenVLA quantize, ~3 Hz pour RT-2-X) sans gain.

### 3.4 Recette d'entrainement : full finetuning end-to-end

OpenVLA fait du **full finetuning** des trois blocs (vision, projector, LLM) en meme temps, sur 64 A100 pendant 14 jours. RT-2 fait du **co-fine-tuning** (alterne batches robotique/web). Le full finetuning est plus simple, converge mieux quand les donnees robotique sont en quantite suffisante.

---

## 4. Fine-tuning LoRA — ou et comment

### 4.1 Le besoin

Tu as OpenVLA pretraine. Tu veux qu'il maitrise une tache specifique : **vider un panier de linge dans une machine a laver** (jamais vue dans Open X-Embodiment). Full finetuning sur RTX 4090 ? **Impossible**, 7B params en fp16 = 14 GB juste pour les poids, plus l'optimizer (Adam = 28 GB de moments), plus les activations. Tu as besoin de ~80 GB VRAM.

**LoRA** (Low-Rank Adaptation, Hu 2021) fige le modele entier et apprend uniquement de **petites matrices low-rank** sur certaines couches. Le papier OpenVLA (section 5, REFERENCES.md #13) montre que LoRA r=32 sur les couches `q_proj` et `v_proj` du Llama2 atteint **97% de la performance du full finetuning** avec **27x moins de params trainables**.

### 4.2 Mathematiques en 30 secondes

Une couche linear a un poids `W` de shape `(d_out, d_in)`. Pendant LoRA :

```
W_effective = W + alpha/r * B @ A
```

ou :
- `A` est `(r, d_in)`, initialisee Gaussian
- `B` est `(d_out, r)`, initialisee a zero (donc au depart `BA = 0` et le modele est intact)
- `r` est le rang (typiquement 8, 16, 32, 64)
- `alpha` est un facteur d'echelle (souvent `alpha = 2*r`)
- **Seules `A` et `B` recoivent des gradients**, `W` est frozen

Pour Llama2 7B, `q_proj` et `v_proj` ont `d_in = d_out = 4096`. Avec `r=32` :
- Params LoRA par couche : `2 * 32 * 4096 = 262 144` (vs `4096*4096 = 16M` pour `W`)
- Llama2 7B a 32 couches, donc 32 * 2 (q et v) = 64 cibles
- Total LoRA params : `64 * 262k = 16,7M` (vs 7B = **0,24% du modele**)

### 4.3 Ou inserer LoRA dans OpenVLA ?

| Bloc | Faut-il y mettre LoRA ? | Pourquoi |
|---|---|---|
| DINOv2 | Non | Deja generaliste, geler |
| SigLIP | Non | Idem |
| Projector MLP | Optionnel | Petit (~10M params), peut etre full-trained |
| Llama2 q_proj, v_proj | **Oui** | Cibles canoniques, recette du papier original LoRA |
| Llama2 mlp gate/up/down | Optionnel | Augmente l'expressivite, +50% de params LoRA |
| Action vocabulary embedding | Possible | Si la tache change le distribution d'actions |

**Recette safe par defaut** (donnee dans le repo openvla) : LoRA r=32, alpha=16, dropout=0.05, sur `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Tu peux tenir sur **24 GB VRAM** (RTX 4090) avec quantization 4-bit du base + LoRA fp16.

### 4.4 Quantization 4-bit pour le deploiement

Une fois fine-tune, tu veux deployer sur un robot dont l'embedded GPU est une RTX 3060 (12 GB). OpenVLA supporte **bitsandbytes 4-bit (NF4)** : chaque poids 16-bit devient 4-bit (4x compression).

- **Memoire** : 7B * 0.5 octet = **3.5 GB de poids** + ~2-3 GB activations = tient en 6-7 GB.
- **Latence** : ~5-7 Hz sur RTX 4090 quantize vs ~15 Hz fp16. Pour du contrôle 5-10 Hz (manipulation lente), c'est suffisant.
- **Cout en performance** : -1 a -3 points de success rate (Table 8 du papier). Acceptable.

> **Convention** : tu fine-tunes en bfloat16 + LoRA, puis tu **mergeais LoRA dans le base** (`peft.merge_and_unload()`), puis tu quantizes en NF4 pour l'inference. Jamais l'inverse.

---

## 5. Limitations connues — ce qu'OpenVLA ne fait PAS

C'est **autant** important que les forces. Si tu deploies OpenVLA sans connaitre ses bornes, tu vas debugguer pendant 3 semaines.

### 5.1 Single-arm uniquement

Open X-Embodiment est **majoritairement** single-arm (WidowX 250, Google Robot, Franka). OpenVLA produit donc des actions 7-D pour **un seul bras**. Les bimanual (ALOHA, Bimanual UR5) sont sous-representees, et OpenVLA generalise mal sur ces embodiements. Pour du bimanual, regarde **pi0** (J21) qui a un design multi-embodiment des le depart.

### 5.2 Frequence de controle basse

L'action head autoregressive predit 7 tokens un par un. Sur RTX 4090 :
- fp16 : ~15 Hz max
- 4-bit : ~5-7 Hz

C'est suffisant pour **picking, placing, push** mais **insuffisant** pour :
- Manipulation contact-rich (insertion de cle, peg-in-hole) : ~50 Hz necessaire
- Locomotion humanoid : ~200 Hz necessaire (cf Helix, J22)

**Helix (Figure)** atteint 200 Hz parce qu'ils utilisent un System1 de 80M params qui regresse les actions en continu. OpenVLA est **System2 only** dans la taxonomie de J22.

### 5.3 Pas de dynamique reactive

OpenVLA ne voit que l'image courante (et eventuellement 1-2 frames d'historique selon la config). Pas de prediction du futur, pas de re-planification rapide. Si l'objet bouge pendant le pick, OpenVLA persiste sur sa trajectoire planifiee jusqu'a ce que la cross-entropy decode autre chose. Ce n'est pas un MPC, ce n'est pas un world-model.

### 5.4 Robuste a l'image, fragile a l'instruction

Dans les ablations (Table 6 du papier), OpenVLA generalise bien sur :
- Nouvelles distractions visuelles (objets jamais vus en arriere-plan)
- Changements d'eclairage moderés

Il echoue sur :
- Nouvelles instructions linguistiques avec **vocabulaire absent** du training (ex : "decant the wine" si "decant" n'est jamais apparu)
- Compositions complexes ("pick the mug **only if** it's red") — pas de raisonnement de haut niveau

### 5.5 Cout du fine-tuning toujours non-trivial

Meme avec LoRA, fine-tune sur RTX 4090 prend **~10-20 heures** pour ~50k steps sur ~5k demos. Sur un cluster 8x A100, tu fais ca en 1h. Le "1 GPU consumer" du papier est vrai mais demande de la patience.

---

## 6. Spaced-repetition — auto-test

> Reponds a chaque question avant de regarder la suivante. Si tu sechais sur 2+, relis la section concernee.

**Q1** — Pourquoi OpenVLA combine DINOv2 ET SigLIP plutot qu'un seul encodeur vision ?
> R : DINOv2 capte la geometrie spatiale 3D (essentiel pour localiser et planifier la trajectoire), SigLIP capte la semantique image-texte (essentiel pour suivre l'instruction "le mug rouge"). L'ablation du papier montre -3 a -4 points si on en retire un. Ils sont concatenes par patch (axe feature), pas en serie.

**Q2** — Combien de tokens un step OpenVLA produit-il ? Comment chaque token devient-il une valeur reelle ?
> R : 7 tokens (3 translation + 3 rotation + 1 gripper). Chaque token est un index parmi 256 bins reutilisant les 256 derniers tokens du vocabulaire Llama2. La detokenization regarde la table de quantiles (1%-99% du dataset OXE) et retourne la valeur reelle correspondante.

**Q3** — Tu as une RTX 4090 (24 GB). Tu veux fine-tuner OpenVLA sur ta tache. Quel hyperparams LoRA poses-tu (rang, alpha, cibles) ?
> R : Recette par defaut : `r=32, alpha=16, dropout=0.05`, cibles `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Quantization 4-bit du base model. Pas de LoRA sur DINOv2/SigLIP (figes). Total ~17M params trainables, tient en 24 GB.

**Q4** — Pourquoi LoRA initialise-t-on `B` a zero et `A` Gaussian ? Que se passerait-il si on initialisait les deux Gaussian ?
> R : Avec `B=0`, on a `BA = 0` donc au depart `W_effective = W` exactement. Le modele commence par etre identique au pretraine, et LoRA "decale" progressivement. Si on initialisait les deux Gaussian, on injecterait du bruit aleatoire dans le forward pass des le step 0 — la loss exploserait au depart et on perdrait le benefice du pretraining.

**Q5** — Donne deux taches ou OpenVLA est attendu de bien marcher, et deux ou il echoue par design.
> R : OK : pick-and-place table-top, push d'un objet, ouvrir un tiroir, manipulation single-arm avec instruction simple. Echec : peg-in-hole rapide (frequence trop basse), bimanual stitching (pas pretraine sur bimanual), locomotion humanoid (pas dans le training data, frequence inadequate), tache avec raisonnement compose ("pick le mug **sauf si** il est rouge").

---

## 7. Checklist sortie de jour

- [ ] Je peux dessiner l'architecture OpenVLA au tableau (4 blocs, dimensions des features)
- [ ] Je sais expliquer en 1 minute pourquoi 7B bat 55B (encodeurs + curation + Llama2 base)
- [ ] Je sais quelles couches recoivent LoRA et pourquoi exactement celles-la
- [ ] Je sais distinguer le merge LoRA / la quantization NF4 et leur ordre dans le pipeline
- [ ] Je connais les 4 limitations majeures (single-arm, frequence basse, pas reactif, sensible au vocab d'instruction)

---

## 8. References

- **Source principale** : REFERENCES.md #13 — Kim, Pertsch et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. https://arxiv.org/abs/2406.09246 + https://github.com/openvla/openvla
- **LoRA original** : Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. https://arxiv.org/abs/2106.09685
- **DINOv2** : Oquab et al., Meta, 2023. https://arxiv.org/abs/2304.07193
- **SigLIP** : Zhai et al., Google, 2023. https://arxiv.org/abs/2303.15343
- **bitsandbytes NF4** : Dettmers et al., *QLoRA*, 2023. https://arxiv.org/abs/2305.14314

> Suite logique : J21 — pi0 et le paradigme flow-matching qui evite la tokenization discrete.

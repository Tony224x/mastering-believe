# Reseaux de Neurones & LLMs — Comprendre l'IA Generative en profondeur

## Scope

Comprendre les mecanismes internes des LLMs, du neurone unique jusqu'aux architectures SOTA. Objectif : pouvoir lire un paper ArXiv sur une nouvelle architecture, comprendre chaque choix, et reimplementer les blocs cles from scratch en Python/PyTorch.

## Carte d'entree — par ou commencer

**3 questions d'auto-diagnostic** (reponds avant de te lancer) :

1. *Sais-tu deriver une backpropagation a la main sur un mini-reseau ?* Non → commence a **J1**. Oui → tu peux survoler J1-J2.
2. *Peux-tu expliquer QKV + softmax scaling et coder une self-attention ?* Non → ton point d'entree reel est **J5-J6**. Oui → vise directement **J8+** (LLMs modernes).
3. *Veux-tu juste les fondations solides, ou la frontiere 2026 (MoE, SSM, quantization, interpretabilite) ?* Fondations → **J1-J14 suffit**. Frontiere → enchaine sur le **bloc J15-J22 (avance, optionnel)**.

**Prerequis minimaux** : Python courant, algebre lineaire de base (matrices, produit scalaire), notions de derivees/regle de la chaine. Pas besoin de GPU : tout le code `02-code/` tourne en CPU (numpy + stdlib).

**Budget temps** : core **J1-J14 ≈ 75-80h** (≈ 3 semaines a temps partiel). Bloc frontiere **J15-J22 ≈ 38h** supplementaires (optionnel).

## Prerequisites

- Python courant
- Algebre lineaire basique (matrices, vecteurs, produit scalaire)
- Notions de calcul differentiel (derivees, chaine)

## Planning (3 semaines)

### Parcours express vs complet

- **Parcours express (core, obligatoire)** : **J1-J14**. Du neurone au Transformer complet, puis LLMs modernes, fine-tuning/RLHF, inference, reasoning, et un capstone. C'est le coeur du domaine : il suffit a atteindre tous les criteres de reussite ci-dessous.
- **Parcours complet (core + frontiere)** : J1-J14 **puis** le bloc **J15-J22 (avance, optionnel)** — MoE, SSM, long-context, quantization, distillation, interpretabilite, VLMs. A faire seulement apres avoir solidement digere le core, ou en piochant a la carte les sujets qui t'interessent.

> **Navigation linéaire** : ne traite PAS J15-J22 comme la suite obligatoire de J14. Le domaine est *fini* a J14 ; le bloc 15-22 est une extension frontiere. Chaque module 15-22 porte le marqueur `[AVANCE — optionnel]` dans son en-tete.

> **Note d'etat (honnete)** : les exercices `03-exercises/` (easy/medium/hard + solutions) ne couvrent pour l'instant que **J1-J3** au niveau medium/hard ; les modules **J4-J14** ont la theorie + le code mais une couverture d'exercices partielle, et **J15-J22 n'ont pas encore d'exercices** (extension prevue). La theorie et le code de tous les modules sont, eux, complets.

### Semaine 1 — Des fondations aux Transformers

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J1 | Le neurone & backpropagation | Perceptron, fonctions d'activation, gradient descent, backprop a la main | 4h |
| J2 | Reseaux denses (MLP) | Forward pass, loss functions, optimizers (SGD, Adam), regularization | 5h |
| J3 | Embeddings & representations | Word2Vec, espaces vectoriels, similarite cosinus, pourquoi ca marche | 5h |
| J4 | Sequence modeling | RNN, LSTM, GRU — pourquoi ils existent, pourquoi ils ont echoue a scale | 5h |
| J5 | Attention mechanism | Self-attention from scratch, QKV, pourquoi l'attention resout le bottleneck | 6h |
| J6 | Transformer architecture | "Attention is All You Need" bloc par bloc, positional encoding, layer norm | 6h |
| J7 | **Implementer un mini-Transformer** | Coder un Transformer complet from scratch en PyTorch (~200 lignes) | 6-8h |

### Semaine 2 — LLMs modernes & techniques avancees

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J8 | Pre-training & Tokenization | BPE, SentencePiece, pre-training objectives (CLM, MLM), scaling laws | 5h |
| J9 | Architecture des LLMs modernes | GPT, LLaMA, Mistral — diff architecturales (RoPE, GQA, SwiGLU, RMSNorm) | 6h |
| J10 | Fine-tuning & Alignment | SFT, RLHF, DPO, constitutional AI — comment on passe de base model a assistant | 5h |
| J11 | Inference optimisee | KV-cache, speculative decoding, quantization (GPTQ, AWQ), Flash Attention | 5h |
| J12 | Multimodalite & au-dela | Vision Transformers, CLIP, modeles multimodaux, architecture encoder-decoder | 5h |
| J13 | Emergent abilities & reasoning | Chain-of-thought, in-context learning, pourquoi ca emerge, scaling hypothesis | 5h |
| J14 | **Capstone** | Lire et decortiquer un paper recent + reimplementer le composant cle | 6h |

### Semaine 3 — Frontiere NN 2026 — **[AVANCE — optionnel]**

> Bloc **J15-J22 : avance et optionnel**. Hors du parcours core (J1-J14). A aborder une fois le core maitrise, ou a la carte. Pas d'exercices a ce stade.

Cette semaine reste **strictement sur les reseaux de neurones** : architectures, training, mecanique interne. Pas d'agentique, pas de RAG, pas de serving — ces sujets sont couverts dans les domaines `agentic-ai` et `system-design`. Ici on creuse les blocs neuronaux qui definissent l'etat de l'art en 2026. Chaque lecon a son code standalone runnable sans GPU.

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J15 | Test-time compute & reasoning models | o1/o3/R1, GRPO, reasoning vs LLM classique, training pour reasoning, self-consistency | 5h |
| J16 | Mixture of Experts (MoE) | Mixtral, DeepSeek-V3, sparse routing, top-k gating, load balancing, expert parallelism | 5h |
| J17 | State Space Models | Mamba, S6, RWKV, alternative a l'attention, complexite lineaire, hybrides Transformer-Mamba | 5h |
| J18 | Long context & attention scaling | Flash Attention 2/3, RoPE scaling, YaRN, ring attention, sliding window, attention sinks | 5h |
| J19 | Quantization deep dive | INT8/INT4, GPTQ, AWQ, QLoRA, calibration, GGUF, perplexity vs vitesse | 5h |
| J20 | Distillation & donnees synthetiques | SLMs specialises, pipeline synthetic data, SFT/DPO, filtrage, contamination, break-even | 4h |
| J21 | Mechanistic interpretability | Circuits, sparse autoencoders (SAEs), probing, induction heads, superposition | 5h |
| J22 | Vision-language models | ViT, CLIP, LLaVA, SigLIP, cross-attention vs token concat, image tokenization | 5h |

**Prerequis pour Semaine 3** : avoir fait J1-J13 (la Semaine 2 des LLMs modernes) ou equivalent. Les lecons de frontiere supposent la maitrise des Transformers, attention, pre-training, SFT/RLHF et inference basique.

## Au-dela de la theorie

- **`02-code/`** : un script standalone par module theorique, runnable sans GPU.
- **`03-exercises/`** : exercices progressifs (easy → medium → hard) + solutions, couvrant les modules 01-14 en 1:1. Les modules 15-22 n'ont pas encore d'exercices. L'espace `workspace/` (gitignore) est reserve a tes propres solutions.
- **`04-projects/`** : espace libre pour mini-projets et capstones supplementaires lies au domaine.
- **`05-projets-guides/`** : 3 projets guides appliques au contexte logistique LogiSim/FleetSim (voir [`shared/logistics-context.md`](../../shared/logistics-context.md)) — detection de quasi-collisions, imitation learning, rapport EOD genere par LLM.
- **`01-theory-qd/`** : version Quarkdown enrichie de la theorie (math LaTeX, mermaid, callouts), buildable en site statique via `quarkdown/scripts/build-all.ps1`. Les `.md` de `01-theory/` restent la source-of-truth.

## Criteres de reussite

- [ ] Implementer un Transformer from scratch sans regarder de reference
- [ ] Expliquer le mecanisme d'attention avec un schema et les maths (QKV, softmax, scaling)
- [ ] Lire un paper ArXiv sur une nouvelle architecture et identifier les innovations en < 30 min
- [ ] Expliquer pourquoi RoPE > positional encoding sinusoidal, pourquoi GQA > MHA, etc.
- [ ] Calculer le nombre de parametres d'un modele a partir de son architecture
- [ ] Expliquer le pipeline complet : pre-training → SFT → RLHF → deployment

## Concepts fondamentaux a maitriser

### Maths essentielles (pas plus)
- Produit matriciel & pourquoi QK^T fonctionne comme une similarite
- Softmax & temperature — controle de la distribution
- Cross-entropy loss — pourquoi c'est THE loss pour la generation
- Gradient descent — intuition geometrique, pas juste la formule

### Les 7 blocs d'un LLM
1. Tokenizer (BPE)
2. Embedding layer + positional encoding
3. Multi-head self-attention
4. Feed-forward network (MLP)
5. Layer normalization
6. Residual connections
7. Output projection (unembedding) + softmax

## Ressources externes

- **Andrej Karpathy — "Let's build GPT from scratch"** (YouTube) — gold standard pedagogique
- **"Attention is All You Need"** (Vaswani et al., 2017) — paper fondateur
- **The Illustrated Transformer** (Jay Alammar) — visualisations
- **Lilian Weng's blog** — surveys techniques de reference
- **Umar Jamil** (YouTube) — implementations detaillees paper par paper

# Plan figé domaine `neural-networks-llm` (14 modules core + bloc bonus J15-J22)

> Curriculum figé, dérivé du [`README.md`](README.md). Source de vérité pour la structure du domaine.
> **Double palier** : **core J1-J14** (du neurone au capstone — le domaine est *fini* ici) + **bloc frontière J15-J22 « [AVANCÉ — optionnel] »** (extension NN 2026, à la carte, après maîtrise du core). Le bloc frontière n'est PAS la suite obligatoire de J14.
> **Convention de slug** : pour un module N, `01-theory/NN-x.md`, `02-code/NN-x.py`, `01-theory-qd/NN-x.qd` et les exercices/solutions partagent le même slug numéroté. Tous les modules (01-22) ont théorie + code standalone runnable sans GPU (numpy + stdlib).
> **REFERENCES.md** : non encore produit (phase 2). Les ressources canoniques sont listées dans le `README.md` (section « Ressources externes »).
> **État des exercices** : `03-exercises/` couvre J1-J3 au niveau medium/hard ; J4-J14 ont une couverture partielle ; J15-J22 n'ont pas encore d'exercices (extension prévue).

---

## Core J1-J14 (parcours obligatoire)

### Semaine 1 — Des fondations aux Transformers

## J1 — Le neurone & backpropagation

- **Concepts clés** : perceptron, fonctions d'activation, gradient descent, backpropagation calculée à la main, MSE vs cross-entropy.
- **Acquis fin de module** : dériver une backprop sur un mini-réseau, expliquer le rôle des activations et de la descente de gradient.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `01-neurone-et-backpropagation`
- **Temps** : ~4h

## J2 — Réseaux denses (MLP)

- **Concepts clés** : forward pass, loss functions, optimizers (SGD, Adam), régularisation.
- **Acquis fin de module** : implémenter un MLP from scratch, choisir loss et optimizer, raisonner overfitting/régularisation.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `02-reseaux-denses-mlp`
- **Temps** : ~5h

## J3 — Embeddings & représentations

- **Concepts clés** : Word2Vec, espaces vectoriels, similarité cosinus, intuition de pourquoi ça marche.
- **Acquis fin de module** : expliquer ce qu'est un embedding, calculer une similarité cosinus, raisonner sur la géométrie des représentations.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `03-embeddings-representations`
- **Temps** : ~5h

## J4 — Sequence modeling

- **Concepts clés** : RNN, LSTM, GRU — pourquoi ils existent, pourquoi ils ont échoué à scale (vanishing gradient, séquentialité).
- **Acquis fin de module** : expliquer le bottleneck des RNN et ce qui motive l'attention.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `04-sequence-modeling-rnn`
- **Temps** : ~5h

## J5 — Attention mechanism

- **Concepts clés** : self-attention from scratch, QKV, softmax scaling, pourquoi l'attention résout le bottleneck du contexte.
- **Acquis fin de module** : coder une self-attention, expliquer QK^T comme similarité et le rôle du scaling.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `05-attention-mechanism`
- **Temps** : ~6h

## J6 — Transformer architecture

- **Concepts clés** : « Attention is All You Need » bloc par bloc, positional encoding, layer norm, résiduels, multi-head, comptage de paramètres.
- **Acquis fin de module** : décrire chaque bloc d'un Transformer et calculer le nombre de paramètres d'un bloc.
- **Stack** : Python (numpy + stdlib ; le `02-code` illustre aussi l'API PyTorch SDPA)
- **Slug** : `06-transformer-architecture`
- **Temps** : ~6h

## J7 — Implémenter un mini-Transformer (capstone Semaine 1)

- **Concepts clés** : coder un Transformer complet from scratch (~200 lignes), entraînement char-level, génération (greedy / sampling).
- **Acquis fin de module** : produire un mini-Transformer entraînable et générer du texte avec.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `07-mini-transformer`
- **Temps** : ~6-8h

### Semaine 2 — LLMs modernes & techniques avancées

## J8 — Pre-training & Tokenization

- **Concepts clés** : BPE, SentencePiece, objectifs de pre-training (CLM, MLM), scaling laws.
- **Acquis fin de module** : expliquer la tokenization BPE et les objectifs/scaling laws du pre-training.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `08-pretraining-tokenization`
- **Temps** : ~5h

## J9 — Architecture des LLMs modernes

- **Concepts clés** : RoPE, RMSNorm, SwiGLU, GQA (core 2024) ; MLA, MoE fine-grained, SSM hybrides (frontière, approfondissement). Encadré « pont / ordre de lecture » en tête du module.
- **Acquis fin de module** : expliquer pourquoi RoPE > sinusoïdal, GQA > MHA, et situer les innovations frontière.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `09-llms-modernes-architectures`
- **Temps** : ~6h

## J10 — Fine-tuning & Alignment

- **Concepts clés** : SFT, RLHF, DPO, constitutional AI — comment on passe d'un base model à un assistant.
- **Acquis fin de module** : décrire le pipeline base model → SFT → RLHF/DPO et les tradeoffs.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `10-fine-tuning-alignment`
- **Temps** : ~5h

## J11 — Inference optimisée

- **Concepts clés** : KV-cache, speculative decoding, quantization (GPTQ, AWQ), Flash Attention, phases prefill/decode.
- **Acquis fin de module** : expliquer pourquoi l'inférence est memory-bound et les leviers d'optimisation principaux.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `11-inference-optimisee`
- **Temps** : ~5h

## J12 — Multimodalité & au-delà

- **Concepts clés** : Vision Transformers, CLIP, modèles multimodaux, architecture encoder-decoder.
- **Acquis fin de module** : expliquer comment une image devient des tokens et comment CLIP aligne texte/image.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `12-multimodalite`
- **Temps** : ~5h

## J13 — Emergent abilities & reasoning

- **Concepts clés** : chain-of-thought, in-context learning, hypothèse de scaling, débat émergence réelle vs artefact de mesure.
- **Acquis fin de module** : expliquer ce qu'est une capacité émergente et le débat Wei vs Schaeffer.
- **Stack** : Python (numpy + stdlib)
- **Slug** : `13-emergent-abilities-reasoning`
- **Temps** : ~5h

## J14 — Capstone (fin du core)

- **Concepts clés** : lire et décortiquer un paper récent + réimplémenter le composant clé (ex. bloc RoPE + GQA + KV cache).
- **Acquis fin de module** : lire un paper d'architecture en < 30 min et réimplémenter une brique from scratch.
- **Stack** : Python (le `02-code/14-capstone.py` illustre RoPE + GQA + KV cache, convention RoPE interleaved documentée)
- **Slug** : `14-capstone`
- **Temps** : ~6h

---

## Bloc bonus J15-J22 — Frontière NN 2026 — **[AVANCÉ — optionnel]**

> Hors du parcours core. Reste **strictement sur les réseaux de neurones** (pas d'agentique, RAG ni serving — voir `agentic-ai` / `system-design`). Chaque module porte le marqueur `[AVANCÉ — optionnel]` en en-tête. Prérequis global : J1-J13. Pas d'exercices à ce stade.

## J15 — Test-time compute & reasoning models

- **Concepts clés** : o1/o3/R1, GRPO, reasoning vs LLM classique, training pour reasoning, self-consistency.
- **Slug** : `15-test-time-compute-reasoning-models`
- **Temps** : ~5h

## J16 — Mixture of Experts (MoE)

- **Concepts clés** : Mixtral, DeepSeek-V3, sparse routing, top-k gating, load balancing, expert parallelism.
- **Slug** : `16-mixture-of-experts`
- **Temps** : ~5h

## J17 — State Space Models

- **Concepts clés** : Mamba, S6, RWKV, alternative à l'attention, complexité linéaire, hybrides Transformer-Mamba.
- **Slug** : `17-state-space-models`
- **Temps** : ~5h

## J18 — Long context & attention scaling

- **Concepts clés** : Flash Attention 2/3, RoPE scaling, YaRN, ring attention, sliding window, attention sinks.
- **Slug** : `18-long-context-attention-scaling`
- **Temps** : ~5h

## J19 — Quantization deep dive

- **Concepts clés** : INT8/INT4, GPTQ, AWQ, QLoRA, calibration, GGUF, perplexité vs vitesse. Sections lourdes (PTQ, outliers, GGUF) marquées « Approfondissement (optionnel) ».
- **Slug** : `19-quantization`
- **Temps** : ~5h

## J20 — Distillation & données synthétiques

- **Concepts clés** : SLMs spécialisés, pipeline synthetic data, SFT/DPO, filtrage, contamination, break-even.
- **Slug** : `20-distillation-synthetic-data-slms`
- **Temps** : ~4h

## J21 — Mechanistic interpretability

- **Concepts clés** : circuits, sparse autoencoders (SAEs), probing, induction heads, superposition.
- **Slug** : `21-mechanistic-interpretability`
- **Temps** : ~5h

## J22 — Vision-language models

- **Concepts clés** : ViT, CLIP, LLaVA, SigLIP, cross-attention vs token concat, image tokenization.
- **Slug** : `22-vision-language-models`
- **Temps** : ~5h

---

## Critères de réussite (transversaux, valides à la fin du core J1-J14)

1. Implémenter un Transformer from scratch sans référence.
2. Expliquer l'attention avec schéma + maths (QKV, softmax, scaling).
3. Lire un paper d'architecture et identifier les innovations en < 30 min.
4. Expliquer pourquoi RoPE > sinusoïdal, GQA > MHA, etc.
5. Calculer le nombre de paramètres d'un modèle depuis son architecture.
6. Expliquer le pipeline complet : pre-training → SFT → RLHF → déploiement.

# Reseaux de Neurones & LLMs — Comprendre l'IA Generative en profondeur

## Scope

Comprendre les mecanismes internes des LLMs, du neurone unique jusqu'aux architectures SOTA. Objectif : pouvoir lire un paper ArXiv sur une nouvelle architecture, comprendre chaque choix, et reimplementer les blocs cles from scratch en Python/PyTorch.

## Prerequisites

- Python courant
- Algebre lineaire basique (matrices, vecteurs, produit scalaire)
- Notions de calcul differentiel (derivees, chaine)

## Planning (2 semaines)

### Semaine 1 — Des fondations aux Transformers

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J1 | Le neurone & backpropagation | Perceptron, fonctions d'activation, gradient descent, backprop a la main | 4h |
| J2 | Reseaux denses (MLP) | Forward pass, loss functions, optimizers (SGD, Adam), regularization | 3h |
| J3 | Embeddings & representations | Word2Vec, espaces vectoriels, similarite cosinus, pourquoi ca marche | 3h |
| J4 | Sequence modeling | RNN, LSTM, GRU — pourquoi ils existent, pourquoi ils ont echoue a scale | 3h |
| J5 | Attention mechanism | Self-attention from scratch, QKV, pourquoi l'attention resout le bottleneck | 4h |
| J6 | Transformer architecture | "Attention is All You Need" bloc par bloc, positional encoding, layer norm | 4h |
| J7 | **Implementer un mini-Transformer** | Coder un Transformer complet from scratch en PyTorch (~200 lignes) | 4h |

### Semaine 2 — LLMs modernes & techniques avancees

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J8 | Pre-training & Tokenization | BPE, SentencePiece, pre-training objectives (CLM, MLM), scaling laws | 3h |
| J9 | Architecture des LLMs modernes | GPT, LLaMA, Mistral — diff architecturales (RoPE, GQA, SwiGLU, RMSNorm) | 4h |
| J10 | Fine-tuning & Alignment | SFT, RLHF, DPO, constitutional AI — comment on passe de base model a assistant | 3h |
| J11 | Inference optimisee | KV-cache, speculative decoding, quantization (GPTQ, AWQ), Flash Attention | 3h |
| J12 | Multimodalite & au-dela | Vision Transformers, CLIP, modeles multimodaux, architecture encoder-decoder | 3h |
| J13 | Emergent abilities & reasoning | Chain-of-thought, in-context learning, pourquoi ca emerge, scaling hypothesis | 3h |
| J14 | **Capstone** | Lire et decortiquer un paper recent + reimplementer le composant cle | 4h |

### Semaine 3 (optionnelle) — Frontiere 2026 & Apply AI Engineering

Cette semaine couvre les sujets qui separent un "AI engineer 2024" d'un **"applied AI engineer 2026"**. Frontiere technique recente (reasoning models, MCP, xgrammar) et patterns produit qui dominent en prod. Chaque lecon a son code standalone runnable sans GPU.

| Jour | Module | Focus | Temps |
|------|--------|-------|-------|
| J15 | Test-time compute & reasoning models | o1/o3/R1, GRPO, reasoning vs LLM classique, pattern planner-executor, self-consistency | 5h |
| J16 | Agentic LLMs en production | Tool use, MCP, computer use, boucle agentique robuste, context rot, human-in-the-loop | 5h |
| J17 | RAG 2026 avance | Hybrid (BM25+dense+RRF), contextual retrieval, reranker, agentic RAG, GraphRAG, groundedness | 5h |
| J18 | Context engineering & prompt caching | Cache hit rate, placement des blocs, economie tokens, long-context vs RAG, compaction | 4h |
| J19 | Production inference serving | vLLM, SGLang, continuous batching, PagedAttention, speculative decoding, FP8, disaggregated | 5h |
| J20 | Distillation & donnees synthetiques | SLMs specialises, pipeline synthetic data, SFT/DPO, filtrage, contamination, break-even | 4h |
| J21 | Evals & observability LLM | 5 niveaux d'eval, LLM-as-judge + biais, pairwise, monitoring prod, red teaming, Goodhart | 4h |
| J22 | Structured outputs & constrained gen | JSON mode, tool calling, FSA-based decoding, xgrammar/llguidance, reasoning-then-structured | 4h |

**Prerequis pour Semaine 3** : avoir fait J1-J13 (la Semaine 2 des LLMs modernes) ou equivalent. Les lecons de frontiere supposent la maitrise des Transformers, attention, pre-training, SFT/RLHF et inference basique.

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

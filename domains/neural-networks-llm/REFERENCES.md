# REFERENCES — `neural-networks-llm` (22 modules)

> Sources de tier-1 par module : papiers fondateurs, manuels canoniques, docs officielles.
> Source de vérité pour les modules J1..J22 (core J1-J14 + frontière J15-J22).
> Toutes les références ont été vérifiées (titre / auteurs / année / arXiv id) via WebSearch/WebFetch (juin 2026).
> Convention de citation : `Auteurs (année). *Titre*. arXiv:id / venue — note de pertinence.`
> Marqueur `(à vérifier)` ajouté aux rares entrées dont un détail reste incertain.

---

## Module 01 — Le neurone & backpropagation

- Rumelhart, Hinton, Williams (1986). *Learning representations by back-propagating errors*. Nature 323:533-536 — papier fondateur de la backpropagation moderne ; la base de tout ce module.
- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press — https://www.deeplearningbook.org/ — ch. 6 (deep feedforward networks) et ch. 8 (optimization) : référence manuel pour neurone, activation, gradient.
- Nielsen (2015). *Neural Networks and Deep Learning*. http://neuralnetworksanddeeplearning.com/ — ch. 1-2 : dérivation visuelle et intuitive de la backprop, idéal en complément.
- Karpathy (2022). *The spelled-out intro to neural networks and backpropagation: building micrograd*. https://www.youtube.com/watch?v=VMj-3S1tku0 — backprop codée from scratch ligne à ligne, gold standard pédagogique.

## Module 02 — Réseaux denses (MLP)

- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press — ch. 6-8 : MLP, loss functions, régularisation (dropout, weight decay), optimisation.
- Kingma, Ba (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980 (ICLR 2015) — l'optimiseur par défaut des réseaux modernes ; à comprendre après SGD/momentum.
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR 15:1929-1958 — régularisation canonique des MLP.
- Ioffe, Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. arXiv:1502.03167 — normalisation des activations, prérequis conceptuel de LayerNorm/RMSNorm vus plus tard.

## Module 03 — Embeddings & représentations

- Mikolov, Chen, Corrado, Dean (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781 — word2vec (CBOW & skip-gram), le papier fondateur des embeddings de mots.
- Mikolov, Sutskever, Chen, Corrado, Dean (2013). *Distributed Representations of Words and Phrases and their Compositionality*. arXiv:1310.4546 — negative sampling + arithmétique vectorielle (king - man + woman ≈ queen).
- Pennington, Socher, Manning (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014 — https://nlp.stanford.edu/projects/glove/ — alternative basée sur les co-occurrences globales, contrepoint à word2vec.

## Module 04 — Sequence modeling (RNN, LSTM, GRU)

- Hochreiter, Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation 9(8):1735-1780 — papier fondateur du LSTM et des gating mechanisms (réponse au vanishing gradient).
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. arXiv:1406.1078 — introduit le GRU et l'architecture encoder-decoder.
- Sutskever, Vinyals, Le (2014). *Sequence to Sequence Learning with Neural Networks*. arXiv:1409.3215 — seq2seq à base de LSTM, le paradigme que l'attention viendra débloquer.
- Karpathy (2015). *The Unreasonable Effectiveness of Recurrent Neural Networks*. http://karpathy.github.io/2015/05/21/rnn-effectiveness/ — intuition et limites des RNN, pourquoi ils échouent à scale.

## Module 05 — Attention mechanism

- Bahdanau, Cho, Bengio (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473 (ICLR 2015) — premier mécanisme d'attention (additive), résout le bottleneck du vecteur de contexte fixe.
- Vaswani et al. (2017). *Attention Is All You Need*. arXiv:1706.03762 — formalise la scaled dot-product attention (QKV, softmax scaling) et la self-attention pure.
- Alammar (2018). *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/ — visualisations de référence de QKV et multi-head attention.

## Module 06 — Transformer architecture

- Vaswani et al. (2017). *Attention Is All You Need*. arXiv:1706.03762 — l'architecture de référence bloc par bloc (multi-head attention, FFN, positional encoding sinusoïdal, residual + LayerNorm).
- Ba, Kiros, Hinton (2016). *Layer Normalization*. arXiv:1607.06450 — la normalisation utilisée dans le bloc Transformer.
- He, Zhang, Ren, Sun (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385 — origine des residual connections, brique 6 des 7 blocs d'un LLM.
- Phuong, Hutter (2022). *Formal Algorithms for Transformers*. arXiv:2207.09238 — pseudo-code rigoureux de chaque composant, utile pour réimplémenter sans ambiguïté.

## Module 07 — Implémenter un mini-Transformer

- Karpathy (2023). *Let's build GPT: from scratch, in code, spelled out*. https://www.youtube.com/watch?v=kCc8FmEb1nY — construction d'un GPT complet from scratch, le gold standard pour ce module.
- Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT — implémentation de référence ~300 lignes, lisible et entraînable.
- Rush et al. (2018). *The Annotated Transformer*. https://nlp.seas.harvard.edu/annotated-transformer/ — le papier Vaswani annoté avec code PyTorch exécutable ligne à ligne.

## Module 08 — Pre-training & tokenization

- Sennrich, Haddow, Birch (2015). *Neural Machine Translation of Rare Words with Subword Units*. arXiv:1508.07909 — introduit le BPE pour la tokenization sous-mot.
- Kudo, Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. arXiv:1808.06226 (EMNLP 2018 demo) — tokenizer language-agnostic entraîné directement sur du texte brut.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf — objectif CLM (causal language modeling) et émergence du zero-shot.
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361 — lois de puissance loss vs (params, data, compute).
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556 — révise les scaling laws : params et tokens doivent croître à parts égales.

## Module 09 — Architecture des LLMs modernes

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864 — RoPE, le positional encoding rotatif standard (remplace le sinusoïdal).
- Zhang, Sennrich (2019). *Root Mean Square Layer Normalization* (RMSNorm). arXiv:1910.07467 — normalisation utilisée par LLaMA/Mistral à la place de LayerNorm.
- Shazeer (2020). *GLU Variants Improve Transformer* (SwiGLU). arXiv:2002.05202 — FFN à gated linear units, standard 2024.
- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245 — Grouped Query Attention, réduit le KV cache.
- Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971 — recette open d'un LLM moderne (RoPE + RMSNorm + SwiGLU).
- Grattafiori et al. / Meta (2024). *The Llama 3 Herd of Models*. arXiv:2407.21783 — détails d'architecture et d'entraînement d'une famille SOTA 2024.

## Module 10 — Fine-tuning & alignment

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). arXiv:2203.02155 — le pipeline SFT → reward model → RLHF (PPO) de référence.
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (DPO). arXiv:2305.18290 — alignment sans RL ni reward model explicite, simple classification loss.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 — fine-tuning paramètre-efficient par adaptateurs bas-rang.
- Bai et al. / Anthropic (2022). *Constitutional AI: Harmlessness from AI Feedback*. arXiv:2212.08073 — RLAIF, alignment guidé par une "constitution" plutôt que par feedback humain direct.

## Module 11 — Inference optimisée

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135 — attention exacte IO-aware par tiling SRAM/HBM.
- Leviathan, Kalman, Matias (2022). *Fast Inference from Transformers via Speculative Decoding*. arXiv:2211.17192 — draft model + vérification, accélère le decoding sans perte de qualité.
- Frantar, Ashkboos, Hoefler, Alistarh (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323 — PTQ 3-4 bits via information de second ordre.
- Lin et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv:2306.00978 (MLSys 2024 best paper) — quantization protégeant les 1% de poids saillants.
- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). arXiv:2309.06180 — gestion mémoire du KV cache façon pagination OS.

## Module 12 — Multimodalité & au-delà

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT). arXiv:2010.11929 — patches d'image comme tokens, le Transformer appliqué à la vision.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). arXiv:2103.00020 — alignement image-texte par contrastive learning, base du zero-shot vision.
- Liu, Li, Wu, Lee (2023). *Visual Instruction Tuning* (LLaVA). arXiv:2304.08485 — connecte un encodeur visuel à un LLM via instruction tuning multimodal.
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. arXiv:2204.14198 — cross-attention image→texte, modèle multimodal few-shot de référence.

## Module 13 — Emergent abilities & reasoning

- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv:2201.11903 — le CoT, raisonnement par étapes intermédiaires.
- Wei et al. (2022). *Emergent Abilities of Large Language Models*. arXiv:2206.07682 — formalise la notion de capacités émergentes avec l'échelle.
- Schaeffer, Miranda, Koyejo (2023). *Are Emergent Abilities of Large Language Models a Mirage?*. arXiv:2304.15004 — contrepoint critique : l'émergence dépendrait de la métrique choisie.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171 — échantillonner plusieurs CoT et voter, gain robuste sur le raisonnement.

## Module 14 — Capstone (lire et décortiquer un paper récent)

- Grattafiori et al. / Meta (2024). *The Llama 3 Herd of Models*. arXiv:2407.21783 — paper complet et lisible, candidat idéal pour l'exercice de décorticage.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report*. arXiv:2412.19437 — concentré de la frontière 2025 (MLA + MoE + MTP), parfait pour relier toutes les notions du domaine.
- Phuong, Hutter (2022). *Formal Algorithms for Transformers*. arXiv:2207.09238 — gabarit pour formaliser proprement le composant clé à réimplémenter.

---

# Bloc frontière (J15-J22) — [AVANCÉ — optionnel]

## Module 15 — Test-time compute & reasoning models

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948 (→ Nature, 2025) — RL pur (R1-Zero) faisant émerger le raisonnement ; modèle ouvert de référence.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300 — introduit GRPO (Group Relative Policy Optimization), l'algo RL des reasoning models.
- Snell, Lee, Xu, Kumar (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314 — compute-optimal au test time : penser plus > grossir le modèle.
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171 — best-of-N / vote, brique de base du test-time scaling.
- OpenAI (2024). *Learning to Reason with LLMs* (o1). https://openai.com/index/learning-to-reason-with-llms/ — annonce o1, premier modèle grand public à scaler le test-time compute (détails techniques limités).

## Module 16 — Mixture of Experts (MoE)

- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. arXiv:1701.06538 — couche MoE sparse à top-k gating, papier fondateur moderne.
- Fedus, Zoph, Shazeer (2021). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. arXiv:2101.03961 — routing top-1, load balancing, expert parallelism.
- Jiang et al. / Mistral (2024). *Mixtral of Experts*. arXiv:2401.04088 — SMoE 8x7B open (47B params, 13B actifs), MoE accessible.
- DeepSeek-AI (2024). *DeepSeek-V3 Technical Report*. arXiv:2412.19437 — fine-grained + shared experts, load balancing auxiliary-loss-free, MoE 671B/37B SOTA.

## Module 17 — State Space Models (SSM)

- Gu, Goel, Ré (2021). *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). arXiv:2111.00396 — SSM structuré, fondation théorique de la lignée Mamba.
- Gu, Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752 — sélectivité (S6), complexité linéaire, alternative à l'attention.
- Dao, Gu (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2 / SSD). arXiv:2405.21060 — dualité attention↔SSM, framework SSD.
- Peng et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era*. arXiv:2305.13048 (Findings EMNLP 2023) — RNN linéaire parallélisable, alternative SSM-like.
- Lieber et al. / AI21 (2024). *Jamba: A Hybrid Transformer-Mamba Language Model*. arXiv:2403.19887 — hybride attention+Mamba+MoE à grande échelle.

## Module 18 — Long context & attention scaling

- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691 — meilleure répartition du travail sur GPU, suite directe de FlashAttention.
- Shah et al. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. arXiv:2407.08608 — exploite asynchronie + FP8 des GPU Hopper.
- Peng, Quesnelle, Fan, Shippole (2023). *YaRN: Efficient Context Window Extension of Large Language Models*. arXiv:2309.00071 — extension de contexte RoPE (NTK-by-parts), 10x moins de tokens.
- Liu, Zaharia, Abbeel (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context*. arXiv:2310.01889 — distribue la séquence sur N devices, contexte ~illimité.
- Beltagy, Peters, Cohan (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150 — sliding window + global attention, origine des patterns d'attention sparse/locale.
- Xiao et al. (2023). *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM). arXiv:2309.17453 — attention sinks, streaming à contexte long sans réentraînement.

## Module 19 — Quantization deep dive

- Frantar, Ashkboos, Hoefler, Alistarh (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323 — PTQ INT3/INT4 par information de second ordre.
- Lin et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv:2306.00978 (MLSys 2024 best paper) — scaling per-channel guidé par les activations.
- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314 — NF4 + double quantization + paged optimizers ; fine-tune un 65B sur un seul GPU 48 Go.
- Dettmers, Lewis, Belkada, Zettlemoyer (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. arXiv:2208.07339 — INT8 avec gestion des outliers, base de bitsandbytes.
- llama.cpp / GGUF. https://github.com/ggml-org/llama.cpp et https://github.com/ggml-org/ggml/blob/master/docs/gguf.md — format GGUF et schémas de quantization k-quants utilisés en pratique côté inference locale.

## Module 20 — Distillation & données synthétiques (SLMs)

- Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531 — knowledge distillation par soft targets et température, papier fondateur.
- Sanh, Debut, Chaumond, Wolf (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108 — distillation appliquée à un Transformer NLP, recette canonique.
- Gunasekar et al. / Microsoft (2023). *Textbooks Are All You Need* (phi-1). arXiv:2306.11644 — données synthétiques "textbook-quality", thèse des SLMs spécialisés.
- Abdin et al. / Microsoft (2024). *Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*. arXiv:2404.14219 — SLM SOTA via data engineering, break-even taille/qualité.
- Taori et al. (2023). *Alpaca: A Strong, Replicable Instruction-Following Model*. https://crfm.stanford.edu/2023/03/13/alpaca.html — Self-Instruct/distillation depuis un modèle plus fort (contamination à surveiller).

## Module 21 — Mechanistic interpretability

- Elhage et al. / Anthropic (2021). *A Mathematical Framework for Transformer Circuits*. https://transformer-circuits.pub/2021/framework/index.html — circuits, QK/OV, induction heads, le socle de la mech-interp.
- Olsson et al. / Anthropic (2022). *In-context Learning and Induction Heads*. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html — rôle causal des induction heads dans l'in-context learning.
- Elhage et al. / Anthropic (2022). *Toy Models of Superposition*. arXiv:2209.10652 — la superposition (plus de features que de neurones), motivation des SAEs.
- Bricken et al. / Anthropic (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. https://transformer-circuits.pub/2023/monosemantic-features/ — sparse autoencoders sur un transformer 1-couche, features monosémantiques.
- Templeton et al. / Anthropic (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. https://transformer-circuits.pub/2024/scaling-monosemanticity/ — SAEs passés à l'échelle d'un modèle de production, features steerables.

## Module 22 — Vision-language models (VLM)

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT). arXiv:2010.11929 — l'encodeur visuel à patches, brique de tout VLM.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). arXiv:2103.00020 — pré-entraînement contrastif image-texte (softmax).
- Zhai, Mustafa, Kolesnikov, Beyer (2023). *Sigmoid Loss for Language Image Pre-Training* (SigLIP). arXiv:2303.15343 (ICCV 2023) — loss sigmoïde pairwise, encodeur visuel des VLM récents.
- Liu, Li, Wu, Lee (2023). *Visual Instruction Tuning* (LLaVA). arXiv:2304.08485 — token concat (projection visuelle → LLM), architecture VLM la plus répandue.
- Alayrac et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. arXiv:2204.14198 — alternative cross-attention (gated xattn), à opposer au token concat de LLaVA.

---

## Ressources transversales

### Manuels & livres

- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press — https://www.deeplearningbook.org/ — manuel canonique des fondations (J1-J6).
- Bishop, Bishop (2023). *Deep Learning: Foundations and Concepts*. Springer — https://www.bishopbook.com/ — manuel moderne, couvre Transformers et diffusion.
- Jurafsky, Martin. *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/ — chapitres LLM/Transformers/embeddings tenus à jour, gratuits.
- Nielsen (2015). *Neural Networks and Deep Learning*. http://neuralnetworksanddeeplearning.com/ — intro visuelle gratuite à backprop et MLP.
- Raschka (2024). *Build a Large Language Model (From Scratch)*. Manning — https://github.com/rasbt/LLMs-from-scratch — implémentation pas-à-pas d'un GPT (J7-J10).

### Cours universitaires

- Stanford CS224N — *Natural Language Processing with Deep Learning* (Manning). https://web.stanford.edu/class/cs224n/ — référence embeddings → attention → Transformers (J3-J9).
- Stanford CS336 — *Language Modeling from Scratch* (Hashimoto, Liang), Spring 2025. https://stanford-cs336.github.io/ — construit tout from scratch : tokenizer, Transformer, FlashAttention-2, scaling laws, SFT/GRPO. Le pendant cours du domaine complet.
- Stanford CS25 — *Transformers United*. https://web.stanford.edu/class/cs25/ — séminaires invités sur la frontière (MoE, SSM, interp), bon pour J15-J22.
- MIT 6.S191 — *Introduction to Deep Learning*. http://introtodeeplearning.com/ — bootcamp fondations + génératif, vidéos à jour.

### Blogs, séries & docs

- Karpathy — *Neural Networks: Zero to Hero*. https://karpathy.ai/zero-to-hero.html — micrograd → makemore → nanoGPT → tokenizer, la série pédagogique de référence (J1-J8).
- Alammar — *The Illustrated Transformer / GPT-2 / BERT*. https://jalammar.github.io/ — visualisations canoniques (J5-J9).
- Weng (Lilian) — *Lil'Log*. https://lilianweng.github.io/ — surveys techniques de référence (attention, LLM, diffusion, RLHF, hallucination).
- Phuong, Hutter (2022). *Formal Algorithms for Transformers*. arXiv:2207.09238 — pseudo-code rigoureux, support de réimplémentation pour tout le domaine.
- Hugging Face — *Transformers / Tokenizers / PEFT docs*. https://huggingface.co/docs — docs des libs utilisées en pratique (tokenization, LoRA/QLoRA, quantization).
- PyTorch documentation. https://pytorch.org/docs/stable/ — référence API pour tout le code `02-code/` et `05-projets-guides/`.

---

## Notes

- **Source-of-truth de citation** : les `.md` de `01-theory/` doivent rester cohérents avec les arXiv id listés ici ; toute correction d'une référence dans un `.md` doit aussi se répercuter dans le `.qd` miroir si présent.
- **Frontière mi-2026** : les modules J15-J22 citent l'état de l'art à juin 2026 (DeepSeek-V3/R1, Mamba-2/SSD, FlashAttention-3, SAEs Claude 3 Sonnet, SigLIP). Les modèles propriétaires (o1/o3, GPT-5, Gemini) sont référencés via annonces officielles faute de papier technique complet — détails d'architecture non publics.
- **Pages SAEs Anthropic** : publiées sur `transformer-circuits.pub` (pas d'arXiv id pour Bricken 2023 et Templeton 2024) ; Toy Models of Superposition a un arXiv id (2209.10652).
- **Aucune référence inventée** : chaque entrée arXiv/venue a été vérifiée (titre, auteurs, année, id) via WebSearch en juin 2026. Aucune entrée n'a nécessité le marqueur `(à vérifier)`.

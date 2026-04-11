# Jour 14 — Capstone : lire un paper et reimplementer LLaMA

> **Temps estime** : 6h | **Prerequis** : Jours 1-13 (tous)

---

## 1. Methode : lire un paper en < 30 minutes

Lire un paper ML de maniere exhaustive est inutile et epuisant. 95% des papers contiennent 5% d'info nouvelle. L'objectif est d'extraire cette info rapidement.

### La methode "3 passes"

Inspire de "How to Read a Paper" (Keshav, 2007).

**Pass 1 — 5 minutes : est-ce que ce paper vaut la peine ?**
1. **Titre** : quelle famille de probleme ?
2. **Abstract** : quelle est la contribution claimed ?
3. **Figures** : les graphiques racontent souvent toute l'histoire
4. **Conclusion** : quels sont les resultats et les limites ?
5. **References** : reconnais-tu les papers cites ? Est-ce une famille de travaux que tu suis ?

**Decision** : si le paper n'est pas pertinent pour ton objectif, **stop**. Tu as gagne 90% du temps de ceux qui le liraient en entier.

**Pass 2 — 15 minutes : quelle est la methode ?**
1. **Intro** : quel probleme exact, quel est le gap, quelle est la contribution ?
2. **Related work** : skimming pour voir les baselines
3. **Method** : tu peux sauter les equations au premier tour, ne lire que les phrases
4. **Experiments** : lire les setups et les tableaux principaux
5. **Ablations** : souvent les plus instructives (qu'est-ce qui compte vraiment)

**Pass 3 — 30+ minutes : reimplementation mentale**
1. Relire la methode en detail, en essayant de coder mentalement l'algorithme
2. Identifier les equations cles et leurs dimensions
3. Noter les hyperparameters et leurs valeurs
4. Chercher le code (souvent sur github, parfois dans une appendix)

### La checklist de lecture critique

Pour chaque paper, repondre a ces questions :

1. **Probleme** : quel probleme exactement ? Est-il important ? Mesurable ?
2. **Contribution** : qu'est-ce qui est nouveau ? (methode, dataset, insight, benchmark)
3. **Baseline** : a quoi est-ce compare ? Les baselines sont-elles equitables ?
4. **Resultats** : quel est le gain ? Est-il statistiquement significatif ?
5. **Ablations** : quelles parties de la methode contribuent vraiment ?
6. **Limitations** : qu'est-ce qui ne marche pas ? (les papers honnetes en parlent)
7. **Reproductibilite** : peux-tu reimplementer ca en 1-2 jours ?

---

## 2. Walkthrough : le paper LLaMA (Touvron et al., 2023)

**Titre** : "LLaMA: Open and Efficient Foundation Language Models"

**Abstract (resume)** : Meta releases LLaMA, a family of foundation models from 7B to 65B. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks. LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B. All trained on publicly available datasets.

### Les 3 contributions revendiquees

1. **Performance** : un petit modele (13B) bat un gros modele (GPT-3 175B)
2. **Donnees publiques uniquement** : pas de donnees proprietaires
3. **Efficacite de training** : atteint la performance de Chinchilla

### Les 4 innovations architecturales (compte a l'ouverture du paper)

En lisant la section "Architecture", on identifie 4 changements par rapport au Transformer standard :

**1. RoPE (Rotary Positional Embedding)**
- Citation : Su et al., 2021 "RoFormer"
- Remplace les learned positional embeddings absolues
- Appliquee aux queries et keys avant le produit scalaire d'attention
- Raison : meilleure extrapolation, encode position relative

**2. RMSNorm**
- Citation : Zhang & Sennrich, 2019
- Remplace LayerNorm
- Pas de centrage, pas de bias
- Raison : plus simple, meme performance, plus rapide

**3. SwiGLU**
- Citation : Shazeer, 2020
- Remplace GeLU/ReLU dans le FFN
- Utilise une porte (gate) multiplicative
- Raison : meilleur empiriquement (+0.5-1%)
- Dimension `d_ff = 8/3 * d_model` pour garder le meme nombre de params

**4. Pre-norm**
- Norm **avant** chaque layer (attention, FFN) au lieu d'apres
- Pas nouveau (GPT-2 l'utilisait deja), mais mis en avant dans LLaMA

**NOTE** : LLaMA 1 utilise MHA classique. GQA a ete introduit dans LLaMA 2 pour les 34B et 70B.

### Les chiffres cles a retenir

```
LLaMA-7B:  32 layers, d_model=4096, 32 heads, d_ff=11008, seq=2048, 1T tokens
LLaMA-13B: 40 layers, d_model=5120, 40 heads, d_ff=13824, seq=2048, 1T tokens
LLaMA-33B: 60 layers, d_model=6656, 52 heads, d_ff=17920, seq=2048, 1.4T tokens
LLaMA-65B: 80 layers, d_model=8192, 64 heads, d_ff=22016, seq=2048, 1.4T tokens
```

### Data mix

| Source | % |
|---|---|
| CommonCrawl | 67% |
| C4 | 15% |
| Github | 4.5% |
| Wikipedia | 4.5% |
| Books | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2% |

### Training

- Optimizer : AdamW
- Learning rate : cosine schedule avec warmup
- Weight decay : 0.1
- Gradient clipping : 1.0
- Batch size : 4M tokens effectif

### Resultats (extract)

| Model | MMLU (5-shot) | HumanEval (pass@1) | GSM8K (8-shot) |
|---|---|---|---|
| GPT-3 175B | 43.9% | - | 6.9% |
| Chinchilla 70B | 67.5% | - | 44% |
| LLaMA 7B | 35.1% | 13.0% | 11.0% |
| LLaMA 13B | 46.9% | 15.8% | 17.8% |
| LLaMA 33B | 57.8% | 21.7% | 35.6% |
| LLaMA 65B | 63.4% | 23.7% | 50.9% |

**Observation** : LLaMA 13B > GPT-3 175B sur MMLU. 7B > GPT-3 175B sur HumanEval. C'est la revolution de LLaMA : **petits modeles bien entraines > gros modeles mal entraines**.

---

## 3. Plan de reimplementation

Pour reimplementer LLaMA, on identifie les composants et on les code un par un :

```
mini_llama/
├── tokenizer    (mock: un simple char-level pour le demo)
├── rms_norm     (j9 - deja fait)
├── rope         (j9 - deja fait)
├── attention    (attention + RoPE + GQA + KV cache)
├── ffn_swiglu   (j9 - deja fait)
├── block        (attention + ffn avec pre-norm residual)
├── model        (embedding + N blocks + final norm + lm_head)
└── generate     (boucle autoregressive avec KV cache)
```

Chaque composant a environ 10-50 lignes de code. Le tout fait ~300-400 lignes.

### Les decisions de simplification

Pour rester en 300-400 lignes :
- Pas de mixed precision (fp16/bf16) — tout en fp32
- Pas d'optimisation memoire (pas de gradient checkpointing)
- Pas de training, juste l'inference
- Tokenizer mock (char-level sur petit vocab)
- GQA optionnelle (activable avec n_kv_heads != n_heads)
- KV cache en dict simple, pas de paging

On garde :
- RoPE complet
- RMSNorm
- SwiGLU
- Optional GQA
- Causal mask
- KV cache pour la generation
- Generation autoregressive avec temperature

---

## 4. Checklist de lecture critique generale

Pour tout paper ML, se poser ces questions :

### Questions sur la contribution
1. Le paper claime N nouveautes. Quelles sont-elles exactement ?
2. Chaque nouveaute est-elle ablee ? (c'est-a-dire desactivee pour mesurer sa contribution)
3. Sans cette nouveaute, qu'obtient-on ?

### Questions sur la comparaison
4. Quelles sont les baselines ? Sont-elles a jour ?
5. Les baselines sont-elles implementees dans le meme setup (meme data, meme compute) ?
6. Le modele est-il evalue sur les memes benchmarks que les baselines ?

### Questions sur les chiffres
7. Les ecarts rapportes sont-ils statistiquement significatifs ?
8. Combien de seeds ont ete essayees ?
9. Quelle est la variance des resultats ?

### Questions sur la generalisation
10. Sur combien de benchmarks est-ce teste ?
11. Y a-t-il des benchmarks ou le modele est **pire** que les baselines ?
12. Le modele marche-t-il sur des domaines autres que celui d'entrainement ?

### Questions sur les failles potentielles
13. Data contamination : le modele a-t-il vu les tests pendant le training ?
14. Cherry-picking : les resultats presentes sont-ils choisis ou moyennes ?
15. Prompt engineering : les prompts ont-ils ete specifiquement tunes pour le paper ?

### Red flags frequents

- "Our method outperforms all baselines on [one cherry-picked benchmark]"
- Aucune ablation
- Comparaison avec des baselines pas a jour (> 6 mois)
- "We leave [major issue] for future work" (signale une limitation majeure non resolue)
- Pas de release de code ou de poids

---

## 5. Comment lire la section "Method" efficacement

### 5.1 Identifier les equations cles

Generalement, 3-5 equations portent 90% de la methode. Les trouver :
- Toutes les equations numerotees explicitement `(1), (2), ...`
- Les equations qui definissent la loss
- Les equations qui definissent les forward passes
- Les equations qui definissent les updates de parametres

### 5.2 Verifier les dimensions

Pour chaque equation, ecrire les dimensions de chaque tenseur :

```
Equation: attention = softmax(Q K^T / sqrt(d_k)) V
Dimensions:
  Q: (batch, n_heads, seq, d_k)
  K: (batch, n_heads, seq, d_k)
  V: (batch, n_heads, seq, d_v)
  Q @ K^T: (batch, n_heads, seq, seq)
  softmax(...): (batch, n_heads, seq, seq)
  ... @ V: (batch, n_heads, seq, d_v)
```

Si tu peux ecrire ces dimensions, tu as compris la methode. Sinon, tu n'as pas compris.

### 5.3 Identifier les "subtilites"

Les vrais details qui font marcher ou echouer :
- Initialization des poids (variance, type)
- Scheduling du learning rate
- Regularization (dropout, weight decay, etc.)
- Tricks d'entrainement (gradient clipping, loss scaling)

### 5.4 Identifier les ablations

Dans la section experiences, chercher les tableaux ou on compare des versions du modele avec/sans chaque composant. C'est la que tu apprends ce qui compte **vraiment**.

---

## 6. Flash Cards — Active Recall

### Q1 : Quelles sont les 4 innovations architecturales de LLaMA 1 ?

<details>
<summary>Reponse</summary>

1. **RoPE** (Rotary Positional Embedding) au lieu d'embeddings de position apprises absolues. Applique rotation aux queries et keys avant le produit scalaire d'attention. Encode la position relative.

2. **RMSNorm** au lieu de LayerNorm. Pas de centrage (mean), pas de bias. Plus simple, plus stable en float16.

3. **SwiGLU** au lieu de GeLU dans le FFN. Utilise un gating multiplicatif `Swish(W_gate @ x) * (W_up @ x)`. Dimension `d_ff = 8/3 * d_model` pour matcher les params.

4. **Pre-norm** : la normalization est appliquee AVANT chaque layer, pas apres. Convergence plus stable en profondeur.

Note : LLaMA 1 utilise MHA classique. GQA a ete ajoute dans LLaMA 2 (34B et 70B uniquement).

</details>

### Q2 : Comment lire un paper en moins de 30 minutes ?

<details>
<summary>Reponse</summary>

**Methode "3 passes"** :

**Pass 1 — 5 minutes : est-ce utile pour moi ?**
- Titre, abstract, figures, conclusion, references
- Decision : continue ou stop

**Pass 2 — 15 minutes : quelle est la methode ?**
- Intro, related work, method, experiments, ablations
- Sauter les equations au premier tour

**Pass 3 — 30 minutes : reimplementation mentale**
- Relire la methode en detail
- Ecrire les equations avec leurs dimensions
- Identifier les subtilites (init, scheduling, tricks)

**Checklist** : probleme, contribution, baselines, resultats, ablations, limitations, reproductibilite.

**Red flags** : pas d'ablation, baselines obsoletes, cherry-picking, pas de code.

</details>

### Q3 : Quelle est l'une des plus grandes contributions de LLaMA ?

<details>
<summary>Reponse</summary>

**Demonstration empirique** qu'un **petit modele bien entraine** (13B sur 1T tokens) bat un **gros modele sous-entraine** (GPT-3 175B sur 300B tokens) sur la plupart des benchmarks.

**Preuves** :
- LLaMA 13B > GPT-3 175B sur MMLU (46.9% vs 43.9%)
- LLaMA 7B > GPT-3 175B sur HumanEval (code)

**Impact** :
1. Justification empirique du scaling law Chinchilla (20 tokens/param)
2. Validation du fait que les donnees de haute qualite comptent plus que le volume de parametres
3. Demonstration qu'un modele "accessible" (teneur sur 1-2 GPUs) peut etre au niveau d'un gros modele
4. Debut de l'ecosysteme open-source fort : Vicuna, Alpaca, Guanaco, tous construits sur LLaMA

**Heritage** : LLaMA 2, LLaMA 3, Mistral, Gemma — tous les modeles open-source modernes descendent directement ou s'inspirent de LLaMA.

</details>

### Q4 : Qu'est-ce qu'une ablation et pourquoi c'est important dans un paper ?

<details>
<summary>Reponse</summary>

**Ablation** : une experience ou on desactive UN composant de la methode pour mesurer sa contribution specifique.

**Exemple** : si un paper propose "methode X = A + B + C", les ablations sont :
- Methode sans A : score ?
- Methode sans B : score ?
- Methode sans C : score ?

On compare chaque version a la methode complete pour voir ce qui compte.

**Pourquoi c'est important** :
1. **Comprendre ce qui marche vraiment** : souvent un composant contribue 80% du gain, les autres sont cosmetiques
2. **Detecter le cherry-picking** : si on change tout d'un coup, on ne sait pas ce qui a aide
3. **Reimplementation** : si un composant ne contribue presque rien, on peut le supprimer
4. **Critique du paper** : pas d'ablation = red flag, on ne peut pas juger la methode

**Red flag** : un paper qui n'a pas d'ablations est souvent un paper qui ne veut pas que tu voies que certaines parties ne contribuent rien.

</details>

### Q5 : Pour reimplementer mini-LLaMA en 400 lignes, quels composants faut-il coder ?

<details>
<summary>Reponse</summary>

**Components essentiels** :

1. **RMSNorm** (~10 lignes) : normalisation sans centrage
2. **RoPE** (~20 lignes) : precompute des frequencies + apply to q, k
3. **Attention avec GQA + KV cache** (~60 lignes) : projections, apply RoPE, causal mask, softmax
4. **SwiGLU FFN** (~15 lignes) : 3 linears + silu gate
5. **TransformerBlock** (~15 lignes) : pre-norm attention + ffn avec residuals
6. **Model** (~30 lignes) : token embeddings + stack de blocks + final norm + lm_head
7. **Generate** (~30 lignes) : boucle autoregressive avec temperature sampling
8. **Mock tokenizer** (~20 lignes) : char-level pour demo

**Total** : ~200-400 lignes selon le niveau de commentaires.

**Simplifications** : fp32 uniquement, tokenizer minimal, KV cache simple (list/tensor, pas paging), pas de training.

**Ce qu'on garde fidele** : architecture exacte, tous les composants du paper, generation autoregressive reelle.

</details>
</content>
</invoke>
# Jour 6 — Transformer Architecture : l'assemblage complet

> **Temps estime** : 6h | **Prerequis** : Jour 5 (Attention), Jour 4 (RNN pour comprendre ce qui a change)

---

## 1. "Attention is All You Need" (2017) : le papier qui a tout change

En juin 2017, 8 chercheurs de Google publient un papier titre "Attention Is All You Need". La these centrale :

> Les convolutions et les recurrences ne sont PAS necessaires. Un modele base uniquement sur de l'attention, sans aucun LSTM ni CNN, peut etre meilleur en traduction automatique ET entraine 10x plus vite.

Ce papier donne naissance au **Transformer**, l'architecture qui propulsera BERT (2018), GPT (2018-2024), LLaMA, Claude, Gemini, et presque tous les modeles modernes.

### L'architecture originale (version encoder-decoder)

```
        Source tokens                  Target tokens
              │                               │
      ┌───────┴───────┐              ┌────────┴────────┐
      │ Token embed + │              │ Token embed +   │
      │ Positional    │              │ Positional      │
      └───────┬───────┘              └────────┬────────┘
              │                               │
      ╔═══════▼═══════╗              ╔════════▼════════╗
      ║  ENCODER      ║              ║  DECODER        ║
      ║               ║              ║                 ║
      ║ x N blocks :  ║              ║ x N blocks :    ║
      ║ Self-Attn     ║              ║ Masked Self-Attn║
      ║   + Add&Norm  ║              ║   + Add&Norm    ║
      ║ FFN           ║              ║ Cross-Attn ←────╫──(source memory)
      ║   + Add&Norm  ║              ║   + Add&Norm    ║
      ║               ║              ║ FFN             ║
      ║               ║──────────────╫→  + Add&Norm    ║
      ╚═══════╤═══════╝              ╚════════╤════════╝
              │                               │
         (memory)                      ┌──────▼──────┐
                                       │ Linear       │
                                       │ + softmax    │
                                       └──────┬──────┘
                                              │
                                     next token probabilities
```

Dans ce cours, on se concentre sur le bloc **encoder** (identique aux blocs utilises dans BERT, GPT, LLaMA — sauf que GPT utilise un masked self-attention).

---

## 2. Le bloc Transformer (un seul)

### Vue d'ensemble

Un bloc Transformer se compose de 4 sous-operations :

```
               input (seq_len, d_model)
                      │
              ┌───────▼───────┐
              │  Self-Attn    │  ← Multi-Head Attention
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  Add & Norm   │  ← residual + LayerNorm
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Feed-Forward  │  ← 2 linear layers + activation
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  Add & Norm   │  ← residual + LayerNorm
              └───────┬───────┘
                      │
               output (seq_len, d_model)
```

Chaque bloc prend une sequence en entree, la transforme, et retourne une sequence de **meme shape**. On peut empiler N blocs (typiquement 6, 12, 24, 96).

### Les 4 ingredients

1. **Multi-Head Self-Attention** : ce qu'on a vu au jour 5. Chaque token regarde tous les autres tokens.

2. **Residual connection + LayerNorm** (le "Add & Norm") : stabilise l'apprentissage pour des reseaux tres profonds.

3. **Feed-Forward Network (FFN)** : un petit MLP applique **independamment a chaque position**. C'est la qu'est capturee la plupart de la "connaissance" du modele (facts, patterns).

4. **Residual connection + LayerNorm** : encore une fois pour stabiliser.

Regardons chaque ingredient en detail.

---

## 3. Le Feed-Forward Network (FFN)

### La formule

```
FFN(x) = W_2 @ ReLU(W_1 @ x + b_1) + b_2

Shapes :
  x    : (d_model,)
  W_1  : (d_ff, d_model)
  W_2  : (d_model, d_ff)

Typiquement : d_ff = 4 * d_model
  d_model = 512 → d_ff = 2048
  d_model = 768 → d_ff = 3072
  GPT-3 d_model = 12288, d_ff = 49152
```

C'est un MLP a deux couches qui expand la dimension (x4), applique une non-linearite, puis contract vers d_model.

### Le detail crucial : applique a chaque position INDEPENDAMMENT

Ce FFN est applique au vecteur de chaque position **separement**, avec les MEMES poids. C'est comme une convolution 1x1 dans l'espace temporel.

```
Input (seq_len=5, d_model=512):
  pos 0 : [x00, x01, ..., x0_511]
  pos 1 : [x10, x11, ..., x1_511]
  pos 2 : [x20, x21, ..., x2_511]
  pos 3 : [x30, x31, ..., x3_511]
  pos 4 : [x40, x41, ..., x4_511]

Apply FFN to each position (same W_1, W_2 everywhere) :
  pos 0 : FFN(x_0) = [y00, ..., y0_511]
  pos 1 : FFN(x_1) = [y10, ..., y1_511]
  ...
```

### Pourquoi le FFN est si important ?

Des etudes recentes (Geva et al., 2021 ; Meng et al., 2023) ont montre que **la plupart du "savoir" d'un LLM est stocke dans les poids du FFN**, pas dans l'attention.

**Interpretation** : l'attention route l'information entre positions (quoi regarder). Le FFN transforme l'information a une position donnee (quoi en faire). C'est le FFN qui implemente des regles comme "si le mot courant est 'Paris' et le contexte est 'capitale de ___', alors produire la France".

**Repartition typique des parametres** dans un Transformer :
- Attention : ~33% des parametres
- FFN : ~66% des parametres (a cause du ratio d_ff = 4 * d_model)

---

## 4. Residual Connections (les "skip connections")

### La formule

```
output = x + Sublayer(x)
```

Au lieu de remplacer `x` par `Sublayer(x)`, on **ajoute** le resultat de la sous-couche a l'entree. La sous-couche apprend une **correction** (residual) a appliquer a `x`.

### Pourquoi c'est crucial pour des reseaux profonds ?

**Probleme 1 : vanishing gradient dans les reseaux profonds**

Sans residuals, le gradient doit traverser N couches de multiplication par des matrices de poids. Comme pour un RNN, il vanish ou explose.

Avec residuals, le gradient a TOUJOURS un chemin direct vers les premieres couches :

```
Sans residual : dL/dx_0 = dL/dx_N * ∏_{i=1}^{N} dSublayer_i/dx_{i-1}
                → produit de matrices, peut vanish

Avec residual : dL/dx_0 = dL/dx_N * ∏_{i=1}^{N} (I + dSublayer_i/dx_{i-1})
                → le "I" garantit un gradient de base qui passe toujours
```

**Probleme 2 : apprendre l'identite est tres difficile**

Pour un reseau profond, la fonction optimale est souvent **proche** de l'identite plus quelques corrections. Sans residuals, chaque couche doit apprendre l'identite (difficile avec des matrices aleatoires). Avec residuals, la "fonction par defaut" est l'identite — la sous-couche apprend juste la correction.

**Analogie** : tu donnes ton brouillon a un relecteur. Il ne reecrit pas tout : il ajoute des corrections a la marge. C'est plus efficace que reecrire a chaque fois.

---

## 5. Layer Normalization

### Qu'est-ce que c'est ?

LayerNorm (Ba, Kiros, Hinton, 2016) normalise les activations d'une couche en calculant la moyenne et l'ecart-type **sur la dimension des features** (pas sur la dimension du batch) :

```
Pour un vecteur x = [x_1, x_2, ..., x_d] (d = d_model) :

  mu    = mean(x)                        # scalaire
  sigma = std(x) + eps                   # scalaire (eps pour eviter div 0)

  x_norm = (x - mu) / sigma              # centre et reduit
  output = gamma * x_norm + beta         # scale & shift appris (d dims)
```

`gamma` et `beta` sont des parametres learnable de dim `d_model` qui permettent au reseau de "reannuler" la normalisation si besoin.

### LayerNorm vs BatchNorm : pourquoi pas BatchNorm ?

BatchNorm normalise sur la dimension **batch** : pour chaque feature, on calcule la moyenne/std sur tous les exemples du batch.

| | BatchNorm | LayerNorm |
|---|---|---|
| Axe de normalisation | Batch | Features |
| Depend de la taille du batch | Oui | Non |
| Depend des autres exemples | Oui | Non |
| Fonctionne en inference avec batch=1 | Besoin de running stats | Oui, directement |
| Adapte aux sequences de longueur variable | Non | Oui |
| Utilise dans les Transformers | Non | Oui |

**Pourquoi LayerNorm gagne pour les Transformers** :
1. Les sequences ont des longueurs variables → BatchNorm galere avec les paddings
2. L'inference avec batch=1 est courante (generation autoregressive) → BatchNorm devrait utiliser des statistiques pre-calculees, source de bugs
3. LayerNorm est stable et ne depend de rien d'autre que du vecteur courant

### Pre-norm vs Post-norm

Il y a deux variantes possibles pour placer le LayerNorm :

**Post-norm (version originale)** :
```
output = LayerNorm(x + Sublayer(x))
```

**Pre-norm (version moderne, GPT-2+, LLaMA)** :
```
output = x + Sublayer(LayerNorm(x))
```

La difference est subtile mais importante : en pre-norm, le residual path est "pur" (pas de normalization), ce qui aide a stabiliser l'entrainement de reseaux tres profonds (96+ couches). Les architectures modernes utilisent quasi-exclusivement pre-norm.

---

## 6. Positional Encoding : donner une adresse aux tokens

### Le probleme

L'attention est **permutation-invariante** : si on melange l'ordre des tokens, les scores Q·K ne changent pas (ce sont juste des produits scalaires symetriques). Le modele ne sait donc pas distinguer "le chat mange la souris" de "la souris mange le chat".

Il faut injecter l'information de position **explicitement** dans les embeddings.

### Positional encoding sinusoidal (Vaswani et al., 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Ou `pos` est la position (0, 1, 2, ...), `i` est l'index de la paire sin/cos, et `d_model` est la dimension du modele.

**Intuition** : chaque dimension est un sinus/cosinus a une frequence differente. Les premieres dimensions ont des hautes frequences (varient vite avec la position), les dernieres ont des basses frequences (varient lentement).

C'est une "horloge multi-echelles" qui donne a chaque position une signature unique.

### Ajout au token embedding

```
input = TokenEmbed[token_id] + PE[position]

Shape : (seq_len, d_model)
```

On ADDITIONNE (pas concatene) le positional encoding au token embedding. C'est possible car tous les deux sont de dim `d_model`.

### Alternatives modernes

- **Learned positional embeddings** (GPT-2, BERT) : au lieu d'utiliser des sinusoides, on apprend une matrice `PE` de shape `(max_len, d_model)`. Simple, efficace. Defaut : ne generalise pas aux sequences plus longues que `max_len`.
- **Rotary Position Embedding (RoPE)** (LLaMA, GPT-NeoX) : applique une rotation dans le plan complexe aux Q et K. Excellente generalisation.
- **ALiBi** (MosaicML) : ajoute un biais lineaire `-m*|i-j|` aux scores d'attention. Tres simple, tres efficace.

Pour ce cours, on reste sur le positional encoding sinusoidal (plus pedagogique).

---

## 7. Encoder vs Decoder vs Encoder-Decoder

Il y a 3 "familles" d'architectures Transformer :

### 7.1 Encoder-only (BERT, RoBERTa)

- Utilise seulement l'encoder du Transformer original
- Self-attention **bidirectionnelle** (pas de masking) : chaque token voit TOUS les autres
- Entraine avec **masked language modeling** : on masque 15% des tokens et on predit les mots masques
- **Utilise pour** : classification, NER, question-answering, embeddings de texte

```
Input  : "le [MASK] mange la souris"
Output : le modele predit "chat" pour [MASK]
```

### 7.2 Decoder-only (GPT, LLaMA, Claude)

- Utilise seulement le decoder du Transformer original
- Self-attention **causale** (masquage triangulaire) : chaque token ne voit que le passe
- Entraine avec **next-token prediction** : predire le prochain token
- **Utilise pour** : generation de texte, chatbots, code completion

```
Input  : "le chat mange la"
Output : le modele predit la distribution sur le prochain token (ex: "souris")
```

### 7.3 Encoder-Decoder (T5, BART, original Transformer)

- Utilise les deux : encoder bidirectionnel + decoder causal
- Le decoder a une **cross-attention** qui regarde la sortie de l'encoder
- Entraine avec **sequence-to-sequence** : la sortie depend entierement de l'entree
- **Utilise pour** : traduction, summarization, text-to-text tasks

```
Input  : "The cat eats the mouse"  (encoder)
Output : "Le chat mange la souris" (decoder, generated token by token)
```

### Timeline

| Annee | Modele | Famille | Taille |
|---|---|---|---|
| 2017 | Original Transformer | Encoder-Decoder | 213M |
| 2018 | BERT | Encoder-only | 340M |
| 2018 | GPT-1 | Decoder-only | 117M |
| 2019 | GPT-2 | Decoder-only | 1.5B |
| 2019 | T5 | Encoder-Decoder | 11B |
| 2020 | GPT-3 | Decoder-only | 175B |
| 2023 | LLaMA-2 | Decoder-only | 70B |
| 2024 | Claude 3, GPT-4 | Decoder-only | ~1T (estime) |

**Observation** : la famille decoder-only a gagne. Pourquoi ? Parce qu'elle est plus simple, scale mieux, et peut faire toutes les taches (classification, traduction, generation) en formulant la tache comme une generation de texte.

---

## 8. Comptage des parametres d'un bloc Transformer

Prenons un bloc standard avec `d_model = 512`, `n_heads = 8`, `d_ff = 2048` :

### Attention (Q, K, V + output projection)

```
W_Q : 512 * 512 = 262 144
W_K : 512 * 512 = 262 144
W_V : 512 * 512 = 262 144
W_O : 512 * 512 = 262 144
Total attention : 1 048 576 params (~1.05M)
```

### FFN (2 linear layers)

```
W_1 : 512 * 2048 = 1 048 576
W_2 : 2048 * 512 = 1 048 576
Total FFN : 2 097 152 params (~2.1M)
```

### LayerNorm (2 layers, gamma + beta each)

```
gamma_1 + beta_1 : 2 * 512 = 1024
gamma_2 + beta_2 : 2 * 512 = 1024
Total LN : 2048 params (negligible)
```

### Total pour 1 bloc

```
Attention : 1.05M   (~33%)
FFN       : 2.10M   (~66%)
LayerNorm : 2K      (~0.1%)
TOTAL     : ~3.15M par bloc
```

**Pour un modele entier** (ex: GPT-2 small, 12 blocs) : ~3.15M * 12 = ~38M juste pour les blocs. Plus embedding (vocab * d_model) ≈ 39M de plus. Total = 124M pour GPT-2 small, ce qui colle avec les chiffres officiels.

---

## 9. Flash Cards — Active Recall

### Q1 : Quels sont les 4 ingredients d'un bloc Transformer ? Dans quel ordre ?

<details>
<summary>Reponse</summary>

Dans l'ordre :
1. **Multi-Head Self-Attention** : chaque token regarde tous les autres
2. **Residual + LayerNorm (Add & Norm)** : stabilise la premiere sous-couche
3. **Feed-Forward Network (FFN)** : MLP a 2 couches applique independamment a chaque position
4. **Residual + LayerNorm (Add & Norm)** : stabilise la deuxieme sous-couche

Avec le FFN ayant generalement `d_ff = 4 * d_model` (expansion puis contraction).

L'entree et la sortie d'un bloc ont la meme shape `(seq_len, d_model)`, ce qui permet d'empiler N blocs.

</details>

### Q2 : Pourquoi utilise-t-on LayerNorm au lieu de BatchNorm dans les Transformers ?

<details>
<summary>Reponse</summary>

LayerNorm normalise sur la dimension **features** (chaque token est normalise independamment), alors que BatchNorm normalise sur la dimension **batch**.

Pour les Transformers, LayerNorm est prefere parce que :
1. Les sequences ont des longueurs variables → BatchNorm galere avec les paddings
2. L'inference autoregressive (batch=1, token par token) est courante → BatchNorm aurait besoin de running stats, source de bugs
3. LayerNorm est deterministe et ne depend pas des autres exemples du batch
4. LayerNorm est stable pour tres petits batches (voire batch=1)

En pratique, les architectures modernes (LLaMA, GPT) utilisent une variante appelee RMSNorm, encore plus simple (pas de `beta`).

</details>

### Q3 : Qu'est-ce qu'une residual connection et pourquoi est-ce crucial pour les reseaux profonds ?

<details>
<summary>Reponse</summary>

Une residual connection est l'operation `output = x + Sublayer(x)`. La sous-couche apprend une **correction additive** a appliquer a l'entree, au lieu de la transformer entierement.

Deux raisons fondamentales :

1. **Gradient highway** : sans residual, le gradient doit traverser N matrices de poids en remontant. Il peut vanish (produit de valeurs propres < 1) ou exploser. Avec residual, le gradient a toujours un chemin direct : `d/dx (x + f(x)) = I + f'(x)`, donc au pire `f'(x) = 0` et le gradient passe sans modification.

2. **Identite comme fonction par defaut** : la plupart des transformations optimales sont proches de l'identite plus quelques corrections. Sans residual, chaque couche doit apprendre l'identite (difficile avec des poids aleatoires). Avec residual, l'identite est gratuite.

Les Transformers sans residuals deviennent instables au-dela de ~6 couches. Avec residuals, on peut aller a 96+ couches (GPT-3).

</details>

### Q4 : Qu'est-ce que le Feed-Forward Network (FFN) d'un Transformer et pourquoi represente-t-il 2/3 des parametres ?

<details>
<summary>Reponse</summary>

Le FFN est un MLP a 2 couches applique independamment a chaque position :
```
FFN(x) = W_2 @ ReLU(W_1 @ x + b_1) + b_2
```

Avec typiquement `d_ff = 4 * d_model`. Pour d_model=512, d_ff=2048.

Comptage : `W_1` est `d_model * d_ff = 512*2048 = 1M`, `W_2` est pareil, total FFN = 2M. L'attention a `W_Q, W_K, W_V, W_O` chacune `d_model^2 = 262K`, total = 1M.

Donc FFN/bloc ≈ 2M et Attention/bloc ≈ 1M → FFN represente 2/3 des parametres.

**Pourquoi si gros ?** Des etudes (Geva et al., 2021) ont montre que le FFN agit comme une "memoire cles-valeurs" ou la majorite des faits et patterns lexicaux sont stockes. L'attention route, le FFN transforme.

</details>

### Q5 : Quelle est la difference entre un Transformer encoder-only (BERT), decoder-only (GPT), et encoder-decoder (T5) ?

<details>
<summary>Reponse</summary>

| | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| Attention | Bidirectionnelle | Causale (masquee) | Encoder biDir, decoder causal + cross-attn |
| Tache de pretraining | Masked language modeling (MLM) | Next token prediction | Text-to-text (seq2seq) |
| Exemples | BERT, RoBERTa | GPT, LLaMA, Claude | T5, BART, original Transformer |
| Utilisation | Classification, NER, embeddings | Generation, chatbots | Traduction, summarization |

**Qui a gagne ?** Decoder-only. Plus simple, scale mieux, et peut faire toutes les taches en formulant la tache comme "generer la reponse sachant l'input en contexte". Tous les gros LLMs modernes (GPT-4, Claude, Gemini, LLaMA) sont decoder-only.

</details>

---

## 10. Key Takeaways

1. **Un bloc Transformer = MHA + Add&Norm + FFN + Add&Norm**. On empile N blocs pour obtenir le modele complet.

2. **Le FFN represente ~2/3 des parametres** d'un bloc et contient la majorite du "savoir" du modele. L'attention route, le FFN transforme.

3. **Les residual connections** sont indispensables pour entrainer des reseaux profonds (N > 6). Elles permettent au gradient de circuler directement et font de l'identite la fonction par defaut.

4. **LayerNorm** (pas BatchNorm) est standard dans les Transformers, car il ne depend pas de la taille du batch et marche parfaitement en inference avec batch=1.

5. **Le positional encoding** (sinusoidal, learned, RoPE, ALiBi) est indispensable car l'attention est permutation-invariante.

6. **3 familles d'architectures** : encoder-only (BERT), decoder-only (GPT, LLaMA), encoder-decoder (T5). La famille decoder-only a gagne.

---

**Prochain jour** : J7 — on assemble tout ca pour construire un mini-GPT qui genere du texte.
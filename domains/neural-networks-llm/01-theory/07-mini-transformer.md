# Jour 7 — Mini-Transformer Build (Capstone Week 1)

> **Temps estime** : 6-8h | **Prerequis** : Jours 1-6 (tout)

---

## 1. Objectif : construire un GPT miniature from scratch

Aujourd'hui on assemble tout ce qu'on a vu pour construire un **mini-GPT** qui apprend a generer du texte. Specifications :

- ~200-300 lignes de PyTorch
- Tokenizer caractere (le plus simple possible)
- Token embedding + positional encoding
- N blocs Transformer decoder (self-attention causale + FFN)
- Couche de sortie (projection vers vocab → distribution sur les caracteres)
- Boucle d'entrainement avec cross-entropy loss
- Generation autoregressive (greedy + sampling)
- Entraine sur un petit corpus (une citation repetee, ou Tiny Shakespeare inlined)

C'est le meme principe que GPT-2/GPT-3/Claude, juste 1 million de fois plus petit. Les concepts sont **identiques**.

---

## 2. Architecture detaillee du mini-GPT

```
           input token IDs (batch, seq_len)
                        │
                        ▼
            ┌──────────────────────┐
            │ Token Embedding      │  (vocab_size, n_embed)
            │     +                │
            │ Positional Encoding  │  (max_len, n_embed)
            └──────────┬───────────┘
                       │  (batch, seq_len, n_embed)
                       ▼
            ┌──────────────────────┐
            │ Transformer Block 1  │
            │   - MHA causal       │
            │   - Add & Norm       │
            │   - FFN              │
            │   - Add & Norm       │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Transformer Block 2  │
            └──────────┬───────────┘
                       │
                       ▼
                     ...  (n_layer blocks total)
                       │
                       ▼
            ┌──────────────────────┐
            │ Final LayerNorm      │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Linear n_embed→vocab │
            └──────────┬───────────┘
                       │
                       ▼
              logits (batch, seq_len, vocab_size)
                       │
                       ▼
                softmax + cross-entropy
                 (vs target tokens)
```

### Hyperparametres "CPU-friendly"

```
vocab_size : determine par le corpus (~60 chars pour un corpus simple)
n_embed    : 32        (equiv a GPT-3 12288 / 400)
n_head     : 4
n_layer    : 2
block_size : 32        (longueur max des sequences pendant l'entrainement)
batch_size : 16
lr         : 3e-3
max_iters  : 3000
```

Total parametres : ~30-50K. Suffisant pour apprendre a generer des caracteres qui ressemblent vaguement au texte d'entree.

---

## 3. Le tokenizer caractere : le plus simple du monde

### L'idee

Au lieu de tokens "mots" ou "sous-mots" (BPE, WordPiece), on tokenize par **caractere individuel**. Chaque caractere unique du corpus est un token.

```
Corpus : "le chat mange"

Chars uniques : [' ', 'a', 'c', 'e', 'g', 'h', 'l', 'm', 'n', 't']
vocab_size    : 10

char_to_idx : {' ': 0, 'a': 1, 'c': 2, 'e': 3, ...}
idx_to_char : {0: ' ', 1: 'a', 2: 'c', 3: 'e', ...}

Encode "le chat" :
  l=6, e=3, ' '=0, c=2, h=5, a=1, t=9
  → [6, 3, 0, 2, 5, 1, 9]

Decode [6, 3, 0, 2, 5, 1, 9] :
  → "le chat"
```

### Pros et cons

**Pros** :
- Vocabulaire minuscule (60-100 tokens vs 50K pour GPT)
- Pas d'OOV (out-of-vocabulary) possible — tout caractere connu se tokenize
- Tres simple a implementer (5 lignes de Python)

**Cons** :
- Sequences beaucoup plus longues (un mot = 5-10 tokens au lieu de 1)
- Les patterns linguistiques prennent plus de temps a apprendre
- Gaspillage de compute (le modele doit "reconstruire" les mots a chaque fois)

Pour un mini-GPT pedagogique, c'est le bon choix. Pour un vrai modele, on utilise BPE/SentencePiece.

---

## 4. L'entrainement autoregressif

### Le principe : predire le caractere suivant

A chaque pas, on donne au modele une sequence et il doit predire le caractere SUIVANT a chaque position.

```
Input  (x) : "hello wor"
Target (y) : "ello worl"   (decale de 1)

Pour chaque position i, on veut :
  model(x)[i] ≈ y[i]

Position 0 : 'h' → predit 'e'
Position 1 : 'e' → predit 'l'
Position 2 : 'l' → predit 'l'
...
Position 8 : 'r' → predit 'l'
```

### Le batching

On echantillonne des "chunks" aleatoires du corpus. Pour un `block_size = 32` :

```
Corpus : "Once upon a time, in a galaxy far far away..."
           0    5    10   15   20   25   30   35   40

Random start = 7 :
  x = corpus[7 : 7+32]
  y = corpus[8 : 8+32]

x = "n a time, in a galaxy far far a"
y = " a time, in a galaxy far far aw"
```

On prend `batch_size` chunks comme ca, on les empile, et on a notre batch.

### La loss

Pour chaque position dans le batch, on a :
- Un logit vector de dim `vocab_size` (predit par le modele)
- Un target token (le vrai caractere suivant)

La loss est la cross-entropy moyennee sur toutes les positions :

```
loss = (1/N) * sum over (batch, seq) of CrossEntropy(logits, target)

Ou N = batch_size * seq_len
```

En PyTorch :
```python
logits = model(x)              # (B, T, vocab_size)
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B*T, vocab_size)
    y.view(-1)                      # (B*T,)
)
```

### L'avantage du parallelisme

Le Transformer predit TOUS les caracteres d'un coup (en parallele), grace au masquage causal. Pour un `block_size = 32`, on calcule 32 losses en une seule passe forward. C'est ca qui rend l'entrainement 100x plus rapide qu'un RNN.

---

## 5. La generation autoregressive

### Greedy decoding

Le plus simple : on part d'un caractere (ou d'un prompt), on predit le caractere suivant, on l'ajoute au contexte, on predit le suivant, etc.

```python
def generate(model, start_ids, max_new_tokens):
    ids = start_ids
    for _ in range(max_new_tokens):
        # Crop to last block_size tokens (if too long)
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)       # (B, T, vocab_size)
        logits_last = logits[:, -1, :] # only the LAST position
        # Greedy: take the argmax
        next_id = torch.argmax(logits_last, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
```

**Probleme du greedy** : il produit toujours la meme suite pour le meme prompt. Pas de diversite. Tendance a boucler sur des patterns courts.

### Sampling avec temperature

Au lieu de prendre l'argmax, on echantillonne selon la distribution softmax. La **temperature** T controle le niveau de randomness :

```
probs = softmax(logits / T)

T = 1   : distribution naturelle
T < 1   : distribution plus peakue (plus deterministe)
T > 1   : distribution plus plate (plus aleatoire)
T → 0   : equivalent au greedy
T → ∞   : equivalent a l'uniforme
```

```python
def generate_temperature(model, start_ids, max_new_tokens, temperature=1.0):
    ids = start_ids
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)
        logits_last = logits[:, -1, :] / temperature  # apply temperature
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # sample
        ids = torch.cat([ids, next_id], dim=1)
    return ids
```

### Top-k et top-p (nucleus) sampling

Extensions populaires :
- **Top-k** : on ne garde que les k tokens les plus probables et on renormalise
- **Top-p (nucleus)** : on garde les tokens les plus probables jusqu'a ce que leur somme depasse p (ex: 0.9)

Ces techniques evitent de tirer des tokens ultra-rares qui cassent la coherence.

---

## 6. Ce qu'on va observer

### Phase 1 : random generation (loss ~4.0)

Au debut, avec des poids aleatoires, le modele predit quasi-uniformement. Chaque caractere a ~1/vocab_size de probabilite. Pour vocab_size=60 :

```
loss_initial ≈ -log(1/60) ≈ log(60) ≈ 4.09
```

La generation est du bruit aleatoire.

### Phase 2 : n-gram (loss ~2.5)

Apres quelques centaines d'iterations, le modele apprend les frequences de caracteres (l'espace est frequent, les voyelles aussi) et les bi-grammes simples ("qu" est frequent en francais, "th" en anglais).

La generation commence a ressembler a des "mots" sans signification :
```
eua tre le ch lle la ou al est mont
```

### Phase 3 : petits mots (loss ~1.8)

Apres ~1000 iterations, le modele capture des tri-grammes et des patterns courts. Il genere de vrais mots courts frequents dans le corpus :
```
le chat mange le grand chat mange sur la
```

### Phase 4 : structure (loss ~1.2)

Apres ~2000-3000 iterations, le modele capture la structure de phrase, la ponctuation, les tournures. Mais il **hallucine** toujours — il n'a aucune comprehension semantique, juste des patterns statistiques.

---

## 7. Les limites d'un mini-GPT

Ce qu'on construit aujourd'hui **ne peut pas** :
- Comprendre le sens
- Repondre a des questions
- Raisonner
- Faire de l'arithmetique

Ce qu'il **peut** :
- Apprendre les patterns statistiques d'un corpus
- Generer du texte qui ressemble au corpus
- Montrer que l'architecture Transformer FONCTIONNE meme a petite echelle

### Pourquoi GPT-3/4 est si different ?

- **Scale** : 175B parametres vs 50K ici (3.5 millions de fois plus gros)
- **Data** : 300B tokens (vs ~2000 caracteres ici)
- **Training compute** : ~3000 PFLOP-days (vs ~1 seconde sur CPU ici)
- **Emergent capabilities** : a cette echelle, des comportements qualitativement nouveaux apparaissent (raisonnement, few-shot learning, code)

Mais l'architecture est fondamentalement LA MEME. C'est ca le plus hallucinant.

---

## 8. Checklist finale pour le capstone

Avant de lancer le code, verifier que tu sais expliquer :

- [ ] Pourquoi on utilise un masque causal (et pas bidirectionnel)
- [ ] Comment le token embedding + positional encoding construisent l'input
- [ ] Pourquoi un bloc Transformer preserve la shape (seq_len, d_model)
- [ ] Le role de la final LayerNorm + output projection
- [ ] Comment la cross-entropy calcule la loss sur tous les positions en parallele
- [ ] La difference entre greedy et sampling avec temperature
- [ ] Pourquoi on ne peut pas depasser `block_size` en generation (solution : croper le contexte)

Si tu es fluent sur ces 7 points, tu as la base pour lire le code de nanoGPT, comprendre le code de LLaMA, et implementer ton propre LLM.

---

## 9. Key Takeaways

1. **Un GPT = token embed + pos encoding + N transformer blocks (causal) + final LN + projection**. C'est tout.

2. **Training loop** : predire le token suivant a chaque position, cross-entropy loss, backprop, repeat.

3. **Generation** : autoregressive (token par token). Greedy ou sampling avec temperature.

4. **Parallelisme** : durant l'entrainement, tous les tokens d'une sequence sont predits en parallele grace au masquage causal. Durant la generation, c'est sequentiel (un token a la fois).

5. **Scale compte** : un mini-GPT de 50K parametres est limite a apprendre des patterns de surface. Un GPT-3 de 175B parametres developpe des capacites emergentes. Mais l'architecture est identique.

6. **Tu peux lire le code de nanoGPT** maintenant. Et, avec quelques lignes de plus, tu peux lire le code de LLaMA et GPT-2. L'architecture est stable depuis 2017.

---

**Fin de la semaine 1**. Tu as construit, from scratch :
- Un neurone + backprop
- Un MLP
- Des embeddings
- Un RNN/LSTM
- L'attention
- Un bloc Transformer
- Un mini-GPT complet qui genere du texte

La semaine 2 : pretraining, fine-tuning, RLHF, tokenization avancee, scaling laws.
